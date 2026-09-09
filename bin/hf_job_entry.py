#!/opt/fidelity/venv/bin/python -I
"""Run a sealed HF Jobs capture, measurement, or comparison; never create a Job."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tempfile
import time
import urllib.request

SOURCE = "https://github.com/malaiwah/quant-fidelity-suite"
RAW_SOURCE = "https://raw.githubusercontent.com/malaiwah/quant-fidelity-suite/"
PYTHON = "/opt/fidelity/venv/bin/python"
PLAN = Path("/inputs/plan/plan.json")
OUT = Path("/outputs/result")
CONTRACT = "measurement-cli-v1"
MAX_JSON = 4 * 1024 * 1024
MAX_BOOTSTRAP = 1024 * 1024
ACTIONS = {"root": "capture", "candidate": "measure", "compare": "compare"}
HEX64 = re.compile(r"[0-9a-f]{64}")
REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def load_plan(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("plan contains duplicate JSON keys; prepare a new plan")
            value[key] = item
        return value

    with path.open("rb") as source:
        raw = source.read(MAX_JSON + 1)
    if len(raw) > MAX_JSON:
        raise ValueError("plan exceeds the 4 MiB limit")
    plan = json.loads(raw, object_pairs_hook=unique)
    canonical(plan)  # Reject non-finite numbers anywhere, including exponent overflow.
    if not isinstance(plan, dict):
        raise ValueError("plan must be a JSON object")
    return plan


def validate_plan(plan, action, environ):
    if plan.get("schema") != "qfs.hf-workflow-plan.v1" or plan.get("launch_contract") != CONTRACT:
        raise ValueError("prepare a new plan with the measurement-cli-v1 launch contract")
    expected = plan.get("plan_sha256")
    body = dict(plan)
    body["plan_sha256"] = ""
    if (not isinstance(expected, str) or not HEX64.fullmatch(expected)
            or hashlib.sha256(canonical(body)).hexdigest() != expected):
        raise ValueError("plan self-seal mismatch; restore the approved plan bytes")
    wid = plan.get("workflow_id")
    if (not isinstance(wid, str) or not re.fullmatch(r"[0-9a-f]{32}", wid)
            or environ.get("QFS_WORKFLOW_ID") != wid
            or environ.get("QFS_PLAN_SHA256") != expected):
        raise ValueError("QFS_PLAN_SHA256 and QFS_WORKFLOW_ID must bind this approved plan")
    if ACTIONS.get(plan.get("mode")) != action:
        raise ValueError("action does not match the sealed plan role; use capture for root, measure for candidate, compare for compare")
    source = plan.get("source")
    if (not isinstance(source, dict) or source.get("repository") != SOURCE
            or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision")))
            or any(not HEX64.fullmatch(str(source.get(key))) for key in
                   ("bootstrap_sha256", "worker_sha256", "environment_sha256"))):
        raise ValueError("source requires the fixed public QFS repository, a 40-hex commit and all three source hashes")
    if not re.fullmatch(r"[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}", str(plan.get("image"))):
        raise ValueError("plan requires an immutable image digest")
    owner, output = plan.get("owner"), plan.get("output")
    if (not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", owner)
            or not isinstance(output, dict) or output.get("mount_path") != "/outputs"
            or output.get("prefix") != "runs/" + wid
            or any(not REPOSITORY.fullmatch(str(output.get(key)))
                   or output[key].split("/")[0] != owner for key in ("bucket", "dataset_repository"))):
        raise ValueError("output must name the sealed owner's fresh workflow bucket prefix")
    hardware = plan.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError("plan requires the approved hardware and budget")
    timeout = hardware.get("timeout_seconds")
    if type(timeout) is not int or not 0 < timeout <= 86400 or hardware.get("device") not in ("cpu", "cuda"):
        raise ValueError("hardware requires cpu/cuda and an integer timeout in 1..86400 seconds")
    for key in ("hourly_usd", "max_compute_usd"):
        value = hardware.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("hardware requires finite positive prices and consent ceiling")
        try:
            value = float(value)
        except (ValueError, OverflowError):
            raise ValueError("hardware requires finite positive prices and consent ceiling") from None
        if not math.isfinite(value) or value <= 0:
            raise ValueError("hardware requires finite positive prices and consent ceiling")
    return timeout


def require_no_credentials():
    forbidden = ("HF_TOKEN", "HF_TOKEN_PATH", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN", "OAUTH_TOKEN")
    if any(os.environ.get(name) for name in forbidden):
        raise ValueError("HF credentials must never be passed to this tokenless worker")
    for path in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
        if path.exists():
            raise ValueError("ambient HF credential file is forbidden")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("bootstrap source redirected; only the fixed immutable HTTPS URL is allowed")


def fetch_bootstrap(source, deadline):
    url = RAW_SOURCE + source["revision"] + "/explorer/job_bootstrap.py"
    # No proxy credentials, ambient auth, redirects, or caller-selected URL.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("workflow deadline expired before bootstrap fetch")
    try:
        with opener.open(url, timeout=min(60, remaining)) as response:
            raw = response.read(MAX_BOOTSTRAP + 1)
    except OSError:
        raise ValueError("cannot fetch the fixed public bootstrap within the deadline; check source availability and network access") from None
    if len(raw) > MAX_BOOTSTRAP:
        raise ValueError("bootstrap exceeds the 1 MiB limit")
    if hashlib.sha256(raw).hexdigest() != source["bootstrap_sha256"]:
        raise ValueError("bootstrap SHA-256 mismatch; no downloaded code was executed")
    return raw


def publish_bootstrap(raw):
    directory = Path(tempfile.mkdtemp(prefix="qfs-job-launch-"))
    target = directory / "job_bootstrap.py"
    temporary = directory / ".bootstrap.tmp"
    with temporary.open("xb") as output:
        os.chmod(temporary, 0o600)
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)
    return target


def build_parser():
    parser = argparse.ArgumentParser(prog="qfs-job", description=__doc__,
        epilog="Consumes the approved plan without model, code, credential, or budget overrides. Uses the baked measurement interpreter; installs no dependencies.")
    parser.add_argument("--version", action="version", version="qfs-job " + CONTRACT)
    actions = parser.add_subparsers(dest="action", required=True)
    for action, description in (("capture", "capture and repeat a reference root"),
                                ("measure", "capture a candidate and compare to its reference"),
                                ("compare", "compare sealed reference and candidate captures")):
        sub = actions.add_parser(action, help=description, description=description.capitalize() + " using the sealed HF workflow plan.")
        sub.add_argument("--plan", required=True, help="approved sealed plan; must be /inputs/plan/plan.json")
        sub.add_argument("--out", required=True, help="fresh result directory; must be /outputs/result")
    return parser


def main(argv=None):
    started = time.time()
    start_clock = time.monotonic()
    args = build_parser().parse_args(argv)
    try:
        if args.plan != str(PLAN) or args.out != str(OUT):
            raise ValueError("use the fixed --plan /inputs/plan/plan.json --out /outputs/result paths")
        require_no_credentials()
        if (PLAN.resolve() != PLAN or not stat.S_ISREG(PLAN.stat().st_mode)
                or OUT.parent.resolve() != OUT.parent or not OUT.parent.is_dir()):
            raise ValueError("plan and output mounts must be regular non-symlink paths")
        if any(path.exists() or path.is_symlink() for path in
               (OUT, OUT.parent / "bootstrap.log", OUT.parent / "bootstrap.json", Path("/tmp/qfs-job-source"))):
            raise ValueError("fresh per-attempt output and source paths required; do not resume this launch")
        plan = load_plan(PLAN)
        timeout = validate_plan(plan, args.action, os.environ)
        if not Path(PYTHON).is_file() or not os.access(PYTHON, os.X_OK):
            raise ValueError("baked /opt/fidelity/venv/bin/python is unavailable; use the pinned HF Jobs image")
        deadline = start_clock + timeout

        def expired(signum, frame):
            raise TimeoutError("workflow deadline expired during launcher setup")

        previous = signal.signal(signal.SIGALRM, expired)
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("workflow deadline expired during plan validation")
            signal.setitimer(signal.ITIMER_REAL, remaining)
            raw = fetch_bootstrap(plan["source"], deadline)
            target = publish_bootstrap(raw)
            if time.monotonic() >= deadline:
                raise TimeoutError("workflow deadline expired before bootstrap execution")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        environment = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"):
            environment.pop(name, None)
        environment.update(QFS_WORKFLOW_STARTED=str(started), HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                           PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        os.execve(PYTHON, [PYTHON, "-I", str(target), "--plan", str(PLAN), "--out", str(OUT)], environment)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        # Do not echo plan values, network headers, URLs, or credential-shaped input.
        detail = (str(exc) if type(exc) in (ValueError, TimeoutError) else
                  "check the fixed plan/output mounts and pinned image (" + type(exc).__name__ + ")")
        print("qfs-job: refused: " + detail[:512], file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
