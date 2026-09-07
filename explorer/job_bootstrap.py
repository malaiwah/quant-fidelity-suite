#!/usr/bin/env python3
"""Bootstrap an immutable public QFS worker in a Python 3.12 full-bookworm Job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time

SOURCE = "https://github.com/malaiwah/quant-fidelity-suite"
CHECKOUT = Path("/tmp/qfs-job-source")
LOG = Path("/outputs/bootstrap.log")
RECEIPT = Path("/outputs/bootstrap.json")
LOG_LIMIT = 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sealed(value, field):
    value[field] = ""
    value[field] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def save(path, value):
    with path.open("wb") as target:
        target.write(canonical(value) + b"\n")
        target.flush()
        os.fsync(target.fileno())


def clean(raw):
    value = raw.decode("utf-8", "replace")
    value = re.sub(r"hf_[A-Za-z0-9]{10,}|(?i:bearer)\s+\S+", "[REDACTED]", value)
    return "".join(c for c in value if c in "\n\t" or 32 <= ord(c) != 127).encode()


def run(command, deadline, commands):
    record = {"argv": command, "returncode": None}
    commands.append(record)
    save(RECEIPT, {"schema": "qfs.hf-workflow-bootstrap.v1", "commands": commands})
    print(json.dumps({"stage": "bootstrap", "command": command[:4], "event": "started"}), flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    observed = 0
    try:
        with LOG.open("ab") as target:
            while selector.get_map():
                if time.monotonic() >= deadline:
                    raise TimeoutError("bootstrap consumed the plan deadline")
                for key, _ in selector.select(timeout=0.5):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    observed += len(chunk)
                    target.write(clean(chunk)[:max(0, LOG_LIMIT - target.tell())])
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        selector.close()
        process.stdout.close()
        record.update(returncode=process.returncode, output_bytes_observed=observed)
        save(RECEIPT, {"schema": "qfs.hf-workflow-bootstrap.v1", "commands": commands})
        print(json.dumps({"stage": "bootstrap", "returncode": process.returncode}), flush=True)
    if process.returncode:
        raise RuntimeError("bootstrap command failed with exit " + str(process.returncode))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.plan != "/inputs/plan/plan.json" or args.out != "/outputs/result" or Path("/outputs").resolve() != Path("/outputs"):
        parser.error("bootstrap requires the fixed plan and bucket output paths")
    if Path(args.out).exists() or LOG.exists() or RECEIPT.exists():
        parser.error("fresh per-attempt bucket prefix required")
    started = time.time()
    plan, commands = {}, []
    try:
        if sys.version_info[:2] != (3, 12):
            raise ValueError("worker image must provide Python 3.12")
        forbidden = ("HF_TOKEN", "HF_TOKEN_PATH", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN", "OAUTH_TOKEN")
        if any(os.environ.get(name) for name in forbidden):
            raise ValueError("HF credentials must never be passed to the worker")
        with open(args.plan, "rb") as source:
            payload = source.read(4 * 1024 * 1024 + 1)
        if len(payload) > 4 * 1024 * 1024:
            raise ValueError("plan exceeds bound")
        document = json.loads(payload)
        if not isinstance(document, dict):
            raise ValueError("plan must be a JSON object")
        plan = document
        if plan.get("schema") != "qfs.hf-workflow-plan.v1" or sealed(dict(plan), "plan_sha256")["plan_sha256"] != plan.get("plan_sha256"):
            raise ValueError("plan schema or self-seal mismatch")
        source = plan["source"]
        if source.get("repository") != SOURCE or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision"))):
            raise ValueError("only an immutable commit of the fixed public QFS repository is allowed")
        timeout = plan["hardware"]["timeout_seconds"]
        if type(timeout) is not int or not 0 < timeout <= 86400:
            raise ValueError("invalid runtime deadline")
        deadline = time.monotonic() + timeout - (time.time() - started)
        device = plan["hardware"]["device"]
        if device not in ("cpu", "cuda"):
            raise ValueError("unsupported device")
        if CHECKOUT.exists():
            raise ValueError("refusing an existing source checkout")
        os.environ.update(HF_HUB_DISABLE_IMPLICIT_TOKEN="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1", PYTHONDONTWRITEBYTECODE="1", GIT_TERMINAL_PROMPT="0", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", QFS_WORKFLOW_STARTED=str(started))
        for name in ("PYTHONPATH", "PYTHONHOME", "PIP_EXTRA_INDEX_URL", "PIP_INDEX_URL", "PIP_FIND_LINKS", "PIP_TRUSTED_HOST"):
            os.environ.pop(name, None)
        os.environ["PIP_CONFIG_FILE"] = "/dev/null"
        run(["git", "init", str(CHECKOUT)], deadline, commands)
        run(["git", "-C", str(CHECKOUT), "remote", "add", "origin", SOURCE], deadline, commands)
        run(["git", "-C", str(CHECKOUT), "-c", "core.hooksPath=/dev/null", "fetch", "--depth", "1", "origin", source["revision"]], deadline, commands)
        run(["git", "-C", str(CHECKOUT), "-c", "core.hooksPath=/dev/null", "checkout", "--detach", source["revision"]], deadline, commands)
        worker = CHECKOUT / "explorer/job_worker.py"
        if hashlib.sha256(worker.read_bytes()).hexdigest() != source["worker_sha256"]:
            raise ValueError("immutable checkout worker hash mismatch")
        if (CHECKOUT / "explorer/job_bootstrap.py").read_bytes() != Path(__file__).read_bytes():
            raise ValueError("bootstrap bytes do not match the immutable source revision")
        # Validate the entire plan and read-only mounts before dependency installation.
        sys.path.insert(0, str(CHECKOUT / "explorer"))
        import job_worker
        job_worker.require_no_credentials()
        job_worker.validate_plan(plan, Path(args.out))
        wheel_kind = "cpu" if device == "cpu" else "cu130"
        run([sys.executable, "-m", "pip", "install", "--only-binary=:all:", "--index-url", "https://download.pytorch.org/whl/" + wheel_kind, "torch==2.11.0+" + wheel_kind], deadline, commands)
        run([sys.executable, "-m", "pip", "install", "--only-binary=:all:", "--index-url", "https://pypi.org/simple", "-r", str(CHECKOUT / "explorer/requirements-worker.txt")], deadline, commands)
        from importlib.metadata import distributions
        versions = {distribution.metadata["Name"]: distribution.version for distribution in distributions()}
        save(RECEIPT, {"schema": "qfs.hf-workflow-bootstrap.v1", "commands": commands,
                       "python": sys.version, "installed_versions": dict(sorted(versions.items())),
                       "source_revision": source["revision"], "worker_sha256": source["worker_sha256"],
                       "requirements_sha256": hashlib.sha256((CHECKOUT / "explorer/requirements-worker.txt").read_bytes()).hexdigest()})
        os.execve(sys.executable, [sys.executable, str(worker), "--plan", args.plan, "--out", args.out], os.environ)
    except BaseException as exc:
        output = Path(args.out)
        output.mkdir(mode=0o700)
        for path in (LOG, RECEIPT):
            if path.is_file():
                shutil.copyfile(path, output / path.name)
        result = {"schema": "qfs.hf-workflow-result.v1", "workflow_id": plan.get("workflow_id"), "owner": plan.get("owner"), "mode": plan.get("mode"), "plan_sha256": plan.get("plan_sha256"), "status": "failed", "outputs": {}, "source": plan.get("source", {}), "files": [], "error": {"type": type(exc).__name__, "stage": "bootstrap", "message": clean(str(exc).encode())[:4096].decode("utf-8", "replace")}}
        for path in sorted(output.iterdir()):
            result["files"].append({"path": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        save(output / "result.json", sealed(result, "result_sha256"))
        os.sync()
        print("QFS bootstrap failed; private partial result retained", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
