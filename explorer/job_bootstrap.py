#!/usr/bin/env python3
"""Run an immutable public QFS worker using the verified baked measurement venv."""
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
IMAGE_ROOT = Path("/opt/fidelity")
PYTHON = "/opt/fidelity/venv/bin/python"
RUNTIME = Path("/outputs/image-runtime.json")


def verify_image(environment, *, image_root=IMAGE_ROOT):
    """Verify immutable BUILD material without equating baked and worker sources."""
    sys.path.insert(0, str(CHECKOUT / "bin"))
    from container_manifest import canonical as manifest_canonical, sha256_file
    raw = (image_root / "BUILD.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != environment["build_sha256"]:
        raise ValueError("measurement image BUILD.json hash mismatch")
    build = json.loads(raw)
    if build.get("schema") != "malaiwah.fidelity-image-build.v1" or build.get("probe_errors"):
        raise ValueError("measurement BUILD schema or build probes failed")
    material = {key: build[key] for key in ("pins", "patches_sha256", "bundle_sha256", "pip_freeze_sha256", "suite_revision")}
    content = hashlib.sha256(manifest_canonical(material).encode()).hexdigest()
    if (content != build.get("image_content_sha256") or content != environment["image_content_sha256"]
            or (image_root / "image-pin.txt").read_text().strip() != content
            or build["suite_revision"] != environment["baked_source_revision"]):
        raise ValueError("measurement image content/source identity mismatch")
    for directory, field in (("suite", "bundle_sha256"), ("patches-v2", "patches_sha256")):
        if not build[field]:
            raise ValueError("measurement image has no recorded " + field)
        root = image_root / directory
        for name, expected in build[field].items():
            path = root / name
            if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink()
                    or root.resolve() not in path.resolve().parents or sha256_file(path) != expected):
                raise ValueError("measurement image file identity mismatch: " + name)
    freeze = (image_root / "pip-freeze.txt").read_bytes()
    if hashlib.sha256(freeze).hexdigest() != build["pip_freeze_sha256"]:
        raise ValueError("measurement image baked dependency closure hash mismatch")
    return build, freeze


def verify_runtime(environment, build, freeze, device):
    from importlib import import_module
    from importlib.metadata import distributions
    if (sys.executable != PYTHON or sys.prefix != str(IMAGE_ROOT / "venv")
            or sys.version_info[:2] != (3, 12)
            or ".".join(map(str, sys.version_info[:3])) != build["pins"]["python"]):
        raise ValueError("only the baked Python 3.12 interpreter is allowed; no fallback")
    normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    expected = {}
    for line in freeze.decode().splitlines():
        if not line.strip():
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version or normalize(name) in expected:
            raise ValueError("unsupported or duplicate baked dependency closure entry")
        expected[normalize(name)] = version
    versions = {}
    for distribution in distributions():
        name = normalize(distribution.metadata["Name"])
        if name in versions:
            raise ValueError("duplicate installed distribution: " + name)
        versions[name] = distribution.version
    # Python 3.12 pip freeze omits pip itself, not setuptools/wheel.
    if {name: version for name, version in versions.items() if name != "pip"} != expected:
        raise ValueError("installed dependency closure differs from baked pip-freeze.txt")
    for name, expected_version in environment["capture_dependencies"].items():
        module = import_module(name)
        if versions.get(normalize(name)) != expected_version or getattr(module, "__version__", None) != expected_version:
            raise ValueError("required capture dependency mismatch: " + name)
    torch = import_module("torch")
    if torch.version.cuda != build["pins"]["torch_cuda"]:
        raise ValueError("baked torch CUDA build differs from BUILD.json")
    available = torch.cuda.is_available()
    if device not in ("cpu", "cuda") or device == "cuda" and not available:
        raise ValueError("requested CUDA capability is unavailable; no CPU fallback")
    probe = torch.ones(1, dtype=torch.float64, device=device)
    if (probe + probe).item() != 2:
        raise ValueError("requested device fp64 runtime probe failed")
    return {"python": sys.version, "interpreter": sys.executable,
            "installed_versions": dict(sorted(versions.items())),
            "torch_cuda": torch.version.cuda, "cuda_available": available, "device": device}


def inspect_runtime(device):
    environment = json.loads((CHECKOUT / "explorer/job_environment.json").read_text())
    build, freeze = verify_image(environment)
    observed = verify_runtime(environment, build, freeze, device)
    save(RUNTIME, {**observed, "image": environment["image"],
                   "build_sha256": environment["build_sha256"],
                   "image_content_sha256": build["image_content_sha256"],
                   "baked_source_revision": build["suite_revision"],
                   "pip_freeze_sha256": build["pip_freeze_sha256"]})


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


def run(command, deadline, commands, *, step):
    started = time.monotonic()
    record = {"step": step, "argv": command, "returncode": None,
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
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
        record.update(finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      duration_seconds=time.monotonic() - started)
        save(RECEIPT, {"schema": "qfs.hf-workflow-bootstrap.v1", "commands": commands})
        print(json.dumps({"stage": "bootstrap", "returncode": process.returncode}), flush=True)
    if process.returncode:
        raise RuntimeError("bootstrap command failed with exit " + str(process.returncode))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inspect-runtime", choices=("cpu", "cuda"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.plan != "/inputs/plan/plan.json" or args.out != "/outputs/result" or Path("/outputs").resolve() != Path("/outputs"):
        parser.error("bootstrap requires the fixed plan and bucket output paths")
    if args.inspect_runtime:
        inspect_runtime(args.inspect_runtime)
        return 0
    if Path(args.out).exists() or LOG.exists() or RECEIPT.exists():
        parser.error("fresh per-attempt bucket prefix required")
    started = time.time()
    plan, commands = {}, []
    try:
        if sys.executable != PYTHON or sys.prefix != str(IMAGE_ROOT / "venv") or sys.version_info[:2] != (3, 12):
            raise ValueError("worker requires the baked Python 3.12 venv; no fallback")
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
        if plan.get("launch_contract") != "measurement-venv-v1":
            raise ValueError("worker requires the sealed measurement launch contract")
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
        run(["git", "init", str(CHECKOUT)], deadline, commands, step="initialize-source")
        run(["git", "-C", str(CHECKOUT), "remote", "add", "origin", SOURCE], deadline, commands, step="configure-source")
        run(["git", "-C", str(CHECKOUT), "-c", "core.hooksPath=/dev/null", "fetch", "--depth", "1", "origin", source["revision"]], deadline, commands, step="fetch-source")
        run(["git", "-C", str(CHECKOUT), "-c", "core.hooksPath=/dev/null", "checkout", "--detach", source["revision"]], deadline, commands, step="checkout-source")
        worker = CHECKOUT / "explorer/job_worker.py"
        if hashlib.sha256(worker.read_bytes()).hexdigest() != source["worker_sha256"]:
            raise ValueError("immutable checkout worker hash mismatch")
        if (CHECKOUT / "explorer/job_bootstrap.py").read_bytes() != Path(__file__).read_bytes():
            raise ValueError("bootstrap bytes do not match the immutable source revision")
        environment_raw = (CHECKOUT / "explorer/job_environment.json").read_bytes()
        environment = json.loads(environment_raw)
        if (hashlib.sha256(environment_raw).hexdigest() != source.get("environment_sha256")
                or environment.get("image") != plan["image"]
                or environment.get("interpreter") != PYTHON
                or environment.get("launch_contract") != plan["launch_contract"]):
            raise ValueError("source-bound measurement image environment mismatch")
        # Validate the entire plan and read-only mounts before loading tensor code.
        sys.path.insert(0, str(CHECKOUT / "explorer"))
        import job_worker
        job_worker.require_no_credentials()
        job_worker.validate_plan(plan, Path(args.out))
        run([PYTHON, str(Path(__file__)), "--plan", args.plan, "--out", args.out,
             "--inspect-runtime", device], deadline, commands, step="verify-baked-runtime")
        run([PYTHON, "-m", "pip", "check"], deadline, commands, step="verify-dependency-consistency")
        observed = json.loads(RUNTIME.read_text())
        RUNTIME.unlink()
        save(RECEIPT, {"schema": "qfs.hf-workflow-bootstrap.v1", "commands": commands,
                       **observed, "source_revision": source["revision"],
                       "worker_sha256": source["worker_sha256"],
                       "environment_sha256": source["environment_sha256"]})
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
