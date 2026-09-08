#!/usr/bin/env python3
"""One-shot Lambda native qualification experiment, NEVER production admission.

plan makes read-only provider calls. run consumes that exclusive plan and may
launch one VM only with explicit acceptance of NO provider deadline. USD/time
limits are operational targets: API/network/host failures can exceed them.
The account credential is transported separately to a root-owned 0600 VM file
for a separately supervised exact-instance billing termination backstop.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import tarfile
import time

# -I -S guardian execution intentionally resolves only the frozen run snapshot.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fidelity import lambdaexperiment as core
from fidelity.lambdaapi import LambdaCloud

from fidelity.jlapi import JLError, redact
ROOT = Path(__file__).resolve().parent.parent
RETRIEVAL_RESERVE = 600
CLEANUP_RESERVE = 120
MAX_RESULT_BYTES = 2 * 1024**3
RISK = "Lambda has no provider-enforced deadline; controller/VM/API failure can exceed the operational cost/runtime ceiling. No production admission or financial reconciliation is established."
DEVICE_CONTRACTS = {
    "gpu_1x_a10": ("NVIDIA A10", r"\bA10\b", 24),
    "gpu_1x_a6000": ("NVIDIA RTX A6000", r"\bA6000\b", 48),
    "gpu_1x_h100_pcie": ("NVIDIA H100 PCIe", r"\bH100\b.*\bPCIe\b", 80),
}
EXTRA_WHEELS = (
    ("compressed-tensors", "0.18.0", "https://files.pythonhosted.org/packages/2f/bf/e17189bd834e5a4487c97ed576564a66ab56ccd7840608f49dcecf61abfc/compressed_tensors-0.18.0-py3-none-any.whl", "91168237c2d815614c44dfc354c61c613085c811dabbb2ccdb929e9aaa313b8f"),
    ("loguru", "0.7.3", "https://files.pythonhosted.org/packages/0c/29/0348de65b8cc732daa3e33e67806420b2ae89bdce2b04af740289c5c6c8c/loguru-0.7.3-py3-none-any.whl", "31a33c10c8e1e10422bfd431aeb5d351c7cf7fa671e3c4df004162264b28220c"),
    ("marisa-trie", "1.3.0", "https://files.pythonhosted.org/packages/45/cd/05bed6d02213da7f2fda63e689300b186a7f16f6b982ea19bb7284ecb1ee/marisa_trie-1.3.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl", "31c891ebce899f35936d4ab9f332b69ab762513d5944b0f43f61427e53671d42"),
    ("llguidance", "1.7.0", "https://files.pythonhosted.org/packages/ff/c5/dc74be474a274b6fc493c8811d11c8c81e4975aa238953661bfaa19d679a/llguidance-1.7.0-cp39-abi3-manylinux_2_31_x86_64.whl", "624fa6950cb14610b3ad63ada5c6cf7391ff1c008d14c7bd9b7cc2f3c0cc8568"),
)


class ExperimentFailure(RuntimeError):
    pass


class DeadlineReached(BaseException):
    pass


class Interrupted(BaseException):
    pass


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_file(path, data, mode=0o600):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data.encode() if isinstance(data, str) else data)
        stream.flush()
        os.fsync(stream.fileno())


def evidence(run_dir, name, value):
    new_file(Path(run_dir) / "evidence" / name,
             json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


@contextmanager
def bounded(deadline, maximum=None):
    remaining = deadline - time.time()
    if maximum is not None:
        remaining = min(remaining, maximum)
    if remaining <= 0:
        raise DeadlineReached()
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise DeadlineReached()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield remaining
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def local(argv, deadline, maximum=30):
    with bounded(deadline, maximum) as remaining:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=remaining, check=False)
    if result.returncode:
        raise ExperimentFailure("local supervision command failed: " + Path(argv[0]).name)
    return result.stdout.strip()


def duration(value):
    match = re.fullmatch(r"([1-9][0-9]*)([smh]?)", value)
    if not match:
        raise argparse.ArgumentTypeError("use positive seconds or an integer with s/m/h")
    return int(match[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[match[2]]


def validate_selection(plan):
    # This approved workload has a reviewed physical-device target, not a model
    # guessed from an instance-type spelling. Broader GPU selection is not admission.
    q = plan["quote"]
    device = DEVICE_CONTRACTS.get(plan["gpu_type"])
    if (device is None or q["gpu_count"] != 1 or q["architecture"] != "x86_64"
            or q["vram_gib"] < device[2] or q["storage_gib"] < 200
            or not re.search(device[1], q["gpu_description"], re.I)):
        raise ExperimentFailure("native experiment requires a reviewed single-GPU device contract")
    if plan["deadline_epoch"] - time.time() <= RETRIEVAL_RESERVE + 300:
        raise ExperimentFailure("insufficient common deadline for setup plus retrieval/cleanup reserve")


def snapshot_runtime(run_dir):
    run_dir = Path(run_dir)
    runtime = run_dir / "runtime"
    runtime.mkdir(mode=0o700)
    names = ["bin/lambda_native_experiment.py", "bin/lambda_native_remote_guard.py"]
    names += ["bin/fidelity/" + name for name in
              ("__init__.py", "common.py", "jlapi.py", "sshbase.py", "lambdaapi.py",
               "lambdaexperiment.py", "tlsguard.py", "tls-roots.pem")]
    manifest = {}
    for name in names:
        source = ROOT / name
        if source.is_symlink() or not source.is_file():
            raise ExperimentFailure("runtime source must be a regular file")
        data = source.read_bytes()
        new_file(runtime / name, data, 0o400)
        manifest[name] = hashlib.sha256(data).hexdigest()
    new_file(runtime / "manifest.json", json.dumps(manifest, sort_keys=True), 0o400)
    for directory in sorted((p for p in runtime.rglob("*") if p.is_dir()), reverse=True):
        directory.chmod(0o500)
    runtime.chmod(0o500)
    return runtime, manifest


def start_guard(run_dir, plan, runtime):
    deadline = plan["deadline_epoch"]
    linger = local(["loginctl", "show-user", str(os.getuid()), "--property=Linger", "--value"], deadline)
    if linger != "yes":
        raise ExperimentFailure("a lingering user systemd manager is required before create")
    local(["systemctl", "--user", "show-environment"], deadline)
    unit = plan["name"] + "-guardian.service"
    core.update_control(run_dir, guardian_unit=unit, runtime=str(runtime), controller_python=sys.executable)
    # No imported environment, shell wrapper, controller heartbeat or OMP lifetime
    # dependency. Restart keeps an unresolved liability under supervision.
    local(["systemd-run", "--user", "--unit=" + unit, "--property=Restart=on-failure",
           "--property=RestartSec=5", "--property=UMask=0077",
           "--property=KillMode=control-group", "--property=NoNewPrivileges=yes",
           "--property=WorkingDirectory=" + str(run_dir),
           sys.executable, "-I", "-S", str(runtime / "bin/lambda_native_experiment.py"),
           "guard", "--run-dir", str(run_dir)], deadline)
    ready_deadline = min(deadline, time.time() + 45)
    while time.time() < ready_deadline:
        state = local(["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"], ready_deadline)
        heartbeat = core.load_plan(run_dir).get("guardian", {}).get("heartbeat_epoch", 0)
        if (state == "active" and type(heartbeat) in (int, float)
                and 0 <= time.time() - heartbeat <= 15):
            evidence(run_dir, "guardian-readiness.json", {"unit": unit, "active": True,
                     "linger": True, "observed_epoch": time.time(), "heartbeat_epoch": heartbeat})
            return unit
        time.sleep(1)
    raise ExperimentFailure("guardian did not become active with a fresh durable heartbeat")


def source_bundle(run_dir, fixture_dir=None):
    paths = set()
    for line in (ROOT / "bin/BUNDLE.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.add(line)
    paths.add("engines/tools/selftest_nvfp4_offline.py")
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "engines/tools/nvfp4-evidence").iterdir() if p.is_file())
    if fixture_dir is not None:
        paths.update(("port/tests/glm5_native_forward.py", "bin/requirements-glm-native-py312.lock"))
        paths.update(("engines/tools/exl3_ordered_kda.py", "engines/tools/exl3_ordered_kda.cu",
                      "engines/tools/probe_native_kda.py"))
    bundle = Path(run_dir) / "source.tar.gz"
    manifest = {}
    total = 0
    with tarfile.open(str(bundle), "x:gz") as archive:
        for name in sorted(paths):
            relative = Path(name)
            source = ROOT / relative
            if relative.is_absolute() or ".." in relative.parts or source.resolve() != source.absolute() or not source.is_file():
                raise ExperimentFailure("unsafe or missing source bundle entry")
            data = source.read_bytes()
            total += len(data)
            if total > 512 * 1024**2:
                raise ExperimentFailure("source bundle exceeds fixed upload ceiling")
            item = tarfile.TarInfo(name)
            item.size, item.mode, item.mtime = len(data), 0o400, 0
            archive.addfile(item, io.BytesIO(data))
            manifest[name] = hashlib.sha256(data).hexdigest()
        if fixture_dir is not None:
            fixture = Path(fixture_dir).resolve()
            fixture_manifest = json.loads((fixture / "fixture-manifest.json").read_text())
            if (fixture_manifest.get("fixture_id") != "glm5-next-native-aligned128-random-bf16-v1"
                    or fixture_manifest.get("execution_complete") is not True):
                raise ExperimentFailure("complete independently identified native fixture required")
            for name in sorted(set(fixture_manifest["files"]) | {"fixture-manifest.json"}):
                relative = Path(name)
                source = fixture / relative
                if (relative.is_absolute() or ".." in relative.parts
                        or source.resolve() != source.absolute() or not source.is_file()):
                    raise ExperimentFailure("unsafe fixture artifact")
                data = source.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if name != "fixture-manifest.json":
                    expected = fixture_manifest["files"][name]
                    if len(data) != expected["size_bytes"] or digest != expected["sha256"]:
                        raise ExperimentFailure("fixture artifact differs from its manifest")
                total += len(data)
                if total > 512 * 1024**2:
                    raise ExperimentFailure("source plus fixture exceeds upload ceiling")
                archive_name = "native-fixture/" + name
                item = tarfile.TarInfo(archive_name)
                item.size, item.mode, item.mtime = len(data), 0o400, 0
                archive.addfile(item, io.BytesIO(data))
                manifest[archive_name] = digest
    bundle.chmod(0o400)
    evidence(run_dir, "source-manifest.json", {"files": manifest, "archive_sha256": digest_file(bundle)})
    return bundle


def remote_exec(provider, instance, command, deadline, maximum=300, check=True):
    with bounded(deadline, maximum) as remaining:
        return provider.exec(instance, command, timeout=max(1, remaining), check=check)


def upload(provider, instance, source, destination, deadline):
    with bounded(deadline, 300):
        return provider.upload(instance, str(source), destination)


def arm_remote(provider, instance, run_dir, plan, remote, runtime):
    deadline = plan["deadline_epoch"] - RETRIEVAL_RESERVE
    guard_root = "/var/lib/" + plan["name"]
    unit = plan["name"] + "-deadline.service"
    staging = Path(run_dir) / "remote-control"
    staging.mkdir(mode=0o700)
    binding = {k: plan[k] for k in ("provider_id", "name", "gpu_type", "region", "deadline_epoch")}
    new_file(staging / "binding.json", json.dumps(binding))
    service = ("[Unit]\nDescription=Lambda native experiment exact-instance billing backstop\n"
        "StartLimitIntervalSec=0\n[Service]\nType=simple\nUser=root\nUMask=0077\n"
        "Restart=on-failure\nRestartSec=5\nNoNewPrivileges=yes\nProtectSystem=strict\n"
        "ProtectHome=yes\nPrivateTmp=yes\nReadWritePaths=" + guard_root + "\n"
        "ExecStart=/usr/bin/python3 -I -S " + guard_root + "/guard.py " + guard_root + "\n"
        "[Install]\nWantedBy=multi-user.target\n")
    new_file(staging / "deadline.service", service)
    remote_exec(provider, instance, "umask 077; mkdir -m 700 " + shlex.quote(remote) +
                "; mkdir -m 700 " + shlex.quote(remote + "/results"), deadline)
    upload(provider, instance, runtime / "bin/lambda_native_remote_guard.py", remote + "/guard.py", deadline)
    upload(provider, instance, staging / "binding.json", remote + "/binding.json", deadline)
    upload(provider, instance, staging / "deadline.service", remote + "/deadline.service", deadline)
    trust_files = ("__init__.py", "common.py", "jlapi.py", "sshbase.py",
                   "lambdaapi.py", "tlsguard.py", "tls-roots.pem")
    for name in trust_files:
        upload(provider, instance, runtime / "bin/fidelity" / name,
               remote + "/trust-" + name, deadline)
    install_trust = " && ".join(
        "sudo -n install -m 400 " + shlex.quote(remote + "/trust-" + name) + " "
        + shlex.quote(guard_root + "/fidelity/" + name) for name in trust_files)
    # The ONLY secret transport: SCP a private file into an authenticated 0700
    # staging tree. It is moved under root authority before starting the service.
    upload(provider, instance, Path(run_dir) / "secret/api_key", remote + "/api_key", deadline)
    remote_exec(provider, instance,
        "sudo -n install -d -m 700 " + guard_root + " " + guard_root + "/fidelity && "
        + install_trust + " && "
        "sudo -n install -m 600 " + remote + "/api_key " + guard_root + "/api_key && "
        "rm -f " + remote + "/api_key && "
        "sudo -n install -m 600 " + remote + "/binding.json " + guard_root + "/binding.json && "
        "sudo -n install -m 400 " + remote + "/guard.py " + guard_root + "/guard.py && "
        "sudo -n install -m 644 " + remote + "/deadline.service /etc/systemd/system/" + unit + " && "
        "sudo -n systemctl daemon-reload && sudo -n systemctl enable --now " + unit, deadline)
    ready_deadline = min(deadline, time.time() + 90)
    last_diagnostic = None
    while time.time() < ready_deadline:
        result = remote_exec(provider, instance,
            "sudo -n systemctl is-active " + unit + "; sudo -n cat " + guard_root + "/heartbeat.json",
            ready_deadline, 20, check=False)
        if result["exit_code"] == 0:
            lines = result["stdout"].splitlines()
            try:
                heartbeat = json.loads("\n".join(lines[1:]))
                diagnostic = {"service_state": lines[0], "state": heartbeat.get("state"),
                              "error_class": heartbeat.get("error_class"),
                              "http_status": heartbeat.get("http_status"),
                              "provider_code": heartbeat.get("provider_code")}
                if diagnostic != last_diagnostic:
                    print("Lambda VM backstop: " + json.dumps(diagnostic, sort_keys=True), flush=True)
                    last_diagnostic = diagnostic
                if (lines[0] == "active" and heartbeat["state"] == "armed"
                        and heartbeat["provider_id"] == instance
                        and heartbeat["deadline_epoch"] == plan["deadline_epoch"]
                        and abs(time.time() - heartbeat["heartbeat_epoch"]) <= 15):
                    evidence(run_dir, "remote-guard-readiness.json", heartbeat)
                    return guard_root, unit
            except (ValueError, KeyError, IndexError):
                pass
        time.sleep(2)
    evidence(run_dir, "remote-guard-failure.json", last_diagnostic)
    raise ExperimentFailure("remote billing backstop not observably armed")


STACK_PROBE = r'''import hashlib, importlib.metadata as m, json, os, platform, sys
import torch
assert sys.version_info[:2] == (3,12), sys.version
assert torch.__version__ == '2.11.0+cu128', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == os.environ['QFS_NATIVE_GPU_MODEL']
x = torch.arange(16, device='cuda:0', dtype=torch.float32)
assert (x*x).sum().item() == 1240
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
rows = []
for d in sorted(m.distributions(), key=lambda d: d.metadata['Name'].lower()):
    metadata = d.read_text('METADATA') or ''
    record = d.read_text('RECORD') or ''
    rows.append({'name': d.metadata['Name'], 'version': d.version,
        'requires_dist': d.requires or [],
        'metadata_sha256': hashlib.sha256(metadata.encode()).hexdigest(),
        'record_sha256': hashlib.sha256(record.encode()).hexdigest()})
print(json.dumps({'python': platform.python_version(), 'machine': platform.machine(),
 'torch': torch.__version__, 'cuda': torch.version.cuda,
 'gpu': torch.cuda.get_device_name(0), 'packages': rows}, sort_keys=True))
'''


def workload_script(remote, plan, bundle_digest, model_mode=False, ordered_kda=False):
    q = shlex.quote
    optional = "\n".join(name + " @ " + url + " --hash=sha256:" + digest for name, version, url, digest in EXTRA_WHEELS)
    stop = int(plan["deadline_epoch"] - RETRIEVAL_RESERVE)
    lock_name = "requirements-glm-native-py312.lock" if model_mode else "requirements-cu128-py312.lock"
    ordered_option = " --ordered-kda" if ordered_kda else ""
    # GNU timeout bounds the entire child process tree with ONE absolute stop,
    # including apt, pip, network reads and both actual scientific entrypoints.
    return """#!/bin/bash
set -euo pipefail
umask 077
cd """ + q(remote) + """
export HOME=""" + q(remote) + """
export QFS_NATIVE_GPU_MODEL=""" + q(DEVICE_CONTRACTS[plan["gpu_type"]][0]) + """
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
export PYTHONNOUSERSITE=1 LOGURU_DIAGNOSE=NO
unset PYTHONPATH PYTHONOPTIMIZE QP_PIPELINE_ROOT HF_TOKEN HUGGING_FACE_HUB_TOKEN LAMBDA_API_KEY LAMBDA_KEY_FILE
remaining=$((""" + str(stop) + """ - $(date +%s)))
[ "$remaining" -gt 0 ]
if [ "${1:-}" != bounded ]; then
  set +e
  timeout --signal=TERM --kill-after=10 "$remaining" /bin/bash "$0" bounded >results/workload.log 2>&1
  code=$?
  printf '%s\\n' "$code" >results/workload.exit
  exit "$code"
fi
printf '%s  source.tar.gz\\n' """ + q(bundle_digest) + """ | sha256sum -c -
mkdir -m 700 suite
tar -xzf source.tar.gz -C suite
uname -a >results/system.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >results/device.csv
python3 - <<'PY'
import os, platform, subprocess
assert platform.machine() == 'x86_64', platform.machine()
rows = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'], text=True).strip().splitlines()
assert len(rows) == 1 and rows[0].split(',')[0].strip() == os.environ['QFS_NATIVE_GPU_MODEL'], rows
assert tuple(int(v) for v in rows[0].split(',')[1].strip().split('.')[:2]) >= (570, 26), rows
PY
if ! command -v python3.12 >/dev/null || ! python3.12 -c 'import ensurepip,sysconfig; from pathlib import Path; assert (Path(sysconfig.get_path("include"))/"Python.h").is_file()' 2>/dev/null; then
  sudo -n apt-get update
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common
  sudo -n add-apt-repository -y ppa:deadsnakes/ppa
  sudo -n apt-get update
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y python3.12 python3.12-venv python3.12-dev
fi
if [ """ + ("1" if ordered_kda else "0") + """ = 1 ]; then
  if [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
    python3 - <<'PY'
import hashlib, platform, sys
sys.path.insert(0, 'suite/bin')
release = platform.freedesktop_os_release()
if release.get('ID') != 'ubuntu' or release.get('VERSION_ID') != '22.04':
    raise RuntimeError('Pinned NVIDIA compiler repository requires Ubuntu22.04')
from fidelity.common import safe_urlopen
url = 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb'
with safe_urlopen(url, timeout=30) as response:
    data = response.read(1048576)
if hashlib.sha256(data).hexdigest() != 'd93190d50b98ad4699ff40f4f7af50f16a76dac3bb8da1eaaf366d47898ff8df':
    raise RuntimeError('NVIDIA repository keyring digest mismatch')
with open('cuda-keyring.deb', 'xb') as stream:
    stream.write(data)
PY
    sudo -n dpkg -i cuda-keyring.deb
    sudo -n apt-get update
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-nvcc-12-8=12.8.93-1 cuda-cudart-dev-12-8=12.8.90-1 cuda-cccl-12-8=12.8.90-1 build-essential
  fi
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH="$CUDA_HOME/bin:$PATH"
  export CXX=/usr/bin/g++ MAX_JOBS=4 TORCH_EXTENSIONS_DIR="$PWD/native-build"
  unset CC CUDAHOSTCXX NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS CFLAGS CXXFLAGS LDFLAGS
fi
python3.12 -c 'import sys; assert sys.version_info[:2] == (3,12)'
python3.12 -m venv venv
export PATH="$PWD/venv/bin:$PATH"
venv/bin/python -m pip install --no-input --require-hashes --no-deps --only-binary=:all: -r suite/bin/""" + lock_name + """
cat >optional.lock <<'LOCK'
""" + optional + """
LOCK
venv/bin/python -m pip install --no-input --require-hashes --no-deps --only-binary=:all: -r optional.lock
venv/bin/python - <<'PY' >results/stack-before.json
""" + STACK_PROBE + """PY
set +e
if [ """ + ("1" if model_mode else "0") + """ = 0 ]; then
venv/bin/python suite/engines/tools/exl3_decoder_parity_vs_exllamav3.py --install --out "$PWD/results/native-v2.json" --cache-dir "$PWD/results/payload-cache" --device cuda:0 >results/native.log 2>&1
native=$?
printf '%s\\n' "$native" >results/native.exit
venv/bin/python suite/engines/tools/exl3_decoder_parity_vs_exllamav3.py --install --out "$PWD/results/native-repeat-v2.json" --cache-dir "$PWD/results/payload-cache" --device cuda:0 >results/native-repeat.log 2>&1
repeat=$?
printf '%s\\n' "$repeat" >results/native-repeat.exit
venv/bin/python - <<'PY' >results/native-repeat-check.log 2>&1
import json, os
from pathlib import Path
docs = [json.loads(Path('results/' + name).read_text()) for name in ('native-v2.json', 'native-repeat-v2.json')]
signatures = []
for doc in docs:
    if doc['observation'] != 'complete' or doc['reference_implementation_verified'] is not True:
        raise ValueError('native repeat requires complete verified observations')
    if not doc['modules'] or doc['modules_compared'] != len(doc['modules']):
        raise ValueError('native repeat has incomplete module coverage')
    signature = {k: doc[k] for k in ('producer_sha256', 'torch_version', 'cuda_version', 'device_name', 'expected_reference_commit', 'ours')}
    signature['modules'] = []
    for row in doc['modules']:
        cases = row['native_module_forward']['cases']
        comparisons = [row['pre_hadamard'], row['weight_fp16_primary']] + [case['primary'] for case in cases]
        if not cases or any(comparison['valid'] is not True for comparison in comparisons):
            raise ValueError('nonfinite, empty or invalid native repeat')
        signature['modules'].append({**{k: row[k] for k in ('label','repo','revision','shard','name','codebook','K','shape_in_out','input_sha256','exllamav3')},
            'forward': [{k: case[k] for k in ('rows','input_sha256','native_sha256','dispatch_from_pinned_predicates')} for case in cases]})
    signatures.append(signature)
report = {'schema': 'fidelity.native-repeat-observation.v1', 'complete': True,
    'deterministic': signatures[0] == signatures[1], 'modules': len(docs[0]['modules']),
    'forward_cases': sum(len(row['native_module_forward']['cases']) for row in docs[0]['modules']),
    'native_export_bitwise': all(row['native_export_diagnostic']['comparison']['equal'] is True for doc in docs for row in doc['modules']),
    'historical_primary_qualification': [doc['qualification'] for doc in docs],
    'whole_model_qualified': False, 'cache_qualified': False}
temporary = Path('results/.native-repeat-check.tmp')
temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
os.replace(temporary, 'results/native-repeat-check.json')
print(json.dumps(report, sort_keys=True))
raise SystemExit(0 if report['deterministic'] and report['native_export_bitwise'] else 1)
PY
repeat_check=$?
venv/bin/python suite/engines/tools/selftest_nvfp4_offline.py >results/nvfp4.log 2>&1
nvfp4=$?
printf '%s\\n' "$nvfp4" >results/nvfp4.exit
venv/bin/python - <<'PY' >results/stack-after.json
""" + STACK_PROBE + """PY
stack=$?
[ "$native" = 0 ] && [ "$repeat" = 0 ] && [ "$repeat_check" = 0 ] && [ "$nvfp4" = 0 ] && [ "$stack" = 0 ]
else
if [ """ + ("1" if ordered_kda else "0") + """ = 1 ]; then
  venv/bin/python suite/engines/tools/probe_native_kda.py --out "$PWD/results/kda-isolation.json" --ordered >results/kda-isolation.log 2>&1
  isolation=$?
  printf '%s\\n' "$isolation" >results/kda-isolation.exit
  if [ "$isolation" != 0 ]; then exit "$isolation"; fi
fi
venv/bin/python suite/port/tests/glm5_native_forward.py --model-dir "$PWD/suite/native-fixture" --out "$PWD/results/model-first.json" --device cuda:0""" + ordered_option + """ >results/model-first.log 2>&1
model_first=$?
venv/bin/python suite/port/tests/glm5_native_forward.py --model-dir "$PWD/suite/native-fixture" --out "$PWD/results/model-second.json" --device cuda:0""" + ordered_option + """ >results/model-second.log 2>&1
model_second=$?
venv/bin/python - <<'PY' >results/model-cold-repeat.log 2>&1
import hashlib, json, os
from pathlib import Path
import torch
from safetensors.torch import load_file
docs = [json.loads(Path('results/model-' + name + '.json').read_text()) for name in ('first','second')]
if any(doc['execution_complete'] is not True for doc in docs):
    raise ValueError('full-model execution incomplete; no cold qualification')
if any(doc['runtime'] != docs[0]['runtime']
       or doc['runner_sha256'] != docs[0]['runner_sha256']
       or doc['fixture']['manifest_sha256'] != docs[0]['fixture']['manifest_sha256'] for doc in docs[1:]):
    raise ValueError('cold observations have different runtime or fixture identities')
signatures = []
for name in ('first','second'):
    tensors = load_file('results/model-' + name + '.native.safetensors')
    if not tensors or any(not bool(torch.isfinite(t).all()) for t in tensors.values()):
        raise ValueError('missing/nonfinite full-model tensors')
    signatures.append({key: {'shape': list(t.shape), 'dtype': str(t.dtype),
        'sha256': hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        for key,t in tensors.items()})
report = {'schema':'qfs.glm5-native-cold-repeat.v1','execution_complete':True,
    'cold_process_byte_equal': signatures[0] == signatures[1],
    'same_path_reset_byte_equal': all(doc['observed_same_path_byte_repeatable'] for doc in docs),
    'reference_value_equal': all(doc['observed_reference_value_equality_all_comparisons'] for doc in docs),
    'tensor_count':len(signatures[0]),'whole_model_reference_equivalence_qualified':False,
    'original_published_fixture_qualified':False}
report['native_determinism_qualified'] = report['cold_process_byte_equal'] and report['same_path_reset_byte_equal']
report['qualification_scope'] = 'this new fixture, exact recorded runtime and observed paths only'
temporary = Path('results/.model-cold-repeat.tmp')
temporary.write_text(json.dumps(report,indent=2,allow_nan=False))
os.replace(temporary,'results/model-cold-repeat.json')
print(json.dumps(report,sort_keys=True))
PY
model_comparison=$?
venv/bin/python - <<'PY' >results/stack-after.json
""" + STACK_PROBE + """PY
stack=$?
[ "$model_first" = 0 ] && [ "$model_second" = 0 ] && [ "$model_comparison" = 0 ] && [ "$stack" = 0 ]
fi
"""


def retrieve(provider, instance, run_dir, remote, plan, guard_root, guard_unit, model_mode=False):
    deadline = plan["deadline_epoch"] - CLEANUP_RESERVE
    # Logs are retrieved even if the producer exits 1 (numeric failure). No
    # cache or native evidence is synthesized when execution failed earlier.
    extra = ""
    if guard_root:
        extra = ("sudo -n cat " + guard_root + "/heartbeat.json >results/remote-guard.json; "
                 "sudo -n journalctl -u " + guard_unit + " --no-pager >results/remote-guard.log; ")
    command = ("cd " + shlex.quote(remote) + "; " + extra +
        "tar -czf result.tar.gz results && python3 -c " + shlex.quote(
        "import hashlib,json,os; p='result.tar.gz'; h=hashlib.sha256(); "
        "f=open(p,'rb'); "
        "\nfor b in iter(lambda:f.read(1048576),b''): h.update(b)\n"
        "print(json.dumps({'bytes':os.path.getsize(p),'sha256':h.hexdigest()}))"))
    result = remote_exec(provider, instance, command, deadline, maximum=240)
    transfer = json.loads(result["stdout"])
    size = transfer["bytes"]
    if type(size) is not int or not 0 < size <= MAX_RESULT_BYTES or not re.fullmatch(r"[0-9a-f]{64}", transfer["sha256"]):
        raise ExperimentFailure("result archive transfer identity invalid")
    destination = Path(run_dir) / "result.tar.gz"
    with bounded(deadline) as remaining:
        provider.download_bounded(instance, remote + "/result.tar.gz", str(destination),
                                  expected_bytes=size, max_bytes=MAX_RESULT_BYTES, timeout=remaining)
    if digest_file(destination) != transfer["sha256"]:
        raise ExperimentFailure("downloaded native result archive digest mismatch")
    evidence(run_dir, "retrieval.json", transfer)
    # No untrusted tar extraction. Copy only regular members under results,
    # bound expanded bytes and refuse aliases/links/duplicate paths.
    seen, expanded = set(), 0
    with bounded(deadline), tarfile.open(str(destination), "r:gz") as archive:
        for item in archive:
            path = Path(item.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "results":
                raise ExperimentFailure("unsafe result archive path")
            if item.isdir():
                continue
            if not item.isfile() or item.name in seen:
                raise ExperimentFailure("unsafe result archive member")
            seen.add(item.name)
            expanded += item.size
            if expanded > MAX_RESULT_BYTES:
                raise ExperimentFailure("expanded result archive exceeds ceiling")
            target = Path(run_dir) / path
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as output, archive.extractfile(item) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
    if model_mode:
        documents = [Path(run_dir) / "results" / ("model-" + name + ".json") for name in ("first", "second")]
        if not all(path.is_file() for path in documents):
            return False
        reports = [json.loads(path.read_text()) for path in documents]
        return all(report.get("schema") == "qfs.glm5-native-forward.v1"
                   and report.get("execution_complete") is True for report in reports)
    observation = Path(run_dir) / "results/native-v2.json"
    if observation.is_file():
        native = json.loads(observation.read_text())
        if native.get("schema") != "malaiwah.exl3-decoder-parity-vs-exllamav3.v2":
            raise ExperimentFailure("native output has unexpected schema")
        return native.get("qualification") == "qualified"
    return False


def await_host(provider, instance, plan, run_dir):
    """Observe startup in short steps; never hide provisioning behind a long wait."""
    started = time.time()
    deadline = min(plan["deadline_epoch"] - RETRIEVAL_RESERVE, started + 600)
    observations = []
    next_notice = 0
    previous_state = None
    try:
        while time.time() < deadline:
            with bounded(deadline, 30):
                current = provider.get(instance)
            status = current.status if current is not None else "not_listed"
            ip = current.raw.get("ip") if current is not None else None
            if current is not None and (
                    current.machine_id != instance or current.name != plan["name"]
                    or current.gpu_type != plan["gpu_type"] or current.region != plan["region"]):
                raise ExperimentFailure("startup resource identity differs from this attempt")
            state = (status, bool(ip))
            if state != previous_state or time.time() >= next_notice:
                item = {"elapsed_seconds": round(time.time() - started, 1),
                        "provider_status": status, "ip_assigned": bool(ip)}
                observations.append(item)
                print("Lambda startup: " + json.dumps(item, sort_keys=True), flush=True)
                previous_state, next_notice = state, time.time() + 15
            if status not in ("booting", "active", "not_listed"):
                raise ExperimentFailure("Lambda startup refused provider status: " + str(status))
            if status == "active" and ip:
                provider._ep[str(instance)] = (ip, 22)
                try:
                    with bounded(deadline, 10):
                        provider._await_ssh(ip, 22, wait=5)
                except JLError:
                    pass
                else:
                    with bounded(deadline, 30):
                        pin = provider.ssh_host_ed25519_fingerprint(instance)
                        provider.set_known_hosts(Path(run_dir) / "known_hosts")
                        host = provider.verify_host_key(instance, pin["fingerprint"])
                    return pin, host
            time.sleep(min(5, max(0, deadline - time.time())))
        raise ExperimentFailure("Lambda startup did not reach authenticated SSH within 600 seconds")
    finally:
        evidence(run_dir, "startup-observations.json", observations)


def run_experiment(args):
    if not args.accept_no_provider_deadline:
        # This flag acknowledges provider risk, not numerical equivalence.
        raise ExperimentFailure("run requires --accept-no-provider-deadline")
    run_dir = Path(args.run_dir).absolute()
    plan = core.load_plan(run_dir)
    validate_selection(plan)
    if plan["phase"] != "PLANNED":
        raise ExperimentFailure("one attempt only: run cannot resume or relaunch a spent plan")
    model_mode = args.fixture_dir is not None
    if args.ordered_kda and not model_mode:
        raise ExperimentFailure("--ordered-kda requires the full-model fixture workload")
    provider = core.provider_from_control(run_dir)
    instance = None
    remote = "/home/ubuntu/" + plan["name"]
    unit = guard_root = guard_unit = None
    retrieved = False
    retrieval_attempted = False
    succeeded = False
    failed_stage = "snapshot"
    error_class = None
    error_message = None
    remote_started = False
    previous = {}
    def interrupt(signum, frame):
        raise Interrupted()
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[number] = signal.signal(number, interrupt)
    try:
        with bounded(plan["deadline_epoch"] - RETRIEVAL_RESERVE, 180):
            runtime, manifest = snapshot_runtime(run_dir)
            evidence(run_dir, "runtime-manifest.json", manifest)
            bundle = source_bundle(run_dir, args.fixture_dir)
        failed_stage = "guardian-readiness"
        unit = start_guard(run_dir, plan, runtime)
        print("Independent Lambda guardian ready; validating the sole launch", flush=True)
        failed_stage = "create"
        with bounded(plan["deadline_epoch"] - RETRIEVAL_RESERVE, 180):
            plan = core.create(provider, run_dir)
        instance = plan["provider_id"]
        print("Lambda resource created and durably bound: " + instance, flush=True)
        failed_stage = "host-authentication"
        pin, host = await_host(provider, instance, plan, run_dir)
        evidence(run_dir, "host-authentication.json", {"pin": pin, "verified": host})
        failed_stage = "resource-binding"
        q = plan["quote"]
        with bounded(plan["deadline_epoch"] - RETRIEVAL_RESERVE, 120):
            binding = provider.validate_safe_resource_binding(instance,
                expected_name=plan["name"], instance_type_name=plan["gpu_type"],
                region_name=plan["region"], ssh_key_names=plan["ssh_key_names"],
                storage_gib=q["storage_gib"], gpu_count=1,
                terminate_after=plan["terminate_after"], file_system_names=())
        if binding["observed"]["price_cents_per_hour"] != q["price_cents_per_hour"]:
            raise ExperimentFailure("live launch rate differs from frozen quote")
        evidence(run_dir, "resource-binding.json", {"passed": binding["passed"],
            "provider_id": instance, "price_cents_per_hour": q["price_cents_per_hour"],
            "expected": binding["expected"], "provider_enforced_deadline": False})
        failed_stage = "device-attestation"
        with bounded(plan["deadline_epoch"] - RETRIEVAL_RESERVE, 360):
            attestation = provider.attest_live_resource(instance,
                expected_gpu_model=DEVICE_CONTRACTS[plan["gpu_type"]][0],
                expected_vram_bytes=q["vram_gib"] * 1024**3,
                min_vcpu=q["vcpus"], min_ram_gb=q["memory_gib"], storage_gib=q["storage_gib"],
                root_available_bytes_minimum=50 * 1024**3,
                run_root_available_bytes_minimum=50 * 1024**3,
                expected_gpu_count=1, expected_ssh_key_names=plan["ssh_key_names"])
        evidence(run_dir, "device-attestation.json", attestation)
        if not attestation["ok"]:
            raise ExperimentFailure("live device/resource attestation failed")
        failed_stage = "remote-backstop"
        guard_root, guard_unit = arm_remote(provider, instance, run_dir, plan, remote, runtime)
        remote_started = True
        failed_stage = "native-workload"
        print("Authenticated VM backstop armed; starting fixed native workload", flush=True)
        upload(provider, instance, bundle, remote + "/source.tar.gz", plan["deadline_epoch"] - RETRIEVAL_RESERVE)
        script = Path(run_dir) / "workload.sh"
        new_file(script, workload_script(remote, plan, digest_file(bundle), model_mode, args.ordered_kda), 0o400)
        evidence(run_dir, "workload-provenance.json", {"script_sha256": digest_file(script),
                 "optional_wheels": [{"name": n, "version": v, "url": u, "sha256": d} for n,v,u,d in EXTRA_WHEELS],
                 "not_qualified": ["whole-model", "cache", "GLM architecture"], "risk": RISK})
        upload(provider, instance, script, remote + "/workload.sh", plan["deadline_epoch"] - RETRIEVAL_RESERVE)
        result = remote_exec(provider, instance, "/bin/bash " + remote + "/workload.sh",
            plan["deadline_epoch"] - RETRIEVAL_RESERVE + 15,
            maximum=plan["max_runtime_seconds"], check=False)
        evidence(run_dir, "workload-exit.json", {"exit_code": result["exit_code"]})
        failed_stage = "retrieval"
        print("Native workload stopped; retrieving observations before cleanup", flush=True)
        retrieval_attempted = True
        qualified = retrieve(provider, instance, run_dir, remote, plan, guard_root, guard_unit, model_mode)
        retrieved = True
        succeeded = result["exit_code"] == 0 and qualified
        if not succeeded:
            failed_stage = ("full-model-execution" if model_mode else "historical-native-parity") if not qualified else "workload-validation"
    except (Exception, DeadlineReached, Interrupted) as exc:
        error_class = type(exc).__name__
        error_message = redact(str(exc))[:2000]
        print("Lambda stage failed: " + failed_stage + ": " + error_class
              + (": " + error_message if error_message else ""), flush=True)
    finally:
        # Repeated signals cannot cut off the durable cleanup request. The
        # independent guardians remain responsible after this bounded teardown.
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        if remote_started and not retrieval_attempted and time.time() < plan["deadline_epoch"] - CLEANUP_RESERVE:
            try:
                retrieval_attempted = True
                retrieve(provider, instance, run_dir, remote, plan, guard_root, guard_unit, model_mode)
                retrieved = True
            except (Exception, DeadlineReached) as exc:
                evidence(run_dir, "retrieval-failure.json", {"error_class": type(exc).__name__})
        try:
            with bounded(plan["deadline_epoch"], 15):
                core.request_cleanup(run_dir, "workload_finished" if succeeded else "workload_failed")
        except (Exception, DeadlineReached):
            # Expiry remains independently enforced by both guardians.
            pass
        # Erase the upload staging copy only. Root backstop credential MUST
        # survive until exact absence: removing it earlier disables its purpose.
        if remote_started and time.time() < plan["deadline_epoch"]:
            try:
                remote_exec(provider, instance, "rm -f " + remote + "/api_key",
                            plan["deadline_epoch"], 10, check=False)
            except (Exception, DeadlineReached):
                pass
        closed = False
        try:
            with bounded(plan["deadline_epoch"], CLEANUP_RESERVE):
                while not closed:
                    closed = core.cleanup_once(provider, run_dir)
                    if not closed:
                        time.sleep(2)
        except (Exception, DeadlineReached):
            pass
        if closed and unit:
            # Only confirmed no-liability state permits stopping the guardian.
            try:
                local(["systemctl", "--user", "stop", unit], plan["deadline_epoch"], 10)
            except (Exception, DeadlineReached):
                pass
        if closed:
            # Status/closed plans need no authority. Never remove the user's
            # original credential, only this run's now-unneeded snapshot.
            (run_dir / "secret/api_key").unlink(missing_ok=True)
        evidence(run_dir, "runner-outcome.json", {"native_qualified": succeeded and not model_mode,
            "full_model_execution_complete": succeeded if model_mode else None,
            "results_retrieved": retrieved, "failed_stage": None if succeeded else failed_stage,
            "error_class": error_class, "error_message": error_message, "cleanup_closed": closed,
            "guardian_preserved_for_liability": not closed, "risk": RISK})
        for number, handler in previous.items():
            signal.signal(number, handler)
    print(json.dumps(core.public_report(run_dir), sort_keys=True, indent=2))
    return 0 if succeeded and closed else 1


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="read-only account preflight; NEVER creates a VM")
    plan.add_argument("--run-dir", required=True)
    plan.add_argument("--gpu-type", default="gpu_1x_a6000")
    plan.add_argument("--region", default="us-south-2")
    plan.add_argument("--max-cost", default="25")
    plan.add_argument("--max-runtime", type=duration, default=14400)
    plan.add_argument("--ssh-key", required=True)
    plan.add_argument("--ssh-key-name", action="append", required=True)
    plan.add_argument("--key-file", help="local private credential file; never a key value")
    plan.add_argument("--accept-no-provider-deadline", action="store_true")
    run = commands.add_parser("run", help="one native attempt from an existing plan")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--accept-no-provider-deadline", action="store_true")
    run.add_argument("--fixture-dir", type=Path, help="NEW native-compatible fixture; selects complete GLM5-next text/cache observation")
    run.add_argument("--ordered-kda", action="store_true", help="Isolate stock recurrence and use explicit ordered CUDA backend for the full-model run")
    for command in ("guard", "status"):
        sub = commands.add_parser(command, help="internal guardian" if command == "guard" else "allowlisted local state; no provider calls")
        sub.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            provider = LambdaCloud(key_file=args.key_file, ssh_key=args.ssh_key, ssh_key_names=args.ssh_key_name)
            core.prepare(provider, args.run_dir, gpu_type=args.gpu_type, region=args.region,
                max_cost_usd=args.max_cost, max_runtime_seconds=args.max_runtime,
                ssh_key_names=args.ssh_key_name, accept_risk=args.accept_no_provider_deadline)
            print(json.dumps(core.public_report(args.run_dir), sort_keys=True, indent=2))
            return 0
        if args.command == "run":
            return run_experiment(args)
        if args.command == "guard":
            return core.guard(args.run_dir)
        print(json.dumps(core.public_report(args.run_dir), sort_keys=True, indent=2))
        return 0
    except (Exception, DeadlineReached, Interrupted) as exc:
        # Provider/remote exceptions may contain provider documents or tokens.
        # Only our explicitly authored refusals are safe to show verbatim.
        detail = str(exc) if isinstance(exc, (ExperimentFailure, core.ExperimentError)) else type(exc).__name__
        print("Lambda native experiment failed: " + detail, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
