"""The single chokepoint for every JarvisLabs call.

WHY THE CLI AND NOT THE REST API.  `jl` owns auth, region selection, the
spot-vs-container rules, ssh-key plumbing and upload/download/exec over SSH.
The REST surface behind it is not publicly documented -- the vendor documents
the CLI -- so reimplementing it would make this recipe a maintenance liability,
which is the opposite of "a standard anyone can run".

The cost of that choice is CLI drift.  It is paid down here rather than spread
through the runner: EVERY invocation goes through `JL._call`, which appends
`--json`, parses stdout, and normalises the vendor's `{"error": ...}` shape
into an exception.  A future `--transport api` is a one-function swap, and a
`jl` version bump has exactly one place to break.

Install:  uv tool install jarvislabs        (or pipx install jarvislabs)
Auth:     jl setup --token <token> --yes    (or export JL_API_KEY=...)
"""

from __future__ import annotations

import base64
import calendar
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .common import CommandError, redact, register_secret, run

MIN_VERSION = (0, 2, 17)


class JLError(RuntimeError):
    pass


class JLNotInstalled(JLError):
    pass


class JLUnsupportedByCli(JLError):
    """The `jl` CLI genuinely exposes no way to answer this.

    Distinct from `JLError` on purpose: a caller may record a named provider
    gap and carry on, but must never treat it as a transient failure to retry
    or -- worse -- substitute a local approximation.  Every raise site names
    the subcommands that were probed and what would supply the answer.
    """


class JLCreateResponseError(JLError):
    """A create committed with an exact id but returned unqualified metadata."""

    def __init__(self, message: str, provider_id: str,
                 response: Dict[str, Any]) -> None:
        super().__init__(message)
        self.provider_id = provider_id
        self.response = dict(response)


class JLBillingUnreconcilable(JLError):
    """Cost cannot be closed, and the evidence gathered says exactly why.

    Raised rather than returned because a partial cost is not a closure: the
    caller must record billing as pending, never seal a receipt over it.  The
    `evidence` payload carries every provider figure that WAS obtainable so
    the operator sees a number and its residual gap instead of nothing.
    """

    def __init__(self, message: str, evidence: Dict[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


def _parse_version(text: str) -> tuple:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(g) for g in m.groups()) if m else (0, 0, 0)


@dataclass
class Instance:
    machine_id: Any      # EXACTLY what the provider returned -- see from_json
    status: str
    gpu_type: Optional[str]
    num_gpus: int
    region: Optional[str]
    is_spot: bool
    cost: float          # RUNNING USD TOTAL, not a rate -- see `billed_usd`
    runtime: Any
    fs_id: Optional[int]
    storage_gb: Optional[int]
    name: Optional[str]
    raw: Dict[str, Any]

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Instance":
        """Parse one provider row, keeping the id EXACTLY as it arrived.

        `int(d["machine_id"])` used to stand here.  JarvisLabs happens to
        number its machines (483634), and treating that as a universal truth
        is the first of this project's three portability bugs: `int(pod_id)`
        raised AFTER the resource existed, so the controller died holding an
        instance it had never adopted.  The cast has no upside even here --
        every consumer either compares the id or passes it back as argv, and
        both want the provider's own spelling -- while its downside is a
        billing leak.  So the id is carried opaquely and only ever compared as
        a string (see `_jl_id`).
        """
        machine_id = d.get("machine_id", 0)
        return cls(
            machine_id=machine_id,
            status=str(d.get("status", "")),
            gpu_type=d.get("gpu_type"),
            num_gpus=int(d.get("num_gpus") or 0),
            region=d.get("region"),
            is_spot=bool(d.get("is_spot")),
            cost=float(d.get("cost") or 0.0),
            runtime=d.get("runtime"),
            fs_id=d.get("fs_id"),
            storage_gb=d.get("storage_gb"),
            name=d.get("name"),
            raw=d,
        )

    @property
    def billed_usd(self) -> float:
        """The accumulated dollar total for this instance so far.

        NOTE, and this contradicts an earlier note in engines/HANDOFF.md: `jl get`'s
        `cost` field is a running TOTAL in USD, not an hourly rate.  Verified
        by reconciling live instances against the published rates -- e.g. an
        8x H200 spot box at 2h33m reported 40.897, which is $16.04/h against a
        list rate of 8 x $1.99 = $15.92/h.  A cost model built on the "rate"
        reading would be wrong by a factor of the elapsed hours.
        """
        return self.cost


@dataclass
class GpuOffer:
    gpu_type: str
    region: Optional[str]
    vram_bytes: float
    price: float
    spot: bool
    free_devices: int
    workload_type: Optional[str]
    raw: Dict[str, Any]


# Subcommands that actually accept --yes, verified against `jl <cmd> --help`
# on jl 0.2.17.  Keyed by the leading argv words.
_YES_OK = frozenset({
    ("create",), ("destroy",), ("pause",), ("resume",),
    ("filesystem", "create"), ("filesystem", "remove"),
})


def _takes_yes(argv: Sequence[str]) -> bool:
    head = tuple(a for a in argv[:2] if not a.startswith("-"))
    return head[:2] in _YES_OK or head[:1] in _YES_OK


# `jl create` names the GPU model `--gpu`, not `--gpu-type`; the keyword the
# controller passes is `gpu_type` because that is what `jl gpus --json` calls
# the field.  Mapping is explicit so the mismatch cannot silently return.
_CREATE_FLAG_ALIASES = {"gpu_type": "gpu"}

# The one path a JarvisLabs run writes to: the separable filesystem's mount
# point.  `measure_cloud.Teardown` exports it as FIDELITY_FS_ROOT's parent,
# and the attestation proves it is a DIFFERENT device from the container's
# own disk -- otherwise "100 GB of separable storage" is really the instance
# disk and the capture dies of ENOSPC after the bootstrap is paid for.
JL_WORKSPACE = "/home/jl_fs"

# `jl create --help` (0.2.17) documents exactly these three region pins, and
# every row of the live `jl gpus --json` sits in one of them.  A create with
# no region lets the provider choose, and a receipt then cannot say where the
# number was made -- Hub throughput differed 10x between datacenters.
JL_REGIONS = ("IN1", "IN2", "EU1")

# The safe profile refuses a create whose deadline is too close to be useful,
# for the same reason RunPod does: a box that dies during bootstrap has been
# paid for and produced nothing.
MIN_CREATE_SETUP_SECONDS = 300

# The startup script that makes a JarvisLabs host key KNOWN IN ADVANCE.
#
# Why this exists, from the vendor's own source rather than its docs:
# `jarvislabs/ssh.py` HARDENING_OPTIONS (lines 22-30 of the installed 0.2.17)
# appends `UserKnownHostsFile=/dev/null` and `StrictHostKeyChecking=no` to
# every ssh it builds, and `jarvislabs/cli/instance.py` drives exec, upload,
# download and ssh through `subprocess.call` on exactly those options. So the
# vendor CLI authenticates NO host, ever, and remembers nothing between
# calls: not trust-on-first-use, no trust at all. There is also no boot log
# to read a fingerprint out of, so RunPod's log-anchored approach is
# unavailable here.
#
# Pinning removes the trust instead of tolerating it. The key is generated
# locally, installed by this script through `jl create --script-id`, and the
# expected fingerprint is frozen into the create's request identity BEFORE
# the instance exists -- so only a box that received our authenticated launch
# request can present it. It does NOT make `jl exec` safe: a verifying
# transport of our own must do the checking (see
# `ssh_host_ed25519_fingerprint`), which is why the request identity records
# `channel_verifies_host_key: False`.
#
# `set -eu` and no `set -x`: this script holds a private key, and a traced
# startup script would print it into provider-side logs.
_HOST_KEY_PIN_SCRIPT = """#!/bin/sh
set -eu
umask 077
key=/etc/ssh/ssh_host_ed25519_key
cat > "$key" <<'FIDELITY_PINNED_HOST_KEY'
__PRIVATE_KEY__
FIDELITY_PINNED_HOST_KEY
printf '%s\\n' '__PUBLIC_KEY__' > "$key.pub"
chmod 600 "$key"
chmod 644 "$key.pub"
# Restart sshd if it is already up, so the pinned key is the one served; if
# it has not started yet the key is simply in place before it does.
if command -v systemctl >/dev/null 2>&1; then
    systemctl restart sshd 2>/dev/null || systemctl restart ssh 2>/dev/null || true
else
    kill -HUP "$(cat /var/run/sshd.pid 2>/dev/null || echo 0)" 2>/dev/null || true
fi
"""

# `jl list --json` spells the live states in Title case ("Running", "Paused").
# RunPod says "RUNNING", Lambda says "active", Vast says "running".  Nothing
# may depend on the spelling: comparing "Running" == "RUNNING" is the second
# of this project's three portability bugs, and it made every healthy poll
# count as not-running, so the controller declared a PREEMPTION and tore down
# a box mid-bootstrap.  Folded, never spelled.
_RUNNING_WORDS = frozenset({"running", "ready", "active"})

# Ids come back from `jl --json` as JSON numbers today, and `jl get --help`
# types the argument `<int>`.  Neither fact is allowed to leak into a
# comparison: an id is a string here, always, and `_jl_id` is the only way to
# obtain one.  A digit-string and an int must normalise to the SAME string,
# because "ids compare as ints in a set" is the third portability bug -- the
# "is it really gone?" check reported a live instance as destroyed.
_JL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _jl_id(value: Any, field: str = "JarvisLabs machine id") -> str:
    """Normalise a provider id to its exact string form, or refuse.

    Accepts the JSON integer the CLI emits and the string form a lease stores,
    and maps both to one canonical string.  Never calls `int()`: a provider
    that renumbers to opaque ids must fail this validation up front rather
    than raise from the middle of a teardown.
    """
    if isinstance(value, bool) or value is None:
        raise JLError("%s must be an exact provider id, got %r" % (field, value))
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise JLError("%s must be an exact provider id, got a %s"
                      % (field, type(value).__name__))
    if _JL_ID_RE.fullmatch(text) is None:
        raise JLError("%s has invalid characters or length" % field)
    return text


def _is_running_status(status: Any) -> bool:
    """Case-folded liveness, so no caller ever compares the spelling."""
    return str(status or "").strip().casefold() in _RUNNING_WORDS


def _exact_utc(value: Any, field: str) -> str:
    text = str(value)
    if (len(text) != 20 or not text.endswith("Z")
            or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text) is None):
        raise JLError("%s must be exact UTC YYYY-MM-DDTHH:MM:SSZ, got %r"
                      % (field, value))
    try:
        calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        raise JLError("%s is not a real UTC instant: %r" % (field, value)) from None
    return text


def _utc_epoch(value: str) -> int:
    return calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))


def _finite_decimal(value: Any, field: str, *, nonnegative: bool = True) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise JLError("%s is not an exact decimal: %r" % (field, value)) from None
    if not parsed.is_finite() or (nonnegative and parsed < 0):
        raise JLError("%s is not a finite non-negative decimal: %r"
                      % (field, value))
    return parsed


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise JLError("%s must be a positive integer" % field)
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise JLError("%s must be a positive integer, got %r" % (field, value))
    if parsed <= 0:
        raise JLError("%s must be a positive integer, got %r" % (field, value))
    return parsed


def _canonical_bytes(document: Any) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _seal(document: Dict[str, Any], key: str) -> Dict[str, Any]:
    out = dict(document)
    out[key] = hashlib.sha256(_canonical_bytes(out)).hexdigest()
    return out


# Read-only device attestation, run on the box itself.  Deliberately the same
# observation set and key names as the RunPod attestation, so a JarvisLabs
# receipt and a RunPod receipt carry comparable device evidence -- provider is
# not a comparability axis, the DEVICE MODEL and the rebuilt stack are, and
# this is the read that makes the device term provable rather than asserted.
#
# Two deliberate differences from RunPod's copy: the workspace mount point is
# JarvisLabs' separable filesystem rather than /workspace, and every GPU the
# box reports is checked (RunPod's copy demands exactly one) because
# `jl create --num-gpus` routinely rents 8.
_LIVE_ATTEST_SCRIPT = r'''
import json
import os
import subprocess
import sys
import time

WORKSPACE = "__JL_WORKSPACE__"

def mount(path):
    resolved = os.path.realpath(path)
    stats = os.statvfs(resolved)
    best = None
    with open("/proc/self/mountinfo", "r", encoding="utf-8") as stream:
        for line in stream:
            left, right = line.rstrip("\n").split(" - ", 1)
            fields = left.split()
            point = fields[4].replace("\\040", " ")
            if resolved == point or resolved.startswith(point.rstrip("/") + "/"):
                if best is None or len(point) > len(best[0]):
                    tail = right.split()
                    best = (point, tail[0], tail[1])
    if best is None:
        raise RuntimeError("mountpoint not found for " + resolved)
    return {
        "path": resolved, "mount_point": best[0], "fs_type": best[1],
        "source": best[2], "device": int(os.stat(resolved).st_dev),
        "total_bytes": int(stats.f_blocks * stats.f_frsize),
        "available_bytes": int(stats.f_bavail * stats.f_frsize),
    }

mem_kib = None
with open("/proc/meminfo", "r", encoding="ascii") as stream:
    for line in stream:
        if line.startswith("MemTotal:"):
            mem_kib = int(line.split()[1])
            break
if mem_kib is None:
    raise RuntimeError("MemTotal missing")
limits = [mem_kib * 1024]
for path in ("/sys/fs/cgroup/memory.max",
             "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    try:
        value = open(path, "r", encoding="ascii").read().strip()
        if value != "max":
            parsed = int(value)
            if 0 < parsed < (1 << 60):
                limits.append(parsed)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
# TOTAL VRAM is not the attestable quantity; FREE is. A rented "24 GB" 4090
# was observed with 23,424 of 24,564 MiB already held by four foreign PIDs
# (host 434175, 2026-09-06) -- "24 GB card" was true and useless, and an
# attestation that compares expected VRAM against total passes an
# oversubscribed card. So free and used are read alongside total, and every
# compute process on the device is enumerated: a foreign PID means we are
# sharing silicon, which changes both the memory available and the timings.
smi = subprocess.run(
    ["nvidia-smi",
     "--query-gpu=index,name,memory.total,memory.free,memory.used,"
     "driver_version",
     "--format=csv,noheader,nounits"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
gpus = []
if smi.returncode == 0:
    for line in smi.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 6:
            gpus.append({
                "index": int(fields[0]), "name": fields[1],
                "vram_bytes": int(fields[2]) * 1024 * 1024,
                "vram_free_bytes": int(fields[3]) * 1024 * 1024,
                "vram_used_bytes": int(fields[4]) * 1024 * 1024,
                "driver_version": fields[5],
            })
apps = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
     "--format=csv,noheader,nounits"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
compute_processes = []
if apps.returncode == 0:
    for line in apps.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 3 and fields[0].isdigit():
            compute_processes.append({
                "pid": int(fields[0]),
                "used_bytes": (int(fields[1]) * 1024 * 1024
                               if fields[1].isdigit() else None),
                "gpu_uuid": fields[2],
            })
cuda = {"usable": False, "count": 0, "name": None,
        "vram_bytes": None, "error": None, "interpreter": None}
CUDA_PROBE = (
    "import json, torch\n"
    "out = {'usable': bool(torch.cuda.is_available()),"
    " 'count': int(torch.cuda.device_count()), 'name': None,"
    " 'vram_bytes': None, 'error': None}\n"
    "if out['usable'] and out['count']:\n"
    "    probe = torch.empty(1, device='cuda'); torch.cuda.synchronize(); del probe\n"
    "    props = torch.cuda.get_device_properties(0)\n"
    "    out['name'] = str(props.name); out['vram_bytes'] = int(props.total_memory)\n"
    "print(json.dumps(out))\n")
candidates = [sys.executable]
image_root = os.environ.get("FIDELITY_IMAGE_ROOT", "")
if image_root and os.path.isfile(os.path.join(image_root, "venv", "bin", "python")):
    candidates.append(os.path.join(image_root, "venv", "bin", "python"))
errors = []
for interpreter in candidates:
    try:
        run = subprocess.run([interpreter, "-c", CUDA_PROBE], capture_output=True,
                             text=True, timeout=240)
    except Exception as exc:
        errors.append("%s: %s: %s" % (interpreter, type(exc).__name__, str(exc)[:200]))
        continue
    if run.returncode != 0:
        tail = (run.stderr or "").strip().splitlines()[-1:] or ["exit %d" % run.returncode]
        errors.append("%s: %s" % (interpreter, tail[0][:300]))
        continue
    try:
        cuda = json.loads(run.stdout.strip().splitlines()[-1])
        cuda["interpreter"] = interpreter
    except Exception as exc:
        errors.append("%s: probe output unreadable: %s" % (interpreter, str(exc)[:200]))
        continue
    break
else:
    cuda["error"] = "; ".join(errors)[:300]
remote_time_epoch = int(time.time())
remote_time_utc = time.strftime(
    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(remote_time_epoch))
print(json.dumps({
    "remote_time_epoch": remote_time_epoch,
    "remote_time_utc": remote_time_utc,
    "logical_cpus": len(os.sched_getaffinity(0)),
    "memtotal_bytes": mem_kib * 1024,
    "effective_memory_bytes": min(limits),
    "nvidia_smi_exit_code": smi.returncode,
    "nvidia_smi_error": smi.stderr[:300],
    "gpus": gpus, "cuda": cuda,
    "compute_processes": compute_processes,
    "compute_apps_exit_code": apps.returncode,
    "filesystems": {"container": mount("/"), "workspace": mount(WORKSPACE)},
}, sort_keys=True, separators=(",", ":"), allow_nan=False))
'''.replace("__JL_WORKSPACE__", JL_WORKSPACE)


@dataclass(frozen=True)
class PreparedJLCreate:
    """A create frozen before any provider mutation.

    On a REST provider the frozen thing is an HTTP request.  Here it is the
    exact argv `jl` will be executed with, plus the identity that argv encodes
    and a digest of both -- and, critically, an intent record the CALLER
    persists before submitting.  What makes a lost create response
    reconcilable on a CLI transport is the same thing that makes it
    reconcilable on GraphQL: the request pins an exact, unique instance NAME,
    so a crash between the two halves is answered by listing the account and
    looking for that name.  Nothing about the create is decided after the
    freeze, so the intent record and the mutation cannot disagree.
    """

    argv: Tuple[str, ...]
    request_identity_json: bytes
    name: str
    terminate_after: str
    storage_gb: int
    container_disk_gb: int
    template: str
    dry_run: bool
    # Present only when the create pinned a generated host key. The PUBLIC
    # half and its fingerprint are all that is kept: verification needs
    # nothing else, and not holding the private key is one fewer secret.
    host_key_fingerprint_sha256: Optional[str] = None
    host_key_public: Optional[str] = None
    host_key_script_id: Optional[str] = None
    host_key_script_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        identity = json.loads(self.request_identity_json.decode("utf-8"))
        argv_bytes = _canonical_bytes(list(self.argv))
        return {
            "schema": "fidelity-suite/jarvislabs-prepared-create.v1",
            "request_identity": identity,
            # The API token reaches `jl` through its config file or JL_API_KEY,
            # never argv (verified against `jl create --help`, which has no
            # token flag), so the argv is safe to persist in a lease and to
            # print.  `redact` is still applied: a future flag must not be
            # able to turn this record into a credential leak, and the host-key
            # pinning path deliberately passes a script ID rather than the key.
            "argv": [redact(item) for item in self.argv],
            "argv_sha256": hashlib.sha256(argv_bytes).hexdigest(),
        }


class JL:
    """Thin, auditable wrapper.  `dry` short-circuits every mutating call."""

    # JarvisLabs filesystems outlive their instance, which is why the
    # controller creates a small box and attaches the big fs to it.
    separable_storage = True
    provider = "jarvislabs"

    def __init__(self, *, dry: bool = False, binary: str = "jl",
                 timeout: float = 300.0) -> None:
        self.binary = binary
        self.dry = dry
        self.timeout = timeout
        self._version: Optional[tuple] = None
        register_secret(os.environ.get("JL_API_KEY"))

    # ---- plumbing ---------------------------------------------------------

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def require(self) -> tuple:
        if not self.available():
            raise JLNotInstalled(
                "the `jl` CLI is not on PATH.\n"
                "  install:  uv tool install jarvislabs\n"
                "            (or: pipx install jarvislabs)\n"
                "  auth:     jl setup --token <your-token> --yes\n"
                "            (or: export JL_API_KEY=...)"
            )
        if self._version is None:
            proc = run([self.binary, "--version"], timeout=30, check=False)
            self._version = _parse_version(proc.stdout or proc.stderr)
        if self._version < MIN_VERSION:
            raise JLError(
                "jl %s is older than the pinned minimum %s; upgrade with "
                "`uv tool upgrade jarvislabs`"
                % (".".join(map(str, self._version)), ".".join(map(str, MIN_VERSION)))
            )
        return self._version

    @property
    def version(self) -> str:
        return ".".join(map(str, self._version or (0, 0, 0)))

    def _call(self, argv: Sequence[str], *, mutating: bool = False,
              timeout: Optional[float] = None, check: bool = True) -> Any:
        """Every jl invocation lands here.  Nothing else may shell out to jl."""
        if mutating and self.dry:
            return {"dry_run": True, "argv": list(argv)}
        cmd = [self.binary] + list(argv)
        # Everything after a bare `--` belongs to the REMOTE command, so jl's
        # own flags have to go in front of it.  Appending `--json` blindly
        # hands it to the remote shell instead of to jl.
        cut = cmd.index("--") if "--" in cmd else len(cmd)
        flags = []
        if "--json" not in cmd[:cut]:
            flags.append("--json")
        # Not every mutating jl subcommand takes --yes.  `upload`, `download`
        # and `exec` move data rather than money, so jl never prompts for them
        # and rejects the flag outright ("No such option: --yes"), which would
        # make every bundle upload fail AFTER the instance is already billing.
        # Verified against jl 0.2.17 --help for each subcommand.
        if mutating and "--yes" not in cmd[:cut] and _takes_yes(argv):
            flags.append("--yes")
        cmd = cmd[:cut] + flags + cmd[cut:]
        try:
            proc = run(cmd, timeout=timeout or self.timeout, check=False)
        except Exception as exc:                      # noqa: BLE001
            raise JLError("jl invocation failed: %s" % redact(str(exc))) from None
        out = (proc.stdout or "").strip()
        if proc.returncode != 0 and not out:
            raise JLError(
                "jl %s exited %d: %s"
                % (" ".join(argv[:2]), proc.returncode, redact(proc.stderr or "")[:400])
            )
        try:
            data = json.loads(out) if out else {}
        except json.JSONDecodeError:
            if check and proc.returncode != 0:
                raise JLError(
                    "jl %s exited %d with non-JSON output: %s"
                    % (" ".join(argv[:2]), proc.returncode, redact(out)[:400])
                ) from None
            return out
        if isinstance(data, dict) and data.get("error"):
            raise JLError("jl %s: %s" % (" ".join(argv[:2]), redact(str(data["error"]))))
        return data

    # ---- read-only --------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return self._call(["status"])

    def balance(self) -> Optional[float]:
        try:
            data = self.status()
        except JLError:
            return None
        bal = data.get("balance")
        if isinstance(bal, dict):
            bal = bal.get("balance")
        try:
            return float(bal)
        except (TypeError, ValueError):
            return None

    def list_instances(self) -> List[Instance]:
        """Every instance on the account.

        MUST NOT answer "none" when it does not know.  Four call sites read an empty
        list as a FACT about the account, and each one spends or leaks money on it:

          * `measure_cloud.reaper_sweep` retires every lease whose machine is "gone" --
            the last-resort backstop then never looks at those boxes again;
          * the adopt loop takes "no instance for this job" as licence to CREATE one,
            which is the double-spend its own comment says it exists to prevent;
          * `_find_by_name` is last-resort id recovery for an instance that is ALREADY
            billing, and gives up;
          * the name-deadline sweep silently degrades to leases only.

        `_call` returns `{}` for an empty body on a zero exit, and returns a parsed
        object unchanged when the exit code is non-zero as long as the JSON carries no
        `error` key.  The old `data.get("instances", [])` fallback turned all three of
        those into "the account is empty".  Reproduced against a stub `jl`: an empty
        body, an unrecognised envelope (`{"data": [...]}`), and `exit 2` with
        `{"detail": "authentication failed"}` each retired a live lease.

        The real `jl 0.2.17 list --json` answers with a top-level JSON array, so the
        `{"instances": [...]}` shape is a speculative fallback that has never run; it is
        kept, and everything else now raises rather than reporting an empty account.
        """
        data = self._call(["list"])
        if isinstance(data, list):
            rows: List[Any] = data
        elif isinstance(data, dict) and isinstance(data.get("instances"), list):
            rows = data["instances"]
        else:
            raise JLError(
                "`jl list --json` (jl %s) answered with %s, which is not an instance "
                "list. Refusing to report an empty account: callers treat that as proof "
                "that an instance is gone, and act on it by retiring leases, creating a "
                "second box, or giving up on recovering the id of one that is billing."
                % (self.version,
                   "an empty body" if not data
                   else "an object with keys %r" % (sorted(data)[:8],)
                   if isinstance(data, dict) else "a %s" % type(data).__name__))
        bad = [r for r in rows if not isinstance(r, dict)]
        if bad:
            raise JLError(
                "`jl list --json` returned %d row(s) that are not objects (first: %r); "
                "this client cannot tell which instances are alive from that."
                % (len(bad), redact(str(bad[0]))[:120]))
        return [Instance.from_json(r) for r in rows]

    def get(self, machine_id: int) -> Optional[Instance]:
        try:
            data = self._call(["get", str(machine_id)])
        except JLError:
            return None
        if isinstance(data, list):
            data = data[0] if data else {}
        return Instance.from_json(data) if data else None

    def gpus(self) -> List[GpuOffer]:
        data = self._call(["gpus"])
        rows = data if isinstance(data, list) else data.get("gpus", [])
        offers: List[GpuOffer] = []
        for r in rows or []:
            vram = r.get("vram") or r.get("gpu_ram") or 0
            spot_price = r.get("spot_price")
            free = r.get("num_free_devices")
            if free is None:
                free = r.get("effective_num_free_devices") or 0
            base = {
                "gpu_type": r.get("gpu_type") or r.get("name") or "?",
                "region": r.get("region"),
                "vram_bytes": float(vram) * 1e9,
                "free_devices": int(free or 0),
                "workload_type": r.get("workload_type"),
                "raw": r,
            }
            # `jl 0.2.17 gpus --json` spells the on-demand rate
            # **price_per_hour**, and neither of the two names this used to
            # look for exists in that payload. The effect was total and silent:
            # every JarvisLabs on-demand row produced NO offer, so
            # `select_offer(..., spot=False)` had nothing to choose from and
            # `--on-demand` refused "no available instance fits this lane" on
            # the one provider this suite treats as its reference. Rows with no
            # spot price at all -- the whole EU1 region -- were invisible in
            # both modes. Proven against the live account: 12 offers came back,
            # every one of them spot, and the $3.99 H200 in the printed table
            # was in none of them.
            on_demand = (r.get("price_per_hour") if r.get("price_per_hour")
                         is not None else
                         (r.get("price") if r.get("price") is not None
                          else r.get("on_demand_price")))
            if on_demand is not None:
                offers.append(GpuOffer(price=float(on_demand), spot=False, **base))
            if spot_price is not None:
                offers.append(GpuOffer(price=float(spot_price), spot=True, **base))
        return offers

    # ---- mutating ---------------------------------------------------------

    def create(self, **kw) -> Dict[str, Any]:
        """Historical single-shot create: maps kwargs onto `jl create` flags.

        Kept because `fidelity-bench` and the historical container controller
        call it with loose kwargs and no deadline, and neither is a paid
        measurement.  A PAID create must go through
        `prepare_safe_create` + `submit_prepared_create` instead: this path
        applies none of the safe profile, freezes nothing, and so leaves a
        lost response ambiguous rather than reconcilable.
        """
        argv = ["create"]
        for key, value in kw.items():
            if value is None or value is False:
                continue
            flag = "--" + _CREATE_FLAG_ALIASES.get(key, key).replace("_", "-")
            argv.append(flag) if value is True else argv.extend([flag, str(value)])
        return self._call(argv, mutating=True, timeout=900)

    def destroy(self, machine_id: int) -> Dict[str, Any]:
        return self._call(["destroy", str(machine_id)], mutating=True, timeout=600)

    def pause(self, machine_id: int) -> Dict[str, Any]:
        return self._call(["pause", str(machine_id)], mutating=True, timeout=600)

    def resume(self, machine_id: int, *, spot: bool = False) -> Dict[str, Any]:
        argv = ["resume", str(machine_id)]
        if spot:
            argv.append("--spot")
        return self._call(argv, mutating=True, timeout=900)

    def exec(self, machine_id: int, command: str, *,
             timeout: float = 600, check: bool = True) -> Any:
        """Run a shell command on the instance and CHECK that it worked.

        Two things here are load-bearing.

        `jl exec <id> "<string>"` does not run a shell: jl execs the whole
        string as one program name, so anything with a pipe, a redirect or an
        argument comes back `sh: 1: ...: not found` with exit 127.  The remote
        command has to be passed after a bare `--` as real argv, with the shell
        named explicitly.

        And `jl exec` reports remote failure INSIDE its JSON payload
        (`exit_code`), not through its own process exit status.  Without this
        check every remote command -- mkdir, chmod, the watchdog, the stage
        runs, shredding the token -- fails silently and the controller reports
        each one as ok.
        """
        res = self._call(["exec", str(machine_id), "--", "sh", "-lc", command],
                         mutating=True, timeout=timeout)
        if check and isinstance(res, dict) and res.get("exit_code") not in (0, None):
            raise JLError(
                "remote command exited %s: %s"
                % (res.get("exit_code"),
                   redact((str(res.get("stderr") or res.get("stdout") or ""))[:400])))
        return res

    def exec_stdout(self, machine_id: int, command: str, *,
                    timeout: float = 600, check: bool = True) -> str:
        """Just the remote STDOUT, as a string.

        `jl exec --json` answers {machine_id, command, exit_code, stdout,
        stderr}, and `command` echoes the command back.  Any caller that
        stringifies the whole dict and looks for a token is really searching
        its OWN command text: the probe

            test -f <marker> && echo DONE || echo PENDING

        matched "DONE" on every single call, so every stage of every run was
        declared finished at the first 120-second poll -- setup "completed" in
        2m07s with an empty receipts directory, and the controller marched on
        to fetch, measure and seal against a box that had done nothing.
        Reading stdout, and only stdout, is the whole fix.
        """
        res = self.exec(machine_id, command, timeout=timeout, check=check)
        if isinstance(res, dict):
            return str(res.get("stdout") or "")
        return str(res or "")

    def upload(self, machine_id: int, local: str, remote: str) -> Any:
        return self._call(["upload", str(machine_id), local, remote],
                          mutating=True, timeout=1800)

    def download(self, machine_id: int, remote: str, local: str,
                 *, recursive: bool = True, timeout: float = 900) -> Any:
        argv = ["download", str(machine_id), remote, local]
        if recursive:
            argv.append("-r")
        return self._call(argv, mutating=True, timeout=timeout)

    def run_job(self, machine_id: int, command: str) -> Any:
        """Start a managed run of a SHELL COMMAND on an existing instance.

        `jl run` takes a TARGET first -- a .py/.sh file to upload, or a
        directory -- and a bare command only after `--`:

            jl run <command-string> --on <id>   -> "Target does not exist: ..."
            jl run --on <id> -- sh -lc '<cmd>'  -> runs it

        The first spelling is what this method used to emit, so EVERY stage of
        every cloud run died the moment it was launched, with the instance
        already created and billing. `--yes` keeps it non-interactive and
        `--no-follow` keeps the CLI from attaching (the controller polls
        `jl run status` / `jl run logs` itself, and an attached CLI can pause
        or destroy the instance when the run ends).
        """
        return self._call(["run", "--on", str(machine_id), "--yes", "--no-follow",
                           "--", "sh", "-lc", command],
                          mutating=True, timeout=600)

    def run_status(self, run_id: str) -> Any:
        """The managed run's own STATE and exit code.

        `jl run status --json` answers {state, exit_code, instance_status, ...}
        with state in {running, succeeded, failed, ...}.  Deciding a stage's
        fate from log TEXT instead -- greping the tail for "Traceback" -- lets a
        stage that exits non-zero quietly look identical to a stage still
        working, and the controller then waits for it until --max-runtime.
        """
        return self._call(["run", "status", str(run_id)], timeout=120, check=False)

    def run_logs(self, run_id: str, *, tail: int = 50) -> Any:
        return self._call(["run", "logs", str(run_id), "--tail", str(tail)],
                          timeout=120, check=False)

    def fs_create(self, *, storage: int, region: str, name: Optional[str] = None) -> Any:
        argv = ["filesystem", "create", "--storage", str(storage), "--region", region]
        if name:
            argv.extend(["--name", name])
        return self._call(argv, mutating=True, timeout=600)

    def fs_delete(self, fs_id: int) -> Any:
        # `jl filesystem` exposes list/create/edit/remove -- there is no
        # `delete`.  Getting this wrong leaks a multi-hundred-GB filesystem
        # that keeps billing after the instance is gone, and the failure is
        # swallowed by Teardown's per-step guard, so it is silent.
        return self._call(["filesystem", "remove", str(fs_id)],
                          mutating=True, timeout=600)

    def fs_list(self) -> Any:
        return self._call(["filesystem", "list"])

    # ---- the provider contract: is this the thing I asked for? ------------

    def list_lifecycle_resources(self) -> List[Dict[str, Any]]:
        """Complete exact-id rows; every row is a resource that still exists.

        `jl list` enumerates only instances the account still holds, so being
        listed IS liveness -- including a Paused box, which stops GPU billing
        but keeps its disk image and its storage, and therefore keeps
        charging.  `listed` is True on every row for that reason, and
        `running` is reported separately and case-folded so no caller has to
        know that this provider spells it "Running".

        `list_instances` refuses rather than reporting an empty account when
        it cannot tell, and this inherits that: a caller may treat an empty
        list as proof that nothing is alive.
        """
        rows = []
        for inst in self.list_instances():
            resource_id = _jl_id(inst.machine_id)
            raw = inst.raw or {}
            rows.append({
                "id": resource_id,
                "name": inst.name,
                "status": inst.status,
                "running": _is_running_status(inst.status),
                "listed": True,
                # NOT a rate. `cost` is the running USD total for this
                # instance; see Instance.billed_usd for the reconciliation
                # that established it. Formatted as an exact decimal string so
                # no float ever reaches a receipt.
                "cost_usd_total": format(
                    _finite_decimal(inst.cost, "JarvisLabs instance cost"), "f"),
                "billing_frequency": raw.get("billing_frequency"),
                "gpu_type": inst.gpu_type,
                "gpu_count": inst.num_gpus,
                "region": inst.region,
                "is_spot": inst.is_spot,
                "runtime": inst.runtime,
                # The separable filesystem, if one is attached: a chargeable
                # resource with its OWN id that outlives this instance.
                "filesystem_id": (None if raw.get("fs_id") in (None, "")
                                  else _jl_id(raw.get("fs_id"),
                                              "JarvisLabs filesystem id")),
                "container_disk_gb": inst.storage_gb,
                "template": raw.get("template"),
                "framework_id": raw.get("framework_id"),
                "disk_type": raw.get("disk_type"),
                "public_ip": raw.get("public_ip"),
                "vpc_id": raw.get("vpc_id"),
                "paused_image_size": raw.get("paused_image_size"),
                "raw": raw,
            })
        return rows

    def get_lifecycle_resource(self, provider_id: Any) -> Optional[Dict[str, Any]]:
        """Exact-id detail; a NAME is refused rather than resolved.

        Resolved off the complete listing rather than `jl get`, because the
        answer to "is my exact id still there" must come from the same
        enumeration that proves absence, and because `jl get --help` types its
        argument `<int>` -- handing it a name would make the CLI, not this
        code, decide what the id was.

        A value that matches no id but DOES match a listed name raises: that
        is a caller bug worth surfacing loudly, since acting on a
        name-resolved id is how a teardown aims at the wrong machine.
        """
        wanted = _jl_id(provider_id)
        rows = self.list_lifecycle_resources()
        for row in rows:
            if row["id"] == wanted:
                return row
        named = [row["id"] for row in rows
                 if str(row.get("name") or "") == wanted]
        if named:
            raise JLError(
                "%r is the NAME of JarvisLabs instance %s, not an id; ids are "
                "exact provider ids and are never resolved from names. Pass %s."
                % (wanted, ", ".join(named), named[0]))
        return None

    def validate_safe_resource_binding(
            self, provider_id: Any, *, expected_name: str,
            gpu_type_id: str, secure_cloud: bool, gpu_count: int,
            volume_gb: int, container_disk_gb: int, image_name: str,
            terminate_after: str) -> Dict[str, Any]:
        """Fail unless the live exact-id instance is the one requested.

        Signature is RunPod's so the controller needs no per-provider branch;
        three of its arguments mean something different here and are mapped
        rather than ignored:

        * `secure_cloud` -- JarvisLabs has no secure/community split at all,
          only region pins.  True is REFUSED instead of silently accepted,
          because accepting it would let a caller believe it had asked for an
          isolation property this provider never offered.
        * `image_name` -- containers are launched from a named template
          (`--template pytorch`), not an image reference, so this is checked
          against the instance's `template`/`framework_id`.  A docker
          reference is refused with the flag that does exist.
        * `volume_gb` / `container_disk_gb` -- storage here is two resources,
          not one: the instance's own disk (`storage_gb`, sized by
          `--storage`) and the separable filesystem (`--fs-id`), which
          outlives the instance.  `container_disk_gb` binds the former and
          `volume_gb` the latter, and a `volume_gb` above the container disk
          with no filesystem attached is refused -- that combination is
          exactly the "No space left on device after paid setup" failure.
        """
        wanted = _jl_id(provider_id)
        observed = self.get_lifecycle_resource(wanted)
        if observed is None:
            raise JLError(
                "created JarvisLabs id %s is absent from the complete listing; "
                "if the create response was lost, reconcile by the exact name "
                "%r against `jl list` before creating anything else"
                % (wanted, expected_name))
        if not isinstance(secure_cloud, bool):
            raise JLError("secure_cloud expectation must be an exact bool")
        if secure_cloud:
            raise JLError(
                "JarvisLabs exposes no secure-cloud/community split; pass "
                "secure_cloud=False and pin the datacenter with region=%s"
                % "|".join(JL_REGIONS))
        expected_deadline = _exact_utc(terminate_after, "terminate_after")
        expected_template = str(image_name).strip()
        if not expected_template:
            raise JLError("expected template must be nonempty")
        if "/" in expected_template or "@sha256:" in expected_template:
            raise JLError(
                "%r looks like a container image reference; JarvisLabs "
                "containers take `jl create --template <name>` (see `jl "
                "templates`), and only VMs created with --vm run an image"
                % expected_template)
        expected = {
            "name": str(expected_name),
            "gpu_type": str(gpu_type_id),
            "gpu_count": _positive_int(gpu_count, "gpu_count"),
            "container_disk_gb": _positive_int(
                container_disk_gb, "container_disk_gb"),
            "template": expected_template,
        }
        problems = []
        for key, value in expected.items():
            actual = observed.get(key)
            if key in ("gpu_count", "container_disk_gb"):
                try:
                    actual = int(actual)
                except (TypeError, ValueError):
                    pass
                if not isinstance(actual, int) or actual < value:
                    problems.append("%s expected at least %r, observed %r"
                                    % (key, value, observed.get(key)))
                continue
            if key == "template":
                if str(actual or "").strip().casefold() != value.casefold() and \
                        str(observed.get("framework_id") or "").strip().casefold() \
                        != value.casefold():
                    problems.append(
                        "template expected %r, observed %r/%r"
                        % (value, observed.get("template"),
                           observed.get("framework_id")))
                continue
            if str(actual or "") != value:
                problems.append("%s expected %r, observed %r"
                                % (key, value, observed.get(key)))
        wanted_volume = _positive_int(volume_gb, "volume_gb")
        filesystem_id = observed.get("filesystem_id")
        if filesystem_id is None:
            container_disk = observed.get("container_disk_gb")
            try:
                container_disk = int(container_disk)
            except (TypeError, ValueError):
                container_disk = -1
            if wanted_volume > container_disk:
                problems.append(
                    "volume_gb %d exceeds the instance disk %r and no "
                    "separable filesystem is attached (pass --fs-id)"
                    % (wanted_volume, observed.get("container_disk_gb")))
        else:
            sizes = {vol["id"]: vol["size_gb"]
                     for vol in self.list_network_volumes()}
            if filesystem_id not in sizes:
                problems.append(
                    "attached filesystem %s is absent from `jl filesystem "
                    "list`" % filesystem_id)
            elif sizes[filesystem_id] < wanted_volume:
                problems.append(
                    "filesystem %s is %r GB, below the requested volume_gb %d"
                    % (filesystem_id, sizes[filesystem_id], wanted_volume))
        # The running total, not a rate: validated as a finite non-negative
        # decimal and nothing more. Validating it as a positive per-hour price
        # -- which is what the RunPod binding does with costPerHr -- would
        # reject a freshly created instance that has not accrued a cent yet.
        cost_total = format(
            _finite_decimal(observed.get("cost_usd_total"),
                            "JarvisLabs live instance cost total"), "f")
        if problems:
            raise JLError("JarvisLabs post-create identity mismatch: %s"
                          % "; ".join(problems))
        return {
            "provider": self.provider,
            "provider_id": wanted,
            "passed": True,
            "expected": dict(expected, volume_gb=wanted_volume,
                             secure_cloud=False,
                             terminate_after=expected_deadline),
            "observed": dict(observed, cost_usd_total=cost_total),
            # `jl create --help` has no deadline flag: JarvisLabs accepts no
            # provider-side termination time at all. The deadline is therefore
            # enforced ONLY by the local reaper, and saying so here keeps a
            # receipt from implying a provider guarantee that does not exist.
            "terminate_after_observable": False,
            "terminate_after_enforced_by": "local reaper only",
            "secure_cloud_supported": False,
        }

    def attest_live_resource(
            self, provider_id: Any, *, expected_gpu_model: str,
            expected_vram_bytes: int, min_vcpu: int, min_ram_gb: int,
            volume_gb: int, container_disk_gb: int,
            workspace_available_bytes_minimum: int,
            container_available_bytes_minimum: int) -> Dict[str, Any]:
        """Read-only proof that this box is the DEVICE the root was captured on.

        This is the scientific gate, not a safety check.  Provider is not a
        comparability axis -- two A100s in two clouds agreed bitwise while an
        H200 sat 2.973e-04 nats away, an order of magnitude above the effect
        between adjacent bit-widths -- so what a comparison binds is the GPU
        MODEL and the rebuilt stack.  Without this read, a JarvisLabs number
        cannot be placed beside a root captured elsewhere at all: the device
        term would be an assertion from a provider catalogue string
        ("RTX-PRO6000", "A100-80GB") rather than a reading from the silicon.

        Transport is `jl exec`, which is the only channel this provider gives
        us; the probe is base64'd because `jl exec` hands its string to a
        shell.  Everything it does is a read.  A transport failure is recorded
        in the document, never hidden, and `ok` is false unless every check
        passed -- there is no partial pass.
        """
        wanted = _jl_id(provider_id)
        model = str(expected_gpu_model).strip()
        if not model:
            raise JLError("expected GPU model must be nonempty")
        expected_numbers = {
            "expected_vram_bytes": expected_vram_bytes,
            "min_vcpu": min_vcpu,
            "min_ram_gb": min_ram_gb,
            "volume_gb": volume_gb,
            "container_disk_gb": container_disk_gb,
            "workspace_available_bytes_minimum":
                workspace_available_bytes_minimum,
            "container_available_bytes_minimum":
                container_available_bytes_minimum,
        }
        for key, value in expected_numbers.items():
            _positive_int(value, key)
        expected = dict(expected_numbers, gpu_model=model,
                        workspace_mount_point=JL_WORKSPACE)
        command = (
            "python3 -c 'import base64;"
            "exec(base64.b64decode(\"%s\").decode(\"utf-8\"))'"
            % base64.b64encode(
                _LIVE_ATTEST_SCRIPT.encode("utf-8")).decode("ascii"))
        observed = None
        transport_error = None
        send_epoch = time.time()
        receive_epoch = send_epoch
        if self.dry:
            transport_error = "dry mode cannot attest a live resource"
            receive_epoch = time.time()
        else:
            try:
                raw = self.exec_stdout(wanted, command, timeout=300)
                observed = json.loads(raw.strip().splitlines()[-1])
                if not isinstance(observed, dict):
                    raise JLError("attestation payload is not an object")
            except Exception as exc:                          # noqa: BLE001
                observed = None
                transport_error = redact(str(exc))[:500]
            finally:
                receive_epoch = time.time()
        round_trip_seconds = max(0.0, receive_epoch - send_epoch)
        remote_epoch = (observed.get("remote_time_epoch")
                        if isinstance(observed, dict) else None)
        remote_utc = (observed.get("remote_time_utc")
                      if isinstance(observed, dict) else None)
        remote_utc_epoch = None
        if isinstance(remote_utc, str):
            try:
                remote_utc_epoch = _utc_epoch(
                    _exact_utc(remote_utc, "remote attestation time"))
            except JLError:
                pass
        midpoint_epoch = send_epoch + round_trip_seconds / 2.0
        allowed_skew_seconds = 30.0 + round_trip_seconds
        clock_skew_seconds = (
            abs(float(remote_epoch) - midpoint_epoch)
            if isinstance(remote_epoch, int)
            and not isinstance(remote_epoch, bool) else None)
        clock_ok = bool(
            clock_skew_seconds is not None
            and remote_utc_epoch == remote_epoch
            and clock_skew_seconds <= allowed_skew_seconds)
        clock = {
            # Both ends of this comparison are the INSTANCE's clock against
            # OURS. JarvisLabs publishes no server clock (see
            # server_time_evidence), so this is the only clock cross-check the
            # provider makes possible, and it is labelled for what it is
            # rather than presented as the provider's time.
            "reference": "controller clock vs instance clock",
            "provider_clock_available": False,
            "controller_send_epoch": send_epoch,
            "controller_send_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(send_epoch)),
            "controller_receive_epoch": receive_epoch,
            "controller_receive_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(receive_epoch)),
            "round_trip_seconds": round_trip_seconds,
            "remote_time_epoch": remote_epoch,
            "remote_time_utc": remote_utc,
            "clock_skew_seconds": clock_skew_seconds,
            "allowed_skew_seconds": allowed_skew_seconds,
            "within_bound": clock_ok,
        }
        failures = []
        checks: Dict[str, bool] = {"remote_clock": clock_ok}
        if not isinstance(observed, dict):
            failures.append("live attestation unavailable")
        else:
            exact_observed = {
                "remote_time_epoch", "remote_time_utc",
                "logical_cpus", "memtotal_bytes", "effective_memory_bytes",
                "nvidia_smi_exit_code", "nvidia_smi_error", "gpus", "cuda",
                "compute_processes", "compute_apps_exit_code",
                "filesystems",
            }
            if set(observed) != exact_observed:
                failures.append("live attestation keys differ")
            for key in ("logical_cpus", "memtotal_bytes",
                        "effective_memory_bytes", "nvidia_smi_exit_code"):
                if (isinstance(observed.get(key), bool)
                        or not isinstance(observed.get(key), int)):
                    failures.append("%s is not an exact integer" % key)
            checks["logical_cpu_floor"] = (
                isinstance(observed.get("logical_cpus"), int)
                and not isinstance(observed.get("logical_cpus"), bool)
                and observed["logical_cpus"] >= min_vcpu)
            checks["memory_floor"] = (
                isinstance(observed.get("effective_memory_bytes"), int)
                and not isinstance(observed.get("effective_memory_bytes"), bool)
                and observed["effective_memory_bytes"] >= min_ram_gb * 10 ** 9)
            gpus = observed.get("gpus")
            # `--num-gpus 8` is ordinary here, so every card is checked
            # instead of demanding exactly one. A box that answers with a
            # mixed bag, or with a wrong model in slot 7, fails.
            checks["nvidia_gpus_present"] = (
                observed.get("nvidia_smi_exit_code") == 0
                and isinstance(gpus, list) and len(gpus) >= 1)
            rows = gpus if checks["nvidia_gpus_present"] else []
            for row in rows:
                if not isinstance(row, dict) or set(row) != {
                        "index", "name", "vram_bytes", "vram_free_bytes",
                        "vram_used_bytes", "driver_version"}:
                    failures.append("nvidia-smi GPU keys differ")
                    rows = []
                    break
            vram_floor = expected_vram_bytes * 9 // 10
            vram_ceiling = expected_vram_bytes * 11 // 10
            checks["gpu_model"] = bool(rows) and all(
                isinstance(row.get("name"), str)
                and row["name"].strip().casefold() == model.casefold()
                for row in rows)
            checks["gpu_vram"] = bool(rows) and all(
                isinstance(row.get("vram_bytes"), int)
                and not isinstance(row.get("vram_bytes"), bool)
                and vram_floor <= row["vram_bytes"] <= vram_ceiling
                for row in rows)
            # The card must actually HAVE the memory, not merely be a model
            # that ships with it. A "24 GB" 4090 with 23,424 of 24,564 MiB
            # held by foreign PIDs satisfies gpu_vram and cannot hold the
            # weights; the run would OOM after the bootstrap is paid for.
            checks["gpu_vram_free"] = bool(rows) and all(
                isinstance(row.get("vram_free_bytes"), int)
                and not isinstance(row.get("vram_free_bytes"), bool)
                and row["vram_free_bytes"] >= vram_floor
                for row in rows)
            # A foreign compute process means we are sharing silicon, which
            # changes both the memory available and every timing measured on
            # it. The probe itself holds no CUDA context (the torch check runs
            # in a separate process that exits), so this list must be empty.
            processes = observed.get("compute_processes")
            if not isinstance(processes, list) or any(
                    not isinstance(item, dict)
                    or set(item) != {"pid", "used_bytes", "gpu_uuid"}
                    for item in processes):
                failures.append("compute-process attestation keys differ")
                processes = None
            checks["no_foreign_compute_processes"] = (
                observed.get("compute_apps_exit_code") == 0
                and isinstance(processes, list) and not processes)
            cuda = observed.get("cuda")
            if not isinstance(cuda, dict) or set(cuda) != {
                    "usable", "count", "name", "vram_bytes", "error",
                    "interpreter"}:
                failures.append("CUDA attestation keys differ")
                cuda = {}
            checks["cuda_usable"] = (
                cuda.get("usable") is True
                and cuda.get("count") == len(rows) and bool(rows)
                and isinstance(cuda.get("name"), str)
                and cuda["name"].strip().casefold() == model.casefold()
                and isinstance(cuda.get("vram_bytes"), int)
                and vram_floor <= cuda["vram_bytes"] <= vram_ceiling)
            filesystems = observed.get("filesystems")
            if not isinstance(filesystems, dict) or set(filesystems) != {
                    "container", "workspace"}:
                failures.append("filesystem attestation keys differ")
                filesystems = {}
            filesystem_keys = {
                "path", "mount_point", "fs_type", "source", "device",
                "total_bytes", "available_bytes",
            }
            for role in ("container", "workspace"):
                row = filesystems.get(role)
                if not isinstance(row, dict) or set(row) != filesystem_keys:
                    failures.append("%s filesystem keys differ" % role)
            container = filesystems.get("container", {})
            workspace = filesystems.get("workspace", {})
            checks["container_disk_size"] = (
                isinstance(container.get("total_bytes"), int)
                and container["total_bytes"] >= container_disk_gb * 900_000_000)
            checks["workspace_volume_size"] = (
                isinstance(workspace.get("total_bytes"), int)
                and workspace["total_bytes"] >= volume_gb * 900_000_000)
            # The separable-storage proof. A JarvisLabs filesystem is a
            # different device from the container disk; if these are the same
            # device then the "separable" storage is the instance's own disk,
            # and a 200 GB fetch will fill it after the bootstrap is paid for.
            checks["workspace_is_separate_device"] = (
                workspace.get("path") == JL_WORKSPACE
                and container.get("path") == "/"
                and (workspace.get("device"), workspace.get("source"))
                != (container.get("device"), container.get("source")))
            checks["container_available_bytes"] = (
                isinstance(container.get("available_bytes"), int)
                and not isinstance(container.get("available_bytes"), bool)
                and container["available_bytes"]
                >= container_available_bytes_minimum)
            checks["workspace_available_bytes"] = (
                isinstance(workspace.get("available_bytes"), int)
                and not isinstance(workspace.get("available_bytes"), bool)
                and workspace["available_bytes"]
                >= workspace_available_bytes_minimum)
            failures.extend(
                name for name, passed in sorted(checks.items()) if not passed)
        # What the PROVIDER says this instance is, recorded beside what the
        # silicon says. A disagreement between the catalogue name and
        # nvidia-smi is exactly the case a receipt must be able to show.
        provider_record = {"region": None, "gpu_type": None, "gpu_count": None,
                           "is_spot": None, "filesystem_id": None,
                           "error": None}
        try:
            listed = self.get_lifecycle_resource(wanted)
            if listed is None:
                raise JLError("instance %s is not listed" % wanted)
            for key in ("region", "gpu_type", "gpu_count", "is_spot",
                        "filesystem_id"):
                value = listed.get(key)
                provider_record[key] = (
                    value if isinstance(value, (bool, int)) or value is None
                    else str(value))
        except Exception as exc:                     # noqa: BLE001 - recorded
            provider_record["error"] = "%s: %s" % (type(exc).__name__, exc)
        document = {
            "schema": "fidelity-suite/jarvislabs-live-attestation.v1",
            "provider": self.provider, "provider_id": wanted,
            "observed_at_utc": clock["controller_receive_utc"],
            "clock": clock,
            "provider_record": provider_record,
            "expected": expected, "observed": observed,
            "transport": "jl exec -- sh -lc (read-only probe)",
            "transport_error": transport_error,
            "checks": checks, "failures": sorted(set(failures)),
            "ok": bool(not failures and transport_error is None
                       and checks and all(checks.values())),
        }
        return _seal(document, "attestation_sha256")

    # ---- the provider contract: is anything of mine still alive? ----------

    def list_network_volumes(self) -> List[Dict[str, Any]]:
        """Persistent chargeable filesystems, which OUTLIVE their instance.

        This is real work on JarvisLabs and the opposite of Vast, where the
        volume dies with the pod and the list is legitimately empty.  Here a
        filesystem is created separately (`jl filesystem create --storage
        --region`), attached with `--fs-id`, and survives `jl destroy` -- that
        persistence is exactly what makes a preempted spot box cheap to
        resume, and it is also what makes an orphaned filesystem a chargeable
        resource that NO instance listing will ever show.

        Same discipline as `list_instances`: an unrecognised payload refuses
        rather than reporting zero volumes, because a caller reads zero as
        proof that nothing is left to bill.
        """
        data = self.fs_list()
        if isinstance(data, list):
            rows: List[Any] = data
        elif isinstance(data, dict) and isinstance(
                data.get("filesystems"), list):
            rows = data["filesystems"]
        else:
            raise JLError(
                "`jl filesystem list --json` (jl %s) answered with %s, which "
                "is not a filesystem list. Refusing to report zero volumes: a "
                "JarvisLabs filesystem outlives its instance, so zero is read "
                "as proof that nothing is still billing."
                % (self.version,
                   "an empty body" if not data
                   else "an object with keys %r" % (sorted(data)[:8],)
                   if isinstance(data, dict) else "a %s" % type(data).__name__))
        volumes = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise JLError(
                    "`jl filesystem list --json` returned a row that is not an "
                    "object (%r); this client cannot tell what is billing from "
                    "that" % (redact(str(row))[:120],))
            # `jl list` calls an instance's attached filesystem `fs_id` and its
            # size `storage_gb`; `jl filesystem create` names its inputs
            # --storage/--region/--name. Both spellings are accepted, and a row
            # carrying NEITHER id spelling refuses with the keys it did carry,
            # rather than being silently dropped from the inventory.
            raw_id = next((row[key] for key in ("fs_id", "id", "filesystem_id")
                           if row.get(key) not in (None, "")), None)
            if raw_id is None:
                raise JLError(
                    "a `jl filesystem list` row carries no recognised id "
                    "(keys: %r). Refusing to drop it: an unidentified "
                    "filesystem is still a chargeable resource."
                    % (sorted(row)[:12],))
            volume_id = _jl_id(raw_id, "JarvisLabs filesystem id")
            if volume_id in seen:
                raise JLError(
                    "`jl filesystem list` reported filesystem %s twice"
                    % volume_id)
            seen.add(volume_id)
            size = next((row[key] for key in
                         ("storage", "storage_gb", "size", "size_gb")
                         if row.get(key) not in (None, "")), None)
            volumes.append({
                "id": volume_id,
                "name": row.get("name"),
                "size_gb": (None if size is None
                            else _positive_int(
                                size, "JarvisLabs filesystem size")),
                "region": row.get("region"),
                # A filesystem with no instance attached is the leak case:
                # nothing in `jl list` mentions it and it keeps billing.
                "attached_machine_ids": sorted(
                    _jl_id(value, "JarvisLabs filesystem attachment")
                    for value in (row.get("machine_ids")
                                  or row.get("instances") or [])
                    if value not in (None, "")),
                "raw": row,
            })
        return volumes

    def chargeable_inventory(self) -> Dict[str, Any]:
        """Every chargeable family, with EXPLICIT completeness per family.

        A partial inventory cannot prove no leak, so this never implies
        wholeness: each family says whether it was established, and
        `unknown_families` names the ones that were not.  A caller that reads
        `complete` as false must treat it as an outage and refuse to conclude
        absence -- an empty family that failed to load looks exactly like an
        empty account.

        Three families, which is what `jl status --json` itself counts:
        instances (containers and VMs), filesystems -- which outlive
        instances, so omitting them would make absence provably wrong -- and
        serverless deployments.

        The completeness evidence here is stronger than naming an endpoint.
        `jl status --json` returns independent COUNTERS
        (running_instances/paused_instances/running_vms/paused_vms,
        filesystems), and each family is cross-checked against its counter.
        Two reads that disagree mean the enumeration is not authoritative, and
        that is reported as incompleteness rather than reconciled by choosing
        one.  A transitional state no counter covers can trip this; the
        correct response is another sweep, never a weakened check.
        """
        summary: Dict[str, Any] = {}
        summary_error = None
        try:
            status = self.status()
            resources = status.get("resources")
            if not isinstance(resources, dict):
                raise JLError("`jl status --json` carries no resources object")
            summary = resources
        except JLError as exc:
            summary_error = redact(str(exc))

        def counter(*names: str) -> Optional[int]:
            if summary_error is not None:
                return None
            total = 0
            for name in names:
                value = summary.get(name)
                if isinstance(value, bool) or not isinstance(value, int):
                    return None
                total += value
            return total

        families: Dict[str, Dict[str, Any]] = {}

        def establish(key: str, source: str, load, expected_count) -> None:
            try:
                resources = load()
            except JLError as exc:
                families[key] = {"complete": False, "source": source,
                                 "resources": [], "unknown": redact(str(exc))}
                return
            family = {"complete": True, "source": source,
                      "resources": resources,
                      "account_summary_count": expected_count}
            if expected_count is None:
                family["complete"] = False
                family["unknown"] = (
                    "`jl status --json` did not supply a comparable counter "
                    "(%s), so this enumeration is uncorroborated"
                    % (summary_error or "counter missing or non-integer"))
            elif expected_count != len(resources):
                family["complete"] = False
                family["unknown"] = (
                    "%s listed %d resource(s) but `jl status` counts %d; the "
                    "two provider reads disagree, so this family is not "
                    "authoritative" % (source, len(resources), expected_count))
            families[key] = family

        establish("instances", "jl list --json",
                  self.list_lifecycle_resources,
                  counter("running_instances", "paused_instances",
                          "running_vms", "paused_vms"))
        establish("network_volumes", "jl filesystem list --json",
                  self.list_network_volumes, counter("filesystems"))
        establish("deployments", "jl deploy list --json",
                  self.list_deployments, counter("deployments"))
        unknown = sorted(name for name, family in families.items()
                         if not family["complete"])
        return {
            "schema": "fidelity-suite/jarvislabs-chargeable-inventory.v1",
            "provider": self.provider,
            "observed_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "complete": not unknown,
            "unknown_families": unknown,
            "families": families,
        }

    def list_deployments(self) -> List[Dict[str, Any]]:
        """Serverless deployments, the third chargeable family.

        Beta, and empty on this account, but a deployment bills and no
        instance or filesystem listing shows one.  `jl deploy list --json`
        answers `{"deployments": [...], "region_errors": [...]}` -- and
        `region_errors` is the provider stating its own partiality, so a
        non-empty one refuses rather than under-reporting.
        """
        data = self._call(["deploy", "list"])
        if not isinstance(data, dict) or not isinstance(
                data.get("deployments"), list):
            raise JLError(
                "`jl deploy list --json` (jl %s) answered with %s rather than "
                "a deployments object; refusing to report zero deployments"
                % (self.version,
                   "an empty body" if not data
                   else "keys %r" % (sorted(data)[:8],)
                   if isinstance(data, dict) else type(data).__name__))
        errors = data.get("region_errors") or []
        if errors:
            raise JLError(
                "`jl deploy list` reported %d region error(s) (%s); the "
                "deployment list is partial by the provider's own account"
                % (len(errors), redact(str(errors))[:200]))
        rows = []
        for row in data["deployments"]:
            if not isinstance(row, dict):
                raise JLError(
                    "`jl deploy list` returned a non-object deployment row")
            raw_id = next((row[key] for key in
                           ("deployment_id", "id", "name")
                           if row.get(key) not in (None, "")), None)
            if raw_id is None:
                raise JLError(
                    "a `jl deploy list` row carries no recognised id (keys: "
                    "%r); refusing to drop a billing resource"
                    % (sorted(row)[:12],))
            rows.append({
                "id": _jl_id(raw_id, "JarvisLabs deployment id"),
                "name": row.get("name"),
                "status": row.get("status") or row.get("state"),
                "region": row.get("region"),
                "raw": row,
            })
        return rows

    # ---- the provider contract: two-phase create --------------------------

    def prepare_safe_create(self, **kw) -> PreparedJLCreate:
        """Freeze an exact create request without mutating anything.

        The point of splitting create in two is that a LOST RESPONSE becomes
        reconcilable instead of ambiguous.  On this provider the frozen
        artifact is the argv, and the evidence that survives a crash between
        the halves is the exact `--name`: the caller persists `to_dict()`
        (argv plus its digest plus the request identity) BEFORE submitting,
        and recovery is `jl list` filtered to that name.  So the name must be
        exact and the request must be fully determined here -- nothing may be
        decided inside `submit_prepared_create`, or the recorded intent and
        the mutation could disagree.

        The refusals are the safe profile, and each one is a thing that has
        cost money or trust somewhere: spot preemption discards a capture, an
        unpinned region loses the datacenter a receipt has to name, a
        CALLER-SUPPLIED startup script runs unaudited code before the
        attestation, exposed HTTP ports widen the box's surface, and a CPU VM
        cannot measure anything.

        `pin_host_key=True` is the one startup script this profile will run,
        and it is generated here rather than supplied.  It exists because
        `jl` disables host authentication outright -- `jarvislabs/ssh.py`
        appends `UserKnownHostsFile=/dev/null` and
        `StrictHostKeyChecking=no` to every `exec`/`upload`/`download`/`run`
        -- so there is no trust-on-first-use to mitigate and no boot log to
        read a fingerprint out of.  Pinning removes the trust instead of
        tolerating it: a fresh ED25519 host key is generated, installed by a
        script registered with `jl scripts add`, and named in the create by
        id, so the expected fingerprint is known BEFORE the instance exists
        and only a box that received our authenticated launch can present it.
        See `ssh_host_ed25519_fingerprint` for what that does and does not
        buy while jl's own transport still ignores host keys.
        """
        for key in ("script_id", "script-id", "script_args", "script-args"):
            if kw.get(key) not in (None, "", False):
                raise JLError(
                    "safe JarvisLabs profile refuses a caller-supplied startup "
                    "script argument (%s): it runs unaudited code before "
                    "attest_live_resource can prove what the box is. The only "
                    "startup script this profile runs is the host-key pinning "
                    "script it generates itself (pin_host_key=True)" % key)
        if kw.get("http_ports") not in (None, "", False):
            raise JLError(
                "safe JarvisLabs profile refuses --http-ports: the run needs "
                "no inbound service, and each exposed port is reachable "
                "surface on a box holding an HF token")
        if kw.get("cpu"):
            raise JLError(
                "safe JarvisLabs profile refuses a CPU VM: a measurement needs "
                "the GPU the root was captured on")
        if kw.get("spot", False) is not False:
            raise JLError(
                "safe JarvisLabs profile requires spot exactly false: a "
                "preemption mid-capture discards the run and its spend. Spot "
                "is also containers-only here, so it cannot be combined with "
                "--vm at all")
        name = str(kw.get("name") or "").strip()
        if not name:
            raise JLError(
                "safe JarvisLabs create requires an exact lease name; it is "
                "the only evidence that reconciles a lost create response")
        if name == "Name me":
            raise JLError(
                "safe JarvisLabs create refuses the `jl create` default name "
                "'Name me': a lost response could not be attributed to this "
                "lease, and a second create would double-spend")
        region = str(kw.get("region") or "").strip()
        if region not in JL_REGIONS:
            raise JLError(
                "safe JarvisLabs create requires an exact region pin (one of "
                "%s); an unpinned create lets the provider choose the "
                "datacenter and the receipt then cannot say where the number "
                "was made" % ", ".join(JL_REGIONS))
        gpu = str(kw.get("gpu_type") or kw.get("gpu") or "").strip()
        if not gpu:
            raise JLError(
                "safe JarvisLabs create requires an exact GPU model as `jl "
                "gpus --json` spells it (gpu_type=A100-80GB, H200, "
                "RTX-PRO6000, ...)")
        template = str(kw.get("template") or "pytorch").strip()
        if "/" in template or "@sha256:" in template:
            raise JLError(
                "%r looks like a container image reference; `jl create` takes "
                "--template <name> from `jl templates`. The measurement image "
                "cannot be launched as a JarvisLabs container template"
                % template)
        terminate_after = kw.get("terminate_after")
        if terminate_after is None and kw.get("terminate_after_epoch") is not None:
            terminate_after = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(float(kw["terminate_after_epoch"])))
        if terminate_after is None:
            raise JLError(
                "safe JarvisLabs create requires terminate_after or "
                "terminate_after_epoch. `jl create` has no deadline flag, so "
                "this is enforced by the local reaper alone -- which is "
                "exactly why it may not be omitted")
        terminate_after = _exact_utc(terminate_after, "terminate_after")
        if _utc_epoch(terminate_after) - time.time() < MIN_CREATE_SETUP_SECONDS:
            raise JLError(
                "JarvisLabs terminate_after must be at least %d seconds in the "
                "future; a box torn down during bootstrap has been paid for "
                "and produced nothing" % MIN_CREATE_SETUP_SECONDS)
        storage_gb = _positive_int(
            kw.get("storage", kw.get("storage_gb")), "storage")
        gpu_count = _positive_int(kw.get("num_gpus", 1), "num_gpus")
        fs_id = kw.get("fs_id")
        volume_gb = kw.get("volume_gb")
        if fs_id is not None:
            fs_id = _jl_id(fs_id, "JarvisLabs filesystem id")
        elif volume_gb is not None:
            # The controller sizes separable storage separately on this
            # provider (separable_storage = True). Asking for a volume with no
            # filesystem to put it on is the ENOSPC-after-paid-setup case.
            raise JLError(
                "volume_gb=%r was requested with no fs_id: JarvisLabs "
                "separable storage is a filesystem created by `jl filesystem "
                "create --storage <gb> --region %s` and attached with --fs-id"
                % (volume_gb, region))
        vm = bool(kw.get("vm"))
        argv: List[str] = [
            "create", "--gpu", gpu, "--num-gpus", str(gpu_count),
            "--region", region, "--storage", str(storage_gb),
            "--name", name,
        ]
        if vm:
            argv.append("--vm")
        else:
            argv.extend(["--template", template])
        if fs_id is not None:
            argv.extend(["--fs-id", fs_id])
        pinned = (self._pin_host_key(name)
                  if kw.get("pin_host_key") else None)
        if pinned is not None:
            argv.extend(["--script-id", pinned["script_id"]])
        request_identity = {
            "provider": self.provider,
            "gpu_type": gpu,
            "gpu_count": gpu_count,
            "region": region,
            "workload_type": "vm" if vm else "container",
            "template": template if not vm else "vm",
            "container_disk_gb": storage_gb,
            "filesystem_id": fs_id,
            "is_spot": False,
            "name": name,
            "terminate_after": terminate_after,
            "terminate_after_enforced_by": "local reaper only",
            "http_ports": None,
            # Either None, or the pinning script this method generated. A
            # caller-supplied script id is refused above.
            "script_id": None if pinned is None else pinned["script_id"],
            # Known BEFORE the instance exists, which is the whole point: only
            # a box that received our authenticated launch request can present
            # this key. The script is recorded by DIGEST -- its content carries
            # the private half.
            "host_key_fingerprint_sha256": (
                None if pinned is None else pinned["fingerprint_sha256"]),
            "host_key_public": None if pinned is None else pinned["public"],
            "host_key_script_sha256": (
                None if pinned is None else pinned["script_sha256"]),
            # jl's own transport cannot check it (jarvislabs/ssh.py sets
            # StrictHostKeyChecking=no, UserKnownHostsFile=/dev/null), so a
            # pin is an expectation for a verifying transport, never a claim
            # that this channel authenticated anything.
            "channel_verifies_host_key": False,
        }
        return PreparedJLCreate(
            argv=tuple(argv),
            request_identity_json=_canonical_bytes(request_identity),
            name=name, terminate_after=terminate_after,
            storage_gb=storage_gb, container_disk_gb=storage_gb,
            template=request_identity["template"], dry_run=self.dry,
            host_key_fingerprint_sha256=(
                None if pinned is None else pinned["fingerprint_sha256"]),
            host_key_public=None if pinned is None else pinned["public"],
            host_key_script_id=None if pinned is None else pinned["script_id"],
            host_key_script_sha256=(
                None if pinned is None else pinned["script_sha256"]))

    def _pin_host_key(self, name: str) -> Dict[str, Any]:
        """Generate an ED25519 host key and register the script that installs it.

        The expected fingerprint therefore exists before the instance does,
        which is what removes trust-on-first-use rather than mitigating it.

        The private half necessarily reaches the provider -- it is what the
        box must present -- and it goes over `jl scripts add`, the same
        TLS-authenticated API path as the launch request itself. It is
        registered as a secret so nothing can echo it, written 0600, and the
        local copy is removed once uploaded; only the PUBLIC half and the two
        digests are kept.
        """
        if self.dry:
            raise JLError(
                "cannot pin a host key under dry: registering the pinning "
                "script with `jl scripts add` is a provider mutation, and a "
                "dry create must not leave one behind")
        workdir = tempfile.mkdtemp(prefix="fidelity-jl-hostkey-")
        keyfile = os.path.join(workdir, "ssh_host_ed25519_key")
        try:
            proc = run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "",
                        "-C", "fidelity-pinned-%s" % name, "-f", keyfile],
                       timeout=60, check=False)
            if proc.returncode != 0 or not os.path.isfile(keyfile + ".pub"):
                raise JLError(
                    "ssh-keygen could not generate a pinned host key: %s"
                    % redact((proc.stderr or proc.stdout or "")[:200]))
            with open(keyfile, encoding="utf-8") as handle:
                private = handle.read()
            with open(keyfile + ".pub", encoding="utf-8") as handle:
                public = handle.read().strip()
            register_secret(private)
            shown = run(["ssh-keygen", "-lf", keyfile + ".pub"],
                        timeout=30, check=False)
            fields = (shown.stdout or "").split()
            fingerprint = next((f for f in fields
                                if f.startswith("SHA256:")), None)
            if shown.returncode != 0 or fingerprint is None:
                raise JLError(
                    "ssh-keygen could not fingerprint the pinned host key")
            script = _HOST_KEY_PIN_SCRIPT.replace(
                "__PRIVATE_KEY__", private.strip()).replace(
                "__PUBLIC_KEY__", public)
            script_path = os.path.join(workdir, "pin-host-key-%s.sh" % name)
            with open(script_path, "w", encoding="utf-8") as handle:
                os.chmod(script_path, 0o600)
                handle.write(script)
            response = self._call(
                ["scripts", "add", script_path, "--name",
                 "fidelity-hostkey-%s" % name], mutating=True, timeout=120)
            if isinstance(response, list):
                response = response[0] if response else {}
            raw_id = (response.get("script_id") or response.get("id")
                      or response.get("scriptId")
                      if isinstance(response, dict) else None)
            if raw_id in (None, ""):
                raise JLError(
                    "`jl scripts add` returned no script id (keys: %r), so the "
                    "pinning script cannot be referenced by `jl create "
                    "--script-id`. Refusing to create an instance whose host "
                    "key would be unpinned"
                    % (sorted(response)[:12]
                       if isinstance(response, dict) else type(response).__name__))
            return {
                "script_id": _jl_id(raw_id, "JarvisLabs script id"),
                "fingerprint_sha256": fingerprint,
                "public": public,
                "script_sha256": hashlib.sha256(
                    script.encode("utf-8")).hexdigest(),
            }
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def submit_prepared_create(
            self, prepared: PreparedJLCreate) -> Dict[str, Any]:
        """Submit the frozen argv and qualify the response's identity.

        Nothing is decided here.  A response with no id at all is an ambiguous
        create and the caller must reconcile by name; a response carrying an
        id whose name is NOT the exact requested name is worse -- the create
        committed and we cannot attribute it -- so that raises
        `JLCreateResponseError` with the id, which is the one thing a
        teardown needs.
        """
        if not isinstance(prepared, PreparedJLCreate):
            raise JLError(
                "submit_prepared_create takes a PreparedJLCreate from "
                "prepare_safe_create; a raw kwargs create bypasses the frozen "
                "request and the safe profile")
        if prepared.dry_run or self.dry:
            return {
                "dry_run": True,
                "request": prepared.to_dict()["request_identity"],
                "prepared_create": prepared.to_dict(),
            }
        response = self._call(list(prepared.argv), mutating=True, timeout=900)
        if isinstance(response, list):
            response = response[0] if response else {}
        if not isinstance(response, dict):
            raise JLError(
                "`jl create` returned %s rather than an instance object; if an "
                "instance was created it is NOT adopted -- reconcile by the "
                "exact name %r against `jl list` before creating anything else"
                % (type(response).__name__, prepared.name))
        raw_id = next((response[key] for key in
                       ("machine_id", "id", "instance_id")
                       if response.get(key) not in (None, "")), None)
        if raw_id is None:
            raise JLError(
                "`jl create` returned no machine id (keys: %r); the create may "
                "have committed. Reconcile by the exact name %r against `jl "
                "list` -- do not create a second instance"
                % (sorted(response)[:12], prepared.name))
        machine_id = _jl_id(raw_id, "JarvisLabs create response machine id")
        response_name = response.get("name")
        if (response_name not in (None, "")
                and str(response_name) != prepared.name):
            raise JLCreateResponseError(
                "`jl create` response name %r is not the exact lease name %r; "
                "instance %s exists and is billing"
                % (response_name, prepared.name, machine_id),
                machine_id, response)
        return {
            "machine_id": machine_id,
            "name": response_name,
            "request": prepared.to_dict()["request_identity"],
            "prepared_create": prepared.to_dict(),
            "requested_terminate_after": prepared.terminate_after,
            "storage_gb": prepared.storage_gb,
            "container_disk_gb": prepared.container_disk_gb,
            "template": prepared.template,
        }

    # ---- the provider contract: what did it cost, and whose clock? --------

    def server_time_evidence(
            self, *, max_clock_delta_seconds: float = 30,
            max_evidence_age_seconds: float = 30) -> Dict[str, Any]:
        """REFUSES: JarvisLabs publishes no clock through the CLI.

        A teardown deadline should be encoded against the provider's time, not
        ours, so that a local clock error cannot extend a rental.  RunPod
        supplies it for free: every authenticated response carries an HTTP
        Date header.  A CLI has no headers, and none of the read-only
        subcommands answers with an absolute provider timestamp -- probed on
        jl 0.2.17 against `status`, `list`, `get`, `gpus`, `resources`,
        `cpus`, `templates`, `filesystem list`, `ssh-key list`, `scripts
        list`, `deploy list` and `run list`.  The nearest thing is `jl get`'s
        `runtime`, a coarse human duration ("10 days, 1 hours 07 minutes"),
        which cannot bound a clock offset in either direction, and `jl run
        list` is explicitly "locally tracked".

        So this refuses instead of substituting our own clock, which would
        turn an unverified assumption into recorded evidence.  It is a real
        provider gap, and it is why `attest_live_resource` labels its clock
        comparison "controller clock vs instance clock" rather than calling it
        the provider's time.
        """
        raise JLUnsupportedByCli(
            "JarvisLabs exposes no absolute provider clock through `jl` "
            "(0.2.17): no read-only subcommand returns a server timestamp, and "
            "`jl get`'s `runtime` is a coarse duration. A deadline on this "
            "provider therefore degrades from a provider-attested deadline to "
            "the local reaper's clock plus the on-instance watchdog, which "
            "runs on the box's own clock; the only ways to close it are a "
            "`--transport api` that can read an HTTP Date header, or a vendor "
            "CLI that prints a server time. Refusing rather than reporting our "
            "own clock as the provider's (asked for delta<=%.0fs, age<=%.0fs)"
            % (float(max_clock_delta_seconds), float(max_evidence_age_seconds)))

    def ssh_host_ed25519_fingerprint(
            self, provider_id: Any, *,
            prepared: Optional[PreparedJLCreate] = None,
            timeout: float = 900) -> Dict[str, Any]:
        """The host key PINNED at create, or a refusal -- never first contact.

        RunPod authenticates its host key by reading the fingerprint out of
        `api.runpod.io` over TLS before first contact.  JarvisLabs publishes
        no instance boot log at all (`jl run logs` is a managed run's stdout,
        `jl deploy logs` a serverless deployment's), so that anchor does not
        exist here and reading the key over `jl exec` would authenticate the
        channel with the channel.

        So the only sound answer is a key decided BEFORE the box existed:
        pass the `PreparedJLCreate` from `prepare_safe_create(pin_host_key=
        True)` and this returns that expectation.  Two things it deliberately
        does NOT claim:

        * `channel_verifies_host_key` is False, because `jl` itself appends
          `StrictHostKeyChecking=no` and `UserKnownHostsFile=/dev/null` to
          every ssh it builds (`jarvislabs/ssh.py:22-30`, driven by
          `jarvislabs/cli/instance.py`).  A pin the transport ignores buys
          nothing on its own: a credential-bearing or result-bearing transfer
          must use an ssh invocation that verifies this fingerprint, which is
          why `ssh_endpoint` exists.
        * `verified_on_live_image` is False until a paid run confirms that a
          JarvisLabs image actually honours a `--script-id` startup script
          early enough to replace the host key.  That is the first thing a
          first paid run must check, and it is unverified by construction
          because verifying it requires creating an instance.
        """
        wanted = _jl_id(provider_id)
        if prepared is None or not prepared.host_key_fingerprint_sha256:
            raise JLUnsupportedByCli(
                "JarvisLabs publishes no boot log for instance %s through `jl` "
                "(0.2.17), so its ED25519 host key cannot be read out of one: "
                "`jl run logs` is a managed run's output and `jl deploy logs` "
                "is a serverless deployment's. Do not trust first contact -- "
                "`jl` disables host checking entirely "
                "(jarvislabs/ssh.py:27-28 sets UserKnownHostsFile=/dev/null "
                "and StrictHostKeyChecking=no). Create with "
                "prepare_safe_create(pin_host_key=True) so the expected "
                "fingerprint exists before the instance does, and pass that "
                "prepared create here (asked for timeout %.0fs)"
                % (wanted, float(timeout)))
        return _seal({
            "schema": "fidelity-suite/jarvislabs-host-key-pin.v1",
            "provider": self.provider,
            "provider_id": wanted,
            "algorithm": "ssh-ed25519",
            "fingerprint_sha256": prepared.host_key_fingerprint_sha256,
            "public_key": prepared.host_key_public,
            "source": "pinned by prepare_safe_create before the instance "
                      "existed, installed via jl create --script-id",
            "script_id": prepared.host_key_script_id,
            "script_sha256": prepared.host_key_script_sha256,
            "provider_log_endpoint_origin": None,
            "channel_verifies_host_key": False,
            "verified_on_live_image": False,
            "observed_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, "host_key_proof_sha256")

    def ssh_endpoint(self, provider_id: Any) -> Dict[str, Any]:
        """The provider's own SSH coordinates for a verifying transport.

        Exists so a transport that DOES check host keys can be pointed at the
        instance without going through `jl exec`, which cannot.  Parsed from
        the `ssh_command` the provider puts on the instance row (`ssh
        root@<host> -p <port>`); a paused or still-launching box has none, and
        that refuses rather than returning a guess.
        """
        wanted = _jl_id(provider_id)
        row = self.get_lifecycle_resource(wanted)
        if row is None:
            raise JLError("instance %s is not listed" % wanted)
        command = str((row.get("raw") or {}).get("ssh_command") or "").strip()
        if not command:
            raise JLError(
                "instance %s publishes no ssh_command yet (status %r); a "
                "paused or launching JarvisLabs box has no endpoint, so there "
                "is nothing to verify a host key against"
                % (wanted, row.get("status")))
        parts = command.split()
        target = next((p for p in parts if "@" in p), None)
        port = None
        for flag, value in zip(parts, parts[1:]):
            if flag == "-p":
                port = value
        if target is None:
            raise JLError(
                "cannot parse a user@host out of the provider's ssh_command "
                "for %s" % wanted)
        user, _, host = target.partition("@")
        return {
            "provider": self.provider,
            "provider_id": wanted,
            "user": user,
            "host": host,
            "port": int(port) if port and port.isdigit() else 22,
            "ssh_command": command,
            # Said explicitly so no caller assumes jl's own transport checks
            # it: it does not, and that is the vendor's choice, not ours.
            "channel_verifies_host_key": False,
        }

    def billing_history(self, provider_id: Any, *, start_time: str,
                        end_time: str, bucket_size: str = "hour") -> Dict[str, Any]:
        """REFUSES: `jl` exposes no per-resource billing history at all.

        RunPod answers `GET /v2/billing/pods?podId=&startTime=&endTime=` with
        time-bucketed amounts, which is what makes a receipt's cost official
        rather than estimated.  jl 0.2.17 has no billing subcommand of any
        kind (`jl --help`: Account = logout/status/setup) and no time-windowed
        query anywhere.

        The one per-resource figure the provider does publish is `cost` on the
        instance row -- a running USD TOTAL, corroborated against list rates,
        with `billing_frequency: "hour"` beside it -- and it is readable ONLY
        while the instance exists.  `cost_snapshot` captures it; there is no
        decomposition, no window and no post-destroy read.

        Two substitutes are explicitly refused rather than offered.  A balance
        delta is worthless: `jl status` balance is account-wide, and today's
        campaign ran concurrent lanes, so the delta over any window contains
        other work.  Our own elapsed wall clock is not the provider's billing
        either -- controller wall time overstated pod life by about 2x on
        RunPod today.
        """
        wanted = _jl_id(provider_id)
        raise JLUnsupportedByCli(
            "JarvisLabs has no per-resource billing API through `jl` (0.2.17): "
            "there is no billing subcommand and no time-windowed query, so the "
            "requested %s buckets from %s to %s for instance %s cannot be "
            "retrieved. The only provider figure is `cost` on the instance row "
            "(a running USD total, readable only while the instance exists) -- "
            "capture it with cost_snapshot(%s) BEFORE destroy. A balance delta "
            "is not a substitute: the balance is account-wide and concurrent "
            "lanes make the delta meaningless."
            % (bucket_size, start_time, end_time, wanted, wanted))

    def cost_snapshot(self, provider_id: Any) -> Dict[str, Any]:
        """The provider's own running cost total for one exact id, timestamped.

        This is the best cost evidence JarvisLabs offers and the only input
        `reconcile_billing` can use, so it exists to be called at teardown --
        after that the instance leaves `jl list` and the figure is gone for
        good.

        The observation time is OURS and is named `controller_observed_at_utc`
        for that reason: this provider publishes no clock (see
        `server_time_evidence`), and a controller timestamp presented as the
        provider's would be a fabricated provenance claim in the one field
        whose job is provenance.
        """
        wanted = _jl_id(provider_id)
        row = self.get_lifecycle_resource(wanted)
        if row is None:
            raise JLError(
                "instance %s is not listed, so its cost total is already "
                "unreadable; JarvisLabs keeps no billing record reachable "
                "through `jl`. A snapshot must be taken before destroy."
                % wanted)
        return {
            "schema": "fidelity-suite/jarvislabs-cost-snapshot.v1",
            "provider": self.provider,
            "provider_id": wanted,
            "cost_usd_total": row["cost_usd_total"],
            "currency": "USD",
            "basis": "jl list .cost -- a running USD total, not a rate",
            "billing_frequency": row.get("billing_frequency"),
            "status": row.get("status"),
            "running": row.get("running"),
            "runtime_text": row.get("runtime"),
            "provider_clock_available": False,
            "controller_observed_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    @staticmethod
    def _lease_facts(lease: Dict[str, Any]) -> Dict[str, Any]:
        """Read ids, absence and cost evidence out of ANY stored lease shape.

        137 JarvisLabs leases have already been settled on this account, so a
        reconciler that only understands documents minted by current code is a
        reconciler that fails on everything it will actually meet.  Two shapes
        exist:

        * `fidelity-suite/cloud-lease.v2` -- `create.provider`,
          `provider_resource_ids`, a `history` of transitions and
          `terminal_proof`;
        * the flat lease `measure_cloud.write_lease` wrote (and the JarvisLabs
          sweep still reads) -- `{job_id, name, provider, machine_id, fs_id,
          deadline_epoch, created_at, pid}`, with `machine_id` a JSON INTEGER
          and no history at all.

        The integer is normalised through `_jl_id`, never `int()`, and the
        legacy `fs_id` is carried as a SECOND chargeable resource: a
        filesystem outlives its instance, so a legacy lease that names one
        cannot be closed by proving the machine gone.
        """
        if not isinstance(lease, dict):
            raise JLError("a lease must be a JSON object, got %s"
                          % type(lease).__name__)
        create = lease.get("create")
        create = create if isinstance(create, dict) else {}
        provider = create.get("provider") or lease.get("provider")
        shape = "cloud-lease.v2" if create else "legacy-flat"
        ids = []
        for value in lease.get("provider_resource_ids") or []:
            ids.append(_jl_id(value, "lease provider_resource_ids entry"))
        if not ids and lease.get("machine_id") not in (None, ""):
            ids.append(_jl_id(lease["machine_id"], "legacy lease machine_id"))
        filesystem_ids = []
        if lease.get("fs_id") not in (None, ""):
            filesystem_ids.append(
                _jl_id(lease["fs_id"], "legacy lease fs_id"))
        absence_at = None
        history = lease.get("history")
        if isinstance(history, list):
            for item in history:
                if isinstance(item, dict) and item.get("to") == "ABSENCE_CONFIRMED":
                    absence_at = item.get("at")
        terminal = lease.get("terminal_proof")
        if absence_at is None and isinstance(terminal, dict):
            proof = terminal.get("provider_absence")
            if isinstance(proof, dict):
                absence_at = proof.get("at") or proof.get("observed_at_utc")
        snapshots: Dict[str, Dict[str, Any]] = {}

        def absorb(container: Any) -> None:
            if not isinstance(container, dict):
                return
            rows = container.get("cost_snapshots")
            if not isinstance(rows, list):
                return
            for row in rows:
                if not isinstance(row, dict):
                    continue
                raw_id = row.get("provider_id")
                if raw_id in (None, ""):
                    continue
                key = _jl_id(raw_id, "cost snapshot provider_id")
                observed = (row.get("controller_observed_at_utc")
                            or row.get("observed_at_utc"))
                if observed is None:
                    raise JLError(
                        "a cost snapshot for %s carries no observation time; "
                        "refusing to date it from the local clock" % key)
                snapshots[key] = {
                    "provider_id": key,
                    "cost_usd_total": format(_finite_decimal(
                        row.get("cost_usd_total"),
                        "cost snapshot cost_usd_total for %s" % key), "f"),
                    "controller_observed_at_utc": _exact_utc(
                        observed, "cost snapshot time for %s" % key),
                }

        absorb(lease)
        absorb(lease.get("billing_reconciliation"))
        if isinstance(terminal, dict):
            absorb(terminal.get("provider_absence"))
        if isinstance(history, list):
            for item in history:
                if isinstance(item, dict):
                    absorb(item.get("evidence"))
        return {
            "shape": shape,
            "provider": provider,
            "provider_resource_ids": sorted(set(ids)),
            "filesystem_ids": sorted(set(filesystem_ids)),
            "absence_confirmed_at": absence_at,
            "cost_snapshots": snapshots,
        }

    def reconcile_billing(self, lease: Dict[str, Any], *,
                          now: Optional[float] = None) -> Dict[str, Any]:
        """Close a lease's cost, or refuse with the whole accounting.

        A closure has to be POST-ABSENCE and independently stable, which on
        this provider is a hard problem rather than a query: the per-instance
        `cost` total is readable only while the instance is listed, and once
        the instance is gone JarvisLabs keeps nothing reachable through `jl`
        (see `billing_history`).  So a closure is possible exactly when the
        lease carries a provider cost snapshot per exact id that was taken at
        or after the absence was confirmed.  A snapshot taken BEFORE destroy
        leaves a residual window the provider never prices, and this refuses
        rather than adding our own arithmetic to it -- an hourly-billed
        residual guessed from a local clock is not a receipt.

        Refusals carry `JLBillingUnreconcilable.evidence`, so a caller records
        billing-pending WITH the provider's own figure and its residual gap
        instead of nothing at all.  Absence itself is re-verified here against
        a complete inventory, because a "post-absence" closure over a still
        listed resource would be a false settlement.
        """
        facts = self._lease_facts(lease)
        if facts["provider"] not in (None, "", self.provider):
            raise JLError(
                "this lease names provider %r; refusing to reconcile it "
                "through the jl CLI -- a lease from another provider whose id "
                "happens to be numeric would aim JarvisLabs calls at whatever "
                "machine wears that number" % (facts["provider"],))
        ids = facts["provider_resource_ids"]
        volume_ids = facts["filesystem_ids"]
        if not ids and not volume_ids:
            raise JLError(
                "JarvisLabs billing reconciliation needs at least one exact "
                "provider id; this %s lease names none, so there is nothing to "
                "close and nothing to prove gone" % facts["shape"])
        if facts["absence_confirmed_at"] is None:
            raise JLError(
                "this %s lease carries no provider-absence event, so billing "
                "cannot be closed post-absence. Prove absence first: "
                "chargeable_inventory() must show %s missing from a COMPLETE "
                "inventory (instances and filesystems both -- a filesystem "
                "outlives its instance here)"
                % (facts["shape"], ", ".join(ids + volume_ids)))
        absence_at = _exact_utc(facts["absence_confirmed_at"],
                                "lease absence time")
        inventory = self.chargeable_inventory()
        if not inventory["complete"]:
            raise JLError(
                "cannot re-verify absence: the JarvisLabs inventory is "
                "incomplete (%s). A partial inventory cannot prove no leak, so "
                "billing stays unresolved until a later sweep reads it whole"
                % ", ".join(inventory["unknown_families"]))
        listed_instances = {row["id"] for row in
                            inventory["families"]["instances"]["resources"]}
        listed_volumes = {row["id"] for row in
                          inventory["families"]["network_volumes"]["resources"]}
        # Ids compare as STRINGS on both sides of this set operation. Comparing
        # a lease's integer id against a listing's string id is the third
        # historical portability bug, and it reported a LIVE instance as
        # destroyed -- here it would settle the cost of a box still running.
        still_alive = sorted((set(ids) & listed_instances)
                             | (set(volume_ids) & listed_volumes))
        if still_alive:
            raise JLError(
                "lease claims absence at %s but %s is still listed on the "
                "account; the absence proof is stale or wrong. Destroy the "
                "resource and re-confirm before any cost closure"
                % (absence_at, ", ".join(still_alive)))
        snapshots = facts["cost_snapshots"]
        instant = time.time() if now is None else float(now)
        absence_epoch = _utc_epoch(absence_at)
        missing = [rid for rid in ids if rid not in snapshots]
        pre_absence = sorted(
            (rid, snapshots[rid]["controller_observed_at_utc"])
            for rid in ids
            if rid in snapshots
            and _utc_epoch(snapshots[rid]["controller_observed_at_utc"])
            < absence_epoch)
        total = Decimal("0")
        for rid in ids:
            if rid in snapshots:
                total += Decimal(snapshots[rid]["cost_usd_total"])
        evidence = {
            "schema": "fidelity-suite/jarvislabs-billing-evidence.v1",
            "provider": self.provider,
            "lease_shape": facts["shape"],
            "provider_resource_ids": ids,
            "filesystem_ids": volume_ids,
            "absence_confirmed_at": absence_at,
            "absence_reverified": True,
            "inventory_complete": True,
            "cost_snapshots": [snapshots[rid] for rid in ids if rid in snapshots],
            "snapshot_total_usd": format(total, "f"),
            "provider_clock_available": False,
            "provider_billing_api_available": False,
            "ids_without_snapshot": missing,
            "pre_absence_snapshots": [
                {"provider_id": rid, "controller_observed_at_utc": at,
                 "residual_unpriced_seconds": absence_epoch - _utc_epoch(at)}
                for rid, at in pre_absence],
            "reconciled_at_epoch": instant,
        }
        if missing:
            raise JLBillingUnreconcilable(
                "no provider cost figure exists for %s: JarvisLabs publishes "
                "no billing history, and the instance's `cost` total left with "
                "the instance. This lease's cost is UNRECONCILABLE and its "
                "receipt is not publishable on cost. Future runs must call "
                "cost_snapshot(<id>) at teardown, before destroy"
                % ", ".join(missing), evidence)
        if pre_absence:
            raise JLBillingUnreconcilable(
                "every cost snapshot predates the absence proof (%s), leaving "
                "%d second(s) of hourly-billed time the provider never priced "
                "and `jl` can no longer be asked about. Refusing to add a "
                "local estimate to a provider total: the snapshot total is "
                "$%s and is a LOWER BOUND, not a closure"
                % (", ".join("%s@%s" % pair for pair in pre_absence),
                   max(row["residual_unpriced_seconds"]
                       for row in evidence["pre_absence_snapshots"]),
                   evidence["snapshot_total_usd"]),
                evidence)
        # The digest covers the CLOSURE, not the moment it was computed: two
        # reconciliations of the same settled lease must be byte-identical, or
        # nothing downstream can tell a stable closure from a changing one.
        # `reconciled_at_epoch` is therefore recorded outside the seal.
        stable = dict(evidence)
        stable.pop("reconciled_at_epoch", None)
        sealed = _seal(stable, "closure_sha256")
        sealed["reconciled_at_epoch"] = instant
        return {
            "reconciled": True,
            "provider": self.provider,
            "provider_resource_ids": ids,
            "total_amount": evidence["snapshot_total_usd"],
            "billing_histories": [],
            "evidence": sealed,
        }


def select_offer(
    offers: Sequence[GpuOffer],
    *,
    required_vram_bytes: float,
    gpus: int,
    spot: bool,
    gpu_type: Optional[str] = None,
    region: Optional[str] = None,
) -> tuple:
    """Cheapest offer that actually fits, plus the full audited candidate table.

    Filter order is deliberate.  VRAM first, because a row that does not fit is
    not a candidate at any price; then the spot rule (spot is GPU CONTAINERS
    only, never VMs); then capacity; then price.  Every row and its verdict
    goes into the receipt, so the choice is auditable after the fact rather
    than a number that appeared from nowhere.
    """
    table = []
    viable = []
    for o in offers:
        verdict = "ok"
        if gpu_type and o.gpu_type.lower() != gpu_type.lower():
            verdict = "not requested"
        elif region and (o.region or "") != region:
            verdict = "wrong region"
        elif o.vram_bytes < required_vram_bytes:
            verdict = "too small (%.0f < %.0f GB)" % (
                o.vram_bytes / 1e9, required_vram_bytes / 1e9)
        elif spot and not o.spot:
            verdict = "on-demand row, --spot requested"
        elif not spot and o.spot:
            verdict = "spot row, --on-demand requested"
        elif spot and (o.workload_type not in (None, "", "container")):
            verdict = "spot is containers only (workload_type=%s)" % o.workload_type
        elif o.free_devices < gpus:
            verdict = "no capacity (%d free, need %d)" % (o.free_devices, gpus)
        table.append({
            "gpu_type": o.gpu_type, "region": o.region,
            "vram_gb": round(o.vram_bytes / 1e9, 0), "price": o.price,
            "spot": o.spot, "free": o.free_devices, "verdict": verdict,
        })
        if verdict == "ok":
            viable.append(o)
    if not viable:
        return None, table
    viable.sort(key=lambda o: (o.price * gpus, -o.vram_bytes))
    return viable[0], table
