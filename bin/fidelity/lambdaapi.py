#!/usr/bin/env python3
"""Lambda Cloud backend, duck-typed to `fidelity.jlapi.JL`.

Lambda is the simplest of the three and the least flexible, and both halves of
that matter to this suite:

* **No spot, no bidding.** One published on-demand price per instance type. It
  is the most expensive per GPU-hour of the providers wired up here, and the
  most predictable -- which makes it the right place to run something that must
  not be interrupted, and the wrong place to run something cheap.
* **Storage is NOT selectable.** Every other backend takes the disk size the
  plan computed. A Lambda instance type comes with whatever local disk it comes
  with, so `fs_create` cannot honour a request; it records what was asked for
  and the caller must check the type actually fits. A measurement that needs
  300 GB and lands on a type with less will fail during fetch, after the money
  starts -- so `create` refuses up front when the requested size exceeds what
  the type is known to provide.
* **Instances are region-pinned and capacity is bursty.** `instance-types`
  reports `regions_with_capacity_available`, and launching into a region that
  is not in that list fails. It is queried per launch rather than cached.
* **Filesystems OUTLIVE instances.** `GET /file-systems` enumerates persistent
  shared filesystems that survive every terminate, so an orphaned one is a real
  chargeable leak and `chargeable_inventory()` cannot prove absence without
  them. (Vast's storage is pod-scoped and its volume list is legitimately
  empty; copying that shape here would publish a false absence proof.) This
  backend deliberately attaches none -- `separable_storage` is False because of
  how we drive the provider, not because the provider lacks the feature.
* **Instances are VMs, not containers.** The measurement image cannot be used
  as-is; `bin/bootstrap_measure.sh` rebuilds the stack on the VM. So the
  transport is SSH + an uploaded bundle, and `attest_live_resource` must
  tolerate a box on which torch does not exist YET while still proving the
  DEVICE is the one the root was captured on.
* **There is no provider-side termination deadline.** RunPod takes
  `terminateAfter` and enforces it itself. Lambda's launch request has no such
  field (official OpenAPI 1.10.0), so a Lambda deadline is only ever enforced
  by the controller's watchdog and the reaper. `validate_safe_resource_binding`
  records that as an observed fact rather than letting a caller assume a
  backstop that does not exist.
* **There is no billing API at all.** The official spec publishes no
  per-resource billing or usage endpoint. `billing_history` therefore returns
  the provider's own published rate and the window, explicitly labelled
  NON-AUTHORITATIVE, and `reconcile_billing` returns a closure that says
  `reconciled: false` with the remedy -- it never seals a bill nobody sent.

SSH keys are per-account and must already be registered by NAME; Lambda does
not accept an inline public key at launch. The user is `ubuntu`, not `root`.

VERIFICATION STATUS. Every method added for provider parity (the twelve in
docs/PROVIDER-PARITY.md) is written against the official published OpenAPI
document (`GET /api/v1/openapi.json`, version 1.10.0, retrieved 2026-09-06)
and is exercised only by the offline fixtures in
`bin/selftest_lambda_contract.py`. NOTHING in this file below `available()` has
ever run against a live Lambda account, because no credential exists on the
controller this was written on. Each such method says so in its own docstring.
An unverified implementation labelled as verified is worse than none.
"""
from __future__ import annotations

import base64
import calendar
import email.utils
import hashlib
import json
import math
import os
import re
import secrets
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from .common import register_secret, safe_urlopen
from .jlapi import GpuOffer, Instance, JLError, redact
from .sshbase import SSHTransport

# The official spec names `https://cloud.lambda.ai/` as the production server
# and `https://cloud.lambdalabs.com/` as a deprecated secondary. This code
# keeps the deprecated host because it is the one this repository has actually
# transacted with (a real gpu_1x_gh200 launch); moving a paid path to a host
# nobody here can authenticate against is not an improvement. Both served
# byte-identical OpenAPI documents on 2026-09-06. See the report: the migration
# is a live-verification task, not a rename.
API = "https://cloud.lambdalabs.com/api/v1"
API_PRODUCTION_HOST = "cloud.lambda.ai"
OPENAPI_VERSION = "1.10.0"

# Sole stable default, expanded once, matching the RunPod backend's convention.
# Absent on the controller this was written on, which is why `available()`
# returns False and the whole parity surface below is offline-only.
DEFAULT_KEY_FILE = os.path.abspath(
    os.path.expanduser("~/.config/lambda/api_key"))

# `InstanceStatus` enum, verbatim from the official spec. A status this code
# has never seen is treated as LIVE, not as gone: mistaking an unknown state
# for absence is how an instance leaks.
LAMBDA_INSTANCE_STATUSES = (
    "booting", "active", "unhealthy", "terminated", "terminating", "preempted")
# The only status that proves a machine is not billing. `terminating` still
# has a machine attached; `preempted` is not `terminated` and Lambda sells no
# spot capacity, so it is treated as a state needing an operator, not absence.
LAMBDA_GONE_STATUSES = ("terminated",)

# `POST /instance-operations/launch` is rate-limited to one request per 12
# seconds (five per minute) by the API's own documentation. A 429 on a create
# is an AMBIGUOUS mutation, so this is refused before the request rather than
# discovered after it.
LAUNCH_MIN_INTERVAL_SECONDS = 12.0
MIN_CREATE_SETUP_SECONDS = 300

MAX_JSON_RESPONSE_BYTES = 8 * 1024 * 1024

# Only an idempotent read may be repeated; a create is never retried.
_READ_RETRY_STATUSES = frozenset((429, 500, 502, 503, 504))
_READ_RETRY_BACKOFF_SECONDS = (2.0, 4.0)

# Enumerated official error codes that prove the launch was REFUSED and no
# instance exists. Anything else -- a timeout, a transport error, an
# unparseable body -- keeps the fail-closed "response may have been lost" path.
_DEFINITIVE_LAUNCH_REJECTION_CODES = frozenset((
    "instance-operations/launch/insufficient-capacity",
    "global/quota-exceeded",
    "global/invalid-parameters",
    "instance-operations/launch/file-system-in-wrong-region",
))

_PROVIDER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_FINGERPRINT_RE = re.compile(r"SHA256:[A-Za-z0-9+/]{43}\Z")
_LAMBDA_NAME_RE = re.compile(r"[^\x00-\x1f\x7f]{1,64}\Z")

# Lambda publishes each type's real local disk as `specs.storage_gib`, so it is
# READ, never guessed. It was guessed once, from a hardcoded table that had
# `gpu_1x_a10` at 200 GB by confusing storage_gib with memory_gib -- and a plan
# needing 400 GB was refused on a machine whose root filesystem measured 1.4 TB.
# A false refusal is cheap to notice and expensive to trust, so the table is
# gone and the only fallback is "unknown", which does not refuse.
# `storage_gib` and `memory_gib` are BINARY units in the spec; every byte
# comparison below multiplies by 1024**3, never 1e9.
GIB = 1 << 30


class LambdaError(JLError):
    pass


def _read_key_file(path: str) -> str:
    """Read a credential file whose permissions are CHECKED, not assumed.

    Same shape as `runpodapi._load_key` and Vast's reader: absolute path,
    O_NOFOLLOW so a symlink cannot redirect the read, regular file, mode
    exactly 0600, owned by the current user. This module's refusal text
    promises a 0600 file, and a promise the code does not check is read as
    evidence.
    """
    selected = os.path.expanduser(str(path))
    if not os.path.isabs(selected):
        raise LambdaError("Lambda key file path must be absolute")
    descriptor = None
    try:
        descriptor = os.open(
            selected, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise LambdaError(
                "Lambda key file must be a regular file, not a symlink")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise LambdaError(
                "Lambda key file must have mode 0600: %s" % selected)
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise LambdaError(
                "Lambda key file must be owned by the current user")
        with os.fdopen(descriptor, encoding="utf-8") as handle:
            descriptor = None
            key = handle.read().strip()
    except LambdaError:
        raise
    except (OSError, UnicodeError) as exc:
        raise LambdaError(
            "Lambda key file is unavailable or invalid at %s: %s"
            % (selected, exc))
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not key:
        raise LambdaError("Lambda key file is empty: %s" % selected)
    return key


def _provider_id(value: Any, field: str = "Lambda instance id") -> str:
    """An exact provider id. Lambda ids are opaque strings, never names.

    The official schema types them `string` with no pattern -- the examples are
    32 hex characters, but assuming that is exactly the class of guess that has
    already produced three portability bugs in this tree. So the check is a
    character/length bound, not a format.
    """
    if not isinstance(value, str) or _PROVIDER_ID_RE.fullmatch(value) is None:
        raise LambdaError("%s has invalid characters or length" % field)
    return value


def _exact_utc(value: Any, field: str) -> str:
    text = str(value)
    try:
        parsed = time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        raise LambdaError("%s must be exact UTC YYYY-MM-DDTHH:MM:SSZ" % field)
    if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) != text:
        raise LambdaError("%s must be exact UTC YYYY-MM-DDTHH:MM:SSZ" % field)
    return text


def _utc(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _iso8601_epoch(value: Any, field: str) -> float:
    """Parse the fractional-second ISO 8601 the API emits (`created`, event
    times) without inventing a timezone. A naive stamp is REFUSED: guessing UTC
    for a timestamp that did not say so is how a cost window silently shifts."""
    text = str(value or "").strip()
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2})[Tt ](\d{2}:\d{2}:\d{2})(\.\d+)?"
        r"(Z|z|[+-]\d{2}:?\d{2})", text)
    if match is None:
        raise LambdaError(
            "%s is not an ISO 8601 instant with an explicit offset: %r"
            % (field, text[:64]))
    base = calendar.timegm(time.strptime(
        "%sT%s" % (match.group(1), match.group(2)), "%Y-%m-%dT%H:%M:%S"))
    fraction = float(match.group(3) or 0.0)
    zone = match.group(4)
    if zone in ("Z", "z"):
        offset = 0
    else:
        sign = 1 if zone[0] == "+" else -1
        digits = zone[1:].replace(":", "")
        offset = sign * (int(digits[:2]) * 3600 + int(digits[2:]) * 60)
    return base + fraction - offset


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LambdaError("%s must be a positive integer" % field)
    return value


def _exact_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LambdaError("%s must be an exact integer" % field)
    return value


def _finite_decimal(value: Any, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise LambdaError("%s is not an exact decimal" % field)
    if not parsed.is_finite() or parsed < 0:
        raise LambdaError("%s is not a finite non-negative decimal" % field)
    return parsed


def _strict_json_loads(raw: str, label: str) -> Any:
    """Refuse duplicate keys and non-finite tokens rather than resolving them."""

    def _pairs(items):
        seen = set()
        for key, _value in items:
            if key in seen:
                raise LambdaError("%s repeats the JSON key %r" % (label, key))
            seen.add(key)
        return dict(items)

    def _constant(token):
        raise LambdaError("%s contains the non-JSON token %r" % (label, token))

    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=_constant)
    except LambdaError:
        raise
    except ValueError as exc:
        raise LambdaError("%s is not valid JSON: %s" % (label, exc))


def _api_error(body: str) -> Tuple[Optional[str], str]:
    """The official error envelope: {"error": {code, message, suggestion?}}.

    Returns (code, human text). A body that does not parse yields (None, text)
    so the caller stays on its ambiguous path.
    """
    try:
        doc = json.loads(body)
    except ValueError:
        return None, redact(body[:300])
    if not isinstance(doc, dict) or not isinstance(doc.get("error"), dict):
        return None, redact(body[:300])
    error = doc["error"]
    code = error.get("code") if isinstance(error.get("code"), str) else None
    parts = [str(error.get(key)) for key in ("message", "suggestion")
             if isinstance(error.get(key), str) and error.get(key)]
    if isinstance(error.get("request_id"), str) and error["request_id"]:
        parts.append("request_id %s" % error["request_id"])
    return code, redact("%s: %s" % (code or "unknown", " -- ".join(parts)))


def _canonical_public_key(value: str) -> str:
    """`<type> <base64>`, comment stripped. Lambda stores a comment ("noname"
    in its own example) and the local .pub carries the operator's; comparing
    the comment would refuse a correctly bound key."""
    lines = str(value).strip().splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise LambdaError("an SSH public key must be exactly one nonempty line")
    fields = lines[0].split()
    if len(fields) < 2 or not fields[0].startswith("ssh-") \
            and not fields[0].startswith("ecdsa-"):
        raise LambdaError("SSH public key is not `<type> <base64>`")
    try:
        base64.b64decode(fields[1], validate=True)
    except Exception:                                     # noqa: BLE001
        raise LambdaError("SSH public key body is not valid base64")
    return "%s %s" % (fields[0], fields[1])


def _fingerprint_of_public_key(public_key: str) -> str:
    """SHA256 fingerprint through ssh-keygen, the same tool sshbase scans with."""
    try:
        result = subprocess.run(
            ["ssh-keygen", "-E", "sha256", "-lf", "-"],
            input=(public_key.rstrip("\n") + "\n").encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LambdaError("cannot fingerprint an SSH public key: %s" % exc)
    fields = result.stdout.decode("utf-8", "replace").strip().split()
    fingerprint = fields[1] if result.returncode == 0 and len(fields) >= 2 else ""
    if _FINGERPRINT_RE.fullmatch(fingerprint) is None:
        raise LambdaError("ssh-keygen returned a noncanonical fingerprint")
    return fingerprint


class _NoMutationRedirect(urllib.request.HTTPRedirectHandler):
    """A launch is never replayed against a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LambdaCreateRejectedError(LambdaError):
    """The provider refused the launch outright and created no instance."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.rejection_code = code


class LambdaCreateResponseError(LambdaError):
    """Launch committed with an exact id but returned unqualified metadata."""

    def __init__(self, message: str, provider_id: str,
                 response: Dict[str, Any]) -> None:
        super().__init__(message)
        self.provider_id = provider_id
        self.response = dict(response)


@dataclass(frozen=True)
class PreparedLambdaCreate:
    """A frozen launch request: built, validated and priced before any mutation.

    `launch_body` carries the cloud-init `user_data` that PINS the instance's
    ED25519 host key, so `to_dict()` deliberately exposes only the request's
    identity and the host key's FINGERPRINT -- never the body, which contains
    the pinned private host key.
    """

    http_request: Any
    http_opener: Any
    launch_body: bytes
    request_identity_json: bytes
    name: str
    instance_type_name: str
    region_name: str
    ssh_key_names: Tuple[str, ...]
    storage_gib: int
    gpu_count: int
    price_cents_per_hour: int
    terminate_after: str
    host_key_fingerprint: str
    host_key_public: str
    dry_run: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "fidelity-suite/lambda-prepared-create.v1",
            "request_identity": json.loads(
                self.request_identity_json.decode("utf-8")),
            "request_identity_sha256": hashlib.sha256(
                self.request_identity_json).hexdigest(),
            "launch_body_sha256": hashlib.sha256(self.launch_body).hexdigest(),
            "name": self.name,
            "terminate_after": self.terminate_after,
            "host_key_fingerprint": self.host_key_fingerprint,
            "dry_run": self.dry_run,
        }


# The read-only attestation probe, run over SSH on the VM before any upload or
# spend. It differs from RunPod's in the two ways a VM differs from a
# container: the filesystem roles are the ROOT disk and the RUN root (a Lambda
# box has no /workspace and no cgroup limit), and the run root's WRITABILITY is
# proven rather than assumed -- `/home` on a Lambda image is root-owned 0755,
# and the controller's old default of /home/jl_fs killed a paid gpu_1x_gh200
# rental two minutes in with EACCES at the bundle upload.
#
# torch may legitimately be ABSENT here: the measurement stack is rebuilt on
# the VM by bootstrap_measure.sh, so a fresh box need not have it yet. The
# probe reports that as `cuda.available: false` with the reason, and the
# attestation treats device identity (nvidia-smi, which is driver-level) as
# the hard gate and the CUDA runtime as a check that only fails when a torch
# was actually found and did not work. Nothing is inferred: whether a torch
# was found is a field.
_LAMBDA_ATTEST_SCRIPT = r'''
import json
import os
import subprocess
import sys
import time

RUN_ROOT = sys.argv[1]

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

def writable(path):
    probe = os.path.join(path, ".fidelity-attest-%d" % os.getpid())
    try:
        handle = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        return {"writable": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    try:
        os.write(handle, b"x")
    finally:
        os.close(handle)
        os.unlink(probe)
    return {"writable": True, "error": None}

mem_kib = None
with open("/proc/meminfo", "r", encoding="ascii") as stream:
    for line in stream:
        if line.startswith("MemTotal:"):
            mem_kib = int(line.split()[1])
            break
if mem_kib is None:
    raise RuntimeError("MemTotal missing")
# FREE VRAM, not total, is the attestable quantity. Host 434175 rented a
# "24 GB" 4090 with 23,424 of its 24,564 MiB already held by four foreign
# PIDs (DecoderParity, 2026-09-06): "24 GB card" was true and useless. So
# memory.used and memory.free are read alongside memory.total, and the
# compute-apps table is captured so a caller can see WHOSE processes hold the
# card rather than only that something does.
smi = subprocess.run(
    ["nvidia-smi",
     "--query-gpu=index,name,memory.total,memory.used,memory.free,"
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
                "vram_used_bytes": int(fields[3]) * 1024 * 1024,
                "vram_free_bytes": int(fields[4]) * 1024 * 1024,
                "driver_version": fields[5],
            })
apps = subprocess.run(
    ["nvidia-smi",
     "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
     "--format=csv,noheader,nounits"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
compute_apps = []
if apps.returncode == 0:
    for line in apps.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 4 and fields[1].isdigit():
            compute_apps.append({
                "gpu_uuid": fields[0], "pid": int(fields[1]),
                "process_name": fields[2][:120],
                "used_memory_bytes": int(fields[3]) * 1024 * 1024
                if fields[3].isdigit() else None,
            })
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
cuda = {"available": False, "usable": False, "count": 0, "name": None,
        "vram_bytes": None, "interpreter": None, "error": None}
candidates = [sys.executable, "/usr/bin/python3"]
venv = os.environ.get("FIDELITY_VENV", "")
if venv and os.path.isfile(os.path.join(venv, "bin", "python")):
    candidates.append(os.path.join(venv, "bin", "python"))
errors = []
seen = set()
for interpreter in candidates:
    if interpreter in seen or not interpreter:
        continue
    seen.add(interpreter)
    try:
        run = subprocess.run([interpreter, "-c", CUDA_PROBE],
                             capture_output=True, text=True, timeout=240)
    except Exception as exc:
        errors.append("%s: %s: %s" % (interpreter, type(exc).__name__, str(exc)[:200]))
        continue
    if run.returncode != 0:
        tail = (run.stderr or "").strip().splitlines()[-1:] or ["exit %d" % run.returncode]
        errors.append("%s: %s" % (interpreter, tail[0][:300]))
        continue
    try:
        parsed = json.loads(run.stdout.strip().splitlines()[-1])
    except Exception as exc:
        errors.append("%s: probe output unreadable: %s" % (interpreter, str(exc)[:200]))
        continue
    cuda.update(parsed)
    cuda["available"] = True
    cuda["interpreter"] = interpreter
    break
if not cuda["available"]:
    cuda["error"] = ("no torch on this VM yet (bootstrap_measure.sh installs "
                     "it): " + "; ".join(errors))[:400]
remote_time_epoch = int(time.time())
print(json.dumps({
    "remote_time_epoch": remote_time_epoch,
    "remote_time_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(remote_time_epoch)),
    "login_user": os.environ.get("USER") or str(os.getuid()),
    "uid": int(os.getuid()),
    "logical_cpus": len(os.sched_getaffinity(0)),
    "memtotal_bytes": mem_kib * 1024,
    "nvidia_smi_exit_code": smi.returncode,
    "nvidia_smi_error": smi.stderr[:300],
    "gpus": gpus, "cuda": cuda, "compute_apps": compute_apps,
    "filesystems": {"root": mount("/"), "run_root": mount(RUN_ROOT)},
    "run_root_write": writable(RUN_ROOT),
}, sort_keys=True, separators=(",", ":"), allow_nan=False))
'''


def _attestation_seal(document: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(document)
    raw = json.dumps(
        out, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")
    out["attestation_sha256"] = hashlib.sha256(raw).hexdigest()
    return out



class LambdaCloud(SSHTransport):
    # False because THIS BACKEND attaches no shared filesystem, so the whole
    # run must fit on the instance's own disk (fixed per instance type). Lambda
    # itself DOES have filesystems that outlive an instance -- see the module
    # docstring and `list_network_volumes` -- and the flag describes our
    # driving of the provider, not the provider.
    # The controller reads this to size `create(storage=)`.
    separable_storage = False
    provider = "lambda"
    ssh_user = "ubuntu"
    RUNS = "/home/ubuntu/.fidruns"
    # Where a RUN may be written. Not cosmetic and not a preference: Lambda logs
    # in as `ubuntu`, and `/home` on its images is root-owned 0755, so the
    # controller's default `/home/jl_fs/...` is EACCES for every command this
    # backend issues. That killed a real gpu_1x_gh200 rental two minutes in, at
    # the bundle upload, with `mkdir: cannot create directory '/home/jl_fs':
    # Permission denied` -- after the boot was paid for and before one line of
    # the measurement ran. This backend attaches no shared filesystem either,
    # so the instance's own disk (3.9 TB on the GH200) is the only place a run
    # can go.
    run_base = "/home/ubuntu"

    def __init__(self, *, dry: bool = False, key_file: Optional[str] = None,
                 ssh_key: Optional[str] = None,
                 ssh_key_names: Optional[List[str]] = None) -> None:
        self.dry = dry
        self._key_file = key_file
        self._key: Optional[str] = None
        self.ssh_key = ssh_key or os.path.expanduser("~/.ssh/id_ed25519")
        self.ssh_key_names = ssh_key_names
        self._ep: Dict[str, tuple] = {}
        self._server_time: Optional[Dict[str, Any]] = None
        self._last_launch_monotonic: Optional[float] = None
        # instance id -> the ED25519 host key this controller PINNED into that
        # instance's cloud-init user_data. Lambda publishes no console or log
        # endpoint, so this is the only out-of-band host-key authority there is.
        self._host_key_pins: Dict[str, Dict[str, Any]] = {}

    def _load_key(self) -> str:
        """Read the API key from a file whose mode is CHECKED, not promised.

        This used to be a bare `open().read()` while the refusal text promised
        a 0600 file. An error message that asserts a guarantee the code does
        not check is worse than silence, because it is read as evidence. The
        checks are RunPod's (`runpodapi._load_key`): absolute path, O_NOFOLLOW,
        regular file, mode exactly 0600, owned by us. `LAMBDA_API_KEY` is
        accepted because an environment variable has no mode to check -- but a
        file is preferred and is what the refusal recommends.
        """
        if self._key:
            return self._key
        selected = self._key_file or os.environ.get("LAMBDA_KEY_FILE") or ""
        environment = os.environ.get("LAMBDA_API_KEY", "").strip()
        if not selected and not environment and os.path.exists(
                DEFAULT_KEY_FILE):
            selected = DEFAULT_KEY_FILE
        if selected:
            self._key = _read_key_file(selected)
        elif environment:
            self._key = environment
        if not self._key:
            raise LambdaError(
                "no Lambda credential: set LAMBDA_KEY_FILE to a 0600 file "
                "owned by you, or LAMBDA_API_KEY, or place the key in %s"
                % DEFAULT_KEY_FILE)
        # Registered so `redact()` scrubs it from every error, log and receipt
        # this module can reach: the key is HTTP Basic material and a leaked
        # one can launch instances.
        register_secret(self._key)
        return self._key

    def _capture_server_time(self, response: Any, url: str) -> None:
        """Record the provider's own clock from an authenticated response.

        A teardown deadline encoded against OUR clock is worthless if the two
        disagree, and Lambda enforces no deadline of its own, so this is the
        only tie between the two clocks a Lambda run has.
        """
        raw = response.headers.get("Date")
        if not isinstance(raw, str) or not raw:
            raise LambdaError("Lambda authenticated response lacks HTTP Date")
        try:
            parsed = email.utils.parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            raise LambdaError("Lambda authenticated response Date is invalid")
        if (parsed is None or parsed.utcoffset() is None
                or parsed.utcoffset().total_seconds() != 0
                or email.utils.format_datetime(parsed, usegmt=True) != raw):
            raise LambdaError(
                "Lambda authenticated response Date is not strict GMT")
        received = time.time()
        self._server_time = {
            "schema": "fidelity-suite/lambda-server-time.v1",
            "endpoint_origin": (
                urllib.parse.urlsplit(url).scheme + "://"
                + str(urllib.parse.urlsplit(url).hostname)),
            "date_header": raw,
            "server_epoch": parsed.timestamp(),
            "local_received_epoch": received,
            "local_minus_server_seconds": received - parsed.timestamp(),
        }

    def _request(self, method: str, path: str, body: Any = None, *,
                 query: Optional[Dict[str, str]] = None,
                 timeout: float = 90) -> Tuple[int, Any]:
        """One authenticated call. Returns (status, document).

        `safe_urlopen` is mandatory rather than cosmetic: the credential travels
        in an `Authorization` header and a cross-origin redirect would hand it
        to whatever host answered. Reads are retried across a bounded transient
        outage; a mutation never is.
        """
        data = json.dumps(body).encode("utf-8") if body is not None else None
        # HTTP Basic with the key as the username and an empty password.
        token = base64.b64encode((self._load_key() + ":").encode()).decode()
        suffix = ("?" + urllib.parse.urlencode(query)) if query else ""
        url = API + path + suffix
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json",
                     "Accept": "application/json",
                     "User-Agent": "quant-fidelity-suite/0.1",
                     "Authorization": "Basic " + token})
        read_only = method == "GET"
        attempts = (0.0,) + (_READ_RETRY_BACKOFF_SECONDS if read_only else ())
        last: Optional[LambdaError] = None
        for index, pause in enumerate(attempts):
            if pause:
                time.sleep(pause)
            try:
                with safe_urlopen(req, timeout=timeout) as resp:
                    raw = resp.read(MAX_JSON_RESPONSE_BYTES + 1)
                    if len(raw) > MAX_JSON_RESPONSE_BYTES:
                        raise LambdaError(
                            "Lambda %s response exceeded %d bytes"
                            % (path, MAX_JSON_RESPONSE_BYTES))
                    if read_only:
                        self._capture_server_time(resp, url)
                    text = raw.decode("utf-8", "strict")
                    document = (_strict_json_loads(
                        text, "Lambda %s response" % path)
                        if text.strip() else {})
                    return int(resp.status), document
            except urllib.error.HTTPError as exc:
                code, detail = _api_error(
                    exc.read(4096).decode("utf-8", "replace"))
                error = LambdaError(
                    "Lambda HTTP %d on %s %s: %s"
                    % (exc.code, method, path, detail))
                setattr(error, "status", int(exc.code))
                setattr(error, "code", code)
                if read_only and exc.code in _READ_RETRY_STATUSES \
                        and index + 1 < len(attempts):
                    last = error
                    continue
                raise error
            except LambdaError:
                raise
            except Exception as exc:                      # noqa: BLE001
                raise LambdaError(
                    "Lambda request failed: %s" % redact(str(exc)))
        raise last or LambdaError("Lambda %s exhausted transient retries" % path)

    def _req(self, method: str, path: str, body: Any = None,
             *, timeout: float = 90) -> Any:
        return self._request(method, path, body, timeout=timeout)[1]

    def _get_data(self, path: str, *, query: Optional[Dict[str, str]] = None,
                  timeout: float = 90) -> Any:
        """`data` out of the official success envelope, or a refusal."""
        document = self._request("GET", path, query=query, timeout=timeout)[1]
        if not isinstance(document, dict) or "data" not in document:
            raise LambdaError(
                "Lambda GET %s did not return the documented {\"data\": ...} "
                "envelope" % path)
        return document["data"]

    # -- identity ----------------------------------------------------------
    def available(self) -> bool:
        try:
            self._load_key()
            return True
        except LambdaError:
            return False

    def require(self) -> tuple:
        if not self.available():
            raise LambdaError("Lambda credential not configured")
        return (0, 0, 0)

    @property
    def version(self) -> str:
        return "lambda-api-v1"

    def status(self) -> Dict[str, Any]:
        return {"instance_types": len(self._req("GET", "/instance-types")
                                      .get("data", {}))}

    def balance(self) -> Optional[float]:
        # Lambda publishes no balance endpoint: it is pay-as-you-go and bills
        # after the fact. Returning None is honest; inventing a number here
        # would make the controller's "can this account pay?" check a lie.
        return None

    def ssh_key_names_available(self) -> List[str]:
        return [k.get("name") for k in
                self._req("GET", "/ssh-keys").get("data", [])]

    # -- catalogue ---------------------------------------------------------
    def gpus(self) -> List[GpuOffer]:
        data = self._req("GET", "/instance-types").get("data", {})
        offers = []
        for name, v in data.items():
            it = v.get("instance_type") or {}
            regions = v.get("regions_with_capacity_available") or []
            specs = it.get("specs") or {}
            gpus = int(specs.get("gpus") or 1)
            vram = float(it.get("gpu_description", "0").split("(")[-1]
                         .split("GB")[0].strip() or 0) if "GB" in (
                             it.get("gpu_description") or "") else 0.0
            for r in (regions or [None]):
                offers.append(GpuOffer(
                    gpu_type=name,
                    region=(r or {}).get("name") if isinstance(r, dict) else r,
                    vram_bytes=vram * (1024 ** 3),
                    price=float(it.get("price_cents_per_hour") or 0) / 100.0,
                    spot=False,
                    free_devices=1 if regions else 0,
                    workload_type="vm",
                    raw={"gpus": gpus,
                         "disk_gb": (specs.get("storage_gib")),
                         "description": it.get("gpu_description"),
                         "available": bool(regions)}))
        return offers

    # -- instances ---------------------------------------------------------
    @staticmethod
    def _to_instance(d: Dict[str, Any]) -> Instance:
        it = d.get("instance_type") or {}
        inst = Instance.from_json({
            "machine_id": 0, "status": d.get("status") or "",
            "gpu_type": it.get("name"),
            "num_gpus": int(((it.get("specs") or {}).get("gpus")) or 1),
            "region": (d.get("region") or {}).get("name"),
            "is_spot": False, "cost": 0.0, "runtime": None, "fs_id": None,
            "storage_gb": ((it.get("specs") or {}).get("storage_gib")),
            "name": d.get("name"),
        })
        inst.machine_id = d.get("id")
        inst.raw["ip"] = d.get("ip")
        inst.raw["price_cents_per_hour"] = it.get("price_cents_per_hour")
        return inst

    def list_instances(self) -> List[Instance]:
        return [self._to_instance(d)
                for d in self._req("GET", "/instances").get("data", [])]

    def get(self, machine_id: Any) -> Optional[Instance]:
        for i in self.list_instances():
            if str(i.machine_id) == str(machine_id):
                return i
        return None

    # A URL WITH A PATH in a create body is a bearer capability: anyone
    # holding a result-sink URL can read a run's results, which is the same
    # insight as the ntfy-topic finding. tlsguard's shared matcher does NOT
    # flag one today (VastParity established that by running the portability
    # suite against the real module before pushing), so this profile carries
    # the property explicitly. DELETE this and its rungs the moment tlsguard
    # covers it -- it is a named policy of the Lambda safe profile, not a
    # second "looks like a secret" matcher.
    _BEARER_URL_RE = re.compile(r"https?://[^\s/]+/\S+")

    def _refuse_credential_payload(self, payload: Dict[str, Any], *,
                                   operation: str = "create") -> None:
        """Refuse before a credential enters Lambda's own records.

        A create body is provider-persisted and lands in the host's
        environment BEFORE the instance exists, so no ordering and no
        attestation can protect it -- there is nothing to attest yet.

        `tlsguard` owns the one implementation of "looks like a secret". It is
        imported lazily and its ABSENCE is a refusal, not a second code path:
        a fail-closed fallback that is the only branch a test ever reaches is
        an untested primary wearing a green badge, which is exactly how
        Vast's `TlsRefusal` escaped its own adapter boundary today. So there
        is one detector, one wrapper, and no shadow implementation.
        """
        for key, value in sorted(payload.items()):
            if isinstance(value, str) and self._BEARER_URL_RE.search(value):
                raise LambdaError(
                    "safe Lambda %s refuses a URL with a path in `%s`: a "
                    "result-sink or webhook URL is a BEARER CAPABILITY -- "
                    "anyone holding it can read this run's results -- and a "
                    "create body is stored by the provider" % (operation, key))
        try:
            from .tlsguard import (TlsRefusal,
                                   refuse_credential_in_provider_payload)
        except ImportError as exc:
            raise LambdaError(
                "cannot check a Lambda %s payload for credentials: "
                "fidelity.tlsguard is not importable (%s). It is in "
                "bin/BUNDLE.txt; refusing rather than transmitting an "
                "unchecked payload into the provider's own records"
                % (operation, exc))
        try:
            refuse_credential_in_provider_payload(
                dict(payload, provider="lambda"), operation=operation)
        except TlsRefusal as exc:
            raise LambdaError(
                "%s -- %s" % (exc.reason, "; ".join(exc.advice)))

    def create(self, **kw) -> Dict[str, Any]:
        # BEFORE the dry short-circuit: a payload that would carry a
        # credential into provider-persisted records is a refusal even when
        # this call would transmit nothing, because the caller's intent is
        # the defect and a dry run is where it should be caught.
        self._refuse_credential_payload(dict(kw), operation="create")
        if self.dry:
            return {"dry_run": True, **kw}
        itype = kw.get("gpu_type") or kw.get("instance_type")
        if not itype:
            raise LambdaError("create requires gpu_type (a Lambda instance type)")
        want_disk = int(kw.get("storage") or kw.get("storage_gb") or 0)
        types_now = self._req("GET", "/instance-types").get("data", {})
        have = (((types_now.get(itype) or {}).get("instance_type") or {})
                .get("specs") or {}).get("storage_gib")
        if want_disk and have and want_disk > have:
            raise LambdaError(
                "instance type %s provides ~%d GB of local disk and this plan "
                "needs %d GB. Lambda disk is fixed per type and cannot be "
                "grown, so this would fail during fetch, after billing starts."
                % (itype, have, want_disk))
        names = self.ssh_key_names or self.ssh_key_names_available()
        if not names:
            raise LambdaError(
                "no SSH key registered on the Lambda account. Lambda attaches "
                "keys BY NAME at launch and accepts no inline public key, so "
                "one must be added in the console first.")
        types = self._req("GET", "/instance-types").get("data", {})
        regions = (types.get(itype) or {}).get(
            "regions_with_capacity_available") or []
        if not regions:
            raise LambdaError("instance type %s has no region with capacity "
                              "right now" % itype)
        region = kw.get("region") or regions[0].get("name")
        got = self._req("POST", "/instance-operations/launch", {
            "region_name": region, "instance_type_name": itype,
            "ssh_key_names": names[:1], "quantity": 1,
            "name": kw.get("name") or "fidcloud"}, timeout=180)
        ids = (got.get("data") or {}).get("instance_ids") or []
        if not ids:
            raise LambdaError("Lambda returned no instance id: %s"
                              % redact(json.dumps(got)[:300]))
        return {"machine_id": ids[0], "region": region}

    def destroy(self, machine_id: Any) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True}
        self._req("POST", "/instance-operations/terminate",
                  {"instance_ids": [str(machine_id)]}, timeout=180)
        return {"terminated": str(machine_id)}

    def pause(self, machine_id: Any) -> Dict[str, Any]:
        raise LambdaError("Lambda has no pause: an instance is running or "
                          "terminated. Use destroy().")

    def resume(self, machine_id: Any, *, spot: bool = False) -> Dict[str, Any]:
        raise LambdaError("Lambda has no resume; a terminated instance is gone.")

    # -- ssh ---------------------------------------------------------------
    def _endpoint(self, machine_id: Any, *, wait: float = 900) -> tuple:
        key = str(machine_id)
        if key in self._ep:
            return self._ep[key]
        deadline = time.time() + wait
        while time.time() < deadline:
            inst = self.get(machine_id)
            if inst is not None and inst.raw.get("ip") \
                    and str(inst.status).lower() in ("active", "running"):
                self._ep[key] = (inst.raw["ip"], 22)
                return self._ep[key]
            time.sleep(10)
        raise LambdaError("instance %s never became reachable within %ds"
                          % (machine_id, int(wait)))

    # -- storage -----------------------------------------------------------
    def fs_create(self, *, storage: int, region: str = "",
                  name: Optional[str] = None) -> Any:
        return {"fs_id": None, "storage_gb": int(storage),
                "note": "lambda disk is fixed per instance type and cannot be "
                        "requested; create() refuses a type too small for it"}

    def fs_delete(self, fs_id: Any) -> Any:
        # A Lambda filesystem IS separable and DOES outlive its instance, so
        # "no separable filesystem on lambda" -- what this used to say -- was
        # simply wrong, and it is the belief that would make
        # `list_network_volumes` trivially empty and a leak invisible. This
        # backend attaches none, so a run has nothing to delete; deleting one
        # that exists is `DELETE /api/v1/filesystems/{id}` and is an OPERATOR
        # decision, because we may not have created it and a mounted
        # filesystem cannot be deleted at all (`is_in_use`).
        return {
            "deleted": False,
            "fs_id": None if fs_id is None else str(fs_id),
            "note": "this backend attaches no Lambda filesystem, so a run "
                    "never has one to delete; a filesystem that does exist "
                    "outlives every instance and is enumerated by "
                    "list_network_volumes(). Deleting one is DELETE "
                    "/api/v1/filesystems/{id} and an operator decision.",
        }

    # ======================================================================
    # Provider parity: the twelve methods in docs/PROVIDER-PARITY.md
    # ----------------------------------------------------------------------
    # OFFLINE PORT. Written against the official published OpenAPI document
    # (version 1.10.0, retrieved 2026-09-06) and exercised only by
    # bin/selftest_lambda_contract.py. No credential exists on this
    # controller, so NONE of the methods below has ever run against a live
    # Lambda account; each says so. What the fixtures can prove is shape,
    # arithmetic and refusal behaviour. What they cannot prove is that Lambda
    # answers the way its own schema says it does.
    # ======================================================================

    # -- what did it cost, and whose clock says so -------------------------
    def server_time_evidence(
            self, *, max_clock_delta_seconds: float = 30,
            max_evidence_age_seconds: float = 30) -> Dict[str, Any]:
        """The provider's own clock, from the last authenticated read.

        UNVERIFIED against a live Lambda account: no credential on this box.
        The `Date` header is HTTP/1.1-mandatory and the parse is strict-GMT
        only, so a proxy that rewrites or omits it refuses rather than
        silently substituting our clock for theirs.

        Lambda gives a clock but NO provider-enforced deadline: its launch
        request has no `terminateAfter`. So even with this evidence the
        guarantee is "provider-attested clock, controller-enforced deadline,
        on-instance watchdog as the real backstop" -- weaker than RunPod's,
        and `provider_enforced_deadline_available` says so in the document
        rather than in a comment.
        """
        evidence = self._server_time
        if evidence is None:
            raise LambdaError(
                "Lambda server time is unavailable; an authenticated read "
                "(status(), list_lifecycle_resources(), "
                "chargeable_inventory()) must succeed before create")
        now = time.time()
        age = now - evidence["local_received_epoch"]
        if not math.isfinite(age) or age < -1 or age > max_evidence_age_seconds:
            raise LambdaError("Lambda server-time evidence is stale")
        delta = evidence["local_minus_server_seconds"]
        if abs(delta) > max_clock_delta_seconds:
            raise LambdaError(
                "local UTC differs from Lambda server UTC by more than %.0fs; "
                "Lambda enforces no termination deadline of its own, so a "
                "skewed controller clock moves the only deadline there is"
                % max_clock_delta_seconds)
        out = dict(evidence)
        out.update({
            "checked_at_epoch": now,
            "evidence_age_seconds": age,
            "max_clock_delta_seconds": float(max_clock_delta_seconds),
            "max_evidence_age_seconds": float(max_evidence_age_seconds),
            "provider_enforced_deadline_available": False,
        })
        return out

    def ssh_host_ed25519_fingerprint(
            self, machine_id: Any, *, timeout: float = 0) -> Dict[str, Any]:
        """The ED25519 host key this controller PINNED for that instance.

        UNVERIFIED against a live Lambda account: no credential on this box,
        and the cloud-init behaviour described below has not been observed on
        a Lambda image.

        This method is not optional on Lambda and it is not cosmetic. Lambda
        execution rides `ssh` (SSHTransport shells out; there is no
        authenticated provider exec channel), so first contact is
        unauthenticated unless the host key is known beforehand -- and the HF
        token is uploaded over that session. An unauthenticated first contact
        would transit a credential through a host nobody verified.

        RunPod authenticates its key by reading the pod's own boot log through
        an authenticated endpoint. Lambda publishes no console and no log
        endpoint at all: OpenAPI 1.10.0 exposes `/instances`,
        `/instances/{id}`, `/instance-types`, `/file-systems`, `/ssh-keys`,
        `/images`, `/regions`, `/firewall-*`, `/audit-events`, `/tickets` and
        the three `instance-operations`, and nothing that returns output from
        inside a box. So there is nothing to read.

        What Lambda has instead is `user_data`. `prepare_safe_create`
        generates a fresh ED25519 host key and pins it through cloud-init, so
        the expected fingerprint is known BEFORE the instance exists and is
        authenticated by construction: only a box that received our launch
        request -- authenticated with our API key over TLS -- can present it.
        `verify_host_key(id, fingerprint)` then proves the live host presents
        exactly that key and writes the per-attempt known_hosts file.

        `timeout` is accepted for signature parity and unused: there is no
        remote log to wait for, and a pin is either present or it is not.
        """
        wanted = _provider_id(machine_id)
        pin = self._host_key_pins.get(wanted)
        if pin is None:
            raise LambdaError(
                "no pinned ED25519 host key for Lambda instance %s. Lambda "
                "publishes no console or log endpoint (OpenAPI %s), so a host "
                "key can only be authenticated by pinning it at launch: "
                "create through prepare_safe_create() / "
                "submit_prepared_create(), which ship the key in cloud-init "
                "user_data. For an instance created outside this controller, "
                "read its fingerprint from the Lambda dashboard and pass it to "
                "verify_host_key() by hand -- never accept a bare keyscan, "
                "because the HF token rides that session."
                % (wanted, OPENAPI_VERSION))
        return {
            "schema": "fidelity-suite/lambda-host-key-pin-evidence.v1",
            "provider": "lambda",
            "provider_id": wanted,
            "algorithm": "ssh-ed25519",
            "fingerprint": pin["fingerprint"],
            "public_key": pin["public_key"],
            "source": "cloud-init user_data pinned at launch",
            "pinned_at_utc": pin["pinned_at_utc"],
            "launch_request_identity_sha256": pin["request_identity_sha256"],
            "provider_log_endpoint": None,
            "trust_on_first_use": False,
        }

    def billing_history(self, machine_id: Any, *, start_time: str,
                        end_time: str, bucket_size: str = "hour",
                        instance_type_name: Optional[str] = None
                        ) -> Dict[str, Any]:
        """The provider's published rate over an exact window. UNPRICED.

        UNVERIFIED against a live Lambda account: no credential on this box.

        There is no billing or usage endpoint in the official API (OpenAPI
        1.10.0 publishes none; the invoice is dashboard-only). Returning
        something shaped like RunPod's `/billing/pods` response would be a
        fabricated receipt.

        So this returns exactly the facts that ARE official -- the instance
        type's published `price_cents_per_hour`, the exact window, and the
        window's length -- and deliberately does NOT multiply them into a
        dollar figure. An hourly-billed residual the provider never priced
        must stay unpriced: a computed cost that looks settled is worse than
        an honest gap. `records` is empty because Lambda publishes none, and
        `unreconcilable_by_provider` says why.

        Lambda sells no spot capacity and runs no bidding, so the published
        on-demand rate is the rate that charges -- unlike Vast, where a real
        2026-09-06 T4 contract billed $0.16667/h against a $0.13556 ask. Read
        what bills, never what was listed; on Lambda those coincide, and that
        is a property of the provider, not an assumption of this code.
        """
        wanted = _provider_id(machine_id)
        start = _exact_utc(start_time, "start_time")
        end = _exact_utc(end_time, "end_time")
        start_epoch = calendar.timegm(time.strptime(start, "%Y-%m-%dT%H:%M:%SZ"))
        end_epoch = calendar.timegm(time.strptime(end, "%Y-%m-%dT%H:%M:%SZ"))
        if end_epoch <= start_epoch:
            raise LambdaError("billing end_time must follow start_time")
        if bucket_size != "hour":
            raise LambdaError("billing bucket_size must be hour")
        type_name = instance_type_name
        if type_name is None:
            observed = self.get_lifecycle_resource(wanted)
            if observed is None:
                raise LambdaError(
                    "Lambda instance %s is no longer listed and no "
                    "instance_type_name was supplied, so even its published "
                    "rate cannot be established. Pass the type recorded in "
                    "the lease." % wanted)
            type_name = observed.get("instance_type_name")
        if not isinstance(type_name, str) or not type_name.strip():
            raise LambdaError("Lambda instance type name is unknown")
        catalogue = self._get_data("/instance-types")
        if not isinstance(catalogue, dict):
            raise LambdaError(
                "Lambda instance-types did not return the documented object")
        entry = catalogue.get(type_name)
        if not isinstance(entry, dict):
            raise LambdaError(
                "Lambda no longer publishes instance type %r, so not even its "
                "rate can be read; the dashboard invoice is the only "
                "remaining cost authority" % type_name)
        cents = _positive_int(
            (entry.get("instance_type") or {}).get("price_cents_per_hour"),
            "Lambda price_cents_per_hour")
        return {
            "schema": "fidelity-suite/lambda-cost-evidence.v1",
            "provider": "lambda",
            "provider_id": wanted,
            "provider_authoritative": False,
            "unreconcilable_by_provider": True,
            "provider_billing_api": None,
            "provider_billing_api_note":
                "no billing or usage endpoint exists in Lambda Cloud API v1 "
                "(OpenAPI %s); the account invoice is dashboard-only"
                % OPENAPI_VERSION,
            "query": {"instance_id": wanted, "start_time": start,
                      "end_time": end, "bucket_size": bucket_size},
            "records": [],
            "instance_type_name": type_name,
            "published_price_cents_per_hour": cents,
            "window_seconds": end_epoch - start_epoch,
            "cost_usd": None,
            "cost_unpriced_reason":
                "the provider published no charge for this resource and this "
                "code will not compute one: a derived total that looks "
                "settled is worse than an honest gap",
            "rate_source": "GET %s/instance-types -> "
                           "instance_type.price_cents_per_hour" % API,
            "retrieved_at_utc": _utc(time.time()),
        }

    def reconcile_billing(self, lease: Dict[str, Any], *,
                          now: Optional[float] = None) -> Dict[str, Any]:
        """A post-absence, independently stable closure that says UNSETTLED.

        UNVERIFIED against a live Lambda account: no credential on this box.

        The stability discipline is RunPod's: the closure is taken only after
        absence has been proven and a 300 s stabilization window has passed,
        and it is retrieved twice and compared, so a moving number is a
        refusal rather than a seal. What differs is the verdict. Lambda
        publishes no per-resource bill, so `settled` is false, the cost stays
        unpriced, and `unreconcilable_by_provider` is true -- the same shape
        JarvisLabs uses, for the same reason. Cost reconciliation protects the
        OPERATOR; the registry publishes no cost field, so this is a stated
        gap in the lease rather than a bar on the measurement.
        """
        ids = sorted({str(value) for value
                      in lease.get("provider_resource_ids") or []
                      if str(value).strip()})
        if not ids:
            raise LambdaError(
                "Lambda cost closure needs at least one exact instance id")
        create = lease.get("create") or {}
        start = _exact_utc(create.get("pre_create_observed_at"),
                           "lease pre_create_observed_at")
        absence = [
            item for item in lease.get("history") or []
            if item.get("to") == "ABSENCE_CONFIRMED"
            or item.get("event") == "EXACT_IDS_ABSENT_FROM_COMPLETE_LISTING"]
        if not absence:
            raise LambdaError("lease has no provider-absence event")
        end = _exact_utc(absence[-1].get("at"), "lease absence time")
        absence_epoch = calendar.timegm(time.strptime(end, "%Y-%m-%dT%H:%M:%SZ"))
        stabilization_seconds = 300
        instant = time.time() if now is None else float(now)
        if instant - absence_epoch < stabilization_seconds:
            raise LambdaError(
                "Lambda cost closure remains inside the %d-second "
                "post-absence stabilization window" % stabilization_seconds)
        type_name = create.get("instance_type_name") or create.get("gpu_type")

        def retrieve() -> List[Dict[str, Any]]:
            return [self.billing_history(
                identifier, start_time=start, end_time=end,
                instance_type_name=type_name) for identifier in ids]

        def closure(histories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            stripped = json.loads(json.dumps(
                histories, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False))
            for history in stripped:
                history.pop("retrieved_at_utc", None)
            return stripped

        first, second = retrieve(), retrieve()
        first_closure, second_closure = closure(first), closure(second)
        if first_closure != second_closure:
            raise LambdaError(
                "Lambda's published rate changed between independent "
                "retrievals; a rate read across a price change is not a "
                "stable closure")
        return {
            "reconciled": False,
            "settled": False,
            "unreconcilable_by_provider": True,
            "provider": "lambda",
            "provider_resource_ids": ids,
            "cost_snapshot": second,
            "total_amount": None,
            "pending_reason":
                "Lambda publishes no per-resource billing endpoint (OpenAPI "
                "%s), so no provider statement exists to reconcile against, "
                "and this code will not substitute arithmetic of its own"
                % OPENAPI_VERSION,
            "remedy": [
                "read the account invoice at cloud.lambda.ai for the billing "
                "period and record it as the authoritative operator cost",
                "the measurement itself is unaffected: the registry publishes "
                "no cost field",
            ],
            "evidence": {
                "schema": "fidelity-suite/lambda-cost-stabilization.v1",
                "absence_confirmed_at": end,
                "minimum_stabilization_seconds": stabilization_seconds,
                "closure_sha256": hashlib.sha256(json.dumps(
                    second_closure, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False).encode("utf-8")).hexdigest(),
                "retrieval_id": secrets.token_hex(12),
                "retrieved_at_utc": _utc(time.time()),
            },
        }

    # -- is anything of mine still alive? ----------------------------------
    @staticmethod
    def _lifecycle_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """One `Instance` from the official schema, flattened to exact fields."""
        if not isinstance(row, dict):
            raise LambdaError("Lambda instance listing contains a non-object row")
        instance_type = row.get("instance_type")
        if instance_type is not None and not isinstance(instance_type, dict):
            raise LambdaError("Lambda instance_type is not an object")
        instance_type = instance_type or {}
        specs = instance_type.get("specs")
        if specs is not None and not isinstance(specs, dict):
            raise LambdaError("Lambda instance_type.specs is not an object")
        specs = specs or {}
        region = row.get("region")
        if region is not None and not isinstance(region, dict):
            raise LambdaError("Lambda region is not an object")
        image = row.get("image")
        if image is not None and not isinstance(image, dict):
            raise LambdaError("Lambda image is not an object")
        status = row.get("status")
        if not isinstance(status, str) or not status:
            raise LambdaError("Lambda instance row has no string status")
        key_names = row.get("ssh_key_names")
        if key_names is not None and (
                not isinstance(key_names, list)
                or any(not isinstance(item, str) for item in key_names)):
            raise LambdaError("Lambda ssh_key_names is not a list of strings")
        fs_names = row.get("file_system_names")
        if fs_names is not None and (
                not isinstance(fs_names, list)
                or any(not isinstance(item, str) for item in fs_names)):
            raise LambdaError(
                "Lambda file_system_names is not a list of strings")
        mounts = row.get("file_system_mounts")
        if mounts is not None and not isinstance(mounts, list):
            raise LambdaError("Lambda file_system_mounts is not a list")
        return {
            "id": _provider_id(row.get("id")),
            "name": row.get("name"),
            "status": status,
            # An unrecognised status counts as LIVE. `terminating` still has a
            # machine attached and `preempted` is not `terminated` on a
            # provider that sells no spot capacity, so neither proves absence.
            "live": status not in LAMBDA_GONE_STATUSES,
            "known_status": status in LAMBDA_INSTANCE_STATUSES,
            "listed": True,
            "instance_type_name": instance_type.get("name"),
            "gpu_description": instance_type.get("gpu_description"),
            "gpu_count": specs.get("gpus"),
            "vcpus": specs.get("vcpus"),
            "memory_gib": specs.get("memory_gib"),
            "storage_gib": specs.get("storage_gib"),
            "architecture": instance_type.get("architecture"),
            "price_cents_per_hour": instance_type.get("price_cents_per_hour"),
            "region_name": (region or {}).get("name"),
            "ssh_key_names": sorted(key_names or []),
            "file_system_names": sorted(fs_names or []),
            "file_system_mounts": list(mounts or []),
            "image_id": (image or {}).get("id"),
            "image_family": (image or {}).get("family"),
            "hostname": row.get("hostname"),
            "ip": row.get("ip"),
            "private_ip": row.get("private_ip"),
            # Lambda has no provider-side deadline field at all. Recorded as
            # an explicit None so a caller cannot read its absence as "not
            # checked" (see validate_safe_resource_binding).
            "terminate_after": None,
            "raw": row,
        }

    def _instance_rows(self) -> List[Dict[str, Any]]:
        data = self._get_data("/instances")
        if not isinstance(data, list):
            raise LambdaError("Lambda instance listing returned non-list data")
        rows, seen = [], set()
        for row in data:
            parsed = self._lifecycle_row(row)
            if parsed["id"] in seen:
                raise LambdaError(
                    "Lambda instance listing repeats id %s" % parsed["id"])
            seen.add(parsed["id"])
            rows.append(parsed)
        return rows

    def list_lifecycle_resources(self) -> List[Dict[str, Any]]:
        """Complete exact-id rows whose status does not prove absence.

        UNVERIFIED against a live Lambda account: no credential on this box.

        `terminated` is the only status filtered out, because it is the only
        one that proves nothing is attached. Statuses this code does not
        recognise are KEPT and flagged `known_status: false`: an unknown state
        read as absence is how an instance leaks.
        """
        return [row for row in self._instance_rows() if row["live"]]

    def get_lifecycle_resource(self, provider_id: Any) -> Optional[Dict[str, Any]]:
        """Exact-id detail through `GET /instances/{id}`; names are not ids.

        UNVERIFIED against a live Lambda account: no credential on this box.

        A 404 means the instance is genuinely unknown to the provider and
        returns None. Every other status raises: "I could not tell" must never
        read as "it is gone".
        """
        wanted = _provider_id(provider_id)
        try:
            data = self._get_data(
                "/instances/%s" % urllib.parse.quote(wanted, safe=""))
        except LambdaError as exc:
            if getattr(exc, "status", None) == 404:
                return None
            raise
        return self._lifecycle_row(data)

    def list_network_volumes(self) -> List[Dict[str, Any]]:
        """Persistent filesystems, which OUTLIVE every instance on Lambda.

        UNVERIFIED against a live Lambda account: no credential on this box.

        This is real work here, not a stub: a Lambda filesystem survives
        terminate, keeps billing storage, and is exactly the persistence that
        makes a preempted box cheap to resume. An empty list from this method
        is a claim, so it is only ever returned when the endpoint answered.

        Note the API's own inconsistency, which is read and not corrected:
        listing is `GET /file-systems` (hyphen) while create/delete are
        `POST /filesystems` and `DELETE /filesystems/{id}`.
        """
        data = self._get_data("/file-systems")
        if not isinstance(data, list):
            raise LambdaError(
                "Lambda filesystem listing returned non-list data")
        volumes, seen = [], set()
        for row in data:
            if not isinstance(row, dict):
                raise LambdaError(
                    "Lambda filesystem listing contains a non-object row")
            volume_id = _provider_id(row.get("id"), "Lambda filesystem id")
            if volume_id in seen:
                raise LambdaError(
                    "Lambda filesystem listing repeats id %s" % volume_id)
            seen.add(volume_id)
            name = row.get("name")
            if not isinstance(name, str) or not name:
                raise LambdaError("Lambda filesystem has no string name")
            in_use = row.get("is_in_use")
            if not isinstance(in_use, bool):
                raise LambdaError(
                    "Lambda filesystem is_in_use is not an exact bool")
            region = row.get("region")
            if not isinstance(region, dict) or not isinstance(
                    region.get("name"), str):
                raise LambdaError("Lambda filesystem region is not an object")
            created = _iso8601_epoch(
                row.get("created"), "Lambda filesystem created")
            used = row.get("bytes_used")
            if used is not None:
                used = _exact_int(used, "Lambda filesystem bytes_used")
            volumes.append({
                "id": volume_id,
                "name": name,
                "region_name": region["name"],
                "mount_point": row.get("mount_point"),
                "is_in_use": in_use,
                "created_utc": _utc(created),
                "bytes_used": used,
                # The API publishes no per-filesystem rate; persistent storage
                # is priced per GB-month on the pricing page only. Left None
                # rather than filled in with a number from a web page.
                "cost_per_hr": None,
                "persists_after_instance_termination": True,
                "raw": row,
            })
        return volumes

    def chargeable_inventory(self) -> Dict[str, Any]:
        """Instances plus filesystems, with EXPLICIT completeness.

        UNVERIFIED against a live Lambda account: no credential on this box.

        A partial inventory cannot prove no leak, so each family carries its
        own `complete` flag and its source endpoint, and a family that could
        not be established is named in `unknown_families` with the refusal
        text. The reaper treats that as an OUTAGE and declines to conclude
        absence, which is the correct behaviour -- fabricating
        `complete: true` would turn an outage into a false absence proof.

        Both families are load-bearing on Lambda. Filesystems outlive
        instances, so an inventory of instances alone would prove nothing
        about an orphaned filesystem still being charged for.
        """
        families: Dict[str, Dict[str, Any]] = {}
        instances_source = "GET %s/instances" % API
        try:
            families["instances"] = {
                "complete": True,
                "source": instances_source,
                "resources": self._instance_rows(),
            }
        except LambdaError as exc:
            families["instances"] = {
                "complete": False,
                "source": instances_source,
                "resources": [],
                "unknown": redact(str(exc)),
            }
        volumes_source = "GET %s/file-systems" % API
        try:
            families["network_volumes"] = {
                "complete": True,
                "source": volumes_source,
                "resources": self.list_network_volumes(),
            }
        except LambdaError as exc:
            families["network_volumes"] = {
                "complete": False,
                "source": volumes_source,
                "resources": [],
                "unknown": redact(str(exc)),
            }
        unknown = sorted(name for name, family in families.items()
                         if not family["complete"])
        return {
            "schema": "fidelity-suite/lambda-chargeable-inventory.v1",
            "provider": "lambda",
            "observed_at_utc": _utc(time.time()),
            "complete": not unknown,
            "unknown_families": unknown,
            "families": families,
        }

    # -- is this the thing I asked for? ------------------------------------
    def validate_safe_resource_binding(
            self, provider_id: Any, *, expected_name: str,
            instance_type_name: str, region_name: str,
            ssh_key_names: Any, storage_gib: int, gpu_count: int,
            terminate_after: str,
            file_system_names: Any = ()) -> Dict[str, Any]:
        """Fail unless the live exact-id instance is the one requested.

        UNVERIFIED against a live Lambda account: no credential on this box.

        The compared fields are LAMBDA's, not RunPod's: there is no
        `gpuTypeId`, no `secureCloud`, no `imageName` and no `volumeInGb` here.
        What Lambda returns for an instance is its instance type (with the
        type's own specs), its region, the SSH key NAMES bound at launch, and
        the filesystems mounted -- and those are what this checks.

        Two Lambda-specific rules:

        * **SSH keys bind by NAME at create.** A name is not a key, so the
          expectation is the exact name set, and `prepare_safe_create` is what
          proves the name maps to the public key we hold. Both halves are
          needed: this one catches a launch that bound a different key, that
          one catches a name that was re-pointed at someone else's key.
        * **Filesystems must be absent.** They outlive the instance, so an
          unexpected mount is a chargeable resource the lease does not know
          about.

        `terminate_after` is validated for shape and then reported as
        `terminate_after_observable: false`, because Lambda has no such field.
        A caller must not read a missing provider deadline as an enforced one.
        """
        observed = self.get_lifecycle_resource(provider_id)
        if observed is None:
            raise LambdaError(
                "created Lambda instance %s is absent from the exact-id "
                "endpoint" % provider_id)
        expected_deadline = _exact_utc(terminate_after, "terminate_after")
        expected_keys = tuple(sorted(str(name) for name in ssh_key_names or ()))
        if not expected_keys:
            raise LambdaError(
                "safe Lambda binding requires the exact SSH key name(s) the "
                "launch was made with")
        expected_fs = tuple(sorted(str(name) for name in file_system_names or ()))
        expected = {
            "name": str(expected_name),
            "instance_type_name": str(instance_type_name),
            "region_name": str(region_name),
            "ssh_key_names": list(expected_keys),
            "file_system_names": list(expected_fs),
            "gpu_count": _positive_int(gpu_count, "gpu_count"),
            "storage_gib": _positive_int(storage_gib, "storage_gib"),
        }
        problems = []
        for key, value in expected.items():
            actual = observed.get(key)
            if key in ("gpu_count", "storage_gib"):
                try:
                    actual = _exact_int(actual, key)
                except LambdaError:
                    problems.append(
                        "%s is not an exact integer, observed %r"
                        % (key, observed.get(key)))
                    continue
            if actual != value:
                problems.append("%s expected %r, observed %r"
                                % (key, value, observed.get(key)))
        if expected_fs == () and observed.get("file_system_mounts"):
            problems.append(
                "file_system_mounts must be empty, observed %r"
                % (observed.get("file_system_mounts"),))
        if not observed.get("live"):
            problems.append("status %r does not describe a live instance"
                            % observed.get("status"))
        if not observed.get("known_status"):
            problems.append(
                "status %r is not in the documented InstanceStatus enum %s"
                % (observed.get("status"), ", ".join(LAMBDA_INSTANCE_STATUSES)))
        rate = observed.get("price_cents_per_hour")
        try:
            _positive_int(rate, "Lambda live price_cents_per_hour")
        except LambdaError:
            problems.append(
                "price_cents_per_hour must be a known positive integer, "
                "observed %r" % (rate,))
        if problems:
            raise LambdaError("Lambda post-create identity mismatch: %s"
                              % "; ".join(problems))
        return {
            "provider_id": str(provider_id),
            "passed": True,
            "expected": dict(expected, terminate_after=expected_deadline),
            "observed": observed,
            "terminate_after_observable": False,
            "provider_enforced_deadline": False,
            "deadline_note":
                "Lambda's launch request has no terminateAfter field (OpenAPI "
                "%s), so this deadline is enforced only by the controller and "
                "the on-instance watchdog" % OPENAPI_VERSION,
        }

    def attest_live_resource(
            self, provider_id: Any, *, expected_gpu_model: str,
            expected_vram_bytes: int, min_vcpu: int, min_ram_gb: int,
            storage_gib: int, root_available_bytes_minimum: int,
            run_root_available_bytes_minimum: int,
            expected_gpu_count: int = 1,
            free_vram_bytes_minimum: Optional[int] = None,
            expected_ssh_key_names: Any = ()) -> Dict[str, Any]:
        """Read-only SSH proof that the box is the DEVICE the root wants.

        UNVERIFIED against a live Lambda account: no credential on this box.
        The probe itself is stdlib-only and has not been run on a Lambda image.

        This is the scientific gate, and it is the reason provider parity is
        legitimate at all: what a comparison binds is the DEVICE MODEL and the
        rebuilt stack, not the company that rented the card (two A100s in two
        clouds agree bitwise; an H200 sits 2.973e-04 nats away --
        docs/ARCHITECTURE-DETERMINISM.md). So the card has to be proven, not
        assumed from a catalogue name.

        A provider STATUS is a claim about the provider's intent and never
        evidence of reachability, so nothing here is inferred from `active`:
        the facts come from the box.

        Four deliberate differences from the RunPod attestation:

        * **Filesystem roles are VM roles.** `/` and the RUN root, not
          container-vs-workspace, and they may legitimately be the same
          device -- a Lambda instance has one disk. The run root's
          WRITABILITY is proven, because that is the check whose absence
          killed a paid gpu_1x_gh200 two minutes into a rental.
        * **torch may not exist yet.** The stack is rebuilt on the VM by
          bootstrap_measure.sh. Device identity comes from nvidia-smi, which
          is driver-level, and is a hard gate; `cuda_usable` is a gate only
          when a torch was actually found. `cuda_probe_available` and
          `revalidate_cuda_after_bootstrap` say which case happened, so a
          caller can re-attest after bootstrap instead of guessing.
        * **GPU count is a parameter.** Lambda sells 1x and 8x types under
          one naming scheme; demanding exactly one would refuse a correct 8x
          box and accept a mislabelled one.
        * **FREE VRAM is gated, not just total.** `free_vram_bytes_minimum`
          defaults to the same 90% of the expected card, so an oversubscribed
          GPU is refused by default rather than only when a caller remembers
          to ask, and `compute_apps` records which foreign processes hold it.
          Total-only checks pass a "24 GB" card with 23,424 of 24,564 MiB
          already taken by four other tenants' PIDs.

        WHAT THIS DOES NOT DO: it performs no TLS peer attestation. On a
        rented VM the host has root, and proving the box is talking to the
        real huggingface.co is a separate, EARLIER step (`bin/fidelity/
        tlsguard.py`, plus the pod-side check in bootstrap) that must hold
        before any credential is transported. That gap is stated here rather
        than filled with a field that would read as attested.
        """
        instance_id = _provider_id(provider_id)
        model = str(expected_gpu_model or "").strip()
        if not model:
            raise LambdaError("expected GPU model must be a nonempty string")
        expected_numbers = {
            "expected_vram_bytes": expected_vram_bytes,
            "min_vcpu": min_vcpu,
            "min_ram_gb": min_ram_gb,
            "storage_gib": storage_gib,
            "root_available_bytes_minimum": root_available_bytes_minimum,
            "run_root_available_bytes_minimum":
                run_root_available_bytes_minimum,
            "expected_gpu_count": expected_gpu_count,
        }
        for key, value in expected_numbers.items():
            _positive_int(value, key)
        expected_keys = tuple(sorted(str(n) for n in expected_ssh_key_names or ()))
        expected = dict(expected_numbers, gpu_model=model,
                        ssh_key_names=list(expected_keys),
                        run_root=self.run_base)
        command = (
            "python3 -c 'import base64,sys;"
            "exec(base64.b64decode(\"%s\").decode(\"utf-8\"))' %s"
            % (base64.b64encode(
                _LAMBDA_ATTEST_SCRIPT.encode("utf-8")).decode("ascii"),
               self.run_base))
        observed = None
        transport_error = None
        send_epoch = time.time()
        receive_epoch = send_epoch
        if self.dry:
            transport_error = "dry mode cannot attest a live resource"
            receive_epoch = time.time()
        else:
            try:
                observed = _strict_json_loads(
                    self.exec_stdout(instance_id, command, timeout=300),
                    "Lambda attestation output")
            except Exception as exc:                      # noqa: BLE001
                transport_error = redact(str(exc))[:500]
            finally:
                receive_epoch = time.time()
        round_trip = max(0.0, receive_epoch - send_epoch)
        remote_epoch = (observed.get("remote_time_epoch")
                        if isinstance(observed, dict) else None)
        remote_utc = (observed.get("remote_time_utc")
                      if isinstance(observed, dict) else None)
        remote_utc_epoch = None
        if isinstance(remote_utc, str):
            try:
                remote_utc_epoch = calendar.timegm(time.strptime(
                    _exact_utc(remote_utc, "remote attestation time"),
                    "%Y-%m-%dT%H:%M:%SZ"))
            except LambdaError:
                pass
        midpoint = send_epoch + round_trip / 2.0
        allowed_skew = 30.0 + round_trip
        skew = (abs(float(remote_epoch) - midpoint)
                if isinstance(remote_epoch, int)
                and not isinstance(remote_epoch, bool) else None)
        clock_ok = bool(skew is not None and remote_utc_epoch == remote_epoch
                        and skew <= allowed_skew)
        clock = {
            "controller_send_epoch": send_epoch,
            "controller_send_utc": _utc(send_epoch),
            "controller_receive_epoch": receive_epoch,
            "controller_receive_utc": _utc(receive_epoch),
            "round_trip_seconds": round_trip,
            "remote_time_epoch": remote_epoch,
            "remote_time_utc": remote_utc,
            "clock_skew_seconds": skew,
            "allowed_skew_seconds": allowed_skew,
            "within_bound": clock_ok,
        }
        failures: List[str] = []
        checks: Dict[str, bool] = {"remote_clock": clock_ok}
        cuda_probe_available = False
        if not isinstance(observed, dict):
            failures.append("live SSH attestation unavailable")
        else:
            exact_observed = {
                "remote_time_epoch", "remote_time_utc", "login_user", "uid",
                "logical_cpus", "memtotal_bytes", "nvidia_smi_exit_code",
                "nvidia_smi_error", "gpus", "cuda", "compute_apps",
                "filesystems", "run_root_write",
            }
            if set(observed) != exact_observed:
                failures.append("live attestation keys differ")
            for key in ("logical_cpus", "memtotal_bytes", "uid",
                        "nvidia_smi_exit_code"):
                if (isinstance(observed.get(key), bool)
                        or not isinstance(observed.get(key), int)):
                    failures.append("%s is not an exact integer" % key)
            checks["logical_cpu_floor"] = (
                isinstance(observed.get("logical_cpus"), int)
                and not isinstance(observed.get("logical_cpus"), bool)
                and observed["logical_cpus"] >= min_vcpu)
            checks["memory_floor"] = (
                isinstance(observed.get("memtotal_bytes"), int)
                and not isinstance(observed.get("memtotal_bytes"), bool)
                and observed["memtotal_bytes"] >= min_ram_gb * 10 ** 9)
            checks["login_user"] = observed.get("login_user") == self.ssh_user
            gpus = observed.get("gpus")
            checks["nvidia_gpu_count"] = (
                observed.get("nvidia_smi_exit_code") == 0
                and isinstance(gpus, list)
                and len(gpus) == expected_gpu_count)
            rows = gpus if checks["nvidia_gpu_count"] else []
            if rows and any(
                    set(row) != {"index", "name", "vram_bytes",
                                 "vram_used_bytes", "vram_free_bytes",
                                 "driver_version"} for row in rows):
                failures.append("nvidia-smi GPU keys differ")
            names = [row.get("name") for row in rows if isinstance(row, dict)]
            vrams = [row.get("vram_bytes") for row in rows
                     if isinstance(row, dict)]
            checks["gpu_model"] = bool(names) and all(
                isinstance(name, str)
                and name.strip().casefold() == model.casefold()
                for name in names)
            vram_floor = expected_vram_bytes * 9 // 10
            vram_ceiling = expected_vram_bytes * 11 // 10
            checks["gpu_vram"] = bool(vrams) and all(
                isinstance(value, int) and not isinstance(value, bool)
                and vram_floor <= value <= vram_ceiling for value in vrams)
            # TOTAL VRAM is what the card has; FREE VRAM is what this run can
            # use, and only the second is attestable. A "24 GB" 4090 with
            # 23,424 of 24,564 MiB held by four foreign PIDs passes every
            # total-based check and cannot run anything (DecoderParity, host
            # 434175, 2026-09-06). The floor defaults to the same 90% of the
            # expected card, so an oversubscribed card is refused by default
            # rather than only when a caller remembers to ask.
            free_floor = (expected_vram_bytes * 9 // 10
                          if free_vram_bytes_minimum is None
                          else _positive_int(free_vram_bytes_minimum,
                                             "free_vram_bytes_minimum"))
            expected["free_vram_bytes_minimum"] = free_floor
            frees = [row.get("vram_free_bytes") for row in rows
                     if isinstance(row, dict)]
            checks["gpu_free_vram"] = bool(frees) and all(
                isinstance(value, int) and not isinstance(value, bool)
                and value >= free_floor for value in frees)
            foreign = observed.get("compute_apps")
            if not isinstance(foreign, list) or any(
                    not isinstance(row, dict) for row in foreign):
                failures.append("compute-apps attestation is not a list of "
                                "objects")
                foreign = []
            # A foreign process on the card is not automatically fatal -- the
            # free-VRAM floor decides that -- but it must be RECORDED, because
            # "who else is on this GPU" is not knowable after the fact.
            checks["no_foreign_compute_apps"] = not foreign
            cuda = observed.get("cuda")
            if not isinstance(cuda, dict) or set(cuda) != {
                    "available", "usable", "count", "name", "vram_bytes",
                    "interpreter", "error"}:
                failures.append("CUDA attestation keys differ")
                cuda = {}
            cuda_probe_available = cuda.get("available") is True
            if cuda_probe_available:
                # A torch WAS found, so it must work and must agree with the
                # driver about which card this is.
                checks["cuda_usable"] = (
                    cuda.get("usable") is True
                    and cuda.get("count") == expected_gpu_count
                    and isinstance(cuda.get("name"), str)
                    and cuda["name"].strip().casefold() == model.casefold()
                    and isinstance(cuda.get("vram_bytes"), int)
                    and vram_floor <= cuda["vram_bytes"] <= vram_ceiling)
            filesystems = observed.get("filesystems")
            if not isinstance(filesystems, dict) or set(filesystems) != {
                    "root", "run_root"}:
                failures.append("filesystem attestation keys differ")
                filesystems = {}
            filesystem_keys = {
                "path", "mount_point", "fs_type", "source", "device",
                "total_bytes", "available_bytes"}
            for role in ("root", "run_root"):
                row = filesystems.get(role)
                if not isinstance(row, dict) or set(row) != filesystem_keys:
                    failures.append("%s filesystem keys differ" % role)
            root = filesystems.get("root", {})
            run_root = filesystems.get("run_root", {})
            # storage_gib is BINARY in the spec. 90% of it allows for the
            # filesystem's own overhead without allowing a wrong type.
            checks["root_disk_size"] = (
                isinstance(root.get("total_bytes"), int)
                and root["total_bytes"] >= storage_gib * GIB * 9 // 10)
            checks["root_available_bytes"] = (
                isinstance(root.get("available_bytes"), int)
                and not isinstance(root.get("available_bytes"), bool)
                and root["available_bytes"] >= root_available_bytes_minimum)
            checks["run_root_available_bytes"] = (
                isinstance(run_root.get("available_bytes"), int)
                and not isinstance(run_root.get("available_bytes"), bool)
                and run_root["available_bytes"]
                >= run_root_available_bytes_minimum)
            checks["run_root_path"] = (
                run_root.get("path") == os.path.realpath(self.run_base))
            write = observed.get("run_root_write")
            if not isinstance(write, dict) or set(write) != {
                    "writable", "error"}:
                failures.append("run-root write probe keys differ")
                write = {}
            checks["run_root_writable"] = write.get("writable") is True
            failures.extend(
                name for name, passed in sorted(checks.items()) if not passed)
        provider_record = {
            "region_name": None, "instance_type_name": None,
            "image_family": None, "image_id": None, "hostname": None,
            "ssh_key_names": None, "file_system_names": None, "error": None}
        try:
            listed = self.get_lifecycle_resource(instance_id)
            if listed is None:
                raise LambdaError("instance %s is not listed" % instance_id)
            for key in ("region_name", "instance_type_name", "image_family",
                        "image_id", "hostname"):
                value = listed.get(key)
                provider_record[key] = str(value) if value is not None else None
            provider_record["ssh_key_names"] = list(
                listed.get("ssh_key_names") or [])
            provider_record["file_system_names"] = list(
                listed.get("file_system_names") or [])
        except Exception as exc:                          # noqa: BLE001
            provider_record["error"] = "%s: %s" % (type(exc).__name__, exc)
        # Lambda binds keys BY NAME at create, so the binding is part of the
        # attestation rather than an afterthought: a box reachable with our
        # key but carrying somebody else's name too is not the box we asked
        # for.
        if expected_keys:
            checks["ssh_key_binding"] = (
                provider_record["error"] is None
                and tuple(provider_record["ssh_key_names"] or ()) == expected_keys)
            if not checks["ssh_key_binding"]:
                failures.append("ssh_key_binding")
        # WHO ATTESTS WHAT. Every `observed` field is the INSTANCE's own
        # report of itself, so this document proves "the box we
        # authenticated said these things", not "these things are true of
        # the hardware". A hostile host with root can lie about nvidia-smi
        # exactly as it can lie about erasing a secret -- a postcondition
        # evaluated by the party it constrains is not independent evidence.
        # What makes the report attributable at all is that it arrives over
        # the SSH channel whose host key was pinned at launch, and what is
        # independent of the box is `provider_record`, read from the Lambda
        # API over TLS. The split is stated in the document rather than left
        # for a reader to assume, because "attested" is read as evidence.
        document = {
            "schema": "fidelity-suite/lambda-live-attestation.v2",
            "provider": "lambda", "provider_id": instance_id,
            "observed_at_utc": clock["controller_receive_utc"],
            "clock": clock,
            "attested_by":
                "the instance itself, over the SSH channel whose ED25519 host "
                "key was pinned at launch (see "
                "ssh_host_ed25519_fingerprint)",
            "box_self_reported_fields": sorted(
                observed.keys() if isinstance(observed, dict) else []),
            "independent_of_the_box": "provider_record, read from the Lambda "
                                      "API over TLS",
            "independently_verifiable": False,
            "provider_record": provider_record,
            "expected": expected, "observed": observed,
            "transport_error": transport_error,
            "cuda_probe_available": cuda_probe_available,
            "revalidate_cuda_after_bootstrap": not cuda_probe_available,
            "checks": checks, "failures": sorted(set(failures)),
            "ok": bool(not failures and transport_error is None
                       and checks and all(checks.values())),
        }
        return _attestation_seal(document)

    def _validated_ssh_public_key(self) -> str:
        """The exact unattended SSH identity, proven to match its private key.

        Same discipline as the RunPod backend: no symlinks, owner-only
        permissions, current-user ownership, and `ssh-keygen -y` over the
        private key must reproduce the .pub. A launch that binds a key we
        cannot use is a box that bills and cannot be reached.
        """
        private_path = os.path.abspath(os.path.expanduser(self.ssh_key))
        public_path = private_path + ".pub"
        private_fd = None
        public_fd = None
        try:
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            private_fd = os.open(private_path, os.O_RDONLY | nofollow)
            public_fd = os.open(public_path, os.O_RDONLY | nofollow)
            private = os.fstat(private_fd)
            public = os.fstat(public_fd)
            if not stat.S_ISREG(private.st_mode):
                raise LambdaError(
                    "SSH private key must be a regular non-symlink file")
            if stat.S_IMODE(private.st_mode) & 0o077:
                raise LambdaError(
                    "SSH private key must not grant group/other access")
            if hasattr(os, "getuid") and private.st_uid != os.getuid():
                raise LambdaError(
                    "SSH private key must be owned by the current user")
            if not stat.S_ISREG(public.st_mode):
                raise LambdaError(
                    "SSH public key must be a regular non-symlink file")
            with os.fdopen(public_fd, encoding="utf-8") as handle:
                public_fd = None
                supplied = _canonical_public_key(handle.read())
            try:
                derived = subprocess.run(
                    ["ssh-keygen", "-y", "-f", "/proc/self/fd/%d" % private_fd],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, timeout=15,
                    pass_fds=(private_fd,))
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise LambdaError(
                    "cannot verify SSH private/public key match: %s" % exc)
            if derived.returncode != 0:
                raise LambdaError("SSH private key is unreadable to ssh-keygen")
            if _canonical_public_key(derived.stdout) != supplied:
                raise LambdaError("SSH .pub does not match the private key")
            return supplied
        except LambdaError:
            raise
        except (OSError, UnicodeError) as exc:
            raise LambdaError(
                "SSH key pair is unavailable or invalid: %s" % exc)
        finally:
            if public_fd is not None:
                os.close(public_fd)
            if private_fd is not None:
                os.close(private_fd)

    def _generate_host_key_pin(self) -> Dict[str, str]:
        """Generate the ED25519 host key this launch will PIN via user_data.

        The private half is registered as a secret so `redact()` scrubs it,
        and it lives only in the frozen launch body -- never in a receipt, a
        lease, `to_dict()` or a log. It is a per-instance throwaway whose only
        job is to make the host's identity known before the host exists.
        """
        with tempfile.TemporaryDirectory(prefix="fidelity-hostkey-") as scratch:
            os.chmod(scratch, 0o700)
            path = os.path.join(scratch, "ssh_host_ed25519_key")
            try:
                generated = subprocess.run(
                    ["ssh-keygen", "-q", "-t", "ed25519", "-N", "",
                     "-C", "fidelity-pinned-host-key", "-f", path],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, timeout=30)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise LambdaError("cannot generate a host key: %s" % exc)
            if generated.returncode != 0:
                raise LambdaError(
                    "ssh-keygen refused to generate a host key: %s"
                    % redact(generated.stderr.strip()[:200]))
            with open(path, encoding="utf-8") as handle:
                private = handle.read()
            with open(path + ".pub", encoding="utf-8") as handle:
                public = _canonical_public_key(handle.read())
        register_secret(private)
        return {
            "private_key": private,
            "public_key": public,
            "fingerprint": _fingerprint_of_public_key(public),
        }

    @staticmethod
    def _host_key_user_data(pin: Dict[str, str]) -> str:
        """cloud-init user_data that installs exactly the pinned host key.

        UNVERIFIED on a live Lambda image: the `ssh_keys` / `ssh_deletekeys` /
        `ssh_genkeytypes` modules are standard cloud-init and Lambda images
        are Ubuntu, but that this runs before sshd first serves a key has not
        been observed here. It is the first thing a paid Lambda run must
        confirm, and until it is confirmed a Lambda run's host key is not
        authenticated.
        """
        indented = "\n".join(
            "      " + line for line in pin["private_key"].splitlines())
        return (
            "#cloud-config\n"
            "ssh_deletekeys: true\n"
            "ssh_genkeytypes: [ed25519]\n"
            "ssh_keys:\n"
            "  ed25519_private: |\n"
            "%s\n"
            "  ed25519_public: %s\n"
            % (indented, pin["public_key"]))

    def prepare_safe_create(self, **kw) -> PreparedLambdaCreate:
        """Build and freeze the launch request before any mutation.

        UNVERIFIED against a live Lambda account: no credential on this box.

        Everything that can be refused is refused here, where nothing is
        billing yet: the type's real disk, the region's live capacity, the GPU
        count, the deadline, the local key pair, and -- the Lambda-specific
        one -- that the SSH key NAME the launch will bind resolves, on the
        account, to the public key we actually hold. Lambda accepts no inline
        key, so a name is the only handle, and a name silently re-pointed at
        another key would produce a billing box nobody can log into.

        The frozen request also carries the cloud-init user_data that pins the
        instance's ED25519 host key (see `ssh_host_ed25519_fingerprint`).
        """
        forbidden = {
            "file_system_names": "a Lambda filesystem outlives the instance, "
                                 "so the safe profile mounts none",
            "file_system_mounts": "a Lambda filesystem outlives the instance, "
                                  "so the safe profile mounts none",
            "env": "the safe Lambda profile is SSH-driven and takes no "
                   "provider environment",
            "docker_cmd": "Lambda instances are VMs; there is no container "
                          "command to launch",
            "user_data": "user_data is reserved for the pinned host key and "
                         "may not be supplied by a caller",
            "firewall_rulesets": "firewall rulesets are an account-level "
                                 "decision, not a per-run one",
        }
        used = sorted(key for key in forbidden if kw.get(key))
        if used:
            raise LambdaError(
                "safe Lambda profile refuses %s"
                % "; ".join("%s (%s)" % (key, forbidden[key]) for key in used))
        # The key-name refusals above catch the CARRIERS. This catches the
        # VALUE wherever it was smuggled -- a token in `name`, in a tag, in a
        # hostname -- because the guard scans the payload rather than a list
        # of field names somebody remembered to extend.
        self._refuse_credential_payload(dict(kw), operation="prepare-create")
        if kw.get("spot", False) is not False:
            raise LambdaError(
                "safe Lambda profile requires spot exactly false; Lambda "
                "sells no spot capacity at all")
        if kw.get("offer", "on-demand") != "on-demand":
            raise LambdaError(
                "safe Lambda profile requires offer exactly on-demand")
        name = str(kw.get("name") or "").strip()
        if not name or _LAMBDA_NAME_RE.fullmatch(name) is None:
            raise LambdaError(
                "safe Lambda create requires an exact lease name of at most "
                "64 printable characters (the API's own limit)")
        itype = str(kw.get("gpu_type") or kw.get("instance_type") or "").strip()
        if not itype:
            raise LambdaError(
                "safe Lambda create requires gpu_type (a Lambda instance type)")
        want_gpus = _positive_int(kw.get("num_gpus", 1), "num_gpus")
        want_disk = _positive_int(
            kw.get("storage") or kw.get("storage_gb"), "storage_gb")
        terminate_after = kw.get("terminate_after")
        if terminate_after is None:
            raise LambdaError(
                "safe Lambda create requires terminate_after: Lambda enforces "
                "no deadline of its own, so the controller must state one and "
                "the watchdog must be built against it")
        terminate_after = _exact_utc(terminate_after, "terminate_after")
        deadline_epoch = calendar.timegm(
            time.strptime(terminate_after, "%Y-%m-%dT%H:%M:%SZ"))
        if deadline_epoch - time.time() < MIN_CREATE_SETUP_SECONDS:
            raise LambdaError(
                "Lambda terminate_after must be at least %d seconds in the "
                "future" % MIN_CREATE_SETUP_SECONDS)
        catalogue = self._get_data("/instance-types")
        if not isinstance(catalogue, dict):
            raise LambdaError(
                "Lambda instance-types did not return the documented object")
        entry = catalogue.get(itype)
        if not isinstance(entry, dict):
            raise LambdaError(
                "Lambda publishes no instance type %r; `gpus()` lists what "
                "exists" % itype)
        instance_type = entry.get("instance_type") or {}
        specs = instance_type.get("specs") or {}
        have_disk = _positive_int(
            specs.get("storage_gib"), "instance type storage_gib")
        have_gpus = _positive_int(specs.get("gpus"), "instance type gpus")
        cents = _positive_int(
            instance_type.get("price_cents_per_hour"),
            "instance type price_cents_per_hour")
        if want_disk > have_disk:
            raise LambdaError(
                "instance type %s provides ~%d GiB of local disk and this plan "
                "needs %d GB. Lambda disk is fixed per type and cannot be "
                "grown, so this would fail during fetch, after billing starts."
                % (itype, have_disk, want_disk))
        if want_gpus != have_gpus:
            raise LambdaError(
                "instance type %s has %d GPU(s) and the plan asked for %d; "
                "Lambda types are fixed shapes, so this is a plan error rather "
                "than a launch option" % (itype, have_gpus, want_gpus))
        regions = entry.get("regions_with_capacity_available")
        if not isinstance(regions, list):
            raise LambdaError(
                "Lambda did not publish regions_with_capacity_available for %s"
                % itype)
        available = [str((row or {}).get("name")) for row in regions
                     if isinstance(row, dict) and row.get("name")]
        region = kw.get("region")
        if region is None:
            raise LambdaError(
                "safe Lambda create requires an exact region: instances are "
                "region-pinned and capacity is per-region. Available for %s "
                "right now: %s" % (itype, ", ".join(available) or "none"))
        region = str(region)
        if region not in available:
            raise LambdaError(
                "instance type %s has no capacity in %r right now (available: "
                "%s); launching into a region that is not listed fails"
                % (itype, region, ", ".join(available) or "none"))
        public_key = self._validated_ssh_public_key()
        names = kw.get("ssh_key_names") or self.ssh_key_names
        registered = self._get_data("/ssh-keys")
        if not isinstance(registered, list):
            raise LambdaError("Lambda ssh-keys did not return a list")
        by_name: Dict[str, str] = {}
        for row in registered:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str):
                raise LambdaError("Lambda ssh-keys row has no string name")
            by_name[row["name"]] = str(row.get("public_key") or "")
        if not names:
            matching = sorted(
                key for key, value in by_name.items()
                if value and _canonical_public_key(value) == public_key)
            if not matching:
                raise LambdaError(
                    "no SSH key on the Lambda account matches %s.pub. Lambda "
                    "attaches keys BY NAME at launch and accepts no inline "
                    "public key, so the key must be registered in the console "
                    "first; registered names: %s"
                    % (self.ssh_key, ", ".join(sorted(by_name)) or "none"))
            names = matching[:1]
        names = [str(item) for item in names]
        # The API states exactly one key must be specified; sending two would
        # be refused after the request, not before it.
        if len(names) != 1:
            raise LambdaError(
                "Lambda accepts exactly one ssh_key_name at launch, got %d"
                % len(names))
        missing = [item for item in names if item not in by_name]
        if missing:
            raise LambdaError(
                "SSH key name(s) %s are not registered on this Lambda "
                "account (registered: %s)"
                % (", ".join(missing), ", ".join(sorted(by_name)) or "none"))
        for item in names:
            stored = by_name[item]
            if not stored:
                raise LambdaError(
                    "Lambda publishes no public_key for registered key %r, so "
                    "the name cannot be shown to bind our key" % item)
            if _canonical_public_key(stored) != public_key:
                raise LambdaError(
                    "Lambda SSH key name %r is registered against a DIFFERENT "
                    "public key than %s.pub. A name is not a key: launching "
                    "with it would bill a box this controller cannot log into."
                    % (item, self.ssh_key))
        pin = self._generate_host_key_pin()
        user_data = self._host_key_user_data(pin)
        body = {
            "region_name": region,
            "instance_type_name": itype,
            "ssh_key_names": names,
            "name": name,
            "user_data": user_data,
        }
        request_identity = {
            "provider": "lambda",
            "region_name": region,
            "instance_type_name": itype,
            "ssh_key_names": names,
            "ssh_public_key_sha256": hashlib.sha256(
                public_key.encode("utf-8")).hexdigest(),
            "name": name,
            "gpu_count": have_gpus,
            "storage_gib": have_disk,
            "requested_storage_gb": want_disk,
            "price_cents_per_hour": cents,
            "file_system_names": [],
            "spot": False,
            "offer": "on-demand",
            "terminate_after": terminate_after,
            "provider_enforced_deadline": False,
            "host_key_fingerprint": pin["fingerprint"],
            "user_data_sha256": hashlib.sha256(
                user_data.encode("utf-8")).hexdigest(),
        }
        launch_body = json.dumps(body).encode("utf-8")
        token = base64.b64encode((self._load_key() + ":").encode()).decode()
        http_request = urllib.request.Request(
            API + "/instance-operations/launch", data=launch_body,
            method="POST",
            headers={"Content-Type": "application/json",
                     "Accept": "application/json",
                     "User-Agent": "quant-fidelity-suite/0.1",
                     "Authorization": "Basic " + token})
        return PreparedLambdaCreate(
            http_request=http_request,
            http_opener=urllib.request.build_opener(_NoMutationRedirect()),
            launch_body=launch_body,
            request_identity_json=json.dumps(
                request_identity, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("utf-8"),
            name=name, instance_type_name=itype, region_name=region,
            ssh_key_names=tuple(names), storage_gib=have_disk,
            gpu_count=have_gpus, price_cents_per_hour=cents,
            terminate_after=terminate_after,
            host_key_fingerprint=pin["fingerprint"],
            host_key_public=pin["public_key"], dry_run=self.dry)

    def submit_prepared_create(
            self, prepared: PreparedLambdaCreate) -> Dict[str, Any]:
        """Submit the frozen launch. Never retried, never redirected.

        UNVERIFIED against a live Lambda account: no credential on this box.

        A LOST response is reconcilable rather than ambiguous because the
        frozen request carries the exact lease name: the caller can list
        instances and look for that name. That is weaker than RunPod's id
        echo and it is the honest position -- Lambda returns
        `data.instance_ids` and nothing to match against beforehand -- so the
        refusal names the reconciliation instead of implying the create did
        not happen.

        An enumerated official error code (insufficient capacity, quota,
        invalid parameters, filesystem region) proves NO instance was created
        and raises `LambdaCreateRejectedError`. Everything else -- a timeout,
        a transport error, an unparseable body -- stays on the fail-closed
        path.
        """
        if prepared.dry_run:
            return {
                "dry_run": True,
                "request": prepared.to_dict()["request_identity"],
                "prepared_create": prepared.to_dict(),
            }
        if self._last_launch_monotonic is not None:
            elapsed = time.monotonic() - self._last_launch_monotonic
            if elapsed < LAUNCH_MIN_INTERVAL_SECONDS:
                raise LambdaError(
                    "Lambda rate-limits launches to one per %.0f seconds and "
                    "the last one was %.1f s ago; a 429 on a create is an "
                    "AMBIGUOUS mutation, so this refuses instead of risking "
                    "one" % (LAUNCH_MIN_INTERVAL_SECONDS, elapsed))
        self._last_launch_monotonic = time.monotonic()
        try:
            with prepared.http_opener.open(
                    prepared.http_request, timeout=180) as response:
                raw = response.read(MAX_JSON_RESPONSE_BYTES + 1)
                if len(raw) > MAX_JSON_RESPONSE_BYTES:
                    raise LambdaError("Lambda launch response is oversized")
                document = _strict_json_loads(
                    raw.decode("utf-8", "strict"), "Lambda launch response")
        except urllib.error.HTTPError as exc:
            code, detail = _api_error(
                exc.read(4096).decode("utf-8", "replace"))
            message = "Lambda launch HTTP %d: %s" % (exc.code, detail)
            if code in _DEFINITIVE_LAUNCH_REJECTION_CODES:
                raise LambdaCreateRejectedError(message, code)
            raise LambdaError(message)
        except LambdaError:
            raise
        except Exception as exc:                          # noqa: BLE001
            raise LambdaError(
                "Lambda launch request failed, and whether an instance was "
                "created is UNKNOWN: reconcile by listing instances and "
                "looking for the exact name %r before retrying. %s"
                % (prepared.name, redact(str(exc))))
        data = document.get("data") if isinstance(document, dict) else None
        if not isinstance(data, dict):
            raise LambdaError(
                "Lambda launch response lacks the documented data object")
        ids = data.get("instance_ids")
        if not isinstance(ids, list) or len(ids) != 1:
            raise LambdaError(
                "Lambda launch returned %r instance ids where exactly one was "
                "requested; reconcile by name %r before any retry"
                % (ids, prepared.name))
        instance_id = _provider_id(ids[0], "Lambda launch response id")
        self._host_key_pins[instance_id] = {
            "fingerprint": prepared.host_key_fingerprint,
            "public_key": prepared.host_key_public,
            "pinned_at_utc": _utc(time.time()),
            "request_identity_sha256": hashlib.sha256(
                prepared.request_identity_json).hexdigest(),
        }
        return {
            "machine_id": instance_id,
            "provider_id": instance_id,
            "region": prepared.region_name,
            "name": prepared.name,
            "instance_type_name": prepared.instance_type_name,
            "price_cents_per_hour": prepared.price_cents_per_hour,
            "terminate_after": prepared.terminate_after,
            "host_key_fingerprint": prepared.host_key_fingerprint,
            "request": json.loads(
                prepared.request_identity_json.decode("utf-8")),
            "prepared_create": prepared.to_dict(),
        }
