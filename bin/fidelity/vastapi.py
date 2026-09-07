#!/usr/bin/env python3
"""Vast.ai backend, duck-typed to `fidelity.jlapi.JL`.

Vast is a MARKETPLACE, not a fleet, and that is the whole character of this
backend. You do not ask for "an A100"; you bid on one specific machine some
specific person owns, with its own disk, its own uplink and its own driver
stack. Two consequences the rest of the suite has to know about:

* **Offers are per-host, and an offer id is not a GPU model.** `create` takes
  the `ask_id` of a bundle that was searched for, so the search and the create
  are one transaction. An offer that vanishes between them is normal.
* **Bitwise determinism claims do not travel here.** docs/CLOUD-PROVIDERS.md
  §3 says it plainly: the cheapness comes from renting whatever a host happens
  to own -- different drivers, different host CPUs, sometimes different silicon
  under one GPU name. That is fine for work whose output is content-digested
  and verified (`verify` recomputes the whole chain before teardown) and is the
  wrong place to ESTABLISH a determinism result.

Disk is chosen at rent time and cannot grow, so `create` asks for the size the
plan computed and refuses an offer that cannot hold it.
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
import socket
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
# DEP-01. Transient HTTP statuses that say 'ask again', not 'no'.
# Applied to GET only; see Vast._req for why a mutation must not be
# retried on these.
_RETRY_STATUSES = frozenset({500, 502, 503, 504})

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from .common import register_secret, safe_urlopen
from .jlapi import GpuOffer, Instance, JLError, redact
from .sshbase import SSHTransport

API = "https://console.vast.ai/api/v0"
DEFAULT_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
# Where the operator keeps the key, and what docs/CLOUD-RECIPES.md documents.
# Expanded once to an absolute path; key bytes stay in the file.
DEFAULT_KEY_FILE = os.path.abspath(
    os.path.expanduser("~/.config/vastai/vast_api_key"))
MIN_CREATE_SETUP_SECONDS = 300
MAX_LOG_RESPONSE_BYTES = 2 * 1024 * 1024
# Vast bills per UTC DAY. `GET /charges/` is filtered by a day index
# (epoch // 86400) and every row's `start`/`end` is that day's midnight;
# nothing finer exists in the official response (verified live 2026-09-06).
VAST_DAY_SECONDS = 86400
# Hosts a capture must be able to reach BEFORE a byte is uploaded or a dollar
# spent. Machine 68004 (Nevada, US) failed the 2026-09-05 Fruit rehearsal at
# the setup stage with SSLEOFError/UNEXPECTED_EOF_WHILE_READING against
# huggingface.co (docs/CLOUD-RECIPES.md around line 297), and on 2026-09-06 it
# was STILL the cheapest rentable Tesla T4 on the marketplace -- so "the
# cheapest offer that fits" lands straight back on it unless something
# refuses.
#
# The recorded CAUSE has been corrected: MitmForensics rented that machine
# back and measured forged UDP DNS injection on its path, with the real
# addresses dialled from the same box verifying a BYTE-IDENTICAL leaf. So
# there is no TLS interceptor there, and this entry must not say there is:
# a misconfigured resolver path and a credential harvester look the same from
# a failed handshake, and only one of them is an accusation. The refusal
# stands on the measured failure, not on a motive.
HUB_PROBE_HOSTS = ("huggingface.co", "cdn-lfs.hf.co")
KNOWN_BAD_MACHINE_IDS = {
    "68004": "machine 68004 (Nevada, US, driver 580.126.09) could not "
             "complete a TLS handshake to huggingface.co and failed the "
             "2026-09-05 capture at setup; measured cause is forged UDP DNS "
             "injection on its path, NOT an interception proxy -- the host "
             "operator is not accused",
}
# Emitted by `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub`. Vast's own
# instance log contains sshd's "Server listening on 0.0.0.0 port 22" but NOT
# this line (verified live on instance 50054271, 2026-09-06); the only ED25519
# strings there are the container's own known-hosts warnings about the
# ssh2.vast.ai JUMP PROXY, whose key is not the instance's.
_HOST_KEY_LOG_RE = re.compile(
    r"256\s+(SHA256:[A-Za-z0-9+/]{43})\s+\S+\s+\(ED25519\)")
_HOST_KEY_LOG_POLL_SECONDS = 4.0
# Vast ids are INTEGERS in every response (`id`, `machine_id`, `new_contract`).
# The canonical form here is the exact decimal string of that integer: a label
# is never an id, and an id never compares as an int against a set of strings.
_INSTANCE_ID_RE = re.compile(r"[1-9][0-9]{0,17}\Z")
_GPU_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._:+()/+-]{0,255}\Z")
_CHARGE_ROW_KEYS = frozenset((
    "start", "end", "type", "source", "description", "amount", "metadata",
    "items"))


class VastError(JLError):
    """Same exception family as the JarvisLabs backend, so callers need no branch."""


class VastCreateResponseError(VastError):
    """A rental committed with an exact contract id but unqualified metadata."""

    def __init__(self, message: str, provider_id: Any,
                 response: Dict[str, Any]) -> None:
        super().__init__(message)
        self.provider_id = str(provider_id)
        self.response = dict(response)


class VastCreateRejectedError(VastError):
    """The provider refused the rental outright and returned no contract."""

    def __init__(self, message: str, reasons) -> None:
        super().__init__(message)
        self.reasons = tuple(sorted(set(reasons)))


def _definitive_create_rejection_reasons(document: Any) -> tuple:
    """Return refusal reasons only when the response PROVES no contract exists.

    A rental Vast REFUSED is not a rental whose response was lost. `PUT
    /asks/<id>/` is one transaction, so `{"success": false, "msg": ...}`
    carrying no integral field anywhere accepted nothing. Everything
    ambiguous -- a non-object body, `success` not exactly `False`, any number
    that could be a contract id, any key naming a contract or instance --
    returns `()` and keeps the fail-closed path, exactly as RunPod's
    enumerated-code rule does.
    """
    if not isinstance(document, dict) or document.get("success") is not False:
        return ()
    for key, value in document.items():
        lowered = str(key).lower()
        if "contract" in lowered or "instance" in lowered or lowered == "id":
            return ()
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, dict, list)):
            return ()
    for field in ("msg", "error"):
        text = document.get(field)
        if isinstance(text, str) and text.strip():
            return (redact(text.strip())[:200],)
    return ()


def _strict_json_loads(raw: str) -> Any:
    def _pairs(items):
        keys = [key for key, _ in items]
        if len(set(keys)) != len(keys):
            raise VastError("Vast returned JSON with duplicate object keys")
        return dict(items)

    def _constant(value):
        raise VastError(
            "Vast returned JSON with the non-finite constant %s" % value)

    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=_constant)
    except VastError:
        raise
    except ValueError as exc:
        raise VastError("Vast returned invalid JSON: %s" % exc)


def _exact_utc(value: Any, field: str) -> str:
    text = str(value)
    try:
        parsed = time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise VastError("%s must be exact UTC YYYY-MM-DDTHH:MM:SSZ" % field)
    if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) != text:
        raise VastError("%s must be exact UTC YYYY-MM-DDTHH:MM:SSZ" % field)
    return text


def _utc_epoch(value: Any, field: str) -> int:
    return calendar.timegm(time.strptime(
        _exact_utc(value, field), "%Y-%m-%dT%H:%M:%SZ"))


def _utc_text(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _terminate_after(kw: Dict[str, Any]) -> Optional[str]:
    text = kw.get("terminate_after")
    epoch = kw.get("terminate_after_epoch")
    if text is not None and epoch is not None:
        raise VastError(
            "pass only one of terminate_after and terminate_after_epoch")
    if epoch is not None:
        try:
            value = float(epoch)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("non-positive or non-finite")
            text = _utc_text(value)
        except (TypeError, ValueError, OverflowError, OSError):
            raise VastError(
                "terminate_after_epoch must be a positive finite epoch")
    return _exact_utc(text, "terminate_after") if text is not None else None


def _provider_id(value: Any, field: str = "Vast instance id") -> str:
    """Canonicalise an exact integral provider id; a label is never an id."""
    if isinstance(value, bool):
        raise VastError("%s must be an exact integral id" % field)
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise VastError("%s must be an exact integral id" % field)
    if _INSTANCE_ID_RE.fullmatch(text) is None:
        raise VastError(
            "%s must be an exact integral id, got %r" % (field, value))
    return text


def _gpu_id(value: Any, field: str = "Vast GPU model") -> str:
    if not isinstance(value, str) or _GPU_ID_RE.fullmatch(value) is None:
        raise VastError("%s has invalid characters or length" % field)
    return value


def _finite_decimal(value: Any, field: str, *, positive: bool = False,
                    nonnegative: bool = False) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise VastError("%s is not an exact decimal" % field)
    if not parsed.is_finite():
        raise VastError("%s is not finite" % field)
    if positive and parsed <= 0:
        raise VastError("%s must be positive" % field)
    if nonnegative and parsed < 0:
        raise VastError("%s must not be negative" % field)
    return parsed


def _day_index(epoch: float) -> int:
    return int(epoch) // VAST_DAY_SECONDS


def _attestation_seal(document: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(document)
    raw = json.dumps(
        out, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")
    out["attestation_sha256"] = hashlib.sha256(raw).hexdigest()
    return out


@dataclass(frozen=True)
class PreparedVastCreate:
    """A rental request built and frozen BEFORE any provider mutation.

    On a marketplace the ask is half the identity: `ask_id` names one host's
    bundle, and an ask that vanishes between search and rental is ordinary.
    So the frozen request carries the ADVERTISED identity of that ask
    (machine, host, GPU model, VRAM, disk, rate) and `validate_safe_resource_
    binding` compares the CONTRACT against it -- an ask that advertised a
    B200 once handed back an H100.
    """
    http_request: Any
    http_opener: Any
    request_body: bytes
    request_identity_json: bytes
    name: str
    ask_id: str
    machine_id: Optional[str]
    disk_gb: int
    storage_gb: int
    container_disk_gb: int
    image_name: str
    terminate_after: str
    duration_seconds: int
    host_key_fingerprint: str
    dry_run: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_identity": json.loads(
                self.request_identity_json.decode("utf-8")),
            "request_identity_sha256": hashlib.sha256(
                self.request_identity_json).hexdigest(),
            "request_body_sha256": hashlib.sha256(
                self.request_body).hexdigest(),
            "url": self.http_request.full_url,
            "method": self.http_request.get_method(),
            "name": self.name,
            "ask_id": self.ask_id,
            "machine_id": self.machine_id,
            "disk_gb": self.disk_gb,
            "storage_gb": self.storage_gb,
            "container_disk_gb": self.container_disk_gb,
            "image_name": self.image_name,
            "terminate_after": self.terminate_after,
            "duration_seconds": self.duration_seconds,
            # The fingerprint the live host MUST present. Public by nature;
            # the private half never enters this document or any log.
            "pinned_host_key_fingerprint": self.host_key_fingerprint,
            "dry_run": self.dry_run,
        }

_LIVE_ATTEST_SCRIPT = r'''
import json
import os
import subprocess
import sys
import time

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
        "error": None,
    }

def filesystem(path):
    try:
        return mount(path)
    except Exception as exc:
        return {"path": path, "mount_point": None, "fs_type": None,
                "source": None, "device": None, "total_bytes": None,
                "available_bytes": None,
                "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}

# NOTE: this script deliberately carries NO TLS peer probe. Judging whether a
# box talks to the REAL Hub is `fidelity.tlsguard`'s single implementation --
# it verifies the chain against our own digest-pinned roots and compares the
# leaf against one the controller verified, neither of which a hand-rolled
# default-context handshake here could do, and two implementations of one
# security property is how they drift. `attest_live_resource` runs tlsguard's
# own collector over this same exec channel and records its verdict.

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
# TOTAL VRAM is not the attestable quantity: FREE is. A rented "24 GB" 4090
# had 23424 of 24564 MiB held by four foreign PIDs (2026-09-06, host
# 434175) -- "24 GB card" was true and useless. Both queries run BEFORE the
# torch probe below, so our own CUDA context can never appear in the
# foreign-process list.
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
    ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
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
                "process_name": fields[2],
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
for extra in ("/usr/bin/python3.12", "/opt/conda/bin/python"):
    if os.path.isfile(extra) and extra not in candidates:
        candidates.append(extra)
errors = []
for interpreter in candidates:
    try:
        run = subprocess.run([interpreter, "-c", CUDA_PROBE], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, timeout=240)
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
    "filesystems": {"root": filesystem("/"),
                    "workspace": filesystem("/workspace")},
}, sort_keys=True, separators=(",", ":"), allow_nan=False))
'''



class Vast(SSHTransport):
    # This provider has NO filesystem that outlives its instance, so the
    # whole run must fit on the instance's own disk: vast disk is chosen at rent time and dies with the instance.
    # The controller reads this to size `create(storage=)`.
    separable_storage = False
    provider = "vast"
    ssh_user = "root"
    RUNS = "/workspace/.fidruns"

    def __init__(self, *, dry: bool = False, key_file: Optional[str] = None,
                 ssh_key: Optional[str] = None) -> None:
        self.dry = dry
        self._key_file = key_file
        self._key: Optional[str] = None
        self.ssh_key = ssh_key or os.path.expanduser("~/.ssh/id_ed25519")
        self._ep: Dict[str, tuple] = {}
        self._server_time: Optional[Dict[str, Any]] = None
        # contract id -> the ED25519 fingerprint pinned at create time.
        self._pinned_host_keys: Dict[str, str] = {}

    # -- transport ---------------------------------------------------------
    def _load_key(self) -> str:
        """Find the credential where the operator actually keeps it.

        The old order was `key_file` -> `$VAST_KEY_FILE` -> `$VAST_API_KEY`
        and nothing else, so a 0600 key sitting at the conventional path was
        NOT FOUND and the refusal told the operator to export a secret into
        their environment -- the opposite of what the 0600-file discipline is
        for. Three sibling tasks worked around it in one afternoon, and
        `docs/CLOUD-RECIPES.md` compounded it by passing the path unexpanded,
        which `os.path.isfile` silently missed. Both are fixed here: the
        conventional path is tried, and `~` is expanded.
        """
        if self._key:
            return self._key
        candidates = [self._key_file, os.environ.get("VAST_KEY_FILE"),
                      DEFAULT_KEY_FILE]
        for candidate in candidates:
            if not candidate:
                continue
            path = os.path.abspath(os.path.expanduser(str(candidate)))
            if not os.path.exists(path):
                # An explicitly requested path that is absent is a refusal,
                # not a silent fall-through to a different credential.
                if candidate is DEFAULT_KEY_FILE:
                    continue
                raise VastError(
                    "Vast key file does not exist: %s" % path)
            self._key = self._read_key_file(path)
            return self._key
        self._key = os.environ.get("VAST_API_KEY", "").strip()
        register_secret(self._key)
        if not self._key:
            raise VastError(
                "no Vast credential: put the key in a 0600 file at %s, point "
                "VAST_KEY_FILE at one, or set VAST_API_KEY"
                % DEFAULT_KEY_FILE)
        return self._key

    @staticmethod
    def _read_key_file(path: str) -> str:
        """Read the credential from a 0600 regular file, never a symlink."""
        if not os.path.isabs(path):
            raise VastError("Vast key file path must be absolute: %s" % path)
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise VastError(
                    "Vast key file must be a regular file, not a symlink")
            if stat.S_IMODE(info.st_mode) != 0o600:
                raise VastError(
                    "Vast key file must have mode 0600: %s" % path)
            if hasattr(os, "getuid") and info.st_uid != os.getuid():
                raise VastError(
                    "Vast key file must be owned by the current user")
            with os.fdopen(fd, encoding="utf-8") as handle:
                fd = None
                key = handle.read().strip()
        except VastError:
            raise
        except (OSError, UnicodeError) as exc:
            raise VastError("Vast key file is unavailable or invalid at %s: %s"
                            % (path, exc))
        finally:
            if fd is not None:
                os.close(fd)
        if not key:
            raise VastError("Vast key file is empty: %s" % path)
        register_secret(key)
        return key

    # Vast enforces roughly one API request per second and answers HTTP 429
    # with a `retry_after`. The banded catalogue search fires five queries back
    # to back, which tripped it immediately -- and it tripped INSIDE the run,
    # after the lease was written, so a rate limit read as a failed run.
    _MIN_INTERVAL = 1.1
    _last_call = 0.0

    def _capture_server_time(self, response: Any, endpoint: str) -> None:
        """Record the provider's own clock from its authenticated response.

        Vast exposes no server-time API FIELD; the authenticated response's
        HTTP `Date` header is the provider's clock (strict GMT, verified live
        2026-09-06 against `GET /users/current/`). A teardown deadline encoded
        against our clock is not a teardown guarantee, so this is captured on
        every read and `server_time_evidence` refuses when it is missing.
        """
        raw = response.headers.get("Date")
        if not isinstance(raw, str) or not raw:
            raise VastError("Vast authenticated response lacks HTTP Date")
        try:
            parsed = email.utils.parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            raise VastError("Vast authenticated response Date is invalid")
        if (parsed is None or parsed.utcoffset() is None
                or parsed.utcoffset().total_seconds() != 0
                or email.utils.format_datetime(parsed, usegmt=True) != raw):
            raise VastError(
                "Vast authenticated response Date is not strict GMT")
        received = time.time()
        server_epoch = parsed.timestamp()
        split = urllib.parse.urlsplit(endpoint)
        self._server_time = {
            "schema": "fidelity-suite/vast-server-time.v1",
            "endpoint_origin": "%s://%s" % (split.scheme, split.hostname),
            "date_header": raw,
            "server_epoch": server_epoch,
            "local_received_epoch": received,
            "local_minus_server_seconds": received - server_epoch,
        }

    def server_time_evidence(
            self, *, max_clock_delta_seconds: float = 30,
            max_evidence_age_seconds: float = 30) -> Dict[str, Any]:
        """The PROVIDER's clock, from its own authenticated response header.

        Live-verified: Vast answers every authenticated request with a strict
        GMT `Date`. No API field carries a server timestamp, so the header is
        the only source; when no read has happened yet this refuses instead of
        substituting our clock.
        """
        evidence = self._server_time
        if evidence is None:
            raise VastError(
                "Vast server time is unavailable; an authenticated read "
                "(status or inventory) must succeed before create")
        now = time.time()
        age = now - evidence["local_received_epoch"]
        if not math.isfinite(age) or age < -1 or age > max_evidence_age_seconds:
            raise VastError("Vast server-time evidence is stale")
        delta = evidence["local_minus_server_seconds"]
        if abs(delta) > max_clock_delta_seconds:
            raise VastError(
                "local UTC differs from Vast server UTC by more than %.0fs"
                % max_clock_delta_seconds)
        out = dict(evidence)
        out.update({
            "checked_at_epoch": now,
            "evidence_age_seconds": age,
            "max_clock_delta_seconds": float(max_clock_delta_seconds),
            "max_evidence_age_seconds": float(max_evidence_age_seconds),
        })
        return out

    @staticmethod
    def _announce_retry(method: str, path: str, why: str, delay: float,
                        tries_left: int) -> None:
        """Say on stderr that a transient is being waited out.

        A silently absorbed retry is indistinguishable from a clean pass in
        summary output, so a run that quietly took four attempts looks
        identical to one that took none -- and the only place the difference
        shows is a log nobody reads. Announce it.
        """
        sys.stderr.write(
            "vast: %s for %s %s -- transient, waiting %.1fs "
            "(%d attempt(s) left)\n"
            % (why, method, path, delay, tries_left - 1))
        sys.stderr.flush()

    def _req(self, method: str, path: str, body: Any = None,
             *, timeout: float = 90, _tries: int = 4) -> Any:
        gap = time.time() - Vast._last_call
        if gap < self._MIN_INTERVAL:
            time.sleep(self._MIN_INTERVAL - gap)
        Vast._last_call = time.time()
        url = API + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "quant-fidelity-suite/0.1",
                                              "Authorization": "Bearer " + self._load_key()})
        try:
            # safe_urlopen, never bare urlopen: this request carries
            # Authorization and must not follow a redirect to another origin.
            with safe_urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                if method == "GET":
                    self._capture_server_time(resp, url)
        except urllib.error.HTTPError as exc:
            payload = exc.read()[:300].decode("utf-8", "replace")
            # DEP-01. This arm used to retry 429 and NOTHING else, which is
            # the same shape as the incident it was written to stop: a
            # transient read as a failed run, INSIDE the run, after the lease
            # was written. A 502/503/504 and every connection reset fell
            # through to `except Exception` and raised hard.
            #
            # But the extension is NOT symmetric, and that asymmetry is the
            # whole design. 429 means the request was REJECTED, so retrying
            # any method is safe. A 5xx or a dropped connection on a MUTATION
            # may mean the mutation SUCCEEDED and only the response was lost
            # -- which for `PUT /asks/{id}/` means an instance now exists and
            # is billing. Retrying that would double-rent, and a leaked
            # instance is a blocker-level defect. That case is already handled
            # correctly one layer up, by the lease store's
            # LOST_CREATE_RESPONSE reconciliation, and it must stay there.
            # So: transient statuses retry on GET only; 429 retries on any
            # method.
            retryable = (exc.code == 429
                         or (method == "GET" and exc.code in _RETRY_STATUSES))
            if retryable and _tries > 1:
                wait = 2.0
                try:
                    # `retry_after` arrives in the BODY, not the header --
                    # measured, and the reason this cannot be a stock
                    # Retry-After helper.
                    wait = max(1.0, float(json.loads(payload).get("retry_after") or 1)) + 1.0
                except Exception:                         # noqa: BLE001
                    pass
                self._announce_retry(method, path, "HTTP %d" % exc.code,
                                     wait, _tries)
                time.sleep(wait)
                return self._req(method, path, body, timeout=timeout,
                                 _tries=_tries - 1)
            raise VastError("Vast HTTP %d on %s: %s"
                            % (exc.code, path, redact(payload)))
        except VastError:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                socket.timeout) as exc:
            # A network fault, not an answer about the request. Same
            # idempotency rule: a GET may be retried, a mutation may not,
            # because a dropped connection cannot tell us whether the
            # provider acted.
            if method == "GET" and _tries > 1:
                self._announce_retry(method, path, type(exc).__name__,
                                     2.0, _tries)
                time.sleep(2.0)
                return self._req(method, path, body, timeout=timeout,
                                 _tries=_tries - 1)
            raise VastError("Vast request failed: %s" % redact(str(exc)))
        except Exception as exc:                          # noqa: BLE001
            raise VastError("Vast request failed: %s" % redact(str(exc)))
        return _strict_json_loads(raw) if raw.strip() else {}

    # -- identity ----------------------------------------------------------
    def available(self) -> bool:
        try:
            self._load_key()
            return True
        except VastError:
            return False

    def require(self) -> tuple:
        if not self.available():
            raise VastError("Vast credential not configured")
        return (0, 0, 0)

    @property
    def version(self) -> str:
        return "vast-rest-v0"

    def status(self) -> Dict[str, Any]:
        return self._req("GET", "/users/current/") or {}

    def balance(self) -> Optional[float]:
        try:
            return float(self.status().get("credit"))
        except Exception:                                 # noqa: BLE001
            return None

    # -- catalogue ---------------------------------------------------------
    # Vast has tens of thousands of live offers and the API returns them
    # cheapest-first. A single unconstrained search therefore returns nothing
    # but 6-13 GB consumer cards, and the controller -- which does its own
    # VRAM filtering on whatever list it is handed -- concluded that Vast had
    # no instance able to hold the model. The catalogue is assembled from
    # SEVERAL banded searches so every tier is represented.
    _VRAM_BANDS = (0, 24, 48, 63, 80)

    def gpus(self, *, min_vram_gb: int = 0, min_disk_gb: int = 300,
             limit: int = 40) -> List[GpuOffer]:
        if not min_vram_gb:
            seen, merged = set(), []
            for band in self._VRAM_BANDS:
                for o in self._search(band, min_disk_gb, max(8, limit // 4)):
                    if o.raw["ask_id"] in seen:
                        continue
                    seen.add(o.raw["ask_id"])
                    merged.append(o)
            return merged
        return self._search(min_vram_gb, min_disk_gb, limit)

    def _search(self, min_vram_gb: int, min_disk_gb: int, limit: int,
                gpu_name: Optional[str] = None) -> List[GpuOffer]:
        q = {"rentable": {"eq": True}, "num_gpus": {"eq": 1},
             "disk_space": {"gte": int(min_disk_gb)},
             "order": [["dph_total", "asc"]], "limit": int(limit),
             "type": "on-demand"}
        if min_vram_gb:
            q["gpu_ram"] = {"gte": int(min_vram_gb) * 1024}
        if gpu_name:
            # Ask Vast for the card BY NAME rather than filtering a generic
            # list. The banded catalogue returns the cheapest few per VRAM
            # band, so a specifically-requested 20 GB card routinely is not in
            # it and "no offer for RTX A4500" was reported while dozens were
            # rentable.
            q["gpu_name"] = {"eq": gpu_name}
        got = self._req("GET", "/bundles/?q=" + urllib.parse.quote(json.dumps(q)))
        offers = []
        for o in (got or {}).get("offers", []):
            offers.append(GpuOffer(
                gpu_type=o.get("gpu_name") or "?",
                region=(o.get("geolocation") or "").strip() or None,
                vram_bytes=float(o.get("gpu_ram") or 0) * (1024 ** 2),
                price=float(o.get("dph_total") or 0), spot=False,
                free_devices=1, workload_type="container",
                # the ask id is the only thing `create` can act on
                raw={"ask_id": o.get("id"), "disk_space": o.get("disk_space"),
                     "cuda": o.get("cuda_max_good"),
                     "inet_down": o.get("inet_down"),
                     "reliability": o.get("reliability2")}))
        return offers

    # -- instances ---------------------------------------------------------
    @staticmethod
    def _elapsed_seconds(d: Dict[str, Any]) -> float:
        """Seconds this contract has actually been alive.

        NOT `duration`: on a live Vast contract that field is the time
        REMAINING on the rental and counts DOWN (15 596 493 -> 15 596 229 over
        four minutes on instance 50054271, 2026-09-06 -- a 180-day contract on
        a box eight minutes old). Costing a run from it overstates spend by
        three orders of magnitude.
        """
        try:
            started = float(d.get("start_date") or 0)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(started) or started <= 0:
            return 0.0
        return max(0.0, time.time() - started)

    @classmethod
    def _to_instance(cls, d: Dict[str, Any]) -> Instance:
        elapsed = cls._elapsed_seconds(d)
        inst = Instance.from_json({
            "machine_id": 0,
            "status": d.get("actual_status") or d.get("cur_state") or "",
            "gpu_type": d.get("gpu_name"), "num_gpus": d.get("num_gpus") or 1,
            "region": d.get("geolocation"), "is_spot": False,
            "cost": float(d.get("dph_total") or 0) * elapsed / 3600.0,
            "runtime": elapsed, "fs_id": None,
            "storage_gb": d.get("disk_space"), "name": d.get("label"),
        })
        inst.machine_id = d.get("id")
        inst.raw["ssh_host"] = d.get("ssh_host")
        inst.raw["ssh_port"] = d.get("ssh_port")
        # The CONTRACT rate, not the ask's. On a marketplace those are two
        # different objects: the ask you searched can be gone by the time the
        # rental lands, and an ask id is not a durable name for one machine --
        # one that advertised a B200 handed back an H100. Anything that prices
        # a run must read what is billing, not what was listed. Live proof
        # (2026-09-06, contract 50054271): the ask listed $0.13556/h and the
        # contract bills $0.16667/h = $0.13333 GPU + $0.03333 disk, 23% more.
        inst.raw["dph_total"] = d.get("dph_total")
        inst.raw["gpu_name"] = d.get("gpu_name")
        inst.raw["contract_seconds_remaining"] = d.get("duration")
        inst.raw["start_date"] = d.get("start_date")
        inst.raw["end_date"] = d.get("end_date")
        return inst

    def list_instances(self) -> List[Instance]:
        got = self._req("GET", "/instances/") or {}
        return [self._to_instance(d) for d in got.get("instances", [])]

    def get(self, machine_id: Any) -> Optional[Instance]:
        wanted = _provider_id(machine_id)
        for i in self.list_instances():
            if str(i.machine_id) == wanted:
                return i
        return None

    # -- lifecycle: is anything of mine still alive? -----------------------
    def _instance_documents(self) -> List[Dict[str, Any]]:
        """Every instance the account holds, validated for exact identity."""
        got = self._req("GET", "/instances/") or {}
        rows = got.get("instances")
        if not isinstance(rows, list):
            raise VastError("Vast instance listing lacks an instances array")
        documents = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise VastError(
                    "Vast instance listing contains a non-object row")
            instance_id = _provider_id(row.get("id"), "Vast instance id")
            if instance_id in seen:
                raise VastError(
                    "Vast instance listing repeats id %s" % instance_id)
            seen.add(instance_id)
            documents.append(row)
        return documents

    def list_lifecycle_resources(self) -> List[Dict[str, Any]]:
        """Complete exact-id rows; every listed instance is still chargeable.

        `status` is `cur_state` -- the CONTRACT's state, lowercase on this
        provider ("running", never "RUNNING"). `actual_status` is carried
        beside it because the two disagree in exactly the window that matters:
        a box reports `cur_state` running with `actual_status` loading while
        its image is still being pulled, and it is billing throughout.
        """
        resources = []
        for row in self._instance_documents():
            volumes = row.get("volume_info")
            resources.append({
                "id": _provider_id(row.get("id")),
                "name": row.get("label"),
                "status": row.get("cur_state"),
                "listed": True,
                "actual_status": row.get("actual_status"),
                "intended_status": row.get("intended_status"),
                "cost_per_hr": row.get("dph_total"),
                "gpu_cost_per_hr": row.get("dph_base"),
                "storage_cost_per_hr": row.get("storage_total_cost"),
                "runtime": self._elapsed_seconds(row),
                "gpu_count": row.get("num_gpus"),
                "gpu_type_id": row.get("gpu_name"),
                "gpu_display_name": row.get("gpu_name"),
                "gpu_ram_mib": row.get("gpu_ram"),
                # A Vast host is one person's machine: the machine and host
                # ids are the only durable names for the silicon and the
                # uplink, and a bad host has already cost a capture.
                "provider_machine_id": (
                    None if row.get("machine_id") is None
                    else str(row.get("machine_id"))),
                "provider_host_id": (
                    None if row.get("host_id") is None
                    else str(row.get("host_id"))),
                "location": row.get("geolocation"),
                "data_center_id": None,
                "verification": row.get("verification"),
                "hosting_type": row.get("hosting_type"),
                "driver_version": row.get("driver_version"),
                "cuda_max_good": row.get("cuda_max_good"),
                # One pod-scoped disk, so both roles resolve to it.
                "volume_gb": row.get("disk_space"),
                "container_disk_gb": row.get("disk_space"),
                "disk_gb": row.get("disk_space"),
                "network_volume_id": (
                    None if not volumes
                    else json.dumps(volumes, sort_keys=True)),
                "image_name": row.get("image_uuid"),
                "start_date": row.get("start_date"),
                # Vast's provider-side deadline: the contract end date. There
                # is no `terminateAfter`; `end_date` is what the provider will
                # act on, so it is what a deadline must be validated against.
                "terminate_after": (
                    None if row.get("end_date") is None
                    else _utc_text(float(row["end_date"]))),
                "end_date": row.get("end_date"),
                "raw": row,
            })
        return resources

    def get_lifecycle_resource(self, provider_id: Any) -> Optional[Dict[str, Any]]:
        """Exact-id detail; a `label` is deliberately not accepted as an id."""
        wanted = _provider_id(provider_id)
        return next((row for row in self.list_lifecycle_resources()
                     if row["id"] == wanted), None)

    def list_network_volumes(self) -> List[Dict[str, Any]]:
        """Enumerate persistent chargeable volumes through the official API.

        This is a REAL enumeration, not a stub returning `[]`. `fs_create` /
        `fs_delete` are right that the disk a rental gets dies with the
        rental -- but Vast separately sells network VOLUMES, `GET /volumes/`
        lists them, offers advertise `avail_vol_ask_id` / `avail_vol_dph` /
        `avail_vol_size`, and a live instance carries `volume_info`. A volume
        outlives the instance that mounted it and keeps charging, so assuming
        this family is empty would make `chargeable_inventory` claim absence
        it never checked. On 2026-09-06 this account held none, which is a
        fact about the account and not about the API.
        """
        got = self._req("GET", "/volumes/") or {}
        rows = got.get("volumes")
        if not isinstance(rows, list):
            raise VastError("Vast volume listing lacks a volumes array")
        resources = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise VastError("Vast volume listing contains a non-object row")
            volume_id = _provider_id(row.get("id"), "Vast volume id")
            if volume_id in seen:
                raise VastError("Vast volume listing repeats id %s" % volume_id)
            seen.add(volume_id)
            size = None
            for field in ("size", "disk_space", "size_gb", "volume_size"):
                if row.get(field) is not None:
                    size = _finite_decimal(
                        row[field], "Vast volume %s" % field, positive=True)
                    break
            if size is None:
                # An unrecognised row shape must REFUSE, so the caller records
                # this family as incomplete rather than counting a chargeable
                # volume it could not size.
                raise VastError(
                    "Vast volume %s exposes no recognised size field (keys: "
                    "%s); size it before treating this family as complete"
                    % (volume_id, ",".join(sorted(row))))
            resources.append({
                "id": volume_id,
                "name": row.get("label") or row.get("name"),
                "size_gb": float(size),
                "cost_per_hr": row.get("dph_total") or row.get("volume_dph"),
                "provider_machine_id": (
                    None if row.get("machine_id") is None
                    else str(row.get("machine_id"))),
                "raw": row,
            })
        return resources

    def chargeable_inventory(self) -> Dict[str, Any]:
        """Instances plus volumes, with EXPLICIT completeness per family.

        A partial inventory cannot prove no leak, so each family says whether
        it was established and names its source endpoint; a family that could
        not be read is named in `unknown_families` and never counted as empty.
        Both families are chargeable on Vast: an instance bills GPU plus disk
        while it exists (and disk alone while stopped), and a network volume
        outlives the instance that mounted it.
        """
        families: Dict[str, Dict[str, Any]] = {}
        try:
            instances = []
            for row in self.list_lifecycle_resources():
                instances.append({
                    "id": row["id"],
                    "name": row["name"],
                    "status": row["status"],
                    "actual_status": row["actual_status"],
                    "cost_per_hr": (
                        None if row["cost_per_hr"] is None else format(
                            _finite_decimal(
                                row["cost_per_hr"],
                                "Vast inventory dph_total", nonnegative=True),
                            "f")),
                    "provider_machine_id": row["provider_machine_id"],
                    "network_volume_id": row["network_volume_id"],
                    "raw": row["raw"],
                })
            families["instances"] = {
                "complete": True,
                "source": "GET %s/instances/" % API,
                "resources": instances,
            }
        except VastError as exc:
            families["instances"] = {
                "complete": False,
                "source": "GET %s/instances/" % API,
                "resources": [],
                "unknown": redact(str(exc)),
            }
        try:
            families["network_volumes"] = {
                "complete": True,
                "source": "GET %s/volumes/" % API,
                "resources": self.list_network_volumes(),
            }
        except VastError as exc:
            families["network_volumes"] = {
                "complete": False,
                "source": "GET %s/volumes/" % API,
                "resources": [],
                "unknown": redact(str(exc)),
            }
        unknown = sorted(name for name, family in families.items()
                         if not family["complete"])
        return {
            "schema": "fidelity-suite/vast-chargeable-inventory.v1",
            "provider": "vast",
            "observed_at_utc": _utc_text(time.time()),
            "complete": not unknown,
            "unknown_families": unknown,
            "families": families,
        }

    # -- is this the thing I asked for? ------------------------------------
    def validate_safe_resource_binding(
            self, provider_id: Any, *, expected_name: str,
            gpu_type_id: str, secure_cloud: bool, gpu_count: int,
            volume_gb: int, container_disk_gb: int, image_name: str,
            terminate_after: str) -> Dict[str, Any]:
        """Fail unless the live exact-id CONTRACT is the rental requested.

        Same signature as RunPod's so the controller needs no special case;
        three arguments mean something different here and the difference is
        checked rather than papered over:

        * `volume_gb` + `container_disk_gb` -- Vast rents ONE pod-scoped disk
          chosen at rent time. The contract must hold their SUM.
        * `secure_cloud` -- a marketplace host is not a secure datacenter and
          exposes no attribute that could prove it were (a live contract
          reported `hosting_type: null`, `verification: "unverified"`), so
          `True` is refused instead of being silently accepted.
        * `terminate_after` -- Vast has no `terminateAfter`; the contract's
          `end_date` is the provider-side deadline. A contract ending LATER
          than the deadline means the provider is not holding our deadline
          at all, which is refused; ending earlier is the provider promising
          to stop sooner and is recorded.
        """
        observed = self.get_lifecycle_resource(provider_id)
        if observed is None:
            raise VastError(
                "created Vast contract %s is absent from the complete listing"
                % _provider_id(provider_id))
        expected_deadline = _exact_utc(terminate_after, "terminate_after")
        deadline_epoch = _utc_epoch(expected_deadline, "terminate_after")
        if not isinstance(secure_cloud, bool):
            raise VastError("secure_cloud expectation must be an exact bool")
        if secure_cloud:
            raise VastError(
                "Vast is a marketplace and cannot attest a secure-datacenter "
                "binding: a live contract reports hosting_type null and "
                "verification %r. Pass secure_cloud=False for vast, or use a "
                "provider whose API states it."
                % observed.get("verification"))
        disk_required = int(volume_gb) + int(container_disk_gb)
        expected = {
            "name": str(expected_name),
            "gpu_type_id": str(gpu_type_id),
            "gpu_count": int(gpu_count),
            "image_name": str(image_name),
            "disk_gb_minimum": disk_required,
            "secure_cloud": False,
        }
        problems = []
        for key in ("name", "gpu_type_id", "image_name"):
            actual = observed.get(key)
            if actual != expected[key]:
                problems.append("%s expected %r, observed %r"
                                % (key, expected[key], actual))
        try:
            observed_gpus = int(observed.get("gpu_count"))
        except (TypeError, ValueError):
            observed_gpus = None
        if observed_gpus != expected["gpu_count"]:
            problems.append("gpu_count expected %r, observed %r"
                            % (expected["gpu_count"], observed.get("gpu_count")))
        try:
            observed_disk = _finite_decimal(
                observed.get("disk_gb"), "Vast contract disk_space",
                positive=True)
        except VastError:
            problems.append("disk_space must be a known positive decimal")
            observed_disk = None
        else:
            observed["disk_gb"] = float(observed_disk)
            if observed_disk < disk_required:
                problems.append(
                    "disk_space %s GB is below the %d GB the plan needs "
                    "(volume %d + container %d); vast disk cannot grow after "
                    "rent time" % (format(observed_disk, "f"), disk_required,
                                   int(volume_gb), int(container_disk_gb)))
        try:
            live_rate = _finite_decimal(
                observed.get("cost_per_hr"), "Vast contract dph_total",
                positive=True)
        except VastError:
            problems.append(
                "cost_per_hr must be a known positive exact decimal")
        else:
            observed["cost_per_hr"] = format(live_rate, "f")
        if observed.get("network_volume_id") not in (None, ""):
            problems.append(
                "no network volume may be attached, observed %r"
                % observed.get("network_volume_id"))
        observed_end = observed.get("end_date")
        try:
            observed_end_epoch = float(observed_end)
        except (TypeError, ValueError):
            observed_end_epoch = None
            problems.append("contract end_date is missing or not a number")
        if observed_end_epoch is not None and observed_end_epoch > deadline_epoch + 120:
            problems.append(
                "contract end_date %s is later than the requested deadline "
                "%s, so the provider is not holding our teardown deadline"
                % (_utc_text(observed_end_epoch), expected_deadline))
        bad_host = KNOWN_BAD_MACHINE_IDS.get(
            str(observed.get("provider_machine_id")))
        if bad_host:
            problems.append("known-bad host: %s" % bad_host)
        if problems:
            raise VastError("Vast post-create identity mismatch: %s"
                            % "; ".join(problems))
        return {
            "provider_id": observed["id"],
            "passed": True,
            "expected": dict(expected, terminate_after=expected_deadline,
                             network_volume_id=None),
            "observed": observed,
            "terminate_after_observable": observed_end_epoch is not None,
            "terminate_after_observed": (
                None if observed_end_epoch is None
                else _utc_text(observed_end_epoch)),
            "terminate_after_earlier_than_requested": (
                observed_end_epoch is not None
                and observed_end_epoch < deadline_epoch),
        }

    def attest_live_resource(
            self, provider_id: Any, *, expected_gpu_model: str,
            expected_vram_bytes: int, min_vcpu: int, min_ram_gb: int,
            volume_gb: int, container_disk_gb: int,
            workspace_available_bytes_minimum: int,
            container_available_bytes_minimum: int) -> Dict[str, Any]:
        """Read-only SSH proof that this box is the DEVICE the root wants.

        This is the scientific gate. Provider is not a comparability axis --
        two A100s in two clouds agreed bitwise while an H200 sat 2.973e-04
        nats away -- so what makes a Vast capture comparable to a root
        captured elsewhere is the GPU MODEL and the rebuilt stack. The
        attestation therefore fails unless nvidia-smi AND torch both report
        the expected model with VRAM inside a +/-10% band.

        Two Vast-specific additions, both paid for in real failures:

        * **Hub identity, judged by `fidelity.tlsguard`.** Machine 68004
          could not complete a TLS handshake to huggingface.co and killed a
          capture at the setup stage on 2026-09-05, after the box was already
          billing. tlsguard's verdict is recorded verbatim here and this
          attestation FAILS when it refuses; the causes it distinguishes
          matter, because 68004 turned out to be forged DNS on its path
          rather than an interception proxy, and a refusal that asserts
          interception accuses a host operator of something not measured.
          The exact-id refusal for a host already known to fail is separate
          and stands on the failure alone.
        * **A status field is a claim about the provider's INTENT, never
          evidence of reachability.** A Vast box reported `cur_state` and
          `actual_status` running for ~14 minutes while its reverse tunnel
          was dead and ssh died at `kex_exchange_identification`
          (2026-09-06, instance 50054271). Reachability is proven by this
          round trip and never by a status field.
        """
        instance_id = _provider_id(provider_id)
        model = _gpu_id(expected_gpu_model, "expected GPU model")
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
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise VastError("%s must be a positive integer" % key)
        expected = dict(expected_numbers, gpu_model=model,
                        hub_probe_hosts=list(HUB_PROBE_HOSTS))
        command = (
            "python3 -c 'import base64;"
            "exec(base64.b64decode(\"%s\").decode(\"utf-8\"))'"
            % base64.b64encode(
                _LIVE_ATTEST_SCRIPT.encode("utf-8")).decode("ascii"))
        observed = None
        transport_error = None
        controller_send_epoch = time.time()
        controller_receive_epoch = controller_send_epoch
        if self.dry:
            transport_error = "dry mode cannot attest a live resource"
            controller_receive_epoch = time.time()
        else:
            try:
                raw = self.exec_stdout(instance_id, command, timeout=300)
                observed = _strict_json_loads(raw)
            except Exception as exc:                      # noqa: BLE001
                transport_error = redact(str(exc))[:500]
            finally:
                controller_receive_epoch = time.time()
        round_trip_seconds = max(
            0.0, controller_receive_epoch - controller_send_epoch)
        remote_epoch = (observed.get("remote_time_epoch")
                        if isinstance(observed, dict) else None)
        remote_utc = (observed.get("remote_time_utc")
                      if isinstance(observed, dict) else None)
        remote_utc_epoch = None
        if isinstance(remote_utc, str):
            try:
                remote_utc_epoch = _utc_epoch(
                    remote_utc, "remote attestation time")
            except VastError:
                pass
        midpoint_epoch = controller_send_epoch + round_trip_seconds / 2.0
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
            "controller_send_epoch": controller_send_epoch,
            "controller_send_utc": _utc_text(controller_send_epoch),
            "controller_receive_epoch": controller_receive_epoch,
            "controller_receive_utc": _utc_text(controller_receive_epoch),
            "round_trip_seconds": round_trip_seconds,
            "remote_time_epoch": remote_epoch,
            "remote_time_utc": remote_utc,
            "clock_skew_seconds": clock_skew_seconds,
            "allowed_skew_seconds": allowed_skew_seconds,
            "within_bound": clock_ok,
        }
        failures: List[str] = []
        checks: Dict[str, bool] = {"remote_clock": clock_ok}
        vram_floor = expected_vram_bytes * 9 // 10
        vram_ceiling = expected_vram_bytes * 11 // 10
        single_filesystem = None
        if not isinstance(observed, dict):
            failures.append("live SSH attestation unavailable")
        else:
            exact_observed = {
                "remote_time_epoch", "remote_time_utc", "logical_cpus",
                "memtotal_bytes", "effective_memory_bytes",
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
            checks["one_nvidia_gpu"] = (
                observed.get("nvidia_smi_exit_code") == 0
                and isinstance(gpus, list) and len(gpus) == 1)
            gpu = gpus[0] if checks["one_nvidia_gpu"] else {}
            if gpu and set(gpu) != {
                    "index", "name", "vram_bytes", "vram_used_bytes",
                    "vram_free_bytes", "driver_version"}:
                failures.append("nvidia-smi GPU keys differ")
            observed_name = gpu.get("name") if isinstance(gpu, dict) else None
            observed_vram = (
                gpu.get("vram_bytes") if isinstance(gpu, dict) else None)
            checks["gpu_model"] = (
                isinstance(observed_name, str)
                and observed_name.strip().casefold() == model.casefold())
            checks["gpu_vram"] = (
                isinstance(observed_vram, int)
                and not isinstance(observed_vram, bool)
                and vram_floor <= observed_vram <= vram_ceiling)
            # FREE VRAM, not total: a rented "24 GB" 4090 came with 23424 of
            # 24564 MiB already held by four foreign PIDs (2026-09-06). Total
            # was honestly advertised and the card was useless, so the gate
            # demands the memory the capture can actually allocate and names
            # every process holding the rest.
            observed_free = (
                gpu.get("vram_free_bytes") if isinstance(gpu, dict) else None)
            checks["gpu_vram_free"] = (
                isinstance(observed_free, int)
                and not isinstance(observed_free, bool)
                and observed_free >= vram_floor)
            processes = observed.get("compute_processes")
            if not isinstance(processes, list) or any(
                    not isinstance(row, dict)
                    or set(row) != {"pid", "used_bytes", "process_name"}
                    for row in processes):
                failures.append("GPU compute-process keys differ")
                processes = None
            checks["no_foreign_gpu_processes"] = (
                observed.get("compute_apps_exit_code") == 0
                and processes == [])
            cuda = observed.get("cuda")
            if not isinstance(cuda, dict) or set(cuda) != {
                    "usable", "count", "name", "vram_bytes", "error",
                    "interpreter"}:
                failures.append("CUDA attestation keys differ")
                cuda = {}
            checks["cuda_usable"] = (
                cuda.get("usable") is True and cuda.get("count") == 1
                and isinstance(cuda.get("name"), str)
                and cuda["name"].strip().casefold() == model.casefold()
                and isinstance(cuda.get("vram_bytes"), int)
                and vram_floor <= cuda["vram_bytes"] <= vram_ceiling)
            filesystems = observed.get("filesystems")
            if not isinstance(filesystems, dict) or set(filesystems) != {
                    "root", "workspace"}:
                failures.append("filesystem attestation keys differ")
                filesystems = {}
            filesystem_keys = {
                "path", "mount_point", "fs_type", "source", "device",
                "total_bytes", "available_bytes", "error",
            }
            for role in ("root", "workspace"):
                row = filesystems.get(role)
                if not isinstance(row, dict) or set(row) != filesystem_keys:
                    failures.append("%s filesystem keys differ" % role)
            root = filesystems.get("root", {})
            workspace = filesystems.get("workspace", {})
            checks["workspace_present"] = (
                isinstance(workspace, dict) and workspace.get("error") is None
                and workspace.get("path") == "/workspace")
            # ONE pod-scoped disk is the normal Vast shape, so when /workspace
            # and / share a device the two byte floors are demands on the SAME
            # free space and must be satisfied TOGETHER, not each alone.
            single_filesystem = bool(
                isinstance(root, dict) and isinstance(workspace, dict)
                and root.get("device") is not None
                and root.get("device") == workspace.get("device"))
            disk_required_gb = int(volume_gb) + int(container_disk_gb)
            if single_filesystem:
                checks["disk_total_bytes"] = (
                    isinstance(root.get("total_bytes"), int)
                    and root["total_bytes"] >= disk_required_gb * 900_000_000)
                checks["disk_available_bytes"] = (
                    isinstance(root.get("available_bytes"), int)
                    and not isinstance(root.get("available_bytes"), bool)
                    and root["available_bytes"]
                    >= workspace_available_bytes_minimum
                    + container_available_bytes_minimum)
            else:
                checks["disk_total_bytes"] = (
                    isinstance(root.get("total_bytes"), int)
                    and isinstance(workspace.get("total_bytes"), int)
                    and root["total_bytes"]
                    >= int(container_disk_gb) * 900_000_000
                    and workspace["total_bytes"]
                    >= int(volume_gb) * 900_000_000)
                checks["disk_available_bytes"] = (
                    isinstance(root.get("available_bytes"), int)
                    and isinstance(workspace.get("available_bytes"), int)
                    and root["available_bytes"]
                    >= container_available_bytes_minimum
                    and workspace["available_bytes"]
                    >= workspace_available_bytes_minimum)
            # No hub checks here: `fidelity.tlsguard` owns that verdict and
            # runs its own collector over this same channel below.
            failures.extend(
                name for name, passed in sorted(checks.items()) if not passed)
        # Whether this box talks to the REAL Hub is `fidelity.tlsguard`'s
        # single implementation, and it is strictly stronger than anything
        # this file could do inline: it verifies the chain against our own
        # digest-pinned roots, compares the leaf against one the controller
        # verified, and separates "this host is lying" from "we could not
        # reach anything". It drives its own collector over this adapter's
        # exec channel, so the probe still runs BEFORE any credential exists
        # on the box. If tlsguard cannot be reached at all the attestation
        # FAILS -- there is no local floor to fall back to, because a second
        # implementation of one security property is how they drift.
        machine = str((self.get_lifecycle_resource(instance_id) or {})
                      .get("provider_machine_id"))
        hub_verdict: Dict[str, Any] = {
            "source": "tlsguard", "ok": False, "verdict": None,
            "failures": [], "disclosures": [], "evidence": None,
            "hosts": list(HUB_PROBE_HOSTS),
        }
        try:
            from . import tlsguard
            if self.dry:
                raise VastError("dry mode cannot attest a live TLS peer")
            judged = tlsguard.attest_before_credential(
                self, instance_id, host_id=machine,
                hosts=HUB_PROBE_HOSTS, timeout=240.0)
            hub_verdict.update({
                "ok": bool(judged.get("ok")),
                "verdict": judged.get("verdict"),
                "failures": judged.get("failures") or [],
                "disclosures": judged.get("disclosures") or [],
                "evidence": judged.get("evidence") or judged,
            })
        except ImportError as exc:
            hub_verdict["failures"] = [{
                "code": "TLSGUARD-UNAVAILABLE",
                "message": "fidelity.tlsguard is not importable: %s"
                           % redact(str(exc))[:200],
                "remedy": "ship bin/fidelity/tlsguard.py (it is in "
                          "bin/BUNDLE.txt); without it nothing attests that "
                          "this box reaches the real Hub",
            }]
        except Exception as exc:                          # noqa: BLE001
            code = getattr(exc, "code", None)
            hub_verdict["failures"] = [{
                "code": str(code) if code else "TLS-ATTESTATION-FAILED",
                "message": redact(str(getattr(exc, "reason", exc)))[:300],
                "remedy": "; ".join(getattr(exc, "advice", ()) or [
                    "attest the TLS peer from this box before any credential "
                    "reaches it"])[:500],
            }]
            hub_verdict["retryable"] = bool(getattr(exc, "retryable", False))
        checks["hub_identity_attested"] = bool(hub_verdict["ok"])
        if not checks["hub_identity_attested"]:
            failures.append("hub_identity_attested")
        provider_record = {
            "provider_machine_id": None, "provider_host_id": None,
            "location": None, "gpu_type_id": None, "gpu_ram_mib": None,
            "driver_version": None, "cuda_max_good": None,
            "status": None, "actual_status": None, "cost_per_hr": None,
            "known_bad_host": None, "error": None,
        }
        try:
            listed = self.get_lifecycle_resource(instance_id)
            if listed is None:
                raise VastError("instance %s is not listed" % instance_id)
            for key in ("provider_machine_id", "provider_host_id", "location",
                        "gpu_type_id", "gpu_ram_mib", "driver_version",
                        "cuda_max_good", "status", "actual_status",
                        "cost_per_hr"):
                value = listed.get(key)
                provider_record[key] = str(value) if value is not None else None
            provider_record["known_bad_host"] = KNOWN_BAD_MACHINE_IDS.get(
                str(listed.get("provider_machine_id")))
            checks["host_not_known_bad"] = provider_record["known_bad_host"] is None
            # The API's advertised VRAM and nvidia-smi's total are exactly the
            # pair an attestation can get wrong; require them to agree.
            advertised_mib = listed.get("gpu_ram_mib")
            advertised = (
                int(advertised_mib) * 1024 * 1024
                if isinstance(advertised_mib, int)
                and not isinstance(advertised_mib, bool) else None)
            checks["provider_vram_agrees"] = (
                advertised is not None
                and vram_floor <= advertised <= vram_ceiling)
            checks["provider_gpu_model_agrees"] = (
                isinstance(listed.get("gpu_type_id"), str)
                and listed["gpu_type_id"].strip().casefold() == model.casefold())
            for name in ("host_not_known_bad", "provider_vram_agrees",
                         "provider_gpu_model_agrees"):
                if not checks[name]:
                    failures.append(name)
        except Exception as exc:  # noqa: BLE001 - recorded verbatim
            provider_record["error"] = "%s: %s" % (type(exc).__name__, exc)
            failures.append("provider_record_unavailable")
        document = {
            "schema": "fidelity-suite/vast-live-attestation.v1",
            "provider": "vast", "provider_id": instance_id,
            "observed_at_utc": clock["controller_receive_utc"],
            "clock": clock,
            "provider_record": provider_record,
            "single_filesystem": single_filesystem,
            "hub_tls_verdict_source": hub_verdict["source"],
            "hub_tls_verdict": hub_verdict,
            "expected": expected, "observed": observed,
            "transport_error": transport_error,
            # Name the ATTESTER of each half, because a postcondition
            # evaluated by the party it constrains is not proof. `observed`
            # is the rented box describing itself; `provider_record` is
            # Vast describing the same box; `checks` are ours over both. The
            # value is that the two independent parties must AGREE
            # (provider_gpu_model_agrees, provider_vram_agrees) -- a single
            # party's word never carries a check on its own.
            "evidence_sources": {
                "observed": "the rented box, self-reported over "
                            "host-key-authenticated SSH; a hostile host can "
                            "report anything and this is not independently "
                            "verifiable",
                "provider_record": "Vast's own API over TLS, independent of "
                                   "the box",
                "checks": "computed by the controller over both halves, "
                          "requiring the two parties to agree",
                "hub_tls_verdict": hub_verdict["source"],
            },
            "checks": checks, "failures": sorted(set(failures)),
            "ok": bool(not failures and transport_error is None
                       and checks and all(checks.values())),
        }
        return _attestation_seal(document)

    # -- two-phase create: build and freeze before any mutation ------------
    def _ask_offer(self, ask_id: str) -> Dict[str, Any]:
        """The advertised identity of one exact ask, by id.

        `ask_contract_id` is the filter that selects an ask by its own id
        (`id` does not; verified live 2026-09-06). The ADVERTISED figures are
        frozen into the request so the post-create binding check can compare
        the contract against what was offered -- an ask that advertised a
        B200 once handed back an H100, and every ask on this account so far
        has under-quoted the rate by ~23% because it excludes the disk.
        """
        query = {"ask_contract_id": {"eq": int(ask_id)}, "type": "on-demand"}
        got = self._req(
            "GET", "/bundles/?q=" + urllib.parse.quote(json.dumps(query)))
        offers = (got or {}).get("offers")
        if not isinstance(offers, list) or len(offers) != 1:
            raise VastError(
                "Vast ask %s is not exactly one rentable on-demand offer any "
                "more; re-search and prepare again -- an offer that vanishes "
                "between search and rental is ordinary on a marketplace"
                % ask_id)
        offer = offers[0]
        if not isinstance(offer, dict) or _provider_id(
                offer.get("id"), "Vast offer id") != ask_id:
            raise VastError("Vast ask lookup returned a different offer id")
        return offer

    def _generate_pinned_host_key(self) -> Dict[str, str]:
        """Generate the ED25519 host key the instance will be REQUIRED to have.

        Trust-on-first-use is removed rather than mitigated: the fingerprint
        is known before the instance exists, and only a box that received our
        TLS-authenticated rental request can hold the private half. The
        private key travels in `env` (never in `onstart`, which the API echoes
        back in the instance object, and never in a log: the instance log
        endpoint hands out a PUBLIC S3 URL).

        UNVERIFIED LIVE: whether Vast's `onstart` completes before the first
        accepted SSH connection has not been measured on a live box, so a
        pinned scan may need to wait for sshd to be re-keyed.
        `ssh_host_ed25519_fingerprint` polls for that and refuses on timeout
        rather than accepting whatever key answers.
        """
        from .sshbase import _bounded_process
        import tempfile
        directory = None
        try:
            directory = tempfile.mkdtemp(prefix="fid-vast-hostkey-")
            os.chmod(directory, 0o700)
            path = os.path.join(directory, "ssh_host_ed25519_key")
            result = _bounded_process(
                ["ssh-keygen", "-t", "ed25519", "-N", "", "-C",
                 "fidelity-vast-pinned", "-f", path],
                timeout=60, stdout_max_bytes=65536,
                stderr_max_bytes=65536, label="ssh-keygen hostkey")
            if result["returncode"] != 0:
                raise VastError("ssh-keygen could not generate a host key")
            with open(path, "rb") as handle:
                private = handle.read()
            listing = _bounded_process(
                ["ssh-keygen", "-E", "sha256", "-lf", path + ".pub"],
                timeout=30, stdout_max_bytes=65536,
                stderr_max_bytes=65536, label="ssh-keygen fingerprint")
            fields = listing["stdout"].strip().split()
            fingerprint = fields[1] if len(fields) >= 2 else ""
            if re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", fingerprint) is None:
                raise VastError(
                    "generated ED25519 fingerprint is noncanonical")
            encoded = base64.b64encode(private).decode("ascii")
            register_secret(encoded)
            return {"fingerprint": fingerprint, "private_key_b64": encoded}
        finally:
            if directory:
                for name in ("ssh_host_ed25519_key",
                             "ssh_host_ed25519_key.pub"):
                    target = os.path.join(directory, name)
                    if os.path.isfile(target):
                        with open(target, "r+b") as handle:
                            length = os.fstat(handle.fileno()).st_size
                            handle.write(b"\0" * length)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.unlink(target)
                os.rmdir(directory)

    # Written by `onstart`, which Vast runs after init on `runtype: "ssh"`.
    # The key material arrives through the environment and is unset the
    # moment it is on disk; sshd is re-keyed with SIGHUP so the reverse
    # tunnel the proxy depends on is not torn down.
    _HOST_KEY_PIN_SNIPPET = (
        "umask 077\n"
        "printf '%s' \"$FIDELITY_VAST_HOST_KEY_B64\" | base64 -d "
        "> /etc/ssh/ssh_host_ed25519_key\n"
        "chmod 600 /etc/ssh/ssh_host_ed25519_key\n"
        "ssh-keygen -y -f /etc/ssh/ssh_host_ed25519_key "
        "> /etc/ssh/ssh_host_ed25519_key.pub\n"
        "unset FIDELITY_VAST_HOST_KEY_B64\n"
        "pkill -HUP sshd || true\n")

    def prepare_safe_create(self, **kw) -> PreparedVastCreate:
        """Build and freeze the rental request before any provider mutation.

        Everything that can refuse, refuses HERE, while nothing is running
        and nothing is billing: the ask must exist as exactly one rentable
        on-demand offer, its host must not be one already known to break the
        Hub, the advertised GPU must be the model the plan chose, the disk
        must hold volume+container, and a teardown deadline must be present
        and far enough out to finish setup.
        """
        forbidden = ("network_volume_id", "volume_id", "network_mounts",
                     "mounts", "volume_info")
        used = [key for key in forbidden if kw.get(key) is not None]
        if used:
            raise VastError(
                "safe Vast profile refuses network/custom volumes: %s -- a "
                "volume outlives the rental and keeps charging"
                % ", ".join(sorted(used)))
        native = [key for key in ("docker_cmd", "docker_entrypoint",
                                  "docker_args", "args")
                  if kw.get(key) not in (None, "")]
        if native:
            raise VastError(
                "safe Vast profile is SSH-driven and refuses native docker "
                "launch: %s -- use create() for the container rehearsal path"
                % ", ".join(sorted(native)))
        if kw.get("env"):
            raise VastError(
                "safe Vast SSH profile refuses caller-supplied provider env")
        if kw.get("spot", False) is not False or kw.get("is_bid", False):
            raise VastError(
                "safe Vast profile requires an on-demand rental, never a bid: "
                "an interruptible contract cannot guarantee a capture")
        if kw.get("offer", "on-demand") != "on-demand":
            raise VastError(
                "safe Vast profile requires offer exactly on-demand")
        region = kw.get("region")
        if region not in (None, "", "marketplace"):
            raise VastError(
                "Vast is a marketplace and has no region tiers to select "
                "(got %r); leave region unset and pin the exact ask instead"
                % region)

        def _positive_int(key, default=None):
            value = kw.get(key, default)
            if isinstance(value, bool):
                raise VastError("%s must be a positive integer" % key)
            if isinstance(value, int):
                parsed = value
            elif isinstance(value, str) and value.isdigit():
                parsed = int(value)
            else:
                raise VastError("%s must be a positive integer" % key)
            if parsed <= 0:
                raise VastError("%s must be a positive integer" % key)
            return parsed

        terminate_after = _terminate_after(kw)
        if terminate_after is None:
            raise VastError(
                "safe Vast create requires terminate_after or "
                "terminate_after_epoch")
        termination_epoch = _utc_epoch(terminate_after, "terminate_after")
        now = time.time()
        if termination_epoch - now < MIN_CREATE_SETUP_SECONDS:
            raise VastError(
                "Vast terminate_after must be at least %d seconds in the "
                "future" % MIN_CREATE_SETUP_SECONDS)
        ask_id = _provider_id(
            kw.get("ask_id") or kw.get("offer_id"), "Vast ask id")
        name = str(kw.get("name") or "").strip()
        if not name:
            raise VastError("safe Vast create requires an exact lease name")
        gpu = _gpu_id(kw.get("gpu_type") or kw.get("gpu"),
                      "Vast create gpu_type")
        volume_gb = _positive_int("storage_gb")
        container_disk_gb = _positive_int("container_disk_gb")
        gpu_count = _positive_int("num_gpus", 1)
        min_vcpu = _positive_int("min_vcpu", 4)
        min_ram_gb = _positive_int("min_ram_gb", 16)
        disk_gb = volume_gb + container_disk_gb
        image = str(kw.get("image") or DEFAULT_IMAGE).strip()
        if not image:
            raise VastError("Vast image must be nonempty")
        offer = self._ask_offer(ask_id)
        machine_id = (None if offer.get("machine_id") is None
                      else str(offer["machine_id"]))
        bad_host = KNOWN_BAD_MACHINE_IDS.get(str(machine_id))
        if bad_host:
            raise VastError(
                "refusing Vast ask %s: %s. Pick another offer -- this host is "
                "routinely the cheapest one that fits, which is exactly how a "
                "capture landed on it before." % (ask_id, bad_host))
        advertised_gpu = str(offer.get("gpu_name") or "")
        if advertised_gpu.strip().casefold() != gpu.strip().casefold():
            raise VastError(
                "Vast ask %s advertises %r, not the %r the plan chose"
                % (ask_id, advertised_gpu, gpu))
        advertised_gpus = offer.get("num_gpus")
        if advertised_gpus != gpu_count:
            raise VastError(
                "Vast ask %s advertises %r GPUs, not %d"
                % (ask_id, advertised_gpus, gpu_count))
        advertised_disk = _finite_decimal(
            offer.get("disk_space"), "Vast offer disk_space", positive=True)
        if advertised_disk < disk_gb:
            raise VastError(
                "Vast ask %s offers %s GB of disk, below the %d GB the plan "
                "needs; vast disk is chosen at rent time and cannot grow"
                % (ask_id, format(advertised_disk, "f"), disk_gb))
        advertised_rate = _finite_decimal(
            offer.get("dph_total"), "Vast offer dph_total", positive=True)
        host_key = self._generate_pinned_host_key()
        # Vast's REST body takes `env` as docker flag text; the key material
        # is the only thing that travels in it and it never enters onstart.
        env_text = "-e FIDELITY_VAST_HOST_KEY_B64=%s" % host_key[
            "private_key_b64"]
        body = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "label": name,
            "runtype": "ssh",
            "onstart": self._HOST_KEY_PIN_SNIPPET,
            "env": env_text,
            # Vast's provider-side deadline. UNVERIFIED LIVE: that Vast
            # honours `duration` on a rental has not been observed here, so
            # `validate_safe_resource_binding` refuses a contract whose
            # end_date is later than the deadline rather than assuming it.
            "duration": int(termination_epoch - now),
        }
        public_key = ""
        candidate = os.path.expanduser(self.ssh_key) + ".pub"
        if os.path.isfile(candidate):
            public_key = open(candidate, encoding="utf-8").read().strip()
        if not public_key:
            raise VastError(
                "safe Vast create needs the controller's SSH public key at "
                "%s: without it the rental accepts no unattended session"
                % candidate)
        body["extra_env"] = {"PUBLIC_KEY": public_key}
        request_identity = {
            "ask_id": ask_id,
            "provider_machine_id": machine_id,
            "provider_host_id": (None if offer.get("host_id") is None
                                 else str(offer["host_id"])),
            "advertised_gpu_name": advertised_gpu,
            "advertised_gpu_ram_mib": offer.get("gpu_ram"),
            "advertised_disk_gb": float(advertised_disk),
            "advertised_dph_total": format(advertised_rate, "f"),
            "advertised_geolocation": offer.get("geolocation"),
            "advertised_cuda_max_good": offer.get("cuda_max_good"),
            "gpu_type_id": gpu,
            "gpu_count": gpu_count,
            "volume_gb": volume_gb,
            "container_disk_gb": container_disk_gb,
            "disk_gb": disk_gb,
            "min_vcpu": min_vcpu,
            "min_ram_gb": min_ram_gb,
            "name": name,
            "image_name": image,
            "is_spot": False,
            "offer": "on-demand",
            "runtype": "ssh",
            "secure_cloud": False,
            "terminate_after": terminate_after,
            "duration_seconds": body["duration"],
            "network_volume_id": None,
            "pinned_host_key_fingerprint": host_key["fingerprint"],
            "public_key_sha256": hashlib.sha256(
                public_key.encode("utf-8")).hexdigest(),
        }
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            API + "/asks/%s/" % ask_id, data=payload, method="PUT",
            headers={"Content-Type": "application/json",
                     "User-Agent": "quant-fidelity-suite/0.1",
                     "Authorization": "Bearer " + self._load_key()})
        identity_json = json.dumps(
            request_identity, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("utf-8")
        return PreparedVastCreate(
            http_request=request,
            http_opener=safe_urlopen,
            request_body=payload,
            request_identity_json=identity_json,
            name=name, ask_id=ask_id, machine_id=machine_id,
            disk_gb=disk_gb, storage_gb=volume_gb,
            container_disk_gb=container_disk_gb, image_name=image,
            terminate_after=terminate_after,
            duration_seconds=body["duration"],
            host_key_fingerprint=host_key["fingerprint"],
            dry_run=self.dry)

    def submit_prepared_create(
            self, prepared: PreparedVastCreate) -> Dict[str, Any]:
        """Submit the frozen request, so a LOST RESPONSE stays reconcilable.

        The response carries `new_contract`, an exact integral id. A refusal
        Vast states explicitly (`success: false` with no integral field) is
        raised as `VastCreateRejectedError`, which means nothing was accepted
        and no id needs hunting; anything ambiguous keeps the fail-closed
        path, and the caller reconciles by searching the account inventory
        for the exact lease `label`.
        """
        if prepared.dry_run:
            return {
                "dry_run": True,
                "request": prepared.to_dict()["request_identity"],
                "prepared_create": prepared.to_dict(),
            }
        gap = time.time() - Vast._last_call
        if gap < self._MIN_INTERVAL:
            time.sleep(self._MIN_INTERVAL - gap)
        Vast._last_call = time.time()
        try:
            with prepared.http_opener(
                    prepared.http_request, timeout=180) as response:
                raw = response.read().decode("utf-8", "replace")
            document = _strict_json_loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            detail = redact(exc.read(300).decode("utf-8", "replace"))
            try:
                parsed = _strict_json_loads(detail)
            except VastError:
                parsed = None
            reasons = _definitive_create_rejection_reasons(parsed)
            if reasons:
                raise VastCreateRejectedError(
                    "Vast refused the rental: %s" % "; ".join(reasons),
                    reasons)
            raise VastError("Vast HTTP %d on the prepared create: %s"
                            % (exc.code, detail))
        except VastError:
            raise
        except Exception as exc:                          # noqa: BLE001
            raise VastError("Vast prepared create request failed: %s"
                            % redact(str(exc)))
        if not isinstance(document, dict):
            raise VastError("Vast create returned non-object JSON")
        if document.get("success") is not True:
            reasons = _definitive_create_rejection_reasons(document)
            message = ("Vast refused the rental: %s"
                       % redact(json.dumps(document)[:300]))
            if reasons:
                raise VastCreateRejectedError(message, reasons)
            raise VastError(message)
        contract = _provider_id(
            document.get("new_contract"), "Vast create new_contract")
        self._pinned_host_keys[contract] = prepared.host_key_fingerprint
        return {
            "machine_id": contract,
            "instance_id": contract,
            "ask_id": prepared.ask_id,
            "name": prepared.name,
            "request": prepared.to_dict()["request_identity"],
            "prepared_create": prepared.to_dict(),
            "requested_terminate_after": prepared.terminate_after,
            "storage_gb": prepared.storage_gb,
            "container_disk_gb": prepared.container_disk_gb,
            "disk_gb": prepared.disk_gb,
            "image_name": prepared.image_name,
            "pinned_host_key_fingerprint": prepared.host_key_fingerprint,
        }

    # -- what did it cost, and whose clock says so? ------------------------
    def _instance_log_text(self, instance_id: str, *, tail: int,
                           timeout: float) -> str:
        """Fetch the authenticated instance log through the official channel.

        `PUT /instances/request_logs/<id>/` answers with a `result_url` the
        host uploads to; the file appears seconds later. Note for anyone
        adding output to a run: that URL is PUBLIC
        (`s3.amazonaws.com/public.vast.ai/instance_logs/<sha>.log`), so a
        secret printed by a run is world-readable to anyone holding it. The
        REQUEST is authenticated, which is what makes the fingerprint line
        below provider-attested rather than trust-on-first-use.
        """
        got = self._req("PUT", "/instances/request_logs/%s/" % instance_id,
                        {"tail": str(int(tail))}, timeout=60)
        url = (got or {}).get("result_url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise VastError(
                "Vast log request returned no https result_url for %s"
                % instance_id)
        deadline = time.monotonic() + max(1.0, float(timeout))
        last = ""
        while True:
            request = urllib.request.Request(
                url, method="GET",
                headers={"User-Agent": "quant-fidelity-suite/0.1"})
            try:
                with safe_urlopen(request, timeout=30) as response:
                    text = response.read(
                        MAX_LOG_RESPONSE_BYTES + 1).decode("utf-8", "replace")
                if len(text) > MAX_LOG_RESPONSE_BYTES:
                    raise VastError(
                        "Vast instance log exceeded %d bytes"
                        % MAX_LOG_RESPONSE_BYTES)
                last = text
                if "No such container" not in text:
                    return text
            except VastError:
                raise
            except Exception as exc:                      # noqa: BLE001
                last = "%s: %s" % (type(exc).__name__, redact(str(exc))[:200])
            if time.monotonic() >= deadline:
                return last
            time.sleep(_HOST_KEY_LOG_POLL_SECONDS)

    def ssh_host_ed25519_fingerprint(
            self, provider_id: Any, *, timeout: float = 900) -> Dict[str, Any]:
        """Authenticate the host key WITHOUT trusting first contact.

        Two sources, in order of strength:

        1. **The key pinned at create time.** `prepare_safe_create` generates
           the ED25519 host key and delivers it through the rental request,
           so the expected fingerprint is known before the instance exists
           and only a box that received our TLS-authenticated request can
           present it. This polls `ssh-keyscan` until the live host presents
           exactly that key, and refuses on timeout -- it never accepts
           whatever answers.
        2. **The provider's authenticated log channel**, as RunPod does.

        LIVE FINDING (2026-09-06, instance 50054271): an unmodified Vast
        image's log contains sshd's "Server listening on 0.0.0.0 port 22"
        but NO `256 SHA256:... (ED25519)` line. The only ED25519 strings
        there are the container's own known-hosts warnings about the
        `ssh2.vast.ai` JUMP PROXY, whose key belongs to Vast and NOT to the
        instance -- accepting one would authenticate the wrong host entirely,
        so they are excluded by construction (the regex demands ssh-keygen's
        `256 SHA256:<43> <comment> (ED25519)` listing form). With no pin and
        no printed line this therefore REFUSES and names both remedies.
        """
        instance_id = _provider_id(provider_id)
        try:
            budget = float(timeout)
        except (TypeError, ValueError, OverflowError):
            raise VastError("Vast host-key timeout must be finite and positive")
        if not math.isfinite(budget) or budget <= 0:
            raise VastError("Vast host-key timeout must be finite and positive")
        pinned = self._pinned_host_keys.get(instance_id)
        deadline = time.monotonic() + budget
        if pinned:
            if self.dry:
                raise VastError("dry-run cannot scan a live SSH host key")
            last_error = None
            while True:
                try:
                    scanned = self.scan_host_key(instance_id)
                except JLError as exc:
                    last_error = redact(str(exc))[:200]
                    scanned = None
                if scanned is not None:
                    if scanned["fingerprint"] == pinned:
                        return {
                            "schema":
                                "fidelity-suite/vast-host-key-evidence.v1",
                            "provider": "vast",
                            "provider_id": instance_id,
                            "source": "pinned-at-create",
                            "endpoint_origin": API.rsplit("/api", 1)[0],
                            "observed_at_utc": _utc_text(time.time()),
                            "fingerprint": scanned["fingerprint"],
                            "pinned_fingerprint": pinned,
                            "host": scanned["host"],
                            "port": scanned["port"],
                            "known_hosts_entry_sha256": hashlib.sha256(
                                scanned["known_hosts_entry"].encode("utf-8")
                            ).hexdigest(),
                        }
                    raise VastError(
                        "Vast instance %s presents ED25519 %s, not the %s "
                        "pinned into its rental request: the box answering is "
                        "not the box we asked for, or its onstart never "
                        "installed the key. Destroy it and record host %s."
                        % (instance_id, scanned["fingerprint"], pinned,
                           (self.get_lifecycle_resource(instance_id) or {})
                           .get("provider_machine_id")))
                if time.monotonic() >= deadline:
                    raise VastError(
                        "Vast instance %s never presented the pinned ED25519 "
                        "host key within %gs (last keyscan error: %s). Vast's "
                        "onstart ordering relative to the first accepted SSH "
                        "connection is unverified, so this may be a slow "
                        "re-key; it is never a licence to accept another key."
                        % (instance_id, budget, last_error))
                time.sleep(_HOST_KEY_LOG_POLL_SECONDS)
        text = self._instance_log_text(
            instance_id, tail=1000,
            timeout=min(120.0, max(1.0, deadline - time.monotonic())))
        for line in text.splitlines():
            match = _HOST_KEY_LOG_RE.fullmatch(line.strip())
            if match is None:
                continue
            canonical = line.strip()
            return {
                "schema": "fidelity-suite/vast-host-key-evidence.v1",
                "provider": "vast",
                "provider_id": instance_id,
                "source": "authenticated-instance-log",
                "endpoint_origin": API.rsplit("/api", 1)[0],
                "observed_at_utc": _utc_text(time.time()),
                "line": canonical,
                "line_sha256": hashlib.sha256(
                    canonical.encode("utf-8")).hexdigest(),
                "fingerprint": match.group(1),
                "pinned_fingerprint": None,
            }
        raise VastError(
            "Vast exposes no authenticated ED25519 host-key fingerprint for "
            "instance %s: its log carries sshd's startup but not a "
            "`256 SHA256:... (ED25519)` listing, and the ED25519 lines it "
            "does carry belong to the ssh2.vast.ai jump proxy, not to this "
            "instance. Either rent through prepare_safe_create, which pins a "
            "generated host key and needs no log, or have the image print "
            "`ssh-keygen -E sha256 -lf /etc/ssh/ssh_host_ed25519_key.pub` at "
            "startup -- and note that Vast's log URL is public, so print the "
            "FINGERPRINT and nothing else." % instance_id)

    def billing_history(self, instance_id: Any, *, start_time: str,
                        end_time: str,
                        bucket_size: str = "day") -> Dict[str, Any]:
        """Validate Vast's official per-instance charge rows for one window.

        `GET /charges/?select_filters={"day":{"gte":D,"lte":D}}` is the only
        per-resource billing response Vast publishes, and it has three
        properties this method is built around, all verified live 2026-09-06
        over 39 rows:

        * **The bucket is one UTC DAY.** There is no hourly form, so a
          `bucket_size` of "hour" is refused rather than approximated.
        * **Server-side selection by instance is silently IGNORED** -- adding
          `source` or `instance_id` to the filter returned the identical 39
          rows -- so the per-instance selection is done HERE on the exact
          `source` string `instance-<id>`, never trusted to the query.
        * **Each row's `amount` equals the sum of its gpu/disk/bwd/bwu items
          exactly**, which is asserted, so a row cannot carry a charge its
          breakdown does not account for.

        `next_token` is refused rather than silently truncating: an
        incomplete page cannot prove a total.
        """
        wanted = _provider_id(instance_id)
        start = _exact_utc(start_time, "start_time")
        end = _exact_utc(end_time, "end_time")
        start_epoch = _utc_epoch(start, "start_time")
        end_epoch = _utc_epoch(end, "end_time")
        if end_epoch <= start_epoch:
            raise VastError("billing end_time must follow start_time")
        if bucket_size != "day":
            raise VastError(
                "Vast bills per UTC day and publishes no finer bucket; pass "
                "bucket_size='day' (got %r)" % bucket_size)
        first_day = _day_index(start_epoch)
        last_day = _day_index(end_epoch)
        query = {"day": {"gte": first_day, "lte": last_day}}
        doc = self._req("GET", "/charges/?select_filters="
                        + urllib.parse.quote(json.dumps(query)))
        if not isinstance(doc, dict) or set(doc) != {
                "success", "count", "total", "results", "next_token"}:
            raise VastError(
                "Vast charge response keys differ from the observed schema")
        if doc["success"] is not True:
            raise VastError("Vast charge response is not a success")
        if doc.get("next_token") is not None:
            raise VastError(
                "Vast charge listing is paginated (next_token present), so "
                "this window cannot be proven complete; narrow the window")
        rows = doc["results"]
        if not isinstance(rows, list):
            raise VastError("Vast charge results is not a list")
        for field in ("count", "total"):
            value = doc[field]
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value != len(rows)):
                raise VastError(
                    "Vast charge metadata %s disagrees with the rows returned"
                    % field)
        source = "instance-%s" % wanted
        matched = []
        totals: Dict[str, Decimal] = {}
        grand = Decimal("0")
        for row in rows:
            if not isinstance(row, dict) or set(row) != set(_CHARGE_ROW_KEYS):
                raise VastError(
                    "Vast charge row keys differ from the observed schema")
            if row.get("type") != "instance":
                raise VastError(
                    "Vast charge row is not an instance charge: %r"
                    % row.get("type"))
            if row.get("source") != source:
                continue
            day_start = row.get("start")
            if (isinstance(day_start, bool)
                    or not isinstance(day_start, int)
                    or day_start != row.get("end")
                    or day_start % VAST_DAY_SECONDS
                    or not first_day * VAST_DAY_SECONDS <= day_start
                    <= last_day * VAST_DAY_SECONDS):
                raise VastError(
                    "Vast charge row is not a UTC day inside the requested "
                    "window: start=%r end=%r"
                    % (row.get("start"), row.get("end")))
            amount = _finite_decimal(
                row.get("amount"), "Vast charge amount", nonnegative=True)
            items = row.get("items")
            if not isinstance(items, list) or not items:
                raise VastError("Vast charge row carries no item breakdown")
            item_sum = Decimal("0")
            for item in items:
                if not isinstance(item, dict) or set(item) != set(
                        _CHARGE_ROW_KEYS):
                    raise VastError("Vast charge item keys differ")
                if item.get("items"):
                    raise VastError("Vast charge item nests further items")
                kind = item.get("type")
                if not isinstance(kind, str) or not kind:
                    raise VastError("Vast charge item has no type")
                value = _finite_decimal(
                    item.get("amount"), "Vast charge item amount",
                    nonnegative=True)
                item_sum += value
                totals[kind] = totals.get(kind, Decimal("0")) + value
            if item_sum != amount:
                raise VastError(
                    "Vast charge row for %s on %s totals %s but its items sum "
                    "to %s" % (source, _utc_text(day_start),
                               format(amount, "f"), format(item_sum, "f")))
            grand += amount
            matched.append(row)
        if not matched:
            raise VastError(
                "Vast has no charge record for instance %s in %s..%s yet; "
                "reconciliation remains unresolved" % (wanted, start, end))
        return {
            "schema": "fidelity-suite/vast-billing-evidence.v1",
            "provider": "vast",
            "instance_id": wanted,
            "query": {"select_filters": query, "start_time": start,
                      "end_time": end, "bucket_size": bucket_size,
                      "source": source},
            "records": matched,
            "metadata": {
                "account_rows_examined": len(rows),
                "matched_row_count": len(matched),
                "day_index_range": [first_day, last_day],
                "selection": "client-side on the exact source string; Vast "
                             "ignores server-side instance filters",
                "totals": {kind: format(value, "f")
                           for kind, value in sorted(totals.items())},
            },
            "total_amount": format(grand, "f"),
            "retrieved_at_utc": _utc_text(time.time()),
        }

    def reconcile_billing(self, lease: Dict[str, Any], *,
                          now: Optional[float] = None) -> Dict[str, Any]:
        """Return only a post-absence, independently stable cost closure.

        Vast's day row keeps moving while the day is open -- the row for the
        current day existed and grew within minutes of a rental starting
        (2026-09-06) -- so a closure taken before UTC midnight could seal a
        partial bill as reconciled. The closure therefore requires the
        instance to be proven absent, the DAY containing that absence to have
        closed, a 300 s stabilization on top, and two independent retrievals
        that agree byte for byte. Until then this raises and the caller
        records billing as pending, which the reaper settles on a later
        sweep. Same shape as RunPod's hour rule with a ~24 h window: a
        scheduling fact, not a defect.

        No local arithmetic is ever substituted: an hourly residual the
        provider has not yet priced stays unpriced, because a computed cost
        that looks settled is worse than an honest gap.
        """
        ids = sorted({_provider_id(value, "lease provider_resource_id")
                      for value in lease.get("provider_resource_ids") or []
                      if str(value).strip()})
        if not ids:
            raise VastError(
                "Vast billing reconciliation needs at least one exact "
                "instance id")
        create = lease.get("create") or {}
        start = _exact_utc(create.get("pre_create_observed_at"),
                           "lease pre_create_observed_at")
        absence_events = [item for item in lease.get("history") or []
                          if item.get("to") == "ABSENCE_CONFIRMED"]
        if not absence_events:
            raise VastError("lease has no provider-absence event")
        end = _exact_utc(absence_events[-1].get("at"), "lease absence time")
        absence_epoch = _utc_epoch(end, "lease absence time")
        stabilization_seconds = 300
        instant = time.time() if now is None else float(now)
        if instant - absence_epoch < stabilization_seconds:
            raise VastError(
                "Vast billing remains inside the 300-second post-absence "
                "stabilization window")
        absence_day_end = (
            absence_epoch // VAST_DAY_SECONDS + 1) * VAST_DAY_SECONDS
        if instant < absence_day_end + stabilization_seconds:
            raise VastError(
                "the Vast UTC day containing the absence (%s) has not closed "
                "and stabilized yet; its charge row is still moving, so a "
                "closure now would seal a partial bill as reconciled. The "
                "reaper settles it on a sweep after %s"
                % (_utc_text(absence_day_end - VAST_DAY_SECONDS)[:10],
                   _utc_text(absence_day_end + stabilization_seconds)))

        def retrieve() -> Dict[str, Any]:
            histories = []
            total = Decimal("0")
            for instance_id in ids:
                history = self.billing_history(
                    instance_id, start_time=start, end_time=end)
                total += _finite_decimal(
                    history["total_amount"],
                    "Vast billing total for %s" % instance_id,
                    nonnegative=True)
                histories.append(history)
            return {
                "reconciled": True,
                "provider": "vast",
                "provider_resource_ids": ids,
                "billing_histories": histories,
                "total_amount": format(total, "f"),
                "evidence": {
                    "schema": "fidelity-suite/vast-billing-retrieval.v1",
                    "retrieval_id": secrets.token_hex(12),
                    "retrieved_at_utc": _utc_text(time.time()),
                },
            }

        def closure(evidence: Dict[str, Any]) -> Dict[str, Any]:
            result = json.loads(json.dumps(
                evidence, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False))
            result.pop("evidence", None)
            for history in result["billing_histories"]:
                history.pop("retrieved_at_utc", None)
            return result

        first = retrieve()
        second = retrieve()
        first_closure = closure(first)
        second_closure = closure(second)
        if first_closure != second_closure:
            raise VastError(
                "Vast billing changed between independent retrievals")
        result = dict(second)
        result["evidence"] = {
            "schema": "fidelity-suite/vast-billing-stabilization.v1",
            "absence_confirmed_at": end,
            "absence_day_closed_at": _utc_text(absence_day_end),
            "minimum_stabilization_seconds": stabilization_seconds,
            "closure_sha256": hashlib.sha256(json.dumps(
                second_closure, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest(),
            "first_retrieval": first["evidence"],
            "second_retrieval": second["evidence"],
        }
        return result

    def _refuse_credential_payload(self, *, env: Any, onstart: Any,
                                   docker_cmd: Any) -> None:
        """Refuse credential-shaped material bound for the create body.

        A create payload is exposed BEFORE the box exists: it lands in Vast's
        own records and in the host's `docker run` environment with no host
        key, no attestation and no TLS check yet possible, because there is no
        instance yet. No ordering fix can protect it, so the adapter -- the
        last place that can tell -- refuses. `docs/CLOUD-RECIPES.md`'s advice
        to prefer `env` over argv was protecting against a DIFFERENT leak
        (argv in a host process list) and is superseded for credentials:
        provider-persisted is worse than process-visible on a box we
        authenticated. Vast container mode may therefore measure PUBLIC
        artifacts and nothing else.

        The judgement is `fidelity.tlsguard`'s, not ours: one implementation
        for all four adapters, whose findings name the JSON path, the key and
        a character count and never the value. If it cannot be imported this
        REFUSES rather than transmitting an unchecked payload.
        """
        payload = {"provider": "vast", "env": env, "onstart": onstart,
                   "docker_cmd": docker_cmd}
        try:
            from .tlsguard import (
                TlsRefusal, refuse_credential_in_provider_payload)
        except ImportError as exc:
            raise VastError(
                "cannot check a Vast create payload for credentials: "
                "fidelity.tlsguard is not importable (%s). It is in "
                "bin/BUNDLE.txt; refusing rather than transmitting an "
                "unchecked payload into the provider's own records."
                % redact(str(exc))[:200])
        try:
            refuse_credential_in_provider_payload(payload, operation="create")
        except TlsRefusal as exc:
            # The guard's own exception must NOT cross this boundary: every
            # caller catches `VastError` (the JLError family), so a TlsRefusal
            # escaping here dies uncaught instead of refusing. LambdaParity
            # caught exactly that on the committed version, where only
            # TypeError was handled -- and it survived testing because the
            # tlsguard-absent path WAS wrapped while the real path was not.
            # A fallback that differs from the primary is the drift nobody
            # looks at, which is why there is no fallback here any more.
            raise VastError("%s -- %s" % (exc.reason, "; ".join(exc.advice)))
        # The bearer-capability property (a URL with a PATH in a create body
        # IS an authorisation) lived here as the Vast profile's own interim
        # policy until `tlsguard.credential_findings` grew it at 13b4eb5.
        # Deleted rather than left beside the shared detector: the reason it
        # was ever here was that the guard did not carry it, and a second
        # implementation kept "for safety" is the drift this file has already
        # been bitten by once today.

    def create(self, **kw) -> Dict[str, Any]:
        """Marketplace rental, container-native or SSH.

        This is the rehearsal/public-artifact path. A CREDENTIAL-BEARING run
        must go through `prepare_safe_create` + `submit_prepared_create` and
        the SSH+bundle transport: a create body is provider-persisted, so a
        token in it is exposed before any attestation is even possible, and
        `_refuse_credential_payload` fails closed here rather than dutifully
        transmitting it.
        """
        self._refuse_credential_payload(
            env=kw.get("env"), onstart=kw.get("onstart"),
            docker_cmd=kw.get("docker_cmd"))
        if self.dry:
            return {"dry_run": True, **kw}
        ask = kw.get("ask_id") or kw.get("offer_id")
        disk = int(kw.get("storage") or kw.get("storage_gb") or 100)
        if not ask:
            # No ask id supplied: search now for the cheapest bundle that fits.
            # Searching and renting must be one transaction on a marketplace --
            # an offer that vanishes in between is ordinary, not an error.
            want = (kw.get("gpu_type") or kw.get("gpu") or "").strip()
            fits = []
            if want:
                # exact name first, then a substring pass over the catalogue
                fits = self._search(int(kw.get("min_vram_gb") or 0), disk, 20,
                                    gpu_name=want)
            if not fits:
                fits = self.gpus(min_vram_gb=int(kw.get("min_vram_gb") or 0),
                                 min_disk_gb=disk)
            if want:
                # HONOUR the requested GPU. Without this the "cheapest that
                # fits" is whatever the marketplace is dumping -- on this
                # account that was a CMP 170HX, a 64 GB MINING card that
                # satisfies a >=63 GB VRAM filter and is useless for this work.
                # The controller already chose a model; renting a different one
                # silently would make `on_validated_hardware` a lie.
                fits = [o for o in fits
                        if want.lower() in (o.gpu_type or "").lower()]
            if not fits:
                raise VastError(
                    "no rentable Vast offer for %s with >=%d GB VRAM and >=%d GB "
                    "disk" % (want or "any GPU",
                              int(kw.get("min_vram_gb") or 0), disk))
        # Container-native mode.  Vast's `runtype: "args"` preserves the
        # image ENTRYPOINT and passes `args` as CMD -- but has no post-start
        # hook, so preparation (target.json, tokenizer, panel binding) cannot
        # run before the capture.  `runtype: "ssh"` replaces the entrypoint
        # with sshd and runs `onstart` AFTER init -- so the full command
        # (prep + entrypoint) goes in `onstart` as a shell script.  The
        # container stays alive for SSH after the script exits; we destroy
        # it when the result arrives.  Secrets travel in `env`, never in
        # onstart text: a provider may echo the command back, but environment
        # variables it does not.  Triggered by `docker_cmd`; when absent the
        # SSH path below is byte-identical.
        docker_cmd = kw.get("docker_cmd")
        if docker_cmd is not None:
            onstart = kw.get("onstart") or ""
            # If onstart is supplied it is a prep script; the docker_cmd
            # (the capture argv) is appended after it so both run in one
            # shell.  If onstart is empty, docker_cmd runs alone.
            if onstart and docker_cmd:
                exec_line = (
                    "exec python3.12 /opt/fidelity/suite/bin/container_entry.py "
                    + " ".join("'%s'" % a.replace("'", "'\\''")
                              for a in docker_cmd))
                full = onstart + "\n" + exec_line
            elif docker_cmd:
                full = (
                    "exec python3.12 /opt/fidelity/suite/bin/container_entry.py "
                    + " ".join("'%s'" % a.replace("'", "'\\''")
                              for a in docker_cmd))
            else:
                full = onstart
            # Vast limits onstart to 4048 chars.  gzip+base64 the prep
            # script and decode it at runtime when the combined text is
            # too long (Vast's own documented workaround).
            if len(full) > 4048 and onstart and docker_cmd:
                import gzip as _gz
                compressed = _gz.compress(onstart.encode("utf-8"))
                encoded = base64.b64encode(compressed).decode("ascii")
                exec_line = (
                    "exec python3.12 /opt/fidelity/suite/bin/container_entry.py "
                    + " ".join("'%s'" % a.replace("'", "'\\''")
                              for a in docker_cmd))
                full = (
                    "echo '%s' | base64 -d | gunzip > /workspace/prep.sh "
                    "&& bash /workspace/prep.sh\n" % encoded) + exec_line
            # Vast's REST API takes `env` as a string in Docker flag
            # format (e.g. "-e KEY=VAL -p 8000:8000"), not a plain dict.
            # The CLI's parse_env converts this to a dict internally, but
            # the PUT body expects the string form.  Secrets stay under env,
            # never in onstart or args.
            env_dict = kw.get("env") or {}
            env_str = " ".join("-e %s=%s" % (k, v) for k, v in env_dict.items())
            body = {"client_id": "me",
                    "image": kw.get("image") or DEFAULT_IMAGE,
                    "disk": disk,
                    "label": kw.get("name") or "fidcloud",
                    "runtype": "ssh",
                    "onstart": full,
                    "env": env_str}
            got = self._req("PUT", "/asks/%s/" % ask, body, timeout=180)
            if not got.get("success"):
                raise VastError("Vast refused the rental: %s"
                                % redact(json.dumps(got)[:300]))
            return {"machine_id": got.get("new_contract"), "ask_id": ask}
        pub = ""
        kp = self.ssh_key + ".pub"
        if os.path.isfile(kp):
            pub = open(kp, encoding="utf-8").read().strip()
        body = {"client_id": "me", "image": kw.get("image") or DEFAULT_IMAGE,
                "disk": disk, "label": kw.get("name") or "fidcloud",
                "runtype": "ssh", "onstart": "", "env": {}}
        if pub:
            body["extra_env"] = {"PUBLIC_KEY": pub}
        got = self._req("PUT", "/asks/%s/" % ask, body, timeout=180)
        if not got.get("success"):
            raise VastError("Vast refused the rental: %s"
                            % redact(json.dumps(got)[:300]))
        cid = got.get("new_contract")
        if pub:
            # Vast attaches keys per-instance, not per-account.
            try:
                self._req("POST", "/instances/%s/ssh/" % cid, {"ssh_key": pub})
            except VastError:
                pass
        return {"machine_id": cid, "ask_id": ask}

    def destroy(self, machine_id: Any) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True}
        self._req("DELETE", "/instances/%s/" % machine_id, {})
        return {"terminated": str(machine_id)}

    def pause(self, machine_id: Any) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True}
        return self._req("PUT", "/instances/%s/" % machine_id, {"state": "stopped"})

    def resume(self, machine_id: Any, *, spot: bool = False) -> Dict[str, Any]:
        if self.dry:
            return {"dry_run": True}
        return self._req("PUT", "/instances/%s/" % machine_id, {"state": "running"})

    # -- ssh ---------------------------------------------------------------
    def _endpoint(self, machine_id: Any, *, wait: float = 900) -> tuple:
        key = str(machine_id)
        if key in self._ep:
            return self._ep[key]
        deadline = time.time() + wait
        while time.time() < deadline:
            inst = self.get(machine_id)
            if inst is not None:
                host = inst.raw.get("ssh_host")
                port = inst.raw.get("ssh_port")
                if host and port and str(inst.status).lower().startswith("run"):
                    # `running` is the CONTRACT's state, not sshd's.
                    self._await_ssh(host, int(port),
                                    wait=max(60.0, deadline - time.time()))
                    self._ep[key] = (host, int(port))
                    return self._ep[key]
            time.sleep(10)
        raise VastError("instance %s never reported a running SSH endpoint "
                        "within %ds" % (machine_id, int(wait)))

    # -- storage -----------------------------------------------------------
    def fs_create(self, *, storage: int, region: str = "",
                  name: Optional[str] = None) -> Any:
        return {"fs_id": None, "storage_gb": int(storage),
                "note": "vast disk is chosen at rent time and dies with the "
                        "instance; requested via create(disk=)"}

    def fs_delete(self, fs_id: Any) -> Any:
        return {"deleted": False, "note": "no separable filesystem on vast"}
