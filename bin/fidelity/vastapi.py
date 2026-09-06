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
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from .common import register_secret, safe_urlopen
from .jlapi import GpuOffer, Instance, JLError, redact
from .sshbase import SSHTransport

API = "https://console.vast.ai/api/v0"
DEFAULT_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
MIN_CREATE_SETUP_SECONDS = 300
MAX_LOG_RESPONSE_BYTES = 2 * 1024 * 1024
# Vast bills per UTC DAY. `GET /charges/` is filtered by a day index
# (epoch // 86400) and every row's `start`/`end` is that day's midnight;
# nothing finer exists in the official response (verified live 2026-09-06).
VAST_DAY_SECONDS = 86400
# Hosts a capture must be able to reach BEFORE a byte is uploaded or a dollar
# spent. Machine 68004 (Nevada, US) proxies huggingface.co with a mismatched
# certificate and answers UNEXPECTED_EOF_WHILE_READING; it failed the
# 2026-09-05 Fruit rehearsal at the setup stage (docs/CLOUD-RECIPES.md around
# line 297). On 2026-09-06 that machine was STILL the cheapest rentable Tesla
# T4 on the marketplace, so "the cheapest offer that fits" lands straight back
# on it unless something refuses -- hence both the id refusal and the live
# reachability probe in `attest_live_resource`.
HUB_PROBE_HOSTS = ("huggingface.co", "cdn-lfs.hf.co")
KNOWN_BAD_MACHINE_IDS = {
    "68004": "machine 68004 (Nevada, US, driver 580.126.09) proxies "
             "huggingface.co with a certificate hostname mismatch and "
             "UNEXPECTED_EOF; it failed the 2026-09-05 capture at setup",
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
            "dry_run": self.dry_run,
        }

_LIVE_ATTEST_SCRIPT = r'''
import json
import os
import socket
import ssl
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

# A Vast host is one person's machine on one person's uplink. Machine 68004
# answered huggingface.co through an SSL proxy with a mismatched certificate
# and UNEXPECTED_EOF, and the capture died at the setup stage after the box
# was already billing. This proves the Hub is reachable WITH a verified
# certificate from THIS host before anything is uploaded or spent.
def hub_probe(host, request_path):
    row = {"host": host, "tls_ok": False, "cert_subject_cn": None,
           "cert_issuer_cn": None, "cert_not_after": None,
           "http_status": None, "error": None}
    try:
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        with socket.create_connection((host, 443), timeout=30) as raw:
            with context.wrap_socket(raw, server_hostname=host) as tls:
                cert = tls.getpeercert() or {}
                subject = {}
                for entry in cert.get("subject", ()):
                    for key, value in entry:
                        subject[key] = value
                issuer = {}
                for entry in cert.get("issuer", ()):
                    for key, value in entry:
                        issuer[key] = value
                row["cert_subject_cn"] = subject.get("commonName")
                row["cert_issuer_cn"] = issuer.get("commonName")
                row["cert_not_after"] = cert.get("notAfter")
                row["tls_ok"] = True
                if request_path:
                    tls.sendall((
                        "HEAD %s HTTP/1.1\r\nHost: %s\r\n"
                        "User-Agent: quant-fidelity-suite/0.1\r\n"
                        "Accept: */*\r\nConnection: close\r\n\r\n"
                        % (request_path, host)).encode("ascii"))
                    head = b""
                    while b"\r\n" not in head and len(head) < 4096:
                        chunk = tls.recv(1024)
                        if not chunk:
                            break
                        head += chunk
                    fields = head.split(b"\r\n", 1)[0].split()
                    if len(fields) >= 2 and fields[1].isdigit():
                        row["http_status"] = int(fields[1])
    except Exception as exc:
        row["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
    return row

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
smi = subprocess.run(
    ["nvidia-smi",
     "--query-gpu=index,name,memory.total,driver_version",
     "--format=csv,noheader,nounits"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
gpus = []
if smi.returncode == 0:
    for line in smi.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 4:
            gpus.append({
                "index": int(fields[0]), "name": fields[1],
                "vram_bytes": int(fields[2]) * 1024 * 1024,
                "driver_version": fields[3],
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
    "filesystems": {"root": filesystem("/"),
                    "workspace": filesystem("/workspace")},
    "hub_reachability": [hub_probe("huggingface.co", "/api/models/gpt2"),
                         hub_probe("cdn-lfs.hf.co", "")],
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

    # -- transport ---------------------------------------------------------
    def _load_key(self) -> str:
        if self._key:
            return self._key
        path = self._key_file or os.environ.get("VAST_KEY_FILE") or ""
        # `~/.config/vastai/vast_api_key` is how the operator spells it and how
        # docs/CLOUD-RECIPES.md spells it; without expanduser the path silently
        # missed and the loader fell through to the environment.
        path = os.path.expanduser(str(path)) if path else ""
        if path:
            self._key = self._read_key_file(path)
        else:
            self._key = os.environ.get("VAST_API_KEY", "").strip()
            register_secret(self._key)
        if not self._key:
            raise VastError("no Vast credential: set VAST_KEY_FILE to a 0600 "
                            "file, or VAST_API_KEY")
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
            if exc.code == 429 and _tries > 1:
                wait = 2.0
                try:
                    wait = max(1.0, float(json.loads(payload).get("retry_after") or 1)) + 1.0
                except Exception:                         # noqa: BLE001
                    pass
                time.sleep(wait)
                return self._req(method, path, body, timeout=timeout,
                                 _tries=_tries - 1)
            raise VastError("Vast HTTP %d on %s: %s"
                            % (exc.code, path, redact(payload)))
        except VastError:
            raise
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
    def _to_instance(d: Dict[str, Any]) -> Instance:
        inst = Instance.from_json({
            "machine_id": 0,
            "status": d.get("actual_status") or d.get("cur_state") or "",
            "gpu_type": d.get("gpu_name"), "num_gpus": d.get("num_gpus") or 1,
            "region": d.get("geolocation"), "is_spot": False,
            "cost": float(d.get("dph_total") or 0)
            * float(d.get("duration") or 0) / 3600.0,
            "runtime": d.get("duration"), "fs_id": None,
            "storage_gb": d.get("disk_space"), "name": d.get("label"),
        })
        inst.machine_id = d.get("id")
        inst.raw["ssh_host"] = d.get("ssh_host")
        inst.raw["ssh_port"] = d.get("ssh_port")
        # The CONTRACT rate, not the ask's. On a marketplace those are two
        # different objects: the ask you searched can be gone by the time the
        # rental lands, and an ask id is not a durable name for one machine --
        # one that advertised a B200 handed back an H100. Anything that prices
        # a run must read what is billing, not what was listed.
        inst.raw["dph_total"] = d.get("dph_total")
        inst.raw["gpu_name"] = d.get("gpu_name")
        return inst

    def list_instances(self) -> List[Instance]:
        got = self._req("GET", "/instances/") or {}
        return [self._to_instance(d) for d in got.get("instances", [])]

    def get(self, machine_id: Any) -> Optional[Instance]:
        for i in self.list_instances():
            if str(i.machine_id) == str(machine_id):
                return i
        return None

    def create(self, **kw) -> Dict[str, Any]:
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
