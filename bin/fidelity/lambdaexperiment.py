"""Private Lambda native experiment; never a production admission protocol.

The credential snapshot is the authority, not an invented account identity.
Lambda provides no billing deadline. An unavailable API or ambiguous launch can
leave liability indefinitely; neither a timeout nor terminate acknowledgement
is absence. All provider mutation is serialized across controller and guardian.
"""
from __future__ import annotations

import contextlib
import fcntl
import hmac
import json
import math
import os
import re
import stat
import tempfile
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .lambdaapi import (LambdaCloud, LambdaCreateRejectedError,
                        LambdaCreateResponseError, _read_key_file)

SCHEMA = "fidelity.lambda-native-experiment.v1"
HEARTBEAT_MAX_AGE = 15
ABSENCE_INTERVAL = 5
MIN_SETUP_SECONDS = 300
MIN_VRAM_GIB = 24
_NOW = time.time
_SLEEP = time.sleep
_PROVIDER_FACTORY = LambdaCloud
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_PHASES = {"PLANNED", "POST_INTENT", "ACTIVE", "AMBIGUOUS", "CLEANING", "CLOSED"}


class ExperimentError(RuntimeError):
    """A refusal with no provider response or secret embedded in the message."""


def _number(value):
    if isinstance(value, bool):
        raise ExperimentError("invalid numeric limit")
    try:
        out = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ExperimentError("invalid numeric limit") from None
    if not out.is_finite() or out <= 0:
        raise ExperimentError("numeric limit must be finite and positive")
    return out


def _integer(value):
    if type(value) is not int or value <= 0:
        raise ExperimentError("expected positive exact integer")
    return value


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ExperimentError("invalid exact provider identifier")
    return value


def _directory(run_dir):
    path = Path(run_dir).absolute()
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ExperimentError("experiment directory must be owned private 0700 directory")
    return path


def _read_json(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ExperimentError("experiment state must be owned regular 0600 file")
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ExperimentError("duplicate state key")
                result[key] = value
            return result
        return json.load(stream, object_pairs_hook=pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(ExperimentError("nonfinite state")))


def _atomic(path, value):
    """Caller holds the state lock, or the independent heartbeat lock."""
    fd, temp = tempfile.mkstemp(prefix=".state-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextlib.contextmanager
def _locked(run_dir, lock_name="state.lock"):
    path = _directory(run_dir)
    fd = os.open(str(path / lock_name), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ExperimentError("invalid state lock")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield path
    finally:
        os.close(fd)


def _save(path, plan):
    saved = dict(plan)
    saved.pop("guardian", None)
    _atomic(path / "plan.json", saved)


def _validate(plan):
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA or plan.get("phase") not in _PHASES:
        raise ExperimentError("invalid experiment schema or phase")
    if not re.fullmatch(r"qfs-native-[0-9a-f]{32}", str(plan.get("name", ""))):
        raise ExperimentError("invalid unique attempt name")
    runtime = _integer(plan["max_runtime_seconds"])
    cost = _number(plan["max_cost_usd"])
    quote = plan["quote"]
    if runtime > 14400 or runtime < MIN_SETUP_SECONDS or cost > 25:
        raise ExperimentError("experiment exceeds approved operational ceiling")
    if quote["architecture"] != "x86_64" or quote["gpu_count"] != 1 or _integer(quote["vram_gib"]) < MIN_VRAM_GIB or _integer(quote["storage_gib"]) < 200:
        raise ExperimentError("invalid native experiment resource shape")
    rate = _integer(quote["price_cents_per_hour"])
    if _number(quote["hourly_usd"]) != Decimal(rate) / 100 or Decimal(rate) * runtime / 360000 > cost:
        raise ExperimentError("invalid or over-budget exact price")
    start, deadline = plan["created_at_epoch"], plan["deadline_epoch"]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, deadline)) or deadline - start != runtime:
        raise ExperimentError("invalid absolute deadline")
    if plan["terminate_after"] != time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline)):
        raise ExperimentError("deadline representations disagree")
    if plan.get("accept_no_provider_deadline") is not True or plan.get("provider_enforced_deadline") is not False:
        raise ExperimentError("missing explicit provider deadline risk acceptance")
    if not isinstance(plan["ssh_key_names"], list) or len(plan["ssh_key_names"]) != 1 or not isinstance(plan["ssh_key_names"][0], str) or not plan["ssh_key_names"][0]:
        raise ExperimentError("invalid SSH name selection")
    for key in ("gpu_type", "region"):
        if not isinstance(plan[key], str) or not plan[key]:
            raise ExperimentError("invalid launch selection")
    baseline = plan["baseline"]
    if baseline.get("complete") is not True:
        raise ExperimentError("incomplete baseline")
    for key in ("instance_ids", "network_volume_ids"):
        identifiers = baseline[key]
        if not isinstance(identifiers, list) or len(set(identifiers)) != len(identifiers):
            raise ExperimentError("invalid baseline identifiers")
        for identifier in identifiers:
            _identifier(identifier)
    if plan.get("provider_id") is not None:
        _identifier(plan["provider_id"])
    cleanup = plan["cleanup"]
    if type(cleanup["requested"]) is not bool or type(cleanup["closed"]) is not bool or type(cleanup["absence_confirmations"]) is not int or cleanup["absence_confirmations"] < 0:
        raise ExperimentError("invalid cleanup state")
    if cleanup["closed"] != (plan["phase"] == "CLOSED"):
        raise ExperimentError("inconsistent closure state")
    if type(plan["post_attempted"]) is not bool or (not plan["post_attempted"] and plan["phase"] not in ("PLANNED", "CLOSED")):
        raise ExperimentError("inconsistent launch state")
    return plan


def load_plan(run_dir):
    path = _directory(run_dir)
    try:
        plan = _validate(_read_json(path / "plan.json"))
        try:
            heartbeat = _read_json(path / "heartbeat.json")
            epoch = heartbeat["heartbeat_epoch"]
            if heartbeat.get("name") != plan["name"] or type(epoch) not in (int, float) or not math.isfinite(epoch):
                raise ExperimentError("invalid guardian heartbeat")
            plan["guardian"] = {"heartbeat_epoch": epoch}
        except FileNotFoundError:
            plan["guardian"] = {"heartbeat_epoch": None}
        try:
            requested = _read_json(path / "cleanup-request.json")
            if requested.get("name") != plan["name"]:
                raise ExperimentError("cleanup request identity differs")
            _request(plan, requested.get("reason"))
        except FileNotFoundError:
            pass
        return plan
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentError("invalid experiment state") from exc


def update_control(run_dir, **fields):
    with _locked(run_dir) as path:
        control = _read_json(path / "control.json")
        if any(key in fields for key in ("key_file", "ssh_key", "ssh_key_names")):
            raise ExperimentError("credential authority and SSH identity are frozen")
        control.update(fields)
        _atomic(path / "control.json", control)


def provider_from_control(run_dir):
    path = _directory(run_dir)
    control = _read_json(path / "control.json")
    if control["key_file"] != str(path / "secret" / "api_key") or not os.path.isabs(control["ssh_key"]):
        raise ExperimentError("invalid private control authority")
    _directory(path / "secret")
    _read_key_file(control["key_file"])
    plan = load_plan(path)
    if control["ssh_key_names"] != plan["ssh_key_names"]:
        raise ExperimentError("SSH identity changed")
    return _PROVIDER_FACTORY(key_file=control["key_file"], ssh_key=control["ssh_key"], ssh_key_names=control["ssh_key_names"])


def _authority(provider, path):
    if getattr(provider, "dry", False):
        raise ExperimentError("experiment does not accept dry providers")
    control = _read_json(path / "control.json")
    if control["key_file"] != str(path / "secret" / "api_key"):
        raise ExperimentError("credential snapshot path changed")
    _directory(path / "secret")
    key = _read_key_file(control["key_file"])
    if not hmac.compare_digest(provider._load_key(), key):
        raise ExperimentError("provider credential differs from frozen authority")
    if str(Path(provider.ssh_key).expanduser().absolute()) != control["ssh_key"]:
        raise ExperimentError("provider SSH identity differs from frozen selection")


def _inventory(provider):
    inventory = provider.chargeable_inventory()
    if inventory.get("complete") is not True or inventory.get("unknown_families") != []:
        raise ExperimentError("chargeable inventory incomplete")
    families = inventory.get("families", {})
    result = {}
    for name in ("instances", "network_volumes"):
        family = families.get(name, {})
        rows = family.get("resources")
        if family.get("complete") is not True or not isinstance(rows, list):
            raise ExperimentError("chargeable inventory family incomplete")
        seen = set()
        for row in rows:
            identifier = _identifier(row.get("id"))
            if identifier in seen:
                raise ExperimentError("duplicate inventory identifier")
            seen.add(identifier)
            if name == "instances" and (not isinstance(row.get("status"), str) or not row["status"]):
                raise ExperimentError("instance status unknown")
        result[name] = rows
    return result


def _quote(provider, gpu_type, region):
    catalogue = provider._get_data("/instance-types")
    entry = catalogue.get(gpu_type) if isinstance(catalogue, dict) else None
    if not isinstance(entry, dict):
        raise ExperimentError("selected instance type unavailable")
    instance = entry.get("instance_type", {})
    specs = instance.get("specs", {})
    regions = entry.get("regions_with_capacity_available")
    if not isinstance(regions, list) or not any(isinstance(row, dict) and row.get("name") == region for row in regions):
        raise ExperimentError("selected exact region lacks capacity")
    if instance.get("name") != gpu_type or instance.get("architecture") != "x86_64" or specs.get("gpus") != 1:
        raise ExperimentError("catalogue must explicitly identify one GPU and x86_64")
    description = instance.get("gpu_description")
    matches = re.findall(r"\((\d+)\s*GB(?:\s[^)]*)?\)", description or "")
    if len(matches) != 1 or int(matches[0]) < MIN_VRAM_GIB or _integer(specs.get("storage_gib")) < 200:
        raise ExperimentError("catalogue must document at least 24 GiB VRAM and 200 GiB local disk")
    cents = _integer(instance.get("price_cents_per_hour"))
    return {"price_cents_per_hour": cents, "hourly_usd": str(Decimal(cents) / 100),
            "gpu_count": _integer(specs.get("gpus")), "architecture": instance["architecture"],
            "gpu_description": description, "vram_gib": int(matches[0]),
            "storage_gib": specs["storage_gib"], "vcpus": _integer(specs.get("vcpus")),
            "memory_gib": _integer(specs.get("memory_gib"))}


def _prepare_request(provider, plan):
    prepared = provider.prepare_safe_create(
        gpu_type=plan["gpu_type"], region=plan["region"], name=plan["name"],
        num_gpus=1, storage=200, spot=False, offer="on-demand",
        ssh_key_names=plan["ssh_key_names"], terminate_after=plan["terminate_after"])
    expected = {"name": plan["name"], "instance_type_name": plan["gpu_type"],
                "region_name": plan["region"], "gpu_count": 1,
                "storage_gib": plan["quote"]["storage_gib"],
                "price_cents_per_hour": plan["quote"]["price_cents_per_hour"],
                "terminate_after": plan["terminate_after"], "dry_run": False}
    if any(getattr(prepared, key) != value for key, value in expected.items()) or tuple(prepared.ssh_key_names) != tuple(plan["ssh_key_names"]):
        raise ExperimentError("prepared launch differs from frozen request or quote")
    # Do not serialize to_dict(): it includes host/userdata/key-derived hashes.
    return prepared, dict(expected, ssh_key_names=plan["ssh_key_names"], file_system_names=[])


def prepare(provider, run_dir, *, gpu_type, region, max_cost_usd,
            max_runtime_seconds, ssh_key_names, accept_risk, now=None):
    if accept_risk is not True:
        raise ExperimentError("explicit acceptance of no provider deadline required")
    runtime = _integer(max_runtime_seconds)
    cost = _number(max_cost_usd)
    if runtime < MIN_SETUP_SECONDS or runtime > 14400 or cost > 25:
        raise ExperimentError("runtime/cost outside approved 300..14400 seconds and USD25 ceiling")
    if not isinstance(ssh_key_names, (tuple, list)) or len(ssh_key_names) != 1:
        raise ExperimentError("exactly one registered SSH key name required")
    if getattr(provider, "dry", False):
        raise ExperimentError("read-only preparation needs a real non-dry provider")
    path = Path(run_dir).absolute()
    os.mkdir(str(path), 0o700)  # Exclusive: even a failed preparation is not reusable.
    with _locked(path):
        os.mkdir(str(path / "secret"), 0o700)
        key = provider._load_key()
        fd = os.open(str(path / "secret" / "api_key"), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(key + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _atomic(path / "control.json", {"key_file": str(path / "secret" / "api_key"),
                "ssh_key": str(Path(provider.ssh_key).expanduser().absolute()),
                "ssh_key_names": list(ssh_key_names)})
        quote = _quote(provider, gpu_type, region)
        if Decimal(quote["price_cents_per_hour"]) * runtime / 360000 > cost:
            raise ExperimentError("exact quoted runtime estimate exceeds operational cost ceiling")
        inventory = _inventory(provider)
        clock = provider.server_time_evidence()
        start = int(_NOW() if now is None else now)
        name = "qfs-native-" + uuid.uuid4().hex
        if any(row.get("name") == name for row in inventory["instances"]):
            raise ExperimentError("unique attempt name already exists in baseline")
        plan = {"schema": SCHEMA, "name": name, "gpu_type": gpu_type, "region": region,
                "quote": quote, "ssh_key_names": list(ssh_key_names),
                "created_at_epoch": start, "deadline_epoch": start + runtime,
                "terminate_after": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start + runtime)),
                "max_cost_usd": str(cost), "max_runtime_seconds": runtime,
                "estimated_runtime_cost_usd": str(Decimal(quote["price_cents_per_hour"]) * runtime / 360000),
                "accept_no_provider_deadline": True, "provider_enforced_deadline": False,
                "authority": "private credential snapshot; no asserted provider account identity",
                "phase": "PLANNED", "post_attempted": False, "provider_id": None,
                "baseline": {"complete": True, "instance_ids": sorted(row["id"] for row in inventory["instances"]),
                             "network_volume_ids": sorted(row["id"] for row in inventory["network_volumes"])},
                "clock": {key: clock[key] for key in ("server_epoch", "local_received_epoch", "local_minus_server_seconds")},
                "cleanup": {"requested": False, "reason": None, "closed": False,
                            "absence_confirmations": 0, "status": "not_requested"}}
        _validate(plan)
        _, plan["prepared_evidence"] = _prepare_request(provider, plan)
        _save(path, plan)
    return load_plan(path)


def _fresh_guard(plan):
    epoch = plan.get("guardian", {}).get("heartbeat_epoch")
    if epoch is None or not 0 <= _NOW() - epoch <= HEARTBEAT_MAX_AGE:
        raise ExperimentError("independent guardian heartbeat missing or stale")


def _request(plan, reason):
    plan["cleanup"]["requested"] = True
    # Reasons are protocol labels, never arbitrary exception strings or paths.
    plan["cleanup"]["reason"] = reason if reason in {
        "deadline", "create_failed", "create_rejected", "preflight_failed", "requested",
        "workload_finished", "workload_failed", "interrupted"} else "requested"


def create(provider, run_dir):
    with _locked(run_dir) as path:
        plan = load_plan(path)
        _authority(provider, path)
        if plan["phase"] != "PLANNED" or plan["post_attempted"] or plan["cleanup"]["requested"]:
            raise ExperimentError("attempt is already spent or cleanup requested; create cannot retry")
        try:
            _fresh_guard(plan)
            if plan["deadline_epoch"] - _NOW() < MIN_SETUP_SECONDS:
                raise ExperimentError("absolute deadline leaves insufficient setup time")
            if _quote(provider, plan["gpu_type"], plan["region"]) != plan["quote"]:
                raise ExperimentError("catalogue drift from frozen quote")
            inventory = _inventory(provider)
            if any(row.get("name") == plan["name"] for row in inventory["instances"]):
                raise ExperimentError("attempt name appeared before launch")
            # Preserve every previously seen foreign ID, including resources
            # appearing between read-only planning and the sole POST.
            for family, field in (("instances", "instance_ids"),
                                  ("network_volumes", "network_volume_ids")):
                plan["baseline"][field] = sorted(set(plan["baseline"][field]).union(
                    row["id"] for row in inventory[family]))
            prepared, evidence = _prepare_request(provider, plan)
            provider.server_time_evidence()
            latest = load_plan(path)
            _fresh_guard(latest)
            if latest["cleanup"]["requested"]:
                raise ExperimentError("cleanup requested during launch preflight")
            if plan["deadline_epoch"] - _NOW() < MIN_SETUP_SECONDS:
                raise ExperimentError("absolute deadline elapsed during preflight")
        except BaseException:
            _request(plan, "preflight_failed")
            plan["phase"] = "CLOSED"
            plan["cleanup"].update(closed=True, status="no_post")
            _save(path, plan)
            raise
        plan.update(phase="POST_INTENT", post_attempted=True, prepared_evidence=evidence,
                    post_intent_epoch=_NOW())
        _save(path, plan)  # Durable before the sole submit, including crash ambiguity.
        try:
            response = provider.submit_prepared_create(prepared)
            identifier = _identifier(response.get("provider_id", response.get("machine_id")))
            plan["provider_id"] = identifier
            _save(path, plan)  # Exact returned ID is durable before any further check.
            if identifier in plan["baseline"]["instance_ids"]:
                raise ExperimentError("returned resource existed before this launch")
            plan["phase"] = "ACTIVE"
            _save(path, plan)
        except LambdaCreateRejectedError:
            _request(plan, "create_rejected")
            plan["phase"] = "CLOSED"
            plan["cleanup"].update(closed=True, status="provider_rejected")
            _save(path, plan)
            raise ExperimentError("provider definitively refused the sole launch") from None
        except BaseException as exc:
            if isinstance(exc, LambdaCreateResponseError):
                try:
                    plan["provider_id"] = _identifier(exc.provider_id)
                except ExperimentError:
                    pass
            _request(plan, "create_failed")
            plan["phase"] = "AMBIGUOUS"
            plan["cleanup"]["status"] = "launch_outcome_unresolved"
            _save(path, plan)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise ExperimentError("launch outcome unresolved; guardian retains cleanup liability; never retry") from None
    return load_plan(path)


def request_cleanup(run_dir, reason):
    # A signal handler may run while create holds state.lock. The independent
    # request lock makes its durable stop request safe without re-entering it.
    with _locked(run_dir, "cleanup-request.lock") as path:
        plan = load_plan(path)
        _request(plan, reason)
        _atomic(path / "cleanup-request.json",
                {"name": plan["name"], "reason": plan["cleanup"]["reason"]})


def _matches(row, plan):
    return all(row.get(key) == plan[field] for key, field in (
        ("name", "name"), ("instance_type_name", "gpu_type"), ("region_name", "region")))


def _unresolved(path, plan, status):
    plan["cleanup"].update(status=status, absence_confirmations=0)
    plan["cleanup"].pop("last_absence_epoch", None)
    _save(path, plan)
    return False


def cleanup_once(provider, run_dir):
    with _locked(run_dir) as path:
        plan = load_plan(path)
        if plan["cleanup"]["closed"]:
            return True
        if _NOW() >= plan["deadline_epoch"]:
            _request(plan, "deadline")
        if not plan["cleanup"]["requested"]:
            return False
        if not plan["post_attempted"]:
            plan["phase"] = "CLOSED"
            plan["cleanup"].update(closed=True, status="no_post")
            _save(path, plan)
            return True
        plan["phase"] = "CLEANING"
        _save(path, plan)
        try:
            _authority(provider, path)
            inventory = _inventory(provider)
            rows = inventory["instances"]
            baseline = set(plan["baseline"]["instance_ids"])
            candidates = [row for row in rows if row.get("name") == plan["name"]]
            identifier = plan["provider_id"]
            if any(row["id"] in baseline or not _matches(row, plan) for row in candidates) or len(candidates) > 1:
                return _unresolved(path, plan, "conflicting_name_candidates")
            if identifier is None:
                if len(candidates) != 1:
                    # No documented queue horizon makes empty inventory definitive.
                    return _unresolved(path, plan, "lost_response_no_identified_resource")
                identifier = candidates[0]["id"]
                plan["provider_id"] = identifier
                plan["reconciled_by_unique_name"] = True
                _save(path, plan)
            if identifier in baseline or (candidates and candidates[0]["id"] != identifier):
                return _unresolved(path, plan, "ownership_conflict")
            listed = next((row for row in rows if row["id"] == identifier), None)
            detail = provider.get_lifecycle_resource(identifier)
            for row in (listed, detail):
                if row is not None and (row.get("id") != identifier or not _matches(row, plan)):
                    return _unresolved(path, plan, "exact_id_binding_conflict")
            gone = lambda row: row is None or row.get("status") == "terminated"
            if gone(listed) and gone(detail):
                cleanup = plan["cleanup"]
                last = cleanup.get("last_absence_epoch")
                current = _NOW()
                if last is None or current - last >= ABSENCE_INTERVAL:
                    cleanup["absence_confirmations"] += 1
                    cleanup["last_absence_epoch"] = current
                cleanup["status"] = "awaiting_repeated_complete_absence"
                if cleanup["absence_confirmations"] >= 2:
                    cleanup.update(closed=True, status="repeated_complete_absence")
                    plan["phase"] = "CLOSED"
                    plan["closed_at_epoch"] = current
                _save(path, plan)
                return cleanup["closed"]
            plan["cleanup"].update(absence_confirmations=0, status="owned_resource_live")
            plan["cleanup"].pop("last_absence_epoch", None)
            # Both complete inventory AND exact-id endpoint must bind ownership.
            if listed is None or detail is None or gone(listed) != gone(detail):
                return _unresolved(path, plan, "inventory_detail_disagree")
            if detail.get("status") == "terminating":
                plan["cleanup"]["status"] = "provider_terminating"
                _save(path, plan)
                return False
            last_terminate = plan["cleanup"].get("terminate_intent_epoch")
            if last_terminate is not None and _NOW() - last_terminate < 30:
                _save(path, plan)
                return False
            plan["cleanup"].update(terminate_intent_epoch=_NOW(), status="terminate_intent")
            _save(path, plan)
            provider.destroy(identifier)
            plan["cleanup"]["status"] = "terminate_acknowledged_not_absence"
            _save(path, plan)
            return False
        except Exception:
            return _unresolved(path, plan, "provider_or_authority_unavailable")


def _remove_closed_credential(path):
    """Remove only this run's snapshot through its owned, non-symlink directory."""
    try:
        fd = os.open(str(path / "secret"), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ExperimentError("invalid closed-run secret directory")
        try:
            os.unlink("api_key", dir_fd=fd)
        except FileNotFoundError:
            pass
        os.fsync(fd)
    finally:
        os.close(fd)


def guard(run_dir, *, poll_seconds=5):
    if not 0 < poll_seconds <= HEARTBEAT_MAX_AGE / 2:
        raise ExperimentError("guardian poll interval must be positive and at most 7.5 seconds")
    path = _directory(run_dir)
    transport_ready = False
    while True:
        try:
            # Re-read the snapshot every iteration: revoked/invalid local authority
            # never produces a fresh liveness claim. No parent heartbeat required.
            plan = load_plan(path)
            if plan["cleanup"]["closed"]:
                _remove_closed_credential(path)
                return 0
            provider = provider_from_control(path)
            if not transport_ready:
                # Prove the frozen guardian can actually use authenticated TLS
                # before it gives create() a fresh readiness heartbeat.
                _inventory(provider)
                provider.server_time_evidence()
                transport_ready = True
            with _locked(path, "heartbeat.lock"):
                _atomic(path / "heartbeat.json", {"name": plan["name"], "heartbeat_epoch": _NOW()})
            if plan["cleanup"]["requested"] or _NOW() >= plan["deadline_epoch"]:
                if cleanup_once(provider, path):
                    _remove_closed_credential(path)
                    return 0
                print("Lambda experiment liability unresolved; independent guardian continues", flush=True)
        except Exception:
            print("Lambda experiment control unavailable; liability unresolved; guardian continues", flush=True)
        _SLEEP(poll_seconds)


def public_report(run_dir):
    plan = load_plan(run_dir)
    report = {key: plan[key] for key in (
        "schema", "name", "gpu_type", "region", "created_at_epoch",
        "deadline_epoch", "terminate_after", "max_cost_usd", "max_runtime_seconds",
        "estimated_runtime_cost_usd", "provider_id", "phase", "post_attempted",
        "accept_no_provider_deadline", "provider_enforced_deadline")}
    report["quote"] = {key: plan["quote"][key] for key in (
        "price_cents_per_hour", "hourly_usd", "gpu_count", "architecture",
        "gpu_description", "vram_gib", "storage_gib", "vcpus", "memory_gib")}
    report["cleanup"] = {key: plan["cleanup"].get(key) for key in (
        "requested", "reason", "closed", "status", "absence_confirmations")}
    report["guardian"] = plan["guardian"]
    report["cost_basis"] = "catalogue runtime estimate only; not an invoice or settled bill"
    report["risk"] = "No provider hard cutoff; failed cleanup or API access may exceed USD25/4h operational ceilings"
    report["production_admission_qualified"] = False
    return report
