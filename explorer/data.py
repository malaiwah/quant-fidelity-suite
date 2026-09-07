"""Read-only, stdlib-only views of one pinned public registry snapshot.

Returned values are ordinary, detached Python data, not HTML or Markdown.
The UI must render registry and user strings as text or structured JSON.
"""
from __future__ import annotations

import importlib.util
import ipaddress
import json
import math
from pathlib import Path
import re
import sys
from types import MappingProxyType
from collections.abc import Mapping
import urllib.error
import urllib.parse
import urllib.request

_ROOT = Path(__file__).resolve().parents[1]
_HF = "https://huggingface.co"
_TIMEOUT = 12.0
_MAX_BYTES = 16 * 1024 * 1024
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,199}\Z")
_PATH_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,255}\Z")


def _private_module(name, path):
    """Isolate transport overrides from CLI clients imported by other tabs."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves its defining module here.
    spec.loader.exec_module(module)
    return module


if str(_ROOT / "bin") not in sys.path:
    sys.path.insert(0, str(_ROOT / "bin"))
_RC = _private_module("fidelity._explorer_registry_client", _ROOT / "bin/fidelity/registry_client.py")
_META = _private_module("fidelity._explorer_hfmeta", _ROOT / "bin/fidelity/hfmeta.py")
_PRED = _private_module("_explorer_registry_predicate", _ROOT / "registry/tools/registry_predicate.py")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("Redirects are disabled for public metadata requests")


def _public_get(url, timeout=_TIMEOUT):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "huggingface.co"
            or parsed.username or parsed.password or parsed.fragment):
        raise ValueError("Only the public Hugging Face metadata host is allowed.")
    # No implicit tokens, cookies, proxies, or redirect destinations. A fresh
    # opener per request has no mutable state shared by concurrent visitors.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    request = urllib.request.Request(url, headers={"User-Agent": "qfs-explorer/1.0"})
    with opener.open(request, timeout=min(timeout, _TIMEOUT)) as response:
        data = response.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise ValueError("Public metadata exceeded the explorer's response limit.")
    return data


def _metadata_get(url, **kwargs):
    return json.loads(_public_get(url, **kwargs))


_RC.HF_ENDPOINT = _META.HF_ENDPOINT = _HF
_RC._http_get = _public_get
_META._get = _metadata_get


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _safe_link(value):
    if not isinstance(value, str) or any(ord(c) < 33 for c in value) or "\\" in value:
        return None
    try:
        url = urllib.parse.urlsplit(value)
        host = url.hostname
        if url.scheme not in ("https", "http") or not host or url.username or url.password:
            return None
        if host == "localhost" or host.endswith((".localhost", ".local")) or "." not in host:
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            pass
        if url.port not in (None, 80, 443):
            return None
    except ValueError:
        return None
    return value


def _parse_target(target):
    message = ("Enter a public Hugging Face model ID (owner/model), optionally @revision "
               "and /path, or an https://huggingface.co/owner/model/tree/revision/path link. "
               "Do not include credentials, query strings, fragments, or other hosts.")
    if not isinstance(target, str) or not target.strip() or len(target) > 1024:
        raise ValueError(message)
    text = target.strip()
    if any(ord(c) < 33 or ord(c) > 126 for c in text) or any(c in text for c in "\\%?#"):
        raise ValueError(message)
    if "://" in text:
        try:
            url = urllib.parse.urlsplit(text)
            if (url.scheme not in ("http", "https")
                    or url.netloc.lower() not in ("huggingface.co", "hf.co")):
                raise ValueError(message)
        except ValueError:
            raise ValueError(message) from None
    elif ":" in text or text.startswith("/"):
        raise ValueError(message)
    # Keep the CLI's target semantics, but refuse its permissive empty-segment
    # normalization and all URL/path metacharacters before any network access.
    raw_path = urllib.parse.urlsplit(text).path if "://" in text else text
    if "//" in raw_path or any(part in (".", "..") for part in raw_path.split("/")):
        raise ValueError(message)
    try:
        parsed = _RC.parse_hf_target(text)
    except (ValueError, TypeError):
        raise ValueError(message) from None
    repo = parsed["repo"]
    if len(repo.split("/")) != 2 or any(not _COMPONENT.fullmatch(p) or ".." in p for p in repo.split("/")):
        raise ValueError(message)
    revision = parsed["revision"]
    if revision is not None and (not _REVISION.fullmatch(revision) or ".." in revision
                                 or "//" in revision or revision.endswith("/")):
        raise ValueError(message)
    hint = parsed["path"]
    if hint and any(not _PATH_PART.fullmatch(p) or p in (".", "..") for p in hint.split("/")):
        raise ValueError(message)
    # Reject @ with no revision, which the CLI parser otherwise accepts.
    if "@" in text and (text.count("@") != 1 or text.split("@", 1)[1].split("/", 1)[0] == ""):
        raise ValueError(message)
    return parsed


class ExplorerRegistry:
    """Load once at startup; methods never reload or mutate this snapshot.

    ``hf`` (default) tries the pinned public mirror first, then explicitly
    discloses a bundled fallback. ``local`` explicitly selects bundled data.
    Metadata lookups resolve current identity, not fresh measurement results.
    """

    def __init__(self, source='hf'):
        if source not in ("hf", "local"):
            raise ValueError("Choose registry source 'hf' or 'local'.")
        try:
            if source == "hf":
                try:
                    snapshot = _RC.load("hf")
                except Exception:
                    snapshot = _RC.load("local")
                    snapshot.notes.append("Public HF registry could not be loaded. Using the bundled registry; it may be ahead of or behind published truth.")
                    snapshot.origin = "bundled registry fallback (not a fresh public snapshot)"
            else:
                snapshot = _RC.load("local")
                snapshot.origin = "bundled registry (explicit local selection; not a fresh public snapshot)"
            snapshot.collections = _freeze(snapshot.collections)
            snapshot.index = _freeze(snapshot.index)
            snapshot.notes = tuple(snapshot.notes)
            self._snapshot = snapshot
            self._prepare()
        except Exception:
            raise ValueError("Registry data could not be loaded safely. Retry later or ask the Space owner to restore the bundled registry.") from None

    def _prepare(self):
        snapshot = self._snapshot
        collections = snapshot.collections
        self._published = {m["id"]: m for m in _RC.rows_for(snapshot, list(collections["artifacts"]))}
        # Predicate evaluates ORIGINAL whole-key membership, including pending
        # rows as the registry does. Lane/model/search filters never certify a
        # previously false or unknown group by discarding inconvenient members.
        full = {}
        self._row_groups = {}
        lanes = {}
        for mid, measurement in collections["measurements"].items():
            key = snapshot.recomputed_key(measurement)
            full.setdefault(key, []).append(mid)
            if mid in self._published:
                lane = snapshot.lane_of(measurement) or ""
                gid = json.dumps([key, lane], separators=(",", ":"))
                self._row_groups[mid] = gid
                lanes.setdefault(gid, (key, lane, []))[2].append(mid)
        predicates = {key: _PRED.group_predicate(collections, ids) for key, ids in full.items()}
        self._groups = {}
        for gid, (key, lane, mids) in lanes.items():
            predicate = predicates[key]
            reasons = list(predicate["reasons"])
            rank_allowed = predicate["comparable"] == "true" and not snapshot.notes
            if snapshot.notes:
                reasons.append("Ranking disabled because this snapshot has loading/integrity notes; see snapshot notes.")
            key_mismatch = any((collections["measurements"][mid].get("comparability") or {}).get("key") != key for mid in full[key])
            if key_mismatch:
                rank_allowed = False
                reasons.append("Ranking disabled: a stored comparability key differs from authoritative row fields.")
            controls = {mid for mid in mids if _PRED._scope_class_of(collections, self._published[mid]) == "unquantized"}
            metric = self._published[mids[0]].get("metric") or {}
            if metric.get("name") not in ("mean_tokenwise_kld", "mean_of_run_means_tokenwise_kld") or metric.get("higher_is_better") is not False:
                rank_allowed = False
                reasons.append("Ranking disabled: this view only orders lower-is-better mean tokenwise KL.")
            ordered = sorted(mids)
            if rank_allowed:
                ordered = sorted(controls) + sorted((mid for mid in mids if mid not in controls), key=lambda mid: (
                    _number((self._published[mid].get("metric") or {}).get("value")) is None,
                    _number((self._published[mid].get("metric") or {}).get("value")) or 0,
                    mid))
            title = "%s | %s" % (snapshot.key_label(key) or key, lane or "no declared lane")
            caveats = set()
            for mid in mids:
                panel = self._published[mid].get("panel_ref")
                for caveat in (_RC._subset_caveat(snapshot, panel), _RC._undisclosed_caveat(snapshot, panel)):
                    if caveat:
                        caveats.add(caveat)
            context = {
                "key": key, "lane": lane or None,
                "key_inputs": snapshot.lib.key_inputs_from_measurement(self._published[mids[0]]),
                "original_predicate": predicate,
                "predicate_scope": "all original members of the full comparability key, before lane or model filtering",
                "full_key_member_count": len(full[key]), "displayed_member_count": len(mids),
                "ranking_allowed": rank_allowed,
                "ordering": "controls first (not ranked), then ascending KL" if rank_allowed else "measurement identity; not a ranking",
                "control_measurement_ids": sorted(controls),
                "caveats": sorted(caveats),
                "rule": "Different keys or lanes are never ranked together. Equal keys alone do not certify like-for-like comparison. Comparability is not a claim of general model quality.",
                "snapshot_notes": list(snapshot.notes),
            }
            self._groups[gid] = _freeze({"title": title, "status": predicate["comparable"],
                                       "reasons": reasons, "rows": [self._row(mid) for mid in ordered],
                                       "context": context})
        self._published = MappingProxyType(self._published)
        self._row_groups = MappingProxyType(self._row_groups)
        self._groups = MappingProxyType(self._groups)

    def _row(self, mid):
        measurement = self._published[mid]
        artifact = self._snapshot.collections["artifacts"].get(measurement.get("artifact_ref")) or {}
        metric = measurement.get("metric") or {}
        return {"id": mid, "artifact": artifact.get("name") or measurement.get("artifact_ref"),
                "revision": (artifact.get("huggingface") or {}).get("revision"),
                "kl": _number(metric.get("value")) if metric.get("name") in ("mean_tokenwise_kld", "mean_of_run_means_tokenwise_kld") else None,
                "top1": _number((measurement.get("auxiliary_metrics") or {}).get("top1_agreement")),
                "classification": (measurement.get("comparability") or {}).get("class") or "unknown"}

    def overview(self) -> dict:
        snapshot = self._snapshot
        return {"snapshot": snapshot.snapshot_id, "origin": snapshot.origin, "notes": list(snapshot.notes),
                "measurement_count": len(self._published), "model_count": len(snapshot.collections["models"]),
                "group_count": len(self._groups)}

    def models(self) -> list[tuple[str, str]]:
        return sorted(((model.get("name") or mid, mid) for mid, model in self._snapshot.collections["models"].items()), key=lambda item: (item[0].casefold(), item[1]))

    def groups(self, model_id='') -> list[tuple[str, str]]:
        if not isinstance(model_id, str) or (model_id and model_id not in self._snapshot.collections["models"]):
            raise ValueError("Choose a model from the registry model list.")
        matching = {gid for mid, gid in self._row_groups.items() if not model_id or self._published[mid].get("model_ref") == model_id}
        return sorted(((self._groups[gid]["title"], gid) for gid in matching), key=lambda item: (item[0], item[1]))

    def group(self, group_id) -> dict:
        if not isinstance(group_id, str) or group_id not in self._groups:
            raise ValueError("Choose a comparability group from the current snapshot.")
        return _plain(self._groups[group_id])

    def detail(self, measurement_id) -> dict:
        if not isinstance(measurement_id, str) or measurement_id not in self._published:
            raise ValueError("Choose a published measurement from the current snapshot.")
        collections = self._snapshot.collections
        measurement = self._published[measurement_id]
        joined = {"measurement": measurement}
        for name, collection in (("artifact", "artifacts"), ("model", "models"), ("panel", "panels"),
                                 ("reference", "references"), ("pipeline", "pipelines")):
            joined[name] = collections[collection].get(measurement.get(name + "_ref"))
        reference = joined["reference"] or {}
        joined["reference_artifact"] = collections["artifacts"].get(reference.get("artifact_ref"))
        joined["reference_pipeline"] = collections["pipelines"].get(reference.get("pipeline_ref"))
        links = set()

        def collect(value):
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if key in ("url", "uri"):
                        link = _safe_link(item)
                        if link:
                            links.add(link)
                    else:
                        collect(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)

        collect(joined)
        joined["source_links"] = sorted(links)
        joined["group_id"] = self._row_groups[measurement_id]
        joined["snapshot"] = self.overview()
        return _plain(joined)

    def lookup(self, target) -> dict:
        parsed = _parse_target(target)
        warnings = list(self._snapshot.notes)
        resolved = None
        try:
            # Unlike resolve_revision, resolve_commit checks even a supplied
            # SHA against HF, so a typo never masquerades as a verified pin.
            resolved = _META.resolve_commit(parsed["repo"], parsed["revision"] or "main")
        except Exception:
            warnings.append("HF metadata is inaccessible or this public model/revision could not be resolved. Registry lookup still works, but target identity is unverified; pinned records are PINNED-UNVERIFIED, never EXACT.")
        match = _RC.match_artifacts(self._snapshot, parsed["repo"], resolved, parsed["path"])
        candidates = match["candidates"]
        selected = {art["id"]: (tier, note) for art, tier, note in candidates}
        rows = []
        gids = set()
        for mid in sorted(self._published):
            measurement = self._published[mid]
            if measurement.get("artifact_ref") in selected:
                row = self._row(mid)
                tier, _ = selected[measurement["artifact_ref"]]
                row["classification"] = "%s / %s" % (tier, row["classification"])
                rows.append(row)
                gids.add(self._row_groups[mid])
        if match["ambiguous"]:
            warnings.append("This repository contains multiple artifact paths. Showing all matching paths, without choosing or merging their comparability groups: " + ", ".join(match["paths"]))
        if rows:
            message = "%d published measurement(s) matched. Identity order only, not a ranking. Open each group for its original comparability predicate; STALE rows describe a different revision, UNPINNED rows do not establish a commit match." % len(rows)
        elif candidates:
            message = "Artifact identity is registered, but no published measurements match this path/revision lookup. No measurement was executed."
        else:
            message = "No registered artifact matches this repository and path in this snapshot. This does not prove it has never been measured. No measurement was executed."
        return {"title": "Registry lookup: " + parsed["repo"], "message": message, "rows": rows,
                "group_ids": sorted(gids),
                "target": {**parsed, "resolved_revision": resolved, "metadata_verified": resolved is not None,
                           "source": _HF + "/" + parsed["repo"],
                           "artifact_matches": [{"id": art["id"], "classification": tier, "note": note} for art, tier, note in candidates]},
                "warnings": warnings}
