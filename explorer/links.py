"""Small, validated evidence URLs; no redirectors or user-selected HTTP hosts."""
from __future__ import annotations

from functools import lru_cache
import json
import re
from urllib.parse import urlencode, urlsplit

from .data import _public_get

REGISTRY = "malaiwah/quant-fidelity-registry"
SHA = re.compile(r"[0-9a-f]{40}\Z")
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
MEASUREMENT = re.compile(r"measurement--[A-Za-z0-9_.-]{1,240}\Z")
MODEL = re.compile(r"model--[A-Za-z0-9_.-]{1,240}\Z")
QUERY_FIELDS = {"measurement", "model", "group", "target", "registry_revision", "tab"}


def explorer_base(host=None):
    if not host:
        return "https://malaiwah-qfs-explorer.hf.space"
    value = host if host.startswith("https://") else "https://" + host
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".hf.space")
            or parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError("Explorer links require a canonical HTTPS hf.space host.")
    return value.rstrip("/")


def parse_query(query):
    pairs = list(query.multi_items()) if hasattr(query, "multi_items") else list(query.items())
    selected = {}
    for key, value in pairs:
        if key == "deep_link":
            raise ValueError("Use QFS measurement links, not Gradio saved-session links.")
        if key not in QUERY_FIELDS:
            continue  # HF/Gradio add their own embedding/theme parameters.
        if key in selected:
            raise ValueError("Repeated deep-link parameters are ambiguous: " + key)
        if not isinstance(value, str) or len(value) > 2048:
            raise ValueError("An evidence-link parameter is invalid or too long.")
        selected[key] = value
    if sum(len(k) + len(v) for k, v in selected.items()) > 4096:
        raise ValueError("Evidence link exceeds the supported query size.")
    revision = selected.get("registry_revision")
    if revision is not None and not SHA.fullmatch(revision):
        raise ValueError("registry_revision must be a full lowercase 40-character commit SHA.")
    if "measurement" in selected and not MEASUREMENT.fullmatch(selected["measurement"]):
        raise ValueError("measurement must be a QFS measurement ID.")
    if "model" in selected and not MODEL.fullmatch(selected["model"]):
        raise ValueError("model must be a QFS model-family ID; use target for an HF model link.")
    if selected.get("tab", "explore") not in ("explore", "costs", "cards", "contribute"):
        raise ValueError("Unknown Explorer tab in the evidence link.")
    return selected


def measurement_url(base, measurement_id, registry_revision, *, tab="explore"):
    if not MEASUREMENT.fullmatch(measurement_id or "") or not SHA.fullmatch(registry_revision or ""):
        raise ValueError("A permanent evidence link requires a measurement ID and immutable registry revision.")
    if tab not in ("explore", "cards"):
        raise ValueError("Measurement links open Explore or Cards.")
    query = {"measurement": measurement_id, "registry_revision": registry_revision}
    if tab != "explore":
        query["tab"] = tab
    return explorer_base(base) + "/?" + urlencode(query)


def evidence_links(base, measurement_id, registry_revision):
    if not MEASUREMENT.fullmatch(measurement_id or "") or not SHA.fullmatch(registry_revision or ""):
        return {}
    return {
        "explorer": measurement_url(base, measurement_id, registry_revision),
        "card_generator": measurement_url(base, measurement_id, registry_revision, tab="cards"),
        "dataset_viewer_live_search": "https://huggingface.co/datasets/%s/viewer/measurements/train?%s" % (REGISTRY, urlencode({"q": measurement_id})),
        "immutable_registry_records": "https://huggingface.co/datasets/%s/resolve/%s/data/measurements.jsonl" % (REGISTRY, registry_revision),
        "viewer_note": "The dataset viewer searches live data; only the Explorer and raw records above pin this registry snapshot.",
    }


def _dataset_repo(url):
    if not isinstance(url, str):
        return None
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or parsed.query or parsed.fragment:
        return None
    parts = parsed.path.strip("/").split("/")
    if len(parts) != 3 or parts[0] != "datasets":
        return None
    repo = "/".join(parts[1:])
    return repo if REPO.fullmatch(repo) else None


@lru_cache(maxsize=16)
def _root_descriptor(repo):
    """Resolve a public descriptor once; return immutable bytes, not a mutable cache."""
    if not REPO.fullmatch(repo):
        raise ValueError("Invalid fidelity dataset repository.")
    meta = json.loads(_public_get("https://huggingface.co/api/datasets/" + repo))
    revision = meta.get("sha")
    if not isinstance(revision, str) or not SHA.fullmatch(revision) or meta.get("private") or meta.get("gated"):
        raise ValueError("Dataset does not expose a public immutable revision.")
    raw = _public_get("https://huggingface.co/datasets/%s/raw/%s/fidelity-dataset.json" % (repo, revision))
    if len(raw) > 256 * 1024:
        raise ValueError("Root descriptor is too large for this metadata-only generator.")
    return revision, raw


def enrich_root_reference(registry, measurement_ids):
    """Add verified descriptor metadata to detached registry data, never its source.

    Registry-declared manifest seal binds the remote descriptor to the selected
    reference. A public URL/name by itself is not sufficient provenance.
    """
    from fidelity import dsformat as F

    for mid in measurement_ids:
        measurement = registry.get("measurements", {}).get(mid)
        if not measurement:
            raise ValueError("Choose a published registry measurement.")
        artifact = registry.get("artifacts", {}).get(measurement.get("artifact_ref")) or {}
        if artifact.get("kind") != "base":
            continue
        reference = registry.get("references", {}).get(measurement.get("reference_ref")) or {}
        if reference.get("artifact_ref") != artifact.get("id"):
            continue  # A different teacher's dataset cannot stand in for this root.
        expected_seal = (reference.get("capture") or {}).get("capture_receipt_sha256")
        expected_head = (reference.get("capture") or {}).get("head_sha256")
        if not expected_seal:
            continue
        for source in reference.get("sources") or []:
            repo = _dataset_repo(source.get("uri")) if source.get("kind") == "dataset_card" else None
            if not repo:
                continue
            try:
                revision, raw = _root_descriptor(repo)
                manifest = json.loads(raw)
                if manifest.get("schema") != F.DATASET_SCHEMA or manifest.get("dataset_sha256") != expected_seal:
                    continue
                if not F.verify_manifest_seal(manifest):
                    continue
                dataset, weights = manifest.get("dataset") or {}, manifest.get("weights") or {}
                identity = artifact.get("huggingface") or {}
                if dataset.get("role") != "root" or dataset.get("repository") != repo:
                    continue
                if weights.get("repository") != identity.get("repository"):
                    continue
                if identity.get("revision") and weights.get("revision") != identity["revision"]:
                    continue
                if (manifest.get("panel") or {}).get("panel_id") != measurement.get("panel_ref"):
                    continue
                if expected_head and (manifest.get("head") or {}).get("tensor_content_sha256") != expected_head:
                    continue
                capture = manifest.get("capture") or {}
                reference["fidelity_dataset"] = {
                    "repository": repo, "revision": revision, "dataset_sha256": expected_seal,
                    "capture_content_digest": capture.get("capture_content_digest"),
                    "form": capture.get("form"), "role": "root",
                }
                break
            except (OSError, ValueError, KeyError):
                continue  # The generator reports the missing root identity; never invent it.
    return registry
