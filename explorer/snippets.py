"""Read-only, registry-backed model-card snippets for QFS Explorer.

The caller supplies the six registry collections from ONE public HF dataset
revision, not browser-supplied measurements. No URL or path in a pasted card is
opened. Generation and local validation run in a bounded, isolated worker.

Hub conventions (consulted 2026-09-07):
https://huggingface.co/docs/hub/model-cards
https://huggingface.co/docs/hub/eval-results
https://datasets-server.huggingface.co/splits?dataset=malaiwah%2Fquant-fidelity-registry
The native viewer recognizes /viewer/measurements/train?q=<id>; it is a live
search, not a revision-pinned row API, and its backend can be unavailable.
"""
from __future__ import annotations

import html
import ipaddress
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

_ROOT = Path(__file__).resolve().parents[1]
_MAX_CARD_BYTES = 128 * 1024
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_SLOTS = threading.BoundedSemaphore(2)
_COLLECTIONS = ("models", "artifacts", "panels", "references", "pipelines", "measurements")
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_HF_DATASET = "https://huggingface.co/datasets/malaiwah/quant-fidelity-registry"
_EVAL_WARNING = (
    "HF still supports model-index in README frontmatter. Its newer .eval_results/*.yaml "
    "format is work in progress and requires a registered benchmark and its actual eval.yaml "
    "task_id. The QFS helper's distribution-fidelity task is not evidence of registration; "
    "no .eval_results file is emitted. Never add verified, verifyToken, or an invented arXiv ID."
)
_DATASET_CAVEAT = (
    "The evaluation repositories named by this fidelity annotation are panels/captures, "
    "not a claim that this model was trained on them. Existing unrelated training-dataset "
    "declarations are preserved. Hugging Face may label the added evaluation links "
    "'Datasets used to train:'; that label does not describe their role here."
)


class SnippetError(ValueError):
    """An actionable refusal safe to display as plain text."""


def _refused(message, *, role="", repository="", links=None, validation=None):
    return {
        "metadata_yaml": "", "markdown_snippet": "", "merged_card": "",
        "eval_results_yaml": "", "warnings": [],
        "validation": validation or {"ok": False, "errors": [message], "warnings": [],
                                      "axes": [], "skipped_axes": ["hub"]},
        "links": links or {}, "role": role, "model_repository": repository,
    }


def _card_limit(text):
    if not isinstance(text, str):
        raise SnippetError("The existing card must be UTF-8 text.")
    if len(text) > _MAX_CARD_BYTES or len(text.encode("utf-8")) > _MAX_CARD_BYTES:
        raise SnippetError("The existing card exceeds the 128 KiB UTF-8 limit; merge it locally.")


def _worker_request(payload, mode):
    if not _SLOTS.acquire(blocking=False):
        return _refused("Card generators are busy. Try again shortly; nothing was submitted.")
    try:
        request = json.dumps(payload, ensure_ascii=True, allow_nan=False)
        if len(request) > _MAX_REQUEST_BYTES:
            return _refused("The registry/card request is too large for the public generator.")
        with tempfile.TemporaryDirectory(prefix="qfs-card-") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-B", str(Path(__file__).resolve()), mode],
                input=request, text=True, encoding="utf-8", capture_output=True,
                cwd=directory, timeout=8, check=False,
            )
        if completed.returncode:
            return _refused("The bundled card validator could not complete. Contact the Space maintainer.")
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired:
        return _refused("Card generation/validation exceeded 8 seconds. Use the local fidelity-card command.")
    except (OSError, ValueError, TypeError, RecursionError):
        return _refused("The card request could not be processed safely. Use ordinary UTF-8 text and registry JSON.")
    finally:
        _SLOTS.release()


def generate_snippet(registry: dict, measurement_ids: list[str], *, explorer_base: str,
                     registry_revision: str, existing_card: str = "") -> dict:
    """Generate pasteable metadata and evidence prose, without writing or publishing.

    ``registry_revision`` is the full HF *registry dataset* commit supplied by the
    snapshot loader (not the QFS source commit). Collection values can be id->row
    mappings or row lists. Refusals return empty paste outputs and validation.ok
    false. Successful merged_card preserves the original body byte-for-byte;
    append markdown_snippet separately where appropriate.

    A root requires a structured ``fidelity_dataset`` on its artifact or matching
    reference row, including repository, revision, dataset_sha256, form and role.
    Descriptor enrichment must come from the trusted snapshot loader, never user
    card YAML. Unstructured source notes/abbreviated hashes do not establish it.
    """
    try:
        _card_limit(existing_card)
        if not isinstance(registry, dict):
            raise SnippetError("The snapshot loader must supply registry collections.")
        if (not isinstance(measurement_ids, list) or not measurement_ids
                or len(measurement_ids) > 32
                or any(not isinstance(mid, str) or not re.fullmatch(r"measurement--[A-Za-z0-9_.-]+", mid)
                       for mid in measurement_ids)):
            raise SnippetError("Select 1–32 registered measurement IDs.")
        if len(set(measurement_ids)) != len(measurement_ids):
            raise SnippetError("Select each measurement only once.")
        if not isinstance(registry_revision, str) or not _SHA.fullmatch(registry_revision):
            raise SnippetError("An immutable full 40-character HF registry revision is required.")
        if not _public_url(explorer_base):
            raise SnippetError("Explorer base must be a public HTTPS URL without credentials.")
    except (SnippetError, UnicodeError) as exc:
        return _refused(str(exc) if isinstance(exc, SnippetError) else "Card text is not valid UTF-8.")
    return _worker_request({"registry": registry, "measurement_ids": measurement_ids,
                            "explorer_base": explorer_base, "registry_revision": registry_revision,
                            "existing_card": existing_card}, "--generate-snippet")


def validate_public_card(card: str, *, confirm_public: bool = False) -> dict:
    """Explicit opt-in POST to HF's fixed validate-yaml endpoint, never automatic.

    Call only for generated, nonprivate examples. The full text leaves the host.
    This reports only Hub YAML acceptance, not measurement or receipt verification.
    """
    if confirm_public is not True:
        return {"axis": "hub", "ran": False, "ok": None,
                "skipped": "Explicit confirm_public=True is required before sending card text to HF."}
    try:
        _card_limit(card)
    except (SnippetError, UnicodeError):
        return {"axis": "hub", "ran": False, "ok": False,
                "errors": ["Public card must be valid UTF-8 text within 128 KiB."]}
    result = _worker_request({"card": card}, "--validate-public-card")
    if "validation" in result:
        return {"axis": "hub", "ran": False, "ok": False,
                "errors": result["validation"]["errors"]}
    return result


def _public_url(value):
    if not isinstance(value, str) or len(value) > 8192 or any(ord(c) < 33 for c in value):
        return None
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        if (parsed.scheme != "https" or not host or parsed.username or parsed.password
                or host == "localhost" or "." not in host or host.endswith((".local", ".localhost"))):
            return None
        try:
            if not ipaddress.ip_address(host).is_global:
                return None
        except ValueError:
            pass
        return value
    except ValueError:
        return None


def _md(value):
    text = html.escape(str(value), quote=False)
    return re.sub(r"([\\`*{}_\[\]()#+.!|>~-])", r"\\\1", text).replace("\n", " ")


def _link(label, url):
    return f"[{_md(label)}](<{quote(url, safe=':/?=&%#@+;,-._~')}>)"


def _safe_frontmatter(text, yaml):
    """Reject alias bombs, duplicate/complex keys and deep YAML before safe_load."""
    _card_limit(text)
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise SnippetError("The existing card has an unclosed YAML frontmatter block.")
    front_text = "\n".join(lines[1:end])
    depth = count = 0
    for event in yaml.parse(front_text):
        count += 1
        if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None):
            raise SnippetError("YAML anchors and aliases are not accepted by this public merger; expand them locally.")
        if getattr(event, "tag", None):
            raise SnippetError("Explicit YAML tags are not accepted by this public merger.")
        if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
            depth += 1
        elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
            depth -= 1
        if depth > 32 or count > 16384:
            raise SnippetError("YAML frontmatter is too deeply nested or complex; merge it locally.")

    class CardLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            output = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str) or key in output:
                    raise SnippetError("YAML mapping keys must be unique strings.")
                output[key] = self.construct_object(value_node, deep=deep)
            return output

    front = yaml.load(front_text, Loader=CardLoader) or {}
    if not isinstance(front, dict):
        raise SnippetError("Model card YAML frontmatter must be a mapping.")
    stack = [front]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if "verified" in item or "verifyToken" in item:
                raise SnippetError("HF-controlled verified/verifyToken claims cannot be carried through this merger.")
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise SnippetError("Card metadata cannot contain NaN or Infinity.")
    for key in ("tags", "datasets", "metrics"):
        if key in front and (not isinstance(front[key], list)
                             or any(not isinstance(v, str) for v in front[key])):
            raise SnippetError(f"Existing {key} metadata must be a list of strings to merge safely.")
    return front


def _registry(raw):
    result = {}
    for name in _COLLECTIONS:
        rows = raw.get(name)
        if isinstance(rows, list):
            indexed = {}
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in indexed:
                    raise SnippetError(f"The snapshot has invalid or duplicate {name} rows.")
                indexed[row["id"]] = row
            rows = indexed
        if not isinstance(rows, dict) or any(not isinstance(row, dict) or row.get("id") != key
                                             for key, row in rows.items()):
            raise SnippetError(f"The snapshot is missing valid {name} collections.")
        result[name] = rows
    if raw.get("_snapshot") is not None:
        if not isinstance(raw["_snapshot"], dict):
            raise SnippetError("Registry snapshot metadata must be a mapping.")
        result["_snapshot"] = raw["_snapshot"]
    return result


def _sources(row):
    return list(row.get("sources") or []) + list((row.get("provenance") or {}).get("sources") or [])


def _evidence_links(rows):
    seen, links = set(), []
    for row in rows:
        for source in _sources(row):
            uri = _public_url(source.get("uri"))
            if not uri or uri in seen:
                continue
            seen.add(uri)
            links.append({"url": uri, "kind": source.get("kind") or "source",
                          "sha256": source.get("sha256"), "note": source.get("note"),
                          "record_id": row["id"]})
    return links


def _root_dataset(artifact, references):
    candidates = [artifact.get("fidelity_dataset")]
    candidates.extend(ref.get("fidelity_dataset") for ref in references
                      if ref.get("artifact_ref") == artifact["id"])
    candidates = [fd for fd in candidates if fd is not None]
    if not candidates:
        raise SnippetError(
            "This is a base/root model, not a quant. Root annotations require an actual fidelity "
            "dataset with repository, sealed dataset_sha256, form and revision recorded in the "
            "trusted registry export. Source links/abbreviated notes alone cannot supply these "
            "fields. No quantized base_model annotation or dataset identity was invented.")
    first = candidates[0]
    if any(fd != first for fd in candidates[1:]):
        raise SnippetError("The selected root rows name different fidelity datasets; select one dataset.")
    if (not isinstance(first, dict) or not _REPO.fullmatch(first.get("repository") or "")
            or not _SHA.fullmatch(first.get("revision") or "")
            or not _DIGEST.fullmatch(first.get("dataset_sha256") or "")
            or first.get("form") not in ("hidden", "logit") or first.get("role") != "root"):
        raise SnippetError("The root fidelity dataset identity is incomplete; a full recorded revision, seal, form and root role are required.")
    return dict(first)


def _local_validation(text, registry, cardmeta, yaml):
    """Reuse QFS invariants plus its schema and the real local Hub parser.

    Do not use cardmeta._roundtrip_axis: that CLI helper searches interpreters and
    writes a shared /tmp script. An isolated public worker needs neither behavior.
    """
    from huggingface_hub import ModelCard
    from jsonschema import Draft202012Validator

    front, _ = cardmeta.split_card(text)
    schema = json.loads((_ROOT / "docs/schema/fidelity-card-annotation.schema.json").read_text())
    schema_errors = [f"x_fidelity {'.'.join(map(str, e.path))}: {e.message}"
                     for e in Draft202012Validator(schema).iter_errors(front["x_fidelity"])]
    schema_axis = {"axis": "schema", "ran": True, "ok": not schema_errors, "errors": schema_errors}
    ours = cardmeta._our_axis(text, registry)
    card = ModelCard(text)
    again = ModelCard(str(card))
    before = json.loads(json.dumps(front, sort_keys=True, default=str))
    after = json.loads(json.dumps(card.data.to_dict(), sort_keys=True, default=str))
    reparsed = json.loads(json.dumps(again.data.to_dict(), sort_keys=True, default=str))
    lost, added = sorted(before.keys() - after.keys()), sorted(after.keys() - before.keys())
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    roundtrip_ok = not (lost or added or changed) and after == reparsed
    roundtrip = {"axis": "roundtrip", "ran": True, "ok": roundtrip_ok,
                 "detail": {"lost": lost, "added": added, "changed": changed,
                            "reparsed_equal": after == reparsed},
                 "errors": [] if roundtrip_ok else ["Hub ModelCard roundtrip changed metadata; refuse a lossy merge."]}
    axes = [schema_axis, ours, roundtrip,
            {"axis": "hub", "ran": False, "ok": None,
             "skipped": "Offline: card text was not sent to HF. validate_public_card is explicit opt-in."}]
    errors = [e for axis in axes for e in axis.get("errors", [])]
    return {"axes": axes, "errors": errors,
            "warnings": [w for axis in axes for w in axis.get("warnings", [])],
            "skipped_axes": ["hub"], "ok": not errors}


def _hf_result_key(result):
    dataset = result.get("dataset") or {}
    return ((result.get("task") or {}).get("type"), dataset.get("type"),
            dataset.get("config"), dataset.get("split"), dataset.get("revision"))


def _generate(payload, cardmeta, yaml):
    registry = _registry(payload["registry"])
    ids, revision = payload["measurement_ids"], payload["registry_revision"]
    existing = payload["existing_card"]
    front = _safe_frontmatter(existing, yaml)
    rows = []
    for mid in ids:
        row = registry["measurements"].get(mid)
        if not row or row.get("status") != "published":
            raise SnippetError(f"{mid} is not a registered published measurement; receipts alone cannot annotate a card.")
        metric = row.get("metric") or {}
        if (metric.get("name") not in ("mean_tokenwise_kld", "mean_of_run_means_tokenwise_kld")
                or metric.get("units") != "nats" or metric.get("direction") != "reference_to_candidate"
                or metric.get("higher_is_better") is not False
                or (row.get("estimator") or {}).get("accumulation_dtype") not in ("float64", "fp64", "float32_reduce_legacy", "unknown", None)
                or not isinstance(metric.get("value"), (int, float)) or isinstance(metric.get("value"), bool)
                or not math.isfinite(metric["value"])):
            raise SnippetError(f"{mid} does not have the recorded KL contract supported by the card helper.")
        rows.append(row)
    artifact_ids = {row.get("artifact_ref") for row in rows}
    if len(artifact_ids) != 1:
        raise SnippetError("The selection spans different artifacts. A card can only attribute results to one measured artifact.")
    artifact = registry["artifacts"].get(next(iter(artifact_ids)))
    if not artifact:
        raise SnippetError("The selected artifact does not resolve in the registry snapshot.")
    repository = (artifact.get("huggingface") or {}).get("repository")
    if not isinstance(repository, str) or not _REPO.fullmatch(repository):
        raise SnippetError("The registry does not record the measured artifact's model repository.")
    if (artifact.get("availability") or {}).get("status") != "public":
        raise SnippetError("The measured model artifact is not recorded as publicly available.")
    kind = artifact.get("kind")
    role = "root" if kind == "base" else "quant" if kind in ("quant", "requantized") else ""
    if not role:
        raise SnippetError("This artifact kind has no unambiguous root/quant annotation role; no quantized relationship was assumed.")
    snapshot = dict(registry.get("_snapshot") or {})
    for key in ("revision", "registry_revision", "hf_revision"):
        if snapshot.get(key) and snapshot[key] != revision:
            raise SnippetError("The supplied registry revision disagrees with the snapshot metadata.")
    snapshot["revision"] = revision
    registry["_snapshot"] = snapshot
    base = urlsplit(payload["explorer_base"])
    links = {"registry_jsonl": f"{_HF_DATASET}/resolve/{revision}/data/measurements.jsonl",
             "registry_revision": revision, "measurements": {}}
    references, evidence_rows = [], [artifact]
    for row in rows:
        connected = []
        for collection, field in (("references", "reference_ref"), ("panels", "panel_ref"),
                                  ("pipelines", "pipeline_ref")):
            target = registry[collection].get(row.get(field))
            if not target:
                raise SnippetError(f"{row['id']} names an unresolved {field}; no provenance was guessed.")
            connected.append(target)
        references.append(connected[0])
        evidence_rows.extend([row, *connected])
        links["measurements"][row["id"]] = {
            "explorer": urlunsplit((base.scheme, base.netloc, base.path,
                                    urlencode({"measurement": row["id"], "registry_revision": revision}), "")),
            "viewer": f"{_HF_DATASET}/viewer/measurements/train?{urlencode({'q': row['id']})}",
            "receipts": _evidence_links([row, *connected]),
        }
    links["sources"] = _evidence_links(evidence_rows)
    warnings = [_DATASET_CAVEAT, _EVAL_WARNING,
                "Native HF viewer links search the live registry and may be unavailable; use the immutable JSONL fallback for exact snapshot values."]
    try:
        dataset = _root_dataset(artifact, references) if role == "root" else None
    except SnippetError as exc:
        return _refused(str(exc), role=role, repository=repository, links=links)
    identities = []
    for mid in ids:
        ref_repo, ref_rev, notes = cardmeta.reference_identity(registry, [mid])
        warnings.extend(notes)
        identities.append((ref_repo, ref_rev))
    if len(set(identities)) != 1:
        raise SnippetError("Selected rows use different reference model identities; make separate annotations.")
    reference_model, reference_revision = identities[0]
    if reference_revision and not _SHA.fullmatch(reference_revision):
        warnings.append("The reference revision is not recorded as a full immutable SHA; the annotation leaves it unknown.")
        reference_revision = None
    if not reference_model:
        warnings.append("Reference model identity is unavailable in this snapshot; x_fidelity.reference_model is null.")
    if not reference_revision:
        warnings.append("Reference revision is unavailable; no immutable reference pin was invented.")
    base_model = None
    if role == "quant":
        parent = registry["artifacts"].get(artifact.get("derived_from_artifact_ref")) or {}
        base_model = (parent.get("huggingface") or {}).get("repository")
        if not isinstance(base_model, str) or not _REPO.fullmatch(base_model) or base_model == repository:
            raise SnippetError("The quant's actual parent model repository is unavailable; the evaluation reference cannot substitute for a base_model relationship.")
        if front.get("base_model") not in (None, base_model, [base_model]):
            raise SnippetError("Existing base_model disagrees with the registry's quant parent; resolve this locally.")
        if front.get("base_model_relation") not in (None, "quantized"):
            raise SnippetError("Existing base_model_relation conflicts with the registry's quant role.")
    elif front.get("base_model_relation") == "quantized":
        raise SnippetError("The selected artifact is a base/root but the existing card claims a quantized relationship.")
    index = cardmeta.build_model_index(registry, ids, repository.rsplit("/", 1)[-1])
    datasets = []
    for result, row in zip(index[0]["results"], rows):
        result["task"]["name"] = "Distribution fidelity (KL divergence vs registered reference)"
        precision = (row.get("estimator") or {}).get("accumulation_dtype")
        if precision not in ("float64", "fp64"):
            warnings.append(f"{row['id']}: estimator precision is {precision or 'unrecorded'}, not fp64. This annotation preserves that limitation; do not compare it as an fp64 result.")
        declared_lane = ((registry["pipelines"][row["pipeline_ref"]].get("lane") or {}).get("name"))
        result["metrics"][0]["args"]["lane_inferred"] = not bool(declared_lane)
        if not declared_lane:
            warnings.append(f"{row['id']}: lane is inferred by the existing registry convention as {cardmeta.lane_of(registry, row)}; the pipeline does not explicitly declare it. This label is not proof of GPU count.")
        panel_repository = result["dataset"]["type"]
        if not isinstance(panel_repository, str) or not _REPO.fullmatch(panel_repository):
            panel = registry["panels"][row["panel_ref"]]
            result["dataset"]["type"] = panel["id"]
            result["dataset"]["args"]["dataset_identifier_kind"] = "qfs_panel_id_not_hf_repository"
            result["dataset"]["args"]["availability"] = (panel.get("availability") or {}).get("status", "unknown")
            warnings.append(f"{row['id']}: no HF dataset repository is recorded for this panel. model-index uses its actual QFS panel ID, not an invented repository; no top-level datasets link is added for it.")
        elif panel_repository not in datasets:
            datasets.append(panel_repository)
        panel_revision = result["dataset"].get("revision")
        if not panel_revision or not _SHA.fullmatch(panel_revision):
            result["dataset"].pop("revision", None)
            warnings.append(f"{row['id']}: no full immutable panel dataset revision is recorded; no pin was invented.")
        result["source"] = {"name": "QFS Explorer — registry snapshot and receipts",
                            "url": links["measurements"][row["id"]]["explorer"]}
        result["metrics"][0]["args"]["comparability_class"] = (row.get("comparability") or {}).get("class", "unknown")
        refusal = cardmeta.attributable_refusal(registry, row, cardmeta.lane_of(registry, row))
        if refusal:
            warnings.append(refusal)
        if role == "root":
            if references[ids.index(row["id"])].get("artifact_ref") != artifact["id"] or row["metric"]["value"] != 0:
                raise SnippetError("Root model-index supports registered exact-zero self comparisons only, not cross-model lineage measurements.")
            result["metrics"][0]["args"]["comparison_kind"] = "self_compare"
            result["metrics"][0]["args"]["exact_zero_asserted"] = True
    if dataset and dataset["repository"] not in datasets:
        datasets.append(dataset["repository"])
    head_policy = (artifact.get("scope") or {}).get("head_policy") or "unknown"
    assignments = [entry for entry in (artifact.get("scope") or {}).get("assignments", [])
                   if entry.get("tensor_class") == "lm_head"]
    bits = {entry.get("bits_per_weight") for entry in assignments}
    head_bits = next(iter(bits)) if len(bits) == 1 else None
    if not isinstance(head_bits, int) or isinstance(head_bits, bool):
        head_bits = None  # Effective bpw (e.g. 8.5) is not an integer head precision.
    fidelity = cardmeta.build_x_fidelity(
        registry, role=role, measurement_ids=ids, artifact_id=artifact["id"],
        reference_model=reference_model, reference_revision=reference_revision,
        fidelity_dataset=dataset, scope_digest=artifact.get("scope_digest"),
        head_quantized=head_policy == "quantized", head_bits=head_bits,
    )
    fidelity["head"]["policy"] = head_policy if head_policy in ("native", "quantized", "shared_reference") else "unknown"
    if head_policy == "unknown":
        fidelity["head"]["quantized"] = None
    fidelity["head"]["note"] = (
        "The registry export does not establish this artifact's normative lm_head tensor-content "
        "digest. No capture head_sha256 or file/container digest was substituted. Cross-artifact "
        "hidden-state replay is not permitted by this annotation."
    )
    warnings.append(fidelity["head"]["note"])
    fidelity["dataset_usage"] = "evaluation_not_training"
    fidelity["evaluation_datasets"] = list(datasets)
    fidelity["evaluation_panels"] = [
        {key: registry["panels"][pid].get(key) for key in ("id", "identity", "availability")}
        for pid in sorted({row["panel_ref"] for row in rows})]
    fidelity["evidence"] = {
        "artifact": {key: artifact.get(key) for key in ("id", "kind", "huggingface", "scope", "disclosures")},
        "measurements": [{key: row.get(key) for key in (
            "id", "measurement_scope", "comparability", "estimator", "provenance", "disclosures",
            "uncertainty", "harness", "quality_gate")} for row in rows],
        "sources": links["sources"],
    }
    old_index = front.get("model-index") or []
    if old_index:
        if not isinstance(old_index, list) or len(old_index) != 1 or old_index[0].get("name") != index[0]["name"]:
            raise SnippetError("Existing model-index names a different model or multiple models; merge it locally without losing attribution.")
        for old_result in old_index[0].get("results") or []:
            cited = {(metric.get("args") or {}).get("measurement_id") for metric in old_result.get("metrics") or []}
            cited.discard(None)
            if cited and not cited.issubset(set(ids)):
                raise SnippetError("Existing card cites other fidelity measurements; include them in the selection to preserve their evidence.")
            if cited:
                regenerated = [metric for result in index[0]["results"]
                               if _hf_result_key(result) == _hf_result_key(old_result)
                               for metric in result.get("metrics") or []]
                if any(not (metric.get("args") or {}).get("measurement_id") and metric not in regenerated
                       for metric in old_result.get("metrics") or []):
                    raise SnippetError("An existing result mixes selected fidelity metrics with unrelated or changed unattributed metrics. Separate that result locally before merging; no metrics were discarded.")
            if not cited:
                index[0]["results"].append(old_result)
    metrics = sorted({metric["type"] for result in index[0]["results"] for metric in result["metrics"]})
    merge_args = dict(model_index=index, x_fidelity=fidelity, datasets=datasets, metrics=metrics,
                      tags=("fidelity", "kl-divergence", "fidelity-provenance"),
                      base_model=base_model, base_model_relation="quantized" if role == "quant" else None)
    merged = cardmeta.merge_card(existing, **merge_args)
    generated = cardmeta.merge_card("", **merge_args)
    metadata, _ = cardmeta.split_card(generated)
    validation = _local_validation(merged, registry, cardmeta, yaml)
    if not validation["ok"]:
        return _refused("The generated card failed local validation.", role=role, repository=repository,
                        links=links, validation=validation)
    warnings.extend(validation["warnings"])
    dataset_caveat = _DATASET_CAVEAT
    if datasets:
        dataset_caveat += "\n\nEvaluation repositories: " + ", ".join(_md(repo) for repo in datasets) + "."
    else:
        dataset_caveat += "\n\nNo HF dataset repository is recorded; model-index uses the logical QFS panel ID. No top-level datasets link was invented."
    warnings[0] = dataset_caveat
    prose = ["## Distribution fidelity evidence", "", dataset_caveat, "",
             f"Measured artifact: {_link(repository, 'https://huggingface.co/' + repository)}. "
             f"Role: **{role}**. Registry snapshot: `{revision}`.", "",
             "These are registry-reported measurements, not an HF-verified badge or a claim of "
             "universal quality. Equal comparability keys alone do not authorize ranking; inspect "
             "the Explorer's full-group predicate, including hardware and pipeline.", ""]
    if dataset:
        dataset_base = "https://huggingface.co/datasets/" + dataset["repository"]
        links["fidelity_dataset"] = {
            "repository": dataset["repository"], "revision": dataset["revision"],
            "dataset_sha256": dataset["dataset_sha256"],
            "tree": dataset_base + "/tree/" + dataset["revision"],
            "manifest": dataset_base + "/blob/" + dataset["revision"] + "/fidelity-dataset.json",
        }
        prose.extend([_link("Pinned fidelity dataset", links["fidelity_dataset"]["tree"]) + " · "
                      + _link("Sealed capture manifest", links["fidelity_dataset"]["manifest"]), ""])
    for row in rows:
        row_links = links["measurements"][row["id"]]
        comparability, provenance = row.get("comparability") or {}, row.get("provenance") or {}
        lane_text = cardmeta.lane_of(registry, row)
        if not ((registry["pipelines"][row["pipeline_ref"]].get("lane") or {}).get("name")):
            lane_text += " (legacy inference; not explicitly declared)"
        prose.extend([
            f"### {_md(row['id'])}", "",
            f"**KL(reference || candidate): {repr(row['metric']['value'])} {_md(row['metric']['units'])}** "
            f"(lower is better); lane: {_md(lane_text)}; "
            f"precision: {_md((row.get('estimator') or {}).get('accumulation_dtype') or 'unrecorded')}; "
            f"comparability: **{_md(comparability.get('class', 'unknown'))}**; "
            f"key: {_md(comparability.get('key', 'unknown'))}; measured by: {_md(provenance.get('measured_by', 'unknown'))}.",
            f"Measurement scope: {_md(json.dumps(row.get('measurement_scope'), ensure_ascii=True, sort_keys=True))}.",
            f"Artifact scope: {_md(artifact.get('scope_digest'))}.",
            _link("Explore this exact registry snapshot", row_links["explorer"]) + " · "
            + _link("HF live dataset search", row_links["viewer"]) + " · "
            + _link("Immutable registry JSONL (find this measurement ID)", links["registry_jsonl"]), "",
        ])
        disclosures = []
        related = [row, artifact, registry["panels"][row["panel_ref"]],
                   registry["references"][row["reference_ref"]], registry["pipelines"][row["pipeline_ref"]]]
        important = sorted({str(d.get("code")) for record in related for d in record.get("disclosures") or []
                            if d.get("severity") in ("caveat", "blocking", "warning")})
        if important:
            prose.extend(["**Caveats:** " + ", ".join(_md(code) for code in important) + ".", ""])
        prose.extend(["<details>", "<summary>Full disclosures and source receipts</summary>", ""])
        for record in related:
            for disclosure in record.get("disclosures") or []:
                text = f"{record['id']}: {disclosure.get('code')} — {disclosure.get('detail')}"
                if text not in disclosures:
                    disclosures.append(text)
        prose.extend(f"- Disclosure: {_md(text)}" for text in disclosures)
        for number, source in enumerate(row_links["receipts"], 1):
            digest = f"; recorded SHA-256 `{source['sha256']}`" if source.get("sha256") else "; digest not recorded"
            prose.append(f"- Evidence {number}: {_link(source['kind'], source['url'])}{digest}. "
                         + (_md(source["note"]) if source.get("note") else ""))
        if not row_links["receipts"]:
            prose.append("- No public receipt/source URL is recorded for this measurement; use the immutable registry row.")
        prose.extend(["", "</details>", ""])
    prose.extend(["### Reading and citing this evidence", "",
                  "Cite the measurement ID and full registry revision above with its Explorer link and "
                  "recorded receipt hashes. Source URLs are quoted as registered, not silently repinned; "
                  "a source on a moving branch must be checked against its recorded digest. "
                  "The native HF viewer is a live search and may be unavailable or display rounded values; "
                  "the immutable registry JSONL and card metadata retain the supplied metric precision.", "",
                  fidelity["head"]["note"], "", _EVAL_WARNING, ""])
    return {"metadata_yaml": yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True, width=100),
            "markdown_snippet": "\n".join(prose), "merged_card": merged,
            "eval_results_yaml": "", "warnings": list(dict.fromkeys(warnings)),
            "validation": validation, "links": links, "role": role, "model_repository": repository}


if __name__ == "__main__":
    if sys.argv[1:] not in (["--generate-snippet"], ["--validate-public-card"]):
        raise SystemExit("This module is a bounded, read-only Explorer card generator.")
    _card_error = SnippetError
    try:
        # Trusted import root only. Never select a module, executable or file from input.
        sys.path.insert(0, str(_ROOT / "bin"))
        import yaml
        from fidelity import cardmeta
        _card_error = cardmeta.CardError

        request = json.loads(sys.stdin.read(_MAX_REQUEST_BYTES + 1))
        if sys.argv[1] == "--validate-public-card":
            _safe_frontmatter(request["card"], yaml)
            output = cardmeta._hub_axis(request["card"], "model")
        else:
            output = _generate(request, cardmeta, yaml)
    except (SnippetError, _card_error) as exc:
        output = _refused(str(exc))
    except Exception:
        # Do not leak tracebacks, local paths, credentials or parser input excerpts.
        output = _refused("The bundled generator could not safely process this card/snapshot. Check its structure or run fidelity-card locally for diagnostics.")
    print(json.dumps(output, ensure_ascii=True, allow_nan=False))
