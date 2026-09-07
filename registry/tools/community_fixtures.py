#!/usr/bin/env python3
"""Import qualified native fixture roots into the maintained registry sources.

Offline: publication must happen first. This tool copies evidence, not weights,
into protocol/community-fixtures and derives the six real registry collections.
seed_registry consumes the SAME sources on every authoritative reseed.

Usage: python registry/tools/community_fixtures.py --manifest root-publications.json
Then run seed_registry.py, registry_validate.py and registry_render.py normally.

Manifest: qfs.community-root-publications.v1, fixtures as agreed by the release
orchestrator. Optional qualification_path defaults to
root_dir/receipts/root-qualification.json. comparison_path must be inside root_dir
or provide comparison_path_in_repo (relative to evidence_repository at its pin).
Each fixture also needs harness_path: a recorded registry harness covering
metric.value, captured when comparison ran, NOT reconstructed from today's tree.
The harness repository may honestly name a dirty parent; its digests are identity.
No quantized or reconstructed candidate is accepted by this native-root importer.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile

import registry_lib as L
import harness_id as H

SOURCE_SCHEMA = "qfs.community-fixture-registry-sources.v1"
MANIFEST_SCHEMA = "qfs.community-root-publications.v1"
SOURCE_DIR = "protocol/community-fixtures"
PURPOSE = ("Synthetic/random-initialized test fixture, not a trained language model, "
           "assistant, quality benchmark, or production-kernel qualification. This "
           "native root is a reference for fixture fidelity only. Its model and "
           "reference identities keep it out of trained-model comparability groups.")


def require(condition, message):
    if not condition:
        raise ValueError("community fixtures: " + message)


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle, parse_constant=L.reject_nonfinite_token)


def digest(path):
    return L.sha256_file(str(path))


def sealed(doc, field):
    value = doc.get(field)
    body = dict(doc, **{field: ""})
    require(isinstance(value, str) and value == L.sha256_hex(L.canonical_json(body)),
            "invalid %s self-seal" % field)


def relative(path):
    p = Path(path)
    require(not p.is_absolute() and ".." not in p.parts and str(p) not in ("", "."),
            "unsafe relative path %r" % str(path))
    return p.as_posix()


def inside(root, path):
    p = Path(root) / relative(path)
    require(p.resolve().is_relative_to(Path(root).resolve()), "path escapes evidence root")
    return p


def pin(repository, revision):
    require(isinstance(repository, str) and re.fullmatch(r"[^/\s]+/[^/\s]+", repository),
            "a public owner/repository is required")
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
            "%s needs a full immutable commit" % repository)


def url(repository, revision, path=None, dataset=True):
    pin(repository, revision)
    base = "https://huggingface.co/" + ("datasets/" if dataset else "") + repository
    return base + ("/resolve/" + revision + "/" + relative(path) if path else "/tree/" + revision)


def slug(text):
    return re.sub(r"[^a-z0-9.-]+", "-", text.lower()).strip(".-")


def check_harness(harness):
    require(harness.get("recorded") is True and "metric.value" in harness.get("covers", []),
            "comparison needs a recorded harness covering metric.value (harness_path)")
    digests = harness.get("code_digests") or []
    require(digests and len({d["role"] for d in digests}) == len(digests),
            "harness needs distinct code roles")
    for d in digests:
        relative(d["path"])
        require(re.fullmatch(r"[0-9a-f]{64}", d["sha256"]), "invalid harness code digest")
    require(harness.get("harness_id") == H.compute_id(
        digests, harness.get("tool_versions") or {}, harness.get("boundary")),
        "comparison harness ID does not recompute")
    repo = harness.get("repository") or {}
    require(re.fullmatch(r"[0-9a-f]{40}", repo.get("commit") or ""),
            "comparison harness needs a full commit or honest dirty parent")
    require(repo.get("commit_role") in ("exact", "parent") and
            (repo["commit_role"] != "parent" or repo.get("dirty") is True),
            "invalid harness repository provenance")


def check_evidence(entry, documents):
    d, runtime, q, c = (documents[k] for k in ("dataset", "runtime", "qualification", "comparison"))
    require(d.get("schema") == "malaiwah.fidelity-dataset.v1", "unsupported root descriptor")
    sealed(d, "dataset_sha256")
    sealed(runtime, "receipt_sha256")
    sealed(q, "receipt_sha256")
    sealed(c, "receipt_sha256")
    require(q.get("schema") == "fidelity.root-qualification-receipt.v1", "unsupported qualification")
    require(c.get("schema") == "malaiwah.fidelity-comparison-receipt.v1", "unsupported comparison")
    require(d["dataset"]["role"] == "root" and d["weights"].get("quantized") is False
            and d["runtime"]["source"] == "native"
            and not runtime["capture_tool"].get("weights_decode")
            and not q["job_contract"].get("candidate"),
            "only native bases may become roots; reconstructed/quant variants retain child roles")
    require(d["dataset"]["author"]["name"] == "malaiwah", "capture authorship is not malaiwah")
    require(d["dataset"]["repository"] == entry["root_repository"]
            and q["dataset_repository"] == entry["root_repository"], "root repository mismatch")
    require(d["weights"]["repository"] == entry["model_repository"]
            and d["weights"]["revision"] == entry["model_revision"], "model pin differs from capture")
    require(d["coverage"]["complete"] is True and d["capture"]["dtype_lossless"] is True,
            "root must be complete and lossless")
    require(runtime["stack_fingerprint"]["device"] == "cpu"
            and d["runtime"]["lane"] == "other"
            and q["job_contract"].get("execution_kind") == "local",
            "fixture importer requires the qualified local CPU lane")
    require(q["job_contract"].get("device") == "cpu", "job does not qualify CPU capture")
    require(q["job_contract"].get("weights_repository") == entry["model_repository"]
            and q["job_contract"].get("weights_revision") == entry["model_revision"]
            and q["job_contract"].get("dtype") in ("bf16", "bfloat16")
            and d["capture"]["dtype"] == "BF16",
            "qualification does not bind the native BF16 model pin")
    require(all(q["reproduction_confirmation"].get(k) is True for k in (
        "two_fresh_processes", "distinct_dataset_roots", "both_independently_verified",
        "exact_zero_comparison", "canonical_dataset_only")), "two-cold qualification is incomplete")
    first, repeat = (q["captures"][k] for k in ("canonical", "repeat"))
    require(first["process_label"] != repeat["process_label"]
            and first["dataset_sha256"] != repeat["dataset_sha256"], "not two independent captures")
    for identity, side in ((first, c["reference"]), (repeat, c["candidate"])):
        require(identity["capture_content_digest"] == d["capture"]["capture_content_digest"]
                and side["capture_content_digest"] == identity["capture_content_digest"]
                and side["dataset_sha256"] == identity["dataset_sha256"],
                "comparison does not bind qualified cold captures")
        require(side["role"] == "root" and side["weights"]["quantized"] is False
                and side["weights"]["checkpoint_identity_sha256"] == d["weights"]["checkpoint_identity_sha256"],
                "self-comparison does not bind native root weights")
    # Publishing attaches qualification and reseals the canonical descriptor. Its
    # pre-publication dataset hash therefore intentionally differs from first.
    require(first["runtime_manifest_sha256"] == d["runtime"]["file_sha256"]
            and first["capture_manifest_sha256"] == d["capture"]["manifest_file_sha256"]
            and first["panel"]["suite_token_hash_sha256"] == d["panel"]["suite_token_hash_sha256"]
            and first["weights_repository"] == entry["model_repository"]
            and first["weights_revision"] == entry["model_revision"],
            "published canonical capture differs from qualification")
    require(c["comparison_kind"] == "reproduction_confirmation"
            and c["metric"]["value"] == 0.0 and c["kl"]["max"] == 0.0
            and c["top1_agreement"] == 1.0, "floor is not measured exact zero")
    require(all(c["self_compare"].get(k) is True for k in (
        "capture_content_digest_equal", "weights_identity_equal", "asserted_exact_zero", "force_compute_agreed")),
        "floor lacks forced exact-zero proof")
    require(c["comparability"]["same_lane"] is True
            and c["comparability"]["usable_as_floor"] is True
            and c["comparability"]["class"] == "strict"
            and not c["comparability"].get("bias"), "floor not scientifically admitted")
    require(c["gates"] and all(g.get("passed") is True and not g.get("overridden_by")
                               for g in c["gates"].values()), "comparison contains overridden/failed gates")
    require(c["panel"]["suite_token_hash_sha256"] == d["panel"]["suite_token_hash_sha256"]
            and c["measurement_scope"]["scored_positions"] == d["capture"]["scored_rows_total"]
            and c["measurement_scope"]["covers_full_panel"] is True, "floor panel/coverage mismatch")
    require(documents["config"].get("quantization_config") in (None, {}), "base config declares a quantizer")
    require(d["scope"]["policy"] == "native" and d["scope"]["head_policy"] == "native"
            and all(a["treatment"] in ("native", "not_present") for a in d["scope"]["assignments"]),
            "root scope is not native")
    check_harness(documents["harness"])
    recorded = {(f["path"], f["sha256"]) for f in documents["harness"]["code_digests"]}
    require(runtime.get("source_files") and all((path, sha) in recorded
            for path, sha in runtime["source_files"].items()),
            "measurement harness must retain the actual captured source-file digests")


def read_entry(source_root, entry):
    documents = {}
    for key, record in entry["files"].items():
        p = inside(source_root, record["path"])
        require(digest(p) == record["sha256"], "maintained %s evidence hash changed" % key)
        if key != "license":
            documents[key] = load(p)
    check_evidence(entry, documents)
    require(entry["files"]["runtime"]["sha256"] == documents["dataset"]["runtime"]["file_sha256"],
            "runtime bytes do not match sealed dataset")
    require(entry["files"]["capture"]["sha256"] == documents["dataset"]["capture"]["manifest_file_sha256"],
            "capture manifest bytes do not match sealed dataset")
    require(entry["files"]["comparison"]["sha256"] == documents["qualification"]["comparison"]["file_sha256"],
            "comparison bytes do not match qualification")
    require(entry["files"]["config"]["sha256"] == documents["dataset"]["weights"]["config_sha256"],
            "config bytes do not match captured weights")
    require(entry["files"]["panel"]["sha256"] == documents["dataset"]["panel"]["panel_file_sha256"]
            and entry["files"]["panel_receipt"]["sha256"]
            == documents["qualification"]["job_contract"]["panel_receipt_file_sha256"],
            "panel bytes do not match qualified published root")
    return documents


def source(S, source_root, record, note=None):
    return S.src("hf_file", record["url"], digest(inside(source_root, record["path"])), note)


def build(entry, source_root, collections, S):
    """Derive records using the registry's maintained constructors, not a catalog."""
    docs = read_entry(source_root, entry)
    d, rt, q, c = (docs[k] for k in ("dataset", "runtime", "qualification", "comparison"))
    cfg = docs["config"].get("text_config") or docs["config"]
    files = entry["files"]
    sources = [source(S, source_root, files[k]) for k in
               ("dataset", "capture", "runtime", "qualification", "comparison", "config", "license")]
    disclosure = [S.disc("record_note", "info", PURPOSE, sources=sources, provenance=True)]
    known_codes = set(load(Path(L.repo_root(__file__)) / "schema/invariants.json")["known_disclosure_codes"])
    for document in (d, c):
        for item in document.get("disclosures", []):
            require(item.get("severity") != "blocking", "receipt contains a blocking disclosure")
            code = item["code"] if item["code"] in known_codes and item["code"] != "no_known_deviations" else "record_note"
            disclosure.append(S.disc(
                code, item["severity"], "Receipt disclosure %s: %s" % (item["code"], item["detail"]),
                affects=item.get("affects_comparability", False), sources=sources, provenance=True))
    model_id = "model--" + slug(entry["model_repository"].replace("/", "."))
    aid = "artifact--" + slug(entry["model_repository"].replace("/", ".")) + "." + entry["model_revision"][:12]
    rid = "reference--fixture." + slug(entry["family"]) + "." + d["dataset_sha256"][:16]
    pid = "pipeline--fixture.cpu." + slug(entry["family"]) + "." + docs["harness"]["harness_id"].split("--", 1)[1]
    mid = "measurement--fixture." + slug(entry["family"]) + ".floor." + c["receipt_sha256"][:16]
    token = d["panel"]["tokenizer"]
    token_files = {f["name"]: f["sha256"] for f in token["files"]
                   if "tokenizer" in f["name"] or f["name"] in ("vocab.json", "merges.txt", "special_tokens_map.json")}
    require(token.get("files_verified") is True and token_files, "tokenizer bytes not verified")
    tokenizer = {"id": "fixture-tokenizer-" + L.sha256_hex(L.canonical_json(token_files))[:16],
                 "repository": token["repository"], "revision": token["revision"],
                 "vocab_size": token["vocab_size"]}
    panel = d["panel"]
    panel_key = L.sha256_hex(L.canonical_json({
        "tokens": panel["suite_token_hash_sha256"], "tokenizer": tokenizer["id"],
        "scoring_window": panel["scoring_window"]}))
    panel_id = "panel--fixture." + panel_key[:24]
    matches = [p for p in collections["panels"]
               if p["identity"]["panel_token_sha256"] == panel["suite_token_hash_sha256"]
               and p["tokenizer"]["id"] == tokenizer["id"]
               and p["structure"]["scoring_window"] == panel["scoring_window"]]
    if matches:
        require(len(matches) == 1, "ambiguous existing panel identity")
        existing = matches[0]
        require(existing["tokenizer"]["id"] == tokenizer["id"]
                and existing["structure"]["scoring_window"] == panel["scoring_window"],
                "shared token content has incompatible tokenizer/scoring window")
        panel_id = existing["id"]
        if model_id not in existing["model_scope"]:
            existing["model_scope"].append(model_id)
            existing["model_scope"].sort()
    else:
        collections["panels"].append({
            "schema_version": S.V, "id": panel_id,
            "name": "Synthetic CPU fixture panel " + panel["suite_token_hash_sha256"][:12],
            "author": S.MAL("panel-author"), "model_scope": [model_id], "tokenizer": tokenizer,
            "structure": {"contexts": panel["contexts"], "context_length": panel["context_length"],
                          "positions_per_context": c["measurement_scope"]["positions_per_context"],
                          "scored_positions_total": panel["scored_positions_total"],
                          "scoring_window": panel["scoring_window"]},
            "identity": {"hash_covers": "token_manifest", "panel_token_sha256": panel["suite_token_hash_sha256"],
                         "panel_receipt_sha256": files["panel_receipt"]["sha256"]},
            "corpus": {"public": True, "lineage": "Synthetic fixture panel; exact source and construction are in the sealed panel receipt.",
                       "sources": [source(S, source_root, files["panel_receipt"])]},
            "contamination": {k: panel["contamination"].get(k) for k in ("checked", "hits", "method")},
            "sealed": True, "availability": {"status": "public", "uri": url(entry["root_repository"], entry["root_revision"], "panel/panel.json")},
            "sources": [source(S, source_root, files["panel"]), source(S, source_root, files["panel_receipt"])],
            "disclosures": disclosure})
    model = {"schema_version": S.V, "id": model_id,
             "name": entry["model_repository"].split("/", 1)[1] + " (random test fixture)",
             "family": "fixture-" + slug(entry["family"]), "publisher": S.MAL("model-publisher"),
             "huggingface": S.hf(entry["model_repository"], entry["model_revision"], "reported_by_author"),
             "architecture": {"kind": cfg.get("model_type") or docs["config"]["model_type"],
                              "total_parameters": None,
                              "hidden_size": cfg.get("hidden_size"), "num_layers": cfg.get("num_hidden_layers"),
                              "vocab_size": cfg.get("vocab_size"),
                              "note": PURPOSE + " Serialized tensor elements: %d; unique parameter count is not asserted." % entry["tensor_elements"]},
             "tokenizer": dict(tokenizer, files_sha256=token_files),
             "canonical_weights": {"artifact_ref": aid, "precision": "bf16"},
             "license": d["dataset"]["license"], "sources": sources, "disclosures": disclosure}
    weights = {f["name"]: f for f in rt["weights"]["checkpoint_files"] if f["name"].endswith(".safetensors")}
    art = S.artifact(aid, model_id, model["name"] + " native BF16", "base", model["huggingface"],
                     "safetensors", "BF16", sum(f["size"] for f in weights.values()), S.codec("bf16", None),
                     S._g53_dataset_scope(d), S.MAL("model-publisher"), sources, disclosure,
                     weights_extra={"size_basis": "repo_weight_files", "shard_count": len(weights),
                                    "shard_sha256": {n: f["sha256"] for n, f in weights.items()},
                                    "config_sha256": d["weights"]["config_sha256"],
                                    "index_sha256": d["weights"].get("index_sha256")},
                     availability={"status": "public", "uri": url(entry["model_repository"], entry["model_revision"], dataset=False)})
    est = c["estimator"]
    dtype = {"float32": "fp32", "float64": "fp64", "bfloat16": "bf16", "float16": "fp16"}
    logits_dtype = dtype.get(est["logits_dtype"], est["logits_dtype"])
    ref = {"schema_version": S.V, "id": rid, "name": model["name"] + " CPU reference",
           "artifact_ref": aid, "panel_ref": panel_id, "reference_kind": "native_bf16",
           "capture": {"stack": rt["stack_fingerprint"]["engine"],
                       "stack_version": rt["stack_fingerprint"]["transformers_version"], "pipeline_ref": pid,
                       "compute_dtype": "bf16", "logits_dtype": logits_dtype,
                       "kv_cache_dtype": d["scope"]["kv_cache_dtype"], "head_source": "own_head",
                       "head_sha256": d["head"]["tensor_content_sha256"],
                       "capture_receipt_sha256": d["dataset_sha256"]},
           "author": S.MAL("measurer"), "logits_available": True,
           "self_consistency": {"floor_measurement_ref": mid,
                                "note": "Two independently verified cold CPU captures; forced self-comparison gives exact zero. Fixture-only floor, not trained-model quality."},
           "sources": sources, "disclosures": disclosure}
    h = docs["harness"]
    cpu = (c["comparator"].get("replay_env") or {}).get("cpu_model")
    require(cpu, "comparison must record the actual replay CPU model")
    lane_disclosure = S.disc("non_sealed_lane", "caveat",
        "Local CPU fixture lane; this exact same-lane reproduction is not a measured offset "
        "against a production sealed GPU lane.", affects=True, sources=sources, provenance=True)
    lane_bias = {"kind": "other", "direction": "unknown", "floor_measurement_ref": None,
                 "estimated_magnitude": None,
                 "detail": "This is the fixture's local CPU reproduction floor. Its offset against "
                           "a production GPU lane was not measured; no cross-lane equivalence is claimed."}
    pipeline = S.pipeline(pid, "Synthetic fixture capture/compare, CPU " + cpu,
                          ["capture", "replay", "scorer"], h["repository"]["url"], h["repository"]["commit"],
                          c["comparator"]["tool"]["entrypoint"], S.MAL("toolchain-author"), disclosure,
                          impl={"dependencies": {k: v for k, v in h["tool_versions"].items() if v is not None}},
                          hardware={"gpu": None, "gpu_count": 0,
                                    "note": "Local CPU capture (runtime device=cpu; no GPU). Replay CPU: " + cpu},
                          lane={"name": d["runtime"]["lane"]},
                          numerics={"accumulation_dtype": dtype.get(est["accumulation_dtype"], est["accumulation_dtype"]),
                                    "two_pass": est["two_pass"], "vocab_chunk": est["vocab_chunk"],
                                    "determinism_controls": ["cold_process_per_run"]}, sources=sources)
    row = S.measurement(mid, model_id, aid, panel_id, rid, pid, c["metric"]["value"],
                        artifacts_map={aid: art}, metric_name=c["metric"]["name"], direction=c["metric"]["direction"],
                        accumulation=est["accumulation_dtype"], stack_relation=est["stack_relation"],
                        head_policy=est["head_policy"], two_pass=est["two_pass"], vocab_chunk=est["vocab_chunk"],
                        logits_dtype=logits_dtype, top1=c["top1_agreement"],
                        scored_positions=c["measurement_scope"]["scored_positions"], contexts=c["measurement_scope"]["contexts"],
                        runs=2, cold=True, identical=True, evidence_kind="hidden_state_tensor_sha256",
                        evidence_hashes=[d["capture"]["capture_content_digest"]],
                        det_note="Qualification verifies two fresh processes; evidence is tensor content, not receipt-file hashes.",
                        sources=sources, receipt_schema=c["schema"], cls="advisory", bias=lane_bias,
                        disclosures=disclosure + [lane_disclosure, S.disc("reduced_run_count", "caveat", "Two qualified cold runs, not five.")])
    row["harness"] = h
    row["comparability"]["usable_as_floor"] = True
    row["measurement_scope"]["positions_per_context"] = c["measurement_scope"]["positions_per_context"]
    for name, record in (("models", model), ("artifacts", art), ("references", ref),
                         ("pipelines", pipeline), ("measurements", row)):
        same = [r for r in collections[name] if r["id"] == record["id"]]
        require(not same or same[0] == record, "conflicting existing %s %s" % (name, record["id"]))
        if not same:
            collections[name].append(record)


def _apply_variants(root, collections, S, roots):
    path = root / "variants.json"
    if not path.exists():
        return
    document = load(path)
    require(document.get("schema") == "qfs.community-format-artifacts.v1",
            "unsupported fixture variant sources")
    for item in document["variants"]:
        pin(item["model_repository"], item["model_revision"])
        parent = next((r for r in roots if r["model_repository"] == item["base_model_repository"]), None)
        require(parent is not None, "variant has no registered native fixture root")
        parent_docs = read_entry(root, parent)
        docs = {}
        sources = []
        for name, record in item["files"].items():
            file = inside(root, record["path"])
            require(digest(file) == record["sha256"], "changed variant evidence")
            docs[name] = load(file)
            sources.append(source(S, root, record))
        d, runtime, comparison = (docs[k] for k in ("dataset", "runtime", "comparison"))
        sealed(d, "dataset_sha256")
        sealed(runtime, "receipt_sha256")
        sealed(comparison, "receipt_sha256")
        require(comparison["candidate"]["dataset_sha256"] == d["dataset_sha256"]
                and comparison["candidate"]["weights"]["checkpoint_identity_sha256"]
                == item["checkpoint_identity_sha256"]
                == runtime["weights"]["checkpoint_identity_sha256"],
                "variant capture/weights identity mismatch")
        require(comparison["reference"]["weights"]["checkpoint_identity_sha256"]
                == parent_docs["dataset"]["weights"]["checkpoint_identity_sha256"],
                "variant was not compared against this native source")
        require(all(g.get("passed") is True and not g.get("overridden_by")
                    for g in comparison["gates"].values()), "variant comparison overrides a gate")
        require(comparison["comparability"]["class"] == "advisory",
                "reconstructed fixture must remain advisory")
        control = item["classification"] in ("exact-control", "near-control")
        require(control or (item["classification"] == "lossy"
                            and comparison["metric"]["value"] > 0),
                "lossy variant has no positive measured comparison")
        model_id = "model--" + slug(item["base_model_repository"].replace("/", "."))
        model = next(m for m in collections["models"] if m["id"] == model_id)
        aid = "artifact--" + slug(item["model_repository"].replace("/", ".")) + "." + item["model_revision"][:12]
        disclosures = [
            S.disc("record_note", "info", PURPOSE + " " + item["format"], sources=sources, provenance=True),
            S.disc("record_note", "info",
                   "RTN FORMAT fixture: optimizer-not-run. This indexes the actual child/control artifact "
                   "and its public capture/receipt, not a new canonical root or a trained-model quality row.",
                   sources=sources, provenance=True),
            S.disc("weights_reconstructed", "caveat",
                   "Stored weights reconstructed for native BF16 CPU execution; serving kernels and "
                   "activation quantization are not validated. Own-head evidence is linked.",
                   affects=True, sources=sources, provenance=True)]
        if control:
            disclosures.append(S.disc("record_note", "info",
                "Unquantized floating-point format control; base storage kind does not make it "
                "the model's canonical root. The canonical_weights pointer remains the native BF16 source.",
                sources=sources, provenance=True))
        projected_scope = S._g53_dataset_scope(d)
        numeric_formats = {"ct-mxfp4": "mxfp4", "ct-nvfp4": "nvfp4",
                           "modelopt-nvfp4": "nvfp4", "fp8-ue8m0": "fp8_e4m3",
                           "modelopt-fp8-block": "fp8_e4m3", "modelopt-fp8-tensor": "fp8_e4m3"}
        for assignment in projected_scope["assignments"]:
            original_format = assignment["format"]
            if original_format in numeric_formats:
                assignment["format"] = numeric_formats[original_format]
                assignment["note"] = (assignment.get("note") or "") + " Stored reader dialect: " + original_format + "."
        projected_scope["policy"] = S.derived_scope_policy(projected_scope["assignments"])
        record = S.artifact(
            aid, model_id, item["name"], "base" if control else "quant",
            S.hf(item["model_repository"], item["model_revision"], "reported_by_author"),
            item["container"], item["format"], sum(f["size"] for f in item["weight_files"]),
            S.codec(item["codec"], None if control else item["bits"],
                    tool=None if control else "QFS CPU RTN format fixture; optimizer-not-run"),
            projected_scope, S.MAL("model-publisher"), sources, disclosures,
            derived_from_artifact_ref=model["canonical_weights"]["artifact_ref"],
            weights_extra={"size_basis": "repo_weight_files", "shard_count": len(item["weight_files"]),
                           "shard_sha256": {f["name"]: f["sha256"] for f in item["weight_files"]},
                           "config_sha256": d["weights"]["config_sha256"]},
            availability={"status": "public", "uri": url(
                item["model_repository"], item["model_revision"], dataset=False)})
        old = [a for a in collections["artifacts"] if a["id"] == aid]
        require(not old or old[0] == record, "conflicting fixture artifact")
        if not old:
            collections["artifacts"].append(record)


def apply(collections_out, S, registry_root=None):
    """Authoritative reseed hook. No source file means the historical seed is unchanged."""
    root = Path(registry_root or L.repo_root(__file__)) / SOURCE_DIR
    manifest_path = root / "sources.json"
    if not manifest_path.exists():
        return collections_out
    manifest = load(manifest_path)
    require(manifest.get("schema") == SOURCE_SCHEMA, "unsupported maintained source manifest")
    collections = {name: copy.deepcopy(records) for name, records in collections_out}
    for entry in sorted(manifest["fixtures"], key=lambda e: e["model_repository"]):
        build(entry, root, collections, S)
    _apply_variants(root, collections, S, manifest["fixtures"])
    return [(name, collections[name]) for name, _ in collections_out]


def stage_entry(item, stage):
    for kind in ("model", "root", "evidence"):
        pin(item[kind + "_repository"], item[kind + "_revision"])
    require(item["model_repository"].startswith("malaiwah/")
            and "-tiny-random-bf16" in item["model_repository"],
            "this importer is only for the publisher's named random BF16 fixtures")
    root, model = Path(item["root_dir"]), Path(item["model_dir"])
    d = load(root / "fidelity-dataset.json")
    require(item.get("harness_path"), "%s requires harness_path recorded at comparison time" % item["family"])
    comparison = Path(item["comparison_path"])
    if comparison.resolve().is_relative_to(root.resolve()):
        comparison_url = url(item["root_repository"], item["root_revision"], comparison.resolve().relative_to(root.resolve()).as_posix())
    else:
        require(item.get("comparison_path_in_repo"), "external comparison needs comparison_path_in_repo in evidence repository")
        comparison_url = url(item["evidence_repository"], item["evidence_revision"], item["comparison_path_in_repo"])
    paths = {
        "dataset": (root / "fidelity-dataset.json", "fidelity-dataset.json"),
        "runtime": (inside(root, d["runtime"]["file"]), d["runtime"]["file"]),
        "capture": (inside(root, d["capture"]["manifest_file"]), d["capture"]["manifest_file"]),
        "panel": (inside(root, d["panel"]["panel_file"]), d["panel"]["panel_file"]),
        "panel_receipt": (inside(root, d["panel"]["panel_receipt_file"]), d["panel"]["panel_receipt_file"]),
        "qualification": (Path(item.get("qualification_path") or root / "receipts/root-qualification.json"), "receipts/root-qualification.json"),
        "config": (model / "config.json", "config.json"),
        "license": (model / "LICENSE", "LICENSE"),
        "comparison": (comparison, "comparison.json"),
        "harness": (Path(item["harness_path"]), "harness.json"),
    }
    entry = {k: item[k] for k in ("family", "model_repository", "model_revision", "root_repository", "root_revision", "evidence_repository", "evidence_revision")}
    entry["files"] = {}
    folder = slug(item["model_repository"].replace("/", ".")) + "." + item["root_revision"][:12]
    for key, (p, rel) in paths.items():
        target_rel = folder + "/" + ("LICENSE" if key == "license" else key + ".json")
        target = stage / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        public_url = (comparison_url if key == "comparison" else
                      url(item["model_repository"], item["model_revision"], rel, dataset=False)
                      if key in ("config", "license") else
                      url(item["root_repository"], item["root_revision"], rel))
        # Harness is locally retained measurement-time provenance, not a claimed
        # public root member; it is used as the row's harness, never a source URL.
        entry["files"][key] = {"path": target_rel, "sha256": digest(target),
                               "url": None if key == "harness" else public_url}
    docs = read_entry(stage, entry)
    source_license = docs["qualification"]["job_contract"].get("weights_license")
    if source_license is not None:
        require(entry["files"]["license"]["sha256"] == source_license["sha256"],
                "model LICENSE differs from the qualification's source license")
    total = 0
    shards = []
    for file in docs["runtime"]["weights"]["checkpoint_files"]:
        p = inside(model, file["name"])
        require(p.stat().st_size == file["size"] and digest(p) == file["sha256"],
                "local model file differs from captured checkpoint: " + file["name"])
        if p.suffix == ".safetensors":
            shards.append(file["name"])
            with open(p, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                require(0 < length <= 100 * 1024 * 1024, "invalid safetensors header length")
                header = json.loads(handle.read(length))
            for name, tensor in header.items():
                if name == "__metadata__":
                    continue
                require(tensor["dtype"] in ("BF16", "F32", "I64", "I32", "I16", "I8", "U8", "BOOL"),
                        "native fixture contains unsupported floating/packed storage: " + name)
                size = 1
                for dimension in tensor["shape"]:
                    size *= dimension
                total += size
    require(shards, "root has no native safetensors shards")
    entry["tensor_elements"] = total
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registry", default=L.repo_root(__file__))
    args = parser.parse_args()
    manifest = load(args.manifest)
    require(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("fixtures"), "unsupported/empty publication manifest")
    repos = [f["model_repository"] for f in manifest["fixtures"]]
    require(len(repos) == len(set(repos)), "duplicate model fixture")
    destination = Path(args.registry) / SOURCE_DIR
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".community-fixtures-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        if destination.exists():
            shutil.copytree(destination, stage, dirs_exist_ok=True)
        old = load(stage / "sources.json") if (stage / "sources.json").exists() else {"schema": SOURCE_SCHEMA, "fixtures": []}
        require(old["schema"] == SOURCE_SCHEMA, "unsupported existing maintained source manifest")
        entries = {e["model_repository"]: e for e in old["fixtures"]}
        for item in manifest["fixtures"]:
            entry = stage_entry(item, stage)
            require(entry["model_repository"] not in entries or entries[entry["model_repository"]] == entry,
                    "refusing to replace an existing fixture registration; retain its historical root")
            entries[entry["model_repository"]] = entry
        result = {"schema": SOURCE_SCHEMA, "fixtures": [entries[k] for k in sorted(entries)]}
        (stage / "sources.json").write_text(L.canonical_json(result) + "\n", encoding="utf-8")
        # No generated rows are silently edited here. The authoritative seed is
        # the only writer: importing these sources is followed by its normal CLI.
        destination.mkdir(exist_ok=True)
        for p in sorted(stage.rglob("*")):
            if p.is_file() and p.name != "sources.json":
                target = destination / p.relative_to(stage)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, target)
        os.replace(stage / "sources.json", destination / "sources.json")
    print("Imported %d qualified fixture sources; run registry/tools/seed_registry.py to write the real registry." % len(manifest["fixtures"]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
