#!/usr/bin/env python3
"""Bounded data-only review intake and authoritative accepted-source reseed hook.

This module is executed only from the trusted application checkout. Repository
snapshots and contributor files are data, never import roots or commands.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import subprocess

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent.parent
sys.path.insert(0, str(HERE))
import registry_lib as L

SOURCE_DIR = "protocol/review-requests"
SOURCE_SCHEMA = "qfs.accepted-registry-input.v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return L.canonical_json(value).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    raw = Path(path).read_bytes()
    require(len(raw) <= 1024 * 1024, "Review JSON exceeds 1 MiB: " + Path(path).name)
    # Maintained inputs are canonical, sealed JSON. Public receipts already pass
    # the controller's bounded duplicate-key/depth parser before this worker.
    value = L.parse_json(raw.decode("utf-8"))
    require(isinstance(value, dict), "Maintained review input must be a JSON object")
    canonical(value)  # Refuse overflowed/non-finite floats as well as NaN tokens.
    return value


def seal(doc, field):
    require(isinstance(doc, dict) and re.fullmatch(r"[0-9a-f]{64}", str(doc.get(field, ""))), "Missing seal: " + field)
    expected = dict(doc, **{field: ""})
    require(sha(canonical(expected)) == doc[field], "Invalid original evidence seal: " + field)


def safe_path(path):
    require(isinstance(path, str) and len(path) <= 240 and not path.startswith("/") and
            all(re.fullmatch(r"[A-Za-z0-9_.-]+", p) and p not in (".", "..") for p in path.split("/")), "Unsafe maintained evidence path")
    return path


def source_url(pub, role):
    return "https://huggingface.co/datasets/%s/resolve/%s/%s" % (pub["repository"], pub["revision"], pub["files"][role]["path"])


def merge(C, record):
    collection = L.collection_of_id(record["id"])
    old = C[collection].get(record["id"])
    require(old is None or old == record, "Conflicting existing record %s. Supply a distinct truthful identity; intake never overwrites existing rows." % record["id"])
    if old is None:
        C[collection][record["id"]] = record
        return True
    return False


def apply(collections_out, S=None, registry_root=None):
    """Preserve accepted, evidence-backed inputs on every authoritative reseed.

    Frozen accepted rows are maintained inputs, not estimator reruns. Their
    producer harness and original receipts remain exactly those reviewed.
    """
    root = Path(registry_root or L.repo_root(__file__))
    C = {name: {r["id"]: copy.deepcopy(r) for r in rows} for name, rows in collections_out}
    sources = root / SOURCE_DIR
    if not sources.exists():
        return collections_out
    for entry_dir in sorted(sources.iterdir()):
        if not entry_dir.is_dir():
            continue
        require(re.fullmatch(r"[0-9a-f]{64}", entry_dir.name), "Unexpected maintained review source directory")
        receipt = load(entry_dir / "acceptance.json")
        seal(receipt, "receipt_sha256")
        require(receipt.get("schema") == "qfs.registry-acceptance.v1" and receipt["request_sha256"] == entry_dir.name, "Invalid maintained acceptance receipt")
        entry = load(entry_dir / "input.json")
        require(entry.get("schema") == SOURCE_SCHEMA and sha((entry_dir / "input.json").read_bytes()) == receipt["input_file_sha256"], "Accepted input bytes changed")
        for path, digest in entry["evidence"].items():
            require(sha((entry_dir / safe_path(path)).read_bytes()) == digest, "Accepted original evidence changed: " + path)
        records = load(entry_dir / "records.json")
        require(sha((entry_dir / "records.json").read_bytes()) == entry["records_file_sha256"], "Accepted records changed")
        for record in records["records"]:
            merge(C, record)
    return [(name, list(C[name].values())) for name, _ in collections_out]


def check_workflow(docs, pub, author):
    result, plan, execution = (docs[k] for k in ("result", "plan", "execution"))
    require(result.get("schema") == "qfs.hf-workflow-result.v1" and plan.get("schema") == "qfs.hf-workflow-plan.v1" and execution.get("schema") == "qfs.hf-jobs-execution.v1", "Not an HF workflow result/plan/execution receipt")
    seal(result, "result_sha256")
    seal(plan, "plan_sha256")
    require(result.get("status") == "complete" and result.get("owner") == plan.get("owner") == author and
            result.get("workflow_id") == plan.get("workflow_id") and result.get("plan_sha256") == plan["plan_sha256"] == execution.get("plan_sha256"), "Result, plan, author and execution identity differ")
    require(execution.get("namespace") == author and execution.get("status") == "COMPLETED" and execution.get("flavor") == plan["hardware"]["flavor"] and
            execution.get("docker_image") == plan["image"] and execution.get("source_revision") == plan["source"]["revision"], "Provider receipt does not bind the completed requested run")
    require(result["source"]["revision"] == plan["source"]["revision"] and result["source"]["repository"] == plan["source"]["repository"] and re.fullmatch(r"[0-9a-f]{40}", result["source"]["revision"]), "Original worker source identity is not immutable")
    require(result.get("mode") == plan.get("mode") and (result["mode"] == "root") == (pub["kind"] == "root"), "Root requests cannot masquerade as measurement submissions")
    manifest = result.get("files")
    require(isinstance(manifest, list) and len(manifest) <= 4096, "Missing original worker file manifest")
    names = [f["path"] for f in manifest]
    require(len(names) == len(set(names)), "Duplicate worker manifest path")
    for name in names:
        safe_path(name)
    if pub["kind"] == "measurement":
        subpath = result["outputs"].get("submission")
        require(isinstance(subpath, str), "Producing Job did not emit a standard sealed submission; do not stamp owner checkout as estimator")
        found = [f for f in manifest if f["path"] == subpath]
        require(len(found) == 1 and found[0]["sha256"] == pub["files"]["submission"]["sha256"], "Submission is not the exact original Job-produced file")
        comparison_path = result["outputs"].get("comparison")
        found = [f for f in manifest if f["path"] == comparison_path]
        require(len(found) == 1 and found[0]["sha256"] == pub["files"]["comparison"]["sha256"],
                "Comparison is not the exact original Job-produced file")


def measurement_records(docs, pub, author, C, receipt_path, provider_verified):
    import _minischema
    import registry_add as add
    sub = docs["submission"]
    require(sub.get("submission_schema") == add.SUBMISSION_SCHEMA, "Use the producing Job's standard sealed submission receipt")
    schemas = _minischema.Registry(str(HERE.parent / "schema"))
    errors = schemas.validate(sub, "submission.schema.json")
    require(not errors, "Submission schema: " + "; ".join(str(e) for e in errors[:20]))
    add.verify_seal(sub)
    require(sub["measurer"].get("handle") == author, "Submission measurer must retain the actual producing caller identity")
    require(sub["produced_by"].get("revision") == docs["result"]["source"]["revision"], "Submission source revision differs from actual Job source")
    sys.path.insert(0, str(SUITE / "bin"))
    from fidelity import dscompare, dsvalidate
    comparison = docs["comparison"]
    report = dsvalidate.validate_receipt(comparison)
    require(not report.errors and comparison.get("comparison_kind") == "measurement",
            "Original candidate comparison failed receipt/schema/scientific checks")
    require(comparison["gates"] and all(g.get("passed") is True and not g.get("overridden_by") for g in comparison["gates"].values()),
            "Measurement has failed or overridden scientific gates")
    require(sub.get("evidence") == [dscompare._evidence_source(comparison[side], side) for side in ("reference", "candidate")],
            "Submission dataset-seal evidence differs from the original comparison")
    require(sub["metric"] == {k: comparison["metric"][k] for k in ("name", "value", "units", "direction")}
            and sub["estimator"] == dscompare._submission_estimator(comparison)
            and sub["reference"]["teacher_receipt_sha256"] == comparison["reference"]["dataset_sha256"],
            "Submission metric, estimator or teacher differs from the producing comparison")
    weights = comparison["candidate"]["weights"]
    require(sub["artifact"]["repository"] == weights["repository"] and sub["artifact"]["revision"] == weights["revision"],
            "Submission artifact does not identify the measured candidate weights")
    # Self-measured is permitted only for an original owner-produced Job, never
    # merely because the reviewer owns the registry. Independent verification is false.
    own_run = provider_verified and author == L.MAINTAINER
    # Legacy submission IDs name an author/tool or artifact/panel, not a run.
    # Build this submission's pipeline rather than silently borrowing an older
    # backend, then give new HF intake records content-bound identities.
    row, extra = add.submission_to_records(
        sub, receipt_path, pub["files"]["submission"]["sha256"],
        dict(C, pipelines={}), maintainer_attribution=own_run)
    legacy_pipeline = row["pipeline_ref"]
    pipeline = next(record for record in extra if record["id"] == legacy_pipeline)
    pipeline_identity = {key: pipeline[key]
                         for key in ("lane", "implementation", "numerics", "hardware", "author")}
    pipeline_identity["estimator"] = sub["estimator"]
    pipeline["id"] = "pipeline--hf-jobs." + sha(canonical(pipeline_identity))[:24]
    row["pipeline_ref"] = pipeline["id"]
    prior_pipeline = C["pipelines"].get(pipeline["id"])
    if prior_pipeline is not None:
        require(all(prior_pipeline.get(key) == pipeline[key]
                    for key in ("lane", "implementation", "numerics", "hardware", "author")),
                "Existing HF pipeline identity conflicts with the submitted implementation")
        extra.remove(pipeline)
    for record in extra:
        if record["id"] == row["artifact_ref"]:
            identity = {key: sub["artifact"].get(key)
                        for key in ("repository", "revision", "path", "scope_digest")}
            record["id"] = "artifact--hf-jobs." + sha(canonical(identity))[:24]
            row["artifact_ref"] = record["id"]
    row["id"] = "measurement--hf-jobs." + sub["receipt_sha256"][:24]
    require(weights.get("model_ref") == row["model_ref"],
            "Measured candidate model differs from the registered reference model")
    model = docs["plan"]["inputs"].get("model")
    if model:
        require(model["repository"] == weights["repository"] and model["revision"] == weights["revision"],
                "Planned model census differs from the measured candidate")
        shards = [f for f in model["files"] if f["path"].endswith((".safetensors", ".gguf"))]
        total = sum(f["bytes"] for f in shards)
        require(shards and total == model["weight_bytes"] == sub["artifact"]["size_bytes"],
                "Submitted weight size differs from the producing plan's exact weight-file census")
        for artifact in extra:
            if artifact["id"] == row["artifact_ref"]:
                artifact["weights"].update(size_basis="repo_weight_files", shard_count=len(shards),
                                           shard_sha256={f["path"]: f["sha256"] for f in shards},
                                           config_sha256=weights["config_sha256"], index_sha256=weights.get("index_sha256"))
                artifact["sources"].append({"kind": "hf_file", "uri": source_url(pub, "plan"),
                                            "sha256": pub["files"]["plan"]["sha256"],
                                            "note": "Original producing plan binds the exact serialized weight-file census and its size basis."})
    require(row["provenance"]["independently_verified"] is False, "Intake must not award independent verification")
    row["comparability"]["class"] = "advisory"
    if sub["lane"] != "sealed-ep8" and row["comparability"]["bias"] is None:
        row["comparability"]["bias"] = {
            "kind": "other", "direction": "unknown", "floor_measurement_ref": None,
            "estimated_magnitude": None,
            "detail": "The original comparison measures its own recorded lane. An offset against the registry production sealed lane was not measured; no cross-lane equivalence is claimed.",
        }
    # The standard receipt's evidence hashes identify DATASET SEALS, not file
    # bytes. Keep that original sealed receipt unchanged; cite its immutable
    # comparison file in the registry rather than laundering those seals into
    # file hashes or dereferencing a mutable /blob/main URL.
    row["provenance"]["sources"] = [
        {"kind": "receipt_file", "uri": receipt_path, "sha256": pub["files"]["submission"]["sha256"],
         "note": "Original sealed Job-produced submission, including original dataset-seal evidence pointers."},
        {"kind": "hf_file", "uri": source_url(pub, "comparison"), "sha256": pub["files"]["comparison"]["sha256"],
         "note": "Original comparison binds both measured dataset seals and the submitted metric; this digest covers file bytes."},
    ]
    # The admitted QFS own-head comparator scores every stored vocabulary column.
    # Record the same explicit policy used by native-root intake.
    row["estimator"]["vocab_masking_policy"] = "full_stored_vocab"
    run_count = row["determinism"].get("run_count")
    if (type(run_count) is int and 0 < run_count < 5
            and not any(L.has_disclosure(row, code) for code in ("reduced_run_count", "single_run"))):
        row["disclosures"].append({
            "code": "reduced_run_count", "severity": "caveat",
            "affects_comparability": False,
            "detail": "The original submission reports %d evaluated comparison run(s), not five. "
                      "Separate cold-capture reproduction is not additional independent evaluation text."
                      % run_count,
        })
    return extra + [row]


def attribution(value, role):
    require(isinstance(value, dict) and value.get("name") and set(value) <= {"name", "handle", "url"}, "Supply original " + role + " metadata as {name,handle,url}; do not infer authorship from repository hosting")
    return {"name": value["name"], "handle": value.get("handle"), "url": value.get("url"), "role": role, "is_registry_maintainer": False}


def root_records(docs, pub, author, C):
    import seed_registry as S
    import community_fixtures as F
    sys.path.insert(0, str(SUITE / "bin"))
    from fidelity import dsvalidate, resultsink, jobcontract
    d, rt, q, c, job = (docs[k] for k in ("dataset", "runtime", "qualification", "comparison", "job"))
    errors = dsvalidate.schema_errors(d, "fidelity-dataset.schema.json")
    require(not errors, "Canonical dataset schema failed: " + "; ".join(errors[:20]))
    for document, field in ((d, "dataset_sha256"), (rt, "receipt_sha256"), (q, "receipt_sha256"), (c, "receipt_sha256")):
        seal(document, field)
    jobcontract.verify_job(job)
    resultsink._validate_root_qualification_semantics(q)
    resultsink._validate_root_evidence(job, q, job_file_sha256=pub["files"]["job"]["sha256"])
    require(q["job_contract"].get("execution_kind") == "hf-jobs" and q["job_contract"].get("candidate") is None,
            "Root registration requires genuine HF Jobs native qualification, not a candidate or a local fixture receipt")
    hf = q.get("hf_execution") or {}
    require(hf.get("plan") == docs["plan"] and hf.get("provider_receipt") == docs["execution"] and hf.get("worker_result_sha256") == docs["result"]["result_sha256"], "Qualification does not bind original HF result/plan/provider evidence")
    require(d.get("schema") == "malaiwah.fidelity-dataset.v1" and d["dataset"]["role"] == "root" and d["weights"].get("quantized") is False and d["runtime"]["source"] == "native" and not rt["capture_tool"].get("weights_decode"), "Only unquantized native captures qualify as roots")
    require(d["dataset"]["author"]["name"] == author, "Preserve original capture author")
    require(d["coverage"]["complete"] is True and d["capture"]["dtype_lossless"] is True and d["capture"]["dtype"] == "BF16", "Native root must be complete lossless BF16")
    require(d["scope"]["policy"] == "native" and d["scope"]["head_policy"] == "native" and all(a["treatment"] in ("native", "not_present") for a in d["scope"]["assignments"]), "Root scope is not genuinely native")
    require(not docs["config"].get("quantization_config"), "Root config declares quantization")
    rep = dsvalidate.validate_receipt(c)
    require(not rep.errors, "Original reproduction comparison failed: " + str(rep.errors[:20]))
    require(c["comparison_kind"] == "reproduction_confirmation" and c["metric"]["value"] == c["kl"]["max"] == 0.0 and c["top1_agreement"] == 1.0,
            "Root needs a forced measured exact-zero two-cold reproduction, not a quantization measurement")
    require(all(q["reproduction_confirmation"].get(k) is True for k in ("two_fresh_processes", "distinct_dataset_roots", "both_independently_verified", "exact_zero_comparison", "canonical_dataset_only")), "Two-cold qualification incomplete")
    require(all(c["self_compare"].get(k) is True for k in ("capture_content_digest_equal", "weights_identity_equal", "asserted_exact_zero", "force_compute_agreed")), "Forced self-comparison evidence incomplete")
    require(c["comparability"]["same_lane"] is True and c["comparability"]["usable_as_floor"] is True and c["comparability"]["class"] == "strict" and not c["comparability"].get("bias"), "Reproduction is not scientifically admitted")
    require(c["gates"] and all(g.get("passed") is True and not g.get("overridden_by") for g in c["gates"].values()), "Failed or overridden scientific gates")
    first, repeat = q["captures"]["canonical"], q["captures"]["repeat"]
    require(first["process_label"] != repeat["process_label"] and first["dataset_sha256"] != repeat["dataset_sha256"], "Captures are not distinct cold processes")
    for identity, side in ((first, c["reference"]), (repeat, c["candidate"])):
        require(side["dataset_sha256"] == identity["dataset_sha256"] and side["capture_content_digest"] == identity["capture_content_digest"] == d["capture"]["capture_content_digest"] and side["weights"]["checkpoint_identity_sha256"] == d["weights"]["checkpoint_identity_sha256"], "Cold comparison identities differ from canonical native capture")
    for role, actual in (("runtime", d["runtime"]["file_sha256"]), ("capture", d["capture"]["manifest_file_sha256"]), ("comparison", q["comparison"]["file_sha256"]), ("config", d["weights"]["config_sha256"]), ("panel", d["panel"]["panel_file_sha256"]), ("panel_receipt", q["job_contract"]["panel_receipt_file_sha256"])):
        require(pub["files"][role]["sha256"] == actual, "Published " + role + " bytes do not match qualification")
    require(first["runtime_manifest_sha256"] == d["runtime"]["file_sha256"] and first["capture_manifest_sha256"] == d["capture"]["manifest_file_sha256"] and first["panel"]["suite_token_hash_sha256"] == d["panel"]["suite_token_hash_sha256"], "Published canonical descriptor differs from qualified capture")
    model_repo, model_rev = d["weights"]["repository"], d["weights"]["revision"]
    require(q["job_contract"]["weights_repository"] == model_repo and q["job_contract"]["weights_revision"] == model_rev, "Qualified native model pin differs")
    meta = pub.get("metadata") or {}
    needed = {"name", "family", "publisher", "panel_author", "toolchain_author", "corpus_lineage", "model_license", "root_repository", "root_revision"}
    require(needed <= meta.keys(), "Missing root metadata: " + ", ".join(sorted(needed - meta.keys())) + ". Supply actual upstream/panel facts; unknown authorship or license cannot be fabricated.")
    require(all(isinstance(meta[k], str) and meta[k].strip() for k in ("name", "family", "corpus_lineage", "model_license")), "Root name/family/corpus lineage/model license must be known nonempty metadata")
    F.pin(meta["root_repository"], meta["root_revision"])
    require(meta["root_repository"] == d["dataset"]["repository"] == q["dataset_repository"], "Canonical public root repository differs from qualification")
    root_url = F.url(meta["root_repository"], meta["root_revision"])
    publisher = attribution(meta["publisher"], "model-publisher")
    panel_author = attribution(meta["panel_author"], "panel-author")
    tool_author = attribution(meta["toolchain_author"], "toolchain-author")
    measurer = S.attr(author, "measurer", author, "https://huggingface.co/" + author, maintainer=author == L.MAINTAINER)
    sources = [S.src("hf_file", source_url(pub, role), f["sha256"]) for role, f in pub["files"].items()]
    disclosures = [S.disc("record_note", "info", "Native HF Jobs root. Receipt integrity and qualification were reviewed; the registry owner did not independently reproduce this run.", provenance=True, sources=sources)]
    known = set(load(HERE.parent / "schema/invariants.json")["known_disclosure_codes"])
    for doc in (d, c):
        for item in doc.get("disclosures", []):
            require(item.get("severity") != "blocking", "Root evidence has a blocking disclosure")
            disclosures.append(S.disc(item["code"] if item["code"] in known and item["code"] != "no_known_deviations" else "record_note", item["severity"], item["code"] + ": " + item["detail"], affects=item.get("affects_comparability", False), provenance=True, sources=sources))
    h = docs["harness"]
    F.check_harness(h)
    recorded = {(f["path"], f["sha256"]) for f in h["code_digests"]}
    require(rt.get("source_files") and all((p, v) in recorded for p, v in rt["source_files"].items()), "Harness does not retain captured code identities")
    require(h["repository"]["commit"] == docs["result"]["source"]["revision"] and h["repository"].get("commit_role") == "exact" and h["repository"].get("dirty") is False, "HF worker harness must retain its exact immutable source checkout")
    token = d["panel"]["tokenizer"]
    token_files = {f["name"]: f["sha256"] for f in token["files"] if "tokenizer" in f["name"] or f["name"] in ("vocab.json", "merges.txt", "special_tokens_map.json")}
    require(token.get("files_verified") is True and token_files, "Verified tokenizer identity is required")
    tokenizer = {"id": "tokenizer-" + sha(canonical(token_files))[:24], "repository": token["repository"], "revision": token["revision"], "vocab_size": token["vocab_size"]}
    model_slug = F.slug(model_repo.replace("/", "."))
    model_id = "model--" + model_slug + "." + model_rev[:12]
    aid = "artifact--" + model_slug + "." + model_rev[:12]
    panel = d["panel"]
    rid = "reference--native." + d["dataset_sha256"][:24]
    pid = "pipeline--hf-jobs." + h["harness_id"].split("--", 1)[1]
    mid = "measurement--native.floor." + c["receipt_sha256"][:24]
    records = []
    weights = {f["name"]: f for f in rt["weights"]["checkpoint_files"] if f["name"].endswith(".safetensors")}
    require(weights, "Native checkpoint safetensor census is missing")
    shard_hashes = {n: f["sha256"] for n, f in weights.items()}
    # Unknown legacy identities stay separate. A known immutable-pin conflict
    # is not a new version and must not be bypassed by choosing another ID.
    for registered in C["artifacts"].values():
        identity = registered["huggingface"]
        if identity["repository"] != model_repo or identity["revision"] != model_rev:
            continue
        known_weights = registered["weights"]
        for key in ("config_sha256", "index_sha256"):
            known_hash, observed_hash = known_weights.get(key), d["weights"].get(key)
            require(not known_hash or not observed_hash or known_hash == observed_hash,
                    "Conflicting native " + key + " at the same immutable model pin")
        require(all(name not in shard_hashes or digest == shard_hashes[name]
                    for name, digest in known_weights.get("shard_sha256", {}).items()),
                "Conflicting native checkpoint hash at the same immutable model pin")
    model_matches = []
    for registered in C["models"].values():
        identity = registered["huggingface"]
        if identity["repository"] != model_repo or identity["revision"] != model_rev:
            continue
        known_token = registered["tokenizer"]
        require(all(name not in token_files or digest == token_files[name]
                    for name, digest in known_token.get("files_sha256", {}).items()),
                "Conflicting tokenizer hash at the same immutable model pin")
        if (all(known_token.get(key) == tokenizer[key] for key in ("repository", "revision", "vocab_size"))
                and known_token.get("files_sha256") == token_files):
            canonical_art = C["artifacts"].get(registered["canonical_weights"]["artifact_ref"])
            require(canonical_art is not None, "Registered model canonical artifact is missing")
            if (canonical_art["model_ref"] == registered["id"]
                    and canonical_art["huggingface"]["repository"] == model_repo
                    and canonical_art["huggingface"]["revision"] == model_rev
                    and canonical_art["weights"].get("shard_sha256") == shard_hashes
                    and all(canonical_art["weights"].get(key) == d["weights"].get(key)
                            for key in ("config_sha256", "index_sha256"))):
                model_matches.append(registered)
    require(len(model_matches) <= 1, "Ambiguous registered native model identity")
    existing_model = model_matches[0] if model_matches else None
    if existing_model:
        model_id = existing_model["id"]
        tokenizer["id"] = existing_model["tokenizer"]["id"]
        aid = existing_model["canonical_weights"]["artifact_ref"]
    else:
        require(model_id not in C["models"], "Native model-version ID collision")
        require(aid not in C["artifacts"], "Native artifact-version ID collision")
    panel_id = "panel--native." + sha(canonical({"model": model_id, "tokenizer": token_files, "tokens": panel["suite_token_hash_sha256"], "scoring_window": panel["scoring_window"]}))[:24]
    matches = [p for p in C["panels"].values() if p["identity"]["panel_token_sha256"] == panel["suite_token_hash_sha256"] and p["tokenizer"]["id"] == tokenizer["id"] and p["structure"]["scoring_window"] == panel["scoring_window"] and model_id in p.get("model_scope", [])]
    require(len(matches) <= 1, "Ambiguous registered panel identity")
    if matches:
        panel_id = matches[0]["id"]
    else:
        require(panel_id not in C["panels"], "Native panel-version ID collision")
        records.append({"schema_version": S.V, "id": panel_id, "name": meta["name"] + " qualified token panel", "author": panel_author, "model_scope": [model_id], "tokenizer": tokenizer,
                        "structure": {"contexts": panel["contexts"], "context_length": panel["context_length"], "positions_per_context": c["measurement_scope"]["positions_per_context"], "scored_positions_total": panel["scored_positions_total"], "scoring_window": panel["scoring_window"]},
                        "identity": {"hash_covers": "token_manifest", "panel_token_sha256": panel["suite_token_hash_sha256"], "panel_receipt_sha256": pub["files"]["panel_receipt"]["sha256"]},
                        "corpus": {"public": True, "lineage": meta["corpus_lineage"], "sources": sources}, "contamination": {k: panel["contamination"].get(k) for k in ("checked", "hits", "method")},
                        "sealed": True, "availability": {"status": "public", "uri": source_url(pub, "panel")}, "sources": sources, "disclosures": disclosures})
    cfg = docs["config"].get("text_config") or docs["config"]
    require(cfg.get("model_type") or docs["config"].get("model_type"), "Native model architecture metadata is missing")
    model = {"schema_version": S.V, "id": model_id, "name": meta["name"], "family": meta["family"], "publisher": publisher,
             "huggingface": S.hf(model_repo, model_rev, "reported_by_author"), "architecture": {"kind": cfg.get("model_type") or docs["config"]["model_type"], "total_parameters": None, "hidden_size": cfg.get("hidden_size"), "num_layers": cfg.get("num_hidden_layers"), "vocab_size": cfg.get("vocab_size"), "note": "Unique parameter count is not established by a serialized tensor-file census."},
             "tokenizer": dict(tokenizer, files_sha256=token_files), "canonical_weights": {"artifact_ref": aid, "precision": "bf16"}, "license": meta["model_license"], "sources": sources, "disclosures": disclosures}
    art = S.artifact(aid, model_id, meta["name"] + " native BF16", "base", model["huggingface"], "safetensors", "BF16", sum(f["size"] for f in weights.values()), S.codec("bf16", None), S._g53_dataset_scope(d), publisher, sources, disclosures,
                     weights_extra={"size_basis": "repo_weight_files", "shard_count": len(weights), "shard_sha256": shard_hashes, "config_sha256": d["weights"]["config_sha256"], "index_sha256": d["weights"].get("index_sha256")},
                     availability={"status": "public", "uri": F.url(model_repo, model_rev, dataset=False)})
    if existing_model:
        registered = C["artifacts"][aid]
        require(registered["scope_digest"] == art["scope_digest"] and registered["huggingface"]["revision"] == model_rev,
                "Registered native artifact scope or revision differs")
        art = registered
    else:
        records.extend((model, art))
    est = c["estimator"]
    dtype = {"float32": "fp32", "float64": "fp64", "bfloat16": "bf16", "float16": "fp16"}
    logits = dtype.get(est["logits_dtype"], est["logits_dtype"])
    stack = rt["stack_fingerprint"]
    device = q["job_contract"]["device"]
    env = job["environment"]
    hardware = {"gpu": env.get("gpu"), "gpu_count": env.get("gpu_count"),
                "note": "HF Jobs flavor %s; capture device %s. Hardware identity is worker-reported, not independently reproduced." % (docs["execution"]["flavor"], device)}
    require(device == "cpu" or hardware["gpu"] and hardware["gpu_count"], "GPU model/count are absent from original replay evidence; supply original runtime metadata, not guessed hardware")
    records.append(S.pipeline(pid, "HF Jobs native capture and reproduction", ["capture", "replay", "scorer"], h["repository"]["url"], h["repository"]["commit"], c["comparator"]["tool"]["entrypoint"], tool_author, disclosures,
                              impl={"dependencies": {k: v for k, v in h["tool_versions"].items() if v is not None}}, hardware=hardware, lane={"name": d["runtime"]["lane"]},
                              numerics={"accumulation_dtype": dtype.get(est["accumulation_dtype"], est["accumulation_dtype"]), "two_pass": est["two_pass"], "vocab_chunk": est["vocab_chunk"], "determinism_controls": ["cold_process_per_run"]}, sources=sources))
    records.append({"schema_version": S.V, "id": rid, "name": meta["name"] + " native reference", "artifact_ref": aid, "panel_ref": panel_id, "reference_kind": "native_bf16",
                    "capture": {"stack": stack["engine"], "stack_version": stack["transformers_version"], "pipeline_ref": pid, "compute_dtype": "bf16", "logits_dtype": logits, "kv_cache_dtype": d["scope"]["kv_cache_dtype"], "head_source": "own_head", "head_sha256": d["head"]["tensor_content_sha256"], "capture_receipt_sha256": d["dataset_sha256"]},
                    "author": measurer, "logits_available": True, "self_consistency": {"floor_measurement_ref": mid, "note": "Two cold captures with forced exact-zero self-comparison; this is reproduction evidence, not model-quality evidence."},
                    "sources": sources + [S.src("dataset_card", root_url)], "disclosures": disclosures})
    ds = list(disclosures) + [S.disc("reduced_run_count", "caveat", "Two qualified cold runs, not five.")]
    if author != L.MAINTAINER:
        ds.append(S.disc("author_reported_only", "caveat", "Original contributor's Job; accepted receipt integrity is not independent reproduction."))
    bias = None
    if d["runtime"]["lane"] != "sealed-ep8":
        ds.append(S.disc("non_sealed_lane", "caveat", "This native reproduction is confined to its recorded HF Jobs lane; no production-lane equivalence is established.", affects=True))
        bias = {"kind": "other", "direction": "unknown", "floor_measurement_ref": None, "estimated_magnitude": None, "detail": "Exact same-lane native reproduction; its offset against other lanes was not measured."}
    row = S.measurement(mid, model_id, aid, panel_id, rid, pid, c["metric"]["value"], artifacts_map={aid: art}, metric_name=c["metric"]["name"], direction=c["metric"]["direction"], accumulation=est["accumulation_dtype"], stack_relation=est["stack_relation"], head_policy=est["head_policy"], two_pass=est["two_pass"], vocab_chunk=est["vocab_chunk"], logits_dtype=logits, top1=c["top1_agreement"],
                        scored_positions=c["measurement_scope"]["scored_positions"], contexts=c["measurement_scope"]["contexts"], runs=2, cold=True, identical=True, evidence_kind="hidden_state_tensor_sha256", evidence_hashes=[d["capture"]["capture_content_digest"]], sources=sources, receipt_schema=c["schema"], cls="advisory", bias=bias, disclosures=ds,
                        measured_by="self-measured" if author == L.MAINTAINER else "third-party-reported", measurer=measurer, verified=False)
    row["harness"] = h
    row["estimator"]["vocab_masking_policy"] = "full_stored_vocab"
    row["comparability"]["usable_as_floor"] = True
    row["measurement_scope"]["positions_per_context"] = c["measurement_scope"]["positions_per_context"]
    records.append(row)
    return records


def validate_snapshot(root):
    import registry_validate as V
    rep = V.Report()
    C = V.check_format_and_schema(str(root), rep, "mini")
    V.check_referential(C, rep)
    V.check_receipts_on_disk(str(root), C, rep)
    V.check_control_chars(C, rep)
    V.check_source_uris(C, rep)
    groups = V.check_comparability(C, rep)
    V.check_provenance(C, rep)
    V.check_determinism(C, rep)
    V.check_scope(C, rep)
    V.check_panels(C, rep)
    V.check_references(C, rep)
    V.check_stats_and_identity(C, rep)
    V.check_disclosures(C, rep, set(load(root / "schema/invariants.json")["known_disclosure_codes"]))
    V.check_harness(str(root), C, rep)
    V.check_provenance_assertions(C, rep)
    V.check_index(str(root), C, groups, rep)
    V.check_index_predicate(str(root), C, groups, rep)
    V.check_prose_keys(str(root), groups, rep)
    require(not rep.errors, "Full registry validation refused: " + json.dumps(rep.errors[:30]))
    joint = subprocess.run([sys.executable, "-I", "-B", str(HERE / "registry_joint_check.py"), "--root", str(root)],
                           capture_output=True, text=True, timeout=90, check=False)
    require(joint.returncode == 0, "Joint-standard validation refused: " + (joint.stdout + joint.stderr)[-12000:])
    return rep.warnings


def render(root, C):
    import registry_render as R
    groups = {}
    for mid, row in C["measurements"].items():
        groups.setdefault(row["comparability"]["key"], []).append(mid)
    head = (root / "README.head.md").read_text()
    require(head.count(R.BEGIN) == head.count(R.END) == 1, "README template must have one generated-table marker pair")
    pre, rest = head.split(R.BEGIN)
    _, post = rest.split(R.END)
    (root / "README.md").write_text(pre + R.BEGIN + "\n\n" + R.render(C, groups) + "\n" + R.END + post)
    index = R.build_index(C, groups)
    for name, _, _ in L.COLLECTIONS:
        index["collections"][name]["sha256"] = L.sha256_file(str(root / "data" / (name + ".jsonl")))
    (root / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def stage(directory):
    directory = Path(directory)
    root = directory / "registry"
    context = load(directory / "context.json")
    envelope = context["envelope"]
    pub = envelope["publication"]
    docs = {key: load(directory / "inputs" / (key + ".json")) for key in pub["files"]}
    check_workflow(docs, pub, envelope["requested_by"])
    provider_verified = context.get("provider_metadata_verified") is True
    require(provider_verified or envelope["requested_by"] != L.MAINTAINER,
            "This owner-authored external request lacks this registry service's provider attestation. Import the original Job into this workspace and revalidate it; existing provenance invariants forbid relabeling an unverified owner claim as someone else's measurement.")
    import registry_validate as V
    baseline_report = V.Report()
    C = V.check_format_and_schema(str(root), baseline_report, "mini")
    require(not baseline_report.errors, "Existing snapshot schema/format errors must be resolved before intake: " + json.dumps(baseline_report.errors[:20]))
    before = copy.deepcopy(C)
    digest = context["request_sha256"]
    require(sha(canonical(envelope)) == digest, "Reviewed request digest differs")
    dest = root / SOURCE_DIR / digest
    require(not dest.exists(), "This exact request is already accepted; inspect its immutable acceptance receipt")
    receipt_path = SOURCE_DIR + "/" + digest + "/submission.json"
    if pub["kind"] == "measurement":
        records = measurement_records(docs, pub, envelope["requested_by"], C, receipt_path, provider_verified)
    else:
        records = root_records(docs, pub, envelope["requested_by"], C)
    provider_notice = ("This service read authenticated HF provider metadata and validated recovered evidence; it did not independently reproduce the model run."
                       if provider_verified else
                       "Provider attestation not independently verified: external author-reported HF Jobs claim. Owner acceptance explicitly acknowledges this limitation; receipt integrity is not independent model reproduction.")
    for record in records:
        if record["id"].startswith("measurement--"):
            record["comparability"]["class"] = "advisory"
            record["disclosures"].append({"code": "record_note", "severity": "caveat",
                                           "detail": provider_notice, "affects_comparability": True})
    added = [record for record in records if merge(C, record)]
    require(added, "All proposed records already exist; no new acceptance is necessary")
    dest.mkdir(parents=True)
    evidence = {}
    for key, pointer in pub["files"].items():
        raw = (directory / "inputs" / (key + ".json")).read_bytes()
        require(sha(raw) == pointer["sha256"], "Staged original evidence changed")
        (dest / (key + ".json")).write_bytes(raw)
        evidence[key + ".json"] = sha(raw)
    (dest / "request.json").write_bytes(canonical(envelope) + b"\n")
    evidence["request.json"] = sha((dest / "request.json").read_bytes())
    (dest / "records.json").write_bytes(canonical({"records": added}) + b"\n")
    input_doc = {"schema": SOURCE_SCHEMA, "request_sha256": digest, "records_file_sha256": sha((dest / "records.json").read_bytes()), "evidence": evidence}
    (dest / "input.json").write_bytes(canonical(input_doc) + b"\n")
    acceptance = {"schema": "qfs.registry-acceptance.v1", "request_sha256": digest, "discussion_id": context["discussion_id"], "registry_repository": context["registry_repository"], "parent_commit": context["registry_head"], "accepted_by": context["reviewed_by"], "original_author": envelope["requested_by"], "input_file_sha256": sha((dest / "input.json").read_bytes()), "record_ids": [r["id"] for r in added], "independently_verified": False, "acceptance_rule": "Published only by explicit authenticated owner acceptance with parent-commit CAS; commit is the immutable acceptance time and identity.", "receipt_sha256": ""}
    acceptance["receipt_sha256"] = sha(canonical(acceptance))
    (dest / "acceptance.json").write_bytes(canonical(acceptance) + b"\n")
    for name, _, _ in L.COLLECTIONS:
        require(all(C[name].get(k) == v for k, v in before[name].items()), "Intake would alter or lose an existing registry row")
        L.write_jsonl(str(root / "data" / (name + ".jsonl")), list(C[name].values()))
    # Publish the maintained hook too: future reseeds must not erase accepted rows.
    seed = root / "tools/seed_registry.py"
    text = seed.read_text()
    marker = "    collections_out = community_fixtures.apply(collections_out, sys.modules[__name__])\n"
    hook = "    import review_requests\n    collections_out = review_requests.apply(collections_out, sys.modules[__name__])\n"
    if hook not in text:
        require(text.count(marker) == 1, "Registry seeder lacks the maintained source-hook boundary; migrate it before intake")
        seed.write_text(text.replace(marker, marker + hook, 1))
    (root / "tools/review_requests.py").write_bytes(Path(__file__).read_bytes())
    render(root, C)
    warnings = validate_snapshot(root)
    warnings.append({"check": "HF_PROVIDER_ATTESTATION", "message": provider_notice})
    # Warnings are retained beside the immutable acceptance, not laundered into
    # a clean-release badge or hidden merely because errors are absent.
    (dest / "validation.json").write_bytes(canonical({"errors": [], "warnings": warnings, "scope": "all existing registry schema and invariant checks; no independent model execution"}) + b"\n")
    original = [(name, list(rows.values())) for name, rows in before.items()]
    restored = {name: {r["id"]: r for r in rows} for name, rows in apply(original, registry_root=root)}
    require(canonical(restored) == canonical(C),
            "Maintained accepted inputs do not reconstruct the staged registry without loss")
    (directory / "preview.json").write_bytes(canonical({"kind": pub["kind"], "record_ids": [r["id"] for r in added], "records": added, "original_author": envelope["requested_by"], "warnings": warnings, "provider_metadata_verified": provider_verified, "owner_acknowledgement": provider_notice, "independently_verified": False, "notice": "All registry schema/invariant checks passed in an isolated snapshot. Warnings remain; no independently verified or clean-release status is awarded."}) + b"\n")


if __name__ == "__main__":
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 ** 2, 64 * 1024 ** 2))
        require(len(sys.argv) == 3 and sys.argv[1] == "stage", "Use the authenticated Explorer review controller")
        stage(sys.argv[2])
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc))
        raise SystemExit(2)
