"""Caller-funded HF Jobs with pinned inputs, provider deadlines and tokenless volumes.

Preparing is read-only. Launch is a separately confirmed mutation. The application
never falls back to an owner token, sends no token to model code, and never equates
COMPLETED with verified/published scientific evidence.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from functools import lru_cache
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone

from .auth import Actor, AuthError
from .data import ExplorerRegistry

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = os.environ.get("QFS_REGISTRY_REPOSITORY", "malaiwah/quant-fidelity-registry")
SOURCE = "https://github.com/malaiwah/quant-fidelity-suite"
SPACE = os.environ.get("SPACE_ID", "local-qfs-explorer")
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")
UUID = re.compile(r"[0-9a-f]{32}\Z")
JOB_ID = re.compile(r"[A-Za-z0-9_-]{8,100}\Z")
MAX_JSON = 16 * 1024 * 1024
MAX_FILES = 20000
MAX_OUTPUT = int(os.environ.get("QFS_MAX_OUTPUT_BYTES", str(4 * 1024**3)))
CACHE = Path(os.environ.get("QFS_JOB_CACHE", tempfile.mkdtemp(prefix="qfs-jobs-")))
CACHE.mkdir(parents=True, exist_ok=True, mode=0o700)
_LOCAL_SIGNING = secrets.token_bytes(32)
_LOCK = threading.RLock()
_PREPARE_SLOTS = threading.BoundedSemaphore(2)
_TERMINAL = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}
_LEDGER_SCHEMA = "qfs.hf-job-ledger.v1"


class JobsError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def seal(value, field):
    value = json.loads(json.dumps(value, allow_nan=False))
    value[field] = ""
    value[field] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def verify_seal(value, field):
    if not isinstance(value, dict) or not HEX.fullmatch(str(value.get(field, ""))):
        raise JobsError("Missing exact document seal: " + field)
    if seal(value, field)[field] != value[field]:
        raise JobsError("Document bytes do not match their seal: " + field)


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    return value


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise JobsError("Expected a nonempty relative artifact path.")
    p = PurePosixPath(value)
    if p.is_absolute() or p.as_posix() != value or any(x in ("", ".", "..") for x in p.parts):
        raise JobsError("Unsafe artifact path.")
    return value


def _identity(repo, revision):
    if not REPO.fullmatch(repo or "") or not SHA.fullmatch(revision or ""):
        raise JobsError("Use owner/repository and an immutable lowercase 40-character commit.")
    return repo, revision


def _api_error(exc, operation):
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return JobsError(operation + " was refused by HF. Check your own Jobs/repository permissions and sign-in scopes.")
    if status == 402:
        return JobsError("HF requires a positive compute-credit balance. No fallback billing account was used.")
    if status == 429:
        return JobsError("HF rate-limited this operation. Retry reading its status later; an ambiguous launch is never resubmitted automatically.")
    return JobsError(operation + " did not complete (" + type(exc).__name__ + "). Check the recorded workflow state; no alternate account or job was substituted.")


def _json_download(actor, repo, revision, name, *, repo_type="model", limit=MAX_JSON, save_to=None):
    from huggingface_hub import hf_hub_download
    _identity(repo, revision); _relative(name)
    infos = actor.client().get_paths_info(repo, [name], repo_type=repo_type, revision=revision)
    if len(infos) != 1 or type(getattr(infos[0], "size", None)) is not int or not 0 <= infos[0].size <= limit:
        raise JobsError("Metadata is missing or exceeds this workspace's safe read limit.")
    with tempfile.TemporaryDirectory(prefix="qfs-metadata-") as td:
        path = Path(hf_hub_download(repo, name, repo_type=repo_type, revision=revision,
                                   token=actor.client().token, cache_dir=td)).resolve()
        if path.stat().st_size != infos[0].size:
            raise JobsError("Metadata bytes differ from the pinned Hub tree.")
        raw = path.read_bytes()
        value = _read_json(path, limit=limit)
        if save_to is not None:
            Path(save_to).write_bytes(raw)
    return value, hashlib.sha256(raw).hexdigest(), len(raw)


def _source_identity():
    environment = json.loads((ROOT / "explorer/job_environment.json").read_text())
    deployment = ROOT / "explorer/deployment.json"
    if deployment.exists():
        data = json.loads(deployment.read_text())
        revision = data.get("source_revision")
    else:
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.SubprocessError):
            raise JobsError("The Space owner must deploy a pinned QFS worker source before enabling Jobs.") from None
    if not SHA.fullmatch(revision or ""):
        raise JobsError("No immutable worker source revision is configured.")
    files = {}
    for name in ("job_worker.py", "job_bootstrap.py"):
        path = ROOT / "explorer" / name
        if not path.is_file():
            raise JobsError("The reviewed Jobs worker is not deployed.")
        key = "worker_sha256" if name == "job_worker.py" else "bootstrap_sha256"
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if deployment.exists():
            if data.get(key) != sha:
                raise JobsError("Deployed worker bytes disagree with the source manifest.")
        else:
            try:
                committed = subprocess.check_output(["git", "show", revision + ":explorer/" + name], cwd=ROOT)
            except subprocess.SubprocessError:
                raise JobsError("Commit and publish the worker before running a paid Job.") from None
            if hashlib.sha256(committed).hexdigest() != sha:
                raise JobsError("Worker source is modified; commit it before planning a Job.")
        files[key] = sha
    image = environment.get("image", "")
    if not re.fullmatch(r"[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}", image):
        raise JobsError("Worker image must be digest-pinned.")
    return {"repository": SOURCE, "revision": revision, **files}, image


def presets():
    catalog = json.loads((ROOT / "engines/coverage.json").read_text())
    roots = {r["model_repository"]: r for r in catalog.get("root_datasets", [])}
    entries = []
    central = {"repository": "malaiwah/qfs-fixture-root-captures-v1", "revision": "f53b204091c988ce4a2161af81886f3018745559"}
    architectures = {e["fixture"]["repository"]: e for e in catalog["architectures"]}
    for repo, row in roots.items():
        entry = architectures.get(repo, {})
        entries.append({"id": "root:" + row["family"], "label": "Native root · " + repo.split("/")[1],
                        "mode": "root", "model_repository": repo, "model_revision": row["model_revision"],
                        "panel": {**central, "path": "roots/" + row["family"] + "/input-panel", "role": "final"},
                        "trusted_code": entry.get("code_pin"), "recommended_flavor": "cpu-basic",
                        "scope_note": "Complete tiny random architecture fixture; synthetic panel, not trained-model quality."})
    for q in catalog.get("quantizers", []):
        parent = roots.get((q.get("source") or {}).get("repository"))
        if parent is None:
            continue
        entries.append({"id": "candidate:" + q["id"], "label": "Quant/format · " + q["fixture"]["repository"].split("/")[1],
                        "mode": "candidate", "model_repository": q["fixture"]["repository"], "model_revision": q["fixture"]["revision"],
                        "reference_repository": parent["repository"], "reference_revision": parent["revision"],
                        "panel": {**central, "path": "roots/" + parent["family"] + "/input-panel", "role": "final"},
                        "recorded_comparison": q["evidence"],
                        "recommended_flavor": "cpu-basic", "scope_note": q["classification"] + "; RTN format test, optimizer-not-run."})
    entries.append({"id": "root:fruit", "label": "Fruit BF16 · heavier trained CI proxy (10.1 GB)", "mode": "root",
                    "model_repository": "malaiwah/GLM-5.2-SIQ-Fruit-bf16", "model_revision": "ef68013aa6e16453cf52b5b77647f72fbe258c3c",
                    "panel": {"kind": "bundled", "path": "engines/panels/panel--fruit.malaiwah.heldout-v1", "role": "final"},
                    "unexpected_allowlist": "engines/tools/layer-outer-evidence/fruit-unexpected-keys.json",
                    "recommended_flavor": "cpu-upgrade", "scope_note": "Heavier trained Fruit proxy; declared indexer/MTP omissions retained. Not an assistant or upstream-model quality claim."})
    return entries


def hardware(actor):
    try:
        rows = actor.client().list_jobs_hardware()
    except Exception as exc:
        raise _api_error(exc, "Hardware lookup") from None
    out = []
    for item in rows:
        row = _plain(item)
        unit = row.get("unit_label")
        price = row.get("unit_cost_micro_usd")
        if unit != "minute" or type(price) is not int or price <= 0:
            continue
        row["hourly_usd"] = str(Decimal(price) * 60 / 1000000)
        row["device"] = "cuda" if row.get("accelerator") else "cpu"
        out.append(row)
    return out


def _model_metadata(actor, repo, revision, *, mode):
    from huggingface_hub import hf_hub_download
    _identity(repo, revision)
    try:
        info = actor.client().model_info(repo, revision=revision, files_metadata=True)
    except Exception as exc:
        raise _api_error(exc, "Model metadata lookup") from None
    if info.sha != revision:
        raise JobsError("The model did not resolve to the requested immutable revision.")
    config, config_sha, config_bytes = _json_download(actor, repo, revision, "config.json")
    if not isinstance(config, dict):
        raise JobsError("Model configuration must be an object.")
    quant = config.get("quantization_config") or config.get("quantization") or (config.get("text_config") or {}).get("quantization_config")
    if mode == "root" and quant:
        raise JobsError("A quantized checkpoint cannot be submitted as a native root. Choose candidate capture.")
    files = []
    for sibling in info.siblings or []:
        name = sibling.rfilename
        _relative(name)
        size = getattr(sibling, "size", None)
        if type(size) is not int or size < 0:
            raise JobsError("The model has an unrecorded file size; resolve its immutable metadata before spending.")
        lfs = getattr(sibling, "lfs", None)
        checksum = getattr(lfs, "sha256", None) if lfs is not None else None
        files.append({"path": name, "bytes": size, "sha256": checksum})
    weights = [f for f in files if f["path"].endswith((".safetensors", ".gguf"))]
    if not weights:
        raise JobsError("This Jobs path requires safetensors or supported GGUF storage; pickle checkpoints are not executed.")
    if mode == "root" and any(f["path"].endswith(".gguf") for f in weights):
        raise JobsError("Native roots require an unquantized safetensors checkpoint, not GGUF.")
    if any(f["path"].endswith(".gguf") for f in weights) and any(f["path"].endswith(".safetensors") for f in weights):
        raise JobsError("This repository contains multiple weight representations. Select a single prepared artifact with its own config/scope.")
    if any("/" in f["path"] for f in weights):
        raise JobsError("Nested/multiple builds need a single prepared checkpoint directory before this one-click path can admit them.")
    for row in weights:
        if not HEX.fullmatch(row.get("sha256") or ""):
            if row["bytes"] > 4 * 1024**2:
                raise JobsError("Weight metadata has no cryptographic content digest; no large weights are downloaded during planning.")
            row["sha256"] = _metadata_digest(actor, repo, revision, row)
    index_sha = index_bytes = None
    metadata_bytes = 0
    for row in files:
        if not HEX.fullmatch(row.get("sha256") or ""):
            metadata_bytes += row["bytes"]
            if row["bytes"] > MAX_JSON or metadata_bytes > 32 * 1024**2:
                raise JobsError("Unhashed non-weight metadata exceeds the planning limit.")
            row["sha256"] = _metadata_digest(actor, repo, revision, row)
    if any(f["path"] == "model.safetensors.index.json" for f in files):
        _, index_sha, index_bytes = _json_download(actor, repo, revision, "model.safetensors.index.json")
    license_names = [name for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE-MODEL", "LICENSE-MODEL.txt")
                     if any(f["path"] == name for f in files)]
    if not license_names:
        raise JobsError("A source license file is required before this app can capture a redistributable head.")
    card = _plain(getattr(info, "card_data", None)) or {}
    if not isinstance(card, dict) and hasattr(info.card_data, "to_dict"):
        card = info.card_data.to_dict()
    return {"repository": repo, "revision": revision, "mount_path": "/inputs/model", "config": config,
            "config_sha256": config_sha, "config_bytes": config_bytes, "index_sha256": index_sha,
            "index_bytes": index_bytes, "files": files, "weight_bytes": sum(f["bytes"] for f in weights),
            "license_file": license_names[0], "license": card.get("license") if isinstance(card, dict) else None,
            "publisher": getattr(info, "author", repo.split("/")[0]), "model_type": config.get("model_type")}


def _dataset_metadata(actor, repo, revision, mount_path):
    _identity(repo, revision)
    try:
        info = actor.client().dataset_info(repo, revision=revision)
    except Exception as exc:
        raise _api_error(exc, "Dataset lookup") from None
    if info.sha != revision:
        raise JobsError("Dataset revision does not match.")
    descriptor, sha, size = _json_download(actor, repo, revision, "fidelity-dataset.json", repo_type="dataset")
    from fidelity import dsformat as F
    if descriptor.get("schema") != F.DATASET_SCHEMA or not F.verify_manifest_seal(descriptor):
        raise JobsError("Reference/candidate is not an intact sealed fidelity dataset.")
    return {"repository": repo, "revision": revision, "mount_path": mount_path,
            "dataset_sha256": descriptor["dataset_sha256"], "manifest_sha256": sha,
            "manifest_bytes": size, "descriptor": descriptor}


def _signing_key():
    secret = os.environ.get("QFS_WORKFLOW_SIGNING_KEY") or os.environ.get("OAUTH_CLIENT_SECRET")
    return hashlib.sha256(("qfs-launch-ticket-v1:" + secret).encode()).digest() if secret else _LOCAL_SIGNING


def _ticket(plan):
    message = {"owner": plan["owner"], "plan_sha256": plan["plan_sha256"], "expires_at": int(time.time()) + 600}
    message["signature"] = hmac.new(_signing_key(), canonical(message), hashlib.sha256).hexdigest()
    return message


def _check_ticket(actor, plan, ticket):
    if not isinstance(ticket, dict):
        raise JobsError("Prepare and review the job before launching it.")
    body = {k: ticket.get(k) for k in ("owner", "plan_sha256", "expires_at")}
    signature = hmac.new(_signing_key(), canonical(body), hashlib.sha256).hexdigest()
    if (not hmac.compare_digest(signature, str(ticket.get("signature", ""))) or body["owner"] != actor.username
            or body["plan_sha256"] != plan.get("plan_sha256") or type(body["expires_at"]) is not int
            or body["expires_at"] < time.time()):
        raise JobsError("This reviewed plan expired or changed. Prepare it again; nothing was launched.")


def _registered(registry, reference, model, scope, codec, bits, actor):
    if reference is None:
        return None
    data = registry.registry_data()
    descriptor = reference["descriptor"]
    candidates = [r for r in data["references"].values()
                  if (r.get("capture") or {}).get("capture_receipt_sha256") == descriptor["dataset_sha256"]]
    if len(candidates) != 1:
        return None
    ref = candidates[0];panel = data["panels"][ref["panel_ref"]]
    return {"registry_repository": REGISTRY, "registry_revision": registry.overview().get("revision"),
            "model_ref": data["artifacts"][ref["artifact_ref"]]["model_ref"], "panel_ref": ref["panel_ref"],
            "reference_ref": ref["id"], "panel": {"panel_ref": ref["panel_ref"],
                "panel_token_sha256": panel["identity"]["panel_token_sha256"]},
            "reference": {"reference_ref": ref["id"], "teacher_receipt_sha256": descriptor["dataset_sha256"]},
            "artifact": {"repository": model["repository"], "revision": model["revision"],
                "container": "gguf" if any(f["path"].endswith(".gguf") for f in model["files"]) else "safetensors",
                "size_bytes": model["weight_bytes"], "codec": {"family": codec, "bits_per_weight_nominal": bits},
                "scope": scope, "producer": {"name": model["publisher"], "handle": model["publisher"],
                    "role": "model-publisher", "url": "https://huggingface.co/" + model["publisher"],
                    "is_registry_maintainer": False}}}


def _prepare(actor, spec, registry=None):
    if not isinstance(actor, Actor) or not isinstance(spec, dict):
        raise JobsError("An authenticated caller and workflow inputs are required.")
    allowed = {"preset", "mode", "model_repository", "model_revision", "reference_repository", "reference_revision",
               "candidate_repository", "candidate_revision", "panel_repository", "panel_revision", "panel_path",
               "scope_json", "codec", "declared_bits", "flavor", "timeout_seconds", "max_compute_usd", "output_repository",
               "review_metadata"}
    if set(spec) - allowed:
        raise JobsError("Unknown workflow input field.")
    if "review_metadata" in spec and not isinstance(spec["review_metadata"], dict):
        raise JobsError("Review metadata must be a JSON object containing original attribution and lineage facts.")
    preset = next((p for p in presets() if p["id"] == spec.get("preset")), None)
    if spec.get("preset") and preset is None:
        raise JobsError("Choose a listed preset or custom workflow.")
    request = {**(preset or {}), **{k: v for k, v in spec.items() if v not in (None, "")}}
    mode = request.get("mode", "root")
    if mode not in ("root", "candidate", "compare"):
        raise JobsError("Choose root capture, candidate measurement, or existing-dataset comparison.")
    source, image = _source_identity()
    timeout = request.get("timeout_seconds", 600)
    if type(timeout) is not int or not 60 <= timeout <= 7200:
        raise JobsError("Job deadline must be 60–7200 seconds.")
    try:
        ceiling = Decimal(str(request.get("max_compute_usd", "0.25")))
    except InvalidOperation:
        raise JobsError("Enter a finite positive maximum compute estimate.") from None
    if not ceiling.is_finite() or not 0 < ceiling <= Decimal(os.environ.get("QFS_MAX_RUN_ESTIMATE_USD", "25")):
        raise JobsError("The compute estimate ceiling is outside this workspace's supported range.")
    flavors = hardware(actor)
    flavor = request.get("flavor") or request.get("recommended_flavor", "cpu-basic")
    hw = next((h for h in flavors if h["name"] == flavor), None)
    if hw is None:
        raise JobsError("The selected hardware is not currently offered by HF Jobs.")
    minutes = (Decimal(timeout) / 60).to_integral_value(rounding=ROUND_CEILING) + 2
    estimate = Decimal(hw["unit_cost_micro_usd"]) * minutes / 1000000
    if estimate > ceiling:
        raise JobsError("This hardware/deadline estimates $%s including two startup minutes, above your $%s ceiling." % (estimate, ceiling))
    actor.require_lifetime(timeout + 120)
    model = None if mode == "compare" else _model_metadata(actor, request.get("model_repository"), request.get("model_revision"), mode=mode)
    reference = None if mode == "root" else _dataset_metadata(actor, request.get("reference_repository"), request.get("reference_revision"), "/inputs/reference")
    candidate = _dataset_metadata(actor, request.get("candidate_repository"), request.get("candidate_revision"), "/inputs/candidate") if mode == "compare" else None
    panel = None
    if mode != "compare":
        panel = dict((preset or {}).get("panel") or {"repository": request.get("panel_repository"), "revision": request.get("panel_revision"), "path": request.get("panel_path"), "role": "final"})
        _relative(panel.get("path"))
        if panel.get("kind") == "bundled":
            source_path = ROOT / panel["path"]
            if not source_path.is_dir():
                raise JobsError("The preset's original token panel is not included in this deployment.")
            panel.update(repository="malaiwah/quant-fidelity-suite", revision=source["revision"], mount_path="/inputs/panel")
        else:
            _identity(panel.get("repository"), panel.get("revision"))
            doc, _, _ = _json_download(actor, panel["repository"], panel["revision"], panel["path"] + "/panel.json", repo_type="dataset")
            if doc.get("schema") != "quant-pipeline.glm53-token-panel.v1":
                raise JobsError("Select the original token-panel tree, not a sealed capture's internal panel directory.")
            panel["mount_path"] = "/inputs/panel"
    tokenizer = None
    if mode == "candidate":
        weights_identity = reference["descriptor"]["weights"]
        tokenizer = _model_metadata(actor, weights_identity["repository"], weights_identity["revision"], mode="root")
        tokenizer["mount_path"] = "/inputs/tokenizer"
    scope = codec = bits = None
    recorded_comparison = None
    if (preset or {}).get("recorded_comparison"):
        recorded = preset["recorded_comparison"]
        recorded_comparison, _, _ = _json_download(actor, recorded["repository"], recorded["revision"], recorded["comparison_path"], repo_type="dataset")
    if mode == "candidate":
        if request.get("scope_json"):
            try: scope = json.loads(request["scope_json"])
            except (TypeError, ValueError): raise JobsError("Scope must be a JSON object.") from None
        else:
            scope, _, _ = _json_download(actor, model["repository"], model["revision"], "scope.json")
        if not isinstance(scope, dict) or not isinstance(scope.get("assignments"), list) or not scope["assignments"]:
            raise JobsError("Candidate capture requires an explicit nonempty intervention scope.")
        config_quant = model["config"].get("quantization_config") or model["config"].get("quantization") or {}
        codec = request.get("codec") or ("gguf-k-quant" if any(f["path"].endswith(".gguf") for f in model["files"]) else "mixed" if config_quant.get("quant_algo") == "MIXED_PRECISION" else "mxfp4" if "mxfp4" in str(config_quant).lower() else "nvfp4" if "nvfp4" in str(config_quant).lower() else "fp8_e4m3" if config_quant.get("quant_method") == "fp8" else "int4")
        bits = request.get("declared_bits")
        if recorded_comparison and not request.get("codec"):
            codec = recorded_comparison["candidate"]["weights"]["codec"]
        if recorded_comparison and bits in (None, ""):
            bits = recorded_comparison["candidate"]["weights"]["declared_bits"]
        if bits in (None, ""):
            observed = {a.get("bits_per_weight") for a in scope["assignments"] if a.get("treatment") == "quantized" and a.get("bits_per_weight") is not None}
            bits = next(iter(observed)) if len(observed) == 1 else config_quant.get("bits", config_quant.get("num_bits"))
        if isinstance(bits, bool) or not isinstance(bits, (int, float)) or not math.isfinite(bits) or not 0 < bits <= 64:
            raise JobsError("Specify the actual nominal bits for this candidate; mixed scopes retain their per-class precision.")
    trusted_code = (preset or {}).get("trusted_code")
    if model and not preset and model["config"].get("auto_map"):
        # Native AutoConfig/AutoModel dispatch may ignore auto_map, but arbitrary Python must never run here.
        known = next((p for p in presets() if p.get("model_repository") == model["repository"] and p.get("model_revision") == model["revision"]), None)
        trusted_code = (known or {}).get("trusted_code")
    allowlist = None
    if (preset or {}).get("unexpected_allowlist"):
        path = preset["unexpected_allowlist"];raw = (ROOT / path).read_bytes();doc = json.loads(raw)
        names = doc if isinstance(doc, list) else doc.get("names", doc.get("unexpected_keys", doc.get("keys")))
        if not isinstance(names, list):
            raise JobsError("The preset allowlist has no exact tensor-name census.")
        allowlist = {"path": path, "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                     "canonical_sorted_names_sha256": hashlib.sha256(canonical(sorted(names))).hexdigest()}
    if model:
        disk = int(re.search(r"\d+", hw["ephemeral_storage"])[0]) * 10**9
        if model["weight_bytes"] * 2 + MAX_OUTPUT + 4 * 1024**3 > disk:
            raise JobsError("Model, bounded outputs and installation margin exceed this hardware's ephemeral disk.")
        if model["weight_bytes"] > 8 * 1024**3 and flavor == "cpu-basic":
            raise JobsError("Use CPU Upgrade or larger for this heavier checkpoint; CPU Basic has insufficient observed margin.")
    workflow_id = secrets.token_hex(16)
    output_repo = request.get("output_repository") or actor.username + "/qfs-capture-" + workflow_id[:12]
    _identity(output_repo, "0" * 40)
    if output_repo.split("/")[0] != actor.username:
        raise JobsError("Outputs must remain in your authenticated personal namespace.")
    if actor.client().repo_exists(output_repo, repo_type="dataset"):
        raise JobsError("Choose a new capture dataset repository; this workflow will not overwrite an existing repository.")
    registry = registry or ExplorerRegistry()
    registered = _registered(registry, reference, model, scope, codec, bits, actor) if mode == "candidate" else None
    if mode == "compare":
        d = candidate["descriptor"]
        weights = d["weights"]
        observed = _model_metadata(actor, weights["repository"], weights["revision"], mode="candidate")
        registered = _registered(registry, reference, observed, d["scope"], weights["codec"], weights["declared_bits"], actor)
    plan = {"schema": "qfs.hf-workflow-plan.v1", "workflow_id": workflow_id, "owner": actor.username, "mode": mode,
            "created_at": datetime.now(timezone.utc).isoformat(), "source": source, "image": image,
            "inputs": {"model": model, "panel": panel, "reference": reference, "candidate": candidate, "tokenizer": tokenizer},
            "output": {"dataset_repository": output_repo, "bucket": actor.username + "/qfs-explorer-results",
                       "prefix": "runs/" + workflow_id, "mount_path": "/outputs"},
            "hardware": {"flavor": flavor, "device": hw["device"], "hourly_usd": hw["hourly_usd"],
                         "unit_cost_micro_usd": hw["unit_cost_micro_usd"], "unit_label": "minute",
                         "timeout_seconds": timeout, "max_compute_usd": str(ceiling), "estimated_max_compute_usd": str(estimate),
                         "quote_time": time.time()},
            "runtime": {"dtype": "bfloat16", "schedule": "layer-outer", "trusted_code": trusted_code,
                        "unexpected_allowlist": allowlist}, "scope": scope, "codec": codec, "declared_bits": bits,
            "registered": registered, "review_metadata": spec.get("review_metadata", {}),
            "limits": {"max_output_bytes": MAX_OUTPUT},
            "notes": ["This estimate is not an account-level hard spending cap. HF enforces the requested timeout; startup, rounding and storage have separate semantics.",
                      "Jobs are billed to " + actor.username + ", not the Space owner. CPU Basic Jobs are not free CPU Basic Space hosting.",
                      "Results persist in your private bucket. No bearer token is passed to model code.",
                      "Capture/reconstruction proves its declared scope, not native serving kernels or model quality."]}
    plan = seal(plan, "plan_sha256")
    return {"plan": plan, "ticket": _ticket(plan)}


def prepare(actor, spec, registry=None):
    if not _PREPARE_SLOTS.acquire(blocking=False):
        raise JobsError("Metadata planning is busy; try again shortly.")
    try:
        return _prepare(actor, spec, registry)
    finally:
        _PREPARE_SLOTS.release()


def _ledger(actor):
    api = actor.client();repo = actor.username + "/qfs-explorer-runs"
    if not api.repo_exists(repo, repo_type="dataset"):
        try:
            api.create_repo(repo, repo_type="dataset", private=True, exist_ok=False)
            initial = api.repo_info(repo, repo_type="dataset")
            doc = {"schema": _LEDGER_SCHEMA, "owner": actor.username, "runs": {}}
            api.upload_file(repo_id=repo, repo_type="dataset", path_in_repo="ledger.json", path_or_fileobj=canonical(doc), parent_commit=initial.sha, commit_message="Initialize private QFS Jobs ledger")
        except Exception:
            # A concurrent initializer is safe only once an actual valid ledger can be read below.
            pass
    info = api.repo_info(repo, repo_type="dataset")
    if not info.private:
        raise JobsError("Your qfs-explorer-runs ledger is public. Make it private before storing workflow metadata.")
    doc, _, _ = _json_download(actor, repo, info.sha, "ledger.json", repo_type="dataset", limit=4 * 1024**2)
    if doc.get("schema") != _LEDGER_SCHEMA or doc.get("owner") != actor.username or not isinstance(doc.get("runs"), dict):
        raise JobsError("The existing private ledger does not belong to this protocol; it was not overwritten.")
    return repo, info.sha, doc


def _save_ledger(actor, repo, head, doc):
    return actor.client().upload_file(repo_id=repo, repo_type="dataset", path_in_repo="ledger.json",
        path_or_fileobj=canonical(doc), parent_commit=head, commit_message="Update QFS Jobs workflow state").oid


def _stage(job):
    stage = getattr(getattr(job, "status", None), "stage", "UNKNOWN")
    return str(getattr(stage, "value", stage)).upper()


def _job_public(job):
    return {"job_id": job.id, "url": job.url, "status": _stage(job),
            "message": getattr(job.status, "message", None), "flavor": str(getattr(job.flavor, "value", job.flavor)),
            "created_at": _plain(job.created_at), "started_at": _plain(getattr(job, "started_at", None)),
            "finished_at": _plain(getattr(job, "finished_at", None)), "durations": _plain(getattr(job, "durations", None)),
            "workflow_id": (job.labels or {}).get("qfs_workflow_id")}


_BOOTSTRAP_FETCH = r'''import hashlib,json,os,urllib.request
from pathlib import Path
p=json.loads(Path('/inputs/plan/plan.json').read_text())
assert p['plan_sha256']==os.environ['QFS_PLAN_SHA256']
body=dict(p);body['plan_sha256']=''
assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==p['plan_sha256']
u='https://raw.githubusercontent.com/malaiwah/quant-fidelity-suite/'+p['source']['revision']+'/explorer/job_bootstrap.py'
raw=urllib.request.urlopen(u,timeout=60).read()
assert hashlib.sha256(raw).hexdigest()==p['source']['bootstrap_sha256']
Path('/tmp/qfs-job-bootstrap.py').write_bytes(raw)
os.execvp('python',['python','/tmp/qfs-job-bootstrap.py','--plan','/inputs/plan/plan.json','--out','/outputs/result'])
'''


def launch(actor, prepared, *, confirm_compute=False):
    if confirm_compute is not True or not isinstance(prepared, dict):
        raise JobsError("Review the named billing account, deadline and cost, then explicitly confirm launch.")
    plan, ticket = prepared.get("plan"), prepared.get("ticket")
    verify_seal(plan, "plan_sha256");_check_ticket(actor, plan, ticket)
    if plan["owner"] != actor.username or time.time() - plan["hardware"]["quote_time"] > 600:
        raise JobsError("The account or hardware quote changed. Prepare a new plan.")
    actor.require_lifetime(plan["hardware"]["timeout_seconds"] + 120)
    source, image = _source_identity()
    if source != plan["source"] or image != plan["image"]:
        raise JobsError("The worker deployment changed. Review a new plan before launching.")
    current = next((h for h in hardware(actor) if h["name"] == plan["hardware"]["flavor"]), None)
    if current is None or current["unit_cost_micro_usd"] > plan["hardware"]["unit_cost_micro_usd"]:
        raise JobsError("HF hardware price or availability changed. Re-plan before spending.")
    from huggingface_hub import Volume
    api = actor.client()
    with _LOCK:
        repo, head, ledger = _ledger(actor)
        wid = plan["workflow_id"]
        if wid in ledger["runs"]:
            prior = ledger["runs"][wid]
            if prior.get("job_id"):
                return inspect(actor, prior["job_id"])
            matches = list(api.list_jobs(namespace=actor.username, labels={"qfs_workflow_id": wid}))
            if len(matches) == 1:
                prior.update(job_id=matches[0].id, state=_stage(matches[0]));_save_ledger(actor, repo, head, ledger)
                return _job_public(matches[0])
            raise JobsError("This launch already has a durable reservation but no unambiguous Job ID. It was NOT submitted again; inspect your HF Jobs page.")
        active = list(api.list_jobs(namespace=actor.username, labels={"qfs_app": "explorer"}, status=["SCHEDULING", "RUNNING"]))
        unresolved = [r for r in ledger["runs"].values() if r.get("state") == "CREATING" and not r.get("job_id")]
        if active or unresolved:
            raise JobsError("One QFS job is already active or has an unresolved creation. Refresh/cancel it before another launch.")
        if len(ledger["runs"]) >= 1000:
            raise JobsError("This workspace ledger reached its safety limit; archive old records before launching more jobs.")
        ledger["runs"][wid] = {"state": "CREATING", "plan": plan, "reserved_at": datetime.now(timezone.utc).isoformat()}
        head = _save_ledger(actor, repo, head, ledger)  # CAS reservation BEFORE any paid create
        bucket = plan["output"]["bucket"]
        try:
            api.create_bucket(bucket, private=True, exist_ok=True)
            if api.bucket_info(bucket).private is not True:
                raise JobsError("The results bucket is public. Nothing private will be written there.")
            prefix = plan["output"]["prefix"]
            additions = [(canonical(plan), prefix + "/inputs/plan.json"), (b"", prefix + "/outputs/.keep")]
            panel = plan["inputs"]["panel"]
            if panel and panel.get("kind") == "bundled":
                directory = ROOT / panel["path"]
                for path in sorted(directory.rglob("*")):
                    if path.is_symlink():raise JobsError("Bundled panel contains a symlink.")
                    if path.is_file(): additions.append((path, prefix + "/inputs/" + panel["path"] + "/" + str(path.relative_to(directory))))
            api.batch_bucket_files(bucket, add=additions)
            volumes = [Volume(type="bucket", source=bucket, path=prefix + "/inputs", mount_path="/inputs/plan", read_only=True),
                       Volume(type="bucket", source=bucket, path=prefix + "/outputs", mount_path="/outputs", read_only=False)]
            if panel and panel.get("kind") == "bundled":
                volumes.append(Volume(type="bucket", source=bucket, path=prefix + "/inputs", mount_path="/inputs/panel", read_only=True))
            for key in ("model", "panel", "reference", "candidate", "tokenizer"):
                value = plan["inputs"].get(key)
                if not value or value.get("kind") == "bundled":continue
                volumes.append(Volume(type="model" if key in ("model", "tokenizer") else "dataset", source=value["repository"],
                    revision=value["revision"], mount_path=value["mount_path"], read_only=True))
        except Exception as exc:
            ledger["runs"][wid]["state"] = "PREPARATION_FAILED";_save_ledger(actor, repo, head, ledger)
            if isinstance(exc, JobsError):raise
            raise _api_error(exc, "Private input/output preparation") from None
        try:
            job = api.run_job(image=plan["image"], command=["python", "-c", _BOOTSTRAP_FETCH],
                flavor=plan["hardware"]["flavor"], timeout=plan["hardware"]["timeout_seconds"], namespace=actor.username,
                env={"QFS_PLAN_SHA256": plan["plan_sha256"], "QFS_WORKFLOW_ID": wid}, secrets={},
                labels={"qfs_app": "explorer", "qfs_workflow_id": wid, "qfs_source": plan["source"]["revision"], "qfs_space": SPACE}, volumes=volumes)
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (400, 401, 402, 403, 422):
                ledger["runs"][wid]["state"] = "REFUSED";_save_ledger(actor, repo, head, ledger)
            # Ambiguous create is deliberately left CREATING. Never blindly retry it.
            raise _api_error(exc, "HF Job creation") from None
        ledger["runs"][wid].update(job_id=job.id, state=_stage(job), submitted_at=datetime.now(timezone.utc).isoformat())
        _save_ledger(actor, repo, head, ledger)
        return {**_job_public(job), "plan": plan, "billing_namespace": actor.username,
                "private_results": "hf://buckets/" + bucket + "/" + prefix + "/outputs"}


def list_runs(actor):
    try:
        return [_job_public(j) for j in actor.client().list_jobs(namespace=actor.username, labels={"qfs_app": "explorer"})]
    except Exception as exc:
        raise _api_error(exc, "Job listing") from None


def _owned_job(actor, job_id):
    actor.require_lifetime(120)
    if not JOB_ID.fullmatch(job_id or ""):
        raise JobsError("Choose an actual HF Job ID.")
    try:job = actor.client().inspect_job(job_id=job_id, namespace=actor.username)
    except Exception as exc:raise _api_error(exc, "Job lookup") from None
    owner = getattr(job.owner, "name", None) if not isinstance(job.owner, dict) else job.owner.get("name")
    if owner != actor.username or (job.labels or {}).get("qfs_app") != "explorer":
        raise JobsError("This Job does not belong to your QFS workspace.")
    return job


def inspect(actor, job_id):
    job = _owned_job(actor, job_id)
    out = _job_public(job)
    out["billing_namespace"] = actor.username
    out["results_verified"] = False
    return out


def logs(actor, job_id):
    _owned_job(actor, job_id)
    try:
        text = "\n".join(str(s) for s in actor.client().fetch_job_logs(job_id=job_id, namespace=actor.username, follow=False, tail=100))[-24000:]
    except Exception as exc:raise _api_error(exc, "Job log read") from None
    return re.sub(r"hf_[A-Za-z0-9_-]{12,}", "<redacted-token>", text)


def cancel(actor, job_id, *, confirm=False):
    if confirm is not True:raise JobsError("Confirm cancellation of this specific Job.")
    _owned_job(actor, job_id)
    actor.client().cancel_job(job_id=job_id, namespace=actor.username)
    return inspect(actor, job_id)


def _download_bucket_manifest(actor, bucket, prefix, destination, *, max_bytes):
    from huggingface_hub import BucketFile
    entries = []
    total = 0
    for item in actor.client().list_bucket_tree(bucket, prefix=prefix + "/", recursive=True):
        if not isinstance(item, BucketFile):continue
        if not item.path.startswith(prefix + "/"):raise JobsError("Bucket returned an out-of-prefix artifact.")
        relative = _relative(item.path[len(prefix) + 1:])
        size = getattr(item, "size", None)
        if type(size) is not int or size < 0:raise JobsError("Unbounded bucket artifact size.")
        total += size
        if total > max_bytes or len(entries) >= MAX_FILES:raise JobsError("Persisted output exceeds this Space's verified-transfer limit; it remains in your private bucket.")
        target = destination / relative;target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        entries.append((item, target))
    if not entries:raise JobsError("No durable output files have arrived yet. COMPLETED alone is not a verified result.")
    actor.client().download_bucket_files(bucket, entries, raise_on_missing_files=True)
    return total


def _fetch_result(actor, job_id, directory):
    job = _owned_job(actor, job_id)
    wid = (job.labels or {}).get("qfs_workflow_id")
    if not UUID.fullmatch(wid or ""):raise JobsError("The Job has no exact workflow identity.")
    repo, head, ledger = _ledger(actor)
    saved = ledger["runs"].get(wid)
    if saved is not None:
        _verify_provider(actor, job, saved["plan"])
    if saved is None:
        raise JobsError("Import this Job's original private ledger/plan first; provider labels alone are not a trusted plan.")
    plan = saved["plan"];verify_seal(plan, "plan_sha256")
    if saved.get("job_id") not in (None, job_id) or plan["owner"] != actor.username:
        raise JobsError("The private ledger identifies a different Job or owner.")
    if str(job.docker_image) != plan["image"] or str(getattr(job.flavor, "value", job.flavor)) != plan["hardware"]["flavor"]:
        raise JobsError("Provider image/flavor does not match the approved plan.")
    if (job.environment or {}).get("QFS_PLAN_SHA256") != plan["plan_sha256"]:
        raise JobsError("Provider environment does not bind the reviewed plan.")
    if _stage(job) not in _TERMINAL:
        raise JobsError("The Job is still active. Refresh logs or cancel; results are not final.")
    _download_bucket_manifest(actor, plan["output"]["bucket"], plan["output"]["prefix"] + "/outputs/result", directory,
                              max_bytes=min(MAX_OUTPUT, plan["limits"]["max_output_bytes"]))
    result = _read_json(directory / "result.json")
    verify_seal(result, "result_sha256")
    if (result.get("schema") != "qfs.hf-workflow-result.v1" or result.get("mode") != plan["mode"]
            or result.get("workflow_id") != wid or result.get("owner") != actor.username or result.get("plan_sha256") != plan["plan_sha256"]):
        raise JobsError("Persisted result belongs to a different workflow.")
    _verify_result_log(actor, job_id, result)
    records = result.get("files")
    if not isinstance(records, list) or len(records) > MAX_FILES:raise JobsError("Invalid result file manifest.")
    seen = set()
    for record in records:
        name = _relative(record["path"])
        if name in seen or name == "result.json":raise JobsError("Duplicate/self-referential output file.")
        seen.add(name);path = directory / name
        if (type(record.get("bytes")) is not int or not HEX.fullmatch(str(record.get("sha256", "")))
                or path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"] or _file_sha(path) != record["sha256"]):
            raise JobsError("Durable result bytes failed verification: " + name)
        if path.suffix == ".json" and path.stat().st_size > MAX_JSON:
            raise JobsError("A persisted JSON sidecar exceeds the safe qualification limit.")
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if _read_json(directory / "plan.json") != plan:
        raise JobsError("The original worker plan differs from the provider-bound ledger plan.")
    if actual != seen | {"result.json"}:raise JobsError("Result tree contains unlisted files.")
    if _stage(job) != "COMPLETED" or result.get("status") != "complete":
        raise JobsError("The Job failed or timed out; partial files remain private and were not promoted to a successful capture.")
    execution = {"schema": "qfs.hf-jobs-execution.v1", "job_id": job_id, "namespace": actor.username,
                 "flavor": plan["hardware"]["flavor"], "docker_image": job.docker_image,
                 "plan_sha256": plan["plan_sha256"], "source_revision": plan["source"]["revision"],
                 "status": _stage(job), "requested_timeout_seconds": plan["hardware"]["timeout_seconds"],
                 "created_at": _plain(job.created_at), "started_at": _plain(getattr(job, "started_at", None)),
                 "finished_at": _plain(getattr(job, "finished_at", None)),
                 "running_seconds": (_plain(getattr(job, "durations", None)) or {}).get("running_secs"),
                 "provider_identity_note": "Controller read authenticated HF Jobs API. Worker hardware is worker-reported, not independent reproduction."}
    (directory / "hf-execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    proof = {"result": result, "plan": plan, "execution": execution, "directory": str(directory)}
    if plan["mode"] in ("root", "candidate"):
        from fidelity.hfjobs import qualify_result
        proof["qualification"] = qualify_result(directory, plan, execution, suite_root=ROOT)
    else:
        from fidelity import dsvalidate
        comparison = directory / _relative(result["outputs"]["comparison"])
        verdict = dsvalidate.validate_receipt(_read_json(comparison))
        if verdict.errors:raise JobsError("The comparison receipt did not pass scientific validation.")
        _verify_comparison(proof)
    saved.update(job_id=job_id, state=saved.get("state") if saved.get("publications") else "VERIFIED",
                 verified_result_sha256=result["result_sha256"])
    _save_ledger(actor, repo, head, ledger)
    return proof


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path, *, limit=MAX_JSON):
    from .job_worker import load_json
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise JobsError("JSON evidence is missing, non-regular or oversized.")
    return load_json(path, maximum=limit)


def _metadata_digest(actor, repo, revision, row):
    from huggingface_hub import hf_hub_download
    with tempfile.TemporaryDirectory(prefix="qfs-model-metadata-") as td:
        path = Path(hf_hub_download(repo, row["path"], revision=revision, token=actor.client().token, cache_dir=td))
        if path.stat().st_size != row["bytes"]:
            raise JobsError("Downloaded metadata differs from its pinned file census.")
        return _file_sha(path)

def _verify_provider(actor, job, plan):
    verify_seal(plan, "plan_sha256")
    source, image = _source_identity()
    if plan.get("source") != source or plan.get("image") != image:
        deployment = ROOT / "explorer/deployment.json"
        approved = _read_json(deployment).get("reviewed_source_revisions", {}) if deployment.is_file() else {}
        pin = approved.get((plan.get("source") or {}).get("revision")) if isinstance(approved, dict) else None
        candidate_source = plan.get("source") or {}
        if (not isinstance(pin, dict) or candidate_source.get("repository") != SOURCE
                or any(candidate_source.get(k) != pin.get(k) for k in ("worker_sha256", "bootstrap_sha256"))
                or plan.get("image") != pin.get("image")):
            raise JobsError("This Job source/image is not in the deployment's explicitly reviewed recovery pins.")
        source = candidate_source
    if (plan.get("owner") != actor.username or job.command != ["python", "-c", _BOOTSTRAP_FETCH]
            or job.arguments or job.secrets or job.space_id
            or job.environment != {"QFS_PLAN_SHA256": plan["plan_sha256"], "QFS_WORKFLOW_ID": plan["workflow_id"]}
            or (job.labels or {}).get("qfs_source") != source["revision"]
            or (job.labels or {}).get("qfs_workflow_id") != plan["workflow_id"]):
        raise JobsError("Provider invocation is not the exact tokenless, pinned QFS worker.")
    trusted = plan["runtime"].get("trusted_code")
    if trusted and not any(p.get("trusted_code") == trusted
                           and p.get("model_repository") == plan["inputs"]["model"]["repository"]
                           and p.get("model_revision") == plan["inputs"]["model"]["revision"] for p in presets()):
        raise JobsError("The imported Job selected unapproved executable model code.")
    prefix, bucket = plan["output"]["prefix"], plan["output"]["bucket"]
    if prefix != "runs/" + plan["workflow_id"] or bucket != actor.username + "/qfs-explorer-results":
        raise JobsError("The output bucket is not this caller's workflow workspace.")
    expected = [
        {"type": "bucket", "source": bucket, "path": prefix + "/inputs", "mount_path": "/inputs/plan", "read_only": True, "revision": None},
        {"type": "bucket", "source": bucket, "path": prefix + "/outputs", "mount_path": "/outputs", "read_only": False, "revision": None},
    ]
    panel = plan["inputs"].get("panel")
    if panel and panel.get("kind") == "bundled":
        expected.append(dict(expected[0], mount_path="/inputs/panel"))
    for key in ("model", "panel", "reference", "candidate", "tokenizer"):
        value = plan["inputs"].get(key)
        if value and value.get("kind") != "bundled":
            _identity(value["repository"], value["revision"])
            expected.append({"type": "model" if key in ("model", "tokenizer") else "dataset",
                             "source": value["repository"], "revision": value["revision"], "path": None,
                             "mount_path": "/inputs/" + key, "read_only": True})
    observed = [{key: getattr(v, key, None) for key in expected[0]} for v in job.volumes or []]
    if sorted(observed, key=canonical) != sorted(expected, key=canonical):
        raise JobsError("Provider mounts differ from the exact approved immutable inputs and private output.")


def _verify_result_log(actor, job_id, result):
    expected = {"status": result["status"], "workflow_id": result["workflow_id"], "result_sha256": result["result_sha256"]}
    found = False
    size = 0
    for chunk in actor.client().fetch_job_logs(job_id=job_id, namespace=actor.username, follow=False, tail=100):
        size += len(str(chunk))
        if size > 1024 * 1024:
            raise JobsError("Provider completion logs exceed the bounded read limit.")
        for line in str(chunk).splitlines():
            try:
                found = json.loads(line) == expected or found
            except ValueError:
                pass
    if not found:
        raise JobsError("Provider logs do not attest these exact persisted result bytes; bucket self-seals alone are not trusted.")


def _verify_comparison(proof):
    from fidelity import dsvalidate
    root = Path(proof["directory"])
    plan, result = proof["plan"], proof["result"]
    comparison = _read_json(root / _relative(result["outputs"]["comparison"]))
    report = dsvalidate.validate_receipt(comparison)
    if (report.errors or not comparison.get("gates")
            or any(g.get("passed") is not True or g.get("overridden_by") for g in comparison["gates"].values())
            or (comparison.get("estimator") or {}).get("head_policy") != "native_head"):
        raise JobsError("Comparison failed its original receipt, own-head or unoverridden scientific gates.")
    for side in ("reference", "candidate"):
        if side == "candidate" and plan["mode"] == "candidate":
            continue
        expected = plan["inputs"][side]
        if comparison[side]["dataset_sha256"] != expected["dataset_sha256"]:
            raise JobsError("Comparison does not identify its actual immutable planned datasets.")
        receipt = _read_json(root / (side + ".verify.json"))
        verify_seal(receipt, "receipt_sha256")
        from fidelity.hfjobs import INPUT_DATASET_ROOT
        if receipt.get("subject") != INPUT_DATASET_ROOT + "/" + side:
            raise JobsError("Verification does not identify the canonical mounted dataset view.")
        if receipt.get("structural_status") != "sealed" or receipt.get("error_count") != 0 or receipt.get("errors"):
            raise JobsError("Original mounted dataset verification did not pass.")
    return comparison


def fetch_result(actor, job_id):
    """Return verified metadata only; all transfer/qualification staging is transient."""
    with _LOCK, tempfile.TemporaryDirectory(prefix="qfs-fetch-", dir=CACHE) as td:
        proof = _fetch_result(actor, job_id, Path(td))
        if "qualification" in proof:
            proof["qualification"] = _read_json(proof["qualification"]["qualification_path"])
        proof.pop("directory")
        return proof


def _review_files(proof, actor):
    root, result = Path(proof["directory"]), proof["result"]
    paths = {"result": "result.json", "plan": "plan.json", "execution": "hf-execution.json"}
    if proof["plan"]["mode"] != "root":
        paths["comparison"] = result["outputs"]["comparison"]
        if result["outputs"].get("submission"):
            paths["submission"] = result["outputs"]["submission"]
        return paths
    d = _read_json(root / "first/fidelity-dataset.json")
    rt = _read_json(root / "first" / _relative(d["runtime"]["file"]))
    q = _read_json(proof["qualification"]["qualification_path"])
    paths.update(job="job.json", dataset="first/fidelity-dataset.json", qualification="qualification.json",
                 runtime="first/" + d["runtime"]["file"], capture="first/" + d["capture"]["manifest_file"],
                 panel="first/" + d["panel"]["panel_file"], comparison=result["outputs"]["reproduction"])
    # Select byte identities, never guess an upstream config/panel receipt filename.
    for role, sha in (("config", d["weights"]["config_sha256"]), ("panel_receipt", q["job_contract"]["panel_receipt_file_sha256"])):
        choices = [row["path"] for row in result["files"] if row["sha256"] == sha and row["path"].endswith(".json")]
        if not choices and role == "config":
            model = proof["plan"]["inputs"]["model"]
            destination = root / "original-config.json"
            _, observed_sha, observed_bytes = _json_download(
                actor, model["repository"], model["revision"], "config.json", save_to=destination)
            if observed_sha != sha or observed_bytes != model["config_bytes"]:
                raise JobsError("Pinned original config differs from the qualified checkpoint.")
            choices = ["original-config.json"]
        if not choices:
            raise JobsError("Original qualified " + role + " bytes are absent from durable evidence.")
        paths[role] = sorted(choices, key=lambda p: (not p.startswith("first/"), p))[0]
    from registry.tools import harness_id
    source = _read_json(root / "source-manifest.json")
    digests = [{"role": "source_%04d" % i, "path": f["path"], "sha256": f["sha256"]}
               for i, f in enumerate(sorted(source["source_files"], key=lambda f: f["path"]))]
    versions = {k: v for k, v in rt["stack_fingerprint"].items() if k.endswith("_version") and isinstance(v, str)}
    harness = {"recorded": True, "boundary": harness_id.BOUNDARY, "covers": ["metric.value"],
               "repository": {"url": source["repository"], "commit": source["revision"], "commit_role": "exact", "dirty": False},
               "code_digests": digests, "tool_versions": versions}
    harness["harness_id"] = harness_id.compute_id(digests, versions)
    (root / "review-harness.json").write_bytes(canonical(harness))
    paths["harness"] = "review-harness.json"
    return paths


def _root_metadata(proof):
    root, plan = Path(proof["directory"]), proof["plan"]
    d = _read_json(root / "first/fidelity-dataset.json")
    data = ExplorerRegistry().registry_data()
    attribution = lambda row: {k: row[k] for k in ("name", "handle", "url") if row.get(k) is not None}
    meta = {}
    models = [m for m in data["models"].values() if m["huggingface"]["repository"] == d["weights"]["repository"]
              and m["huggingface"]["revision"] == d["weights"]["revision"]]
    if len(models) == 1:
        model = models[0]
        meta.update(name=model["name"], family=model["family"], publisher=attribution(model["publisher"]), model_license=model["license"])
    panels = [p for p in data["panels"].values() if p["identity"]["panel_token_sha256"] == d["panel"]["suite_token_hash_sha256"]]
    if len(panels) > 1 and len(models) == 1:
        known_panels = {m["panel_ref"] for m in data["measurements"].values() if m["model_ref"] == models[0]["id"]}
        panels = [p for p in panels if p["id"] in known_panels]
    if len(panels) == 1:
        meta.update(panel_author=attribution(panels[0]["author"]), corpus_lineage=panels[0]["corpus"]["lineage"])
    authors = [attribution(p["author"]) for p in data["pipelines"].values()
               if p["implementation"].get("repository") == SOURCE]
    if authors and all(a == authors[0] for a in authors):
        meta["toolchain_author"] = authors[0]
    supplied = plan.get("review_metadata") or {}
    allowed = {"name", "family", "publisher", "panel_author", "toolchain_author", "corpus_lineage", "model_license"}
    if set(supplied) - allowed:
        raise JobsError("Review metadata has unknown fields; canonical repository/revision are controller-owned.")
    meta.update(supplied)
    return meta


def _verify_uploaded(actor, repository, revision, records, *, private):
    api = actor.client()
    info = api.repo_info(repository, repo_type="dataset", revision=revision)
    if info.sha != revision or info.private is not private or getattr(info, "gated", False):
        raise JobsError("Published revision or explicit visibility differs from the requested new dataset.")
    from fidelity import dshub
    for row in records:
        dshub._read_remote_exact(dshub.resolve_url(repository, revision, row["path"]),
                                row["bytes"], row["sha256"], token=api.token if private else None)

def _upload_tree(actor, repository, files, *, private, state, persist):
    """Exclusive new-repository upload with durable CAS heads, never visibility edits."""
    from huggingface_hub import CommitOperationAdd
    from fidelity import dshub, dsformat
    _identity(repository, "0" * 40)
    if repository.split("/")[0] != actor.username:
        raise JobsError("Publication is restricted to the caller's personal namespace.")
    api = actor.client()
    records = []
    canonical_root = "fidelity-dataset.json" in files
    third_party_receipts = set()
    if canonical_root:
        from fidelity import panel as panel_contract
        panel = _read_json(files["fidelity-dataset.json"]).get("panel") or {}
        receipt = panel.get("panel_receipt_file")
        if (receipt in files and panel.get("panel_receipt_sha256")
                and _read_json(files[receipt]).get("schema") == panel_contract.ARTIFACT_RECEIPT_SCHEMA):
            if panel_contract.verify_third_party_sealed_receipt(files[receipt].read_bytes(), panel["panel_receipt_sha256"]):
                third_party_receipts.add(receipt)
    for name, path in sorted(files.items()):
        _relative(name)
        if dsformat.looks_like_a_credential(name):
            raise JobsError("Credential/private filenames cannot be published.")
        dshub._scan_publish_member(str(path), name, api.token,
                                  textual=canonical_root and name not in third_party_receipts and dshub._textual_publish_member(name))
        if name in {"job.json", "qualification.json", "hf-execution.json", "review-harness.json", "receipts/root-qualification.json"}:
            dshub._scan_publish_member(str(path), name, api.token, textual=True)
        records.append({"path": name, "bytes": path.stat().st_size, "sha256": _file_sha(path)})
    if state.get("revision"):
        _verify_uploaded(actor, repository, state["revision"], records, private=private)
        return state["revision"]
    if not state.get("head"):
        if state.get("creating"):
            raise JobsError("Repository creation was ambiguous. The reservation is retained; no existing repository was overwritten.")
        for kind in ("dataset", "model", "space"):
            if api.repo_exists(repository, repo_type=kind):
                raise JobsError("Publication requires a new repository; an existing namespace object was not modified.")
        state.update(creating=True, repository=repository, private=private)
        persist()
        api.create_repo(repository, repo_type="dataset", private=private, exist_ok=False)
        info = api.repo_info(repository, repo_type="dataset")
        if info.private is not private or not SHA.fullmatch(info.sha or ""):
            raise JobsError("Exclusive repository creation did not return the requested visibility and exact HEAD.")
        state.update(head=info.sha, creating=False)
        persist()
    info = api.repo_info(repository, repo_type="dataset")
    if info.sha != state["head"]:
        # Lost commit response: recover only the exact intended byte tree.
        _verify_uploaded(actor, repository, info.sha, records, private=private)
        revision = info.sha
    else:
        commit = api.create_commit(repository, repo_type="dataset", parent_commit=state["head"],
                                   operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(f)) for p, f in sorted(files.items())],
                                   commit_message="Publish verified QFS workflow evidence")
        revision = commit.oid
    if not SHA.fullmatch(revision or ""):
        raise JobsError("Upload did not return an immutable commit.")
    state["revision"] = revision
    persist()
    _verify_uploaded(actor, repository, revision, records, private=private)
    return revision


def publish_result(actor, job_id, *, visibility="private", confirm_publish=False, confirm_redistribution=False):
    """Publish original verified evidence; public roots additionally get a canonical dataset."""
    if confirm_publish is not True or visibility not in ("private", "public"):
        raise JobsError("Explicitly confirm publication and choose private or public visibility.")
    public = visibility == "public"
    if public and confirm_redistribution is not True:
        raise JobsError("Public capture evidence includes heads, token panels, license and lineage: explicitly confirm redistribution rights.")
    from . import review
    review._identity(actor)
    with _LOCK, tempfile.TemporaryDirectory(prefix="qfs-publish-", dir=CACHE) as td:
        proof = _fetch_result(actor, job_id, Path(td))
        root, plan, result = Path(td), proof["plan"], proof["result"]
        repo, head, ledger = _ledger(actor)
        saved = ledger["runs"][plan["workflow_id"]]
        prior = (saved.get("publications") or {}).get(visibility)
        if prior:
            _check_saved_publication(actor, prior, proof, public=public)
            return prior
        roles = _review_files(proof, actor)
        files = {row["path"]: root / row["path"] for row in result["files"]}
        files.update({"result.json": root / "result.json", "hf-execution.json": root / "hf-execution.json"})
        if "qualification" in proof:
            import fidelity_dataset as fd
            from fidelity.hfjobs import publication_source
            q = proof["qualification"]
            # Qualification time is recorded once so interrupted uploads resume exact bytes.
            recorded = saved.setdefault("publication_qualification", _read_json(q["qualification_path"]))
            Path(q["qualification_path"]).write_bytes(canonical(recorded))
            head = _save_ledger(actor, repo, head, ledger)
            fd._load_qualification(q["qualification_path"], job_path=q["job_path"], dataset=q["dataset_path"],
                                   repository=plan["output"]["dataset_repository"])
            publication_source(q["dataset_path"], q["qualification_path"], q["job_path"])
            files.update({"job.json": Path(q["job_path"]), "qualification.json": Path(q["qualification_path"])})
        files.update({p: root / p for p in roles.values()})
        metadata = _root_metadata(proof) if plan["mode"] == "root" else {}
        pointers = {role: {"path": p, "sha256": _file_sha(root / p)} for role, p in roles.items()}
        if public:
            for pointer in pointers.values():
                if (root / pointer["path"]).stat().st_size > review.MAX_FILE:
                    raise JobsError("Required public review JSON exceeds the 1 MiB review limit; private durable publication remains available.")
        attempt = saved.setdefault("publication_attempts", {}).setdefault(visibility, {})
        def persist():
            nonlocal head
            head = _save_ledger(actor, repo, head, ledger)
        if public and plan["mode"] == "root":
            from fidelity import dsformat as F
            dataset = Path(proof["qualification"]["dataset_path"])
            root_files = {name: dataset / name for name in F.iter_dataset_files(str(dataset), exclude=())}
            root_files["receipts/root-qualification.json"] = Path(proof["qualification"]["qualification_path"])
            root_repo = plan["output"]["dataset_repository"]
            revision = _upload_tree(actor, root_repo, root_files, private=False,
                                    state=attempt.setdefault("root", {}), persist=persist)
            metadata.update(root_repository=root_repo, root_revision=revision)
        repository = actor.username + "/qfs-evidence-" + plan["workflow_id"] + "-" + visibility
        revision = _upload_tree(actor, repository, files, private=not public,
                                state=attempt.setdefault("evidence", {}), persist=persist)
        publication = {"kind": "root" if plan["mode"] == "root" else "measurement", "repository": repository,
                       "revision": revision, "files": pointers, "metadata": metadata,
                       "redistribution_consent": public and confirm_redistribution is True}
        saved.setdefault("publications", {})[visibility] = publication
        saved["state"] = "PUBLISHED_" + visibility.upper()
        persist()
        if public and review._signing_key() is not None and (publication["kind"] == "root" or "submission" in pointers):
            publication = review.attest_publication(actor, publication)
            saved["publications"][visibility] = publication
            persist()
        return publication


def _check_saved_publication(actor, publication, proof, *, public):
    from . import review
    repository, revision = _identity(publication["repository"], publication["revision"])
    if repository.split("/")[0] != actor.username:
        raise JobsError("Saved publication is not in this caller's namespace.")
    info = actor.client().repo_info(repository, repo_type="dataset", revision=revision)
    if info.private is not (not public) or info.sha != revision:
        raise JobsError("Saved publication visibility or immutable revision changed.")
    expected_roles = _review_files(proof, actor)
    if (set(publication.get("files", {})) != set(expected_roles)
            or any(publication["files"][role].get("path") != path
                   or not HEX.fullmatch(str(publication["files"][role].get("sha256", "")))
                   for role, path in expected_roles.items())):
        raise JobsError("Saved publication roles differ from the original verified evidence.")
    with tempfile.TemporaryDirectory(prefix="qfs-published-") as td:
        raw = review._evidence(actor.client(), publication, Path(td), require_public=public)
        if (json.loads(raw["result"]) != proof["result"] or json.loads(raw["plan"]) != proof["plan"]
                or json.loads(raw["execution"]) != proof["execution"]):
            raise JobsError("Saved published receipts do not bind the freshly provider-verified original result.")
        _verify_uploaded(actor, repository, revision, proof["result"]["files"], private=not public)
        original = {r["path"]: r["sha256"] for r in proof["result"]["files"]}
        if "config" in expected_roles:
            original[expected_roles["config"]] = _file_sha(Path(proof["directory"]) / expected_roles["config"])
        for role, pointer in publication["files"].items():
            if role not in {"execution", "job", "qualification", "harness", "result"} and original.get(pointer["path"]) != pointer["sha256"]:
                raise JobsError("Saved publication substituted original worker evidence.")
        if publication["kind"] == "root":
            import fidelity_dataset as fd
            q = proof["qualification"]
            if publication["files"]["harness"]["sha256"] != _file_sha(Path(proof["directory"]) / expected_roles["harness"]):
                raise JobsError("Published harness differs from the original measured source and runtime.")
            Path(q["job_path"]).write_bytes(raw["job"])
            Path(q["qualification_path"]).write_bytes(raw["qualification"])
            fd._load_qualification(q["qualification_path"], job_path=q["job_path"], dataset=q["dataset_path"],
                                   repository=proof["plan"]["output"]["dataset_repository"])
            if json.loads(raw["job"])["hf_execution"]["provider_receipt"] != proof["execution"]:
                raise JobsError("Published qualification has different provider readback.")
            if public:
                meta = publication["metadata"]
                if meta["root_repository"] != proof["plan"]["output"]["dataset_repository"]:
                    raise JobsError("Canonical root publication differs from the original planned dataset.")
                records = [dict(r, path=r["path"][6:]) for r in proof["result"]["files"] if r["path"].startswith("first/")]
                _verify_uploaded(actor, meta["root_repository"], meta["root_revision"], records, private=False)


def request_review(actor, job_id, *, confirm_public=False):
    """Recover by own provider Job ID, then post the saved immutable public evidence."""
    if confirm_public is not True:
        raise JobsError("Explicitly confirm a public registry review request; this does not publish private evidence.")
    from . import review
    with _LOCK, tempfile.TemporaryDirectory(prefix="qfs-review-job-", dir=CACHE) as td:
        proof = _fetch_result(actor, job_id, Path(td))
        repo, head, ledger = _ledger(actor)
        saved = ledger["runs"][proof["plan"]["workflow_id"]]
        publication = (saved.get("publications") or {}).get("public")
        if publication is None:
            raise JobsError("Publish this Job publicly with redistribution consent first; private evidence is never exposed by review.")
        _check_saved_publication(actor, publication, proof, public=True)
        publication = review.attest_publication(actor, publication)
        saved["publications"]["public"] = publication
        head = _save_ledger(actor, repo, head, ledger)
        if saved.get("review_request"):
            return saved["review_request"]
        receipt = review.request_review(actor, publication, confirm_public=True)
        saved["review_request"] = receipt
        saved["state"] = "REVIEW_REQUESTED"
        _save_ledger(actor, repo, head, ledger)
        return receipt
