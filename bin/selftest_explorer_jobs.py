#!/usr/bin/env python3
"""Offline negative-path checks for the Explorer's HF-Jobs subsystem.

EXP-01/SEC-03. A pinned PUBLIC read must be anonymous: an anonymous read is the
evidence the artifact is publicly readable, and only a 401/403 escalates to the
caller's token, so a 404 or a network fault never sends a credential anywhere
(jobs.py re-learned the CLI-22/SEC-03 rule the CLI already enforces).

EXP-02. Required-public review evidence is read anonymously ONLY: a 401/403 is
itself the disclosure that the evidence is not public, so it is a refusal,
never a credential escalation (review.py).

EXP-03. The worker must consume the already-verified canonical dataset views,
never re-read the raw Hub volume -- the exact surface that produced truncated
JSON prefixes in production (job_worker.py).

EXP-04. Attribution values the Explorer infers from registry matching must be
labeled as inference at the explorer layer; the frozen published metadata
shape stays untouched, and explicit review_metadata overrides stay unlabeled.

EXP-05. publish_explorer refuses ambient authentication: the token comes only
from an explicit 0600 file named by --token-file.

EXP-06. A reservation stuck CREATING without a Job ID reconciles ONLY on
positive provider-side absence proof; any doubt keeps refusing, and stale
reservations surface as a distinct status.

Everything runs against a stubbed huggingface_hub: no network, no real HF, no
token. The one token-shaped fixture string is never sent anywhere.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import secrets
import stat
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bin"))

REV = "a" * 40
TOKEN = "hf_fixture_token_never_sent_anywhere"
STALE_WID = "f" * 32


def check(name, value):
    if not value:
        raise AssertionError(name)
    print("  PASS  %s" % name)


def refuses(call, *types_):
    try:
        call()
    except types_ or (ValueError,):
        return True
    return False


# ---------------------------------------------------------------------------
# The stub Hub. huggingface_hub is imported lazily inside every explorer
# function, so a module seeded into sys.modules before the first call is the
# seam: no network, no real credential, and every read records its token.
# ---------------------------------------------------------------------------
RECORD = []
RESPONSES = {}
ns = types.SimpleNamespace


class FakeHTTPError(Exception):
    def __init__(self, status):
        super().__init__("HTTP %s (fixture)" % status)
        self.response = ns(status_code=status)


def _write_download(kwargs, filename, payload):
    target = Path(kwargs["cache_dir"]) / Path(filename).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return str(target)


class FakeApi:
    def __init__(self, endpoint=None, token=None, **_):
        self.endpoint = endpoint
        self.token = token
        RECORD.append({"method": "__init__", "token": token, "args": (), "kwargs": {}})

    def __getattr__(self, name):
        def call(*args, **kwargs):
            RECORD.append({"method": name, "token": self.token, "args": args, "kwargs": kwargs})
            result = RESPONSES[name](self.token, *args, **kwargs)
            if name == "hf_hub_download" and isinstance(result, bytes):
                return _write_download(kwargs, args[1], result)
            return result
        return call


def _fake_module_download(repo_id, filename, *, token=None, cache_dir=None, **_):
    RECORD.append({"method": "hf_hub_download", "token": token, "args": (repo_id, filename), "kwargs": {}})
    payload = RESPONSES["hf_hub_download"](token, repo_id, filename)
    return _write_download({"cache_dir": cache_dir or tempfile.mkdtemp(prefix="qfs-hub-fixture-")}, filename, payload)


def _install_fake_hub():
    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = FakeApi
    hub.hf_hub_download = _fake_module_download

    class Volume:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class BucketFile:
        pass

    class CommitOperationAdd:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    hub.Volume = Volume
    hub.BucketFile = BucketFile
    hub.CommitOperationAdd = CommitOperationAdd
    errors = types.ModuleType("huggingface_hub.errors")

    class EntryNotFoundError(Exception):
        pass

    errors.EntryNotFoundError = EntryNotFoundError
    hub.errors = errors
    hub.EntryNotFoundError = EntryNotFoundError
    sys.modules["huggingface_hub"] = hub
    sys.modules["huggingface_hub.errors"] = errors


_install_fake_hub()

from explorer import jobs, review  # noqa: E402  (needs the stub Hub first)
from explorer.auth import Actor  # noqa: E402


def _reset(**responses):
    RECORD.clear()
    RESPONSES.clear()
    RESPONSES.update(responses)


def _authenticated_reads(token):
    return [r for r in RECORD if r["method"] != "__init__" and r["token"] == token]


def _serve(files):
    def paths(token, repo, names, **_):
        return [ns(path=name, size=len(files[name])) for name in names]

    def download(token, repo, filename, **_):
        return files[filename]

    return paths, download


PAYLOAD = b'{"model_type": "tiny-fixture"}'


# ---------------------------------------------------------------------------
# EXP-01: anonymous-first pinned reads (jobs.py).
# ---------------------------------------------------------------------------
def rung_anonymous_first(actor):
    paths, download = _serve({"config.json": PAYLOAD})
    _reset(get_paths_info=paths, hf_hub_download=download)
    value, sha, size = jobs._json_download(actor, "pub/model", REV, "config.json")
    check("E1a public metadata read returns the pinned bytes", value == json.loads(PAYLOAD) and size == len(PAYLOAD))
    reads = [r for r in RECORD if r["method"] == "get_paths_info"]
    check("E1b a public read probes the tree anonymously exactly once",
          len(reads) == 1 and reads[0]["token"] in (False, None))
    fetches = [r for r in RECORD if r["method"] == "hf_hub_download"]
    check("E1c the anonymous download carries no token",
          len(fetches) == 1 and fetches[0]["token"] in (False, None))
    check("E1d no credential was attached anywhere on a public read", not _authenticated_reads(TOKEN))

    # 401 escalates to the caller's token: a private duplicate still works.
    def gated_paths(token, repo, names, **_):
        if not token:
            raise FakeHTTPError(401)
        return [ns(path=n, size=len(PAYLOAD)) for n in names]

    def gated_download(token, repo, filename, **_):
        if not token:
            raise FakeHTTPError(401)
        return PAYLOAD

    _reset(get_paths_info=gated_paths, hf_hub_download=gated_download)
    value, _, _ = jobs._json_download(actor, "priv/model", REV, "config.json")
    check("E2a a 401 escalates and the private duplicate read succeeds", value == json.loads(PAYLOAD))
    probes = [r["token"] for r in RECORD if r["method"] == "get_paths_info"]
    check("E2b escalation order is anonymous first, caller token second", probes[0] in (False, None) and probes[1] == TOKEN)

    # 403 escalates too.
    def forbidden(token, repo, names, **_):
        if not token:
            raise FakeHTTPError(403)
        return [ns(path=n, size=len(PAYLOAD)) for n in names]

    _reset(get_paths_info=forbidden, hf_hub_download=gated_download)
    jobs._json_download(actor, "priv/model", REV, "config.json")
    check("E2c a 403 escalates exactly like a 401",
          [r["token"] for r in RECORD if r["method"] == "get_paths_info"] == [False, TOKEN])

    # A 404 never sends a credential anywhere.
    _reset(get_paths_info=lambda token, repo, names, **_: (_ for _ in ()).throw(FakeHTTPError(404)),
           hf_hub_download=gated_download)
    check("E3a a 404 refuses without escalation", refuses(lambda: jobs._json_download(actor, "pub/model", REV, "x.json"), jobs.JobsError))
    check("E3b a 404 sent no credential anywhere", not _authenticated_reads(TOKEN))

    # A network fault never sends a credential anywhere either.
    _reset(get_paths_info=lambda token, repo, names, **_: (_ for _ in ()).throw(OSError("connection reset")),
           hf_hub_download=gated_download)
    check("E3c a network fault refuses without escalation", refuses(lambda: jobs._json_download(actor, "pub/model", REV, "x.json"), jobs.JobsError))
    check("E3d a network fault sent no credential anywhere", not _authenticated_reads(TOKEN))

    # Model metadata: the census lookup itself is an anonymous-first read.
    cfg, lic = b'{"model_type": "tiny-fixture"}', b"MIT fixture license\n"
    model_info = ns(sha=REV, author="pub", card_data=None, siblings=[
        ns(rfilename="config.json", size=len(cfg), lfs=None),
        ns(rfilename="LICENSE", size=len(lic), lfs=None),
        ns(rfilename="model.safetensors", size=32, lfs=ns(sha256="ab" * 32))])
    files = {"config.json": cfg, "LICENSE": lic}
    paths, download = _serve(files)

    def gated_info(token, repo, **_):
        if not token:
            raise FakeHTTPError(403)
        return model_info

    _reset(model_info=gated_info, get_paths_info=paths, hf_hub_download=download)
    meta = jobs._model_metadata(actor, "priv/model", REV, mode="root")
    lookups = [r["token"] for r in RECORD if r["method"] == "model_info"]
    check("E4a model census lookup escalates only after an anonymous 403", lookups == [False, TOKEN])
    check("E4b escalated private model metadata resolves",
          meta["publisher"] == "pub" and meta["license_file"] == "LICENSE")
    check("E4c small unhashed metadata digests were fetched anonymously",
          all(r["token"] in (False, None) for r in RECORD if r["method"] == "hf_hub_download"))

    # And a genuinely public model never attaches the token at all.
    _reset(model_info=lambda token, repo, **_: model_info, get_paths_info=paths, hf_hub_download=download)
    jobs._model_metadata(actor, "pub/model", REV, mode="root")
    check("E4d a public model metadata read sends no token", not _authenticated_reads(TOKEN))


# ---------------------------------------------------------------------------
# EXP-02: required-public review evidence is anonymous only (review.py).
# ---------------------------------------------------------------------------
def rung_public_evidence(actor):
    public_info = ns(sha=REV, private=False)
    _reset(repo_info=lambda token, repo, **_: public_info)
    review._public("tester/pub-evidence", REV)
    reads = [r for r in RECORD if r["method"] == "repo_info"]
    check("R1a the public-evidence check reads anonymously", len(reads) == 1 and reads[0]["token"] in (False, None))

    for status in (401, 403):
        _reset(repo_info=lambda token, repo, **_: (_ for _ in ()).throw(FakeHTTPError(status)))
        try:
            review._public("tester/hidden-evidence", REV)
            raised = None
        except ValueError as exc:
            raised = exc
        check("R1b a %d on required-public evidence is a refusal, never an escalation" % status,
              raised is not None and "refused an anonymous read" in str(raised) and "not public" in str(raised))
        check("R1c the %d refusal sent no credential anywhere" % status, not _authenticated_reads(TOKEN))

    # The full evidence fetch of a public publication is anonymous end to end.
    docs = {name: json.dumps({"schema": "qfs.fixture." + name, "owner": "tester"}).encode()
            for name in ("result", "plan", "execution")}
    pointers = {name: {"path": name + ".json", "sha256": hashlib.sha256(doc).hexdigest()}
                for name, doc in docs.items()}
    publication = {"kind": "measurement", "repository": "tester/pub-evidence", "revision": REV,
                   "redistribution_consent": True, "files": pointers}
    files = {p["path"]: docs[name] for name, p in pointers.items()}
    paths, download = _serve(files)
    _reset(repo_info=lambda token, repo, **_: public_info, get_paths_info=paths, hf_hub_download=download)
    raw = review._evidence(actor.client(), publication, Path(tempfile.mkdtemp()), require_public=True)
    check("R1d public evidence downloads anonymously and hashes match",
          set(raw) == set(docs) and not _authenticated_reads(TOKEN))

    # The caller's own PRIVATE publication re-check keeps its authenticated read.
    _reset(repo_info=lambda token, repo, **_: ns(sha=REV, private=True),
           get_paths_info=paths, hf_hub_download=download)
    review._evidence(actor.client(), publication, Path(tempfile.mkdtemp()), require_public=False)
    check("R1e a private publication re-check still uses the caller's token",
          any(r["method"] == "hf_hub_download" and r["token"] == TOKEN for r in RECORD))


# ---------------------------------------------------------------------------
# EXP-03: the worker consumes canonical views, not the raw Hub volume.
# ---------------------------------------------------------------------------
def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class StubRunner:
    """Runs nothing: materializes only the receipts the contract layer reads back."""

    def __init__(self, out, receipts=None):
        self.out = Path(out)
        self.receipts = receipts or {}
        self.ran = []
        self.argv = []

    def measure(self, name, function, *args):
        # The real Runner.measure (upstream 08b8839) times a local preparation
        # phase; the stub records the step and calls through, so the rungs
        # still assert which preparations ran and what they returned.
        self.ran.append(name)
        return function(*args)

    def run(self, name, arguments, *, allowed=(0,)):
        self.ran.append(name)
        self.argv.append([str(a) for a in arguments])
        receipt = self.receipts.get(name)
        if receipt is not None:
            path = self.out / receipt[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(receipt[1]))
        return 0

    def bound(self):
        pass


def _staged_dataset(root, name, *, repo, revision, manifest_extra=None):
    """Both halves of a planned dataset input, exactly as jobs.py stages them:
    verified metadata under the plan volume, and the raw Hub volume whose
    fidelity-dataset.json carries the observed truncated-JSON provider fault."""
    from fidelity import dsformat as F
    manifest = {"schema": F.DATASET_SCHEMA, "format_version": F.FORMAT_VERSION, "dataset_sha256": "",
                "weights": {"repository": repo, "revision": revision},
                "panel": {"suite_token_hash_sha256": "e" * 64},
                "capture": {"capture_content_digest": "c" * 64}}
    manifest.update(manifest_extra or {})
    manifest = F.seal_manifest(manifest)
    raw = root / ("raw-" + name)
    raw.mkdir()
    # The production provider fault this rung exists for: a truncated prefix.
    (raw / F.MANIFEST_NAME).write_bytes(_canonical(manifest)[:9])
    notes = b'{"note": "fixture member"}\n'
    members = {F.MANIFEST_NAME: _canonical(manifest),
               F.CHECKSUMS_NAME: (hashlib.sha256(notes).hexdigest() + "  notes.txt\n").encode(),
               "notes.txt": notes}
    metadata_root = root / "plan" / "datasets" / name
    metadata_root.mkdir(parents=True)
    rows = []
    for member, payload in members.items():
        (metadata_root / member).write_bytes(payload)
        rows.append({"path": member, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    descriptor = {"repository": repo, "revision": revision, "mount_path": str(raw),
                  "dataset_sha256": manifest["dataset_sha256"], "metadata_files": rows}
    return descriptor, manifest


def _prepare_worker_stubs(root):
    import explorer.job_worker as worker
    from fidelity import hfjobs
    worker.PLAN_PATH = root / "plan" / "plan.json"
    worker.PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    hfjobs.INPUT_DATASET_ROOT = str(root / "staged-inputs")
    worker.source_manifest = lambda plan: {
        "schema": "qfs.hf-workflow-source.v1", "repository": worker.SOURCE, "revision": "f" * 40,
        "source_files": [{"path": "explorer/job_worker.py", "bytes": 1, "sha256": "a" * 64},
                         {"path": "bin/BUNDLE.txt", "bytes": 1, "sha256": "b" * 64}]}
    return worker


def rung_worker_canonical_views(root):
    worker = _prepare_worker_stubs(root)
    from fidelity import dsmanifest

    # Compare mode with registered provenance: pre-fix this re-read the RAW
    # candidate volume for the registered-compare manifest.
    cand_repo, cand_rev = "pub/tiny-cand", "b" * 40
    scope = dsmanifest.scope_block(
        assignments=[{"tensor_class": "mlp", "treatment": "quantized", "format": "nvfp4",
                      "bits_per_weight": 4.0}],
        head_policy="native_head", kv_cache_dtype="float32", policy="static")
    reference, _ = _staged_dataset(root, "reference", repo="pub/tiny-ref", revision=REV)
    candidate, candidate_manifest = _staged_dataset(
        root, "candidate", repo=cand_repo, revision=cand_rev,
        manifest_extra={"weights": {"repository": cand_repo, "revision": cand_rev},
                        "panel": {"suite_token_hash_sha256": "e" * 64}, "scope": scope})
    registered = {"model_ref": "model--x", "panel_ref": "panel--x", "reference_ref": "reference--x",
                  "registry_repository": "malaiwah/quant-fidelity-registry", "registry_revision": "d" * 40,
                  "artifact": {"repository": cand_repo, "revision": cand_rev, "container": "safetensors",
                               "codec": {"family": "nvfp4", "bits_per_weight_nominal": 4.0},
                               "scope": scope,
                               "producer": {"name": "pub", "handle": "pub", "url": "https://huggingface.co/pub"}},
                  "panel": {"panel_ref": "panel--x", "panel_token_sha256": "e" * 64},
                  "reference": {"reference_ref": "reference--x", "teacher_receipt_sha256": "9" * 64}}
    comparison_receipt = {
        "comparison_kind": "measurement", "gates": {},
        "metric": {"value": 0.05, "units": "nats", "direction": "lower_is_better"},
        "candidate": {"lane": "other", "form": "hidden", "dataset_sha256": "9" * 64,
                      "repository": cand_repo, "revision": cand_rev, "dataset_id": "fid--cand"},
        "reference": {"lane": "other", "form": "hidden", "dataset_sha256": "8" * 64, "dataset_id": "fid--ref"},
        "estimator": {"accumulation_dtype": "float64", "head_policy": "native_head",
                      "logits_dtype": "float32", "two_pass": True, "vocab_chunk": 8192},
        "determinism": {"run_count": 2, "cold_start_per_run": True},
        "measurement_scope": {"contexts": 4, "scored_positions": 64},
        "kl": {"median": 0.01, "p95": 0.2, "p99": 0.3, "p99_9": 0.4, "max": 1.0},
        "disclosures": []}
    plan = {"schema": "qfs.hf-workflow-plan.v1", "workflow_id": "1" * 32, "owner": "tester",
            "mode": "compare", "source": {"revision": "f" * 40}, "image": "python@sha256:" + "c" * 64,
            "inputs": {"model": None, "panel": None, "reference": reference, "candidate": candidate,
                       "tokenizer": None},
            "output": {"dataset_repository": "tester/qfs-capture-x", "bucket": "tester/qfs-explorer-results",

                       "prefix": "runs/x", "mount_path": "/outputs"},
            "hardware": {"device": "cpu", "flavor": "cpu-basic", "timeout_seconds": 600},
            "runtime": {"dtype": "bfloat16", "schedule": "layer-outer", "trusted_code": None,
                        "unexpected_allowlist": None},
            "registered": registered}
    out = root / "out-compare"
    out.mkdir()
    runner = StubRunner(out, {"comparison": ("comparison/comparison-receipt.json", comparison_receipt)})
    outputs = {}
    worker.workflow(plan, out, runner, outputs)
    check("W1a registered compare completes over the canonical candidate view", outputs.get("submission") is not None)
    check("W1b the submission receipt was emitted from the canonical manifest",
          (out / outputs["submission"]).is_file() and "submission-validation" in runner.ran)
    check("W1c the raw candidate volume was never re-parsed",
          json.loads((out / "candidate.input.json").read_text())["repository"] == cand_repo)

    from fidelity import hfjobs
    hfjobs.INPUT_DATASET_ROOT = str(root / "staged-inputs-candidate")

    # Candidate mode: pre-fix the base-capture block re-read the RAW reference
    # volume; post-fix it consumes the canonical staged manifest. The real
    # fidelity.panel was already imported by the W1 emit_submission chain, so
    # both sys.modules AND the package attribute are pointed at the stub.
    fake_panel = types.ModuleType("fidelity.panel")
    fake_panel.resolve_panel = lambda panel, role=None, tokenizer_root=None: ns(
        to_dict=lambda: {"tokenizer": {"files_verified": True}})
    sys.modules["fidelity.panel"] = fake_panel
    sys.modules["fidelity"].panel = fake_panel
    fake_quant = types.ModuleType("quant_stream")
    fake_quant.quantization = lambda config: {}
    fake_quant.reader_for = lambda config: None
    sys.modules["quant_stream"] = fake_quant

    mount = root / "model-mount"
    mount.mkdir()
    cfg, weights, lic = b'{"model_type": "tiny-fixture"}', b"FIXTUREWEIGHTS" * 4, b"MIT fixture license\n"
    (mount / "config.json").write_bytes(cfg)
    (mount / "model.safetensors").write_bytes(weights)
    (mount / "LICENSE").write_bytes(lic)
    model = {"repository": "pub/tiny", "revision": REV, "mount_path": str(mount),
             "config": json.loads(cfg), "config_sha256": hashlib.sha256(cfg).hexdigest(),
             "config_bytes": len(cfg),
             "files": [{"path": n, "bytes": len(d), "sha256": hashlib.sha256(d).hexdigest()}
                       for n, d in (("config.json", cfg), ("LICENSE", lic), ("model.safetensors", weights))],
             "weight_bytes": len(weights), "index_sha256": None, "index_bytes": None, "license_file": "LICENSE"}
    panel_mount = root / "panel-mount"
    (panel_mount / "panel-x").mkdir(parents=True)
    (panel_mount / "panel-x" / "panel.json").write_bytes(b'{"schema": "quant-pipeline.glm53-token-panel.v1"}')
    reference2, reference2_manifest = _staged_dataset(root, "reference-c", repo="pub/tiny-ref", revision=REV)
    candidate_plan = {"schema": "qfs.hf-workflow-plan.v1", "workflow_id": "2" * 32, "owner": "tester",
                      "mode": "candidate", "source": {"revision": "f" * 40}, "image": "python@sha256:" + "c" * 64,
                      "inputs": {"model": model,
                                 "panel": {"repository": "pub/panel", "revision": REV, "path": "panel-x",
                                           "role": "final", "mount_path": str(panel_mount), "kind": "hub"},
                                 "reference": reference2, "candidate": None,
                                 "tokenizer": {"repository": "pub/tok", "revision": REV,
                                               "mount_path": str(root / "tokenizer-mount")}},
                      "output": {"dataset_repository": "tester/qfs-capture-y", "bucket": "tester/qfs-explorer-results",
                                 "prefix": "runs/y", "mount_path": "/outputs"},
                      "hardware": {"device": "cpu", "flavor": "cpu-basic", "timeout_seconds": 600},
                      "runtime": {"dtype": "bfloat16", "schedule": "layer-outer", "trusted_code": None,
                                  "unexpected_allowlist": None},
                      "scope": scope, "codec": "nvfp4", "declared_bits": 4.0, "registered": None}
    out2 = root / "out-candidate"
    out2.mkdir()
    runner2 = StubRunner(out2, {
        "reproduction": ("reproduction/comparison-receipt.json",
                         {"comparison_kind": "reproduction_confirmation",
                          "self_compare": {"force_compute_agreed": True}}),
        "comparison": ("comparison/comparison-receipt.json",
                       {"comparison_kind": "measurement", "gates": {}})})
    outputs2 = {}
    worker.workflow(candidate_plan, out2, runner2, outputs2)
    base = next((a for a in runner2.argv if "--base-capture" in a), None)
    check("W2a candidate capture completes over the canonical reference view", outputs2.get("comparison") is not None)
    check("W2b the base-capture block was passed", base is not None)
    if base:
        payload = json.loads(base[base.index("--base-capture") + 1])
        check("W2c base capture carries the canonical manifest's sealed dataset identity",
              payload["dataset_sha256"] == reference2_manifest["dataset_sha256"]
              and payload["capture_content_digest"] == reference2_manifest["capture"]["capture_content_digest"])


# ---------------------------------------------------------------------------
# EXP-04: inferred attribution is labeled at the explorer layer.
# ---------------------------------------------------------------------------
def rung_attribution_labeling(root):
    tok = "e" * 64
    data = {
        "models": {"m--tiny": {"id": "m--tiny", "name": "Tiny", "family": "tiny-fam", "license": "mit",
                               "huggingface": {"repository": "pub/tiny", "revision": REV},
                               "publisher": {"name": "Publisher", "handle": "publisher",
                                             "url": "https://huggingface.co/publisher"}}},
        "panels": {"p--x": {"id": "p--x",
                            "author": {"name": "PanelAuthor", "handle": "panelauthor",
                                       "url": "https://huggingface.co/panelauthor"},
                            "identity": {"panel_token_sha256": tok},
                            "corpus": {"lineage": "public corpus v1"}}},
        "pipelines": {"pipe--x": {"implementation": {"repository": jobs.SOURCE},
                                  "author": {"name": "Toolchain", "handle": "toolchain",
                                             "url": "https://huggingface.co/toolchain"}}},
        "measurements": {"meas--x": {"model_ref": "m--tiny", "panel_ref": "p--x"}}}

    class FakeRegistry:
        def __init__(self, *a, **k):
            pass

        def registry_data(self):
            return data

    original = jobs.ExplorerRegistry
    jobs.ExplorerRegistry = FakeRegistry
    try:
        capture = root / "capture-root"
        (capture / "first").mkdir(parents=True)
        (capture / "first" / "fidelity-dataset.json").write_text(json.dumps(
            {"weights": {"repository": "pub/tiny", "revision": REV},
             "panel": {"suite_token_hash_sha256": tok}}))
        proof = {"directory": str(capture), "execution": {"job_id": "job_fixture01"},
                 "result": {"outputs": {}},
                 "plan": {"review_metadata": {}, "mode": "root", "owner": "tester",
                          "source": {"repository": jobs.SOURCE, "revision": REV}}}
        meta, inferred = jobs._root_metadata(proof)
        check("A1a inferred attribution fields are all disclosed",
              set(inferred) == {"publisher", "panel_author", "corpus_lineage", "toolchain_author"})
        check("A1b each disclosure states inferred=true with a matching basis",
              all(entry.get("inferred") is True and isinstance(entry.get("basis"), str) and entry["basis"]
                  for entry in inferred.values()))
        check("A1c the flat metadata keeps its frozen published shape",
              set(meta) == {"name", "family", "publisher", "panel_author", "toolchain_author",
                            "corpus_lineage", "model_license"}
              and jobs._validate_review_metadata(meta) == [])
        supplied = {"name": "PanelAuthor", "handle": "panelauthor", "url": "https://huggingface.co/panelauthor"}
        proof2 = {"directory": str(capture), "plan": {"review_metadata": {"panel_author": supplied}, "mode": "root"}}
        meta2, inferred2 = jobs._root_metadata(proof2)
        check("A1d an explicit review_metadata override is authoritative and unlabeled",
              meta2["panel_author"] == supplied and "panel_author" not in inferred2)
        check("A1e explicit saved publication metadata also unlabeled at the publish layer",
              "toolchain_author" not in jobs._attribution_disclosure(
                  inferred, {"toolchain_author": supplied}))
        card = jobs._publication_card(proof, meta, "public", inferred=inferred)
        frozen_block = ("```json\n" + json.dumps(meta, indent=2, ensure_ascii=False).replace("`", "\\u0060")
                        + "\n```").encode()
        check("A1f the card's frozen attribution block is unchanged", frozen_block in card)
        check("A1g the card discloses which attribution values were inferred",
              b"Attribution disclosure" in card and b"panel_author" in card and b"corpus_lineage" in card)
        bare = jobs._publication_card(proof, meta, "public")
        check("A1h no disclosure line when nothing was inferred",
              b"Attribution disclosure" not in bare and frozen_block in bare)
    finally:
        jobs.ExplorerRegistry = original


# ---------------------------------------------------------------------------
# EXP-05: publish_explorer requires an explicit 0600 token file.
# ---------------------------------------------------------------------------
def rung_publish_token_file(root):
    spec = importlib.util.spec_from_file_location(
        "publish_explorer_fixture", ROOT / "bin" / "publish_explorer.py")
    publish = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publish)

    _reset()
    os.environ["HF_TOKEN"] = "hf_env_canary_must_never_be_used"
    try:
        sys.argv = ["publish_explorer.py", "--repo", "tester/qfs-explorer"]
        try:
            publish.main()
            raised = None
        except SystemExit as exc:
            raised = exc
        check("P1a no --token-file refuses with an actionable remedy",
              raised is not None and "--token-file" in str(raised) and "Remedy" in str(raised))
        check("P1b the refusal never constructed an HF client", not RECORD)

        token_path = root / "hf-token-0644"
        token_path.write_text("hf_fixture_token_never_sent_anywhere\n")
        os.chmod(token_path, 0o644)
        sys.argv = ["publish_explorer.py", "--repo", "tester/qfs-explorer", "--token-file", str(token_path)]
        try:
            publish.main()
            raised = None
        except SystemExit as exc:
            raised = exc
        check("P1c a non-0600 token file is refused", raised is not None and "0600" in str(raised))
        check("P1d the refused 0644 file never reached an HF client", not RECORD)

        token_path = root / "hf-token"
        fd = os.open(str(token_path), os.O_WRONLY | os.O_CREAT, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(TOKEN + "\n")
        real_subprocess = publish.subprocess
        publish.subprocess = ns(
            check_output=lambda cmd, *a, **k: (("f" * 40 + "\n") if cmd[:2] == ["git", "rev-parse"]
                                               else (ROOT / "explorer" / cmd[-1].split("explorer/")[-1]).read_bytes()))

        def _entry_not_found(token, *a, **k):
            raise sys.modules["huggingface_hub"].EntryNotFoundError()

        deployment = ROOT / "explorer/deployment.json"
        existed = deployment.exists()
        _reset(create_repo=lambda token, *a, **_: None,
               get_space_runtime=lambda token, *a, **_: ns(hardware="cpu-basic", requested_hardware="cpu-basic"),
               hf_hub_download=_entry_not_found,
               upload_folder=lambda token, *a, **_: ns(oid="e" * 40))
        sys.argv = ["publish_explorer.py", "--repo", "tester/qfs-explorer", "--token-file", str(token_path)]
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                publish.main()
        finally:
            publish.subprocess = real_subprocess
            if not existed and deployment.exists():
                deployment.unlink()
        clients = [r["token"] for r in RECORD if r["method"] == "__init__"]
        check("P1e the 0600 token file is the only credential source", clients == [TOKEN])
        check("P1f the token is never echoed", TOKEN not in captured.getvalue())
    finally:
        os.environ.pop("HF_TOKEN", None)


# ---------------------------------------------------------------------------
# EXP-06: stale-CREATING reconciliation is bounded, fail-closed and visible.
# ---------------------------------------------------------------------------
def _ledger_doc(runs):
    return _canonical({"schema": "qfs.hf-job-ledger.v1", "owner": "tester", "runs": runs})


def _launch_plan(actor, source, image):
    plan = {"schema": "qfs.hf-workflow-plan.v1", "workflow_id": secrets.token_hex(16),
            "owner": actor.username, "mode": "compare", "created_at": "2026-09-08T00:00:00+00:00",
            "source": source, "image": image,
            "inputs": {"model": None, "panel": None, "reference": None, "candidate": None, "tokenizer": None},
            "output": {"dataset_repository": actor.username + "/qfs-capture-x",
                       "bucket": actor.username + "/qfs-explorer-results", "prefix": "runs/x",
                       "mount_path": "/outputs"},
            "hardware": {"flavor": "cpu-basic", "device": "cpu", "timeout_seconds": 60,
                         "unit_cost_micro_usd": 167, "quote_time": time.time()}}
    plan = jobs.seal(plan, "plan_sha256")
    return {"plan": plan, "ticket": jobs._ticket(plan)}


def rung_stale_reconciliation(actor, root):
    source_fixture = {"repository": jobs.SOURCE, "revision": "f" * 40,
                      "worker_sha256": "a" * 64, "bootstrap_sha256": "b" * 64}
    image_fixture = "python@sha256:" + "c" * 64
    real_identity = jobs._source_identity
    jobs._source_identity = lambda: (source_fixture, image_fixture)
    stale_run = {"state": "CREATING", "plan": {"workflow_id": STALE_WID},
                 "reserved_at": "2026-09-07T00:00:00+00:00"}
    saved_ledgers = []

    def ledger_responses(workflow_query):
        def list_jobs(token, *args, **kwargs):
            labels = kwargs.get("labels") or {}
            if "qfs_workflow_id" in labels:
                return workflow_query(labels["qfs_workflow_id"])
            return []
        paths, download = _serve({"ledger.json": _ledger_doc({STALE_WID: stale_run})})
        return {
            "list_jobs_hardware": lambda token, *a, **_: [{"name": "cpu-basic", "unit_label": "minute",
                                                           "unit_cost_micro_usd": 167, "accelerator": None}],
            "repo_exists": lambda token, *a, **_: True,
            "repo_info": lambda token, *a, **kwargs: ns(sha="d" * 40, private=True),
            "get_paths_info": paths, "hf_hub_download": download,
            "upload_file": lambda token, *a, **kwargs: (saved_ledgers.append(json.loads(kwargs["path_or_fileobj"]))
                                                        or ns(oid="e" * 40)),
            "list_jobs": list_jobs,
            "create_bucket": lambda token, *a, **_: None,
            "bucket_info": lambda token, *a, **_: ns(private=True),
            "batch_bucket_files": lambda token, *a, **_: None,
            "run_job": lambda token, **kwargs: ns(
                id="job_fixture01", url="https://huggingface.co/jobs/tester/job_fixture01",
                status=ns(stage="SCHEDULING", message=None), flavor="cpu-basic", created_at=None,
                labels={"qfs_app": "explorer", "qfs_workflow_id": kwargs["env"]["QFS_WORKFLOW_ID"]}),
            "inspect_job": lambda token, **kwargs: ns(
                id="job_fixture01", url="https://huggingface.co/jobs/tester/job_fixture01",
                status=ns(stage="COMPLETED", message=None), flavor="cpu-basic", created_at=None,
                owner=ns(name="tester"), labels={"qfs_app": "explorer", "qfs_workflow_id": STALE_WID}),
        }

    try:
        # Absence proof clears the stale reservation and the launch proceeds.
        _reset(**ledger_responses(lambda wid: []))
        prepared = _launch_plan(actor, source_fixture, image_fixture)
        result = jobs.launch(actor, prepared, confirm_compute=True)
        check("S1a provider-side absence proof clears the stale CREATING reservation",
              result["job_id"] == "job_fixture01")
        wid = prepared["plan"]["workflow_id"]
        check("S1b the reconciled ledger records RECONCILED_ABSENT, not a silent delete",
              saved_ledgers and saved_ledgers[-1]["runs"][STALE_WID]["state"] == "RECONCILED_ABSENT"
              and saved_ledgers[-1]["runs"][wid]["job_id"] == "job_fixture01")

        # A listing error is doubt: keep refusing, name the reconciliation.
        _reset(**ledger_responses(lambda wid: (_ for _ in ()).throw(OSError("listing unavailable"))))
        saved_before = len(saved_ledgers)
        prepared = _launch_plan(actor, source_fixture, image_fixture)
        try:
            jobs.launch(actor, prepared, confirm_compute=True)
            raised = None
        except jobs.JobsError as exc:
            raised = exc
        check("S1c an unprovable reservation keeps refusing with the reconciliation remedy",
              raised is not None and "still unresolved" in str(raised)
              and "positively confirms no Job exists" in str(raised))
        check("S1d doubt never cleared anything and never submitted a paid create",
              len(saved_ledgers) == saved_before
              and not [r for r in RECORD if r["method"] == "run_job"])

        # A Job that actually exists for the stale workflow is not absence: refuse.
        _reset(**ledger_responses(lambda wid: [ns(id="job_stale01")]))
        prepared = _launch_plan(actor, source_fixture, image_fixture)
        check("S1e provider-side presence keeps refusing",
              refuses(lambda: jobs.launch(actor, prepared, confirm_compute=True), jobs.JobsError))
        check("S1f presence never cleared the reservation",
              not [r for r in RECORD if r["method"] == "run_job"])

        # Stale reservations surface as a distinct status in list_runs and inspect.
        _reset(**ledger_responses(lambda wid: []))
        runs = jobs.list_runs(actor)
        stale = [r for r in runs if r.get("status") == "UNRESOLVED_CREATING"]
        check("S1g list_runs surfaces the stale reservation as a distinct status",
              len(stale) == 1 and stale[0]["workflow_id"] == STALE_WID)
        inspected = jobs.inspect(actor, "job_fixture01")
        check("S1h inspect names the unresolved reservation", inspected["unresolved_reservations"] == [STALE_WID])
    finally:
        jobs._source_identity = real_identity


def main():
    with tempfile.TemporaryDirectory(prefix="qfs-selftest-explorer-") as td:
        root = Path(td)
        actor = Actor("tester", TOKEN)
        try:
            rung_anonymous_first(actor)
            rung_public_evidence(actor)
            rung_worker_canonical_views(root)
            rung_attribution_labeling(root)
            rung_publish_token_file(root)
            rung_stale_reconciliation(actor, root)
        except AssertionError as exc:
            print("selftest_explorer_jobs: FAIL: %s" % exc)
            return 1
        except Exception as exc:  # a stubbed-Hub rung must never traceback
            print("selftest_explorer_jobs: FAIL: unexpected %s: %s" % (type(exc).__name__, exc))
            return 1
    print("selftest_explorer_jobs: all explorer job-safety rungs passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
