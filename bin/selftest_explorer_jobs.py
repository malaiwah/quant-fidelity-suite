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

EXP-07. Results execute and inventory locally, then explicitly checkpoint to the
bucket. Stale bucket listings cannot omit declared outputs or sealed sidecars;
final-byte corruption, interruptions and output limits must never seal success.

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

    panel = (ROOT / "engines/panels/panel--qwen38.malaiwah.suite-v5-shard0-1m/panel.json").read_bytes()
    try:
        review._parse(panel.decode())
        paste_refused = False
    except ValueError:
        paste_refused = True
    check("R2 generic paste limits remain unchanged for a full panel", paste_refused)
    for content, permitted in (
            (panel, True),
            (json.dumps({"rows": [{"tokens": [0] * 128} for _ in range(512)]}).encode(), False)):
        paths, download = _serve({"panel.json": content})
        _reset(repo_info=lambda token, repo, **_: public_info,
               get_paths_info=paths, hf_hub_download=download)
        publication = {"kind": "measurement", "repository": "tester/pub-evidence", "revision": REV,
                       "files": {"panel": {"path": "panel.json", "sha256": hashlib.sha256(content).hexdigest()}}}
        with tempfile.TemporaryDirectory() as td:
            try:
                fetched = review._evidence(actor.client(), publication, Path(td))
                accepted = fetched["panel"] == content
            except ValueError:
                accepted = False
        check("R2 typed panel evidence has bounded capacity and preserves exact bytes",
              accepted is permitted and not _authenticated_reads(TOKEN))


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
# EXP-07: local inventory, explicit durable publication, and bounded failure.
# ---------------------------------------------------------------------------
def rung_worker_result_inventory(root):
    import signal
    spec = importlib.util.spec_from_file_location("qfs_worker_result_fixture", ROOT / "explorer/job_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    real_walk, real_path = os.walk, worker.Path
    capture = b"sealed cold capture fixture\n"
    tokenwise = b"comparison sidecar fixture\n"
    payloads = {}
    outputs = {"first": "first", "repeat": "repeat",
               "reproduction": "reproduction/comparison-receipt.json",
               "comparison": "comparison/comparison-receipt.json",
               "submission": "receipts/tester/submission-receipt.json"}
    for name in ("first", "repeat"):
        payloads[name + "/capture/data.bin"] = capture
        payloads[name + "/checksums.txt"] = (hashlib.sha256(capture).hexdigest() + "  capture/data.bin\n").encode()
        payloads[name + "/fidelity-dataset.json"] = b'{"fixture":"sealed capture closure"}\n'
    for name in ("reproduction", "comparison"):
        payloads[name + "/tokenwise-kld.npy"] = tokenwise
        payloads[name + "/comparison-receipt.json"] = _canonical({
            "comparison_kind": "measurement" if name == "comparison" else "reproduction_confirmation",
            "tokenwise": {"path": "tokenwise-kld.npy", "bytes": len(tokenwise),
                         "sha256": hashlib.sha256(tokenwise).hexdigest()}})
    payloads[outputs["submission"]] = b'{"fixture":"validated submission"}\n'
    # An undeclared nested sidecar defends recursive inventory, not a patch
    # hard-coding the three paths absent from the September private pilot.
    payloads["audit/arbitrary/deep/sidecar.bin"] = b"closure includes every local file\n"

    def attempt(label, fault=None):
        base = root / ("worker-result-" + label)
        mount = base / "outputs"
        mount.mkdir(parents=True)
        destination = mount / "result"
        plan_path = base / "plan.json"
        plan = {"schema": "qfs.hf-workflow-plan.v1", "workflow_id": "3" * 32,
                "owner": "tester", "mode": "candidate", "plan_sha256": "d" * 64,
                "source": {}, "image": "python@sha256:" + "c" * 64,
                "hardware": {"timeout_seconds": 30}, "runtime": {},
                "limits": {"max_output_bytes": 8192 if fault == "budget" else 1024 * 1024}}
        plan_path.write_bytes(_canonical(plan))
        worker.PLAN_PATH, worker.OUT_PATH = plan_path, destination
        # Virtualize only the fixed mount boundary; main's admitted CLI spellings
        # and its real publication/exception/signal paths remain in use.
        worker.Path = lambda value: mount if str(value) == "/outputs" else real_path(value)
        worker.validate_plan = lambda plan, out: None
        worker.require_no_credentials = lambda: None

        def stale_walk(path, *args, **kwargs):
            if real_path(path) == destination or destination in real_path(path).parents:
                # A directory can be directly readable while readdir still
                # reports its pre-publication empty state.
                yield str(path), [], []
                return
            yield from real_walk(path, *args, **kwargs)

        def fixture_workflow(plan, local, runner, declared):
            declared.update(outputs)
            # A real child writes the first sealed artifact; its completed-stage
            # checkpoint must survive later failure or a deadline interrupt.
            early = {key: value for key, value in payloads.items() if key.startswith("first/")}
            code = ("import sys; from pathlib import Path; root=Path(sys.argv[1]); "
                    "files=" + repr(early) + "; "
                    "[( (root/name).parent.mkdir(parents=True,exist_ok=True), "
                    "(root/name).write_bytes(data)) for name,data in files.items()]")
            runner.run("fixture-capture", [sys.executable, "-c", code, str(local)])
            if fault == "interrupt":
                os.kill(os.getpid(), signal.SIGTERM)
            if fault == "child-failure":
                runner.run("fixture-failure", [sys.executable, "-c", "raise SystemExit(17)"])
            for name, data in payloads.items():
                if name in early:
                    continue
                if fault == "missing-declared" and name == outputs["submission"]:
                    continue
                if fault == "missing-sidecar" and name == "comparison/tokenwise-kld.npy":
                    continue
                target = local / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            if fault == "corruption":
                (destination / "first/capture/data.bin").write_bytes(b"changed after checkpoint")
            if fault == "budget":
                (local / "oversized.bin").write_bytes(b"x" * plan["limits"]["max_output_bytes"])
            runner.bound()

        worker.workflow = fixture_workflow
        stdout = io.StringIO()
        previous_started = os.environ.pop("QFS_WORKFLOW_STARTED", None)
        try:
            worker.os.walk = stale_walk
            with contextlib.redirect_stdout(stdout):
                code = worker.main(["--plan", str(plan_path), "--out", str(destination)])
        finally:
            worker.os.walk = real_walk
            worker.Path = real_path
            if previous_started is not None:
                os.environ["QFS_WORKFLOW_STARTED"] = previous_started
        result = json.loads((destination / "result.json").read_bytes())
        return code, result, destination, stdout.getvalue(), plan

    code, result, destination, logged, _ = attempt("stale-listing")
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
    listed = {item["path"] for item in result["files"]}
    check("W3a stale durable enumeration cannot seal an incomplete recursive inventory",
          code == 0 and result["status"] == "complete" and actual == listed | {"result.json"}
          and set(payloads).issubset(listed))
    check("W3b every published file matches the final sealed size and hash",
          all((destination / item["path"]).stat().st_size == item["bytes"]
              and hashlib.sha256((destination / item["path"]).read_bytes()).hexdigest() == item["sha256"]
              for item in result["files"]))
    check("W3c the completion anchor names the durable manifest's self-seal",
          worker.seal(dict(result), "result_sha256")["result_sha256"] == result["result_sha256"]
          and any(json.loads(line).get("result_sha256") == result["result_sha256"]
                  for line in logged.splitlines()))
    for fault in ("missing-declared", "missing-sidecar", "corruption", "child-failure", "interrupt", "budget"):
        code, result, destination, logged, plan = attempt(fault, fault)
        check("W3d %s cannot emit a successful result" % fault,
              code == 1 and result["status"] == "failed"
              and not any(json.loads(line).get("status") == "complete" for line in logged.splitlines()))
        if fault in ("child-failure", "interrupt", "budget"):
            check("W3e %s preserves the completed private capture checkpoint" % fault,
                  (destination / "first/capture/data.bin").read_bytes() == capture)
        if fault == "budget":
            check("W3f failure publication respects the durable byte ceiling",
                  sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
                  <= plan["limits"]["max_output_bytes"])
    # Expiry before publication must refuse without beginning an unbounded copy.
    expired_local, expired_out = root / "expired-local", root / "expired-durable"
    expired_local.mkdir()
    expired_out.mkdir()
    (expired_local / "payload").write_bytes(capture)
    expired = worker._Publication(expired_local, expired_out, 1024, time.monotonic() - 1)
    check("W3g an expired publication deadline retains no success manifest",
          refuses(expired.checkpoint, TimeoutError) and not list(expired_out.iterdir()))

    check("W3h a strict inventory refuses a directory enumeration failure",
          refuses(lambda: list(worker.tree(root / "missing-evidence-tree")), FileNotFoundError))

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


def _launch_plan(actor, source, image, max_active_jobs=1):
    plan = {"schema": "qfs.hf-workflow-plan.v1", "workflow_id": secrets.token_hex(16),
            "owner": actor.username, "mode": "compare", "created_at": "2026-09-08T00:00:00+00:00",
            "source": source, "image": image, "launch_contract": "measurement-cli-v1",
            "runtime": {"replay": jobs.job_resources.replay_policy({"device": "cpu"})},
            "limits": {"max_active_jobs": max_active_jobs},
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
    image_fixture = json.loads((ROOT / "explorer/job_environment.json").read_text())["image"]
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
        def run_job(token, **kwargs):
            # HF rejects image-derived names over 100 characters before create.
            # A digest-pinned measurement image must supply its own short name.
            name = kwargs.get("labels", {}).get("name", kwargs["image"])
            if not isinstance(name, str) or not 1 <= len(name) <= 100:
                raise FakeHTTPError(400)
            return ns(id="job_fixture01", url="https://huggingface.co/jobs/tester/job_fixture01",
                      status=ns(stage="SCHEDULING", message=None), flavor="cpu-basic", created_at=None,
                      labels={"qfs_app": "explorer", "qfs_workflow_id": kwargs["env"]["QFS_WORKFLOW_ID"]})
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
            "run_job": run_job,
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

        for mode, action in (("root", "capture"), ("candidate", "measure"), ("compare", "compare")):
            _reset(**ledger_responses(lambda wid: []))
            prepared = _launch_plan(actor, source_fixture, image_fixture)
            plan = prepared["plan"]
            plan["mode"] = mode
            plan["output"]["prefix"] = "runs/" + plan["workflow_id"]
            plan = jobs.seal(plan, "plan_sha256")
            jobs.launch(actor, {"plan": plan, "ticket": jobs._ticket(plan)}, confirm_compute=True)
            invocation = next(row["kwargs"] for row in RECORD if row["method"] == "run_job")
            check("S1 action command reaches provider for " + mode,
                  invocation["command"] == ["/usr/local/bin/qfs-job", action, "--plan",
                                             "/inputs/plan/plan.json", "--out", "/outputs/result"])
            job = ns(command=invocation["command"], arguments=[], secrets={}, space_id=None,
                     docker_image=invocation["image"], environment=invocation["env"],
                     labels=invocation["labels"], volumes=invocation["volumes"])
            jobs._verify_provider(actor, job, plan)
            for index, wrong in ((0, "/tmp/qfs-job"), (1, "root"), (3, "/tmp/plan.json"),
                                 (5, "/outputs/elsewhere")):
                job.command = list(invocation["command"])
                job.command[index] = wrong
                check("S1 provider refuses changed command field %d for %s" % (index, mode),
                      refuses(lambda: jobs._verify_provider(actor, job, plan), jobs.JobsError))
            job.command = invocation["command"]
            job.docker_image = "python@sha256:" + "0" * 64
            check("S1 CLI recovery refuses changed image",
                  refuses(lambda: jobs._verify_provider(actor, job, plan), jobs.JobsError))
        for contract in ("python-bootstrap-v1", "measurement-venv-v1", "capture"):
            _reset(**ledger_responses(lambda wid: []))
            plan = _launch_plan(actor, source_fixture, image_fixture)["plan"]
            plan["launch_contract"] = contract
            plan = jobs.seal(plan, "plan_sha256")
            check("S1 new launch refuses legacy or alias contract " + contract,
                  refuses(lambda: jobs.launch(actor, {"plan": plan, "ticket": jobs._ticket(plan)},
                                              confirm_compute=True), jobs.JobsError)
                  and not [row for row in RECORD if row["method"] == "run_job"])

        # Default single-job admission, explicitly bounded two-job race, and a
        # third Job refused. No provider resource is created by this fixture.
        for count, maximum, admitted in ((1, 1, False), (1, 2, True), (2, 2, False)):
            responses = ledger_responses(lambda wid: [])
            responses["list_jobs"] = lambda token, count=count, **kwargs: (
                [] if "qfs_workflow_id" in kwargs.get("labels", {}) else
                [ns(id="active-job-%d" % index) for index in range(count)])
            _reset(**responses)
            race = _launch_plan(actor, source_fixture, image_fixture, maximum)
            if admitted:
                jobs.launch(actor, race, confirm_compute=True)
                methods = [row["method"] for row in RECORD]
                check("S2 bounded race reserves before the second create",
                      methods.count("run_job") == 1 and methods.index("upload_file") < methods.index("run_job"))
            else:
                check("S2 active limit %d refuses count %d" % (maximum, count),
                      refuses(lambda: jobs.launch(actor, race, confirm_compute=True), jobs.JobsError)
                      and not [row for row in RECORD if row["method"] == "run_job"])

        responses = ledger_responses(lambda wid: [])
        paths, download = _serve({"ledger.json": _ledger_doc({})})
        responses.update(get_paths_info=paths, hf_hub_download=download)
        responses["upload_file"] = lambda token, **kwargs: (_ for _ in ()).throw(OSError("CAS conflict"))
        _reset(**responses)
        race = _launch_plan(actor, source_fixture, image_fixture, 2)
        check("S2 a losing CAS cannot submit compute",
              refuses(lambda: jobs.launch(actor, race, confirm_compute=True), OSError)
              and not [row for row in RECORD if row["method"] == "run_job"])

        # Even an empty provider listing cannot release a concurrent controller's
        # reserved slot before that controller submits its paid create.
        guarded = dict(stale_run, creation_guard="provider-absence-is-not-abort-proof")
        responses = ledger_responses(lambda wid: [])
        paths, download = _serve({"ledger.json": _ledger_doc({STALE_WID: guarded})})
        responses.update(get_paths_info=paths, hf_hub_download=download)
        _reset(**responses)
        race = _launch_plan(actor, source_fixture, image_fixture, 2)
        check("S2 in-flight guarded reservation cannot be cleared by apparent absence",
              refuses(lambda: jobs.launch(actor, race, confirm_compute=True), jobs.JobsError)
              and not [row for row in RECORD if row["method"] in ("run_job", "upload_file")])

        # A listing error is doubt: keep refusing, name the reconciliation.
        _reset(**ledger_responses(lambda wid: (_ for _ in ()).throw(OSError("listing unavailable"))))
        saved_before = len(saved_ledgers)
        prepared = _launch_plan(actor, source_fixture, image_fixture, 2)
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
        prepared = _launch_plan(actor, source_fixture, image_fixture, 2)
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


def rung_baked_runtime(actor, root):
    from unittest.mock import patch
    from explorer import job_bootstrap as bootstrap
    import container_manifest

    image_root = root / "baked-image"
    (image_root / "suite").mkdir(parents=True)
    (image_root / "patches-v2").mkdir()
    (image_root / "suite/source.py").write_bytes(b"baked source, not worker source")
    (image_root / "patches-v2/SERIES").write_bytes(b"patch fixture")
    freeze = b"torch==2.11.0+cu130\nnumpy==2.5.2\n"
    (image_root / "pip-freeze.txt").write_bytes(freeze)
    build = {"schema": "malaiwah.fidelity-image-build.v1", "suite_revision": "1" * 40,
             "pins": {"python": "3.12.3", "torch_cuda": "13.0"}, "probe_errors": {},
             "bundle_sha256": {"source.py": hashlib.sha256((image_root / "suite/source.py").read_bytes()).hexdigest()},
             "patches_sha256": {"SERIES": hashlib.sha256(b"patch fixture").hexdigest()},
             "pip_freeze_sha256": hashlib.sha256(freeze).hexdigest()}
    material = {key: build[key] for key in ("pins", "patches_sha256", "bundle_sha256", "pip_freeze_sha256", "suite_revision")}
    build["image_content_sha256"] = hashlib.sha256(container_manifest.canonical(material).encode()).hexdigest()
    raw = json.dumps(build).encode()
    (image_root / "BUILD.json").write_bytes(raw)
    (image_root / "image-pin.txt").write_text(build["image_content_sha256"] + "\n")
    environment = {"build_sha256": hashlib.sha256(raw).hexdigest(),
                   "image_content_sha256": build["image_content_sha256"], "baked_source_revision": "1" * 40,
                   "capture_dependencies": {"torch": "2.11.0+cu130", "numpy": "2.5.2"}}
    observed, closure = bootstrap.verify_image(environment, image_root=image_root)
    check("B1 verified baked provenance remains independent of worker checkout",
          observed["suite_revision"] == "1" * 40 and closure == freeze)

    launcher_path = image_root / "qfs-job"
    launcher_path.write_bytes(b"#!/opt/fidelity/venv/bin/python\n")
    launcher = {"schema": "qfs.hf-job-launcher.v1", "launch_contract": "measurement-cli-v1",
                "launcher_path": "/usr/local/bin/qfs-job",
                "launcher_sha256": hashlib.sha256(launcher_path.read_bytes()).hexdigest(),
                "launcher_source_revision": "2" * 40,
                "base_image": "ghcr.io/malaiwah/quant-fidelity-measure@sha256:" + "3" * 64,
                **{key: environment[key] for key in ("build_sha256", "image_content_sha256", "baked_source_revision")}}
    launcher_raw = jobs.canonical(launcher)
    (image_root / "JOBS.json").write_bytes(launcher_raw)
    cli_environment = {**environment, **launcher, "schema": "qfs.hf-job-environment.v1",
                       "interpreter": bootstrap.PYTHON,
                       "image": "ghcr.io/malaiwah/quant-fidelity-measure@sha256:" + "4" * 64,
                       "launcher_manifest_sha256": hashlib.sha256(launcher_raw).hexdigest()}
    verified = bootstrap.verify_launcher(cli_environment, image_root=image_root, launcher_path=launcher_path)
    check("B2 Jobs overlay source is distinct from unchanged baked source",
          verified["launcher_source_revision"] == "2" * 40 and build["suite_revision"] == "1" * 40)
    for field, value in (("launcher_sha256", "0" * 64), ("launcher_manifest_sha256", "0" * 64),
                         ("launcher_source_revision", "0" * 40), ("base_image", "other@sha256:" + "0" * 64),
                         ("build_sha256", "0" * 64), ("launcher_path", "/tmp/qfs-job"),
                         ("interpreter", "/usr/bin/python"), ("launch_contract", "capture")):
        check("B2 launcher refuses changed " + field,
              refuses(lambda: bootstrap.verify_launcher(dict(cli_environment, **{field: value}),
                                                         image_root=image_root, launcher_path=launcher_path), ValueError))
    launcher_path.write_bytes(b"modified launcher")
    check("B2 actual executable tampering refuses",
          refuses(lambda: bootstrap.verify_launcher(cli_environment, image_root=image_root,
                                                     launcher_path=launcher_path), ValueError))
    launcher_path.write_bytes(b"#!/opt/fidelity/venv/bin/python\n")
    (image_root / "JOBS.json").write_bytes(launcher_raw + b" ")
    check("B2 actual manifest tampering refuses",
          refuses(lambda: bootstrap.verify_launcher(cli_environment, image_root=image_root,
                                                     launcher_path=launcher_path), ValueError))
    (image_root / "JOBS.json").write_bytes(launcher_raw)

    deployment_root = root / "cli-deployment"
    deployed = deployment_root / "explorer"
    deployed.mkdir(parents=True)
    for name in ("job_worker.py", "job_bootstrap.py"):
        (deployed / name).write_bytes((ROOT / "explorer" / name).read_bytes())
    (deployed / "job_environment.json").write_bytes(jobs.canonical(cli_environment))
    deployment = {"source_revision": "5" * 40, **{
        field: hashlib.sha256((deployed / name).read_bytes()).hexdigest()
        for name, field in (("job_worker.py", "worker_sha256"), ("job_bootstrap.py", "bootstrap_sha256"),
                            ("job_environment.json", "environment_sha256"))}}
    (deployed / "deployment.json").write_bytes(jobs.canonical(deployment))
    with patch.object(jobs, "ROOT", deployment_root):
        identity, image = jobs._source_identity()
        check("B2 source identity binds the complete CLI environment",
              image == cli_environment["image"] and identity["environment_sha256"] == deployment["environment_sha256"])
        (deployed / "job_environment.json").write_bytes(jobs.canonical(dict(cli_environment, launcher_sha256="0" * 64)))
        check("B2 environment launcher pin mutation invalidates deployed source identity",
              refuses(jobs._source_identity, jobs.JobsError))
        for field, value in (("launch_contract", "measurement-venv-v1"), ("launcher_path", "/tmp/qfs-job"),
                             ("interpreter", "/usr/bin/python"), ("launcher_manifest_sha256", ""),
                             ("launcher_source_revision", "main"), ("base_image", "image:latest")):
            (deployed / "job_environment.json").write_bytes(jobs.canonical(dict(cli_environment, **{field: value})))
            deployment["environment_sha256"] = hashlib.sha256((deployed / "job_environment.json").read_bytes()).hexdigest()
            (deployed / "deployment.json").write_bytes(jobs.canonical(deployment))
            check("B2 deployment refuses unreviewed " + field, refuses(jobs._source_identity, jobs.JobsError))
    for name, replacement in (("BUILD.json", raw + b" "), ("image-pin.txt", b"0" * 64),
                              ("pip-freeze.txt", freeze.replace(b"2.5.2", b"2.5.3")),
                              ("suite/source.py", b"changed baked code"),
                              ("patches-v2/SERIES", b"changed baked patch")):
        path = image_root / name
        original = path.read_bytes()
        path.write_bytes(replacement)
        try:
            check("B2 changed " + name + " refuses before runtime",
                  refuses(lambda: bootstrap.verify_image(environment, image_root=image_root), ValueError))
        finally:
            path.write_bytes(original)

    class Scalar:
        def __add__(self, other):
            return self
        def item(self):
            return 2

    torch = ns(__version__="2.11.0+cu130", version=ns(cuda="13.0"),
               cuda=ns(is_available=lambda: False), float64="fp64",
               ones=lambda *args, **kwargs: Scalar())
    modules = {"torch": torch, "numpy": ns(__version__="2.5.2")}
    wheels = [ns(metadata={"Name": name}, version=version) for name, version in
              (("torch", "2.11.0+cu130"), ("numpy", "2.5.2"), ("pip", "26.0"))]
    with patch.object(sys, "executable", bootstrap.PYTHON), patch.object(sys, "prefix", "/opt/fidelity/venv"), \
            patch.object(sys, "version_info", (3, 12, 3)), \
            patch("importlib.metadata.distributions", return_value=wheels), \
            patch("importlib.import_module", side_effect=lambda name: modules[name]):
        runtime = bootstrap.verify_runtime(environment, build, freeze, "cpu")
        check("B3 CUDA wheel on CPU reports actual unavailable CUDA",
              runtime["device"] == "cpu" and runtime["cuda_available"] is False and runtime["torch_cuda"] == "13.0")
        check("B4 requested missing CUDA fails, never falls back",
              refuses(lambda: bootstrap.verify_runtime(environment, build, freeze, "cuda"), ValueError))
        for attribute, value in (("executable", "/usr/bin/python3.12"), ("prefix", "/usr"),
                                 ("version_info", (3, 11, 9))):
            with patch.object(sys, attribute, value):
                check("B5 wrong baked " + attribute + " refuses",
                      refuses(lambda: bootstrap.verify_runtime(environment, build, freeze, "cpu"), ValueError))
        wheels[1].version = "2.5.3"
        check("B6 mismatched installed dependency fails instead of installation",
              refuses(lambda: bootstrap.verify_runtime(environment, build, freeze, "cpu"), ValueError))
        wheels[1].version = "2.5.2"
        with patch.object(importlib, "import_module", side_effect=RuntimeError("native import failed")):
            check("B7 native import exceptions are failures, not skips",
                  refuses(lambda: bootstrap.verify_runtime(environment, build, freeze, "cpu"), RuntimeError))

    # Recovery authenticates historical provider evidence, even after worker edits.
    # Keep the actual measurement-cli canary as an input independent of the map:
    # deleting its reviewed pin must not silently remove this recovery regression.
    sources = {**jobs._reviewed_job_sources(), "4ba2f4d9bfe52249dc7190ac068abeb4950b9b72": {
        "worker_sha256": "007aa9fb5e2d2e1ea6a46947188d98958d52a5b666341dac9446cb3fa8434ff3",
        "bootstrap_sha256": "ad04862b0d042f9753afa1451f3a5aafb335712210ea601935b44325cab89086",
        "environment_sha256": "79d103191da67a0d344f351eecb9792aa5d7f06635c362df143489e724cc360c",
        "image": "ghcr.io/malaiwah/quant-fidelity-measure@sha256:e03ccb8c67a54fc206ada6092cac426867fc9c2b26554a8887beb48fcf1e0683",
        "launch_contract": "measurement-cli-v1"}}
    for revision, pin in sources.items():
        source = {"repository": jobs.SOURCE, "revision": revision,
                  **{key: pin[key] for key in ("worker_sha256", "bootstrap_sha256", "environment_sha256") if key in pin}}
        plan = _launch_plan(actor, source, pin["image"])["plan"]
        contract = pin.get("launch_contract", "python-bootstrap-v1")
        if contract == "python-bootstrap-v1":
            plan.pop("launch_contract")
        else:
            plan["launch_contract"] = contract
        plan["output"]["prefix"] = "runs/" + plan["workflow_id"]
        plan = jobs.seal(plan, "plan_sha256")
        job = ns(command=jobs._launch_command(contract, plan["mode"]), arguments=[], secrets={}, space_id=None,
                 docker_image=pin["image"],
                 environment={"QFS_PLAN_SHA256": plan["plan_sha256"], "QFS_WORKFLOW_ID": plan["workflow_id"]},
                 labels={"qfs_source": revision, "qfs_workflow_id": plan["workflow_id"]},
                 volumes=[ns(type="bucket", source=plan["output"]["bucket"],
                             path=plan["output"]["prefix"] + "/" + name, mount_path=mount,
                             read_only=readonly, revision=None)
                          for name, mount, readonly in (("inputs", "/inputs/plan", True), ("outputs", "/outputs", False))])
        jobs._verify_provider(actor, job, plan)
        job.command = ["/tmp/arbitrary-python", "-c", jobs._BOOTSTRAP_FETCH]
        check("B8 recovery refuses an arbitrary interpreter for " + revision[:7],
              refuses(lambda: jobs._verify_provider(actor, job, plan), jobs.JobsError))
        job.command = jobs._launch_command(contract, plan["mode"])
        job.docker_image = "python@sha256:" + "0" * 64
        check("B9 recovery refuses a different provider image for " + revision[:7],
              refuses(lambda: jobs._verify_provider(actor, job, plan), jobs.JobsError))


def rung_bootstrap_no_install(root, bootstrap=None):
    """Exercise successful handoff with an immutable checkout and no installer."""
    from unittest.mock import patch
    if bootstrap is None:
        from explorer import job_bootstrap as bootstrap
    base = root / "bootstrap-handoff"
    outputs = base / "outputs"
    outputs.mkdir(parents=True)
    checkout = base / "checkout"
    plan_path = base / "plan.json"
    environment = json.loads((ROOT / "explorer/job_environment.json").read_bytes())
    environment["launch_contract"] = "measurement-cli-v1"
    environment_raw = jobs.canonical(environment)
    worker_raw = b"# immutable worker fixture\n"
    plan = jobs.seal({"schema": "qfs.hf-workflow-plan.v1", "launch_contract": "measurement-cli-v1",
                     "image": environment["image"], "hardware": {"timeout_seconds": 60, "device": "cpu"},
                     "source": {"repository": jobs.SOURCE, "revision": "a" * 40,
                                "worker_sha256": hashlib.sha256(worker_raw).hexdigest(),
                                "environment_sha256": hashlib.sha256(environment_raw).hexdigest()}}, "plan_sha256")
    plan_path.write_bytes(jobs.canonical(plan))
    launched = []
    attempted_installs = []
    remaining_budgets = []
    runtime = {"baked_source_revision": environment["baked_source_revision"],
               "image_content_sha256": environment["image_content_sha256"],
               "installed_versions": {"numpy": "2.5.2", "torch": "2.11.0+cu130"}}

    def mapped_path(value):
        if str(value) == "/inputs/plan/plan.json":
            return plan_path
        if str(value) == "/outputs" or str(value).startswith("/outputs/"):
            return outputs / str(value).removeprefix("/outputs").lstrip("/")
        return Path(value)

    def execute(command, deadline, commands, *, step):
        commands.append({"step": step, "argv": command, "returncode": 0})
        remaining_budgets.append(deadline - time.monotonic())
        if command[1:4] == ["-m", "pip", "install"]:
            attempted_installs.append(command)
            raise RuntimeError("package installation is forbidden in measurement Jobs")
        if step == "initialize-source":
            (checkout / "explorer").mkdir(parents=True)
            (checkout / "explorer/job_worker.py").write_bytes(worker_raw)
            (checkout / "explorer/job_bootstrap.py").write_bytes(Path(bootstrap.__file__).read_bytes())
            (checkout / "explorer/job_environment.json").write_bytes(environment_raw)
        if step == "verify-baked-runtime":
            (outputs / "image-runtime.json").write_bytes(jobs.canonical(runtime))

    with contextlib.ExitStack() as stack:
        for name, value in (("CHECKOUT", checkout), ("LOG", outputs / "bootstrap.log"),
                            ("RECEIPT", outputs / "bootstrap.json"), ("RUNTIME", outputs / "image-runtime.json"),
                            ("Path", mapped_path), ("run", execute)):
            stack.enter_context(patch.object(bootstrap, name, value, create=True))
        stack.enter_context(patch.object(bootstrap, "open", lambda path, mode: open(mapped_path(path), mode), create=True))
        stack.enter_context(patch.object(sys, "executable", "/opt/fidelity/venv/bin/python"))
        stack.enter_context(patch.object(sys, "prefix", "/opt/fidelity/venv"))
        stack.enter_context(patch.object(sys, "version_info", (3, 12, 3)))
        stack.enter_context(patch.object(sys, "path", list(sys.path)))
        stack.enter_context(patch.dict(os.environ, {"QFS_WORKFLOW_STARTED": str(time.time() - 3)}, clear=True))
        stack.enter_context(patch.dict(sys.modules, {"job_worker": ns(require_no_credentials=lambda: None,
                                                                    validate_plan=lambda *args: None)}))
        stack.enter_context(patch.object(os, "sync", lambda: None))
        stack.enter_context(patch.object(os, "execve", lambda path, argv, env: launched.append((path, argv))))
        bootstrap.main(["--plan", "/inputs/plan/plan.json", "--out", "/outputs/result"])
    check("B10 immutable worker handoff succeeds without per-Job installs",
          not attempted_installs and launched and launched[0][0] == "/opt/fidelity/venv/bin/python")
    check("B10 launcher fetch time remains charged to the bootstrap deadline",
          remaining_budgets and all(0 < remaining <= 57 for remaining in remaining_budgets))
    receipt = json.loads((outputs / "bootstrap.json").read_text())
    check("B11 final bootstrap evidence separates installed image runtime and worker source",
          receipt["baked_source_revision"] == environment["baked_source_revision"]
          and receipt["source_revision"] == "a" * 40
          and receipt["installed_versions"]["numpy"] == "2.5.2")


def hf_intake_identity_regression(root):
    """Repeated HF measurements retain old rows and their actual backend identities."""
    import copy
    sys.path.insert(0, str(ROOT / "registry/tools"))
    import review_requests as intake
    from fidelity import common, dscompare

    C = intake.L.load_registry(os.path.join(root, "data"))
    before = copy.deepcopy(C)
    fixture = Path(root) / "protocol/review-requests" / (
        "7717c50133ead6ca569dba9be4ca5fd775627c2750a58afe3f698cb7a131043d")
    request = json.loads((fixture / "request.json").read_text())
    publication = request["publication"]
    docs = {role: json.loads((fixture / (role + ".json")).read_text())
            for role in publication["files"]}
    author = docs["submission"]["measurer"]["handle"]
    first = intake.measurement_records(
        docs, publication, author, C, "receipts/intake-first.json", True)
    first_row = next(row for row in first if row["id"].startswith("measurement--"))
    assert first_row["id"] not in C["measurements"]
    assert first_row["determinism"]["run_count"] == 1
    assert intake.L.has_disclosure(first_row, "reduced_run_count")
    for row in first:
        intake.merge(C, row)

    changed = copy.deepcopy(docs)
    changed_publication = copy.deepcopy(publication)
    comparison = changed["comparison"]
    environment = comparison["comparator"]["replay_env"]
    if comparison["comparator"]["replay_backend"].startswith("numpy:"):
        environment["blas_threads"] = int(environment.get("blas_threads") or 1) + 1
    else:
        environment["device_name"] = "synthetic-other-device"
    comparison = changed["comparison"] = common.seal(comparison)
    submission = changed["submission"]
    submission["estimator"] = dscompare._submission_estimator(comparison)
    changed["submission"] = common.seal(submission)
    for role in ("comparison", "submission"):
        changed_publication["files"][role]["sha256"] = intake.sha(intake.canonical(changed[role]))
    second = intake.measurement_records(
        changed, changed_publication, author, C, "receipts/intake-second.json", True)
    second_row = next(row for row in second if row["id"].startswith("measurement--"))
    assert second_row["id"] != first_row["id"]
    assert second_row["pipeline_ref"] != first_row["pipeline_ref"]
    for row in second:
        intake.merge(C, row)
    assert all(C[collection][key] == record
               for collection, records in before.items()
               for key, record in records.items())


def hf_root_version_regression(root):
    """A pinned native root cannot reassign an unpinned historical model."""
    import copy
    sys.path.insert(0, str(ROOT / "registry/tools"))
    import review_requests as intake
    import community_fixtures as fixtures

    fixture = Path(root) / "protocol/review-requests" / (
        "ccab4f22762da93d7415f03842c4c7e325a987b90498f1e5f0e9ea20fae24b66")
    request = json.loads((fixture / "request.json").read_text())
    publication = request["publication"]
    docs = {role: json.loads((fixture / (role + ".json")).read_text())
            for role in publication["files"]}
    accepted = json.loads((fixture / "records.json").read_text())["records"]
    model = next(row for row in accepted if row["id"].startswith("model--"))
    model_repo = docs["dataset"]["weights"]["repository"]
    model_rev = docs["dataset"]["weights"]["revision"]
    version_id = "model--" + fixtures.slug(model_repo.replace("/", ".")) + "." + model_rev[:12]
    # Only the synthetic registry history changes; accepted evidence stays sealed.
    old_ids = {row["id"]: row["id"] + ".legacy" for row in accepted
               if row["id"].startswith(("artifact--", "reference--", "measurement--"))}

    def historical(value):
        if isinstance(value, dict):
            return {key: historical(item) for key, item in value.items()}
        if isinstance(value, list):
            return [historical(item) for item in value]
        return old_ids.get(value, value) if isinstance(value, str) else value

    for legacy_revision in (None, "0" * 40):
        C = {name: {} for name, _, _ in intake.L.COLLECTIONS}
        for row in historical(accepted):
            intake.merge(C, row)
        legacy = C["models"][model["id"]]
        legacy["huggingface"]["revision"] = legacy_revision
        legacy["tokenizer"]["files_sha256"] = {"tokenizer.json": "0" * 64}
        C["artifacts"][legacy["canonical_weights"]["artifact_ref"]]["huggingface"]["revision"] = legacy_revision
        before = copy.deepcopy(C)
        records = intake.root_records(docs, publication, request["requested_by"], C)
        assert C == before, "Root intake mutated historical records before merge"
        for row in records:
            intake.merge(C, row)
        assert all(C[collection][key] == row
                   for collection, rows in before.items() for key, row in rows.items())
        pinned = C["models"][version_id]
        assert pinned["huggingface"]["revision"] == model_rev
        assert pinned["tokenizer"]["files_sha256"] == model["tokenizer"]["files_sha256"]
        native = C["artifacts"][pinned["canonical_weights"]["artifact_ref"]]
        assert native["model_ref"] == version_id
        reference = next(row for row in records if row["id"].startswith("reference--"))
        assert reference["artifact_ref"] == native["id"]
        assert C["panels"][reference["panel_ref"]]["model_scope"] == [version_id]
        floor = next(row for row in records if row["id"].startswith("measurement--"))
        assert floor["model_ref"] == version_id and floor["reference_ref"] == reference["id"]
        assert floor["panel_ref"] == reference["panel_ref"]
        # Exact pinned identities are reused, rather than allocating another model.
        repeated = intake.root_records(docs, publication, request["requested_by"], C)
        assert not any(row["id"].startswith(("model--", "artifact--", "panel--")) for row in repeated)
        assert next(row for row in repeated if row["id"].startswith("reference--")) == reference
        assert C["models"][model["id"]] == before["models"][model["id"]]

        for target in ("tokenizer", "checkpoint", "config"):
            conflict = copy.deepcopy(C)
            if target == "tokenizer":
                conflict["models"][version_id]["tokenizer"]["files_sha256"]["tokenizer.json"] = "0" * 64
            elif target == "checkpoint":
                hashes = conflict["artifacts"][native["id"]]["weights"]["shard_sha256"]
                hashes[next(iter(hashes))] = "0" * 64
            else:
                conflict["artifacts"][native["id"]]["weights"]["config_sha256"] = "0" * 64
            conflict_before = copy.deepcopy(conflict)
            assert refuses(lambda: intake.root_records(
                docs, publication, request["requested_by"], conflict)), target
            assert conflict == conflict_before
        for collection, identity in (("models", version_id), ("artifacts", native["id"])):
            collision = copy.deepcopy(before)
            row = copy.deepcopy(legacy if collection == "models" else native)
            row["id"] = identity
            row["huggingface"]["revision"] = "0" * 40
            collision[collection][identity] = row
            assert refuses(lambda: intake.root_records(
                docs, publication, request["requested_by"], collision)), collection


def rung_encoded_evidence_credential(root):
    from unittest.mock import patch
    from fidelity import dshub

    member = root / "encoded-bootstrap.json"
    encoded = "".join("\\u%04x" % ord(char) for char in TOKEN)
    member.write_text('{"path":"/tmp/qfs-worker/evidence","secret":"' + encoded + '"}')

    def forbidden_network(*args, **kwargs):
        raise AssertionError("encoded credential reached publication preflight")

    actor = ns(username="tester", client=lambda: ns(token=TOKEN, repo_exists=forbidden_network))
    refused = False
    with patch.dict(sys.modules, {"huggingface_hub": ns(CommitOperationAdd=object)}):
        try:
            jobs._upload_tree(actor, "tester/evidence", {"bootstrap.json": member},
                              private=False, state={}, persist=forbidden_network)
        except dshub.HubError:
            refused = True
    check("P2 encoded credentials refuse even when original worker paths are retained", refused)


def rung_owner_acceptance_public_check(root):
    from unittest.mock import patch
    from explorer import review

    for private in (False, True):
        directory = root / ("owner-accept-private" if private else "owner-accept-public")
        stage = directory / "registry"
        stage.mkdir(parents=True)
        raw = b'{}\n'
        (stage / "staged.json").write_bytes(raw)
        commits = []
        api = ns(repo_info=lambda *a, **k: ns(sha=REV),
                 create_commit=lambda *a, **k: (commits.append(k) or ns(oid="b" * 40, commit_url="https://example.invalid/commit")),
                 change_discussion_status=lambda *a, **k: None)
        actor = ns(username="malaiwah")
        ticket = "a" * 32
        state = {"actor": actor.username, "expires_at": time.time() + 300,
                 "discussion_id": 1, "digest": "c" * 64, "head": REV,
                 "directory": str(directory), "changes": {"staged.json": review._sha(raw)},
                 "preview": {"warnings": []}}
        _reset(repo_info=lambda token, repo, **_: ns(sha=REV, private=private, gated=False))
        refused = False
        with patch.object(review, "_identity", return_value=api), \
                patch.object(review, "_request", return_value=(None, None, "c" * 64)), \
                patch.dict(review._TICKETS, {ticket: state}, clear=True):
            try:
                result = review.accept_request(actor, ticket, confirm_accept=True)
            except ValueError:
                refused = True
        reads = [row for row in RECORD if row["method"] == "repo_info"]
        check("I3 owner acceptance checks public visibility anonymously",
              reads and all(row["token"] is False for row in reads))
        check("I3 private registry refuses; public registry commits under inspected parent",
              (refused and not commits) if private else
              (not refused and len(commits) == 1 and commits[0]["parent_commit"] == REV
               and result["independently_verified"] is False))


def main():
    with tempfile.TemporaryDirectory(prefix="qfs-selftest-explorer-") as td:
        root = Path(td)
        actor = Actor("tester", TOKEN)
        try:
            rung_anonymous_first(actor)
            rung_public_evidence(actor)
            rung_worker_canonical_views(root)
            rung_worker_result_inventory(root)
            rung_attribution_labeling(root)
            rung_publish_token_file(root)
            rung_stale_reconciliation(actor, root)
            rung_baked_runtime(actor, root)
            rung_bootstrap_no_install(root)
            hf_intake_identity_regression(ROOT / "registry")
            print("  PASS  I1 repeated HF intake preserves existing rows and backend identities")
            hf_root_version_regression(ROOT / "registry")
            print("  PASS  I2 native root versions preserve historical identities and refuse pin conflicts")
            rung_encoded_evidence_credential(root)
            rung_owner_acceptance_public_check(root)
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
