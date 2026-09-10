#!/usr/bin/env python3
"""Offline behavior regressions for sealed Jobs budgets; no Hub/GPU/paid work."""
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bin"))
from explorer import job_resources as R
from explorer import jobs
from explorer.auth import Actor
from explorer import job_worker


def refuses(function, text=None):
    try:
        function()
    except ValueError as exc:
        if text and text not in str(exc):
            raise AssertionError("wrong refusal: " + str(exc)) from exc
        return
    raise AssertionError("operation did not refuse")


def fixture():
    config = {"hidden_size": 1, "vocab_size": 2,
              "text_config": {"hidden_size": 5120, "vocab_size": 248320}}
    tensors = {"lm_head.weight": {"shape": [248320, 5120]},
               "model.embed_tokens.weight": {"shape": [248320, 5120]},
               "model.layers.0.mlp.weight": {"shape": [17408, 5120]}}
    model = {"files": [{"bytes": 54_000_000_000}], "resource_geometry": R.model_geometry(config, tensors)}
    binding = {"panel": {"contexts": 512, "context_length": 2048, "scored_positions_total": 512 * 2047},
               "tokenizer": {"vocab_size": 248320, "files_verified": True},
               "content": {"manifest": [{"bytes": 12_000_000}]}}
    hardware = {"device": "cuda", "ram": "142 GB", "ephemeral_storage": "1000 GB",
                "accelerator": {"type": "gpu", "quantity": "1", "vram": "80 GB"}}
    return model, binding, hardware


#: The reviewed rented battery and which suites take the pipeline tree as a
#: flag, pinned here instead of derived from the implementation constants, so
#: an unreviewed edit to either list fails this rung rather than following it.
SELFTEST_BATTERY = (("engines/tools/selftest_exl3hf_offline.py", False),
                    ("engines/tools/selftest_trellis_decode_offline.py", False),
                    ("engines/tools/selftest_gguf_offline.py", True),
                    ("engines/tools/selftest_nvfp4_offline.py", True))


class RecordingRunner:
    """Record the exact battery invocation; never execute a suite."""

    def __init__(self):
        self.environment = {"PATH": "/usr/bin"}
        self.calls = []

    def run(self, step, arguments, *, allowed=(0,)):
        self.calls.append({"step": step, "argv": [str(item) for item in arguments],
                           "environment": dict(self.environment)})


@contextlib.contextmanager
def torch_visibility(available):
    """State CUDA visibility without a device, and without importing Torch here.

    The controller battery must stay importable on stock Python with no tensor
    stack; only `selftest_environment`'s narrow Torch surface is stood in for.
    """
    module = sys.modules.get("torch")
    substitute = module is None
    if substitute:
        module = types.ModuleType("torch")
        module.__version__ = "0.0.0+fixture"
        module.cuda = types.SimpleNamespace(is_available=lambda: False,
                                            get_device_name=lambda index: "",
                                            get_device_capability=lambda index: (0, 0))
        sys.modules["torch"] = module
    try:
        with patch.object(module.cuda, "is_available", return_value=available), \
                patch.object(module.cuda, "get_device_name", return_value="Fixture Device"), \
                patch.object(module.cuda, "get_device_capability", return_value=(9, 0)):
            yield
    finally:
        if substitute:
            del sys.modules["torch"]


def battery_image(root, *, pipeline=True, oracle=False, omit=()):
    """A temporary immutable-source tree and image closure; no real suite bytes."""
    source, image = root / "source", root / "image"
    for suite, _ in SELFTEST_BATTERY:
        if suite in omit:
            continue
        path = source / suite
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env python3\n# stand-in for " + suite + "\n")
    package = image / "pipeline/src/quant_pipeline"
    if pipeline:
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
    else:
        image.mkdir(parents=True)
    if oracle:
        (image / "exllamav3").mkdir()
    return source, image


def expected_battery(source, image, *, native):
    """The exact reviewed argv, receipt fields and digests for one sealed plan."""
    records = []
    for suite, flagged in SELFTEST_BATTERY:
        path = source / suite
        argv = [sys.executable, str(path)]
        if flagged:
            argv.extend(["--pipeline-root", str(image / "pipeline")])
        oracle = native and suite.endswith("selftest_exl3hf_offline.py")
        if oracle:
            argv.append("--require-live-native")
        step = "selftest-" + Path(suite).stem
        records.append({"suite": suite, "step": step, "argv": argv,
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "log": step + ".log", "native_oracle_required": oracle})
    return records


def selftest_worker_stage():
    """The rented battery runs exactly the reviewed suites, or refuses outright."""
    assert tuple(name for name, _ in SELFTEST_BATTERY) == tuple(job_worker.SELFTEST_SUITES)
    assert frozenset(name for name, flagged in SELFTEST_BATTERY if flagged) == job_worker.SELFTEST_PIPELINE_FLAG
    for name, _ in SELFTEST_BATTERY:
        assert (ROOT / name).is_file(), "reviewed suite absent from this checkout: " + name
    cpu_plan = {"mode": "selftest", "hardware": {"device": "cpu"}}
    cuda_plan = {"mode": "selftest", "hardware": {"device": "cuda"}}
    with tempfile.TemporaryDirectory(prefix="qfs-selftest-stage-") as td:
        base = Path(td).resolve()

        def case(label, *, pipeline=True, oracle=False, omit=()):
            root = base / label
            root.mkdir()
            source, image = battery_image(root, pipeline=pipeline, oracle=oracle, omit=omit)
            out = root / "out"
            out.mkdir()
            return source, image, out, RecordingRunner()

        # An image with no importable pipeline can only re-skip the rungs the
        # rental exists to execute; nothing runs and no receipt is written.
        source, image, out, runner = case("no-pipeline", pipeline=False)
        with patch.object(job_worker, "ROOT", source), patch.object(job_worker, "IMAGE_ROOT", image), \
                torch_visibility(False):
            refuses(lambda: job_worker.selftest_stage(cpu_plan, out, runner), "importable quant_pipeline")
        assert runner.calls == [] and not (out / "selftest").exists()

        source, image, out, runner = case("no-device")
        with patch.object(job_worker, "ROOT", source), patch.object(job_worker, "IMAGE_ROOT", image), \
                torch_visibility(False):
            refuses(lambda: job_worker.selftest_stage(cuda_plan, out, runner), "no visible CUDA device")
        assert runner.calls == [] and not (out / "selftest").exists()

        # A suite missing from the immutable source is a refusal, not a shorter
        # battery: the preceding suite has already run when it is discovered.
        source, image, out, runner = case("missing-suite", omit=("engines/tools/selftest_trellis_decode_offline.py",))
        with patch.object(job_worker, "ROOT", source), patch.object(job_worker, "IMAGE_ROOT", image), \
                torch_visibility(False):
            refuses(lambda: job_worker.selftest_stage(cpu_plan, out, runner),
                    "reviewed suite missing from the immutable source: engines/tools/selftest_trellis_decode_offline.py")
        assert [call["step"] for call in runner.calls] == ["selftest-selftest_exl3hf_offline"]
        assert not (out / "selftest/report.json").exists()

        # The sealed plan's device decides, and the image must actually carry
        # the oracle: a cpu plan on a CUDA-visible host must not acquire it,
        # and the reviewed overlay (no /opt/fidelity/exllamav3) must report the
        # gap instead of failing the battery for a known-absent package.
        for label, plan, visible, oracle in (("host", cpu_plan, False, False),
                                             ("device-oracle", cuda_plan, True, True),
                                             ("device-no-oracle", cuda_plan, True, False),
                                             ("host-on-device", cpu_plan, True, True)):
            native = oracle and plan["hardware"]["device"] == "cuda"
            source, image, out, runner = case(label, oracle=oracle)
            with patch.object(job_worker, "ROOT", source), patch.object(job_worker, "IMAGE_ROOT", image), \
                    torch_visibility(visible):
                report = job_worker.selftest_stage(plan, out, runner)
            expected = expected_battery(source, image, native=native)
            assert report["suites"] == expected
            assert [call["step"] for call in runner.calls] == [item["step"] for item in expected]
            assert [call["argv"] for call in runner.calls] == [item["argv"] for item in expected]
            demanded = [item["suite"] for item in expected if "--require-live-native" in item["argv"]]
            assert demanded == (["engines/tools/selftest_exl3hf_offline.py"] if native else [])
            assert [item["suite"] for item in expected if "--pipeline-root" in item["argv"]] == [
                "engines/tools/selftest_gguf_offline.py", "engines/tools/selftest_nvfp4_offline.py"]
            # The two flagless suites must still import the image's pipeline.
            assert runner.environment["QP_PIPELINE_ROOT"] == str(image / "pipeline")
            assert runner.environment["PYTHONPATH"] == str(image / "pipeline/src")
            assert all(call["environment"]["QP_PIPELINE_ROOT"] == str(image / "pipeline")
                       and call["environment"]["PYTHONPATH"] == str(image / "pipeline/src")
                       for call in runner.calls)
            durable = json.loads((out / "selftest/report.json").read_text())
            assert durable == report
            assert durable["schema"] == "qfs.hf-workflow-selftest.v1" and durable["suite_count"] == 4
            assert durable["native_oracle_required"] is native
            # The note describes what ran: demanded-and-executed, absent from
            # the image, or present but not demanded by this plan's device.
            note = durable["native_oracle_note"]
            if native:
                assert "required to execute" in note and "does not exercise" not in note
            else:
                assert "does not exercise the native oracle" in note and "required to execute" not in note
                assert ("does not demand the oracle" in note) is oracle
            environment = durable["environment"]
            assert environment["cuda_available"] is visible and environment["pipeline_present"] is True
            assert environment["exllamav3_present"] is oracle
            assert environment["pipeline_root"] == str(image / "pipeline")
            assert (environment["device_name"] == "Fixture Device") is visible
            for item in durable["suites"]:
                assert item["source_sha256"] == job_worker.digest(source / item["suite"])
    print("PASS rented battery refuses a pipelineless image, an absent CUDA device and a missing reviewed suite")
    print("PASS rented battery runs the four reviewed suites in order with exact flags, environment and digests")


def selftest_output_declaration():
    """A battery result declares the battery receipt and nothing else."""
    with tempfile.TemporaryDirectory(prefix="qfs-selftest-outputs-") as td:
        local = Path(td).resolve()
        (local / "selftest").mkdir()
        report = local / "selftest/report.json"
        report.write_text(json.dumps({"schema": "qfs.hf-workflow-selftest.v1", "suite_count": 4}))
        records = [job_worker.row(report, local)]
        declared = {"selftest": "selftest/report.json"}
        battery = {"mode": "selftest"}
        job_worker._output_coverage(battery, local, declared, records)
        refuses(lambda: job_worker._output_coverage(battery, local, {}, records), "required outputs")
        refuses(lambda: job_worker._output_coverage(battery, local, dict(declared, comparison="comparison/receipt.json"),
                                                    records), "required outputs")
        refuses(lambda: job_worker._output_coverage(battery, local, declared, []), "omits or changes")
        # A measurement plan may not smuggle the battery receipt into its result.
        for mode, expected in (("root", {"first", "repeat", "reproduction"}),
                               ("candidate", {"first", "repeat", "reproduction", "comparison"}),
                               ("compare", {"comparison"})):
            outputs = dict({key: key + "/receipt.json" for key in expected}, **declared)
            refuses(lambda mode=mode, outputs=outputs: job_worker._output_coverage(
                {"mode": mode}, local, outputs, records), "required outputs")
    print("PASS selftest result declares exactly the battery receipt; measurement modes refuse it")


def selftest_mode_resources():
    """The battery reserves a working floor, not a capture-fit claim."""
    margin = 64 * 1024**2
    cuda = {"device": "cuda", "ram": "142 GB", "ephemeral_storage": "1000 GB",
            "accelerator": {"type": "gpu", "quantity": "1", "vram": "80 GB"}}
    cpu = {"device": "cpu", "ram": "16 GB", "ephemeral_storage": "100 GB"}

    def battery(hardware, maximum=margin):
        return R.plan_resources("selftest", None, None, None, None, None, hardware, maximum)

    accelerated = battery(cuda)
    assert accelerated["minimum_output_bytes"] == margin
    assert {key for key, value in accelerated["output_components"].items() if value} == {"metadata_margin_bytes"}
    assert accelerated["cpu_ram_required_bytes"] == 4 * R.GIB
    assert accelerated["gpu_required_bytes"] == 2 * R.GIB
    assert accelerated["capture_required_bytes"] == accelerated["replay_required_bytes"] == 0
    assert accelerated["checkpoint_bytes"] == accelerated["canonical_dataset_bytes"] == 0
    assert accelerated["tokenizer_source_bytes"] == 0 and accelerated["runtime_qualified"] is False
    host = battery(cpu)
    assert host["cpu_ram_required_bytes"] == 4 * R.GIB and host["gpu_required_bytes"] == 0
    assert host["minimum_output_bytes"] == margin
    refuses(lambda: battery(dict(cpu, ram="3 GB")), "CPU RAM")
    refuses(lambda: battery(dict(cuda, accelerator={"type": "gpu", "quantity": "1", "vram": "1 GB"})),
            "single-device VRAM")
    refuses(lambda: battery(cuda, margin - 1), "below")
    print("PASS selftest resources hold the 4 GiB CPU floor, CUDA-only 2 GiB allowance and metadata-only output minimum")


RECOVERY_JOB = "fixture-recovery-job"


@contextlib.contextmanager
def hub_bucket_stub():
    """Stand in only the bucket-listing type; no Hub client is ever contacted."""
    previous = sys.modules.get("huggingface_hub")
    module = types.ModuleType("huggingface_hub")

    class BucketFile:
        def __init__(self, path, size):
            self.path, self.size = path, size

    module.BucketFile = BucketFile
    sys.modules["huggingface_hub"] = module
    try:
        yield BucketFile
    finally:
        if previous is None:
            del sys.modules["huggingface_hub"]
        else:
            sys.modules["huggingface_hub"] = previous


def recovery_tree(mode, artifacts, outputs):
    """A sealed plan and the exact durable tree a finished Job would leave behind."""
    workflow = "b" * 32
    plan = jobs.seal({"schema": "qfs.hf-workflow-plan.v1", "mode": mode, "workflow_id": workflow,
                      "owner": "fixture-owner", "image": "fixture@sha256:" + "a" * 64,
                      "hardware": {"flavor": "l4x1", "timeout_seconds": 900},
                      "output": {"bucket": "fixture-bucket", "prefix": "runs/" + workflow},
                      "source": {"revision": "a" * 40},
                      "limits": {"max_output_bytes": 64 * 1024 * 1024},
                      "plan_sha256": ""}, "plan_sha256")
    files = {"plan.json": json.dumps(plan, indent=2).encode()}
    files.update(artifacts)
    records = [{"path": name, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
               for name, blob in sorted(files.items())]
    result = jobs.seal({"schema": "qfs.hf-workflow-result.v1", "mode": mode, "workflow_id": workflow,
                        "owner": plan["owner"], "plan_sha256": plan["plan_sha256"], "status": "complete",
                        "outputs": outputs, "files": records, "result_sha256": ""}, "result_sha256")
    files["result.json"] = json.dumps(result, indent=2).encode()
    return plan, result, files


def drive_recovery(plan, result, files):
    """Run the real `jobs._fetch_result` over a synthetic bucket transfer.

    Qualification and comparison validation are sentinels, never executed: this
    rung asks which post-transfer path a sealed mode reaches, not whether a
    measurement qualifies. Returns the proof, the reached paths and the ledger row.
    """
    reached = []

    def qualification(directory, sealed, execution, *, suite_root=None):
        reached.append("qualification")
        return {"qualification_path": str(Path(directory) / "qualification.json")}

    def validate_receipt(document):
        reached.append("dsvalidate")
        return types.SimpleNamespace(errors=[])

    def verify_comparison(proof):
        reached.append("comparison")

    job = types.SimpleNamespace(labels={"qfs_workflow_id": plan["workflow_id"]},
                                docker_image=plan["image"], flavor=plan["hardware"]["flavor"],
                                environment={"QFS_PLAN_SHA256": plan["plan_sha256"]},
                                status=types.SimpleNamespace(stage="COMPLETED"),
                                created_at="2026-01-01T00:00:00Z", started_at=None,
                                finished_at=None, durations=None)
    saved = {"plan": plan, "job_id": RECOVERY_JOB}
    actor = Actor(plan["owner"], False)
    attestation = json.dumps({"status": result["status"], "workflow_id": result["workflow_id"],
                              "result_sha256": result["result_sha256"]})
    prefix = plan["output"]["prefix"] + "/outputs/result"

    def download(bucket, entries, **kw):
        for item, target in entries:
            target.write_bytes(files[item.path[len(prefix) + 1:]])

    with tempfile.TemporaryDirectory(prefix="qfs-recovery-") as td, hub_bucket_stub() as BucketFile, \
            patch.object(Actor, "client") as client, \
            patch.object(jobs, "_owned_job", return_value=job), \
            patch.object(jobs, "_ledger", return_value=("repo", "head", {"runs": {plan["workflow_id"]: saved}})), \
            patch.object(jobs, "_verify_provider"), patch.object(jobs, "_save_ledger"), \
            patch.object(jobs, "_verify_comparison", verify_comparison), \
            patch("fidelity.hfjobs.qualify_result", qualification), \
            patch("fidelity.dsvalidate.validate_receipt", validate_receipt), \
            patch.object(R.shutil, "disk_usage", return_value=types.SimpleNamespace(free=100 * R.GIB)):
        client.return_value.list_bucket_tree.return_value = [
            BucketFile(prefix + "/" + name, len(blob)) for name, blob in sorted(files.items())]
        client.return_value.download_bucket_files.side_effect = download
        client.return_value.fetch_job_logs.return_value = [attestation]
        proof = jobs._fetch_result(actor, RECOVERY_JOB, Path(td))
        proof.pop("directory")
    return proof, reached, saved


def refuses_recovery(call, text):
    """Recovery must refuse, and refuse as a refusal: never KeyError/AttributeError."""
    try:
        call()
    except Exception as exc:
        if not isinstance(exc, jobs.JobsError):
            raise AssertionError("recovery raised %s instead of refusing: %s" % (type(exc).__name__, exc)) from exc
        if text not in str(exc):
            raise AssertionError("wrong refusal: " + str(exc)) from exc
        return
    raise AssertionError("recovery accepted a damaged battery receipt")


def selftest_recovery_branch():
    """Battery recovery proves its own receipt and qualifies no measurement."""
    report = {"schema": "qfs.hf-workflow-selftest.v1", "suite_count": 2, "native_oracle_required": False,
              "suites": [{"suite": "engines/tools/selftest_exl3hf_offline.py", "source_sha256": "1" * 64},
                         {"suite": "engines/tools/selftest_gguf_offline.py", "source_sha256": "2" * 64}]}
    declared = {"selftest": "selftest/report.json"}

    def battery(document):
        return recovery_tree("selftest", {"selftest/report.json": json.dumps(document, indent=2).encode()}, declared)

    plan, result, files = battery(report)
    proof, reached, saved = drive_recovery(plan, result, files)
    # The battery receipt is the proof; a battery qualifies nothing and has no
    # comparison to validate, so neither downstream path may be entered.
    assert proof["selftest"] == report
    assert "qualification" not in proof and reached == []
    assert proof["plan"] == plan and proof["result"] == result
    assert proof["execution"]["job_id"] == RECOVERY_JOB and proof["execution"]["status"] == "COMPLETED"
    assert saved["state"] == "VERIFIED" and saved["verified_result_sha256"] == result["result_sha256"]

    # A declared receipt that never arrived is a refusal, not a fabricated pass.
    absent = recovery_tree("selftest", {}, declared)
    refuses_recovery(lambda: drive_recovery(*absent), "JSON evidence is missing")
    damaged = (dict(report, schema="qfs.hf-workflow-selftest.v2"),
               dict(report, suite_count=0, suites=[]),
               dict(report, suite_count=3),
               dict(report, suite_count=3, suites="abc"),
               dict(report, suites=[dict(report["suites"][0], source_sha256="0" * 63 + "z"), report["suites"][1]]))
    for document in damaged:
        refuses_recovery(lambda document=document: drive_recovery(*battery(document)),
                         "not an intact selftest report")

    # Control: the measurement modes still reach their own post-transfer path.
    for mode in ("root", "candidate"):
        proof, reached, _ = drive_recovery(*recovery_tree(
            mode, {"capture/receipt.json": b"{}"}, {"first": "capture/receipt.json"}))
        assert reached == ["qualification"] and "selftest" not in proof
        assert proof["qualification"]["qualification_path"].endswith("qualification.json")
    proof, reached, _ = drive_recovery(*recovery_tree(
        "compare", {"comparison/receipt.json": b"{}"}, {"comparison": "comparison/receipt.json"}))
    assert reached == ["dsvalidate", "comparison"]
    assert "selftest" not in proof and "qualification" not in proof
    print("PASS battery recovery stores the intact receipt, qualifies nothing and refuses a damaged report")
    print("PASS measurement recovery still reaches qualification and comparison validation")


def main():
    model, binding, hardware = fixture()
    def plan(maximum=32 * R.GIB, hw=hardware, mdl=model, panel=binding, reference=None, tokenizer=None):
        return R.plan_resources("root" if reference is None else "candidate", mdl, panel, reference, None, tokenizer, hw, maximum)
    resources = plan()
    assert resources["replay"] == R.replay_policy(hardware)
    assert resources["replay"]["device"] == resources["replay"]["replay_device"] == "cuda"
    assert resources["replay_required_bytes"] >= 128 * 248320 * 8 * 8 + model["resource_geometry"]["head_bf16_bytes"] * 8
    legacy = {"hardware": hardware, "runtime": {}}
    assert R.resolve_replay(legacy) == R.replay_policy({"device": "cpu"})
    for field, value in (("device", "cpu"), ("replay_dtype", "float64"), ("vocab_chunk", 128),
                         ("chunk_positions", True), ("replay_device", "auto")):
        invalid = dict(resources["replay"], **{field: value})
        refuses(lambda invalid=invalid: R.resolve_replay({"hardware": hardware, "runtime": {"replay": invalid}}))
    refuses(lambda: R.replay_policy({"device": "cpu"}, "cuda"), "CPU hardware")
    for value in (0, 3, True, 2.0, "2"):
        refuses(lambda value=value: R.active_job_limit(value), "1..2")
    assert R.active_job_limit() == 1 and R.active_job_limit(2) == 2
    assert 24 * R.GIB < resources["minimum_output_bytes"] < 32 * R.GIB
    refuses(lambda: plan(4 * R.GIB), "below")
    refuses(lambda: plan(resources["minimum_output_bytes"] - 1), "below")
    assert plan(resources["minimum_output_bytes"])["minimum_output_bytes"] == resources["minimum_output_bytes"]
    for value in (0, -1, True, 4.0, "4294967296", float("nan"), float("inf"), 64 * R.GIB + 1):
        refuses(lambda value=value: plan(value), "max_output_bytes")
    assert R.output_limit(64 * R.GIB) == 64 * R.GIB
    assert R.byte_quantity("1.5 TiB", "disk") == 1536 * R.GIB
    assert R.byte_quantity("80 GB", "GPU") == 80_000_000_000
    for value in (None, "80", "4x24 GB", "NaN GB", "-1 GB"):
        refuses(lambda value=value: R.byte_quantity(value, "memory"))
    multi = copy.deepcopy(hardware)
    multi["accelerator"].update(quantity="4", vram="96 GB")
    assert R.hardware_capacity(multi)["gpu_bytes"] == 24_000_000_000
    refuses(lambda: plan(hw=multi), "single-device VRAM")
    multi["accelerator"]["vram"] = "4 × 24 GB"
    assert R.hardware_capacity(multi)["gpu_bytes"] == 24_000_000_000
    for field in ("ram", "ephemeral_storage", "accelerator"):
        missing = copy.deepcopy(hardware)
        missing.pop(field)
        refuses(lambda missing=missing: plan(hw=missing))
    small_ram = dict(hardware, ram="16 GB")
    refuses(lambda: plan(hw=small_ram), "CPU RAM")
    small_disk = dict(hardware, ephemeral_storage="200 GB")
    refuses(lambda: plan(hw=small_disk), "ephemeral disk")
    reference = {"artifact_bytes": 10 * R.GIB}
    tokenizer = {"files": [{"bytes": 5 * R.GIB}]}
    candidate = plan(reference=reference, tokenizer=tokenizer)
    assert candidate["disk_required_bytes"] - resources["disk_required_bytes"] == 30 * R.GIB
    bad_panel = copy.deepcopy(binding)
    bad_panel["tokenizer"]["files_verified"] = False
    refuses(lambda: plan(panel=bad_panel), "Verified panel tokenizer")
    refuses(lambda: R.model_geometry({"text_config": {"hidden_size": 5120, "vocab_size": 248320}},
                                      {"lm_head.weight": {"shape": [128, 5120]}}), "head tensor")
    print("PASS output payload boundaries, nested config/head identity, unit parsing, CPU/GPU separation and disk copies")

    # The consumer accepts an explicit larger cap into a sealed prepared plan;
    # external metadata are the injection seams, never a fake paid provider.
    actor = Actor("budget-fixture", "hf_not_a_real_token", source="test")
    quote = dict(hardware, name="a100-large", hourly_usd="2.50002", unit_cost_micro_usd=41667)
    inventory = json.loads((ROOT / "engines/tools/layer-outer-evidence/qwen38-27b-unexpected-keys.json").read_text())
    provenance = json.loads((ROOT / "engines/tools/layer-outer-evidence/qwen38-27b-unexpected-keys.json.provenance.json").read_text())
    native = dict(model, repository=provenance["repository"], revision=provenance["revision"],
                  config_sha256=provenance["config_sha256"],
                  index_sha256=provenance["index_sha256"],
                  config={"hidden_size": 5120, "vocab_size": 248320})
    inputs = {"preset": "root:qwen38-27b", "max_output_bytes": 32 * R.GIB,
              "max_compute_usd": "6", "timeout_seconds": 7200}
    with patch.object(jobs, "_source_identity", return_value=({"revision": "a" * 40}, "fixture@sha256:" + "a" * 64)), \
         patch.object(jobs, "hardware", return_value=[quote]), \
         patch.object(jobs, "_model_metadata", return_value=native), \
         patch.object(jobs, "_resolve_planning_panel", return_value=binding), \
         patch.object(jobs, "_planning_geometry", return_value=model["resource_geometry"]), \
         patch.object(Actor, "require_lifetime"), patch.object(Actor, "client") as api:
        api.return_value.repo_exists.return_value = False
        prepared = jobs.prepare(actor, inputs, registry=object())
        jobs.verify_seal(prepared["plan"], "plan_sha256")
        assert prepared["plan"]["limits"]["max_output_bytes"] == 32 * R.GIB
        assert prepared["plan"]["runtime"]["replay"] == resources["replay"]
        assert prepared["plan"]["limits"]["max_active_jobs"] == 1
        assert jobs.prepare(actor, dict(inputs, max_active_jobs=2), registry=object())["plan"]["limits"]["max_active_jobs"] == 2
        numpy_plan = jobs.prepare(actor, dict(inputs, replay_device="numpy"), registry=object())["plan"]
        assert numpy_plan["runtime"]["replay"]["replay_device"] == "numpy"
        assert numpy_plan["resources"]["replay"]["device"] == "cpu"
        tampered = copy.deepcopy(prepared["plan"])
        tampered["runtime"]["replay"]["device"] = "cpu"
        refuses(lambda: jobs.verify_seal(tampered, "plan_sha256"), "seal")
        default_inputs = dict(inputs)
        default_inputs.pop("max_output_bytes")
        refuses(lambda: jobs.prepare(actor, default_inputs, registry=object()), "below")
        first_run = dict(inputs, timeout_seconds=14220, max_compute_usd="10")
        roomy = jobs.prepare(actor, first_run, registry=object())["plan"]
        assert roomy["hardware"]["estimated_max_compute_usd"] == "9.958413"
        assert roomy["hardware"]["timeout_seconds"] == 14220
        refuses(lambda: jobs.prepare(actor, dict(first_run, timeout_seconds=14280), registry=object()),
                "above your")
        refuses(lambda: jobs.prepare(actor, dict(first_run, timeout_seconds=86401), registry=object()),
                "deadline")
        api.return_value.run_job.assert_not_called()
        api.return_value.create_bucket.assert_not_called()
        api.return_value.batch_bucket_files.assert_not_called()
        hydrated_preset = next(p for p in jobs.presets() if p["id"] == "candidate:qwen38-27b-k5k6-hydrated")
        hydrated_provenance = json.loads((ROOT / (hydrated_preset["unexpected_allowlist"] + ".provenance.json")).read_text())
        hydrated = dict(native, **{field: hydrated_provenance[field]
                                  for field in ("repository", "revision", "config_sha256", "index_sha256")},
                        config={"hidden_size": 5120, "vocab_size": 248320,
                                "quantization_config": {"quant_method": "exl3", "bits": 4.0, "head_bits": 6}})
        def candidate_metadata(actor, repository, revision, *, mode):
            return dict(hydrated if mode == "candidate" else native, repository=repository, revision=revision)
        def reference_metadata(actor, repository, revision, mount_path):
            jobs._identity(repository, revision)
            return {"repository": repository, "revision": revision, "mount_path": mount_path,
                    "artifact_bytes": 10 * R.GIB,
                    "descriptor": {"weights": {"repository": native["repository"], "revision": native["revision"]}}}
        hydrated_inputs = dict(inputs, preset=hydrated_preset["id"],
                               reference_repository="caller/actual-sealed-reference", reference_revision="b" * 40)
        with patch.object(jobs, "_model_metadata", side_effect=candidate_metadata), \
             patch.object(jobs, "_dataset_metadata", side_effect=reference_metadata), \
             patch.object(jobs, "_registered", return_value=None):
            candidate_plan = jobs.prepare(actor, hydrated_inputs, registry=object())["plan"]
            jobs.verify_seal(candidate_plan, "plan_sha256")
            assert candidate_plan["codec"] == "exl3-mcg" and candidate_plan["declared_bits"] == 4.0
            assert candidate_plan["scope"]["policy"] == "mixed" and candidate_plan["scope"]["head_policy"] == "quantized"
            assignments = {entry["tensor_class"]: entry for entry in candidate_plan["scope"]["assignments"]}
            assert {name: assignments[name]["bits_per_weight"] for name in
                    ("mlp.gate", "mlp.up", "mlp.down", "attn.qkv", "attn.o", "lm_head")} == {
                        "mlp.gate": 5, "mlp.up": 5, "mlp.down": 6, "attn.qkv": 6, "attn.o": 6, "lm_head": 6}
            assert assignments["mtp"]["treatment"] == "quantized" and candidate_plan["scope"]["mtp_included"] is True
            assert candidate_plan["inputs"]["panel"]["path"] == prepared["plan"]["inputs"]["panel"]["path"]
            assert candidate_plan["inputs"]["reference"]["repository"] == hydrated_inputs["reference_repository"]
            assert candidate_plan["inputs"]["reference"]["revision"] == hydrated_inputs["reference_revision"]
            assert candidate_plan["inputs"]["model"]["repository"] == "malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated"
            assert candidate_plan["inputs"]["model"]["revision"] == "853acef0b24961b269cdcf32b1ebb405649b545b"
            unbound_reference = dict(hydrated_inputs)
            unbound_reference.pop("reference_repository");unbound_reference.pop("reference_revision")
            refuses(lambda: jobs.prepare(actor, unbound_reference, registry=object()), "immutable")
            for field, value in (("model_repository", native["repository"]), ("model_revision", native["revision"])):
                refuses(lambda field=field, value=value: jobs.prepare(
                    actor, dict(hydrated_inputs, **{field: value}), registry=object()), "binding mismatch")
        candidate_allow = candidate_plan["runtime"]["unexpected_allowlist"]
        root_allow = prepared["plan"]["runtime"]["unexpected_allowlist"]
        assert candidate_allow["path"] != root_allow["path"]
        # Equal logical names do not authorize the BF16 inventory's foreign provenance.
        assert candidate_allow["canonical_sorted_names_sha256"] == root_allow["canonical_sorted_names_sha256"]
        refuses(lambda: job_worker.vetted_unexpected_inventory(root_allow, hydrated), "binding mismatch")
        refuses(lambda: job_worker.vetted_unexpected_inventory(candidate_allow, native), "binding mismatch")
        for field in ("repository", "revision", "config_sha256", "index_sha256"):
            wrong = dict(hydrated, **{field: "foreign"})
            refuses(lambda wrong=wrong: job_worker.vetted_unexpected_inventory(candidate_allow, wrong), "binding mismatch")
        api.return_value.run_job.assert_not_called()
        api.return_value.create_bucket.assert_not_called()
        api.return_value.batch_bucket_files.assert_not_called()
    print("PASS hydrated candidate keeps mixed scope, requires caller reference, and refuses foreign inventory provenance")
    print("PASS larger explicit cap prepares sealed no-spend plan; omitted cap refuses large payload")

    # A fresh process retains no controller global/config override. Recovery must
    # honor the sealed 32 GiB cap even when that process's default is 4 GiB.
    with tempfile.TemporaryDirectory(prefix="qfs-budget-regression-") as td:
        path = Path(td) / "plan.json"
        path.write_text(json.dumps(prepared["plan"]))
        code = """import json,sys
from unittest.mock import patch
from types import SimpleNamespace
from explorer import jobs,job_resources as r
plan=json.load(open(sys.argv[1]));jobs.verify_seal(plan,'plan_sha256')
assert jobs.MAX_OUTPUT == 4*r.GIB
with patch.object(r.shutil,'disk_usage',return_value=SimpleNamespace(free=100*r.GIB)):
    assert r.retrieval_limit(plan,'.') == 32*r.GIB
with patch.object(r.shutil,'disk_usage',return_value=SimpleNamespace(free=68*r.GIB-1)):
    try:r.retrieval_limit(plan,'.')
    except ValueError:pass
    else:raise AssertionError('recovery accepted insufficient disk')
# Exercise the actual result-fetch transfer gate, with a sparse transport file.
# Five GiB must reach verification under a 32 GiB sealed plan, not be rejected
# by a restarted process's default four GiB. No tensor verification is claimed.
import tempfile,types
from pathlib import Path
from explorer.auth import Actor
hub=types.ModuleType('huggingface_hub')
class BucketFile:pass
hub.BucketFile=BucketFile
sys.modules['huggingface_hub']=hub
entry=BucketFile();entry.path=plan['output']['prefix']+'/outputs/result/capture.bin';entry.size=5*r.GIB
class ReachedVerification(Exception):pass
job=SimpleNamespace(labels={'qfs_workflow_id':plan['workflow_id']},docker_image=plan['image'],
    flavor=plan['hardware']['flavor'],environment={'QFS_PLAN_SHA256':plan['plan_sha256']},
    status=SimpleNamespace(stage='COMPLETED'))
saved={'plan':plan,'job_id':'fixture-job'}
actor=Actor(plan['owner'],False)
with tempfile.TemporaryDirectory() as td, patch.object(Actor,'client') as client:
    client.return_value.list_bucket_tree.return_value=[entry]
    def download(bucket,entries,**kw):
        for item,target in entries:
            with target.open('wb') as stream:stream.truncate(item.size)
    client.return_value.download_bucket_files.side_effect=download
    with patch.object(jobs,'_owned_job',return_value=job), patch.object(jobs,'_ledger',return_value=('', '', {'runs':{plan['workflow_id']:saved}})), patch.object(jobs,'_verify_provider'), patch.object(jobs,'_read_json',side_effect=ReachedVerification), patch.object(r.shutil,'disk_usage',return_value=SimpleNamespace(free=100*r.GIB)):
        try:jobs._fetch_result(actor,'fixture-job',Path(td))
        except ReachedVerification:pass
        else:raise AssertionError('fetch did not reach verification')
        assert (Path(td)/'capture.bin').stat().st_size == 5*r.GIB
"""
        subprocess.run([sys.executable, "-c", code, str(path)], cwd=ROOT,
                       env=dict(os.environ, QFS_MAX_OUTPUT_BYTES=str(4 * R.GIB)), check=True)
    print("PASS cross-process sealed-cap recovery and local disk underflow")
    worker_plan = {"limits": {"max_output_bytes": 32 * R.GIB}, "resources": resources, "hardware": hardware}
    with patch.object(R.shutil, "disk_usage") as disk, patch.object(R, "available_cpu_memory", return_value=R.GIB):
        disk.return_value.free = 2000 * R.GIB
        refuses(lambda: R.check_worker_resources(worker_plan, ".", "."), "CPU/cgroup")
    with patch.object(R.shutil, "disk_usage") as disk:
        disk.return_value.free = 1
        refuses(lambda: R.check_worker_resources(worker_plan, ".", "."), "scratch disk")
    assert R.check_worker_resources({}, ".", ".") is None
    print("PASS actual worker resource refusal and unchanged legacy-plan admission")
    # Exercise the actual worker model-binding consumer, not just plan assembly.
    sys.path.insert(0, str(ROOT / "engines/tools"))
    with tempfile.TemporaryDirectory(prefix="qfs-inventory-binding-") as td:
        root = Path(td)
        mount, out = root / "model", root / "out"
        mount.mkdir();out.mkdir()
        config = {"model_type": "qwen3_5"}
        (mount / "config.json").write_text(json.dumps(config))
        (mount / "model.safetensors").write_bytes(b"synthetic-census-only")
        (mount / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
        model_meta = {"repository": provenance["repository"], "revision": provenance["revision"],
                      "mount_path": str(mount), "config": config,
                      "config_sha256": job_worker.digest(mount / "config.json"),
                      "config_bytes": (mount / "config.json").stat().st_size,
                      "index_sha256": job_worker.digest(mount / "model.safetensors.index.json"),
                      "index_bytes": (mount / "model.safetensors.index.json").stat().st_size,
                      "weight_bytes": (mount / "model.safetensors").stat().st_size,
                      "files": [job_worker.row(path, mount) for path in sorted(mount.iterdir())]}
        model_meta["metadata_files"] = [row for row in model_meta["files"]
                                        if not row["path"].endswith(job_worker.WEIGHT_SUFFIXES)]
        metadata_stage = root / "inputs/datasets/model"
        metadata_stage.mkdir(parents=True)
        for row in model_meta["metadata_files"]:
            (metadata_stage / row["path"]).write_bytes((mount / row["path"]).read_bytes())
        name = "engines/tools/layer-outer-evidence/qwen38-27b-unexpected-keys.json"
        path = root / name;path.parent.mkdir(parents=True)
        document = copy.deepcopy(inventory)
        evidence = dict(provenance, config_sha256=model_meta["config_sha256"], index_sha256=model_meta["index_sha256"])
        Path(str(path) + ".provenance.json").write_text(json.dumps(evidence))
        path.write_text(json.dumps(document))
        (root / "engines/coverage.json").write_text('{"architectures":[]}')
        allow = {"path": name, "artifact_sha256": job_worker.digest(path),
                 "canonical_sorted_names_sha256": jobs.hashlib.sha256(jobs.canonical(sorted(document))).hexdigest()}
        worker_plan = {"mode": "root", "inputs": {"model": model_meta},
                       "runtime": {"unexpected_allowlist": allow, "trusted_code": None}}
        with patch.object(job_worker, "ROOT", root), \
                patch.object(job_worker, "PLAN_PATH", root / "inputs/plan.json"), \
                patch("fidelity.hfjobs.INPUT_DATASET_ROOT", str(root / "canonical")):
            bound = job_worker.model_binding(worker_plan, out)
            assert {p.name: p.read_bytes() for p in bound.iterdir()} == {p.name: p.read_bytes() for p in mount.iterdir()}
            assert (out / "unexpected-tensors.json").read_bytes() == path.read_bytes()
            assert json.loads((out / "unexpected-tensors.provenance.json").read_text()) == evidence
            from engines.tools.hf_capture import load_unexpected_tensor_allowlist
            captured = load_unexpected_tensor_allowlist(str(out / "unexpected-tensors.json"),
                                                       allow["artifact_sha256"], allow["canonical_sorted_names_sha256"])
            assert captured["expected_keys"] == sorted(inventory)
            for field in ("repository", "revision", "config_sha256", "index_sha256"):
                wrong = dict(model_meta, **{field: "foreign"})
                refuses(lambda wrong=wrong: job_worker.vetted_unexpected_inventory(allow, wrong), "binding mismatch")
            refuses(lambda: job_worker.vetted_unexpected_inventory(dict(allow, canonical_sorted_names_sha256="0"*64), model_meta), "capture CLI")
            refuses(lambda: job_worker.vetted_unexpected_inventory(dict(allow, path="engines/coverage.json"), model_meta), "vetted")
            path.write_text(json.dumps({"schema": "qfs.exact-unexpected-tensors.v1", "names": document,
                                        "repository": model_meta["repository"], "revision": model_meta["revision"],
                                        "evidence": evidence}))
            old_format = dict(allow, artifact_sha256=job_worker.digest(path))
            refuses(lambda: job_worker.vetted_unexpected_inventory(old_format, model_meta), "capture CLI")
    print("PASS exact vetted Qwen inventory reaches real worker binding; foreign identities and inventories refuse")
    selftest_mode_resources()
    selftest_worker_stage()
    selftest_output_declaration()
    selftest_recovery_branch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
