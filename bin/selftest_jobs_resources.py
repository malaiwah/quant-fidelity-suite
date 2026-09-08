#!/usr/bin/env python3
"""Offline behavior regressions for sealed Jobs budgets; no Hub/GPU/paid work."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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


def main():
    model, binding, hardware = fixture()
    def plan(maximum=32 * R.GIB, hw=hardware, mdl=model, panel=binding, reference=None, tokenizer=None):
        return R.plan_resources("root" if reference is None else "candidate", mdl, panel, reference, None, tokenizer, hw, maximum)
    resources = plan()
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
                      "index_sha256": job_worker.digest(mount / "model.safetensors.index.json"),
                      "index_bytes": (mount / "model.safetensors.index.json").stat().st_size,
                      "weight_bytes": (mount / "model.safetensors").stat().st_size,
                      "files": [job_worker.row(path, mount) for path in sorted(mount.iterdir())]}
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
        with patch.object(job_worker, "ROOT", root):
            assert job_worker.model_binding(worker_plan, out) == mount
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
