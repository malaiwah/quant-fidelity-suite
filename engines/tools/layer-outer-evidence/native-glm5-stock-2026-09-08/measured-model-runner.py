#!/usr/bin/env python3
"""Observe complete, unmodified EXL3 GLM5-next text forwards, not qualification.

Requires a NEW fixture from engines/tools/build_glm5_native_fixture.py, Python
3.12, exllamav3 1.4.8+cu128.torch2.11.0, Torch 2.11.0/CUDA 12.8, FLA and Triton.
The caller must independently verify the release wheel digest below. This tool
records installed code hashes; a package version is not proof of wheel origin.

Exit 0 means all requested executions and finite comparisons completed. It does
NOT mean reference equality or determinism. Exit 1 writes failure evidence;
existing output files are never replaced. Run separate fresh processes with new
--out paths for cold comparisons. No vision, MTP or production-model claim.

Native API basis: turboderp-org/exllamav3 at SOURCE_COMMIT, model/model.py,
model/model_ls.py, cache/cache.py, cache/recurrent_util.py,
modules/gated_delta_net.py and architecture/glm5_next.py. GDNState.reset() only
changes position; fresh allocation, not reset(), clears its backing storage.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import shutil
import tempfile

SOURCE_COMMIT = "6ff3a17ea7f3d0026b273d43239398d57f71b788"
WHEEL_SHA256 = "7134a38e6584aac4f0668c8bdfefe430e0f42bfb3ff2fd6f65249a1e1d405779"
FIXTURE_ID = "glm5-next-native-aligned128-random-bf16-v1"
PATHS = ("uncached", "cached_full", "forward_decode", "prefill_decode")
LAYERS = ["linear_attention"] * 3 + ["deepseek_sparse_attention", "linear_attention"]
TORCH = None


class Refusal(RuntimeError):
    """A controlled, path-free evidence error."""


def require(condition, message):
    if not condition:
        raise Refusal(message)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def fixture_inputs(root, report):
    manifest = read_json(root / "fixture-manifest.json")
    require(manifest.get("schema") == "qfs.glm5-native-fixture.v1", "Unsupported fixture manifest schema")
    require(manifest.get("fixture_id") == FIXTURE_ID, "Requires NEW KDA128 fixture identity, not published tiny fixture")
    files = manifest.get("files")
    require(isinstance(files, dict) and bool(files), "Missing fixture file inventory")
    required = {"config.json", "processor_config.json", "inputs.json", "reference.safetensors",
                "reference-repeat.safetensors", "reference-metadata.json"}
    require(required <= files.keys(), "Fixture artifact coverage is incomplete")
    for name, entry in files.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() == name,
                "Unsafe fixture inventory path")
        path = root / relative
        require(path.is_file() and not path.is_symlink() and root in path.resolve().parents,
                "Missing or indirect fixture artifact")
        require(path.stat().st_size == entry["size_bytes"] and sha256_file(path) == entry["sha256"],
                "Fixture artifact content hash mismatch")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == set(files) | {"fixture-manifest.json"}, "Unmanifested files in fixture directory")
    cfg = read_json(root / "config.json")
    tc = cfg["text_config"]
    expected = {"num_hidden_layers": 5, "hidden_size": 128, "vocab_size": 266,
                "max_position_embeddings": 256, "n_routed_experts": 8,
                "num_experts_per_tok": 2, "hc_mult": 4, "num_nextn_predict_layers": 0,
                "qk_rope_head_dim": 0, "q_lora_rank": 128, "kv_lora_rank": 128,
                "qk_nope_head_dim": 128, "v_head_dim": 128, "index_head_dim": 128,
                "intermediate_size": 256, "moe_intermediate_size": 128}
    for key, value in expected.items():
        require(tc.get(key) == value, "Unsupported fixture text config: " + key)
    require(cfg.get("architectures") == ["Glm5NextForConditionalGeneration"], "Unsupported model architecture")
    require(not cfg.get("auto_map") and not tc.get("auto_map"), "Remote model code is forbidden")
    require(cfg.get("tie_word_embeddings") is False, "Full untied vocabulary head required")
    require(tc.get("layer_types") == LAYERS, "All five hybrid blocks must be preserved")
    mlps = tc.get("mlp_layer_types", ["dense" if n < tc.get("first_k_dense_replace", 3) else "sparse" for n in range(5)])
    require(mlps == ["dense"] * 3 + ["sparse"] * 2, "Dense/dense/dense/MoE/MoE schedule required")
    la = tc["linear_attn_config"]
    require(la.get("head_dim") == 128 and la.get("num_heads") == 4,
            "Stock gdn.cu channelwise decode requires key/value head_dim128; D16 is unsupported")
    require(tc.get("index_topk") == 8 and tc.get("index_kpool") == 4, "Sealed sparse-index geometry required")
    panel = read_json(root / "inputs.json")
    expected_panel = {"a": [[1] + list(range(10, 73))], "b": [[1] + list(reversed(range(10, 73))) ]}
    require(panel == expected_panel, "Input panel differs from sealed A/B contract")
    require(all(type(t) is int for batch in panel.values() for row in batch for t in row), "Panel must contain integer IDs")
    report["fixture"] = {"fixture_id": FIXTURE_ID, "manifest_sha256": sha256_file(root / "fixture-manifest.json"),
                         "files": files, "original_published_fixture_qualification": False,
                         "text_config": {k: tc[k] for k in expected}, "layer_types": LAYERS,
                         "mlp_layer_types": mlps, "linear_attn_config": la}
    return panel, files


def installed_identity():
    versions = {name: metadata.version(name) for name in
                ("exllamav3", "torch", "flash-linear-attention", "triton", "safetensors")}
    require(versions["exllamav3"].split("+")[0] == "1.4.8", "Requires exllamav3 1.4.8")
    require(versions["torch"].split("+")[0] == "2.11.0", "Requires Torch 2.11.0")
    require(sys.version_info[:2] == (3, 12), "Pinned release wheel requires Python 3.12")
    dist = metadata.distribution("exllamav3")
    hashes = {}
    for entry in dist.files or []:
        name = entry.as_posix()
        if (name.startswith("exllamav3/") or name.startswith("exllamav3_ext")) and name.endswith((".py", ".so")):
            require(".." not in entry.parts, "Unsafe installed native package entry")
            digest = sha256_file(Path(dist.locate_file(entry)))
            require(entry.hash is not None and entry.hash.mode == "sha256", "Native installed file lacks SHA256 RECORD identity")
            expected = base64.urlsafe_b64decode(entry.hash.value + "=" * (-len(entry.hash.value) % 4)).hex()
            require(digest == expected, "Installed EXL3 code differs from wheel RECORD; patched runtime refused")
            hashes[name] = digest
    require("exllamav3/architecture/glm5_next.py" in hashes and any(n.endswith(".so") for n in hashes),
            "Incomplete native installation identity")
    return {"versions": versions, "python": platform.python_version(), "platform": platform.system(),
            "machine": platform.machine(), "expected_source_commit": SOURCE_COMMIT,
            "expected_wheel_sha256": WHEEL_SHA256, "wheel_origin_verified_by_runner": False,
            "wheel_verification_requirement": "Caller must verify release wheel SHA256 externally",
            "installed_exllamav3_code_sha256": hashes, "installed_RECORD_matches": True,
            "source_patches_allowed": False}


def runtime_device(device, runtime):
    torch = TORCH
    require(torch.__version__.split("+")[0] == "2.11.0" and torch.version.cuda == "12.8",
            "Requires actual Torch 2.11.0 with CUDA 12.8 build")
    require(re.fullmatch(r"cuda:[0-9]+", device) is not None, "Explicit CUDA ordinal required")
    require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    runtime.update({"torch_import_version": torch.__version__, "torch_cuda": torch.version.cuda,
                    "device": device, "device_name": props.name,
                    "compute_capability": [props.major, props.minor], "total_memory_bytes": props.total_memory,
                    "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()})
    env = {}
    # Never serialize arbitrary environment values, tokens, or directory names.
    for name in ("CUDA_LAUNCH_BLOCKING", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
                 "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "TRITON_INTERPRET", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED"):
        value = os.environ.get(name)
        if value is not None:
            env[name] = value if re.fullmatch(r"[0-9:]+", value) else "present_non_numeric_redacted"
    runtime["kernel_environment"] = env
    runtime["additional_kernel_environment_names"] = sorted(n for n in os.environ if
        n.startswith(("EXL", "FLA_", "TRITON_", "CUDA_", "TORCH_")) and n not in env)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10, check=True)
        drivers = sorted(set(result.stdout.split()))
        require(bool(drivers) and all(re.fullmatch(r"[0-9.]+", d) for d in drivers), "Invalid driver version response")
        runtime["driver_versions"] = drivers
    except (OSError, subprocess.SubprocessError):
        raise Refusal("Cannot record NVIDIA driver version") from None


def tensor_bytes(tensor):
    return tensor.contiguous().view(TORCH.uint8).numpy().tobytes()


def checked_logits(value, positions, width=266):
    torch = TORCH
    require(isinstance(value, torch.Tensor), "Forward did not return an actual logits tensor")
    require(tuple(value.shape) == (1, positions, width), "Full-vocabulary logits shape is incomplete")
    require(value.is_floating_point(), "Logits must be floating point")
    value = value.detach().to("cpu").contiguous().clone()
    require(bool(torch.isfinite(value).all().item()), "Nonfinite logits refused")
    return value


def compare(reference, candidate):
    torch = TORCH
    require(reference.shape == candidate.shape, "Comparison shape mismatch")
    r, c = reference.to(torch.float64), candidate.to(torch.float64)
    log_r, log_c = torch.log_softmax(r, dim=-1), torch.log_softmax(c, dim=-1)
    # No clipping, renormalization, noise-floor subtraction, or acceptance threshold.
    kl = (log_r.exp() * (log_r - log_c)).sum(dim=-1)
    delta = (r - c).abs()
    result = {"byte_equal": reference.dtype == candidate.dtype and tensor_bytes(reference) == tensor_bytes(candidate),
              "value_equal": bool(torch.equal(r, c)),
              "different_values": int(torch.count_nonzero(r != c).item()),
              "max_abs_difference": delta.max().item(), "mean_abs_difference": delta.mean().item(),
              "kl_reference_to_candidate_nats_mean": kl.mean().item(),
              "kl_reference_to_candidate_nats_min": kl.min().item(),
              "kl_reference_to_candidate_nats_max": kl.max().item(),
              "kl_reference_to_candidate_nats_per_position": kl.squeeze(0).tolist(),
              "positions": r.shape[1], "vocabulary": r.shape[2], "compute_dtype": "cpu_float64"}
    require(all(math.isfinite(v) for v in result.values() if type(v) is float) and
            all(math.isfinite(v) for v in result["kl_reference_to_candidate_nats_per_position"]),
            "Nonfinite comparison statistic refused")
    return result


def native_inventory(model, cache):
    require(type(model).__name__ == "Glm5NextModel", "Native text model class substitution refused")
    require(not model.config.layer_map and len(model.fwd_modules) == len(model.modules), "Layer remapping refused")
    require(all(m is model.modules[i] and instance == 0 and idx == i
                for i, (m, instance, idx) in enumerate(model.fwd_modules)), "Native forward module coverage hole")
    blocks = [m for m in model.modules if m.layer_idx is not None]
    require([m.layer_idx for m in blocks] == list(range(5)), "Incomplete five-block model")
    require(model.modules[-1].key == "lm_head" and model.modules[-1].out_features_unpadded == 266,
            "Complete untied vocabulary head not present")
    require(len(cache.layers) == 1 and len(cache.recurrent_layers) == 4, "Incomplete MLA/KDA cache layer coverage")
    require(all(type(layer).__name__ == "CacheLayer_MLA_fp16" for layer in cache.layers.values()),
            "Requires real unquantized MLA latent cache")
    return {"class": type(model).__name__, "modules": [{"key": m.key, "class": type(m).__name__,
            "layer_idx": m.layer_idx} for m in model.modules],
            "cache_layers": [type(l).__name__ for l in cache.layers.values()],
            "recurrent_layers": [type(l).__name__ for l in cache.recurrent_layers.values()],
            "num_experts": model.config.num_experts, "experts_per_token": model.config.num_experts_per_tok,
            "expert_coverage_scope": "All 8 experts loaded per MoE block; no claim that this panel routes through every expert",
            "cache_capacity": 256, "max_batch_size": 1, "max_history": 0}


def run_case(model, cache, ids, kind, call_records):
    torch = TORCH
    if kind == "uncached":
        value = checked_logits(model.forward(ids, {"attn_mode": "flash_attn_nc"}), 64, model.modules[-1].out_features)
        call_records.append({"method": "forward", "past_len": None, "input_tokens": 64, "scored_tokens": 64})
        return value
    require(list(cache.free_list) == [0], "Leaked or duplicated recurrent state slot before case")
    states = None
    params = None
    outputs = []
    try:
        chunks = [(0, 64)] if kind == "cached_full" else [(0, 32)] + [(n, n + 1) for n in range(32, 64)]
        for start, stop in chunks:
            params = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, 256), "past_len": start}
            if states is not None:
                require(len(states) == 1 and states[0].position == start, "Recurrent state position mismatch before call")
                params["recurrent_states"] = states
            is_prefill = kind == "prefill_decode" and start == 0
            try:
                result = (model.prefill if is_prefill else model.forward)(ids[:, start:stop], params)
            finally:
                # Allocation occurs inside prepare_inputs; capture even on a failed forward.
                states = params.get("recurrent_states", states)
            require(isinstance(states, list) and len(states) == 1 and states[0].position == stop,
                    "Native recurrent state did not advance to the consumed position")
            if is_prefill:
                require(result is None, "Pinned LS prefill return contract changed")
            else:
                outputs.append(checked_logits(result, stop - start, model.modules[-1].out_features))
            call_records.append({"method": "prefill" if is_prefill else "forward", "past_len": start,
                                 "input_tokens": stop - start, "scored_tokens": 0 if is_prefill else stop - start,
                                 "recurrent_position_after": states[0].position, "slot": states[0].slot})
        return checked_logits(torch.cat(outputs, dim=1), 32 if kind == "prefill_decode" else 64,
                              model.modules[-1].out_features)
    finally:
        if states is not None:
            for state in states:
                state.free()
        require(list(cache.free_list) == [0], "Recurrent states not freed exactly once")
        # Only after freeing active states. This does not clear storage; next p0
        # allocation calls get_new_state(clear=True). MLA valid lengths start at0.
        cache.reset_states()


def execute(args, report, tensors):
    global TORCH
    root = args.model_dir.resolve()
    report["stage"] = "fixture_identity"
    panel, files = fixture_inputs(root, report)
    report["stage"] = "runtime_identity"
    report["runtime"] = installed_identity()
    import torch
    TORCH = torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    # Read CPU artifacts before native construction or GPU allocation.
    refs = load_file(str(root / "reference.safetensors"), device="cpu")
    repeats = load_file(str(root / "reference-repeat.safetensors"), device="cpu")
    expected_keys = {p + "_" + k for p in ("a", "b") for k in ("uncached", "cached")}
    require(set(refs) == expected_keys and set(repeats) == expected_keys, "Incomplete independent reference coverage")
    for key in expected_keys:
        refs[key] = checked_logits(refs[key], 64)
        repeats[key] = checked_logits(repeats[key], 64)
    report["reference_repeatability"] = {k: compare(refs[k], repeats[k]) for k in sorted(expected_keys)}
    report["reference_metadata_sha256"] = files["reference-metadata.json"]["sha256"]
    checkpoint_files = [n for n in files if n.endswith(".safetensors") and n not in
                        ("reference.safetensors", "reference-repeat.safetensors")]
    require(bool(checkpoint_files), "No checkpoint shards")
    tensor_keys = set()
    for name in checkpoint_files:
        with safe_open(str(root / name), framework="pt", device="cpu") as shard:
            keys = set(shard.keys())
            require(not tensor_keys.intersection(keys), "Duplicate checkpoint tensor keys")
            tensor_keys.update(keys)
    require("lm_head.weight" in tensor_keys, "Missing independent full vocabulary head weights")
    for layer in (3, 4):
        for expert in range(8):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
                require(key in tensor_keys, "Incomplete serialized MoE expert set")
    runtime_device(args.device, report["runtime"])
    report["stage"] = "native_construction"
    from exllamav3 import Config, Model, Cache
    import exllamav3
    package_init = metadata.distribution("exllamav3").locate_file("exllamav3/__init__.py")
    require(Path(exllamav3.__file__).resolve() == Path(package_init).resolve(),
            "Imported EXL3 shadows the verified installed distribution")
    view = tempfile.TemporaryDirectory(prefix="glm5-native-weights-", dir=args.out.parent)
    model = None
    try:
        view_root = Path(view.name)
        view_names = checkpoint_files + ["config.json", "generation_config.json",
                                         "processor_config.json", "tokenizer.json", "tokenizer_config.json"]
        for name in view_names:
            shutil.copy2(root / name, view_root / name)
            require(sha256_file(view_root / name) == files[name]["sha256"], "Checkpoint view changed artifact bytes")
        report["native_checkpoint_view"] = {"files": sorted(view_names),
            "reference_logits_excluded": True, "artifact_bytes_unchanged": True}
        config = Config.from_directory(str(view_root))
        model = Model.from_config(config, component="text")
        # Attach real cache layers before native loading.
        cache = Cache(model, max_num_tokens=256, max_batch_size=1, max_history=0)
        report["native_model"] = native_inventory(model, cache)
        report["stage"] = "native_load"
        model.load(device=args.device, tensor_p=False, progressbar=False,
                   max_chunk_size=256, max_output_size=256, max_batch_size=1)
        require(cache.initialized and not model.loaded_tp, "Native load did not initialize single-device cache")
        report["native_model"]["load_completed"] = True
        report["cases"] = {}
        ids = {k: torch.tensor(v, dtype=torch.long) for k, v in panel.items()}
        with torch.inference_mode():
            for repeat in range(2):
                # Interleaving A and B dirties shared MLA/recurrent storage before
                # a same-path repeat, exercising fresh-state isolation.
                for kind in PATHS:
                    for panel_key in ("a", "b"):
                        name = f"{panel_key}_{kind}_r{repeat}"
                        report["stage"] = name
                        calls = []
                        report["cases"][name] = {"calls": calls, "complete": False,
                            "panel": panel_key, "path": kind, "repeat": repeat,
                            "scored_token_indices": list(range(32 if kind == "prefill_decode" else 0, 64))}
                        raw = run_case(model, cache, ids[panel_key], kind, calls)
                        # Linear pads266 to384 at this pin; keep the actual returned
                        # tensor and score every real vocabulary entry, no padding.
                        value = checked_logits(raw[:, :, :266], raw.shape[1])
                        tensors[name + "__native_raw"] = raw
                        torch.cuda.synchronize(args.device)
                        tensors[name] = value
                        first = 32 if kind == "prefill_decode" else 0
                        comparisons = {ref_path: compare(refs[f"{panel_key}_{ref_path}"][:, first:, :], value)
                                       for ref_path in ("uncached", "cached")}
                        report["cases"][name].update({"complete": True, "reference_comparisons": comparisons,
                            "shape": list(value.shape), "dtype": str(value.dtype),
                            "native_raw_shape": list(raw.shape),
                            "native_raw_tensor_key": name + "__native_raw",
                            "native_raw_tensor_bytes_sha256": hashlib.sha256(tensor_bytes(raw)).hexdigest(),
                            "scored_vocabulary_slice": [0, 266],
                            "tensor_bytes_sha256": hashlib.sha256(tensor_bytes(value)).hexdigest()})
        expected_cases = {f"{p}_{k}_r{r}" for p in ("a", "b") for k in PATHS for r in range(2)}
        expected_tensor_keys = expected_cases | {n + "__native_raw" for n in expected_cases}
        require(set(tensors) == expected_tensor_keys and all(v["complete"] for v in report["cases"].values()),
                "Requested case coverage incomplete")
        report["same_path_reset_repeatability"] = {f"{p}_{k}": compare(tensors[f"{p}_{k}_r0"], tensors[f"{p}_{k}_r1"])
            for p in ("a", "b") for k in PATHS}
        report["raw_return_tensor_reset_byte_equality"] = {
            f"{p}_{k}": (
                tensors[f"{p}_{k}_r0__native_raw"].dtype == tensors[f"{p}_{k}_r1__native_raw"].dtype
                and tensor_bytes(tensors[f"{p}_{k}_r0__native_raw"]) == tensor_bytes(tensors[f"{p}_{k}_r1__native_raw"])
            ) for p in ("a", "b") for k in PATHS
        }
        report["cross_path_comparisons"] = {f"{p}_{k}_r{r}": compare(
            tensors[f"{p}_uncached_r{r}"][:, 32 if k == "prefill_decode" else 0:, :], tensors[f"{p}_{k}_r{r}"])
            for p in ("a", "b") for k in PATHS[1:] for r in range(2)}
        report["observed_same_path_byte_repeatable"] = (
            all(v["byte_equal"] for v in report["same_path_reset_repeatability"].values())
            and all(report["raw_return_tensor_reset_byte_equality"].values())
        )
        report["observed_reference_value_equality_all_comparisons"] = all(
            c["value_equal"] for case in report["cases"].values() for c in case["reference_comparisons"].values())
        report["observed_reference_byte_equality_all_comparisons"] = all(
            c["byte_equal"] for case in report["cases"].values() for c in case["reference_comparisons"].values())
        report["execution_complete"] = True
        report["stage"] = "complete"
    finally:
        try:
            if model is not None:
                model.unload()
        finally:
            view.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New JSON file; companion <stem>.native.safetensors is also new")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    require(args.out.suffix == ".json", "--out must name a new .json file")
    tensor_path = args.out.with_name(args.out.stem + ".native.safetensors")
    require(not args.out.exists() and not tensor_path.exists(), "Output artifact already exists")
    require(args.out.parent.is_dir(), "Output parent directory must already exist")
    require(args.model_dir.resolve() not in args.out.resolve().parents, "Evidence must be outside immutable fixture directory")
    report = {"schema": "qfs.glm5-native-forward.v1", "started_utc": datetime.now(timezone.utc).isoformat(),
              "execution_complete": False, "observed_same_path_byte_repeatable": False,
              "observed_reference_value_equality_all_comparisons": False,
              "observed_reference_byte_equality_all_comparisons": False,
              "whole_model_reference_equivalence_qualified": False, "native_determinism_qualified": False,
              "cold_process_repeatability_observed": False, "vision_qualified": False,
              "original_published_fixture_qualified": False,
              "interpretation": "Execution, exact observed equality, and qualification are separate. No tolerance is invented. "
                                "One process cannot establish cold determinism; all results concern this new random fixture only.",
              "runner_sha256": sha256_file(Path(__file__)), "stage": "initialization"}
    tensors = {}
    started = time.monotonic()
    # Reserve both new output names before expensive work; never clobber an artifact.
    with args.out.open("x", encoding="utf-8") as output:
        try:
            with tensor_path.open("xb") as tensor_output:
                try:
                    execute(args, report, tensors)
                except Exception as error:
                    report["execution_complete"] = False
                    message = str(error).replace(str(args.model_dir.resolve()), "<fixture>")
                    message = re.sub(r"/(?:home|Users|private|tmp)/[^\s:'\"]+", "<private-path>", message)
                    report["error"] = {"type": type(error).__name__, "message": message[:4000]}
                    print(type(error).__name__ + ": " + message[:4000], file=sys.stderr)
                if tensors:
                    from safetensors.torch import save
                    tensor_output.write(save(tensors, metadata={"schema": "qfs.glm5-native-logits.v1",
                        "fixture_manifest_sha256": report["fixture"]["manifest_sha256"]}))
            report["native_tensor_artifact"] = {"filename": tensor_path.name, "sha256": sha256_file(tensor_path),
                "size_bytes": tensor_path.stat().st_size, "keys": sorted(tensors), "valid_safetensors": bool(tensors)}
        except Exception as error:
            report["execution_complete"] = False
            report["artifact_error_type"] = type(error).__name__
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({"execution_complete": report["execution_complete"], "stage": report["stage"],
                      "same_path_byte_repeatable": report["observed_same_path_byte_repeatable"],
                      "reference_value_equal": report["observed_reference_value_equality_all_comparisons"]}))
    return 0 if report["execution_complete"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refusal as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
