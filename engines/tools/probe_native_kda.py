#!/usr/bin/env python3
"""Bounded stock/ordered CUDA KDA repeatability experiment; no tolerance gates.

Run --out NEW_JSON [--ordered] in the pinned native CUDA environment. Every
invocation is real CUDA; the independent CPU FP64 recurrence is diagnostic only.
Actual inputs, every output/state, and FP64 references accompany the JSON in
NEW_JSON.safetensors. Neither stock package files nor old artifacts are changed.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import json
import math
from pathlib import Path
import platform
import sys

COMMIT = "6ff3a17ea7f3d0026b273d43239398d57f71b788"
SOURCE_BASE = f"https://raw.githubusercontent.com/turboderp-org/exllamav3/{COMMIT}/"
SOURCE_PINS = {
    "exllamav3/modules/gated_delta_net_fn/gated_delta_rule.py":
        "82e057e970e5b8d92a5d76ce5878db0a978e8dad9fc3c1444bdb1479c4206758",
    "exllamav3/exllamav3_ext/gdn.cu":
        "7efcb8c6836e7d2ed0689bd0faa2c7792cbc6c85b744abcef4452daa631b5b30",
}
REPEATS = 64
CASES = ("mixed_sign_mild_decay", "mixed_sign_medium_decay", "mixed_sign_strong_decay")
SIGNATURE = ["mixed_qkv", "g", "beta", "recurrent_state", "out",
             "num_k_heads", "num_v_heads", "k_head_dim", "v_head_dim",
             "recurrent_slots", "history"]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": sha256(path)}


def raw_bytes(tensor) -> bytes:
    import torch
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_record(tensor) -> dict:
    import torch
    raw = raw_bytes(tensor)
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "elements": tensor.numel(), "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "nonfinite_elements": (int((~torch.isfinite(tensor)).sum().item())
                                   if tensor.is_floating_point() else 0)}


def stock_backend():
    """Refuse implicit JIT and mismatched source; identify the actual wheel binary."""
    distribution = importlib.metadata.distribution("exllamav3")
    if distribution.version != "1.4.8+cu128.torch2.11.0":
        raise RuntimeError(f"Require the pinned 1.4.8+cu128.torch2.11.0 wheel, got {distribution.version}")
    package_spec = importlib.util.find_spec("exllamav3")
    if package_spec is None or not package_spec.origin:
        raise RuntimeError("Cannot locate installed exllamav3 package")
    package_root = Path(package_spec.origin).resolve().parent
    sources = {}
    for relative, expected in SOURCE_PINS.items():
        path = package_root / relative.removeprefix("exllamav3/")
        source_record = file_record(path)
        if source_record["sha256"] != expected:
            raise RuntimeError(f"Installed source bytes differ from pinned commit: {path}")
        sources[relative] = {**source_record, "pinned_url": SOURCE_BASE + relative}
    spec = importlib.util.find_spec("exllamav3_ext")
    if spec is None or not spec.origin or not any(
            spec.origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        raise RuntimeError("Require installed native exllamav3_ext binary; refusing stock JIT")
    ext = importlib.import_module("exllamav3_ext")
    binary_path = Path(ext.__file__).resolve()
    if binary_path != Path(spec.origin).resolve():
        raise RuntimeError("Loaded native binary differs from resolved import spec")
    function = ext.cuda_recurrent_gated_delta_rule
    if not inspect.isbuiltin(function) or function.__module__ != ext.__name__:
        raise RuntimeError("Stock binding is not the expected native builtin")
    # RECORD links the loaded bytes to the installed wheel, not just its version string.
    record_entry = next((entry for entry in distribution.files or ()
                         if Path(distribution.locate_file(entry)).resolve() == binary_path), None)
    if record_entry is None or record_entry.hash is None or record_entry.hash.mode != "sha256":
        raise RuntimeError("Native extension has no SHA256 entry in the installed wheel RECORD")
    binary = file_record(binary_path)
    record_digest = base64.urlsafe_b64encode(bytes.fromhex(binary["sha256"])).decode().rstrip("=")
    if record_digest != record_entry.hash.value:
        raise RuntimeError("Loaded stock extension bytes differ from installed wheel RECORD")
    direct_url = distribution.read_text("direct_url.json")
    metadata = {
        "runtime_identity": "stock_exllamav3_releasev1.4.8_native_cuda",
        "changed_runtime": False, "distribution_version": distribution.version,
        "pinned_source_commit": COMMIT, "source_identity_verified": sources,
        "source_hash_normalization": "none; exact upstream response bytes and installed wheel bytes",
        "binary": binary, "wheel_record_entry": str(record_entry),
        "wheel_record_sha256_verified": True,
        "direct_url": json.loads(direct_url) if direct_url else None,
        "native_module": ext.__name__, "function": function.__name__,
        "binding_doc": function.__doc__, "ordinary_positional_signature": SIGNATURE,
        "identity_limit": "Source and wheel RECORD verified; no claim of reproducible binary build from source.",
    }
    return function, metadata


def fixed_inputs(case_index: int) -> dict:
    """Integer arithmetic on CPU, power-of-two scaling, then explicit dtype casts."""
    import torch

    def signed(shape, multiplier, offset, modulus, denominator, dtype):
        index = torch.arange(math.prod(shape), dtype=torch.int64, device="cpu")
        # Odd numerators exclude zero; the signed pattern also excludes all-positive reductions.
        numerator = 2 * ((index * multiplier + offset) % modulus) - (modulus - 1)
        return (numerator.to(torch.float64) / denominator).to(dtype).reshape(shape).contiguous()

    offset = 29 + 113 * case_index
    q = signed((1, 1, 4, 128), 37, offset, 512, 512, torch.bfloat16)
    k = signed((1, 1, 4, 128), 73, offset + 61, 512, 512, torch.bfloat16)
    v = signed((1, 1, 4, 128), 97, offset + 131, 512, 256, torch.bfloat16)
    state = signed((1, 1, 4, 128, 128), 157, offset + 211,
                   1024, 4096, torch.float32)
    g_index = torch.arange(512, dtype=torch.int64).reshape(1, 1, 4, 128)
    g = (-(((g_index * 19 + offset) % 251) + 1).to(torch.float64)
         / (4096, 512, 64)[case_index]).to(torch.float32)
    beta = (torch.tensor([31, 101, 173, 239], dtype=torch.float64)
            .roll(case_index) / 256).to(torch.bfloat16).reshape(1, 1, 4)
    return {"mixed_qkv": torch.cat([q.flatten(2), k.flatten(2), v.flatten(2)], dim=-1),
            "g": g, "beta": beta, "recurrent_state": state,
            "recurrent_slots": torch.tensor([0], dtype=torch.int32)}


def mathematical_reference(inputs: dict) -> dict:
    """FP64 mathematical KDA, independent of CUDA reductions and BF16 RZ store."""
    import torch
    q, k, v = [part.reshape(1, 1, 4, 128).to(torch.float64)
               for part in inputs["mixed_qkv"].split(512, dim=-1)]
    q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    decayed = (inputs["recurrent_state"][:, 0].to(torch.float64)
               * inputs["g"][:, 0].to(torch.float64).exp().unsqueeze(-1))
    prediction = (decayed * k[:, 0].unsqueeze(-1)).sum(dim=-2)
    delta = ((v[:, 0] - prediction)
             * inputs["beta"][:, 0].to(torch.float64).unsqueeze(-1))
    state = decayed + k[:, 0].unsqueeze(-1) * delta.unsqueeze(-2)
    out = (state * q[:, 0].unsqueeze(-1)).sum(dim=-2) / math.sqrt(128)
    return {"out": out.unsqueeze(1).contiguous(),
            "recurrent_state": state.unsqueeze(1).contiguous()}


def error_metrics(actual, reference) -> dict:
    import torch
    a, b = actual.to(torch.float64), reference.to(torch.float64)
    finite = torch.isfinite(a) & torch.isfinite(b)
    finite_count = int(finite.sum().item())
    result = {"elements": a.numel(), "finite_pairs": finite_count,
              "nonfinite_pairs": a.numel() - finite_count}
    if finite_count:
        difference = (a[finite] - b[finite]).abs()
        result.update(max_abs=float(difference.max().item()),
                      mean_abs=float(difference.mean().item()),
                      rms=float(difference.square().mean().sqrt().item()))
    else:
        result.update(max_abs=None, mean_abs=None, rms=None)
    return result


def execute_backend(name, function, metadata, cases, references, tensors):
    import torch
    case_reports = {}
    with torch.inference_mode():
        for case_name, cpu_inputs in cases.items():
            base = {key: value.to("cuda:0") for key, value in cpu_inputs.items()}
            expected_bytes = {key: raw_bytes(value) for key, value in cpu_inputs.items()}
            first_bytes = None
            first_values = None
            repeats = []
            groups = {"out": {}, "recurrent_state": {}, "joint": {}}
            for repeat in range(REPEATS):
                live = {key: value.clone() for key, value in base.items()}
                before_bytes = {key: raw_bytes(value) for key, value in live.items()}
                if before_bytes != expected_bytes:
                    raise RuntimeError(f"Fresh input/state bytes differ: {name}/{case_name}/{repeat}")
                # Poison every output element so a missing native store is observable.
                out = torch.full((1, 1, 4, 128), float("nan"), dtype=torch.bfloat16, device="cuda:0")
                function(live["mixed_qkv"], live["g"], live["beta"],
                         live["recurrent_state"], out, 4, 4, 128, 128,
                         live["recurrent_slots"], False)
                torch.cuda.synchronize(0)
                values = {"out": out.cpu().contiguous(),
                          "recurrent_state": live["recurrent_state"].cpu().contiguous()}
                result_bytes = {key: raw_bytes(value) for key, value in values.items()}
                if first_bytes is None:
                    first_bytes, first_values = result_bytes, values
                records, keys = {}, {}
                for key, value in values.items():
                    tensor_key = f"{name}.{case_name}.repeat_{repeat:03d}.{key}"
                    tensors[tensor_key] = value
                    keys[key] = tensor_key
                    records[key] = tensor_record(value)
                    groups[key].setdefault(records[key]["sha256"], []).append(repeat)
                joint = hashlib.sha256(result_bytes["out"] + result_bytes["recurrent_state"]).hexdigest()
                groups["joint"].setdefault(joint, []).append(repeat)
                readonly = {key: raw_bytes(value) == expected_bytes[key]
                            for key, value in live.items() if key != "recurrent_state"}
                repeats.append({
                    "repeat": repeat, "fresh_inputs_and_initial_state_byte_identical": True,
                    "readonly_inputs_unchanged": readonly, "tensors": keys,
                    "result_records": records, "joint_sha256": joint,
                    "exact_bytes_equal_first": {key: result_bytes[key] == first_bytes[key]
                                                for key in values},
                    "delta_vs_first": {key: error_metrics(value, first_values[key])
                                       for key, value in values.items()},
                    "fp64_reference_error": {key: error_metrics(value, references[case_name][key])
                                             for key, value in values.items()},
                })
            all_equal = {key: all(row["exact_bytes_equal_first"][key] for row in repeats)
                         for key in ("out", "recurrent_state")}
            finite = all(record["nonfinite_elements"] == 0 for row in repeats
                         for record in row["result_records"].values())
            readonly_unchanged = all(all(row["readonly_inputs_unchanged"].values()) for row in repeats)
            case_reports[case_name] = {
                "invocations": len(repeats), "input_tensor_prefix": f"inputs.{case_name}",
                "all_output_bytes_equal": all_equal["out"],
                "all_final_state_bytes_equal": all_equal["recurrent_state"],
                "joint_byte_repeatability_observed": all(all_equal.values()),
                "all_results_finite": finite, "all_readonly_inputs_unchanged": readonly_unchanged,
                "unique_result_counts": {key: len(value) for key, value in groups.items()},
                "unique_result_hashes_and_repeat_indices": groups, "repeats": repeats,
            }
    return {"identity": metadata, "cases": case_reports,
            "total_invocations": sum(case["invocations"] for case in case_reports.values()),
            "joint_byte_repeatability_observed_all_cases": all(
                case["joint_byte_repeatability_observed"] for case in case_reports.values())}


def run(out_path: Path, ordered: bool) -> dict:
    artifact_path = Path(str(out_path) + ".safetensors")
    if out_path.exists() or out_path.is_symlink() or artifact_path.exists() or artifact_path.is_symlink():
        raise ValueError("Refusing existing JSON or companion tensor destination")
    if not out_path.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    import torch
    from safetensors.torch import save_file
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("This experiment requires real NVIDIA CUDA; no fallback exists")
    function, stock_identity = stock_backend()
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    cases = {name: fixed_inputs(index) for index, name in enumerate(CASES)}
    references = {name: mathematical_reference(inputs) for name, inputs in cases.items()}
    tensors = {}
    for case_name, inputs in cases.items():
        tensors.update({f"inputs.{case_name}.{key}": value for key, value in inputs.items()})
        tensors.update({f"reference_fp64.{case_name}.{key}": value
                        for key, value in references[case_name].items()})
    report = {
        "schema": "qfs.native-kda-isolation.v1", "started_utc": datetime.now(timezone.utc).isoformat(),
        "generator": file_record(Path(__file__)),
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "executable": sys.executable,
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "exllamav3", "safetensors")},
            "torch_cuda": torch.version.cuda, "torch_build_config": torch.__config__.show(),
            "device_index": 0, "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "device_properties": str(properties), "device_uuid": str(getattr(properties, "uuid", "unavailable")),
            "torch_compiled_arches": torch.cuda.get_arch_list(),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        },
        "coverage": {
            "case_names": list(CASES), "repeats_per_case_per_backend": REPEATS,
            "requested_backends": ["stock", "ordered"] if ordered else ["stock"],
            "batch_size": 1, "sequence_length": 1, "num_k_heads": 4, "num_v_heads": 4,
            "k_head_dim": 128, "v_head_dim": 128, "history": False, "slots": [0],
            "channelwise_g": True, "native_v_split": 4,
            "warmup_invocations_excluded": 0, "synchronize_after_each_invocation": True,
            "fresh_clones_all_inputs_and_initial_state": True,
            "output_initialized_to_nan": True,
            "input_generation": "CPU int64 modular arithmetic, odd nonzero signed numerators, power-of-two scaling; no RNG",
            "reference": "CPU FP64 from actual BF16/FP32 input values: q/k L2 norm eps=1e-6; exp(g) row decay; delta update; q projection / sqrt(128)",
            "reference_limit": "Diagnostics only, not a fallback or expected bit pattern. Native uses FP32 fast math and round-toward-zero BF16 output; FP64 reference output is unquantized.",
            "scope_limit": "Finite one-token 4-head corpus only; exact repeats do not establish universal determinism. No atomicAdd causality claim.",
            "numerical_tolerance_gate": None,
        },
        "inputs": {name: {key: {"tensor": f"inputs.{name}.{key}", **tensor_record(value)}
                          for key, value in inputs.items()} for name, inputs in cases.items()},
        "backends": {},
    }
    report["backends"]["stock"] = execute_backend(
        "stock", function, stock_identity, cases, references, tensors)
    if ordered:
        from exl3_ordered_kda import load_ordered_backend
        ordered_function, ordered_identity = load_ordered_backend()
        identity = {"runtime_identity": "ordered_kda_separate_native_cuda_changed_runtime",
                    "changed_runtime": True, "loader_metadata": ordered_identity,
                    "ordinary_positional_signature": SIGNATURE}
        report["backends"]["ordered"] = execute_backend(
            "ordered", ordered_function, identity, cases, references, tensors)
        comparisons = {}
        for case_name in CASES:
            rows = []
            for repeat in range(REPEATS):
                paired = {}
                for key in ("out", "recurrent_state"):
                    stock = tensors[f"stock.{case_name}.repeat_{repeat:03d}.{key}"]
                    changed = tensors[f"ordered.{case_name}.repeat_{repeat:03d}.{key}"]
                    paired[key] = {"exact_bytes_equal": raw_bytes(stock) == raw_bytes(changed),
                                   **error_metrics(changed, stock)}
                rows.append({"repeat": repeat, **paired})
            comparisons[case_name] = {
                "paired_repeats": rows,
                "max_abs_across_paired_repeats": {
                    key: max((row[key]["max_abs"] for row in rows if row[key]["max_abs"] is not None), default=None)
                    for key in ("out", "recurrent_state")},
                "nonfinite_pairs_across_repeats": {
                    key: sum(row[key]["nonfinite_pairs"] for row in rows)
                    for key in ("out", "recurrent_state")},
            }
        report["ordered_vs_stock_diagnostics"] = comparisons
    report["coverage"]["completed_invocations"] = sum(
        backend["total_invocations"] for backend in report["backends"].values())
    report["coverage"]["expected_invocations"] = len(CASES) * REPEATS * (2 if ordered else 1)
    report["coverage"]["complete"] = (
        report["coverage"]["completed_invocations"] == report["coverage"]["expected_invocations"])
    report["tensor_inventory"] = {key: tensor_record(value) for key, value in sorted(tensors.items())}
    report["tensor_artifact"] = {"path": str(artifact_path.resolve()), "tensor_count": len(tensors)}
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    # Strict JSON serialization happens before reserving either destination.
    json.dumps(report, allow_nan=False)
    with artifact_path.open("xb"):
        pass
    save_file(tensors, str(artifact_path), metadata={"schema": report["schema"], "source_commit": COMMIT})
    report["tensor_artifact"].update(file_record(artifact_path))
    encoded = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with out_path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return {"json": str(out_path.resolve()), "json_sha256": sha256(out_path),
            "tensor_artifact": report["tensor_artifact"], "coverage": report["coverage"],
            "observed_joint_byte_repeatability": {
                name: backend["joint_byte_repeatability_observed_all_cases"]
                for name, backend in report["backends"].items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="New JSON path; parent must exist")
    parser.add_argument("--ordered", action="store_true", help="Also run the separate ordered CUDA runtime")
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.ordered), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
