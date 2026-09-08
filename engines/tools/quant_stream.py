"""Metadata-planned, module-at-a-time reconstruction for streamed capture.

Storage components are never routed as model parameters. Virtual native-weight
slots route through the model's own converter; each slot fetches its exact
components (including shared ancestor scales) and releases its decoded temporary.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os

METHODS = {"affine_surface": "affine-weight-reconstruction", "microscale_surface": "microfloat-weight-reconstruction"}


def quantization(config):
    declarations = []
    for node in (config, config.get("text_config") or {}):
        for field in ("quantization_config", "quantization"):
            value = node.get(field)
            if value:
                if not isinstance(value, dict):
                    raise ValueError("quantization declaration must be an object")
                declarations.append(value)
    if any(value != declarations[0] for value in declarations[1:]):
        raise ValueError("conflicting top/text quantization declarations")
    return declarations[0] if declarations else {}


def reader_for(config):
    quant = quantization(config)
    method = str(quant.get("quant_method", "")).lower()
    if method in {"gptq", "awq", "mlx"} or (not method and "bits" in quant and "group_size" in quant):
        return "affine_surface"
    if method == "compressed-tensors":
        return "affine_surface" if quant.get("format") == "pack-quantized" else "microscale_surface"
    if method == "fp8" and quant.get("scale_fmt") == "ue8m0":
        return "microscale_surface"
    if method == "modelopt":
        # Preserve the independently qualified flagship reader's exact contract.
        # Other geometries use the format reader, never the flagship's constants.
        if quant.get("quant_algo") == "MIXED_PRECISION":
            return "microscale_surface"
        if quant.get("quant_algo") == "NVFP4":
            from nvfp4_surface import GEOMETRIES, geometry_for_config
            try:
                geometry_for_config(config)
            except ValueError:
                geometry = GEOMETRIES.get(config.get("model_type"))
                if geometry is not None and tuple(config.get("architectures") or ()) != geometry.architectures:
                    raise
                return "microscale_surface"
    return None


def checkpoint_plan(config, model_dir, header_reader, gate=None):
    reader = reader_for(config)
    if reader is None:
        return None
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    weight_map = {}
    if os.path.isfile(index_path):
        with open(index_path) as handle:
            weight_map = json.load(handle).get("weight_map") or {}
    if gate is not None and not weight_map:
        raise ValueError("gated quantized capture requires a complete safetensors index")
    names = sorted(set(weight_map.values())) if weight_map else sorted(
        name for name in os.listdir(model_dir) if name.endswith(".safetensors"))
    if not names:
        raise ValueError("recognized quantization has no safetensors shards")
    if gate is not None:
        # Exhaustive format admission needs all headers before model construction.
        # This barrier is explicit; legacy race formats retain incremental opening.
        gate.wait_for_shards(names)
    metadata = {}
    for name in names:
        path = os.path.join(model_dir, name)
        header, required = header_reader(path)
        if os.stat(path).st_size < required:
            raise ValueError("short quantized shard: " + name)
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            if key in metadata or (weight_map and weight_map.get(key) != name):
                raise ValueError("duplicate/misindexed quantized tensor: " + key)
            metadata[key] = {"shape": list(entry["shape"]), "dtype": entry["dtype"]}
            if key.endswith(".weight_shape"):
                if entry["shape"] not in ([2], [3]) or entry["dtype"] not in {"I32", "I64"}:
                    raise ValueError("invalid small weight_shape tensor: " + key)
                from safetensors import safe_open
                with safe_open(path, framework="pt", device="cpu") as handle:
                    metadata[key]["value"] = handle.get_tensor(key).tolist()
    if weight_map and set(weight_map) != set(metadata):
        raise ValueError("quantized index/header tensor sets differ")
    surface = importlib.import_module(reader)
    plan = surface.plan_modules(config, metadata)
    if plan is None:
        raise ValueError("selected quantized reader did not recognize its declaration")
    consumed = set(plan["consumed"])
    import affine_surface
    import microscale_surface
    quant_roles = affine_surface._QUANT_SUFFIXES | microscale_surface.QUANT_ROLES | {"scale"}
    for key, meta in metadata.items():
        # No unscaled float8 or integer-packed weight can pass as a native float.
        if key not in consumed and key.endswith(".weight") and meta["dtype"] not in {"BF16", "F16", "F32", "F64"}:
            raise ValueError("unconsumed encoded weight: " + key)
        if key not in consumed and (key.rsplit(".", 1)[-1] in quant_roles or "quantizer" in key):
            raise ValueError("unconsumed quantization state: " + key)
    plan["quantization_config"] = quantization(config)
    plan["_reader"] = reader
    plan["_metadata"] = metadata
    plan["_shards"] = names
    plan["_gate_header_barrier"] = gate is not None
    return plan


class WeightSlot:
    __slots__ = ("name", "spec")

    def __init__(self, name, spec):
        self.name = name
        self.spec = spec


def slots(plan):
    return {name: WeightSlot(name, spec) for name, spec in plan["modules"].items()}


def materialize(subset, plan, payload_slices, dtype, device, stats, eager, sink=None, expected_shape=None):
    surface = importlib.import_module(plan["_reader"])
    out = {}
    for name, value in subset.items():
        if not isinstance(value, WeightSlot):
            out[name] = value
            continue
        spec = value.spec
        expected = expected_shape(name) if expected_shape else None
        if expected is not None and tuple(spec["shape"]) != tuple(expected):
            raise ValueError(f"decoded shape for {name}: {spec['shape']} differs from model {expected}")
        missing = set(spec["components"].values()) - payload_slices.keys()
        if missing:
            raise ValueError("missing packed components: " + ", ".join(sorted(missing)))
        payload = {role: eager(payload_slices[key]) for role, key in spec["components"].items()}
        stats["bytes_read"] = stats.get("bytes_read", 0) + sum(
            tensor.numel() * tensor.element_size() for tensor in payload.values())
        decoded = surface.decode_module(payload, spec, dtype=dtype, device=device)
        del payload
        stats["decoded_modules"] += 1
        stats["formats"][spec["format"]] = stats["formats"].get(spec["format"], 0) + 1
        stats["components"].update(spec["components"].values())
        stats["decoded_names"].add(name)
        if sink is None or not sink(name, decoded):
            out[name] = decoded
        del decoded
    return out


def source_files(plan):
    names = ["quant_stream.py", "affine_surface.py", "microscale_surface.py"]
    if plan["_reader"] == "affine_surface" and any(s["format"].startswith("mlx-") for s in plan["modules"].values()):
        names.append("mlx_surface.py")
    if plan["_reader"] == "microscale_surface":
        names.append("nvfp4_surface.py")
    return names


def evidence(plan, stats, dtype_name):
    sources = {}
    for name in source_files(plan):
        with open(os.path.join(os.path.dirname(__file__), name), "rb") as handle:
            sources["engines/tools/" + name] = hashlib.sha256(handle.read()).hexdigest()
    public = {key: value for key, value in plan.items() if not key.startswith("_")}
    digest = hashlib.sha256(json.dumps(public, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    quant = plan["quantization_config"]
    activation_declared = quant.get("activation_scheme") not in (None, "", "none")
    activation_declared = activation_declared or bool(quant.get("kv_cache_scheme"))
    for group in (quant.get("config_groups") or {}).values():
        activation_declared = activation_declared or any(
            group.get(role) is not None for role in ("input_activations", "output_activations"))
    # A flat ModelOpt block declares static FP4 activations with a bare
    # `with_input_scale: true` (nvfp4_surface.modelopt_weight_declaration
    # reads it as a W4A4 declaration), and the reader's plan names the stored
    # activation-scale components it consumed. Either one means the artifact
    # is activation-declared, so the activation-not-captured disclosure must
    # survive here instead of being silently erased.
    import microscale_surface
    activation_plan = plan.get("activation_quantization")
    stored = activation_plan.get("stored_components", ()) if isinstance(activation_plan, dict) else ()
    activation_declared = activation_declared or quant.get("with_input_scale") is True
    activation_declared = activation_declared or any(
        key.rsplit(".", 1)[-1] in microscale_surface.ACTIVATION_ROLES for key in stored)
    return {"method": METHODS[plan["_reader"]], "reference": plan["reference"],
            "output_dtype": dtype_name, "quantization_config": plan["quantization_config"],
            "modules_decoded": stats["decoded_modules"], "modules_planned": len(plan["modules"]),
            "formats_decoded": dict(sorted(stats["formats"].items())),
            "components_consumed": len(stats["components"]), "components_planned": len(plan["consumed"]),
            "modules_not_decoded": sorted(set(plan["modules"]) - stats["decoded_names"]),
            "components_not_consumed": sorted(set(plan["consumed"]) - stats["components"]),
            "checkpoint_bytes_read": stats.get("bytes_read", 0),
            "activation_quantization": plan.get("activation_quantization") if activation_declared else None,
            "reader_evidence": {"source_files": sources, "decode_plan_sha256": digest,
                                "gate_header_barrier": plan["_gate_header_barrier"]},
            "scope": "weights_reconstructed", "comparison_class": "advisory",
            "scientific_disclosure": "Stored-weight reconstruction only; no native quantized GEMM, activation arithmetic, or quantizer optimization quality claim."}
