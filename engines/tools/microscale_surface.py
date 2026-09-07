"""Metadata-planned weights-only floating microformats (no native quantized GEMM).

Covered: compressed-tensors 0.18 MXFP4 (U8 weight_packed, U8 UE8M0
weight_scale, K groups of 32), CT NVFP4 (groups of 16, E4M3 scales divided
by FP32 weight_global_scale), ModelOpt NVFP4 (same bytes, scales multiplied
by weight_scale_2), and DeepSeek-style block E4M3 + UE8M0 scales.
Matrices and explicit leading expert axes [E,N,K] are supported. No swizzles.
ModelOpt MIXED_PRECISION dispatches explicit ancestor targets to NVFP4,
FP8 per-tensor (including shared parent PLE shard scales), or FP8_PB_WO
128x128 blocks with BF16/F32 scales. Other constituents are refused.

FP4 byte 0x21 decodes [0.5, 1.0]: low nibble first; nibble 8 is -0.
UE8M0 byte 0 is 2**-127, NOT zero; 255 is reserved NaN and refused.
FP32 scale arithmetic precedes FP32 element multiplication, then one output
cast. This is not CT's default BF16-intermediate decompressor or GPU parity.

References (source audited, never executed by the reader):
https://github.com/vllm-project/compressed-tensors/tree/0.18.0/src/compressed_tensors/compressors
  mxfp4/base.py, mx_utils.py, nvfp4/base.py, nvfp4/helpers.py
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py
  fp8_gemm_kernel: scales_b[ceil(N/128),ceil(K/128)] cast to FP32.
https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/blob/fc694b54fb0174e0913e6adf86691ef85a4ead47/config.json
  quantized_layers; model-fp8-mtp-ple.safetensors header confirms shared
  ngram_embedding.weight_scale BF16 [1] and MTP weight_scale_inv BF16 grids.

DeepSeek V4's I8 .weight + F8_E8M0 .scale FP4 dialect is deliberately
REFUSED: it is not CT weight_packed, and no packed-nibble oracle has been
validated here. GPT-OSS blocks/scales, MLX MXFP4, CT MXFP8, ModelOpt MXFP8
and online transforms are not implemented. Ordinary FP8 + FP32 checkpoints
stay on the legacy decoder; mixed ModelOpt block FP8 reuses its arithmetic.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

REFERENCE = "compressed-tensors@0.18.0; DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py; nvidia/Qwen3.8-Flash-Next-NVFP4@fc694b54fb0174e0913e6adf86691ef85a4ead47/config.json+headers"
FP8 = {"F8_E4M3", "F8_E4M3FN"}
UE8 = {"U8", "F8_E8M0", "F8_E8M0FNU"}
ACTIVATION_ROLES = {"input_scale", "input_global_scale", "input_scale_2", "output_scale", "output_global_scale"}
QUANT_ROLES = {"weight_packed", "weight_scale", "weight_scale_inv", "weight_global_scale", "weight_scale_2", "weight_shape", "weight_zero_point", "weight_g_idx", "qweight", "qzeros", "g_idx", "k_scale", "v_scale", "input_zero_point", "output_zero_point", "activation_scale"} | ACTIVATION_ROLES


def _fail(message):
    return ValueError("microfloat: " + message)


def _quantization(config):
    candidates = [config.get("quantization_config")]
    for name in ("text_config", "language_config"):
        nested = config.get(name)
        if isinstance(nested, Mapping):
            candidates.append(nested.get("quantization_config"))
    candidates = [q for q in candidates if q is not None]
    if not candidates:
        return {}
    if any(not isinstance(q, Mapping) for q in candidates):
        raise _fail("quantization_config must be a mapping")
    if any(q != candidates[0] for q in candidates[1:]):
        raise _fail("conflicting top-level/text quantization_config")
    return candidates[0]


def _shape(meta, key):
    if key not in meta:
        raise _fail(f"missing component {key}")
    shape = meta[key].get("shape")
    if not isinstance(shape, (list, tuple)) or any(type(x) is not int or x <= 0 for x in shape):
        raise _fail(f"invalid shape for {key}: {shape!r}")
    return list(shape)


def _require(meta, key, shape, dtypes):
    if _shape(meta, key) != shape or meta[key].get("dtype") not in dtypes:
        raise _fail(f"{key} must have shape {shape}, dtype {sorted(dtypes)}; got {meta[key]}")


def _matches(target, module):
    if not isinstance(target, str):
        raise _fail("quantization targets must be strings")
    if target == "Linear":
        return True
    if target.startswith("re:"):
        return re.match(target[3:], module) is not None
    return target == module or module.endswith("." + target)


def _ct_group(quant, module):
    groups = quant.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise _fail("CT requires nonempty config_groups")
    if any(_matches(t, module) for t in quant.get("ignore", [])):
        raise _fail(f"packed module {module} conflicts with ignore declaration")
    matched = []
    for group in groups.values():
        if not isinstance(group, Mapping):
            raise _fail("invalid CT group")
        targets = group.get("targets", [])
        if any(_matches(t, module) for t in targets):
            matched.append(group)
    if not matched:
        raise _fail(f"no CT group targets {module}")
    if any(g != matched[0] for g in matched[1:]):
        raise _fail(f"ambiguous overlapping CT groups for {module}")
    group = matched[0]
    weights = group.get("weights") or {}
    if not isinstance(weights, Mapping):
        raise _fail(f"invalid CT weights declaration for {module}")
    fmt = group.get("format") or quant.get("format")
    if fmt not in ("mxfp4-pack-quantized", "nvfp4-pack-quantized"):
        raise _fail(f"unsupported CT format {fmt!r} for {module}")
    size = 32 if fmt.startswith("mx") else 16
    if (weights.get("num_bits") != 4 or weights.get("type") != "float"
            or weights.get("group_size") != size or weights.get("symmetric") is not True
            or weights.get("dynamic") not in (None, False)
            or weights.get("strategy") not in ("group", "tensor_group")
            or weights.get("actorder") not in (None, False)
            or weights.get("block_structure") is not None):
        raise _fail(f"unsupported CT FP4 weight declaration for {module}: {weights}")
    expected_scale = {None, "torch.uint8", "uint8"} if size == 32 else {None, "torch.float8_e4m3fn", "float8_e4m3fn"}
    if weights.get("scale_dtype") not in expected_scale:
        raise _fail(f"unsupported CT scale dtype for {module}")
    if size == 32 and weights.get("strategy") != "group":
        raise _fail("MXFP4 requires group strategy, without a global scale")
    return fmt, group


def _modelopt_layer(quant, module):
    """Longest explicit ancestor target; no substring or regex guessing."""
    layers = quant["quantized_layers"]
    matches = [name for name in layers if module == name or module.startswith(name + ".")]
    if not matches:
        return None
    target = max(matches, key=len)
    declaration = layers[target]
    algorithm = declaration["quant_algo"].upper()
    matched_groups = [g for g in (quant.get("config_groups") or {}).values()
                      if any(module == t or module.startswith(t + ".") for t in g.get("targets", []))]
    if len(matched_groups) > 1:
        raise _fail(f"ambiguous ModelOpt config_groups for {module}")
    group = matched_groups[0] if matched_groups else {}
    weights = group.get("weights") or {}
    if weights:
        bits = 4 if algorithm == "NVFP4" else 8
        size = declaration.get("group_size")
        if (weights.get("num_bits") != bits or weights.get("type") != "float"
                or weights.get("dynamic") not in (None, False)
                or weights.get("group_size") != size):
            raise _fail(f"ModelOpt group/per-layer declaration conflict for {module}")
    return target, declaration, group.get("input_activations")


def plan_modules(config_dict, tensor_metadata):
    """Return a JSON plan; None only for unrecognized quant_method.

    Caller dispatches CT integer formats to affine_surface and ordinary FP8
    scales to the legacy FP8 reader. Recognized but unsupported formats fail.
    No tensors or model code are loaded during planning.
    """
    quant = _quantization(config_dict)
    method = str(quant.get("quant_method", "")).lower()
    if method not in ("compressed-tensors", "modelopt", "fp8"):
        return None
    if quant.get("transform_config") or quant.get("sparsity_config"):
        raise _fail("transforms/sparsity must not be silently dropped")
    groups = quant.get("config_groups") or {}
    if not isinstance(groups, Mapping):
        raise _fail("config_groups must be a mapping")
    for group in groups.values():
        if not isinstance(group, Mapping) or not isinstance(group.get("targets", []), list):
            raise _fail("each quantization group must declare a target list")
        if any(not isinstance(t, str) for t in group.get("targets", [])):
            raise _fail("group targets must be strings")
        if group.get("weights") is not None and not isinstance(group["weights"], Mapping):
            raise _fail("group weights must be a mapping")
    if not isinstance(quant.get("ignore", []), list):
        raise _fail("ignore must be a list")
    if method == "modelopt":
        from nvfp4_surface import MO_ONLINE_TRANSFORM_KEYS, modelopt_weight_declaration
        modelopt_mixed = str(quant.get("quant_algo", "")).upper() == "MIXED_PRECISION"
        if any(quant.get(k) not in (None, False) for k in MO_ONLINE_TRANSFORM_KEYS):
            raise _fail("ModelOpt online transforms are unsupported")
        if modelopt_mixed:
            layers = quant.get("quantized_layers")
            if not isinstance(layers, Mapping) or not layers:
                raise _fail("ModelOpt mixed precision requires quantized_layers")
            for target, declaration in layers.items():
                if not isinstance(target, str) or not isinstance(declaration, Mapping):
                    raise _fail("invalid ModelOpt per-layer declaration")
                if set(declaration) - {"quant_algo", "group_size"}:
                    raise _fail(f"unsupported ModelOpt per-layer settings for {target}")
                algo = str(declaration.get("quant_algo", "")).upper()
                if algo not in ("NVFP4", "FP8", "FP8_PB_WO"):
                    raise _fail(f"unimplemented mixed ModelOpt constituent {target}: {algo}")
                expected_group = {"NVFP4": 16, "FP8_PB_WO": 128, "FP8": None}[algo]
                if declaration.get("group_size") != expected_group:
                    raise _fail(f"unsupported {algo} group size for {target}")
            activation = {k: v.get("input_activations") for k, v in groups.items()}
        else:
            if str(quant.get("quant_algo", "")).upper() != "NVFP4" or quant.get("quantized_layers"):
                raise _fail("unsupported homogeneous ModelOpt algorithm")
            declaration = modelopt_weight_declaration(quant)
            activation = declaration["input_activations"]
    elif method == "fp8":
        if quant.get("scale_fmt") != "ue8m0" or quant.get("fmt") != "e4m3":
            raise _fail("only FP8 e4m3 with explicit scale_fmt=ue8m0 belongs to this reader")
        block = quant.get("weight_block_size")
        if not isinstance(block, (list, tuple)) or len(block) != 2 or any(type(x) is not int or x <= 0 for x in block):
            raise _fail("FP8 weight_block_size must contain two positive integers")
        activation = {"activation_scheme": quant.get("activation_scheme"), "scale_fmt": "ue8m0"}
    else:
        if not isinstance(groups, Mapping) or not groups:
            raise _fail("CT requires config_groups")
        activation = {k: {a: v.get(a) for a in ("input_activations", "output_activations")}
                      for k, v in groups.items() if isinstance(v, Mapping)}
    modules = {}
    consumed = set()
    for key, meta in tensor_metadata.items():
        if method == "compressed-tensors":
            if not key.endswith(".weight_packed"):
                continue
            module = key.rsplit(".", 1)[0]
            fmt, group = _ct_group(quant, module)
            size = 32 if fmt.startswith("mx") else 16
            global_role = None if size == 32 else "weight_global_scale"
        elif method == "modelopt":
            if not key.endswith(".weight"):
                continue
            module = key.rsplit(".", 1)[0]
            modelopt_activation = activation
            if modelopt_mixed:
                selected = _modelopt_layer(quant, module)
                if selected is None:
                    continue
                target, layer_declaration, modelopt_activation = selected
                algorithm = layer_declaration["quant_algo"].upper()
                if algorithm != "NVFP4":
                    shape = _shape(tensor_metadata, key)
                    if len(shape) not in (2, 3) or meta.get("dtype") not in FP8:
                        raise _fail(f"{algorithm} needs E4M3 matrix/leading-expert storage: {key}")
                    if algorithm == "FP8":
                        # NVIDIA PLE shards share ONE parent weight_scale.
                        candidates = list(dict.fromkeys([module + ".weight_scale", target + ".weight_scale"]))
                        found = [k for k in candidates if k in tensor_metadata]
                        if len(found) != 1:
                            raise _fail(f"{key} requires one unambiguous local/declared-parent FP8 scale")
                        scale_key = found[0]
                        if _shape(tensor_metadata, scale_key) not in ([], [1]):
                            raise _fail(f"per-tensor FP8 scale must be scalar: {scale_key}")
                        fmt = "modelopt-fp8-tensor"
                        block_size = None
                    else:
                        scale_key = module + ".weight_scale_inv"
                        scale_shape = shape[:-2] + [math.ceil(shape[-2] / 128), math.ceil(shape[-1] / 128)]
                        _require(tensor_metadata, scale_key, scale_shape, {"F32", "BF16"})
                        fmt, block_size = "modelopt-fp8-block", [128, 128]
                    if tensor_metadata[scale_key].get("dtype") not in ("F32", "BF16"):
                        raise _fail(f"ModelOpt FP8 scale must be F32/BF16: {scale_key}")
                    components = {"weight": key, "weight_scale": scale_key}
                    for role in ACTIVATION_ROLES:
                        if module + "." + role in tensor_metadata:
                            components[role] = module + "." + role
                    if isinstance(modelopt_activation, Mapping) and modelopt_activation.get("dynamic") is False and "input_scale" not in components:
                        raise _fail(f"{module}: static FP8 activations require input_scale")
                    modules[key] = {"format": fmt, "shape": shape, "block_size": block_size,
                                    "components": components, "modelopt_target": target}
                    consumed.update(components.values())
                    continue
            elif meta.get("dtype") != "U8":
                continue
            fmt, size, global_role = "modelopt-nvfp4", 16, "weight_scale_2"
        else:
            if not key.endswith(".weight"):
                continue
            module = key.rsplit(".", 1)[0]
            scale_names = [module + "." + s for s in ("scale", "weight_scale_inv", "weight_scale") if module + "." + s in tensor_metadata]
            if not scale_names and meta.get("dtype") not in FP8:
                continue
            if meta.get("dtype") not in FP8:
                raise _fail(f"unsupported DeepSeek FP4/other packed weight {key} ({meta.get('dtype')}); not CT/GPT-OSS MXFP4")
            if len(scale_names) != 1:
                raise _fail(f"{key} requires exactly one UE8M0 scale component")
            shape = _shape(tensor_metadata, key)
            if len(shape) not in (2, 3):
                raise _fail(f"only matrix/leading-expert FP8 supported: {key}")
            scale_shape = shape[:-2] + [math.ceil(shape[-2] / block[0]), math.ceil(shape[-1] / block[1])]
            _require(tensor_metadata, scale_names[0], scale_shape, UE8)
            spec = {"format": "fp8-ue8m0", "shape": shape, "block_size": list(block),
                    "components": {"weight": key, "weight_scale": scale_names[0]}}
            modules[key] = spec
            consumed.update(spec["components"].values())
            continue
        packed_shape = _shape(tensor_metadata, key)
        if len(packed_shape) not in (2, 3) or meta.get("dtype") != "U8":
            raise _fail(f"{key} requires rank-2/3 U8 packed storage")
        shape = packed_shape[:-1] + [packed_shape[-1] * 2]
        if shape[-1] % size:
            raise _fail(f"{key}: K={shape[-1]} must be divisible by {size}; padded FP4 tails are undeclared")
        canonical = module + ".weight"
        if canonical != key and canonical in tensor_metadata:
            raise _fail(f"both dense and packed weight present for {module}")
        scale_key = module + ".weight_scale"
        _require(tensor_metadata, scale_key, shape[:-1] + [shape[-1] // size], UE8 if size == 32 else FP8)
        components = {"weight_packed": key, "weight_scale": scale_key}
        if global_role:
            global_key = module + "." + global_role
            global_shape = _shape(tensor_metadata, global_key)
            permitted = [[], [1]]
            if len(shape) == 3:
                permitted += [[shape[0]], [shape[0], 1], [shape[0], 1, 1]]
            if global_shape not in permitted or tensor_metadata[global_key].get("dtype") != "F32":
                raise _fail(f"{global_key} must be FP32 scalar or one scalar per leading expert")
            components[global_role] = global_key
        shape_key = module + ".weight_shape"
        if shape_key in tensor_metadata:
            _require(tensor_metadata, shape_key, [len(shape)], {"I32", "I64"})
            if tensor_metadata[shape_key].get("value") != shape:
                raise _fail(f"{shape_key} value must be supplied and equal {shape}")
            components["weight_shape"] = shape_key
        for role in ACTIVATION_ROLES:
            act_key = module + "." + role
            if act_key in tensor_metadata:
                _shape(tensor_metadata, act_key)
                if tensor_metadata[act_key].get("dtype") not in ({"F32", "F16", "BF16"} | FP8 | UE8):
                    raise _fail(f"unsupported activation scale dtype: {act_key}")
                components[role] = act_key
        required_activation = None
        if method == "compressed-tensors":
            declared_input = group.get("input_activations")
            if isinstance(declared_input, Mapping) and declared_input.get("dynamic") is False:
                required_activation = "input_global_scale"
        elif isinstance(modelopt_activation, Mapping) and modelopt_activation.get("dynamic") is False:
            required_activation = "input_scale"
        if required_activation and required_activation not in components:
            raise _fail(f"{module}: static activation declaration requires {required_activation}")
        modules[canonical] = {"format": fmt, "shape": shape, "group_size": size,
                              "components": components, "nibble_order": "low-first"}
        consumed.update(components.values())
    # Any quantization state not accounted for is an error, never silently discarded.
    for key, meta in tensor_metadata.items():
        role = key.rsplit(".", 1)[-1]
        quant_state = (role in QUANT_ROLES or role.startswith(("weight_scale", "weight_zero", "input_scale", "input_global_scale"))
                       or "quantizer" in key)
        if key not in consumed and (quant_state or (role == "weight" and meta.get("dtype") in (FP8 | {"U8", "I8"}))):
            raise _fail(f"unconsumed/unsupported quantization component {key}")
    if not modules:
        raise _fail(f"recognized {method} configuration but no supported encoded weights")
    for spec in modules.values():
        spec["component_metadata"] = {role: dict(tensor_metadata[key])
                                      for role, key in spec["components"].items()}
    return {"format": "microfloat", "method": method, "modules": modules,
            "consumed": sorted(consumed), "reference": REFERENCE,
            "quantization_config": dict(quant),
            "activation_quantization": {"scope": "weights-reconstructed/advisory; activation-not-captured",
                "applied_to_weights": False, "declared": activation,
                "kv_cache_scheme": quant.get("kv_cache_scheme"),
                "stored_components": sorted(k for k in consumed if k.rsplit(".", 1)[-1] in ACTIVATION_ROLES)}}


def _ue8m0(value):
    import torch
    if str(value.dtype) in ("torch.float8_e8m0fnu",):
        value = value.view(torch.uint8)
    if value.dtype != torch.uint8:
        raise _fail(f"UE8M0 needs raw U8 or float8_e8m0fnu, got {value.dtype}")
    if (value == 255).any():
        raise _fail("reserved UE8M0 NaN exponent 255")
    return torch.ldexp(torch.ones_like(value, dtype=torch.float32), value.to(torch.int32) - 127)


def decode_module(payload, spec, *, dtype, device):
    """Decode planned components in FP32; refuse nonfinite input/output.

    Activation components are validated/consumed but NEVER multiplied into W.
    FP8 byte views (U8) may replace native float8 tensors on unsupported devices.
    """
    import torch
    from nvfp4_surface import dequant_nvfp4, f8e4m3_to_float32, unpack_e2m1
    if set(payload) != set(spec["components"]):
        raise _fail("payload roles do not exactly match the module plan")
    for role, meta in spec["component_metadata"].items():
        if list(payload[role].shape) != list(meta["shape"]):
            raise _fail(f"{role} payload shape disagrees with planned metadata")
    if dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        raise _fail("output dtype must be a floating reconstruction dtype")
    shape = list(spec["shape"])
    fmt = spec["format"]
    for role in ACTIVATION_ROLES:
        if role in payload:
            value = payload[role]
            if value.dtype == torch.uint8 or str(value.dtype) == "torch.float8_e8m0fnu":
                _ue8m0(value)
            elif not torch.isfinite(value.to(torch.float32)).all():
                raise _fail(f"nonfinite activation state {role}")
    if "weight_shape" in payload and payload["weight_shape"].tolist() != shape:
        raise _fail("weight_shape payload disagrees with plan")
    if fmt in ("fp8-ue8m0", "modelopt-fp8-block", "modelopt-fp8-tensor"):
        weight = payload["weight"].to(device)
        if list(weight.shape) != shape or (weight.dtype != torch.uint8 and str(weight.dtype) != "torch.float8_e4m3fn"):
            raise _fail("FP8 payload shape/dtype mismatch")
        raw_scale = payload["weight_scale"].to(device)
        if fmt == "fp8-ue8m0":
            scale = _ue8m0(raw_scale)
        else:
            if raw_scale.dtype not in (torch.float32, torch.bfloat16):
                raise _fail("ModelOpt FP8 scale must be F32/BF16")
            scale = raw_scale.to(torch.float32)
            if not torch.isfinite(scale).all() or (scale < 0).any():
                raise _fail("ModelOpt FP8 scales must be finite and nonnegative")
        if fmt == "modelopt-fp8-tensor":
            result = f8e4m3_to_float32(weight) * scale.reshape(())
        elif fmt == "modelopt-fp8-block":
            # Reuse the existing FP8 block decoder, including its padded-tail math.
            from layer_outer import dequantize_block_fp8
            native = weight.view(torch.float8_e4m3fn) if weight.dtype == torch.uint8 else weight
            if len(shape) == 2:
                result = dequantize_block_fp8(native, scale, torch.float32, spec["block_size"])
            else:
                result = torch.empty(shape, dtype=torch.float32, device=device)
                for expert in range(shape[0]):
                    result[expert] = dequantize_block_fp8(native[expert], scale[expert], torch.float32, spec["block_size"])
        else:
            bm, bk = spec["block_size"]
            expected = shape[:-2] + [math.ceil(shape[-2] / bm), math.ceil(shape[-1] / bk)]
            if list(scale.shape) != expected:
                raise _fail("FP8 scale payload shape mismatch")
            weight = f8e4m3_to_float32(weight)
            rows = torch.arange(shape[-2], device=device) // bm
            cols = torch.arange(shape[-1], device=device) // bk
            result = weight * scale[..., rows[:, None], cols[None, :]]
    else:
        packed = payload["weight_packed"].to(device)
        scale = payload["weight_scale"].to(device)
        size = spec["group_size"]
        if list(packed.shape) != shape[:-1] + [shape[-1] // 2] or packed.dtype != torch.uint8:
            raise _fail("FP4 packed payload shape/dtype mismatch")
        if list(scale.shape) != shape[:-1] + [shape[-1] // size]:
            raise _fail("FP4 scale payload shape mismatch")
        rows = math.prod(shape[:-1])
        if fmt == "mxfp4-pack-quantized":
            decoded = unpack_e2m1(packed.reshape(rows, -1)).reshape(*shape[:-1], -1, size)
            result = (decoded * _ue8m0(scale).unsqueeze(-1)).reshape(shape)
        elif fmt in ("nvfp4-pack-quantized", "modelopt-nvfp4"):
            if str(scale.dtype) != "torch.float8_e4m3fn" and scale.dtype != torch.uint8:
                raise _fail("NVFP4 scale payload must be E4M3 or its U8 byte view")
            scale32 = f8e4m3_to_float32(scale)
            if not torch.isfinite(scale32).all() or (scale32 < 0).any():
                raise _fail("NVFP4 scales must be finite and nonnegative")
            role = "weight_global_scale" if fmt == "nvfp4-pack-quantized" else "weight_scale_2"
            global_scale = payload[role].to(device)
            if global_scale.dtype != torch.float32 or not torch.isfinite(global_scale).all() or (global_scale <= 0).any():
                raise _fail("NVFP4 global scales must be finite positive FP32")
            if global_scale.numel() == 1:
                result = dequant_nvfp4(packed.reshape(rows, -1), scale32.reshape(rows, -1), **{role: global_scale}).reshape(shape)
            elif len(shape) == 3 and list(global_scale.shape) in ([shape[0]], [shape[0], 1], [shape[0], 1, 1]):
                result = torch.empty(shape, dtype=torch.float32, device=device)
                globals_flat = global_scale.reshape(-1)
                for expert in range(shape[0]):
                    result[expert] = dequant_nvfp4(packed[expert], scale32[expert], **{role: globals_flat[expert]})
            else:
                raise _fail("global scale payload must be scalar or one scalar per leading expert")
        else:
            raise _fail(f"unknown module format {fmt!r}")
    if not torch.isfinite(result).all():
        raise _fail("nonfinite decoded weight (NaN code or FP32 overflow)")
    result = result.to(dtype=dtype)
    if not torch.isfinite(result).all():
        raise _fail("output dtype overflow")
    return result
