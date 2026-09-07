#!/usr/bin/env python3
"""Weight reconstruction, not quantizer optimization or serving-kernel emulation.

GPTQ v1/v2 INT4, AWQ GEMM INT4, compressed-tensors pack-quantized INT4/INT8,
and MLX affine INT4/INT8. Canonical names are never rewritten: only the final
storage suffix changes to .weight. Callers retain every key outside consumed.
Config-level activation declarations are disclosed but are not simulated.
GEMV, Marlin, EXL, planar packing, mixed GPTQ dynamic overrides and CT activation
ordering are refused. Tensor geometry cannot identify these alternative layouts.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

REFERENCES = {
    "gptq": "https://github.com/ModelCloud/GPTQModel/blob/46626c213c4a4c31ca1760124dfe400bc7a03ad9/gptqmodel/utils/model_dequant.py",
    "awq": "https://github.com/casper-hansen/AutoAWQ/blob/88e4c76b20755db275574e6a03c83c84ba3bece5/awq/utils/packing_utils.py",
    "compressed-tensors": "https://github.com/vllm-project/compressed-tensors/blob/099fa98fea7f3533a8e304a081795a1136ef67c0/src/compressed_tensors/compressors/pack_quantized/base.py",
    "mlx": "mlx_surface.dequant_affine (MLX affine q * scale + bias)",
}
_FLOATS = {"F16", "BF16", "F32"}
_QUANT_SUFFIXES = {"qweight", "qzeros", "scales", "biases", "g_idx", "weight_packed", "weight_shape", "weight_scale", "weight_zero_point", "weight_g_idx", "input_scale", "input_zero_point", "input_global_scale", "output_scale", "output_zero_point", "q_perm", "q_invperm", "q_scale", "q_scale_max"}


def _fail(message):
    return ValueError(f"affine_surface: {message}")


def _integer(value, label, *, allow_minus_one=False):
    if type(value) is not int or (value <= 0 and not (allow_minus_one and value == -1)):
        raise _fail(f"{label} must be a positive integer" + (" or -1" if allow_minus_one else ""))
    return value


def _quant_config(config):
    candidates = []
    def visit(node, path):
        if not isinstance(node, Mapping):
            return
        for field in ("quantization_config", "quantization"):
            value = node.get(field)
            if value is not None:
                if not isinstance(value, Mapping):
                    raise _fail(f"{path}{field} must be an object")
                candidates.append((path + field, dict(value)))
        for field in ("text_config", "language_config", "llm_config"):
            visit(node.get(field), path + field + ".")
    visit(config, "")
    if not candidates:
        return None
    first = candidates[0][1]
    if any(value != first for _, value in candidates[1:]):
        raise _fail("ambiguous conflicting quantization declarations: " + ", ".join(p for p, _ in candidates))
    return first


def _meta(metadata, key, *, shape=None, dtypes=None, ranks=None):
    if key not in metadata:
        raise _fail(f"missing component {key}")
    item = metadata[key]
    if not isinstance(item, Mapping):
        raise _fail(f"metadata for {key} must be an object")
    actual = item.get("shape")
    if not isinstance(actual, (list, tuple)) or any(type(x) is not int or x <= 0 for x in actual):
        raise _fail(f"invalid shape for {key}: {actual}")
    actual = list(actual)
    if shape is not None and actual != list(shape):
        raise _fail(f"{key} shape {actual} != expected {list(shape)}")
    if ranks is not None and len(actual) not in ranks:
        raise _fail(f"{key}: rank {len(actual)} unsupported")
    if dtypes is not None and item.get("dtype") not in dtypes:
        raise _fail(f"{key}: dtype {item.get('dtype')} not in {sorted(dtypes)}")
    return actual


def _matches(target, prefix):
    if not isinstance(target, str):
        raise _fail("compressed-tensors targets must be strings")
    if target == "Linear":
        # Only packed modules are considered; no model class is inferred for floats.
        return True
    if target.startswith("re:"):
        try:
            return re.match(target[3:], prefix) is not None
        except re.error as exc:
            raise _fail(f"invalid target regex {target}: {exc}") from exc
    return prefix == target


def _ct_scheme(quant, prefix):
    groups = quant.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise _fail("compressed-tensors config_groups missing")
    matches = []
    for name, group in groups.items():
        if not isinstance(group, Mapping) or not isinstance(group.get("targets"), list):
            raise _fail(f"invalid config group {name}")
        if any(_matches(target, prefix) for target in group["targets"]):
            matches.append(group)
    if not matches:
        raise _fail(f"no compressed-tensors scheme targets {prefix}")
    if any(group != matches[0] for group in matches[1:]):
        raise _fail(f"ambiguous compressed-tensors schemes for {prefix}")
    ignored = quant.get("ignore", [])
    if not isinstance(ignored, list):
        raise _fail("compressed-tensors ignore must be a list")
    if any(_matches(target, prefix) for target in ignored):
        raise _fail(f"packed module {prefix} is declared ignored")
    return matches[0]


def plan_modules(config_dict, tensor_metadata):
    """Plan from safetensors metadata; CT weight_shape needs its small value list.

    No tensor payloads or model code are loaded here. Recognized but unsupported
    declarations fail closed. CT format routing must precede this reader when
    multiple readers support the compressed-tensors method.
    """
    quant = _quant_config(config_dict)
    if quant is None:
        return None
    method = str(quant.get("quant_method", "")).lower()
    if not method and "bits" in quant and "group_size" in quant:
        method = "mlx"
    if method not in REFERENCES:
        return None
    if method == "compressed-tensors" and quant.get("format") != "pack-quantized":
        raise _fail(f"unsupported compressed-tensors format {quant.get('format')}")
    if method in {"gptq", "awq"}:
        if quant.get("bits") != 4:
            raise _fail(f"{method} supports only 4 bits")
        if quant.get("dynamic"):
            raise _fail("GPTQ dynamic overrides are unsupported")
        if str(quant.get("pack_dtype", "int32")).removeprefix("torch.") != "int32":
            raise _fail("only int32 packed GPTQ/AWQ is supported")
        if method == "gptq":
            if "checkpoint_format" in quant and "format" in quant and str(quant["checkpoint_format"]).lower() != str(quant["format"]).lower():
                raise _fail("conflicting GPTQ format and checkpoint_format")
            fmt = str(quant.get("checkpoint_format", quant.get("format", "gptq"))).lower()
            if fmt not in {"gptq", "gptq_v2"}:
                raise _fail(f"unsupported GPTQ checkpoint_format {fmt}")
        else:
            fmt = str(quant.get("version", "gemm")).lower()
            if fmt != "gemm" or quant.get("zero_point", True) is not True:
                raise _fail("only asymmetric AWQ GEMM is supported, not GEMV/Marlin")
    modules, consumed, activations = {}, set(), []
    suffix = ".qweight" if method in {"gptq", "awq"} else ".weight_packed" if method == "compressed-tensors" else ".scales"
    for key in sorted(tensor_metadata):
        if not key.endswith(suffix):
            continue
        prefix = key[:-len(suffix)]
        components = {}
        def component(role, tail, **kwargs):
            stored = prefix + "." + tail
            result = _meta(tensor_metadata, stored, **kwargs)
            components[role] = stored
            return result
        spec = {"components": components, "format": method, "bits": 4}
        if method in {"gptq", "awq"}:
            packed = component("weight", "qweight", ranks={2}, dtypes={"I32"})
            n, k = (packed[1], packed[0] * 8) if method == "gptq" else (packed[1] * 8, packed[0])
            gs = _integer(quant.get("group_size"), "group_size", allow_minus_one=True)
            gs = k if gs == -1 else gs
            groups = math.ceil(k / gs)
            component("scales", "scales", shape=[groups, n], dtypes=_FLOATS)
            component("zeros", "qzeros", shape=[groups, math.ceil(n / 8)], dtypes={"I32"})
            if method == "gptq" and prefix + ".g_idx" in tensor_metadata:
                component("g_idx", "g_idx", shape=[k], dtypes={"I32", "I64"})
            elif method == "gptq" and quant.get("desc_act", False):
                raise _fail(f"desc_act requires original-column g_idx: {prefix}")
            spec.update(shape=[n, k], group_size=gs, zero_offset=1 if method == "gptq" and fmt == "gptq" else 0, format=f"{method}-{fmt}-int4")
        elif method == "mlx":
            per_tensor = quant.get("per_tensor", {})
            if not isinstance(per_tensor, Mapping):
                raise _fail("MLX per_tensor overrides must be an object")
            overrides = per_tensor.get(prefix, per_tensor.get(prefix + ".weight", quant.get(prefix, {})))
            if prefix in quant and prefix in per_tensor and quant[prefix] != per_tensor[prefix]:
                raise _fail(f"conflicting MLX per-module overrides for {prefix}")
            if not isinstance(overrides, Mapping):
                raise _fail(f"packed MLX module has disabled/invalid override: {prefix}")
            bits = _integer(overrides.get("bits", quant.get("bits")), "bits")
            if bits not in {4, 8} or overrides.get("mode", quant.get("mode", "affine")) != "affine":
                raise _fail("only MLX affine 4/8-bit is supported")
            gs = _integer(overrides.get("group_size", quant.get("group_size")), "group_size")
            scales_shape = component("scales", "scales", ranks={2, 3}, dtypes=_FLOATS)
            component("biases", "biases", shape=scales_shape, dtypes=_FLOATS)
            shape = [*scales_shape[:-1], scales_shape[-1] * gs]
            if shape[-1] * bits % 32:
                raise _fail("MLX input dimension is not whole packed words")
            component("weight", "weight", shape=[*shape[:-1], shape[-1] * bits // 32], dtypes={"U32", "I32"})
            spec.update(shape=shape, bits=bits, group_size=gs, format=f"mlx-affine-int{bits}")
        else:
            scheme = _ct_scheme(quant, prefix)
            weights = scheme.get("weights")
            if not isinstance(weights, Mapping) or weights.get("type") != "int" or weights.get("num_bits") not in {4, 8}:
                raise _fail(f"{prefix}: only CT signed INT4/INT8 is supported")
            bits = weights["num_bits"]
            lanes = 32 // bits
            if weights.get("actorder") not in (None, False) or weights.get("dynamic", False):
                raise _fail("CT activation ordering/dynamic weights unsupported")
            symmetric = weights.get("symmetric", True)
            if type(symmetric) is not bool:
                raise _fail("CT symmetric must be boolean")
            packed = component("weight", "weight_packed", ranks={2, 3}, dtypes={"I32"})
            component("weight_shape", "weight_shape", shape=[len(packed)], dtypes={"I32", "I64"})
            shape = tensor_metadata[components["weight_shape"]].get("value")
            if not isinstance(shape, (list, tuple)) or len(shape) != len(packed) or any(type(x) is not int or x <= 0 for x in shape):
                raise _fail(f"{prefix}: weight_shape requires actual positive integer values")
            shape = list(shape)
            if packed != [*shape[:-1], math.ceil(shape[-1] / lanes)]:
                raise _fail(f"{prefix}: packed shape inconsistent with weight_shape")
            strategy = weights.get("strategy", "tensor")
            if strategy == "group":
                gs = _integer(weights.get("group_size"), "group_size")
                scale_shape = [*shape[:-1], math.ceil(shape[-1] / gs)]
            elif strategy == "channel":
                gs, scale_shape = shape[-1], [*shape[:-1], 1]
            elif strategy == "tensor":
                gs, scale_shape = shape[-1], [1]
            else:
                raise _fail(f"unsupported CT strategy {strategy}")
            component("scales", "weight_scale", shape=scale_shape, dtypes=_FLOATS)
            zp_key = prefix + ".weight_zero_point"
            if not symmetric:
                zp_shape = [*shape[:-2], math.ceil(shape[-2] / lanes), scale_shape[-1]] if strategy != "tensor" else [1]
                component("zeros", "weight_zero_point", shape=zp_shape, dtypes={"I32"} if strategy != "tensor" else {"I8"})
            elif zp_key in tensor_metadata:
                component("zeros", "weight_zero_point", shape=scale_shape, dtypes={"I8"})
            for which in ("input_activations", "output_activations"):
                if scheme.get(which) is not None:
                    activations.append({"module": prefix, "kind": which, "declaration": scheme[which], "status": "activation-not-captured"})
            spec.update(shape=shape, bits=bits, group_size=gs, strategy=strategy, symmetric=symmetric, format=f"ct-pack-quantized-int{bits}")
        canonical = prefix + ".weight"
        if canonical in tensor_metadata and components.get("weight") != canonical:
            raise _fail(f"both packed and canonical weight stored: {canonical}")
        spec["component_metadata"] = {role: {"shape": list(tensor_metadata[stored]["shape"]), "dtype": tensor_metadata[stored]["dtype"]} for role, stored in components.items()}
        modules[canonical] = spec
        consumed.update(components.values())
    if not modules:
        raise _fail(f"recognized {method} declaration has no packed modules")
    orphans = [key for key in tensor_metadata if key.rsplit(".", 1)[-1] in _QUANT_SUFFIXES and key not in consumed]
    if orphans:
        raise _fail("orphan/unsupported quantization state: " + ", ".join(orphans[:12]))
    return {"format": "affine-weight-reconstruction-v1", "method": method, "modules": modules, "consumed": sorted(consumed), "activation_quantization": activations or None, "reference": REFERENCES[method]}


def _unpack_last(value, count, *, bits=4, awq=False):
    import torch
    lanes = 32 // bits
    shifts = torch.arange(lanes, device=value.device, dtype=torch.int32) * bits
    unpacked = (value.to(torch.int32).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    if awq:
        order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=value.device)
        unpacked = unpacked.index_select(-1, order)
    return unpacked.flatten(-2)[..., :count]


def decode_module(payload, spec, *, dtype, device):
    """Decode actual storage in FP32, then cast exactly once to requested dtype.

    GPTQ g_idx maps original K columns to groups; it is NOT an instruction to
    permute the stored weights. Reapplying argsort(g_idx) would corrupt desc_act.
    Work chunks bound temporary gathered group scales for noncontiguous g_idx.
    """
    import torch
    if set(payload) != set(spec["components"]):
        raise _fail("payload roles differ from planned components")
    dtype_map = {"I32": torch.int32, "I64": torch.int64, "I8": torch.int8, "U32": torch.uint32, "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
    for role, tensor in payload.items():
        meta = spec["component_metadata"][role]
        if list(tensor.shape) != meta["shape"] or tensor.dtype != dtype_map[meta["dtype"]]:
            raise _fail(f"payload {role} differs from planned shape/dtype")
    data = {role: tensor.to(device=device) for role, tensor in payload.items()}
    scale = data["scales"].float()
    if not torch.isfinite(scale).all() or (scale < 0).any():
        raise _fail("scales must be finite and nonnegative")
    shape, gs, fmt = spec["shape"], spec["group_size"], spec["format"]
    n, k = shape[-2:]
    output = torch.empty(shape, dtype=dtype, device=device)
    if fmt.startswith("mlx-"):
        from mlx_surface import dequant_affine
        biases = data["biases"].float()
        if not torch.isfinite(biases).all():
            raise _fail("MLX biases must be finite")
        weight_rows = data["weight"].reshape(-1, data["weight"].shape[-1])
        scale_rows = scale.reshape(-1, scale.shape[-1])
        bias_rows = biases.reshape(-1, biases.shape[-1])
        output_rows = output.reshape(-1, k)
        for start in range(0, weight_rows.shape[0], 256):
            end = min(weight_rows.shape[0], start + 256)
            decoded = dequant_affine(weight_rows[start:end], scale_rows[start:end], bias_rows[start:end], bits=spec["bits"], group_size=gs)
            output_rows[start:end] = decoded.to(dtype=dtype)
        return output
    if fmt.startswith(("gptq-", "awq-")):
        awq = fmt.startswith("awq-")
        zeros = ((_unpack_last(data["zeros"], n, awq=awq) + spec["zero_offset"]) & 15).float()
        groups = scale.shape[0]
        g_idx = data.get("g_idx")
        if g_idx is not None and ((g_idx < 0).any() or (g_idx >= groups).any()):
            raise _fail("g_idx outside scale group bounds")
        for start in range(0, k, 256):
            end = min(k, start + 256)
            if awq:
                codes = _unpack_last(data["weight"][start:end], n, awq=True)
            else:
                # K chunks start on whole words; GPTQ packs down K, not N.
                codes = _unpack_last(data["weight"][start // 8:math.ceil(end / 8)].T, end - start).T
            idx = g_idx[start:end].long() if g_idx is not None else torch.arange(start, end, device=device) // gs
            decoded = (codes.float() - zeros.index_select(0, idx)) * scale.index_select(0, idx)
            output[..., start:end] = decoded.T.to(dtype=dtype)
        return output
    if fmt not in {"ct-pack-quantized-int4", "ct-pack-quantized-int8"}:
        raise _fail(f"unsupported decode format {fmt}")
    if data["weight_shape"].tolist() != shape:
        raise _fail("weight_shape payload differs from planning value")
    bits = spec["bits"]
    lanes, offset = 32 // bits, 1 << (bits - 1)
    zeros = data.get("zeros")
    if spec["symmetric"]:
        if zeros is not None and torch.count_nonzero(zeros):
            raise _fail("nonzero zero_point for symmetric CT scheme")
        zeros = None
    elif spec["strategy"] != "tensor":
        zeros = (_unpack_last(zeros.transpose(-2, -1), n, bits=bits).transpose(-2, -1) - offset).float()
    else:
        if ((zeros.int() < -offset) | (zeros.int() >= offset)).any():
            raise _fail(f"CT tensor zero_point outside signed INT{bits}")
        zeros = zeros.float()
    # Groupwise broadcast: no K-wide repeat of scales/zero points, supports tails.
    for start in range(0, k, gs):
        end = min(k, start + gs)
        # group_size need not align with a packed word boundary.
        word_start = start // lanes
        codes = _unpack_last(data["weight"][..., word_start:math.ceil(end / lanes)], end - word_start * lanes, bits=bits)[..., start - word_start * lanes:] - offset
        group = start // gs
        s = scale if spec["strategy"] == "tensor" else scale[..., group:group + 1]
        z = 0 if zeros is None else zeros if spec["strategy"] == "tensor" else zeros[..., group:group + 1]
        output[..., start:end] = ((codes.float() - z) * s).to(dtype=dtype)
    return output
