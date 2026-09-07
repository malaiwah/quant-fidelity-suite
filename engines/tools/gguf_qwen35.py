"""Dense qwen35 canonical GGUF -> native Qwen3_5ForCausalLM text view.

ABI audited at llama.cpp 9a7570587ce908b0073a0458877205b80627f393,
conversion/qwen.py Qwen3NextModel / _LinearAttentionVReorderBase. GGUF has
no V-layout version field: a file-hash-bound qwen35-layout.json attestation
is required. Shape/name matches alone cannot distinguish historical layouts.
This is not qwen35moe, qwen4_exp, vision, MTP, or llama.cpp serving execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import gguf_surface as gg

CONVERTER_REVISION = "9a7570587ce908b0073a0458877205b80627f393"
LAYOUT = "qwen35-split-qkv-z-grouped-to-tiled-v1"


def _dict(config):
    return dict(config) if isinstance(config, Mapping) else config.to_dict()


def text_config(config):
    original = _dict(config)
    if original.get("model_type") not in ("qwen3_5", "qwen3_5_text"):
        raise ValueError("qwen35: only dense Qwen3.5 text/wrapper config is supported")
    text = dict(original.get("text_config", original))
    if text.get("model_type") != "qwen3_5_text":
        raise ValueError("qwen35: expected dense qwen3_5_text, not MoE or Qwen4")
    for key in ("hidden_size", "num_hidden_layers", "vocab_size", "layer_types"):
        if key in original and "text_config" in original and original[key] != text.get(key):
            raise ValueError(f"qwen35: conflicting top/text {key}")
    if text.get("attention_bias", False) or text.get("num_experts", 0):
        raise ValueError("qwen35: attention bias / expert layout not supported")
    if text.get("mtp_num_hidden_layers", 0) or original.get("mtp_num_hidden_layers", 0):
        raise ValueError("qwen35: MTP config is outside this text-only surface")
    return text


def model_view(config):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    original = _dict(config)
    text = text_config(original)
    text["architectures"] = ["Qwen3_5ForCausalLM"]
    derived = Qwen3_5TextConfig.from_dict(text)
    evidence = {
        "schema": "qfs.qwen35-native-text-view.v1", "scope": "text-only",
        "original_model_type": original.get("model_type"),
        "original_architectures": original.get("architectures"),
        "native_class": "Qwen3_5ForCausalLM", "derived_config": json.loads(derived.to_json_string(use_diff=False)),
        "original_config_sha256": gg._sha256_bytes(gg._canonical_json(original)),
        "vision": "not present in language GGUF; not borrowed or synthesized",
        "mtp": "not supported", "identity_policy": "original config and packed files retained",
    }
    return Qwen3_5ForCausalLM, derived, evidence


@dataclass(frozen=True)
class TensorSpec:
    stored: str
    shape: tuple[int, ...]
    stored_shape: tuple[int, ...]
    layer: int
    transform: str = "identity"


def canonical_plan(config):
    c = text_config(config)
    h, f, n, vocab = (int(c[k]) for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size"))
    kh, vh, kd, vd, kernel = (int(c[k]) for k in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim"))
    ah, ak, ad = int(c["num_attention_heads"]), int(c["num_key_value_heads"]), int(c["head_dim"])
    if min(h, f, n, vocab, kh, vh, kd, vd, kernel, ah, ak, ad) < 1 or vh % kh or ah % ak:
        raise ValueError("qwen35: invalid attention geometry")
    if kernel == 1:
        raise ValueError("qwen35: singleton-kernel converter squeeze vintage is unsupported")
    types = c.get("layer_types")
    if not isinstance(types, (list, tuple)) or len(types) != n or set(types) - {"linear_attention", "full_attention"}:
        raise ValueError("qwen35: explicit complete layer_types required")
    out = {}
    def add(name, stored, shape, layer=-1, transform="identity", stored_shape=None):
        out[name] = TensorSpec(stored, tuple(shape), tuple(stored_shape or shape), layer, transform)
    add("model.embed_tokens.weight", "token_embd.weight", (vocab, h))
    add("model.norm.weight", "output_norm.weight", (h,), transform="norm_offset")
    add("lm_head.weight", "output.weight", (vocab, h))
    for i, kind in enumerate(types):
        p, s = f"model.layers.{i}.", f"blk.{i}."
        for hf, stored in (("input_layernorm", "attn_norm"), ("post_attention_layernorm", "post_attention_norm")):
            add(p + hf + ".weight", s + stored + ".weight", (h,), i, "norm_offset")
        for hf, stored, shape in (("gate_proj", "ffn_gate", (f, h)), ("up_proj", "ffn_up", (f, h)), ("down_proj", "ffn_down", (h, f))):
            add(p + "mlp." + hf + ".weight", s + stored + ".weight", shape, i)
        if kind == "full_attention":
            # Upstream leaves q_proj unchanged: [head, query+gate, head_dim].
            # It is NOT all query rows followed by all gate rows.
            for hf, stored, shape, transform in (
                ("q_proj", "attn_q", (2 * ah * ad, h), "identity"),
                ("k_proj", "attn_k", (ak * ad, h), "identity"),
                ("v_proj", "attn_v", (ak * ad, h), "identity"),
                ("o_proj", "attn_output", (h, ah * ad), "identity"),
                ("q_norm", "attn_q_norm", (ad,), "norm_offset"),
                ("k_norm", "attn_k_norm", (ad,), "norm_offset"),
            ):
                add(p + "self_attn." + hf + ".weight", s + stored + ".weight", shape, i, transform)
        else:
            for hf, stored, shape, transform in (
                ("in_proj_qkv.weight", "attn_qkv.weight", (2 * kh * kd + vh * vd, h), "qkv"),
                ("in_proj_z.weight", "attn_gate.weight", (vh * vd, h), "v_rows"),
                ("in_proj_a.weight", "ssm_alpha.weight", (vh, h), "head_rows"),
                ("in_proj_b.weight", "ssm_beta.weight", (vh, h), "head_rows"),
                ("A_log", "ssm_a", (vh,), "negative_exp"),
                ("dt_bias", "ssm_dt.bias", (vh,), "head_rows"),
                ("norm.weight", "ssm_norm.weight", (vd,), "identity"),
                ("out_proj.weight", "ssm_out.weight", (h, vh * vd), "v_columns"),
            ):
                add(p + "linear_attn." + hf, s + stored, shape, i, transform)
            channels = 2 * kh * kd + vh * vd
            add(p + "linear_attn.conv1d.weight", s + "ssm_conv1d.weight", (channels, 1, kernel), i, "conv", (channels, kernel))
    return out


@dataclass(frozen=True)
class Qwen35Surface:
    container: Any
    config: dict
    plan: dict[str, TensorSpec]
    layout: dict
    file_records: tuple
    file_hash_verification: str
    architecture: str = "qwen35"


def load_gguf_surface(files, config, require_file_hashes=True, **kwargs):
    container = gg.GgufContainer([gg.GgufFile(str(p)) for p in files])
    if container.architecture != "qwen35":
        raise ValueError("qwen35: only canonical dense qwen35 supported, not qwen35moe/qwen4_exp")
    if container.remote:
        raise ValueError("qwen35: local hash-bound converter-layout attestation required")
    c = text_config(config)
    plan = canonical_plan(config)
    kv = container.kv
    expected = {
        "block_count": c["num_hidden_layers"], "embedding_length": c["hidden_size"],
        "feed_forward_length": c["intermediate_size"], "attention.head_count": c["num_attention_heads"],
        "attention.head_count_kv": c["num_key_value_heads"], "attention.key_length": c["head_dim"],
        "attention.value_length": c["head_dim"], "ssm.conv_kernel": c["linear_conv_kernel_dim"],
        "ssm.state_size": c["linear_key_head_dim"], "ssm.group_count": c["linear_num_key_heads"],
        "ssm.time_step_rank": c["linear_num_value_heads"],
        "ssm.inner_size": c["linear_value_head_dim"] * c["linear_num_value_heads"],
    }
    rope = c.get("rope_parameters") or c.get("rope_scaling") or {}
    expected["rope.dimension_count"] = int(c["head_dim"] * rope.get("partial_rotary_factor", c.get("partial_rotary_factor", 0.25)))
    expected["rope.dimension_sections"] = list(rope.get("mrope_section", [11, 11, 10])) + [0]
    expected["attention.layer_norm_rms_epsilon"] = c["rms_norm_eps"]
    if "rope_theta" in rope:
        expected["rope.freq_base"] = rope["rope_theta"]
    for key, value in expected.items():
        actual = kv.get("qwen35." + key)
        if isinstance(value, float):
            import math
            same = isinstance(actual, (int, float)) and math.isclose(actual, value, rel_tol=1e-6)
        else:
            same = actual == value
        if not same:
            raise ValueError(f"qwen35: header {key}={actual!r}, config requires {value!r}")
    if kv.get("qwen35.nextn_predict_layers", 0):
        raise ValueError("qwen35: MTP tensors are outside the declared text view")
    recurrent = kv.get("qwen35.attention.recurrent_layers")
    want = [t == "linear_attention" for t in c["layer_types"]]
    if recurrent is None:
        interval = kv.get("qwen35.full_attention_interval")
        if not isinstance(interval, int) or interval < 1:
            raise ValueError("qwen35: missing recurrent schedule")
        recurrent = [(i + 1) % interval != 0 for i in range(c["num_hidden_layers"])]
    if recurrent != want:
        raise ValueError("qwen35: recurrent/full attention schedule mismatch")
    if c.get("tie_word_embeddings") and "output.weight" not in container.tensors:
        head = plan["lm_head.weight"]
        plan["lm_head.weight"] = TensorSpec("token_embd.weight", head.shape, head.stored_shape, -1)
    required = {spec.stored for spec in plan.values()}
    if required != set(container.tensors):
        raise ValueError(f"qwen35: incomplete canonical text census; missing={sorted(required - set(container.tensors))}; unknown={sorted(set(container.tensors) - required)}")
    for name, spec in plan.items():
        row = container.tensors[spec.stored]
        if row["type"] not in gg.SUPPORTED_TYPES:
            raise ValueError(f"qwen35: unsupported type {row['type']} for {spec.stored}")
        if gg.hf_shape_of(row) != spec.stored_shape:
            raise ValueError(f"qwen35: unsupported layout for {spec.stored}: {gg.hf_shape_of(row)} != {spec.stored_shape}")
    if c.get("tie_word_embeddings") and "output.weight" in container.tensors:
        embedding, head = (container.tensors[k] for k in ("token_embd.weight", "output.weight"))
        if embedding["type"] != head["type"] or embedding["bytes"] != head["bytes"]:
            raise ValueError("qwen35: tied head has a distinct encoding; equivalence is unproven")
        for offset in range(0, head["bytes"], 8 << 20):
            size = min(8 << 20, head["bytes"] - offset)
            if container.read_tensor_range("token_embd.weight", offset, size) != container.read_tensor_range("output.weight", offset, size):
                raise ValueError("qwen35: tied config has unequal own embedding/head tensors")
    gg.audit_container(container)
    root = Path(container.files[0].location).resolve().parent
    layout_path = root / "qwen35-layout.json"
    layout = json.loads(layout_path.read_text())
    if layout.get("schema") != "qfs.qwen35-converter-layout.v1" or layout.get("layout") != LAYOUT or layout.get("converter_revision") != CONVERTER_REVISION or layout.get("source_projection_layout") != "split_qkv_z":
        raise ValueError("qwen35: unproven converter vintage/fused QKVZ layout; canonical split ABI attestation required")
    records = []
    bound = layout.get("files", {})
    for f in container.files:
        sha = gg._sha256_file(Path(f.location))
        if bound.get(f.name) != sha:
            raise ValueError(f"qwen35: converter-layout attestation not bound to current {f.name}")
        records.append({"name": f.name, "bytes": f.size, "sha256": sha})
    return Qwen35Surface(container, c, plan, layout, tuple(records), "full")


def slots(surface):
    return {name: spec.layer for name, spec in surface.plan.items()}


def _untile(tensor, axis, kh, ratio, dim):
    shape = list(tensor.shape)
    expanded = shape[:axis] + [ratio, kh, dim] + shape[axis + 1:]
    return tensor.reshape(expanded).transpose(axis, axis + 1).reshape(shape).contiguous()


def materialize_layer(surface, layer, *, torch_dtype=None, device=None, stats=None, only=None):
    import torch
    c = surface.config
    kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
    ratio, vd, qk = vh // kh, c["linear_value_head_dim"], 2 * kh * c["linear_key_head_dim"]
    names = {name for name, spec in surface.plan.items() if spec.layer == layer}
    if only is not None:
        if set(only) - names:
            raise ValueError("qwen35: requested names outside canonical layer")
        names = set(only)
    out = {}
    for name in sorted(names):
        spec = surface.plan[name]
        x = gg.load_decoded_tensor(surface.container, spec.stored, device=device)
        t = spec.transform
        if t in ("qkv", "conv"):
            x = torch.cat((x[:qk], _untile(x[qk:], 0, kh, ratio, vd)), dim=0)
            if t == "conv":
                x = x.unsqueeze(1)
        elif t == "v_rows":
            x = _untile(x, 0, kh, ratio, vd)
        elif t == "v_columns":
            x = _untile(x, 1, kh, ratio, vd)
        elif t in ("head_rows", "negative_exp"):
            x = _untile(x, 0, kh, ratio, 1)
            if t == "negative_exp":
                if not bool(torch.isfinite(x).all()) or not bool((x < 0).all()):
                    raise ValueError(f"qwen35: {spec.stored} must be finite strictly negative for log(-A)")
                x = torch.log(-x)
        elif t == "norm_offset":
            x = x - 1.0
        if tuple(x.shape) != spec.shape or not bool(torch.isfinite(x).all()):
            raise ValueError(f"qwen35: invalid reconstructed {name}")
        out[name] = x.to(dtype=torch_dtype or torch.float32).contiguous()
        if stats is not None:
            row = surface.container.tensors[spec.stored]
            stats["tensors_decoded"] = stats.get("tensors_decoded", 0) + 1
            stats["gguf_bytes_read"] = stats.get("gguf_bytes_read", 0) + int(row["bytes"])
            types = stats.setdefault("ggml_types", {})
            types[row["type"]] = types.get(row["type"], 0) + 1
    if stats is not None:
        stats["official_tensors_produced"] = stats.get("official_tensors_produced", 0) + len(out)
    return out


def decode_contract(surface):
    contract = gg.decode_contract(surface.container, surface.layout.get("build", "qwen35-text"))
    contract["quantization_config"].update({
        "decode": "GGUF blocks to fp32, inverse norm offsets/negative exponent/V-head tile order, then requested capture dtype; native text forward, not GGUF serving kernels",
        "model_view": "Qwen3_5ForCausalLM text-only; vision/MTP absent",
        "layout": surface.layout,
        "canonical_tensor_count": len(surface.plan),
        "classification": "weights_reconstructed",
    })
    return contract
