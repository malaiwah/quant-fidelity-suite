#!/usr/bin/env python3
# GLM-5.3-Flash (glm5_next) layer-parity harness for the exllamav3 port.
#
# Purpose
# -------
# Validates the three new/modified compute paths of the glm5_next port against pure-torch
# reference oracles built from the checkpoint-true math (vLLM glm5_next + FLA KDA semantics,
# MLX glm5_next NoPE-MLA / kpool-indexer / sigmoid-noaux_tc MoE):
#
#   kda   KimiDeltaAttention layer  (module key  model.language_model.layers.<K>.self_attn)
#           - prefill  (T = --seq) and short (T = 32).
#             Kernel dispatch is implementation-dependent, not instrumented here.
#   dsa   NoPE MLA + kpool indexer  (module key  model.language_model.layers.<D>.self_attn)
#           - nc       (T = --seq, cache-less calibration path, dense-exact since T <= index_topk)
#           - cached   (T = --seq, paged fp16 latent cache, prefill kernels with D_r = 0)
#           - decode   (T = 96, chunk = 1, flash-decoding kernel with D_r = 0)
#           - sparse   (--long-dsa: T = index_topk + 256, exercises kpool top-k selection;
#                       reference implements the full pooled-indexer math; looser tolerance,
#                       near-tie pool selections may legitimately differ)
#   moe   288-expert sigmoid noaux_tc block  (module key  model.language_model.layers.<D>.mlp)
#           - batch    (T = --moe-tokens) and single token (T = 1)
#           - the reference computes BOTH activation conventions:
#               vLLM:  clamp(g, max=10) * sigmoid(clamp(g, max=10)) * clamp(u, +-10)
#               exl3:  min(silu(g), 10) * clamp(u, +-10)
#             The post-activation clamp difference is bounded by ~4.5e-4 per unit
#             up-value (~4.5e-3 after |up| <= 10); both conventions are reported.
#   hc    mHC HyperConnection mix/apply against independent checkpoint tensors.
#           Loaded fn/base/scale must preserve checkpoint values exactly.
#
# The reference reads checkpoint tensors independently and upcasts to fp32.
# Native GEMM loaders may convert bf16 to fp16; exactness requires representable
# values (including the fp16 subnormal boundary). Load/rounding defects therefore
# remain part of parity, not an assumed-away difference. mHC loads are checked exactly.
#
# Environment
# -----------
#   - Full parity needs: CUDA GPU, exllamav3 (with the glm5_next port registered), triton,
#     flash-linear-attention (fla.ops.kda), safetensors.
#   - CPU/reference runs require explicit --ref-only. They may exit 0 for reference
#     checks, but are UNQUALIFIED / NON-NATIVE and cannot certify the port.
#   - Missing CUDA, construction errors and native execution errors fail native runs.
#   - The full-size MoE test loads 288 experts in fp16 (~14.6 GB VRAM). With less free VRAM,
#     pass --moe-experts 32: BOTH sides are truncated to the same first-N experts (router
#     included), so the comparison remains a valid implementation-parity check.
#
# Usage
# -----
#   python port/tests/glm5_layer_parity.py --model-dir /path/to/bf16
#   python port/tests/glm5_layer_parity.py --tests kda,moe --moe-experts 32
#   python port/tests/glm5_layer_parity.py --tests dsa --long-dsa
#
# Exit code 0 = requested layer coverage passed, or explicit unqualified reference-only;
# 1 = failures/missing native coverage. Neither mode qualifies whole-model serving.

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

# --------------------------------------------------------------------------------------------
# Checkpoint access
# --------------------------------------------------------------------------------------------

class ShardReader:
    """Lazy tensor loader over a sharded safetensors checkpoint (bf16)."""

    def __init__(self, model_dir: str):
        from safetensors import safe_open
        self._safe_open = safe_open
        self.model_dir = model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            single = os.path.join(model_dir, "model.safetensors")
            assert os.path.exists(single), f"No safetensors index or file in {model_dir}"
            self.weight_map = None
            self._single = "model.safetensors"
        self._handles = {}

    def _handle(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = self._safe_open(os.path.join(self.model_dir, shard), framework = "pt", device = "cpu")
            self._handles[shard] = h
        return h

    def has(self, name: str) -> bool:
        if self.weight_map is not None:
            return name in self.weight_map
        return name in self._handle(self._single).keys()

    def get(self, name: str, device = "cpu", dtype: torch.dtype | None = None) -> torch.Tensor:
        shard = self.weight_map[name] if self.weight_map is not None else self._single
        t = self._handle(shard).get_tensor(name)
        if dtype is not None:
            t = t.to(dtype)
        return t.to(device)


def load_text_config(model_dir: str, tests=None) -> SimpleNamespace:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    tests = set(tests or ("kda", "dsa", "moe", "hc"))
    ns = SimpleNamespace(
        arch=cfg.get("architectures", ["?"])[0],
        hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        rms_norm_eps=tc["rms_norm_eps"],
        layer_types=tc["layer_types"],
        mlp_layer_types=tc.get("mlp_layer_types", []),
    )
    if "kda" in tests:
        lac = tc["linear_attn_config"]
        for name, source in (("kda_num_heads", "num_heads"), ("kda_head_dim", "head_dim"),
                             ("kda_conv_k", "short_conv_kernel_size"),
                             ("kda_lower_bound", "gate_lower_bound")):
            setattr(ns, name, lac[source])
    if "dsa" in tests:
        ns.num_q_heads = tc["num_attention_heads"]
        for name in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
                     "v_head_dim", "index_n_heads", "index_head_dim", "index_topk", "index_kpool"):
            setattr(ns, name, tc[name])
        ns.index_tail = tc.get("index_kpool_always_select_tail", True)
        if ns.qk_rope_head_dim != 0:
            raise ValueError("DSA harness requires NoPE (qk_rope_head_dim=0)")
    if "moe" in tests:
        for name in ("n_routed_experts", "num_experts_per_tok", "moe_intermediate_size",
                     "intermediate_size", "routed_scaling_factor", "n_shared_experts"):
            setattr(ns, name, tc[name])
        ns.swiglu_limit = tc.get("swiglu_limit", 10.0)
        if (tc.get("scoring_func", "sigmoid") != "sigmoid"
                or tc.get("topk_method", "noaux_tc") != "noaux_tc"
                or tc.get("norm_topk_prob", True) is not True):
            raise ValueError("MoE harness requires normalized sigmoid noaux_tc routing")
    if "hc" in tests:
        for name in ("hc_mult", "hc_sinkhorn_iters", "hc_eps"):
            setattr(ns, name, tc[name])
        ns.hc_post_mult = tc.get("mhc_post_mult_value", 2.0)
    return ns


KEY_PREFIX = "model.language_model"


# --------------------------------------------------------------------------------------------
# Reference oracles (pure torch, fp32)
# --------------------------------------------------------------------------------------------

def rms_norm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + eps) * w.float()


def l2norm_ref(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # FLA / MLX convention: x / sqrt(sum(x^2) + eps)  (eps inside the sqrt, on the sum)
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim = True) + eps)


def causal_conv_silu_ref(mixed: torch.Tensor, conv_w: torch.Tensor,
                         conv_ctx: torch.Tensor | None = None):
    """Depthwise causal conv1d + silu over [T, C] with kernel [C, K]; zero (or carried)
    initial state. Returns (y [T, C], new_ctx [K-1, C])."""
    T, C = mixed.shape
    K = conv_w.shape[-1]
    if conv_ctx is None:
        conv_ctx = mixed.new_zeros(K - 1, C)
    u = torch.cat([conv_ctx, mixed], dim = 0)                       # [T + K - 1, C]
    y = F.conv1d(u.T.unsqueeze(0), conv_w.unsqueeze(1), groups = C) # [1, C, T]
    y = F.silu(y[0].T)
    return y, u[-(K - 1):].clone()


def ref_kda_forward(R: ShardReader, key: str, x: torch.Tensor, tc: SimpleNamespace,
                    device = "cpu", state: dict | None = None):
    """Reference Kimi Delta Attention layer (vLLM glm5_next kda.py + fla.ops.kda math).

    x: [1, T, hidden] (any float dtype). Returns ([1, T, hidden] fp32, carry_state).
    carry_state carries (recurrent S, conv context) for the split-consistency self-check.
    """
    H, D = tc.kda_num_heads, tc.kda_head_dim
    P = H * D
    lb = tc.kda_lower_bound
    T = x.shape[1]
    xf = x[0].to(device = device, dtype = torch.float32)

    W = lambda n: R.get(f"{key}.{n}", device).float()

    q = xf @ W("q_proj.weight").T                                   # [T, P]
    k = xf @ W("k_proj.weight").T
    v = xf @ W("v_proj.weight").T
    mixed = torch.cat([q, k, v], dim = -1)                          # [T, 3P]

    conv_w = torch.cat([
        R.get(f"{key}.q_conv1d.weight", device),
        R.get(f"{key}.k_conv1d.weight", device),
        R.get(f"{key}.v_conv1d.weight", device),
    ], dim = 0).float().squeeze(1)                                  # [3P, K]
    conv_ctx = state["conv_ctx"] if state is not None else None
    y, new_ctx = causal_conv_silu_ref(mixed, conv_w, conv_ctx)

    q, k, v = y.split(P, dim = -1)
    q = q.view(T, H, D)
    k = k.view(T, H, D)
    v = v.view(T, H, D)

    # Forget gate: g_log = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
    # (fla kda_gate_fwd_kernel, SAFE_GATE branch; per KEY channel)
    fa = xf @ W("f_a_proj.weight").T                                # [T, 128]
    g1 = (fa @ W("f_b_proj.weight").T).view(T, H, D)
    g1 = g1 + R.get(f"{key}.dt_bias", device).float().view(H, D)
    a_log = R.get(f"{key}.A_log", device).float().view(H, 1)
    g_log = lb * torch.sigmoid(torch.exp(a_log) * g1)               # [T, H, D], in (lb, 0)

    beta = torch.sigmoid(xf @ W("b_proj.weight").T)                 # [T, H]

    qn = l2norm_ref(q)
    kn = l2norm_ref(k)
    scale = D ** -0.5

    S = state["S"].clone() if state is not None else xf.new_zeros(H, D, D)
    outs = []
    for t in range(T):
        S = S * torch.exp(g_log[t]).unsqueeze(-1)                   # per-key-channel decay
        kv_mem = torch.einsum("hkv,hk->hv", S, kn[t])
        delta = (v[t] - kv_mem) * beta[t].unsqueeze(-1)
        S = S + kn[t].unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append(torch.einsum("hkv,hk->hv", S, qn[t]) * scale)
    o = torch.stack(outs, dim = 0)                                  # [T, H, D]

    # Gated output norm: rmsnorm(o) * w * sigmoid(g2), eps = rms_norm_eps, fp32
    g2 = ((xf @ W("g_a_proj.weight").T) @ W("g_b_proj.weight").T).view(T, H, D)
    w_norm = R.get(f"{key}.o_norm.weight", device).float()
    on = o * torch.rsqrt(o.pow(2).mean(-1, keepdim = True) + tc.rms_norm_eps)
    on = on * w_norm * torch.sigmoid(g2)

    out = on.reshape(T, P) @ W("o_proj.weight").T
    return out.unsqueeze(0), {"S": S, "conv_ctx": new_ctx}


def ref_kpool_topk_indices(R: ShardReader, key: str, xf: torch.Tensor, qr: torch.Tensor,
                           tc: SimpleNamespace, device = "cpu"):
    """kpool lightning-indexer selection (MLX glm5_next reference math, B = 1, no padding).

    Returns a boolean allow-mask [T, T] (True = key visible to query) implementing:
    softmax(gate + ape)-pooled keys, relu'd per-head scores, weights_proj * H^-0.5 head mix,
    top (index_topk // kpool) complete visible pools + always-select-tail."""
    T = xf.shape[0]
    hd, Hn, kp = tc.index_head_dim, tc.index_n_heads, tc.index_kpool
    W = lambda n: R.get(f"{key}.indexer.{n}", device).float()

    q = (qr @ W("wq_b.weight").T).view(T, Hn, hd)
    k = xf @ W("wk.weight").T                                        # [T, hd]
    k = F.layer_norm(k, (hd,), W("k_norm.weight"), W("k_norm.bias"), 1e-6)
    gate = xf @ W("index_kpool_compress_gate").T                     # [T, hd]
    ape = R.get(f"{key}.indexer.index_kpool_compress_ape", device).float()   # [kp, hd]
    w_heads = (xf @ W("weights_proj.weight").T) * (Hn ** -0.5)       # [T, Hn]

    # Pool the keys: complete pools only (trailing partial pool is invalid)
    Pn = (T + kp - 1) // kp
    pad = Pn * kp - T
    if pad:
        k_p = torch.cat([k, k.new_zeros(pad, hd)])
        g_p = torch.cat([gate, gate.new_zeros(pad, hd)])
    else:
        k_p, g_p = k, gate
    gk = k_p.view(Pn, kp, hd)
    gg = g_p.view(Pn, kp, hd)
    logits = gg + ape.unsqueeze(0)
    if pad:
        valid_slot = torch.arange(Pn * kp, device = device).view(Pn, kp) < T
        logits = logits.masked_fill(~valid_slot.unsqueeze(-1), -1e30)
    probs = torch.softmax(logits, dim = 1)
    pool_keys = (probs * gk).sum(dim = 1)                            # [Pn, hd]
    pool_valid = torch.ones(Pn, dtype = torch.bool, device = device)
    if pad:
        pool_valid[-1] = False                                       # incomplete pool
    pool_end = torch.clamp(torch.arange(Pn, device = device) * kp + kp - 1, max = T - 1)

    select_k = min(tc.index_topk // kp, Pn)
    softmax_scale = hd ** -0.5
    allow = torch.zeros(T, T, dtype = torch.bool, device = device)
    scores_hp = torch.einsum("thd,pd->thp", q, pool_keys) * softmax_scale
    scores_hp = torch.clamp(scores_hp, min = 0.0)
    index_scores_all = torch.einsum("th,thp->tp", w_heads, scores_hp)  # [T, Pn]
    for i in range(T):
        cand = pool_valid & (pool_end <= i)
        s = index_scores_all[i].masked_fill(~cand, -1e30)
        kk = min(select_k, Pn)
        sel = torch.topk(s, kk).indices
        sel = sel[cand[sel]]                                         # drop -1e30 picks
        for p in sel.tolist():
            a = p * kp
            allow[i, a : min(a + kp, i + 1)] = True
        # always-select-tail: the (i + 1) mod kp trailing tokens
        tail = (i + 1) % kp
        if tc.index_tail and tail:
            allow[i, i + 1 - tail : i + 1] = True
    return allow


def ref_dsa_forward(R: ShardReader, key: str, x: torch.Tensor, tc: SimpleNamespace,
                    device = "cpu", sparse: bool = False):
    """Reference NoPE MLA (dense; exact for T <= index_topk where the indexer selects all).
    With sparse = True, applies the kpool indexer allow-mask (T > index_topk regime)."""
    Hq = tc.num_q_heads
    dn, dv = tc.qk_nope_head_dim, tc.v_head_dim
    eps = tc.rms_norm_eps
    T = x.shape[1]
    xf = x[0].to(device = device, dtype = torch.float32)
    W = lambda n: R.get(f"{key}.{n}", device).float()

    qa = xf @ W("q_a_proj.weight").T
    qr = rms_norm_ref(qa, R.get(f"{key}.q_a_layernorm.weight", device), eps)
    q = (qr @ W("q_b_proj.weight").T).view(T, Hq, dn)

    ckv = xf @ W("kv_a_proj_with_mqa.weight").T                      # [T, kv_lora]
    ckv = rms_norm_ref(ckv, R.get(f"{key}.kv_a_layernorm.weight", device), eps)
    kv = (ckv @ W("kv_b_proj.weight").T).view(T, Hq, dn + dv)
    k, v = kv[..., :dn], kv[..., dn:]

    scores = torch.einsum("qhd,khd->hqk", q, k) * (dn ** -0.5)
    pos = torch.arange(T, device = device)
    allow = pos.unsqueeze(1) >= pos.unsqueeze(0)                     # allow[q, k] = k <= q
    if sparse:
        # Also exercise the dense boundary through the actual sparse selector.
        allow = allow & ref_kpool_topk_indices(R, key, xf, qr, tc, device)
    scores = scores.masked_fill(~allow.unsqueeze(0), float("-inf"))
    p = torch.softmax(scores, dim = -1)
    o = torch.einsum("hqk,khd->qhd", p, v).reshape(T, Hq * dv)
    out = o @ W("o_proj.weight").T
    return out.unsqueeze(0)


def _act_vllm(g, u, limit):
    # SiluAndMulWithClamp forward_native: asymmetric clamp on the gate (max only)
    g = torch.clamp(g, max = limit)
    return g * torch.sigmoid(g) * torch.clamp(u, min = -limit, max = limit)


def _act_exl3(g, u, limit):
    # exllamav3 silu_mul with act_limit: post-activation clamp
    return torch.minimum(F.silu(g), torch.tensor(limit, dtype = g.dtype, device = g.device)) \
        * torch.clamp(u, min = -limit, max = limit)


def ref_moe_forward(R: ShardReader, key: str, x: torch.Tensor, tc: SimpleNamespace,
                    num_experts: int, device = "cpu"):
    """Reference sigmoid noaux_tc MoE (MLX DeepseekV32MoE math + vLLM clamp activation).

    Selection by (sigmoid scores + e_score_correction_bias); weights from UNBIASED scores of
    the selected experts, normalized, * routed_scaling_factor. Experts streamed one at a time
    from the checkpoint (bf16 -> fp32), never all resident. Returns (out_vllm, out_exl3act,
    diagnostics)."""
    top_k = tc.num_experts_per_tok
    if not 1 <= top_k <= num_experts <= tc.n_routed_experts:
        raise ValueError("expert count must cover top-k and not exceed checkpoint experts")
    rsf = tc.routed_scaling_factor
    limit = tc.swiglu_limit
    T = x.shape[1]
    xf = x[0].to(device = device, dtype = torch.float32)

    gate_w = R.get(f"{key}.gate.weight", device).float()[:num_experts]
    esb = R.get(f"{key}.gate.e_score_correction_bias", device).float()[:num_experts]

    logits = xf @ gate_w.T                                           # fp32 router
    scores = torch.sigmoid(logits)
    biased = scores + esb
    sel = torch.topk(biased, top_k, dim = -1).indices                # [T, top_k]
    w = scores.gather(-1, sel)
    w = w / w.sum(-1, keepdim = True) * rsf

    # tie margin diagnostic: gap between the top_k-th and (top_k + 1)-th biased scores
    srt = torch.sort(biased, dim = -1, descending = True).values
    margin = (srt[:, top_k - 1] - srt[:, top_k]) if top_k < num_experts else None

    out_a = torch.zeros_like(xf)
    out_b = torch.zeros_like(xf)
    flat_sel = sel.reshape(-1)
    flat_tok = torch.arange(T, device = device).repeat_interleave(top_k)
    flat_w = w.reshape(-1)
    for e in flat_sel.unique().tolist():
        m = flat_sel == e
        toks = flat_tok[m]
        coef = flat_w[m].unsqueeze(-1)
        Wg = R.get(f"{key}.experts.{e}.gate_proj.weight", device).float()
        Wu = R.get(f"{key}.experts.{e}.up_proj.weight", device).float()
        Wd = R.get(f"{key}.experts.{e}.down_proj.weight", device).float()
        xt = xf[toks]
        g = xt @ Wg.T
        u = xt @ Wu.T
        out_a.index_add_(0, toks, (_act_vllm(g, u, limit) @ Wd.T) * coef)
        out_b.index_add_(0, toks, (_act_exl3(g, u, limit) @ Wd.T) * coef)
        del Wg, Wu, Wd

    # shared expert
    Wg = R.get(f"{key}.shared_experts.gate_proj.weight", device).float()
    Wu = R.get(f"{key}.shared_experts.up_proj.weight", device).float()
    Wd = R.get(f"{key}.shared_experts.down_proj.weight", device).float()
    g = xf @ Wg.T
    u = xf @ Wu.T
    out_a = out_a + _act_vllm(g, u, limit) @ Wd.T
    out_b = out_b + _act_exl3(g, u, limit) @ Wd.T

    diag = {"min_margin": margin.min().item() if margin is not None else None,
            "median_margin": margin.median().item() if margin is not None else None}
    return out_a.unsqueeze(0), out_b.unsqueeze(0), diag


# ---- mHC reference: verbatim vLLM mhc_torch math (refs/mhc_torch.py) -----------------------

def mhc_pre_torch(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                  hc_post_mult_value, sinkhorn_repeat):
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    x = residual_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim = -1, keepdim = True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult:].reshape(num_tokens, hc_mult, hc_mult) * hc_scale[2] \
        + hc_base[2 * hc_mult:].reshape(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim = -1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim = -2, keepdim = True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim = -1, keepdim = True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim = -2, keepdim = True) + hc_sinkhorn_eps)

    layer_input = torch.sum(pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim = 1)
    return (
        post_mix.reshape(*outer_shape, hc_mult),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        layer_input.reshape(*outer_shape, hidden_size),
    )


def mhc_post_torch(x, residual, post_layer_mix, comb_res_mix):
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh", comb_res_mix.to(torch.float32), residual.to(torch.float32))
    post_term = post_layer_mix.to(torch.float32).unsqueeze(-1) * x.unsqueeze(-2).to(torch.float32)
    return mixed_residual + post_term


# --------------------------------------------------------------------------------------------
# Metrics / reporting
# --------------------------------------------------------------------------------------------

def metrics(out: torch.Tensor, ref: torch.Tensor) -> dict:
    if out.shape != ref.shape:
        raise ValueError(f"shape mismatch {tuple(out.shape)} vs {tuple(ref.shape)}")
    if not out.numel() or not torch.isfinite(out).all() or not torch.isfinite(ref).all():
        raise ValueError("comparisons require nonempty, finite tensors")
    a = out.detach().double().flatten().cpu()
    b = ref.detach().double().flatten().cpu()
    diff = (a - b).abs()
    ref_scale = b.abs().max().clamp_min(1e-12)
    cos = 1.0 if torch.equal(a, b) else float(
        (a @ b) / (a.norm().clamp_min(1e-12) * b.norm().clamp_min(1e-12)))
    m = {
        "max_abs": float(diff.max()),
        "rel_max": float(diff.max() / ref_scale),
        "rmse": float(diff.pow(2).mean().sqrt()),
        "cosine": cos,
    }
    if out.dim() >= 2:
        o2 = out.detach().double().reshape(-1, out.shape[-1]).cpu()
        r2 = ref.detach().double().reshape(-1, ref.shape[-1]).cpu()
        per_tok = (o2 - r2).abs().amax(-1) / ref_scale
        m["worst_tok_rel"] = float(per_tok.max())
        m["median_tok_rel"] = float(per_tok.median())
    return m


class Report:
    def __init__(self, ref_only=False, required=None, limitations=None):
        self.rows = []
        self.failures = 0
        self.skips = 0
        self.ref_only = ref_only
        if isinstance(required, dict):
            raise ValueError("required coverage must contain exact case names, not counts")
        self.required = set(required or ())
        self.limitations = list(limitations or ())
        self.tolerances = {}
        self.run_config = {}

    def add(self, name: str, m: dict | None, tol_rel: float, tol_cos: float,
            note: str = "", skip: str | None = None):
        if not math.isfinite(tol_rel) or tol_rel < 0 or not math.isfinite(tol_cos) or not -1 <= tol_cos <= 1:
            raise ValueError("invalid comparison tolerances")
        if any(row[0] == name for row in self.rows):
            raise ValueError(f"duplicate comparison case: {name}")
        if m is not None and (not m or not all(math.isfinite(v) for v in m.values())):
            raise ValueError("comparison metrics must be nonempty and finite")
        if skip is None and (m is None or not {"max_abs", "rel_max", "rmse", "cosine"} <= m.keys()):
            raise ValueError("comparison is missing required metrics")
        self.tolerances[name] = {"rel_max": tol_rel, "cosine_min": tol_cos}
        if skip is not None:
            status = "SKIP" if self.ref_only else "FAIL"
            self.rows.append((name, None, note or skip, status))
            self.skips += 1
            if not self.ref_only:
                self.failures += 1
            print(f"[{status}] {name}: {skip}")
            return
        ok = (m["rel_max"] <= tol_rel) and (m["cosine"] >= tol_cos)
        status = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        self.rows.append((name, m, note, status))
        wt = f" worst_tok_rel={m['worst_tok_rel']:.3e} med_tok_rel={m['median_tok_rel']:.3e}" \
            if "worst_tok_rel" in m else ""
        print(f"[{status}] {name}: max_abs={m['max_abs']:.4e} rel_max={m['rel_max']:.4e} "
              f"rmse={m['rmse']:.4e} cosine={m['cosine']:.6f}{wt}"
              f"  (tol rel<={tol_rel:g} cos>={tol_cos:g}){('  ' + note) if note else ''}")

    def summary(self) -> int:
        print("\n" + "=" * 100)
        print(f"{'test':44s} {'rel_max':>10s} {'cosine':>10s} {'status':>8s}  note")
        print("-" * 100)
        for name, m, note, status in self.rows:
            if m is None:
                print(f"{name:44s} {'-':>10s} {'-':>10s} {status:>8s}  {note}")
            else:
                print(f"{name:44s} {m['rel_max']:>10.3e} {m['cosine']:>10.6f} {status:>8s}  {note}")
        print("=" * 100)
        print(f"{self.failures} failure(s), {self.skips} skip(s)")
        missing = sorted(self.required - self.observed())
        empty_native = not self.ref_only and not self.required
        if empty_native:
            print("MISSING NATIVE COVERAGE: no required cases declared")
        if missing:
            print("MISSING NATIVE COVERAGE: " + ", ".join(missing))
        if self.ref_only:
            print("UNQUALIFIED / NON-NATIVE: reference-only checks; no native parity claim.")
        elif not self.failures and not missing and not empty_native:
            print("Requested layer comparisons passed; NOT whole-model/native-serving qualification.")
        for limitation in self.limitations:
            print("LIMITATION: " + limitation)
        return 1 if self.failures or empty_native or (missing and not self.ref_only) else 0

    def observed(self):
        return {name for name, m, _, _ in self.rows if m is not None and name in self.required}

    def finish(self, output=None):
        code = self.summary()
        if output:
            evidence = {
                "schema": "glm5-layer-parity-v1",
                "scope": "native-layer-comparisons",
                "status": "failed" if code else (
                    "reference-only-unqualified" if self.ref_only else "requested-layer-cases-passed"),
                "whole_model_qualified": False,
                "native_serving_qualified": False,
                "required_cases": sorted(self.required),
                "observed_cases": sorted(self.observed()),
                "passed_cases": sorted(name for name, _, _, status in self.rows
                                       if status == "PASS" and name in self.required),
                "missing_cases": sorted(self.required - self.observed()),
                "limitations": self.limitations,
                "run_config": self.run_config,
                "comparisons": [
                    {"case": name, "metrics": m, "note": note, "status": status,
                     "tolerances": self.tolerances[name]}
                    for name, m, note, status in self.rows],
            }
            directory = os.path.dirname(os.path.abspath(output))
            fd, temporary = tempfile.mkstemp(prefix=".layer-parity-", suffix=".json", dir=directory)
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(evidence, stream, indent=2, allow_nan=False)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, output)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return code


# --------------------------------------------------------------------------------------------
# exllamav3 side
# --------------------------------------------------------------------------------------------

def import_exl3():
    import exllamav3  # noqa: F401 (needs the CUDA extension)
    from exllamav3 import Config, Model
    return Config, Model


def build_exl3_model(model_dir: str):
    Config, Model = import_exl3()
    config = Config.from_directory(model_dir)
    model = Model.from_config(config, component = "text")
    return config, model


@contextmanager
def exl3_load_module(model, key: str, device: torch.device):
    module = model.find_module(key)
    try:
        module.load(device)
        yield module
    finally:
        module.unload()


def exl3_run_dsa_cached(module, x: torch.Tensor, device: torch.device, chunk: int | None = None):
    """Paged fp16 latent cache prefill/decode, mirroring tests/test_mla.py::run_module."""
    from exllamav3.cache import CacheLayer_MLA_fp16
    from exllamav3.constants import PAGE_SIZE
    bsz, S, _ = x.shape
    if bsz != 1 or S < 1 or (chunk is not None and chunk < 1):
        raise ValueError("cached comparison requires batch=1, positive sequence and chunk")
    npages = (S + PAGE_SIZE - 1) // PAGE_SIZE
    layer = CacheLayer_MLA_fp16(None, module, 0, npages * PAGE_SIZE)
    try:
        layer.alloc(device)
        bt = torch.arange(npages, dtype=torch.int32, device=device).view(1, npages)
        chunk = S if chunk is None else chunk
        seqlens = torch.zeros((bsz,), dtype=torch.int32, device=device)
        outs = []
        for a in range(0, S, chunk):
            b = min(a + chunk, S)
            params = {
                "attn_mode": "flash_attn",
                "cache": layer,
                "block_table": bt,
                "cache_seqlens": seqlens,
                "positions": seqlens.clone(),
            }
            outs.append(module.forward(x[:, a:b].contiguous(), params))
            seqlens = seqlens + (b - a)
        return torch.cat(outs, dim=1)
    finally:
        layer.free()


def exl3_run_moe_truncated(R: ShardReader, key: str, x: torch.Tensor,
                           tc: SimpleNamespace, n_experts: int, device: torch.device):
    """Direct BlockSparseMLP with the first n_experts (router truncated identically), served
    through a FakeSTC so it runs without loading the full 288-expert layer."""
    from exllamav3.modules import BlockSparseMLP, GatedMLP
    try:
        from exllamav3.model.config import InferParams
        infer_params = InferParams()
    except Exception:
        infer_params = SimpleNamespace(
            no_reconstruct = False, moe_cpu_offload = 0, draft_moe_cpu_offload = 0,
            moe_cpu_split = 0, moe_cpu_component = "text")

    class FakeSTC:
        def __init__(self, tensors):
            self.tensors = tensors

        def has_tensor(self, k):
            return k in self.tensors

        def has_tensor_group(self, k, subkeys):
            if isinstance(k, list):
                return all(self.has_tensor_group(kk, subkeys) for kk in k)
            return all(
                (f"{k}.{sk}" in self.tensors if isinstance(sk, str)
                 else any(f"{k}.{s}" in self.tensors for s in sk))
                for sk in subkeys)

        def get_tensor(self, k, device = None, optional = False, allow_bf16 = False,
                       float2half = False, no_defer = False, transpose = False, pad_to = None,
                       fidx = None):
            if k not in self.tensors:
                if optional:
                    return None
                raise ValueError(f"Required tensor {k} not found")
            t = self.tensors[k].to(device if device is not None else "cpu")
            if float2half and t.dtype in (torch.float32, torch.float64, torch.bfloat16):
                t = t.half()
            if transpose:
                t = t.T.contiguous()
            if pad_to is not None:
                pad = []
                for i in range(len(pad_to) - 1, -1, -1):
                    pad += [0, max(0, pad_to[i] - t.shape[i])]
                if any(pad):
                    t = F.pad(t, pad)
            return t.contiguous()

    class FakeConfig:
        def __init__(self, tensors):
            self.stc = FakeSTC(tensors)
            self.infer_params = infer_params

    t = {
        f"{key}.gate.weight": R.get(f"{key}.gate.weight")[:n_experts],
        f"{key}.gate.e_score_correction_bias":
            R.get(f"{key}.gate.e_score_correction_bias")[:n_experts],
    }
    for i in range(n_experts):
        for p in ("gate_proj", "up_proj", "down_proj"):
            t[f"{key}.experts.{i}.{p}.weight"] = R.get(f"{key}.experts.{i}.{p}.weight")
    for p in ("gate_proj", "up_proj", "down_proj"):
        t[f"{key}.shared_experts.{p}.weight"] = R.get(f"{key}.shared_experts.{p}.weight")

    fc = FakeConfig(t)
    module = BlockSparseMLP(
        config = fc,
        key = key,
        hidden_size = tc.hidden_size,
        intermediate_size = tc.moe_intermediate_size,
        num_experts = n_experts,
        num_experts_per_tok = tc.num_experts_per_tok,
        key_up = "experts.{expert_idx}.up_proj",
        key_gate = "experts.{expert_idx}.gate_proj",
        key_down = "experts.{expert_idx}.down_proj",
        key_routing_gate = "gate",
        key_e_score_bias = "gate.e_score_correction_bias",
        qmap = None,
        interm_dtype = torch.half,
        out_dtype = torch.float,
        activation_fn = "silu",
        act_limit = tc.swiglu_limit,
        router_type = "dots",
        routed_scaling_factor = tc.routed_scaling_factor,
        shared_experts = GatedMLP(
            config = fc,
            key = f"{key}.shared_experts",
            hidden_size = tc.hidden_size,
            intermediate_size = tc.moe_intermediate_size * tc.n_shared_experts,
            key_up = "up_proj",
            key_gate = "gate_proj",
            key_down = "down_proj",
            qmap = None,
            interm_dtype = torch.half,
            out_dtype = torch.float,
            activation_fn = "silu",
            act_limit = tc.swiglu_limit,
        ),
    )
    try:
        module.load(device)
        return module.forward(x, {})
    finally:
        module.unload()


def exl3_run_hc(model, key: str, resid_bf16: torch.Tensor, y_half: torch.Tensor,
                device: torch.device, source: ShardReader):
    """Check native loads against independent checkpoint values before mix/apply."""
    hc = model.find_module(key)
    try:
        hc.load(device)
        tensors = tuple(source.get(f"{key}_{name}").float().to(device)
                        for name in ("fn", "base", "scale"))
        for name, expected in zip(("fn", "base", "scale"), tensors):
            loaded = getattr(hc, name)
            if loaded.shape != expected.shape or not torch.equal(loaded.float(), expected):
                raise RuntimeError(f"mHC checkpoint load parity failed: {key}_{name}")
        streams = resid_bf16.float().to(device).contiguous()
        post, comb, collapsed = hc.mix(streams, {})
        applied = hc.apply_(streams.clone(), y_half.to(device), post, comb, {})
        return post, comb, collapsed, applied, tensors
    finally:
        hc.unload()


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def pick_layers(tc: SimpleNamespace, args):
    tests = set(args.tests.split(",")) if isinstance(args.tests, str) else set(args.tests)
    result = []
    for group, kinds, expected in (
        ("kda", getattr(tc, "layer_types", []), "linear_attention"),
        ("dsa", getattr(tc, "layer_types", []), "deepseek_sparse_attention"),
        ("moe", getattr(tc, "mlp_layer_types", []), "sparse"),
    ):
        if group not in tests:
            result.append(None)
            continue
        index = getattr(args, group + "_layer")
        if index == -1:
            try:
                index = kinds.index(expected)
            except ValueError:
                raise ValueError(f"no {expected} layer for requested {group}") from None
        if index < 0 or index >= len(kinds) or kinds[index] != expected:
            raise ValueError(f"layer {index} is not {expected}")
        result.append(index)
    return tuple(result)


def required_cases(tests, long_dsa=False):
    cases = {
        "kda": ["prefill-vs-ref", "short-vs-ref"],
        "dsa": ["nc-vs-ref", "cached-prefill-vs-ref", "decode-vs-ref"]
               + (["sparse-kpool-vs-ref"] if long_dsa else []),
        "moe": ["batch-vs-ref(vllm-act)", "batch-vs-ref(exl3-act)", "bsz1-vs-ref(vllm-act)"],
        "hc": [f"{tag}-{part}" for tag in ("decode(R=8)", "prefill(R=48)")
               for part in ("mix-post", "mix-comb", "mix-collapsed", "apply")],
    }
    return {f"{group}/{case}" for group in tests for case in cases[group]}


def main():
    ap = argparse.ArgumentParser(description = "GLM-5.3-Flash layer-parity harness")
    ap.add_argument("--model-dir", default = "/home/glm53k6/models/bf16")
    ap.add_argument("--device", default = "cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ref-device", default = None,
                    help = "device for the reference math (default: same as --device if cuda, else cpu)")
    ap.add_argument("--tests", default = "kda,dsa,moe,hc",
                    help = "comma list from {kda,dsa,moe,hc}")
    ap.add_argument("--kda-layer", type = int, default = -1)
    ap.add_argument("--dsa-layer", type = int, default = -1)
    ap.add_argument("--moe-layer", type = int, default = -1)
    ap.add_argument("--hc-layer", type=int, default=0,
                    help="mHC layer, independent of attention/MLP kinds")
    ap.add_argument("--seq", type = int, default = 512,
                    help = "prefill length for kda/dsa (must be <= index_topk for exact DSA parity)")
    ap.add_argument("--moe-tokens", type = int, default = 64)
    ap.add_argument("--moe-experts", type = int, default = 0,
                    help = "0 = auto (288 if >18 GiB free VRAM else 32). N < 288 truncates BOTH sides identically.")
    ap.add_argument("--long-dsa", action = "store_true",
                    help = "additionally test the sparse kpool-indexer regime at T = index_topk + 256")
    ap.add_argument("--seed", type = int, default = 17)
    ap.add_argument("--ref-only", action = "store_true",
                    help = "UNQUALIFIED / NON-NATIVE: run reference oracles + self-checks only")
    ap.add_argument("--output", help="atomically write machine-readable layer evidence (not model qualification)")
    args = ap.parse_args()
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    if not tests or len(tests) != len(set(tests)) or set(tests) - {"kda", "dsa", "moe", "hc"}:
        ap.error("--tests must be a nonempty, unique list from {kda,dsa,moe,hc}")
    args.tests = tests
    if args.seq < 1 or ("kda" in tests and args.seq < 2) or args.moe_tokens < 1:
        ap.error("sequence/token lengths must be positive; KDA split check requires --seq >= 2")
    if args.moe_experts < 0:
        ap.error("--moe-experts must be zero (auto) or positive")
    if args.long_dsa and "dsa" not in tests:
        ap.error("--long-dsa requires --tests dsa")

    torch.manual_seed(args.seed)
    tc = load_text_config(args.model_dir, tests)
    R = ShardReader(args.model_dir)
    rep = Report(args.ref_only, required_cases(tests, args.long_dsa), [
        "Layer-local comparisons only: whole-model logits, generation and serving are untested.",
        "Native KDA carried-state/cache parity is untested; dispatch is not instrumented.",
        "DSA sparse cached prefill/decode, noncontiguous pages, cache reuse and multi-batch are untested.",
    ])
    if not args.long_dsa:
        rep.limitations.append("Long sparse DSA is untested (--long-dsa not requested).")

    device = torch.device(args.device)
    cuda_ok = device.type == "cuda" and torch.cuda.is_available()
    ref_device = args.ref_device or (str(device) if cuda_ok else "cpu")

    try:
        kda_l, dsa_l, moe_l = pick_layers(tc, args)
    except ValueError as error:
        ap.error(str(error))
    if "hc" in tests and not 0 <= args.hc_layer < len(tc.layer_types):
        ap.error("--hc-layer must name an existing layer")
    kda_key = f"{KEY_PREFIX}.layers.{kda_l}.self_attn"
    dsa_key = f"{KEY_PREFIX}.layers.{dsa_l}.self_attn"
    moe_key = f"{KEY_PREFIX}.layers.{moe_l}.mlp"
    hc_key = f"{KEY_PREFIX}.layers.{args.hc_layer}.hc_attn"

    if "dsa" in tests and (tc.index_topk < 1 or tc.index_kpool < 1
                           or args.seq > tc.index_topk or not tc.index_tail):
        ap.error("dense DSA requires positive index geometry, --seq <= index_topk and always-select-tail")
    n_experts = args.moe_experts
    if "moe" in tests:
        if not 1 <= tc.num_experts_per_tok <= tc.n_routed_experts:
            ap.error("invalid checkpoint routed-expert/top-k counts")
        if n_experts == 0:
            full_memory = cuda_ok and torch.cuda.mem_get_info(device)[0] > 18 * (1 << 30)
            n_experts = tc.n_routed_experts if full_memory else min(
                max(32, tc.num_experts_per_tok), tc.n_routed_experts)
        if not tc.num_experts_per_tok <= n_experts <= tc.n_routed_experts:
            ap.error("--moe-experts must cover routing top-k and not exceed checkpoint experts")
        rep.limitations.append(
            f"MoE experts requested for comparison: {n_experts}/{tc.n_routed_experts}; "
            f"selection={'auto' if args.moe_experts == 0 else 'explicit'}; "
            + ("truncated router/expert population does NOT qualify full MoE."
               if n_experts != tc.n_routed_experts else "full expert population requested."))
    rep.run_config = dict(vars(args), effective_moe_experts=n_experts if "moe" in tests else None,
                          selected_kda_layer=kda_l, selected_dsa_layer=dsa_l, selected_moe_layer=moe_l)

    print(f"model:      {args.model_dir}  ({tc.arch})")
    print(f"layers:     kda={kda_l}  dsa={dsa_l}  moe={moe_l}")
    print(f"device:     exl3={device}  ref={ref_device}")
    print(f"tests:      {tests}  seq={args.seq}  moe_tokens={args.moe_tokens}\n")

    # Inputs (identical bytes on both sides: generated fp16 and upcast where needed)
    H = tc.hidden_size
    x_kda = torch.randn(1, args.seq, H).half()
    x_kda_short = torch.randn(1, 32, H).half()
    x_dsa = torch.randn(1, args.seq, H).half()
    decode_length = min(96, tc.index_topk) if "dsa" in tests else 1
    x_dsa_dec = torch.randn(1, decode_length, H).half()
    x_moe = torch.randn(1, args.moe_tokens, H).half()
    x_moe1 = x_moe[:, :1].contiguous()

    # ---- exllamav3 availability ------------------------------------------------------------
    model = None
    exl3_err = None
    if not args.ref_only and cuda_ok:
        try:
            _, model = build_exl3_model(args.model_dir)
        except Exception as e:
            exl3_err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
    elif not cuda_ok:
        exl3_err = "no CUDA device; use --ref-only for unqualified reference checks"
    else:
        exl3_err = "--ref-only"
    if model is None and not args.ref_only:
        rep.add("native/construction", None, 0, 0, skip=exl3_err)
        sys.exit(rep.finish(args.output))

    # ---- KDA -------------------------------------------------------------------------------
    if "kda" in tests:
        t0 = time.time()
        ref_kda, _ = ref_kda_forward(R, kda_key, x_kda, tc, ref_device)
        ref_kda_s, _ = ref_kda_forward(R, kda_key, x_kda_short, tc, ref_device)
        print(f"(kda reference computed in {time.time() - t0:.1f}s)")

        # Oracle self-check: state-carried split run must agree within fp32 tolerance.
        half = args.seq // 2
        y1, st = ref_kda_forward(R, kda_key, x_kda[:, :half], tc, ref_device)
        y2, _ = ref_kda_forward(R, kda_key, x_kda[:, half:], tc, ref_device, state = st)
        m = metrics(torch.cat([y1, y2], dim = 1), ref_kda)
        rep.add("kda/oracle-split-consistency", m, tol_rel = 1e-4, tol_cos = 0.999999,
                note = "reference self-check")

        if model is not None:
            try:
                with exl3_load_module(model, kda_key, device) as mod:
                    out = mod.forward(x_kda.to(device), {})
                    rep.add("kda/prefill-vs-ref", metrics(out, ref_kda),
                            tol_rel=3e-2, tol_cos=0.999, note=f"T={args.seq}; dispatch unverified")
                    out_s = mod.forward(x_kda_short.to(device), {})
                    rep.add("kda/short-vs-ref", metrics(out_s, ref_kda_s),
                            tol_rel=3e-2, tol_cos=0.999, note="T=32; dispatch unverified")
            except Exception as e:
                traceback.print_exc()
                rep.add("kda/exl3", None, 0, 0, skip = f"exl3 KDA failed: {type(e).__name__}: {e}")
        else:
            rep.add("kda/exl3", None, 0, 0, skip = exl3_err)

    # ---- DSA -------------------------------------------------------------------------------
    if "dsa" in tests:
        t0 = time.time()
        ref_dsa = ref_dsa_forward(R, dsa_key, x_dsa, tc, ref_device)
        ref_dsa_dec = ref_dsa_forward(R, dsa_key, x_dsa_dec, tc, ref_device)
        print(f"(dsa reference computed in {time.time() - t0:.1f}s)")

        if model is not None:
            try:
                with exl3_load_module(model, dsa_key, device) as mod:
                    positions = torch.zeros((1,), dtype=torch.int32, device=device)
                    out_nc = mod.forward(x_dsa.to(device),
                                         {"attn_mode": "flash_attn_nc", "positions": positions})
                    rep.add("dsa/nc-vs-ref", metrics(out_nc, ref_dsa),
                            tol_rel=1e-2, tol_cos=0.999, note=f"T={args.seq} dense (<= index_topk)")
                    out_c = exl3_run_dsa_cached(mod, x_dsa.to(device), device)
                    rep.add("dsa/cached-prefill-vs-ref", metrics(out_c, ref_dsa),
                            tol_rel=1e-2, tol_cos=0.999, note="paged fp16 cache, D_r=0 kernels")
                    out_d = exl3_run_dsa_cached(mod, x_dsa_dec.to(device), device, chunk=1)
                    rep.add("dsa/decode-vs-ref", metrics(out_d, ref_dsa_dec),
                            tol_rel=1e-2, tol_cos=0.999, note=f"T={decode_length}, chunk=1; dispatch unverified")
                    if args.long_dsa:
                        T_long = tc.index_topk + 256
                        x_long = torch.randn(1, T_long, H).half()
                        ref_sp = ref_dsa_forward(R, dsa_key, x_long, tc, ref_device, sparse=True)
                        out_sp = mod.forward(x_long.to(device),
                                             {"attn_mode": "flash_attn_nc", "positions": positions})
                        rep.add("dsa/sparse-kpool-vs-ref", metrics(out_sp, ref_sp),
                                tol_rel=1e-1, tol_cos=0.99,
                                note=f"T={T_long} > index_topk; near-tie pool picks may differ")
            except Exception as e:
                traceback.print_exc()
                rep.add("dsa/exl3", None, 0, 0, skip = f"exl3 DSA failed: {type(e).__name__}: {e}")
        else:
            rep.add("dsa/exl3", None, 0, 0, skip = exl3_err)

    # ---- MoE -------------------------------------------------------------------------------
    if "moe" in tests:
        full = n_experts == tc.n_routed_experts

        t0 = time.time()
        ref_a, ref_b, diag = ref_moe_forward(R, moe_key, x_moe, tc, n_experts, ref_device)
        ref1_a, ref1_b, _ = ref_moe_forward(R, moe_key, x_moe1, tc, n_experts, ref_device)
        print(f"(moe reference computed in {time.time() - t0:.1f}s; experts={n_experts}, "
              f"router tie margin: {diag}; null means every expert selected)")

        if model is not None:
            try:
                if full:
                    with exl3_load_module(model, moe_key, device) as mod:
                        out = mod.forward(x_moe.to(device), {})
                        out1 = mod.forward(x_moe1.to(device), {})
                else:
                    out = exl3_run_moe_truncated(R, moe_key, x_moe.to(device),
                                                 tc, n_experts, device)
                    out1 = exl3_run_moe_truncated(R, moe_key, x_moe1.to(device),
                                                  tc, n_experts, device)
                rep.add("moe/batch-vs-ref(vllm-act)", metrics(out, ref_a),
                        tol_rel=5e-2, tol_cos=0.998, note=f"T={args.moe_tokens}; experts={n_experts}")
                rep.add("moe/batch-vs-ref(exl3-act)", metrics(out, ref_b),
                        tol_rel=5e-2, tol_cos=0.998, note="same output, exl3 clamp convention")
                rep.add("moe/bsz1-vs-ref(vllm-act)", metrics(out1, ref1_a),
                        tol_rel=5e-2, tol_cos=0.998, note="T=1 routing path")
            except Exception as e:
                traceback.print_exc()
                rep.add("moe/exl3", None, 0, 0, skip = f"exl3 MoE failed: {type(e).__name__}: {e}")
        else:
            rep.add("moe/exl3", None, 0, 0, skip = exl3_err)

        # Oracle self-check: the two activation conventions must agree to ~1e-3 rel
        m = metrics(ref_b, ref_a)
        rep.add("moe/oracle-act-convention-delta", m, tol_rel = 1e-2, tol_cos = 0.9999,
                note = "vLLM vs exl3 clamp (expected tiny)")

    # ---- mHC -------------------------------------------------------------------------------
    if "hc" in tests:
        # Two input shapes target the native small-R and large-R paths. Native fn_h
        # rounding is part of the numerical comparison, not assumed lossless.
        hc_cases = [("decode(R=8)", 8, 2e-3), ("prefill(R=48)", 48, 1e-3)]
        if model is not None:
            for tag, S, tol in hc_cases:
                resid = (torch.randn(1, S, tc.hc_mult, H) * 0.5).bfloat16()
                y_site = (torch.randn(1, S, H) * 0.5).half()
                try:
                    post, comb, collapsed, applied, (fn, base, scale) = \
                        exl3_run_hc(model, hc_key, resid, y_site, device, R)
                    r_post, r_comb, r_coll = mhc_pre_torch(
                        resid.to(device), fn, scale, base,
                        rms_eps = tc.rms_norm_eps, hc_pre_eps = tc.hc_eps,
                        hc_sinkhorn_eps = tc.hc_eps, hc_post_mult_value = tc.hc_post_mult,
                        sinkhorn_repeat = tc.hc_sinkhorn_iters)
                    rep.add(f"hc/{tag}-mix-post", metrics(post, r_post),
                            tol_rel = tol, tol_cos = 0.99999)
                    rep.add(f"hc/{tag}-mix-comb", metrics(comb, r_comb),
                            tol_rel = tol, tol_cos = 0.99999)
                    rep.add(f"hc/{tag}-mix-collapsed", metrics(collapsed.float(), r_coll),
                            tol_rel = 1e-2, tol_cos = 0.9999, note = "ext returns half")
                    r_applied = mhc_post_torch(y_site.to(device), resid.float().to(device),
                                               r_post, r_comb)
                    rep.add(f"hc/{tag}-apply", metrics(applied, r_applied),
                            tol_rel = 1e-2, tol_cos = 0.9999)
                except Exception as e:
                    traceback.print_exc()
                    rep.add(f"hc/{tag}", None, 0, 0,
                            skip = f"exl3 HC failed: "
                                   f"{type(e).__name__}: {e}")
        else:
            rep.add("hc/exl3", None, 0, 0, skip = exl3_err)

    sys.exit(rep.finish(args.output))


if __name__ == "__main__":
    main()
