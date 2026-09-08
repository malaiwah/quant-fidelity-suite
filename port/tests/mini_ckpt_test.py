#!/usr/bin/env python3
"""CPU-only temporary checkpoint and independent oracle controls; no native qualification."""
import json
import os
import sys
import tempfile
from unittest.mock import patch

import torch
from safetensors.torch import save_file
import glm5_layer_parity as hp


def run(mini):
    H = 64            # hidden
    KH, KD, CK = 4, 8, 4          # kda heads, head_dim, conv kernel
    QLR, KVLR, NOPE, VH, NQ = 32, 16, 16, 16, 4
    INH, IHD, ITOPK, KP = 4, 8, 16, 4
    NE, TOPK_E, MI, II = 8, 4, 32, 128

    cfg = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": {
            "model_type": "glm5_next_text",
            "hidden_size": H,
            "num_hidden_layers": 2,
            "rms_norm_eps": 1e-5,
            "layer_types": ["linear_attention", "deepseek_sparse_attention"],
            "mlp_layer_types": ["dense", "sparse"],
            "linear_attn_config": {"num_heads": KH, "head_dim": KD,
                                   "short_conv_kernel_size": CK, "gate_lower_bound": -5.0},
            "num_attention_heads": NQ,
            "q_lora_rank": QLR, "kv_lora_rank": KVLR,
            "qk_nope_head_dim": NOPE, "qk_rope_head_dim": 0, "v_head_dim": VH,
            "index_n_heads": INH, "index_head_dim": IHD, "index_topk": ITOPK,
            "index_kpool": KP, "index_kpool_always_select_tail": True,
            "n_routed_experts": NE, "num_experts_per_tok": TOPK_E,
            "moe_intermediate_size": MI, "intermediate_size": II,
            "routed_scaling_factor": 2.5, "n_shared_experts": 1, "swiglu_limit": 10.0,
            "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": True,
            "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
        },
    }
    with open(os.path.join(mini, "config.json"), "w") as f:
        json.dump(cfg, f)

    torch.manual_seed(3)
    def r(*s, scale = 0.08, dtype = torch.bfloat16):
        return (torch.randn(*s) * scale).to(dtype)

    p = "model.language_model.layers"
    t = {}
    # layer 0: KDA
    k0 = f"{p}.0.self_attn"
    P = KH * KD
    t[f"{k0}.q_proj.weight"] = r(P, H)
    t[f"{k0}.k_proj.weight"] = r(P, H)
    t[f"{k0}.v_proj.weight"] = r(P, H)
    t[f"{k0}.o_proj.weight"] = r(H, P)
    t[f"{k0}.b_proj.weight"] = r(KH, H)
    t[f"{k0}.f_a_proj.weight"] = r(KD, H)
    t[f"{k0}.f_b_proj.weight"] = r(P, KD)
    t[f"{k0}.g_a_proj.weight"] = r(KD, H)
    t[f"{k0}.g_b_proj.weight"] = r(P, KD)
    t[f"{k0}.q_conv1d.weight"] = r(P, 1, CK, scale = 0.4)
    t[f"{k0}.k_conv1d.weight"] = r(P, 1, CK, scale = 0.4)
    t[f"{k0}.v_conv1d.weight"] = r(P, 1, CK, scale = 0.4)
    t[f"{k0}.A_log"] = torch.randn(KH).float() * 0.5
    t[f"{k0}.dt_bias"] = torch.randn(P).float() * 0.5
    t[f"{k0}.o_norm.weight"] = (torch.randn(KD) * 0.1 + 1).bfloat16()
    # layer 1: DSA
    k1 = f"{p}.1.self_attn"
    t[f"{k1}.q_a_proj.weight"] = r(QLR, H)
    t[f"{k1}.q_a_layernorm.weight"] = (torch.randn(QLR) * 0.1 + 1).bfloat16()
    t[f"{k1}.q_b_proj.weight"] = r(NQ * NOPE, QLR)
    t[f"{k1}.kv_a_proj_with_mqa.weight"] = r(KVLR, H)
    t[f"{k1}.kv_a_layernorm.weight"] = (torch.randn(KVLR) * 0.1 + 1).bfloat16()
    t[f"{k1}.kv_b_proj.weight"] = r(NQ * (NOPE + VH), KVLR)
    t[f"{k1}.o_proj.weight"] = r(H, NQ * VH)
    t[f"{k1}.indexer.wq_b.weight"] = r(INH * IHD, QLR)
    t[f"{k1}.indexer.wk.weight"] = r(IHD, H)
    t[f"{k1}.indexer.k_norm.weight"] = (torch.randn(IHD) * 0.1 + 1).float()
    t[f"{k1}.indexer.k_norm.bias"] = (torch.randn(IHD) * 0.05).float()
    t[f"{k1}.indexer.weights_proj.weight"] = r(INH, H)
    t[f"{k1}.indexer.index_kpool_compress_gate"] = r(IHD, H)
    t[f"{k1}.indexer.index_kpool_compress_ape"] = torch.randn(KP, IHD).float() * 0.1
    # layer 1 MoE
    m1 = f"{p}.1.mlp"
    t[f"{m1}.gate.weight"] = r(NE, H)
    t[f"{m1}.gate.e_score_correction_bias"] = torch.randn(NE).float() * 0.1
    for i in range(NE):
        t[f"{m1}.experts.{i}.gate_proj.weight"] = r(MI, H)
        t[f"{m1}.experts.{i}.up_proj.weight"] = r(MI, H)
        t[f"{m1}.experts.{i}.down_proj.weight"] = r(H, MI)
    t[f"{m1}.shared_experts.gate_proj.weight"] = r(MI, H)
    t[f"{m1}.shared_experts.up_proj.weight"] = r(MI, H)
    t[f"{m1}.shared_experts.down_proj.weight"] = r(H, MI)
    # hc tensors (layer 0)
    hc3 = 2 * 4 + 16
    t[f"{p}.0.hc_attn_fn"] = r(hc3, 4 * H, scale = 0.05)
    t[f"{p}.0.hc_attn_base"] = torch.randn(hc3).bfloat16() * 0.1
    t[f"{p}.0.hc_attn_scale"] = (torch.randn(3) * 0.1 + 1).bfloat16()

    save_file(t, os.path.join(mini, "model.safetensors"))
    wm = {k: "model.safetensors" for k in t}
    with open(os.path.join(mini, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": wm}, f)
    print("mini checkpoint written:", mini, f"({len(t)} tensors)")

    # Independent checkpoint reader: every stored tensor must retain its bytes.
    tc = hp.load_text_config(mini)
    R = hp.ShardReader(mini)
    for name, expected in t.items():
        torch.testing.assert_close(R.get(name), expected, rtol=0, atol=0)

    x_long = torch.randn(1, ITOPK + KP * 3 + 1, H).half()
    out_sparse = hp.ref_dsa_forward(R, k1, x_long, tc, "cpu", sparse=True)
    out_dense = hp.ref_dsa_forward(R, k1, x_long, tc, "cpu", sparse=False)
    assert torch.isfinite(out_sparse).all() and torch.isfinite(out_dense).all()
    assert not torch.allclose(out_sparse, out_dense), "sparse selection had no effect"
    # Causal invariance, including an incomplete trailing pool.
    for sparse in (False, True):
        short = hp.ref_dsa_forward(R, k1, x_long[:, :-2], tc, "cpu", sparse=sparse)
        full = out_sparse if sparse else out_dense
        torch.testing.assert_close(short, full[:, :-2], rtol=2e-5, atol=2e-6)
    x_short = torch.randn(1, ITOPK, H).half()
    out_dense_s = hp.ref_dsa_forward(R, k1, x_short, tc, "cpu", sparse=False)
    out_sparse_s = hp.ref_dsa_forward(R, k1, x_short, tc, "cpu", sparse=True)
    torch.testing.assert_close(out_sparse_s, out_dense_s, rtol=0, atol=0)

    # Independent convolution oracle (explicit causal sum, no conv1d).
    mixed = torch.randn(9, 3)
    weights = torch.randn(3, 4)
    expected = torch.zeros_like(mixed)
    for token in range(9):
        for tap in range(4):
            source = token + tap - 3
            if source >= 0:
                expected[token] += mixed[source] * weights[:, tap]
    expected = torch.nn.functional.silu(expected)
    actual, _ = hp.causal_conv_silu_ref(mixed, weights)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    left, state = hp.causal_conv_silu_ref(mixed[:5], weights)
    right, _ = hp.causal_conv_silu_ref(mixed[5:], weights, state)
    torch.testing.assert_close(torch.cat((left, right)), expected, rtol=1e-5, atol=1e-6)
    reset, _ = hp.causal_conv_silu_ref(mixed[5:], weights)
    assert not torch.allclose(reset, right), "cache control failed to distinguish reset state"

    resid = torch.randn(1, 4, 4, H).bfloat16()
    fn = R.get(f"{p}.0.hc_attn_fn").float()
    base = R.get(f"{p}.0.hc_attn_base").float()
    scale = R.get(f"{p}.0.hc_attn_scale").float()
    post, comb, coll = hp.mhc_pre_torch(resid, fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20)
    for sums in (comb.sum(-1), comb.sum(-2)):
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=0, atol=5e-6)
    y = torch.randn(1, 4, H).half()
    applied = hp.mhc_post_torch(y, resid.float(), post, comb)
    # Independent explicit stream accumulation catches a transposed connection matrix.
    expected = post.unsqueeze(-1) * y.float().unsqueeze(-2)
    for destination in range(4):
        for source in range(4):
            expected[:, :, destination] += comb[:, :, source, destination, None] * resid[:, :, source].float()
    torch.testing.assert_close(applied, expected, rtol=1e-5, atol=1e-6)
    # Zero logits have an analytic mix, independent of checkpoint/random values.
    post0, comb0, coll0 = hp.mhc_pre_torch(resid, torch.zeros_like(fn), scale,
                                          torch.zeros_like(base), 1e-5, 1e-6, 1e-6, 2.0, 20)
    torch.testing.assert_close(post0, torch.ones_like(post0), rtol=0, atol=0)
    torch.testing.assert_close(comb0, torch.full_like(comb0, .25), rtol=0, atol=1e-6)
    torch.testing.assert_close(coll0, resid.float().sum(-2) * (.5 + 1e-6), rtol=1e-6, atol=1e-6)
    print("Independent load, causal/cache, sparse-boundary and mHC oracle checks OK")
    output = os.path.join(mini, "layer-evidence.json")
    argv = ["glm5_layer_parity.py", "--model-dir", mini, "--ref-only", "--seq", "16",
            "--moe-tokens", "8", "--moe-experts", "8", "--device", "cpu", "--output", output]
    with patch.object(sys, "argv", argv):
        try:
            hp.main()
        except SystemExit as error:
            if error.code != 0:
                raise AssertionError(f"reference harness failed: {error.code}") from error
    with open(output) as stream:
        evidence = json.load(stream)
    assert evidence["status"] == "reference-only-unqualified"
    assert not evidence["whole_model_qualified"] and not evidence["native_serving_qualified"]
    assert evidence["missing_cases"] == evidence["required_cases"]


if __name__ == "__main__":
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="glm5-mini-") as mini:
        run(mini)
