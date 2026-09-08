#!/usr/bin/env python3
"""Offline (no GPU, no weights download, no network) refusal-contract
validation for the affine weight-reconstruction surface, the qwen35 GGUF
materializer, and quant_stream's activation-evidence disclosure.

Proves, on this machine, in seconds:

  1. NONFINITE DECODE REFUSAL (affine_surface.decode_module) - every decode
     branch (MLX affine, GPTQ int4, compressed-tensors pack-quantized int4)
     decodes in FP32 and casts exactly once to the requested capture dtype.
     A FINITE scale/code pair whose product overflows that dtype (F16 scale
     1e4 x code 15 overflows FP16; F32 scale 3e38 x code 7 overflows FP32)
     is REFUSED after the cast, never clamped into plausibility. In-range
     controls decode to exact known values, so each refusal is the overflow
     and not a broken decode.
  2. CT ADMISSION REFUSAL (affine_surface.plan_modules) - a pack-quantized
     compressed-tensors checkpoint plans cleanly, but the same checkpoint
     declaring transform_config or sparsity_config -- state this reader
     would silently drop, since it decodes codes/zero-points/scales only --
     is refused BY NAME, exactly as microscale_surface refuses it.
  3. POST-CAST REFUSAL (gguf_qwen35.materialize_layer) - a finite FP32
     reconstruction that overflows the requested capture dtype (1e5 into
     FP16) is refused after the cast and named; an in-range value
     round-trips through the same cast.
  4. ACTIVATION DISCLOSURE (quant_stream.evidence) - a flat ModelOpt W4A4
     declaring `with_input_scale: true` (which nvfp4_surface reads as a
     static FP4 activation declaration) and a plan carrying stored
     activation-scale components both keep their activation-not-captured
     disclosure instead of having it silently erased, while a config that
     genuinely declares nothing still reports activation_quantization: null.

Torch is required (rungs 1 and 3 drive real decode tensors); without it the
whole suite SKIPs loudly, naming the dependency, instead of reporting an
empty run as coverage.

Run:  python3 engines/tools/selftest_affine_stream_offline.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def refuses(fragment, call, *args, **kwargs):
    """Assert `call` raises a ValueError whose message NAMES the problem."""
    try:
        call(*args, **kwargs)
    except ValueError as exc:
        assert fragment in str(exc), (
            "refusal %r does not name %r" % (str(exc), fragment))
        return str(exc)
    raise AssertionError("expected a refusal mentioning %r, got none" % (fragment,))


# ---------------------------------------------------------------------------
# rung 1: the decode contract is "refuse, never clamp" AFTER the final cast.
# ---------------------------------------------------------------------------
def _rung_affine_nonfinite(torch):
    import affine_surface as affine
    passed = []

    # MLX affine branch: F16 scales/biases, 4-bit codes. 15 * 1e4 = 1.5e5.
    mlx_spec = {
        "components": {"weight": "m.weight", "scales": "m.scales", "biases": "m.biases"},
        "component_metadata": {"weight": {"shape": [2, 1], "dtype": "U32"},
                               "scales": {"shape": [2, 1], "dtype": "F16"},
                               "biases": {"shape": [2, 1], "dtype": "F16"}},
        "shape": [2, 8], "group_size": 8, "bits": 4, "format": "mlx-affine-int4"}
    mlx_payload = {"weight": torch.full((2, 1), 0xFFFFFFFF, dtype=torch.uint32),
                   "scales": torch.full((2, 1), 1e4, dtype=torch.float16),
                   "biases": torch.zeros((2, 1), dtype=torch.float16)}
    refuses("m.weight", affine.decode_module, mlx_payload, mlx_spec,
            dtype=torch.float16, device="cpu")
    ok = affine.decode_module(
        {**mlx_payload, "scales": torch.full((2, 1), 0.5, dtype=torch.float16)},
        mlx_spec, dtype=torch.float16, device="cpu")
    assert ok.dtype == torch.float16 and bool((ok == 7.5).all()), (
        "mlx in-range control decoded the wrong values")
    passed.append("1a mlx-affine int4: finite F16 scale 1e4 x code 15 refused in "
                  "FP16 after the cast (in-range control: 15 x 0.5 = 7.5 exactly)")

    # GPTQ branch: int32 qweight/qzeros packed down K. 15 * 1e4 = 1.5e5.
    gptq_spec = {
        "components": {"weight": "m.qweight", "scales": "m.scales", "zeros": "m.qzeros"},
        "component_metadata": {"weight": {"shape": [1, 2], "dtype": "I32"},
                               "scales": {"shape": [1, 2], "dtype": "F32"},
                               "zeros": {"shape": [1, 1], "dtype": "I32"}},
        "shape": [2, 8], "group_size": 8, "format": "gptq-gptq_v2-int4",
        "zero_offset": 0}
    gptq_payload = {"weight": torch.full((1, 2), -1, dtype=torch.int32),
                    "scales": torch.full((1, 2), 1e4, dtype=torch.float32),
                    "zeros": torch.zeros((1, 1), dtype=torch.int32)}
    refuses("m.qweight", affine.decode_module, gptq_payload, gptq_spec,
            dtype=torch.float16, device="cpu")
    ok = affine.decode_module({**gptq_payload, "scales": torch.full((1, 2), 0.5)},
                              gptq_spec, dtype=torch.float16, device="cpu")
    assert ok.dtype == torch.float16 and bool((ok == 7.5).all()), (
        "gptq in-range control decoded the wrong values")
    passed.append("1b gptq int4: finite F32 scale 1e4 x code 15 refused in FP16 "
                  "after the cast (in-range control: 15 x 0.5 = 7.5 exactly)")

    # CT pack-quantized branch: symmetric INT4, tensor strategy, FP32 target.
    # (15 - 8 offset) * 3e38 = 2.1e39 overflows FP32 itself.
    ct_spec = {
        "components": {"weight": "m.weight_packed", "weight_shape": "m.weight_shape",
                       "scales": "m.weight_scale"},
        "component_metadata": {"weight": {"shape": [2, 1], "dtype": "I32"},
                               "weight_shape": {"shape": [2], "dtype": "I32"},
                               "scales": {"shape": [1], "dtype": "F32"}},
        "shape": [2, 8], "group_size": 8, "bits": 4,
        "format": "ct-pack-quantized-int4", "strategy": "tensor", "symmetric": True}
    ct_payload = {"weight": torch.full((2, 1), -1, dtype=torch.int32),
                  "weight_shape": torch.tensor([2, 8], dtype=torch.int32),
                  "scales": torch.tensor([3e38], dtype=torch.float32)}
    refuses("m.weight_packed", affine.decode_module, ct_payload, ct_spec,
            dtype=torch.float32, device="cpu")
    ok = affine.decode_module({**ct_payload, "scales": torch.tensor([1.0])},
                              ct_spec, dtype=torch.float32, device="cpu")
    assert ok.dtype == torch.float32 and bool((ok == 7.0).all()), (
        "ct in-range control decoded the wrong values")
    passed.append("1c ct-pack int4: finite F32 scale 3e38 x code 7 refused even "
                  "in an FP32 capture (in-range control: 7 x 1.0 = 7.0 exactly)")
    return passed


# ---------------------------------------------------------------------------
# rung 2: CT admission must not accept declarations it would silently drop.
# ---------------------------------------------------------------------------
def _rung_ct_admission():
    import affine_surface as affine
    quant = {"quant_method": "compressed-tensors", "format": "pack-quantized",
             "config_groups": {"group_0": {"targets": ["Linear"], "weights": {
                 "num_bits": 4, "type": "int", "symmetric": True,
                 "strategy": "tensor", "dynamic": False}}}}
    metadata = {"m.weight_packed": {"shape": [2, 1], "dtype": "I32"},
                "m.weight_shape": {"shape": [2], "dtype": "I32", "value": [2, 8]},
                "m.weight_scale": {"shape": [1], "dtype": "F32"}}
    plan = affine.plan_modules({"quantization_config": dict(quant)}, metadata)
    assert plan["method"] == "compressed-tensors" and list(plan["modules"]) == ["m.weight"], (
        "the transform-free control checkpoint must plan cleanly")
    refuses("silently dropped", affine.plan_modules,
            {"quantization_config": {**quant, "transform_config": {"sequence": []}}},
            metadata)
    refuses("silently dropped", affine.plan_modules,
            {"quantization_config": {**quant, "sparsity_config": {"format": "dense-sparse"}}},
            metadata)
    return ["2 ct admission: transform_config and sparsity_config refused BY NAME "
            "(control checkpoint without them plans cleanly)"]


# ---------------------------------------------------------------------------
# rung 3: the qwen35 capture cast re-checks finiteness on its result.
# ---------------------------------------------------------------------------
def _rung_qwen35_post_cast(torch):
    import types

    import gguf_surface as ggmod
    import gguf_qwen35 as q35

    name = "model.layers.0.mlp.up_proj.weight"
    surface = types.SimpleNamespace(
        config={"linear_num_key_heads": 1, "linear_num_value_heads": 1,
                "linear_value_head_dim": 1, "linear_key_head_dim": 1},
        plan={name: q35.TensorSpec(stored="blk.0.up_proj.weight", shape=(1, 1),
                                   stored_shape=(1, 1), layer=0)},
        container=types.SimpleNamespace(tensors={}))
    # The decoded tensor is fed, not fetched: this rung targets the capture
    # cast, not the container reader, so the loader stands in for the bytes.
    box = {"x": torch.full((1, 1), 1e5, dtype=torch.float32)}  # finite, > FP16 max
    original = q35.gg.load_decoded_tensor
    q35.gg.load_decoded_tensor = lambda container, stored, device=None: box["x"].clone()
    try:
        refuses(name, q35.materialize_layer, surface, 0, torch_dtype=torch.float16)
        box["x"] = torch.full((1, 1), 1.0, dtype=torch.float32)
        ok = q35.materialize_layer(surface, 0, torch_dtype=torch.float16)
    finally:
        q35.gg.load_decoded_tensor = original
    assert q35.gg.load_decoded_tensor is original and ggmod.load_decoded_tensor is original, (
        "the patched loader was not restored")
    assert ok[name].dtype == torch.float16 and float(ok[name].item()) == 1.0, (
        "the in-range control value did not round-trip the cast")
    return ["3 qwen35 materialize: finite 1e5 refused in FP16 after the capture "
            "cast and named (in-range control round-trips)"]


# ---------------------------------------------------------------------------
# rung 4: an activation declaration must not be erased from the evidence.
# ---------------------------------------------------------------------------
def _rung_stream_disclosure():
    import quant_stream as qs

    def evidence(quant, activation_plan, reader="microscale_surface",
                 module_format="modelopt-nvfp4"):
        plan = {"_reader": reader, "reference": "pinned",
                "quantization_config": quant,
                "modules": {"a.weight": {"format": module_format}},
                "consumed": ["a.weight"] + list(
                    (activation_plan or {}).get("stored_components", ())),
                "activation_quantization": activation_plan,
                "_gate_header_barrier": True}
        stats = {"decoded_modules": 1, "formats": {module_format: 1},
                 "components": {"a.weight"}, "decoded_names": {"a.weight"},
                 "bytes_read": 4096}
        return qs.evidence(plan, stats, "bfloat16")

    declared = {"scope": "weights-reconstructed/advisory; activation-not-captured",
                "applied_to_weights": False,
                "declared": {"dynamic": False, "num_bits": 4, "type": "float"},
                "kv_cache_scheme": None, "stored_components": []}
    flat_modelopt = {"quant_method": "modelopt", "quant_algo": "NVFP4",
                     "group_size": 16}
    # (a) a flat ModelOpt W4A4: `with_input_scale: true` IS the declaration.
    ev = evidence({**flat_modelopt, "with_input_scale": True},
                  {**declared, "stored_components": []})
    assert ev["activation_quantization"] is not None, (
        "flat ModelOpt with_input_scale: true had its activation disclosure erased")
    # (b) stored activation-scale components declare, with_input_scale absent.
    ev = evidence(dict(flat_modelopt), {**declared, "stored_components": ["a.input_scale"]})
    assert ev["activation_quantization"] is not None, (
        "stored activation-scale components had their disclosure erased")
    # control: a config that declares nothing keeps reporting null.
    ev = evidence({"quant_method": "gptq", "bits": 4, "group_size": 128}, None,
                  reader="affine_surface", module_format="gptq-gptq_v2-int4")
    assert ev["activation_quantization"] is None, (
        "an undeclared config suddenly reported activation evidence")
    # control: with_input_scale: false is not a declaration either.
    ev = evidence({**flat_modelopt, "with_input_scale": False},
                  {**declared, "stored_components": []})
    assert ev["activation_quantization"] is None, (
        "with_input_scale: false was read as an activation declaration")
    return ["4 quant_stream evidence: flat ModelOpt with_input_scale: true and "
            "stored activation scales keep the activation-not-captured "
            "disclosure; undeclared configs still report null"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    try:
        import torch
    except ImportError:
        print("  SKIP  affine/stream surfaces offline: torch is not importable "
              "under %s -- rungs 1 and 3 drive real decode tensors; install "
              "torch or export FIDELITY_PYTHON" % sys.executable)
        print("selftest_affine_stream_offline: SKIP (torch missing)")
        return 0

    passed = []
    passed.extend(_rung_affine_nonfinite(torch))
    passed.extend(_rung_ct_admission())
    passed.extend(_rung_qwen35_post_cast(torch))
    passed.extend(_rung_stream_disclosure())
    for line in passed:
        print("  PASS  %s" % line)
    print("selftest_affine_stream_offline: %d rungs" % len(passed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
