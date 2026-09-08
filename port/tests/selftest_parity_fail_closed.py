#!/usr/bin/env python3
"""Offline native-parity refusal contracts; requires torch, not CUDA or safetensors."""
import contextlib
import io
import json
import os
import tempfile
import sys
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

import torch
import glm5_layer_parity as hp


class ParityRefusalTest(unittest.TestCase):
    def run_harness(self, constructor=None, forward=None, ref_only=False, extra=()):
        config = SimpleNamespace(hidden_size=4, hc_mult=4, arch="fixture",
                                 layer_types=["ordinary_attention"], mlp_layer_types=["dense"])
        argv = ["parity", "--tests", "hc", "--seq", "2", "--device", "cuda:0", *extra]
        if ref_only:
            argv.append("--ref-only")
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.object(hp, "load_text_config", return_value=config))
            stack.enter_context(patch.object(hp, "ShardReader"))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
            stack.enter_context(patch.object(hp, "build_exl3_model", side_effect=constructor,
                                            return_value=(None, object())))
            stack.enter_context(patch.object(hp, "exl3_run_hc", side_effect=forward))
            stack.enter_context(contextlib.redirect_stdout(output))
            stack.enter_context(contextlib.redirect_stderr(output))
            with self.assertRaises(SystemExit) as exit_info:
                hp.main()
        return exit_info.exception.code, output.getvalue()

    def test_constructor_error_is_failure(self):
        code, _ = self.run_harness(constructor=RuntimeError("constructor defect"))
        self.assertEqual(code, 1)

    def test_native_forward_error_is_failure(self):
        code, _ = self.run_harness(forward=RuntimeError("native forward defect"))
        self.assertEqual(code, 1)

    def test_reference_only_is_explicitly_unqualified(self):
        code, output = self.run_harness(ref_only=True,
                                        constructor=RuntimeError("must not construct"))
        self.assertEqual(code, 0)
        # No unrelated KDA, DSA or sparse MLP exists in this fixture.
        self.assertNotIn("FAIL", output)

    def test_absent_requested_comparisons_fail_even_without_errors(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hp.Report(required=hp.required_cases(["hc"])).summary(), 1)

    def test_duplicates_and_substitute_cases_cannot_fill_coverage(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = hp.Report(required=hp.required_cases(["kda"]))
            values = hp.metrics(torch.ones(1, 2), torch.ones(1, 2))
            report.add("kda/prefill-vs-ref", values, 0, 1)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                report.add("kda/prefill-vs-ref", values, 0, 1)
            report.add("kda/unrelated", values, 0, 1)
            self.assertEqual(report.summary(), 1)
            report.add("kda/short-vs-ref", values, 0, 1)
            self.assertEqual(report.summary(), 0)

    def test_bad_geometry_empty_and_nonfinite_comparisons_refused(self):
        cases = [(torch.ones(2, 3), torch.ones(3, 2)),
                 (torch.empty(0), torch.empty(0)),
                 (torch.tensor([float("nan")]), torch.ones(1)),
                 (torch.ones(1), torch.tensor([float("inf")]))]
        for actual, expected in cases:
            with self.subTest(shape=actual.shape), self.assertRaises(ValueError):
                hp.metrics(actual, expected)
        self.assertEqual(hp.metrics(torch.zeros(2), torch.zeros(2))["cosine"], 1)

    def test_bad_metrics_and_tolerances_refused(self):
        values = hp.metrics(torch.ones(2), torch.ones(2))
        for rel, cos in ((-1, 0), (float("inf"), 0), (0, float("nan")), (0, 1.01)):
            with self.subTest(rel=rel, cos=cos), self.assertRaises(ValueError):
                hp.Report().add("case", values, rel, cos)
        with self.assertRaises(ValueError):
            hp.Report().add("case", dict(values, max_abs=float("nan")), 0, 1)

    def test_invalid_cli_lengths_refused(self):
        for option, value in (("--seq", "0"), ("--moe-tokens", "-1"), ("--moe-experts", "-2")):
            with self.subTest(option=option):
                code, _ = self.run_harness(ref_only=True, extra=(option, value))
                self.assertEqual(code, 2)

    def test_reference_receipt_preserves_missing_native_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "evidence.json")
            code, _ = self.run_harness(ref_only=True, extra=("--output", path))
            self.assertEqual(code, 0)
            with open(path) as stream:
                evidence = json.load(stream)
            self.assertEqual(evidence["status"], "reference-only-unqualified")
            self.assertEqual(evidence["missing_cases"], sorted(hp.required_cases(["hc"])))
            self.assertEqual(evidence["observed_cases"], [])
            self.assertFalse(evidence["whole_model_qualified"])
            self.assertFalse(evidence["native_serving_qualified"])

    def test_hc_config_does_not_require_unrequested_architectures(self):
        config = {"hidden_size": 4, "num_hidden_layers": 1, "rms_norm_eps": 1e-5,
                  "layer_types": ["ordinary_attention"], "hc_mult": 4,
                  "hc_sinkhorn_iters": 20, "hc_eps": 1e-6}
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "config.json"), "w") as stream:
                json.dump(config, stream)
            tc = hp.load_text_config(directory, ["hc"])
            selected = hp.pick_layers(tc, SimpleNamespace(tests=["hc"]))
            self.assertEqual(selected, (None, None, None))

    def test_expert_count_must_cover_router_without_silent_truncation(self):
        tc = SimpleNamespace(num_experts_per_tok=2, n_routed_experts=4)
        for count in (0, 1, 5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                hp.ref_moe_forward(None, "layer", torch.ones(1, 1, 2), tc, count)

    def test_module_cleanup_including_partial_load(self):
        for failure in ("load", "forward"):
            module = SimpleNamespace(
                load=lambda device: None, unload=lambda: released.append(True))
            released = []
            if failure == "load":
                def load(device):
                    raise RuntimeError("partial load")
                module.load = load
            model = SimpleNamespace(find_module=lambda key: module)
            with self.assertRaises(RuntimeError):
                with hp.exl3_load_module(model, "module", torch.device("cpu")):
                    raise RuntimeError("forward error")
            self.assertEqual(released, [True])

    def test_paged_cache_chunk_positions_and_error_cleanup(self):
        # CPU cache double is a control for the harness's paged-call semantics,
        # not evidence for native cache arithmetic.
        released = []
        class Cache:
            def __init__(self, *args):
                self.tokens = []
            def alloc(self, device):
                pass
            def free(self):
                released.append(True)
        cache_module = ModuleType("exllamav3.cache")
        cache_module.CacheLayer_MLA_fp16 = Cache
        constants = ModuleType("exllamav3.constants")
        constants.PAGE_SIZE = 4
        def forward(x, params):
            cache = params["cache"]
            expected_position = sum(t.shape[1] for t in cache.tokens)
            self.assertEqual(params["positions"].item(), expected_position)
            self.assertEqual(params["cache_seqlens"].item(), expected_position)
            cache.tokens.append(x)
            return torch.cat(cache.tokens, dim=1).cumsum(1)[:, -x.shape[1]:]
        x = torch.arange(1, 8).float().view(1, 7, 1)
        with patch.dict(sys.modules, {"exllamav3.cache": cache_module,
                                      "exllamav3.constants": constants}):
            for chunk in (None, 1, 3):
                actual = hp.exl3_run_dsa_cached(SimpleNamespace(forward=forward), x,
                                              torch.device("cpu"), chunk)
                torch.testing.assert_close(actual, x.cumsum(1), rtol=0, atol=0)
            def broken(x, params):
                raise RuntimeError("cache forward failed")
            with self.assertRaisesRegex(RuntimeError, "cache forward"):
                hp.exl3_run_dsa_cached(SimpleNamespace(forward=broken), x, torch.device("cpu"))
        self.assertEqual(len(released), 4)

    def test_sparse_pool_selection_and_partial_tail_oracle(self):
        tc = SimpleNamespace(index_head_dim=2, index_n_heads=1, index_kpool=2,
                             index_topk=2, index_tail=True)
        tensors = {
            "wq_b.weight": torch.tensor([[1.], [0.]]),
            "wk.weight": torch.tensor([[1., 0., 0.], [0., 1., 0.]]),
            "k_norm.weight": torch.ones(2), "k_norm.bias": torch.zeros(2),
            "index_kpool_compress_gate": torch.zeros(2, 3),
            "index_kpool_compress_ape": torch.zeros(2, 2),
            "weights_proj.weight": torch.tensor([[0., 0., 1.]]),
        }
        reader = SimpleNamespace(get=lambda name, device: tensors[name.split("indexer.")[1]].to(device))
        x = torch.tensor([[-1., 1., 1.], [-1., 1., 1.], [1., -1., 1.],
                          [1., -1., 1.], [0., 0., 1.], [0., 0., 1.], [1., -1., 1.]])
        mask = hp.ref_kpool_topk_indices(reader, "layer", x, torch.ones(7, 1), tc)
        expected = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1]], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expected))

    def test_checkpoint_load_corruption_refuses_before_forward(self):
        released = []
        source = {"hc_fn": torch.tensor([[1.001]], dtype=torch.float32),
                  "hc_base": torch.tensor([0.123456]), "hc_scale": torch.tensor([1.0])}

        class Module:
            def load(self, device):
                for name in ("fn", "base", "scale"):
                    # Models an accidental fp32 -> fp16 -> fp32 load conversion.
                    setattr(self, name, source["hc_" + name].half().float())

            def mix(self, *args):
                raise AssertionError("corrupted load reached native forward")

            def unload(self):
                released.append(True)

        model = SimpleNamespace(find_module=lambda key: Module())
        reader = SimpleNamespace(get=source.__getitem__)
        with self.assertRaisesRegex(RuntimeError, "checkpoint load parity"):
            hp.exl3_run_hc(model, "hc", torch.ones(1, 1, 1, 1),
                           torch.ones(1, 1, 1), torch.device("cpu"), reader)
        self.assertEqual(released, [True])


if __name__ == "__main__":
    unittest.main()
