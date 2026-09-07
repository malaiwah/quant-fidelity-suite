#!/usr/bin/env python3
"""Offline native-parity refusal contracts; requires torch, not CUDA or safetensors."""
import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import glm5_layer_parity as hp


class ParityRefusalTest(unittest.TestCase):
    def run_harness(self, constructor=None, forward=None, ref_only=False):
        config = SimpleNamespace(hidden_size=4, hc_mult=4, index_topk=16, arch="fixture",
                                 layer_types=["linear_attention", "deepseek_sparse_attention"],
                                 mlp_layer_types=["dense", "sparse"])
        argv = ["parity", "--tests", "hc", "--seq", "2", "--device", "cuda:0"]
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
        self.assertIn("UNQUALIFIED / NON-NATIVE", output)

    def test_absent_requested_comparisons_fail_even_without_errors(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hp.Report(required={"hc": 8}).summary(), 1)

    def test_checkpoint_load_corruption_refuses_before_forward(self):
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
                pass

        model = SimpleNamespace(find_module=lambda key: Module())
        reader = SimpleNamespace(get=source.__getitem__)
        with self.assertRaisesRegex(RuntimeError, "checkpoint load parity"):
            hp.exl3_run_hc(model, "hc", torch.ones(1, 1, 1, 1),
                           torch.ones(1, 1, 1), torch.device("cpu"), reader)


if __name__ == "__main__":
    unittest.main()
