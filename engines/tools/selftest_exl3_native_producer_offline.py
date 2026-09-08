#!/usr/bin/env python3
"""Offline numeric regressions for the v2 native-module producer, not GPU evidence.

Run with torch and numpy. No vendor import, network or CUDA execution is needed.
The small synthetic linear below computes its outputs from an explicit formula,
not from the producer's reference tensors or comparison results.
"""
from __future__ import annotations

import contextlib
import copy
import io
import importlib.util
import json
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import exl3_decoder_parity_vs_exllamav3 as xp


class FormulaLinear:
    """Independent three-input/two-output arithmetic, exactly representable."""

    def __init__(self, fault=None):
        self.config = SimpleNamespace(infer_params=SimpleNamespace(no_reconstruct=False))
        self._fused_reconstruct = True
        self.fault = fault

    def forward(self, x, params, out_dtype):
        # Not matmul and not a stored/echoed expected output.
        a, b, c = x[:, 0].double(), x[:, 1].double(), x[:, 2].double()
        result = torch.stack((2 * a - b + c / 2, -a / 2 + 3 * b + 2 * c), dim=1).to(out_dtype)
        if self.fault == "numeric":
            result[0, 0] += 1
        elif self.fault == "dtype":
            result = result.float()
        elif self.fault == "nonfinite":
            result[0, 0] = float("inf")
        elif self.fault == "empty":
            result = result[:0]
        return result


PLAN = (
    ("synthetic", "offline/a", "a" * 40, "fixture", "linear-a", "mcg"),
    ("synthetic", "offline/a", "a" * 40, "fixture", "linear-b", "mul1"),
)


def fixture(row, fault=None):
    # Two independent ways of deriving the same explicit coefficient matrix.
    weight = torch.tensor([[2, -0.5], [-1, 3], [0.5, 2]], dtype=torch.float32)
    observed_weight = FormulaLinear().forward(torch.eye(3, dtype=torch.float16), {}, torch.float16)
    record = dict(zip(("label", "repo", "revision", "shard", "name", "codebook"), row))
    record.update(
        shape_in_out=[3, 2],
        pre_hadamard=xp.compare(weight.half(), observed_weight),
        weight_fp16_primary=xp.compare(weight.half(), observed_weight),
        native_module_forward=xp.run_forward(FormulaLinear(fault), weight, observed_weight, "cpu"),
    )
    return record


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"[ok] {name}")


def range_fetch_contract():
    class Response:
        def __init__(self, status=206, content_range="bytes 10-17/1024", body=b"payload!"):
            self.status = status
            self.headers = {"Content-Range": content_range}
            self.body = body
            self.reads = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, limit=None):
            if limit is None or limit > 9:
                raise AssertionError("unbounded shard body read")
            self.reads += 1
            return self.body[:limit]

    for response in (Response(status=200), Response(content_range="bytes 11-18/1024")):
        with patch.object(xp.urllib.request, "urlopen", return_value=response):
            try:
                xp._http("https://example.invalid/shard", 10, 17, tries=1)
            except xp.ParityError:
                pass
            else:
                raise AssertionError("server ignored the requested range")
        check("incorrect HTTP range is refused before reading its body", response.reads == 0)
    for body in (b"short", b"too-long!"):
        with patch.object(xp.urllib.request, "urlopen", return_value=Response(body=body)):
            try:
                xp._http("https://example.invalid/shard", 10, 17, tries=1)
            except xp.ParityError:
                pass
            else:
                raise AssertionError("wrong response byte count was accepted")
        check("truncated or oversized ranged body is refused", True)
    with patch.object(xp.urllib.request, "urlopen", return_value=Response()):
        check("exact range uses a bounded body read",
              xp._http("https://example.invalid/shard", 10, 17, tries=1) == b"payload!")
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "cache"
        header = {"tensor.suh": {"dtype": "F16", "shape": [2], "data_offsets": [0, 8]},
                  "__header_len__": 32}
        with patch.object(xp, "_http", side_effect=AssertionError("fetch before geometry validation")):
            try:
                xp.fetch_tensor(cache, "fixture/repo", "a" * 40, "fixture", header, "tensor.suh")
            except xp.ParityError:
                pass
            else:
                raise AssertionError("tensor byte count disagrees with its geometry")
        check("bad tensor geometry causes no fetch or cache mutation", not cache.exists())


def main():
    range_fetch_contract()
    reference = torch.tensor([1.0], dtype=torch.float16)
    mismatched = torch.tensor([1.0001], dtype=torch.float32)
    check("dtype conversion cannot hide a numeric difference", not xp.compare(reference, mismatched)["equal"])
    check("equal numeric values in different dtypes are not bitwise parity",
          not xp.compare(reference, reference.float())["equal"])
    for value in (float("nan"), float("inf"), -float("inf")):
        tensor = torch.tensor([value], dtype=torch.float16)
        comparison = xp.compare(tensor, tensor.clone())
        check(f"nonfinite {value} cannot qualify", not comparison["equal"] and not comparison["valid"])
        json.dumps(comparison, allow_nan=False)
    empty = xp.compare(torch.empty(0), torch.empty(0))
    check("empty parity is invalid", not empty["equal"] and not empty["valid"])
    signed_zero = xp.compare(torch.tensor([0.0]), torch.tensor([-0.0]))
    check("bitwise parity distinguishes signed zero", not signed_zero["equal"]
          and signed_zero["byte_differing_elements"] == 1 and signed_zero["max_abs_diff"] == 0)

    records = [fixture(row) for row in PLAN]
    result = xp.summarize(records, PLAN, reference_verified=True)
    check("complete independent numeric execution fixture qualifies its declared module scope",
          result["qualification"] == "qualified")
    check("module fixture never qualifies whole model/cache",
          result["stages"]["whole_model_cache"]["qualified"] is False)
    check("numeric agreement without verified native implementation cannot qualify",
          xp.summarize(records, PLAN)["qualification"] == "not-qualified")
    for fault in ("numeric", "dtype", "nonfinite", "empty"):
        bad = [fixture(PLAN[0], fault), records[1]]
        check(f"native forward {fault} cannot qualify",
              xp.summarize(bad, PLAN, reference_verified=True)["qualification"] == "not-qualified")

    for name, subset, required in (
        ("empty plan", [], ()),
        ("no observations", [], PLAN),
        ("partial module coverage", records[:1], PLAN),
        ("duplicate module replacing missing module", [records[0], records[0]], PLAN),
        ("unexpected extra module", records + [records[0]], PLAN),
    ):
        check(name, xp.summarize(subset, required, reference_verified=True)["qualification"] == "not-qualified")

    for name in ("missing", "skipped", "partial", "wrong-branch", "truncated-weight", "weight-failed", "pre-failed"):
        bad = copy.deepcopy(records)
        forward = bad[0]["native_module_forward"]
        if name == "missing":
            del bad[0]["native_module_forward"]
        elif name == "skipped":
            forward["execution"] = "skipped"
        elif name == "partial":
            forward["cases"].pop()
            forward["observed_rows"].pop()
        elif name == "wrong-branch":
            forward["cases"][-1]["dispatch_from_pinned_predicates"] = "direct"
            forward["observed_dispatch"][-1] = "direct"
        elif name == "truncated-weight":
            bad[0]["weight_fp16_primary"] = xp.compare(torch.ones(1, 2).half(), torch.ones(1, 2).half())
        else:
            field = "weight_fp16_primary" if name == "weight-failed" else "pre_hadamard"
            bad[0][field] = xp.compare(torch.ones(3, 2).half(), torch.zeros(3, 2).half())
        check(f"{name} cannot qualify despite other passing evidence",
              xp.summarize(bad, PLAN, reference_verified=True)["qualification"] == "not-qualified")

    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "observation.json"
        existing = Path(tmp) / "existing.json"
        existing.write_text("do not overwrite", encoding="utf-8")
        xp.write_observation(fresh, result)
        check("complete observation appears as valid JSON", json.loads(fresh.read_text()) == result)
        try:
            xp.write_observation(fresh, {"replacement": True})
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing observation overwritten")
        check("exclusive publication preserves existing evidence", json.loads(fresh.read_text()) == result)
        fresh.unlink()
        try:
            xp.write_observation(fresh, {"nonfinite": float("nan")})
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite observation accepted")
        check("serialization failure creates no artifact or staging debris",
              not fresh.exists() and not list(Path(tmp).glob(".native-observation-*")))

        payloads = {
            "exllamav3/__init__.py": b"version stays unchanged",
            "exllamav3/modules/quant/exl3.py": b"native linear implementation",
            "exllamav3/modules/quant/exl3_lib/quantize.py": b"native transforms",
            "exllamav3/util/hadamard.py": b"native matrix",
            "exllamav3_ext.fixture.so": b"native binary",
        }
        wheel = Path(tmp) / "fixture.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, content in payloads.items():
                archive.writestr(name, content)
        manifest = xp.wheel_payloads(wheel)
        installed = Path(tmp) / "installed"
        for name, content in payloads.items():
            target = installed / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        xp.verify_installed_payloads(installed, manifest)
        model_spec = importlib.util.spec_from_file_location(
            "native_model_identity_test", xp.TOOLS.parent.parent / "port/tests/glm5_native_forward.py")
        model_identity = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_identity)
        distribution = SimpleNamespace(locate_file=lambda name: installed / name)
        hashes = {row["path"]: row["sha256"] for row in manifest}
        extension = installed / "exllamav3_ext.fixture.so"
        linear = installed / "exllamav3/modules/quant/exl3.py"
        xp.verify_imported_payloads(installed, manifest,
                                    extension_path=extension, linear_path=linear)
        model_identity.verify_loaded_extension(
            SimpleNamespace(__file__=str(extension)), distribution, hashes)
        for shadow in (installed / "exllamav3_ext.untracked.so", Path(tmp) / extension.name):
            shadow.write_bytes(b"different imported CUDA implementation")
            # The installed wheel remains valid. It must not certify this
            # other binary merely because it was imported under the same name.
            xp.verify_installed_payloads(installed, manifest)
            for verifier, refusal in (
                    (lambda: xp.verify_imported_payloads(
                        installed, manifest, extension_path=shadow, linear_path=linear), xp.ParityError),
                    (lambda: model_identity.verify_loaded_extension(
                        SimpleNamespace(__file__=str(shadow)), distribution, hashes), model_identity.Refusal)):
                try:
                    verifier()
                except refusal:
                    pass
                else:
                    raise AssertionError("installed wheel certified a different imported CUDA binary")
            check("untracked or external imported CUDA binary cannot inherit wheel identity", True)
        changed = installed / "exllamav3/modules/quant/exl3.py"
        original = changed.read_bytes()
        changed.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        try:
            xp.verify_installed_payloads(installed, manifest)
        except xp.ParityError:
            pass
        else:
            raise AssertionError("matching version concealed changed implementation")
        check("unchanged version cannot conceal a changed installed implementation",
              (installed / "exllamav3/__init__.py").read_bytes() == payloads["exllamav3/__init__.py"])
        # Patching availability is a platform-independent refusal scenario, not a
        # simulated successful GPU result. Any attempted I/O raises immediately.
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(xp, "_http", side_effect=AssertionError("network before refusal")), \
                patch.object(xp, "module_record", side_effect=AssertionError("fetch before refusal")), \
                patch.object(xp, "import_exllamav3", side_effect=AssertionError("vendor import before refusal")):
            for args in (["--out", str(fresh)], ["--out", str(existing)],
                         ["--out", str(xp.FROZEN_RECEIPT)]):
                try:
                    xp.main(args)
                except xp.ParityError:
                    pass
                else:
                    raise AssertionError(f"unsafe invocation was accepted: {args}")
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    xp.main([])
                except SystemExit as exc:
                    check("output destination is mandatory", exc.code == 2)
                else:
                    raise AssertionError("default output still accepted")
        check("hardware refusal creates no observation", not fresh.exists())
        check("existing evidence remains unchanged", existing.read_text(encoding="utf-8") == "do not overwrite")
    print("All native-producer offline regressions passed (synthetic CPU execution only; no native qualification).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
