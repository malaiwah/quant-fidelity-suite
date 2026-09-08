#!/usr/bin/env python3
"""Real-tensor parity of `exl3hf_surface.decode_payload_hf` against exllamav3's OWN reconstruction.

    engines/tools/exl3_decoder_parity_vs_exllamav3.py --install \\
        --out /tmp/exl3-native-observation-v2.json

One command on a CUDA host (python 3.12, torch 2.11.0 cu12x/cu13x): it installs
the pinned exllamav3 v1.4.2 release wheel (URL + sha256 recorded below and in
the receipt), range-fetches ONLY the payload tensors of the modules in
MODULE_PLAN (a few MB per module, never a shard), decodes each with our
decoder and with exllamav3's `LinearEXL3` (the class its loader builds from
stored `trellis/suh/svh/<codebook>` tensors, `modules/linear.py::load_exl3`),
and writes a new, explicitly addressed v2 observation. Existing destinations
and the frozen v1 receipt are never overwritten.

Two stages are compared, because exllamav3 rounds differently from us:

* pre_hadamard: `LinearEXL3.get_inner_weight_tensor()` = `exllamav3_ext.reconstruct`
  (`exllamav3_ext/quant/reconstruct.cu`), the trellis unpack + codebook + tile
  layout as fp16 [in, out]. Pure integer/LUT work on both sides, so this stage
  is asserted BITWISE. It is the stage the served GEMM kernel consumes.
* weight: `LinearEXL3.get_weight_tensor()` = the above, then exllamav3's
  `preapply_had_l` (fp32 matmul, cast to fp16), `*= suh` (fp16), `preapply_had_r`
  (fp32, cast to fp16), `*= svh` (fp16): four fp16 roundings. Ours keeps fp32
  through both Hadamards and rounds once, so the fp16-cast weights may differ
  at the fp16 ULP. The primary comparison remains our original decoder cast
  explicitly once to fp16; diagnostics never override primary failure.

Per module the receipt also carries a COMMITTED WINDOW: the first 8x8 trellis
tiles plus the matching 128 `suh`/`svh` values (base64) and the digests of
exllamav3's two outputs on that window, so `selftest_exl3hf_offline.py` can
re-assert the pre_hadamard stage bitwise on any host, and the whole result on
a host where `import exllamav3` succeeds.

Native-module forward is also measured with deterministic fp16 inputs at
1, 8, 145 and 1024 rows through LinearEXL3.forward(x, {}, out_dtype=fp16).
The independent dense reference uses CPU fp64 accumulation of our original
reconstructed fp32 weights, then casts once to fp16. Exact parity is required;
different native arithmetic is observed, not excused with a tolerance.
These sampled, bias-free modules do NOT qualify an entire model or its cache.
Exit 0 means all required primary stages qualify; exit 1 means a completed
observation did not qualify; exit 2 means a prerequisite/execution error.

Why this needs a GPU: exllamav3 1.4.2 has no CPU reconstruction. The PyPI
wheel is pure python and `import exllamav3` (`exllamav3/__init__.py` ->
`model.config` -> ... -> `exllamav3/ext.py:147`) JIT-builds the CUDA extension
through `torch.utils.cpp_extension.load` unless a precompiled `exllamav3_ext`
is installed; without a toolkit that dies in `_join_cuda_home` ("CUDA_HOME
environment variable is not set"). With one, `reconstruct` is still a
`__global__` kernel launched on the current CUDA stream
(`reconstruct.cu:11-84,108-109`); `exllamav3_ext/cpu/` holds an int8 MoE GEMM,
not a reconstruct. `--fetch-only` runs the fetch and our half anywhere.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import math
import os
import re
import platform
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import exl3hf_surface as xs  # noqa: E402

FROZEN_SCHEMA = "malaiwah.exl3-decoder-parity-vs-exllamav3.v1"
FROZEN_RECEIPT = TOOLS / "layer-outer-evidence" / "exl3-decoder-parity-vs-exllamav3.json"
SCHEMA = "malaiwah.exl3-decoder-parity-vs-exllamav3.v2"
FORWARD_ROWS = (1, 8, 145, 1024)
FORWARD_BRANCHES = ("direct", "direct", "reconstruct-unfused", "reconstruct-fused")

EXLLAMAV3_VERSION = "1.4.2"
EXLLAMAV3_COMMIT = "5f3c537ca9d89893d771256f5c43c93656553fbb"  # git tag v1.4.2
EXLLAMAV3_RELEASE = "https://github.com/turboderp-org/exllamav3/releases/download/v1.4.2/"
# Release wheels carrying a precompiled `exllamav3_ext` (no JIT build), keyed by the
# CUDA major of the torch that will import them. Digests taken 2026-09-05.
EXLLAMAV3_WHEELS = {
    12: ("exllamav3-1.4.2+cu128.torch2.11.0-cp312-cp312-linux_x86_64.whl",
         "1cca2df47f671938a3ee508cad8eaae9e170a7202f541021b13c9803a0a0550a"),
    13: ("exllamav3-1.4.2+cu132.torch2.11.0-cp312-cp312-linux_x86_64.whl",
         "fb131e9c97ec270f5d72e28e4331197b0360fa55f10c67d87b6418a0a029fc7d"),
}
WHEEL_TORCH = "2.11.0"
WHEEL_PYTHON = (3, 12)
RECONSTRUCT_ENTRYPOINT = (
    "exllamav3.modules.quant.exl3.LinearEXL3.get_weight_tensor "
    "(get_inner_weight_tensor -> exllamav3_ext.reconstruct, exllamav3_ext/quant/reconstruct.cu; "
    "then exl3_lib.quantize.preapply_had_l/_r and the suh/svh fp16 multiplies)"
)

MAX_HEADER_BYTES = 64 << 20
MAX_RANGE_BYTES = 512 << 20
WINDOW_TILES = 8  # 8 x 8 tiles = one 128 x 128 Hadamard block, the smallest self-contained window

# (label, repo, revision, shard, module, codebook[, objects]). One module per (release, K)
# for davidsyoung, the two Fruit expert-0 modules the reconstruction receipt
# already names, and one drowzeys module per codebook. K is read from the
# header and recorded; it is not an input. The optional 7th element names
# where an object lives when it is NOT `<module>.<object>` in the same shard:
# {"suh"|"svh": (shard, tensor name)} -- the layer-shared rotation vectors of
# the GLM-5.2 layouts (`layer_outer.exl3_rotation_groups`; evidence in
# layer-outer-evidence/glm52-exl3-layouts-parity.json), which exllamav3's
# LinearEXL3 takes as plain suh/svh tensors once resolved by name.
MODULE_PLAN: Tuple[Tuple[Any, ...], ...] = (
    ("fruit", "malaiwah/GLM-5.2-SIQ-Fruit", "c1798e3676fa16b4a874381171adab1e3033fbd5",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.0.down_proj.rank0", "mcg"),
    ("fruit", "malaiwah/GLM-5.2-SIQ-Fruit", "c1798e3676fa16b4a874381171adab1e3033fbd5",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.0.gate_proj.rank0", "mcg"),
    ("dy30", "davidsyoung/GLM-5.3-EXL3-TR3-3.0bpw", "eeab94eb6e95b4e4d13d94af55ab3c420d6f52d3",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.0.down_proj.rank0", "mcg"),
    ("dy325", "davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw", "6d6bd738c0c1635513e0bd0fdf0302049bd820a9",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.0.down_proj.rank0", "mcg"),
    ("dy325", "davidsyoung/GLM-5.3-EXL3-TR3-3.25bpw", "6d6bd738c0c1635513e0bd0fdf0302049bd820a9",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.3.down_proj.rank0", "mcg"),
    # 3.42's layer-3 expert-0/3 down_proj rank0 are byte-identical to 3.25's (same tier,
    # same atoms), so 3.42 contributes an expert that is K4 here and K3 in 3.25, on rank 1.
    ("dy342", "davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw", "99c6f951333d2b38f1efefa533c7afadf0d376e3",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.20.gate_proj.rank1", "mcg"),
    ("dy342", "davidsyoung/GLM-5.3-EXL3-TR3-3.42bpw", "99c6f951333d2b38f1efefa533c7afadf0d376e3",
     "model-layer-003.safetensors", "model.layers.3.mlp.experts.3.up_proj.rank1", "mcg"),
    ("drowzeys", "drowzeys/keys-GLM-5.3-EXL3", "ebf3c8bb0ed869b8f96a6ade9c8d365a49bdbad5",
     "model-00001-of-00041.safetensors", "model.layers.3.mlp.experts.0.gate_proj", "mcg"),
    ("drowzeys", "drowzeys/keys-GLM-5.3-EXL3", "ebf3c8bb0ed869b8f96a6ade9c8d365a49bdbad5",
     "model-00002-of-00041.safetensors", "model.layers.4.mlp.experts.0.gate_proj", "mul1"),
    # GLM-5.2 shared_h_v1: the down_proj rank's svh is the layer's shared vector.
    ("willfalco", "willfalco/GLM-5.2-EXL3-TR3-3.42bpw", "700c99dfa75d61cba4dda1ce9a36478bc217728d",
     "model-layer-010.safetensors", "model.layers.10.mlp.experts.0.down_proj.rank0", "mcg",
     {"svh": ("model-layer-010.safetensors",
              "model.layers.10.mlp.experts.shared_h.down_proj.rank0.svh")}),
    # jpsequeira keeps the shared vectors in their own shard; plus its exl3 wq_b (K6).
    ("jpsequeira", "jpsequeira/GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2",
     "b92479840ef92fbeb7d774187f91cf5a2a659ade",
     "projection-mixed-layer-010.safetensors", "model.layers.10.mlp.experts.0.down_proj.rank0", "mcg",
     {"svh": ("shared-h-layer-010.safetensors",
              "model.layers.10.mlp.experts.shared_h.down_proj.rank0.svh")}),
    ("jpsequeira", "jpsequeira/GLM-5.2-EXL3-TR3-3.40bpw-KVarN-K4V2",
     "b92479840ef92fbeb7d774187f91cf5a2a659ade",
     "exl3-exemption-layer-010.safetensors", "model.layers.10.self_attn.indexer.wq_b", "mcg"),
    # brandonmusic r7_shared: unsharded experts; gate_up_suh serves gate AND up, down_svh down.
    ("brandonmusic", "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
     "7c73450f05a151439d0f184f216b1eefcc394a31",
     "r7-experts-layer-010.safetensors", "model.layers.10.mlp.experts.0.down_proj", "mcg",
     {"svh": ("r7-experts-layer-010.safetensors", "model.layers.10.mlp.experts.r7_shared.down_svh")}),
    ("brandonmusic", "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
     "7c73450f05a151439d0f184f216b1eefcc394a31",
     "r7-experts-layer-010.safetensors", "model.layers.10.mlp.experts.0.up_proj", "mcg",
     {"suh": ("r7-experts-layer-010.safetensors", "model.layers.10.mlp.experts.r7_shared.gate_up_suh")}),
    # brandonmusic's dense-6 non-routed module (K6), stock layout.
    ("brandonmusic", "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
     "7c73450f05a151439d0f184f216b1eefcc394a31",
     "model-layer-010.safetensors", "model.layers.10.self_attn.q_b_proj", "mcg"),
)

_NP_DTYPE = {"I16": "<i2", "I32": "<i4", "F16": "<f2"}
_EXPECTED_DTYPE = {"trellis": "I16", "suh": "F16", "svh": "F16", "mcg": "I32", "mul1": "I32"}


class ParityError(RuntimeError):
    pass


def _fail(message: str) -> ParityError:
    return ParityError(f"exl3_decoder_parity_vs_exllamav3: {message}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_tensor(value) -> str:
    import numpy as np

    return sha256_bytes(np.ascontiguousarray(value.detach().cpu().contiguous().numpy()).tobytes())


# --------------------------------------------------------------------------
# ranged fetch (header + exact tensor byte spans; never a shard)
# --------------------------------------------------------------------------
def _http(url: str, start: Optional[int] = None, end: Optional[int] = None, tries: int = 4) -> bytes:
    if (type(start) is not int or type(end) is not int or start < 0 or end < start
            or end - start + 1 > MAX_RANGE_BYTES):
        raise _fail("payload request needs an explicit bounded byte range")
    expected = end - start + 1
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as response:
                content_range = re.fullmatch(
                    r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", response.headers.get("Content-Range", ""))
                if (response.status != 206 or content_range is None
                        or tuple(map(int, content_range.groups()[:2])) != (start, end)
                        or int(content_range.group(3)) <= end):
                    raise _fail("server did not honor the exact requested byte range")
                length = response.headers.get("Content-Length")
                if length is not None and length != str(expected):
                    raise _fail("ranged response declares an unexpected byte count")
                data = response.read(expected + 1)
            if len(data) != expected:
                raise _fail(f"range {start}-{end} of {url} returned {len(data)} bytes")
            return data
        except ParityError:
            raise
        except Exception:  # noqa: BLE001 - retried, re-raised on the last attempt
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise _fail("unreachable")


def shard_header(cache: Path, repo: str, revision: str, shard: str) -> Dict[str, Any]:
    """The safetensors header of one shard by two range requests, cached on disk."""
    if xs._REVISION.fullmatch(revision) is None:
        raise _fail(f"{repo}: revision must be the immutable 40-hex commit, got {revision!r}")
    path = cache / repo.replace("/", "__") / revision / f"{shard}.header.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{shard}"
    (length,) = struct.unpack("<Q", _http(url, 0, 7))
    if length <= 0 or length > MAX_HEADER_BYTES:
        raise _fail("safetensors header length is outside the bounded observation contract")
    header = json.loads(_http(url, 8, 8 + length - 1))
    header["__header_len__"] = length
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(header, sort_keys=True), encoding="utf-8")
    return header


def fetch_tensor(cache: Path, repo: str, revision: str, shard: str, header: Dict[str, Any], name: str,
                 obj: Optional[str] = None):
    """One tensor's exact bytes by range -> (torch tensor, sha256, byte count).

    `obj` names the payload object the tensor stands for when the tensor's own
    suffix does not (a layer-shared `down_svh` / `gate_up_suh` / `rank0.svh`)."""
    import numpy as np
    import torch

    entry = header.get(name)
    if entry is None:
        raise _fail(f"{repo}@{revision[:8]} {shard} has no tensor {name}")
    expected = _EXPECTED_DTYPE[obj or name.rsplit(".", 1)[1]]
    if entry["dtype"] != expected:
        raise _fail(f"{name} is {entry['dtype']}, expected {expected}")
    start, end = entry["data_offsets"]
    shape = entry.get("shape")
    if (not isinstance(shape, list) or any(type(size) is not int or size <= 0 for size in shape)
            or type(start) is not int or type(end) is not int or start < 0 or end <= start
            or end - start != math.prod(shape) * np.dtype(_NP_DTYPE[expected]).itemsize
            or end - start > MAX_RANGE_BYTES):
        raise _fail(f"{name}: invalid or oversized tensor geometry/byte range")
    path = cache / repo.replace("/", "__") / revision / f"{name}.bin"
    if path.exists():
        raw = path.read_bytes()
        if len(raw) != end - start:
            raise _fail(f"cached {path} is {len(raw)} bytes, header says {end - start}; delete it")
    else:
        base = 8 + int(header["__header_len__"])
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{shard}"
        raw = _http(url, base + start, base + end - 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, path)
    array = np.frombuffer(raw, dtype=_NP_DTYPE[entry["dtype"]]).copy().reshape(entry["shape"])
    return torch.from_numpy(array), sha256_bytes(raw), len(raw)


# --------------------------------------------------------------------------
# our decoder, and the pre-Hadamard stage it passes through
# --------------------------------------------------------------------------
def ours_pre_hadamard(trellis, codebook: str):
    """fp16 [in, out]: unpack + codebook LUT + tile layout, exactly as decode_payload_hf
    composes them (exl3hf_surface.py, decode_payload_hf, before the first Hadamard)."""
    import torch

    bits = trellis.shape[-1] // 16
    states = xs.unpack_trellis_states_anybits(trellis, bits)
    indices = (states.to(torch.int64) & 0xFFFF).long()
    values = xs.codebook_lut(codebook, states.device).index_select(0, indices.flatten()).reshape_as(states)
    values = values.index_select(-1, torch.argsort(xs._permutation(states.device)))
    k_tiles, n_tiles, _ = values.shape
    return values.reshape(k_tiles, n_tiles, 16, 16).permute(0, 2, 1, 3).reshape(k_tiles * 16, n_tiles * 16).contiguous()


def ours_weight(trellis, suh, svh, codebook: str):
    """fp32 [in, out] (exllamav3 orientation) from decode_payload_hf's [out, in]."""
    return xs.decode_payload_hf(trellis, suh, svh, codebook=codebook).T.contiguous()


def native_export_reference(pre, suh, svh, device):
    """Independent unpack/LUT input, native export rounding, shared torch BLAS.

    This is a separate numerical program, not a change to the historical
    fp32 decoder. It does not emulate fused reconstruction or direct GEMM.
    """
    import torch

    if (pre.ndim != 2 or pre.dtype != torch.float16
            or suh.dtype != torch.float16 or svh.dtype != torch.float16):
        raise _fail("native export reference requires fp16 matrix and rotations")
    k, n = pre.shape
    if (not k or not n or k % 128 or n % 128
            or tuple(suh.shape) != (k,) or tuple(svh.shape) != (n,)):
        raise _fail("native export reference requires exact 128-block geometry")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in (pre, suh, svh)):
        raise _fail("native export reference inputs must be finite")
    h = xs._hadamard(device, torch.float32)
    weight = (h @ pre.to(device).float().view(k // 128, 128, n)).half().view(k, n)
    weight *= suh.to(device)[:, None]
    weight = (weight.float().view(k, n // 128, 128) @ h).half().view(k, n)
    weight *= svh.to(device)[None, :]
    return weight.cpu().contiguous()


def compare(ours, theirs) -> Dict[str, Any]:
    """Strict finite, nonempty, same-dtype byte parity; fp64 numeric diagnostics.

    Never cast the observed tensor to conceal a dtype or precision mismatch.
    Explicitly rounded reference comparisons must be named by their caller.
    """
    import torch

    ours = ours.detach().cpu().contiguous()
    theirs = theirs.detach().cpu().contiguous()
    same_shape = tuple(ours.shape) == tuple(theirs.shape)
    same_dtype = ours.dtype == theirs.dtype
    finite = bool(torch.isfinite(ours).all() and torch.isfinite(theirs).all())
    valid = same_shape and same_dtype and ours.numel() > 0 and finite
    byte_equal = same_shape and same_dtype and torch.equal(
        ours.reshape(-1).view(torch.uint8), theirs.reshape(-1).view(torch.uint8))
    result = {
        "equal": bool(valid and byte_equal),
        "valid": bool(valid),
        "ours_dtype": str(ours.dtype), "theirs_dtype": str(theirs.dtype),
        "ours_shape": list(ours.shape), "theirs_shape": list(theirs.shape),
        "same_dtype": same_dtype, "same_shape": same_shape, "finite": finite,
        "elements": int(ours.numel()), "observed_elements": int(theirs.numel()),
        "max_abs_diff": None, "mean_abs_diff": None, "rmse": None,
        "differing_elements": None, "byte_differing_elements": None,
        "first_difference": None,
    }
    if same_shape and ours.numel():
        a, b = ours.double(), theirs.double()
        unequal = a != b
        result["differing_elements"] = int(unequal.sum())
        if same_dtype:
            unequal = (ours.reshape(-1).view(torch.uint8).reshape(-1, ours.element_size())
                       != theirs.reshape(-1).view(torch.uint8).reshape(-1, theirs.element_size())).any(dim=1)
            result["byte_differing_elements"] = int(unequal.sum())
        if bool(unequal.any()):
            index = int(unequal.reshape(-1).nonzero()[0])
            av, bv = a.reshape(-1)[index], b.reshape(-1)[index]
            result["first_difference"] = {
                "flat_index": index,
                "ours": float(av) if bool(torch.isfinite(av)) else None,
                "theirs": float(bv) if bool(torch.isfinite(bv)) else None,
            }
        if finite:
            diff = (a - b).abs()
            result.update(max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                          rmse=float(diff.square().mean().sqrt()))
    return result


def window_of(trellis, suh, svh):
    k = min(WINDOW_TILES, trellis.shape[0])
    n = min(WINDOW_TILES, trellis.shape[1])
    if k != WINDOW_TILES or n != WINDOW_TILES:
        raise _fail(f"module smaller than one {WINDOW_TILES}x{WINDOW_TILES}-tile window: {tuple(trellis.shape)}")
    return (trellis[:k, :n, :].contiguous(), suh[: k * 16].contiguous(), svh[: n * 16].contiguous())


def b64(tensor) -> str:
    import numpy as np

    return base64.b64encode(np.ascontiguousarray(tensor.cpu().numpy()).tobytes()).decode("ascii")

def unb64(text: str, dtype: str, shape) -> Any:
    import numpy as np
    import torch

    return torch.from_numpy(np.frombuffer(base64.b64decode(text), dtype=_NP_DTYPE[dtype]).copy().reshape(shape))


# --------------------------------------------------------------------------
# exllamav3: install the pinned wheel, import, reconstruct
# --------------------------------------------------------------------------
def wheel_for_this_stack():
    import torch

    if sys.version_info[:2] != WHEEL_PYTHON:
        raise _fail(f"release wheels are cp{WHEEL_PYTHON[0]}{WHEEL_PYTHON[1]}; this is python {platform.python_version()}")
    if torch.__version__.split("+", 1)[0] != WHEEL_TORCH:
        raise _fail(f"release wheels are built for torch {WHEEL_TORCH}; this is {torch.__version__}")
    if not torch.cuda.is_available() or not torch.version.cuda:
        raise _fail("no CUDA device: exllamav3's reconstruct is a CUDA kernel (see module docstring)")
    major = int(torch.version.cuda.split(".")[0])
    if major not in EXLLAMAV3_WHEELS:
        raise _fail(f"no pinned wheel for CUDA {torch.version.cuda}")
    name, digest = EXLLAMAV3_WHEELS[major]
    return {"name": name, "url": EXLLAMAV3_RELEASE + name.replace("+", "%2B"), "sha256": digest,
            "cuda_tag": name.split("+")[1].split(".")[0]}


def wheel_payloads(path):
    """Bind installed implementation bytes to the verified wheel, not its version label."""
    payloads = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename
            if info.is_dir() or not (
                    name.startswith("exllamav3/") or name.startswith("exllamav3_ext.")):
                continue
            if name.startswith("/") or ".." in Path(name).parts:
                raise _fail("wheel contains an unsafe implementation path")
            raw = archive.read(info)
            payloads.append({"path": name, "bytes": len(raw), "sha256": sha256_bytes(raw)})
    names = {row["path"] for row in payloads}
    required = {"exllamav3/__init__.py", "exllamav3/modules/quant/exl3.py",
                "exllamav3/modules/quant/exl3_lib/quantize.py", "exllamav3/util/hadamard.py"}
    if not required <= names or not any(name.startswith("exllamav3_ext.") for name in names):
        raise _fail("wheel lacks the required native implementation closure")
    return payloads


def verify_installed_payloads(root, payloads):
    if not payloads:
        raise _fail("installed implementation manifest is empty")
    root = Path(root).resolve()
    for row in payloads:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise _fail("installed implementation manifest has an unsafe path")
        path = root / relative
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
                or path.stat().st_size != row["bytes"]
                or xs._sha256_file(path) != row["sha256"]):
            raise _fail("installed implementation bytes differ from the verified wheel: " + row["path"])


def verify_imported_payloads(root, payloads, *, extension_path, linear_path):
    """Bind the active implementation, not merely files beside it, to the wheel."""
    root = Path(root).resolve()
    inventory = {row["path"]: row for row in payloads}
    for actual, expected_name in (
            (extension_path, None),
            (linear_path, "exllamav3/modules/quant/exl3.py")):
        if not actual:
            raise _fail("imported native implementation has no file identity")
        path = Path(actual)
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError:
            raise _fail("imported native implementation is outside the verified wheel") from None
        if expected_name is None:
            if not relative.startswith("exllamav3_ext.") or "/" in relative:
                raise _fail("imported extension is not a verified top-level wheel binary")
        elif relative != expected_name:
            raise _fail("imported linear implementation does not match the verified wheel path")
        row = inventory.get(relative)
        if (row is None or path.is_symlink() or not path.is_file()
                or path.stat().st_size != row["bytes"]
                or xs._sha256_file(path) != row["sha256"]):
            raise _fail("imported native implementation bytes differ from the verified wheel")


def write_observation(path, receipt):
    """Publish complete JSON exclusively; no empty/partial or overwritten evidence."""
    text = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-observation-", dir=path.parent) as temporary:
        staged = Path(temporary) / "observation.json"
        with staged.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staged, 0o644)
        os.link(staged, path)


def install_exllamav3(log) -> Dict[str, Any]:
    wheel = wheel_for_this_stack()
    with tempfile.TemporaryDirectory(prefix="exl3wheel-") as tmp:
        path = Path(tmp) / wheel["name"]
        log(f"fetching {wheel['url']}")
        with urllib.request.urlopen(urllib.request.Request(wheel["url"]), timeout=600) as response, path.open("wb") as out:
            digest = hashlib.sha256()
            for chunk in iter(lambda: response.read(1 << 22), b""):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != wheel["sha256"]:
            raise _fail(f"wheel digest {digest.hexdigest()} != pinned {wheel['sha256']}")
        wheel["implementation_files"] = wheel_payloads(path)
        log(f"wheel sha256 verified; pip install {wheel['name']}")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-input",
                        "--force-reinstall", "--no-deps", str(path)], check=True)
    wheel["installed_by"] = "current interpreter -m pip install --force-reinstall --no-deps <verified wheel>"
    return wheel


def import_exllamav3(expect_precompiled: bool = True):
    """Require the pinned version; --install additionally requires its precompiled extension."""
    try:
        exllamav3 = importlib.import_module("exllamav3")
    except Exception as exc:  # noqa: BLE001 - the receipt names the failure
        raise _fail(f"import exllamav3 failed: {type(exc).__name__}: {exc}") from exc
    version = getattr(exllamav3, "__version__", None) or importlib.import_module("exllamav3.version").__version__
    if version != EXLLAMAV3_VERSION:
        raise _fail(f"exllamav3 {version} imported; this parity pins {EXLLAMAV3_VERSION}")
    ext = importlib.import_module("exllamav3.ext").exllamav3_ext
    ext_file = getattr(ext, "__file__", "") or ""
    precompiled = bool(ext_file) and Path(ext_file).resolve().parent == Path(exllamav3.__file__).resolve().parent.parent
    if expect_precompiled and not precompiled:
        raise _fail(f"exllamav3_ext at {ext_file!r} is not the release wheel's precompiled extension")
    exl3 = importlib.import_module("exllamav3.modules.quant.exl3")
    return exllamav3, exl3, {
        "version": version, "package_file": "exllamav3/__init__.py",
        "extension_file": Path(ext_file).name, "extension_precompiled": precompiled,
        "extension_sha256": xs._sha256_file(Path(ext_file)),
        "linear_source_file": "exllamav3/modules/quant/exl3.py",
        "linear_source_sha256": xs._sha256_file(Path(exl3.__file__))}


def native_linear(exl3, trellis, suh, svh, codebook: str, marker, device):
    """Build the pinned vendor class, not a torch.nn.Module wrapper."""
    trellis = trellis.to(device)
    suh = suh.to(device)
    svh = svh.to(device)
    marker = marker.to(device)
    linear = exl3.LinearEXL3(
        None, trellis.shape[0] * 16, trellis.shape[1] * 16,
        suh=suh, svh=svh, trellis=trellis,
        mcg=marker if codebook == "mcg" else None,
        mul1=marker if codebook == "mul1" else None,
    )
    if int(linear.K) != trellis.shape[-1] // 16 or bool(linear.mcg) != (codebook == "mcg") or bool(linear.mul1) != (codebook == "mul1"):
        raise _fail("LinearEXL3 did not adopt the payload's K/codebook")
    return linear


def theirs_reconstruct(exl3, trellis, suh, svh, codebook: str, marker, device):
    """Actual vendor reconstruction used for full modules and frozen windows."""
    linear = native_linear(exl3, trellis, suh, svh, codebook, marker, device)
    return (linear.get_inner_weight_tensor().detach().cpu().contiguous(),
            linear.get_weight_tensor().detach().cpu().contiguous())


def forward_input(rows: int, in_features: int):
    """Explicit CPU-generated dyadic inputs, no RNG or device-dependent seed."""
    import torch

    i = torch.arange(rows * in_features, dtype=torch.int64).reshape(rows, in_features)
    return (((i * 17 + (i // in_features) * 13) % 61 - 30).float() / 32).half().contiguous()


def dense_forward(x, weight):
    """Independent CPU fp64 matmul of [in,out] weights, rounded once to fp16."""
    return (x.double().cpu() @ weight.double().cpu()).half().contiguous()


def run_forward(linear, weight, native_weight, device) -> Dict[str, Any]:
    import torch

    weight = weight.detach().cpu().double()
    native_weight = native_weight.detach().cpu().double()
    cases = []
    for rows in FORWARD_ROWS:
        x = forward_input(rows, weight.shape[0])
        expected = dense_forward(x, weight)
        with torch.inference_mode():
            actual = linear.forward(x.to(device), {}, out_dtype=torch.float16).detach().cpu().contiguous()
        if rows <= 144 or linear.config.infer_params.no_reconstruct:
            branch = "direct"
        else:
            branch = "reconstruct-fused" if rows >= 1024 and linear._fused_reconstruct else "reconstruct-unfused"
        cases.append({
            "rows": rows, "input_shape": list(x.shape), "input_dtype": str(x.dtype),
            "dispatch_from_pinned_predicates": branch,
            "input_sha256": sha256_tensor(x),
            "reference_sha256": sha256_tensor(expected), "native_sha256": sha256_tensor(actual),
            "primary": compare(expected, actual),
            "native_weight_dense_diagnostic": compare(dense_forward(x, native_weight), actual),
        })
    return {
        "scope": "native-module-forward", "execution": "observed",
        "entrypoint": "LinearEXL3.forward(x, {}, out_dtype=torch.float16)",
        "input_formula": "fp16(((flat_index*17 + row_index*13) % 61 - 30) / 32)",
        "reference": "CPU fp64 x @ decode_payload_hf(...).T, cast once to fp16; no bias",
        "required_rows": list(FORWARD_ROWS), "observed_rows": [c["rows"] for c in cases],
        "required_dispatch": list(FORWARD_BRANCHES),
        "observed_dispatch": [c["dispatch_from_pinned_predicates"] for c in cases],
        "dispatch_basis": "pinned forward/reconstruct_hgemm predicates and live module config; not kernel instrumentation",
        "cases": cases,
    }


def summarize(records, plan=None, *, reference_verified=False) -> Dict[str, Any]:
    """Fail closed on absent, duplicate, incomplete or invalid stage observations."""
    if plan is None:
        plan = MODULE_PLAN
    def key(row):
        return tuple(row[:6])

    def record_key(record):
        return tuple(record.get(k) for k in ("label", "repo", "revision", "shard", "name", "codebook"))

    required = [key(row) for row in plan]
    observed = [record_key(record) for record in records]
    coverage_complete = (bool(required) and len(set(required)) == len(required)
                         and len(observed) == len(required) and set(observed) == set(required))

    def matched(comparison, shape):
        elements = shape[0] * shape[1]
        return (elements > 0 and comparison.get("valid") is True and comparison.get("equal") is True
                and comparison.get("finite") is True and comparison.get("same_dtype") is True
                and comparison.get("ours_dtype") == comparison.get("theirs_dtype") == "torch.float16"
                and comparison.get("ours_shape") == comparison.get("theirs_shape") == shape
                and comparison.get("elements") == comparison.get("observed_elements") == elements)

    pre_pass, weight_pass, forward_pass = [], [], []
    pre_seen, weight_seen, forward_seen = [], [], []
    for record, identity in zip(records, observed):
        shape = record.get("shape_in_out", [])
        if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
            continue
        for name, passed, seen in (("pre_hadamard", pre_pass, pre_seen),
                                   ("weight_fp16_primary", weight_pass, weight_seen)):
            comparison = record.get(name, {})
            if comparison:
                seen.append(identity)
            if matched(comparison, shape):
                passed.append(identity)
        forward = record.get("native_module_forward", {})
        cases = forward.get("cases", [])
        rows = [case.get("rows") for case in cases]
        expected_rows = list(FORWARD_ROWS)
        if (forward.get("execution") == "observed" and rows == expected_rows
                and forward.get("observed_rows") == expected_rows):
            forward_seen.append(identity)
        if (forward.get("execution") == "observed" and bool(expected_rows)
                and forward.get("scope") == "native-module-forward"
                and rows == expected_rows and forward.get("required_rows") == expected_rows
                and forward.get("observed_rows") == expected_rows
                and forward.get("required_dispatch") == list(FORWARD_BRANCHES)
                and forward.get("observed_dispatch") == list(FORWARD_BRANCHES)
                and [case.get("dispatch_from_pinned_predicates") for case in cases] == list(FORWARD_BRANCHES)
                and all(case.get("input_shape") == [case["rows"], shape[0]]
                        and case.get("input_dtype") == "torch.float16"
                        and matched(case.get("primary", {}), [case["rows"], shape[1]]) for case in cases)):
            forward_pass.append(identity)

    def stage(scope, passed, seen, prerequisite=True):
        qualified = coverage_complete and len(passed) == len(required) and prerequisite
        return {"scope": scope, "qualified": qualified,
                "status": "qualified" if qualified else "not-qualified",
                "required_modules": [list(k) for k in required],
                "observed_modules": [list(k) for k in seen],
                "passing_modules": [list(k) for k in passed]}

    pre = stage("payload-decode", pre_pass, pre_seen, reference_verified is True)
    weight = stage("complete-reconstructed-weight", weight_pass, weight_seen, pre["qualified"])
    forward = stage("native-module-forward", forward_pass, forward_seen, weight["qualified"])
    forward["required_rows_per_module"] = list(FORWARD_ROWS)
    forward["required_dispatch_per_module"] = list(FORWARD_BRANCHES)
    forward["case_coverage"] = [
        {"module": list(record_key(record)), "required_rows": list(FORWARD_ROWS),
         "observed_rows": [case.get("rows") for case in record.get("native_module_forward", {}).get("cases", [])]}
        for record in records
    ]
    return {
        "observation": "complete" if coverage_complete and all(
            len(seen) == len(required) for seen in (pre_seen, weight_seen, forward_seen)) else "incomplete",
        "qualification": "qualified" if forward["qualified"] else "not-qualified",
        "reference_implementation_verified": reference_verified is True,
        "coverage": {"complete": coverage_complete, "required_modules": [list(k) for k in required],
                     "observed_modules": [list(k) for k in observed]},
        "stages": {"payload_decode": pre, "reconstructed_weight": weight, "native_module_forward": forward,
                   "whole_model_cache": {
                       "scope": "whole-model/cache", "status": "not-assessed", "qualified": False,
                       "required_coverage": ["all model modules", "end-to-end forward", "prefill/decode cache transitions"],
                       "observed_coverage": [],
                   }},
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def module_record(cache: Path, plan_row, log) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fetch one module, decode it with our decoder; return (record, tensors)."""
    import torch

    label, repo, revision, shard, module, codebook = plan_row[:6]
    objects = dict(plan_row[6]) if len(plan_row) > 6 else {}
    header = shard_header(cache, repo, revision, shard)
    tensors, digests, fetched = {}, {}, 0
    for obj in ("trellis", "suh", "svh", codebook):
        obj_shard, obj_name = objects.get(obj, (shard, f"{module}.{obj}"))
        obj_header = header if obj_shard == shard else shard_header(cache, repo, revision, obj_shard)
        tensors[obj], digests[obj], n = fetch_tensor(cache, repo, revision, obj_shard, obj_header,
                                                     obj_name, obj=obj)
        fetched += n
    if tensors[codebook].numel() != 1:
        raise _fail(f"{module}: codebook marker must contain exactly one value")
    marker = int(tensors[codebook].reshape(-1)[0])
    if marker != xs.CODEBOOK_OBJECTS[codebook]:
        raise _fail(f"{module}: {codebook} marker {marker} != {xs.CODEBOOK_OBJECTS[codebook]}")
    trellis, suh, svh = tensors["trellis"], tensors["suh"], tensors["svh"]
    if trellis.ndim != 3 or not trellis.numel() or trellis.shape[-1] not in (32, 48, 64, 80, 96, 112, 128):
        raise _fail(f"{module}: invalid trellis shape {tuple(trellis.shape)}")
    if suh.ndim != 1 or svh.ndim != 1 or not bool(torch.isfinite(suh).all() and torch.isfinite(svh).all()):
        raise _fail(f"{module}: suh/svh must be finite one-dimensional rotations")
    bits = trellis.shape[-1] // 16
    in_features, out_features = trellis.shape[0] * 16, trellis.shape[1] * 16
    if suh.numel() != in_features or svh.numel() != out_features:
        raise _fail(f"{module}: suh/svh lengths {suh.numel()}/{svh.numel()} != {in_features}/{out_features}")
    pre = ours_pre_hadamard(trellis, codebook)
    weight = ours_weight(trellis, suh, svh, codebook)
    w_trellis, w_suh, w_svh = window_of(trellis, suh, svh)
    log(f"{label} {module} K{bits} {codebook} [{in_features},{out_features}] fetched {fetched} B")
    record = {
        "label": label, "repo": repo, "revision": revision, "shard": shard, "name": module,
        "codebook": codebook, "K": bits, "marker": marker,
        "shape_in_out": [in_features, out_features], "elements": int(in_features * out_features),
        "input_sha256": digests, "input_bytes": fetched,
        "objects": {obj: {"shard": s, "name": n} for obj, (s, n) in objects.items()},
        "ours": {"pre_hadamard_sha256": sha256_tensor(pre),
                 "weight_fp16_sha256": sha256_tensor(weight.to(torch.float16))},
        "window": {
            "k_tiles": WINDOW_TILES, "n_tiles": WINDOW_TILES,
            "trellis_shape": list(w_trellis.shape), "trellis_i16_b64": b64(w_trellis),
            "suh_f16_b64": b64(w_suh), "svh_f16_b64": b64(w_svh),
            "ours_pre_hadamard_sha256": sha256_tensor(ours_pre_hadamard(w_trellis, codebook)),
        },
    }
    return record, {"trellis": trellis, "suh": suh, "svh": svh, "marker": tensors[codebook],
                    "pre": pre, "weight": weight, "window": (w_trellis, w_suh, w_svh)}


def run_parity(exl3, record: Dict[str, Any], tensors: Dict[str, Any], device) -> None:
    import torch

    codebook = record["codebook"]
    linear = native_linear(exl3, tensors["trellis"], tensors["suh"], tensors["svh"],
                           codebook, tensors["marker"], device)
    t_pre = linear.get_inner_weight_tensor().detach().cpu().contiguous()
    t_weight = linear.get_weight_tensor().detach().cpu().contiguous()
    record["pre_hadamard"] = compare(tensors["pre"], t_pre)
    record["weight_fp16_primary"] = compare(tensors["weight"].to(torch.float16), t_weight)
    exported = native_export_reference(tensors["pre"], tensors["suh"], tensors["svh"], device)
    record["native_export_diagnostic"] = {
        "reference": "independent-unpack-lut-native-export-rounding.v1",
        "shared_backend": "torch fp32 matmul on the observed device",
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "comparison": compare(exported, t_weight),
        "reference_sha256": sha256_tensor(exported),
        "affects_historical_primary": False,
        "not_covered": ["fused reconstruction", "direct GEMM", "whole-model", "cache"],
    }
    record["weight_fp32_diagnostic"] = compare(tensors["weight"], t_weight.float())
    record["native_module_forward"] = run_forward(linear, tensors["weight"], t_weight, device)
    record["exllamav3"] = {"pre_hadamard_sha256": sha256_tensor(t_pre), "weight_fp16_sha256": sha256_tensor(t_weight)}
    w_trellis, w_suh, w_svh = tensors["window"]
    w_pre, w_weight = theirs_reconstruct(exl3, w_trellis, w_suh, w_svh, codebook, tensors["marker"], device)
    window = record["window"]
    window["exllamav3_pre_hadamard_sha256"] = sha256_tensor(w_pre)
    window["exllamav3_weight_fp16_sha256"] = sha256_tensor(w_weight)
    window["pre_hadamard"] = compare(ours_pre_hadamard(w_trellis, codebook), w_pre)
    # Hash-only native weights cannot diagnose rounding after the GPU is gone.
    # Retain both contexts: a standalone block can dispatch differently from the
    # same block inside a full reconstruction.
    window["native_weight_tensors"] = {
        label: {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "byteorder": sys.byteorder, "sha256": sha256_tensor(tensor),
                "data_b64": b64(tensor)}
        for label, tensor in (
            ("isolated_window", w_weight),
            ("full_matrix_slice", t_weight[:w_weight.shape[0], :w_weight.shape[1]]),
        )
    }
    window["weight_fp16"] = compare(ours_weight(w_trellis, w_suh, w_svh, codebook).to(torch.float16), w_weight)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="required new destination; existing files and frozen v1 are refused")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("FIDELITY_SCRATCH", tempfile.gettempdir())) / "exl3parity-cache",
                        help="fetched payload bytes and shard headers (re-runs are offline)")
    parser.add_argument("--install", action="store_true",
                        help="install and byte-verify the pinned native reference wheel; required for qualification")
    parser.add_argument("--fetch-only", action="store_true",
                        help="populate the payload cache and print our digests; no native observation receipt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.fetch_only and (args.install or args.out is not None):
        parser.error("--fetch-only cannot be combined with --install or --out")
    if not args.fetch_only and args.out is None:
        parser.error("--out is required for a native observation")
    if args.out is not None and (args.out.resolve() == FROZEN_RECEIPT.resolve()
                                 or args.out.exists() or args.out.is_symlink()):
        raise _fail("--out must be a new path, never the frozen v1 receipt or an existing destination")

    def log(message: str) -> None:
        print(f"[exl3-parity] {message}", flush=True)

    import torch

    started = time.monotonic()
    if not MODULE_PLAN:
        raise _fail("module plan is empty")
    wheel = None
    if not args.fetch_only:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available() or not torch.version.cuda:
            raise _fail("native observation requires an available CUDA device; no payloads fetched")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise _fail(f"CUDA device index {index} is unavailable; no payloads fetched")
        torch.cuda.set_device(device)
        if any(name == "exllamav3" or name.startswith("exllamav3.")
               or name == "exllamav3_ext" for name in sys.modules):
            raise _fail("native qualification requires a fresh process before importing the reference")
        wheel = install_exllamav3(log) if args.install else None
        package, exl3, imported = import_exllamav3(expect_precompiled=args.install)
        reference_verified = False
        if wheel is not None:
            verify_installed_payloads(Path(package.__file__).resolve().parent.parent,
                                      wheel["implementation_files"])
            active_extension = importlib.import_module("exllamav3.ext").exllamav3_ext
            verify_imported_payloads(
                Path(package.__file__).resolve().parent.parent, wheel["implementation_files"],
                extension_path=getattr(active_extension, "__file__", None),
                linear_path=getattr(exl3, "__file__", None))
            reference_verified = True
        if exl3.AUTO_RECONSTRUCT_THRESHOLD != 144:
            raise _fail("vendor forward threshold differs from the pinned coverage contract")
    records: List[Dict[str, Any]] = []
    tensors: List[Dict[str, Any]] = []
    for row in MODULE_PLAN:
        record, held = module_record(args.cache_dir, row, log)
        records.append(record)
        tensors.append(held)
    if args.fetch_only:
        for record in records:
            print(json.dumps({k: record[k] for k in ("label", "name", "K", "codebook", "input_sha256", "ours")}, sort_keys=True))
        log(f"fetch-only: {len(records)} modules, {sum(r['input_bytes'] for r in records)} bytes, "
            f"{time.monotonic() - started:.1f}s; cache populated, no observation receipt written")
        return 0

    for record, held in zip(records, tensors):
        run_parity(exl3, record, held, device)
        primary = record["weight_fp16_primary"]
        log(f"{record['name']}: pre_hadamard equal={record['pre_hadamard']['equal']} "
            f"full_weight equal={primary['equal']} max_abs_diff={primary['max_abs_diff']} "
            f"differing={primary['differing_elements']}/{primary['elements']}")
    for held in tensors:
        held.clear()

    receipt = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "engines/tools/exl3_decoder_parity_vs_exllamav3.py",
        "producer_sha256": xs._sha256_file(Path(__file__)),
        "exllamav3_version": EXLLAMAV3_VERSION,
        "expected_reference_commit": EXLLAMAV3_COMMIT,
        "exllamav3": {**imported, "wheel": wheel, "reconstruct_entrypoint": RECONSTRUCT_ENTRYPOINT},
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "python_version": platform.python_version(),
        "ours": {"module": "engines/tools/exl3hf_surface.py", "function": "decode_payload_hf",
                 "code_sha256": xs._sha256_file(TOOLS / "exl3hf_surface.py"),
                 "mcg_lut_sha256": xs.MCG_LUT_SHA256},
        "modules_compared": len(records),
        "codebooks": sorted({r["codebook"] for r in records}),
        "k_values": sorted({r["K"] for r in records}),
        **summarize(records, reference_verified=reference_verified),
        "modules": records,
        "note": (
            "pre_hadamard compares exllamav3_ext.reconstruct (fp16 [in,out]) with our unpack+LUT+tile "
            "layout bitwise. weight compares LinearEXL3.get_weight_tensor (four fp16 roundings) with "
            "decode_payload_hf cast to fp16 once; weight_fp32_diagnostic compares against our unrounded output. "
            "Native-module forward uses independent dense CPU fp64 accumulation rounded once to fp16; "
            "vendor-weight diagnostics cannot override any primary failure. No whole-model/cache qualification."
        ),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    write_observation(args.out, receipt)
    log(f"wrote {args.out}: modules={len(records)} observation={receipt['observation']} "
        f"qualification={receipt['qualification']} scope=native-module-forward")
    return 0 if receipt["qualification"] == "qualified" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - CLI errors are never completed observations
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
