"""Explicitly changed native CUDA runtime for ordered channelwise128 KDA.

Call load_ordered_backend() with the intended CUDA device current. It returns
(callable, metadata); the callable has the stock ordinary binding's positional
signature, mutates state/output, and returns None. Importing this module does
not import Torch, inspect GPUs, compile anything, or alter the stock wheel.

Requires CUDA-enabled PyTorch, its compatible CUDA toolkit (nvcc and headers),
a compatible C++17 compiler, Ninja, and NVIDIA compute capability >= 8.0.
Builds only the adjacent standalone CUDA translation unit in Torch's extension
cache. There is no CPU, Torch-math, stock-kernel, or alternate-hardware fallback.

The pinned kernel's SUBK=4 partial arithmetic and fast-math flags are retained;
shared atomic reductions are intentionally replaced with ascending-sub-k FP32
adds. This is not promised bit-identical to stock or portable across compiler/
GPU versions. Repeatability is an experimental question, not asserted here.
History, CUDA graph capture, noncontiguous inputs, writable aliasing, invalid
slots, and duplicate batch slots are rejected. Slots are copied synchronously
to the host for bounds/uniqueness validation, not for recurrence computation.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


_PIN = "6ff3a17ea7f3d0026b273d43239398d57f71b788"
_UPSTREAM = f"https://raw.githubusercontent.com/turboderp-org/exllamav3/{_PIN}"


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _compiler_identity(command: list[str]) -> dict:
    resolved = shutil.which(command[0])
    if resolved is None:
        raise RuntimeError(f"Required compiler is not available: {command[0]}")
    version = subprocess.run(
        [*command, "--version"], check=True, capture_output=True, text=True,
    )
    return {
        "command": command,
        "executable": str(Path(resolved).resolve()),
        "version": version.stdout.strip(),
        "version_stderr": version.stderr.strip(),
    }


def load_ordered_backend():
    """Lazily compile/load for the current CUDA device; return (callable, identity)."""
    import torch

    if torch.version.hip or torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("Ordered KDA requires CUDA-enabled Torch on an NVIDIA CUDA device")
    return _load_for_device(torch.cuda.current_device())


@lru_cache(maxsize=None)
def _load_for_device(device_index: int):
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        capability = torch.cuda.get_device_capability(device)
        if capability < (8, 0):
            raise RuntimeError("Ordered KDA requires NVIDIA compute capability >= 8.0")
        if CUDA_HOME is None:
            raise RuntimeError("Ordered KDA requires a compatible CUDA toolkit; CUDA_HOME was not found")
        ninja = shutil.which("ninja")
        if ninja is None:
            raise RuntimeError("Ordered KDA requires Ninja on PATH")
        source = Path(__file__).resolve().with_suffix(".cu")
        python_source = Path(__file__).resolve()
        for flag_variable in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
            if os.environ.get(flag_variable):
                raise RuntimeError(f"Unset {flag_variable}: ordered KDA requires controlled CUDA flags")
        # Explicit SASS architecture prevents an ambient TORCH_CUDA_ARCH_LIST
        # from producing an extension for some other visible GPU. No PTX fallback.
        arch = f"{capability[0]}{capability[1]}"
        cuda_flags = [
            "-lineinfo", "-O3", "--use_fast_math",
            "-Xcudafe", "--diag_suppress=177",
            "-Xcudafe", "--diag_suppress=20012",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
        ]
        host_command = shlex.split(os.environ.get("CXX", "c++"))
        if not host_command:
            raise RuntimeError("CXX must identify a C++ compiler")
        host_identity = _compiler_identity(host_command)
        nvcc_identity = _compiler_identity([str(Path(CUDA_HOME) / "bin" / "nvcc")])
        cuda_host = os.environ.get("CUDAHOSTCXX")
        cuda_host_identity = None
        if cuda_host:
            cuda_host_identity = _compiler_identity([cuda_host])
            cuda_flags.extend(["-ccbin", cuda_host])
        # cpp_extension itself honors CC for NVCC's host compiler when -ccbin
        # is absent. Record that environment rather than pretending CXX governs it.
        cc = os.environ.get("CC")
        cc_identity = _compiler_identity(shlex.split(cc)) if cc else None
        build_plan = {
            "runtime": "exl3_ordered_channelwise128_cuda",
            "changed_runtime": True,
            "upstream_release": "v1.4.8",
            "upstream_commit": _PIN,
            "upstream_kernel_url": f"{_UPSTREAM}/exllamav3/exllamav3_ext/gdn.cu",
            "upstream_wrapper_url": f"{_UPSTREAM}/exllamav3/modules/gated_delta_net_fn/gated_delta_rule.py",
            "upstream_build_url": f"{_UPSTREAM}/setup.py",
            "upstream_kernel": "cuda_recurrent_gated_delta_rule_kernel_128<false,V_SPLIT,true>",
            "cuda_source_sha256": _sha256(source),
            "loader_source_sha256": _sha256(python_source),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            "torch_build_configuration": torch.__config__.show(),
            "cuda_home": str(Path(CUDA_HOME).resolve()),
            "host_compiler": host_identity,
            "nvcc": nvcc_identity,
            "cuda_host_compiler_override": cuda_host_identity,
            "cc_environment_compiler": cc_identity,
            "extra_cflags": ["-Ofast"],
            "extra_cuda_cflags": cuda_flags,
            "sass_architecture": f"sm_{arch}",
            "ptx_included": False,
            "ninja_executable": str(Path(ninja).resolve()),
            "build_environment": {
                key: os.environ.get(key)
                for key in (
                    "CXX", "CC", "CUDAHOSTCXX", "TORCH_EXTENSIONS_DIR", "MAX_JOBS",
                    "TORCH_CUDA_ARCH_LIST", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
                    "CFLAGS", "CXXFLAGS", "LDFLAGS",
                )
            },
            "ordered_reduction": "four indexed FP32 partials; __fadd_rn from +0 in bt=0,1,2,3 order",
            "arithmetic": "pinned partial expressions; --use_fast_math; BF16 inputs; FP32 state; BF16 output RZ",
            "intentional_kernel_changes": [
                "memory-dot shared atomicAdd replaced by indexed partial1[bt][t] writes and ascending-bt __fadd_rn",
                "output-dot shared atomicAdd replaced by indexed partial2[bt][t] writes and ascending-bt __fadd_rn",
                "sh_red warp sums written only by bt=0 instead of four concurrent identical-value writers",
                "sh_q/sh_k/sh_g written only by bt=0 instead of four concurrent identical-value writers",
                "barrier added after memory-dot combine before consumers; end-of-sequence-iteration barrier added",
                "atomic accumulator zeroing removed; fixed-order combines instead start at positive FP32 zero",
            ],
            "causal_scope": (
                "This intervention changes atomic reduction order AND removes stock identical-value "
                "shared write/write races. It cannot by itself isolate atomics as the exclusive "
                "cause of any measured stock nondeterminism or whole-model difference."
            ),
            "limitations": [
                "channelwise g only; k_head_dim=v_head_dim=128; history=False",
                "one matched CUDA device per returned callable; compute capability >=8.0",
                "contiguous tensors with exact native dtypes; no writable aliases",
                "unique in-bounds slots; synchronous host slot validation",
                "no CUDA graph capture; no CPU/Torch arithmetic or hardware fallback",
                "changed reduction order; neither stock bit parity nor repeatability claimed without measurement",
            ],
        }
        plan_digest = hashlib.sha256(
            json.dumps(build_plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        module_name = f"exl3_ordered_kda_{plan_digest[:20]}"
        extension = load(
            name=module_name,
            sources=[str(source)],
            extra_cflags=build_plan["extra_cflags"],
            extra_cuda_cflags=cuda_flags,
            with_cuda=True,
            verbose=False,
        )
        binary = Path(extension.__file__).resolve()
        ninja_file = binary.parent / "build.ninja"
        properties = torch.cuda.get_device_properties(device)
        metadata = {
            **build_plan,
            "build_plan_sha256": plan_digest,
            "module_name": module_name,
            "module_loaded_name": extension.__name__,
            "cuda_source_path": str(source),
            "loader_source_path": str(python_source),
            "binary_path": str(binary),
            "binary_sha256": _sha256(binary),
            "build_directory": str(binary.parent),
            "build_ninja_sha256": _sha256(ninja_file),
            "build_ninja": ninja_file.read_text(),
            "device_index": device_index,
            "device_name": properties.name,
            "device_uuid": str(getattr(properties, "uuid", "unavailable")),
            "device_compute_capability": list(capability),
            "device_total_memory": properties.total_memory,
            **dict(extension.runtime_identity()),
        }

    def cuda_recurrent_gated_delta_rule(
        mixed_qkv, g, beta, recurrent_state, out,
        num_k_heads, num_v_heads, k_head_dim, v_head_dim,
        recurrent_slots, history,
    ):
        if not isinstance(mixed_qkv, torch.Tensor) or mixed_qkv.device != device:
            raise RuntimeError(f"Ordered KDA callable was built for {device}; mixed_qkv must match")
        return extension.cuda_recurrent_gated_delta_rule(
            mixed_qkv, g, beta, recurrent_state, out,
            num_k_heads, num_v_heads, k_head_dim, v_head_dim,
            recurrent_slots, history,
        )

    return cuda_recurrent_gated_delta_rule, metadata
