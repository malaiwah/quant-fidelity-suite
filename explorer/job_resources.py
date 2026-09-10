"""No-spend resource accounting shared by Jobs preparation and its tokenless worker.

Admission floors account for known payloads, not a proof of peak memory or runtime.
No model code or optional tensor library is imported on controller startup.
"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import re
import shutil

GIB = 1024**3
DEFAULT_OUTPUT_BYTES = 4 * GIB
MAX_OUTPUT_BYTES = 64 * GIB
DISK_MARGIN_BYTES = 4 * GIB
SCHEMA = "qfs.hf-job-resources.v1"


def replay_policy(hardware, selection="auto"):
    """Resolve a new plan's explicit numerical backend, never an error fallback."""
    if hardware.get("device") not in ("cpu", "cuda"):
        raise ValueError("Replay requires declared CPU or CUDA hardware.")
    if selection not in ("auto", "numpy", "cuda"):
        raise ValueError("replay_device must be auto, numpy or cuda.")
    backend = ("cuda" if hardware["device"] == "cuda" else "numpy") if selection == "auto" else selection
    if backend == "cuda" and hardware["device"] != "cuda":
        raise ValueError("CUDA replay cannot run on CPU hardware.")
    return {"device": "cuda" if backend == "cuda" else "cpu", "replay_device": backend,
            "replay_dtype": "float32", "vocab_chunk": 8192, "chunk_positions": 128}


def resolve_replay(plan):
    """Missing policy means the exact historical CPU/numpy path, only for recovery."""
    runtime = plan.get("runtime", {})
    if "replay" not in runtime:
        return replay_policy({"device": "cpu"}, "numpy")
    policy = runtime["replay"]
    if not isinstance(policy, dict):
        raise ValueError("The sealed replay policy must be an explicit object.")
    expected = replay_policy(plan["hardware"], policy.get("replay_device"))
    if (policy != expected or any(type(policy.get(key)) is not int
                                  for key in ("vocab_chunk", "chunk_positions"))):
        raise ValueError("Unsupported or inconsistent sealed replay policy.")
    return dict(expected)


def active_job_limit(value=1):
    if type(value) is not int or not 1 <= value <= 2:
        raise ValueError("max_active_jobs must be an integer in 1..2; this is not a batch scheduler.")
    return value


def cuda_replay_smoke():
    """Bounded full-vocabulary controls through the actual CUDA replay/estimator."""
    import math
    import numpy as np
    import torch
    from fidelity import dscompare
    if not torch.cuda.is_available():
        raise ValueError("Required CUDA replay smoke needs an actual CUDA device; no CPU fallback.")
    vocab = 8193  # Cross the sealed 8192-column GEMM boundary.
    hidden = np.ones((2, 1), dtype=np.float32)
    reference = np.zeros((1, vocab), dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 0] = 1.0
    left = dscompare._TorchReplay(reference, "cuda", "float32", 8192)
    right = dscompare._TorchReplay(candidate, "cuda", "float32", 8192)
    a, b = left.replay(hidden), right.replay(hidden)
    zeros, _, backend = dscompare.token_kld(a, a.clone(), "cuda")
    values, _, observed_backend = dscompare.token_kld(a, b, "cuda")
    expected = math.log1p(math.expm1(1.0) / vocab) - 1.0 / vocab
    if not np.array_equal(zeros, np.zeros(2, dtype=np.float64)):
        raise ValueError("CUDA replay zero-control failed.")
    if not np.allclose(values, expected, rtol=1e-9, atol=1e-12) or not np.all(values > 0):
        raise ValueError("CUDA replay full-vocabulary known-answer control failed.")
    if backend != "torch:k6_kld_report._token_kld" or observed_backend != backend:
        raise ValueError("CUDA smoke did not exercise the required Torch estimator.")
    refused = []
    for label, operation in (
            ("replay", lambda: left.replay(np.full((2, 1), np.nan, dtype=np.float32))),
            ("estimator", lambda: dscompare.token_kld(a, torch.full_like(b, float("inf")), "cuda"))):
        try:
            operation()
        except dscompare.Refusal as exc:
            if exc.code != "non_finite":
                raise
            refused.append(label)
        else:
            raise ValueError("CUDA %s accepted non-finite input." % label)
    torch.cuda.synchronize()
    result = {"schema": "qfs.cuda-replay-smoke.v1", "device": "cuda",
              "replay_dtype": "float32", "estimator_backend": backend,
              "vocab_size": vocab, "positions": 2, "vocab_chunk": 8192,
              "known_answer": expected, "observed": values.tolist(),
              "zero_control": zeros.tolist(), "nonfinite_refusals": refused,
              "replay_environment": left.env,
              "scope": "Bounded synthetic CUDA replay/estimator controls, not CPU/CUDA parity or model-forward correctness."}
    del left, right, a, b
    torch.cuda.empty_cache()
    return result

def output_limit(value):
    if type(value) is not int or not 0 < value <= MAX_OUTPUT_BYTES:
        raise ValueError("max_output_bytes must be an integer in 1..%d (64 GiB)." % MAX_OUTPUT_BYTES)
    return value


def byte_quantity(value, label):
    """Require units; decimal GB and binary GiB are deliberately distinct."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(B|[KMGT]i?B)\s*", str(value), re.I)
    if not match:
        raise ValueError("Missing or ambiguous advertised %s; explicit byte units are required." % label)
    unit = match[2].upper()
    power = "BKMGT".index(unit[0]) if unit != "B" else 0
    result = Decimal(match[1]) * ((1024 if "I" in unit else 1000) ** power)
    if result <= 0 or result != result.to_integral_value():
        raise ValueError("Invalid advertised %s byte quantity." % label)
    return int(result)


def hardware_capacity(hardware):
    result = {"disk_bytes": byte_quantity(hardware.get("ephemeral_storage"), "ephemeral storage"),
              "cpu_ram_bytes": byte_quantity(hardware.get("ram"), "CPU RAM"), "gpu_bytes": 0}
    if hardware["device"] == "cuda":
        accelerator = hardware.get("accelerator")
        if not isinstance(accelerator, dict) or accelerator.get("type") != "gpu":
            raise ValueError("Explicit GPU type, count and memory are required.")
        quantity = str(accelerator.get("quantity", ""))
        if not re.fullmatch(r"[1-9]\d*", quantity):
            raise ValueError("Missing or invalid advertised GPU count.")
        count = int(quantity)
        # HF's structured vram field is aggregate (e.g. 4 A10Gs: 96 GB).
        # An explicitly multiplied form instead states the per-device size.
        value = str(accelerator.get("vram", ""))
        match = re.fullmatch(r"\s*(\d+)\s*[x×]\s*(.+)", value, re.I)
        if match:
            if int(match[1]) != count:
                raise ValueError("Advertised GPU counts disagree.")
            per_device = byte_quantity(match[2], "per-device VRAM")
        else:
            total = byte_quantity(value, "aggregate VRAM")
            if total % count:
                raise ValueError("Advertised aggregate VRAM cannot be divided into equal devices.")
            per_device = total // count
        result["gpu_bytes"] = per_device
        result["gpu_count"] = count
    return result


def positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError("Verified positive integer %s is required for resource admission." % label)
    return value


def model_geometry(config, tensors):
    text = config.get("text_config") or config
    hidden = positive(text.get("hidden_size"), "text hidden_size")
    vocab = positive(text.get("vocab_size"), "text vocab_size")
    head_shapes = [row["shape"] for name, row in tensors.items() if name.endswith("lm_head.weight")]
    if head_shapes and any(shape != [vocab, hidden] for shape in head_shapes):
        raise ValueError("Checkpoint head tensor geometry differs from the actual text configuration.")
    resident = 0
    layers = {}
    for name, row in tensors.items():
        elements = 1
        for dim in row["shape"]:
            elements *= positive(dim, "tensor dimension")
        # Native BF16 materialization; packed shapes are only a floor, not a decode-fit claim.
        size = elements * 2
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if match:
            key = name[:match.end()]
            layers[key] = layers.get(key, 0) + size
        else:
            resident += size
    return {"hidden_size": hidden, "vocab_size": vocab, "head_bf16_bytes": 2 * hidden * vocab,
            "resident_bf16_floor_bytes": max(resident, 2 * hidden * vocab),
            "largest_layer_bf16_floor_bytes": max(layers.values(), default=0),
            "tensor_count": len(tensors), "basis": "Pinned config plus safetensors header shapes; packed/decode and activation overhead remain unproven."}


def plan_resources(mode, model, binding, reference, candidate, tokenizer, hardware, maximum, replay=None):
    maximum = output_limit(maximum)
    capacity = hardware_capacity(hardware)
    replay = resolve_replay({"hardware": hardware, "runtime": {"replay": replay or replay_policy(hardware)}})
    capture = replay_bytes = 0
    datasets = [item for item in (reference, candidate) if item]
    dataset_bytes = sum(positive(item.get("artifact_bytes"), "canonical dataset bytes") for item in datasets)
    checkpoint_bytes = sum(row["bytes"] for row in (model or {}).get("files", []))
    tokenizer_bytes = sum(row["bytes"] for row in (tokenizer or {}).get("files", []))
    components = {"cold_hidden_bytes": 0, "own_head_bytes": 0, "panel_bytes": 0,
                  "comparator_array_bytes": 0, "metadata_margin_bytes": 64 * 1024**2}
    gpu = 0
    cpu = 2 * GIB
    if mode == "selftest":
        # The reviewed battery builds its own small fixtures: no panel, no
        # checkpoint, no capture. Reserve a device allowance for the CUDA decode
        # rungs and the metadata margin, and nothing that pretends to be a
        # capture-fit claim.
        cpu = max(cpu, 4 * GIB)
        gpu = 2 * GIB if hardware["device"] == "cuda" else 0
    elif model:
        geometry = model["resource_geometry"]
        panel = binding["panel"]
        contexts = positive(panel["contexts"], "panel contexts")
        length = positive(panel["context_length"], "panel context length")
        positions = positive(panel["scored_positions_total"], "panel scored positions")
        hidden, vocab = geometry["hidden_size"], geometry["vocab_size"]
        if binding["tokenizer"]["vocab_size"] != vocab or not binding["tokenizer"]["files_verified"]:
            raise ValueError("Verified panel tokenizer vocabulary differs from checkpoint text vocabulary.")
        components.update(cold_hidden_bytes=2 * positions * hidden * 2,
                          own_head_bytes=2 * geometry["head_bf16_bytes"],
                          panel_bytes=3 * sum(row["bytes"] for row in binding["content"]["manifest"]) + 2 * contexts * length * 17,
                          comparator_array_bytes=positions * 8 * (2 if mode == "candidate" else 1))
        # Schedule retains every window's layer state on its capture device. Include
        # an extra state set and one full-vocab epilogue, not only a single window.
        activation = 2 * (contexts + 1) * length * hidden * 2 + length * vocab * 4
        capture = geometry["resident_bf16_floor_bytes"] + 2 * geometry["largest_layer_bf16_floor_bytes"] + activation + 2 * GIB
        # Replay concatenates vocabulary tiles into full-vocabulary logits before
        # fp64 normalization/reduction. A vocab tile is NOT its memory bound.
        replay_positions = replay["chunk_positions"] if replay["device"] == "cuda" else max(length, replay["chunk_positions"])
        replay_bytes = geometry["head_bf16_bytes"] * 8 + replay_positions * vocab * 8 * 8 + positions * 8 * 4 + 2 * GIB
        cpu = max(cpu, geometry["head_bf16_bytes"] * 8 + 2 * GIB,
                  replay_bytes if replay["device"] == "cpu" else 0,
                  capture if hardware["device"] == "cpu" else 2 * geometry["largest_layer_bf16_floor_bytes"] + 2 * GIB)
        gpu = max(capture, replay_bytes if replay["device"] == "cuda" else 0) if hardware["device"] == "cuda" else 0
    else:
        # Input captures already include heads; reserve their complete bytes
        # rather than guessing an absent model's geometry.
        positions = max(positive(item["descriptor"]["panel"]["scored_positions_total"], "dataset scored positions") for item in datasets)
        components["comparator_array_bytes"] = positions * 8
        vocab = max(positive(item["descriptor"]["capture"]["vocab_size"], "dataset vocabulary") for item in datasets)
        replay_bytes = dataset_bytes * 4 + replay["chunk_positions"] * vocab * 8 * 8 + 2 * GIB
        if replay["device"] == "cuda":
            gpu = replay_bytes
        else:
            cpu = max(cpu, replay_bytes)
    cpu = max(cpu, dataset_bytes * 4 + 2 * GIB)
    if datasets and replay["device"] == "cuda":
        # Candidate own heads can differ from the captured model's geometry.
        gpu = max(gpu, dataset_bytes * 4 + replay_bytes)
    minimum = sum(components.values())
    if maximum < minimum:
        raise ValueError("Output budget %d bytes is below the accounted payload plus metadata margin %d bytes; raise max_output_bytes." % (maximum, minimum))
    # Mounted input cache + canonical copies, scratch + durable copies + one
    # full-cap checkpoint temporary. Do not assume bucket/FUSE storage is free.
    disk = 2 * (checkpoint_bytes + tokenizer_bytes + dataset_bytes) + 3 * maximum + DISK_MARGIN_BYTES
    for label, required, available in (("ephemeral disk", disk, capacity["disk_bytes"]),
                                       ("CPU RAM", cpu, capacity["cpu_ram_bytes"]),
                                       ("single-device VRAM", gpu, capacity["gpu_bytes"])):
        if required > available:
            raise ValueError("Accounted %s requirement %d bytes exceeds advertised %d bytes. This worker cannot pool multiple GPUs." % (label, required, available))
    return {"schema": SCHEMA, "output_components": components, "minimum_output_bytes": minimum,
            "disk_required_bytes": disk, "cpu_ram_required_bytes": cpu, "gpu_required_bytes": gpu,
            "checkpoint_bytes": checkpoint_bytes, "tokenizer_source_bytes": tokenizer_bytes,
            "canonical_dataset_bytes": dataset_bytes, "advertised": capacity,
            "replay": replay, "capture_required_bytes": capture, "replay_required_bytes": replay_bytes,
            "runtime_qualified": False,
            "limitations": "Accounted planning floors with explicit margins, NOT a peak-memory bound or runtime/fit proof. Architecture-specific states, allocator/workspace/decode peaks and throughput are unqualified. Worker rechecks actual available resources before capture."}


def retrieval_limit(plan, destination):
    maximum = output_limit(plan["limits"]["max_output_bytes"])
    # Transfer temporaries and subsequent verification/publication work can coexist.
    free = shutil.disk_usage(destination).free
    if free < 2 * maximum + DISK_MARGIN_BYTES:
        raise ValueError("Local recovery disk needs two approved output copies plus 4 GiB headroom; results remain private and durable.")
    return maximum


def available_cpu_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        values[key] = value.strip()
    # Linux /proc kB means KiB (unlike advertised provider KB).
    match = re.fullmatch(r"(\d+)\s+kB", values.get("MemAvailable", ""))
    if match is None:
        raise ValueError("Actual available CPU memory is unknown.")
    available = int(match[1]) * 1024
    root = Path("/sys/fs/cgroup")
    locations = {(root, "memory.max", "memory.current"),
                 (root / "memory", "memory.limit_in_bytes", "memory.usage_in_bytes")}
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, path = line.split(":", 2)
        if controllers == "" or "memory" in controllers.split(","):
            base = root if controllers == "" else root / "memory"
            location = base / path.lstrip("/")
            if ".." not in location.parts:
                while location == base or base in location.parents:
                    locations.add((location, "memory.max" if controllers == "" else "memory.limit_in_bytes",
                                   "memory.current" if controllers == "" else "memory.usage_in_bytes"))
                    if location == base:
                        break
                    location = location.parent
    for location, limit_name, usage_name in locations:
        try:
            limit = (location / limit_name).read_text().strip()
            if limit != "max":
                used = int((location / usage_name).read_text())
                available = min(available, max(0, int(limit) - used))
        except FileNotFoundError:
            continue
    return available


def check_worker_resources(plan, scratch, durable):
    resources = plan.get("resources")
    if resources is None:
        return None  # Historical sealed v1 plans keep their original contract.
    if resources.get("schema") != SCHEMA:
        raise ValueError("Unsupported sealed resource contract.")
    policy = resolve_replay(plan)
    if "replay" in plan.get("runtime", {}) and resources.get("replay") != policy:
        raise ValueError("Resource admission does not bind the sealed replay policy.")
    required_disk = positive(resources.get("disk_required_bytes"), "worker disk requirement")
    observed = {"scratch_free_bytes": shutil.disk_usage(scratch).free,
                "durable_free_bytes": shutil.disk_usage(durable).free,
                "cpu_available_bytes": available_cpu_memory(), "gpu_free_bytes": 0}
    if observed["scratch_free_bytes"] < required_disk:
        raise ValueError("Actual worker scratch disk is below the sealed admission requirement.")
    if observed["durable_free_bytes"] < 2 * output_limit(plan["limits"]["max_output_bytes"]):
        raise ValueError("Actual durable mount disk is below approved output plus checkpoint headroom.")
    if observed["cpu_available_bytes"] < positive(resources.get("cpu_ram_required_bytes"), "worker CPU requirement"):
        raise ValueError("Actual available CPU/cgroup memory is below the sealed admission requirement.")
    if plan["hardware"]["device"] == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise ValueError("The sealed plan requires an actual CUDA device.")
        observed["gpu_free_bytes"] = int(torch.cuda.mem_get_info(0)[0])
        if observed["gpu_free_bytes"] < positive(resources.get("gpu_required_bytes"), "worker GPU requirement"):
            raise ValueError("Actual cuda:0 free VRAM is below the sealed single-device requirement.")
    return observed
