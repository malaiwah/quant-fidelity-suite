#!/usr/bin/env python3
"""Build a NEW KDA128 GLM5-Next fixture and independent complete CPU references.

Run in Python 3.12 with torch==2.11.0+cpu, transformers==5.16.1,
tokenizers==0.23.2 and safetensors==0.8.0; no FLA, causal-conv1d or hub kernels:
    python engines/tools/build_glm5_native_fixture.py --out NEW_DIRECTORY

This is not the published tiny fixture: KDA head dimensions and authored media
metadata have a new identity. Real vision weights are saved, but only the entire
text model is exercised. No source weights, remote code, or downloads are used.
The manifest is written last; a partial directory is never a completed fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

SEED = 20260907
FIXTURE_ID = "glm5-next-native-aligned128-random-bf16-v1"
PUBLISHED_REVISION = "4c31348e3beb1a8a6bd73d055464362953d768ea"
PUBLISHED_BASE = (
    "https://huggingface.co/malaiwah/glm5-next-tiny-random-bf16/resolve/"
    + PUBLISHED_REVISION + "/"
)
NATIVE_COMMIT = "6ff3a17ea7f3d0026b273d43239398d57f71b788"
TRANSFORMERS_BASE = "https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/"
VERSIONS = {
    "torch": "2.11.0+cpu", "transformers": "5.16.1",
    "tokenizers": "0.23.2", "safetensors": "0.8.0",
}
SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>", "<image>", "<video>",
    "<image_start>", "<image_end>", "<video_start>", "<video_end>",
]
KDA_LAYERS = (0, 1, 2, 4)
LAYER_TYPES = ["linear_attention"] * 3 + ["deepseek_sparse_attention", "linear_attention"]
MLP_TYPES = ["dense"] * 3 + ["sparse"] * 2

# New fixture dimensions match the native 128-wide linear/cache layout.
# Topology, expert population, streams, vocabulary and input panel are retained.
TEXT_PARAMETERS = {
    "vocab_size": 266, "hidden_size": 128, "intermediate_size": 256,
    "moe_intermediate_size": 128, "num_hidden_layers": 5,
    "num_attention_heads": 4, "num_key_value_heads": 4,
    "n_shared_experts": 1, "n_routed_experts": 8, "num_experts_per_tok": 2,
    "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
    "routed_scaling_factor": 2.5, "q_lora_rank": 128, "kv_lora_rank": 128,
    "qk_nope_head_dim": 128, "qk_rope_head_dim": 0, "v_head_dim": 128,
    "index_n_heads": 4, "index_head_dim": 128, "index_topk": 8, "index_kpool": 4,
    "index_kpool_always_select_tail": True, "indexer_types": ["full"] * 5,
    "layer_types": LAYER_TYPES, "mlp_layer_types": MLP_TYPES,
    "linear_head_dim": 128, "linear_num_heads": 4, "linear_conv_kernel_dim": 4,
    "linear_lower_bound": -5.0,
    "linear_attn_config": {
        "head_dim": 128, "num_heads": 4, "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0, "safe_gate": True,
    },
    "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
    "num_nextn_predict_layers": 0, "max_position_embeddings": 256,
    "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
    "tie_word_embeddings": False, "attention_dropout": 0.0,
    "initializer_range": 0.02, "dtype": "float32", "attention_bias": False,
    "hidden_act": "silu", "rms_norm_eps": 1e-5, "swiglu_limit": 10.0,
    "output_router_logits": False, "router_aux_loss_coef": 0.001, "use_cache": True,
}
VISION_PARAMETERS = {
    "depth": 1, "hidden_size": 32, "num_heads": 4, "intermediate_size": 64,
    "out_hidden_size": 128, "projection_intermediate_size": 128, "image_size": 16,
    "patch_size": 4, "spatial_merge_size": 2, "temporal_patch_size": 2,
    "dtype": "float32", "attention_bias": True, "attention_dropout": 0.0,
    "hidden_act": "silu", "in_channels": 3, "initializer_range": 0.02,
    "rms_norm_eps": 1e-5, "swiglu_limit": 10.0,
}
WRAPPER_PARAMETERS = {
    "tie_word_embeddings": False, "image_token_id": 4, "video_token_id": 5,
    "image_start_token_id": 6, "image_end_token_id": 7,
    "video_start_token_id": 8, "video_end_token_id": 9,
    "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "dtype": "float32",
}
# Authored metadata, NOT copied/inferred from the processor-less published artifact.
# Four tokens correspond to this fixture's 16x16 canvas / (4*2)^2.
PROCESSOR_PARAMETERS = {
    "processor_class": "Glm5NextProcessor",
    "image_processor": {
        "image_processor_type": "Glm5NextImageProcessor",
        "patch_size": 4, "temporal_patch_size": 2, "merge_size": 2,
        "patch_expand_factor": 1, "min_image_tokens": 4, "max_image_tokens": 4,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "do_resize": True, "resample": 3, "size": {"longest_edge": 1},
        "default_to_square": False, "do_rescale": True,
        "rescale_factor": 1 / 255, "do_normalize": True, "do_convert_rgb": True,
    },
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def raw_bytes(tensor) -> bytes:
    import torch
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_record(name: str, tensor) -> dict:
    import torch
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError("Nonfinite tensor: " + name)
    raw = raw_bytes(tensor)
    return {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "elements": tensor.numel(), "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def clean_load(path: Path):
    import torch
    from transformers import Glm5NextForConditionalGeneration
    model, info = Glm5NextForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False,
        attn_implementation="eager", output_loading_info=True,
    )
    required = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    if not required.issubset(info) or any(info[key] for key in required):
        raise ValueError("Incomplete or unclean Transformers loading: " + repr(info))
    if model.config.text_config._attn_implementation != "eager":
        raise ValueError("Independent reference requires eager attention")
    return model.cpu().eval(), {
        key: sorted(value) if isinstance(value, set) else value for key, value in info.items()
    }


def check_geometry(model) -> list[dict]:
    import torch
    text = model.config.text_config
    if (text.num_hidden_layers, text.hidden_size, text.vocab_size,
        text.max_position_embeddings, text.hc_mult, text.num_nextn_predict_layers,
        text.n_routed_experts, text.num_experts_per_tok) != (5, 128, 266, 256, 4, 0, 8, 2):
        raise ValueError("Incomplete or changed text configuration")
    if text.layer_types != LAYER_TYPES or text.mlp_layer_types != MLP_TYPES:
        raise ValueError("Changed block schedule")
    if len(model.model.language_model.layers) != 5:
        raise ValueError("All five text blocks are required")
    if model.get_output_embeddings().weight.shape != (266, 128):
        raise ValueError("Full vocabulary head required")
    if model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr():
        raise ValueError("Untied head required")
    geometry = []
    for index, layer in enumerate(model.model.language_model.layers):
        attention = layer.self_attn
        item = {"layer": index, "attention_class": type(attention).__name__,
                "mlp_class": type(layer.mlp).__name__}
        if layer.attn_hc.hc_mult != 4 or layer.ffn_hc.hc_mult != 4:
            raise ValueError("Both four-stream residual connections are required")
        if index in KDA_LAYERS:
            dimensions = (attention.num_heads, attention.head_dim, attention.conv_kernel_size)
            if dimensions != (4, 128, 4) or tuple(attention.conv1d.weight.shape) != (1536, 1, 4):
                raise ValueError("KDA128 geometry was not retained")
            item["kda_heads_head_dim_conv"] = list(dimensions)
        else:
            dimensions = (attention.q_lora_rank, attention.kv_lora_rank,
                          attention.qk_nope_head_dim, attention.qk_rope_head_dim, attention.v_head_dim)
            if dimensions != (128, 128, 128, 0, 128):
                raise ValueError("Authored native-aligned MLA dimensions required")
            item["mla_q_kv_nope_rope_v"] = list(dimensions)
        if index >= 3:
            experts = layer.mlp.experts
            if (tuple(experts.gate_up_proj.shape), tuple(experts.down_proj.shape),
                layer.mlp.gate.top_k) != ((8, 256, 128), (8, 128, 128), 2):
                raise ValueError("Complete eight-expert/top2 MoE required")
        geometry.append(item)
    state = model.state_dict()
    if not any(name.startswith("model.visual.") for name in state):
        raise ValueError("Complete initialized vision tower missing")
    if any("shared_head" in name or ".layers.5." in name or ".layers.45." in name for name in state):
        raise ValueError("Unexpected orphan MTP tensors")
    for name, tensor in state.items():
        tensor_record(name, tensor)
        if tensor.device.type != "cpu":
            raise ValueError("Reference tensors must remain on CPU")
    return geometry


def compare(left, right) -> dict:
    import torch
    if left.shape != right.shape:
        raise ValueError("Comparison shape mismatch")
    difference = (left.double() - right.double()).abs()
    log_p = torch.log_softmax(left.double(), dim=-1)
    log_q = torch.log_softmax(right.double(), dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    return {"byte_equal": left.dtype == right.dtype and raw_bytes(left) == raw_bytes(right),
            "unequal_elements": int((left != right).sum()),
            "max_abs_difference": float(difference.max()),
            "kl_reference_to_candidate_fp64_mean": float(kl.mean()),
            "kl_reference_to_candidate_fp64_max": float(kl.max())}


def check_cache(cache, position: int) -> dict:
    import torch
    if cache is None or len(cache.layers) != 5 or cache.get_seq_length(3) != position:
        raise ValueError(f"Incomplete reference cache at position {position}")
    for index in KDA_LAYERS:
        if not cache.has_previous_state(index):
            raise ValueError(f"Missing recurrent state for layer {index}")
        layer = cache.layers[index]
        conv, recurrent = layer.conv_states[0], layer.recurrent_states[0]
        if conv is None or recurrent is None or tuple(recurrent.shape) != (1, 4, 128, 128):
            raise ValueError(f"Invalid KDA state geometry at layer {index}")
        if not bool(torch.isfinite(conv).all()) or not bool(torch.isfinite(recurrent).all()):
            raise ValueError(f"Nonfinite KDA state at layer {index}")
    return {"position": position, "cache_class": type(cache).__name__,
            "layer_classes": [type(layer).__name__ for layer in cache.layers]}


def reference_pass(model, inputs: dict) -> tuple[dict, dict]:
    import torch
    tensors, metadata = {}, {}
    counts = {f"layer_{index}": 0 for index in range(5)}
    counts["full_head"] = 0
    handles = []

    def counter(name):
        def hook(module, args, result):
            counts[name] += 1
        return hook

    for index, layer in enumerate(model.model.language_model.layers):
        handles.append(layer.register_forward_hook(counter(f"layer_{index}")))
    handles.append(model.get_output_embeddings().register_forward_hook(counter("full_head")))

    def logits(output, length):
        tensor = output.logits.detach().cpu().contiguous().clone()
        if tensor.shape != (1, length, 266):
            raise ValueError("Incomplete scored token/vocabulary coverage")
        tensor_record("reference logits", tensor)
        return tensor

    try:
        with torch.inference_mode():
            for label, tokens in inputs.items():
                ids = torch.tensor(tokens, dtype=torch.long, device="cpu")
                for name in counts:
                    counts[name] = 0
                output = model(input_ids=ids, use_cache=False, return_dict=True, logits_to_keep=0)
                if output.past_key_values is not None:
                    raise ValueError("Uncached call unexpectedly returned state")
                tensors[label + "_uncached"] = logits(output, 64)
                if any(count != 1 for count in counts.values()):
                    raise ValueError("Uncached full-model coverage hole")
                uncached_counts = counts.copy()
                for name in counts:
                    counts[name] = 0
                output = model(input_ids=ids[:, :32], use_cache=True, return_dict=True, logits_to_keep=0)
                cache = output.past_key_values
                positions = [check_cache(cache, 32)]
                chunks = [logits(output, 32)]
                for position in range(32, 64):
                    output = model(input_ids=ids[:, position:position + 1], past_key_values=cache,
                                   use_cache=True, return_dict=True, logits_to_keep=0)
                    if output.past_key_values is not cache:
                        raise ValueError("Reference cache identity changed during carry")
                    positions.append(check_cache(cache, position + 1))
                    chunks.append(logits(output, 1))
                tensors[label + "_cached"] = torch.cat(chunks, dim=1).contiguous()
                if any(count != 33 for count in counts.values()):
                    raise ValueError("Cached full-model coverage hole")
                metadata[label] = {
                    "uncached_forward_counts": uncached_counts,
                    "cached_forward_counts": counts.copy(), "cache_positions": positions,
                    "uncached_vs_cached": compare(tensors[label + "_uncached"], tensors[label + "_cached"]),
                }
                del output, cache, chunks
    finally:
        for handle in handles:
            handle.remove()
    return tensors, metadata


def build(destination: Path) -> dict:
    # Keep model/reference imports lazy, CPU-only and offline, including hub kernels.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    import transformers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import (
        Glm5NextConfig, Glm5NextTextConfig, Glm5NextVisionConfig,
        Glm5NextForConditionalGeneration, PreTrainedTokenizerFast,
    )

    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Refusing existing destination: {destination}")
    observed = {name: importlib.metadata.version(name) for name in VERSIONS}
    if observed != VERSIONS or platform.python_version_tuple()[:2] != ("3", "12"):
        raise ValueError(f"Require Python3.12 and {VERSIONS}; observed {platform.python_version()}, {observed}")
    forbidden = [name for name in ("fla", "causal_conv1d", "kernels") if importlib.util.find_spec(name)]
    if torch.version.cuda is not None or forbidden:
        raise ValueError(f"Use isolated CPU Torch without FLA/causal-conv1d/hub kernels: {forbidden}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    destination.mkdir(parents=True, exist_ok=False)

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    if len(alphabet) != 256:
        raise ValueError("Byte alphabet must have 256 entries")
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS + alphabet)}
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>", pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<bos>", 1), ("<eos>", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
        unk_token="<unk>", additional_special_tokens=SPECIAL_TOKENS[4:],
        model_max_length=256, clean_up_tokenization_spaces=False,
    )
    if len(tokenizer) != 266:
        raise ValueError("Vocabulary must have exactly 266 entries")
    tokenizer.save_pretrained(destination)
    text = Glm5NextTextConfig(**TEXT_PARAMETERS)
    vision = Glm5NextVisionConfig(**VISION_PARAMETERS)
    config = Glm5NextConfig(text_config=text, vision_config=vision, **WRAPPER_PARAMETERS)
    if (text.linear_head_dim, text.linear_num_heads, text.linear_conv_kernel_dim,
        text.linear_lower_bound) != (128, 4, 4, -5.0):
        raise ValueError("Nested KDA configuration changed geometry")
    initialized = Glm5NextForConditionalGeneration(config).cpu().eval()
    check_geometry(initialized)
    original = initialized.state_dict()
    # Do not .bfloat16() the model: native strict FP32-retained values must not round.
    # save_pretrained supplies real checkpoint conversion mappings (KDA/experts/mHC).
    with tempfile.TemporaryDirectory(prefix="glm5-kda128-fp32-") as temporary:
        initialized.save_pretrained(temporary, safe_serialization=True, max_shard_size="10MB")
        model, first_info = clean_load(Path(temporary))
    loaded = model.state_dict()
    if original.keys() != loaded.keys():
        raise ValueError("State keys changed during native BF16 conversion")
    for name, tensor in original.items():
        if not torch.equal(tensor.to(loaded[name].dtype), loaded[name]):
            raise ValueError("Initialized value changed during conversion: " + name)
    del initialized, original
    retained = [name for name, tensor in loaded.items() if tensor.dtype == torch.float32]
    required_retained = {
        f"model.language_model.layers.{index}.self_attn.{suffix}"
        for index in KDA_LAYERS
        for suffix in ("conv1d.weight", "forget_gate.dt_bias", "forget_gate.A_log")
    } | {f"model.language_model.layers.{index}.mlp.gate.e_score_correction_bias" for index in (3, 4)}
    if not required_retained.issubset(retained):
        raise ValueError("Required FP32 retention lost: " + repr(required_retained - set(retained)))
    model.config.dtype = torch.bfloat16
    model.config.text_config.dtype = torch.bfloat16
    model.config.vision_config.dtype = torch.bfloat16
    model.save_pretrained(destination, safe_serialization=True, max_shard_size="10MB")
    reloaded, reload_info = clean_load(destination)
    final = reloaded.state_dict()
    if loaded.keys() != final.keys():
        raise ValueError("State keys changed during final serialization")
    for name, tensor in final.items():
        if tensor.dtype != loaded[name].dtype or raw_bytes(tensor) != raw_bytes(loaded[name]):
            raise ValueError("Save/reload changed tensor bytes: " + name)
    geometry = check_geometry(reloaded)
    del model, loaded, final

    restored_tokenizer = transformers.AutoTokenizer.from_pretrained(
        destination, local_files_only=True, trust_remote_code=False,
    )
    sample = "Byte coverage: café 日本語\n\t\\ punctuation!"
    if restored_tokenizer.decode(restored_tokenizer.encode(sample, add_special_tokens=False)) != sample:
        raise ValueError("Tokenizer roundtrip failed")
    write_json(destination / "processor_config.json", PROCESSOR_PARAMETERS)
    inputs = {"a": [[1] + list(range(10, 73))], "b": [[1] + list(reversed(range(10, 73)))]}
    write_json(destination / "inputs.json", inputs)
    reference, pass_metadata = reference_pass(reloaded, inputs)
    repeat, repeat_metadata = reference_pass(reloaded, inputs)
    save_file(dict(sorted(reference.items())), destination / "reference.safetensors")
    save_file(dict(sorted(repeat.items())), destination / "reference-repeat.safetensors")
    runtime = {
        "versions": observed, "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "torch_cuda": torch.version.cuda,
        "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "attention": "eager", "device": "cpu", "inference_mode": True,
        "optional_kernel_packages": forbidden,
        "installed_distributions": dict(sorted(
            (distribution.metadata["Name"], distribution.version)
            for distribution in importlib.metadata.distributions() if distribution.metadata["Name"]
        )),
    }
    reference_metadata = {
        "schema": "qfs.glm5-native-reference.v1", "fixture_id": FIXTURE_ID,
        "runtime": runtime, "scope": "complete text model, five blocks and untied full-vocabulary head",
        "uncached_schedule": [64], "cached_schedule": [32] + [1] * 32,
        "cache_policy": "Fresh DynamicCache per input/pass; carry actual returned cache through decode",
        "scored_shape": [1, 64, 266], "passes": {"first": pass_metadata, "repeat": repeat_metadata},
        "repeat_comparisons": {name: compare(reference[name], repeat[name]) for name in sorted(reference)},
        "tensors": [tensor_record(name, reference[name]) for name in sorted(reference)],
        "repeat_tensors": [tensor_record(name, repeat[name]) for name in sorted(repeat)],
        "reference_equivalence_to_native": False, "native_execution_observed": False,
    }
    write_json(destination / "reference-metadata.json", reference_metadata)
    checkpoint_tensors = []
    checkpoints = sorted(destination.glob("model*.safetensors"))
    if not checkpoints:
        raise ValueError("No saved model checkpoint")
    seen = set()
    for checkpoint in checkpoints:
        with safe_open(checkpoint, framework="pt", device="cpu") as saved:
            for name in sorted(saved.keys()):
                if name in seen:
                    raise ValueError("Duplicate checkpoint tensor: " + name)
                seen.add(name)
                record = tensor_record(name, saved.get_tensor(name))
                record["file"] = checkpoint.name
                checkpoint_tensors.append(record)
    for index in KDA_LAYERS:
        prefix = f"model.language_model.layers.{index}.self_attn."
        required = {prefix + suffix for suffix in (
            "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "A_log", "dt_bias",
        )}
        if not required.issubset(seen):
            raise ValueError("Actual KDA checkpoint conversion incomplete: " + repr(required - seen))
    for index in (3, 4):
        for expert in range(8):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.language_model.layers.{index}.mlp.experts.{expert}.{projection}.weight"
                if name not in seen:
                    raise ValueError("Actual expert checkpoint conversion incomplete: " + name)
    shutil.copyfile(__file__, destination / "build_glm5_native_fixture.py")
    installed_sources = {}
    for module_name in (
        "transformers.models.glm5_next.modeling_glm5_next",
        "transformers.models.glm5_next.configuration_glm5_next", "transformers.cache_utils",
    ):
        module = importlib.import_module(module_name)
        source_path = Path(inspect.getfile(module))
        installed_sources[module_name] = {"sha256": sha256(source_path), "size_bytes": source_path.stat().st_size}
    files = {}
    for path in sorted(destination.iterdir()):
        if not path.is_file() or path.is_symlink():
            raise ValueError("Unexpected non-file fixture content: " + str(path))
        files[path.name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema": "qfs.glm5-native-fixture.v1", "fixture_id": FIXTURE_ID,
        "seed": SEED, "files": files, "generator_sha256": sha256(Path(__file__)),
        "runtime": runtime, "installed_source_files": installed_sources,
        "authored_config": {"text": TEXT_PARAMETERS, "vision": VISION_PARAMETERS,
                            "wrapper": WRAPPER_PARAMETERS, "processor": PROCESSOR_PARAMETERS},
        "serialized_config": json.loads((destination / "config.json").read_text(encoding="utf-8")),
        "source_basis": {
            "published_builder": PUBLISHED_BASE + "build_fixture.py",
            "published_builder_sha256": "2958ca3099779a8d81911f2f9d2fe1dba31609880e21e1bad2d501df6938364c",
            "published_config": PUBLISHED_BASE + "config.json",
            "native_commit": NATIVE_COMMIT,
            "native_kernel_constraint": f"https://github.com/turboderp-org/exllamav3/blob/{NATIVE_COMMIT}/exllamav3/exllamav3_ext/gdn.cu",
            "transformers_model": TRANSFORMERS_BASE + "models/glm5_next/modeling_glm5_next.py",
            "transformers_config": TRANSFORMERS_BASE + "models/glm5_next/configuration_glm5_next.py",
            "processor_schema": TRANSFORMERS_BASE + "models/glm5_next/image_processing_glm5_next.py",
            "native_processor_reader": f"https://github.com/turboderp-org/exllamav3/blob/{NATIVE_COMMIT}/exllamav3/architecture/glm4v.py",
        },
        "changes_from_published": [
            {"path": "text_config.linear_head_dim", "published": 16, "authored": 128,
             "reason": "Stock native channelwise recurrent KDA requires key/value head_dim128"},
            {"path": "text_config.linear_attn_config.head_dim", "published": 16, "authored": 128,
             "reason": "Keep nested and normalized KDA dimensions consistent"},
            {"path": "text_config.internal_linear_dimensions",
             "published": {"hidden_size": 64, "intermediate_size": 128, "moe_intermediate_size": 32,
                           "q_lora_rank": 32, "kv_lora_rank": 16, "qk_nope_head_dim": 16,
                           "v_head_dim": 16, "index_head_dim": 16},
             "authored": {key: TEXT_PARAMETERS[key] for key in ("hidden_size", "intermediate_size",
                          "moe_intermediate_size", "q_lora_rank", "kv_lora_rank",
                          "qk_nope_head_dim", "v_head_dim", "index_head_dim")},
             "reason": "Separate native-compatible fixture; prevent padded internal linear widths from changing adjacent operator geometry"},
            {"path": "vision_config.output_projection", "published": [64, 64],
             "authored": [128, 128], "reason": "Match the new text residual width; vision remains unqualified"},
            {"path": "processor_config.json", "published": "absent", "authored": PROCESSOR_PARAMETERS,
             "reason": "Explicit NEW text fixture media metadata required by stock native config reader; not published provenance"},
        ],
        "processor_provenance": {
            "kind": "authored_new_fixture_parameters", "api_executed": False,
            "geometry_basis": "Retained vision patch4/temporal2/merge2/image16; authored fixed four-token image budget",
            "normalization_basis": "Glm5NextImageProcessor OpenAI CLIP defaults in Transformers5.16.1",
            "vision_or_processor_execution_qualified": False,
        },
        "initialization": "Complete native FP32 random initialization; save_pretrained and native BF16 reload with strict FP32 retention; no source weights or repairs",
        "native_class": type(reloaded).__name__, "native_module_geometry": geometry,
        "parameter_count": sum(parameter.numel() for parameter in reloaded.parameters()),
        "tensor_count": len(checkpoint_tensors), "checkpoint_tensors": checkpoint_tensors,
        "fp32_retained_in_memory": retained,
        "fp32_saved_tensors": [record["name"] for record in checkpoint_tensors if record["dtype"] == "torch.float32"],
        "conversion_note": "In-memory conv1d/forget_gate/fused experts and mHC names differ from disk; disk inventory is authoritative and comes from save_pretrained",
        "loading_info": {"fp32_to_bf16": first_info, "final_reload": reload_info},
        "save_reload_byte_equal": True, "untied_head": True,
        "tokenizer": {"kind": "independent byte BPE without merges", "vocab_size": len(tokenizer),
                      "special_token_ids": {token: tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS}},
        "execution_complete": True, "execution_scope": "Transformers CPU text reference only",
        "native_execution_observed": False, "published_fixture_qualification": False,
        "limitations": ["NEW random fixture, not published tiny weights or geometry", "No trained language or vision quality claim",
                        "CPU repeat observations are not native GPU determinism or reference equivalence",
                        "Native loader, all kernels, and native/reference metrics require separate execution"],
        "hash_coverage": "Every artifact file except this manifest; hash this manifest externally to avoid recursive self-hashing",
    }
    write_json(destination / "fixture-manifest.json", manifest)
    return {"fixture_id": FIXTURE_ID, "destination": str(destination),
            "manifest_sha256": sha256(destination / "fixture-manifest.json"),
            "checkpoint_tensor_count": len(checkpoint_tensors),
            "reference_keys": sorted(reference), "cpu_repeat": reference_metadata["repeat_comparisons"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="New nonexistent fixture directory")
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.out), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
