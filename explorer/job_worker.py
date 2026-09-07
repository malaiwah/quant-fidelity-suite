#!/usr/bin/env python3
"""Tokenless, fail-closed HF Jobs capture/compare worker; no publication authority."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/malaiwah/quant-fidelity-suite"
PLAN_PATH = Path("/inputs/plan/plan.json")
OUT_PATH = Path("/outputs/result")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
LOG_LIMIT = 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def seal(value, field):
    value[field] = ""
    value[field] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_json(path, maximum=16 * 1024 * 1024):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    with Path(path).open("rb") as source:
        raw = source.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("JSON exceeds bound: " + str(path))
    value = json.loads(raw, object_pairs_hook=unique)
    canonical(value)
    return value


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        output.write(canonical(value) + b"\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(p in (".", "..") for p in path.parts):
        raise ValueError("unsafe relative path: " + value)
    return path


def regular(path):
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode) or path.resolve() != path.absolute():
        raise ValueError("expected non-symlink regular file: " + str(path))
    return path


def row(path, root):
    regular(path)
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}


def tree(root):
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            if (Path(directory) / name).is_symlink():
                raise ValueError("symlink directory is not an evidence artifact")
        for name in sorted(files):
            yield regular(Path(directory) / name)


def require_no_credentials():
    forbidden = {"HF_TOKEN", "HF_TOKEN_PATH", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN", "OAUTH_TOKEN"}
    if any(os.environ.get(key) for key in forbidden):
        raise ValueError("worker must not receive HF credentials")
    for path in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
        if path.exists():
            raise ValueError("ambient HF credential file is forbidden")


def validate_plan(plan, out):
    if plan.get("schema") != "qfs.hf-workflow-plan.v1":
        raise ValueError("unsupported plan schema")
    expected = plan.get("plan_sha256")
    if not HEX64.fullmatch(str(expected)) or seal(dict(plan), "plan_sha256")["plan_sha256"] != expected:
        raise ValueError("plan self-seal mismatch")
    if not re.fullmatch(r"[0-9a-f]{32}", str(plan.get("workflow_id"))):
        raise ValueError("invalid workflow identity")
    owner = plan.get("owner")
    if not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,95}", owner):
        raise ValueError("invalid personal account identity")
    source = plan["source"]
    if source.get("repository") != SOURCE or not HEX40.fullmatch(str(source.get("revision"))):
        raise ValueError("source must be an immutable commit of the fixed QFS repository")
    if source.get("worker_sha256") != digest(Path(__file__).resolve()):
        raise ValueError("worker source SHA-256 mismatch")
    output = plan["output"]
    if (out != OUT_PATH or out.parent.resolve() != Path("/outputs") or output.get("mount_path") != "/outputs"
            or not REPOSITORY.fullmatch(str(output.get("dataset_repository")))
            or output["dataset_repository"].split("/")[0] != owner
            or not REPOSITORY.fullmatch(str(output.get("bucket")))
            or output["bucket"].split("/")[0] != owner
            or output.get("prefix") != "runs/" + plan["workflow_id"]):
        raise ValueError("account/output identity or fixed output path mismatch")
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", str(plan.get("image"))):
        raise ValueError("immutable image digest required")
    hardware = plan["hardware"]
    timeout = hardware.get("timeout_seconds")
    if type(timeout) is not int or not 0 < timeout <= 86400:
        raise ValueError("timeout must be an integer in 1..86400 seconds")
    if hardware.get("device") not in ("cpu", "cuda"):
        raise ValueError("unsupported device")
    for key in ("hourly_usd", "max_compute_usd"):
        value = float(hardware[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("finite positive price and consent ceiling required")
    if type(plan["limits"].get("max_output_bytes")) is not int or plan["limits"]["max_output_bytes"] <= 0:
        raise ValueError("positive output bound required")
    runtime = plan["runtime"]
    if runtime.get("dtype") != "bfloat16" or runtime.get("schedule") != "layer-outer":
        raise ValueError("only the declared BF16 layer-outer runtime is admitted")
    mode = plan.get("mode")
    required = {"root": {"model", "panel"}, "candidate": {"model", "panel", "reference"}, "compare": {"reference", "candidate"}}.get(mode)
    if required is None or set(plan["inputs"]) != {"model", "panel", "reference", "candidate", "tokenizer"}:
        raise ValueError("invalid action/input contract")
    if mode == "candidate":
        required = required | {"tokenizer"}
    for name, value in plan["inputs"].items():
        if name not in required:
            if value is not None:
                raise ValueError("extraneous input: " + name)
            continue
        if (not isinstance(value, dict) or value.get("mount_path") != "/inputs/" + name
                or not REPOSITORY.fullmatch(str(value.get("repository")))
                or not HEX40.fullmatch(str(value.get("revision")))):
            raise ValueError("invalid immutable input descriptor: " + name)
        mount = Path(value["mount_path"])
        if not mount.is_dir() or mount.resolve() != mount or not os.statvfs(mount).f_flag & os.ST_RDONLY:
            raise ValueError("input must be a read-only, non-symlink mount: " + name)
        if name in ("reference", "candidate") and not HEX64.fullmatch(str(value.get("dataset_sha256"))):
            raise ValueError("immutable dataset identity missing")
    if mode == "candidate":
        if not isinstance(plan.get("scope"), dict) or not plan["scope"] or not isinstance(plan.get("codec"), str) or not plan["codec"]:
            raise ValueError("candidate requires actual scope and codec")
    elif plan.get("scope") is not None or plan.get("codec") is not None:
        raise ValueError("scope/codec are only valid for candidate capture")
    bits = plan.get("declared_bits")
    if bits is not None and (isinstance(bits, bool) or not isinstance(bits, (int, float)) or not math.isfinite(bits) or not 0 < bits <= 64):
        raise ValueError("invalid nominal bits")
    if mode != "compare":
        panel = plan["inputs"]["panel"]
        relative(panel["path"])
        if panel.get("role") != "final":
            raise ValueError("only an exact final token panel is admitted")


def source_manifest(plan):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip()
    if revision != plan["source"]["revision"]:
        raise ValueError("checkout revision differs from the source plan")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL, timeout=30)
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, timeout=30).decode().split("\0")
    selected = sorted(p for p in paths if p and (p.endswith(".py") or p == "bin/BUNDLE.txt" or p.startswith("explorer/requirements-worker") or p == "engines/coverage.json" or p.startswith("engines/tools/layer-outer-evidence/") and "unexpected-keys.json" in p))
    if "explorer/job_worker.py" not in selected or "bin/BUNDLE.txt" not in selected:
        raise ValueError("worker and bundle contract must be tracked in the immutable source")
    return {"schema": "qfs.hf-workflow-source.v1", "repository": SOURCE, "revision": revision, "source_files": [row(ROOT / p, ROOT) for p in selected]}


def model_binding(plan, out):
    sys.path.insert(0, str(ROOT / "engines/tools"))
    import quant_stream
    model = plan["inputs"]["model"]
    mount = Path(model["mount_path"])
    config_path = regular(mount / "config.json")
    config = load_json(config_path)
    if digest(config_path) != model["config_sha256"] or config != model["config"]:
        raise ValueError("mounted model configuration differs from metadata admission")
    files = model["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("complete model file census required")
    names = set()
    for item in files:
        relative(item["path"])
        if item["path"] in names:
            raise ValueError("duplicate model census path")
        names.add(item["path"])
        path = regular(mount / item["path"])
        if type(item["bytes"]) is not int or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError("model file census mismatch: " + item["path"])
    weight_names = {p.relative_to(mount).as_posix() for p in tree(mount) if p.suffix in (".safetensors", ".bin", ".pt", ".pth", ".gguf")}
    supported_suffixes = (".safetensors", ".gguf") if plan["mode"] == "candidate" else (".safetensors",)
    if not weight_names or not weight_names <= names or any(not name.endswith(supported_suffixes) for name in weight_names):
        raise ValueError("only completely inventoried native safetensors or supported candidate storage is admitted")
    if any(name.endswith(".gguf") for name in weight_names) and any(name.endswith(".safetensors") for name in weight_names):
        raise ValueError("mixed checkpoint representations are not a single measured artifact")
    if sum((mount / name).stat().st_size for name in weight_names) != model["weight_bytes"]:
        raise ValueError("model total weight bytes mismatch")
    index_path = mount / "model.safetensors.index.json"
    if index_path.exists():
        regular(index_path)
        if digest(index_path) != model["index_sha256"] or index_path.stat().st_size != model["index_bytes"]:
            raise ValueError("model index identity mismatch")
        shards = set(load_json(index_path)["weight_map"].values())
        if shards != weight_names:
            raise ValueError("index shard set differs from mounted weights")
    elif model.get("index_sha256") is not None or model.get("index_bytes") not in (None, 0):
        raise ValueError("declared model index is missing")
    quant = quant_stream.quantization(config)
    method = str(quant.get("quant_method", "")).lower()
    if quant and quant_stream.reader_for(config) is None and method not in ("fp8", "exl3") and not (method == "modelopt" and quant.get("quant_algo") == "NVFP4") and (config.get("hybrid_tr3_tail") or {}).get("format") != "exl3-trellis":
        raise ValueError("quantization declaration has no supported streaming reader")
    if plan["mode"] == "root" and (quant or config.get("hybrid_tr3_tail")):
        raise ValueError("a quantized checkpoint cannot be declared a native root")
    catalog = load_json(ROOT / "engines/coverage.json")
    code = plan["runtime"].get("trusted_code")
    if code:
        matches = [entry for entry in catalog["architectures"] if config.get("model_type") in entry["model_types"] and entry.get("runtime_mode") == "pinned_code"]
        admitted = False
        for entry in matches:
            tested = entry.get("tested_runtime", {})
            admitted |= code == entry.get("code_pin")
            repository = tested.get("code_repository", entry["fixture"]["repository"])
            revision = tested.get("code_revision", tested.get("model_and_code_revision"))
            admitted |= code == {"repository": repository, "revision": revision}
        if not admitted:
            raise ValueError("custom code is not the exact vetted architecture catalog pin")
    allowlist = plan["runtime"].get("unexpected_allowlist")
    if allowlist:
        name = str(relative(allowlist["path"]))
        if name not in ("engines/tools/layer-outer-evidence/fruit-unexpected-keys.json", "engines/tools/layer-outer-evidence/fruit-fp8-unexpected-keys.json"):
            raise ValueError("unexpected tensor inventory is not a vetted Fruit artifact")
        path = regular(ROOT / name)
        provenance = load_json(str(path) + ".provenance.json")
        names_hash = hashlib.sha256(canonical(sorted(load_json(path)))).hexdigest()
        if (digest(path) != allowlist["artifact_sha256"] or names_hash != allowlist["canonical_sorted_names_sha256"]
                or provenance["config_sha256"] != model["config_sha256"] or provenance["index_sha256"] != model["index_sha256"]):
            raise ValueError("Fruit inventory/config/index binding mismatch")
        shutil.copyfile(path, out / "unexpected-tensors.json")
        shutil.copyfile(str(path) + ".provenance.json", out / "unexpected-tensors.provenance.json")
    return mount


def safe_log(raw):
    text = raw.decode("utf-8", "replace")
    text = re.sub(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|$))", "", text)
    text = re.sub(r"hf_[A-Za-z0-9]{10,}|(?i:bearer)\s+\S+", "[REDACTED]", text)
    return "".join(c for c in text if c in "\n\t" or ord(c) >= 32 and ord(c) != 127).encode("utf-8")


class Runner:
    def __init__(self, plan, out, deadline):
        self.plan, self.out, self.deadline = plan, out, deadline
        self.commands = []
        self.maximum = plan["limits"]["max_output_bytes"]
        self.environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HOME="/tmp/qfs-worker-hf", PYTHONDONTWRITEBYTECODE="1", STACKPRINT_IMAGE_PIN=plan["image"], FIDELITY_IMAGE_REFERENCE=plan["image"])
        self.environment.pop("PYTHONPATH", None)
        self.environment.pop("PYTHONHOME", None)
        if not plan["runtime"].get("trusted_code"):
            self.environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    def bound(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("plan runtime deadline exceeded")
        if sum(p.stat().st_size for p in tree(self.out)) > self.maximum:
            raise ValueError("plan output byte bound exceeded; partial evidence retained")

    def run(self, name, arguments, *, allowed=(0,)):
        self.bound()
        command = {"step": name, "argv": [str(a) for a in arguments], "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "returncode": None}
        self.commands.append(command)
        save(self.out / "commands.json", self.commands)
        process = subprocess.Popen(command["argv"], cwd=ROOT, env=self.environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        written, seen = 0, 0
        pending = b""
        try:
            with (self.out / (name + ".log")).open("wb") as log:
                while selector.get_map():
                    self.bound()
                    for key, _ in selector.select(timeout=0.5):
                        data = os.read(key.fileobj.fileno(), 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            data, pending = pending, b""
                        else:
                            pending += data
                            if b"\n" not in pending and len(pending) < 131072:
                                continue
                            end = pending.rfind(b"\n") + 1 if b"\n" in pending else len(pending)
                            data, pending = pending[:end], pending[end:]
                        clean = safe_log(data)
                        seen += len(clean)
                        piece = clean[:max(0, LOG_LIMIT - written)]
                        log.write(piece)
                        written += len(piece)
                process.wait(timeout=max(0.01, self.deadline - time.monotonic()))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            selector.close()
            process.stdout.close()
            command.update(returncode=process.returncode, log_bytes_retained=written, log_bytes_observed=seen, log_truncated=seen > written)
            save(self.out / "commands.json", self.commands)
        if process.returncode not in allowed:
            raise RuntimeError(name + " refused or failed (exit " + str(process.returncode) + "); see bounded log and raw receipts")
        self.bound()


def workflow(plan, out, runner, outputs):
    sys.path.insert(0, str(ROOT / "bin"))
    from fidelity import dsformat, jobcontract
    manifest = source_manifest(plan)
    save(out / "source-manifest.json", manifest)
    save(out / "harness.json", jobcontract.finalize_bundle_manifest(manifest["source_files"], SOURCE + "@" + manifest["revision"]))
    tool = [sys.executable, ROOT / "bin/fidelity_dataset.py"]
    inputs = plan["inputs"]
    mode = plan["mode"]
    for name in ("reference", "candidate"):
        if inputs.get(name):
            path = Path(inputs[name]["mount_path"])
            observed = dsformat.load_manifest(str(path))
            if observed["dataset_sha256"] != inputs[name]["dataset_sha256"]:
                raise ValueError(name + " dataset identity mismatch")
            runner.run("verify-" + name, [*tool, "verify", path, "--verify-tensors", "--json", out / (name + ".verify.json")])
            save(out / (name + ".input.json"), inputs[name])
    if mode != "compare":
        model = model_binding(plan, out)
        descriptor = inputs["panel"]
        panel = Path(descriptor["mount_path"]) / str(relative(descriptor["path"]))
        if panel.resolve() != panel or not panel.is_dir():
            raise ValueError("unsafe or missing raw token-panel tree")
        for path in tree(panel):
            runner.bound()
            destination = out / "input-panel" / path.relative_to(panel)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        from fidelity import panel as panel_api
        tokenizer_root = Path(inputs["tokenizer"]["mount_path"]) if mode == "candidate" else model
        resolved = panel_api.resolve_panel(panel, role="final", tokenizer_root=tokenizer_root).to_dict()
        save(out / "panel-binding.json", resolved)
        binding = load_json(out / "panel-binding.json")
        if not binding["tokenizer"]["files_verified"]:
            raise ValueError("tokenizer files were not exactly verified")
        license_name = inputs["model"].get("license_file")
        if license_name not in ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE-MODEL", "LICENSE-MODEL.txt"):
            raise ValueError("source checkpoint has no admitted original license identity")
        license_path = regular(model / license_name)
        if not 0 < license_path.stat().st_size <= 1048576:
            raise ValueError("source license must be nonempty UTF-8 and <=1 MiB")
        license_path.read_text(encoding="utf-8")
        common = [sys.executable, ROOT / "engines/tools/hf_capture.py", "--model", model,
                  "--model-revision", inputs["model"]["revision"], "--weights-repository", inputs["model"]["repository"],
                  "--panel", panel, "--panel-role", "final", "--panel-binding", out / "panel-binding.json",
                  "--panel-binding-sha256", digest(out / "panel-binding.json"), "--panel-tokenizer-root", tokenizer_root,
                  "--role", "root" if mode == "root" else "quant", "--lane", "other", "--device", plan["hardware"]["device"],
                  "--dtype", "bfloat16", "--schedule", "layer-outer", "--dataset-id", "fidelity--" + plan["workflow_id"],
                  "--dataset-name", "HF workflow " + plan["workflow_id"], "--repository", plan["output"]["dataset_repository"],
                  "--author", plan["owner"], "--dataset-license", "other", "--weights-license-file", license_path,
                  "--weights-license-sha256", digest(license_path), "--weights-license-bytes", license_path.stat().st_size]
        common.extend(["--panel-repository", descriptor["repository"], "--panel-revision", descriptor["revision"]])
        if mode == "candidate":
            save(out / "scope.json", plan["scope"])
            common.extend(["--scope-file", out / "scope.json", "--codec", plan["codec"]])
            reference_manifest = dsformat.load_manifest(inputs["reference"]["mount_path"])
            base_capture = {"dataset_sha256": reference_manifest["dataset_sha256"],
                            "capture_content_digest": reference_manifest["capture"]["capture_content_digest"],
                            "repository": inputs["reference"]["repository"], "revision": inputs["reference"]["revision"],
                            "note": "Verified immutable reference mounted read-only before both cold captures"}
            common.extend(["--base-capture", canonical(base_capture).decode()])
            if plan.get("declared_bits") is not None:
                common.extend(["--declared-bits", plan["declared_bits"]])
        if plan.get("registered"):
            common.extend(["--model-ref", plan["registered"]["model_ref"]])
        code = plan["runtime"].get("trusted_code")
        if code:
            common.extend(["--trust-remote-code", "--code-repository", code["repository"], "--code-revision", code["revision"]])
        allowlist = plan["runtime"].get("unexpected_allowlist")
        if allowlist:
            common.extend(["--unexpected-tensors-allowlist", ROOT / allowlist["path"], "--unexpected-tensors-allowlist-sha256", allowlist["artifact_sha256"], "--unexpected-tensors-name-sha256", allowlist["canonical_sorted_names_sha256"]])
        for name in ("first", "repeat"):
            label = plan["workflow_id"] + "-" + name
            runner.run("capture-" + name, [*common, "--out", out / name, "--run-name", label, "--cold-run", label, "--memory-report", out / (name + ".memory.json")])
            runner.run("verify-" + name, [*tool, "verify", out / name, "--verify-tensors", "--json", out / (name + ".verify.json")])
            outputs[name] = name
        runner.run("reproduction", [*tool, "compare", "--reference", out / "first", "--candidate", out / "repeat", "--out", out / "reproduction", "--device", "cpu", "--replay-device", "numpy", "--replay-dtype", "float32", "--vocab-chunk", "8192", "--verify-tensors", "--self-compare", "--force-compute", "--reference-label", plan["workflow_id"] + "-first", "--candidate-label", plan["workflow_id"] + "-repeat"])
        reproduction = load_json(out / "reproduction/comparison-receipt.json")
        if reproduction["comparison_kind"] != "reproduction_confirmation" or reproduction["self_compare"].get("force_compute_agreed") is not True:
            raise ValueError("two cold captures did not pass forced exact numerical self-control")
        outputs["reproduction"] = "reproduction/comparison-receipt.json"
    if mode != "root":
        runner.run("comparison", [*tool, "compare", "--reference", inputs["reference"]["mount_path"], "--candidate", out / "first" if mode == "candidate" else inputs["candidate"]["mount_path"], "--out", out / "comparison", "--device", "cpu", "--replay-device", "numpy", "--replay-dtype", "float32", "--vocab-chunk", "8192", "--verify-tensors", "--own-heads"], allowed=(0, 2))
        outputs["comparison"] = "comparison/comparison-receipt.json"
        comparison = load_json(out / outputs["comparison"])
        if not all(g.get("passed") is True and not g.get("overridden_by") for g in comparison["gates"].values()):
            raise ValueError("comparison failed or overrode a scientific gate")
        registered = plan.get("registered")
        if registered and comparison["comparison_kind"] == "measurement":
            from fidelity import dscompare
            for field in ("model_ref", "panel_ref", "reference_ref", "registry_repository", "registry_revision", "artifact", "panel", "reference"):
                if not registered.get(field):
                    raise ValueError("incomplete registered submission provenance: " + field)
            if not HEX40.fullmatch(registered["registry_revision"]):
                raise ValueError("registered provenance lacks immutable revision")
            candidate_manifest = dsformat.load_manifest(str(out / "first") if mode == "candidate" else inputs["candidate"]["mount_path"])
            artifact = registered["artifact"]
            if any(artifact.get(key) != candidate_manifest["weights"].get(key) for key in ("repository", "revision")):
                raise ValueError("submission artifact differs from the actually measured weights")
            if registered["panel"].get("panel_ref") != registered["panel_ref"] or registered["reference"].get("reference_ref") != registered["reference_ref"]:
                raise ValueError("registered submission references disagree")
            if registered["panel"].get("panel_token_sha256") != candidate_manifest["panel"]["suite_token_hash_sha256"]:
                raise ValueError("registered panel token identity differs from measured capture")
            from fidelity import dsmanifest
            scope = artifact["scope"]
            expected_scope = dsmanifest.scope_block(assignments=scope["assignments"], head_policy=scope["head_policy"], kv_cache_dtype=scope["kv_cache_dtype"], policy=scope["policy"])
            if candidate_manifest["scope"] != expected_scope:
                raise ValueError("registered artifact scope differs from the actually measured intervention")
            save(out / "submission-provenance.json", registered)
            measurer = {"name": plan["owner"], "handle": plan["owner"], "url": "https://huggingface.co/" + plan["owner"], "is_artifact_author": registered["artifact"]["repository"].split("/")[0] == plan["owner"]}
            dscompare.emit_submission(comparison, str(out / "comparison/submission-receipt.json"), measurer=measurer, artifact=registered["artifact"], panel=registered["panel"], reference=registered["reference"])
            runner.run("submission-validation", [sys.executable, ROOT / "registry/tools/registry_validate.py", "--submission", out / "comparison/submission-receipt.json"])
            outputs["submission"] = "comparison/submission-receipt.json"
    runner.bound()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if Path(args.plan) != PLAN_PATH or out != OUT_PATH or out.parent.resolve() != Path("/outputs"):
        parser.error("only --plan /inputs/plan/plan.json --out /outputs/result is admitted")
    if out.exists() or out.is_symlink():
        parser.error("refusing an existing result tree; use a fresh per-attempt bucket prefix")
    out.mkdir(mode=0o700)
    plan, outputs, error = {}, {}, None
    status = "failed"
    started = time.monotonic()
    def deadline_signal(signum, frame):
        raise TimeoutError("plan runtime deadline exceeded")
    try:
        require_no_credentials()
        document = load_json(regular(PLAN_PATH), 4 * 1024 * 1024)
        if not isinstance(document, dict):
            raise ValueError("plan must be a JSON object")
        plan = document
        final_deadline = None
        validate_plan(plan, out)
        remaining = plan["hardware"]["timeout_seconds"] - max(0, time.time() - float(os.environ.get("QFS_WORKFLOW_STARTED", time.time())))
        if remaining <= 0:
            raise TimeoutError("bootstrap consumed the plan runtime deadline")
        signal.signal(signal.SIGALRM, deadline_signal)
        signal.signal(signal.SIGTERM, deadline_signal)
        final_deadline = time.monotonic() + remaining
        execution_seconds = remaining - min(30.0, remaining / 10)
        signal.setitimer(signal.ITIMER_REAL, execution_seconds)
        save(out / "plan.json", plan)
        for name in ("bootstrap.log", "bootstrap.json"):
            path = out.parent / name
            if path.exists():
                regular(path)
                if path.stat().st_size > LOG_LIMIT:
                    raise ValueError("bootstrap evidence exceeds its bound")
                shutil.copyfile(path, out / name)
        runner = Runner(plan, out, final_deadline - min(30.0, remaining / 10))
        workflow(plan, out, runner, outputs)
        status = "complete"
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": safe_log(str(exc).encode())[:4096].decode("utf-8", "replace")}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result = {"schema": "qfs.hf-workflow-result.v1", "workflow_id": plan.get("workflow_id"), "owner": plan.get("owner"), "mode": plan.get("mode"), "plan_sha256": plan.get("plan_sha256"), "status": status, "outputs": outputs,
              "source": plan.get("source") if isinstance(plan.get("source"), dict) else {}, "files": []}
    manifest_path = out / "source-manifest.json"
    if manifest_path.is_file():
        result["source"] = load_json(manifest_path)
        result["source"].pop("schema", None)
    if error:
        result["error"] = error
    try:
        if "final_deadline" in locals() and final_deadline is not None:
            signal.setitimer(signal.ITIMER_REAL, max(0.01, final_deadline - time.monotonic() - 1))
        result["files"] = sorted((row(p, out) for p in tree(out) if p != out / "result.json"), key=lambda r: r["path"])
        maximum = (plan.get("limits") or {}).get("max_output_bytes")
        if status == "complete" and sum(item["bytes"] for item in result["files"]) + len(canonical(result)) + 128 > maximum:
            raise ValueError("complete result including its manifest exceeds the plan output bound")
    except BaseException as exc:
        result["status"] = status = "failed"
        result["error"] = {"type": type(exc).__name__, "message": "Cannot safely inventory partial evidence: " + safe_log(str(exc).encode())[:4096].decode("utf-8", "replace")}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    save(out / "result.json", seal(result, "result_sha256"))
    os.sync()
    print(json.dumps({"status": status, "workflow_id": result["workflow_id"], "result_sha256": result["result_sha256"]}), flush=True)
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
