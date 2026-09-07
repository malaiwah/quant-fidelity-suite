"""Architecture/format evidence catalog and metadata-only capture recipes.

This module never loads model code, downloads weights, rents compute or publishes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys

from . import panel

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "engines" / "coverage.json"
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
AUTHOR = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
DATASET_ID = re.compile(r"fidelity--[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def load_catalog():
    with CATALOG.open(encoding="utf-8") as source:
        catalog = json.load(source)
    if catalog.get("schema") != "qfs.coverage-catalog.v1":
        raise ValueError("Unsupported coverage catalog schema")
    return catalog


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def prepare_recipe(args):
    catalog = load_catalog()
    entry = next((e for e in catalog["architectures"] if e["id"] == args.architecture), None)
    if entry is None:
        raise ValueError("Unknown architecture; use fidelity-dataset architectures to list tested families")
    for value, label in ((args.model_repository, "model repository"), (args.dataset_repository, "dataset repository")):
        if not REPO_ID.fullmatch(value):
            raise ValueError(label + " must be owner/repository")
    if not SHA40.fullmatch(args.model_revision):
        raise ValueError("model revision must be a full lowercase 40-character commit")
    if not AUTHOR.fullmatch(args.author):
        raise ValueError("author must be your actual HF handle; no maintainer identity is inferred")
    if not DATASET_ID.fullmatch(args.dataset_id):
        raise ValueError("dataset ID must begin fidelity-- and contain only letters, digits, dots, underscores or hyphens")
    model_dir = Path(args.model_dir).resolve(strict=True)
    tokenizer_dir = Path(args.tokenizer_root or args.model_dir).resolve(strict=True)
    panel_dir = Path(args.panel).resolve(strict=True)
    if not all(p.is_dir() for p in (model_dir, tokenizer_dir, panel_dir)):
        raise ValueError("model, tokenizer and panel inputs must be existing local directories")
    config_path = model_dir / "config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("model_type") not in entry["model_types"]:
        raise ValueError("Model config does not belong to the selected architecture: " + str(config.get("model_type")))
    if args.role == "quant" and (not args.scope_file or not args.reference):
        raise ValueError("A quant recipe requires --scope-file describing the real intervention and --reference fidelity dataset")
    if args.role == "root" and (args.scope_file or args.reference):
        raise ValueError("Root recipes produce their own two-capture control; use role quant for a reference/candidate comparison")
    scope_path = Path(args.scope_file).resolve(strict=True) if args.scope_file else None
    if scope_path and not isinstance(json.loads(scope_path.read_text()), dict):
        raise ValueError("scope-file must contain a scope JSON object")
    license_path = Path(args.weights_license_file or model_dir / "LICENSE").resolve(strict=True)
    if not license_path.is_file() or not 0 < license_path.stat().st_size <= 1048576:
        raise ValueError("weights license must be a regular nonempty file <=1 MiB")
    license_raw = license_path.read_bytes()
    license_raw.decode("utf-8")
    binding = panel.resolve_panel(panel_dir, role=args.panel_role, tokenizer_root=tokenizer_dir).to_dict()
    if not binding["tokenizer"]["files_verified"]:
        raise ValueError("The panel's tokenizer files were not verified")
    code_args = []
    if args.trust_remote_code:
        if not args.code_repository or not REPO_ID.fullmatch(args.code_repository) or not args.code_revision or not SHA40.fullmatch(args.code_revision):
            raise ValueError("Explicit custom-code consent also requires --code-repository and a full --code-revision")
        code_args = ["--trust-remote-code", "--code-repository", args.code_repository, "--code-revision", args.code_revision]
    elif args.code_repository or args.code_revision:
        raise ValueError("Code pins do not imply consent; explicitly add --trust-remote-code after reviewing the code")
    elif entry.get("runtime_mode") == "pinned_code":
        raise ValueError("This tested runtime needs explicit code consent and pin; inspect the architecture's tested_runtime entry")
    if args.bits is not None and not 0 < args.bits <= 64:
        raise ValueError("Declared bits must be positive and <=64")
    destination = Path(args.out).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Refusing to overwrite a recipe directory: " + str(destination))
    destination.mkdir(parents=True, mode=0o700)
    binding_path = destination / "panel-binding.json"
    _write_json(binding_path, binding)
    binding_sha = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    program = str(ROOT / "engines/tools/hf_capture.py")
    dataset_tool = str(ROOT / "bin/fidelity_dataset.py")
    common = ["--model", str(model_dir), "--weights-repository", args.model_repository,
              "--model-revision", args.model_revision, "--panel", str(panel_dir),
              "--panel-role", args.panel_role, "--panel-binding", str(binding_path),
              "--panel-binding-sha256", binding_sha, "--panel-tokenizer-root", str(tokenizer_dir),
              "--role", args.role, "--lane", args.lane, "--device", args.device,
              "--dtype", "bfloat16", "--schedule", args.schedule,
              "--dataset-id", args.dataset_id, "--dataset-name", args.dataset_name or args.dataset_id,
              "--repository", args.dataset_repository, "--author", args.author,
              "--dataset-license", "other", "--weights-license-file", str(license_path),
              "--weights-license-sha256", hashlib.sha256(license_raw).hexdigest(),
              "--weights-license-bytes", str(len(license_raw)), *code_args]
    if args.drop_parallel_plan:
        common.append("--drop-parallel-plan")
    if args.sanity_expect is not None:
        common.extend(["--sanity-expect", args.sanity_expect])
    if scope_path:
        common.extend(["--scope-file", str(scope_path)])
    if args.codec:
        common.extend(["--codec", args.codec])
    if args.bits is not None:
        common.extend(["--declared-bits", str(args.bits)])
    commands = []
    for side in (("first", "repeat") if args.role == "root" else ("candidate",)):
        output = destination / side
        label = args.dataset_id + "-" + side
        commands.append({"step": "capture-" + side, "argv": [sys.executable, program, *common,
            "--out", str(output), "--run-name", label, "--cold-run", label,
            "--memory-report", str(destination / (side + ".memory.json"))]})
        commands.append({"step": "verify-" + side, "argv": [sys.executable, dataset_tool, "verify", str(output), "--verify-tensors", "--json", str(destination / (side + ".verify.json"))]})
    comparison = [sys.executable, dataset_tool, "compare", "--reference",
                  str(destination / "first") if args.role == "root" else args.reference,
                  "--candidate", str(destination / ("repeat" if args.role == "root" else "candidate")),
                  "--out", str(destination / "comparison"), "--device", "cpu", "--replay-device", "numpy",
                  "--replay-dtype", "float32", "--vocab-chunk", "8192", "--verify-tensors"]
    comparison.extend(["--self-compare", "--force-compute"] if args.role == "root" else ["--own-heads"])
    if args.role == "root":
        comparison.extend(["--reference-label", args.dataset_id + "-first",
                           "--candidate-label", args.dataset_id + "-repeat"])
    commands.append({"step": "compare", "argv": comparison})
    warnings = [
        "Prepared commands only: no model/code was loaded, no weights downloaded, no hardware rented and nothing published.",
        "Fixture success is architecture-path evidence, not qualification of every original FP8/MXFP4/GGUF export or production GPU kernel.",
        "Read the selected support entry and original model/code licenses. Captures include output-head weights; respect their redistribution terms.",
        "A panel is fixed token IDs, not candidate retokenization. For a quantization claim, establish shared tokenizer/model ancestry; the recipe does not invent it.",
        "Use an appropriate representative panel for real measurements. Tiny synthetic panels are pipeline checks, not quality rankings.",
        "The shell stops on any nonzero exit, including validation warnings (exit2). Inspect receipts before continuing; do not suppress refusals.",
        "This local workflow does not qualify or publish a production root or add a registry row. Existing qualification/submission rules still apply.",
    ]
    if code_args:
        warnings.append("CUSTOM CODE: execution commands will run Python from the specified immutable repository pin. Hash verification is provenance, not a sandbox. No code was executed while preparing this recipe.")
    document = {"schema": "qfs.capture-workflow.v1", "architecture": entry["id"], "model_repository": args.model_repository,
                "model_revision": args.model_revision, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "dataset_repository": args.dataset_repository, "author": args.author, "role": args.role,
                "panel_binding_sha256": binding_sha, "source_catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
                "commands": commands, "warnings": warnings, "support": entry}
    _write_json(destination / "workflow.json", document)
    script = ["#!/usr/bin/env bash", "set -euo pipefail", "# Review workflow.json and source licenses before executing.", ""]
    for command in commands:
        script.extend(["# " + command["step"], shlex.join(command["argv"]), ""])
    with (destination / "run.sh").open("x", encoding="utf-8") as output:
        output.write("\n".join(script))
    os.chmod(destination / "run.sh", 0o700)
    return document


def configure_parser(subparsers):
    command = subparsers.add_parser("architectures", help="list verified architecture/format coverage or prepare a local capture recipe")
    actions = command.add_subparsers(dest="architecture_action")
    listing = actions.add_parser("list", help="show coverage and public fixtures; no model execution")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=run_cli)
    show = actions.add_parser("show", help="show one architecture's exact evidence and restrictions")
    show.add_argument("architecture")
    show.set_defaults(func=run_cli)
    recipe = actions.add_parser("prepare", help="write bound capture/compare commands without running them")
    for flag in ("architecture", "model-dir", "model-repository", "model-revision", "panel", "dataset-repository", "dataset-id", "author", "out"):
        recipe.add_argument("--" + flag, required=True)
    recipe.add_argument("--role", choices=("root", "quant"), default="root")
    recipe.add_argument("--dataset-name")
    recipe.add_argument("--device", default="cpu")
    recipe.add_argument("--schedule", choices=("window-outer", "layer-outer"), default="layer-outer")
    recipe.add_argument("--lane", choices=("sealed-ep8", "streaming", "local-mps", "local-cuda-budget", "other"), default="other")
    recipe.add_argument("--panel-role", default="final")
    recipe.add_argument("--tokenizer-root")
    recipe.add_argument("--weights-license-file")
    recipe.add_argument("--scope-file")
    recipe.add_argument("--reference")
    recipe.add_argument("--codec")
    recipe.add_argument("--bits", type=float)
    recipe.add_argument("--sanity-expect")
    recipe.add_argument("--drop-parallel-plan", action="store_true")
    recipe.add_argument("--trust-remote-code", action="store_true")
    recipe.add_argument("--code-repository")
    recipe.add_argument("--code-revision")
    recipe.set_defaults(func=run_cli)
    command.set_defaults(func=run_cli)


def run_cli(args):
    try:
        if args.architecture_action == "prepare":
            result = prepare_recipe(args)
            print("Prepared %s/run.sh and workflow.json. No execution occurred." % args.out)
            for warning in result["warnings"]:
                print("  " + warning)
            return 0
        catalog = load_catalog()
        if args.architecture_action == "show":
            match = next((e for e in catalog["architectures"] if e["id"] == args.architecture), None)
            if match is None:
                raise ValueError("Unknown architecture: " + args.architecture)
            print(json.dumps(match, indent=2))
        elif getattr(args, "json", False):
            print(json.dumps(catalog, indent=2))
        else:
            print("QFS tested architecture paths — not blanket quantizer or paid-execution admission")
            for entry in catalog["architectures"]:
                print("%-20s %-16s %s" % (entry["id"], entry["runtime_mode"], entry["fixture"]["repository"]))
            print("Use architectures show NAME for original-format limitations and immutable fixture evidence.")
        return 0
    except (ValueError, OSError, KeyError, panel.PanelError) as exc:
        print("REFUSED [capture_workflow]: " + str(exc))
        return 3
