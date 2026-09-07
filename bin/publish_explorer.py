#!/usr/bin/env python3
"""Publish the CPU-only Explorer. Uses the caller's local HF authentication.

Run from a committed checkout:
    python bin/publish_explorer.py --repo malaiwah/qfs-explorer

Only app/source/schema/registry metadata is uploaded: no weights, captures,
credentials or local run directories. This never upgrades to paid hardware.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import subprocess


ALLOW_PATTERNS = [
    "app.py", "requirements.txt", "README.md", "LICENSE", "llms.txt",
    "explorer/*.py", "explorer/*.json", "explorer/requirements-worker.txt", "bin/fidelity/*.py",
    "bin/fidelity_dataset.py", "bin/BUNDLE.txt", "engines/tools/*.py",
    "bin/jointstd/*.py",
    "engines/panels/panel--fruit.malaiwah.heldout-v1/**",
    "engines/tools/layer-outer-evidence/fruit*unexpected-keys.json*",
    "registry/tools/*.py", "registry/data/*.jsonl", "registry/index.json",
    "registry/receipts/**", "registry/protocol/**", "registry/README.head.md",
    "registry/publication-audit.json",
    "registry/schema/*.json", "docs/schema/*.json",
    "docs/CARD-ANNOTATION-SPEC.md",
    "docs/THIRD-PARTY-QUICKSTART.md",
    "engines/coverage.json", "engines/quant-coverage-audit.json",
    "registry/docs/examples/dione-q4.submission.json",
]


def main():
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="malaiwah/qfs-explorer")
    parser.add_argument("--private", action="store_true", help="Create a private Space instead of a public one")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    image = json.loads((root / "explorer/job_environment.json").read_text())["image"]
    deployment = {"schema": "qfs.explorer-deployment.v1", "source_revision": revision, "image": image}
    for name, key in (("job_worker.py", "worker_sha256"), ("job_bootstrap.py", "bootstrap_sha256")):
        path = root / "explorer" / name
        committed = subprocess.check_output(["git", "show", revision + ":explorer/" + name], cwd=root)
        if committed != path.read_bytes():
            raise SystemExit("Commit the reviewed worker before publishing the Space.")
        deployment[key] = hashlib.sha256(committed).hexdigest()
    api = HfApi()
    # hardware is only requested when creating a new repository. Never upgrade an
    # existing owner's Space, change its visibility or replace its secrets.
    api.create_repo(repo_id=args.repo, repo_type="space", space_sdk="gradio",
                    space_hardware="cpu-basic", private=args.private, exist_ok=True)
    runtime = api.get_space_runtime(args.repo)
    states = [str(value) for value in (runtime.hardware, runtime.requested_hardware) if value is not None]
    if not states or any(value not in ("cpu-basic", "SpaceHardware.CPU_BASIC") for value in states):
        raise SystemExit("Refusing to publish without CPU-Basic-only current/requested hardware on %s (%s). No hardware was changed."
                         % (args.repo, states))
    reviewed = {}
    try:
        previous = json.loads(Path(hf_hub_download(
            args.repo, "explorer/deployment.json", repo_type="space", token=api.token)).read_text())
    except EntryNotFoundError:
        previous = {}
    local_manifest = root / "explorer/deployment.json"
    prior_manifests = [previous]
    if local_manifest.exists():
        prior_manifests.append(json.loads(local_manifest.read_text()))
    for prior in prior_manifests:
        reviewed.update(prior.get("reviewed_source_revisions", {}))
        if prior.get("source_revision") and all(prior.get(k) for k in ("worker_sha256", "bootstrap_sha256", "image")):
            reviewed[prior["source_revision"]] = {k: prior[k] for k in ("worker_sha256", "bootstrap_sha256", "image")}
    reviewed[revision] = {k: deployment[k] for k in ("worker_sha256", "bootstrap_sha256", "image")}
    deployment["reviewed_source_revisions"] = reviewed
    local_manifest.write_text(json.dumps(deployment, indent=2) + "\n")
    commit = api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=root,
                               allow_patterns=ALLOW_PATTERNS,
                               ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
                               commit_message="Publish QFS Explorer on CPU Basic")
    print("Space: https://huggingface.co/spaces/%s" % args.repo)
    print("Deployment commit: %s" % commit.oid)


if __name__ == "__main__":
    main()
