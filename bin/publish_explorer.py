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


ALLOW_PATTERNS = [
    "app.py", "requirements.txt", "README.md", "LICENSE", "llms.txt",
    "explorer/*.py", "explorer/pricing.json", "bin/fidelity/*.py",
    "registry/tools/*.py", "registry/data/*.jsonl", "registry/index.json",
    "registry/schema/*.json", "docs/schema/*.json",
    "registry/docs/examples/dione-q4.submission.json",
]


def main():
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="malaiwah/qfs-explorer")
    parser.add_argument("--private", action="store_true", help="Create a private Space instead of a public one")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
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
    commit = api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=root,
                               allow_patterns=ALLOW_PATTERNS,
                               ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
                               commit_message="Publish QFS Explorer on CPU Basic")
    print("Space: https://huggingface.co/spaces/%s" % args.repo)
    print("Deployment commit: %s" % commit.oid)


if __name__ == "__main__":
    main()
