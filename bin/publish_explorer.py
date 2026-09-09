#!/usr/bin/env python3
"""Publish the CPU-only Explorer with an explicit file-sourced HF credential.

Run from a committed checkout:
    python bin/publish_explorer.py --repo malaiwah/qfs-explorer --token-file ~/.config/qfs/hf-token

Only app/source/schema/registry metadata is uploaded: no weights, captures,
credentials or local run directories. This never upgrades to paid hardware.
The HF token is read ONLY from the 0600 owner-only file named by --token-file:
never a command-line argument, never $HF_TOKEN or the ambient login cache, and
never echoed, logged or bundled.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import os
import stat
import subprocess


ALLOW_PATTERNS = [
    "app.py", "requirements.txt", "README.md", "LICENSE", "llms.txt",
    "explorer/*.py", "explorer/*.json", "explorer/requirements-worker.txt", "bin/fidelity/*.py",
    "bin/fidelity_dataset.py", "bin/BUNDLE.txt", "engines/tools/*.py",
    "bin/jointstd/*.py",
    "engines/tools/*.json", "engines/tools/**/*.json", "engines/scopes/*.json", "engines/scopes/**/*.json",
    "engines/panels/**",
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


def _read_token_file(path):
    """Read the token from a 0600 owner-only file; never argv, never echo."""
    if not path:
        raise SystemExit(
            "Refusing to publish: --token-file <path> is required.\n"
            "  Remedy: create an owner-only credential file first, e.g.\n"
            "    install -m 600 /dev/stdin ~/.config/qfs/hf-token   # paste the HF token, then Ctrl-D\n"
            "  and pass its path with --token-file. The token itself is never accepted as a\n"
            "  command-line argument and ambient $HF_TOKEN / the HF login cache is never used.")
    if not hasattr(os, "O_NOFOLLOW"):
        raise SystemExit("Refusing to publish: reading an explicit token file requires O_NOFOLLOW support.")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise SystemExit("Refusing to publish: could not open the token file %r: %s" % (path, exc))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SystemExit("Refusing to publish: the token file is not a regular file.")
        if info.st_uid != os.getuid():
            raise SystemExit("Refusing to publish: the token file must be owned by the current user.")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise SystemExit("Refusing to publish: the token file mode must be exactly 0600 (got 0%o). Remedy: chmod 600 %r" % (stat.S_IMODE(info.st_mode), path))
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            token = handle.read().strip()
    finally:
        if fd >= 0:
            os.close(fd)
    if not token:
        raise SystemExit("Refusing to publish: the token file is empty.")
    return token


def main():
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="malaiwah/qfs-explorer")
    parser.add_argument("--private", action="store_true", help="Create a private Space instead of a public one")
    parser.add_argument("--token-file", help="path to a 0600 owner-only file containing the HF token (required; the token itself is never an argument and ambient HF_TOKEN is never used)")
    args = parser.parse_args()
    token = _read_token_file(args.token_file)
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    environment_raw = (root / "explorer/job_environment.json").read_bytes()
    if subprocess.check_output(["git", "show", revision + ":explorer/job_environment.json"], cwd=root) != environment_raw:
        raise SystemExit("Commit the reviewed Job environment before publishing the Space.")
    environment = json.loads(environment_raw)
    image = environment["image"]
    deployment = {"schema": "qfs.explorer-deployment.v1", "source_revision": revision, "image": image,
                  "launch_contract": environment.get("launch_contract", "python-bootstrap-v1"),
                  "environment_sha256": hashlib.sha256(environment_raw).hexdigest()}
    for name, key in (("job_worker.py", "worker_sha256"), ("job_bootstrap.py", "bootstrap_sha256")):
        path = root / "explorer" / name
        committed = subprocess.check_output(["git", "show", revision + ":explorer/" + name], cwd=root)
        if committed != path.read_bytes():
            raise SystemExit("Commit the reviewed worker before publishing the Space.")
        deployment[key] = hashlib.sha256(committed).hexdigest()
    api = HfApi(endpoint="https://huggingface.co", token=token)
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
            args.repo, "explorer/deployment.json", repo_type="space", token=token)).read_text())
    except EntryNotFoundError:
        previous = {}
    local_manifest = root / "explorer/deployment.json"
    prior_manifests = [previous]
    if local_manifest.exists():
        prior_manifests.append(json.loads(local_manifest.read_text()))
    for prior in prior_manifests:
        reviewed.update(prior.get("reviewed_source_revisions", {}))
        if prior.get("source_revision") and all(prior.get(k) for k in ("worker_sha256", "bootstrap_sha256", "image")):
            reviewed[prior["source_revision"]] = {
                k: prior[k] for k in ("worker_sha256", "bootstrap_sha256", "image",
                                     "launch_contract", "environment_sha256") if k in prior}
    reviewed[revision] = {k: deployment[k] for k in ("worker_sha256", "bootstrap_sha256", "image",
                                                   "launch_contract", "environment_sha256")}
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
