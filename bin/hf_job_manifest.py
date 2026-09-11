#!/usr/bin/env python3
"""Record the HF Jobs launcher overlay without rewriting the baked runtime BUILD."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

BASE_IMAGE = "ghcr.io/malaiwah/quant-fidelity-measure@sha256:5b1f3bb6e899d55d9a21c9765243cec7c26aea5f5b1777d16dc801d92ffc8783"
BASE_BUILD = "d9115aea633c6aa123f0c84cec2bccb4c5b94a6528707a0d8e6bfeab521cb0a1"
BASE_CONTENT = "f10f26179ffa2342941b7fc2b17fdd9a41c20a3120a3d0844bc76afab22d2b1d"
BASE_SOURCE = "987e1146c2337a918f847391cbf685f20a6ad82b"
LAUNCHER = "/usr/local/bin/qfs-job"


def generate(image_root, launcher, revision, base_image):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("launcher source revision must be an immutable 40-hex commit")
    if base_image != BASE_IMAGE:
        raise ValueError("HF Jobs overlay requires the reviewed exact measurement base image")
    raw = (image_root / "BUILD.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_BUILD:
        raise ValueError("base BUILD.json differs from the reviewed baked runtime")
    build = json.loads(raw)
    if (build.get("schema") != "malaiwah.fidelity-image-build.v1" or build.get("probe_errors")
            or build.get("image_content_sha256") != BASE_CONTENT
            or build.get("suite_revision") != BASE_SOURCE
            or (image_root / "image-pin.txt").read_text().strip() != BASE_CONTENT):
        raise ValueError("base content/source identity differs from the reviewed baked runtime")
    if launcher.is_symlink() or not stat.S_ISREG(launcher.stat().st_mode) or stat.S_IMODE(launcher.stat().st_mode) != 0o755:
        raise ValueError("launcher must be an installed 0755 regular file")
    return {"schema": "qfs.hf-job-launcher.v1", "launch_contract": "measurement-cli-v1",
            "launcher_path": LAUNCHER, "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
            "launcher_source_revision": revision, "base_image": base_image,
            "build_sha256": BASE_BUILD, "image_content_sha256": BASE_CONTENT,
            "baked_source_revision": BASE_SOURCE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("/opt/fidelity"))
    parser.add_argument("--launcher", type=Path, default=Path(LAUNCHER))
    parser.add_argument("--suite-revision", required=True)
    parser.add_argument("--base-image", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = generate(args.image_root, args.launcher, args.suite_revision, args.base_image)
        target = args.image_root / "JOBS.json"
        if target.exists() or target.is_symlink():
            raise ValueError("refusing to replace an existing launcher manifest")
        raw = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        fd, name = tempfile.mkstemp(prefix=".JOBS-", dir=args.image_root)
        try:
            with os.fdopen(fd, "wb") as output:
                os.fchmod(output.fileno(), 0o644)
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, target)
        finally:
            Path(name).unlink(missing_ok=True)
        print("JOBS.json sha256=" + hashlib.sha256(raw).hexdigest())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
