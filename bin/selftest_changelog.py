#!/usr/bin/env python3
"""A generated changelog stays current after its own commit, without hiding code changes."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE = Path(__file__).with_name("changelog.py")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="qfs-changelog-") as temporary:
        root = Path(temporary)
        (root / "bin").mkdir()
        shutil.copy2(SOURCE, root / "bin" / "changelog.py")
        code = root / "example.py"
        code.write_text("VALUE = 1\n", encoding="utf-8")

        def git(*args):
            return subprocess.run(
                ["git", "-C", str(root), "-c", "user.name=QFS selftest",
                 "-c", "user.email=selftest@example.invalid", "-c", "commit.gpgsign=false",
                 "-c", "core.hooksPath=" + str(root / ".git" / "hooks"), *args],
                check=True, capture_output=True, text=True, timeout=20)

        def changelog(*args):
            return subprocess.run(
                [sys.executable, str(root / "bin" / "changelog.py"), *args],
                capture_output=True, text=True, timeout=20)

        git("init", "-q")
        git("add", "bin/changelog.py", "example.py")
        git("commit", "-qm", "science: initial fixture behavior")
        assert changelog("--all", "--out", str(root / "CHANGELOG.md")).returncode == 0
        first = (root / "CHANGELOG.md").read_bytes()
        git("add", "CHANGELOG.md")
        git("commit", "-qm", "docs: generated history snapshot")
        checked = changelog("--check")
        assert checked.returncode == 0, checked.stderr
        assert (root / "CHANGELOG.md").read_bytes() == first

        code.write_text("VALUE = 2\n", encoding="utf-8")
        git("add", "example.py")
        git("commit", "-qm", "changelog: real code change still belongs in history")
        assert changelog("--check").returncode == 1
        assert changelog("--all", "--out", str(root / "CHANGELOG.md")).returncode == 0
        assert b"real code change still belongs in history" in (root / "CHANGELOG.md").read_bytes()
        git("add", "CHANGELOG.md")
        git("commit", "-qm", "docs: refresh generated history")
        assert changelog("--check").returncode == 0
    print("PASS: changelog self-commit stability and real-change detection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
