#!/usr/bin/env python3
"""Stage the exact bundle and execute setup selftests with explicit coverage.

A missing external quant_pipeline can be a skipped prerequisite. An arbitrary
failure whose earlier output mentions that package is still a failure. Bundle
completeness is not native/GPU qualification; nested skips remain visible.
"""
import ast
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from selftest_partition import is_internal_skip

SUITE = Path(__file__).resolve().parent.parent
FAILED = []


def check(label, ok, detail=""):
    print("  %s  %s%s" % ("PASS" if ok else "FAIL", label,
                          (" -- " + detail) if detail and not ok else ""))
    if not ok:
        FAILED.append(label)


def bundle_entries():
    text = (SUITE / "bin" / "BUNDLE.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.startswith("#")]


def setup_selftests():
    text = (SUITE / "bin" / "bootstrap_measure.sh").read_text(encoding="utf-8")
    return sorted(set(re.findall(r"(selftest_[a-z0-9_]+\.py)", text)))


def run_setup_selftest(script):
    """Run a real child; only an explicit terminal dependency refusal is a skip."""
    script = Path(script)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(script.parent),
                          capture_output=True, text=True, timeout=900)
    output = proc.stdout + proc.stderr
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    terminal = lines[-1] if lines else ""
    missing_pipeline = (
        terminal == "pass --pipeline-root (tree containing quant_pipeline)"
        or re.fullmatch(r"ModuleNotFoundError: No module named ['\"]quant_pipeline(?:\.[A-Za-z0-9_.]+)?['\"]",
                        terminal) is not None)
    skipped = [line for line in lines if is_internal_skip(line)]
    status = ("skip" if proc.returncode != 0 and missing_pipeline else
              "fail" if proc.returncode != 0 else "partial" if skipped else "pass")
    return {"status": status, "returncode": proc.returncode, "output": output,
            "skip_notices": skipped, "reason": terminal if status == "skip" else None}


def import_gaps(stage):
    package = stage / "bin" / "fidelity"
    gaps = []
    for module in sorted(package.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"), str(module))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.ImportFrom):
                if node.level >= 1:
                    if node.module:
                        targets.append(node.module.split(".")[0])
                    else:
                        targets.extend(alias.name for alias in node.names)
                elif node.module and node.module.split(".")[0] == "fidelity":
                    parts = node.module.split(".")
                    if len(parts) > 1:
                        targets.append(parts[1])
                    else:
                        targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                targets.extend(alias.name.split(".")[1] for alias in node.names
                               if alias.name.startswith("fidelity."))
            for name in targets:
                if not (package / (name + ".py")).is_file():
                    gaps.append("%s:%d -> fidelity.%s" % (module.name, node.lineno, name))
    return gaps


def main():
    FAILED.clear()
    skipped = 0
    print("== the tree an instance actually receives ==")
    entries = bundle_entries()
    absent = [entry for entry in entries if not (SUITE / entry).is_file()]
    check("every BUNDLE.txt entry exists in the repo", not absent, str(absent[:4]))
    with tempfile.TemporaryDirectory(prefix="fidbundle-") as directory:
        stage = Path(directory)
        for relative in entries:
            source = SUITE / relative
            if not source.is_file():
                continue
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        staged = sum(1 for path in stage.rglob("*") if path.is_file())
        check("bundle stages cleanly (%d files)" % staged,
              staged == len(entries) - len(absent))
        print("\n== and every bundled fidelity module's imports are bundled ==")
        gaps = import_gaps(stage)
        check("every intra-package import resolves in the bundle", not gaps,
              "\n        ".join(gaps[:6]))
        print("\n== setup selftests executed from the staged bundle ==")
        names = setup_selftests()
        check("bootstrap names selftests", bool(names))
        for name in names:
            script = stage / "engines" / "tools" / name
            if not script.is_file():
                print("  SKIP %s: bootstrap's optional file is not bundled" % name)
                skipped += 1
                continue
            result = run_setup_selftest(script)
            if result["status"] == "skip":
                print("  SKIP %s: %s" % (name, result["reason"]))
                skipped += 1
            else:
                check("%s executes from the bundle" % name, result["status"] != "fail",
                      "\n        ".join(result["output"].strip().splitlines()[-6:]))
            for notice in result["skip_notices"]:
                print("  SKIP %s nested coverage: %s" % (name, notice))
                skipped += 1
    print()
    if FAILED:
        print("selftest_bundle_complete: %d FAILED" % len(FAILED))
        return 1
    print("selftest_bundle_complete: bundle checks passed; %d skip notice(s), not native qualification" % skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
