#!/usr/bin/env python3
"""Consumer regressions for qualified clean-scope publication; no generated files."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin/emit_clean_scope_report.py"
spec = importlib.util.spec_from_file_location("clean_scope_report", SCRIPT)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths = ["registry/protocol/window-selection.brandonmusic-final25.json",
                 "registry/protocol/glm53-joint-kld-protocol.v1.json"]
        paths += ["registry/protocol/per-window/%s.json" % key for key, *_ in E.SERIES]
        for relative in paths:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        analysis = root / "docs/joint-standard/analysis"
        analysis.mkdir(parents=True)
        paired_path = analysis / "paired.K6-vs-FP8.selected.json"
        paired = json.loads((ROOT / "docs/joint-standard/analysis/paired.K6-vs-FP8.selected.json").read_text())
        paired["document_level"]["sampling_assumption"] = "independent documents conditional on this collection"
        paired["document_map_source"] = "/Users/example/private/document-map.json"
        paired["contract_a"]["source"] = "/home/example/private/capture.json"
        paired_path.write_text(json.dumps(paired))
        out, md = root / "report.json", root / "report.md"
        command = [sys.executable, str(SCRIPT), "--root", str(root), "--out", str(out), "--markdown", str(md)]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        report = json.loads(out.read_text())
        emitted = report["paired"]["comparisons"][0]["scopes"]["clean17"]
        for key in ("document_level", "inference_unit", "window_stats_are", "cross_lane"):
            assert emitted[key] == paired[key], key
        markdown = md.read_text()
        for text in (paired["window_stats_are"], paired["cross_lane"]["bridge"],
                     paired["document_level"]["sampling_assumption"],
                     paired["document_level"]["t_interval_note"]):
            assert text in markdown, text
        assert "/Users/" not in markdown and "/home/" not in markdown
        assert "/Users/" not in out.read_text() and "/home/" not in out.read_text()
        assert emitted["document_map_source"] == (
            "producer-local-path-sha256:" +
            hashlib.sha256(paired["document_map_source"].encode()).hexdigest())
        assert emitted["contract_a"]["source"] != emitted["document_map_source"]
        assert emitted["source_analysis"] == {
            "path": paired_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(paired_path.read_bytes()).hexdigest()}
        assert "ratio_a_over_b" not in emitted
        assert "sign_test_p" not in emitted
        assert emitted["descriptive_window_diagnostics"]["sign_test_p"] == paired["sign_test_p"]
        for row in report["rows"]:
            source = E.load(root / ("registry/protocol/per-window/%s.json" % row["series"]))
            mean, _, _ = E.scope_mean(source["per_window"])
            assert row["scopes"]["panel25"]["mean_kld_nats"] == mean
        # Invalid input must not replace either prior publication.
        before = (out.read_bytes(), md.read_bytes())
        del paired["mean_diff"]
        paired_path.write_text(json.dumps(paired))
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0
        assert (out.read_bytes(), md.read_bytes()) == before
        # A failed rename leaves the old artifact whole and no scratch debris.
        with mock.patch.object(E.os, "replace", side_effect=OSError("rename refused")):
            try:
                E.atomic_text(out, "replacement")
            except OSError:
                pass
            else:
                raise AssertionError("rename failure was ignored")
        assert out.read_bytes() == before[0]
        assert not list(root.glob(".clean-scope-*.tmp"))
    print("clean-scope qualification, refusal, and atomic-publication regressions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
