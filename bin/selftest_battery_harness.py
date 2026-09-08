#!/usr/bin/env python3
"""Execute the real battery's dispatch and summary with tiny local commands.

These checks test exit status, retained per-rung output and visible skips. They
replace regex assertions that merely found a variable or a plausible shell
spelling. Only the production prelude/helpers and footer are loaded; none of the
network, account, model or full-suite rungs runs recursively.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BATTERY = ROOT / "bin" / "selftest_all.sh"


class BatteryDispatchTest(unittest.TestCase):
    def run_battery(self, commands):
        source = BATTERY.read_text(encoding="utf-8")
        prelude, separator, _ = source.partition('echo "== selftests (offline) =="')
        self.assertTrue(separator, "cannot isolate production battery dispatch")
        summary = source[source.rindex('\necho "selftest_all:'):]
        environment = dict(os.environ, FIDELITY_PYTHON=sys.executable)
        result = subprocess.run(
            ["bash", "-c", prelude + "\n" + commands + "\n" + summary,
             str(BATTERY)],
            cwd=str(ROOT), env=environment, text=True, capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.stderr, "", result.stderr)
        counts = re.search(
            r"selftest_all: (\d+) passed, (\d+) failed, (\d+) skipped,\s*"
            r"(\d+) internal skip", result.stdout,
        )
        self.assertIsNotNone(counts, result.stdout)
        return result, tuple(int(value) for value in counts.groups())

    def test_success_and_expected_refusal_both_run(self):
        result, counts = self.run_battery(
            't success 0 printf "success-output\\n"\n'
            't expected-refusal 3 bash -c "echo refused-output; exit 3"\n'
            'cat "$LOG_DIR"/*.log'
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(counts, (2, 0, 0, 0))
        self.assertIn("success-output", result.stdout)
        self.assertIn("refused-output", result.stdout)

    def test_unexpected_failure_remains_failure_and_next_rung_runs(self):
        result, counts = self.run_battery(
            't broken 0 bash -c "echo actual-error; exit 7"\n'
            't after-failure 0 printf "after-error\\n"\n'
            'cat "$LOG_DIR"/*.log'
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertEqual(counts, (1, 1, 0, 0))
        self.assertIn("actual-error", result.stdout)
        self.assertIn("after-error", result.stdout)

    def test_outer_skip_never_counts_as_a_pass(self):
        result, counts = self.run_battery('s unavailable "missing test fixture"')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(counts, (0, 0, 1, 0))
        self.assertIn("missing test fixture", result.stdout)

    def test_each_internal_skip_format_is_visible(self):
        notices = (
            "SKIP missing independent oracle",
            "[skip] quant_pipeline unavailable",
            "accelerator: SKIPPED (no CUDA)",
            "reference comparison SKIPPED: dependency absent",
            "SKIPPED missing fifth prerequisite",
            "SKIP missing sixth prerequisite",
            "SKIP seventh-prerequisite-must-remain-visible",
            "[SKIP] native oracle unavailable",
            "0 failure(s), 4 skip(s)",
        )
        for notice in notices:
            with self.subTest(notice=notice):
                result, counts = self.run_battery(
                    "t partial 0 printf '%s\\n' " + repr(notice)
                )
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertEqual(counts, (1, 0, 0, 1))
                self.assertIn(notice, result.stdout)
        result, counts = self.run_battery(
            "t partial 0 printf '%s\\n' " + " ".join(repr(n) for n in notices)
        )
        self.assertEqual(counts, (1, 0, 0, len(notices)))
        for notice in notices:
            self.assertIn(notice, result.stdout)

    def test_zero_skip_summary_is_not_a_skipped_test(self):
        for notice in ("11 passed, 0 failed, 0 skipped", "0 failure(s), 0 skip(s)"):
            with self.subTest(notice=notice):
                result, counts = self.run_battery(
                    "t complete 0 printf '%s\\n' " + repr(notice))
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertEqual(counts, (1, 0, 0, 0))

    def test_nonzero_skip_summary_is_not_hidden(self):
        result, counts = self.run_battery(
            't partial 0 printf "140 passed, 0 failed, 6 skipped\\n"'
        )
        self.assertEqual(counts, (1, 0, 0, 1))
        self.assertIn("6 skipped", result.stdout)

    def test_passing_labels_and_empty_json_are_not_skips(self):
        for notice in (
                '{"failed": [], "passed": true, "skipped": []}',
                "[ok] public K6 cannot skip shard binding",
                "PASS exact p (pre-fix: None, 'skipped above 2000')"):
            with self.subTest(notice=notice):
                result, counts = self.run_battery(
                    "t complete 0 printf '%s\\n' " + repr(notice))
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertEqual(counts, (1, 0, 0, 0))

    def test_json_skip_evidence_is_visible(self):
        notice = '{"passed": true, "skipped": ["native-oracle-unavailable"]}'
        result, counts = self.run_battery(
            "t partial 0 printf '%s\\n' " + repr(notice))
        self.assertEqual(counts, (1, 0, 0, 1))
        self.assertIn("native-oracle-unavailable", result.stdout)


class BundleCoverageTest(unittest.TestCase):
    def child(self, text):
        from selftest_bundle_complete import run_setup_selftest
        with tempfile.TemporaryDirectory(prefix="qfs-bundle-coverage-") as directory:
            script = Path(directory) / "selftest_child.py"
            script.write_text(text, encoding="utf-8")
            return run_setup_selftest(script)

    def test_prior_pipeline_skip_cannot_hide_a_later_failure(self):
        result = self.child(
            "print('[skip] quant_pipeline is unavailable')\n"
            "raise AssertionError('decoded tensor is wrong')\n")
        self.assertEqual(result["status"], "fail")
        self.assertNotEqual(result["returncode"], 0)

    def test_pipeline_words_cannot_hide_a_missing_fixture(self):
        result = self.child(
            "print('quant_pipeline --pipeline-root')\n"
            "open('a-required-fixture-that-does-not-exist')\n")
        self.assertEqual(result["status"], "fail")

    def test_explicit_dependency_absence_is_skip_not_pass(self):
        result = self.child(
            "raise SystemExit('pass --pipeline-root (tree containing quant_pipeline)')\n")
        self.assertEqual(result["status"], "skip")
        self.assertNotEqual(result["returncode"], 0)

    def test_success_with_nested_skips_remains_partial(self):
        result = self.child("print('SKIP native oracle unavailable')\n")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["skip_notices"]), 1)
        self.assertEqual(self.child("assert 2 + 2 == 4\n")["status"], "pass")


if __name__ == "__main__":
    unittest.main()
