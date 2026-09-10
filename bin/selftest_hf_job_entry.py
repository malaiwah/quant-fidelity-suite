#!/usr/bin/env python3
"""Offline admission and real-process handoff checks for the baked HF Jobs CLI."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_job_entry as entry
REAL_FETCH = entry.fetch_bootstrap

BOOTSTRAP = b'''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--out');a=p.parse_args()
plan=json.loads(Path(a.plan).read_text())
Path(a.out).write_text(json.dumps({'mode':plan['mode'],'plan_sha256':plan['plan_sha256'],'started':float(os.environ['QFS_WORKFLOW_STARTED']),'finished':time.time(),'injected_pythonpath':os.environ.get('PYTHONPATH')}))
'''


def sealed(plan):
    plan = copy.deepcopy(plan)
    plan["plan_sha256"] = ""
    plan["plan_sha256"] = hashlib.sha256(entry.canonical(plan)).hexdigest()
    return plan


def fixture(mode="root"):
    return sealed({"schema": "qfs.hf-workflow-plan.v1", "launch_contract": entry.CONTRACT,
        "workflow_id": "a" * 32, "owner": "fixture", "mode": mode,
        "source": {"repository": entry.SOURCE, "revision": "b" * 40,
                   "bootstrap_sha256": hashlib.sha256(BOOTSTRAP).hexdigest(),
                   "worker_sha256": "c" * 64, "environment_sha256": "d" * 64},
        "image": "ghcr.io/fixture/image@sha256:" + "e" * 64,
        "output": {"mount_path": "/outputs", "prefix": "runs/" + "a" * 32,
                   "bucket": "fixture/results", "dataset_repository": "fixture/capture"},
        "hardware": {"timeout_seconds": 60, "device": "cpu", "hourly_usd": "0.01", "max_compute_usd": "1.00"}})


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)
        self.plan = self.root / "plan.json"
        self.out = self.root / "result"
        self.document = fixture()
        self.plan.write_bytes(entry.canonical(self.document))
        self.environment = {"HOME": str(self.root), "QFS_PLAN_SHA256": self.document["plan_sha256"],
                            "QFS_WORKFLOW_ID": self.document["workflow_id"]}
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.scratch.cleanup)
        self.stack.enter_context(patch.dict(os.environ, self.environment, clear=True))
        self.stack.enter_context(patch.object(entry, "PLAN", self.plan))
        self.stack.enter_context(patch.object(entry, "OUT", self.out))
        self.stack.enter_context(patch.object(entry, "PYTHON", sys.executable))
        self.fetch = self.stack.enter_context(patch.object(entry, "fetch_bootstrap", side_effect=AssertionError("network before admission")))
        self.execute = self.stack.enter_context(patch.object(entry.os, "execve", side_effect=AssertionError("unapproved execution")))

    def launch(self, action="capture", extra=()):
        with contextlib.redirect_stderr(io.StringIO()):
            return entry.main([action, "--plan", str(self.plan), "--out", str(self.out), *extra])

    def update(self, document):
        self.document = sealed(document)
        self.plan.write_bytes(entry.canonical(self.document))
        os.environ["QFS_PLAN_SHA256"] = self.document["plan_sha256"]

    def refused_before_fetch(self, action="capture"):
        self.assertEqual(self.launch(action), 3)
        self.fetch.assert_not_called()
        self.execute.assert_not_called()

    def test_action_and_seal_refuse_before_network(self):
        self.refused_before_fetch("measure")
        self.plan.write_bytes(self.plan.read_bytes().replace(b'"root"', b'"candidate"'))
        self.refused_before_fetch("measure")

    def test_environment_identity_cannot_be_replaced_by_self_seal(self):
        os.environ["QFS_WORKFLOW_ID"] = "f" * 32
        self.refused_before_fetch()
        os.environ["QFS_WORKFLOW_ID"] = self.document["workflow_id"]
        os.environ["QFS_PLAN_SHA256"] = "0" * 64
        self.refused_before_fetch()

    def test_mutable_or_foreign_source_is_not_fetchable(self):
        for key, value in (("revision", "main"), ("repository", "https://example.org/foreign"), ("bootstrap_sha256", "x" * 64)):
            with self.subTest(key=key):
                document = fixture()
                document["source"][key] = value
                self.update(document)
                self.refused_before_fetch()

    def test_full_json_refuses_duplicates_nonfinite_and_oversize(self):
        original = self.plan.read_bytes()
        for raw in (original[:-1] + b',"nested":{"x":1,"x":2}}',
                    original[:-1] + b',"nested":[NaN]}',
                    original[:-1] + b',"nested":[1e999]}',
                    original + b"[]", b" " * (entry.MAX_JSON + 1)):
            with self.subTest(raw_bytes=len(raw)):
                self.plan.write_bytes(raw)
                self.refused_before_fetch()

    def test_credentials_deadline_and_existing_output_refuse(self):
        os.environ["HF_TOKEN"] = "fixture-not-a-real-token"
        self.refused_before_fetch()
        del os.environ["HF_TOKEN"]
        document = fixture()
        document["hardware"]["timeout_seconds"] = True
        self.update(document)
        self.refused_before_fetch()
        self.update(fixture())
        self.out.mkdir()
        self.refused_before_fetch()

    def test_missing_baked_interpreter_and_symlink_plan_refuse(self):
        with patch.object(entry, "PYTHON", str(self.root / "absent-python")):
            self.refused_before_fetch()
        renamed = self.root / "original.json"
        self.plan.rename(renamed)
        self.plan.symlink_to(renamed)
        self.refused_before_fetch()

    def test_bootstrap_hash_is_checked_before_publication(self):
        self.fetch.side_effect = None
        self.fetch.side_effect = lambda source, deadline: fetch_from_bytes(source, deadline, BOOTSTRAP + b"# altered\n")
        with patch.object(entry, "publish_bootstrap") as publish:
            self.assertEqual(self.launch(), 3)
            publish.assert_not_called()
        self.execute.assert_not_called()

    def test_fresh_atomic_bootstrap_never_adopts_existing_file(self):
        first = entry.publish_bootstrap(BOOTSTRAP)
        second = entry.publish_bootstrap(b"different")
        self.addCleanup(shutil.rmtree, first.parent)
        self.addCleanup(shutil.rmtree, second.parent)
        self.assertNotEqual(first.parent, second.parent)
        self.assertEqual(first.read_bytes(), BOOTSTRAP)
        self.assertEqual(second.read_bytes(), b"different")
        self.assertEqual(stat.S_IMODE(first.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o600)

    def test_redirect_and_download_bound_refuse(self):
        with self.assertRaises(ValueError):
            entry.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://example.org/bootstrap.py")
        with self.assertRaises(ValueError):
            fetch_from_bytes(self.document["source"], time.monotonic() + 30, b"x" * (entry.MAX_BOOTSTRAP + 1))

    def test_help_and_unknown_model_override(self):
        result = subprocess.run([sys.executable, "-I", entry.__file__, "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for action in ("capture", "measure", "compare", "selftest"):
            self.assertIn(action, result.stdout)
        with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(io.StringIO()):
            entry.build_parser().parse_args(["capture", "--plan", str(self.plan), "--out", str(self.out), "--model", "foreign/model"])
        self.assertEqual(refused.exception.code, 2)

    def test_action_matches_exactly_one_sealed_plan_role(self):
        roles = (("root", "capture"), ("candidate", "measure"), ("compare", "compare"), ("selftest", "selftest"))
        for mode, expected in roles:
            document = fixture(mode)
            environ = {"QFS_WORKFLOW_ID": document["workflow_id"], "QFS_PLAN_SHA256": document["plan_sha256"]}
            for _, action in roles:
                with self.subTest(mode=mode, action=action):
                    if action == expected:
                        self.assertEqual(entry.validate_plan(document, action, environ),
                                         document["hardware"]["timeout_seconds"])
                    else:
                        with self.assertRaises(ValueError):
                            entry.validate_plan(document, action, environ)

    def test_selftest_role_mismatch_refuses_before_network(self):
        self.update(fixture("selftest"))
        for action in ("capture", "measure", "compare"):
            with self.subTest(action=action):
                self.refused_before_fetch(action)
        for mode in ("root", "candidate", "compare"):
            with self.subTest(mode=mode):
                self.update(fixture(mode))
                self.refused_before_fetch("selftest")

    def test_selftest_keeps_fixed_paths_identity_binding_and_tokenless_launch(self):
        self.update(fixture("selftest"))

        def invoke(plan_path, out_path):
            with contextlib.redirect_stderr(io.StringIO()):
                return entry.main(["selftest", "--plan", plan_path, "--out", out_path])

        self.assertEqual(invoke(str(self.root / "copy.json"), str(self.out)), 3)
        self.assertEqual(invoke(str(self.plan), str(self.root / "elsewhere")), 3)
        for extra in (["--model", "foreign/model"], ["--token", "fixture-not-a-real-token"],
                      ["--suite", "selftest_gguf_offline"]):
            with self.subTest(extra=extra[0]), self.assertRaises(SystemExit) as refused, \
                    contextlib.redirect_stderr(io.StringIO()):
                entry.build_parser().parse_args(["selftest", "--plan", str(self.plan), "--out", str(self.out), *extra])
            self.assertEqual(refused.exception.code, 2)
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"):
            os.environ[name] = "fixture-not-a-real-token"
            with self.subTest(credential=name):
                self.refused_before_fetch("selftest")
            del os.environ[name]
        unsealed = dict(self.document, hardware=dict(self.document["hardware"], hourly_usd="0.02"))
        self.plan.write_bytes(entry.canonical(unsealed))
        self.refused_before_fetch("selftest")
        self.plan.write_bytes(entry.canonical(self.document))
        for name, value in (("QFS_PLAN_SHA256", "0" * 64), ("QFS_WORKFLOW_ID", "f" * 32)):
            original = os.environ[name]
            os.environ[name] = value
            with self.subTest(binding=name):
                self.refused_before_fetch("selftest")
            os.environ[name] = original
        # Only the sealed plan bytes, fixed mounts and tokenless environment reach the fetch.
        self.fetch.side_effect = ValueError("admission reached the pinned bootstrap fetch")
        self.assertEqual(self.launch("selftest"), 3)
        self.assertEqual(self.fetch.call_count, 1)
        self.execute.assert_not_called()

    def test_valid_actions_execute_verified_bootstrap_in_real_process(self):
        for mode, action in entry.ACTIONS.items():
            with self.subTest(mode=mode):
                self.update(fixture(mode))
                self.out.unlink(missing_ok=True)
                environment = dict(os.environ, PYTHONPATH="/unapproved-pythonpath", QFS_WORKFLOW_STARTED="1")
                before = time.time()
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--handoff", str(self.root), action],
                                        env=environment, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                observed = json.loads(self.out.read_text())
                self.assertEqual(observed["mode"], mode)
                self.assertEqual(observed["plan_sha256"], self.document["plan_sha256"])
                self.assertGreaterEqual(observed["started"], before)
                self.assertLess(observed["started"], observed["finished"] - 0.02)
                self.assertIsNone(observed["injected_pythonpath"])


def fetch_from_bytes(source, deadline, raw):
    class Opener:
        def open(self, url, timeout):
            if url != entry.RAW_SOURCE + source["revision"] + "/explorer/job_bootstrap.py":
                raise AssertionError("unexpected source URL")
            return io.BytesIO(raw)
    with patch.object(entry.urllib.request, "build_opener", return_value=Opener()):
        return REAL_FETCH(source, deadline)


def handoff(root, action):
    entry.PLAN, entry.OUT, entry.PYTHON = root / "plan.json", root / "result", sys.executable
    original_mkdtemp = tempfile.mkdtemp
    class Opener:
        def open(self, url, timeout):
            time.sleep(0.03)  # Setup time must remain inside the worker's deadline.
            return io.BytesIO(BOOTSTRAP)
    with patch.object(entry.urllib.request, "build_opener", return_value=Opener()), \
            patch.object(entry.tempfile, "mkdtemp", side_effect=lambda **kw: original_mkdtemp(dir=root, **kw)):
        return entry.main([action, "--plan", str(entry.PLAN), "--out", str(entry.OUT)])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--handoff":
        raise SystemExit(handoff(Path(sys.argv[2]), sys.argv[3]))
    unittest.main()
