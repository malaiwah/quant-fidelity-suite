#!/usr/bin/env python3
"""Offline behavioral regressions for the isolated Lambda experiment protocol."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fidelity import lambdaexperiment as experiment
from fidelity.lambdaapi import LambdaCreateRejectedError


class Clock:
    def __init__(self):
        self.now = 1800000000

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Provider:
    dry = False
    ssh_key = "/offline/ssh_identity"

    def __init__(self, clock):
        self.clock = clock
        self.rows = []
        self.volumes = []
        self.posts = 0
        self.terminated = []
        self.outcome = "success"
        self.complete = True
        self.read_error = False
        self.detail_override = False
        self.detail = None
        self.run_dir = None
        self.prepare_hook = None
        self.catalogue = {"gpu_1x_a6000": {
            "instance_type": {"name": "gpu_1x_a6000", "architecture": "x86_64",
                              "gpu_description": "NVIDIA RTX A6000 (48 GB)",
                              "price_cents_per_hour": 109,
                              "specs": {"gpus": 1, "storage_gib": 200,
                                        "vcpus": 14, "memory_gib": 100}},
            "regions_with_capacity_available": [{"name": "us-south-2"}]}}

    def _load_key(self):
        return "offline-secret-not-a-real-api-key"

    def _get_data(self, path):
        if path != "/instance-types":
            raise AssertionError("unexpected catalogue endpoint")
        return copy.deepcopy(self.catalogue)

    def chargeable_inventory(self):
        if self.read_error:
            raise OSError("opaque provider failure with private data")
        return {"complete": self.complete,
                "unknown_families": [] if self.complete else ["network_volumes"],
                "families": {
                    "instances": {"complete": True, "resources": copy.deepcopy(self.rows)},
                    "network_volumes": {"complete": self.complete, "resources": copy.deepcopy(self.volumes)}}}

    def server_time_evidence(self):
        return {"server_epoch": self.clock(), "local_received_epoch": self.clock(),
                "local_minus_server_seconds": 0}

    def prepare_safe_create(self, **kw):
        if self.prepare_hook:
            self.prepare_hook()
        return SimpleNamespace(name=kw["name"], instance_type_name=kw["gpu_type"],
                               region_name=kw["region"], gpu_count=1, storage_gib=200,
                               price_cents_per_hour=self.catalogue[kw["gpu_type"]]["instance_type"]["price_cents_per_hour"],
                               terminate_after=kw["terminate_after"], dry_run=False,
                               ssh_key_names=tuple(kw["ssh_key_names"]))

    def row(self, identifier="owned", name=None, **changes):
        row = {"id": identifier, "name": name or experiment.load_plan(self.run_dir)["name"],
               "instance_type_name": "gpu_1x_a6000", "region_name": "us-south-2",
               "status": "active", "raw": {"jupyter_token": "never-publish-token"}}
        row.update(changes)
        return row

    def submit_prepared_create(self, prepared):
        durable = experiment.load_plan(self.run_dir)
        if durable["phase"] != "POST_INTENT" or not durable["post_attempted"]:
            raise AssertionError("POST called without durable intent")
        self.posts += 1
        if self.outcome == "rejected":
            raise LambdaCreateRejectedError("refused", "instance-operations/launch/insufficient-capacity")
        if self.outcome != "lost_without_resource":
            self.rows.append(self.row(name=prepared.name))
        if self.outcome.startswith("lost"):
            raise OSError("response lost; PRIVATE_TOKEN")
        if self.outcome == "invalid":
            return {"provider_id": []}
        if self.outcome == "interrupt":
            raise KeyboardInterrupt()
        return {"provider_id": "owned"}

    def get_lifecycle_resource(self, identifier):
        if self.read_error:
            raise OSError("offline")
        if self.detail_override:
            return self.detail
        return next((copy.deepcopy(row) for row in self.rows if row["id"] == identifier), None)

    def destroy(self, identifier):
        self.terminated.append(identifier)
        self.rows = [row for row in self.rows if row["id"] != identifier]
        return {"terminated": identifier}


class ExperimentBehavior(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock = Clock()
        self.clock_patch = patch.object(experiment, "_NOW", self.clock)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.provider = Provider(self.clock)
        self.path = Path(self.temp.name) / "new-attempt"
        self.provider.run_dir = self.path

    def prepare(self):
        return experiment.prepare(self.provider, self.path, gpu_type="gpu_1x_a6000",
                                  region="us-south-2", max_cost_usd="25",
                                  max_runtime_seconds=3600, ssh_key_names=["ephemeral"], accept_risk=True)

    def heartbeat(self):
        plan = experiment.load_plan(self.path)
        experiment._atomic(self.path / "heartbeat.json", {"name": plan["name"], "heartbeat_epoch": self.clock()})

    def launch(self):
        self.prepare()
        self.heartbeat()
        return experiment.create(self.provider, self.path)

    def close_after_terminate(self):
        self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.clock.sleep(5)
        self.assertTrue(experiment.cleanup_once(self.provider, self.path))

    def test_lost_response_reconciles_owned_resource_without_second_launch(self):
        self.provider.outcome = "lost"
        self.prepare()
        self.heartbeat()
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)
        self.close_after_terminate()
        self.assertEqual(self.provider.posts, 1)
        self.assertEqual(self.provider.terminated, ["owned"])
        self.assertTrue(experiment.public_report(self.path)["cleanup"]["closed"])

    def test_lost_response_empty_inventory_never_becomes_proven_absence(self):
        self.provider.outcome = "lost_without_resource"
        self.prepare()
        self.heartbeat()
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)
        for _ in range(3):
            self.clock.sleep(20000)
            self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.assertEqual(self.provider.terminated, [])
        self.assertFalse(experiment.public_report(self.path)["cleanup"]["closed"])

    def test_foreign_baseline_resources_and_filesystems_survive(self):
        self.provider.rows = [self.provider.row("foreign", name="other-run")]
        self.provider.volumes = [{"id": "foreign-volume", "raw": {"sensitive": "private"}}]
        self.launch()
        experiment.request_cleanup(self.path, "workload_finished")
        self.close_after_terminate()
        self.assertEqual([row["id"] for row in self.provider.rows], ["foreign"])
        self.assertEqual([row["id"] for row in self.provider.volumes], ["foreign-volume"])
        self.assertEqual(self.provider.terminated, ["owned"])

    def test_same_name_baseline_is_refused_before_launch(self):
        attempt = "a" * 32
        self.provider.rows = [self.provider.row("foreign", name="qfs-native-" + attempt)]
        with patch.object(experiment.uuid, "uuid4", return_value=SimpleNamespace(hex=attempt)):
            with self.assertRaises(experiment.ExperimentError):
                self.prepare()
        self.assertEqual(self.provider.posts, 0)
        self.assertEqual(self.provider.terminated, [])

    def test_returned_baseline_id_never_authorizes_termination(self):
        self.provider.rows = [self.provider.row("owned", name="foreign-run")]
        self.prepare()
        self.heartbeat()
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)
        self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.assertEqual(self.provider.terminated, [])

    def test_multiple_or_mismatched_candidates_are_not_adopted(self):
        self.provider.outcome = "lost"
        self.prepare()
        self.heartbeat()
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)
        self.provider.rows.append(self.provider.row("conflicting"))
        self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.provider.rows = [self.provider.row("conflicting", region_name="other-region")]
        self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.assertEqual(self.provider.terminated, [])

    def test_quote_drift_refuses_before_post_and_spends_attempt(self):
        self.prepare()
        self.heartbeat()
        self.provider.catalogue["gpu_1x_a6000"]["instance_type"]["price_cents_per_hour"] = 110
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)
        self.assertEqual(self.provider.posts, 0)
        self.assertTrue(experiment.load_plan(self.path)["cleanup"]["closed"])
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)

    def test_quote_drift_inside_prepare_refuses_before_post(self):
        self.prepare()
        self.heartbeat()
        self.provider.prepare_hook = lambda: self.provider.catalogue["gpu_1x_a6000"]["instance_type"].update(price_cents_per_hour=110)
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)
        self.assertEqual(self.provider.posts, 0)

    def test_guard_required_and_stale_heartbeat_is_not_authority(self):
        self.prepare()
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)
        self.assertEqual(self.provider.posts, 0)
        self.path = Path(self.temp.name) / "different-attempt"
        self.provider.run_dir = self.path
        self.prepare()
        self.heartbeat()
        self.clock.sleep(16)
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)
        self.assertEqual(self.provider.posts, 0)

    def test_deadline_expiring_inside_preparation_prevents_launch(self):
        self.prepare()
        self.heartbeat()
        def delayed():
            self.clock.sleep(3500)
            self.heartbeat()
        self.provider.prepare_hook = delayed
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)
        self.assertEqual(self.provider.posts, 0)

    def test_cleanup_request_during_preflight_is_durable_and_prevents_post(self):
        self.prepare()
        self.heartbeat()
        self.provider.prepare_hook = lambda: experiment.request_cleanup(self.path, "interrupted")
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)
        self.assertEqual(self.provider.posts, 0)
        self.assertTrue(experiment.load_plan(self.path)["cleanup"]["requested"])
        self.assertTrue(experiment.cleanup_once(self.provider, self.path))

    def test_interrupt_after_post_preserves_cleanup_liability(self):
        self.provider.outcome = "interrupt"
        self.prepare()
        self.heartbeat()
        with self.assertRaises(KeyboardInterrupt):
            experiment.create(self.provider, self.path)
        self.assertTrue(experiment.load_plan(self.path)["cleanup"]["requested"])
        self.close_after_terminate()
        self.assertEqual(self.provider.posts, 1)

    def test_invalid_launch_response_remains_reconcilable_not_retryable(self):
        self.provider.outcome = "invalid"
        self.prepare()
        self.heartbeat()
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)
        self.assertIsNone(experiment.load_plan(self.path)["provider_id"])
        self.close_after_terminate()
        self.assertEqual(self.provider.posts, 1)
        self.assertEqual(self.provider.terminated, ["owned"])

    def test_terminate_exception_does_not_imply_absence(self):
        self.launch()
        experiment.request_cleanup(self.path, "requested")
        with patch.object(self.provider, "destroy", side_effect=OSError("response lost")):
            self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.assertFalse(experiment.load_plan(self.path)["cleanup"]["closed"])
        self.clock.sleep(30)
        self.close_after_terminate()
        self.assertEqual(self.provider.terminated, ["owned"])

    def test_incomplete_inventory_resets_absence_and_outage_stays_unresolved(self):
        self.launch()
        experiment.request_cleanup(self.path, "requested")
        self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.provider.complete = False
        self.clock.sleep(5)
        self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.provider.complete = True
        self.provider.read_error = True
        self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.provider.read_error = False
        self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.clock.sleep(5)
        self.assertTrue(experiment.cleanup_once(self.provider, self.path))

    def test_inventory_absence_cannot_override_live_exact_id(self):
        self.launch()
        self.provider.detail_override = True
        self.provider.detail = self.provider.rows[0]
        self.provider.rows = []
        experiment.request_cleanup(self.path, "requested")
        self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.clock.sleep(5)
        self.assertFalse(experiment.cleanup_once(self.provider, self.path))
        self.assertEqual(self.provider.terminated, [])

    def test_guard_enforces_absolute_deadline_without_parent(self):
        plan = self.launch()
        self.clock.now = plan["deadline_epoch"]
        with patch.object(experiment, "_PROVIDER_FACTORY", return_value=self.provider), patch.object(experiment, "_SLEEP", self.clock.sleep):
            self.assertEqual(experiment.guard(self.path), 0)
        self.assertEqual(self.provider.terminated, ["owned"])
        self.assertTrue(experiment.load_plan(self.path)["cleanup"]["closed"])
        self.assertFalse((self.path / "secret/api_key").exists())

    def test_guard_removes_snapshot_from_already_closed_run_without_provider(self):
        self.launch()
        experiment.request_cleanup(self.path, "workload_finished")
        self.close_after_terminate()
        self.assertTrue((self.path / "secret/api_key").is_file())
        with patch.object(experiment, "_PROVIDER_FACTORY",
                          side_effect=AssertionError("closed run contacted provider")):
            self.assertEqual(experiment.guard(self.path), 0)
        self.assertFalse((self.path / "secret/api_key").exists())

    def test_refused_launch_closes_without_assuming_network_error_is_refusal(self):
        self.provider.outcome = "rejected"
        self.prepare()
        self.heartbeat()
        with self.assertRaises(experiment.ExperimentError):
            experiment.create(self.provider, self.path)
        self.assertTrue(experiment.cleanup_once(self.provider, self.path))
        self.assertEqual(self.provider.terminated, [])

    def test_new_directory_and_exact_credential_authority_required(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.heartbeat()
        with patch.object(self.provider, "_load_key", return_value="different-account-credential"):
            with self.assertRaises(experiment.ExperimentError):
                experiment.create(self.provider, self.path)
        self.assertEqual(self.provider.posts, 0)

    def test_public_state_excludes_private_provider_documents_and_paths(self):
        self.launch()
        experiment.request_cleanup(self.path, "PRIVATE_TOKEN /private/controller/path")
        report = json.dumps(experiment.public_report(self.path))
        plan = (self.path / "plan.json").read_text()
        for secret in (self.provider._load_key(), self.provider.ssh_key, str(self.path), "never-publish-token", "PRIVATE_TOKEN"):
            self.assertNotIn(secret, report)
            self.assertNotIn(secret, plan)
        self.assertEqual((self.path / "secret" / "api_key").stat().st_mode & 0o777, 0o600)

    def test_guardian_cannot_advertise_readiness_without_authenticated_transport(self):
        self.prepare()
        self.provider.read_error = True
        with patch.object(experiment, "_PROVIDER_FACTORY", return_value=self.provider), \
                patch.object(experiment, "_SLEEP", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                experiment.guard(self.path)
        self.assertFalse((self.path / "heartbeat.json").exists())
        self.assertEqual(self.provider.posts, 0)

    def test_active_service_without_first_heartbeat_is_not_ready(self):
        import lambda_native_experiment as runner
        plan = self.prepare()
        def service(argv, *args, **kwargs):
            return "yes" if argv[0] == "loginctl" else "active"
        with patch.object(runner, "local", side_effect=service), \
                patch.object(runner.time, "time", self.clock), \
                patch.object(runner.time, "sleep", self.clock.sleep):
            with self.assertRaises(runner.ExperimentFailure):
                runner.start_guard(self.path, plan, self.path / "runtime")
        self.assertFalse((self.path / "evidence/guardian-readiness.json").exists())
        self.assertEqual(self.provider.posts, 0)

    def test_first_durable_heartbeat_releases_readiness_wait(self):
        import lambda_native_experiment as runner
        plan = self.prepare()
        def service(argv, *args, **kwargs):
            return "yes" if argv[0] == "loginctl" else "active"
        def publish(seconds):
            self.clock.sleep(seconds)
            self.heartbeat()
        with patch.object(runner, "local", side_effect=service), \
                patch.object(runner.time, "time", self.clock), \
                patch.object(runner.time, "sleep", publish):
            runner.start_guard(self.path, plan, self.path / "runtime")
        self.assertEqual(self.clock(), plan["created_at_epoch"] + 1)
        self.assertEqual(self.provider.posts, 0)


    def test_unhealthy_startup_refuses_promptly_with_diagnostic_evidence(self):
        import lambda_native_experiment as runner
        plan = self.launch()
        def state(status):
            return SimpleNamespace(machine_id="owned", name=plan["name"],
                                   gpu_type=plan["gpu_type"], region=plan["region"],
                                   status=status, raw={"ip": None})
        provider = SimpleNamespace(get=unittest.mock.Mock(
            side_effect=[state("booting"), state("unhealthy")]))
        with patch.object(runner.time, "time", self.clock), \
                patch.object(runner.time, "sleep", self.clock.sleep):
            with self.assertRaises(runner.ExperimentFailure):
                runner.await_host(provider, "owned", plan, self.path)
        observations = json.loads((self.path / "evidence/startup-observations.json").read_text())
        self.assertEqual([row["provider_status"] for row in observations], ["booting", "unhealthy"])
        self.assertEqual(self.clock(), plan["created_at_epoch"] + 5)

    def test_startup_identity_mismatch_never_reaches_ssh(self):
        import lambda_native_experiment as runner
        plan = self.launch()
        provider = SimpleNamespace(get=lambda _: SimpleNamespace(
            machine_id="owned", name="foreign", gpu_type=plan["gpu_type"],
            region=plan["region"], status="active", raw={"ip": "192.0.2.5"}),
            _await_ssh=unittest.mock.Mock(side_effect=AssertionError("foreign SSH attempt")))
        with patch.object(runner.time, "time", self.clock):
            with self.assertRaises(runner.ExperimentFailure):
                runner.await_host(provider, "owned", plan, self.path)
        provider._await_ssh.assert_not_called()


class RemoteGuardBehavior(unittest.TestCase):
    def setUp(self):
        import lambda_native_remote_guard as remote
        self.remote = remote
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clock = Clock()
        self.binding = {"provider_id": "owned", "name": "qfs-native-" + "a" * 32,
                        "gpu_type": "gpu_1x_a6000", "region": "us-south-2",
                        "deadline_epoch": self.clock() - 1}
        for name, data in (("binding.json", json.dumps(self.binding)), ("api_key", "fixture-only")):
            path = self.root / name
            path.write_text(data)
            path.chmod(0o600)
        self.owned = {"id": "owned", "name": self.binding["name"],
                      "instance_type": {"name": self.binding["gpu_type"]},
                      "region": {"name": self.binding["region"]}, "status": "active"}
        self.foreign = dict(self.owned, id="foreign", name="unrelated")
        self.posts = []
        self.live = True

    def request(self, method, path, body=None):
        if method == "GET":
            if path == "/instances":
                return [self.foreign] + ([self.owned] if self.live else [])
            return self.owned if self.live else None
        self.posts.append(body)
        self.live = False
        return {"terminated_instances": [self.owned]}

    def test_deadline_terminates_only_bound_instance_and_waits_for_absence(self):
        backstop = self.remote.Backstop(self.root)
        with patch.object(backstop, "request", side_effect=self.request), \
                patch.object(self.remote.time, "time", self.clock), \
                patch.object(self.remote.time, "sleep", self.clock.sleep):
            self.assertEqual(backstop.run(), 0)
        self.assertEqual(self.posts, [{"instance_ids": ["owned"]}])
        self.assertFalse((self.root / "api_key").exists())
        state = json.loads((self.root / "heartbeat.json").read_text())
        self.assertEqual(state["state"], "closed")
        self.assertEqual(state["confirmations"], 2)

    def test_wrong_exact_id_binding_never_arms_or_terminates(self):
        self.owned["name"] = "foreign-owner"
        backstop = self.remote.Backstop(self.root)
        with patch.object(backstop, "request", side_effect=self.request):
            with self.assertRaises(self.remote.Refusal):
                backstop.run()
        self.assertEqual(self.posts, [])
        self.assertFalse((self.root / "heartbeat.json").exists())

    def test_duplicate_inventory_cannot_authorize_termination(self):
        backstop = self.remote.Backstop(self.root)
        def duplicate(method, path, body=None):
            return [self.owned, self.owned] if path == "/instances" else self.request(method, path, body)
        with patch.object(backstop, "request", side_effect=duplicate):
            with self.assertRaises(self.remote.Refusal):
                backstop.run()
        self.assertEqual(self.posts, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
