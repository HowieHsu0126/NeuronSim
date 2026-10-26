"""CPU-only contract tests for the Merlin AppWorld bootstrap.

Run in a Merlin CPU allocation. HDFS and GPU sampling calls are mocked; the
small real subprocesses exercise CLI parsing, owner exit handling and neutral
receipt admission. They do not load a model or native task.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from population_dynamics.appworld_benchmark_development import (
    load_benchmark_config, validate_model_qualification,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_appworld_benchmark_merlin.py"
SPEC = importlib.util.spec_from_file_location("appworld_merlin_bootstrap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class ArtifactStore:
    """Capture immutable publication records without calling HDFS."""

    def __init__(self, work: Path, output: Path) -> None:
        self.prefix = "hdfs://fixture/checkpoints/run/appworld_gpu_development"
        self.work = work
        self.artifacts = output
        self.event_lock = threading.Lock()
        self.uploads: list[tuple[Path, str]] = []

    def publish(self, path: Path, relative: str, **kwargs: object) -> dict[str, object]:
        self.uploads.append((path, relative))
        return {"uri": self.prefix + "/" + relative, **bootstrap.file_identity(path)}


def wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for the test condition")
        time.sleep(0.005)


def uniform_attempt(**changes) -> dict:
    return {
        "agent": "collaboration_uniform", "split": "train", "eligible_index": 0,
        "seed": 41, "verification_budget": 8, "max_model_calls": 50, **changes,
    }


def official_attempt(**changes) -> dict:
    return {
        "agent": "official_full_code", "split": "train", "eligible_index": 0,
        "seed": 41, **changes,
    }


def valid_inputs() -> dict:
    native_environment = {
        "native_python": str(bootstrap.NATIVE_PREFIX / "native-env/bin/python"),
        "source_checkout": str(bootstrap.NATIVE_PREFIX / "source"),
    }
    native = {"exit_code": 0, "persisted": True, "environment": native_environment}
    data = {
        "exit_code": 0, "persisted": True,
        "environment": {
            **native_environment,
            "data_root": str(bootstrap.DATA_PREFIX / "benchmark"),
            "appworld_cache": str(bootstrap.DATA_PREFIX / "cache"),
        },
    }
    model = {
        "exit_code": 0, "persisted": True,
        "environment": {
            "python": str(bootstrap.MODEL_PREFIX / "model-env/bin/python"),
            "status": "MODEL_PACKAGES_PREPARED_GPU_VALIDATION_REQUIRED",
        },
    }
    return {
        "native_runtime": native,
        "data_admission": data,
        "model_runtime": model,
        "source_metadata": {"sha": "pinned-model-revision"},
        "model_inputs": {
            "status": "MODEL_SNAPSHOT_VERIFIED_GPU_ADMISSION_REQUIRED",
            "exit_code": 0,
            "persisted": True,
            "full_snapshot_verified": True,
            "model_revision": "pinned-model-revision",
            "runtime_receipt": copy.deepcopy(model),
            "files": {"config.json": {"bytes": 1, "sha256": "a" * 64}},
            "model_hdfs_prefix": "hdfs://fixture/model_inputs/model",
        },
    }


class TemporaryArtifacts(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.output = self.work / "artifacts"
        self.output.mkdir()
        self.store = ArtifactStore(self.work, self.output)
        self.report = {
            "run_id": "fixture-run", "trial_id": "fixture-trial",
            "status": "RUNNING", "stage": "fixture",
        }

    def make_monitor(self, publish):
        def factory(prefix, work, output, *, timeout, cancel_event):
            return SimpleNamespace(
                publish=lambda path, relative: publish(path, relative, cancel_event),
                event_lock=None,
            )

        with mock.patch.object(bootstrap, "HdfsStore", side_effect=factory):
            return bootstrap.GpuMonitor(self.store, self.output, self.report)


class GpuMonitorTests(TemporaryArtifacts):
    def test_blocked_publication_does_not_block_sampling_or_drain_queue_on_stop(self) -> None:
        entered = threading.Event()
        calls = []

        def publish(path, relative, cancel_event):
            calls.append(relative)
            entered.set()
            if not cancel_event.wait(3):
                raise AssertionError("Publication was not cancelled at shutdown")
            raise InterruptedError("test shutdown")

        monitor = self.make_monitor(publish)
        gpu = subprocess.CompletedProcess(["nvidia-smi"], 0, "fixture-gpu\n", "")
        with mock.patch.object(bootstrap, "GPU_SAMPLE_SECONDS", 0.02), \
                mock.patch.object(bootstrap.subprocess, "run", return_value=gpu):
            monitor.start()
            try:
                self.assertTrue(entered.wait(1))
                wait_until(lambda: len(monitor.snapshots) >= 4)
                self.assertEqual(len(calls), 1)
            finally:
                started = time.monotonic()
                monitor.stop()
                self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(len(calls), 1)
        self.assertFalse(monitor.sampling_thread.is_alive())
        self.assertFalse(monitor.publication_thread.is_alive())
        states = [row["publication_status"] for row in monitor.snapshots]
        self.assertEqual(states[0], "CANCELLED_AT_SHUTDOWN")
        self.assertTrue(all(state == "LOCAL_ONLY_AT_SHUTDOWN" for state in states[1:]))
        self.assertEqual(monitor.errors, [])
        for row in monitor.snapshots:
            value = json.loads((self.output / row["path"]).read_text())
            self.assertFalse(value["resumable_checkpoint"])
            self.assertFalse(value["active_task_databases_included"])
            self.assertEqual(value["gpu"]["stdout"], "fixture-gpu\n")

    def test_gpu_and_publication_errors_retain_local_diagnostics(self) -> None:
        def publish(path, relative, cancel_event):
            raise OSError("fixture transport failure")

        monitor = self.make_monitor(publish)
        gpu = subprocess.CompletedProcess(["nvidia-smi"], 1, "", "fixture GPU probe failure")
        with mock.patch.object(bootstrap, "GPU_SAMPLE_SECONDS", 0.02), \
                mock.patch.object(bootstrap.subprocess, "run", return_value=gpu):
            monitor.start()
            try:
                wait_until(lambda: any(row["kind"] == "publication" for row in monitor.errors))
            finally:
                monitor.stop()
        self.assertTrue(any(row["kind"] == "gpu_sample" for row in monitor.errors))
        failed = [row for row in monitor.snapshots if row["publication_status"] == "FAILED"]
        self.assertTrue(failed)
        self.assertIsNone(failed[0]["persisted_at"])
        self.assertTrue((self.output / failed[0]["path"]).is_file())

    def test_sampling_error_is_reported_and_shutdown_completes(self) -> None:
        monitor = self.make_monitor(lambda path, relative, stop: {})
        with mock.patch.object(bootstrap, "GPU_SAMPLE_SECONDS", 0.02), \
                mock.patch.object(monitor, "_sample", side_effect=OSError("fixture sample failure")):
            monitor.start()
            try:
                wait_until(lambda: bool(monitor.errors))
            finally:
                monitor.stop()
        self.assertEqual(monitor.errors[0]["kind"], "sampling")
        self.assertFalse(monitor.publication_thread.is_alive())

    def test_partial_startup_failure_stops_started_worker(self) -> None:
        monitor = self.make_monitor(lambda path, relative, stop: {})
        with mock.patch.object(
            monitor.sampling_thread, "start", side_effect=RuntimeError("fixture thread failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "fixture thread failure"):
                monitor.start()
        self.assertTrue(monitor.stop_event.is_set())
        self.assertFalse(monitor.publication_thread.is_alive())
        self.assertEqual(monitor.errors[0]["kind"], "startup")
        monitor.stop()

    def test_stop_before_start_is_idempotent(self) -> None:
        monitor = self.make_monitor(lambda path, relative, stop: {})
        monitor.stop()
        monitor.stop()
        with self.assertRaisesRegex(RuntimeError, "ALREADY_STARTED_OR_STOPPED"):
            monitor.start()

    def test_diagnostic_hdfs_cancellation_reaps_current_client(self) -> None:
        stop = threading.Event()
        store = bootstrap.HdfsStore(
            self.store.prefix, self.work, self.output, timeout=60, cancel_event=stop,
        )
        process = mock.Mock(pid=123, returncode=None)
        process.poll.return_value = None
        process.wait.return_value = -int(signal.SIGINT)

        def communicate(timeout):
            stop.set()
            raise subprocess.TimeoutExpired("fixture hdfs", timeout)

        process.communicate.side_effect = communicate
        with mock.patch.object(bootstrap.subprocess, "Popen", return_value=process), \
                mock.patch.object(bootstrap.os, "killpg") as kill, \
                mock.patch.object(store, "record") as record:
            with self.assertRaisesRegex(InterruptedError, "PUBLICATION_STOPPED"):
                store.hdfs("-put", "fixture-local", "hdfs://fixture/remote")
        kill.assert_called_once_with(123, signal.SIGINT)
        cleanup = record.call_args.args[0]["cleanup"]
        self.assertTrue(cleanup["owner_terminated"])
        self.assertTrue(cleanup["reaped"])


class ChildOwnerTests(TemporaryArtifacts):
    def test_already_exited_owner_is_reaped_without_signal(self) -> None:
        process = mock.Mock(pid=123)
        process.poll.return_value = 0
        process.wait.return_value = 0
        with mock.patch.object(bootstrap.os, "killpg") as kill:
            cleanup = bootstrap.stop_child(process)
        kill.assert_not_called()
        process.wait.assert_called_once_with(timeout=0)
        self.assertTrue(cleanup["owner_terminated"])
        self.assertTrue(cleanup["reaped"])
        self.assertEqual(cleanup["returncode"], 0)

    def test_exit_between_poll_and_signal_is_reaped(self) -> None:
        process = mock.Mock(pid=123)
        process.poll.return_value = None
        process.wait.return_value = 0
        with mock.patch.object(bootstrap.os, "killpg", side_effect=ProcessLookupError):
            cleanup = bootstrap.stop_child(process)
        self.assertTrue(cleanup["owner_terminated"])
        self.assertTrue(cleanup["reaped"])
        self.assertEqual(cleanup["signals"][0]["status"], "PROCESS_GROUP_ALREADY_EXITED")

    def test_cleanup_escalates_and_confirms_killed_owner(self) -> None:
        process = mock.Mock(pid=123)
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired("fixture", 30),
            subprocess.TimeoutExpired("fixture", 10),
            -int(signal.SIGKILL),
        ]
        with mock.patch.object(bootstrap.os, "killpg") as kill:
            cleanup = bootstrap.stop_child(process)
        self.assertEqual(
            [call.args[1] for call in kill.call_args_list],
            [signal.SIGINT, signal.SIGTERM, signal.SIGKILL],
        )
        self.assertTrue(cleanup["owner_terminated"])
        self.assertTrue(cleanup["reaped"])

    def test_cleanup_reports_unconfirmed_owner(self) -> None:
        process = mock.Mock(pid=123)
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("fixture", 10)
        with mock.patch.object(bootstrap.os, "killpg"):
            cleanup = bootstrap.stop_child(process)
        self.assertFalse(cleanup["owner_terminated"])
        self.assertFalse(cleanup["reaped"])
        self.assertIsNone(cleanup["returncode"])

    def test_failed_child_is_reaped_and_nonzero_exit_is_preserved(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CHILD_STAGE_FAILED"):
            bootstrap.run_command(
                "fixture_failure", [sys.executable, "-c", "raise SystemExit(3)"],
                dict(os.environ), self.work, self.output, self.store, self.report, [],
            )
        record = self.report["stages"]["fixture_failure"]
        self.assertEqual(record["exit_code"], 3)
        self.assertTrue(record["outputs_closed"])
        self.assertTrue(record["cleanup"]["reaped"])
        self.assertTrue(record["cleanup"]["owner_terminated"])

    def test_successful_child_publishes_closed_output(self) -> None:
        target = self.output / "fixture_result.json"
        bootstrap.run_command(
            "fixture_success",
            [sys.executable, "-c",
             "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{}\\n')", str(target)],
            dict(os.environ), self.work, self.output, self.store, self.report, [target],
        )
        record = self.report["stages"]["fixture_success"]
        self.assertEqual(record["exit_code"], 0)
        self.assertTrue(record["outputs_closed"])
        self.assertIn("stages/fixture_result.json", [row[1] for row in self.store.uploads])

    def test_unconfirmed_cleanup_publishes_only_stage_receipt(self) -> None:
        process = mock.Mock(pid=123, returncode=None)
        process.wait.side_effect = KeyboardInterrupt
        cleanup = {
            "scope": "owner_process", "pid": 123, "signals": [],
            "owner_terminated": False, "reaped": False, "returncode": None,
        }
        payload = self.output / "development"
        payload.mkdir()
        with mock.patch.object(bootstrap.subprocess, "Popen", return_value=process), \
                mock.patch.object(bootstrap, "stop_child", return_value=cleanup), \
                mock.patch.object(bootstrap, "publish_closed", return_value={}) as publish:
            with self.assertRaisesRegex(RuntimeError, "OWNER_TERMINATION_UNCONFIRMED"):
                bootstrap.run_command(
                    "fixture_interrupted", ["fixture-command"], {}, self.work,
                    self.output, self.store, self.report, [payload],
                )
        paths = publish.call_args.args[2]
        self.assertEqual(paths, [self.output / "stage_receipts/fixture_interrupted.json"])
        self.assertFalse(self.report["stages"]["fixture_interrupted"]["outputs_closed"])
        self.assertEqual(
            self.report["unclosed_output_paths"], ["fixture_interrupted.log", "development"],
        )


class AdmissionTests(TemporaryArtifacts):
    def test_complete_admission_format_is_accepted(self) -> None:
        # File authenticity is subsequently checked by the existing snapshot verifier.
        bootstrap.validate_inputs(valid_inputs())

    def test_invalid_or_incomplete_model_input_receipts_are_rejected(self) -> None:
        changes = [
            {"status": "RUNNING"},
            {"exit_code": 1},
            {"persisted": False},
            {"full_snapshot_verified": False},
            {"model_revision": "different-revision"},
            {"runtime_receipt": {}},
            {"files": {}},
            {"model_hdfs_prefix": "/tmp/unadmitted-model"},
        ]
        for change in changes:
            with self.subTest(change=change):
                inputs = valid_inputs()
                inputs["model_inputs"].update(change)
                with self.assertRaises(ValueError):
                    bootstrap.validate_inputs(inputs)

    def test_incomplete_runtime_and_changed_prefix_are_rejected(self) -> None:
        inputs = valid_inputs()
        inputs["native_runtime"]["persisted"] = False
        with self.assertRaisesRegex(ValueError, "COMPLETED_PERSISTED_INPUT_REQUIRED"):
            bootstrap.validate_inputs(inputs)
        inputs = valid_inputs()
        inputs["model_runtime"]["environment"]["python"] = "/tmp/other-env/bin/python"
        with self.assertRaisesRegex(ValueError, "ORIGINAL_RUNTIME_PREFIX_OR_STATUS_DIFFERS"):
            bootstrap.validate_inputs(inputs)

    def test_complete_file_uri_mapping_without_common_prefix_is_accepted(self) -> None:
        inputs = valid_inputs()
        intake = inputs["model_inputs"]
        intake["files"]["model.safetensors"] = {"bytes": 2, "sha256": "b" * 64}
        identities = copy.deepcopy(intake["files"])
        del intake["model_hdfs_prefix"]
        locations = {
            "config.json": "hdfs://fixture/old-run/model/config.json",
            "model.safetensors": "hdfs://fixture/new-run/model/model.safetensors",
        }
        intake["model_file_uris"] = locations
        bootstrap.validate_inputs(inputs)
        self.assertEqual(bootstrap.model_download_uris(intake), locations)
        self.assertEqual(intake["files"], identities)

    def test_complete_mapping_takes_precedence_over_legacy_prefix(self) -> None:
        intake = valid_inputs()["model_inputs"]
        intake["model_file_uris"] = {
            "config.json": "hdfs://fixture/verified-source/config.json",
        }
        self.assertEqual(
            bootstrap.model_download_uris(intake)["config.json"],
            "hdfs://fixture/verified-source/config.json",
        )

    def test_malformed_or_partial_mapping_is_rejected_even_with_prefix(self) -> None:
        malformed = [
            None,
            [],
            {},
            {"config.json": "hdfs://fixture/config.json"},
            {"config.json": "hdfs://fixture/config.json",
             "model.safetensors": "hdfs://fixture/model.safetensors",
             "extra": "hdfs://fixture/extra"},
            {"config.json": None, "model.safetensors": "hdfs://fixture/model.safetensors"},
            {"config.json": 17, "model.safetensors": "hdfs://fixture/model.safetensors"},
            {"config.json": "/tmp/config.json", "model.safetensors": "hdfs://fixture/model.safetensors"},
            {"config.json": "s3://fixture/config.json", "model.safetensors": "hdfs://fixture/model.safetensors"},
        ]
        for locations in malformed:
            with self.subTest(locations=locations):
                inputs = valid_inputs()
                inputs["model_inputs"]["files"]["model.safetensors"] = {
                    "bytes": 2, "sha256": "b" * 64,
                }
                inputs["model_inputs"]["model_file_uris"] = locations
                with self.assertRaisesRegex(ValueError, "COMPLETE_HDFS_MODEL_FILE_URIS_REQUIRED"):
                    bootstrap.validate_inputs(inputs)

    def test_prefix_only_receipt_retains_legacy_download_locations(self) -> None:
        inputs = valid_inputs()
        bootstrap.validate_inputs(inputs)
        intake = inputs["model_inputs"]
        self.assertEqual(
            bootstrap.model_download_uris(intake),
            {"config.json": "hdfs://fixture/model_inputs/model/config.json"},
        )

    def test_scratch_admission_preserves_eighty_gib_boundary(self) -> None:
        minimum = 80 * 1024**3
        with mock.patch.object(
            bootstrap.shutil, "disk_usage",
            return_value=SimpleNamespace(free=minimum, total=512 * 1024**3),
        ):
            admission = bootstrap.require_scratch_space(self.work)
        self.assertEqual(admission["required_free_bytes"], minimum)
        self.assertEqual(admission["status"], "PASS")
        with mock.patch.object(
            bootstrap.shutil, "disk_usage",
            return_value=SimpleNamespace(free=minimum - 1, total=512 * 1024**3),
        ):
            with self.assertRaisesRegex(ValueError, "INSUFFICIENT_SCRATCH_SPACE"):
                bootstrap.require_scratch_space(self.work)


class DevelopmentAttemptTests(TemporaryArtifacts):
    def test_default_react_declaration_remains_unchanged(self) -> None:
        self.assertEqual(bootstrap.resolve_development_attempt({}), {
            "agent": "official_react_code", "split": "train", "eligible_index": 0,
            "seed": 41, "method": "official_react_code", "neuronsim_method": False,
        })

    def test_explicit_official_declarations_preserve_methods_splits_and_seeds(self) -> None:
        for agent in ("official_react_code", "official_full_code", "official_function_calling"):
            for split in ("train", "dev"):
                for seed in (41, 42, 43):
                    with self.subTest(agent=agent, split=split, seed=seed):
                        declared = official_attempt(agent=agent, split=split, seed=seed, eligible_index=7)
                        inputs = {**valid_inputs(), "development_attempt": declared}
                        original = copy.deepcopy(inputs)
                        bootstrap.validate_inputs(inputs)
                        resolved = bootstrap.resolve_development_attempt(inputs)
                        self.assertEqual(inputs, original)
                        self.assertEqual(resolved, {**declared, "method": agent, "neuronsim_method": False})

    def test_official_declarations_reject_missing_extra_or_invalid_fields(self) -> None:
        for agent in ("official_react_code", "official_full_code", "official_function_calling"):
            valid = official_attempt(agent=agent)
            malformed = [{key: value for key, value in valid.items() if key != missing}
                         for missing in valid]
            malformed += [{**valid, key: value} for key in
                          ("verification_budget", "max_model_calls", "planner_calls")
                          for value in (None, 1)]
            malformed.append({**valid, "extra": "unknown"})
            for key, values in {
                "agent": ("official_unknown", "collaboration_fixed", "neuronsim", None, []),
                "split": ("test_normal", "test_challenge", "", None, []),
                "seed": (40, 44, True, 41.0, "41"),
                "eligible_index": (-1, False, 0.0, "0"),
            }.items():
                malformed += [{**valid, key: value} for value in values]
            for declared in malformed:
                with self.subTest(agent=agent, declared=declared):
                    with self.assertRaisesRegex(ValueError, "DEVELOPMENT_ATTEMPT"):
                        bootstrap.validate_inputs({**valid_inputs(), "development_attempt": declared})

    def test_explicit_uniform_declaration_preserves_input_and_accepts_budget_boundaries(self) -> None:
        for declared in (uniform_attempt(), uniform_attempt(seed=43, eligible_index=2,
                                                           verification_budget=0, max_model_calls=1)):
            with self.subTest(declared=declared):
                inputs = valid_inputs()
                inputs["development_attempt"] = declared
                original = copy.deepcopy(inputs)
                bootstrap.validate_inputs(inputs)
                resolved = bootstrap.resolve_development_attempt(inputs)
                self.assertEqual(inputs, original)
                self.assertEqual({key: resolved[key] for key in declared}, declared)
                self.assertFalse(resolved["neuronsim_method"])

    def test_malformed_declarations_reject_instead_of_falling_back_to_react(self) -> None:
        valid = uniform_attempt()
        malformed = [None, {}, [], "collaboration_uniform"]
        malformed += [{key: value for key, value in valid.items() if key != missing}
                      for missing in valid]
        malformed += [uniform_attempt(planner_calls=None), uniform_attempt(extra="unknown")]
        for key, values in {
            "agent": ("official_react_code", "collaboration_fixed", "neuronsim"),
            "split": ("dev", "test_normal", "test_challenge"),
            "seed": (40, 44, True, 41.0, "41"),
            "eligible_index": (-1, False, 0.0, "0"),
            "verification_budget": (-1, True, 8.0, "8"),
            "max_model_calls": (0, -1, True, 50.0, "50"),
        }.items():
            malformed += [uniform_attempt(**{key: value}) for value in values]
        for declared in malformed:
            with self.subTest(declared=declared):
                inputs = valid_inputs()
                inputs["development_attempt"] = declared
                with self.assertRaisesRegex(ValueError, "DEVELOPMENT_ATTEMPT"):
                    bootstrap.validate_inputs(inputs)

    def test_selection_fragment_uses_declared_index_and_seed_with_native_demo_exclusion(self) -> None:
        config = {
            "schema": "appworld_benchmark_v1",
            "development_execution": {
                "api_predictor": {"demo_task_ids": ["traincase_2"]},
                "methods": {
                    "official_react_code": {},
                    "official_full_code": {"demo_task_ids": ["devcase_2"]},
                    "official_function_calling": {},
                },
            },
        }
        manifest = {
            "schema": "appworld_split_membership_v1",
            "appworld_version": "synthetic-fixture",
            "data_root": str(self.work),
            "splits": {
                split: {
                    "task_ids": [f"{split}case_{variant}" for variant in (1, 2, 3)],
                    "task_count": 3, "scenario_count": 1,
                    "membership_file": str(self.work / f"{split}.txt"),
                    "membership_sha256": "a" * 64,
                }
                for split in ("train", "dev")
            },
        }
        bootstrap.write_json(self.output / "bound_config.yaml", config)
        bootstrap.write_json(self.work / "split_manifest.json", manifest)
        for split, seed in (("train", 41), ("dev", 43)):
            with self.subTest(split=split, seed=seed):
                result = subprocess.run(
                    [sys.executable, "-c", bootstrap.SELECT_TASK, str(self.work), split, "1", str(seed)],
                    env=dict(os.environ, PYTHONPATH=str(SCRIPT.parents[1]), CUDA_VISIBLE_DEVICES=""),
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                selection = json.loads((self.output / "selection.json").read_text())
                self.assertEqual(selection["task_id"], f"{split}case_3")
                self.assertEqual(selection["eligible_index"], 1)
                self.assertEqual(selection["native_membership_index"], 2)
                self.assertEqual(selection["excluded_demo_task_ids"], [f"{split}case_2"])
                self.assertEqual(selection["split"], split)
                self.assertEqual(selection["seed"], seed)
                self.assertFalse(selection["task_content_read"])
                self.assertFalse(selection["scores_read"])
                self.assertEqual((self.output / "selected_task_id.txt").read_text(), f"{split}case_3\n")


class HdfsPublicationRetryTests(TemporaryArtifacts):
    def test_eof_read_failure_is_not_treated_as_missing(self) -> None:
        store = bootstrap.HdfsStore(self.store.prefix, self.work, self.output)
        failed = subprocess.CompletedProcess(
            ["hdfs", "dfs", "-stat"], 1, "",
            "java.io.EOFException while reading: FileNotFoundException",
        )
        with mock.patch.object(store, "hdfs", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "HDFS_STATE_UNRESOLVED"):
                store.remote_size("hdfs://fixture/payload")
        missing = subprocess.CompletedProcess(
            ["hdfs", "dfs", "-stat"], 1, "", "FileNotFoundException: No such file",
        )
        with mock.patch.object(store, "hdfs", return_value=missing):
            self.assertIsNone(store.remote_size("hdfs://fixture/payload"))

    def test_eof_datanode_write_failure_retries_after_reconciliation(self) -> None:
        payload = self.work / "payload.json"
        payload.write_text("{}\n")
        store = bootstrap.HdfsStore(self.store.prefix, self.work, self.output)
        puts = []

        def hdfs(*arguments, check=True):
            if arguments[0] == "-mkdir":
                return subprocess.CompletedProcess(arguments, 0, "", "")
            self.assertEqual(arguments[0], "-put")
            puts.append(arguments)
            if len(puts) == 1:
                return subprocess.CompletedProcess(
                    arguments, 1, "", "datanode write failed: java.io.EOFException",
                )
            return subprocess.CompletedProcess(arguments, 0, "", "")

        with mock.patch.object(store, "hdfs", side_effect=hdfs), \
                mock.patch.object(
                    store, "remote_size", side_effect=[None, None, None, payload.stat().st_size],
                ), \
                mock.patch.object(store, "retry_pause") as pause:
            record = store.publish(payload, "payload.json")
        self.assertEqual(len(puts), 2)
        pause.assert_called_once_with(5)
        self.assertEqual(record["sha256"], bootstrap.file_identity(payload)["sha256"])

    def test_denied_or_unrelated_eof_does_not_retry(self) -> None:
        payload = self.work / "payload.json"
        payload.write_text("{}\n")
        for message in (
            "Permission denied: datanode write failed: java.io.EOFException",
            "authentication failure: datanode java.io.EOFException",
            "java.io.EOFException",
        ):
            with self.subTest(message=message):
                store = bootstrap.HdfsStore(self.store.prefix, self.work, self.output)
                puts = []

                def hdfs(*arguments, check=True):
                    if arguments[0] == "-mkdir":
                        return subprocess.CompletedProcess(arguments, 0, "", "")
                    self.assertEqual(arguments[0], "-put")
                    puts.append(arguments)
                    return subprocess.CompletedProcess(arguments, 1, "", message)

                with mock.patch.object(store, "hdfs", side_effect=hdfs), \
                        mock.patch.object(store, "remote_size", side_effect=[None, None]), \
                        mock.patch.object(store, "retry_pause") as pause:
                    with self.assertRaisesRegex(RuntimeError, "NON_RETRYABLE_HDFS_PUBLICATION"):
                        store.publish(payload, "payload.json")
                self.assertEqual(len(puts), 1)
                pause.assert_not_called()

    def test_eof_write_failure_does_not_override_identity_mismatch(self) -> None:
        payload = self.work / "payload.json"
        payload.write_text("{}\n")
        size = payload.stat().st_size
        store = bootstrap.HdfsStore(self.store.prefix, self.work, self.output)
        puts = []

        def hdfs(*arguments, check=True):
            if arguments[0] == "-mkdir":
                return subprocess.CompletedProcess(arguments, 0, "", "")
            self.assertEqual(arguments[0], "-put")
            puts.append(arguments)
            return subprocess.CompletedProcess(
                arguments, 1, "", "datanode write failure: java.io.EOFException",
            )

        with mock.patch.object(store, "hdfs", side_effect=hdfs), \
                mock.patch.object(store, "remote_size", side_effect=[None, size]), \
                mock.patch.object(store, "remote_identity", return_value={
                    "bytes": size, "sha256": "0" * 64,
                }), \
                mock.patch.object(store, "retry_pause") as pause:
            with self.assertRaisesRegex(ValueError, "PUBLISHED_IDENTITY_DIFFERS_AFTER_FAILURE"):
                store.publish(payload, "payload.json")
        self.assertEqual(len(puts), 1)
        pause.assert_not_called()


class FinalPublicationTests(TemporaryArtifacts):
    def create_decisive_files(self) -> None:
        for name in bootstrap.DECISIVE_ARTIFACTS:
            bootstrap.write_json(
                self.output / name,
                {"fixture": True, "task_goal_completed": False},
            )

    def test_decisive_artifacts_and_bundle_are_bound_before_terminal_receipt(self) -> None:
        self.create_decisive_files()
        (self.work / "native_runtime.tar.gz").write_bytes(b"input archive")
        (self.work / "model").mkdir()
        (self.work / "model/weights").write_bytes(b"input weights")
        bootstrap.publish_final(self.store, self.work, self.output, self.report)
        self.assertEqual(self.store.uploads[-1][1], "receipt.json")
        receipt = json.loads((self.work / "receipt.json").read_text())
        self.assertTrue(receipt["persisted"])
        self.assertEqual(set(receipt["decisive_artifacts"]), set(bootstrap.DECISIVE_ARTIFACTS))
        for name, row in receipt["decisive_artifacts"].items():
            self.assertEqual(row["sha256"], bootstrap.file_identity(self.output / name)["sha256"])
            self.assertEqual(row["uri"], self.store.prefix + "/stages/" + name)
        with tarfile.open(self.work / "outputs.tar.gz") as archive:
            names = archive.getnames()
            self.assertNotIn("native_runtime.tar.gz", names)
            self.assertFalse(any(name.startswith("model/") for name in names))
            manifest_file = archive.extractfile("output_manifest.json")
            assert manifest_file is not None
            manifest = json.load(manifest_file)
            for name, row in manifest["files"].items():
                member = archive.extractfile("artifacts/" + name)
                assert member is not None
                content = member.read()
                self.assertEqual(len(content), row["bytes"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), row["sha256"])
            score_file = archive.extractfile(
                "artifacts/development/attempt/development_score.json",
            )
            assert score_file is not None
            self.assertFalse(json.load(score_file)["task_goal_completed"])

    def test_outputs_of_unconfirmed_owner_are_excluded(self) -> None:
        self.create_decisive_files()
        (self.output / "development_pipeline.log").write_text("still mutable")
        self.report["unclosed_output_paths"] = ["development", "development_pipeline.log"]
        bootstrap.publish_final(self.store, self.work, self.output, self.report)
        self.assertEqual(
            set(self.report["decisive_artifacts"]), {"qualification/qualification.json"},
        )
        manifest = json.loads((self.work / "output_manifest.json").read_text())
        self.assertEqual(
            manifest["unclosed_outputs_excluded"], ["development", "development_pipeline.log"],
        )
        self.assertFalse(any(
            name.startswith("development") for name in manifest["files"]
        ))
        with tarfile.open(self.work / "outputs.tar.gz") as archive:
            self.assertFalse(any(
                name.startswith("artifacts/development") for name in archive.getnames()
            ))


class InputDataLinkPublicationTests(TemporaryArtifacts):
    def setUp(self) -> None:
        super().setUp()
        self.data_root = self.work / "admitted_native"
        self.data_directory = self.data_root / "data"
        self.data_directory.mkdir(parents=True)
        (self.data_directory / "input_only.txt").write_text("Synthetic input; never an output.")
        self.attempt = self.output / "development/attempt"
        self.attempt.mkdir(parents=True)
        self.data_link = self.attempt / "data"
        self.data_link.symlink_to(self.data_directory, target_is_directory=True)
        self.data_archive = {
            "uri": "hdfs://fixture/admitted/data.tar.gz", "sha256": "b" * 64,
        }
        bootstrap.write_json(self.work / "native_preparation.receipt.json", {
            "status": "PASS_DATA_VERSION_AND_NATIVE_SPLIT_ADMISSION",
            "exit_code": 0, "persisted": True,
            "data_root": str(self.data_root),
            "data_hdfs_uri": self.data_archive["uri"],
            "data_bundle_sha256": self.data_archive["sha256"],
            "environment": {
                "status": "PREPARED_RUNTIME_AND_SPLIT_METADATA",
                "data_root": str(self.data_root),
            },
        })
        bootstrap.write_json(self.output / "input_admission.json", {
            "status": "RESTORED_INPUT_IDENTITIES_VERIFIED",
            "data_archive": self.data_archive,
        })
        bootstrap.write_json(self.attempt / "execution_receipt.json", {
            "schema": "appworld_development_execution_v1",
            "run_root": str(self.attempt), "data_root": str(self.data_root),
            "native_data_directory": str(self.data_directory),
            "status": "agent_failure", "terminal": True, "formal_eligible": False,
            "scoring_status": "not_run", "native_tgc": None,
        })
        bootstrap.write_json(self.output / "development/model_server/transport_records.json", {
            "fixture": True, "records": [],
        })

    def forbid_input_traversal(self):
        original = bootstrap.os.scandir

        def guarded(path):
            candidate = Path(path)
            if (candidate.is_relative_to(self.data_directory)
                    or candidate.is_relative_to(self.data_link)):
                raise AssertionError("Input data link or target was traversed")
            return original(path)

        return mock.patch.object(bootstrap.os, "scandir", side_effect=guarded)

    def test_closed_publication_skips_only_the_admitted_input_link(self) -> None:
        with self.forbid_input_traversal():
            records = bootstrap.publish_closed(
                self.store, self.output, [self.output / "development"],
            )
            direct = bootstrap.publish_closed(self.store, self.output, [self.data_link])
        self.assertEqual(direct, {})
        self.assertIn("development/attempt/execution_receipt.json", records)
        self.assertIn("development/model_server/transport_records.json", records)
        self.assertFalse(any(
            name == "development/attempt/data" or name.startswith("development/attempt/data/")
            for name in records
        ))
        self.assertFalse(any(path.is_relative_to(self.data_directory) for path, _ in self.store.uploads))

    def test_failed_unscored_attempt_keeps_terminal_receipt_and_excludes_input_data(self) -> None:
        self.report.update(status="FAILED", exit_code=1, error="fixture native execution failure")
        bootstrap.write_json(self.output / "qualification/qualification.json", {
            "fixture": True, "status": "PASS",
        })
        bootstrap.write_json(self.output / "development/pipeline.json", {
            "status": "INCOMPLETE_DEVELOPMENT_ATTEMPT",
        })
        bootstrap.write_json(self.attempt / "development_score.json", {
            "scoring_status": "not_run", "native_task_result": None,
        })
        with self.forbid_input_traversal():
            bootstrap.publish_final(self.store, self.work, self.output, self.report)
        self.assertEqual(self.store.uploads[-1][1], "receipt.json")
        receipt = json.loads((self.work / "receipt.json").read_text())
        self.assertTrue(receipt["persisted"])
        self.assertEqual(receipt["status"], "FAILED")
        self.assertEqual(receipt["exit_code"], 1)
        self.assertEqual(set(receipt["decisive_artifacts"]), set(bootstrap.DECISIVE_ARTIFACTS))
        manifest = json.loads((self.work / "output_manifest.json").read_text())
        self.assertEqual(manifest["input_links_excluded"], {
            "development/attempt/data": {
                "target": str(self.data_directory), "data_archive": self.data_archive,
            },
        })
        with tarfile.open(self.work / "outputs.tar.gz") as archive:
            self.assertFalse(any(member.issym() for member in archive.getmembers()))
            self.assertFalse(any(
                name == "artifacts/development/attempt/data"
                or name.startswith("artifacts/development/attempt/data/")
                for name in archive.getnames()
            ))
            member = archive.extractfile("artifacts/development/attempt/execution_receipt.json")
            assert member is not None
            execution = json.load(member)
            self.assertEqual(execution["status"], "agent_failure")
            self.assertIsNone(execution["native_tgc"])
            self.assertEqual(execution["scoring_status"], "not_run")

    def test_known_link_with_wrong_target_is_rejected(self) -> None:
        self.data_link.unlink()
        self.data_link.symlink_to(self.work / "different_data", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "TARGET_OR_OWNER_DIFFERS"):
            bootstrap.publish_closed(self.store, self.output, [self.data_link])
        with self.assertRaisesRegex(ValueError, "TARGET_OR_OWNER_DIFFERS"):
            bootstrap.publish_final(self.store, self.work, self.output, self.report)

    def test_missing_or_mismatched_admission_cannot_exempt_the_link(self) -> None:
        cases = (
            (self.work / "native_preparation.receipt.json", "persisted", False),
            (self.output / "input_admission.json", "data_archive",
             {"uri": self.data_archive["uri"], "sha256": "0" * 64}),
            (self.attempt / "execution_receipt.json", "native_data_directory",
             str(self.work / "different_data")),
            (self.attempt / "execution_receipt.json", "run_root", str(self.work / "different_attempt")),
        )
        for path, key, value in cases:
            with self.subTest(path=path.name, field=key):
                original = path.read_text()
                changed = json.loads(original)
                changed[key] = value
                bootstrap.write_json(path, changed)
                try:
                    with self.assertRaises(ValueError):
                        bootstrap.publish_closed(self.store, self.output, [self.data_link])
                finally:
                    path.write_text(original)
        path = self.output / "input_admission.json"
        original = path.read_text()
        path.unlink()
        try:
            with self.assertRaisesRegex(ValueError, "ADMISSION_METADATA_REQUIRED"):
                bootstrap.publish_closed(self.store, self.output, [self.data_link])
        finally:
            path.write_text(original)

    def test_other_output_symlinks_remain_rejected(self) -> None:
        unknown = self.output / "unexpected_data"
        unknown.symlink_to(self.data_directory, target_is_directory=True)
        with self.forbid_input_traversal():
            with self.assertRaisesRegex(ValueError, "OUTPUT_SYMLINK_CANNOT_BE_PUBLISHED"):
                bootstrap.publish_closed(self.store, self.output, [unknown])
            with self.assertRaisesRegex(ValueError, "OUTPUT_BUNDLE_CONTAINS_SYMLINK"):
                bootstrap.publish_final(self.store, self.work, self.output, self.report)


class QualificationReuseTests(TemporaryArtifacts):
    def setUp(self) -> None:
        super().setUp()
        self.case_index = 0
        self.model = {
            "repo_id": "fixture-model", "revision": "a" * 40,
            "serving_backend": "vllm", "serving_version": "0.11.2",
            "tool_call_parser": "qwen3_coder",
        }
        self.config = {
            "schema": "appworld_benchmark_v1", "seeds": [41, 42, 43],
            "model": self.model,
            "model_qualification": {"tensor_parallel_size": 2, "dtype": "bfloat16"},
        }
        self.qualification = {
            "schema": "appworld_model_runtime_qualification_v1",
            "scope": "neutral_model_transport_only", "status": "PASS",
            "transport_probe_status": "PASS", "tool_calls_validated": True,
            "seeds": [41, 42, 43], "model_identity": self.model,
            "config": self.config, "runtime": {"packages": {"vllm": "0.11.2"}},
            "server_cleanup": {
                "server_parent_reaped": True, "forced_kill": False,
                "process_group_term_sent": True, "returncode_before_termination": None,
            },
            "run_id": "original-run", "trial_id": "original-trial",
            "hostname": "original-host",
            "checkpoint_path": "hdfs://fixture/checkpoints/original-run",
            "model_directory": "/tmp/original-run/model",
            "runtime_receipt": "/tmp/original-run/model_package_environment.json",
            "source_metadata": "/tmp/original-run/source_metadata.json",
            "snapshot_config": "/tmp/original-run/local_actor.yaml",
        }
        self.source_uri = (
            "hdfs://fixture/checkpoints/original-run/appworld_gpu_development/"
            "stages/qualification/qualification.json"
        )
        self.set_payload(self.qualification)

    def set_payload(self, qualification) -> None:
        self.payload = (json.dumps(qualification, indent=3) + "\n").encode()
        self.reference = {
            "uri": self.source_uri, "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }

    def run_case(self, reference, development_attempt=None):
        self.case_index += 1
        work = self.work / f"case_{self.case_index}"
        output = work / "artifacts"
        output.mkdir(parents=True)
        bootstrap.write_json(output / "bound_config.yaml", self.config)
        store = ArtifactStore(work, output)
        commands = []
        report = {"status": "RUNNING", "run_id": "new-run", "trial_id": "new-trial"}
        case = SimpleNamespace(work=work, output=output, store=store, commands=commands, report=report)
        self.last_case = case

        def download(*arguments, check=True):
            if arguments[:2] != ("-get", self.source_uri):
                raise RuntimeError("fixture source URI mismatch")
            Path(arguments[2]).write_bytes(self.payload)
            return subprocess.CompletedProcess(arguments, 0, "", "")

        store.hdfs = mock.Mock(side_effect=download)

        def run(stage, command, environment, work, output, store, report, closed_outputs):
            commands.append((stage, command))
            if stage == "native_binding":
                bootstrap.write_json(output / "bound_native_environment.json", {
                    "status": "PREPARED_RUNTIME_AND_SPLIT_METADATA",
                    "appworld_cache": str(work / "cache"),
                })
            elif stage == "qualification_reuse_admission":
                self.assertEqual(command[2], bootstrap.ADMIT_QUALIFICATION)
                validate_model_qualification(
                    load_benchmark_config(command[-2]),
                    json.loads(Path(command[-1]).read_text()),
                )
            elif stage == "model_qualification":
                directory = Path(command[command.index("--output") + 1])
                directory.mkdir()
                (directory / "qualification.json").write_bytes(self.payload)
            elif stage == "select_training_task":
                bootstrap.write_json(output / "selection.json", {
                    "task_id": "fixture_train_task", "split": command[-3],
                    "eligible_index": int(command[-2]), "seed": int(command[-1]),
                })
                (output / "selected_task_id.txt").write_text("fixture_train_task\n")
            elif stage == "development_pipeline":
                directory = Path(command[command.index("--output") + 1])
                bootstrap.write_json(directory / "pipeline.json", {
                    "status": "SCORED_DEVELOPMENT_ATTEMPT",
                })
                bootstrap.write_json(directory / "attempt/development_score.json", {
                    "fixture": True, "native_tgc": False,
                })
            else:
                raise AssertionError("Unexpected stage: " + stage)

        with mock.patch.object(bootstrap, "run_command", side_effect=run):
            bootstrap.run_development(
                sys.executable, sys.executable, dict(os.environ),
                work, output, store, report, qualification_reference=reference,
                development_attempt=development_attempt,
            )
        return case

    def test_valid_reuse_preserves_bytes_and_original_provenance(self) -> None:
        case = self.run_case(self.reference)
        path = case.output / "qualification/qualification.json"
        self.assertEqual(path.read_bytes(), self.payload)
        self.assertEqual(json.loads(path.read_text())["run_id"], "original-run")
        self.assertEqual(json.loads(path.read_text())["hostname"], "original-host")
        self.assertEqual(json.loads(path.read_text())["model_directory"], "/tmp/original-run/model")
        self.assertEqual(case.report["run_id"], "new-run")
        self.assertEqual(case.report["qualification_reuse"], {
            "reference": self.reference, "neutral_probe_repeated": False,
        })
        self.assertEqual([row[0] for row in case.commands], [
            "native_binding", "qualification_reuse_admission",
            "select_training_task", "development_pipeline",
        ])
        command = case.commands[-1][1]
        self.assertEqual(command[command.index("--model-qualification") + 1], str(path))

    def test_default_route_still_executes_neutral_qualification(self) -> None:
        case = self.run_case(None)
        self.assertEqual([row[0] for row in case.commands], [
            "native_binding", "model_qualification",
            "select_training_task", "development_pipeline",
        ])
        self.assertNotIn("qualification_reuse", case.report)
        case.store.hdfs.assert_not_called()
        self.assertIn(
            "population_dynamics.appworld_benchmark_model_runtime", case.commands[1][1],
        )
        selection_command, pipeline_command = case.commands[-2][1], case.commands[-1][1]
        self.assertEqual(selection_command[-3:], ["train", "0", "41"])
        self.assertEqual(pipeline_command[pipeline_command.index("--agent") + 1], "official_react_code")
        self.assertEqual(pipeline_command[pipeline_command.index("--seed") + 1], "41")
        self.assertNotIn("--verification-budget", pipeline_command)
        self.assertNotIn("--max-model-calls", pipeline_command)
        self.assertEqual(case.report["method"], "official_react_code")
        self.assertEqual(case.report["eligible_index"], 0)
        self.assertEqual(case.report["seed"], 41)
        self.assertFalse(case.report["neuronsim_method"])

    def test_uniform_dispatch_preserves_declared_selection_seed_budgets_and_metadata(self) -> None:
        for seed, index in ((41, 0), (42, 7), (43, 2)):
            with self.subTest(seed=seed, index=index):
                declared = uniform_attempt(seed=seed, eligible_index=index)
                case = self.run_case(self.reference, development_attempt=declared)
                selection = json.loads((case.output / "selection.json").read_text())
                self.assertEqual(selection["eligible_index"], index)
                self.assertEqual(selection["seed"], seed)
                command = case.commands[-1][1]
                expected = {
                    "--agent": "collaboration_uniform", "--split": "train",
                    "--task-id": "fixture_train_task", "--seed": str(seed),
                    "--verification-budget": "8", "--max-model-calls": "50",
                }
                self.assertEqual({key: command[command.index(key) + 1] for key in expected}, expected)
                self.assertNotIn("--planner-calls", command)
                self.assertEqual({key: case.report[key] for key in declared}, declared)
                self.assertEqual(case.report["method"],
                                 "development_full_history_uniform_verification_budget_8")
                self.assertFalse(case.report["neuronsim_method"])
                self.assertEqual(case.report["qualification_reuse"]["reference"], self.reference)
                self.assertEqual([row[0] for row in case.commands], [
                    "native_binding", "qualification_reuse_admission",
                    "select_training_task", "development_pipeline",
                ])

    def test_official_dispatch_preserves_selection_and_qualification_without_budget_overrides(self) -> None:
        for agent in ("official_react_code", "official_full_code", "official_function_calling"):
            for split, seed, index in (("train", 42, 3), ("dev", 43, 1)):
                with self.subTest(agent=agent, split=split):
                    declared = official_attempt(agent=agent, split=split, seed=seed, eligible_index=index)
                    original = copy.deepcopy(declared)
                    case = self.run_case(self.reference, development_attempt=declared)
                    self.assertEqual(declared, original)
                    self.assertEqual(case.commands[-2][1][-3:], [split, str(index), str(seed)])
                    command = case.commands[-1][1]
                    expected = {
                        "--agent": agent, "--split": split, "--task-id": "fixture_train_task",
                        "--seed": str(seed),
                        "--model-qualification": str(case.output / "qualification/qualification.json"),
                    }
                    self.assertEqual({key: command[command.index(key) + 1] for key in expected}, expected)
                    for option in ("--verification-budget", "--max-model-calls", "--planner-calls"):
                        self.assertNotIn(option, command)
                    self.assertEqual(case.report["method"], agent)
                    self.assertEqual({key: case.report[key] for key in declared}, declared)
                    self.assertFalse(case.report["neuronsim_method"])
                    self.assertEqual(case.report["qualification_reuse"]["reference"], self.reference)
                    self.assertEqual([row[0] for row in case.commands], [
                        "native_binding", "qualification_reuse_admission",
                        "select_training_task", "development_pipeline",
                    ])

    def test_invalid_official_dispatch_stops_before_native_binding(self) -> None:
        for declared in (official_attempt(verification_budget=None),
                         official_attempt(split="test_normal"),
                         official_attempt(seed=True)):
            with self.subTest(declared=declared):
                with self.assertRaisesRegex(ValueError, "DEVELOPMENT_ATTEMPT"):
                    self.run_case(self.reference, development_attempt=declared)
                self.assertEqual(self.last_case.commands, [])
                self.last_case.store.hdfs.assert_not_called()

    def test_invalid_dispatch_stops_before_native_binding(self) -> None:
        with self.assertRaisesRegex(ValueError, "DEVELOPMENT_ATTEMPT"):
            self.run_case(self.reference, development_attempt=uniform_attempt(seed=True))
        self.assertEqual(self.last_case.commands, [])
        self.last_case.store.hdfs.assert_not_called()

    def test_optional_reference_rejects_malformed_metadata(self) -> None:
        inputs = valid_inputs()
        inputs["model_qualification"] = self.reference
        bootstrap.validate_inputs(inputs)
        malformed = [
            None, {}, {**self.reference, "uri": "https://fixture/qualification.json"},
            {**self.reference, "uri": "hdfs://"}, {**self.reference, "bytes": True},
            {**self.reference, "bytes": 0}, {**self.reference, "sha256": "bad"},
            {**self.reference, "extra": "unbound"},
        ]
        for reference in malformed:
            with self.subTest(reference=reference):
                inputs = valid_inputs()
                inputs["model_qualification"] = reference
                with self.assertRaisesRegex(ValueError, "REQUIRES_HDFS_URI_BYTES_SHA256"):
                    bootstrap.validate_inputs(inputs)

    def test_uri_bytes_and_hash_mismatches_stop_before_task_execution(self) -> None:
        cases = [
            {**self.reference, "uri": "hdfs://fixture/wrong/qualification.json"},
            {**self.reference, "bytes": self.reference["bytes"] + 1},
            {**self.reference, "sha256": "0" * 64},
        ]
        for reference in cases:
            with self.subTest(reference=reference):
                with self.assertRaises((RuntimeError, ValueError)):
                    self.run_case(reference)
                self.assertEqual([row[0] for row in self.last_case.commands], ["native_binding"])
                self.assertFalse(
                    (self.last_case.output / "qualification/qualification.json").exists()
                )

    def test_existing_validator_rejects_non_pass_and_changed_identity(self) -> None:
        for change in (
            {"status": "FAIL"},
            {"tool_calls_validated": False},
            {"model_identity": {**self.model, "revision": "b" * 40}},
            {"config": {**self.config, "model_qualification": {"tensor_parallel_size": 1}}},
        ):
            with self.subTest(change=change):
                self.set_payload({**self.qualification, **change})
                with self.assertRaises(ValueError):
                    self.run_case(self.reference)
                self.assertEqual([row[0] for row in self.last_case.commands], [
                    "native_binding", "qualification_reuse_admission",
                ])

    def test_admission_fragment_runs_the_existing_validator_on_cpu(self) -> None:
        config_path = self.work / "validation_config.yaml"
        qualification_path = self.work / "qualification.json"
        bootstrap.write_json(config_path, self.config)
        qualification_path.write_bytes(self.payload)
        command = [
            sys.executable, "-c", bootstrap.ADMIT_QUALIFICATION,
            str(config_path), str(qualification_path),
        ]
        environment = dict(
            os.environ, PYTHONPATH=str(SCRIPT.parents[1]), CUDA_VISIBLE_DEVICES="",
        )
        result = subprocess.run(
            command, env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        bootstrap.write_json(qualification_path, {**self.qualification, "status": "FAIL"})
        result = subprocess.run(
            command, env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("real neutral model qualification has not passed", result.stderr)


class EntrypointTests(unittest.TestCase):
    def test_embedded_python_fragments_compile(self) -> None:
        for name in ("PREPARE_CONFIG", "VERIFY_SNAPSHOT", "SELECT_TASK", "ADMIT_QUALIFICATION"):
            with self.subTest(fragment=name):
                compile(getattr(bootstrap, name), "<" + name + ">", "exec")

    def test_cli_help_succeeds_without_gpu_allocation(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--inputs", result.stdout)


if __name__ == "__main__":
    unittest.main()
