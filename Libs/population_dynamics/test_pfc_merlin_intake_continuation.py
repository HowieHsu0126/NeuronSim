"""Tests for continuation selection, accepted evidence and private cleanup."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import pfc_merlin_intake_continuation as continuation


class ContinuationTests(unittest.TestCase):
    def source(self):
        return "hdfs://checkpoints/" + continuation.SOURCE_RUN_ID

    def index(self, values):
        return dict(run_id=continuation.SOURCE_RUN_ID, trial_id=continuation.SOURCE_TRIAL_ID,
                    completed=[dict(index=index, run_id=continuation.SOURCE_RUN_ID,
                         trial_id=continuation.SOURCE_TRIAL_ID, exit_code=0, persisted=True,
                         checkpoint_path=self.source() + f"/pfc/task_{index:04d}") for index in values])

    def prior(self, values):
        return dict(schema=continuation.SCHEMA, status="PASS", accepted_task_indices=list(values),
                    accepted_episode_count=len(values), source_simulations=0, production_episodes=0,
                    source_qualification_granted=False, scaling_qualified=False,
                    scientific_parameter_domain_qualified=False, failures=[],
                    execution=dict(run_id="real-run", trial_id="real-trial"))

    def test_ready_selection_excludes_accepted_and_uses_bounded_batches(self):
        self.assertEqual(continuation.select_ready(self.index([1, 1005, 2005]), {1}, self.source(), 1), [1005])
        template = dict(prior_acceptance="old", return_tasks=[], domain_base="/domain")
        canonical = dict(source_root="/domain/source_results", run_id=continuation.SOURCE_RUN_ID,
                         trial_id=continuation.SOURCE_TRIAL_ID)
        value = continuation.batch_config(template, canonical, self.source(), [1005, 2005], Path("prior"))
        self.assertEqual([row["xie_job"] for row in value["return_tasks"]], ["643652_5", "643653_5"])
        self.assertEqual(template["return_tasks"], [])

    def test_invalid_ready_rows_fail_without_selection(self):
        for kind in ("duplicate", "bool_index", "failed", "unpersisted", "run", "trial", "path", "status"):
            with self.subTest(kind=kind):
                value = self.index([1])
                row = value["completed"][0]
                if kind == "duplicate": value["completed"].append(dict(row))
                elif kind == "bool_index": row["index"] = True
                elif kind == "failed": row["exit_code"] = 1
                elif kind == "unpersisted": row["persisted"] = False
                elif kind == "run": row["run_id"] = "other"
                elif kind == "trial": row["trial_id"] = "other"
                elif kind == "path": row["checkpoint_path"] = "hdfs://other"
                else: value["status"] = "FAILED"
                with self.assertRaises(ValueError):
                    continuation.select_ready(value, set(), self.source(), 48)

    def test_partial_failed_batch_preserves_success_count_and_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "acceptance.json"
            prefix = "hdfs://continuation/batch_0001"
            batch = self.prior([0, 1])
            batch.update(status="FAILED", new_accepted_episode_count=1,
                         completed=[dict(index=1, acceptance_uri=prefix + "/pfc_scientific_intake/task_0001_acceptance.json",
                                         acceptance_sha256="a" * 64)],
                         failures=[dict(index=2, error_type="ValueError", error="retained failure")])
            continuation.atomic_json(path, batch)
            state = self.prior([0])
            state.update(status="RUNNING", batches=[], prior_accepted_episode_count=1, new_accepted_episode_count=0)
            result = continuation.merge_batch(state, batch, [1, 2], path, prefix, "real-run", "real-trial")
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["accepted_task_indices"], [0, 1])
            self.assertEqual(result["new_accepted_episode_count"], 1)
            self.assertEqual(result["failures"][0]["index"], 2)
            self.assertEqual(result["batches"][0]["run_id"], "real-run")

    def test_execution_identity_and_recovered_receipt_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "acceptance.json"
            prefix = "hdfs://continuation/batch_0001"
            batch = self.prior([0, 1])
            batch.update(status="PASS", new_accepted_episode_count=1,
                         completed=[dict(index=1, acceptance_uri=prefix + "/pfc_scientific_intake/task_0001_acceptance.json",
                                         acceptance_sha256="a" * 64)])
            continuation.atomic_json(path, batch)
            state = self.prior([0]); state.update(status="RUNNING", batches=[], prior_accepted_episode_count=1)
            wrong = copy.deepcopy(batch); wrong["execution"]["run_id"] = "wrong"
            with self.assertRaisesRegex(ValueError, "BATCH_MERLIN_EXECUTION_IDENTITY_CHANGED"):
                continuation.merge_batch(copy.deepcopy(state), wrong, [1], path, prefix, "real-run", "real-trial")
            recovered = copy.deepcopy(batch)
            recovered.update(status="FAILED", recovered_publication=True,
                             publication_error=dict(error_type="OSError", error="HDFS final publication failed"))
            continuation.atomic_json(path, recovered)
            result = continuation.merge_batch(state, recovered, [1], path, prefix, "real-run", "real-trial")
            self.assertEqual(result["accepted_episode_count"], 2)
            self.assertTrue(result["batches"][0]["acceptance_uri"].endswith("/acceptance_recovery.json"))
            self.assertEqual(result["status"], "FAILED")

    def test_successful_batches_accumulate_without_repeating_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, work = self.fixture(root, list(range(2998)))
            value = continuation.read_json(config); value["batch_size"] = 1
            continuation.atomic_json(config, value)
            calls = []
            def download(uri, path):
                value = self.index([2998, 2999]); continuation.atomic_json(path, value); return value
            def intake_run(config_path, results):
                value = continuation.read_json(config_path)
                prior = continuation.read_json(value["prior_acceptance"])
                selected = [row["index"] for row in value["return_tasks"]]; calls.extend(selected)
                accepted = prior["accepted_task_indices"] + selected
                prefix = os.environ["RH2_CHECKPOINT_HDFS"]
                report = self.prior(accepted)
                report.update(new_accepted_episode_count=1,
                    completed=[dict(index=selected[0], acceptance_uri=prefix + f"/pfc_scientific_intake/task_{selected[0]:04d}_acceptance.json", acceptance_sha256="a" * 64)])
                continuation.atomic_json(results / "acceptance.json", report); return report
            with patch.dict(os.environ, self.environment(), clear=True), \
                    patch.object(continuation, "filesystem", return_value="ext2/ext3"), \
                    patch.object(continuation.intake, "download_json", side_effect=download), \
                    patch.object(continuation.intake, "hdfs"), \
                    patch.object(continuation.intake, "run", side_effect=intake_run):
                result = continuation.run(config, work)
            self.assertEqual(calls, [2998, 2999])
            self.assertEqual(result["accepted_episode_count"], 3000)
            self.assertEqual(result["new_accepted_episode_count"], 2)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(result["batches"]), 2)

    def test_private_cleanup_rejects_preexisting_and_outside_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("source_results", "observations"):
                (root / name).mkdir()
            template = dict(domain_base=str(root))
            task = dict(index=1, source_root=str(root / "source_results"))
            allowed = (root / "source_results", root / "observations")
            with patch.object(continuation, "filesystem", return_value="ext2/ext3"):
                paths = continuation.owned_paths(template, [task])
                for path in paths:
                    path.mkdir()
                    (path / "private.dat").write_text("private")
                protected = root / "frozen"
                protected.mkdir()
                with self.assertRaisesRegex(ValueError, "CLEANUP_PATH_OUTSIDE_ALLOWLIST"):
                    continuation.cleanup_owned([protected], allowed)
                cleaned = continuation.cleanup_owned(paths, allowed)
                self.assertEqual(len(cleaned), 2)
                self.assertTrue(protected.is_dir())
                (root / "source_results" / "task_0001").mkdir()
                with self.assertRaisesRegex(ValueError, "CLEANUP_TASK_PATH_PREEXISTS"):
                    continuation.owned_paths(template, [task])
            with patch.object(continuation, "filesystem", return_value="nfs"):
                with self.assertRaisesRegex(ValueError, "CLEANUP_ROOT_NOT_PRIVATE_LOCAL"):
                    continuation.owned_paths(template, [task])

    def fixture(self, root, accepted):
        domain = root / "domain"
        for name in ("source_results", "observations"):
            (domain / name).mkdir(parents=True)
        task = dict(index=622, source_root=str(domain / "source_results"),
                    run_id=continuation.SOURCE_RUN_ID, trial_id=continuation.SOURCE_TRIAL_ID,
                    xie_job="643651_622", receipt_uri=self.source() + "/pfc/task_0622/receipt.json",
                    bundle_uri=self.source() + "/pfc/task_0622/result.tar.gz")
        template = root / "template.json"
        continuation.atomic_json(template, dict(domain_base=str(domain), return_tasks=[task],
                                                 prior_acceptance=str(root / "frozen_prior.json")))
        prior = root / "prior.json"
        continuation.atomic_json(prior, self.prior(accepted))
        config = root / "config.json"
        continuation.atomic_json(config, dict(template_config=str(template), prior_acceptance=str(prior),
                    completion_index_uri=self.source() + "/pfc/progress.json",
                    expected_source_run_id=continuation.SOURCE_RUN_ID,
                    expected_source_trial_id=continuation.SOURCE_TRIAL_ID, batch_size=48,
                    poll_seconds=180, target=3000))
        work = root / "work"
        work.mkdir()
        return config, work

    def environment(self):
        return dict(RH2_JOB_RUN_ID="real-run", ARNOLD_TRIAL_ID="real-trial",
                    ARNOLD_MONITOR_ROLE="head", ARNOLD_APPLIED_GPU_NUM="0", ARNOLD_HEAD_CPU="2",
                    RH2_CHECKPOINT_HDFS="hdfs://continuation-real-run", CUDA_VISIBLE_DEVICES="")

    def test_complete_prior_never_calls_scientific_intake(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, work = self.fixture(Path(temporary), list(range(3000)))
            with patch.dict(os.environ, self.environment(), clear=True), \
                    patch.object(continuation, "filesystem", return_value="ext2/ext3"), \
                    patch.object(continuation.intake, "hdfs"), patch.object(continuation.intake, "run") as intake_run:
                result = continuation.run(config, work)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["accepted_episode_count"], 3000)
            self.assertEqual(result["production_episodes"], 0)
            intake_run.assert_not_called()

    def test_partial_batch_stops_after_publishing_cumulative_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, work = self.fixture(root, [0])
            calls = []
            def download(uri, path):
                value = self.index([0, 1, 2])
                continuation.atomic_json(path, value)
                return value
            def intake_run(config_path, results):
                value = continuation.read_json(config_path)
                calls.append([row["index"] for row in value["return_tasks"]])
                prefix = os.environ["RH2_CHECKPOINT_HDFS"]
                report = self.prior([0, 1])
                report.update(status="FAILED", new_accepted_episode_count=1,
                              completed=[dict(index=1, acceptance_uri=prefix + "/pfc_scientific_intake/task_0001_acceptance.json",
                                              acceptance_sha256="a" * 64)],
                              failures=[dict(index=2, error_type="ValueError", error="source rejection")])
                continuation.atomic_json(results / "acceptance.json", report)
                return report
            with patch.dict(os.environ, self.environment(), clear=True), \
                    patch.object(continuation, "filesystem", return_value="ext2/ext3"), \
                    patch.object(continuation.intake, "download_json", side_effect=download), \
                    patch.object(continuation.intake, "hdfs"), \
                    patch.object(continuation.intake, "run", side_effect=intake_run), \
                    patch.object(continuation.time, "sleep") as sleep:
                result = continuation.run(config, work)
            self.assertEqual(calls, [[1, 2]])
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["accepted_episode_count"], 2)
            self.assertEqual(continuation.read_json(work / "acceptance.json")["failures"][0]["index"], 2)
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
