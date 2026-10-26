"""Validate declared sequential development collection on Merlin CPU."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

from population_dynamics import test_appworld_benchmark_merlin as fixture
from population_dynamics import appworld_benchmark_pipeline as native_pipeline

bootstrap = fixture.bootstrap


class DevelopmentBatchTests(fixture.TemporaryArtifacts):
    def test_single_and_ordered_batch_admission(self):
        declared = [fixture.official_attempt(seed=43), fixture.uniform_attempt(eligible_index=1)]
        original = copy.deepcopy(declared)
        inputs = {**fixture.valid_inputs(), "development_attempts": declared}
        bootstrap.validate_inputs(inputs)
        actual = bootstrap.resolve_development_attempts(inputs)
        self.assertEqual([row["seed"] for row in actual], [43, 41])
        self.assertEqual(declared, original)
        self.assertEqual(bootstrap.resolve_development_attempts({}),
                         [bootstrap.resolve_development_attempt({})])

    def test_invalid_rosters_stop_before_preparation(self):
        invalid = [
            {}, None, [], "all", [None],
            [fixture.official_attempt(split="test_normal")],
            [fixture.official_attempt(seed=44)],
            [fixture.official_attempt(max_model_calls=1)],
            [fixture.uniform_attempt(), fixture.uniform_attempt(verification_budget=7)],
        ]
        with mock.patch.object(bootstrap, "prepare_development_context") as prepare:
            for declared in invalid:
                with self.subTest(declared=declared), self.assertRaises(ValueError):
                    bootstrap.run_development_batch(
                        sys.executable, sys.executable, {}, self.work, self.output,
                        self.store, self.report, development_attempts=declared,
                    )
            prepare.assert_not_called()
        with self.assertRaisesRegex(ValueError, "EXCLUSIVE"):
            bootstrap.validate_inputs({
                **fixture.valid_inputs(), "development_attempt": fixture.official_attempt(),
                "development_attempts": [fixture.official_attempt()],
            })

    def run_batch(self, *, outcomes=(False, True), failure=None, bad_score=None,
                  fail_publication=False, selection_failure=False):
        attempts = [fixture.official_attempt(eligible_index=index, seed=41 + index)
                    for index in range(len(outcomes))]
        commands = []
        self.commands = commands
        self.selections = []
        real_run = bootstrap.run_command
        real_publish = self.store.publish
        context = {
            "common": ["--config", str(self.output / "bound_config.yaml")],
            "native_environment": self.output / "bound_native_environment.json",
            "split_manifest": self.work / "split_manifest.json",
            "qualification": self.output / "qualification",
            "environment": dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        }

        def select(attempt, **kwargs):
            if selection_failure and attempt["eligible_index"] == 1:
                raise ValueError("fixture selection failure")
            self.selections.append(attempt["eligible_index"])
            return {"task_id": f"fixture_{attempt['eligible_index']}",
                    "split": attempt["split"], "seed": attempt["seed"],
                    "eligible_index": attempt["eligible_index"],
                    "native_membership_index": attempt["eligible_index"] + 1,
                    "excluded_demo_task_ids": ["fixture_demo"],
                    "task_content_read": False, "scores_read": False}

        remote_identities = {}

        def publish(path, relative, **kwargs):
            identity = bootstrap.file_identity(path)
            previous = remote_identities.get(relative)
            if previous is not None and previous != identity and not kwargs.get("replace", False):
                raise ValueError("EXISTING_PUBLISHED_IDENTITY_DIFFERS: " + relative)
            if kwargs.get("replace", False):
                self.assertEqual(relative, "stages/development_batch.json")
            remote_identities[relative] = identity
            if fail_publication and relative.endswith("/pipeline.json"):
                raise RuntimeError("fixture publication failure")
            return real_publish(path, relative, **kwargs)

        def run(stage, command, environment, work, output, store, report, closed_outputs):
            self.assertEqual(len(self.selections), len(attempts))
            index = len(commands)
            if index:
                self.assertEqual(report["development_plan"][index-1]["state"],
                                 "SCORED_SUCCESS" if outcomes[index-1] else "SCORED_NEGATIVE")
                self.assertEqual(store.uploads[-1][1], "stages/development_batch.json")
            commands.append(command)
            directory = Path(command[command.index("--output") + 1])
            # The software harness removes allocation variables. Supply an explicit
            # synthetic identity while exercising the real scratch-path validator.
            with mock.patch.dict(os.environ, {
                "RH2_JOB_RUN_ID": "fixture-run", "ARNOLD_TRIAL_ID": "fixture-trial",
                "RH2_CHECKPOINT_HDFS": "hdfs://fixture/checkpoints/fixture-run",
            }):
                native_pipeline._check_execution_location(directory)
            self.assertFalse(directory.exists(), "Native pipeline must own its fresh attempt directory")
            identity = {
                "task_id": command[command.index("--task-id") + 1],
                "seed": int(command[command.index("--seed") + 1]),
                "split": command[command.index("--split") + 1],
                "method": command[command.index("--agent") + 1],
            }
            execution = {**identity, "schema": "appworld_development_execution_v1",
                         "status": "completed", "usage_complete": True, "lm_call_count": 2}
            score = {**identity, "schema": "appworld_development_score_v1",
                     "scoring_status": "completed", "native_tgc": outcomes[index]}
            pipeline = {
                "status": "SCORED_DEVELOPMENT_ATTEMPT", "native_tgc": outcomes[index],
                "processes": {
                    "actor": {"exit_code": 0, "cleanup": {"parent_reaped": True}},
                    "scorer": {"exit_code": 0, "cleanup": {"parent_reaped": True}},
                },
                "model_server": {"status": "STOPPED",
                                 "server_cleanup": {"server_parent_reaped": True}},
            }
            if bad_score == "identity":
                score["seed"] = 43
            elif bad_score == "server_cleanup":
                pipeline["model_server"]["server_cleanup"]["server_parent_reaped"] = False
            elif bad_score == "boolean":
                score["native_tgc"] = 0
            elif bad_score == "actor_failure":
                execution["status"] = "agent_failure"
                pipeline["processes"]["actor"]["exit_code"] = 1
            if failure == index:
                pipeline["status"] = "FAILED"
            files = {"pipeline.json": pipeline, "attempt/execution_receipt.json": execution,
                     "attempt/development_score.json": score}
            script = (
                "import json,pathlib,sys\n"
                "root=pathlib.Path(sys.argv[1]); files=json.loads(sys.argv[2])\n"
                "for name,record in files.items():\n"
                " p=root/name;p.parent.mkdir(parents=True,exist_ok=True);"
                "p.write_text(json.dumps(record)+'\\n')\n"
                "raise SystemExit(int(sys.argv[3]))\n"
            )
            real_run(stage, [sys.executable, "-c", script, str(directory),
                             json.dumps(files), "1" if failure == index else "0"],
                     environment, work, output, store, report, closed_outputs)

        self.store.publish = publish
        with mock.patch.object(bootstrap, "prepare_development_context", return_value=context) as prepare, \
             mock.patch.object(bootstrap, "select_attempt", side_effect=select), \
             mock.patch.object(bootstrap, "run_command", side_effect=run):
            try:
                bootstrap.run_development_batch(
                    sys.executable, sys.executable, {}, self.work, self.output,
                    self.store, self.report, development_attempts=attempts,
                    qualification_reference={"fixture": True},
                )
            finally:
                self.prepare_call_count = prepare.call_count

    def test_preserves_negative_then_success_and_publishes_before_advancing(self):
        self.run_batch()
        self.assertEqual(self.prepare_call_count, 1)
        self.assertEqual(self.report["status"], "SCORED_DEVELOPMENT_BATCH")
        self.assertEqual(self.report["development_attempts"], 2)
        self.assertEqual([row["state"] for row in self.report["development_plan"]],
                         ["SCORED_NEGATIVE", "SCORED_SUCCESS"])
        roots = [command[command.index("--output") + 1] for command in self.commands]
        self.assertEqual(len(set(roots)), 2)
        self.assertTrue(all("population_dynamics.appworld_benchmark_pipeline" in c
                            for c in self.commands))
        self.assertEqual([row["selection"]["native_membership_index"]
                          for row in self.report["development_plan"]], [1, 2])
        self.assertFalse(self.report["automatic_attempt_retries"])
        self.assertFalse(self.report["model_server_reuse"])
        bootstrap.publish_final(self.store, self.work, self.output, self.report)
        manifest = json.loads((self.work / "output_manifest.json").read_text())
        self.assertEqual(manifest["scope"], "declared_development_batch_and_diagnostics")
        self.assertEqual(sum(name.endswith("development_score.json")
                             for name in manifest["files"]), 2)

    def test_execution_failure_preserves_first_result_and_unstarted_denominator(self):
        with self.assertRaisesRegex(RuntimeError, "CHILD_STAGE_FAILED"):
            self.run_batch(outcomes=(False, False, False), failure=1)
        self.assertEqual(len(self.commands), 2)
        self.assertEqual([row["state"] for row in self.report["development_plan"]],
                         ["SCORED_NEGATIVE", "UNKNOWN", "NOT_STARTED"])
        self.assertEqual(self.report["development_coverage"]["planned"], 3)
        self.assertEqual(self.report["development_attempts"], 1)

    def test_all_selections_must_pass_before_first_actor(self):
        with self.assertRaisesRegex(ValueError, "fixture selection failure"):
            self.run_batch(selection_failure=True)
        self.assertEqual(self.commands, [])
        self.assertTrue(all(row["state"] == "NOT_STARTED"
                            for row in self.report["development_plan"]))

    def test_publication_failure_stops_before_next_attempt(self):
        with self.assertRaisesRegex(RuntimeError, "fixture publication failure"):
            self.run_batch(fail_publication=True)
        self.assertEqual(len(self.commands), 1)
        self.assertEqual([row["state"] for row in self.report["development_plan"]],
                         ["UNKNOWN", "NOT_STARTED"])

    def test_nonzero_actor_can_have_a_valid_independent_negative_score(self):
        self.run_batch(outcomes=(False,), bad_score="actor_failure")
        self.assertEqual(self.report["development_plan"][0]["state"], "SCORED_NEGATIVE")

    def test_changed_score_identity_stops_before_next_attempt(self):
        with self.assertRaisesRegex(ValueError, "IDENTITY_DIFFERS"):
            self.run_batch(bad_score="identity")
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(self.report["development_plan"][1]["state"], "NOT_STARTED")

    def test_unconfirmed_server_cleanup_stops_before_next_attempt(self):
        with self.assertRaisesRegex(ValueError, "PROCESS_CLOSURE_UNCONFIRMED"):
            self.run_batch(bad_score="server_cleanup")
        self.assertEqual(len(self.commands), 1)

    def test_non_boolean_score_stops_before_next_attempt(self):
        with self.assertRaisesRegex(ValueError, "COMPLETED_SCORE_REQUIRED"):
            self.run_batch(bad_score="boolean")


class RegisteredDataLinkTests(fixture.TemporaryArtifacts):
    def setUp(self):
        super().setUp()
        self.data_root = self.work / "admitted"
        self.data_directory = self.data_root / "data"
        self.data_directory.mkdir(parents=True)
        self.attempt = self.output / "development/official_full_code/train/seed_41/eligible_1/attempt"
        self.attempt.mkdir(parents=True)
        self.data_link = self.attempt / "data"
        self.data_link.symlink_to(self.data_directory, target_is_directory=True)
        self.store.native_attempt_roots = (self.attempt.relative_to(self.output).as_posix(),)
        archive = {"uri": "hdfs://fixture/data.tar.gz", "sha256": "b"*64}
        bootstrap.write_json(self.work/"native_preparation.receipt.json", {
            "status": "PASS_DATA_VERSION_AND_NATIVE_SPLIT_ADMISSION",
            "exit_code": 0, "persisted": True, "data_root": str(self.data_root),
            "data_hdfs_uri": archive["uri"], "data_bundle_sha256": archive["sha256"],
            "environment": {"status": "PREPARED_RUNTIME_AND_SPLIT_METADATA",
                            "data_root": str(self.data_root)},
        })
        bootstrap.write_json(self.output/"input_admission.json", {
            "status": "RESTORED_INPUT_IDENTITIES_VERIFIED", "data_archive": archive,
        })
        bootstrap.write_json(self.attempt/"execution_receipt.json", {
            "schema": "appworld_development_execution_v1", "run_root": str(self.attempt),
            "data_root": str(self.data_root), "native_data_directory": str(self.data_directory),
        })

    def test_only_exact_registered_link_is_admitted(self):
        self.assertIsNotNone(bootstrap.admitted_native_data_link(self.store, self.output, self.data_link))
        other = self.output / "development/unregistered/data"
        other.parent.mkdir(parents=True)
        other.symlink_to(self.data_directory, target_is_directory=True)
        self.assertIsNone(bootstrap.admitted_native_data_link(self.store, self.output, other))
        with self.assertRaisesRegex(ValueError, "OUTPUT_SYMLINK_CANNOT_BE_PUBLISHED"):
            bootstrap.publish_closed(self.store, self.output, [other])

    def test_registered_input_link_is_excluded_from_published_outputs(self):
        (self.data_directory/"input_only.txt").write_text("input must not be included")
        bootstrap.publish_closed(self.store, self.output, [self.attempt])
        bootstrap.publish_final(self.store, self.work, self.output, self.report)
        manifest = json.loads((self.work/"output_manifest.json").read_text())
        relative = self.data_link.relative_to(self.output).as_posix()
        self.assertIn(relative, manifest["input_links_excluded"])
        self.assertFalse(any("input_only" in name for name in manifest["files"]))

    def test_registered_link_still_rejects_wrong_target(self):
        self.data_link.unlink()
        self.data_link.symlink_to(self.work, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "TARGET_OR_OWNER_DIFFERS"):
            bootstrap.admitted_native_data_link(self.store, self.output, self.data_link)


class SelectionDirectoryTests(fixture.TemporaryArtifacts):
    def test_real_selector_keeps_distinct_paths_and_original_native_indices(self):
        config = {"schema": "appworld_benchmark_v1", "development_execution": {
            "api_predictor": {"demo_task_ids": ["case_1"]},
            "methods": {"official_react_code": {}, "official_full_code": {},
                        "official_function_calling": {}},
        }}
        manifest = {
            "schema": "appworld_split_membership_v1", "appworld_version": "fixture",
            "data_root": str(self.work), "splits": {"train": {
                "task_ids": ["case_1", "case_2", "case_3"], "task_count": 3,
                "scenario_count": 1, "membership_file": str(self.work/"train.txt"),
                "membership_sha256": "a"*64,
            }},
        }
        bootstrap.write_json(self.output/"bound_config.yaml", config)
        bootstrap.write_json(self.work/"split_manifest.json", manifest)
        for index, seed in ((0, 41), (1, 43)):
            directory = self.output / "selections" / str(index)
            result = subprocess.run(
                [sys.executable, "-c", bootstrap.SELECT_TASK, str(self.work),
                 "train", str(index), str(seed), str(directory)],
                env=dict(os.environ, PYTHONPATH=str(fixture.SCRIPT.parents[1]),
                         CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads((directory/"selection.json").read_text())
            self.assertEqual(row["native_membership_index"], index+1)
            self.assertEqual(row["seed"], seed)
            self.assertFalse(row["task_content_read"])
            self.assertFalse(row["scores_read"])
        self.assertFalse((self.output/"selection.json").exists())
