"""Causal receipt conversion checks; run on Merlin CPU, without a model or task.

Synthetic fixtures provide an already-admitted membership mapping. The shared
membership validator has its own tests; the parent also checks the builder with
the retained real manifest and train receipts.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from population_dynamics import appworld_benchmark_closure_data as data
from population_dynamics import appworld_benchmark_scoring as scoring


class ClosureDataTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "schema": "appworld_split_membership_v1", "appworld_version": "fixture-version",
            "data_root": "/fixture/data",
            "splits": {"train": {"task_ids": ["demo_1", "task_2"], "membership_sha256": "d" * 64}},
        }
        payload = json.dumps(self.manifest).encode()
        self.manifest_identity = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        memberships = {"train": set(self.manifest["splits"]["train"]["task_ids"])}
        for module in (data, scoring):
            patcher = mock.patch.object(module, "validate_split_manifest", return_value=memberships)
            patcher.start()
            self.addCleanup(patcher.stop)
        evaluator = mock.patch.object(scoring, "_load_evaluator", side_effect=AssertionError("No native scoring"))
        evaluator.start()
        self.addCleanup(evaluator.stop)

    def fixture(self, *, allocated=True, status="completed", seed=41, success=False, uniform=False):
        budget = 2
        policy = {"name": "fixed_verification_control", "planner_calls": 1, "score_conditioned": False}
        method = "development_full_history_fixed_verification_budget_2_planner_1"
        allocation = {"planner": 1, "executor": 1}
        if uniform:
            policy = {"name": "uniform_verification_collection", "seed": seed, "draw_index": 1,
                      "score_conditioned": False, "assignment_probability": 1 / 3}
            method = "development_full_history_uniform_verification_budget_2"
            allocation = data.UniformVerificationPolicy(seed, 1)(state_reports={}, total_budget=budget)
        histories, exports, native = {}, [], []
        used = {"planner": 0, "executor": 0}
        for actor in ("planner", "executor"):
            events = [
                {"index": 0, "kind": "task_instruction", "payload": {"text": "synthetic task"}},
                {"index": 1, "kind": "action_rejection", "payload": {"reason": actor.upper() + "_PRIVATE"}},
            ]
            if allocated:
                representation = {"representation": "full_visible_history", "actor": actor,
                                  "events": deepcopy(events)}
                exports.append({"actor": actor, "prefix_events": len(events),
                                "representation": representation, "wall_seconds": 0.001})
                events.append({"index": 2, "kind": "state_report_delivered",
                               "payload": {"representation": deepcopy(representation)}})
                events.append({"index": 3, "kind": "verification_allocation",
                               "payload": {"assigned_calls": allocation[actor]}})
                if allocation[actor]:
                    result_id = actor + "/call_1"
                    observation = {"ok": True, "result": {"error": "FUTURE_VALUE"},
                                   "raw_output": '{"error":"FUTURE_VALUE"}'}
                    row = {"actor": actor, "result_id": result_id, "app": "fixture", "api": "inspect",
                           "arguments": {}, "verification": True, "status": "returned",
                           "observation": observation, "raw_output": observation["raw_output"],
                           "native_wall_seconds": 0.01}
                    native.append(row)
                    events.append({"index": len(events), "kind": "tool_attempt",
                                   "payload": {name: deepcopy(row[name]) for name in
                                               ("result_id", "app", "api", "arguments", "verification")}})
                    events.append({"index": len(events), "kind": "tool_observation",
                                   "payload": {"result_id": result_id, "observation": deepcopy(observation)}})
                    used[actor] += 1
            histories[actor] = {"actor": actor, "events": events}
        receipt = {
            "schema": "appworld_development_execution_v1", "scope": "development_only",
            "formal_eligible": False, "terminal": True, "split": "train", "task_id": "task_2",
            "method": method, "seed": seed, "status": status, "usage_complete": True,
            "native_tgc": None, "scoring_status": "not_run", "target_load_ground_truth": False,
            "membership_verified": True, "run_root": "/fixture/run", "data_root": "/fixture/data",
            "experiment_name": "development", "appworld_version": "fixture-version",
            "native_kwargs": {"load_ground_truth": False}, "neuronsim_method": False,
            "state_encoder": "full_visible_history_collection_not_learned",
            "verification_budget": budget, "allocation_policy": policy,
            "host_budget": {"max_model_calls": 50, "max_completion_tokens": 8192, "max_output_tokens": 100000},
            "source": {"repository": "fixture/appworld", "revision": "1" * 40,
                       "archive_manifest": {"sha256": "b" * 64}},
            "model_identity": {"repo_id": "fixture/model", "revision": "2" * 40,
                               "language_model_training": False},
            "host": {"status": status, "stop_reason": "executor_stopped", "native_task_completed": False,
                     "allocation": allocation if allocated else None, "state_exports": exports,
                     "verification_used": used, "native_attempts": native, "actor_histories": histories},
        }
        score = {
            "schema": "appworld_development_score_v1", "scope": "development_only", "formal_eligible": False,
            **{name: receipt[name] for name in ("split", "task_id", "method", "seed")},
            "status": status, "execution_status": status, "usage_complete": True,
            "native_tgc": success, "scoring_status": "completed", "evaluator_called": True,
            "membership_verified": True, "native_evaluation_report": "/fixture/report.md",
            "native_evaluation_version_file": "/fixture/version.txt",
            "native_task_result": {"success": success, "num_tests": 1,
                                   "passes": ["check"] if success else [],
                                   "failures": [] if success else ["check"]},
            "provenance": {**{name: receipt[name] for name in
                              ("appworld_version", "run_root", "data_root", "experiment_name")},
                           "membership_sha256": "d" * 64},
        }
        admission = {
            "status": "RESTORED_INPUT_IDENTITIES_VERIFIED", "formal_task_contents_inspected": False,
            "split_manifest": dict(self.manifest_identity), "model_revision": "2" * 40,
            "source_archive_manifest_sha256": "b" * 64, "data_archive": {"sha256": "c" * 64},
            "source_sha256": {"Libs/population_dynamics/" + name: "a" * 64
                              for name in data._SOURCE_MODULES},
        }
        return {"execution_receipt": receipt, "score_receipt": score, "input_admission": admission}

    def identity(self, record):
        return {name: record["execution_receipt"][name] for name in ("split", "task_id", "method", "seed")}

    def build(self, records, plan=None):
        return data.build_closure_dataset(
            self.manifest, manifest_identity=self.manifest_identity,
            plan=[self.identity(record) for record in records] if plan is None else plan, attempts=records)

    def test_null_agent_failure_and_missing_attempt_stay_in_planned_denominator(self):
        record = self.fixture(allocated=False, status="agent_failure", uniform=True)
        plan = [self.identity(record), {**self.identity(record), "seed": 42}]
        result = self.build([record], plan)
        self.assertEqual(result["coverage"]["planned_attempts"], 2)
        self.assertEqual(result["coverage"]["by_coverage_status"], {"missing": 1, "settled": 1})
        self.assertEqual(result["coverage"]["settled_allocation_null_attempts"], 1)
        self.assertEqual(result["coverage"]["official_scored_attempts"], 1)
        self.assertEqual(result["coverage"]["official_successes"], 0)
        self.assertFalse(result["coverage"]["planned_scoring_complete"])
        self.assertIsNone(result["attempts"][0]["allocation"])
        self.assertEqual(result["attempts"][0]["native_membership_index"], 1)
        self.assertEqual(result["decision_examples"], [])

    def test_features_keep_actor_prefixes_private_and_future_labels_separate(self):
        record = self.fixture()
        result = self.build([record])
        example = result["decision_examples"][0]
        planner = json.dumps(example["features"]["actor_prefixes"]["planner"])
        executor = json.dumps(example["features"]["actor_prefixes"]["executor"])
        self.assertNotIn("EXECUTOR_PRIVATE", planner)
        self.assertNotIn("PLANNER_PRIVATE", executor)
        self.assertNotIn("FUTURE_VALUE", json.dumps(example["features"]))
        self.assertIn("FUTURE_VALUE", json.dumps(example["labels"]))
        self.assertEqual(example["treatment"]["allocation"], {"planner": 1, "executor": 1})
        self.assertIsNone(example["treatment"]["assignment_probability"])
        self.assertFalse(example["labels"]["official_score"]["native_tgc"])
        self.assertEqual({row["return_kind"] for row in example["labels"]["future_native_observations"]},
                         {"json_native_return"})
        self.assertNotIn("success", example["labels"]["future_native_observations"][0])
        self.assertEqual(result["attempts"][0]["host_status"], "completed")
        self.assertEqual(result["attempts"][0]["official_score"]["native_tgc"], False)

    def test_missing_recorded_propensity_is_not_reconstructed(self):
        record = self.fixture(uniform=True)
        del record["execution_receipt"]["allocation_policy"]["assignment_probability"]
        result = self.build([record])
        self.assertIsNone(result["attempts"][0]["recorded_assignment_probability"])
        self.assertIsNone(result["decision_examples"][0]["treatment"]["assignment_probability"])

    def test_uniform_membership_index_is_not_renumbered_for_a_tranche(self):
        record = self.fixture(allocated=False, uniform=True)
        self.assertEqual(self.build([record])["attempts"][0]["native_membership_index"], 1)
        record["execution_receipt"]["allocation_policy"]["draw_index"] = 0
        with self.assertRaisesRegex(ValueError, "original native train index"):
            self.build([record])

    def test_future_or_partial_state_exports_are_rejected(self):
        original = self.fixture()
        for change in ("count", "representation", "allocation_event", "duplicate_actor"):
            record = deepcopy(original)
            host = record["execution_receipt"]["host"]
            if change == "count":
                host["state_exports"][0]["prefix_events"] += 1
            elif change == "representation":
                host["state_exports"][0]["representation"]["events"].append(
                    {"index": 2, "kind": "tool_observation", "payload": {"leak": "future"}})
            elif change == "allocation_event":
                host["actor_histories"]["planner"]["events"][3]["payload"]["assigned_calls"] = 0
            else:
                host["state_exports"][1]["actor"] = "planner"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build([record])

    def test_unknown_unfinished_and_infrastructure_attempts_remain_unscored(self):
        records = []
        for seed, status, terminal in ((41, "unknown", True), (42, "running", False),
                                       (43, "infrastructure_failure", True)):
            record = self.fixture(seed=seed)
            receipt = record["execution_receipt"]
            receipt.update(status=status, terminal=terminal)
            receipt["host"]["status"] = status
            record["score_receipt"] = None
            records.append(record)
        result = self.build(records)
        self.assertEqual(result["coverage"]["by_coverage_status"],
                         {"infrastructure_failure": 1, "unfinished": 1, "unknown": 1})
        self.assertEqual(result["coverage"]["official_scored_attempts"], 0)
        self.assertEqual(result["decision_examples"], [])
        self.assertTrue(all(row["official_score"]["native_tgc"] is None for row in result["attempts"]))

    def test_score_identity_and_native_target_must_agree(self):
        original = self.fixture()
        changes = (
            lambda score: score["provenance"].update(membership_sha256="e" * 64),
            lambda score: score.update(native_tgc=True),
            lambda score: score.update(evaluator_called=False),
            lambda score: score.update(seed=42),
        )
        for change in changes:
            record = deepcopy(original)
            change(record["score_receipt"])
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build([record])

    def test_scored_success_after_agent_failure_remains_a_native_success(self):
        record = self.fixture(allocated=False, status="agent_failure", success=True)
        result = self.build([record])
        self.assertEqual(result["attempts"][0]["execution_status"], "agent_failure")
        self.assertTrue(result["attempts"][0]["official_score"]["native_tgc"])
        self.assertEqual(result["coverage"]["official_successes"], 1)

    def test_evaluator_failure_is_unknown_not_a_negative_target(self):
        record = self.fixture()
        record["score_receipt"].update(status="evaluator_failure", scoring_status="evaluator_failure",
                                       native_tgc=None, native_task_result=None)
        result = self.build([record])
        self.assertIsNone(result["decision_examples"][0]["labels"]["official_score"]["native_tgc"])
        self.assertFalse(result["decision_examples"][0]["official_target_available"])

    def test_execution_score_leakage_and_formal_split_are_rejected(self):
        record = self.fixture()
        record["execution_receipt"]["native_tgc"] = True
        with self.assertRaisesRegex(ValueError, "execution/score boundary"):
            self.build([record])
        record = self.fixture()
        record["execution_receipt"]["split"] = "test_challenge"
        with self.assertRaisesRegex(ValueError, "only train"):
            self.build([record])

    def test_unplanned_and_duplicate_attempts_are_rejected(self):
        record = self.fixture()
        with self.assertRaisesRegex(ValueError, "duplicate execution"):
            self.build([record, deepcopy(record)], [self.identity(record)])
        with self.assertRaisesRegex(ValueError, "unplanned"):
            self.build([record], [{**self.identity(record), "seed": 42}])

    def test_data_and_source_bindings_are_checked(self):
        original = self.fixture()
        changes = (
            lambda record: record["input_admission"]["split_manifest"].update(sha256="e" * 64),
            lambda record: record["input_admission"].update(source_archive_manifest_sha256="e" * 64),
            lambda record: record["input_admission"].update(model_revision="3" * 40),
            lambda record: record["input_admission"]["source_sha256"].clear(),
        )
        for change in changes:
            record = deepcopy(original)
            change(record)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build([record])
        second = self.fixture(seed=42)
        second["input_admission"]["data_archive"]["sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "mixed data/model/source"):
            self.build([original, second])

    def test_usage_and_native_ledger_cannot_be_silently_coerced(self):
        for change in ("used", "duplicate", "wrong_observation", "unknown"):
            record = self.fixture()
            host = record["execution_receipt"]["host"]
            if change == "used":
                host["verification_used"]["planner"] = 0
            elif change == "duplicate":
                host["native_attempts"].append(deepcopy(host["native_attempts"][0]))
            elif change == "wrong_observation":
                host["native_attempts"][0]["observation"]["result"] = {"changed": True}
            else:
                host["native_attempts"][0]["status"] = "unknown"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build([record])

    def test_cli_uses_explicit_relative_paths_and_refuses_output_overwrite(self):
        record = self.fixture(allocated=False, status="agent_failure")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {"manifest.json": self.manifest, "plan.json": [self.identity(record)],
                     "execution.json": record["execution_receipt"], "score.json": record["score_receipt"],
                     "admission.json": record["input_admission"],
                     "attempts.json": [{"execution_receipt": "execution.json", "score_receipt": "score.json",
                                        "input_admission": "admission.json"}]}
            for name, value in files.items():
                (root / name).write_text(json.dumps(value))
            args = ["--split-manifest", str(root / "manifest.json"), "--plan", str(root / "plan.json"),
                    "--attempts", str(root / "attempts.json"), "--output", str(root / "dataset.json")]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(data.main(args), 0)
                self.assertEqual(data.main(args), 2)
            result = json.loads((root / "dataset.json").read_text())
            self.assertEqual(result["coverage"]["planned_attempts"], 1)
            self.assertEqual(result["coverage"]["settled_allocation_null_attempts"], 1)
            self.assertEqual(result["input_files"]["split_manifest"]["sha256"], self.manifest_identity["sha256"])
            self.assertIn("execution_receipt", result["attempts"][0]["input_files"])


if __name__ == "__main__":
    unittest.main()
