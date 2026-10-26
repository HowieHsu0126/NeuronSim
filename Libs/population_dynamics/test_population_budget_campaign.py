"""Keep deployment budget continuation bound to its completed parent campaign."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import run_population_trajectory_campaign as runner


CONFIG = dict(training_seeds=[41, 42, 43], representation="recurrent", maximum_epochs=20,
              early_stopping_patience=4, device="cpu", cpu_threads=2)
FORECAST = dict(joint_samples=16, origins_ms=[1000., 3000., 5000.], horizons_ms=[20., 80., 500.])
ENVIRONMENT = dict(SLURM_JOB_ID="budget-fixture", SLURMD_NODENAME="cpu04",
                   SLURM_JOB_PARTITION="cpu", SLURM_JOB_GPUS="",
                   SLURM_ARRAY_JOB_ID="700000", SLURM_ARRAY_TASK_ID="1")


class PopulationBudgetCampaignTests(unittest.TestCase):
    def fixture(self, root):
        repo, work = root / "repo", root / "work"
        campaign = repo / "Output/neural_population_dynamics/model/trajectory_development/run_606139"
        parent = campaign / "seed_42"
        parent.mkdir(parents=True)
        software, episode = repo / runner.SOFTWARE, repo / runner.EPISODE
        for directory in (software, episode, repo / runner.COHORT):
            directory.mkdir(parents=True)
            runner.write_json(directory / "delivery.json", dict(status="PASS"))
        source_name = "Libs/population_dynamics/population_trajectory_training.py"
        source = parent / source_name
        source.parent.mkdir(parents=True)
        source.write_text("# accepted fixture numerical source\n")
        source_hash = runner.digest(source)
        sources = {Path(source_name).name: source_hash}
        input_hashes = {"train/design.json": "a" * 64, "train/acquisition.json": "b" * 64,
                        "train_fields/dataset.json": "c" * 64, "train_fields/episodes.csv": "d" * 64}
        binding = dict(config=CONFIG, seed=42, acquisition_design_sha256=input_hashes["train/design.json"],
                       acquisition_ledger_sha256=input_hashes["train/acquisition.json"],
                       dataset_sha256=input_hashes["train_fields/dataset.json"],
                       episode_table_sha256=input_hashes["train_fields/episodes.csv"])
        training = parent / "training"
        training.mkdir()
        (training / "latest.pt").write_bytes(b"committed epoch 20, Adam, variational RNG")
        (training / "best.pt").write_bytes(b"selected epoch 19")
        runner.write_json(training / "epochs.json", [dict(epoch=i) for i in range(20)])
        (training / "optimization.csv").write_text("epoch,loss\n19,1.0\n")
        (training / "validation.csv").write_text("epoch,loss\n19,1.0\n")
        runner.write_json(training / "run.json", dict(status="FITTED_DEVELOPMENT", training_seed=42,
            binding=binding, completed_epochs=20, effective_maximum_epochs=20, continuation=None,
            trainer_sources=sources, calibration_outcomes_accessed=False, test_outcomes_accessed=False,
            latest_checkpoint_sha256=runner.digest(training / "latest.pt"),
            best_checkpoint_sha256=runner.digest(training / "best.pt")))
        runner.write_json(parent / "training_configuration.json", CONFIG)
        runner.write_json(parent / "source_snapshot.json", dict(software_job_id="606023", episode_job_id="606074",
            software_delivery_sha256=runner.digest(software / "delivery.json"),
            episode_delivery_sha256=runner.digest(episode / "delivery.json")))
        runner.write_json(parent / "inputs.json", dict(delivery_sha256=runner.digest(repo / runner.COHORT / "delivery.json"),
            files=[dict(path=name, sha256=sha256) for name, sha256 in input_hashes.items()]))
        names = [source_name, "training_configuration.json", "source_snapshot.json", "inputs.json",
                 *["training/" + name for name in ("run.json", "latest.pt", "best.pt", "epochs.json",
                                                   "optimization.csv", "validation.csv")]]
        rows = [dict(path=name, bytes=(parent / name).stat().st_size, sha256=runner.digest(parent / name))
                for name in names]
        proof = dict(status="PASS", exit_code=0, job_id="completed-parent", training_seed=42,
                     representation="recurrent", device="cpu", core_neurons=100000, complete_episodes=72,
                     episode_ms=6000, fit_episodes=56, validation_episodes=16, verified_forecast_origins=48,
                     score_rows=144, calibration_or_test_accessed=False, scientific_claim_qualified=False, files=rows)
        runner.write_json(parent / "delivery.json", proof)

        def prerequisite(local_repo, relative, job):
            directory = software if relative == runner.SOFTWARE else episode
            return directory, {}, {source_name: next(row for row in rows if row["path"] == source_name)}

        def stage_software(local_repo, local_work):
            local_work.mkdir(parents=True)
            configs = local_work / "Libs/configs"
            configs.mkdir(parents=True)
            (configs / runner.CONFIGS[0]).write_text(yaml.safe_dump(CONFIG))
            (configs / runner.CONFIGS[1]).write_text(yaml.safe_dump(FORECAST))
            staged_source = local_work / source_name
            staged_source.parent.mkdir(parents=True)
            staged_source.write_text(source.read_text())
            return lambda *args, **kwargs: dict(status="PASS")

        return repo, work, campaign, parent, proof, prerequisite, stage_software

    def test_cli_forwards_seed_specific_budget_request(self):
        arguments = ["campaign", "--repo", "/tmp/neuronsim/budget-cli/repo", "--work", "/tmp/neuronsim/budget-cli/work",
                     "--budget-parent-campaign", "/tmp/neuronsim/budget-cli/parent", "--total-epochs", "80"]
        signature = inspect.signature(runner.run)
        with mock.patch.object(sys, "argv", arguments), mock.patch.object(runner, "run", return_value=0) as execute:
            with self.assertRaises(SystemExit) as result:
                runner.main()
        self.assertEqual(result.exception.code, 0)
        bound = signature.bind(*execute.call_args.args, **execute.call_args.kwargs)
        bound.apply_defaults()
        self.assertEqual(bound.arguments["budget_parent_campaign"], Path("/tmp/neuronsim/budget-cli/parent"))
        self.assertEqual(bound.arguments["total_epochs"], 80)
        self.assertEqual(bound.arguments["representation"], "recurrent")

    def test_invalid_budget_cli_rejects_before_any_execution(self):
        variants = [[], ["--total-epochs", "20"], ["--total-epochs", "80", "--representation", "fixed"],
                    ["--total-epochs", "80", "--pfc-preparation", "/tmp/pfc"],
                    ["--total-epochs", "80", "--prepare-pfc"],
                    ["--total-epochs", "80", "--cost-point"],
                    ["--total-epochs", "80", "--recovery-from", "/tmp/attempt"]]
        for extra in variants:
            with self.subTest(extra=extra):
                arguments = ["campaign", "--repo", "/tmp/neuronsim/budget-cli/repo", "--work", "/tmp/neuronsim/budget-cli/work",
                             "--budget-parent-campaign", "/tmp/neuronsim/budget-cli/parent", *extra]
                with mock.patch.object(sys, "argv", arguments), mock.patch.object(runner, "run") as execute, \
                        mock.patch.object(runner, "run_cost_point") as cost, mock.patch.object(runner, "prepare_pfc") as prepare:
                    with self.assertRaises(SystemExit) as result:
                        runner.main()
                self.assertEqual(result.exception.code, 2)
                execute.assert_not_called()
                cost.assert_not_called()
                prepare.assert_not_called()
        with mock.patch.object(sys, "argv", ["campaign", "--repo", "/tmp/repo", "--total-epochs", "80"]), \
                mock.patch.object(runner, "run") as execute:
            with self.assertRaises(SystemExit) as result:
                runner.main()
        self.assertEqual(result.exception.code, 2)
        execute.assert_not_called()

    def test_continuation_stages_only_parent_training_and_keeps_full_forecast_protocol(self):
        with tempfile.TemporaryDirectory(dir="/tmp/neuronsim", prefix="neuronsim-budget-") as temporary:
            repo, work, campaign, parent, proof, prerequisite, software = self.fixture(Path(temporary))
            before = {row["path"]: runner.digest(parent / row["path"]) for row in proof["files"]}
            calls = []

            def phase(local, name, command, probe):
                calls.append((name, command))
                if name == "fit":
                    self.assertNotIn("--resume", command)
                    local_parent = Path(command[command.index("--resume-from") + 1])
                    self.assertEqual(local_parent, work / "prerequisites/budget_parent/training")
                    self.assertEqual(command[command.index("--total-epochs") + 1], "80")
                    self.assertEqual((local_parent / "latest.pt").read_bytes(), (parent / "training/latest.pt").read_bytes())
                    self.assertFalse((local / "training").exists())
                    runner.write_json(local / "training/run.json", dict(status="FITTED_DEVELOPMENT", training_seed=42,
                        fitting_episodes=56, validation_episodes=16, completed_epochs=80, stop_reason="epoch_budget",
                        effective_maximum_epochs=80, binding=dict(config=CONFIG), continuation=dict(total_epochs=80,
                        base_maximum_epochs=20, parent_completed_epochs=20,
                        parent_checkpoint_sha256=runner.digest(local_parent / "latest.pt"))))
                elif name == "evaluate":
                    self.assertEqual(yaml.safe_load((local / "Libs/configs" / runner.CONFIGS[1]).read_text()), FORECAST)
                    runner.write_json(local / "forecasts/evaluation.json", dict(status="EVALUATED_DEVELOPMENT", score_rows=144))
                else:
                    self.assertIn("--all-origins", command)
                    runner.write_json(local / "replay.json", dict(status="PASS", verified_origins=48))
                return 0

            with mock.patch.dict(os.environ, ENVIRONMENT, clear=True), \
                    mock.patch.object(runner, "prerequisite", side_effect=prerequisite), \
                    mock.patch.object(runner, "stage_software", side_effect=software), \
                    mock.patch.object(runner, "stage_inputs"), mock.patch.object(runner, "phase", side_effect=phase):
                self.assertEqual(runner.run(repo, work, "cpu", budget_parent_campaign=campaign, total_epochs=80), 0)
            target = repo / "Output/neural_population_dynamics/model/trajectory_development/optimization_budget_continuation/run_700000/seed_42"
            delivery = json.loads((target / "delivery.json").read_text())
            self.assertEqual(delivery["status"], "PASS")
            self.assertEqual(delivery["effective_maximum_epochs"], 80)
            self.assertEqual(delivery["optimization_budget_continuation"]["parent"], str(parent))
            self.assertEqual(delivery["verified_forecast_origins"], 48)
            self.assertEqual(delivery["core_neurons"], 100000)
            self.assertFalse(delivery["scientific_claim_qualified"])
            self.assertEqual(json.loads((work / "training_configuration.json").read_text())["maximum_epochs"], 20)
            self.assertEqual([name for name, _ in calls], ["fit", "evaluate", "replay"])
            self.assertFalse((work / "prerequisites/budget_parent/forecasts").exists())
            self.assertEqual(before, {name: runner.digest(parent / name) for name in before})

    def test_legacy_delivery_requires_authenticated_recurrent_training_binding(self):
        for representation in ("recurrent", "fixed"):
            with self.subTest(representation=representation), tempfile.TemporaryDirectory(dir="/tmp/neuronsim", prefix="neuronsim-budget-") as temporary:
                repo, work, campaign, parent, proof, prerequisite, _ = self.fixture(Path(temporary))
                proof.pop("representation")
                if representation == "fixed":
                    for name in ("training_configuration.json", "training/run.json"):
                        path = parent / name
                        value = json.loads(path.read_text())
                        if name == "training_configuration.json":
                            value["representation"] = representation
                        else:
                            value["binding"]["config"]["representation"] = representation
                        runner.write_json(path, value)
                        row = next(row for row in proof["files"] if row["path"] == name)
                        row.update(bytes=path.stat().st_size, sha256=runner.digest(path))
                runner.write_json(parent / "delivery.json", proof)
                delivery_before = (parent / "delivery.json").read_bytes()
                with mock.patch.object(runner, "prerequisite", side_effect=prerequisite):
                    if representation == "recurrent":
                        accepted = runner.inspect_budget_parent(repo, campaign, 42, "cpu", 80)
                        self.assertEqual(accepted["parent"], str(parent))
                        self.assertEqual(accepted["config"]["representation"], "recurrent")
                    else:
                        with self.assertRaisesRegex(ValueError, "terminal frozen 20-epoch"):
                            runner.inspect_budget_parent(repo, campaign, 42, "cpu", 80)
                self.assertFalse(work.exists())
                self.assertEqual((parent / "delivery.json").read_bytes(), delivery_before)

    def recovery_fixture(self, root):
        repo, work, campaign, parent, proof, prerequisite, software = self.fixture(root)
        with mock.patch.object(runner, "prerequisite", side_effect=prerequisite):
            budget = runner.inspect_budget_parent(repo, campaign, 42, "cpu", 80)
        failed = repo / "Output/neural_population_dynamics/model/trajectory_development/optimization_budget_continuation/run_700000/seed_42"
        shutil.copytree(parent, failed)
        shutil.copytree(parent, failed / "prerequisites/budget_parent")
        continuation = dict(schema="neural_population_trajectory_budget_continuation_v1",
            base_maximum_epochs=20, total_epochs=80, parent_completed_epochs=20,
            parent_checkpoint_sha256=budget["parent_checkpoint_sha256"],
            parent_artifact_sha256={"run.json": runner.digest(parent / "training/run.json")})
        report = json.loads((failed / "training/run.json").read_text())
        report.update(status="RUNNING", attempt=2, completed_epochs=71, effective_maximum_epochs=80,
                      best_epoch=69, continuation=continuation)
        runner.write_json(failed / "training/run.json", report)
        runner.write_json(failed / "training/attempts/attempt_002.json", report)
        lineage = {key: value for key, value in budget.items()
                   if key not in ("config", "trainer_sources", "training_files", "parent_metadata")}
        failed_proof = dict(proof, status="RESOURCE_BLOCKED", exit_code=3, job_id="failed-child",
                            array_job_id="700000", optimization_budget_continuation=lineage)
        self.refresh_recovery_manifest(failed, failed_proof)
        return repo, work, failed, failed_proof, software

    def refresh_recovery_manifest(self, failed, proof):
        proof["files"] = [dict(path=str(path.relative_to(failed)), bytes=path.stat().st_size,
                               sha256=runner.digest(path)) for path in sorted(failed.rglob("*"))
                          if path.is_file() and path != failed / "delivery.json"]
        runner.write_json(failed / "delivery.json", proof)

    def test_resource_recovery_preserves_effective_budget_and_parent_prerequisites(self):
        with tempfile.TemporaryDirectory(dir="/tmp/neuronsim", prefix="neuronsim-budget-recovery-") as temporary:
            repo, work, failed, proof, software = self.recovery_fixture(Path(temporary))
            before = (failed / "delivery.json").read_bytes()
            calls = []

            def phase(local, name, command, probe):
                calls.append((name, command))
                if name == "fit":
                    self.assertIn("--resume", command)
                    self.assertNotIn("--resume-from", command)
                    self.assertNotIn("--total-epochs", command)
                    report = json.loads((local / "training/run.json").read_text())
                    self.assertEqual(report["completed_epochs"], 71)
                    self.assertEqual(report["effective_maximum_epochs"], 80)
                    self.assertEqual(report["best_epoch"], 69)
                    self.assertTrue((local / "training/attempts/attempt_002.json").is_file())
                    self.assertEqual((local / "prerequisites/budget_parent/delivery.json").read_bytes(),
                                     (failed / "prerequisites/budget_parent/delivery.json").read_bytes())
                    report.update(status="FITTED_DEVELOPMENT", attempt=3, completed_epochs=80,
                                  fitting_episodes=56, validation_episodes=16, stop_reason="epoch_budget")
                    runner.write_json(local / "training/run.json", report)
                elif name == "evaluate":
                    runner.write_json(local / "forecasts/evaluation.json", dict(status="EVALUATED_DEVELOPMENT", score_rows=144))
                else:
                    runner.write_json(local / "replay.json", dict(status="PASS", verified_origins=48))
                return 0

            environment = {key: value for key, value in ENVIRONMENT.items() if not key.startswith("SLURM_ARRAY_")}
            with mock.patch.dict(os.environ, environment, clear=True), \
                    mock.patch.object(runner, "stage_software", side_effect=software), \
                    mock.patch.object(runner, "stage_inputs"), mock.patch.object(runner, "phase", side_effect=phase):
                self.assertEqual(runner.run(repo, work, "cpu", 42, "700000", "700000_1", recovery_from=failed), 0)
            target = failed.parent / "resource_attempts/seed_42_job_budget-fixture"
            delivery = json.loads((target / "delivery.json").read_text())
            self.assertEqual(delivery["optimization_budget_continuation"], proof["optimization_budget_continuation"])
            self.assertEqual(delivery["resource_recovery"]["maximum_epochs"], 80)
            self.assertEqual(delivery["effective_maximum_epochs"], 80)
            self.assertEqual([name for name, _ in calls], ["fit", "evaluate", "replay"])
            self.assertEqual((failed / "delivery.json").read_bytes(), before)

    def test_resource_recovery_rejects_changed_budget_or_missing_parent_cost_binding(self):
        for change in ("ceiling", "lineage", "parent_receipt", "parent_payload"):
            with self.subTest(change=change), tempfile.TemporaryDirectory(dir="/tmp/neuronsim", prefix="neuronsim-budget-reject-") as temporary:
                repo, work, failed, proof, software = self.recovery_fixture(Path(temporary))
                software(repo, work)
                runner.write_json(work / "training_configuration.json", CONFIG)
                if change == "ceiling":
                    proof["optimization_budget_continuation"]["total_epochs"] = 100
                elif change == "lineage":
                    proof.pop("optimization_budget_continuation")
                elif change == "parent_receipt":
                    (failed / "prerequisites/budget_parent/delivery.json").unlink()
                self.refresh_recovery_manifest(failed, proof)
                if change == "parent_payload":
                    (failed / "prerequisites/budget_parent/training/run.json").write_text("{}")
                with self.assertRaises(ValueError):
                    runner.stage_recovery(failed, work, 42, "700000", "cpu", "recurrent")

    def test_invalid_parent_rejects_without_staging_or_output_writes(self):
        for fault in ("wrong_seed", "explicit_fixed", "wrong_source", "missing_delivery", "missing_checkpoint_receipt", "wrong_data"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(dir="/tmp/neuronsim", prefix="neuronsim-budget-") as temporary:
                repo, work, campaign, parent, proof, prerequisite, _ = self.fixture(Path(temporary))
                if fault == "wrong_seed":
                    proof["training_seed"] = 41
                    runner.write_json(parent / "delivery.json", proof)
                elif fault == "explicit_fixed":
                    proof["representation"] = "fixed"
                    runner.write_json(parent / "delivery.json", proof)
                elif fault == "wrong_source":
                    (parent / "Libs/population_dynamics/population_trajectory_training.py").write_text("# changed\n")
                elif fault == "missing_delivery":
                    (parent / "delivery.json").unlink()
                else:
                    report_path = parent / "training/run.json"
                    report = json.loads(report_path.read_text())
                    if fault == "missing_checkpoint_receipt":
                        report.pop("latest_checkpoint_sha256")
                    else:
                        report["binding"]["dataset_sha256"] = "e" * 64
                    runner.write_json(report_path, report)
                    row = next(row for row in proof["files"] if row["path"] == "training/run.json")
                    row.update(bytes=report_path.stat().st_size, sha256=runner.digest(report_path))
                    runner.write_json(parent / "delivery.json", proof)
                with mock.patch.dict(os.environ, ENVIRONMENT, clear=True), \
                        mock.patch.object(runner, "prerequisite", side_effect=prerequisite), \
                        mock.patch.object(runner, "stage_software") as stage, mock.patch.object(runner, "archive") as archive:
                    with self.assertRaises(ValueError):
                        runner.run(repo, work, "cpu", budget_parent_campaign=campaign, total_epochs=80)
                stage.assert_not_called()
                archive.assert_not_called()
                self.assertFalse(work.exists())
                self.assertFalse((repo / "Output/neural_population_dynamics/model/trajectory_development/optimization_budget_continuation").exists())


if __name__ == "__main__":
    unittest.main()
