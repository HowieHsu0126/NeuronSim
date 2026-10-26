"""Keep representation comparisons within the accepted population campaign."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import run_population_trajectory_campaign as runner


# Use the complete-run fixture pattern accepted by training relocation job
# 607064. No training, native execution or external data is needed here.
SOURCE_CONFIG = """training_seeds: [41, 42, 43]
representation: recurrent
maximum_epochs: 20
patience: 4
"""
FORECAST_CONFIG = """joint_samples: 16
origins_ms: [1000.0, 3000.0, 5000.0]
horizons_ms: [20.0, 80.0, 500.0]
"""
ENVIRONMENT = dict(SLURM_JOB_ID="representation-fixture", SLURMD_NODENAME="cpu04",
                   SLURM_JOB_PARTITION="cpu", SLURM_JOB_GPUS="")


class PopulationRepresentationCampaignTests(unittest.TestCase):
    def test_cost_point_consumes_actual_evaluator_output_schema_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, target = root / "work", root / "delivered"
            calls = []
            def command(args, **kwargs):
                calls.append(args)
                if "Libs.population_dynamics.population_trajectory_evaluation" in args:
                    self.assertEqual(kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
                    output = Path(args[args.index("--output") + 1])
                    runner.write_json(output / "cost_point.json", dict(
                        status="EVALUATED_OBSERVED_HISTORY_POSTERIOR_COST_POINT",
                        generated_unit_count_values=160000000, all_unit_outputs_saved=True,
                        checkpoint_sha256="a" * 64, hardware=dict(device="cuda")))
                return mock.Mock(returncode=0)
            env = dict(SLURM_JOB_ID="cost-fixture", SLURMD_NODENAME="gpu03",
                       SLURM_JOB_PARTITION="gpu1", SLURM_JOB_GPUS="0")
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(runner, "stage_cost_point", return_value=dict(checkpoint_sha256="a" * 64)), \
                    mock.patch.object(runner.subprocess, "run", side_effect=command):
                self.assertEqual(runner.run_cost_point(root, work, target, root / "software"), 0)
            proof = json.loads((target / "delivery.json").read_text())
            self.assertEqual(proof["status"], "PASS")
            self.assertFalse(proof["training_performed"])
            self.assertFalse(proof["matched_fidelity_qualified"])
            self.assertEqual(len(calls), 2)
            self.assertIn("--cost-point", calls[1])

    def complete_fixture(self, root, representation=None):
        repo, work = root / "repo", root / "work"
        work.mkdir()
        configs = work / "Libs/configs"
        configs.mkdir(parents=True)
        source = configs / runner.CONFIGS[0]
        source.write_text(SOURCE_CONFIG)
        (configs / runner.CONFIGS[1]).write_text(FORECAST_CONFIG)
        calls = []

        def phase(local, name, command, probe):
            calls.append((name, command))
            if name == "fit":
                runner.write_json(local / "training/run.json", dict(
                    status="FITTED_DEVELOPMENT", training_seed=42,
                    fitting_episodes=56, validation_episodes=16,
                    completed_epochs=20, stop_reason="maximum_epochs"))
            elif name == "evaluate":
                runner.write_json(local / "forecasts/evaluation.json", dict(
                    status="EVALUATED_DEVELOPMENT", score_rows=144))
            elif name == "replay":
                runner.write_json(local / "replay.json", dict(status="PASS", verified_origins=48))
            else:
                self.fail("unexpected phase: " + name)
            return 0

        def probe(*args, **kwargs):
            return dict(status="PASS")

        settings = {} if representation is None else {"representation": representation}
        with mock.patch.dict(os.environ, ENVIRONMENT, clear=True), \
                mock.patch.object(runner, "stage_software", return_value=probe) as software, \
                mock.patch.object(runner, "stage_inputs") as inputs, \
                mock.patch.object(runner, "phase", side_effect=phase):
            code = runner.run(repo, work, "cpu", 42, "606139", "606139_1", **settings)
        software.assert_called_once_with(repo, work)
        inputs.assert_called_once_with(repo, work)
        self.assertEqual(code, 0)
        self.assertEqual([name for name, _ in calls], ["fit", "evaluate", "replay"])
        self.assertEqual(source.read_text(), SOURCE_CONFIG)
        self.assertEqual((configs / runner.CONFIGS[1]).read_text(), FORECAST_CONFIG)
        fit_command = calls[0][1]
        self.assertEqual(fit_command[fit_command.index("--seed") + 1], "42")
        self.assertEqual(fit_command[fit_command.index("--config") + 1],
                         "Libs/configs/neural_population_state_trajectory_execution.yaml")
        self.assertIn("--all-origins", calls[2][1])
        return repo, work

    def check_delivery_and_config(self, target, work, representation):
        delivery = json.loads((target / "delivery.json").read_text())
        self.assertEqual(delivery["status"], "PASS")
        self.assertEqual(delivery["representation"], representation)
        self.assertEqual(delivery["training_seed"], 42)
        self.assertEqual(delivery["array_job_id"], "606139")
        self.assertEqual(delivery["replaces_unstarted_job"], "606139_1")
        self.assertEqual(delivery["fit_episodes"], 56)
        self.assertEqual(delivery["validation_episodes"], 16)
        self.assertEqual(delivery["episode_ms"], 6000)
        self.assertEqual(delivery["verified_forecast_origins"], 48)
        self.assertFalse(delivery["scientific_claim_qualified"])
        self.assertFalse(delivery["calibration_or_test_accessed"])
        expected = yaml.safe_load(SOURCE_CONFIG)
        expected.update(device="cpu", cpu_threads=2, representation=representation)
        self.assertEqual(json.loads((work / "training_configuration.json").read_text()), expected)
        self.assertEqual(yaml.safe_load((work / "Libs/configs/neural_population_state_trajectory_execution.yaml").read_text()), expected)
        self.assertEqual(json.loads((target / "training_configuration.json").read_text()), expected)
        self.assertEqual((work / "archive_complete").read_text(), str(target))

    def test_default_recurrent_keeps_existing_campaign_output_and_execution_config(self):
        with tempfile.TemporaryDirectory(dir="/tmp/neuronsim", prefix="neuronsim-representation-") as temp:
            repo, work = self.complete_fixture(Path(temp))
            target = repo / "Output/neural_population_dynamics/model/trajectory_development/run_606139/seed_42"
            self.check_delivery_and_config(target, work, "recurrent")
            with mock.patch.dict(os.environ, ENVIRONMENT, clear=True), self.assertRaises(FileExistsError):
                runner.run(repo, work, "cpu", 42, "606139", "606139_1")

    def test_fixed_representation_has_separate_output_and_preserves_recurrent_evidence(self):
        with tempfile.TemporaryDirectory(dir="/tmp/neuronsim", prefix="neuronsim-representation-") as temp:
            root = Path(temp)
            existing = root / "repo/Output/neural_population_dynamics/model/trajectory_development/run_606139/seed_42/delivery.json"
            existing.parent.mkdir(parents=True)
            existing.write_text('{"existing_recurrent_evidence": true}\n')
            repo, work = self.complete_fixture(root, "fixed")
            target = repo / "Output/neural_population_dynamics/model/trajectory_development/representation_fixed/run_606139/seed_42"
            self.check_delivery_and_config(target, work, "fixed")
            self.assertEqual(existing.read_text(), '{"existing_recurrent_evidence": true}\n')
            with mock.patch.dict(os.environ, ENVIRONMENT, clear=True), self.assertRaises(FileExistsError):
                runner.run(repo, work, "cpu", 42, "606139", "606139_1", representation="fixed")

    def test_cli_forwards_default_and_explicit_representation_with_seed_identity(self):
        for representation in (None, "fixed", "recurrent"):
            with self.subTest(representation=representation):
                arguments = ["campaign", "--repo", "/tmp/neuronsim/representation-cli/repo",
                             "--work", "/tmp/neuronsim/representation-cli/work", "--device", "cpu",
                             "--seed", "42", "--campaign-array-id", "606139", "--replaces-job", "606139_1"]
                if representation is not None:
                    arguments.extend(["--representation", representation])
                original_signature = inspect.signature(runner.run)
                with mock.patch.object(sys, "argv", arguments), \
                        mock.patch.object(runner, "run", return_value=0) as execute:
                    with self.assertRaises(SystemExit) as exit_status:
                        runner.main()
                self.assertEqual(exit_status.exception.code, 0)
                execute.assert_called_once()
                forwarded = original_signature.bind(*execute.call_args.args, **execute.call_args.kwargs)
                forwarded.apply_defaults()
                self.assertEqual(forwarded.arguments["representation"], representation or "recurrent")
                self.assertEqual(forwarded.arguments["seed"], 42)
                self.assertEqual(forwarded.arguments["campaign_array_id"], "606139")
                self.assertEqual(forwarded.arguments["replaces_job"], "606139_1")
                self.assertEqual(forwarded.arguments["device"], "cpu")

    def test_invalid_cli_representation_rejects_before_execution(self):
        arguments = ["campaign", "--repo", "/tmp/neuronsim/representation-cli/repo",
                     "--work", "/tmp/neuronsim/representation-cli/work", "--representation", "unsupported"]
        with mock.patch.object(sys, "argv", arguments), mock.patch.object(runner, "run") as execute:
            with self.assertRaises(SystemExit) as exit_status:
                runner.main()
        self.assertEqual(exit_status.exception.code, 2)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
