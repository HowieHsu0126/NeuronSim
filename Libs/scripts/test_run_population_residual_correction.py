"""Native runner contracts using metadata fixtures and mocked child execution.

These checks do not load a model, inspect a tensor or run scientific evaluation.
Execute the suite on Merlin with the staged runner and its existing helpers.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import run_population_residual_correction as runner
from population_dynamics.population_residual_refit_provenance import FAMILY_SCHEMA


REPO = Path(__file__).resolve().parents[2]
PARENT_SOURCES = {
    "model_implementation_sha256": "population_state.py",
    "observation_implementation_sha256": "refractory_observation.py",
    "loader_implementation_sha256": "native_renewal_data.py",
    "latent_model_implementation_sha256": "poisson_state_space.py",
    "native_counts_implementation_sha256": "native_count_windows.py",
}


class NativeFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-runner-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo, self.fitting, self.generation, self.refits, self.work = (
            self.root / name for name in ("repo", "fitting", "generation", "refits", "work"))
        self.helper = runner.campaign_helper(REPO)
        self.config = self.repo / "Libs/configs/fitting.yaml"
        config = dict(training_seeds=[41, 42, 43], max_epochs=80, source_units=100000)
        self.write(self.config, yaml.safe_dump(config).encode())
        source_names = [*PARENT_SOURCES.values(), "population_residual_memory.py",
                        "population_trajectory_evaluation.py", "population_forecast_replay.py"]
        for name in source_names:
            self.write(self.repo / "Libs/population_dynamics" / name, ("# fixture " + name + "\n").encode())
        for name in ("Libs/__init__.py", "Libs/population_dynamics/__init__.py",
                     "Libs/scripts/run_population_residual_correction.py",
                     "Libs/scripts/run_population_trajectory_campaign.py"):
            self.write(self.repo / name, b"# fixture\n")
        self.helper.software_closure = lambda repo, modules: [
            "Libs/__init__.py", "Libs/population_dynamics/__init__.py",
            *["Libs/population_dynamics/" + name for name in source_names]]
        self.sources = {name: self.helper.digest(self.repo / "Libs/population_dynamics" / name)
                        for name in source_names}
        self.forecast = dict(origins_ms=[1000., 3000., 5000.], horizons_ms=[20., 80., 500.],
                             joint_samples=16, formal_evaluation=False)
        forecast_path = self.generation / "generation_config" / runner.FORECAST
        self.write(forecast_path, yaml.safe_dump(self.forecast).encode())
        cases = [dict(case_id=f"train_block_{block:03d}_familiar_{template}", block=block,
                      template_id=f"familiar_{template}") for block in range(18) for template in range(4)]
        self.helper.write_json(self.fitting / "cohort/train/design.json", dict(acquired_split="train", cases=cases))
        for name in ("train/acquisition.json", "train_fields/dataset.json", "train_fields/episodes.csv"):
            self.write(self.fitting / "cohort" / name, b"{}\n")
        cohort_names = ["train/design.json", "train/acquisition.json", "train_fields/dataset.json", "train_fields/episodes.csv"]
        fitting_names = ["cohort/" + name for name in cohort_names]
        generation_names = ["generation_config/" + runner.FORECAST]
        cohort_rows = [self.row(self.fitting / "cohort", name) for name in cohort_names]
        for case in cases:
            names = [f"train/{case['case_id']}/{name}" for name in ("report.json", "spikes.npz", "initial_parameters.npz")]
            names += [f"train_fields/{case['case_id']}/{name}" for name in ("summary.json", "spatial_fields.npz")]
            root, destination = (self.fitting, fitting_names) if case["block"] < 14 else (self.generation, generation_names)
            for name in names:
                self.write(root / "cohort" / name, (name + "\n").encode())
                destination.append("cohort/" + name)
                cohort_rows.append(self.row(root / "cohort", name))
        self.helper.write_json(self.fitting / "cohort/delivery.json", dict(
            status="VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE", completed_episodes=72, cortical_units=100000,
            episode_duration_ms=6000., source_dt_ms=.1, field_bin_ms=5., native_rng_family="per_edge_philox4_32_10",
            files=cohort_rows))
        fitting_names.append("cohort/delivery.json")
        self.parent_bindings = {}
        for seed in runner.SEEDS:
            parent = self.fitting / f"parent/seed_{seed}"
            self.write(parent / "training/best.pt", f"opaque checkpoint seed {seed}".encode())
            checkpoint_sha = self.helper.digest(parent / "training/best.pt")
            binding = dict(seed=seed, config=dict(cpu_threads=2),
                acquisition_design_sha256=self.helper.digest(self.fitting / "cohort/train/design.json"),
                dataset_sha256=self.helper.digest(self.fitting / "cohort/train_fields/dataset.json"),
                **{key: self.sources[name] for key, name in PARENT_SOURCES.items()})
            self.parent_bindings[seed] = binding
            self.helper.write_json(parent / "training/run.json", dict(status="FITTED_DEVELOPMENT", training_seed=seed,
                best_checkpoint_sha256=checkpoint_sha, binding=binding))
            rows = [self.row(parent, name) for name in ("training/best.pt", "training/run.json")]
            rows.append(dict(self.row(self.generation, "generation_config/" + runner.FORECAST),
                             path="Libs/configs/" + runner.FORECAST))
            self.helper.write_json(parent / "delivery.json", dict(status="PASS", exit_code=0, training_seed=seed,
                source_units=100000, fit_episodes=56, validation_episodes=16, complete_origins=48,
                verified_origins=48, score_rows=144, files=rows))
            fitting_names += [f"parent/seed_{seed}/" + name for name in ("delivery.json", "training/run.json", "training/best.pt")]
        common = dict(schema="native_residual_observation_inputs_v1", training_seeds=[41, 42, 43], source_units=100000,
                      transfer_complete=False, local_payload_checks_performed=False)
        self.helper.write_json(self.fitting / "manifest.json", dict(**common,
            scope="complete_fitting_and_internal_selection", blocks=list(range(14)),
            files=[self.row(self.fitting, name) for name in fitting_names]))
        self.helper.write_json(self.generation / "manifest.json", dict(**common,
            scope="native_generation_additional_inputs", blocks=list(range(14, 18)),
            cohort_delivery_sha256=self.helper.digest(self.fitting / "cohort/delivery.json"),
            forecast_config_sha256=self.helper.digest(forecast_path),
            files=[self.row(self.generation, name) for name in generation_names]))
        for seed in runner.SEEDS:
            for mode in runner.MODES:
                cell = self.refits / f"seed_{seed}" / mode
                self.write(cell / "predictor.npz", f"opaque predictor {seed} {mode}".encode())
                binding = dict(phase="refit", seed=seed, mode=mode, config=config, sources=self.sources,
                    input_manifest_sha256=self.helper.digest(self.fitting / "manifest.json"),
                    config_sha256=self.helper.digest(self.config), selection_sha256="selection-fixture",
                    parent_checkpoint_sha256=self.helper.digest(self.fitting / f"parent/seed_{seed}/training/best.pt"),
                    parent_binding_sha256=hashlib.sha256(json.dumps(self.parent_bindings[seed],
                        sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    fit_cases=[case["case_id"] for case in cases[:56]], validation_cases=[])
                self.helper.write_json(cell / "fit.json", dict(schema="native_residual_observation_fit_v1",
                    status="FITTED_DEVELOPMENT", phase="refit", completed_epochs=43, selected_epoch=43,
                    parent_unchanged=True, scientific_claim_qualified=False, run_id=f"run-{seed}-{mode}",
                    trial_id=f"trial-{seed}-{mode}", predictor_sha256=self.helper.digest(cell / "predictor.npz"), binding=binding))

    @staticmethod
    def write(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def row(self, root, name):
        return dict(path=name, bytes=(root / name).stat().st_size, sha256=self.helper.digest(root / name))

    def stage(self):
        return runner.stage_native(self.helper, self.repo, self.fitting, self.generation,
                                   self.refits, self.config, 41, self.work)

    def test_complete_native_family_stages_parent_frozen_forecast_and_all_reports(self):
        before = (self.fitting / "manifest.json").read_bytes()
        forecast, identities, provenance = self.stage()
        self.assertEqual(forecast, self.forecast)
        self.assertEqual(set(identities), {"current", "history"})
        self.assertEqual(provenance["fit_schema"], FAMILY_SCHEMA)
        self.assertNotIn("fit_job_id", provenance)
        family = json.loads((self.work / "provenance/fit.json").read_text())
        self.assertEqual(len(family["fits"]), 6)
        for row in family["fits"]:
            self.assertEqual(self.helper.digest(self.work / row["path"]), row["sha256"])
        self.assertEqual((self.work / "Libs/configs" / runner.FORECAST).read_bytes(),
                         (self.generation / "generation_config" / runner.FORECAST).read_bytes())
        self.assertEqual(before, (self.fitting / "manifest.json").read_bytes())
        self.assertFalse(json.loads(before)["transfer_complete"])
        self.assertTrue((self.work / "inputs/train/train_block_017_familiar_3/spikes.npz").is_file())

    def test_rejects_changed_additional_source_bytes(self):
        (self.generation / "cohort/train/train_block_014_familiar_0/spikes.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "staged input identity"):
            self.stage()

    def test_rejects_input_manifest_different_from_refit_binding(self):
        path = self.fitting / "manifest.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "fitting input/configuration"):
            self.stage()

    def test_rejects_fitting_config_different_from_refit_binding(self):
        self.config.write_text(self.config.read_text() + "# changed bytes\n")
        with self.assertRaisesRegex(ValueError, "fitting input/configuration"):
            self.stage()

    def test_rejects_changed_runtime_source(self):
        (self.repo / "Libs/population_dynamics/population_residual_memory.py").write_text("# different\n")
        with self.assertRaisesRegex(ValueError, "runtime source differs"):
            self.stage()

    def test_rejects_missing_other_seed_refit(self):
        (self.refits / "seed_43/history/fit.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.stage()

    def test_rejects_other_seed_predictor_corruption(self):
        (self.refits / "seed_43/history/predictor.npz").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "predictor and parent binding"):
            self.stage()

    def test_rejects_changed_parent_binding(self):
        path = self.refits / "seed_42/current/fit.json"
        report = json.loads(path.read_text())
        report["binding"]["parent_binding_sha256"] = "other-parent"
        self.helper.write_json(path, report)
        with self.assertRaisesRegex(ValueError, "predictor and parent binding"):
            self.stage()

    def test_rejects_self_consistent_but_nonparent_forecast(self):
        path = self.generation / "generation_config" / runner.FORECAST
        path.write_text(path.read_text() + "# changed bytes\n")
        manifest_path = self.generation / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["forecast_config_sha256"] = self.helper.digest(path)
        manifest["files"] = [self.row(self.generation, row["path"]) if row["path"].startswith("generation_config/")
                             else row for row in manifest["files"]]
        self.helper.write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "parent.*frozen matched forecast"):
            self.stage()


class NativeExecutionTests(unittest.TestCase):
    def execute(self, root, fail_phase=None, verify_failure=False):
        helper = runner.campaign_helper(REPO)
        commands = []

        def phase(work, name, command, probe):
            commands.append((name, command))
            code = 2 if name == fail_phase else 0
            helper.write_json(work / "phases" / name / "receipt.json", dict(status="FAILED" if code else "PASS", exit_code=code))
            return code

        helper.phase = phase
        identity = dict(backend="merlin", run_id="actual-run-fixture", trial_id="actual-trial-fixture", host="actual-host-fixture")
        with patch.object(runner, "merlin_identity", return_value=identity), \
                patch.object(runner, "campaign_helper", return_value=helper), \
                patch.object(runner, "native_probe", return_value=dict(status="PASS", reasons=[])), \
                patch.object(runner, "stage_native", return_value=({}, dict(current={}, history={}), dict(parent_checkpoint_sha256="parent"))), \
                patch.object(runner, "verify_mode", side_effect=ValueError("incomplete replay") if verify_failure else None,
                             return_value=dict(status="PASS", exit_code=0)) as verify, \
                patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(io.StringIO()):
            code = runner.run_native(root / "repo", root / "fitting", root / "generation", root / "refits",
                root / "config.yaml", 41, root / "work", root / "output", "cuda")
        return code, json.loads((root / "output/delivery.json").read_text()), commands, verify.call_count

    def test_success_runs_both_fresh_evaluations_and_all_origin_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            code, delivery, commands, verifications = self.execute(Path(directory))
        self.assertEqual(code, 0)
        self.assertEqual(delivery["status"], "PASS")
        self.assertEqual(delivery["run_id"], "actual-run-fixture")
        self.assertNotIn("job_id", delivery)
        self.assertEqual(verifications, 2)
        self.assertEqual([name for name, _ in commands], ["current_evaluate", "current_replay", "history_evaluate", "history_replay"])
        for name, command in commands:
            self.assertIn("--correction", command)
            if name.endswith("replay"):
                self.assertIn("--all-origins", command)
            else:
                self.assertEqual(command[command.index("--config") + 1], "Libs/configs/" + runner.FORECAST)

    def test_failed_evaluation_cannot_be_overwritten_by_other_mode_success(self):
        with tempfile.TemporaryDirectory() as directory:
            code, delivery, commands, verifications = self.execute(Path(directory), fail_phase="current_evaluate")
        self.assertEqual((code, delivery["status"]), (2, "FAILED"))
        self.assertEqual(delivery["modes"]["history"]["status"], "PASS")
        self.assertEqual(verifications, 1)
        self.assertNotIn("current_replay", [name for name, _ in commands])

    def test_replay_failure_prevents_overall_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            code, delivery, _, verifications = self.execute(Path(directory), fail_phase="history_replay")
        self.assertEqual((code, delivery["status"]), (2, "FAILED"))
        self.assertEqual(verifications, 1)

    def test_verifier_rejection_prevents_overall_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            code, delivery, _, verifications = self.execute(Path(directory), verify_failure=True)
        self.assertEqual((code, delivery["status"]), (2, "FAILED"))
        self.assertEqual(verifications, 2)

    def test_native_execution_requires_real_run_trial_host(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(runner.platform, "node", return_value="worker"):
            with self.assertRaisesRegex(ValueError, "actual Merlin"):
                runner.merlin_identity()
        with patch.dict(os.environ, RH2_JOB_RUN_ID="run", ARNOLD_TRIAL_ID="trial"), \
                patch.object(runner.platform, "node", return_value="mgmt2"):
            with self.assertRaisesRegex(ValueError, "actual Merlin"):
                runner.merlin_identity()


class CommandLineTests(unittest.TestCase):
    common = ["--repo", "/repo", "--work", "/tmp/work", "--output", "/output", "--seed", "41"]

    def test_existing_legacy_cli_remains_available(self):
        with patch.object(runner, "run", return_value=0) as legacy, \
                patch.object(runner, "run_native") as native, self.assertRaises(SystemExit) as raised:
            runner.main(self.common + ["--preparation", "/prepared", "--fit-job", "123"])
        self.assertEqual(raised.exception.code, 0)
        legacy.assert_called_once()
        native.assert_not_called()

    def test_explicit_native_cli_routes_without_slurm_fit_id(self):
        with patch.object(runner, "run") as legacy, patch.object(runner, "run_native", return_value=0) as native, \
                self.assertRaises(SystemExit) as raised:
            runner.main(self.common + ["--native-refits", "/refits", "--fitting-inputs", "/fitting",
                "--generation-inputs", "/generation", "--fitting-config", "/config.yaml"])
        self.assertEqual(raised.exception.code, 0)
        native.assert_called_once()
        legacy.assert_not_called()

    def test_rejects_mixed_native_and_legacy_arguments(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            runner.main(self.common + ["--native-refits", "/refits", "--fit-job", "123"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
