"""Delivered-data integration and rejection checks for the mean comparison."""
import csv
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from .braid_population_comparison import (compare, _training_cost, DeliveredCell,
    _population_cost_receipt, _population_origin_cost_columns, _score_table)
from .lif_neural_field import sampled_mean_field_mse
from .provenance import sha256_file


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def delivery(cell, seed):
    rows = [dict(path=str(path.relative_to(cell)), bytes=path.stat().st_size,
                 sha256=sha256_file(path)) for path in cell.rglob("*")
            if path.is_file() and path.name != "delivery.json"]
    write_json(cell / "delivery.json", dict(status="PASS", exit_code=0, training_seed=seed,
                                            scientific_claim_qualified=False, files=rows))


def fixture(root):
    source = dict(acquisition_design_sha256="a"*64, acquisition_ledger_sha256="b"*64,
                  dataset_sha256="c"*64, episode_table_sha256="d"*64,
                  episode_bindings={f"train_block_{b:03d}_familiar_{t}": {} for b in range(18) for t in range(4)},
                  native_source_identity=dict(native_library_sha256="e"*64, dt_ms=.1))
    config = dict(origins_ms=[1000., 3000., 5000.], horizons_ms=[20., 80., 500.], joint_samples=16)
    fields = np.array(["field_A", "field_B"])
    units = np.array([50000, 50000])
    sd = np.array([1., 2.])
    for seed in (41, 42, 43):
        b, p = (root / name / f"seed_{seed}" for name in ("braid", "population"))
        for cell in (b, p):
            (cell / "forecasts").mkdir(parents=True)
        np.savez(p / "forecasts/fitting_field_scale.npz", standard_deviation_hz=sd,
                 unit_count=units, field_ids=fields)
        common = dict(status="FITTED_DEVELOPMENT", training_seed=seed, stop_reason="epoch_budget",
                      completed_epochs=20, best_epoch=19, parameter_count=10,
                      elapsed_seconds=100., peak_process_rss_bytes=1000)
        write_json(b / "training/run.json", dict(common, fit_episodes=56, validation_episodes=16))
        write_json(b / "training/committed.json", dict(checkpoint="synthetic_checkpoint",
                   binding=dict(dataset=source,training_seed=seed)))
        p_binding = dict(source, config=dict(fit_blocks=list(range(14)), validation_blocks=list(range(14,18))))
        write_json(p / "training/run.json", dict(common, fitting_episodes=56, validation_episodes=16,
            binding=p_binding, representation="recurrent", best_checkpoint_sha256="f"*64,
            model_configuration=dict(unit_ids=list(range(100000)), field_names=fields.tolist(),
                unit_field_index=[0]*50000+[1]*50000, bin_seconds=.005)))
        b_binding = dict(cohort=source, origins_ms=config["origins_ms"], horizons_ms=config["horizons_ms"],
                         source_unit_ids=list(range(100000)), source_field_ids=fields.tolist(),
                         fitting_scale_sha256=sha256_file(p / "forecasts/fitting_field_scale.npz"),
                         training_committed_sha256=sha256_file(b / "training/committed.json"))
        b_rows, p_rows, costs, b_artifacts, p_artifacts = [], [], [], {}, {}
        for block, template, origin in itertools.product(range(14,18), range(4), config["origins_ms"]):
            case = f"train_block_{block:03d}_familiar_{template}"
            name = f"{case}_origin_{origin:g}.npz"
            truth = np.ones((100,2), dtype=np.int64)
            samples = np.ones((16,100,2), dtype=np.int64)
            samples[::2] = 0; samples[1::2] = 2
            means = truth.astype(float) + .5
            np.savez(b / "forecasts" / name, expected_field_counts=means,
                     observed_field_counts=truth.astype(float), field_ids=fields, unit_count=units)
            np.savez(p / "forecasts" / name, sampled_field_counts=samples, observed_field_counts=truth)
            b_artifacts[name] = sha256_file(b / "forecasts" / name)
            p_artifacts[name] = sha256_file(p / "forecasts" / name)
            for horizon in config["horizons_ms"]:
                identity = dict(seed=seed, block=block, case_id=case, origin_ms=origin, horizon_ms=horizon)
                b_rows.append(dict(identity, method="braid_mean", unit_bin_mean_mse=.25,
                    fitting_standardized_field_rate_mse=float(np.mean((.5/(units*.005*sd))**2))))
                p_rows.append(dict(identity, method="population", unit_bin_mean_mse_corrected=-.01,
                                   inference_seconds=1., full_unit_generation_seconds=2., field_projection_seconds=3.))
            costs.append(dict(block=block, case_id=case, origin_ms=origin, generated_unit_mean_values=10000000,
                input_loading_seconds=1., inference_seconds=2., full_unit_mean_forecast_seconds=3.,
                field_projection_and_scoring_seconds=4., serialization_and_hash_seconds=5., origin_total_seconds=15.))
        write_rows(b / "forecasts/scores.csv", b_rows)
        write_rows(p / "forecasts/scores.csv", p_rows)
        write_rows(b / "forecasts/origin_costs.csv", costs)
        for name in ("scores.csv", "origin_costs.csv"):
            b_artifacts[name] = sha256_file(b / "forecasts" / name)
        b_eval = dict(status="EVALUATED_DEVELOPMENT", training_seed=seed, score_rows=144, origins=48,
                      binding=b_binding, artifacts=b_artifacts)
        p_eval = dict(status="EVALUATED_DEVELOPMENT", training_seed=seed, score_rows=144,
                      acquisition_design_sha256=source["acquisition_design_sha256"],
                      dataset_sha256=source["dataset_sha256"], checkpoint_sha256="f"*64,
                      configuration=config, formal_test_accessed=False, artifacts=p_artifacts,
                      scores_sha256=sha256_file(p / "forecasts/scores.csv"))
        for cell, evaluation, replay_name in ((b,b_eval,"forecasts/replay.json"),(p,p_eval,"replay.json")):
            write_json(cell / "forecasts/evaluation.json", evaluation)
            replay = dict(status="PASS", training_seed=seed, verified_origins=48, verified_score_rows=144,
                          evaluation_sha256=sha256_file(cell / "forecasts/evaluation.json"))
            replay.update(dict(all_origin_fields_boundary_and_metrics_exact=True) if cell==b else
                          dict(all_origin_fields_and_latents_exact=True, all_forecast_unit_metrics_replayed=True))
            write_json(cell / replay_name, replay)
            delivery(cell, seed)



def prior_population_fixture(root):
    """Attach real delivered prior ancestry to the dedicated comparison fixtures."""
    import shutil
    import torch
    from .population_forecast_analysis import PRIOR_REFIT_SCHEMA, PRIOR_REFIT_OBJECTIVE
    population = root / "population"
    for seed in (41, 42, 43):
        cell = population / f"seed_{seed}"
        initial = json.loads((cell / "training/run.json").read_text())
        model = {"encoder.weight": torch.tensor([1.]), "unit_loading": torch.tensor([2.]),
                 "local_weight": torch.tensor([3.])}
        config = dict(initial["binding"]["config"], schema="neural_population_state_trajectory_training_v1",
            model_family="population", representation="recurrent", maximum_epochs=20, early_stopping_patience=4)
        binding = dict(initial["binding"], config=config, seed=seed, observation_implementation_sha256="fixture-observation")
        original = dict(initial, binding=binding, model_family="population", attempt=0, elapsed_seconds=100.,
            effective_maximum_epochs=20, continuation=None, hardware=dict(device="cpu"), trainer_sources={"trainer": "original"})
        constructor = initial["model_configuration"]
        parent_base = root / "ancestry/base" / f"seed_{seed}"
        parent = root / "ancestry/parent" / f"seed_{seed}"

        def rows(folder):
            return [dict(path=str(path.relative_to(folder)), bytes=path.stat().st_size, sha256=sha256_file(path))
                    for path in folder.rglob("*") if path.is_file() and path != folder / "delivery.json"]

        def save_train(folder, run, payload):
            (folder / "training").mkdir(parents=True, exist_ok=True)
            torch.save(payload, folder / "training/best.pt")
            shutil.copyfile(folder / "training/best.pt", folder / "training/latest.pt")
            run.update(best_checkpoint_sha256=sha256_file(folder / "training/best.pt"),
                       latest_checkpoint_sha256=sha256_file(folder / "training/latest.pt"))
            write_json(folder / "training/run.json", run)

        def save_parent(folder, **extra):
            write_json(folder / "delivery.json", dict(status="PASS", exit_code=0, training_seed=seed,
                scientific_claim_qualified=False, calibration_or_test_accessed=False,
                core_neurons=100000, complete_episodes=72, episode_ms=6000, files=rows(folder), **extra))

        original_checkpoint = dict(binding=binding, model_configuration=constructor, model=model, next_epoch=20, continuation=None)
        save_train(parent_base, original, original_checkpoint)
        save_parent(parent_base)
        continuation = dict(schema="neural_population_trajectory_budget_continuation_v1", base_maximum_epochs=20,
            total_epochs=80, parent_effective_maximum_epochs=20, parent_completed_epochs=20,
            inherited_early_stopping_patience=4, parent_binding=binding,
            parent_checkpoint_sha256=original["latest_checkpoint_sha256"],
            parent_artifact_sha256={"run.json": sha256_file(parent_base / "training/run.json"),
                                   "best.pt": original["best_checkpoint_sha256"]},
            parent_source_hashes=original["trainer_sources"], current_source_hashes=original["trainer_sources"], source_migration={})
        parent_run = dict(original, completed_epochs=80, best_epoch=79, effective_maximum_epochs=80,
                          continuation=continuation, attempt=1, elapsed_seconds=80.)
        save_train(parent, parent_run, dict(original_checkpoint, next_epoch=80, continuation=continuation))
        for name in ("delivery.json", "training/run.json"):
            target = parent / "prerequisites/budget_parent" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent_base / name, target)
        save_parent(parent, optimization_budget_continuation=dict(parent=str(parent_base),
            parent_delivery_sha256=sha256_file(parent_base / "delivery.json"),
            parent_checkpoint_sha256=original["latest_checkpoint_sha256"], parent_completed_epochs=20, total_epochs=80))
        initialization = dict(mode="selected_parent_prior_refit", parent_training="/removed/previous/parent_training",
            parent_checkpoint_sha256=parent_run["best_checkpoint_sha256"],
            parent_report_sha256=sha256_file(parent / "training/run.json"),
            parent_selected_epoch=79, parent_completed_epochs=80, parent_binding=binding,
            trainable_parameter_names=["local_weight", "field_mix_logits", "control_weight", "transition_bias",
                                       "decay_logit", "process_log_variance"], optimizer_and_random_streams_reinitialized=True)
        prior_config = dict(config, schema=PRIOR_REFIT_SCHEMA, maximum_epochs=80, early_stopping_patience=12,
            training_sequence="complete_episode_causal_posterior_cache", observation_objective=PRIOR_REFIT_OBJECTIVE,
            prior_origins_ms=[1000, 3000, 5000], prior_horizons_ms=[5, 20, 80, 500],
            prior_selection="fixed_draw_validation_multistep_kl", prior_target_gradient="stopped",
            parent_checkpoint_sha256={str(s): parent_run["best_checkpoint_sha256"] for s in (41, 42, 43)})
        run = dict(original, binding=dict(binding, config=prior_config), completed_epochs=80, best_epoch=79,
            effective_maximum_epochs=80, attempt=1, elapsed_seconds=6., trainer_sources={"trainer": "prior"},
            units=100000, training_objective=PRIOR_REFIT_OBJECTIVE, selection_metric="validation_multistep_prior_kl",
            refit_initialization=initialization, complete_episode_updates=False, complete_source_episode_encoding=True,
            posterior_targets_fixed=True, observation_parameters_refitted=False,
            observation_likelihood_evaluated_during_refit=False, fitting_origin_windows=168, validation_origin_windows=48,
            frozen_parameters_unchanged_in_attempt=True, calibration_outcomes_accessed=False, test_outcomes_accessed=False)
        save_train(cell, run, dict(binding=run["binding"], model_configuration=constructor,
            model=dict(model, local_weight=torch.tensor([4.])), next_epoch=80, continuation=None, initialization=initialization))
        write_json(cell / "training/attempts/attempt_000.json", dict(run, attempt=0, elapsed_seconds=4.))
        (cell / "parent_training").mkdir()
        for name in ("run.json", "best.pt"):
            shutil.copyfile(parent / "training" / name, cell / "parent_training" / name)
        parent_rows = json.loads((parent / "delivery.json").read_text())["files"]
        write_json(cell / "prerequisites/prior_parent.json", dict(parent_cell=str(parent),
            delivery_sha256=sha256_file(parent / "delivery.json"),
            files=[row for row in parent_rows if row["path"] in ("training/run.json", "training/best.pt")]))
        evaluation = json.loads((cell / "forecasts/evaluation.json").read_text())
        evaluation["checkpoint_sha256"] = run["best_checkpoint_sha256"]
        write_json(cell / "forecasts/evaluation.json", evaluation)
        replay = json.loads((cell / "replay.json").read_text())
        replay["evaluation_sha256"] = sha256_file(cell / "forecasts/evaluation.json")
        write_json(cell / "replay.json", replay)
        write_json(cell / "delivery.json", dict(status="PASS", exit_code=0, training_seed=seed,
            scientific_claim_qualified=False, parent_cell=str(parent), parent_delivery_sha256=sha256_file(parent / "delivery.json"),
            files=rows(cell)))
    return population


class ComparisonChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_cli_and_negative_correction(self):
        import subprocess
        import sys
        output = self.root / "complete"
        child = subprocess.run([sys.executable, "-m", "Libs.population_dynamics.braid_population_comparison",
            "--braid-campaign", str(self.root / "braid"), "--population-campaign", str(self.root / "population"),
            "--output", str(output)], capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)
        report = json.loads((output / "comparison.json").read_text())
        self.assertEqual((report["score_rows"],report["origin_cost_rows"]),(432,144))
        self.assertFalse(report["samples_regenerated"])
        self.assertFalse(report["scientific_claim_qualified"])
        with (output / "mean_comparison_by_block.csv").open() as stream:
            blocks = list(csv.DictReader(stream))
        self.assertEqual(len(blocks),36)
        self.assertEqual({int(row["block"]) for row in blocks},{14,15,16,17})
        self.assertTrue(all(int(row["matched_conditions"])==12 for row in blocks))
        with (output / "mean_comparison_all_seeds.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertLess(float(rows[0]["population_field_mean_mse_corrected"]),0)
        self.assertEqual(float(rows[0]["population_unit_mean_mse_corrected"]),-.01)
        self.assertAlmostEqual(float(rows[0]["population_minus_braid_unit_mean_mse"]),-.26)
        with (output / "costs_by_origin.csv").open() as stream:
            costs = list(csv.DictReader(stream))
        self.assertTrue(all(row["population_origin_total_seconds"] == "" for row in costs))
        self.assertTrue(all("unmeasured" in row["cost_scope"] for row in costs))
        self.assertEqual(report["population_measured_evaluation_cost_seeds"], [])
        self.assertFalse(report["speedup_qualified"])
        self.assertFalse(report["full_end_to_end_cost_matched"])

    def measured_fixture(self, seed=41):
        from .test_population_forecast_analysis import _measured_report
        cell = self.root / f"population/seed_{seed}"
        run = json.loads((cell / "training/run.json").read_text())
        evaluation = json.loads((cell / "forecasts/evaluation.json").read_text())
        scores = _score_table(cell / "forecasts/scores.csv", seed, "population")
        validation = [dict(block=block, case_id=f"train_block_{block:03d}_familiar_{template}")
                      for block, template in itertools.product(range(14, 18), range(4))]
        run["binding"]["config"]["training_sequence"] = "complete_six_second_episode"
        run["binding"]["native_source_identity"]["dt_ms"] = .1
        run["binding"]["episode_bindings"] = {case["case_id"]: {} for case in validation}
        source = SimpleNamespace(validation=validation, bin_ms=5., unit_ids=range(100000),
            field_ids=["field_A", "field_B"], source_identity=dict(dt_ms=.1),
            truth={case["case_id"]: range(1200) for case in validation})
        numeric_rows = [dict(case_id=case, block=block, origin_ms=origin, horizon_ms=horizon,
            inference_seconds=1., full_unit_generation_seconds=2., field_projection_seconds=3.,
            peak_gpu_allocated_bytes=None) for block, case, origin, horizon in scores]
        evaluation = _measured_report(source, evaluation["configuration"], numeric_rows, evaluation)
        evaluation["configuration"]["field_metric_dtype"] = "float64"
        for origin in evaluation["origin_costs"]:
            origin["origin_total_seconds"] = 7.
        evaluation["elapsed_seconds"] = 400.
        evaluation["function_total_seconds"] = 402.
        return run, evaluation, scores

    def test_measured_costs_preserve_once_per_episode_charge_and_order(self):
        run, evaluation, scores = self.measured_fixture()
        receipt = _population_cost_receipt(evaluation, run, scores)
        first = _population_origin_cost_columns(receipt, "train_block_014_familiar_0", 1000.)
        second = _population_origin_cost_columns(receipt, "train_block_014_familiar_0", 3000.)
        self.assertEqual(first["population_episode_preparation_seconds"], .4)
        self.assertIsNone(second["population_episode_preparation_seconds"])
        self.assertAlmostEqual(first["population_other_origin_costs"], .11)
        self.assertEqual(first["population_origin_total_seconds"], 7.)
        self.assertEqual(first["population_generated_unit_count_values"], 160000000)
        self.assertIn("fixture", first["population_evaluation_hardware"])
        evaluation["origin_costs"].reverse()
        evaluation["episode_costs"].reverse()
        reordered = _population_cost_receipt(evaluation, run, scores)
        self.assertEqual(_population_origin_cost_columns(reordered, "train_block_014_familiar_0", 1000.), first)
        charge = sum(_population_origin_cost_columns(reordered, case, origin)["population_episode_preparation_seconds"] or 0
                     for block, case, origin, horizon in scores if horizon == 500.)
        self.assertAlmostEqual(charge, 16 * .4)

    def test_incomplete_or_mismatched_measured_costs_rejected(self):
        for failure in ("missing_episode", "wrong_timer", "wrong_bin_count", "wrong_source", "wrong_field_dtype"):
            with self.subTest(failure=failure):
                run, evaluation, scores = self.measured_fixture()
                if failure == "missing_episode": evaluation["episode_costs"].pop()
                elif failure == "wrong_timer": evaluation["origin_costs"][0]["inference_seconds"] = .5
                elif failure == "wrong_bin_count": evaluation["episode_costs"][0]["native_bins"] = 1199
                elif failure == "wrong_field_dtype": evaluation["configuration"]["field_metric_dtype"] = "float32"
                else: run["binding"]["episode_bindings"].pop("train_block_014_familiar_0")
                with self.assertRaises(ValueError): _population_cost_receipt(evaluation, run, scores)

    def test_mixed_measured_legacy_complete_consumer(self):
        cell = self.root / "population/seed_41"
        paths = [cell / name for name in ("training/run.json", "forecasts/evaluation.json", "replay.json", "delivery.json")]
        backups = {path: path.read_bytes() for path in paths}
        try:
            run, evaluation, scores = self.measured_fixture()
            # Cohort/source comparison must remain valid after fixture enrichment.
            braid = self.root / "braid/seed_41"
            extra = [braid / name for name in ("forecasts/evaluation.json", "forecasts/replay.json", "training/committed.json", "delivery.json")]
            backups.update({path: path.read_bytes() for path in extra})
            b_eval = json.loads(extra[0].read_text())
            b_eval["binding"]["cohort"]["native_source_identity"] = run["binding"]["native_source_identity"]
            b_eval["binding"]["cohort"]["episode_bindings"] = run["binding"]["episode_bindings"]
            committed = json.loads(extra[2].read_text())
            committed["binding"]["dataset"] = b_eval["binding"]["cohort"]
            write_json(extra[2], committed)
            b_eval["binding"]["training_committed_sha256"] = sha256_file(extra[2])
            write_json(extra[0], b_eval)
            b_replay = json.loads(extra[1].read_text()); b_replay["evaluation_sha256"] = sha256_file(extra[0])
            write_json(extra[1], b_replay); delivery(braid, 41)
            write_json(paths[0], run); write_json(paths[1], evaluation)
            replay = json.loads(paths[2].read_text()); replay["evaluation_sha256"] = sha256_file(paths[1])
            write_json(paths[2], replay); delivery(cell, 41)
            output = self.root / "mixed_costs"
            result = compare(self.root / "braid", self.root / "population", output)
            self.assertEqual(result["population_measured_evaluation_cost_seeds"], [41])
            self.assertFalse(result["speedup_qualified"])
            with (output / "costs_by_origin.csv").open() as stream: costs = list(csv.DictReader(stream))
            new = [row for row in costs if int(row["seed"]) == 41]
            old = [row for row in costs if int(row["seed"]) != 41]
            self.assertEqual(len(new), 48)
            self.assertTrue(all(row["population_origin_total_seconds"] == "7.0" for row in new))
            self.assertTrue(all(row["population_origin_total_seconds"] == "" for row in old))
            self.assertAlmostEqual(sum(float(row["population_episode_preparation_seconds"] or 0) for row in new), 6.4)
            with (output / "population_evaluation_costs.csv").open() as stream: totals = list(csv.DictReader(stream))
            self.assertEqual(totals[0]["setup_seconds"], "1.0")
            self.assertTrue(all(row["function_total_seconds"] == "" for row in totals[1:]))
        finally:
            for path, value in backups.items(): path.write_bytes(value)

    def test_bound_corruption_rejected_before_output(self):
        cell = self.root / "braid/seed_41"
        path = cell / "forecasts/scores.csv"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"changed")
            output = self.root / "corrupt"
            with self.assertRaisesRegex(ValueError,"input changed"):
                compare(self.root / "braid",self.root / "population",output)
            self.assertFalse(output.exists())
        finally:
            path.write_bytes(original)

    def test_different_truth_rejected(self):
        cell = self.root / "population/seed_41"
        path = cell / "forecasts/train_block_014_familiar_0_origin_1000.npz"
        evaluation = cell / "forecasts/evaluation.json"
        proof = cell / "delivery.json"
        backups = {p:p.read_bytes() for p in (path,evaluation,proof)}
        try:
            with np.load(path) as values:
                samples = values["sampled_field_counts"].copy()
                truth = values["observed_field_counts"].copy()
            truth[0,0] = 3
            np.savez(path,sampled_field_counts=samples,observed_field_counts=truth)
            record = json.loads(evaluation.read_text())
            record["artifacts"][path.name] = sha256_file(path)
            write_json(evaluation,record)
            # Keep replay's evaluation binding valid to reach the truth check.
            replay_path = cell / "replay.json"
            backups[replay_path] = replay_path.read_bytes()
            replay = json.loads(replay_path.read_text())
            replay["evaluation_sha256"] = sha256_file(evaluation)
            write_json(replay_path,replay);delivery(cell,41)
            with self.assertRaisesRegex(ValueError,"truth differs"):
                compare(self.root / "braid",self.root / "population",self.root / "different")
        finally:
            for p,data in backups.items():p.write_bytes(data)

    def test_missing_registered_seed_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            compare(empty,self.root / "population",self.root / "missing")

    def test_prepared_launcher_complete_flow(self):
        import os
        import subprocess
        import sys
        repo = Path(__file__).parents[2]
        output = self.root / "launcher_output"
        env = dict(os.environ, SLURMD_NODENAME="cpu04", SLURM_JOB_PARTITION="cpu",
                   SLURM_JOB_ID="fixture", SLURM_JOB_GPUS="", SLURM_STEP_GPUS="")
        child = subprocess.run([sys.executable, str(repo / "Libs/scripts/run_braid_population_comparison.py"),
            "--repo",str(repo),"--braid-campaign",str(self.root / "braid"),
            "--population-campaign",str(self.root / "population"),"--output",str(output)],
            env=env, capture_output=True, text=True)
        self.assertEqual(child.returncode,0,child.stdout+child.stderr)
        proof = json.loads((output / "run_fixture/delivery.json").read_text())
        self.assertEqual(proof["status"],"PASS")
        report = json.loads((output / "run_fixture/comparison/comparison.json").read_text())
        self.assertEqual(report["score_rows"],432)

    def test_interrupted_attempt_cost_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = dict(attempt=1,training_seed=42,binding={"source":"a"},elapsed_seconds=9.,
                       parameter_count=10,completed_epochs=20,best_epoch=19,stop_reason="epoch_budget",
                       peak_process_rss_bytes=100)
            write_json(root / "training/attempts/attempt_000.json",
                       dict(attempt=0,training_seed=42,binding=run["binding"],elapsed_seconds=7.))
            delivery(root,42)
            result = _training_cost(DeliveredCell(root,42),run,"population_recurrent")
            self.assertEqual(result["recorded_training_seconds"],16.)
            self.assertEqual(result["training_attempts"],2)


if __name__ == "__main__":
    unittest.main()

def test_dedicated_braid_prior_comparison_keeps_source_metrics_and_inherited_cost(tmp_path):
    fixture(tmp_path)
    population = prior_population_fixture(tmp_path)
    result = compare(tmp_path / "braid", population, tmp_path / "prior_comparison")
    assert result["candidate_method"] == "population_recurrent_prior_refit"
    assert result["score_rows"] == 432 and result["distribution_comparison_performed"] is False
    assert result["samples_regenerated"] is False and result["speedup_qualified"] is False
    assert result["population_training_bindings"]["41"]["prior_refit_lineage"]["parent_completed_epochs"] == 80
    with (tmp_path / "prior_comparison/training_costs.csv").open() as stream:
        costs = [row for row in csv.DictReader(stream) if row["method"] == "population_recurrent_prior_refit"]
    assert len(costs) == 3 and all(float(row["recorded_training_seconds"]) == 190. for row in costs)
    assert all(float(row["parent_training_elapsed_seconds"]) == 180. for row in costs)
    assert all(row["launcher_staging_seconds"] == "" for row in costs)
    with (tmp_path / "prior_comparison/mean_comparison_all_seeds.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 432 and all(row["candidate_method"] == result["candidate_method"] for row in rows)
    assert all(float(row["population_unit_mean_mse_corrected"]) == -.01 for row in rows)

def test_prior_cost_receipt_preserves_complete_native_measurement_and_rejects_partial_encoding(tmp_path):
    import pytest
    from .test_population_forecast_analysis import _measured_report
    fixture(tmp_path)
    population = prior_population_fixture(tmp_path)
    cell = population / "seed_41"
    run = json.loads((cell / "training/run.json").read_text())
    evaluation = json.loads((cell / "forecasts/evaluation.json").read_text())
    evaluation["configuration"]["field_metric_dtype"] = "float64"
    scores = _score_table(cell / "forecasts/scores.csv", 41, "population")
    validation = [dict(block=block, case_id=f"train_block_{block:03d}_familiar_{template}")
                  for block, template in itertools.product(range(14, 18), range(4))]
    source = SimpleNamespace(validation=validation, bin_ms=5., unit_ids=range(100000),
        field_ids=["field_A", "field_B"], source_identity=dict(dt_ms=.1),
        truth={case["case_id"]: range(1200) for case in validation})
    numeric = [dict(case_id=case, block=block, origin_ms=origin, horizon_ms=horizon,
        inference_seconds=1., full_unit_generation_seconds=2., field_projection_seconds=3.,
        peak_gpu_allocated_bytes=None) for block, case, origin, horizon in scores]
    evaluation = _measured_report(source, evaluation["configuration"], numeric, evaluation)
    for origin in evaluation["origin_costs"]: origin["origin_total_seconds"] = 7.
    evaluation["elapsed_seconds"], evaluation["function_total_seconds"] = 400., 402.
    receipt = _population_cost_receipt(evaluation, run, scores)
    assert len(receipt["measured_evaluation_costs"]["origin_costs"]) == 48
    assert _population_origin_cost_columns(receipt, validation[0]["case_id"], 1000.)["population_generated_unit_count_values"] == 160000000
    run["complete_source_episode_encoding"] = False
    with pytest.raises(ValueError, match="complete source encoding"):
        _population_cost_receipt(evaluation, run, scores)


def test_braid_mixture_comparison_uses_registered_profile_and_full_inherited_cost(tmp_path):
    from .test_population_forecast_analysis import _set_prior_fixture_profile
    from .population_forecast_analysis import PRIOR_MIXTURE_OBJECTIVE
    fixture(tmp_path)
    population = prior_population_fixture(tmp_path)
    for seed in (41, 42, 43):
        _set_prior_fixture_profile(population / f"seed_{seed}", mixture=True)
    output = tmp_path / "mixture_comparison"
    report = compare(tmp_path / "braid", population, output)
    assert report["candidate_method"] == "population_recurrent_predictive_mixture"
    assert report["score_rows"] == 432 and report["scientific_claim_qualified"] is False
    assert all(row["training_objective"] == PRIOR_MIXTURE_OBJECTIVE
               and row["prediction_particles"] == 8 and row["posterior_target_draws"] == 4
               for row in report["population_training_bindings"].values())
    with (output / "training_costs.csv").open() as stream:
        costs = [row for row in csv.DictReader(stream) if row["method"] == report["candidate_method"]]
    assert len(costs) == 3 and all(float(row["recorded_training_seconds"]) == 190. for row in costs)


def test_braid_comparison_rejects_mixed_prior_objectives_across_seeds(tmp_path):
    import pytest
    from .test_population_forecast_analysis import _set_prior_fixture_profile
    fixture(tmp_path)
    population = prior_population_fixture(tmp_path)
    _set_prior_fixture_profile(population / "seed_41", mixture=True)
    output = tmp_path / "mixed_objectives"
    with pytest.raises(ValueError, match="same prior objective"):
        compare(tmp_path / "braid", population, output)
    assert not output.exists()
