"""Shared output budgets, elementwise identity replay and inference causality."""
import json
import csv
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from . import poisson_state_space
from .poisson_state_space import ControlledPoissonLDS, LatentRandomStreams
from .population_trajectory_evaluation import (
    _evaluation_hardware, _peak_process_rss_bytes, evaluate, sample_poisson_counts,
)
from .population_forecast_replay import replay
from .provenance import sha256_file


class ObservedEpisodeFixture:
    """Small explicit observation-boundary substitute, not simulated evidence."""
    def __init__(self, episode, fields, **kwargs):
        self.unit_ids = np.array([11, 17, 29])
        self.unit_field_index = np.array([0, 1, 0])
        self.minimum_interval_ticks = np.array([2, 3, 2])
        self.field_ids = np.array(['["L23E",0,0]', '["L23I",0,0]'])
        self.source_identity = {"scope": "explicit_test_observation_boundary", "graph_manifest_sha256": "a" * 64}
        self.binding = dict(episode_report_sha256=sha256_file(episode / "report.json"),
                            field_summary_sha256=sha256_file(fields / "summary.json"))
        self.bin_ms, self.dt_ms, self.bin_ticks, self.duration_ticks = 5., 1., 5, 60

    def window(self, start, stop):
        n = (stop - start) // self.bin_ticks
        counts = np.zeros((n, 3), dtype=np.float32)
        counts[::2, 0] = 1
        counts[1::2, 1] = 1
        return counts, np.zeros((n, 1), dtype=np.float32)

    def ready_at(self, tick):
        return np.zeros(3, dtype=np.int64)


class MatchedEvaluationTests(unittest.TestCase):
    def model(self):
        return ControlledPoissonLDS([11, 17, 29], 1, 2, 3, .005, initialization_seed=41).double()

    def test_count_only_baseline_matches_public_draws_and_all_rng_states(self):
        model = self.model()
        history = dict(counts=np.ones((2, 4, 3)), stimulus=np.zeros((2, 4, 1)))
        state = model.infer_state(history)
        future = np.ones((2, 6, 1))
        a_rng = LatentRandomStreams.from_seeds(301, 302, 303)
        b_rng = LatentRandomStreams.from_seeds(301, 302, 303)
        complete = model.simulate(state, future, a_rng, 7)
        counts, latent, initial = sample_poisson_counts(model, state, future, b_rng, 7)
        for a, b in [(counts, complete.counts), (latent, complete.latent_states), (initial, complete.initial_latent_states)]:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for name in ("initial", "dynamics", "observation"):
            torch.testing.assert_close(getattr(a_rng, name).get_state(), getattr(b_rng, name).get_state(), rtol=0, atol=0)

    def test_runtime_metadata_preserves_random_state(self):
        model = self.model()
        before = torch.get_rng_state().clone()
        hardware = _evaluation_hardware(model, torch.device("cpu"))
        peak = _peak_process_rss_bytes()
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(hardware["device"], "cpu")
        self.assertEqual(hardware["model_parameter_devices"], ["cpu"])
        self.assertEqual(hardware["model_parameter_dtypes"], ["torch.float64"])
        self.assertGreater(peak, 0)

    def fixture(self, root):
        acquisition, fields, training = [root / name for name in ("acquisition", "fields", "training")]
        for p in (acquisition, fields, training):
            p.mkdir()
        cases = [dict(case_id=f"train_block_{i:03d}_familiar_0", block=i, split="train",
                      family="trajectory", template_id="familiar_0") for i in (0, 1)]
        receipts = []
        for case in cases:
            native = acquisition / case["case_id"]
            view = fields / case["case_id"]
            native.mkdir(); view.mkdir()
            (native / "report.json").write_text(json.dumps({"test_case": case["case_id"]}))
            (view / "summary.json").write_text(json.dumps({"test_view": case["case_id"]}))
            receipts.append(dict(case_id=case["case_id"], split="train",
                                 raw_report_sha256=sha256_file(native / "report.json"),
                                 field_summary_sha256=sha256_file(view / "summary.json")))
        (acquisition / "design.json").write_text(json.dumps(dict(acquired_split="train", cases=cases, graph_manifest_sha256="a"*64)))
        with (fields / "episodes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case_id", "split", "raw_report_sha256", "field_summary_sha256"])
            writer.writeheader()
            writer.writerows(receipts)
        (fields / "dataset.json").write_text(json.dumps(dict(scope="test_fixture", status="BUILT", split="train",
                    completed_episodes=len(cases), episode_table_sha256=sha256_file(fields / "episodes.csv"))))
        config = dict(fit_blocks=[0], validation_blocks=[1], training_seeds=[41, 42, 43],
                      latent_size=2, encoder_size=3, bin_ms=5., cpu_threads=1)
        binding = dict(config=config, seed=41, dataset_sha256=sha256_file(fields / "dataset.json"),
                       acquisition_design_sha256=sha256_file(acquisition / "design.json"),
                       model_implementation_sha256=sha256_file(Path(poisson_state_space.__file__)))
        model = self.model()
        with torch.no_grad():
            model.loading.zero_()
            model.log_rate_bias.copy_(torch.tensor([20., 40., 4000.]).double().log())
        torch.save(dict(binding=binding, model=model.state_dict()), training / "best.pt")
        (training / "run.json").write_text(json.dumps(dict(status="FITTED_DEVELOPMENT", binding=binding,
                                                   best_checkpoint_sha256=sha256_file(training / "best.pt"))))
        settings = yaml.safe_load((Path(__file__).parents[1] / "configs/neural_population_matched_forecast.yaml").read_text())
        settings.update(origins_ms=[10.], horizons_ms=[5., 20.], joint_samples=4, temporal_lags_bins=[1])
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(settings))
        return acquisition, fields, training, path

    @staticmethod
    def scale(*args):
        return np.zeros(2), np.ones(2), dict(unit_count=np.array([2, 1]),
                      field_ids=np.array(['["L23E",0,0]', '["L23I",0,0]']))

    def test_complete_forecast_replay_detects_within_field_unit_swap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            acquisition, fields, training, config = self.fixture(root)
            forecasts = root / "forecasts"
            with patch("Libs.population_dynamics.population_trajectory_evaluation.NativeRenewalEpisode", ObservedEpisodeFixture), \
                 patch("Libs.population_dynamics.population_trajectory_evaluation.fitting_scale", self.scale), \
                 patch("Libs.population_dynamics.population_forecast_replay.NativeRenewalEpisode", ObservedEpisodeFixture):
                result = evaluate(acquisition, fields, training, config, forecasts, "poisson", "cpu")
                self.assertEqual(result["score_rows"], 2)
                self.assertFalse(result["likelihood_scores_compared"])
                first = replay(acquisition, fields, training, forecasts, root / "replay.json", "cpu")
                self.assertEqual(first["status"], "EXACT_FULL_UNIT_AND_INDEPENDENT_FIELD_REPLAY")
                path = forecasts / "full_unit_samples.npz"
                with np.load(path, allow_pickle=False) as a:
                    arrays = {k: a[k] for k in a.files}
                old = arrays["sampled_unit_counts"].copy()
                arrays["sampled_unit_counts"] = old[:, :, [2, 1, 0]]
                self.assertFalse(np.array_equal(arrays["sampled_unit_counts"], old))
                # Units11 and29 share field0: projected fields alone cannot
                # detect this identity error, but unitwise replay must do so.
                np.testing.assert_array_equal(old[:, :, [0, 2]].sum(-1),
                                              arrays["sampled_unit_counts"][:, :, [0, 2]].sum(-1))
                np.savez_compressed(path, **arrays)
                result["artifacts"]["full_unit_samples.npz"] = sha256_file(path)
                (forecasts / "evaluation.json").write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, "elementwise replay"):
                    replay(acquisition, fields, training, forecasts, root / "bad_replay.json", "cpu")

    def test_cost_records_count_each_episode_and_longest_origin_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            acquisition, fields, training, config = self.fixture(root)
            settings = yaml.safe_load(config.read_text())
            settings["origins_ms"] = [10., 30.]
            config.write_text(yaml.safe_dump(settings))
            forecasts = root / "forecasts"
            with patch("Libs.population_dynamics.population_trajectory_evaluation.NativeRenewalEpisode", ObservedEpisodeFixture), \
                 patch("Libs.population_dynamics.population_trajectory_evaluation.fitting_scale", self.scale), \
                 patch("Libs.population_dynamics.population_forecast_replay.NativeRenewalEpisode", ObservedEpisodeFixture):
                result = evaluate(acquisition, fields, training, config, forecasts, "poisson", "cpu")
                replayed = replay(acquisition, fields, training, forecasts, root / "replay.json", "cpu")
            self.assertEqual(replayed["status"], "EXACT_FULL_UNIT_AND_INDEPENDENT_FIELD_REPLAY")
            self.assertEqual(json.loads((forecasts / "evaluation.json").read_text()), result)
            self.assertEqual((result["episodes"], result["origins"], result["score_rows"]), (1, 2, 4))
            self.assertEqual(len(result["episode_costs"]), 1)
            episode = result["episode_costs"][0]
            self.assertEqual(episode["origins_ms"], [10., 30.])
            self.assertEqual((episode["native_bins"], episode["unit_count"]), (12, 3))
            self.assertEqual((episode["count_dtype"], episode["stimulus_dtype"]), ("float32", "float32"))
            self.assertEqual((episode["count_tensor_dtype"], episode["stimulus_tensor_dtype"]),
                             ("torch.float32", "torch.float32"))
            self.assertEqual((episode["count_tensor_device"], episode["stimulus_tensor_device"]), ("cpu", "cpu"))
            with (forecasts / "scores.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            identities = {(cost["case_id"], cost["origin_ms"]) for cost in result["origin_costs"]}
            self.assertEqual(len(identities), 2)
            self.assertEqual(result["generated_unit_count_values"], 2 * 4 * 4 * 3)
            for cost in result["origin_costs"]:
                siblings = [row for row in rows if row["case_id"] == cost["case_id"]
                            and float(row["origin_ms"]) == cost["origin_ms"]]
                self.assertEqual({float(row["horizon_ms"]) for row in siblings}, {5., 20.})
                self.assertEqual(cost["horizons_ms"], [5., 20.])
                self.assertEqual((cost["generated_horizon_ms"], cost["generated_bins"], cost["draws"],
                                  cost["unit_count"], cost["field_count"], cost["score_rows"]), (20., 4, 4, 3, 2, 2))
                self.assertEqual(cost["generated_unit_count_values"], 4 * 4 * 3)
                self.assertEqual((cost["generated_unit_count_dtype"], cost["generated_unit_count_device"]),
                                 ("torch.int64", "cpu"))
                self.assertEqual((cost["latent_dtype"], cost["latent_device"]), ("torch.float64", "cpu"))
                self.assertEqual((cost["field_count_dtype"], cost["field_metric_dtype"]), ("int64", "float64"))
                self.assertIsNone(cost["peak_gpu_allocated_bytes"])
                self.assertGreater(cost["peak_process_rss_bytes"], 0)
                for name in ("inference_seconds", "full_unit_generation_seconds", "field_projection_seconds"):
                    self.assertEqual({row[name] for row in siblings}, {str(cost[name])})
                with np.load(forecasts / siblings[0]["ensemble_artifact"], allow_pickle=False) as arrays:
                    self.assertEqual(arrays["sampled_field_counts"].shape, (4, 4, 2))
                    self.assertEqual(arrays["sampled_field_counts"].dtype, np.int64)
                    self.assertEqual(arrays["latent_states"].dtype, np.float64)
                subtotal = sum(cost[name] for name in (
                    "inference_seconds", "full_unit_generation_seconds", "field_projection_seconds",
                    "scoring_seconds", "serialization_seconds", "hash_seconds", "csv_seconds"))
                self.assertGreaterEqual(cost["origin_total_seconds"], subtotal)
            self.assertEqual([cost["stored_full_unit_counts"] for cost in result["origin_costs"]], [True, False])
            self.assertFalse(result["cost_scope"]["speedup_qualified"])
            self.assertIn("final evaluation.json write excluded", result["cost_scope"]["function_total_seconds"])
            self.assertIn("process-lifetime", result["cost_scope"]["peak_process_rss_bytes"])
            self.assertEqual(result["hardware"]["model_parameter_dtypes"], ["torch.float64"])
            self.assertEqual(result["hardware"]["model_parameter_devices"], ["cpu"])
            for values in (result, episode, *result["origin_costs"]):
                for key, value in values.items():
                    if key.endswith("_seconds"):
                        self.assertTrue(math.isfinite(value), key)
                        self.assertGreaterEqual(value, 0, key)
            self.assertGreaterEqual(result["peak_process_rss_bytes"],
                                    max(cost["peak_process_rss_bytes"] for cost in result["origin_costs"]))
            self.assertGreaterEqual(result["function_total_seconds"],
                                    result["setup_seconds"] + result["elapsed_seconds"] + result["score_hash_seconds"])

    def test_changed_fitting_summary_is_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            acquisition, fields, training, config = self.fixture(root)
            (fields / "train_block_000_familiar_0" / "summary.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "canonical field index"):
                evaluate(acquisition, fields, training, config, root / "rejected", "poisson", "cpu")
            self.assertFalse((root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
