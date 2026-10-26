"""Focused cache-lineage, retained-array, window and aggregation checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from population_dynamics import population_readout_diagnostic as module
from population_dynamics.test_population_readout_moments import affine_parent, example


def posterior_fixture():
    checkpoint = dict(binding=dict(seed=41, model_implementation_sha256="model", cohort_identity="cohort"),
                      initialization=dict(parent_checkpoint_sha256="initialization-parent"))
    mean = torch.arange(1200, dtype=torch.float32).reshape(1, 1200, 1, 1).expand(1, 1200, 128, 4)
    cache = dict(binding=dict(case_id="train_block_014_familiar_0", parent_checkpoint_sha256="initialization-parent",
        model_implementation_sha256="model", bin_ms=5, cohort=dict(cohort_identity="cohort")),
        mean=mean, variance=torch.ones((1, 1200, 128, 4)))
    return cache, checkpoint


def native_rows(origin=1000., block=14):
    return {h: dict(model="history", seed=41, block=block, case_id=f"train_block_{block:03d}_familiar_0",
        template_id="familiar_0", origin_ms=origin, horizon_ms=h, method="population_residual_history",
        hardware_device="cpu", ensemble_artifact=f"train_block_{block:03d}_familiar_0_origin_{origin:g}.npz",
        energy_score=h/1000, peak_gpu_allocated_bytes=None) for h in (20., 80., 500.)}


class PopulationReadoutDiagnosticTests(unittest.TestCase):
    def test_complete_kernel_output_passes_driver_column_contract(self):
        moments = module.trajectory_readout_moments(affine_parent(), *example())
        names = module._moment_columns(moments, None)
        self.assertEqual(len(names), 25)
        self.assertEqual(module._moment_columns(moments, names), names)
        with self.assertRaisesRegex(ValueError, "complete consistent"):
            module._moment_columns(moments, tuple(reversed(names)))
        del moments["log_unit_mean_draw_hazard"]
        with self.assertRaisesRegex(ValueError, "complete consistent"):
            module._moment_columns(moments, None)

    def test_cache_uses_initialization_lineage_and_future_window_is_not_shifted(self):
        cache, checkpoint = posterior_fixture()
        mean, variance = module.posterior_window(cache, checkpoint, "train_block_014_familiar_0", 1000.)
        self.assertEqual(mean.shape, (100, 128, 4))
        self.assertEqual(float(mean[0, 0, 0]), 200.)
        self.assertEqual(float(mean[3, 0, 0]), 203.)
        self.assertEqual(float(mean[-1, 0, 0]), 299.)
        self.assertTrue(torch.all(variance == 1))
        cache["binding"]["parent_checkpoint_sha256"] = "trained-selected-checkpoint"
        with self.assertRaisesRegex(ValueError, "initialization-parent"):
            module.posterior_window(cache, checkpoint, "train_block_014_familiar_0", 1000.)

    def test_cache_identity_shape_and_variance_rejections(self):
        for mutation in ("case", "cohort", "shape", "variance", "nonfinite"):
            cache, checkpoint = posterior_fixture()
            if mutation == "case":
                cache["binding"]["case_id"] = "train_block_015_familiar_0"
            elif mutation == "cohort":
                cache["binding"]["cohort"]["cohort_identity"] = "different"
            elif mutation == "shape":
                cache["mean"] = cache["mean"][:, :1199]
            elif mutation == "variance":
                cache["variance"][0, 0, 0, 0] = 0
            else:
                cache["variance"][0, 0, 0, 0] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                module.posterior_window(cache, checkpoint, "train_block_014_familiar_0", 1000.)

    def test_retained_loader_never_materializes_full_unit_member(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trajectory.npz"
            np.savez(path, latent_states=np.zeros((16, 100, 128, 4), dtype=np.float32),
                initial_latent_states=np.ones((16, 1, 128, 4), dtype=np.float32),
                full_unit_samples=np.array([{"must_not_be_unpickled": True}], dtype=object))
            samples, initial = module.load_retained_trajectory(path)
            self.assertEqual(samples.shape, (16, 100, 128, 4))
            self.assertEqual(initial.shape, (16, 128, 4))
            self.assertTrue(torch.all(initial == 1))

    def test_missing_corrupt_and_invalid_retained_arrays_reject(self):
        for mutation in ("missing", "shape", "nonfinite", "corrupt"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"trajectory.npz"
                arrays = dict(latent_states=np.zeros((16, 100, 128, 4), dtype=np.float32),
                              initial_latent_states=np.zeros((16, 1, 128, 4), dtype=np.float32))
                if mutation == "missing":
                    del arrays["initial_latent_states"]
                elif mutation == "shape":
                    arrays["latent_states"] = arrays["latent_states"][:, :99]
                elif mutation == "nonfinite":
                    arrays["latent_states"][0, 0, 0, 0] = np.nan
                if mutation == "corrupt":
                    path.write_bytes(b"not a retained array file")
                else:
                    np.savez(path, **arrays)
                with self.assertRaisesRegex(ValueError, "invalid retained trajectory"):
                    module.load_retained_trajectory(path)

    def test_prefix_endpoint_and_maximum_are_distinct_with_native_na(self):
        moments = dict(mean_displacement_unit_mean=np.arange(1., 101.),
                       decomposition_closure_max_absolute=np.arange(100., 0., -1))
        times, rows, metrics = module.origin_tables(moments, native_rows(), ["energy_score", "peak_gpu_allocated_bytes"])
        first = rows[0]
        self.assertEqual(first["prefix_mean_mean_displacement_unit_mean"], 2.5)
        self.assertEqual(first["endpoint_mean_displacement_unit_mean"], 4.)
        self.assertEqual(first["prefix_max_decomposition_closure_max_absolute"], 100.)
        self.assertEqual(first["endpoint_decomposition_closure_max_absolute"], 97.)
        self.assertEqual(times[0]["absolute_bin_end_ms"], 1005.)
        self.assertEqual(first["native_energy_score"], .02)
        self.assertIsNone(first["native_peak_gpu_allocated_bytes"])
        self.assertIn("prefix_max_decomposition_closure_max_absolute", metrics)
        bad_native = native_rows()
        bad_native[80.]["origin_ms"] = 3000.
        with self.assertRaisesRegex(ValueError, "one exact origin"):
            module.origin_tables(moments, bad_native, ["energy_score"])

    def test_aggregation_preserves_blocks_na_and_maximum_closure(self):
        rows, metrics = [], None
        for origin, block, moment, closure in ((1000., 14, 2., 9.), (3000., 14, 4., 3.), (1000., 15, 8., 1.)):
            _, added, metrics = module.origin_tables(
                dict(mean_displacement_unit_mean=np.full(100, moment), decomposition_closure_max_absolute=np.full(100, closure)),
                native_rows(origin, block), ["energy_score", "peak_gpu_allocated_bytes"])
            rows.extend(added)
        summary = module.summarize_origins(rows, metrics)
        at20 = {row["metric"]: row for row in summary if row["horizon_ms"] == 20.}
        self.assertAlmostEqual(at20["prefix_mean_mean_displacement_unit_mean"]["value"], 14/3)
        self.assertEqual(at20["prefix_max_decomposition_closure_max_absolute"]["value"], 9.)
        self.assertEqual(at20["prefix_max_decomposition_closure_max_absolute"]["statistic"], "maximum")
        self.assertIsNone(at20["native_peak_gpu_allocated_bytes"]["value"])
        self.assertEqual(at20["native_peak_gpu_allocated_bytes"]["structural_na_origins"], 3)
        blocks = module.summarize_origins(rows, metrics, by_block=True)
        block14 = next(row for row in blocks if row["block"] == 14 and row["horizon_ms"] == 20.
                       and row["metric"] == "prefix_mean_mean_displacement_unit_mean")
        self.assertEqual(block14["value"], 3.)
        self.assertEqual(block14["total_origins"], 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.summarize_origins(rows+[rows[0]], metrics)

    def hazard_table_fixture(self):
        native = native_rows()
        for horizon, row in native.items():
            row.update(native_sampled_expected_spikes_per_unit=horizon / 1000,
                       native_observed_spikes_per_unit=horizon / 2000,
                       prefix_bins=int(horizon / 5))
        identity = {key: native[20.][key] for key in
                    ("model", "seed", "block", "case_id", "template_id", "origin_ms")}
        times = [dict(identity, elapsed_ms=5 * (step + 1), absolute_bin_end_ms=1000 + 5 * (step + 1),
                      log_unit_mean_draw_hazard=np.log(2.), log_unit_mean_mean_path_hazard=0.,
                      log_unit_mean_posterior_mean_hazard=np.log(3.)) for step in range(100)]
        return times, list(native.values())

    def test_integrated_hazard_uses_arithmetic_rates_and_seconds(self):
        times, native = self.hazard_table_fixture()
        rows, metrics = module.prefix_hazard_exposures(times, native)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(metrics), 5)
        for row in rows:
            seconds = row["horizon_ms"] / 1000
            self.assertAlmostEqual(row["integrated_draw_hazard_per_unit"], 2 * seconds)
            self.assertAlmostEqual(row["integrated_mean_path_hazard_per_unit"], seconds)
            self.assertAlmostEqual(row["integrated_posterior_mean_hazard_per_unit"], 3 * seconds)
            self.assertEqual(row["native_sampled_expected_spikes_per_unit"], seconds)
        reversed_rows, _ = module.prefix_hazard_exposures(list(reversed(times)), native)
        self.assertEqual(rows, reversed_rows)

    def test_integrated_hazard_keeps_temporal_burst_before_averaging(self):
        times, native = self.hazard_table_fixture()
        for row in times:
            row["log_unit_mean_draw_hazard"] = 0.
        times[-1]["log_unit_mean_draw_hazard"] = np.log(101.)
        rows, _ = module.prefix_hazard_exposures(times, native)
        self.assertAlmostEqual(rows[-1]["integrated_draw_hazard_per_unit"], 1.)
        self.assertAlmostEqual(rows[-1]["integrated_mean_path_hazard_per_unit"], .5)
        wrong = .5 * np.exp(np.mean([row["log_unit_mean_draw_hazard"] for row in times]))
        self.assertGreater(rows[-1]["integrated_draw_hazard_per_unit"], wrong)

    def test_integrated_hazard_rejects_missing_duplicate_and_unmatched_windows(self):
        for mutation in ("missing_time", "duplicate_time", "absolute_time", "nonfinite_rate",
                         "missing_horizon", "duplicate_horizon", "origin", "clock", "native_count"):
            times, native = self.hazard_table_fixture()
            if mutation == "missing_time": times.pop()
            elif mutation == "duplicate_time": times.append(times[0])
            elif mutation == "absolute_time": times[0]["absolute_bin_end_ms"] += 5
            elif mutation == "nonfinite_rate": times[0]["log_unit_mean_draw_hazard"] = float("nan")
            elif mutation == "missing_horizon": native.pop()
            elif mutation == "duplicate_horizon": native.append(native[0])
            elif mutation == "origin": native[0]["origin_ms"] = 3000.
            elif mutation == "clock": native[0]["prefix_bins"] += 1
            else: native[0]["native_observed_spikes_per_unit"] = -1.
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                module.prefix_hazard_exposures(times, native)

    def test_merlin_receipt_checks_real_environment_and_hostname(self):
        environment = dict(RH2_JOB_RUN_ID="run-fixture", ARNOLD_TRIAL_ID="trial-fixture", RH2_CHECKPOINT_HDFS="hdfs://fixture/checkpoints")
        receipt = dict(execution_backend="merlin", environment=environment, hostname="fixture-worker")
        with patch.dict(module.os.environ, environment), patch.object(module.platform, "node", return_value="fixture-worker"):
            self.assertEqual(module.merlin_runtime(receipt)["environment"], environment)
            bad = json.loads(json.dumps(receipt))
            bad["environment"]["ARNOLD_TRIAL_ID"] = "another-trial"
            with self.assertRaisesRegex(ValueError, "receipt differs"):
                module.merlin_runtime(bad)
        with patch.dict(module.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "actual Merlin"):
                module.merlin_runtime(receipt)


if __name__ == "__main__":
    unittest.main()
