"""Temporal indexing, block exclusion and complete fitting-log contracts."""
import unittest

import numpy as np
import torch

from .population_prior_residual_diagnostic import (
    cached_residuals, leave_block_out_center, residual_moments, fit_history,
)


class ResidualDiagnosticTests(unittest.TestCase):
    def test_transition_uses_next_bin_controls(self):
        class Model:
            field_names = ("A",)
            latent_channels = input_size = 1
            def transition_mean(self, state, controls):
                return state + controls[:, :, None]
        mean = torch.tensor([0., 2., 5., 9.]).reshape(1, 4, 1, 1)
        controls = torch.tensor([99., 2., 3., 4.]).reshape(1, 4, 1)
        residual, persistence = cached_residuals(Model(), mean, torch.ones_like(mean), controls)
        np.testing.assert_array_equal(residual, np.zeros((3, 1, 1)))
        self.assertAlmostEqual(persistence, 29 / 3)
        with self.assertRaises(ValueError):
            cached_residuals(Model(), mean, -torch.ones_like(mean), controls)

    def test_block_centering_excludes_queried_block(self):
        values = np.arange(3 * 2 * 4 * 2).reshape(3, 2, 4, 2, 1).astype(float)
        expected = values[1] - (values[0] + values[2]) / 2
        np.testing.assert_array_equal(leave_block_out_center(values, 1), expected)
        changed = values.copy()
        changed[1] += 100
        np.testing.assert_allclose(leave_block_out_center(changed, 1) - expected, 100)
        with self.assertRaises(ValueError):
            leave_block_out_center(values[:1], 0)

    def test_shared_template_time_effect_is_removed(self):
        shared = np.arange(24.).reshape(2, 4, 3, 1)
        values = np.stack([shared + offset for offset in (-2., 0., 2.)])
        for block, expected in enumerate((-3., 0., 3.)):
            np.testing.assert_allclose(leave_block_out_center(values, block), expected)

    def test_covariance_and_lag_preserve_episode_boundaries(self):
        values = np.array([[[1., 2.], [3., 1.], [2., 5.], [7., 4.]],
                           [[100., 60.], [70., 45.], [90., 22.], [40., 10.]]])[..., None]
        summary, arrays = residual_moments(values, ["A", "B"])
        np.testing.assert_allclose(arrays["covariance"][0], np.cov(values[..., 0].reshape(-1, 2).T, bias=True))
        np.testing.assert_allclose(arrays["correlation"][0], np.corrcoef(values[..., 0].reshape(-1, 2).T))
        for field in range(2):
            expected = np.corrcoef(values[:, :-1, field, 0].ravel(), values[:, 1:, field, 0].ravel())[0, 1]
            self.assertAlmostEqual(arrays["lag1_correlation"][field, 0], expected)
        self.assertEqual(summary["lag_pairs"], 6)
        self.assertEqual(summary["cross_population_correlation_pairs"], 1)
        self.assertEqual(summary["within_population_correlation_pairs"], 0)
        self.assertIsNone(summary["within_population_mean_correlation"])
        self.assertAlmostEqual(summary["mse"], summary["squared_bias"] + summary["mean_coordinate_variance"])

    def test_zero_variance_correlations_remain_undefined(self):
        summary, arrays = residual_moments(np.ones((2, 4, 2, 1)), ["A", "B"])
        self.assertTrue(np.isnan(arrays["correlation"]).all())
        self.assertTrue(np.isnan(arrays["lag1_correlation"]).all())
        self.assertEqual(summary["cross_population_defined_correlations"], 0)
        self.assertIsNone(summary["lag1_mean"])
        self.assertEqual(summary["mse"], 1.)

    def test_complete_fitting_history_and_ess_semantics(self):
        config = dict(fit_blocks=[0], bin_ms=5., prior_horizons_ms=[5., 20.], prediction_particles=8)
        rows = [dict(epoch=epoch, case_id=f"train_block_000_familiar_{template}", gradient_norm=2.,
                     prior_observation_nll_h1=1., prior_observation_nll_h4=3.,
                     prior_observation_particle_ess_h1=8., prior_observation_particle_ess_h4=1.)
                for epoch in range(2) for template in range(4)]
        actual = fit_history(rows, config, 41, 2)
        self.assertEqual(len(actual), 4)
        self.assertEqual(actual[0]["ess_mean"], 8)
        self.assertEqual(actual[1]["case_mean_ess_within_1e_6_of_one_fraction"], 1)
        for bad in (rows[:-1], rows + rows[:1], [dict(row, case_id="train_block_014_familiar_0") for row in rows],
                    [dict(row, prior_observation_particle_ess_h1=9.) for row in rows]):
            with self.subTest(bad=bad[0]), self.assertRaises(ValueError):
                fit_history(bad, config, 41, 2)


if __name__ == "__main__":
    unittest.main()
