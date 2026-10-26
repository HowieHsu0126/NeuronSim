"""Analytical and complete-history contracts for cached mean diagnostics."""
import unittest
import numpy as np

from .population_prior_mean_diagnostic import mean_envelopes, signed_violation, epoch_means


class PriorMeanDiagnosticTests(unittest.TestCase):
    def test_two_step_bound_equals_extreme_bounded_drives(self):
        mean = np.array([2., -3., .2])
        decay = np.array([.5, .8, 0.])
        limits = mean_envelopes(mean, decay, 2)
        for sign, side in ((-1, "lower"), (1, "upper")):
            value = mean.copy()
            for _ in range(2):
                value = decay * value + (1 - decay) * sign
            np.testing.assert_allclose(limits["current_" + side], value, atol=1e-15)
        np.testing.assert_array_equal(limits["structural_lower"], [-1., -3., -1.])
        np.testing.assert_array_equal(limits["structural_upper"], [2., 1., 1.])

    def test_current_bounds_are_inside_structural_bounds_including_endpoints(self):
        mean = np.array([-2., -.3, .4, 3.])
        for steps in (1, 4, 16, 100):
            for decay in (0., .1, .99, 1.):
                values = mean_envelopes(mean, np.full_like(mean, decay), steps)
                self.assertTrue(np.all(values["current_lower"] >= values["structural_lower"] - 1e-15))
                self.assertTrue(np.all(values["current_upper"] <= values["structural_upper"] + 1e-15))

    def test_signed_violation_preserves_direction(self):
        actual = signed_violation([-2., -.5, 3.], [-1., -1., -1.], [1., 1., 1.])
        np.testing.assert_array_equal(actual, [-1., 0., 2.])

    def test_invalid_inputs_rejected(self):
        for decay, steps in (([1.1], 2), ([np.nan], 2), ([.5], 0), ([.5], True)):
            with self.subTest(decay=decay, steps=steps), self.assertRaises(ValueError):
                mean_envelopes([0.], decay, steps)
        with self.assertRaises(ValueError):
            signed_violation([0.], [2.], [1.])

    def test_complete_history_rejects_missing_or_duplicated_case_draw(self):
        config = dict(fit_blocks=[0], validation_blocks=[1], maximum_epochs=2, validation_draws=2)
        terms = ("loss", "prior_kl_h1", "prior_kl_h4", "prior_kl_h16", "prior_kl_h100")
        rows = [dict(epoch=str(epoch), case_id=f"train_block_001_familiar_{template}", draw=str(draw),
                     **{term: str(epoch + 1) for term in terms})
                for epoch in range(2) for template in range(4) for draw in range(2)]
        result = epoch_means(rows, "selection", 41, config)
        self.assertEqual([row["loss"] for row in result], [1., 2.])
        for bad in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                epoch_means(bad, "selection", 41, config)


if __name__ == "__main__":
    unittest.main()
