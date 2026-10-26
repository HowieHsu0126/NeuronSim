"""Causal temporal alignment and non-regressive diagnostic decision tests."""
import unittest
import numpy as np

from .population_prior_residual_prediction import causal_inputs, paired_decisions, validate_config


class ResidualPredictionEntryTests(unittest.TestCase):
    def test_history_and_control_use_only_boundary_information(self):
        mean = np.arange(10.).reshape(5, 2, 1)
        residual = np.arange(8.).reshape(4, 2, 1) + 100
        controls = np.arange(5.).reshape(5, 1) + 1000
        X, Y = causal_inputs(mean, residual, controls)
        np.testing.assert_array_equal(X[0], [2., 3., 1002., 100., 101.])
        np.testing.assert_array_equal(X[-1], [6., 7., 1004., 104., 105.])
        np.testing.assert_array_equal(Y, [[102., 103.], [104., 105.], [106., 107.]])
        changed = residual.copy()
        changed[-1] += 999
        np.testing.assert_array_equal(causal_inputs(mean, changed, controls)[0], X)
        future = mean.copy()
        future[-1] += 999
        np.testing.assert_array_equal(causal_inputs(future, residual, controls)[0], X)

    def test_invalid_complete_episode_inputs_reject(self):
        mean = np.ones((5, 2, 1))
        residual, controls = np.ones((4, 2, 1)), np.ones((5, 1))
        for m, r, u in ((mean[:3], residual, controls), (mean, residual[:3], controls),
                        (mean, residual, controls[:-1]), (mean*np.nan, residual, controls)):
            with self.subTest(shape=m.shape), self.assertRaises(ValueError):
                causal_inputs(m, r, u)

    @staticmethod
    def config():
        return dict(training_seeds=[41, 42, 43], fitting_blocks=list(range(4)),
                    bootstrap_stream_seeds=[41, 42, 43], bootstrap_draws_per_stream=40,
                    confidence_level=.95)

    def rows(self, scores):
        return [dict(seed=s, block=b, predictor=p, normalized_mse=v)
                for s in (41, 42, 43) for b in range(4) for p, v in scores.items()]

    def test_history_decision_requires_improvement_over_original(self):
        result = paired_decisions(self.rows(dict(zero=1., current=3., history=2.)), self.config())
        self.assertTrue(result["comparisons"]["history_vs_current"]["directional_support"])
        self.assertFalse(result["comparisons"]["history_vs_zero"]["directional_support"])
        self.assertEqual(result["next_action"], "TESTED_LINEAR_FEATURES_DO_NOT_RESOLVE_RESIDUAL_DEPENDENCE")

    def test_seed_fits_are_averaged_inside_each_source_block(self):
        result = paired_decisions(self.rows(dict(zero=3., current=2., history=1.)), self.config())
        item = result["comparisons"]["history_vs_current"]
        self.assertEqual(item["conditional_block_bootstrap_interval"], [1., 1.])
        self.assertEqual(len(item["per_block_seed_mean_improvement"]), 4)
        self.assertEqual(result["next_action"], "TEST_CAUSAL_MEMORY_CORRECTION_IN_FREE_GENERATION")

    def test_incomplete_and_duplicate_comparisons_reject(self):
        rows = self.rows(dict(zero=3., current=2., history=1.))
        for bad in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                paired_decisions(bad, self.config())

    def test_unregistered_protocol_rejects(self):
        with self.assertRaises(ValueError):
            validate_config(dict(training_seeds=[41, 42, 44], fitting_blocks=list(range(15))))


if __name__ == "__main__":
    unittest.main()
