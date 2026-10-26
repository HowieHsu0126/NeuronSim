import unittest

import numpy as np

from .trajectory_scores import energy_score, energy_score_terms, variogram_score


class TrajectoryScoreTests(unittest.TestCase):
    def test_energy_uses_distinct_pair_denominator(self):
        # Two scalar draws at0/2 and an observation at1 give 1 - 1 = 0.
        self.assertAlmostEqual(energy_score(np.array([[0.], [2.]]), np.array([1.])), 0.)
        self.assertAlmostEqual(energy_score(np.zeros((3, 2)), np.array([1., 1.])), 1.)

    def test_energy_terms_preserve_exact_legacy_arithmetic(self):
        for shape in ((2, 1), (16, 4, 8), (16, 100, 128)):
            samples = np.sin(np.arange(np.prod(shape), dtype=np.float64)).reshape(shape)
            observed = np.cos(np.arange(np.prod(shape[1:]), dtype=np.float64)).reshape(shape[1:])
            x, y = samples.reshape(len(samples), -1), observed.reshape(-1)
            first = np.linalg.norm(x - y, axis=1).mean()
            pair_sum = sum(np.linalg.norm(x[i + 1:] - x[i], axis=1).sum() for i in range(len(x) - 1))
            legacy = float((first - pair_sum / (len(x) * (len(x) - 1))) / np.sqrt(y.size))
            terms = energy_score_terms(samples, observed)
            self.assertEqual(energy_score(samples, observed), legacy)
            self.assertEqual(terms["energy_score"], legacy)
            self.assertAlmostEqual(terms["draw_to_observation"] - terms["subtracted_draw_to_draw"], legacy, places=14)

    def test_energy_terms_use_distinct_pairs_and_coordinate_normalization(self):
        terms = energy_score_terms(np.array([[0., 0.], [2., 2.]]), np.array([1., 1.]))
        self.assertEqual(terms, dict(draw_to_observation=1., subtracted_draw_to_draw=1., energy_score=0.))

    def test_variogram_mc_correction_can_be_negative(self):
        observed = np.array([[0., 1.]])
        draws = np.array([[[0., 0.]], [[0., 2.]]])
        self.assertAlmostEqual(variogram_score(draws, observed, [0], [1]), -1.)
        fixed = np.repeat(observed[None], 4, axis=0)
        self.assertAlmostEqual(variogram_score(fixed, observed, [0], [1]), 0.)


if __name__ == "__main__":
    unittest.main()
