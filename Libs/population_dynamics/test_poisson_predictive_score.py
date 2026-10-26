"""Independent quadrature check of the count-marginal forecasting score."""
import math
import unittest

import numpy as np
from scipy.integrate import quad
import torch

from .poisson_trajectory_evaluation import poisson_lognormal_scores


class PredictiveScoreTests(unittest.TestCase):
    def test_gaussian_quadrature_matches_scalar_density_integration(self):
        observed = np.array([[0., 1., 4.], [2., 0., 1.]])
        means = np.array([[-2., -1., .5], [.1, -3., -.5]])
        variance = np.array([[.1, .3, .2], [.5, .15, .4]])
        reference = []
        for t in range(2):
            scores = []
            for count, mu, var in zip(observed[t], means[t], variance[t]):
                def density(z):
                    log_intensity = mu + math.sqrt(var) * z
                    return math.exp(count * log_intensity - math.exp(log_intensity)
                                    - math.lgamma(count + 1) - z * z / 2) / math.sqrt(2 * math.pi)
                value, error = quad(density, -12., 12., epsabs=1e-12, epsrel=1e-12)
                self.assertLess(error, 1e-10)
                scores.append(-math.log(value))
            reference.append(np.mean(scores))
        actual = poisson_lognormal_scores(torch.tensor(observed), torch.tensor(means),
                                         torch.tensor(variance), 32)
        np.testing.assert_allclose(actual, reference, atol=1e-8, rtol=0)

    def test_zero_latent_variance_is_exact_poisson(self):
        observed = torch.tensor([[0., 3.]], dtype=torch.float64)
        means = torch.tensor([[-2., .5]], dtype=torch.float64)
        result = poisson_lognormal_scores(observed, means, torch.zeros_like(means), 32)
        expected = (means.exp() - observed * means + torch.lgamma(observed + 1)).mean(-1).numpy()
        np.testing.assert_allclose(result, expected, atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
