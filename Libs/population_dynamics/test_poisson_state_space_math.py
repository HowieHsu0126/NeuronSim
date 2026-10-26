"""Independent Gaussian quadrature and derivative checks for the PLDS ELBO."""
import math
import unittest
from itertools import product

import numpy as np
import torch

from .poisson_state_space import expected_observation_nll, initial_kl, transition_kl


def tensor(value, requires_grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=requires_grad)


def normal_logpdf(value, mean, variance):
    return float(-0.5 * np.sum(np.log(2.0 * np.pi * variance) + (value - mean)**2 / variance))


def gaussian_expectation(function, mean, variance, order=12):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    weights = weights / math.sqrt(math.pi)
    scale = np.sqrt(2.0 * variance)
    total = 0.0
    for indices in product(range(order), repeat=mean.size):
        indices = np.asarray(indices, dtype=int)
        point = mean + scale * nodes[indices]
        total += float(np.prod(weights[indices])) * function(point)
    return float(total)


class AnalyticELBOTests(unittest.TestCase):
    def setUp(self):
        self.mean = np.array([[[0.20, -0.10], [-0.15, 0.31]]])
        self.previous_mean = np.array([[[0.00, 0.12], [0.18, -0.08]]])
        self.prior_mean = np.array([[0.03, -0.08]])
        self.drive = np.array([[[0.04, -0.02], [-0.03, 0.06]]])
        self.y = np.array([[[0.0, 2.0], [3.0, 1.0]]])
        self.bias = np.array([0.20, -0.30])
        self.dt = 0.037
        self.parameters = {
            "loading": np.array([[0.70, -0.40], [-0.25, 0.90]]),
            "transition": np.array([[0.72, -0.35], [0.21, 0.61]]),
            "variance": np.array([[[0.12, 0.19], [0.07, 0.16]]]),
            "previous_variance": np.array([[[0.27, 0.09], [0.11, 0.23]]]),
            "prior_variance": np.array([0.31, 0.14]),
            "process_variance": np.array([0.23, 0.08]),
        }

    def numpy_terms(self, p):
        observation = np.empty(self.mean.shape[:-1])
        transition = np.empty(self.mean.shape[:-1])
        initial = np.empty(self.mean.shape[0])
        for batch in range(self.mean.shape[0]):
            mean, variance = self.mean[batch, 0], p['variance'][batch, 0]
            prior_mean = self.prior_mean[batch]
            initial[batch] = gaussian_expectation(
                lambda z: normal_logpdf(z, mean, variance)
                - normal_logpdf(z, prior_mean, p['prior_variance']), mean, variance, order=3)
        for index in np.ndindex(self.mean.shape[:-1]):
            mean, variance = self.mean[index], p['variance'][index]
            y = self.y[index]
            log_factorial = np.array([math.lgamma(float(count) + 1.0) for count in y])

            def observation_nll(z):
                log_intensity = math.log(self.dt) + p['loading'] @ z + self.bias
                return float(np.sum(np.exp(log_intensity) - y * log_intensity + log_factorial))

            observation[index] = gaussian_expectation(observation_nll, mean, variance)
            previous_mean, previous_variance = self.previous_mean[index], p['previous_variance'][index]
            drive, dimension = self.drive[index], mean.size

            def transition_log_ratio(joint):
                previous, current = joint[:dimension], joint[dimension:]
                conditional_mean = p['transition'] @ previous + drive
                return (normal_logpdf(current, mean, variance)
                        - normal_logpdf(current, conditional_mean, p['process_variance']))

            transition[index] = gaussian_expectation(
                transition_log_ratio, np.concatenate([previous_mean, mean]),
                np.concatenate([previous_variance, variance]), order=3)
        return observation, initial, transition

    def torch_terms(self, p):
        return (expected_observation_nll(tensor(self.y), tensor(self.mean), p['variance'],
                                        p['loading'], tensor(self.bias), self.dt),
                initial_kl(tensor(self.mean[:, 0]), p['variance'][:, 0],
                           tensor(self.prior_mean), p['prior_variance']),
                transition_kl(tensor(self.mean), p['variance'], tensor(self.previous_mean),
                              p['previous_variance'], p['transition'], tensor(self.drive), p['process_variance']))

    def check_term(self, term_index, expected_shape):
        p = {name: tensor(value) for name, value in self.parameters.items()}
        actual, expected = self.torch_terms(p)[term_index], self.numpy_terms(self.parameters)[term_index]
        self.assertEqual(tuple(actual.shape), expected_shape)
        self.assertEqual(actual.dtype, torch.float64)
        np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-11, atol=1e-11)

    def test_observation_expectation_with_nonunit_bin_width(self):
        self.check_term(0, (1, 2))

    def test_initial_kl_against_gaussian_quadrature(self):
        self.check_term(1, (1,))

    def test_transition_includes_previous_state_uncertainty(self):
        self.check_term(2, (1, 2))

    def test_gradients_against_independent_quadrature_differences(self):
        p = {name: tensor(value, requires_grad=True) for name, value in self.parameters.items()}
        sum(term.sum() for term in self.torch_terms(p)).backward()
        epsilon = 1e-6
        for name, value in self.parameters.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(p[name].grad)
                actual = p[name].grad.detach().numpy()
                self.assertTrue(np.isfinite(actual).all())
                expected = np.empty_like(value)
                for index in np.ndindex(value.shape):
                    plus = {k: v.copy() for k, v in self.parameters.items()}
                    minus = {k: v.copy() for k, v in self.parameters.items()}
                    plus[name][index] += epsilon; minus[name][index] -= epsilon
                    upper = sum(float(t.sum()) for t in self.numpy_terms(plus))
                    lower = sum(float(t.sum()) for t in self.numpy_terms(minus))
                    expected[index] = (upper - lower) / (2.0 * epsilon)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6, err_msg=name)


if __name__ == '__main__':
    unittest.main()
