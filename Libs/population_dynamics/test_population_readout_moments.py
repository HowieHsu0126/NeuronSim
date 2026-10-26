"""Analytic affine-readout, covariance, and state-preservation tests."""
import math
import unittest

import torch

from population_dynamics.population_readout_moments import trajectory_readout_moments
from population_dynamics.population_state import ControlledPopulationState
from population_dynamics.test_population_residual_memory import make_parent


def example():
    samples = torch.tensor([
        [[[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]]],
        [[[2., 0.], [0., 2.]], [[3., 1.], [1., 3.]]],
    ], dtype=torch.float64)
    initial = torch.tensor([[[-1., 0.], [0., -1.]], [[1., 0.], [0., 1.]]], dtype=torch.float64)
    q_mean = torch.zeros((2, 2, 2), dtype=torch.float64)
    q_variance = torch.tensor([[[1., 4.], [9., 16.]], [[4., 1.], [16., 9.]]], dtype=torch.float64)
    scale = torch.tensor([1., 2., 1., 2.], dtype=torch.float64)
    return samples, initial, q_mean, q_variance, scale


def affine_parent(metadata=False):
    parent = make_parent()
    loading = torch.tensor([[1., 2.], [-1., .5], [0., 3.], [2., -2.]])
    bias = torch.tensor([1., -2., .25, .5])
    if metadata:
        config = parent.configuration()
        def schema(name):
            return dict(name=name, unit="dimensionless", meaning="synthetic fixture indicator", transform="identity")
        config["metadata"] = dict(
            unit_feature_schema=[schema(f"indicator_{index}") for index in range(4)],
            field_feature_schema=[schema("field_indicator")],
            unit_features=torch.eye(4).tolist(), field_features=[[0.], [1.]],
            control_features=[dict(name="stimulus", unit="dimensionless")],
            source_identity=dict(circuit_id="fixture", source_revision="fixture",
                                 connectivity_identity="fixture", parameter_identity="fixture"),
        )
        parent = ControlledPopulationState(**config).eval()
        with torch.no_grad():
            parent.shared_unit_readout.weight.copy_(torch.cat((loading, bias[:, None]), dim=1).T)
            parent.shared_unit_readout.bias.zero_()
    else:
        with torch.no_grad():
            parent.unit_loading.copy_(loading)
            parent.log_hazard_bias.copy_(bias)
    return parent


class PopulationReadoutMomentTests(unittest.TestCase):
    def test_arithmetic_hazard_uses_rates_before_unit_reduction(self):
        result = trajectory_readout_moments(affine_parent(), *example(), unit_chunk_size=3)
        bias = [1., -2., .25, .5]
        draw_displacements = [([0., 0., 0., 0.], [2., -2., 6., -4.]),
                              ([3., -.5, 3., 0.], [5., -2.5, 9., -4.])]
        q_rate = sum(math.exp(value) for value in bias) / 4
        for time, draws in enumerate(draw_displacements):
            draw_rate = sum(math.exp(bias[unit] + draw[unit]) for draw in draws for unit in range(4)) / 8
            mean_rate = sum(math.exp(bias[unit] + sum(draw[unit] for draw in draws) / 2)
                            for unit in range(4)) / 4
            expected = {
                "log_unit_mean_draw_hazard": math.log(draw_rate),
                "log_unit_mean_mean_path_hazard": math.log(mean_rate),
                "log_unit_mean_posterior_mean_hazard": math.log(q_rate),
                "unit_mean_hazard_mean_path_log_ratio": math.log(mean_rate / q_rate),
                "unit_mean_hazard_empirical_jensen_gap": math.log(draw_rate / mean_rate),
                "unit_mean_hazard_draw_to_posterior_mean_log_ratio": math.log(draw_rate / q_rate),
            }
            for name, value in expected.items():
                self.assertAlmostEqual(float(result[name][time]), value, places=12)
            self.assertNotAlmostEqual(float(result["unit_mean_hazard_mean_path_log_ratio"][time]),
                                      float(result["mean_displacement_unit_mean"][time]), places=3)

    def test_arithmetic_hazard_handles_rates_beyond_exponential_range(self):
        parent = affine_parent()
        reference = trajectory_readout_moments(parent, *example())
        with torch.no_grad():
            parent.log_hazard_bias.add_(1000.)
        shifted = trajectory_readout_moments(parent, *example(), unit_chunk_size=1)
        for name in ("draw", "mean_path", "posterior_mean"):
            key = f"log_unit_mean_{name}_hazard"
            torch.testing.assert_close(shifted[key], reference[key] + 1000., atol=1e-10, rtol=0)
        for key in ("unit_mean_hazard_mean_path_log_ratio", "unit_mean_hazard_empirical_jensen_gap",
                    "unit_mean_hazard_draw_to_posterior_mean_log_ratio"):
            torch.testing.assert_close(shifted[key], reference[key], atol=1e-10, rtol=0)

    def test_analytic_readout_and_diagonal_q_decomposition(self):
        result = trajectory_readout_moments(affine_parent(), *example(), unit_chunk_size=3)
        empirical_jensen = torch.tensor([math.log(math.cosh(value)) for value in (1., 1., 3., 2.)], dtype=torch.float64)
        q_adjustments = torch.tensor([[8.5, 1., 72., 50.], [4., 2.125, 40.5, 50.]], dtype=torch.float64)
        mean_displacements = torch.tensor([[1., -1., 3., -2.], [4., -1.5, 6., -2.]], dtype=torch.float64)
        stochastic_displacements = empirical_jensen - q_adjustments
        total_gap = mean_displacements + stochastic_displacements
        expected = dict(
            standardized_latent_mean_mse=torch.tensor([.3125, 1.5625], dtype=torch.float64),
            standardized_latent_ensemble_variance_mean=torch.full((2,), .625, dtype=torch.float64),
            standardized_latent_lag_one_covariance_mean=torch.full((2,), .625, dtype=torch.float64),
            empirical_log_hazard_variance_unit_mean=torch.full((2,), 7.5, dtype=torch.float64),
            empirical_jensen_gap_unit_mean=empirical_jensen.mean().expand(2),
            posterior_lognormal_adjustment_unit_mean=q_adjustments.mean(dim=1),
            posterior_mean_log_hazard_unit_mean=torch.full((2,), -.0625, dtype=torch.float64),
            predicted_mean_log_hazard_unit_mean=mean_displacements.mean(dim=1)-.0625,
            empirical_log_expected_hazard_unit_mean=mean_displacements.mean(dim=1)-.0625+empirical_jensen.mean(),
            posterior_log_expected_hazard_unit_mean=q_adjustments.mean(dim=1)-.0625,
            mean_displacement_unit_mean=mean_displacements.mean(dim=1),
            mean_displacement_unit_mean_square=mean_displacements.square().mean(dim=1),
            stochastic_gap_displacement_unit_mean=stochastic_displacements.mean(dim=1),
            stochastic_gap_displacement_unit_mean_square=stochastic_displacements.square().mean(dim=1),
            total_log_expected_hazard_gap_unit_mean=total_gap.mean(dim=1),
            total_log_expected_hazard_gap_unit_mean_square=total_gap.square().mean(dim=1),
            decomposition_cross_term_unit_mean=(2*mean_displacements*stochastic_displacements).mean(dim=1),
        )
        for name, value in expected.items():
            with self.subTest(moment=name):
                torch.testing.assert_close(result[name], value, atol=1e-12, rtol=1e-12)
        self.assertLess(float(result["decomposition_closure_max_absolute"].max()), 1e-12)
        self.assertLess(float(result["decomposition_mean_square_closure"].abs().max()), 1e-10)

    def test_initial_state_enters_same_draw_lag_covariance(self):
        samples = torch.zeros((3, 2, 2, 2), dtype=torch.float64)
        samples[:, 0, 0, 0] = torch.tensor([-1., 0., 1.])
        samples[:, 1, 0, 0] = torch.tensor([2., 0., -2.])
        initial = torch.zeros((3, 2, 2), dtype=torch.float64)
        initial[:, 0, 0] = torch.tensor([-2., 0., 2.])
        result = trajectory_readout_moments(affine_parent(), samples, initial,
            torch.zeros((2, 2, 2)), torch.ones((2, 2, 2)), torch.ones(4))
        torch.testing.assert_close(result["standardized_latent_lag_one_covariance_mean"], torch.tensor([.5, -.5], dtype=torch.float64))
        torch.testing.assert_close(result["standardized_latent_ensemble_variance_mean"], torch.tensor([.25, 1.], dtype=torch.float64))

    def test_constant_draws_keep_zero_empirical_variance_and_nonzero_q_adjustment(self):
        samples = torch.full((3, 2, 2, 2), .5, dtype=torch.float64)
        result = trajectory_readout_moments(affine_parent(), samples, samples[:, 0], samples[0],
                                           torch.ones((2, 2, 2)), torch.ones(4))
        for name in ("standardized_latent_mean_mse", "standardized_latent_ensemble_variance_mean",
                     "standardized_latent_lag_one_covariance_mean", "empirical_log_hazard_variance_unit_mean",
                     "empirical_jensen_gap_unit_mean", "mean_displacement_unit_mean_square"):
            torch.testing.assert_close(result[name], torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
        torch.testing.assert_close(result["posterior_lognormal_adjustment_unit_mean"], torch.full((2,), 2.90625, dtype=torch.float64))
        torch.testing.assert_close(result["stochastic_gap_displacement_unit_mean"], torch.full((2,), -2.90625, dtype=torch.float64), atol=1e-14, rtol=0)

    def test_metadata_affine_readout_and_chunk_sizes_match(self):
        reference = trajectory_readout_moments(affine_parent(), *example(), unit_chunk_size=4)
        metadata_parent = affine_parent(metadata=True)
        for chunk in (1, 3, 5):
            result = trajectory_readout_moments(metadata_parent, *example(), unit_chunk_size=chunk)
            for name in reference:
                torch.testing.assert_close(result[name], reference[name], atol=1e-10, rtol=1e-12)

    def test_parent_inputs_and_global_rng_are_unchanged(self):
        parent = affine_parent()
        inputs = example()
        inputs[0].requires_grad_(True)
        before = {name: value.detach().clone() for name, value in parent.state_dict().items()}
        flags = [parameter.requires_grad for parameter in parent.parameters()]
        modes = [child.training for child in parent.modules()]
        original_inputs = [value.detach().clone() for value in inputs]
        rng = torch.get_rng_state().clone()
        result = trajectory_readout_moments(parent, *inputs)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(flags, [parameter.requires_grad for parameter in parent.parameters()])
        self.assertEqual(modes, [child.training for child in parent.modules()])
        for name, value in parent.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        for value, original in zip(inputs, original_inputs):
            self.assertTrue(torch.equal(value, original))
        for value in result.values():
            self.assertEqual(value.shape, (2,))
            self.assertEqual(value.dtype, torch.float64)
            self.assertFalse(value.requires_grad)

    def test_shape_nonfinite_and_nonpositive_inputs_reject(self):
        parent, inputs = affine_parent(), example()
        bad = [
            (0, inputs[0][:1]), (0, inputs[0][0]), (1, inputs[1][:1]),
            (2, inputs[2][:1]), (3, inputs[3][:, :1]), (4, torch.ones(3)),
            (3, torch.zeros_like(inputs[3])), (3, -inputs[3]), (4, torch.zeros(4)),
            (0, inputs[0].to(torch.complex128)),
        ]
        for index, changed in bad:
            arguments = list(inputs)
            arguments[index] = changed
            with self.subTest(index=index, shape=changed.shape), self.assertRaises(ValueError):
                trajectory_readout_moments(parent, *arguments)
        for index in range(5):
            arguments = [value.clone() for value in inputs]
            arguments[index].reshape(-1)[0] = float("nan")
            with self.subTest(nonfinite=index), self.assertRaisesRegex(ValueError, "finite"):
                trajectory_readout_moments(parent, *arguments)
        for chunk in (0, -1, True, 1.5):
            with self.subTest(chunk=chunk), self.assertRaisesRegex(ValueError, "positive integer"):
                trajectory_readout_moments(parent, *inputs, unit_chunk_size=chunk)


if __name__ == "__main__":
    unittest.main()
