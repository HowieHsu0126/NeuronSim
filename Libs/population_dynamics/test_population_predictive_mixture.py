import math
import unittest
from unittest.mock import patch

import torch

from .population_predictive_mixture import (
    joint_gaussian_mixture_log_density,
    prior_predictive_mixture_loss,
)
from .population_prior_objective import (
    PRIOR_PARAMETER_NAMES,
    configure_prior_parameters,
    prior_rollout_loss,
)
from .test_population_state import example


class PopulationPredictiveMixtureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = example().double()
        generator = torch.Generator().manual_seed(43)
        self.mean = torch.randn(2, 10, 3, 2, dtype=torch.float64, generator=generator)
        self.variance = .1 + torch.rand(self.mean.shape, dtype=torch.float64, generator=generator)
        self.controls = torch.linspace(-1, 1, 20, dtype=torch.float64).reshape(2, 10, 1)

    @staticmethod
    def streams(initial_seed=41, dynamics_seed=42, target_seed=43):
        return tuple(torch.Generator().manual_seed(seed)
                     for seed in (initial_seed, dynamics_seed, target_seed))

    def objective(self, *, mean=None, variance=None, controls=None,
                  origins=(2, 5), horizons=(1, 3), streams=None,
                  prediction_particles=8, posterior_target_draws=4):
        return prior_predictive_mixture_loss(
            self.model, self.mean if mean is None else mean,
            self.variance if variance is None else variance,
            self.controls if controls is None else controls,
            origins, horizons, *(self.streams() if streams is None else streams),
            prediction_particles=prediction_particles, posterior_target_draws=posterior_target_draws)

    def assert_terms_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-12, atol=1e-12)

    def replay(self, origins, horizons, streams, particles=8, target_draws=4):
        # Independent replay retains separate batch and origin axes, including
        # source-field/channel coordinates, instead of flattening cases.
        initial, dynamics, target = streams
        shape = (particles, self.mean.shape[0], len(origins), *self.mean.shape[2:])
        target_shape = (target_draws, *shape[1:])
        boundary = [origin - 1 for origin in origins]
        state = self.mean[:, boundary] + self.variance[:, boundary].sqrt() * torch.randn(
            shape, dtype=self.mean.dtype, generator=initial)
        process_variance = self.model.process_log_variance.exp()
        terms = {}
        for step in range(1, max(horizons) + 1):
            indices = [origin + step - 1 for origin in origins]
            prior_mean = self.model.transition_mean(state, self.controls[:, indices])
            if step in horizons:
                qmean, qvariance = self.mean[:, indices], self.variance[:, indices]
                draws = qmean + qvariance.sqrt() * torch.randn(
                    target_shape, dtype=self.mean.dtype, generator=target)
                normal = torch.distributions.Normal(prior_mean, process_variance.sqrt())
                joint_component_log_density = normal.log_prob(draws[:, None]).sum((-1, -2))
                log_density = torch.logsumexp(joint_component_log_density, 1) - math.log(particles)
                entropy = torch.distributions.Normal(qmean, qvariance.sqrt()).entropy().sum((-1, -2))
                terms[f"prior_mixture_kl_h{step}"] = (-entropy - log_density.mean(0)).mean()
            if step < max(horizons):
                state = prior_mean + process_variance.sqrt() * torch.randn(
                    shape, dtype=self.mean.dtype, generator=dynamics)
        terms["loss"] = torch.stack([terms[f"prior_mixture_kl_h{h}"] for h in horizons]).mean()
        return terms

    def test_single_component_and_identical_components_match_joint_normal_density(self):
        samples = torch.linspace(-2, 2, 72, dtype=torch.float64).reshape(3, 4, 3, 2)
        means = torch.linspace(-.4, .4, 24, dtype=torch.float64).reshape(1, 4, 3, 2)
        variance = torch.linspace(.4, 1.1, 6, dtype=torch.float64).reshape(3, 2)
        expected = torch.distributions.Independent(
            torch.distributions.Normal(means[0], variance.sqrt()), 2).log_prob(samples)
        for components in (means, means.expand(5, -1, -1, -1)):
            torch.testing.assert_close(
                joint_gaussian_mixture_log_density(samples, components, variance),
                expected, rtol=1e-12, atol=1e-12)

    def test_joint_mixture_cannot_mix_different_particles_per_coordinate(self):
        samples = torch.tensor([[[[4., -4.]]]], dtype=torch.float64)
        means = torch.tensor([[[[4., 4.]]], [[[-4., -4.]]]], dtype=torch.float64)
        variance = torch.full((1, 2), .25, dtype=torch.float64)
        per_coordinate = torch.distributions.Normal(means, variance.sqrt()).log_prob(samples[:, None])
        joint = torch.logsumexp(per_coordinate.sum((-1, -2)), 1) - math.log(2)
        coordinatewise = (torch.logsumexp(per_coordinate, 1) - math.log(2)).sum((-1, -2))
        actual = joint_gaussian_mixture_log_density(samples, means, variance)
        torch.testing.assert_close(actual, joint, rtol=1e-12, atol=1e-12)
        self.assertGreater(float((coordinatewise - actual).min()), 100)

    def test_matches_particle_major_replay_source_mapping_and_equal_horizon_weights(self):
        origins, horizons = (5, 2), (4, 1, 2)
        actual_streams, expected_streams = self.streams(), self.streams()
        original_transition = self.model.transition_mean
        seen_shapes = []
        def transition(state, controls):
            seen_shapes.append((tuple(state.shape), tuple(controls.shape)))
            return original_transition(state, controls)
        with patch.object(self.model, "transition_mean", side_effect=transition):
            actual = self.objective(origins=origins, horizons=horizons, streams=actual_streams)
        expected = self.replay(origins, horizons, expected_streams)
        self.assert_terms_equal(actual, expected)
        self.assertEqual(seen_shapes, [((32, 3, 2), (32, 1))] * 4)
        for actual_generator, expected_generator in zip(actual_streams, expected_streams):
            self.assertTrue(torch.equal(actual_generator.get_state(), expected_generator.get_state()))

    def test_single_particle_mc_converges_to_original_analytic_conditional_kl(self):
        with torch.no_grad():
            self.model.process_log_variance.zero_()
        actual_streams = self.streams()
        actual = self.objective(horizons=(1,), streams=actual_streams,
                                prediction_particles=1, posterior_target_draws=65536)
        reference_streams = self.streams()
        expected = prior_rollout_loss(
            self.model, self.mean, self.variance, self.controls,
            (2, 5), (1,), reference_streams[0], reference_streams[1])
        torch.testing.assert_close(actual["loss"], expected["loss"], rtol=.01, atol=.03)
        for actual_generator, expected_generator in zip(actual_streams[:2], reference_streams[:2]):
            self.assertTrue(torch.equal(actual_generator.get_state(), expected_generator.get_state()))

    def test_control_endpoints_and_intermediate_posterior_do_not_reset_paths(self):
        arguments = dict(origins=(2,), horizons=(1, 3))
        original = self.objective(**arguments)
        mean, variance, controls = self.mean.clone(), self.variance.clone(), self.controls.clone()
        unused = [0, 3, 5, 6, 7, 8, 9]
        mean[:, unused] += 100
        variance[:, unused] += 100
        controls[:, :2] += 100
        controls[:, 5:] -= 100
        self.assert_terms_equal(original, self.objective(
            mean=mean, variance=variance, controls=controls, **arguments))
        changed_target = self.mean.clone()
        changed_target[:, 2] += 10
        after_target = self.objective(mean=changed_target, **arguments)
        self.assertFalse(torch.equal(original["prior_mixture_kl_h1"], after_target["prior_mixture_kl_h1"]))
        self.assertTrue(torch.equal(original["prior_mixture_kl_h3"], after_target["prior_mixture_kl_h3"]))
        changed_control = self.controls.clone()
        changed_control[:, 4] += 20
        after_control = self.objective(controls=changed_control, **arguments)
        self.assertTrue(torch.equal(original["prior_mixture_kl_h1"], after_control["prior_mixture_kl_h1"]))
        self.assertFalse(torch.equal(original["prior_mixture_kl_h3"], after_control["prior_mixture_kl_h3"]))

    def test_frozen_posterior_readout_and_controls_with_all_six_prior_gradients(self):
        counts = torch.zeros(2, 10, 7, dtype=torch.float64)
        counts[:, 1::3, ::2] = 1
        controls = self.controls.clone().requires_grad_()
        mean, variance = self.model.encode(counts, controls)
        mean.retain_grad()
        variance.retain_grad()
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        parameters = configure_prior_parameters(self.model)
        optimizer = torch.optim.SGD(parameters, lr=.001)
        with patch.object(self.model, "encode", side_effect=AssertionError("posterior must stay fixed")), \
                patch.object(self.model, "log_hazard", side_effect=AssertionError("no observation draws")):
            self.objective(mean=mean, variance=variance, controls=controls)["loss"].backward()
        self.assertIsNone(mean.grad)
        self.assertIsNone(variance.grad)
        self.assertIsNone(controls.grad)
        for name, parameter in self.model.named_parameters():
            if name in PRIOR_PARAMETER_NAMES:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
            else:
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)
        optimizer.step()
        for name, parameter in self.model.named_parameters():
            self.assertEqual(torch.equal(before[name], parameter), name not in PRIOR_PARAMETER_NAMES, name)

    def test_checkpointed_stream_continuation_and_global_rng_unchanged(self):
        streams = self.streams()
        global_before = torch.get_rng_state().clone()
        first = self.objective(streams=streams)
        continuation = tuple(generator.get_state().clone() for generator in streams)
        second = self.objective(streams=streams)
        after_second = tuple(generator.get_state().clone() for generator in streams)
        self.assertFalse(torch.equal(first["loss"], second["loss"]))
        for generator, state in zip(streams, continuation):
            generator.set_state(state)
        replayed = self.objective(streams=streams)
        self.assert_terms_equal(second, replayed)
        for generator, state in zip(streams, after_second):
            self.assertTrue(torch.equal(generator.get_state(), state))
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))

    def test_target_draw_count_does_not_change_prediction_streams(self):
        first, second = self.streams(), self.streams()
        self.objective(streams=first, posterior_target_draws=4)
        self.objective(streams=second, posterior_target_draws=7)
        for left, right in zip(first[:2], second[:2]):
            self.assertTrue(torch.equal(left.get_state(), right.get_state()))
        self.assertFalse(torch.equal(first[2].get_state(), second[2].get_state()))
        one_step = self.streams()
        before = one_step[1].get_state().clone()
        self.objective(streams=one_step, horizons=(1,))
        self.assertTrue(torch.equal(one_step[1].get_state(), before))

    def test_invalid_particle_and_target_counts_rejected_before_rng_advances(self):
        for field in ("prediction_particles", "posterior_target_draws"):
            for value in (0, -1, True, 1.5, None):
                with self.subTest(field=field, value=value):
                    streams = self.streams()
                    before = tuple(generator.get_state().clone() for generator in streams)
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        self.objective(streams=streams, **{field: value})
                    for generator, state in zip(streams, before):
                        self.assertTrue(torch.equal(generator.get_state(), state))

    def test_all_three_streams_require_distinct_objects_seeds_and_local_device(self):
        shared = torch.Generator().manual_seed(41)
        for streams in ((None, *self.streams()[1:]),
                        (self.streams()[0], None, self.streams()[2]),
                        (*self.streams()[:2], None),
                        (shared, self.streams()[1], shared),
                        self.streams(target_seed=42), self.streams(target_seed=41)):
            with self.subTest(streams=streams):
                before = tuple(generator.get_state().clone() if generator is not None else None
                               for generator in streams)
                with self.assertRaisesRegex(ValueError, "random streams"):
                    self.objective(streams=streams)
                for generator, state in zip(streams, before):
                    if generator is not None:
                        self.assertTrue(torch.equal(generator.get_state(), state))
        with self.assertRaisesRegex(ValueError, "device-local"):
            self.objective(mean=self.mean.to("meta"), variance=self.variance.to("meta"),
                           controls=self.controls.to("meta"))

    def test_invalid_native_inputs_and_process_variance_do_not_advance_rng(self):
        cases = [{"origins": (2, 2)}, {"horizons": (1, 0)}, {"origins": (8,)},
                 {"variance": torch.zeros_like(self.variance)},
                 {"mean": self.mean[:, :, :2], "variance": self.variance[:, :, :2]},
                 {"controls": self.controls.float()}]
        for arguments in cases:
            with self.subTest(arguments=tuple(arguments)):
                streams = self.streams()
                before = tuple(generator.get_state().clone() for generator in streams)
                with self.assertRaises(ValueError):
                    self.objective(streams=streams, **arguments)
                for generator, state in zip(streams, before):
                    self.assertTrue(torch.equal(generator.get_state(), state))
        with torch.no_grad():
            self.model.process_log_variance.fill_(float("-inf"))
        streams = self.streams()
        before = tuple(generator.get_state().clone() for generator in streams)
        with self.assertRaises(FloatingPointError):
            self.objective(streams=streams)
        for generator, state in zip(streams, before):
            self.assertTrue(torch.equal(generator.get_state(), state))

    def test_finite_negative_monte_carlo_estimate_is_not_clamped_or_rejected(self):
        with torch.no_grad():
            self.model.process_log_variance.zero_()
        streams = self.streams()
        original_randn = torch.randn
        def draw(shape, **kwargs):
            if kwargs["generator"] is streams[2]:
                return torch.zeros(shape, dtype=kwargs["dtype"], device=kwargs["device"])
            return original_randn(shape, **kwargs)
        with patch("torch.randn", side_effect=draw), patch.object(
                self.model, "transition_mean", side_effect=lambda state, controls: torch.zeros_like(state)):
            result = self.objective(mean=torch.zeros_like(self.mean), variance=torch.ones_like(self.variance),
                                    streams=streams, horizons=(1,), prediction_particles=1,
                                    posterior_target_draws=1)
        torch.testing.assert_close(result["loss"], torch.tensor(-3., dtype=torch.float64))

    def test_nonfinite_predictive_density_raises(self):
        with patch.object(self.model, "transition_mean", side_effect=lambda state, controls:
                          torch.full_like(state, float("inf"))):
            with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
                self.objective(horizons=(1,))


if __name__ == "__main__":
    unittest.main()
