import unittest
from unittest.mock import patch

import torch

from .population_prior_objective import (
    PRIOR_PARAMETER_NAMES,
    configure_prior_parameters,
    prior_rollout_loss,
)
from .test_population_state import example


class PopulationPriorObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = example().double()
        data_generator = torch.Generator().manual_seed(43)
        self.mean = torch.randn(2, 10, 3, 2, dtype=torch.float64, generator=data_generator)
        self.variance = .1 + torch.rand(
            self.mean.shape, dtype=torch.float64, generator=data_generator)
        self.controls = torch.linspace(-1, 1, 20, dtype=torch.float64).reshape(2, 10, 1)

    @staticmethod
    def streams(initial_seed=41, dynamics_seed=42):
        return (torch.Generator().manual_seed(initial_seed),
                torch.Generator().manual_seed(dynamics_seed))

    def objective(self, *, mean=None, variance=None, controls=None,
                  origins=(2, 5), horizons=(1, 3), streams=None):
        return prior_rollout_loss(
            self.model, self.mean if mean is None else mean,
            self.variance if variance is None else variance,
            self.controls if controls is None else controls,
            origins, horizons, *(self.streams() if streams is None else streams))

    def replay(self, origins, horizons, streams):
        # Independent Gaussian formula, preserving explicit batch/origin axes.
        shape = (self.mean.shape[0], len(origins), *self.mean.shape[2:])
        boundary = [origin - 1 for origin in origins]
        initial, dynamics = streams
        state = self.mean[:, boundary] + self.variance[:, boundary].sqrt() * torch.randn(
            shape, dtype=self.mean.dtype, generator=initial)
        process_variance = self.model.process_log_variance.exp()
        terms = {}
        for step in range(1, max(horizons) + 1):
            indices = [origin + step - 1 for origin in origins]
            prior_mean = self.model.transition_mean(state, self.controls[:, indices])
            if step in horizons:
                target_mean, target_variance = self.mean[:, indices], self.variance[:, indices]
                per_coordinate = .5 * (
                    torch.log(process_variance / target_variance) - 1
                    + (target_variance + (target_mean - prior_mean).square()) / process_variance)
                terms[f"prior_kl_h{step}"] = per_coordinate.sum((-1, -2)).mean()
            if step < max(horizons):
                state = prior_mean + process_variance.sqrt() * torch.randn(
                    shape, dtype=self.mean.dtype, generator=dynamics)
        terms["loss"] = torch.stack([terms[f"prior_kl_h{h}"] for h in horizons]).mean()
        return terms

    def assert_terms_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-12, atol=1e-12)

    def test_matches_exact_stochastic_replay_and_equal_horizon_weight(self):
        origins, horizons = (2, 5), (4, 1, 2)
        actual_streams, expected_streams = self.streams(), self.streams()
        actual = self.objective(origins=origins, horizons=horizons, streams=actual_streams)
        expected = self.replay(origins, horizons, expected_streams)
        self.assert_terms_equal(actual, expected)
        for actual_generator, expected_generator in zip(actual_streams, expected_streams):
            self.assertTrue(torch.equal(
                actual_generator.get_state(), expected_generator.get_state()))

    def test_one_step_integrates_last_noise_without_consuming_dynamics_rng(self):
        streams = self.streams()
        dynamics_before = streams[1].get_state().clone()
        result = self.objective(horizons=(1,), streams=streams)
        self.assert_terms_equal(result, self.replay((2, 5), (1,), self.streams()))
        self.assertTrue(torch.equal(dynamics_before, streams[1].get_state()))
        self.assert_terms_equal(
            result, self.objective(horizons=(1,), streams=self.streams(dynamics_seed=43)))
        changed_initial = self.objective(horizons=(1,), streams=self.streams(initial_seed=43))
        self.assertFalse(torch.equal(result["loss"], changed_initial["loss"]))

    def test_control_indices_and_unused_posterior_bins(self):
        origins, horizons = (2,), (1, 3)
        result = self.objective(origins=origins, horizons=horizons)
        changed_controls = self.controls.clone()
        changed_controls[:, :2] += 100
        changed_controls[:, 5:] -= 100
        changed_mean = self.mean.clone()
        changed_variance = self.variance.clone()
        unused = [0, 3, 5, 6, 7, 8, 9]
        changed_mean[:, unused] += 100
        changed_variance[:, unused] += 100
        self.assert_terms_equal(result, self.objective(
            mean=changed_mean, variance=changed_variance, controls=changed_controls,
            origins=origins, horizons=horizons))
        first_control_changed = self.controls.clone()
        first_control_changed[:, 2] += 20
        self.assertFalse(torch.equal(result["prior_kl_h1"], self.objective(
            controls=first_control_changed, origins=origins, horizons=horizons)["prior_kl_h1"]))
        last_control_changed = self.controls.clone()
        last_control_changed[:, 4] += 20
        after = self.objective(controls=last_control_changed, origins=origins, horizons=horizons)
        self.assertTrue(torch.equal(result["prior_kl_h1"], after["prior_kl_h1"]))
        self.assertFalse(torch.equal(result["prior_kl_h3"], after["prior_kl_h3"]))

    def test_targets_are_detached_and_only_prior_parameters_update(self):
        counts = torch.zeros(2, 10, 7, dtype=torch.float64)
        counts[:, 1::3, ::2] = 1
        controls = self.controls.clone().requires_grad_()
        mean, variance = self.model.encode(counts, controls)
        mean.retain_grad()
        variance.retain_grad()
        before = {name: parameter.detach().clone()
                  for name, parameter in self.model.named_parameters()}
        # A stale encoder gradient must not survive freezing.
        next(self.model.encoder.parameters()).grad = torch.ones_like(
            next(self.model.encoder.parameters()))
        parameters = configure_prior_parameters(self.model)
        self.assertEqual(tuple(name for name, parameter in self.model.named_parameters()
                               if parameter.requires_grad), PRIOR_PARAMETER_NAMES)
        self.assertEqual(tuple(id(parameter) for parameter in parameters),
                         tuple(id(getattr(self.model, name)) for name in PRIOR_PARAMETER_NAMES))
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
            if name in PRIOR_PARAMETER_NAMES:
                self.assertFalse(torch.equal(before[name], parameter), name)
            else:
                self.assertTrue(torch.equal(before[name], parameter), name)

    def test_generator_state_restore_replays_continuation_without_global_rng(self):
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

    def test_process_noise_affects_later_horizons(self):
        first = self.objective(horizons=(1, 4))
        changed = self.objective(horizons=(1, 4), streams=self.streams(dynamics_seed=43))
        self.assertTrue(torch.equal(first["prior_kl_h1"], changed["prior_kl_h1"]))
        self.assertFalse(torch.equal(first["prior_kl_h4"], changed["prior_kl_h4"]))

    def test_rejects_invalid_origins_horizons_before_consuming_rng(self):
        cases = [
            {"origins": ()}, {"origins": (0,)}, {"origins": (-1,)},
            {"origins": (2, 2)}, {"origins": (True,)}, {"origins": (2.0,)},
            {"origins": 2}, {"origins": (8,), "horizons": (3,)},
            {"horizons": ()}, {"horizons": (0,)}, {"horizons": (-1,)},
            {"horizons": (1, 1)}, {"horizons": (False,)}, {"horizons": (1.5,)},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                streams = self.streams()
                before = tuple(generator.get_state().clone() for generator in streams)
                with self.assertRaises(ValueError):
                    self.objective(**arguments, streams=streams)
                for generator, state in zip(streams, before):
                    self.assertTrue(torch.equal(generator.get_state(), state))

    def test_rejects_malformed_posterior_and_control_tensors(self):
        cases = [
            {"mean": self.mean[:, :, 0]},
            {"mean": self.mean[:, :0]},
            {"variance": self.variance[:, :-1]},
            {"mean": self.mean[:, :, :2], "variance": self.variance[:, :, :2]},
            {"controls": self.controls[:, :-1]},
            {"controls": self.controls.float()},
            {"mean": self.mean.float(), "variance": self.variance.float(),
             "controls": self.controls.float()},
            {"variance": torch.zeros_like(self.variance)},
            {"variance": torch.full_like(self.variance, float("nan"))},
            {"mean": torch.full_like(self.mean, float("inf"))},
            {"controls": torch.full_like(self.controls, float("nan"))},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    self.objective(**arguments)

    def test_rejects_invalid_generators_and_device_mismatch(self):
        shared = torch.Generator().manual_seed(41)
        for streams in ((None, shared), (shared, None), (shared, shared),
                        self.streams(initial_seed=41, dynamics_seed=41)):
            with self.subTest(streams=streams):
                with self.assertRaisesRegex(ValueError, "random streams"):
                    self.objective(streams=streams)
        # Meta tensors exercise device validation without a CUDA dependency.
        with self.assertRaisesRegex(ValueError, "device-local"):
            self.objective(mean=self.mean.to("meta"), variance=self.variance.to("meta"),
                           controls=self.controls.to("meta"))

    def test_rejects_non_native_or_missing_parameter_profile_before_freeze(self):
        self.model.metadata = {}
        before = tuple(parameter.requires_grad for parameter in self.model.parameters())
        with self.assertRaisesRegex(ValueError, "native unshared"):
            configure_prior_parameters(self.model)
        with self.assertRaisesRegex(ValueError, "native unshared"):
            self.objective()
        self.assertEqual(before, tuple(parameter.requires_grad for parameter in self.model.parameters()))
        self.model.metadata = None
        del self.model.field_mix_logits
        with self.assertRaisesRegex(ValueError, "field_mix_logits"):
            configure_prior_parameters(self.model)
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))

    def test_rejects_invalid_process_variance_without_advancing_rng(self):
        with torch.no_grad():
            self.model.process_log_variance.fill_(float("-inf"))
        streams = self.streams()
        before = tuple(generator.get_state().clone() for generator in streams)
        with self.assertRaises(FloatingPointError):
            self.objective(streams=streams)
        for generator, state in zip(streams, before):
            self.assertTrue(torch.equal(generator.get_state(), state))


if __name__ == "__main__":
    unittest.main()
