"""Time, variance and likelihood contracts for checkpoint-only diagnostics."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from .population_state import ControlledPopulationState
from .population_posterior_diagnostic import diagnose_native_plan, diagnostic_boundary_vector, transition_delta_variance
from .refractory_observation import renewal_window, observation_nll


def model_example():
    return ControlledPopulationState(unit_ids=[0, 1], unit_field_index=[0, 1],
        field_names=["E", "I"], minimum_interval_ticks=[3, 3],
        tick_seconds=.001, bin_seconds=.005, input_size=1, latent_channels=2,
        encoder_size=3, initialization_seed=41, unit_chunk_size=1).double().eval()


def native_example():
    data = SimpleNamespace(unit_ids=np.array([0, 1]), field_ids=np.array(["E", "I"]),
        unit_field_index=np.array([0, 1]), minimum_interval_ticks=np.array([3, 3]),
        dt_ms=1., bin_ms=5., bin_ticks=5, duration_ticks=20,
        event_ticks=np.array([4, 7, 10, 14]), event_units=np.array([0, 0, 1, 0]),
        pulse_ticks=np.array([5]))
    def window(start, stop):
        count = np.zeros(((stop - start) // 5, 2), dtype=np.float64)
        selected = (data.event_ticks >= start) & (data.event_ticks < stop)
        np.add.at(count, ((data.event_ticks[selected] - start) // 5, data.event_units[selected]), 1)
        pulse = data.pulse_ticks[(data.pulse_ticks >= start) & (data.pulse_ticks < stop)]
        controls = np.bincount((pulse - start) // 5, minlength=len(count)).astype(float)[:, None]
        return count, controls
    data.window = window
    data.renewal_window = lambda start, stop: renewal_window(data, start, stop, data.minimum_interval_ticks)
    def ready(tick):
        last = np.full(2, -3)
        before = data.event_ticks < tick
        np.maximum.at(last, data.event_units[before], data.event_ticks[before])
        return np.maximum(last + 3 - tick, 0)
    data.ready_at = ready
    return data


class PosteriorDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = model_example()

    def test_delta_variance_matches_full_autograd_jacobian(self):
        mean = torch.tensor([[[.1, -.3], [.8, .2]]], dtype=torch.float64)
        variance = torch.tensor([[[.2, .7], [.1, .4]]], dtype=torch.float64)
        controls = torch.tensor([[1.]], dtype=torch.float64)
        jacobian = torch.autograd.functional.jacobian(
            lambda flat: self.model.transition_mean(flat.reshape_as(mean), controls).flatten(),
            mean.flatten()).reshape(4, 4)
        expected = jacobian.square() @ variance.flatten()
        actual = transition_delta_variance(self.model, mean, variance, controls).flatten()
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
        self.assertTrue(torch.any(jacobian[:2, 2:] != 0))

    def test_zero_variance_and_invalid_variance(self):
        mean = torch.zeros(1, 2, 2, dtype=torch.float64)
        controls = torch.zeros(1, 1, dtype=torch.float64)
        self.assertTrue(torch.equal(transition_delta_variance(self.model, mean, mean, controls), mean))
        with self.assertRaises(ValueError):
            transition_delta_variance(self.model, mean, mean - 1, controls)

    def test_native_boundary_exposure_and_event_nll(self):
        data = native_example()
        whole = data.renewal_window(0, 15)
        after = data.renewal_window(5, 15)
        np.testing.assert_array_equal(after.counts, whole.counts[1:])
        np.testing.assert_array_equal(after.eligible_ticks, [[1, 5], [5, 3]])
        np.testing.assert_array_equal(after.initial_ready_in_ticks, [2, 0])
        np.testing.assert_array_equal(after.terminal_ready_in_ticks, [2, 0])
        log_rate = torch.zeros(2, 2, dtype=torch.float64)
        actual = observation_nll(torch.tensor(after.counts), torch.tensor(after.eligible_ticks), log_rate, .001)
        wanted = ((after.eligible_ticks - after.counts) * .001
                  - after.counts * np.log(-np.expm1(-.001))).sum(-1)
        np.testing.assert_allclose(actual.numpy(), wanted, atol=1e-12, rtol=1e-12)

    def test_full_diagnostic_uses_current_controls_and_all_units(self):
        native = native_example()
        before = torch.get_rng_state().clone()
        result = diagnose_native_plan(self.model, native, 5, 15)
        counts, controls = native.window(0, 15)
        with torch.no_grad():
            mean, _ = self.model.encode(counts[None], controls[None])
            prior = self.model.transition_mean(mean[:, :-1], torch.tensor(controls[None, 1:]))
            rates = self.model.log_hazard(prior)[0]
            expected = observation_nll(torch.tensor(counts[1:]),
                torch.tensor(native.renewal_window(0, 15).eligible_ticks[1:]), rates, .001)
        np.testing.assert_allclose(result["one_step_event_nll"], expected.numpy(), atol=1e-12)
        np.testing.assert_allclose(result["one_step_event_nll_per_unit_bin"], expected.numpy() / 2)
        np.testing.assert_array_equal(result["controls"], [[1], [0]])
        np.testing.assert_array_equal(result["observed_total_spikes"], [1, 2])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        no_pulse = self.model.transition_mean(mean[:, :-1], torch.zeros(1, 2, 1, dtype=torch.float64))
        self.assertTrue(torch.any(prior[:, 0] != no_pulse[:, 0]))
        torch.testing.assert_close(prior[:, 1], no_pulse[:, 1], atol=0, rtol=0)

    def test_rejects_source_map_and_incomplete_prefix(self):
        native = native_example()
        with self.assertRaises(ValueError): diagnose_native_plan(self.model, native, 0, 15)
        native.unit_field_index = np.array([1, 0])
        with self.assertRaises(ValueError): diagnose_native_plan(self.model, native, 5, 15)

    def test_boundary_witness_normalizes_readiness_as_float64(self):
        values = dict(boundary_mean=np.ones((1, 2, 2), dtype=np.float32),
            boundary_variance=np.full((1, 2, 2), .1, dtype=np.float32),
            boundary_ready_in_ticks=np.array([[2, 0]], dtype=np.int64))
        actual = diagnostic_boundary_vector(values, self.model)
        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_array_equal(actual[:, -2:], [[2 / 3, 0]])
        np.testing.assert_array_equal(actual[:, 4:8], values["boundary_variance"].reshape(1, 4))


if __name__ == "__main__":
    unittest.main()
