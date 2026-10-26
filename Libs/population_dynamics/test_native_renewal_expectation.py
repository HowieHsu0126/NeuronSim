"""Independent event-path enumeration and gradients for native count means."""
import itertools
import math

import numpy as np
import pytest
import torch

from .native_renewal_expectation import expected_sequence_counts


def enumerate_means(log_rates, initial_ready, interval, bin_ticks, tick_seconds):
    probabilities = -np.expm1(-np.exp(np.asarray(log_rates)) * tick_seconds)
    ticks = len(probabilities) * bin_ticks
    result, total_mass = np.zeros(len(probabilities)), 0.
    for path in itertools.product((0, 1), repeat=ticks):
        ready, mass = initial_ready, 1.
        counts = np.zeros(len(probabilities))
        for tick, event in enumerate(path):
            if tick < ready:
                if event:
                    mass = 0.
                    break
            else:
                probability = probabilities[tick // bin_ticks]
                mass *= probability if event else 1 - probability
                if event:
                    counts[tick // bin_ticks] += 1
                    ready = tick + interval
        result += mass * counts
        total_mass += mass
    assert total_mass == pytest.approx(1., abs=1e-13)
    return result


@pytest.mark.parametrize("interval", [1, 2, 3, 5, 9])
def test_exhaustive_paths_with_varying_hazards_and_every_boundary_readiness(interval):
    for hazards in ([.05, 1.1, 4.], [2., .3, .02], [.5, .5, .5]):
        rates = torch.tensor(np.log(np.asarray(hazards) / .001), dtype=torch.float64).reshape(3, 1)
        for initial_ready in range(interval):
            result = expected_sequence_counts(rates, torch.tensor([initial_ready]), torch.tensor([interval]), 2, .001)
            exact = enumerate_means(rates[:, 0].numpy(), initial_ready, interval, 2, .001)
            np.testing.assert_allclose(result[:, 0].numpy(), exact, rtol=1e-12, atol=1e-13)


def test_shuffled_intervals_and_leading_dimensions_preserve_every_unit():
    rates = (4. + torch.arange(48, dtype=torch.float64).reshape(2, 2, 3, 4) * .03).transpose(0, 1)
    intervals = torch.tensor([3, 1, 5, 2])
    ready = torch.tensor([[[0, 0, 4, 1], [2, 0, 0, 0]], [[1, 0, 2, 1], [0, 0, 3, 0]]])
    result = expected_sequence_counts(rates, ready, intervals, 2, .001)
    assert result.shape == rates.shape
    for leading in np.ndindex(rates.shape[:-2]):
        for unit, interval in enumerate(intervals.tolist()):
            exact = enumerate_means(rates[leading][:, unit].numpy(), int(ready[leading][unit]), interval, 2, .001)
            np.testing.assert_allclose(result[leading][:, unit].numpy(), exact, rtol=1e-12, atol=1e-13)


def test_cross_bin_refractory_exclusion_and_earlier_rate_gradients():
    probabilities = torch.tensor([.8, .4, .7], dtype=torch.float64)
    rates = torch.log(-torch.log1p(-probabilities) / .001).reshape(3, 1).requires_grad_()
    result = expected_sequence_counts(rates, torch.tensor([0]), torch.tensor([3]), 1, .001)
    expected = torch.stack((probabilities[0], (1 - probabilities[0]) * probabilities[1],
                            (1 - probabilities[0]) * (1 - probabilities[1]) * probabilities[2]))
    torch.testing.assert_close(result[:, 0], expected, rtol=1e-12, atol=1e-13)
    gradient = torch.autograd.grad(result[-1, 0], rates)[0][:, 0]
    assert gradient[0] < 0 and gradient[1] < 0 and gradient[2] > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_interval_one_matches_analytic_bin_means_and_preserves_dtype_device(dtype):
    rates = torch.linspace(-4., 4., 12, dtype=dtype).reshape(2, 3, 2)
    ready = torch.zeros((2, 2), dtype=torch.int32)
    result = expected_sequence_counts(rates, ready, torch.ones(2, dtype=torch.int32), 7, .001)
    expected = -torch.expm1(-rates.exp() * .001) * 7
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype == rates.dtype and result.device == rates.device


def test_saturated_finite_hazard_releases_at_the_exact_minimum_interval():
    rates = torch.full((3, 1), math.log(1e9), dtype=torch.float64)
    result = expected_sequence_counts(rates, torch.tensor([2]), torch.tensor([3]), 3, .001)
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)


def test_gradcheck_through_delayed_events_and_model_bin_boundaries():
    rates = (4. + torch.arange(18, dtype=torch.float64).reshape(2, 3, 3) * .01).requires_grad_()
    ready, intervals = torch.tensor([[0, 1, 0], [0, 2, 1]]), torch.tensor([1, 3, 2])
    assert torch.autograd.gradcheck(lambda value: expected_sequence_counts(value, ready, intervals, 2, .001), (rates,))


def test_inputs_and_rng_state_are_unchanged():
    rates = torch.tensor([[4., 5.], [6., 3.]], dtype=torch.float64, requires_grad=True)
    ready, intervals = torch.tensor([2, 0]), torch.tensor([3, 1])
    saved = [value.detach().clone() for value in (rates, ready, intervals)]
    rng = torch.get_rng_state().clone()
    first = expected_sequence_counts(rates, ready, intervals, 3, .001)
    second = expected_sequence_counts(rates, ready, intervals, 3, .001)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.requires_grad and rates.grad is None
    for actual, expected in zip((rates, ready, intervals), saved):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["not_tensor", "rank", "empty", "integer_rate", "nonfinite_rate",
    "ready_shape", "ready_dtype", "negative_ready", "ready_equals_interval", "interval_shape", "interval_dtype", "zero_interval"])
def test_invalid_rates_and_source_boundary_inputs_fail(change):
    rates = torch.zeros((2, 2), dtype=torch.float64)
    ready, intervals = torch.zeros(2, dtype=torch.int64), torch.tensor([2, 3])
    if change == "not_tensor":
        rates = [[0., 0.], [0., 0.]]
    elif change == "rank":
        rates = rates[0]
    elif change == "empty":
        rates = rates[:0]
    elif change == "integer_rate":
        rates = rates.to(torch.int64)
    elif change == "nonfinite_rate":
        rates[0, 0] = float("nan")
    elif change == "ready_shape":
        ready = ready[None]
    elif change == "ready_dtype":
        ready = ready.to(torch.float64)
    elif change == "negative_ready":
        ready[0] = -1
    elif change == "ready_equals_interval":
        ready[0] = intervals[0]
    elif change == "interval_shape":
        intervals = intervals[None]
    elif change == "interval_dtype":
        intervals = intervals.to(torch.float64)
    else:
        intervals[0] = 0
    with pytest.raises(ValueError, match="(floating log rates|boundary readiness)"):
        expected_sequence_counts(rates, ready, intervals, 2, .001)


@pytest.mark.parametrize("parameter,value", [
    ("bin_ticks", 0), ("bin_ticks", True), ("bin_ticks", 2.0),
    ("tick_seconds", 0.), ("tick_seconds", -.001), ("tick_seconds", float("nan")),
    ("tick_seconds", float("inf")), ("tick_seconds", True),
])
def test_invalid_native_clock_fails(parameter, value):
    clock = dict(bin_ticks=2, tick_seconds=.001)
    clock[parameter] = value
    with pytest.raises(ValueError, match="native bin"):
        expected_sequence_counts(torch.zeros((2, 1)), torch.tensor([0]), torch.tensor([2]), **clock)


@pytest.mark.parametrize("log_rate", [-1000., 1000.])
def test_hazard_underflow_and_overflow_fail_without_clipping(log_rate):
    with pytest.raises(FloatingPointError, match="hazard outside"):
        expected_sequence_counts(torch.full((2, 1), log_rate, dtype=torch.float64),
                                 torch.tensor([0]), torch.tensor([2]), 2, .001)
