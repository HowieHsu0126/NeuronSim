"""Analytical native-tick probabilities and full-unit chunk accounting."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location("activity_diagnostic", Path(__file__).with_name("population_activity_diagnostic.py"))
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class IdentifiedHazards:
    unit_ids = (11, 22, 33, 44, 55)
    field_names = ("E", "I")
    latent_channels = 1
    tick_seconds = .1
    bin_ticks = 4
    bin_seconds = .4

    def __init__(self):
        rates = torch.tensor([1., 2., 3., 4., 10.], dtype=torch.float64)
        self.rates = rates[None, None].expand(2, 3, -1).clone()
        self.rates[1] *= 2
        self.rates[:, 1] *= 3

    def log_hazard(self, latent, start=0, stop=None):
        return self.rates[..., start:stop].log()


def test_native_tick_probability_is_distinct_from_raw_intensity():
    model = IdentifiedHazards()
    latent = torch.zeros((2, 3, 2, 1), dtype=torch.float64)
    result = diagnostic.decode_activity(model, latent, (1, 3), unit_chunk_size=2)
    raw = (model.rates * model.bin_seconds).sum(-1).cumsum(1).numpy()
    finite = (-torch.expm1(-model.rates * model.tick_seconds) * model.bin_ticks).sum(-1).cumsum(1).numpy()
    for bins in (1, 3):
        np.testing.assert_allclose(result[bins]["raw_integrated_intensity"], raw[:, bins - 1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result[bins]["all_ticks_available_expectation"], finite[:, bins - 1], rtol=1e-12, atol=1e-12)
        assert np.all(result[bins]["raw_integrated_intensity"] > result[bins]["all_ticks_available_expectation"])


def test_chunk_remainder_preserves_every_identified_unit_and_draw():
    model = IdentifiedHazards()
    latent = torch.zeros((2, 3, 2, 1), dtype=torch.float64)
    whole = diagnostic.decode_activity(model, latent, (1, 2, 3), unit_chunk_size=5)
    for chunk in (1, 2, 4, 8):
        pieces = diagnostic.decode_activity(model, latent, (1, 2, 3), unit_chunk_size=chunk)
        for bins in (1, 2, 3):
            for key in whole[bins]:
                np.testing.assert_allclose(pieces[bins][key], whole[bins][key], rtol=1e-12, atol=1e-12)


def test_gaussian_tick_bounds_enclose_exact_deterministic_probability():
    rate = np.array([1e-4, 1., 20., 1e4])
    for tick in (.0001, .1):
        result = diagnostic.gaussian_tick_bounds(np.log(rate), np.zeros_like(rate), tick)
        exact = -np.expm1(-rate * tick)
        np.testing.assert_allclose(result["expected_rate_hz"], rate, rtol=1e-14)
        assert np.all(result["probability_lower"] <= exact + 1e-15)
        assert np.all(exact <= result["probability_upper"] + 1e-15)
        assert np.all(result["probability_lower"] >= 0)
        assert np.all(result["probability_upper"] <= 1)


def test_gaussian_variance_changes_both_analytic_moments():
    result = diagnostic.gaussian_tick_bounds(np.array([np.log(2.)]), np.array([np.log(4.)]), .1)
    np.testing.assert_allclose(result["expected_rate_hz"], [4.], rtol=1e-14)
    np.testing.assert_allclose(result["first_tick_hazard_moment"], [.4], rtol=1e-14)
    np.testing.assert_allclose(result["second_tick_hazard_moment"], [.64], rtol=1e-14)
    np.testing.assert_allclose(result["probability_lower"], [.08], rtol=1e-14)
    np.testing.assert_allclose(result["probability_upper"], [.4], rtol=1e-14)


class GaussianReadout:
    unit_ids = (11, 22, 33)
    field_names = ("E", "I")
    latent_channels = 1
    tick_seconds = .01
    bin_ticks = 4
    bin_seconds = .04
    unit_field_index = torch.tensor([0, 1, 0])
    unit_loading = torch.tensor([[1.], [2.], [1.]], dtype=torch.float64)
    bias = torch.tensor([0., 0., np.log(2.)], dtype=torch.float64)

    def log_hazard(self, latent, start=0, stop=None):
        selected = latent.index_select(-2, self.unit_field_index[start:stop])
        return (selected * self.unit_loading[start:stop]).sum(-1) + self.bias[start:stop]


def test_posterior_bounds_keep_variance_exposure_and_remainder_units():
    model = GaussianReadout()
    mean = torch.tensor([[[[np.log(2.)], [np.log(3.)]], [[np.log(2.)], [np.log(3.)]]]], dtype=torch.float64)
    variance = torch.tensor([[[[.25], [.5]], [[.25], [.5]]]], dtype=torch.float64)
    exposure = np.array([[4, 3, 0], [0, 0, 0]])
    expected_rate = np.array([2 * np.exp(.125), 9 * np.exp(1.), 4 * np.exp(.125)])
    first = expected_rate * .01
    second = np.array([4 * np.exp(.5), 81 * np.exp(4.), 16 * np.exp(.5)]) * .01**2
    lower, upper = np.maximum(0, first - .5 * second), np.minimum(1, first)
    for chunk in (1, 2, 3, 4):
        result = diagnostic.posterior_activity_bounds(model, mean, variance, exposure, unit_chunk_size=chunk)
        np.testing.assert_allclose(result["mean_state_integrated_intensity"], [15 * .04] * 2, rtol=1e-14)
        np.testing.assert_allclose(result["posterior_integrated_intensity"], [expected_rate.sum() * .04] * 2, rtol=1e-14)
        np.testing.assert_allclose(result["all_ticks_expectation_lower"], [lower.sum() * 4] * 2, rtol=1e-14)
        np.testing.assert_allclose(result["all_ticks_expectation_upper"], [upper.sum() * 4] * 2, rtol=1e-14)
        np.testing.assert_allclose(result["observed_path_expectation_lower"], [lower @ exposure[0], 0.], rtol=1e-14)
        np.testing.assert_allclose(result["observed_path_expectation_upper"], [upper @ exposure[0], 0.], rtol=1e-14)


def test_posterior_bounds_reject_invalid_native_exposure():
    model = GaussianReadout()
    mean = torch.zeros((1, 1, 2, 1), dtype=torch.float64)
    for exposure in (np.array([[5, 0, 0]]), np.array([[.5, 0, 0]]), np.array([[1, 2]])):
        with pytest.raises(ValueError, match="eligible exposure"):
            diagnostic.posterior_activity_bounds(model, mean, mean, exposure)
