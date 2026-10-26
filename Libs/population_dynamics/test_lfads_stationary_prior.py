"""Independent density/grid checks for the fixed LFADS stationary AR family."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from Libs.population_dynamics.lfads_stationary_prior import (
    StationaryARFitError, StationaryARMoments, fit_stationary_ar_prior, stationary_ar_nll,
)


def paths():
    base = np.array([[3, 1.4, .9, .2, -.2], [-2, -1.2, -.8, -.2, .1],
                     [1, .6, .5, .3, .2]], dtype=np.float32)
    return base[:, :, None] * np.array([.5, 1, 2, 3], dtype=np.float32)


def direct_nll(z, phi, variance):
    """Direct first-density plus transition-density oracle without moments."""
    z = z.astype(np.float64)
    total = 0.0
    for sequence in z:
        stationary_variance = variance / (1.0 - phi * phi)
        total += .5 * (math.log(2 * math.pi * stationary_variance)
                       + sequence[0] ** 2 / stationary_variance)
        for previous, current in zip(sequence[:-1], sequence[1:]):
            total += .5 * (math.log(2 * math.pi * variance)
                           + (current - phi * previous) ** 2 / variance)
    return total


def direct_profile(z, phi):
    z = z.astype(np.float64)
    q = math.fsum(float(v) for v in np.square(z[:, 1:] - phi * z[:, :-1]).flat)
    q += (1 - phi * phi) * math.fsum(float(v) for v in np.square(z[:, 0]).flat)
    variance = q / z.size
    return direct_nll(z, phi, variance), variance


def test_density_includes_stationary_first_sample_and_all_normalizers():
    data = paths()
    moment = StationaryARMoments(sequence_length=5).update(data)
    phi = np.array([.2, .4, .6, .8])
    variance = np.array([.3, .4, .5, .6])
    logtau = (-np.log(-np.log(phi)))[None]
    logvar = np.log(variance)[None]
    actual = stationary_ar_nll(moment, logtau, logvar)
    expected = [direct_nll(data[:, :, c], phi[c], variance[c]) for c in range(4)]
    np.testing.assert_allclose(actual["per_channel"], expected, rtol=2e-14, atol=2e-14)
    assert actual["total"] == pytest.approx(math.fsum(expected), rel=2e-14)
    transition_only = .5 * (
        data.shape[0] * (data.shape[1] - 1) * np.log(2 * math.pi * variance[0])
        + np.square(data[:, 1:, 0].astype(float) - phi[0] * data[:, :-1, 0]).sum() / variance[0])
    assert abs(actual["per_channel"][0] - transition_only) > 1
    assert actual["sequence_count"] == 3 and actual["sequence_length"] == 5


def test_streamed_full_paths_equal_one_batch_without_modifying_input():
    data = paths()
    saved = data.copy()
    batch = StationaryARMoments().update(data)
    streamed = StationaryARMoments(sequence_length=5)
    for sequence in data:
        streamed.update(sequence[None])
    assert batch.as_dict() == streamed.as_dict()
    first, second = fit_stationary_ar_prior(batch), fit_stationary_ar_prior(streamed)
    for name in ("logatau", "logevars"):
        np.testing.assert_array_equal(first[name], second[name])
        assert first[name].shape == (1, 4) and first[name].dtype == np.float32
    assert first["diagnostics"] == second["diagnostics"]
    json.dumps(first["diagnostics"], allow_nan=False)
    np.testing.assert_array_equal(data, saved)


def test_global_profile_optimum_agrees_with_direct_density_grid():
    data = paths()
    moment = StationaryARMoments().update(data)
    initial_logtau = np.full((1, 4), 0., dtype=np.float32)
    initial_logvar = np.full((1, 4), 0., dtype=np.float32)
    fit = fit_stationary_ar_prior(moment, initial_logtau, initial_logvar)
    diagnostics = fit["diagnostics"]
    for channel, row in enumerate(diagnostics["channels"]):
        expected, expected_variance = direct_profile(data[:, :, channel], row["phi"])
        assert row["profile_nll"] == pytest.approx(expected, rel=2e-13, abs=2e-13)
        assert row["innovation_variance"] == pytest.approx(expected_variance, rel=2e-13)
        grid_minimum = min(direct_profile(data[:, :, channel], phi)[0]
                           for phi in np.linspace(.0001, .9999, 3001))
        assert row["profile_nll"] <= grid_minimum + 1e-11
        for displacement in (-1e-4, 1e-4):
            assert expected < direct_profile(data[:, :, channel], row["phi"] + displacement)[0]
        transition_only_phi = np.sum(data[:, 1:, channel].astype(float) * data[:, :-1, channel])
        transition_only_phi /= np.square(data[:, :-1, channel].astype(float)).sum()
        assert abs(row["phi"] - transition_only_phi) > .1
    assert diagnostics["float32_candidate_nll"]["total"] < diagnostics["initial_parameter_nll"]["total"]
    assert diagnostics["continuous_optimum_nll"]["total"] <= diagnostics["float32_candidate_nll"]["total"] + 1e-11
    assert diagnostics["native_tensorflow_validated"] is False
    assert diagnostics["scientific_claim_qualified"] is False


def test_two_step_paths_have_linear_stationary_equation():
    data = np.array([[1., .4], [-2., -.8], [.5, .3]], dtype=np.float32)
    data = np.repeat(data[:, :, None], 4, axis=2)
    moment = StationaryARMoments(sequence_length=2).update(data)
    fit = fit_stationary_ar_prior(moment)
    a = np.square(data[:, :, 0].astype(float)).sum()
    b = np.sum(data[:, 0, 0].astype(float) * data[:, 1, 0])
    assert moment.as_dict()["C"] == [0.] * 4
    for row in fit["diagnostics"]["channels"]:
        assert row["phi"] == pytest.approx(2 * b / a, rel=2e-14)
        assert row["profile_nll"] == pytest.approx(
            direct_profile(data[:, :, 0], row["phi"])[0], rel=2e-14)


@pytest.mark.parametrize(("base", "reason"), [
    (np.zeros((2, 4), dtype=np.float32), "ZERO_VARIANCE_UNBOUNDED"),
    (np.array([[1, -1, 1, -1], [-2, 2, -2, 2]], dtype=np.float32),
     "NONINTERIOR_PHI_ZERO_OPTIMUM"),
    (np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32),
     "NONINTERIOR_PHI_ZERO_OPTIMUM"),
    (np.array([[1, 1, 1], [-2, -2, -2]], dtype=np.float32),
     "PHI_ONE_VARIANCE_ZERO_UNBOUNDED"),
])
def test_inadmissible_boundary_and_unbounded_fits_are_not_clipped(base, reason):
    data = np.repeat(base[:, :, None], 4, axis=2)
    with pytest.raises(StationaryARFitError, match=reason) as caught:
        fit_stationary_ar_prior(StationaryARMoments().update(data))
    assert [row["channel"] for row in caught.value.diagnostics["failures"]] == list(range(4))


@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_float32_source_range_failure_is_explicit(scale):
    data = (paths() * np.float32(scale)).astype(np.float32)
    with pytest.raises(StationaryARFitError, match="FLOAT32_SOURCE_ADMISSIBLE"):
        fit_stationary_ar_prior(StationaryARMoments().update(data))


@pytest.mark.parametrize("bad", [
    np.ones((2, 3, 4), dtype=np.float64),
    np.ones((2, 3, 4), dtype=np.int32),
    np.ones((2, 3), dtype=np.float32),
    np.ones((2, 3, 3), dtype=np.float32),
    np.ones((2, 1, 4), dtype=np.float32),
    np.ones((0, 3, 4), dtype=np.float32),
    [[[1., 2., 3., 4.], [1., 2., 3., 4.]]],
])
def test_wrong_dtype_shape_and_incomplete_paths_rejected(bad):
    with pytest.raises(ValueError):
        StationaryARMoments().update(bad)


def test_failed_update_preserves_existing_moments():
    data = paths()
    moment = StationaryARMoments().update(data[:1])
    before = moment.as_dict()
    changed = data.copy()
    changed[-1, -1, -1] = np.nan
    with pytest.raises(ValueError, match="FINITE_NATIVE"):
        moment.update(changed)
    assert moment.as_dict() == before
    with pytest.raises(ValueError, match="LENGTH_CHANGED"):
        moment.update(data[:, :-1])
    assert moment.as_dict() == before


def test_empty_moments_bad_lengths_and_parameter_errors():
    for length in (0, 1, 2.5, True):
        with pytest.raises(ValueError, match="FULL_SEQUENCE_LENGTH"):
            StationaryARMoments(sequence_length=length)
    with pytest.raises(ValueError, match="NONEMPTY"):
        fit_stationary_ar_prior(StationaryARMoments())
    moment = StationaryARMoments().update(paths())
    zero_logs = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="BOTH_INITIAL"):
        fit_stationary_ar_prior(moment, initial_logatau=zero_logs)
    for bad in (zero_logs[0], np.zeros((1, 4), dtype=np.int32),
                np.full((1, 4), np.nan), np.full((1, 4), np.inf)):
        with pytest.raises(ValueError, match="LOG_PARAMETER"):
            stationary_ar_nll(moment, bad, zero_logs)
    for extreme in (-1000., 1000.):
        with pytest.raises(ValueError, match="INTERIOR"):
            stationary_ar_nll(moment, np.full((1, 4), extreme), zero_logs)

