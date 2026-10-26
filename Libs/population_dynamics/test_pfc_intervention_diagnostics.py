"""Check scientific reductions and identity failures for saved PFC diagnostics."""
import numpy as np
import pytest

from . import pfc_intervention_diagnostics as diagnostic
from .trajectory_scores import energy_score


def test_decomposition_matches_existing_score_and_scale():
    samples = np.arange(48, dtype=float).reshape(4, 4, 3) / 7
    references = np.arange(36, dtype=float).reshape(3, 4, 3) / 11
    result = diagnostic.components(samples, references)
    assert result["reference_energy_score"] == pytest.approx(
        np.mean([energy_score(samples, row) for row in references]), abs=1e-14)
    assert result["reference_energy_score"] == result["target_distance"]-result["dispersion_subtraction"]
    doubled = diagnostic.components(samples*2, references*2)
    for key in result:
        assert doubled[key] == pytest.approx(result[key]*2)


def test_identical_draws_have_zero_dispersion():
    result = diagnostic.components(np.ones((16, 4, 2)), np.zeros((8, 4, 2)))
    assert result == dict(target_distance=1., dispersion_subtraction=0.,
                          reference_energy_score=1., ensemble_mean_reference_distance=1.)


@pytest.mark.parametrize("samples,references", [
    (np.ones((1, 4, 2)), np.zeros((8, 4, 2))),
    (np.ones((2, 4, 2)), np.zeros((8, 2, 4))),
    (np.full((2, 4, 2), np.nan), np.zeros((8, 4, 2))),
])
def test_invalid_score_inputs_rejected(samples, references):
    with pytest.raises(ValueError, match="FINITE_MATCHING"):
        diagnostic.components(samples, references)


def test_ranks_retain_ties():
    assert diagnostic.ranks([3, 1, 1, 2]) == [(4., 1), (1.5, 2), (1.5, 2), (3., 1)]


def test_complete_grid_rejects_duplicate_or_missing_identity():
    expected = [(41, "a"), (42, "a"), (43, "a")]
    rows = [dict(seed=seed, case=case) for seed, case in expected]
    assert len(diagnostic.index_rows(rows, ("seed", "case"), expected)) == 3
    for broken in (rows[:-1], rows+[rows[0]], rows[:-1]+[dict(seed=44, case="a")]):
        with pytest.raises(ValueError, match="COMPLETE_UNIQUE"):
            diagnostic.index_rows(broken, ("seed", "case"), expected)


def config(strength=.5, duration=50):
    return dict(stimulus_start_ms=100, stimulus_duration_ms=200, stimulus_strength_nA=.5,
                distractor_start_ms=400, distractor_duration_ms=duration, distractor_strength_nA=strength)


def test_phases_partition_full_task_and_ignore_inactive_stimulus():
    assert diagnostic.phase_windows([config()]) == [
        (100., 300., 0, 40), (300., 400., 40, 60),
        (400., 450., 60, 70), (450., 5000., 70, 980)]
    assert diagnostic.phase_windows([config(strength=0)]) == [(100., 300., 0, 40), (300., 5000., 40, 980)]
    with pytest.raises(ValueError, match="BOUNDARIES_MUST_ALIGN"):
        diagnostic.phase_windows([config(duration=51)])


def test_temporal_contrasts_preserve_spatial_error():
    native = np.array([[1., -1.], [2., -2.]])
    samples = np.stack([native+.5, native-.5])
    result = diagnostic.temporal_metrics(samples, native, np.zeros_like(native), np.zeros_like(native))
    np.testing.assert_array_equal(result["mean_error_rms"], [0., 0.])
    np.testing.assert_allclose(result["ensemble_spread_rms"], np.sqrt(.5))
    np.testing.assert_array_equal(result["native_contrast_rms"], [1., 2.])
    np.testing.assert_array_equal(result["contrast_error_rms"], [0., 0.])


def test_saved_arrays_reject_field_permutation_and_unit_permutation(tmp_path):
    fields, units = np.arange(17), np.arange(100000)
    path = tmp_path/"source.npz"
    np.savez(path, field_rate_hz=np.zeros((980, 17)), field_ids=fields, unit_ids=units)
    diagnostic.field_arrays(path, fields, units, native=True)
    with pytest.raises(ValueError, match="FIELD_ORDER"):
        diagnostic.field_arrays(path, fields[::-1], units, native=True)
    with pytest.raises(ValueError, match="UNIT_ORDER"):
        diagnostic.field_arrays(path, fields, units[::-1], native=True)


def test_saved_reference_rejects_invalid_frozen_scale(tmp_path):
    path = tmp_path/"reference.npz"
    np.savez(path, field_rate_hz=np.zeros((8, 980, 17)), field_ids=np.arange(17),
             scale_hz=np.zeros(17), block_ids=np.arange(8))
    with pytest.raises(ValueError, match="POSITIVE_FROZEN"):
        diagnostic.field_arrays(path, reference=True)
