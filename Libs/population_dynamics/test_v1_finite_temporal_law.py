"""Finite-seed ensemble moments, native bins and silent denominators."""
import numpy as np
import pytest
from .v1_finite_temporal_law import temporal_counts, temporal_moments, ensemble_covariance


def test_native_bins_keep_multiplicity_and_endpoint():
    counts, terminal = temporal_counts(np.array([.125, .125, 1., 2999.999, 3000.]))
    assert counts.shape == (3000,) and counts.sum() == 4 and terminal == 1
    assert counts[0] == 2 and counts[1] == 1 and counts[-1] == 1


def test_finite_reference_matches_direct_seed_ensemble_centering():
    table = np.tile(np.array([[0,1,2,0],[2,1,0,0]], dtype=np.int32), (5000,1))
    result = temporal_moments(table, lags=(0,1,2))
    assert result['mean_count'] == 3 and result['total_count_variance'] == 0
    centered = table.astype(float)-table.mean(axis=0)
    expected=[np.mean(centered[:,:4-lag]*centered[:,lag:]) for lag in (0,1,2)]
    np.testing.assert_array_equal(result['seed_ensemble_lag_covariance'], expected)
    assert expected == [.5,0,-.5] and result['ensemble_ddof'] == 0


def test_observed_ensemble_preserves_silent_units_and_ddof():
    values=np.array([[[0,0],[0,1]],[[0,0],[2,3]],[[0,0],[4,5]]])
    result=ensemble_covariance(values,0,ddof=1)
    np.testing.assert_array_equal(result,[0,4])
    np.testing.assert_array_equal(ensemble_covariance(values,1,ddof=1),[0,4])


@pytest.mark.parametrize('events',[np.array([0.]),np.array([-1.]),np.array([3001.]),np.array([np.nan])])
def test_invalid_native_event_support_rejects(events):
    with pytest.raises(ValueError):temporal_counts(events)


def test_partial_primary_seed_support_rejects():
    with pytest.raises(ValueError,match='COMPLETE_PRIMARY_SEEDS'):
        temporal_moments(np.zeros((9999,4),dtype=np.int32),lags=(0,))
