"""Dense direct covariance oracle for event-based complete-population descriptions."""
import numpy as np
import pytest

from .benchmark_activity_statistics import event_statistics


def events(counts):
    units,bins=np.nonzero(counts)
    repeats=counts[units,bins]
    return np.repeat(bins,repeats),np.repeat(units,repeats)


def test_sparse_covariance_matches_direct_all_unit_population_and_lag_matrices():
    counts=np.array([[2,0,1,3,0,1],[0,1,0,2,1,0],[1,0,0,0,0,1],[0,0,0,0,0,0]])
    ticks,units=events(counts)
    ticks=np.r_[ticks,6];units=np.r_[units,0]
    populations=np.array(["E","E","I","I"])
    summary,arrays=event_statistics(ticks,units,populations,duration_ticks=6,
        start_tick=0,stop_tick=6,bin_ticks=1,lag_bins=(0,1,3,5,9))
    assert summary["events"]==counts.sum() and summary["silent_units"]==1
    np.testing.assert_array_equal(arrays["unit_totals"],counts.sum(axis=1))
    population=np.stack([counts[:2].sum(axis=0),counts[2:].sum(axis=0)],axis=1)
    for index,lag in enumerate((0,1,3)):
        a,b=counts[:,:6-lag],counts[:,lag:]
        centered_a=a-a.mean(axis=1,keepdims=True)
        centered_b=b-b.mean(axis=1,keepdims=True)
        direct=centered_a@centered_b.T/(a.shape[1]-1)
        np.testing.assert_allclose(arrays["unit_count_autocovariance"][index],np.diag(direct),atol=1e-14)
        mask=~np.eye(4,dtype=bool)
        assert summary["lags"][index]["mean_distinct_unit_count_covariance"]==pytest.approx(direct[mask].mean())
        x,y=population[:6-lag],population[lag:]
        expected=(x-x.mean(axis=0)).T@(y-y.mean(axis=0))/(len(x)-1)
        np.testing.assert_allclose(arrays["population_count_covariance"][index],expected,atol=1e-14)
    assert all(row["temporal_count_covariance"] is None for row in summary["lags"][3:])
    assert np.isnan(arrays["unit_count_autocovariance"][3:]).all()


def test_exact_window_binning_keeps_silent_units_and_excludes_terminal_events():
    ticks=np.array([0,3,4,4,7,8,11,12])
    units=np.array([0,0,0,1,1,0,0,1])
    summary,arrays=event_statistics(ticks,units,np.array(["E","I","I"]),
        duration_ticks=12,start_tick=4,stop_tick=12,bin_ticks=4,lag_bins=(0,1))
    np.testing.assert_array_equal(arrays["population_counts"],[[1,2],[2,0]])
    assert summary["events"]==5 and summary["silent_units"]==1
    assert summary["ordered_distinct_unit_pairs"]==6


def test_negative_pair_covariance_is_retained():
    counts=np.array([[1,0,1,0],[0,1,0,1]])
    ticks,units=events(counts)
    summary,_=event_statistics(ticks,units,np.array(["E","E"]),duration_ticks=4,
        start_tick=0,stop_tick=4,bin_ticks=1,lag_bins=(0,))
    assert summary["lags"][0]["mean_distinct_unit_count_covariance"]==pytest.approx(-1/3)


@pytest.mark.parametrize("changes",[{"bin_ticks":0},{"bin_ticks":2.0},{"stop_tick":7},
                                  {"lag_bins":(0,0)},{"lag_bins":(-1,)},{"event_units":np.array([3])},
                                  {"ticks":np.array([7])},{"ticks":np.array([0.5])},
                                  {"ticks":np.array([-1])},{"event_units":np.array([-1])}])
def test_invalid_clock_identity_and_lag_are_rejected(changes):
    args=dict(ticks=np.array([0]),event_units=np.array([0]),populations=np.array(["E","I"]),
              duration_ticks=6,start_tick=0,stop_tick=6,bin_ticks=1,lag_bins=(0,1))
    args.update(changes)
    with pytest.raises(ValueError,match="EVENT_CLOCK_UNIT_OR_LAG_INVALID"):
        event_statistics(**args)


def test_population_indices_follow_source_inventory_instead_of_alphabetical_order():
    counts = np.array([[1, 0, 2, 0], [0, 1, 0, 2], [2, 0, 0, 0], [0, 0, 0, 0]])
    ticks, unit_indices = events(counts)
    summary, arrays = event_statistics(
        ticks, unit_indices, np.array(["Pyramidal", "Axo", "Pyramidal", "Axo"]),
        duration_ticks=4, start_tick=0, stop_tick=4, bin_ticks=1, lag_bins=(0, 1))
    assert arrays["population_names"].tolist() == ["Pyramidal", "Axo"]
    assert summary["population_names"] == ["Pyramidal", "Axo"]
    np.testing.assert_array_equal(arrays["population_index"], [0, 1, 0, 1])
    np.testing.assert_array_equal(arrays["population_counts"],
                                  np.stack([counts[[0, 2]].sum(axis=0), counts[[1, 3]].sum(axis=0)], axis=1))
    assert summary["silent_units"] == 1


def test_population_reindexing_preserves_all_unit_covariance():
    counts = np.array([[2, 0, 1, 1], [0, 2, 0, 1], [1, 1, 0, 0]])
    results = []
    for order in [np.arange(3), np.array([2, 0, 1])]:
        ticks, unit_indices = events(counts[order])
        summary, _ = event_statistics(
            ticks, unit_indices, np.array(["Z", "A", "Z"])[order],
            duration_ticks=4, start_tick=0, stop_tick=4, bin_ticks=1, lag_bins=(0, 1))
        results.append(summary)
    for left, right in zip(results[0]["lags"], results[1]["lags"]):
        assert left["temporal_count_covariance"] == pytest.approx(right["temporal_count_covariance"])
        assert left["mean_unit_count_autocovariance"] == pytest.approx(right["mean_unit_count_autocovariance"])
        assert left["mean_distinct_unit_count_covariance"] == pytest.approx(right["mean_distinct_unit_count_covariance"])
