import numpy as np
import pytest

from .v1_clock_binning_comparison import (
    BIN_MS, LAGS_MS, START_MS, STOP_MS, binned_descriptions,
)


def event_fixture(fine=False):
    return dict(unit_index=np.array([0, 0, 1, 0, 1, 0, 0]),
        time_ms=np.array([501., 501.125 if fine else 501.25, 505., 511.,
                          2990.875 if fine else 2990.75, 2991., 3000.]))


def test_bins_preserve_silent_units_endpoints_and_both_native_clocks():
    populations = np.array(["E", "E", "I"])
    a = binned_descriptions(event_fixture(), populations, .25)
    b = binned_descriptions(event_fixture(True), populations, .125)
    for bin_ms in BIN_MS:
        summary, arrays = a[bin_ms]
        assert summary["units"] == 3 and summary["events"] == 5
        assert summary["bins"] == (STOP_MS-START_MS)//bin_ms
        assert arrays["unit_totals"].tolist() == [3, 2, 0]
        assert arrays["population_names"].tolist() == ["E", "I"]
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], b[bin_ms][1][name])


def test_sparse_statistics_match_independent_dense_event_counts():
    events = event_fixture()
    result = binned_descriptions(events, np.array(["E", "E", "I"]), .25)
    for bin_ms in BIN_MS:
        counts = np.zeros((3, (STOP_MS-START_MS)//bin_ms), dtype=np.int64)
        for unit, time in zip(events["unit_index"], events["time_ms"]):
            if START_MS <= time < STOP_MS:
                counts[unit, int((time-START_MS)//bin_ms)] += 1
        arrays = result[bin_ms][1]
        pop = np.column_stack([counts[:2].sum(axis=0), counts[2]])
        np.testing.assert_array_equal(arrays["population_counts"], pop)
        for index, lag_ms in enumerate(LAGS_MS):
            lag = lag_ms//bin_ms
            overlap = counts.shape[1]-lag
            left, right = counts[:, :overlap], counts[:, lag:lag+overlap]
            centered_left = left-left.mean(axis=1, keepdims=True)
            centered_right = right-right.mean(axis=1, keepdims=True)
            expected = (centered_left*centered_right).sum(axis=1)/(overlap-1)
            np.testing.assert_allclose(arrays["unit_count_autocovariance"][index], expected,
                                       rtol=1e-12, atol=1e-15)
            x, y = pop[:overlap], pop[lag:lag+overlap]
            expected_pop = (x-x.mean(axis=0)).T @ (y-y.mean(axis=0))/(overlap-1)
            np.testing.assert_allclose(arrays["population_count_covariance"][index], expected_pop,
                                       rtol=1e-12, atol=1e-15)


def test_non_native_times_and_unsupported_clock_fail():
    events = event_fixture()
    events["time_ms"][0] = 501.1
    with pytest.raises(ValueError, match="NOT_ON_NATIVE_CLOCK"):
        binned_descriptions(events, np.array(["E", "E", "I"]), .25)
    with pytest.raises(ValueError, match="CLOCK_NOT_SUPPORTED"):
        binned_descriptions(event_fixture(), np.array(["E", "E", "I"]), .1)
