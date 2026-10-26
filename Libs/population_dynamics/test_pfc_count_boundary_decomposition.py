"""Exact half-open count identities; toy event lists are not scientific samples."""
import numpy as np
import pytest

from . import pfc_count_boundary_decomposition as boundary


def ev(rows):
    return np.asarray(rows, dtype=np.int64).reshape(-1, 2)


def run(a, b):
    return boundary.decompose(ev(a), ev(b), 3, 8000, [400, 4000, 8000],
                              np.array([0, 0, 1]), np.array([2, 1]))


def test_retiming_inside_window_changes_only_open_closed_disagreement_interval():
    summary, rows = run([(92, 0)], [(89, 0)])
    assert rows["segments"] == [dict(unit=0, start_exclusive_tick=89,
        stop_inclusive_tick=92, delta=1, unresolved_at_horizon=False)]
    assert summary["total_changed_unit_windows"] == 0
    assert summary["unresolved_excursions_at_horizon"] == 0
    assert rows["excursions"][0]["compensation_event_tick"] == 92


def test_event_at_boundary_is_excluded_from_previous_window():
    summary, rows = run([(400, 0)], [(399, 0)])
    changed = [x for x in rows["changed"] if x["width_ticks"] == 400]
    assert [(x["start_tick"], x["delta"], x["D_start"], x["D_stop"]) for x in changed] == [
        (0, 1, 0, 1), (400, -1, 1, 0)]
    assert summary["unresolved_excursions_at_horizon"] == 0


def test_final_tick_unresolved_imbalance_is_present_at_horizon():
    summary, rows = run([], [(7999, 0)])
    assert summary["unresolved_excursions_at_horizon"] == 1
    assert rows["segments"][0]["stop_inclusive_tick"] == 8000
    assert all(x["stop_tick"] == 8000 and x["delta"] == 1 for x in rows["changed"])
    with pytest.raises(ValueError, match="EVENT_SCHEMA"):
        run([], [(8000, 0)])


def test_magnitude_changes_split_segments_without_inventing_new_excursions():
    summary, rows = run([(300, 0), (600, 0)], [(0, 0), (200, 0)])
    assert [x["delta"] for x in rows["segments"]] == [1, 2, 1]
    assert summary["excursions"] == 1
    assert rows["excursions"][0]["maximum_absolute_imbalance"] == 2
    assert rows["excursions"][0]["compensation_event_tick"] == 600


def test_identical_simultaneous_events_cancel_without_segments():
    summary, rows = run([(0, 0), (400, 1)], [(400, 1), (0, 0)])
    assert not rows["segments"] and summary["total_changed_unit_windows"] == 0


def test_signed_field_cdf_cancellation_preserves_touched_observations():
    summary, rows = run([(10, 0)], [(10, 1)])
    field = next(x for x in rows["fields"] if x["width_ticks"] == 400 and x["field"] == 0)
    assert field["touched_observations"] == 2
    assert field["denominator"] == 40 and field["maximum_absolute_cdf_numerator"] == 0
    assert summary["unresolved_excursions_at_horizon"] == 2
    assert any(x["delta"] == -1 for x in rows["changed"])
    assert any(x["delta"] == 1 for x in rows["changed"])


def test_duplicate_events_are_rejected():
    with pytest.raises(ValueError, match="DUPLICATE_EVENT"):
        run([(1, 0), (1, 0)], [])


def test_raw_prefix_clipping_preserves_strict_stop(tmp_path):
    p = tmp_path / "events.npz"
    np.savez(p, spike_ticks=np.array([0, 7999, 8000]), spike_unit_indices=np.array([0, 1, 2]))
    config = dict(excitatory_neurons=2, inhibitory_neurons=1, duration_ms=5000., dt_ms=.0125)
    np.testing.assert_array_equal(boundary.read_events(p, config, 8000), ev([(0, 0), (7999, 1)]))
