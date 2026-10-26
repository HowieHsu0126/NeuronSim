"""Boundary, multiplicity, mapping, and frozen-score regressions."""
import json

import numpy as np
import pytest

from . import pfc_response_event_attribution as attribution
from .provenance import sha256_file


def ev(ticks, unit=0):
    return np.array([(t, unit) for t in ticks], dtype=np.int64).reshape(-1, 2)


def fixture_streams(unit=0):
    return [ev([100, 600, 1000, 1100], unit), ev([400, 600, 800], unit),
            ev([400, 600, 800], unit), ev([100, 600, 1000], unit)]


def test_four_stream_multiplicity_and_compensation():
    streams = fixture_streams()
    jumps, info = attribution.signed_jumps(streams, 3, 1200)
    assert jumps == {0: {100: 2, 400: -2, 800: -2, 1000: 2, 1100: 1}}
    assert info["maximum_jump_magnitude"] == 2
    assert info["cancelled_coordinates"] == 1
    summary, arrays, contributors, bounds, segments, excursions = attribution.attribute(streams, 3, 400, 800, 1200)
    assert [tuple(r[k] for k in attribution.SEGMENT_FIELDS) for r in segments] == [
        (0, 100, 400, 2, False), (0, 800, 1000, -2, False), (0, 1100, 1200, 1, True)]
    np.testing.assert_array_equal(arrays["component_window_counts"][:, 0], [1, 2, 2, 1])
    np.testing.assert_array_equal(arrays["cumulative_signed_counts"][:, 0], [2, 0, 1])
    assert summary["window"]["signed_sum"] == -2
    assert summary["unresolved_excursions_at_horizon"] == 1
    assert [r["boundary_tick"] for r in bounds] == [400]


@pytest.mark.parametrize("start,stop,expected", [(0,400,2), (400,800,-2), (800,1200,1)])
def test_half_open_complete_windows(start, stop, expected):
    result = attribution.attribute(fixture_streams(80000), 100000, start, stop, 1200)
    assert result[0]["window"]["signed_sum"] == expected
    assert result[1]["unit_signed_counts"].shape == (100000,)
    assert result[1]["unit_signed_counts"][80000] == expected
    assert np.count_nonzero(result[1]["unit_signed_counts"]) == 1


def test_zero_score_retains_control_cancellation_and_all_units():
    streams = [ev([10]), ev([20]), ev([]), ev([])]
    summary, arrays, contributors, _, _, _ = attribution.attribute(streams, 3, 0, 100, 100)
    assert summary["window"]["signed_sum"] == 0
    assert summary["cancelled_control_change_units"] == 1
    assert len(contributors) == 1 and contributors[0]["cue_clock_count"] == contributors[0]["omission_clock_count"] == 1
    assert arrays["component_window_counts"].shape == (4,3)
    assert arrays["component_window_counts"].sum() == 2


def test_empty_events_and_exact_time_cancellation():
    for streams in ([ev([]) for _ in range(4)], [ev([0,10]) for _ in range(4)]):
        summary, arrays, _, _, segments, excursions = attribution.attribute(streams, 3, 0, 100, 100)
        assert summary["window"]["signed_sum"] == 0
        assert not segments and not excursions
        assert not np.any(arrays["unit_signed_counts"])


def test_positive_negative_unit_cancellation_is_not_omitted():
    streams = [ev([10],0), ev([10],1), ev([]), ev([])]
    summary, arrays, contributors, *_ = attribution.attribute(streams, 3, 0, 100, 100)
    assert summary["window"] == dict(signed_sum=0, positive_sum=1, negative_sum=-1, absolute_sum=2,
        positive_units=1, negative_units=1, nonzero_units=2, max_abs=1)
    assert len(contributors) == 2


def test_duplicate_event_and_horizon_event_rejected():
    for bad in (ev([0,0]), ev([100])):
        with pytest.raises(ValueError):
            attribution.attribute([bad, ev([]), ev([]), ev([])], 3, 0, 100, 100)


@pytest.mark.parametrize("clock,factor", [(.05,4),(.025,2),(.0125,1)])
def test_common_physical_clock_keeps_boundary_events(tmp_path, clock, factor):
    path = tmp_path / "raw.npz"
    np.savez(path, spike_ticks=np.array([0,1,2]), spike_unit_indices=np.array([0,1,1]))
    actual = attribution.read_events(path, clock, .0125, 3, 1.)
    np.testing.assert_array_equal(actual[:,0], [0,factor,2*factor])
    assert attribution.count_window(actual,3,factor,2*factor).sum() == 1
    with pytest.raises(ValueError, match="NONINTEGER_PHYSICAL_CLOCK"):
        attribution.read_events(path, .03, .0125, 3, 1.)


@pytest.mark.parametrize("count,score", [(-30,-.00030000000000002247), (26,.000260000000000038),
                                       (9,.00008999999999997899), (-6,-.000060000000000004494)])
def test_original_and_repeat_scores_require_the_exact_integer(count, score):
    assert attribution.score_identity(count,100000.,score,1e-12) < 1e-12
    with pytest.raises(ValueError, match="SAVED_SCORE_RECONSTRUCTION"):
        attribution.score_identity(count+1,100000.,score,1e-12)


def test_historical_coarse_delivery_requires_its_own_completion_fields(tmp_path):
    folder=tmp_path/"coarse"
    folder.mkdir()
    path=folder/"delivery.json"
    payload=dict(status="ARCHIVED_PFC_SOURCE_EXECUTION", result_status="PFC_SOURCE_EXECUTION_COMPLETE",
                 source_execution_complete=True, exit_code=0, files=[])
    path.write_text(json.dumps(payload))
    spec=dict(path="coarse", status=payload["status"], delivery_sha256=sha256_file(path))
    assert attribution.archive_metadata(tmp_path, dict(archives=dict(coarse_cue=spec))) == {"coarse_cue": {}}
    payload["source_execution_complete"]=False
    path.write_text(json.dumps(payload))
    spec["delivery_sha256"]=sha256_file(path)
    with pytest.raises(ValueError, match="COARSE_DELIVERY_COMPLETION"):
        attribution.archive_metadata(tmp_path, dict(archives=dict(coarse_cue=spec)))
