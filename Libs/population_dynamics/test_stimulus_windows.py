"""Checks of annotation identity, bin exposure and concurrent presentation semantics."""
import inspect

import numpy as np
import pandas as pd
import pytest

from .stimulus_windows import encode_stimulus_windows


def tables(rows):
    """Rows: id, start, stop, name, condition, frame, trial."""
    presentations = pd.DataFrame([row[:4] + (row[5],) for row in rows],
                                columns=["presentation_id", "start_time", "stop_time", "stimulus_name", "frame"])
    members = pd.DataFrame([(row[0], row[6]) for row in rows], columns=["presentation_id", "trial_id"])
    trials = pd.DataFrame(list(dict.fromkeys((row[6], row[4], row[3]) for row in rows)),
                          columns=["trial_id", "condition_id", "stimulus_name"])
    return presentations, members, trials


def index(result, condition, frame=None):
    return next(feature["feature_index"] for feature in result.features
                if feature["condition_id"] == condition and feature["frame"] == frame)


def test_frame_splits_and_movie_identity():
    data = tables([
        ("p0", 0., .6, "natural_movie_one", "movie_one", 0, "t0"),
        ("p1", .6, 1., "natural_movie_one", "movie_one", 1, "t0"),
        ("p2", 2., 3., "natural_movie_two", "movie_two", 0, "t1"),
        ("p3", 8., 9., "natural_movie_one", "unused", 12, "t2"),
    ])
    result = encode_stimulus_windows(*data, np.array([[0., .5, 1.], [2., 2.5, 3.]]))
    assert result.values.dtype == np.float64
    assert result.values.shape == (2, 2, 4)
    np.testing.assert_allclose(result.values[0, :, index(result, "movie_one", 0)], [1., .2])
    np.testing.assert_allclose(result.values[0, :, index(result, "movie_one", 1)], [0., .8])
    np.testing.assert_array_equal(result.values[1, :, index(result, "movie_two", 0)], [1., 1.])
    assert not any(feature["condition_id"] == "unused" for feature in result.features)
    np.testing.assert_allclose(result.values[:, :, 0], 0, atol=1e-15)
    np.testing.assert_array_equal(result.overlap_fraction, 0)


def test_unknown_is_not_explicit_spontaneous_and_half_open_boundaries():
    data = tables([("p0", 1., 2., "spontaneous", "rest", np.nan, "t0"),
                   ("p1", 3., 4., "blank", "blank", np.nan, "t1")])
    result = encode_stimulus_windows(*data, np.array([[0., 1., 2., 3.]]))
    assert result.features[0]["feature_kind"] == "unannotated_fraction"
    assert result.features[0]["stimulus_name"] is None
    np.testing.assert_array_equal(result.values[0, :, 0], [1., 0., 1.])
    np.testing.assert_array_equal(result.values[0, :, index(result, "rest")], [0., 1., 0.])
    assert len(result.features) == 2  # an annotation starting at right edge is excluded


def test_concurrency_preserves_exposure_and_uses_union_for_unannotated():
    data = tables([("p0", .0, .75, "pulse", "a", None, "t0"),
                   ("p1", .25, .75, "pulse", "a", None, "t1"),
                   ("p2", .5, 1., "pulse", "b", None, "t2")])
    result = encode_stimulus_windows(*data, np.array([[0., 1., 2.]]))
    np.testing.assert_array_equal(result.values[0, :, 0], [0., 1.])
    np.testing.assert_array_equal(result.values[0, :, index(result, "a")], [1.25, 0.])
    np.testing.assert_array_equal(result.values[0, :, index(result, "b")], [.5, 0.])
    np.testing.assert_array_equal(result.overlap_fraction, [[.5, 0.]])
    assert result.values[0, 0].sum() == 1.75


def test_vocabulary_and_values_are_independent_of_table_order_and_activity_columns():
    data = tables([("p0", 0., 1., "pulse", "z", None, "t0"),
                   ("p1", 1., 2., "pulse", "a", None, "t1")])
    first = encode_stimulus_windows(*data, np.array([[0., 1., 2.]]))
    altered = [table.iloc[::-1].copy() for table in data]
    altered[0]["response"] = [np.nan, 1e50]
    second = encode_stimulus_windows(*altered, np.array([[0., 1., 2.]]))
    assert first.features == second.features
    assert first.features[1]["condition_id"] == "a"
    np.testing.assert_array_equal(first.values, second.values)
    assert list(inspect.signature(encode_stimulus_windows).parameters) == [
        "presentations", "trial_members", "trials", "absolute_edges_seconds"]


def test_empty_annotations_mean_unannotated():
    result = encode_stimulus_windows(*tables([]), np.array([[2., 3., 4.]]))
    np.testing.assert_array_equal(result.values, np.ones((1, 2, 1)))
    np.testing.assert_array_equal(result.overlap_fraction, np.zeros((1, 2)))


@pytest.mark.parametrize("edges", [np.array([0., 1.]), np.array([[0.]]), np.array([[0., 0.]]),
                                   np.array([[1., 0.]]), np.array([[0., np.nan]]),
                                   np.array([[0., np.inf]]), np.array([[False, True]]),
                                   np.array([["0", "1"]])])
def test_invalid_edges_rejected(edges):
    with pytest.raises(ValueError):
        encode_stimulus_windows(*tables([]), edges)


@pytest.mark.parametrize("frame", [-1, .25, np.nan, np.inf, True, "1", None])
def test_invalid_movie_frames_rejected(frame):
    data = tables([("p0", 0., 1., "natural_movie_one", "a", frame, "t0")])
    with pytest.raises(ValueError, match="MOVIE_FRAME"):
        encode_stimulus_windows(*data, np.array([[0., 1.]]))


@pytest.mark.parametrize("start,stop", [(1., 1.), (2., 1.), (np.nan, 1.), (0., np.inf), ("0", 1.), (True, 2.)])
def test_invalid_presentation_time_rejected(start, stop):
    data = tables([("p0", start, stop, "pulse", "a", None, "t0")])
    with pytest.raises(ValueError):
        encode_stimulus_windows(*data, np.array([[0., 1.]]))


@pytest.mark.parametrize("mutation", ["duplicate_presentation", "duplicate_member", "missing_member",
                                      "unknown_member", "unknown_trial", "duplicate_trial",
                                      "invalid_id", "condition_name", "trial_name", "session"])
def test_identity_mapping_rejects_ambiguity(mutation):
    p, m, t = tables([("p0", 0., 1., "pulse", "a", None, "t0"),
                      ("p1", 1., 2., "other", "b", None, "t1")])
    if mutation == "duplicate_presentation":
        p = pd.concat([p, p.iloc[:1]])
    elif mutation == "duplicate_member":
        m = pd.concat([m, m.iloc[:1]])
    elif mutation == "missing_member":
        m = m.iloc[:1]
    elif mutation == "unknown_member":
        m.loc[0, "presentation_id"] = "unknown"
    elif mutation == "unknown_trial":
        m.loc[0, "trial_id"] = "unknown"
    elif mutation == "duplicate_trial":
        t = pd.concat([t, t.iloc[:1]])
    elif mutation == "invalid_id":
        p.loc[0, "presentation_id"] = " "
    elif mutation == "condition_name":
        t.loc[1, "condition_id"] = "a"
    elif mutation == "trial_name":
        t.loc[0, "stimulus_name"] = "other"
    elif mutation == "session":
        m["session_id"] = [1, 1]
        t["session_id"] = [2, 1]
    with pytest.raises(ValueError):
        encode_stimulus_windows(p, m, t, np.array([[0., 1.]]))
