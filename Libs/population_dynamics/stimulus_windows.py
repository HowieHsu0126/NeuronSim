"""Encode source stimulus exposure on exact absolute-time windows.

This module uses annotations only. It does not infer recording support or certify
that a stimulus is causally qualified for a scientific analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StimulusWindows:
    values: np.ndarray  # trial, time bin, feature; fractional bin occupancy
    features: list[dict[str, Any]]
    overlap_fraction: np.ndarray  # bin fraction with at least two source rows


def _column_ids(frame: pd.DataFrame, column: str, table: str) -> list[str | int]:
    result: list[str | int] = []
    for value in frame[column]:
        if isinstance(value, str) and value.strip():
            result.append(value)
        elif isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= 0:
            result.append(int(value))
        else:
            raise ValueError(f"INVALID_ID:{table}.{column}")
    return result


def _check_table(frame: pd.DataFrame, name: str, columns: set[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"DATAFRAME_REQUIRED:{name}")
    if not frame.columns.is_unique or not columns.issubset(frame.columns):
        raise ValueError(f"REQUIRED_UNIQUE_COLUMNS:{name}:{','.join(sorted(columns))}")


def _times(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column].to_numpy()
    if any(not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError(f"FINITE_NUMERIC_TIME_REQUIRED:{column}")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"FINITE_NUMERIC_TIME_REQUIRED:{column}")
    return values


def _coverage_integrals(starts: np.ndarray, stops: np.ndarray,
                        edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integrate union and multiplicity >= 2, retaining concurrent annotations."""
    if len(starts) == 0:
        zeros = np.zeros(edges.shape, dtype=np.float64)
        return zeros, zeros.copy()
    times, positions = np.unique(np.concatenate((starts, stops)), return_inverse=True)
    changes = np.bincount(positions, weights=np.concatenate((np.ones(len(starts)), -np.ones(len(stops)))))
    active = np.cumsum(changes)[:-1]
    durations = np.diff(times)
    union = np.concatenate(([0.0], np.cumsum(durations * (active >= 1))))
    overlap = np.concatenate(([0.0], np.cumsum(durations * (active >= 2))))
    return (np.interp(edges, times, union), np.interp(edges, times, overlap))


def encode_stimulus_windows(presentations: pd.DataFrame,
                            trial_members: pd.DataFrame,
                            trials: pd.DataFrame,
                            absolute_edges_seconds: np.ndarray) -> StimulusWindows:
    """Return annotation exposures for each supplied row of absolute bin edges.

    Every presentation must map uniquely through ``trial_members.trial_id`` to
    a source ``trials.condition_id``. Extra source trials are permitted; missing
    or conflicting presentation memberships are rejected. Optional shared
    identity fields are checked when supplied. Window rows need not correspond
    to rows of ``trials``: history can overlap annotations from preceding trials.

    Feature zero records the fraction without any annotation. Explicit
    spontaneous/blank stimuli retain their own condition feature. Movies use
    (condition_id, frame), other stimuli use condition_id. Vocabulary is sorted
    deterministically using only annotations intersecting the requested windows.
    Concurrent annotations each retain their full exposure, so feature sums can
    exceed one; ``overlap_fraction`` reports their time overlap separately.
    """
    _check_table(presentations, "presentations", {"presentation_id", "start_time", "stop_time", "stimulus_name"})
    _check_table(trial_members, "trial_members", {"presentation_id", "trial_id"})
    _check_table(trials, "trials", {"trial_id", "condition_id"})
    edges = np.asarray(absolute_edges_seconds)
    if (edges.ndim != 2 or edges.shape[1] < 2 or edges.dtype.kind not in "fiu"
            or not np.isfinite(edges).all()):
        raise ValueError("FINITE_TWO_DIMENSIONAL_BIN_EDGES_REQUIRED")
    edges = edges.astype(np.float64, copy=False)
    if not np.isfinite(edges).all() or np.any(np.diff(edges, axis=1) <= 0):
        raise ValueError("STRICTLY_INCREASING_BIN_EDGES_REQUIRED")

    presentation_ids = _column_ids(presentations, "presentation_id", "presentations")
    member_ids = _column_ids(trial_members, "presentation_id", "trial_members")
    member_trials = _column_ids(trial_members, "trial_id", "trial_members")
    trial_ids = _column_ids(trials, "trial_id", "trials")
    conditions = _column_ids(trials, "condition_id", "trials")
    if len(set(presentation_ids)) != len(presentation_ids):
        raise ValueError("DUPLICATE_PRESENTATION_ID")
    if len(set(member_ids)) != len(member_ids):
        raise ValueError("DUPLICATE_PRESENTATION_MEMBERSHIP")
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("DUPLICATE_TRIAL_ID")
    if set(member_ids) != set(presentation_ids):
        raise ValueError("PRESENTATION_MEMBERSHIP_MISMATCH")
    if not set(member_trials).issubset(trial_ids):
        raise ValueError("UNKNOWN_MEMBER_TRIAL_ID")
    member_lookup = dict(zip(member_ids, range(len(member_ids))))
    trial_lookup = dict(zip(trial_ids, range(len(trial_ids))))
    member_rows = [member_lookup[value] for value in presentation_ids]
    trial_rows = [trial_lookup[member_trials[row]] for row in member_rows]
    source_conditions = [conditions[row] for row in trial_rows]
    source_names = presentations["stimulus_name"].tolist()
    if any(not isinstance(name, str) or not name.strip() for name in source_names):
        raise ValueError("STIMULUS_NAME_REQUIRED")
    if "stimulus_name" in trials:
        names = trials["stimulus_name"].tolist()
        if any(name != names[row] for name, row in zip(source_names, trial_rows)):
            raise ValueError("STIMULUS_IDENTITY_MISMATCH")
    for field in ("session_id", "animal_id", "split"):
        tables = [(presentations, list(range(len(presentations))), "presentations"),
                  (trial_members, member_rows, "trial_members"),
                  (trials, trial_rows, "trials")]
        aligned = []
        for table, rows, name in tables:
            if field in table:
                ids = _column_ids(table, field, name)
                aligned.append([ids[row] for row in rows])
        if any(values != aligned[0] for values in aligned[1:]):
            raise ValueError(f"IDENTITY_MAPPING_MISMATCH:{field}")
    if "condition_id" in presentations:
        if _column_ids(presentations, "condition_id", "presentations") != source_conditions:
            raise ValueError("CONDITION_IDENTITY_MISMATCH")

    starts = _times(presentations, "start_time")
    stops = _times(presentations, "stop_time")
    if np.any(stops <= starts):
        raise ValueError("POSITIVE_PRESENTATION_DURATION_REQUIRED")
    names_by_condition: dict[str | int, str] = {}
    keys: list[tuple[str | int, int | None]] = []
    frames = presentations["frame"].tolist() if "frame" in presentations else [None] * len(presentations)
    for condition, name, frame in zip(source_conditions, source_names, frames):
        if condition in names_by_condition and names_by_condition[condition] != name:
            raise ValueError("CONDITION_STIMULUS_NAME_CONFLICT")
        names_by_condition[condition] = name
        if name.startswith("natural_movie"):
            if (not isinstance(frame, Real) or isinstance(frame, (bool, np.bool_))
                    or not np.isfinite(frame) or frame < 0 or frame != int(frame)):
                raise ValueError("NONNEGATIVE_INTEGER_MOVIE_FRAME_REQUIRED")
            keys.append((condition, int(frame)))
        else:
            keys.append((condition, None))

    rows_per_window = [np.flatnonzero((starts < row[-1]) & (stops > row[0])) for row in edges]
    selected_rows = sorted({int(row) for rows in rows_per_window for row in rows})
    selected_keys = {keys[row] for row in selected_rows}
    def sort_key(key: tuple[str | int, int | None]) -> tuple[Any, ...]:
        condition, frame = key
        return (0 if isinstance(condition, int) else 1, condition, -1 if frame is None else frame)
    vocabulary = sorted(selected_keys, key=sort_key)
    features: list[dict[str, Any]] = [{"feature_index": 0, "feature_kind": "unannotated_fraction",
                                     "stimulus_name": None, "condition_id": None, "frame": None}]
    for index, (condition, frame) in enumerate(vocabulary, start=1):
        features.append({"feature_index": index, "feature_kind": "condition" if frame is None else "movie_frame",
                         "stimulus_name": names_by_condition[condition], "condition_id": condition, "frame": frame})
    feature_lookup = {key: index for index, key in enumerate(vocabulary, start=1)}
    values = np.zeros((len(edges), edges.shape[1] - 1, len(features)), dtype=np.float64)
    widths = np.diff(edges, axis=1)
    union_integrals, overlap_integrals = _coverage_integrals(starts[selected_rows], stops[selected_rows], edges)
    values[:, :, 0] = 1.0 - np.diff(union_integrals, axis=1) / widths
    overlap_fraction = np.diff(overlap_integrals, axis=1) / widths
    bin_indices = np.arange(edges.shape[1] - 1)[None, :]
    for window_index, rows in enumerate(rows_per_window):
        if not len(rows):
            continue
        row_edges = edges[window_index]
        durations = np.maximum(0.0, np.minimum(stops[rows, None], row_edges[None, 1:])
                               - np.maximum(starts[rows, None], row_edges[None, :-1]))
        feature_indices = np.asarray([feature_lookup[keys[row]] for row in rows])[:, None]
        np.add.at(values[window_index], (bin_indices, feature_indices), durations / widths[window_index])
    return StimulusWindows(values=values, features=features, overlap_fraction=overlap_fraction)
