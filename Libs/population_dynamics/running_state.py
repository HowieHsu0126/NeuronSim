"""Whole-interval alignment of exported running speed before stimulus onset.

Source values are already in cm/s. No interpolation, zero fill, pupil processing
or brain-state thresholding is performed. Interval timestamp causality does not
establish causality of any upstream filtering used to produce the speed values.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class RunningIntervals:
    start_seconds: np.ndarray
    stop_seconds: np.ndarray
    speed_cm_s: np.ndarray
    unit: str


def _finite_vector(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
        raise ValueError('FINITE_NUMERIC_INTERVAL_VECTOR_REQUIRED:' + name)
    return array.astype(np.float64, copy=False)


def validate_intervals(intervals: RunningIntervals):
    if intervals.unit != 'cm/s':
        raise ValueError('RUNNING_SPEED_PHYSICAL_UNIT_MUST_BE_CM_PER_SECOND')
    start = _finite_vector(intervals.start_seconds, 'start_seconds')
    stop = _finite_vector(intervals.stop_seconds, 'stop_seconds')
    speed = _finite_vector(intervals.speed_cm_s, 'speed_cm_s')
    if start.shape != stop.shape or start.shape != speed.shape:
        raise ValueError('RUNNING_INTERVAL_SHAPES_MUST_MATCH')
    if np.any(stop <= start):
        raise ValueError('STRICTLY_POSITIVE_RUNNING_INTERVAL_DURATION_REQUIRED')
    if np.any(np.diff(start) < 0):
        raise ValueError('RUNNING_INTERVAL_SOURCE_ORDER_MUST_BE_ASCENDING')
    if np.any(start[1:] < stop[:-1]):
        raise ValueError('OVERLAPPING_RUNNING_INTERVALS_REJECTED')
    return start, stop, speed


def read_exported_running_csv(path: Path, source_metadata: dict[str, Any]) -> RunningIntervals:
    """Read the intake export without reapplying its source conversion/offset."""
    if source_metadata.get('unit') != 'cm/s':
        raise ValueError('RUNNING_SPEED_PHYSICAL_UNIT_MUST_BE_CM_PER_SECOND')
    starts, stops, speeds = [], [], []
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        required = {'start_time', 'stop_time', 'value', 'value_is_finite'}
        if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not required.issubset(reader.fieldnames)):
            raise ValueError('EXPORTED_RUNNING_INTERVAL_COLUMNS_REQUIRED')
        for index, row in enumerate(reader):
            if row['value_is_finite'] not in ('True', 'False'):
                raise ValueError('EXPORTED_RUNNING_FINITE_FLAG_INVALID:' + str(index))
            try:
                start, stop, speed = (float(row[name]) for name in ('start_time', 'stop_time', 'value'))
            except (TypeError, ValueError) as error:
                raise ValueError('EXPORTED_RUNNING_NUMERIC_VALUE_INVALID:' + str(index)) from error
            if row['value_is_finite'] != 'True' or not all(math.isfinite(v) for v in (start, stop, speed)):
                raise ValueError('NONFINITE_EXPORTED_RUNNING_INTERVAL_REJECTED:' + str(index))
            starts.append(start)
            stops.append(stop)
            speeds.append(speed)
    result = RunningIntervals(np.asarray(starts, dtype=float), np.asarray(stops, dtype=float),
                              np.asarray(speeds, dtype=float), source_metadata['unit'])
    validate_intervals(result)
    return result


def align_running_state(intervals: RunningIntervals, onsets_seconds: Sequence[float],
                        history_seconds: float, *, valid_intervals=None) -> list[dict[str, Any]]:
    """Summarize only valid intervals fully inside [onset-history, onset).

    An interval ending exactly at onset is admissible: its half-open support
    contains no future time. A boundary-crossing interval is wholly excluded.
    Optional validity flags represent explicit source quality exclusions; all
    interval timestamps and numeric values must still be finite and valid.
    Means divide by covered duration, never by the full window with zero fill.
    """
    start, stop, speed = validate_intervals(intervals)
    onsets = _finite_vector(onsets_seconds, 'onsets_seconds')
    if isinstance(history_seconds, (bool, np.bool_)) or not isinstance(history_seconds, (int, float, np.number)):
        raise ValueError('POSITIVE_FINITE_RUNNING_HISTORY_REQUIRED')
    history = float(history_seconds)
    if not math.isfinite(history) or history <= 0:
        raise ValueError('POSITIVE_FINITE_RUNNING_HISTORY_REQUIRED')
    if valid_intervals is None:
        valid = np.ones(start.shape, dtype=bool)
    else:
        valid = np.asarray(valid_intervals)
        if valid.dtype.kind != 'b' or valid.shape != start.shape:
            raise ValueError('EXPLICIT_BOOLEAN_INTERVAL_VALIDITY_REQUIRED')
    results = []
    for onset in onsets:
        onset = float(onset)
        left = onset - history
        if not math.isfinite(left) or left >= onset:
            raise ValueError('RUNNING_HISTORY_NOT_RESOLVED_ON_SOURCE_CLOCK')
        begin = int(np.searchsorted(stop, left, side='right'))
        end = int(np.searchsorted(start, onset, side='left'))
        cursor, covered, signed, absolute = left, [], [], []
        included, exclusions, gaps = [], [], []
        for index in range(begin, end):
            a, b = max(left, float(start[index])), min(onset, float(stop[index]))
            if a > cursor:
                gaps.append(dict(start_seconds=cursor, stop_seconds=a))
            reasons = []
            if start[index] < left:
                reasons.append('crosses_history_start')
            if stop[index] > onset:
                reasons.append('ends_after_onset')
            if not valid[index]:
                reasons.append('source_marked_invalid')
            if reasons:
                exclusions.append(dict(interval_index=index, start_seconds=float(start[index]),
                    stop_seconds=float(stop[index]), overlap_start_seconds=a, overlap_stop_seconds=b,
                    reasons=reasons))
            else:
                duration = float(stop[index] - start[index])
                included.append(index)
                covered.append(duration)
                signed.append(float(speed[index]) * duration)
                absolute.append(abs(float(speed[index])) * duration)
            cursor = b
        if cursor < onset:
            gaps.append(dict(start_seconds=cursor, stop_seconds=onset))
        duration = math.fsum(covered)
        signed_mean = math.fsum(signed) / duration if duration else None
        absolute_mean = math.fsum(absolute) / duration if duration else None
        if duration and not (math.isfinite(signed_mean) and math.isfinite(absolute_mean)):
            raise ValueError('RUNNING_WEIGHTED_MEAN_NONFINITE')
        excluded_seconds = math.fsum(item['overlap_stop_seconds'] - item['overlap_start_seconds'] for item in exclusions)
        gap_seconds = math.fsum(item['stop_seconds'] - item['start_seconds'] for item in gaps)
        results.append(dict(onset_seconds=onset, window_start_seconds=left, window_stop_seconds=onset,
            requested_history_seconds=history, window_duration_seconds=onset-left,
            covered_seconds=duration, coverage_fraction=duration/(onset-left),
            signed_mean_speed_cm_s=signed_mean, absolute_mean_speed_cm_s=absolute_mean,
            included_interval_indices=included, included_interval_count=len(included),
            missing_source_seconds=gap_seconds, excluded_interval_seconds=excluded_seconds,
            uncovered_seconds=gap_seconds+excluded_seconds, missing_source_intervals=gaps,
            excluded_intervals=exclusions, timestamp_causality_verified=True,
            inclusion_rule='whole intervals: start >= onset-history and stop <= onset',
            averaging_denominator='covered valid interval duration only'))
    return results
