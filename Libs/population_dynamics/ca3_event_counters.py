"""Exact streamed native/AER reconciliation, shared by retrieval consumers."""
from pathlib import Path

import numpy as np

from .ca3_native_observation import iter_intervals


def compare_event_counters(path: Path, events: np.ndarray, units: int, result: dict,
                           *, expected_stop_ms: int) -> None:
    """Compare every unit/ms count, including silence and repeated events.

    Buffer one native interval. Exhaust the trace to require its terminal footer.
    Record the first mismatch before rejecting the recording.
    """
    events = np.asarray(events)
    if (events.ndim != 2 or events.shape[1] != 2 or events.dtype.kind not in "iu"
            or type(units) is not int or units <= 0
            or type(expected_stop_ms) is not int or expected_stop_ms <= 0
            or np.any(events < 0)
            or np.any(events[:, 0] >= expected_stop_ms) or np.any(events[:, 1] >= units)):
        raise ValueError("CA3_OBSERVATION_EVENT_COUNTER_INPUT_INVALID")
    ordered = events[np.argsort(events[:, 0], kind="stable")]
    per_ms = np.bincount(ordered[:, 0], minlength=expected_stop_ms)
    offsets = np.concatenate(([0], np.cumsum(per_ms, dtype=np.int64)))
    tested = 0
    for interval in iter_intervals(path, expected_stop_ms=expected_stop_ms):
        tick = interval.start_ms
        expected = np.bincount(ordered[offsets[tick]:offsets[tick + 1], 1], minlength=units)
        if interval.native_counts.shape != (units,) or not np.array_equal(expected, interval.native_counts):
            if interval.native_counts.shape == (units,):
                differing = np.flatnonzero(expected != interval.native_counts)
                result["first_counter_mismatch"] = dict(start_ms=tick, stop_ms=tick + 1,
                    differing_units=int(len(differing)),
                    examples=[dict(global_unit_id=int(i), aer_count=int(expected[i]),
                                   native_count=int(interval.native_counts[i])) for i in differing[:8]])
            raise ValueError("CA3_OBSERVATION_PER_UNIT_MILLISECOND_COUNTER_MISMATCH")
        tested += units
    result["unit_millisecond_counts_compared"] = tested
    result["native_intervals_compared"] = expected_stop_ms
