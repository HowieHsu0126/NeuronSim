"""Supplemental source-indexed population readout O(Y, M).

The same fixed mapping applies to source and generated unit trajectories. Labels
identify source groups; they do not establish spatial coordinates or recording
coverage. Missing units remain in denominators and never become silent units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

INT64_MAX = np.iinfo(np.int64).max


def _readonly(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    result.flags.writeable = False
    return result


def _unit_ids(values) -> np.ndarray:
    ids = np.asarray(values)
    if (ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu")
            or any(isinstance(value, (bool, np.bool_)) for value in values)):
        raise ValueError("UNIQUE_INTEGER_UNIT_IDS_REQUIRED")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("DUPLICATE_UNIT_ID")
    return ids if ids.size else np.empty(0, dtype=np.int64)


def _exact_sum(values: np.ndarray) -> np.ndarray:
    # Ordinary counts take the vectorized path. Large values need an exact
    # overflow check before conversion, including sparse but representable sums.
    if np.any(np.max(values, axis=-1) > INT64_MAX // values.shape[-1]):
        subtotal = values.astype(object).sum(axis=-1)
        if np.any(subtotal > INT64_MAX):
            raise ValueError("FIELD_COUNT_SUM_OVERFLOW")
        return np.asarray(subtotal, dtype=np.int64)
    return values.sum(axis=-1, dtype=np.int64)


@dataclass(frozen=True)
class FieldReadout:
    count_sum: np.ndarray
    observed_unit_count: np.ndarray
    total_unit_count: np.ndarray
    observed_rate_hz: np.ndarray
    complete: np.ndarray
    field_ids: np.ndarray
    supplied_unit_ids: np.ndarray


@dataclass(frozen=True)
class PopulationFieldMapping:
    unit_ids: np.ndarray
    field_ids: np.ndarray
    group_index: np.ndarray
    definition: str

    @classmethod
    def from_groups(cls, unit_ids, group_labels, *, definition: str,
                    field_ids=None) -> "PopulationFieldMapping":
        """Fix source membership; an explicit ordered vocabulary retains empty fields."""
        ids = _unit_ids(unit_ids)
        if not len(ids):
            raise ValueError("SOURCE_UNIT_MEMBERSHIP_REQUIRED")
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError("SOURCE_GROUP_DEFINITION_REQUIRED")
        if isinstance(group_labels, (str, bytes)):
            raise ValueError("ONE_SOURCE_STRING_LABEL_PER_UNIT_REQUIRED")
        labels = list(group_labels)
        if (len(labels) != len(ids)
                or any(not isinstance(label, str) or not label.strip() for label in labels)):
            raise ValueError("ONE_SOURCE_STRING_LABEL_PER_UNIT_REQUIRED")
        if field_ids is None:
            names = sorted(set(labels))
        else:
            if isinstance(field_ids, (str, bytes)):
                raise ValueError("EXPLICIT_FIELD_VOCABULARY_REQUIRED")
            names = list(field_ids)
            if (not names or any(not isinstance(name, str) or not name.strip() for name in names)
                    or len(set(names)) != len(names)):
                raise ValueError("UNIQUE_NONEMPTY_FIELD_IDS_REQUIRED")
            if set(labels) - set(names):
                raise ValueError("FIELD_VOCABULARY_MISSING_SOURCE_GROUP")
        fields = np.asarray(names)
        lookup = {label: index for index, label in enumerate(fields)}
        groups = np.asarray([lookup[label] for label in labels], dtype=np.int64)
        return cls(_readonly(ids), _readonly(fields), _readonly(groups), definition)

    def project(self, chunks: Iterable[tuple], *, bin_seconds: float) -> FieldReadout:
        """Sum observed counts, with mean rate per observed unit in each field.

        Chunks contain (unit_ids, counts[..., units], observed_mask). A None mask
        declares every supplied entry observed. An empty stream has no shape;
        use an explicit (..., 0) chunk to represent wholly absent observations.
        """
        if (isinstance(bin_seconds, (bool, np.bool_))
                or not isinstance(bin_seconds, (int, float, np.integer, np.floating))
                or not np.isfinite(bin_seconds) or bin_seconds <= 0):
            raise ValueError("FINITE_POSITIVE_BIN_SECONDS_REQUIRED")
        bin_seconds = float(bin_seconds)
        if not np.isfinite(bin_seconds) or bin_seconds <= 0:
            raise ValueError("FINITE_POSITIVE_BIN_SECONDS_REQUIRED")
        source_index = {int(unit): index for index, unit in enumerate(self.unit_ids)}
        seen = np.zeros(len(self.unit_ids), dtype=bool)
        total = np.bincount(self.group_index, minlength=len(self.field_ids))
        prefix = None
        count_sum = observed_count = None
        for supplied, counts, observed_mask in chunks:
            ids = _unit_ids(supplied)
            try:
                positions = np.asarray([source_index[int(unit)] for unit in ids], dtype=np.int64)
            except KeyError as exc:
                raise ValueError(f"UNKNOWN_SOURCE_UNIT_ID:{exc.args[0]}") from exc
            if np.any(seen[positions]):
                raise ValueError("UNIT_SUPPLIED_IN_MULTIPLE_CHUNKS")
            values = np.asarray(counts)
            if values.ndim < 1 or values.shape[-1] != len(ids) or values.dtype.kind not in "iuf":
                raise ValueError("NUMERIC_COUNTS_WITH_UNIT_AXIS_REQUIRED")
            if prefix is None:
                prefix = values.shape[:-1]
                count_sum = np.zeros((*prefix, len(self.field_ids)), dtype=np.int64)
                observed_count = np.zeros_like(count_sum)
            elif values.shape[:-1] != prefix:
                raise ValueError("CHUNK_PREFIX_SHAPE_MISMATCH")
            mask = np.ones(values.shape, dtype=bool) if observed_mask is None else np.asarray(observed_mask)
            if mask.dtype.kind != "b" or mask.shape != values.shape:
                raise ValueError("BOOLEAN_OBSERVATION_MASK_SHAPE_REQUIRED")
            available = values[mask]
            if (not np.isfinite(available).all() or np.any(available < 0)
                    or (available.dtype.kind == "f" and np.any(available != np.floor(available)))):
                raise ValueError("OBSERVED_COUNTS_MUST_BE_FINITE_NONNEGATIVE_INTEGERS")
            if (np.any(available >= 2**63) if available.dtype.kind == "f"
                    else np.any(available > INT64_MAX)):
                raise ValueError("OBSERVED_COUNT_EXCEEDS_INT64")
            integers = np.where(mask, values, 0).astype(np.int64)
            groups = self.group_index[positions]
            order = np.argsort(groups, kind="stable")
            boundaries = np.r_[0, np.flatnonzero(np.diff(groups[order])) + 1, len(order)]
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                if start == stop:
                    continue
                members = order[start:stop]
                group = groups[members[0]]
                subtotal = _exact_sum(integers[..., members])
                if np.any(count_sum[..., group] > INT64_MAX - subtotal):
                    raise ValueError("FIELD_COUNT_SUM_OVERFLOW")
                count_sum[..., group] += subtotal
                observed_count[..., group] += mask[..., members].sum(axis=-1, dtype=np.int64)
            seen[positions] = True
        if prefix is None:
            raise ValueError("CHUNK_REQUIRED_TO_ESTABLISH_OBSERVATION_SHAPE")
        rate = np.full(count_sum.shape, np.nan, dtype=np.float64)
        available = observed_count > 0
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            np.divide(count_sum, observed_count, out=rate, where=available)
            rate[available] /= bin_seconds
        if not np.isfinite(rate[available]).all():
            raise ValueError("OBSERVED_RATE_NOT_FINITE")
        return FieldReadout(count_sum, observed_count, total, rate, observed_count == total,
                            self.field_ids, _readonly(self.unit_ids[seen]))
