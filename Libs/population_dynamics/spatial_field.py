"""Source-defined periodic square top-hat cells over fixed neural populations.

Cell support is disjoint and half-open, with the positive outer edge identified
with the negative edge. This is a count-density observation, not interpolation,
an LFP forward model, or a claim that source recording coverage is qualified.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import numpy as np

from .population_field import FieldReadout, PopulationFieldMapping, _readonly, _unit_ids


@dataclass(frozen=True)
class SpatialFieldReadout:
    field: FieldReadout
    density_hz_per_mm2: np.ndarray


@dataclass(frozen=True)
class PeriodicPopulationGrid:
    mapping: PopulationFieldMapping
    population_labels: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    centers_mm: np.ndarray
    cell_area_mm2: float
    extent_mm: float
    cells_per_axis: int

    @classmethod
    def from_positions(cls, unit_ids, populations, positions_mm, *, extent_mm: float,
                       cells_per_axis: int, definition: str) -> "PeriodicPopulationGrid":
        """Create all population × y × x cells from source positions in mm.

        Only exact +extent/2 boundary values wrap; out-of-domain input is rejected.
        The resulting fixed map is shared by source and generated trajectories.
        """
        ids = _unit_ids(unit_ids)
        if (isinstance(extent_mm, (bool, np.bool_))
                or not isinstance(extent_mm, (int, float, np.integer, np.floating))
                or not np.isfinite(extent_mm) or extent_mm <= 0):
            raise ValueError("FINITE_POSITIVE_EXTENT_MM_REQUIRED")
        if type(cells_per_axis) is not int or cells_per_axis < 1:
            raise ValueError("POSITIVE_INTEGER_CELLS_PER_AXIS_REQUIRED")
        extent = float(extent_mm)
        half, width = extent / 2.0, extent / cells_per_axis
        area = width * width
        if not np.isfinite(extent) or not np.isfinite(area) or area <= 0 or half <= 0:
            raise ValueError("SPATIAL_CELL_AREA_NOT_REPRESENTABLE")
        if isinstance(populations, (str, bytes)):
            raise ValueError("ONE_SOURCE_POPULATION_LABEL_PER_UNIT_REQUIRED")
        labels = list(populations)
        if (len(labels) != len(ids) or not labels
                or any(not isinstance(label, str) or not label.strip() for label in labels)):
            raise ValueError("ONE_SOURCE_POPULATION_LABEL_PER_UNIT_REQUIRED")
        positions = np.asarray(positions_mm)
        if (positions.shape != (len(ids), 2) or positions.dtype.kind not in "iuf"
                or not np.isfinite(positions).all()):
            raise ValueError("FINITE_SOURCE_POSITIONS_N_BY_2_REQUIRED")
        if np.any((positions < -half) | (positions > half)):
            raise ValueError("SOURCE_POSITION_OUTSIDE_DECLARED_SQUARE")
        edges = np.linspace(-half, half, cells_per_axis + 1)
        if np.any(np.diff(edges) <= 0):
            raise ValueError("SPATIAL_CELL_EDGES_NOT_RESOLVABLE")
        wrapped = np.where(positions == half, -half, positions)
        indices = np.searchsorted(edges, wrapped, side="right") - 1
        ordered_populations = sorted(set(labels))
        population_index = {label: i for i, label in enumerate(ordered_populations)}
        n = cells_per_axis
        unit_fields = np.asarray([population_index[label] for label in labels]) * n * n + indices[:, 1] * n + indices[:, 0]
        field_populations = np.repeat(np.asarray(ordered_populations), n * n)
        ix = np.tile(np.arange(n, dtype=np.int64), len(ordered_populations) * n)
        iy = np.tile(np.repeat(np.arange(n, dtype=np.int64), n), len(ordered_populations))
        field_ids = np.asarray([json.dumps([str(label), int(y), int(x)], ensure_ascii=False,
                                           separators=(",", ":"))
                                for label, y, x in zip(field_populations, iy, ix)])
        mapping = PopulationFieldMapping.from_groups(ids, field_ids[unit_fields],
                                                     definition=definition, field_ids=field_ids)
        centers = (edges[:-1] + edges[1:]) / 2.0
        centers_mm = np.column_stack((centers[ix], centers[iy]))
        return cls(mapping, _readonly(field_populations), _readonly(ix), _readonly(iy),
                   _readonly(centers_mm), area, extent, n)

    def project(self, chunks: Iterable[tuple], *, bin_seconds: float) -> SpatialFieldReadout:
        """Return exact grouped counts and complete-input spike-rate density.

        A known empty cell has zero density and undefined per-unit rate. A cell
        with missing source units has undefined density, even if observed units
        are silent. Completeness describes supplied input, not recording coverage.
        """
        field = self.mapping.project(chunks, bin_seconds=bin_seconds)
        density = np.full(field.count_sum.shape, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            np.divide(field.count_sum, float(bin_seconds), out=density, where=field.complete)
            density[field.complete] /= self.cell_area_mm2
        if not np.isfinite(density[field.complete]).all():
            raise ValueError("COMPLETE_CELL_DENSITY_NOT_FINITE")
        return SpatialFieldReadout(field, density)

    def project_events(self, spike_unit_ids, spike_ticks, *, observed_unit_ids,
                       bin_edges_ticks, dt_seconds: float) -> SpatialFieldReadout:
        """Project events with a fixed, explicitly declared observation set.

        Uses O(events + units + bins*fields) memory, without a bins*units array.
        All bins share the declared observed units. Events outside [first,last)
        are excluded, including the final right boundary. Recording completeness
        must be established by the caller; absence of events alone is insufficient.
        """
        supplied = _unit_ids(observed_unit_ids)
        senders, times, edges = map(np.asarray, (spike_unit_ids, spike_ticks, bin_edges_ticks))
        if (senders.ndim != 1 or senders.dtype.kind not in "iu" or times.shape != senders.shape
                or times.dtype.kind not in "iu" or np.any(times < 0)):
            raise ValueError("INTEGER_EVENT_IDENTITIES_AND_NONNEGATIVE_TICKS_REQUIRED")
        if (edges.ndim != 1 or len(edges) < 2 or edges.dtype.kind not in "iu"
                or np.any(edges < 0) or np.any(edges[1:] <= edges[:-1])
                or not np.all(np.diff(edges) == edges[1]-edges[0])):
            raise ValueError("UNIFORM_INCREASING_NONNEGATIVE_INTEGER_BIN_EDGES_REQUIRED")
        if (isinstance(dt_seconds, (bool, np.bool_)) or not np.isfinite(dt_seconds) or dt_seconds <= 0):
            raise ValueError("FINITE_POSITIVE_TICK_DURATION_REQUIRED")
        ordered = np.argsort(self.mapping.unit_ids)
        source_ids = self.mapping.unit_ids[ordered]
        def positions(ids):
            where = np.searchsorted(source_ids, ids)
            if np.any(where >= len(source_ids)) or not np.array_equal(source_ids[where], ids):
                raise ValueError("UNKNOWN_SOURCE_UNIT_ID")
            return ordered[where]
        observed_positions, event_positions = positions(supplied), positions(senders)
        observed = np.zeros(len(source_ids), dtype=bool)
        observed[observed_positions] = True
        if not observed[event_positions].all():
            raise ValueError("EVENT_FROM_UNOBSERVED_UNIT")
        n_fields = len(self.mapping.field_ids)
        total = np.bincount(self.mapping.group_index, minlength=n_fields)
        observed_count = np.bincount(self.mapping.group_index[observed_positions], minlength=n_fields)
        counts = np.zeros((len(edges)-1, n_fields), dtype=np.int64)
        inside = (times >= edges[0]) & (times < edges[-1])
        bins = np.searchsorted(edges, times[inside], side="right")-1
        np.add.at(counts, (bins, self.mapping.group_index[event_positions[inside]]), 1)
        observed_count = np.broadcast_to(observed_count, counts.shape).copy()
        complete = observed_count == total
        seconds = float(edges[1]-edges[0]) * float(dt_seconds)
        rates = np.full(counts.shape, np.nan)
        density = np.full(counts.shape, np.nan)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            np.divide(counts, observed_count*seconds, out=rates, where=observed_count > 0)
            np.divide(counts, seconds*self.cell_area_mm2, out=density, where=complete)
        if (not np.isfinite(rates[observed_count > 0]).all()
                or not np.isfinite(density[complete]).all()):
            raise ValueError("EVENT_FIELD_RATES_NOT_FINITE")
        field = FieldReadout(counts, observed_count, total, rates, complete,
                             self.mapping.field_ids, _readonly(self.mapping.unit_ids[observed]))
        return SpatialFieldReadout(field, density)
