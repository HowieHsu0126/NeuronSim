"""Stream coupled fine-clock inputs without changing native coarse RNG draws.

The fine seeded trajectories differ from native fine-clock runs. This adapter
preserves the coupling kernel's marginal law and source unit mapping; neural
fidelity and numerical-error reduction require separate scientific execution.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .pfc_clock_coupling import CoupledClockRows
from .pfc_native_drive import _MAX_WORKING_BLOCK_BYTES, _seed, create_drive

# Reserve space for one parent row, both refinement levels and the temporary
# int64 arrays used by the existing binomial splitter. Output blocks are uint16.
_REFINEMENT_WORK_BYTES_PER_UNIT = 128


def _max_rows(n_e: int, n_i: int) -> int:
    if any(type(value) is not int or value <= 0 for value in (n_e, n_i)):
        raise ValueError("PFC_COUPLED_DRIVE_GEOMETRY_INVALID")
    units = n_e + n_i
    rows = (_MAX_WORKING_BLOCK_BYTES - _REFINEMENT_WORK_BYTES_PER_UNIT * units) // (2 * units)
    if rows < 1:
        raise ValueError("PFC_COUPLED_DRIVE_REFINEMENT_EXCEEDS_MEMORY_BOUND")
    return rows


class CoupledFineDrive:
    """A forward-only fine-clock consumer with at most one parent-row cache.

    Only consumed fine rows enter event_totals. The native coarse cursor may be
    one parent row ahead while its children remain buffered. Returned arrays own
    their data through a new packed allocation and never alias this cache.
    """

    def __init__(self, rows: CoupledClockRows, dt_seconds: float, factor: int):
        coarse = rows.coarse
        if (type(factor) is not int or factor not in (2, 4)
                or type(dt_seconds) not in (int, float)
                or not np.isfinite(dt_seconds) or dt_seconds <= 0
                or dt_seconds != coarse.dt_seconds / factor):
            raise ValueError("PFC_COUPLED_DRIVE_EXACT_HALF_OR_QUARTER_CLOCK_REQUIRED")
        if coarse.ticks != 0:
            raise ValueError("PFC_COUPLED_DRIVE_FRESH_COARSE_STREAM_REQUIRED")
        self.n_e, self.n_i = coarse.n_e, coarse.n_i
        self.max_rows = _max_rows(self.n_e, self.n_i)
        self.dt_seconds = dt_seconds
        self.total_ticks = coarse.total_ticks * factor
        self.factor = factor
        self.ticks = 0
        self._rows = rows
        self._coarse_total_ticks = coarse.total_ticks
        self._events = {"E": 0, "I": 0}
        self._buffer: np.ndarray | None = None
        self._buffer_offset = 0

    @property
    def remaining(self) -> int:
        return self.total_ticks - self.ticks

    @property
    def event_totals(self) -> dict[str, int]:
        return dict(self._events)

    def _check_source_cursor(self) -> None:
        expected = (self.ticks + self.factor - 1) // self.factor
        if (self._rows.coarse.ticks != expected
                or self._rows.coarse.total_ticks != self._coarse_total_ticks):
            raise ValueError("PFC_COUPLED_DRIVE_SOURCE_CURSOR_CHANGED_OR_REUSED")

    def take(self, rows: int) -> dict[str, np.ndarray]:
        """Return consecutive complete E/I rows; reject requests before draws."""
        if (isinstance(rows, (bool, np.bool_))
                or not isinstance(rows, (int, np.integer)) or rows <= 0):
            raise ValueError("PFC_COUPLED_DRIVE_ROWS_MUST_BE_POSITIVE_INTEGER")
        count = int(rows)
        if count > self.remaining:
            raise ValueError("PFC_COUPLED_DRIVE_REQUEST_EXCEEDS_REMAINING_TICKS")
        if count > self.max_rows:
            raise ValueError("PFC_COUPLED_DRIVE_REQUEST_EXCEEDS_MEMORY_BOUND")
        self._check_source_cursor()
        # Allocate before advancing any RNG. Each call has independent storage.
        packed = np.empty((count, self.n_i + self.n_e), dtype=np.uint16)
        written = 0
        while written < count:
            if self._buffer is None:
                # The existing kernel generates both levels on every coarse
                # tick, including when this consumer selects only middle rows.
                self._buffer = self._rows.take_row()[1 if self.factor == 2 else 2]
                self._buffer_offset = 0
            available = self.factor - self._buffer_offset
            copied = min(count - written, available)
            packed[written:written + copied] = self._buffer[
                self._buffer_offset:self._buffer_offset + copied]
            written += copied
            self._buffer_offset += copied
            if self._buffer_offset == self.factor:
                self._buffer = None
                self._buffer_offset = 0
        inhibitory, excitatory = packed[:, :self.n_i], packed[:, self.n_i:]
        self._events["I"] += int(inhibitory.sum(dtype=np.uint64))
        self._events["E"] += int(excitatory.sum(dtype=np.uint64))
        self.ticks += count
        return {"E": excitatory, "I": inhibitory}

    def metadata(self) -> dict[str, Any]:
        return dict(
            input_mode="coupled_binomial_clock_refinement_stream",
            scope="external_input_generator",
            coupling=self._rows.metadata(),
            n_e=self.n_e, n_i=self.n_i, dt_seconds=self.dt_seconds,
            coarse_dt_seconds=self._rows.coarse.dt_seconds,
            refinement_factor=self.factor, total_ticks=self.total_ticks,
            ticks_generated=self.ticks, external_event_totals=self.event_totals,
            buffered_fine_rows=0 if self._buffer is None else self.factor - self._buffer_offset,
            initial_source_draws_preserved=True,
            coarse_source_seed_sequence_preserved=True,
            fine_source_seed_sequence_preserved=False,
            fine_binomial_marginals_preserved_by_construction=True,
            all_refinement_levels_generated=True,
            initial_voltage_archive_order="E_then_I", initial_voltage_units="mV",
            packed_draw_order="coarse_tick_then_level_then_subtick_I_then_E",
            returned_source_unit_order="ascending_zero_based_within_E_and_I",
            returned_dtype="uint16", max_rows=self.max_rows,
            working_block_bound_bytes=_MAX_WORKING_BLOCK_BYTES,
            refinement_work_bytes_reserved_per_unit=_REFINEMENT_WORK_BYTES_PER_UNIT,
            trajectory_retained=False,
            neural_error_reduction_established=False,
        )

    def final_state_arrays(self) -> dict[str, np.ndarray]:
        """Export coarse/auxiliary RNG arrays only after every child is used."""
        self._check_source_cursor()
        if (self.ticks != self.total_ticks or self._rows.coarse.remaining != 0
                or self._buffer is not None or self._buffer_offset != 0):
            raise ValueError("PFC_COUPLED_DRIVE_FINAL_STATE_REQUIRES_COMPLETE_CONSUMPTION")
        return {name: value.copy() for name, value in self._rows.state_arrays().items()}


def create_coupled_drive(
    config: Mapping[str, Any],
    case: Mapping[str, Any],
    coarse_dt_ms: float,
    refinement_seeds: Sequence[int],
    *,
    row_adapter: Callable[[CoupledClockRows], Any] | None = None,
) -> tuple[np.ndarray, CoupledFineDrive]:
    """Return source initial mV and a /2 or /4 coupled fine-clock adapter.

    The coarse config differs only in dt_ms: full duration, case and initial
    source draw order are retained. Auxiliary seeds must be distinct from each
    other and every source RNG seed. Invalid clocks/seeds are rejected before
    the native constructor draws initial voltages. An optional row_adapter
    wraps the fresh coupled row stream before the fine-clock consumer is built;
    it must preserve the same coarse stream and provide its public row interface.
    """
    fine_dt_ms, duration_ms = config["dt_ms"], config["duration_ms"]
    if any(type(value) not in (int, float) or not np.isfinite(value) or value <= 0
           for value in (fine_dt_ms, coarse_dt_ms, duration_ms)):
        raise ValueError("PFC_COUPLED_DRIVE_CLOCK_INVALID")
    factors = [factor for factor in (2, 4) if fine_dt_ms == coarse_dt_ms / factor]
    if len(factors) != 1:
        raise ValueError("PFC_COUPLED_DRIVE_EXACT_HALF_OR_QUARTER_CLOCK_REQUIRED")
    factor = factors[0]
    coarse_ticks = int(round(duration_ms / coarse_dt_ms))
    if (coarse_ticks < 1 or not np.isclose(coarse_ticks * coarse_dt_ms, duration_ms, rtol=0, atol=1e-9)
            or int(round(duration_ms / fine_dt_ms)) != coarse_ticks * factor):
        raise ValueError("PFC_COUPLED_DRIVE_DURATION_OFF_COARSE_CLOCK")
    _max_rows(config["excitatory_neurons"], config["inhibitory_neurons"])
    if not isinstance(refinement_seeds, Sequence) or len(refinement_seeds) != 2:
        raise ValueError("PFC_COUPLED_DRIVE_TWO_REFINEMENT_SEEDS_REQUIRED")
    seeds = tuple(_seed(value) for value in refinement_seeds)
    if len(set(seeds)) != 2:
        raise ValueError("PFC_COUPLED_DRIVE_DISTINCT_REFINEMENT_SEEDS_REQUIRED")
    policy = config.get("random_stream_policy", "source_global")
    seed_names = (("source_rng_seed",) if policy == "source_global"
                  else ("initial_rng_seed", "external_noise_seed"))
    source_seeds = {_seed(case.get(name, config.get(name))) for name in seed_names}
    if set(seeds) & source_seeds:
        raise ValueError("PFC_COUPLED_DRIVE_REFINEMENT_STREAMS_MUST_NOT_REUSE_SOURCE_SEEDS")
    if row_adapter is not None and not callable(row_adapter):
        raise ValueError("PFC_COUPLED_DRIVE_ROW_ADAPTER_MUST_BE_CALLABLE")
    coarse_config = dict(config, dt_ms=coarse_dt_ms)
    initial, coarse = create_drive(coarse_config, case)
    coupled = CoupledClockRows(coarse, seeds)
    if row_adapter is not None:
        coupled = row_adapter(coupled)
        if (getattr(coupled, "coarse", None) is not coarse
                or any(not callable(getattr(coupled, name, None))
                       for name in ("take_row", "state_arrays", "metadata"))):
            raise ValueError("PFC_COUPLED_DRIVE_ROW_ADAPTER_INTERFACE_OR_COARSE_CHANGED")
    return initial, CoupledFineDrive(coupled, fine_dt_ms * 1e-3, factor)
