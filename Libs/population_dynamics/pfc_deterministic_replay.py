"""Execute pinned PFC equations with recorded or source-stream external inputs.

NumPy is the source arithmetic comparator. Explicit CUDA uses Torch float64;
its FFT, prefix sum and transcendental rounding are separately measured. This
is an execution comparator, not a generative model or scientific qualification.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from collections.abc import Mapping
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from .pfc_input_pairing import _array, _relative, _resolve_cases, _under
from .provenance import freeze_json, sha256_file
from . import pfc_kernel_parameters as kernel_parameters

SOURCE_ENTRY_SHA256 = "416a914b9a358ae4b685d1a7119d14d777f31c70253f88e01b85d7ede993d720"
FLOAT_RTOL = 1e-10
FLOAT_ATOL_SI = 1e-12
_STIMULUS_DT = 1.0 * 1e-3
_SCHEDULE = ["fft_nmda", "stimulus", "summed_nmda", "update_E", "update_I",
             "threshold_E", "threshold_I", "gaba_projection", "gaba_projection",
             "input_I", "input_E", "reset_E", "reset_I"]


@dataclass(frozen=True)
class FocalSpikeSuppression:
    """Reference-only E-spike intervention after a retained first spike.

    Voltage and AMPA remain internal to the focal unit. Its absent output
    spikes must be removed before refractory updates, recording and resets.
    All nonfocal dynamics continue through the original full-network runner.
    """

    unit_index: int
    after_tick: int
    stop_tick: int

    def validate(self, n_e, total_ticks, backend):
        if (backend != "numpy" or type(self.unit_index) is not int
                or not 0 <= self.unit_index < n_e or type(self.after_tick) is not int
                or type(self.stop_tick) is not int
                or not 0 <= self.after_tick < self.stop_tick <= total_ticks):
            raise ValueError("PFC_REFERENCE_SUPPRESSION_IDENTITY_OR_INTERVAL")


class _Arithmetic:
    """Explicit array/device boundary; no automatic backend fallback."""

    def __init__(self, backend):
        if backend not in ("numpy", "cuda"):
            raise ValueError("PFC_REPLAY_BACKEND_UNSUPPORTED")
        self.backend, self.torch = backend, None
        if backend == "cuda":
            self.torch = importlib.import_module("torch")
            if not self.torch.cuda.is_available():
                raise RuntimeError("PFC_REPLAY_EXPLICIT_CUDA_UNAVAILABLE")
            self.torch.cuda.synchronize()
        self.xp = np if self.torch is None else self.torch

    def array(self, values):
        values = np.asarray(values)
        if self.torch is None:
            return values.copy()
        return self.torch.as_tensor(values.copy(), device="cuda")

    def host(self, values):
        if self.torch is None:
            return np.asarray(values).copy()
        return values.detach().cpu().numpy().copy()

    def integer_steps(self, elapsed, dt):
        # Pinned Brian timestep(): shift by 0.001dt and truncate, not round.
        shifted = (elapsed + 1e-3 * dt) / dt
        return (np.asarray(shifted, dtype=np.int64) if self.torch is None
                else shifted.to(dtype=self.torch.int64))

    def rfft(self, values):
        return self.xp.fft.rfft(values)

    def irfft(self, values):
        return self.xp.fft.irfft(values)

    def cumulative_last(self, values):
        if self.torch is None:
            return np.cumsum(values, dtype=np.float64)[-1]
        return self.torch.cumsum(values, dim=0, dtype=self.torch.float64)[-1]

    def indices(self, mask):
        if self.torch is None:
            return np.flatnonzero(mask).astype(np.int64, copy=False)
        return self.torch.nonzero(mask, as_tuple=True)[0].cpu().numpy().copy()

    def device_indices(self, indices):
        return indices if self.torch is None else self.torch.as_tensor(indices, device="cuda")

    def metadata(self):
        result = dict(backend=self.backend, state_dtype="float64", numpy_version=np.__version__,
                      kernel_normalization="source math operations and scipy.special.erf",
                      synaptic_gaba="repeated +1 in ascending inhibitory spike order; II excludes self",
                      external_inputs="recorded uint16 vectors, converted exactly to float64; no resampling",
                      fft="numpy.fft.rfft/irfft" if self.torch is None else "torch.fft.rfft/irfft CUDA float64",
                      summed_nmda="numpy sequential float64 cumsum" if self.torch is None
                                  else "torch CUDA float64 cumsum; rounding can differ from sequential source",
                      exp="numpy.exp" if self.torch is None else "torch.exp CUDA float64",
                      operation_execution="eager operations; no compiled/fused replacement",
                      full_unit_spike_indices_copied_to_host_each_tick=self.torch is not None)
        if self.torch is not None:
            result.update(torch_version=self.torch.__version__, cuda_version=self.torch.version.cuda,
                          cuda_device=self.torch.cuda.get_device_name(self.torch.cuda.current_device()),
                          cuda_process_peak_allocated_bytes=self.torch.cuda.max_memory_allocated())
        return result


def _dimensions(config):
    n_e, n_i = config["excitatory_neurons"], config["inhibitory_neurons"]
    dt_ms, duration_ms = config["dt_ms"], config["duration_ms"]
    if (any(type(n) is not int or not 0 < n < 2**31 for n in (n_e, n_i))
            or n_e < 2 or n_e % 2
            or any(type(x) not in (int, float) or not np.isfinite(x) or x <= 0
                   for x in (dt_ms, duration_ms))):
        raise ValueError("PFC_REPLAY_INVALID_DIMENSIONS_OR_ODD_NATIVE_FFT_SIZE")
    dt = dt_ms * 1e-3
    ticks = int(round(duration_ms / dt_ms))
    stimulus_stride = int(round(_STIMULUS_DT / dt))
    if (ticks <= 0 or not np.isclose(ticks * dt_ms, duration_ms, rtol=0, atol=1e-9)
            or stimulus_stride < 1 or not np.isclose(stimulus_stride * dt, _STIMULUS_DT, rtol=0, atol=1e-15)):
        raise ValueError("PFC_REPLAY_REQUIRES_COMMENSURATE_NATIVE_CLOCKS")
    if config.get("synapse_execution", "factorized_uniform_projections") != "factorized_uniform_projections":
        raise ValueError("PFC_REPLAY_REQUIRES_QUALIFIED_FACTORIZED_SOURCE_EXECUTION")
    return n_e, n_i, dt, ticks, stimulus_stride


def _constants(config, population):
    scale = config.get("weight_scaling_factor", 2.0)
    if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("PFC_REPLAY_SOURCE_WEIGHT_SCALE_INVALID")
    # Preserve the pinned source's scalar SI conversion and multiplication order.
    e = population == "E"
    return dict(cm=(0.5 if e else 0.2) * 1e-9,
                g_leak=(25.0 if e else 20.0) * 1e-9,
                g_external=(3.1 if e else 2.38) * 1e-9,
                g_inhibitory=((1.336 if e else 1.024) * 1e-9) * scale,
                g_nmda=((0.381 if e else 0.292) * 1e-9) * scale,
                e_leak=-70.0 * 1e-3, e_gaba=-70.0 * 1e-3,
                e_ampa=0.0 * 1e-3, e_nmda=0.0 * 1e-3,
                tau_ampa=.9 * 2.0 * 1e-3, tau_gaba=10.0 * 1e-3,
                tau_nmda_s=.65 * 100.0 * 1e-3, tau_nmda_x=.94 * 2.0 * 1e-3,
                alpha_nmda=0.5 * 1e3, refractory=(2.0 if e else 1.0) * 1e-3,
                threshold=-50.0 * 1e-3, reset=-60.0 * 1e-3)


def _kernel(n_e, config=None):
    return kernel_parameters.source_kernel(n_e, config)


def _target_indices(n_e, center, width):
    if not np.isfinite(center) or not np.isfinite(width) or width < 0:
        raise ValueError("PFC_REPLAY_STIMULUS_GEOMETRY_INVALID")
    middle = int(round(n_e / 360. * center))
    half = int(round(n_e / 360. * width / 2))
    return np.asarray([idx % n_e for idx in range(middle - half, middle + half + 1)], dtype=np.int64)


def _control(config, case, n_e, arithmetic):
    result = {}
    for prefix, default_center in (("stimulus", 180.0), ("distractor", 90.0)):
        amplitude = case.get(prefix + "_strength_nA", config.get(prefix + "_strength_nA", .07 if prefix == "stimulus" else 0.0))
        start = config.get(prefix + "_start_ms", 0.0) * 1e-3
        duration = config.get(prefix + "_duration_ms", 0.0) * 1e-3
        if any(not np.isfinite(x) or x < 0 for x in (amplitude, start, duration)):
            raise ValueError("PFC_REPLAY_STIMULUS_PROGRAM_INVALID")
        indices = _target_indices(n_e, config.get(prefix + "_center_deg", default_center),
                                  config.get(prefix + "_width_deg", 40.0))
        result[prefix] = dict(start=start, end=start + duration, amplitude=amplitude * 1e-9,
                              indices=arithmetic.device_indices(indices))
    return result


def _held_current(current, program, operation_time):
    cue, distractor = program["stimulus"], program["distractor"]
    cue_active = cue["start"] <= operation_time < cue["end"]
    if cue_active:
        current[cue["indices"]] = cue["amplitude"]
    else:
        current[:] = 0.0
    distractor_active = distractor["start"] <= operation_time < distractor["end"]
    if distractor_active:
        current[distractor["indices"]] = distractor["amplitude"]
    # No clearing of non-cue cells while cue remains active: native persistence.
    return cue_active, distractor_active


def _applied_current_metadata(n_e, dt, ticks, stimulus_stride):
    return dict(schema="pfc_applied_current_segments_v1", variable="I_stim", population="E", units="A",
                observed_unit_count=n_e, dt_seconds=dt, duration_ticks=ticks,
                stimulus_operation_stride_ticks=stimulus_stride,
                timing="after_source_stimulus_operation_before_rk2",
                capture_rule="initial_source_operation_and_cue_or_distractor_active_state_transitions",
                source_indices="all_E_source_indices_ascending", archive_indices="E_population_archive_indices_in_E_then_I_unit_order",
                segment_bounds="start_inclusive_stop_exclusive",
                segment_interval_semantics="held by source current-update semantics; not repeated measurements",
                coverage="full E I_stim vectors at segment starts; excludes inhibitory, synaptic and total membrane currents")


def _increment(v, a, b, q, current, constants, dt, arithmetic):
    c = constants
    nmda = ((-c["g_nmda"]) * (q * ((-c["e_nmda"]) + v))) / (
        1.0 + (0.280112044817927 * arithmetic.xp.exp((-62.0) * v)))
    passive = (((c["g_external"] * (a * ((-c["e_ampa"]) + v)))
                + (c["g_inhibitory"] * (b * ((-c["e_gaba"]) + v))))
               + (c["g_leak"] * ((-c["e_leak"]) + v)))
    rhs = nmda - passive if current is None else (nmda + current) - passive
    return (dt / c["cm"]) * rhs


def _rk2(state, constants, t, dt, arithmetic, population):
    a, b, v, q = (state[key] for key in ("s_AMPA", "s_GABA", "v", "s_NMDA_total"))
    refractory_ticks = int(np.asarray((constants["refractory"] + 1e-3 * dt) / dt, dtype=np.int64))
    available = arithmetic.integer_steps(t - state["lastspike"], dt) >= refractory_ticks
    la, lb = (-dt) / constants["tau_ampa"], (-dt) / constants["tau_gaba"]
    ka, kb = la * a, lb * b
    current = state["I_stim"] if population == "E" else None
    kv = arithmetic.xp.where(available, _increment(v, a, b, q, current, constants, dt, arithmetic), 0.0)
    a_mid, b_mid, v_mid = (0.5 * ka) + a, (0.5 * kb) + b, (0.5 * kv) + v
    next_a, next_b = (la * a_mid) + a, (lb * b_mid) + b
    next_v = arithmetic.xp.where(available, v + _increment(
        v_mid, a_mid, b_mid, q, current, constants, dt, arithmetic), v)
    if population == "E":
        s, x = state["s_NMDA"], state["x"]
        inverse_tau = 1.0 / constants["tau_nmda_s"]
        alpha, lx = constants["alpha_nmda"], (-dt) / constants["tau_nmda_x"]
        ks = dt * ((alpha * (x * (1.0 - s))) - (inverse_tau * s))
        kx = lx * x
        next_s = (dt * ((alpha * (((0.5 * kx) + x) * ((1.0 + (0.5 * (-ks))) - s)))
                        - (inverse_tau * ((0.5 * ks) + s)))) + s
        next_x = (lx * ((0.5 * kx) + x)) + x
        state.update(s_NMDA=next_s, x=next_x)
    state.update(v=next_v, s_AMPA=next_a, s_GABA=next_b, not_refractory=available)


_STATE_FIELDS = ("s_AMPA", "s_GABA", "s_NMDA_total", "v", "lastspike", "not_refractory")
_E_STATE_FIELDS = ("s_NMDA", "x", "I_stim")
_RNG_FIELDS = ("algorithm", "key", "position", "has_gauss", "cached_gaussian")


def _snapshot_integer(arrays, key):
    value = np.asarray(arrays[key])
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError("PFC_RESUME_INTEGER_SNAPSHOT_REQUIRED:" + key)
    return int(value)


def _rng_state(arrays, prefix=""):
    if any(prefix + key not in arrays for key in _RNG_FIELDS):
        raise ValueError("PFC_RESUME_COMPLETE_MT19937_SNAPSHOT_REQUIRED:" + prefix)
    values = {key: np.asarray(arrays[prefix + key]) for key in _RNG_FIELDS}
    if (values["algorithm"].shape != () or values["algorithm"].item() != "MT19937"
            or values["key"].dtype != np.uint32 or values["key"].shape != (624,)
            or values["position"].dtype != np.int64 or values["position"].shape != ()
            or not 0 <= values["position"].item() <= 624
            or values["has_gauss"].dtype != np.int8 or values["has_gauss"].shape != ()
            or values["has_gauss"].item() not in (0, 1)
            or values["cached_gaussian"].dtype != np.float64
            or values["cached_gaussian"].shape != () or not np.isfinite(values["cached_gaussian"])):
        raise ValueError("PFC_RESUME_MT19937_SNAPSHOT_SCHEMA:" + prefix)
    return ("MT19937", values["key"].copy(), int(values["position"]),
            int(values["has_gauss"]), float(values["cached_gaussian"]))


def _resume_drive_kind(drive):
    # Frozen fresh-run packages do not stage coupled-drive helpers. Load those
    # classes only for an explicit continuation snapshot or restore operation.
    from .pfc_native_drive import NativeDrive
    if isinstance(drive, NativeDrive):
        return "native"
    from .pfc_coupled_drive import CoupledFineDrive
    from .pfc_clock_coupling import CoupledClockRows
    if isinstance(drive, CoupledFineDrive) and type(drive._rows) is CoupledClockRows:
        return "coupled"
    raise ValueError("PFC_RESUME_NATIVE_OR_UNWRAPPED_COUPLED_DRIVE_REQUIRED")


def snapshot_drive_state(drive):
    """Copy native/coupled cursors, input totals, RNGs and buffered child rows.

    The coupled schema matches existing first-passage prefix archives.
    Delivered input replacements are separate from native drive totals.
    """
    kind = _resume_drive_kind(drive)
    common = dict(fine_ticks=np.asarray(drive.ticks), fine_total_ticks=np.asarray(drive.total_ticks),
                  events_E=np.asarray(drive.event_totals["E"]), events_I=np.asarray(drive.event_totals["I"]))
    if kind == "native":
        from .pfc_native_drive import _state_arrays
        return {**_state_arrays(drive.get_state()), **common,
                "prefix_kind": np.asarray("native_numpy_stream")}
    drive._check_source_cursor()
    rows = drive._rows
    buffer = (np.empty((0, drive.n_e + drive.n_i), dtype=np.uint16)
              if drive._buffer is None else drive._buffer.copy())
    return {**{key: value.copy() for key, value in rows.state_arrays().items()}, **common,
            "prefix_kind": np.asarray("coupled_binomial_clock_refinement_stream"),
            "coarse_ticks": np.asarray(rows.coarse.ticks),
            "coarse_events_E": np.asarray(rows.coarse.event_totals["E"]),
            "coarse_events_I": np.asarray(rows.coarse.event_totals["I"]),
            "refinement_factor": np.asarray(drive.factor), "buffer_offset": np.asarray(drive._buffer_offset),
            "buffered_parent_children": buffer}


def restore_drive_state(drive, arrays, *, start_tick=None, event_totals=None):
    """Restore a fresh matched drive without regenerating prefix inputs.

    Native five-array legacy snapshots require explicit start_tick and native
    event_totals. Coupled prefix archives contain both cursors, all three RNGs,
    native totals and buffered children. Construct the drive with the original
    full-duration configuration, seeds and refinement settings. The caller
    verifies source/configuration provenance before using the restored drive.
    """
    kind = _resume_drive_kind(drive)
    arrays = dict(arrays)
    if drive.ticks != 0:
        raise ValueError("PFC_RESUME_RESTORE_REQUIRES_FRESH_DRIVE")
    if "fine_ticks" in arrays:
        saved_tick = _snapshot_integer(arrays, "fine_ticks")
        if start_tick is not None and start_tick != saved_tick:
            raise ValueError("PFC_RESUME_DRIVE_CURSOR_DISAGREEMENT")
        start_tick = saved_tick
        if _snapshot_integer(arrays, "fine_total_ticks") != drive.total_ticks:
            raise ValueError("PFC_RESUME_DRIVE_DURATION_CHANGED")
        saved_totals = {p: _snapshot_integer(arrays, "events_" + p) for p in ("E", "I")}
        if event_totals is not None and dict(event_totals) != saved_totals:
            raise ValueError("PFC_RESUME_NATIVE_INPUT_TOTAL_DISAGREEMENT")
        event_totals = saved_totals
    if (type(start_tick) is not int or not 0 <= start_tick <= drive.total_ticks
            or not isinstance(event_totals, Mapping) or set(event_totals) != {"E", "I"}
            or any(type(value) is not int or value < 0 for value in event_totals.values())):
        raise ValueError("PFC_RESUME_NATIVE_CURSOR_AND_TOTALS_REQUIRED")
    if kind == "native":
        if "prefix_kind" in arrays and np.asarray(arrays["prefix_kind"]).item() != "native_numpy_stream":
            raise ValueError("PFC_RESUME_NATIVE_INPUT_KIND_CHANGED")
        state = _rng_state(arrays)
        drive._rng.set_state(state)
    else:
        if (np.asarray(arrays["prefix_kind"]).item() != "coupled_binomial_clock_refinement_stream"
                or _snapshot_integer(arrays, "refinement_factor") != drive.factor):
            raise ValueError("PFC_RESUME_COUPLED_REFINEMENT_CHANGED")
        coarse_tick = _snapshot_integer(arrays, "coarse_ticks")
        offset = _snapshot_integer(arrays, "buffer_offset")
        buffer = np.asarray(arrays["buffered_parent_children"])
        if (coarse_tick != (start_tick + drive.factor - 1) // drive.factor
                or offset != start_tick % drive.factor or buffer.dtype != np.uint16
                or buffer.shape != (drive.factor if offset else 0, drive.n_e + drive.n_i)
                or np.any(buffer > 1000)):
            raise ValueError("PFC_RESUME_COUPLED_CURSOR_OR_BUFFER_CHANGED")
        coarse_totals = {p: _snapshot_integer(arrays, "coarse_events_" + p) for p in ("E", "I")}
        if any(value < 0 for value in coarse_totals.values()):
            raise ValueError("PFC_RESUME_COARSE_INPUT_TOTAL_INVALID")
        states = [_rng_state(arrays, name + "__") for name in ("coarse", "refinement_1", "refinement_2")]
        rows = drive._rows
        rows.coarse._rng.set_state(states[0])
        rows.coarse.ticks, rows.coarse._events = coarse_tick, coarse_totals
        for rng, state in zip(rows.children, states[1:]):
            rng.set_state(state)
        drive._buffer = buffer.copy() if offset else None
        drive._buffer_offset = offset
    drive.ticks, drive._events = start_tick, dict(event_totals)
    return drive


def _resume_neural_state(terminal_state, prefix_raw, start_tick, n_e, n_i, dt, arithmetic):
    if not isinstance(terminal_state, Mapping) or not isinstance(prefix_raw, Mapping):
        raise ValueError("PFC_RESUME_COMPLETE_TERMINAL_AND_PREFIX_EVENTS_REQUIRED")
    required_state = {p + "__" + field for p in ("E", "I")
                      for field in _STATE_FIELDS + (_E_STATE_FIELDS if p == "E" else ())}
    if (not required_state.issubset(terminal_state)
            or not {"spike_ticks", "spike_unit_indices", "native_spike_counts"}.issubset(prefix_raw)):
        raise ValueError("PFC_RESUME_COMPLETE_TERMINAL_AND_PREFIX_EVENTS_REQUIRED")
    populations = {}
    for population, size, offset in (("E", n_e, 0), ("I", n_i, n_e)):
        state = {}
        for field in _STATE_FIELDS + (_E_STATE_FIELDS if population == "E" else ()):
            key = population + "__" + field
            value = np.asarray(terminal_state[key])
            if (value.shape != (size,) or value.dtype != (bool if field == "not_refractory" else np.float64)
                    or not np.isfinite(value).all()):
                raise ValueError("PFC_RESUME_TERMINAL_STATE_SCHEMA:" + key)
            state[field] = arithmetic.array(value)
        for name, expected in (("source_indices", np.arange(size, dtype=np.int32)),
                               ("archive_indices", np.arange(size, dtype=np.int32) + offset)):
            key = population + "__" + name
            if key in terminal_state:
                value = np.asarray(terminal_state[key])
                if value.dtype != expected.dtype or not np.array_equal(value, expected):
                    raise ValueError("PFC_RESUME_UNIT_IDENTITY_CHANGED:" + key)
        populations[population] = state
    tick_array = np.asarray(prefix_raw["spike_ticks"])
    ids = np.asarray(prefix_raw["spike_unit_indices"])
    if (tick_array.dtype != np.int64 or ids.dtype != np.int64 or tick_array.ndim != 1
            or ids.shape != tick_array.shape or np.any(tick_array < 0) or np.any(tick_array >= start_tick)
            or np.any(ids < 0) or np.any(ids >= n_e + n_i)):
        raise ValueError("PFC_RESUME_PREFIX_EVENT_SCHEMA")
    if len(ids) > 1 and np.any(np.diff(tick_array * (n_e + n_i) + ids) <= 0):
        raise ValueError("PFC_RESUME_PREFIX_EVENTS_NOT_CANONICAL")
    counts = np.bincount(ids, minlength=n_e + n_i).astype(np.int64)
    recorded_counts = np.asarray(prefix_raw["native_spike_counts"])
    if recorded_counts.dtype != np.int64 or not np.array_equal(recorded_counts, counts):
        raise ValueError("PFC_RESUME_PREFIX_SPIKE_COUNTS_CHANGED")
    if "spike_seconds" in prefix_raw and not np.array_equal(prefix_raw["spike_seconds"], tick_array * dt):
        raise ValueError("PFC_RESUME_PREFIX_EVENT_CLOCK_CHANGED")
    for population, size, offset in (("E", n_e, 0), ("I", n_i, n_e)):
        key = population + "__spike_counts"
        if key in terminal_state and not np.array_equal(terminal_state[key], counts[offset:offset + size]):
            raise ValueError("PFC_RESUME_TERMINAL_SPIKE_COUNTS_CHANGED")
    return populations, [tick_array.copy()], [ids.copy()]


def simulate(config, case, initial_voltage_mV, counts_E=None, counts_I=None, backend="numpy", *, drive=None,
             observe_applied_current=False, focal_spike_suppression=None, terminal_state=None,
             start_tick=0, stop_tick=None, prefix_raw=None, drive_state=None, native_event_totals=None,
             prefix_inputs_used=None, prefix_suppressed_reference=None, prefix_applied_current_observation=None):
    """Execute source ticks and return full host arrays plus measured cost.

    Input is either two complete recorded arrays or one matched source drive.
    stop_tick is an optional absolute exclusive execution boundary. Resume uses
    complete terminal_state, start_tick, cumulative prefix_raw and
    prefix_inputs_used. A fresh drive can be restored from drive_state; an
    already restored drive must have ticks == start_tick. Native legacy RNG
    arrays additionally need native_event_totals. Config duration stays the full
    source duration. Only [start_tick, stop_tick) is executed on continuation.
    The source terminal states, neural spikes and targets are never model inputs.
    Streamed generation is included in wall time and separately measured.
    Optional applied-current snapshots copy actual backend E state after control
    updates at segment starts. Recording overhead is included in wall time; full
    current vectors are never transferred on unchanged control ticks.
    A FocalSpikeSuppression creates an explicitly labeled conditional reference;
    it is not a source-law episode. Its first-spike state and recurrent forcing
    are captured without retaining full-network state snapshots.
    """
    started = time.perf_counter()
    n_e, n_i, dt, ticks, stimulus_stride = _dimensions(config)
    end_tick = ticks if stop_tick is None else stop_tick
    resuming = terminal_state is not None
    if (type(start_tick) is not int or type(end_tick) is not int
            or not 0 <= start_tick < end_tick <= ticks
            or resuming != (start_tick > 0)
            or (not resuming and any(value is not None for value in
                (prefix_raw, drive_state, native_event_totals, prefix_inputs_used,
                 prefix_suppressed_reference, prefix_applied_current_observation)))):
        raise ValueError("PFC_RESUME_TERMINAL_STATE_AND_ABSOLUTE_CLOCK_REQUIRED")
    suppression = focal_spike_suppression
    if suppression is not None:
        if not isinstance(suppression, FocalSpikeSuppression):
            raise ValueError("PFC_REFERENCE_SUPPRESSION_TYPE")
        suppression.validate(n_e, ticks, backend)
    suppressed_threshold_ticks = []
    conditional_initial = None
    reference_forcing = []
    initial = np.asarray(initial_voltage_mV)
    if not resuming and (initial.dtype != np.float64 or initial.shape != (n_e + n_i,)
            or not np.isfinite(initial).all() or np.any(initial < -60.0) or np.any(initial >= -50.0)):
        raise ValueError("PFC_REPLAY_INITIAL_VOLTAGE_SUPPORT_INVALID")
    source_drive = drive
    generation_seconds = 0.0
    current_observation_seconds = 0.0
    current_ticks, current_states, current_values = [], [], []
    if drive_state is not None:
        if source_drive is None:
            raise ValueError("PFC_RESUME_RNG_REQUIRES_SOURCE_DRIVE")
        restore_drive_state(source_drive, drive_state, start_tick=start_tick, event_totals=native_event_totals)
    if source_drive is None:
        for values, size in ((counts_E, n_e), (counts_I, n_i)):
            if not isinstance(values, np.ndarray) or values.dtype != np.uint16 or values.shape != (ticks, size):
                raise ValueError("PFC_REPLAY_COMPLETE_UINT16_INPUT_REQUIRED")
    elif (counts_E is not None or counts_I is not None
          or source_drive.n_e != n_e or source_drive.n_i != n_i
          or source_drive.dt_seconds != dt or source_drive.ticks != start_tick
          or source_drive.total_ticks < end_tick):
        raise ValueError("PFC_STREAM_REQUIRES_MATCHED_SOURCE_DRIVE_AT_START_TICK")
    arithmetic = _Arithmetic(backend)
    constants = {p: _constants(config, p) for p in ("E", "I")}
    if resuming:
        populations, event_ticks, event_ids = _resume_neural_state(
            terminal_state, prefix_raw, start_tick, n_e, n_i, dt, arithmetic)
    else:
        populations = {}
        for population, size, offset in (("E", n_e, 0), ("I", n_i, n_e)):
            state = {name: arithmetic.array(np.zeros(size, dtype=np.float64))
                     for name in ("s_AMPA", "s_GABA", "s_NMDA_total")}
            state.update(v=arithmetic.array(initial[offset:offset + size] * 0.001),
                         lastspike=arithmetic.array(np.full(size, -1e4, dtype=np.float64)),
                         not_refractory=arithmetic.array(np.ones(size, dtype=bool)))
            if population == "E":
                state.update({name: arithmetic.array(np.zeros(size, dtype=np.float64))
                              for name in ("s_NMDA", "x", "I_stim")})
            populations[population] = state
        event_ticks, event_ids = [], []
    e, i = populations["E"], populations["I"]
    program = _control(config, case, n_e, arithmetic)
    fft_kernel = arithmetic.rfft(arithmetic.array(_kernel(n_e, config)))
    chunk_rows = max(1, min(256, (32 * 1024 * 1024) // (8 * (n_e + n_i))))
    totals = {"E": 0, "I": 0}
    if resuming:
        if (not isinstance(prefix_inputs_used, Mapping)
                or not {"ticks_consumed", "no_resampling", "input_mode", "E", "I"}.issubset(prefix_inputs_used)
                or prefix_inputs_used["ticks_consumed"] != start_tick
                or prefix_inputs_used["no_resampling"] != (source_drive is None)
                or prefix_inputs_used["input_mode"] != ("recorded" if source_drive is None else "native_numpy_stream")):
            raise ValueError("PFC_RESUME_CUMULATIVE_DELIVERED_INPUTS_REQUIRED")
        for population, size in (("E", n_e), ("I", n_i)):
            prior = prefix_inputs_used[population]
            if (not isinstance(prior, Mapping) or not {"values", "events"}.issubset(prior)
                    or prior["values"] != start_tick * size or type(prior["events"]) is not int or prior["events"] < 0):
                raise ValueError("PFC_RESUME_DELIVERED_INPUT_TOTAL_INVALID:" + population)
            totals[population] = prior["events"]
        if suppression is not None and prefix_suppressed_reference is not None:
            prior = prefix_suppressed_reference
            if any(prior[key] != getattr(suppression, key) for key in ("unit_index", "after_tick", "stop_tick")):
                raise ValueError("PFC_RESUME_SUPPRESSION_IDENTITY_CHANGED")
            conditional_initial = prior["conditional_initial"]
            suppressed_threshold_ticks = np.asarray(prior["threshold_candidate_ticks"]).tolist()
            forcing = prior["forcing"]
            for index in range(len(forcing["tick"])):
                reference_forcing.append({key: np.asarray(value)[index].item() for key, value in forcing.items()})
        elif suppression is not None and start_tick > suppression.after_tick:
            raise ValueError("PFC_RESUME_PREFIX_SUPPRESSION_CAPTURE_REQUIRED")
        if observe_applied_current:
            if prefix_applied_current_observation is None:
                raise ValueError("PFC_RESUME_PREFIX_CURRENT_OBSERVATION_REQUIRED")
            prior = prefix_applied_current_observation["arrays"]
            if (len(prior["segment_stop_ticks"]) == 0 or prior["segment_stop_ticks"][-1] != start_tick
                    or np.asarray(prior["current_A"]).shape != (len(prior["segment_start_ticks"]), n_e)):
                raise ValueError("PFC_RESUME_PREFIX_CURRENT_OBSERVATION_SCHEMA")
            current_ticks = np.asarray(prior["segment_start_ticks"]).tolist()
            current_states = list(zip(np.asarray(prior["cue_active"]).tolist(), np.asarray(prior["distractor_active"]).tolist()))
            current_values = [np.array(value, copy=True) for value in prior["current_A"]]
    context = nullcontext() if arithmetic.torch is None else arithmetic.torch.no_grad()
    with context:
        for tick in range(start_tick, end_tick):
            if tick == start_tick or tick % chunk_rows == 0:
                block_start = tick
                stop = min((tick // chunk_rows + 1) * chunk_rows, end_tick)
                if source_drive is None:
                    blocks = {"E": counts_E[tick:stop], "I": counts_I[tick:stop]}
                else:
                    generation_started = time.perf_counter()
                    blocks = source_drive.take(stop - tick)
                    generation_seconds += time.perf_counter() - generation_started
                drive = {}
                for population, size in (("E", n_e), ("I", n_i)):
                    block = np.asarray(blocks[population])
                    if block.dtype != np.uint16 or block.shape != (stop - tick, size):
                        raise ValueError("PFC_STREAM_INPUT_BLOCK_SCHEMA_CHANGED")
                    if np.any(block > 1000):
                        raise ValueError(f"PFC_REPLAY_INPUT_SUPPORT_INVALID:{population}:tick{tick}")
                    totals[population] += int(block.sum(dtype=np.uint64))
                    drive[population] = arithmetic.array(block.astype(np.float64))
            t = tick * dt
            # start: source FFT precedes held stimulus, both before groups.
            e["s_NMDA_total"] = arithmetic.irfft(fft_kernel * arithmetic.rfft(e["s_NMDA"]))
            if tick % stimulus_stride == 0:
                control_state = _held_current(e["I_stim"], program, (tick // stimulus_stride) * _STIMULUS_DT)
                if observe_applied_current and (not current_states or control_state != current_states[-1]):
                    observation_started = time.perf_counter()
                    # Copy the state that the following RK2 consumes, including
                    # any native distractor residual outside the cue cells.
                    current_values.append(arithmetic.host(e["I_stim"]))
                    current_ticks.append(tick)
                    current_states.append(control_state)
                    current_observation_seconds += time.perf_counter() - observation_started
            # groups/-1: sum the OLD E NMDA state before either RK2 update.
            i["s_NMDA_total"][:] = arithmetic.cumulative_last(e["s_NMDA"])
            if suppression is not None and suppression.after_tick < tick < suppression.stop_tick:
                reference_forcing.append(dict(tick=tick, time_seconds=t,
                    **{key: float(e[key][suppression.unit_index])
                       for key in ("s_GABA", "s_NMDA_total", "I_stim", "s_NMDA", "x")}))
            _rk2(e, constants["E"], t, dt, arithmetic, "E")
            _rk2(i, constants["I"], t, dt, arithmetic, "I")
            spikes = {}
            for population, offset in (("E", 0), ("I", n_e)):
                state = populations[population]
                selected = arithmetic.indices((state["v"] > constants[population]["threshold"])
                                              & state["not_refractory"])
                if (suppression is not None and population == "E"
                        and suppression.after_tick < tick < suppression.stop_tick):
                    retained = selected != suppression.unit_index
                    if not np.all(retained):
                        suppressed_threshold_ticks.append(tick)
                        selected = selected[retained]
                spikes[population] = selected
                selected_device = arithmetic.device_indices(selected)
                state["not_refractory"][selected_device] = False
                state["lastspike"][selected_device] = t
                if len(selected):
                    event_ids.append(selected + offset)
                    event_ticks.append(np.full(len(selected), tick, dtype=np.int64))
            # synapses/-1: preserve each +1 and native ascending presynaptic order.
            for index in spikes["I"]:
                i["s_GABA"][:int(index)] += 1.0
                i["s_GABA"][int(index) + 1:] += 1.0
            for _ in spikes["I"]:
                e["s_GABA"][:] += 1.0
            if suppression is not None and tick == suppression.after_tick:
                u = suppression.unit_index
                if u not in spikes["E"]:
                    raise ValueError("PFC_REFERENCE_FIRST_SPIKE_NOT_RETAINED")
                state = {key: value[u].item() for key, value in e.items()}
                state.update(v=constants["E"]["reset"], x=float(e["x"][u] + 1.0))
                conditional_initial = dict(unit_index=u, population="E", first_spike_tick=tick,
                    window_stop_tick=suppression.stop_tick, state=state,
                    timing="first_spike_reset_with_preinput_ampa", fresh_input_included=False,
                    voltage_units="V", time_units="s")
            # synapses/0: native I input, then E input; effects enter the next RK2.
            i["s_AMPA"][:] += drive["I"][tick - block_start]
            e["s_AMPA"][:] += drive["E"][tick - block_start]
            # resets: E increments post-RK2 x; I only resets voltage.
            for population in ("E", "I"):
                selected = arithmetic.device_indices(spikes[population])
                populations[population]["v"][selected] = constants[population]["reset"]
                if population == "E":
                    populations[population]["x"][selected] += 1.0
        if arithmetic.torch is not None:
            arithmetic.torch.cuda.synchronize()
    spike_ticks = np.concatenate(event_ticks) if event_ticks else np.empty(0, dtype=np.int64)
    spike_ids = np.concatenate(event_ids) if event_ids else np.empty(0, dtype=np.int64)
    order = np.lexsort((spike_ids, spike_ticks))
    spike_ticks, spike_ids = spike_ticks[order], spike_ids[order]
    counts = np.bincount(spike_ids, minlength=n_e + n_i).astype(np.int64, copy=False)
    if np.any(counts > np.iinfo(np.int32).max):
        raise ValueError("PFC_REPLAY_NATIVE_MONITOR_COUNT_RANGE_EXCEEDED")
    terminal = {}
    for population, size, offset in (("E", n_e, 0), ("I", n_i, n_e)):
        terminal.update({population + "__" + name: arithmetic.host(value)
                         for name, value in populations[population].items()})
        terminal[population + "__source_indices"] = np.arange(size, dtype=np.int32)
        terminal[population + "__archive_indices"] = np.arange(size, dtype=np.int32) + offset
        terminal[population + "__spike_counts"] = counts[offset:offset + size].astype(np.int32)
    raw = dict(spike_unit_indices=spike_ids, spike_ticks=spike_ticks, spike_seconds=spike_ticks * dt,
               native_spike_counts=counts,
               terminal_voltage_mV=np.concatenate((terminal["E__v"], terminal["I__v"])) / 0.001)
    current_observation = None
    if observe_applied_current:
        observation_started = time.perf_counter()
        current_observation = dict(
            arrays=dict(source_unit_indices=np.arange(n_e, dtype=np.int64),
                        archive_unit_indices=np.arange(n_e, dtype=np.int64),
                        segment_start_ticks=np.asarray(current_ticks, dtype=np.int64),
                        segment_stop_ticks=np.asarray(current_ticks[1:] + [end_tick], dtype=np.int64),
                        cue_active=np.asarray([state[0] for state in current_states], dtype=bool),
                        distractor_active=np.asarray([state[1] for state in current_states], dtype=bool),
                        current_A=np.stack(current_values)),
            metadata=_applied_current_metadata(n_e, dt, end_tick, stimulus_stride))
        current_observation_seconds += time.perf_counter() - observation_started
    elapsed = time.perf_counter() - started
    arithmetic_path = arithmetic.metadata()
    if source_drive is not None:
        if source_drive.ticks != end_tick:
            raise ValueError("PFC_STREAM_FINAL_CLOCK_CHANGED")
        arithmetic_path["external_inputs"] = "source NumPy MT19937 binomial blocks; native per-tick I-then-E draw order"
    result = dict(raw=raw, terminal_state=terminal, elapsed_seconds=elapsed,
                cost=dict(wall_seconds=elapsed, backend=backend, arithmetic_path=arithmetic_path,
                          native_input_generation_seconds=generation_seconds,
                          applied_current_capture_seconds=current_observation_seconds,
                          applied_current_capture_cost_scope="snapshot copies including backend synchronization, and array assembly; transition detection included in wall time",
                          scope="simulate: validation, setup, input generation if streamed, transfer, all ticks, full output extraction and optional current observation; excludes file hashing and reference comparison"),
                inputs_used=dict(ticks_consumed=end_tick, no_resampling=source_drive is None,
                                 input_mode="recorded" if source_drive is None else "native_numpy_stream",
                                 E=dict(values=end_tick * n_e, events=totals["E"]),
                                 I=dict(values=end_tick * n_i, events=totals["I"])))
    if resuming or stop_tick is not None:
        result["continuation"] = dict(start_tick=start_tick, stop_tick=end_tick, full_source_ticks=ticks,
                                      executed_ticks=end_tick - start_tick, reused_prefix_ticks=start_tick,
                                      terminal_state_resumed=resuming, cumulative_events_and_inputs=True)
        result["cost"]["scope"] = "simulate invocation: validation, setup, input generation, ticks in [start_tick, stop_tick), full cumulative output extraction; excludes prior prefix execution"
    if suppression is not None:
        expected_rows = max(0, min(end_tick, suppression.stop_tick) - suppression.after_tick - 1)
        if (len(reference_forcing) != expected_rows
                or (end_tick > suppression.after_tick and conditional_initial is None)):
            raise ValueError("PFC_REFERENCE_SUPPRESSION_CAPTURE_INCOMPLETE")
        names = ("tick", "time_seconds", "s_GABA", "s_NMDA_total", "I_stim", "s_NMDA", "x")
        result["suppressed_reference"] = dict(conditional_initial=conditional_initial,
            forcing={key: np.asarray([row[key] for row in reference_forcing], dtype=np.int64 if key == "tick" else float)
                     for key in names},
            threshold_candidate_ticks=np.asarray(suppressed_threshold_ticks, dtype=np.int64),
            unit_index=suppression.unit_index, after_tick=suppression.after_tick, stop_tick=suppression.stop_tick,
            source_law_trajectory=False, network_feedback_recomputed=True,
            voltage_and_ampa_are_unobserved_reference_coordinates=True)
        if end_tick < suppression.stop_tick:
            result["suppressed_reference"]["capture_complete"] = False
    if current_observation is not None:
        result["applied_current_observation"] = current_observation
    return result


def _schedule_contract(generated, receipt, dt):
    rows = generated["native_network_schedule"]
    if generated["full_network_operation_schedule"] != "captured_during_first_native_input_draw" or not rows:
        raise ValueError("PFC_REPLAY_CAPTURED_NATIVE_SCHEDULE_REQUIRED")
    names, abstract = {}, {}
    for population in ("E", "I"):
        code = {row["kind"]: row for row in generated["populations"][population]}
        if set(code) != {"state_updater", "threshold:spike", "reset:spike"}:
            raise ValueError("PFC_REPLAY_GENERATED_RUNNER_SET_CHANGED")
        abstract[population] = {kind: row["abstract_code"] for kind, row in code.items()}
        for kind, role, when in (("state_updater", "update", "groups"),
                                 ("threshold:spike", "threshold", "thresholds"),
                                 ("reset:spike", "reset", "resets")):
            row = code[kind]
            if row["code_object_class"] != "NumpyCodeObject" or row["dt_seconds"] != dt or row["when"] != when or row["order"] != 0:
                raise ValueError("PFC_REPLAY_GENERATED_RUNNER_CLOCK_CHANGED")
            names[row["name"]] = role + "_" + population
        reset = "v=v_reset_excit; x+=1.0" if population == "E" else "v=v_reset_inhib"
        threshold = "_cond = (v>v_firing_threshold_" + ("excit" if population == "E" else "inhib") + ") and not_refractory"
        refractory = "0.002000" if population == "E" else "0.001000"
        if (code["reset:spike"]["abstract_code"] != reset or code["threshold:spike"]["abstract_code"] != threshold
                or not code["state_updater"]["abstract_code"].startswith(
                    "not_refractory = timestep(t - lastspike, dt) >= timestep(" + refractory + ", dt)\n")):
            raise ValueError("PFC_REPLAY_GENERATED_SPIKE_OR_REFRACTORY_RULE_CHANGED")
    provider_names = {row["name"]: "input_" + row["population"] for row in receipt["providers"]}
    schedule = []
    ignored = {"NeuronGroup", "StateMonitor", "SpikeMonitor", "PopulationRateMonitor"}
    for row in rows:
        if row["active"] is not True:
            raise ValueError("PFC_REPLAY_INACTIVE_NATIVE_COMPONENT")
        kind = row["component_class"]
        if kind in ignored:
            continue
        if kind == "NetworkOperation":
            functions = {"update_nmda_sum": "fft_nmda", "stimulate_network": "stimulus"}
            role = functions[row["function_name"]]
            if row["when"] != "start" or row["order"] != 0 or row["dt_seconds"] != (_STIMULUS_DT if role == "stimulus" else dt):
                raise ValueError("PFC_REPLAY_NATIVE_OPERATION_CLOCK_CHANGED")
        elif kind == "UniformProjection":
            if row["function_name"] != "apply_projection" or row["order"] != -1 or row["dt_seconds"] != dt:
                raise ValueError("PFC_REPLAY_NATIVE_PROJECTION_CHANGED")
            role = {"groups": "summed_nmda", "synapses": "gaba_projection"}[row["when"]]
        elif kind == "PoissonInput":
            role = provider_names[row["name"]]
            if row["when"] != "synapses" or row["order"] != 0 or row["dt_seconds"] != dt:
                raise ValueError("PFC_REPLAY_NATIVE_INPUT_SCHEDULE_CHANGED")
        elif kind in ("StateUpdater", "Thresholder", "Resetter"):
            role = names[row["name"]]
        else:
            raise ValueError("PFC_REPLAY_UNSUPPORTED_NATIVE_COMPONENT:" + kind)
        schedule.append(role)
    canonical = list(schedule)
    # Actual cue cases reverse E/I name order. These three pairs are disjoint:
    # both fixed NMDA inputs precede updates; neither threshold changes the other
    # population; both resets follow all projections. No other swap is allowed.
    for offset, role in ((3, "update"), (5, "threshold"), (11, "reset")):
        expected = [role + "_E", role + "_I"]
        if set(canonical[offset:offset + 2]) != set(expected):
            raise ValueError("PFC_REPLAY_NATIVE_COMPONENT_ORDER_CHANGED")
        canonical[offset:offset + 2] = expected
    if canonical != _SCHEDULE:
        raise ValueError("PFC_REPLAY_NATIVE_COMPONENT_ORDER_CHANGED")
    equation_hash = hashlib.sha256(json.dumps(abstract, sort_keys=True).encode()).hexdigest()
    return dict(semantic_schedule=schedule, executed_schedule=_SCHEDULE,
                independent_population_pair_order_normalized=canonical != schedule,
                generated_abstract_equations_sha256=equation_hash)


def _array_comparison(case_id, family, key, actual, expected):
    exact = family == "raw" and key != "terminal_voltage_mV"
    if family == "terminal_state":
        exact = key.split("__", 1)[1] in ("lastspike", "not_refractory", "I_stim", "source_indices", "archive_indices", "spike_counts")
    atol = 0.0 if exact else (FLOAT_ATOL_SI / 0.001 if family == "raw" else FLOAT_ATOL_SI)
    row = dict(case_id=case_id, family=family, array=key, comparison="exact" if exact else "float_tolerance",
               rtol=0.0 if exact else FLOAT_RTOL, atol=atol, passed=False,
               actual_shape=None if actual is None else list(actual.shape),
               expected_shape=None if expected is None else list(expected.shape),
               actual_dtype=None if actual is None else str(actual.dtype),
               expected_dtype=None if expected is None else str(expected.dtype),
               values_compared=0, unequal_values=None, failing_values=None, max_abs_error=None,
               actual_nonfinite_values=None, expected_nonfinite_values=None, bitwise_equal=False)
    if actual is None or expected is None or actual.shape != expected.shape:
        return row
    row["values_compared"] = int(actual.size)
    finite = np.isfinite(actual) & np.isfinite(expected)
    row["actual_nonfinite_values"] = int((~np.isfinite(actual)).sum())
    row["expected_nonfinite_values"] = int((~np.isfinite(expected)).sum())
    row["unequal_values"] = int(np.count_nonzero(actual != expected))
    row["bitwise_equal"] = actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
    if np.any(finite):
        row["max_abs_error"] = float(np.max(np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))))
    elif actual.size == 0:
        row["max_abs_error"] = 0.0
    acceptable = (actual == expected) if exact else np.isclose(actual, expected, rtol=FLOAT_RTOL, atol=atol)
    row["failing_values"] = int(np.count_nonzero(~(acceptable & finite)))
    row["passed"] = actual.dtype == expected.dtype and row["failing_values"] == 0
    return row


def _save_arrays(path, values):
    with path.open("xb") as stream:
        np.savez(stream, **values)


def replay(validation: Path, pairing_path: Path, output: Path, backend="numpy"):
    validation, pairing_path, output = Path(validation), Path(pairing_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_PFC_DETERMINISTIC_REPLAY", backend=backend,
                  created_utc=datetime.now(timezone.utc).isoformat(), expected_cases=8, completed_cases=0,
                  passed_cases=0, cases=[], arrays=[], consumed_files=[],
                  implementation_sha256=sha256_file(Path(__file__)),
                  fixed_float_tolerance=dict(rtol=FLOAT_RTOL, atol_SI_or_dimensionless=FLOAT_ATOL_SI),
                  scope="captured-input full-unit source execution comparator", source_inputs_replayed=True,
                  generative_model=False, scientific_fidelity_qualified=False, gpu_parity_qualified=False,
                  production_episodes=0)
    try:
        pairing = json.loads(pairing_path.read_text())
        delivery_path = validation / "delivery.json"
        delivery = json.loads(delivery_path.read_text())
        report.update(pairing_sha256=sha256_file(pairing_path), delivery_sha256=sha256_file(delivery_path))
        if (pairing["status"] != "PASS_NATIVE_INPUT_PAIRING" or pairing["input_identity_mapping_complete"] is not True
                or not all(value is True for value in pairing["gates"].values())
                or pairing["denominator"]["validated_cases"] != 8
                or pairing["delivery_sha256"] != report["delivery_sha256"]
                or delivery["status"] != "ARCHIVED_NATIVE_INPUT_OBSERVATION_PARITY"):
            raise ValueError("PFC_REPLAY_QUALIFIED_PAIRED_INPUTS_REQUIRED")
        declared = {_relative(row["path"]): row for row in delivery["files"]}
        if len(declared) != len(delivery["files"]):
            raise ValueError("PFC_REPLAY_DUPLICATE_DELIVERY_ENTRY")
        paired_files = {row["path"]: row for row in pairing["consumed_files"]}
        checked = {}

        def consume(name, digest=None):
            name = _relative(name)
            entry = declared[name]
            if (digest is not None and entry["sha256"] != digest
                    or name in paired_files and paired_files[name]["sha256"] != entry["sha256"]):
                raise ValueError("PFC_REPLAY_INPUT_LINEAGE_CHANGED:" + name)
            if name not in checked:
                path = validation / name
                if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                    raise ValueError("PFC_REPLAY_INPUT_CONTENT_CHANGED:" + name)
                checked[name] = dict(path=name, sha256=entry["sha256"], bytes=entry["bytes"])
                report["consumed_files"].append(checked[name])
            return validation / name

        def load(name, digest=None):
            return json.loads(consume(name, digest).read_text())

        prior = load("comparison/comparison.json")
        execution = load("capture/execution.json", prior["capture_execution_sha256"])
        native = load("capture/native/report.json", execution["native_report_sha256"])
        inputs = load("capture/inputs/report.json", execution["observation_report_sha256"])
        if (prior["status"] != "PASS_NATIVE_INPUT_OBSERVATION_PARITY" or prior != delivery["comparison"]
                or execution["status"] != "RECORDED_NATIVE_PFC_INPUT_OBSERVATION" or execution["mode"] != "capture"
                or native["status"] != "NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS"
                or inputs["status"] != "NATIVE_SOURCE_INPUT_OBSERVATION_COMPLETE" or inputs["mode"] != "capture"
                or any(doc["source_entry_sha256"] != SOURCE_ENTRY_SHA256 for doc in (native, inputs, pairing))
                or native["brian2_version"] != "2.5.4" or inputs["brian2_version"] != "2.5.4"
                or execution["config_sha256"] != pairing["config_sha256"]):
            raise ValueError("PFC_REPLAY_PINNED_SOURCE_OBSERVATION_REQUIRED")
        consume(pairing["config_file"], pairing["config_sha256"])
        config = native["resolved_config"]
        cases = _resolve_cases(config)
        ids = [row["case_id"] for row in cases]
        if (len(cases) != 8 or len(set(ids)) != 8 or execution["cases"] != 8
                or any([row["case_id"] for row in doc["cases"]] != ids for doc in (native, inputs, prior))
                or [row["case"] for row in pairing["cases"]] != cases):
            raise ValueError("PFC_REPLAY_FULL_SOURCE_CASE_SET_REQUIRED")
        n_e, n_i, dt, ticks, _ = _dimensions(config)
        report.update(source_entry_sha256=SOURCE_ENTRY_SHA256, config_sha256=execution["config_sha256"],
                      resolved_config=config, source_numpy_version=native["numpy_version"],
                      cases=[dict(case_id=row["case_id"], status="NOT_RUN") for row in cases])
        freeze_json(output / "run.json", {key: value for key, value in report.items() if key not in ("arrays", "consumed_files")})
        equation_hash = None
        for number, case in enumerate(cases):
            row, native_case, paired = inputs["cases"][number], native["cases"][number], pairing["cases"][number]
            case_output = output / case["case_id"]
            case_output.mkdir()
            receipt = load(_under("capture/inputs", row["receipt_file"]), row["receipt_sha256"])
            if (receipt["status"] != "NATIVE_INPUT_OBSERVATION_CASE_COMPLETE" or receipt["mode"] != "capture"
                    or receipt["case"] != case or receipt["source_entry_sha256"] != SOURCE_ENTRY_SHA256
                    or paired["validation_status"] != "verified"):
                raise ValueError("PFC_REPLAY_CASE_BINDING_CHANGED")
            for key in ("initial_rng_seed", "external_noise_seed", "source_rng_seed", "stimulus_strength_nA", "distractor_strength_nA"):
                if native_case[key] != case[key]:
                    raise ValueError("PFC_REPLAY_NATIVE_CASE_PARAMETER_CHANGED:" + key)
            generated_name = _under("capture/inputs", row["generated_code_file"])
            if receipt["generated_code_file"] != row["generated_code_file"] or receipt["generated_code_sha256"] != row["generated_code_sha256"]:
                raise ValueError("PFC_REPLAY_GENERATED_SOURCE_BINDING_CHANGED")
            generated = load(generated_name, row["generated_code_sha256"])
            contract = _schedule_contract(generated, receipt, dt)
            if equation_hash is not None and equation_hash != contract["generated_abstract_equations_sha256"]:
                raise ValueError("PFC_REPLAY_EQUATIONS_CHANGED_BETWEEN_CASES")
            equation_hash = contract["generated_abstract_equations_sha256"]
            initial_name = _under("capture/native", native_case["initial_voltage_file"])
            if paired["initial_voltage_file"] != initial_name or paired["initial_voltage_sha256"] != native_case["initial_voltage_sha256"]:
                raise ValueError("PFC_REPLAY_INITIAL_INPUT_BINDING_CHANGED")
            initial_path = consume(initial_name, native_case["initial_voltage_sha256"])
            providers = {p["population"]: p for p in receipt["providers"]}
            if len(receipt["providers"]) != 2 or set(providers) != {"E", "I"}:
                raise ValueError("PFC_REPLAY_INPUT_POPULATION_SET_CHANGED")
            paths = {}
            for population, units in (("E", n_e), ("I", n_i)):
                provider, mapping = providers[population], paired["populations"][population]
                name = _under("capture/inputs", provider["counts_file"])
                if (mapping["path"] != name or mapping["sha256"] != provider["counts_sha256"]
                        or mapping["shape"] != [ticks, units] or provider["counts_shape"] != [ticks, units]
                        or provider["observed_sampler_calls"] != ticks or provider["expected_sample_rows"] != ticks
                        or provider["dt_seconds"] != dt or provider["units"] != units
                        or provider["target_variable"] != "s_AMPA" or provider["N"] != 1000
                        or provider["rate_Hz"] != 1.4 or provider["probability"] != 1.4 * dt
                        or provider["weight_dimensionless"] != 1.0
                        or provider["source_unit_order"] != "ascending_zero_based_population_index"):
                    raise ValueError("PFC_REPLAY_NATIVE_DRIVE_GEOMETRY_CHANGED")
                paths[population] = consume(name, provider["counts_sha256"])
            with _array(initial_path, (n_e + n_i,), np.float64) as initial:
                with _array(paths["E"], (ticks, n_e), np.uint16) as e_counts:
                    with _array(paths["I"], (ticks, n_i), np.uint16) as i_counts:
                        result = simulate(config, case, initial, e_counts, i_counts, backend=backend)
            if result["inputs_used"]["ticks_consumed"] != ticks:
                raise ValueError("PFC_REPLAY_INCOMPLETE_DRIVE_CONSUMPTION")
            for population, units in (("E", n_e), ("I", n_i)):
                if (result["inputs_used"][population]["values"] != ticks * units
                        or result["inputs_used"][population]["events"] != providers[population]["captured_external_events"]):
                    raise ValueError("PFC_REPLAY_DRIVE_CONSUMPTION_MISMATCH")
            _save_arrays(case_output / "raw.npz", result["raw"])
            _save_arrays(case_output / "terminal_state.npz", result["terminal_state"])
            # Reference neural outputs are first loaded only AFTER execution.
            raw_path = consume(_under("capture/native", native_case["raw_file"]), native_case["raw_sha256"])
            state_path = consume(_under("capture/inputs", row["terminal_state_file"]), row["terminal_state_sha256"])
            comparisons = []
            for family, reference_path in (("raw", raw_path), ("terminal_state", state_path)):
                with np.load(reference_path, allow_pickle=False) as reference:
                    for key in sorted(set(reference.files) | set(result[family])):
                        comparisons.append(_array_comparison(case["case_id"], family, key,
                            result[family].get(key), reference[key] if key in reference.files else None))
            passed = all(value["passed"] for value in comparisons)
            case_report = dict(case_id=case["case_id"], case=case,
                status="PASS_NUMERICAL_SOURCE_COMPARISON" if passed else "FAIL_NUMERICAL_SOURCE_COMPARISON",
                backend=backend, source_entry_sha256=SOURCE_ENTRY_SHA256,
                generated_code_file=generated_name, generated_code_sha256=row["generated_code_sha256"],
                source_contract=contract, pairing_sha256=report["pairing_sha256"], delivery_sha256=report["delivery_sha256"],
                implementation_sha256=report["implementation_sha256"], cost=result["cost"], inputs_used=result["inputs_used"],
                raw_file="raw.npz", raw_sha256=sha256_file(case_output / "raw.npz"),
                terminal_state_file="terminal_state.npz", terminal_state_sha256=sha256_file(case_output / "terminal_state.npz"),
                reference_raw_sha256=native_case["raw_sha256"], reference_terminal_state_sha256=row["terminal_state_sha256"],
                comparisons=comparisons, scientific_fidelity_qualified=False, gpu_parity_qualified=False,
                production_episodes=0)
            freeze_json(case_output / "comparison.json", case_report)
            report["cases"][number] = dict(case_id=case["case_id"], status=case_report["status"],
                report_file=case["case_id"] + "/comparison.json", report_sha256=sha256_file(case_output / "comparison.json"),
                all_arrays_bitwise_equal=all(value["bitwise_equal"] for value in comparisons),
                elapsed_seconds=result["elapsed_seconds"])
            report["arrays"].extend(comparisons)
            report["completed_cases"] += 1
            report["passed_cases"] += int(passed)
            print(json.dumps(report["cases"][number]), flush=True)
        report["status"] = ("PASS_PFC_DETERMINISTIC_REPLAY" if report["passed_cases"] == 8
                            else "FAIL_PFC_DETERMINISTIC_REPLAY")
    except Exception as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error))
        report["status"] = "FAILED_PFC_DETERMINISTIC_REPLAY"
    columns = ("case_id", "family", "array", "comparison", "rtol", "atol", "passed", "actual_shape", "expected_shape",
               "actual_dtype", "expected_dtype", "values_compared", "unequal_values", "failing_values", "max_abs_error",
               "actual_nonfinite_values", "expected_nonfinite_values", "bitwise_equal")
    with (output / "array_comparisons.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report["arrays"])
    report["array_comparisons_file"] = "array_comparisons.csv"
    report["array_comparisons_sha256"] = sha256_file(output / "array_comparisons.csv")
    report["not_completed_cases"] = report["expected_cases"] - report["completed_cases"]
    freeze_json(output / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--pairing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("numpy", "cuda"), default="numpy")
    args = parser.parse_args()
    report = replay(args.validation, args.pairing, args.output, args.backend)
    print(json.dumps({key: report.get(key) for key in
                     ("status", "backend", "expected_cases", "completed_cases", "passed_cases", "failure")}, indent=2))
    return 0 if report["status"] == "PASS_PFC_DETERMINISTIC_REPLAY" else 1


if __name__ == "__main__":
    sys.exit(main())
