"""Census of archived PFC first-spike timing strata on a full physical frame.

These exact archived-frame weights are not source-law probabilities.  No source
simulation, first-passage integration or population confidence interval is run.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .pfc_delay_analysis import PHASES
from .pfc_deterministic_replay import _constants
from .pfc_first_passage_reference import SpikeTape, validate_tape
from .pfc_fluctuation_analysis import unit_mapping


_ARRAY_ORDER = "window_major_then_ascending_source_unit_index"
_FRAME_KEYS = (
    "excitatory_neurons", "inhibitory_neurons", "excitatory_preference_bins",
    "field_index", "phase", "phase_start_ms", "phase_stop_ms", "window_ms",
    "time_edges_ms", "duration_ms", "field_units", "windows", "frame_size", "array_order",
)
_WINDOW_ARRAYS = ("window_start_tick", "window_stop_tick")
_VALUE_ARRAYS = (
    "observed_spike_count", "first_spike_tick", "refractory_ticks",
    "post_refractory_remaining_ticks", "time_bin", "multispike",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("PFC_HISTORY_TIMING_CENSUS_" + message)


def _ticks(ms: float, dt_ms: float, label: str) -> int:
    require(type(ms) in (int, float) and np.isfinite(ms) and ms >= 0,
            label + "_TIME_INVALID")
    ticks = round(ms / dt_ms)
    require(np.isclose(ticks * dt_ms, ms, rtol=0, atol=1e-12), label + "_OFF_CLOCK")
    return ticks


def _edges_ticks(edges: Sequence[float], dt_ms: float) -> np.ndarray:
    require(len(edges) == 4 and edges[0] == 0
            and all(edges[i] < edges[i + 1] for i in range(3)), "THREE_POSITIVE_TIME_BINS")
    return np.asarray([_ticks(value, dt_ms, "TIME_EDGE") for value in edges], dtype=np.int64)


def _counts(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    counts, bins = arrays["observed_spike_count"], arrays["time_bin"]
    no_first = counts == 0
    bin_counts = np.bincount(bins.astype(np.int64) + 1, minlength=4)
    return dict(
        observed_spikes=int(counts.sum()), no_first_spike_windows=int(no_first.sum()),
        refractory_zero_windows=int(((bins == -1) & ~no_first).sum()),
        eligible_windows=int((bins >= 0).sum()), multispike_windows=int((counts >= 2).sum()),
        time_bin_counts={str(i - 1): int(n) for i, n in enumerate(bin_counts)},
        time_bin_weights={str(i - 1): float(n / counts.size) for i, n in enumerate(bin_counts)},
    )


def census(
    config: Mapping[str, Any], raw: Mapping[str, np.ndarray], *, field_index: int,
    phase: str, window_ms: float, time_edges_ms: Sequence[float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Enumerate every E-unit window, including both kinds of structural zero.

    Per-window values are flattened in window-major, ascending source-unit order.
    The source-unit index vector and the window bounds are each stored once.
    Remaining ticks are signed for observed first spikes, and zero when absent.
    Actual multispike windows remain eligible and are explicitly flagged.
    """
    tape = validate_tape(config, raw)
    mapping, sizes = unit_mapping(tape.n_e, tape.n_i, config["excitatory_preference_bins"])
    return _census_from_tape(config, tape, mapping, sizes, field_index=field_index,
                             phase=phase, window_ms=window_ms, time_edges_ms=time_edges_ms)


def census_many(
    config: Mapping[str, Any], raw: Mapping[str, np.ndarray],
    requests: Iterable[Mapping[str, Any]],
) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """Yield full-frame censuses in request order from one validated source tape.

    Each request supplies the four keyword arguments accepted by ``census``.
    Source validation and unit mapping run once, on the first iteration, even
    when requests is empty. Requests are consumed one at a time; results are
    not accumulated. Duplicate requests produce independent, equal outputs.
    Keep source inputs and pending requests unchanged until iteration finishes.
    """
    tape = validate_tape(config, raw)
    mapping, sizes = unit_mapping(tape.n_e, tape.n_i, config["excitatory_preference_bins"])
    for request in requests:
        yield _census_from_tape(config, tape, mapping, sizes, **request)


def _census_from_tape(
    config: Mapping[str, Any], tape: SpikeTape, mapping: np.ndarray, sizes: np.ndarray,
    *, field_index: int, phase: str, window_ms: float, time_edges_ms: Sequence[float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Enumerate one physical frame using the shared, validated source arrays."""
    dt_ms = config["dt_ms"]
    require(type(field_index) is int and 0 <= field_index < len(sizes) - 1,
            "SOURCE_EXCITATORY_FIELD_REQUIRED")
    require(phase in PHASES, "SOURCE_PHASE_REQUIRED")
    start_ms, stop_ms = PHASES[phase]
    start, stop = (_ticks(value, dt_ms, "PHASE") for value in (start_ms, stop_ms))
    width = _ticks(window_ms, dt_ms, "WINDOW")
    require(width > 0 and 0 <= start < stop <= tape.total_ticks
            and start % width == stop % width == 0, "COMPLETE_PHYSICAL_FRAME_REQUIRED")
    edges = _edges_ticks(time_edges_ms, dt_ms)
    units = np.flatnonzero(mapping == field_index)
    require(units.size == sizes[field_index] and np.all(units < tape.n_e), "SOURCE_UNIT_MAPPING")
    windows = (stop - start) // width
    frame_size = windows * units.size
    starts = np.arange(start, stop, width, dtype=np.int64)
    stops = starts + width

    events = tape.events
    selected = events[(events[:, 0] >= start) & (events[:, 0] < stop)
                      & (mapping[events[:, 1]] == field_index)]
    slots = np.searchsorted(units, selected[:, 1])
    flat = ((selected[:, 0] - start) // width) * units.size + slots
    counts = np.bincount(flat, minlength=frame_size).astype(np.int64, copy=False)
    first = np.full(frame_size, tape.total_ticks, dtype=np.int64)
    np.minimum.at(first, flat, selected[:, 0])
    first[counts == 0] = -1
    refractory_seconds = _constants(config, "E")["refractory"]
    refractory = int(np.asarray(
        (refractory_seconds + 1e-3 * tape.dt_seconds) / tape.dt_seconds, dtype=np.int64))
    refractory_ticks = np.full(frame_size, refractory, dtype=np.int64)
    remaining = np.where(counts > 0, np.repeat(stops, units.size) - first - refractory, 0)
    eligible = (counts > 0) & (remaining > 0)
    require(np.all(remaining[eligible] <= edges[-1]), "ELIGIBLE_TIME_OUTSIDE_FROZEN_BINS")
    require(np.all(eligible[counts >= 2]), "MULTISPIKE_SOURCE_REFRACTORY_INCONSISTENCY")
    bins = np.full(frame_size, -1, dtype=np.int8)
    bins[eligible] = np.searchsorted(edges[1:], remaining[eligible], side="left")
    arrays = dict(unit_indices=units, window_start_tick=starts, window_stop_tick=stops,
                  observed_spike_count=counts, first_spike_tick=first,
                  refractory_ticks=refractory_ticks, post_refractory_remaining_ticks=remaining,
                  time_bin=bins, multispike=counts >= 2)
    require(counts.size == frame_size and int(counts.sum()) == len(selected), "FULL_FRAME_DENOMINATOR")
    metadata = dict(
        excitatory_neurons=tape.n_e, inhibitory_neurons=tape.n_i,
        excitatory_preference_bins=config["excitatory_preference_bins"], field_index=field_index,
        phase=phase, phase_start_ms=start_ms, phase_stop_ms=stop_ms, window_ms=float(window_ms),
        time_edges_ms=[float(value) for value in time_edges_ms], dt_ms=dt_ms,
        duration_ms=config["duration_ms"], field_units=int(units.size), windows=int(windows),
        frame_size=int(frame_size), refractory_ticks=refractory, array_order=_ARRAY_ORDER,
        window_bounds="start_inclusive_stop_exclusive", source_population="E",
        unit_coordinate="source_preferred_angle_field; not soma position",
        remaining_ticks_scope="stop-first-refractory for an observed first spike; zero placeholder when absent",
        weight_scope="Exact conditional archived-frame census weights; not source-law probabilities",
        **_counts(arrays),
    )
    return metadata, arrays


def _validate_census(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    """Check each clock's physical frame and assignments before pairing."""
    require(metadata["array_order"] == _ARRAY_ORDER and metadata["phase"] in PHASES,
            "PHYSICAL_FRAME_IDENTITY")
    require([metadata["phase_start_ms"], metadata["phase_stop_ms"]] == PHASES[metadata["phase"]],
            "SOURCE_PHASE_BOUNDS")
    dt = metadata["dt_ms"]
    require(type(dt) in (int, float) and np.isfinite(dt) and dt > 0, "CLOCK_INVALID")
    width = _ticks(metadata["window_ms"], dt, "WINDOW")
    start, stop = (_ticks(metadata[key], dt, "PHASE") for key in ("phase_start_ms", "phase_stop_ms"))
    require(width > 0 and 0 <= start < stop <= _ticks(metadata["duration_ms"], dt, "DURATION")
            and start % width == stop % width == 0, "COMPLETE_PHYSICAL_FRAME_REQUIRED")
    mapping, _ = unit_mapping(metadata["excitatory_neurons"], metadata["inhibitory_neurons"],
                              metadata["excitatory_preference_bins"])
    require(type(metadata["field_index"]) is int
            and 0 <= metadata["field_index"] < metadata["excitatory_preference_bins"], "SOURCE_EXCITATORY_FIELD_REQUIRED")
    units = np.flatnonzero(mapping == metadata["field_index"])
    windows = (stop - start) // width
    size = windows * units.size
    require(metadata["field_units"] == units.size and metadata["windows"] == windows
            and metadata["frame_size"] == size, "FULL_FRAME_DENOMINATOR")
    for key in ("unit_indices", *_WINDOW_ARRAYS, *_VALUE_ARRAYS[:-1]):
        value = np.asarray(arrays[key])
        require(value.ndim == 1 and value.dtype.kind in "iu", "INTEGER_ARRAY_REQUIRED:" + key)
    require(np.array_equal(arrays["unit_indices"], units), "SOURCE_UNIT_MAPPING")
    starts = np.arange(start, stop, width, dtype=np.int64)
    require(np.array_equal(arrays["window_start_tick"], starts)
            and np.array_equal(arrays["window_stop_tick"], starts + width), "WINDOW_GRID_CHANGED")
    require(all(np.asarray(arrays[key]).shape == (size,) for key in _VALUE_ARRAYS), "FULL_FRAME_ARRAY_SHAPES")
    counts, first = arrays["observed_spike_count"], arrays["first_spike_tick"]
    present = counts > 0
    require(np.all(counts >= 0) and np.all(first[~present] == -1)
            and np.all(first[present] >= np.repeat(starts, units.size)[present])
            and np.all(first[present] < np.repeat(starts + width, units.size)[present]), "FIRST_SPIKE_SUPPORT")
    dt_seconds = dt * 1e-3
    refractory = int(np.asarray((_constants({}, "E")["refractory"] + 1e-3 * dt_seconds)
                               / dt_seconds, dtype=np.int64))
    require(metadata["refractory_ticks"] == refractory
            and np.all(arrays["refractory_ticks"] == refractory), "SOURCE_REFRACTORY_CHANGED")
    remaining = np.where(present, np.repeat(starts + width, units.size) - first - refractory, 0)
    require(np.array_equal(arrays["post_refractory_remaining_ticks"], remaining), "REMAINING_TIME_CHANGED")
    eligible = present & (remaining > 0)
    edges = _edges_ticks(metadata["time_edges_ms"], dt)
    require(np.all(remaining[eligible] <= edges[-1]), "ELIGIBLE_TIME_OUTSIDE_FROZEN_BINS")
    bins = np.full(size, -1, dtype=np.int8)
    bins[eligible] = np.searchsorted(edges[1:], remaining[eligible], side="left")
    require(np.array_equal(arrays["time_bin"], bins)
            and np.asarray(arrays["multispike"]).dtype.kind == "b"
            and np.array_equal(arrays["multispike"], counts >= 2)
            and np.all(eligible[counts >= 2]), "TIMING_ASSIGNMENT_CHANGED")
    require(all(metadata[key] == value for key, value in _counts(arrays).items()), "CENSUS_COUNTS_CHANGED")


def pair(
    coarse_meta: Mapping[str, Any], coarse_arrays: Mapping[str, np.ndarray],
    fine_meta: Mapping[str, Any], fine_arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Pair all physical IDs, preserving clock-specific zeros and first spikes."""
    _validate_census(coarse_meta, coarse_arrays)
    _validate_census(fine_meta, fine_arrays)
    require(all(coarse_meta[key] == fine_meta[key] for key in _FRAME_KEYS), "PAIRED_PHYSICAL_FRAME_CHANGED")
    require(np.array_equal(coarse_arrays["unit_indices"], fine_arrays["unit_indices"]), "PAIRED_SOURCE_UNITS_CHANGED")
    coarse_bin, fine_bin = coarse_arrays["time_bin"], fine_arrays["time_bin"]
    joint = ((coarse_bin.astype(np.int16) + 1) * 4 + fine_bin + 1).astype(np.int8)
    counts = np.bincount(joint, minlength=16)
    size = coarse_meta["frame_size"]
    require(joint.size == size and counts.size == 16 and int(counts.sum()) == size, "FULL_JOINT_DENOMINATOR")
    arrays = {"unit_indices": np.array(coarse_arrays["unit_indices"], copy=True), "joint_time_bin": joint}
    for clock, original in (("coarse", coarse_arrays), ("fine", fine_arrays)):
        arrays.update({clock + "_" + key: np.array(original[key], copy=True)
                       for key in (*_WINDOW_ARRAYS, *_VALUE_ARRAYS)})
    summary = dict(
        frame_size=size, field_units=coarse_meta["field_units"], windows=coarse_meta["windows"],
        clocks={"coarse": dict(coarse_meta), "fine": dict(fine_meta)},
        joint_time_bin_formula="(coarse_time_bin+1)*4+(fine_time_bin+1)",
        joint_time_bin_counts={str(i): int(count) for i, count in enumerate(counts)},
        joint_time_bin_weights={str(i): float(count / size) for i, count in enumerate(counts)},
        coarse_zero_fine_eligible_windows=int(((coarse_bin == -1) & (fine_bin >= 0)).sum()),
        fine_zero_coarse_eligible_windows=int(((fine_bin == -1) & (coarse_bin >= 0)).sum()),
        weight_scope="Exact conditional paired archived-frame census weights; not source-law probabilities",
        array_order=_ARRAY_ORDER, discarded_physical_windows=0,
        population_confidence_interval_computed=False, future_stratified_precision_established=False,
        scientific_fidelity_qualified=False, confirmation_execution_released=False,
        production_episodes=0, new_independent_source_blocks=0, new_neural_trajectories=0,
    )
    return summary, arrays


def run_pair(
    settings: Mapping[str, Any], coarse_dir: Path, fine_dir: Path, output: Path,
) -> dict[str, Any]:
    """Read a pair of caller-authenticated, staged full source archives."""
    require(settings["field_index"] == 5 and settings["phase"] == "cue"
            and settings["window_ms"] == 5.0 and settings["time_edges_ms"] == [0., 1., 2., 3.]
            and settings["expected_units"] == 100000, "REGISTERED_FULL_SCALE_FRAME_REQUIRED")
    seed = settings["source_rng_seed"]
    require(type(seed) is int and settings["case_id"] == f"pfc_{seed}_strong_cue", "REGISTERED_SOURCE_CASE")
    clocks = {}
    source_cases = []
    for clock, directory, expected_dt in (("coarse", Path(coarse_dir), .05), ("fine", Path(fine_dir), .025)):
        config = yaml.safe_load((directory / "config.yaml").read_text())
        report = json.loads((directory / "report.json").read_text())
        require(report["resolved_config"] == config, "SOURCE_CONFIG_REPORT_MISMATCH:" + clock)
        selected = report["selected_case"]
        require(selected["source_rng_seed"] == seed and selected["case_id"] == settings["case_id"],
                "SOURCE_CASE_MISMATCH:" + clock)
        source_cases.append(selected)
        require(config["excitatory_neurons"] + config["inhibitory_neurons"] == settings["expected_units"]
                and config["duration_ms"] == 5000. and config["dt_ms"] == expected_dt
                and settings["clocks"][clock]["dt_ms"] == expected_dt, "SOURCE_DIMENSIONS_OR_CLOCK_CHANGED:" + clock)
        with np.load(directory / "raw.npz", allow_pickle=False) as archive:
            # The tape validator needs no state arrays or reconstructed neural trajectories.
            keys = ("spike_ticks", "spike_unit_indices", "native_spike_counts", "spike_seconds")
            raw = {key: archive[key] for key in keys if key in archive.files}
            clocks[clock] = census(config, raw, **{key: settings[key] for key in
                                   ("field_index", "phase", "window_ms", "time_edges_ms")})
    require(source_cases[0] == source_cases[1], "PAIRED_SOURCE_CASE_CHANGED")
    summary, arrays = pair(*clocks["coarse"], *clocks["fine"])
    require(summary["field_units"] == 5000 and summary["windows"] == 40
            and summary["frame_size"] == 200000, "ALL200000_PHYSICAL_WINDOWS_REQUIRED")
    summary.update(status="COMPLETE_PFC_HISTORY_TIMING_CENSUS", source_rng_seed=seed,
                   case_id=settings["case_id"], selected_case=source_cases[0],
                   input_binding_scope="Source completion and delivery hashes are authenticated by the caller",
                   assignments_file="frame_assignments.npz")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "frame_assignments.npz", **arrays)
    (output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "coarse", "fine", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    report = run_pair(settings, args.coarse, args.fine, args.output)
    print(json.dumps({key: report[key] for key in ("status", "source_rng_seed", "frame_size", "joint_time_bin_counts")}))


if __name__ == "__main__":
    main()
