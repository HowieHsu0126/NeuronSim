"""Recompute registered V1 observation bins from accepted native events."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from . import benchmark_activity_statistics as activity
from . import v1_clock_comparison as clock
from .provenance import freeze_json, sha256_file
from .v1_clock_joint_activity import accepted_comparison
from .v1_clock_noise_comparison import change_summary, select_blocks, load_accepted_panel

# Registered in neural_circuit_benchmark_tasks.md before this analysis.
BIN_MS = (1, 5, 10)
LAGS_MS = (0, 10, 100)
START_MS, STOP_MS = 501, 2991


def binned_descriptions(events: dict, populations: np.ndarray, dt_ms: float):
    """Keep one physical interval and verify exact nested-bin count conservation."""
    if dt_ms not in (clock.REFERENCE_DT_MS, clock.FINE_DT_MS):
        raise ValueError("BINNING_NATIVE_CLOCK_NOT_SUPPORTED")
    times = np.asarray(events["time_ms"])
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("BINNING_NATIVE_EVENT_TIMES_INVALID")
    ticks = np.rint(times / dt_ms).astype(np.int64)
    if not np.array_equal(ticks * dt_ms, times):
        raise ValueError("BINNING_EVENT_NOT_ON_NATIVE_CLOCK")
    result = {}
    for bin_ms in BIN_MS:
        summary, arrays = activity.event_statistics(ticks, events["unit_index"], populations,
            duration_ticks=int(clock.DURATION_MS / dt_ms), start_tick=int(START_MS / dt_ms),
            stop_tick=int(STOP_MS / dt_ms), bin_ticks=int(bin_ms / dt_ms),
            lag_bins=tuple(lag // bin_ms for lag in LAGS_MS))
        result[bin_ms] = (summary, arrays)
    base = result[1][1]
    for bin_ms in BIN_MS:
        arrays = result[bin_ms][1]
        expected = base["population_counts"].reshape(-1, bin_ms, len(base["population_names"])).sum(axis=1)
        if (not np.array_equal(arrays["population_counts"], expected)
                or not np.array_equal(arrays["unit_totals"], base["unit_totals"])):
            raise ValueError("BINNING_NESTED_COUNTS_NOT_EXACT")
    return result


def native_descriptions(archive: Path, consumed: dict, dt_ms: float):
    """Stage the existing admitted native payload; never invoke the simulator."""
    if sha256_file(archive / "delivery.json") != consumed["delivery.json"]["sha256"]:
        raise ValueError("BINNING_NATIVE_DELIVERY_CHANGED")
    with tempfile.TemporaryDirectory(prefix="neuronsim-binning-input-") as temporary:
        work = Path(temporary)
        for name in ("primary/native/cortical_state.npz", "primary/native/cortical_spikes.npz",
                     clock.SOURCE_TYPE_CSV):
            target = work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive / name, target)
            if target.stat().st_size != consumed[name]["bytes"] or sha256_file(target) != consumed[name]["sha256"]:
                raise ValueError("BINNING_NATIVE_INPUT_CHANGED:" + name)
        state = clock.align_state(clock.load_arrays(work / "primary/native/cortical_state.npz"))
        events = clock.align_spikes(clock.load_arrays(work / "primary/native/cortical_spikes.npz"), state, dt_ms)
        names, codes = clock.source_populations(work / clock.SOURCE_TYPE_CSV, state["node_type_id"])
        if len(state["source_id"]) != 230924 or len(names) != 21:
            raise ValueError("BINNING_COMPLETE_NATIVE_POPULATION_REQUIRED")
        populations = names[codes]
        descriptions = binned_descriptions(events, populations, dt_ms)
        identity = dict(source_id=state["source_id"], node_type_id=state["node_type_id"],
                        initial_vm_mV=state["initial_vm_mV"], populations=populations)
        events_count = len(events["time_ms"])
    return identity, descriptions, events_count


def compare(panel: Path, circuit_id: str, first_block: int, second_block: int, output: Path):
    acceptance_path = panel / "panel_acceptance.json"
    acceptance, plan, task, _ = load_accepted_panel(panel)
    submitted = json.loads((panel / "envelope_submission.json").read_text())
    expected = {row["episode_id"]: row["report_sha256"] for row in task["evidence"]}
    jobs = {row["identity"]["episode_id"]: row for row in submitted["cases"]}
    blocks = select_blocks(plan["cases"], circuit_id, first_block, second_block)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="FAILED_V1_CLOCK_BINNING_COMPARISON", exit_code=1,
        acceptance=str(acceptance_path.resolve()), acceptance_sha256=sha256_file(acceptance_path),
        implementation={name: sha256_file(path) for name, path in (
            ("binning", Path(__file__)), ("activity_statistics", Path(activity.__file__)),
            ("clock_alignment", Path(clock.__file__)),
            ("clock_noise_metrics", Path(sys.modules[change_summary.__module__].__file__)))},
        command=sys.argv, circuit_id=circuit_id, block_indices=[first_block, second_block],
        conditions=10, units=230924, populations=21, bin_ms=list(BIN_MS), lag_ms=list(LAGS_MS),
        window=dict(start_ms=START_MS, stop_ms_exclusive=STOP_MS),
        full_native_duration_ms=3000, noise_block_pairs=1,
        new_source_simulations=0, new_unique_episodes=0, production_episodes=0,
        scientific_qualification=False, scientific_bins_frozen=False, tolerances_frozen=False,
        interpretation="Registered local-window observation-bin diagnostic on one matched noise-block pair. All bins and conditions are retained. This is not a noise-distribution estimate or a fidelity/equivalence/convergence decision.")
    rows, inputs = [], []
    try:
        for a_case, b_case in zip(*blocks):
            arms, identities = [], []
            for case in (a_case, b_case):
                key = case["identity"]["episode_id"]
                comparison, digest = accepted_comparison(case, jobs[key])
                if digest != expected[key]:
                    raise ValueError("BINNING_ACCEPTED_COMPARISON_CHANGED:" + key)
                for side, archive, dt in (("reference", Path(case["reference"]), .25),
                                          ("fine", Path(jobs[key]["native_archive"]), .125)):
                    identity, descriptions, native_events = native_descriptions(
                        archive, comparison["inputs"][side]["consumed"], dt)
                    identities.append(identity)
                    arms.append(descriptions)
                    inputs.append(dict(episode_id=key, side=side, native_archive=str(archive),
                        comparison_sha256=digest, native_events=native_events,
                        retained_events=descriptions[1][0]["events"],
                        silent_units=descriptions[1][0]["silent_units"]))
            for identity in identities[1:]:
                for name in identities[0]:
                    if not np.array_equal(identity[name], identities[0][name]):
                        raise ValueError("BINNING_CIRCUIT_OR_RESET_IDENTITY_DIFFERS:" + name)
            for bin_ms in BIN_MS:
                arrays = [arm[bin_ms][1] for arm in arms]
                for array in arrays[1:]:
                    for name in ("population_index", "population_names", "population_unit_counts", "lag_bins"):
                        if not np.array_equal(array[name], arrays[0][name]):
                            raise ValueError("BINNING_SOURCE_POPULATION_MAP_DIFFERS")
                metrics = [("unit_totals", None, "spikes_in_window"),
                           ("population_counts", None, "spikes_per_bin")]
                metrics.extend((name, index, "spike_count_squared")
                    for name in ("unit_count_autocovariance", "population_count_covariance")
                    for index in range(len(LAGS_MS)))
                for metric, index, unit in metrics:
                    values = [array[metric] for array in arrays]
                    if index is not None:
                        values = [value[index] for value in values]
                    rows.append(dict(condition_index=a_case["identity"]["condition_index"],
                        bin_ms=bin_ms, metric=metric, lag_ms=None if index is None else LAGS_MS[index],
                        value_unit=unit, **change_summary(*values)))
            print(json.dumps(dict(condition_index=a_case["identity"]["condition_index"], status="DESCRIBED")), flush=True)
        path = output / "comparison.csv"
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result.update(status="COMPLETE_V1_CLOCK_BINNING_COMPARISON", exit_code=0,
            comparison_rows=len(rows), native_arms=len(inputs), inputs=inputs,
            exact_nested_bin_count_checks=len(inputs)*len(BIN_MS),
            artifacts={path.name: dict(bytes=path.stat().st_size, sha256=sha256_file(path))})
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        freeze_json(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--circuit-id", required=True)
    parser.add_argument("--first-block", type=int, required=True)
    parser.add_argument("--second-block", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = compare(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ("status", "comparison_rows", "native_arms", "scientific_qualification")}))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
