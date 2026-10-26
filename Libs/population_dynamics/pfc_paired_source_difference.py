"""Locate weak-cue/omission differences in existing complete PFC source episodes.

This diagnostic compares recorded events and terminal arrays. It does not infer
continuous state synchronization, numerical equivalence, or biological absence.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .pfc_delay_analysis import read, require, write_json, _check_readout
from .pfc_scale_analysis import (
    PHASES, _source_registration, _source_header, _source_values,
    _source_manifest, _source_bound_file, scalar_readout,
)
from .provenance import sha256_file


def _event_counts(raw, total_units):
    ticks = np.asarray(raw["spike_ticks"])
    units = np.asarray(raw["spike_unit_indices"])
    require(ticks.ndim == units.ndim == 1 and ticks.shape == units.shape,
            "event arrays require matching one-dimensional shapes")
    require(ticks.dtype.kind in "iu" and units.dtype.kind in "iu",
            "integer event arrays required")
    limit = np.iinfo(np.int64).max
    require(not any(np.any(a < 0) or np.any(a > limit) for a in (ticks, units))
            and not np.any(units >= total_units), "event indices outside valid range")
    require(not np.any(ticks[1:] < ticks[:-1]), "event ticks must be nondecreasing")
    events = np.empty(len(ticks), dtype=[("tick", np.int64), ("unit", np.int64)])
    events["tick"], events["unit"] = ticks, units
    return np.unique(events, return_counts=True)


def compare_events(left_raw, right_raw, total_units, start_tick, stop_tick):
    """Compare canonical (tick, unit) sequences including duplicate multiplicity.

    Within-tick storage order is ignored. The late interval is [start, stop).
    Whole-episode differences include events outside that interval.
    """
    values = (total_units, start_tick, stop_tick)
    require(all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                for v in values), "integer unit count and window bounds required")
    total_units, start_tick, stop_tick = map(int, values)
    require(total_units > 0 and 0 <= start_tick < stop_tick <= np.iinfo(np.int64).max,
            "positive unit count and nonempty int64 window required")
    left, lc = _event_counts(left_raw, total_units)
    right, rc = _event_counts(right_raw, total_units)
    events = np.union1d(left, right)
    delta = np.zeros(len(events), dtype=np.int64)
    delta[np.searchsorted(events, left)] += lc
    delta[np.searchsorted(events, right)] -= rc
    changed = delta != 0

    def late(events, counts):
        mask = (events["tick"] >= start_tick) & (events["tick"] < stop_tick)
        selected, amounts = events[mask], counts[mask]
        vector = np.zeros(total_units, dtype=np.int64)
        np.add.at(vector, selected["unit"], amounts)
        return selected, amounts, vector

    le, lm, lv = late(left, lc)
    re, rm, rv = late(right, rc)
    return dict(
        whole_episode_left_only_events=int(delta[delta > 0].sum()),
        whole_episode_right_only_events=int(-delta[delta < 0].sum()),
        last_differing_tick=int(events["tick"][changed][-1]) if changed.any() else None,
        late_event_sequence_equal=bool(np.array_equal(le, re) and np.array_equal(lm, rm)),
        late_unit_counts_equal=bool(np.array_equal(lv, rv)),
        late_units_with_differing_counts=int(np.count_nonzero(lv != rv)),
    )


def compare_terminal(left, right):
    """Keep cumulative counters and unit identities separate from dynamic state."""
    require(set(left) == set(right), "terminal field keys differ")
    result = {}
    for name in sorted(left):
        a, b = np.asarray(left[name]), np.asarray(right[name])
        require(a.shape == b.shape and a.dtype == b.dtype,
                "terminal shape or dtype differs: " + name)
        field = name.rsplit("__", 1)[-1]
        role = {"spike_counts": "cumulative_spike_counts",
                "source_indices": "source_indices",
                "archive_indices": "archive_indices"}.get(field, "dynamic_state")
        if a.dtype.kind == "b":
            maximum = None
        elif a.dtype.kind in "iu":
            x, y = a.astype(np.uint64), b.astype(np.uint64)
            if a.dtype.kind == "i":
                x ^= np.uint64(1 << 63)
                y ^= np.uint64(1 << 63)
            maximum = int(np.max(np.maximum(x, y) - np.minimum(x, y), initial=np.uint64(0)))
        elif a.dtype.kind == "f":
            require(np.isfinite(a).all() and np.isfinite(b).all(),
                    "nonfinite terminal values: " + name)
            with np.errstate(over="raise", invalid="raise"):
                maximum = np.max(np.abs(a.astype(np.longdouble) - b.astype(np.longdouble)), initial=0)
            require(np.isfinite(maximum) and maximum <= np.finfo(float).max,
                    "terminal difference exceeds finite float range: " + name)
            maximum = float(maximum)
        else:
            raise TypeError("unsupported terminal dtype: " + name)
        result[name] = dict(role=role, differing_values=int(np.count_nonzero(a != b)),
                            max_absolute_difference=maximum)
    return result


def load_reference(root, config_sha, registered):
    """Bind the consumed readout files to their original complete replay."""
    receipt = read(root / "delivery.json")
    require(receipt.get("status") == "ARCHIVED_PFC_SOURCE_READOUTS"
            and receipt.get("exit_code") == 0, "completed phase readout required")
    manifest = _source_manifest(receipt)
    paths = {name: _source_bound_file(root, manifest, "analysis/" + name)
             for name in ("analysis.json", "readout_replay.json", "responses.csv", "phase_unit_counts.npz")}
    report, replay = read(paths["analysis.json"]), read(paths["readout_replay.json"])
    require(report["status"] == "DESCRIBED_SOURCE_EXECUTION_DIAGNOSTIC"
            and replay["status"] == "PASS" and report["complete_source_episodes"] == len(registered)
            and report["registered_source_cases"] == len(registered)
            and not report["missing_source_cases"] and not report["unresolved_input_paths"]
            and report["config_sha256"] == replay["config_sha256"] == config_sha
            and report["readout_replay_sha256"] == sha256_file(paths["readout_replay.json"]),
            "complete matching registered readout replay required")
    bindings = {row["case_id"]: row for row in report["source_bindings"]}
    require(len(bindings) == len(report["source_bindings"]) and set(bindings) == set(registered)
            and replay["source_bindings"] == report["source_bindings"],
            "complete unique replayed source bindings required")
    for name in ("responses.csv", "phase_unit_counts.npz"):
        require(report["artifacts"][name] == replay["artifacts"][name] == sha256_file(paths[name]),
                "readout/replay artifact identity mismatch: " + name)
    with paths["responses.csv"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    late_rows = [row for row in rows if row["phase"] == "late_delay"]
    require(len(late_rows) == len(registered) and {row["case_id"] for row in late_rows} == set(registered),
            "one late readout per registered case required")
    return report, bindings, {row["case_id"]: row for row in late_rows}, paths["phase_unit_counts.npz"]


def compare_input_pair(left, right):
    result = dict(
        exact_pre_cue_events=bool(np.array_equal(left["prefix_ticks"], right["prefix_ticks"])
                                 and np.array_equal(left["prefix_ids"], right["prefix_ids"])),
        exact_initial_voltage=bool(np.array_equal(left["initial"], right["initial"])),
        exact_final_source_rng=bool(set(left["final_rng"]) == set(right["final_rng"])
            and all(np.array_equal(left["final_rng"][k], right["final_rng"][k]) for k in left["final_rng"])),
        same_source_input_consumption=left["inputs_used"] == right["inputs_used"],
    )
    require(all(result.values()), "weak-cue/omission source pairing mismatch")
    return result


def run(source_runs, config_path, reference, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="RUNNING", new_source_simulations=0, production_episodes=0,
                  scientific_fidelity_qualified=False, working_memory_qualified=False,
                  confidence_intervals=False, p_values=False, q_values=False)
    try:
        config, registered, seeds, dimensions = _source_registration(config_path)
        config_sha = sha256_file(config_path)
        reference_report, bindings, late_rows, count_path = load_reference(Path(reference), config_sha, registered)
        expected = {f"pfc_{seed}_{condition}" for seed in seeds for condition in ("cue", "omission")}
        headers = {}
        for source in source_runs:
            status, header = _source_header(source, config, registered, config_sha)
            require(header is not None and status["status"] == "COMPLETE", "completed source required")
            case_id = status["case_id"]
            require(case_id in expected and case_id not in headers,
                    "exactly the registered weak-cue/omission sources are required")
            require(all(header["binding"][key] == bindings[case_id][key]
                        for key in ("report_sha256", "delivery_sha256", "source_runner_sha256")),
                    "source differs from verified readout binding")
            headers[case_id] = header
        require(set(headers) == expected, "all six weak-cue/omission sources required")
        start_ms, stop_ms = PHASES["late_delay"]
        start_tick, stop_tick = (round(value / config["dt_ms"]) for value in (start_ms, stop_ms))
        pairs, replayed_scalars = [], 0
        with np.load(count_path, allow_pickle=False) as reference_counts:
            for seed in seeds:
                loaded, observed_bindings, independent = {}, {}, {}
                for condition in ("cue", "omission"):
                    case_id = f"pfc_{seed}_{condition}"
                    header = headers[case_id]
                    raw, common, paired = _source_values(header, config, registered, dimensions, config_sha)
                    require(common == reference_report["common_execution_binding"],
                            "source program differs from original readout")
                    observed = header["binding"]["applied_current_observation"]
                    require(observed["status"] == "VERIFIED_ACTUAL_CURRENT", "observed source currents required")
                    with np.load(header["source_run"] / "terminal_state.npz", allow_pickle=False) as stored:
                        terminal = {key: stored[key] for key in stored.files}
                    chosen = (raw["spike_ticks"] >= start_tick) & (raw["spike_ticks"] < stop_tick)
                    ids = raw["spike_unit_indices"][chosen]
                    counts = np.bincount(ids, minlength=dimensions["coupled_core_units"])
                    require(np.array_equal(counts, reference_counts[case_id + "_late_delay"]),
                            "all-unit late counts differ from original readout")
                    scalar = scalar_readout(ids, dimensions["n_e"], dimensions["n_i"], stop_ms-start_ms)
                    replayed_scalars += _check_readout(late_rows[case_id], scalar)
                    independent[condition] = scalar
                    loaded[condition] = (raw, paired, terminal)
                    observed_bindings[condition] = header["binding"]
                left, right = loaded["cue"], loaded["omission"]
                events = compare_events(left[0], right[0], dimensions["coupled_core_units"], start_tick, stop_tick)
                last = events["last_differing_tick"]
                pairs.append(dict(source_rng_seed=seed, left_condition="cue", right_condition="omission",
                    source_bindings=observed_bindings, pairing=compare_input_pair(left[1], right[1]),
                    events=events, last_differing_event_ms=None if last is None else last*config["dt_ms"],
                    late_readouts=independent, terminal=compare_terminal(left[2], right[2])))
        result.update(status="COMPLETE_PAIRED_SOURCE_DIFFERENCE", pairs=pairs,
            config_sha256=config_sha, reference_archive=str(reference),
            reference_delivery_sha256=sha256_file(Path(reference) / "delivery.json"),
            registered_source_cases=len(registered), diagnosed_source_cases=len(headers),
            joint_rng_blocks=seeds, coupled_core_units=dimensions["coupled_core_units"],
            dt_ms=config["dt_ms"], duration_ms=config["duration_ms"],
            late_interval_ms=[start_ms, stop_ms], late_interval_ticks=[start_tick, stop_tick],
            late_unit_count_values_verified=len(headers)*dimensions["coupled_core_units"],
            independently_recomputed_scalar_values=replayed_scalars,
            terminal_units=dict(v="V", I_stim="A", lastspike="s",
                                synaptic_state_variables="dimensionless", not_refractory="boolean"),
            event_equality_semantics="canonical tick/unit sequence equality with multiplicity; within-tick storage order ignored",
            interpretation="Recorded finite episodes only. Terminal dynamic state is separate from counters and identities. No continuous-state synchronization, equivalence, biological absence or working-memory qualification.",
            implementation_sha256=sha256_file(Path(__file__)))
    except Exception as error:
        result.update(status="FAILED", error=str(error), error_type=type(error).__name__)
        write_json(output / "report.json", result)
        raise
    write_json(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-readouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.source_run, args.config, args.reference_readouts, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
