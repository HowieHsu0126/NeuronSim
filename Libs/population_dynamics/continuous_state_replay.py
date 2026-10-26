"""Replay continuous state observations against unobserved full/prefix runs.

This is an offline observation-contract check, not a source-fidelity test.
The caller supplies a source episode and one independently acquired prefix
per observation. No simulator state is reconstructed or used for prediction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .nest_gpu_spatial import Execution, STATE_FIELDS
from .provenance import sha256_file


def _checked_report(folder):
    report = json.loads((folder / "report.json").read_text())
    if report["status"] != "EXECUTED" or not all(report["checks"].values()):
        raise ValueError("completed native episode with passing checks required")
    # Bind the data consumed here; unrelated import diagnostics need not be staged.
    for name in ("spikes.npz", "spike_counts.npz", "states.npz", "initial_parameters.npz", "terminal_states.npz"):
        if sha256_file(folder / name) != report["artifacts"][name]:
            raise ValueError(f"native artifact changed: {folder / name}")
    return report


def _equal_arrays(left, right, keys=None):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        selected = set(a.files) if keys is None else set(keys)
        if keys is None and selected != set(b.files):
            raise ValueError("array fields differ")
        for key in selected:
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"array mismatch: {left.name}/{key}")
        return sum(a[key].size for key in selected if key in STATE_FIELDS)


def replay(observed: Path, original: Path, prefixes: dict[int, Path]):
    observed_report = _checked_report(observed)
    source_report = _checked_report(original)
    a = asdict(Execution(**observed_report["config"]))
    b = asdict(Execution(**source_report["config"]))
    declared_ticks = a.pop("state_observation_ticks")
    b.pop("state_observation_ticks")
    # Source RNG paths relocate, but their content binding must remain identical.
    a.pop("source_rng_library"); b.pop("source_rng_library")
    for key in ("graph_manifest_sha256", "initial_state_rng_library_sha256"):
        if observed_report[key] != source_report[key]:
            raise ValueError(f"source binding differs: {key}")
    if a != b or declared_ticks != sorted(prefixes):
        raise ValueError("observer comparison must match the original episode and all requested ticks")
    checks = {}
    for name in ("spikes.npz", "spike_counts.npz", "states.npz", "initial_parameters.npz", "terminal_states.npz"):
        _equal_arrays(observed / name, original / name)
        checks[name] = True
    receipt = json.loads((observed / "state_observations.json").read_text())
    if [row["tick"] for row in receipt["observations"]] != declared_ticks:
        raise ValueError("observation inventory differs from execution config")
    values = 0
    for row in receipt["observations"]:
        prefix = prefixes[row["tick"]]
        report = _checked_report(prefix)
        if report["graph_manifest_sha256"] != source_report["graph_manifest_sha256"]:
            raise ValueError("prefix source graph differs")
        for key in ("initial_state_seed", "background_seed", "initial_state_rng_library_sha256"):
            if report[key] != source_report[key]:
                raise ValueError(f"prefix source realization differs: {key}")
        if not np.isclose(report["config"]["duration_ms"], row["time_ms"], rtol=0, atol=1e-9):
            raise ValueError("prefix duration differs from observation tick")
        path = observed / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError("state observation content changed")
        boundary = json.loads((prefix / "boundary_capture.json").read_text())
        if (boundary["status"] != "CAPTURED" or boundary["terminal_tick"] != row["tick"]
                or sha256_file(prefix / "terminal_propagators.npz") != boundary["propagators_sha256"]
                or sha256_file(prefix / "report.json") != boundary["source_report_sha256"]):
            raise ValueError("prefix boundary/propagator binding differs")
        keys = ["unit_ids", *STATE_FIELDS]
        values += _equal_arrays(path, prefix / "terminal_states.npz", keys)
        _equal_arrays(path, prefix / "terminal_propagators.npz", ["unit_ids", "P11ex", "P11in"])
        _equal_arrays(prefix / "initial_parameters.npz", original / "initial_parameters.npz")
        with np.load(prefix / "spikes.npz", allow_pickle=False) as p, np.load(original / "spikes.npz", allow_pickle=False) as o:
            before = o["spike_ticks"] <= row["tick"]
            for key in ("spike_unit_ids", "spike_ticks", "raw_spike_times_ms"):
                if not np.array_equal(p[key], o[key][before]):
                    raise ValueError(f"prefix events differ: {key}")
        checks[f"events_through_tick_{row['tick']}"] = True
        checks[f"state_at_tick_{row['tick']}"] = True
    return dict(status="PASS", checks=checks, observation_ticks=declared_ticks,
                observed_cortical_units=receipt["cortical_units"],
                full_state_values=values,
                full_terminal_state_values=receipt["cortical_units"] * len(STATE_FIELDS),
                native_phase=receipt["native_phase"], full_hidden_state=False,
                observed_report_sha256=sha256_file(observed / "report.json"),
                original_report_sha256=sha256_file(original / "report.json"),
                scientific_reference_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--prefix", action="append", required=True, metavar="TICK=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prefixes = {}
    for item in args.prefix:
        tick, path = item.split("=", 1)
        if int(tick) in prefixes:
            raise ValueError("repeated prefix tick")
        prefixes[int(tick)] = Path(path)
    result = replay(args.observed, args.original, prefixes)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
