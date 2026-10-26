"""Compare fresh native PFC processes and independently check spike support."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .provenance import freeze_json, sha256_file


def compare(first: Path, second: Path, output: Path):
    reports = [json.loads((root / "report.json").read_text()) for root in (first, second)]
    for field in ("resolved_config", "source_revision", "source_entry_sha256", "brian2_version", "numpy_version"):
        if reports[0][field] != reports[1][field]:
            raise ValueError("PFC_NATIVE_REPLAY_BINDING_CHANGED:" + field)
    if [r["case_id"] for r in reports[0]["cases"]] != [r["case_id"] for r in reports[1]["cases"]]:
        raise ValueError("PFC_NATIVE_REPLAY_CASE_SET_CHANGED")
    n_e = reports[0]["resolved_config"]["excitatory_neurons"]
    dt = reports[0]["resolved_config"]["dt_ms"]
    n_i = reports[0]["resolved_config"]["inhibitory_neurons"]
    separate = reports[0]["resolved_config"].get("random_stream_policy", "source_global") == "independent_initial_external"
    checked, initial_checked, violations = 0, 0, 0
    for a, b in zip(reports[0]["cases"], reports[1]["cases"]):
        if separate:
            if any(a.get(key) != b.get(key) for key in ("initial_rng_seed", "external_noise_seed")):
                raise ValueError("PFC_INITIAL_EXTERNAL_REPLAY_SEEDS_CHANGED")
            initial_name = a["case_id"] + "_initial_voltage.npy"
            values = []
            for root, row in ((first,a), (second,b)):
                if row.get("initial_voltage_file") != initial_name:
                    raise ValueError("PFC_INITIAL_VOLTAGE_REPLAY_PATH_CHANGED")
                path = root / initial_name
                if sha256_file(path) != row["initial_voltage_sha256"]:
                    raise ValueError("PFC_INITIAL_VOLTAGE_REPLAY_FILE_CHANGED")
                values.append(np.load(path, allow_pickle=False))
            # Independent source-order reconstruction; source draws I before E,
            # while every archived per-unit array is ordered E then I.
            rng = np.random.RandomState(a["initial_rng_seed"])
            initial_i = rng.uniform(-60., -50., size=n_i)
            initial_e = rng.uniform(-60., -50., size=n_e)
            expected = np.concatenate((initial_e, initial_i))
            if any(value.dtype != expected.dtype or value.shape != expected.shape
                   or value.tobytes() != expected.tobytes() for value in values):
                raise ValueError("PFC_INITIAL_VOLTAGE_RNG_REPLAY_FAILED")
            initial_checked += len(expected)
        paths = [first / a["raw_file"], second / b["raw_file"]]
        if any(sha256_file(path) != row["raw_sha256"] for path, row in zip(paths, (a, b))):
            raise ValueError("PFC_NATIVE_REPLAY_RAW_FILE_CHANGED")
        with np.load(paths[0], allow_pickle=False) as original, np.load(paths[1], allow_pickle=False) as replay:
            if set(original.files) != set(replay.files):
                raise ValueError("PFC_NATIVE_REPLAY_COLUMNS_CHANGED")
            for name in original.files:
                if not np.array_equal(original[name], replay[name]):
                    raise ValueError("PFC_FRESH_PROCESS_ARRAY_REPLAY_FAILED:" + name)
                checked += original[name].size
            if not np.isfinite(original["terminal_voltage_mV"]).all():
                raise ValueError("PFC_TERMINAL_STATE_NONFINITE")
            ids, ticks = original["spike_unit_indices"], original["spike_ticks"]
            order = np.lexsort((ticks, ids))
            ids, ticks = ids[order], ticks[order]
            same = ids[1:] == ids[:-1]
            # The pinned source defines 2 ms E and 1 ms I absolute refractory periods.
            minimum_ticks = np.rint(np.where(ids[1:] < n_e, 2., 1.) / dt).astype(np.int64)
            violations += int(np.count_nonzero(same & (np.diff(ticks) < minimum_ticks)))
    result = dict(status="PASS" if violations == 0 else "FAILED_REFRACTORY_SUPPORT",
                  original_report_sha256=sha256_file(first / "report.json"),
                  replayed_report_sha256=sha256_file(second / "report.json"),
                  source_revision=reports[0]["source_revision"],
                  comparison_implementation_sha256=sha256_file(Path(__file__)),
                  fresh_process_array_values_checked=checked,
                  fresh_process_initial_voltage_values_checked=initial_checked,
                  source_initial_voltage_rng_replay_values=initial_checked,
                  refractory_violations=violations, core_units=n_e+reports[0]["resolved_config"]["inhibitory_neurons"],
                  complete_episodes=len(reports[0]["cases"]),
                  replay_runs_count_as_additional_samples=False,
                  scientific_fidelity_qualified=False, full_scale_qualified=False)
    freeze_json(output, result)
    if violations:
        raise ValueError("PFC_NATIVE_REFRACTORY_VIOLATIONS")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.original, args.replay, args.output)))


if __name__ == "__main__":
    main()
