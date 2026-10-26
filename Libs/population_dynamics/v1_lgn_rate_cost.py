"""Measure source LGN rate execution against saved real rate outputs."""
from __future__ import annotations

import argparse
import cProfile
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import pstats
import time

import h5py
import numpy as np

from .provenance import freeze_json, sha256_file
from .v1_configured_inputs import delivered_file
from .v1_lgn_runtime import SourceNodes, gray_movie_support, load_prepared_runtime
from .v1_lgn_source import read_input_construction_cells
from .v1_lgn_transfer import numeric_transfer_support
from .v1_nest_input_replay import bound_json


def profile_rates(runtime: Path, construction: Path, reference: Path, output: Path,
                  *, numeric_transfer: bool = False, all_units: bool = False) -> dict:
    """Time source calls and reject any difference from the saved rate rows."""
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_LGN_RATE_COST_DIAGNOSTIC", production_episodes=0,
                  coupled_neural_episodes=0, profiled_execution=not all_units,
                  numeric_transfer=numeric_transfer,
                  scientific_qualification=False,
                  started_utc=datetime.now(timezone.utc).isoformat())
    try:
        previous, _ = bound_json(reference, "probe/probe.json")
        if previous["status"] != "SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT":
            raise ValueError("LGN_RATE_EXECUTED_REFERENCE_REQUIRED")
        cells, _, binding = read_input_construction_cells(construction)
        if binding != previous.get("input_construction_binding"):
            raise ValueError("LGN_RATE_COST_CONSTRUCTION_BINDING_DIFFERS")
        if not all_units:
            cells = cells.groupby("model_id", sort=False).head(1)
        ids = [int(value) for value in cells.index]
        rows = {int(value): index for index, value in enumerate(previous["source_unit_ids"])}
        if len(cells) != (17400 if all_units else 14) or not set(ids).issubset(rows):
            raise ValueError("LGN_RATE_COST_ALL_MODEL_REPRESENTATIVES_REQUIRED")
        reference_rates = delivered_file(reference, "probe/firing_rates_f_tot.h5")
        with h5py.File(reference_rates, "r") as saved:
            expected_clock = saved["time"][:]
            expected_dtype = saved["firing_rates_Hz"].dtype
        functions, _ = load_prepared_runtime(runtime)
        stimulus = previous["config"]["stimulus"]
        prefix = output / "firing_rates"
        profiler = cProfile.Profile() if not all_units else None
        start = time.perf_counter()
        support = numeric_transfer_support() if numeric_transfer else nullcontext()
        with gray_movie_support(functions, stimulus["contrast_percent"]), support:
            arguments = (SourceNodes(cells), "gratings", str(prefix), stimulus["duration_s"],
                         stimulus["gray_screen_s"], stimulus["spatial_frequency_cycles_per_degree"],
                         stimulus["temporal_frequency_hz"], stimulus["direction_deg"],
                         stimulus["contrast_percent"])
            if profiler is None:
                functions.calculate_firing_rate(*arguments)
            else:
                profiler.runcall(functions.calculate_firing_rate, *arguments)
        elapsed = time.perf_counter() - start
        if profiler is not None:
            profiler.dump_stats(str(output / "source_calls.prof"))
        with h5py.File(str(prefix) + "_f_tot.h5", "r") as generated, h5py.File(reference_rates, "r") as saved:
            clock = generated["time"][:]
            rates = generated["firing_rates_Hz"]
            rates_exact = rates.shape == (len(ids), len(expected_clock))
            for index in range(0, len(ids), 256):
                selected = ids[index:index + 256]
                expected_rates = np.stack([saved["firing_rates_Hz"][rows[value]] for value in selected])
                rates_exact = rates_exact and np.array_equal(rates[index:index + len(selected)], expected_rates)
            rate_shape = list(rates.shape)
            rate_dtype = rates.dtype
        exact = dict(clock=np.array_equal(clock, expected_clock),
                     rates=rates_exact,
                     clock_dtype=clock.dtype == expected_clock.dtype,
                     rate_dtype=rate_dtype == expected_dtype)
        stats = pstats.Stats(profiler) if profiler is not None else None
        calls = []
        for (path, line, function), (primitive, total, own, cumulative, _) in (stats.stats.items() if stats else []):
            if "lgnmodel" in path or path.endswith("lgn_functions.py"):
                calls.append(dict(module=Path(path).name, line=line, function=function,
                                  primitive_calls=primitive, total_calls=total,
                                  own_seconds=own, cumulative_seconds=cumulative))
        report.update(reference_archive=str(reference), construction_binding=binding,
                      reference_report_sha256=sha256_file(reference / "probe/probe.json"),
                      prepared_runtime_sha256=sha256_file(runtime / "prepared.json"),
                      stimulus=stimulus, source_unit_ids=ids,
                      source_models=cells["model_id"].tolist(), rate_shape=rate_shape,
                      elapsed_seconds=elapsed, exact={key: bool(value) for key, value in exact.items()},
                      source_calls=sorted(calls, key=lambda row: row["cumulative_seconds"], reverse=True),
                      interpretation=("Full saved population replay; no scientific qualification is implied."
                                      if all_units else "Instrumented representative execution; full-population speed is not measured."))
        if not all(exact.values()):
            raise ValueError("LGN_RATE_COST_REFERENCE_VALUES_DIFFER")
        report["status"] = "PASS_LGN_RATE_COST_DIAGNOSTIC"
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        freeze_json(output / "cost.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "construction", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--numeric-transfer", action="store_true")
    parser.add_argument("--all-units", action="store_true")
    args = parser.parse_args()
    report = profile_rates(args.runtime, args.construction, args.reference, args.output,
                           numeric_transfer=args.numeric_transfer, all_units=args.all_units)
    print(json.dumps({key: report[key] for key in ("status", "elapsed_seconds", "exact")}))


if __name__ == "__main__":
    main()
