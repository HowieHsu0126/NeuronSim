"""Profile an existing PFC coupled input generator without neural simulation."""
from __future__ import annotations

import argparse
import cProfile
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pstats
import shutil
import socket
import sys
import tempfile
import time
import traceback

COARSE_TICKS = 64


class TimedRandomState:
    """Time original binomial calls without changing arguments or returned data."""

    def __init__(self, rng):
        self.rng = rng
        self.calls = 0
        self.wall_seconds = 0.0

    def binomial(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self.rng.binomial(*args, **kwargs)
        finally:
            self.wall_seconds += time.perf_counter() - started
            self.calls += 1

    def get_state(self):
        return self.rng.get_state()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()
    job = os.environ["SLURM_JOB_ID"]
    if socket.gethostname().split(".")[0] != "cpu04" or os.environ.get("SLURM_JOB_GPUS"):
        raise RuntimeError("CPU04_WITHOUT_GPU_REQUIRED")
    archive = args.output_parent / ("run_" + job)
    if archive.exists():
        raise RuntimeError("PROFILE_ARCHIVE_ALREADY_EXISTS")
    work = Path(tempfile.mkdtemp(prefix="neuronsim-pfc-input-profile-" + job + ".", dir="/tmp"))
    os.environ["TMPDIR"] = str(work)
    os.environ["XDG_CACHE_HOME"] = str(work / "cache")
    sys.dont_write_bytecode = True
    report = dict(status="FAILED_PFC_COUPLED_INPUT_PROFILE", job_id=job,
                  node=socket.gethostname(), source_archive=str(args.source_archive),
                  case_id=args.case_id, coarse_ticks_per_clock=COARSE_TICKS,
                  neural_simulations=0, production_episodes=0, source_code_modified=False,
                  scientific_fidelity_qualified=False, matched_scientific_speedup_established=False,
                  scope="Two bounded input-generator prefixes; original full-duration configuration retained.",
                  created_utc=datetime.now(timezone.utc).isoformat(), inputs=[], profiles=[])
    exit_code = 1
    delivered = False
    try:
        shutil.copy2(Path(__file__), work / "profile_pfc_coupled_input.py")
        shutil.copy2(Path(os.environ["NRNSIM_PROFILE_BATCH"]), work / "launcher.sbatch")
        delivery_path = args.source_archive / "delivery.json"
        delivery = json.loads(delivery_path.read_text())
        if delivery["status"] != "COMPLETE_COUPLED_CLOCK_RECURRENT_GAIN_BLOCK" or delivery["verified"] is not True:
            raise RuntimeError("ACCEPTED_SOURCE_BLOCK_REQUIRED")
        manifest = {row["path"]: row for row in delivery["files"]}
        report["source_delivery_sha256"] = digest(delivery_path)
        selected = sorted(name for name in manifest if name.startswith("Libs/") and name.endswith(".py"))
        report_paths = sorted(name for name in manifest if name.startswith("source_cases/")
                              and name.endswith("/" + args.case_id + "/report.json"))
        if len(report_paths) != 2:
            raise RuntimeError("TWO_ACCEPTED_FINE_CLOCK_REPORTS_REQUIRED")
        for name in selected + report_paths:
            row = manifest[name]
            target = work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.source_archive / name, target)
            if target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
                raise RuntimeError("SOURCE_BINDING_CHANGED:" + name)
            report["inputs"].append(dict(path=name, sha256=row["sha256"], bytes=row["bytes"]))
        sys.path.insert(0, str(work / "Libs"))
        from population_dynamics.resource_probe import probe
        health = probe(work / "resource_preflight.json", 4)
        if health["status"] != "PASS":
            raise RuntimeError("COMPUTE_RESOURCE_PREFLIGHT_FAILED:" + ",".join(health["reasons"]))
        import numpy as np
        from population_dynamics.pfc_coupled_drive import create_coupled_drive
        report["numpy_version"] = np.__version__
        prefix_states = []
        initials = []
        factors = []
        for name in report_paths:
            source = json.loads((work / name).read_text())
            if (source["status"] != "COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE"
                    or source["source_execution_complete"] is not True
                    or source["coupled_core_units"] != 100000
                    or source["numpy_version"] != np.__version__):
                raise RuntimeError("QUALIFIED_100K_GENERATOR_CONFIGURATION_REQUIRED")
            config, case = source["resolved_config"], source["selected_case"]
            initial, drive = create_coupled_drive(config, case, source["coarse_dt_ms"], source["refinement_seeds"])
            if case["case_id"] != args.case_id or drive._rows.coarse.remaining < COARSE_TICKS:
                raise RuntimeError("PROFILE_CASE_OR_PREFIX_INVALID")
            factor = drive.factor
            factors.append(factor)
            # Exactly the block-size expression used by the archived neural engine.
            chunk_rows = max(1, min(256, (32 * 1024 * 1024) // (8 * (drive.n_e + drive.n_i))))
            target_ticks = COARSE_TICKS * factor
            # Keep an uninstrumented reference outside the measured interval.
            reference_initial, reference = create_coupled_drive(
                config, case, source["coarse_dt_ms"], source["refinement_seeds"])
            expected = {population: np.empty((target_ticks, size), dtype=np.uint16)
                        for population, size in (("E", drive.n_e), ("I", drive.n_i))}
            while reference.ticks < target_ticks:
                offset = reference.ticks
                block = reference.take(min(chunk_rows, target_ticks - offset))
                for population in expected:
                    expected[population][offset:reference.ticks] = block[population]
            if not np.array_equal(initial, reference_initial):
                raise RuntimeError("TIMING_INSTRUMENTATION_INITIAL_VALUES_DIFFER")
            timers = [TimedRandomState(drive._rows.coarse._rng)]
            timers.extend(TimedRandomState(rng) for rng in drive._rows.children)
            drive._rows.coarse._rng = timers[0]
            drive._rows.children = timers[1:]
            profiler = cProfile.Profile()
            wall = 0.0
            while drive.ticks < target_ticks:
                offset = drive.ticks
                started = time.perf_counter()
                profiler.enable()
                counts = drive.take(min(chunk_rows, target_ticks - offset))
                profiler.disable()
                wall += time.perf_counter() - started
                for population in expected:
                    if not np.array_equal(counts[population], expected[population][offset:drive.ticks]):
                        raise RuntimeError("TIMING_INSTRUMENTATION_DELIVERED_COUNTS_DIFFER")
                del counts
            reference_state = reference._rows.state_arrays()
            actual_state = drive._rows.state_arrays()
            if (set(reference_state) != set(actual_state) or not all(
                    reference_state[key].dtype == actual_state[key].dtype
                    and reference_state[key].shape == actual_state[key].shape
                    and reference_state[key].tobytes() == actual_state[key].tobytes()
                    for key in reference_state)
                    or reference.event_totals != drive.event_totals
                    or reference._rows.coarse.event_totals != drive._rows.coarse.event_totals):
                raise RuntimeError("TIMING_INSTRUMENTATION_COUNTERS_OR_RNG_STATES_DIFFER")
            if [timer.calls for timer in timers] != [COARSE_TICKS, 3 * COARSE_TICKS, 6 * COARSE_TICKS]:
                raise RuntimeError("EXPECTED_COARSE_AND_REFINEMENT_CALL_COUNTS_CHANGED")
            del expected, reference
            if drive._rows.coarse.ticks != COARSE_TICKS or drive._buffer is not None:
                raise RuntimeError("COMPLETE_COARSE_PREFIX_REQUIRED")
            profile_dir = work / "profiles" / ("factor_" + str(factor))
            profile_dir.mkdir(parents=True)
            profiler.dump_stats(str(profile_dir / "calls.prof"))
            stats = pstats.Stats(profiler)
            rows = []
            for (filename, line, function), (primitive, total, self_seconds, cumulative, _) in stats.stats.items():
                rows.append(dict(filename=filename, line=line, function=function,
                                 primitive_calls=primitive, total_calls=total,
                                 self_seconds=self_seconds, cumulative_seconds=cumulative))
            rows.sort(key=lambda row: row["self_seconds"], reverse=True)
            with (profile_dir / "calls.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            materialization = [row for row in rows if any(term in row["function"]
                                                        for term in ("astype", "concatenate", "stack", "empty"))]
            # These are prefix states, not the final states of a 5000ms episode.
            state = {key: value.copy() for key, value in drive._rows.state_arrays().items()}
            np.savez(profile_dir / "prefix_rng_states.npz", **state)
            prefix_states.append(state)
            initials.append(initial)
            report["profiles"].append(dict(
                factor=factor, dt_ms=source["dt_ms"], coarse_dt_ms=source["coarse_dt_ms"],
                source_rng_seed=source["source_rng_seed"], refinement_seeds=source["refinement_seeds"],
                n_e=drive.n_e, n_i=drive.n_i, configured_duration_ms=config["duration_ms"],
                prefix_duration_ms=COARSE_TICKS * source["coarse_dt_ms"],
                fine_ticks=drive.ticks, coarse_ticks=drive._rows.coarse.ticks,
                chunk_rows=chunk_rows, measured_wall_seconds=wall,
                profile_total_self_seconds=stats.total_tt,
                binomial_call_wall_seconds=sum(timer.wall_seconds for timer in timers),
                binomial_call_count=sum(timer.calls for timer in timers),
                binomial_calls_by_stream=[dict(stream=stream, calls=timer.calls, wall_seconds=timer.wall_seconds)
                                         for stream, timer in zip(("coarse", "refinement_1", "refinement_2"), timers)],
                instrumentation_initial_values_exact=True,
                instrumentation_count_values_compared=target_ticks * (drive.n_e + drive.n_i),
                instrumentation_delivered_counts_exact=True,
                instrumentation_event_totals_and_prefix_rng_states_exact=True,
                visible_materialization_self_seconds=sum(row["self_seconds"] for row in materialization),
                fine_event_totals=drive.event_totals, coarse_event_totals=drive._rows.coarse.event_totals,
                profiler_tables=str((profile_dir / "calls.csv").relative_to(work)),
                binomial_time_scope="Direct timing of original NumPy calls includes internal allocation and profiler overhead; not a pure RNG-kernel measurement.",
                materialization_time_scope="Exclusive self-time of visible astype/concatenate/stack/empty calls; not all allocation cost."))
        if sorted(factors) != [2, 4] or not np.array_equal(initials[0], initials[1]):
            raise RuntimeError("PAIRED_CLOCK_PREFIX_CONFIGURATION_CHANGED")
        left, right = prefix_states
        state_equal = set(left) == set(right) and all(
            left[key].dtype == right[key].dtype and left[key].shape == right[key].shape
            and left[key].tobytes() == right[key].tobytes() for key in left)
        if not state_equal:
            raise RuntimeError("PAIRED_CLOCK_PREFIX_RNG_STATES_DIFFER")
        report.update(status="PROFILED_EXISTING_PFC_COUPLED_INPUT_PREFIX",
                      initial_values_exact=True, coarse_and_auxiliary_prefix_rng_states_exact=True,
                      state_arrays_compared=len(left), full_source_episode_executed=False,
                      instrumentation_count_values_compared=sum(row["instrumentation_count_values_compared"]
                                                                for row in report["profiles"]),
                      instrumentation_count_replay_exact=True)
        exit_code = 0
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        report.update(exit_code=exit_code, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(work / "report.json", report)
        archive.mkdir(parents=True, exist_ok=False)
        files = []
        for source in sorted(work.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(work)
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            sha = digest(source)
            if digest(target) != sha:
                raise RuntimeError("PROFILE_ARCHIVE_COPY_MISMATCH:" + str(relative))
            files.append(dict(path=str(relative), sha256=sha, bytes=source.stat().st_size))
        write_json(archive / "delivery.json", dict(status=report["status"], job_id=job, verified=True, files=files))
        delivered = True
        print(json.dumps(dict(status=report["status"], exit_code=exit_code, archive=str(archive))))
        if delivered:
            shutil.rmtree(work)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
