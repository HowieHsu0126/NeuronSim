"""Finite primary-seed temporal moments for the accepted full gray LGN drive.

The reference is an independent uniform primary-seed hypothesis. The implemented
starting-seed vector is verified separately. Seed-ensemble covariance is centered
across invocations at each time, not across time within an episode. These input
realizations do not count as coupled cortical episodes or establish equivalence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time

import h5py
import numpy as np
import yaml

from .provenance import freeze_json, sha256_file
from .resource_probe import probe
from .v1_driver_statistics import finite_seed_count_lookup, source_count_moments
from .v1_lgn_runtime import load_prepared_runtime
from .v1_nest_input_replay import bound_json
from .v1_nest_time_semantics import encode_precise_times

SEEDS = 10000
DURATION_MS = 3000
LAGS = (0, 1, 10, 100)


def temporal_counts(events_ms: np.ndarray, *, duration_ms: int = DURATION_MS) -> tuple[np.ndarray, int]:
    """Count native-decoded events in 1ms half-open bins; retain the endpoint."""
    events = np.asarray(events_ms)
    if events.ndim != 1 or events.dtype.kind != "f" or not np.isfinite(events).all():
        raise ValueError("FINITE_TEMPORAL_EVENT_DOMAIN_REQUIRED")
    decoded = encode_precise_times(events, 1000.)["decoded_times_ms"]
    if np.any(decoded < 0) or np.any(decoded > duration_ms):
        raise ValueError("FINITE_TEMPORAL_EVENT_CLOCK_REQUIRED")
    endpoint = int(np.count_nonzero(decoded == duration_ms))
    counts = np.bincount(np.floor(decoded[decoded < duration_ms]).astype(np.int64), minlength=duration_ms)
    if int(counts.sum()) + endpoint != len(events):
        raise ValueError("FINITE_TEMPORAL_EVENT_COUNT_DIFFERS")
    return counts.astype(np.int32), endpoint


def ensemble_covariance(values: np.ndarray, lag: int, *, ddof: int) -> np.ndarray:
    """Mean over time of per-time covariance centered across seed invocations.

    Input shape is invocation x unit x time. Unit output retains every unit.
    """
    values = np.asarray(values)
    if (values.ndim != 3 or values.dtype.kind not in "iu" or np.any(values < 0)
            or type(lag) is not int or not 0 <= lag < values.shape[2]
            or ddof not in (0, 1) or values.shape[0] <= ddof):
        raise ValueError("FINITE_TEMPORAL_ENSEMBLE_SHAPE_REQUIRED")
    a = values[:, :, :values.shape[2]-lag].astype(np.float64)
    b = values[:, :, lag:].astype(np.float64)
    a -= a.mean(axis=0); b -= b.mean(axis=0)
    return np.einsum("nut,nut->u", a, b) / ((values.shape[0]-ddof)*a.shape[2])


def temporal_moments(table: np.ndarray, lags=LAGS) -> dict:
    """Exact moments over the complete finite primary-seed support."""
    values = np.asarray(table)
    if values.ndim != 2 or values.shape[0] != SEEDS or values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError("FINITE_TEMPORAL_COMPLETE_PRIMARY_SEEDS_REQUIRED")
    mean = values.mean(axis=0)
    total = values.sum(axis=1, dtype=np.int64)
    return dict(mean_count=float(total.mean()), total_count_variance=float(total.var(ddof=0)),
        mean_time_count=mean, seed_ensemble_lag_covariance=np.asarray([
            ensemble_covariance(values[:, None, :], lag, ddof=0)[0] for lag in lags]),
        lag_ms=list(lags), finite_seed_count=SEEDS, ensemble_ddof=0)


def run(config_path: Path, output: Path) -> dict:
    if os.environ.get("SLURMD_NODENAME") != "cpu03" or not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("FINITE_TEMPORAL_CPU03_WITHOUT_GPU_REQUIRED")
    if output.exists():
        raise FileExistsError("FINITE_TEMPORAL_FRESH_OUTPUT_REQUIRED")
    config = yaml.safe_load(config_path.read_text())
    if (config.get("scope") != "full_gray_finite_temporal_law_development" or config.get("training_seeds_used") != []
            or config.get("bin_ms") != 1 or config.get("lag_ms") != list(LAGS)):
        raise ValueError("FINITE_TEMPORAL_DECLARED_DIAGNOSTIC_REQUIRED")
    prior = Path(config["finite_count_archive"])
    old, old_delivery = bound_json(prior, "analysis/report.json")
    if old_delivery.get("exit_code") != 0 or old.get("status") != "COMPLETE_FULL_GRAY_FINITE_PRIMARY_SEED_COUNT_LAW_NOT_QUALIFIED":
        raise ValueError("FINITE_TEMPORAL_ACCEPTED_COUNT_LAW_REQUIRED")
    count_path = prior / "analysis/finite_seed_counts.npz"
    row = next(x for x in old_delivery["files"] if x["path"] == "analysis/finite_seed_counts.npz")
    if sha256_file(count_path) != row["sha256"]:
        raise ValueError("FINITE_TEMPORAL_COUNT_LAW_CHANGED")
    with np.load(count_path, allow_pickle=False) as saved:
        previous = {key: saved[key] for key in saved.files}
    rates_root = Path(config["rate_archive"])
    rate, delivery = bound_json(rates_root, "probe/probe.json")
    rate_path = rates_root / "probe/firing_rates_f_tot.h5"
    if (delivery.get("exit_code") != 0 or sha256_file(rate_path) != old["rate_array_sha256"]
            or rate["source_unit_ids"] != previous["source_ids"].tolist()
            or rate["config"]["stimulus"]["contrast_percent"] != 0):
        raise ValueError("FINITE_TEMPORAL_BOUND_FULL_GRAY_RATE_REQUIRED")
    with h5py.File(rate_path, "r") as saved:
        clock, rates = saved["time"][:], saved["firing_rates_Hz"][:]
    if rates.shape != (17400, 3001):
        raise ValueError("FINITE_TEMPORAL_ALL_LGN_UNITS_REQUIRED")
    groups = previous["rate_class"]
    representatives = [int(np.flatnonzero(groups == g)[0]) for g in range(14)]
    for group, first in enumerate(representatives):
        if not np.array_equal(rates[groups == group], np.broadcast_to(rates[first], rates[groups == group].shape)):
            raise ValueError("FINITE_TEMPORAL_EXACT_RATE_CLASS_DIFFERS")
    if source_count_moments(clock, rates[representatives])["poisson"].any():
        raise ValueError("FINITE_TEMPORAL_PRIMARY_BRANCH_REQUIRED")
    starts = previous["starting_noise_seeds"]
    replayed = np.asarray([np.random.RandomState(int(seed)).randint(SEEDS, size=17400) for seed in starts])
    if not np.array_equal(replayed, previous["primary_seeds"]):
        raise ValueError("FINITE_TEMPORAL_STARTING_SEED_VECTOR_DIFFERS")
    output.mkdir(parents=True)
    report = dict(status="FAILED", coupled_cortical_episodes=0, production_episodes=0, scientific_qualification=False,
        bin_ms=1, lag_ms=list(LAGS), sampler_calls=0, full_lgn_units=17400, primary_seeds_per_class=SEEDS,
        rate_classes=14, starting_seed_realizations=100, independent_seed_reference_is_hypothesis=True,
        seed_ensemble_covariance_centering="center across invocations at each time, then mean across overlapping times",
        source_count_archive_sha256=sha256_file(count_path), source_rate_sha256=sha256_file(rate_path))
    before_numpy, before_python = np.random.get_state(), random.getstate()
    started = time.monotonic()
    try:
        if probe(output / "resource_preflight.json", required_memory_gib=4)["status"] != "PASS":
            raise RuntimeError("FINITE_TEMPORAL_RESOURCE_STOP")
        functions, _ = load_prepared_runtime(Path(config["prepared_runtime"]))
        means, reference_lags, totals = [], [], []
        population_counts = np.zeros((100, 14, DURATION_MS), dtype=np.int64)
        summed_unit_covariance = np.zeros((14, len(LAGS)))
        endpoints = np.zeros((100, 17400), dtype=np.int32)
        time_ms = clock * 1000.
        with h5py.File(output / "finite_temporal_counts.h5", "x") as store:
            store.attrs.update(bin_ms=1, endpoint_policy="3000ms retained separately", primary_seed_count=SEEDS)
            for group, representative in enumerate(representatives):
                table = np.empty((SEEDS, DURATION_MS), dtype=np.int32)
                endpoint_table = np.empty(SEEDS, dtype=np.int32)
                for seed in range(SEEDS):
                    events = np.asarray(functions.f_rate_to_spike_train(time_ms, rates[representative], seed, time_ms[0], time_ms[-1], .1), dtype=np.float64)
                    table[seed], endpoint_table[seed] = temporal_counts(events)
                    report["sampler_calls"] += 1
                    if int(table[seed].sum()) + int(endpoint_table[seed]) != previous["class_counts"][group, seed]:
                        raise ValueError("FINITE_TEMPORAL_PRIOR_COUNT_LAW_DIFFERS")
                native = store.create_group("class_" + str(group).zfill(2))
                native.create_dataset("counts", data=table, compression="gzip", shuffle=True, chunks=(32, DURATION_MS))
                native.create_dataset("terminal_counts", data=endpoint_table, compression="gzip")
                moment = temporal_moments(table)
                means.append(moment["mean_time_count"]); reference_lags.append(moment["seed_ensemble_lag_covariance"])
                totals.append(dict(rate_class=group, mean_count=moment["mean_count"], total_count_variance=moment["total_count_variance"]))
                positions = np.flatnonzero(groups == group)
                for offset in range(0, len(positions), 16):
                    selected = positions[offset:offset+16]
                    primary_seeds = previous["primary_seeds"][:, selected]
                    values = table[primary_seeds]
                    population_counts[:, group] += values.sum(axis=1, dtype=np.int64)
                    endpoints[:, selected] = endpoint_table[primary_seeds]
                    for index, lag in enumerate(LAGS):
                        summed_unit_covariance[group, index] += ensemble_covariance(values, lag, ddof=1).sum()
                print(json.dumps(dict(rate_class_complete=group, sampler_calls=report["sampler_calls"])), flush=True)
                if group % 4 == 3 and probe(output / "resource_latest.json", required_memory_gib=2)["status"] != "PASS":
                    raise RuntimeError("FINITE_TEMPORAL_RESOURCE_STOP")
        sizes = np.bincount(groups, minlength=14)
        reference_lags = np.asarray(reference_lags)
        pair_covariances, population_covariances = [], []
        pairs = sizes[:, None]*sizes[None, :]; pairs[np.diag_indices(14)] -= sizes
        for index, lag in enumerate(LAGS):
            a = population_counts[:, :, :DURATION_MS-lag].astype(float)
            b = population_counts[:, :, lag:].astype(float)
            a -= a.mean(axis=0); b -= b.mean(axis=0)
            covariance = np.einsum("ngt,nht->gh", a, b) / (99*(DURATION_MS-lag))
            distinct = covariance.copy(); distinct[np.diag_indices(14)] -= summed_unit_covariance[:, index]
            pair_covariances.append(np.divide(distinct, pairs, out=np.full((14,14), np.nan), where=pairs != 0))
            population_covariances.append(covariance)
        reconstructed = population_counts.sum(axis=2)
        expected = previous["source_model_total_counts"]
        endpoint_population = np.column_stack([endpoints[:, groups == g].sum(axis=1) for g in range(14)])
        if not np.array_equal(reconstructed + endpoint_population, expected):
            raise ValueError("FINITE_TEMPORAL_FULL_POPULATION_COUNT_READBACK_DIFFERS")
        np.savez_compressed(output / "temporal_summary.npz", class_mean_time_count=np.asarray(means),
            class_seed_ensemble_lag_covariance=reference_lags, rate_class=groups, source_ids=previous["source_ids"],
            starting_noise_seeds=starts, primary_seeds=previous["primary_seeds"], population_counts=population_counts,
            terminal_counts=endpoints, population_covariance=np.asarray(population_covariances),
            distinct_unit_mean_covariance=np.asarray(pair_covariances), summed_unit_covariance=summed_unit_covariance,
            ordered_distinct_unit_pairs=pairs, class_population_sizes=sizes, lag_ms=np.asarray(LAGS))
        report.update(status="COMPLETE_FULL_GRAY_FINITE_TEMPORAL_LAW_NOT_QUALIFIED", count_law_reconciled=True,
            full_starting_seed_vectors_exact=True, all_population_temporal_totals_exact=True,
            finite_class_total_moments=totals,
            independent_primary_seed_reference=dict(distinct_unit_covariance=0,
                population_seed_ensemble_lag_covariance=[np.diag(sizes*reference_lags[:,i]).tolist() for i in range(len(LAGS))]),
            observed_population_seed_ensemble_covariance=np.asarray(population_covariances).tolist(),
            full_event_sequences_saved=False, temporal_count_realizations_are_cortical_episodes=False,
            joint_law_qualified=False, scientific_equivalence="NOT_EVALUATED_NO_FROZEN_TOLERANCE",
            interpretation="Exact finite primary-seed temporal count law and realized starting-seed population moments. Independent primary-seed reference is a declared hypothesis, not a conclusion. No biological or cortical fidelity claim.")
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        np.random.set_state(before_numpy); random.setstate(before_python)
        report.update(elapsed_seconds=time.monotonic()-started, recorded_utc=datetime.now(timezone.utc).isoformat(),
            artifacts={p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p)) for p in output.iterdir() if p.is_file() and p.name != "report.json"})
        freeze_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.config, args.output)
    print(json.dumps({k:result[k] for k in ("status", "sampler_calls", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
