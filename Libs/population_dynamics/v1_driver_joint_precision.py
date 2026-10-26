"""Invocation-level precision of full-gray finite-seed driver comparisons.

The finite independent-primary-seed law is a hypothesis. Bootstrap resamples
complete starting-seed invocations jointly across populations, preserving each
invocation's complete temporal record. Conditional descriptive intervals do not
establish invocation independence, equivalence, or biological replication.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix
import yaml

from .provenance import freeze_json, sha256_file

EXPERIMENT_SEEDS = (41, 42, 43)


def comparison_scores(population_counts: np.ndarray, class_mean_time: np.ndarray,
                      class_lag_covariance: np.ndarray, class_sizes: np.ndarray,
                      class_total_variance: np.ndarray, lag_ms: np.ndarray) -> dict:
    values, means, lags, sizes, variance, lag_values = map(np.asarray,
        (population_counts,class_mean_time,class_lag_covariance,class_sizes,class_total_variance,lag_ms))
    if (values.ndim != 3 or values.dtype.kind not in "iu" or np.any(values < 0)
            or values.shape[0] < 2 or means.shape != values.shape[1:]
            or means.dtype.kind != "f" or not np.isfinite(means).all() or np.any(means < 0)
            or sizes.shape != (values.shape[1],) or sizes.dtype.kind not in "iu" or np.any(sizes < 1)
            or variance.shape != sizes.shape or variance.dtype.kind != "f" or not np.isfinite(variance).all() or np.any(variance <= 0)
            or lags.shape != (values.shape[1],len(lag_values)) or not np.isfinite(lags).all()
            or lag_values.dtype.kind not in "iu" or np.any(lag_values < 0) or np.any(lag_values >= values.shape[2])
            or not np.any(lag_values == 0)):
        raise ValueError("V1_JOINT_PRECISION_BOUND_ARRAYS_REQUIRED")
    zero = int(np.flatnonzero(lag_values == 0)[0])
    scale_total = sizes * variance
    exact_mean = means * sizes[:,None]
    residual = values.astype(float)-exact_mean[None,:,:]
    total_residual = residual.sum(axis=2)
    standardized_total = total_residual / np.sqrt(scale_total)[None,:]
    total_population_scores = standardized_total[:,:,None]*standardized_total[:,None,:]
    scale_time = sizes * lags[:,zero]
    if np.any(scale_time <= 0):
        raise ValueError("V1_JOINT_PRECISION_NONZERO_TEMPORAL_VARIANCE_REQUIRED")
    temporal_scores = []
    reference_temporal = []
    for index,lag in enumerate(lag_values):
        a,b=residual[:,:,:values.shape[2]-lag],residual[:,:,lag:]
        normalizer=np.sqrt(scale_time[:,None]*scale_time[None,:])
        temporal_scores.append(np.einsum("ngt,nht->ngh",a,b)/((values.shape[2]-lag)*normalizer[None,:,:]))
        reference_temporal.append(np.diag(sizes*lags[:,index]/scale_time))
    # The whole-population normalization includes every class and every unit.
    pooled = total_residual.sum(axis=1) / np.sqrt(scale_total.sum())
    return dict(mean_scores=standardized_total,total_covariance_scores=total_population_scores,
        total_covariance_reference=np.eye(values.shape[1]),
        temporal_covariance_scores=np.stack(temporal_scores,axis=1),
        temporal_covariance_reference=np.asarray(reference_temporal),
        pooled_mean_scores=pooled,pooled_second_moment_scores=pooled*pooled,
        scale_total=scale_total,scale_time=scale_time)


def joint_bootstrap(scores: np.ndarray, *, draws: int, seed: int) -> dict:
    scores=np.asarray(scores)
    if scores.ndim < 1 or scores.shape[0] < 2 or not np.isfinite(scores).all() or type(draws) is not int or draws < 100 or type(seed) is not int or seed not in EXPERIMENT_SEEDS:
        raise ValueError("V1_JOINT_PRECISION_BOOTSTRAP_INPUT_REQUIRED")
    rng=np.random.default_rng(seed)
    result=np.empty((draws,*scores.shape[1:]),dtype=float)
    for start in range(0,draws,100):
        # One multinomial weight vector resamples a complete invocation. The
        # same vector applies to every population and lag in that invocation.
        weights=rng.multinomial(scores.shape[0],np.full(scores.shape[0],1/scores.shape[0]),size=min(100,draws-start))
        result[start:start+len(weights)]=np.tensordot(weights,scores,axes=(1,0))/scores.shape[0]
    return dict(estimate=scores.mean(axis=0),lower=np.quantile(result,.025,axis=0),upper=np.quantile(result,.975,axis=0),
        draws=draws,bootstrap_seed=seed,unit="complete starting-seed invocation",method="pointwise percentile empirical bootstrap")


def require_experiment_seeds(config: dict) -> list[int]:
    seeds = config.get("bootstrap_seeds")
    if seeds != list(EXPERIMENT_SEEDS) or any(type(seed) is not int for seed in seeds):
        raise ValueError("V1_JOINT_PRECISION_EXPERIMENT_SEEDS_MUST_BE_41_42_43")
    if "bootstrap_seed" in config:
        raise ValueError("V1_JOINT_PRECISION_SINGLE_EXPERIMENT_SEED_NOT_ALLOWED")
    return seeds


def finite_population_counts(table: np.ndarray, primary_seeds: np.ndarray) -> np.ndarray:
    """Sum finite count paths with exact integer seed multiplicities."""
    table, seeds = np.asarray(table), np.asarray(primary_seeds)
    if (table.ndim != 2 or seeds.ndim != 2 or not all(table.shape) or not all(seeds.shape)
            or table.dtype.kind not in "iu" or seeds.dtype.kind not in "iu"
            or np.any(table < 0) or np.any(seeds < 0) or np.any(seeds >= table.shape[0])):
        raise ValueError("V1_JOINT_PRECISION_FINITE_COUNT_INPUT_REQUIRED")
    rows = np.repeat(np.arange(seeds.shape[0]), seeds.shape[1])
    multiplicities = csr_matrix((np.ones(seeds.size, dtype=np.int64), (rows, seeds.ravel())),
                                shape=(seeds.shape[0], table.shape[0]))
    return (multiplicities @ csr_matrix(table).astype(np.int64)).toarray()


def replay_starting_seeds(selection: dict, previous: np.ndarray) -> np.ndarray:
    if not isinstance(selection, dict) or set(selection) != {"first_starting_noise_seed", "invocations"}:
        raise ValueError("V1_JOINT_PRECISION_REPLAY_SELECTION_REQUIRED")
    first, count = selection["first_starting_noise_seed"], selection["invocations"]
    if (type(first) is not int or type(count) is not int or first < 0 or count < 2
            or first + count > 2**32):
        raise ValueError("V1_JOINT_PRECISION_REPLAY_SEED_RANGE_INVALID")
    starts = np.arange(first, first+count, dtype=np.int64)
    if np.intersect1d(starts, previous).size:
        raise ValueError("V1_JOINT_PRECISION_REPLAY_SEEDS_OVERLAP_HISTORY")
    return starts


def replay_population(source: Path, analysis: dict, saved: dict, selection: dict) -> dict:
    """Use the accepted finite paths without repeating spike-sampler calls."""
    starts = replay_starting_seeds(selection, saved["starting_noise_seeds"])
    groups, sizes = saved["rate_class"], saved["class_population_sizes"]
    old_seeds = saved["primary_seeds"]
    if (len(groups) != 17400 or len(sizes) != 14 or old_seeds.shape != (len(saved["starting_noise_seeds"]),17400)
            or not np.array_equal(np.bincount(groups, minlength=14), sizes)):
        raise ValueError("V1_JOINT_PRECISION_FULL_LGN_REPLAY_REQUIRED")
    table_path = source/"analysis/finite_temporal_counts.h5"
    if sha256_file(table_path) != analysis["artifacts"]["finite_temporal_counts.h5"]["sha256"]:
        raise ValueError("V1_JOINT_PRECISION_FINITE_TABLE_CHANGED")
    seeds = np.asarray([np.random.RandomState(int(seed)).randint(10000,size=17400)
                        for seed in starts], dtype=np.int32)
    population = np.zeros((len(starts),14,3000),dtype=np.int64)
    endpoints = np.zeros((len(starts),14),dtype=np.int64)
    checked = 0
    with h5py.File(table_path,"r") as store:
        for group in range(14):
            table = store[f"class_{group:02d}/counts"][:]
            terminal = store[f"class_{group:02d}/terminal_counts"][:]
            if table.shape != (10000,3000) or terminal.shape != (10000,):
                raise ValueError("V1_JOINT_PRECISION_COMPLETE_FINITE_TABLE_REQUIRED")
            selected = groups == group
            historical = finite_population_counts(table, old_seeds[:,selected])
            if (not np.array_equal(historical,saved["population_counts"][:,group])
                    or not np.array_equal(terminal[old_seeds[:,selected]],saved["terminal_counts"][:,selected])):
                raise ValueError("V1_JOINT_PRECISION_HISTORICAL_TEMPORAL_REPLAY_DIFFERS")
            checked += historical.size
            population[:,group] = finite_population_counts(table,seeds[:,selected])
            endpoints[:,group] = terminal[seeds[:,selected]].sum(axis=1,dtype=np.int64)
    return dict(starting_noise_seeds=starts,primary_seeds=seeds,population_counts=population,
                terminal_population_counts=endpoints,historical_population_values_verified=checked,
                finite_table_sha256=sha256_file(table_path))


def run(config_path: Path, output: Path) -> dict:
    if not os.environ.get("SLURM_JOB_ID") or output.exists():
        raise ValueError("V1_JOINT_PRECISION_FRESH_SLURM_OUTPUT_REQUIRED")
    config=yaml.safe_load(config_path.read_text())
    if config.get("scope") != "full_gray_joint_driver_precision_development" or config.get("training_seeds_used") != []:
        raise ValueError("V1_JOINT_PRECISION_DECLARED_DEVELOPMENT_REQUIRED")
    experiment_seeds = require_experiment_seeds(config)
    source=Path(config["temporal_archive"]);acceptance=Path(config["temporal_acceptance"])
    receipt=json.loads(acceptance.read_text());analysis=json.loads((source/"analysis/report.json").read_text())
    row=json.loads((source/"report.json").read_text())
    if (receipt["status"] != "PASS_FINITE_GRAY_TEMPORAL_LAW_AND_FULL_SOURCE_READBACK"
            or receipt["exit_code"] != 0 or receipt["source_report_sha256"] != sha256_file(source/"report.json")
            or row["exit_code"] != 0 or analysis["status"] != "COMPLETE_FULL_GRAY_FINITE_TEMPORAL_LAW_NOT_QUALIFIED"):
        raise ValueError("V1_JOINT_PRECISION_ACCEPTED_TEMPORAL_LAW_REQUIRED")
    if row["output_artifacts"]["analysis/report.json"]["sha256"] != sha256_file(source/"analysis/report.json"):
        raise ValueError("V1_JOINT_PRECISION_BOUND_ANALYSIS_CHANGED")
    path=source/"analysis/temporal_summary.npz"
    if sha256_file(path) != analysis["artifacts"]["temporal_summary.npz"]["sha256"]:
        raise ValueError("V1_JOINT_PRECISION_TEMPORAL_ARRAY_CHANGED")
    with np.load(path,allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    population=saved["population_counts"];means=saved["class_mean_time_count"];lags=saved["class_seed_ensemble_lag_covariance"]
    sizes=saved["class_population_sizes"];lag_values=saved["lag_ms"];starts=saved["starting_noise_seeds"]
    replay = None
    if "population_replay" in config:
        if os.environ.get("SLURMD_NODENAME") != "cpu03":
            raise ValueError("V1_JOINT_PRECISION_POPULATION_REPLAY_REQUIRES_CPU03")
        replay = replay_population(source,analysis,saved,config["population_replay"])
        population, starts = replay["population_counts"], replay["starting_noise_seeds"]
    variance=np.asarray([row["total_count_variance"] for row in analysis["finite_class_total_moments"]])
    scores=comparison_scores(population,means,lags,sizes,variance,lag_values)
    output.mkdir(parents=True)
    replay_receipt = None
    if replay is not None:
        artifact = output/"population_replay.npz"
        np.savez_compressed(artifact, **{key: replay[key] for key in
            ("starting_noise_seeds","primary_seeds","population_counts","terminal_population_counts")},
            source_ids=saved["source_ids"],rate_class=saved["rate_class"],class_population_sizes=sizes)
        replay_receipt = dict(selection=config["population_replay"],
            historical_population_values_verified=replay["historical_population_values_verified"],
            finite_table_sha256=replay["finite_table_sha256"],artifact=artifact.name,
            artifact_sha256=sha256_file(artifact),source_primary_seed_vector_calls=len(starts),
            source_spike_sampler_calls=0,background_generator_included=False)
    per_seed = []
    for experiment_seed in experiment_seeds:
        tables = {name:joint_bootstrap(scores[name],draws=config["bootstrap_draws"],seed=experiment_seed)
                  for name in ("mean_scores","total_covariance_scores","temporal_covariance_scores","pooled_mean_scores","pooled_second_moment_scores")}
        arrays = {name+"_"+key:table[key] for name,table in tables.items() for key in ("estimate","lower","upper")}
        arrays.update(total_covariance_reference=scores["total_covariance_reference"],temporal_covariance_reference=scores["temporal_covariance_reference"],
            total_standardization_variance=scores["scale_total"],temporal_standardization_variance=scores["scale_time"],starting_noise_seeds=starts,lag_ms=lag_values)
        artifact = output / ("precision_seed_"+str(experiment_seed)+".npz")
        np.savez_compressed(artifact,**arrays)
        pooled=tables["pooled_second_moment_scores"]
        widths={name:dict(maximum=float(np.max(table["upper"]-table["lower"])),median=float(np.median(table["upper"]-table["lower"]))) for name,table in tables.items()}
        per_seed.append(dict(bootstrap_seed=experiment_seed,pooled_standardized_second_moment=float(pooled["estimate"]),
            pooled_second_moment_pointwise_interval=[float(pooled["lower"]),float(pooled["upper"])],pointwise_interval_widths=widths,
            artifact=artifact.name,artifact_sha256=sha256_file(artifact)))
    report=dict(status="COMPLETE_DESCRIPTIVE_FULL_GRAY_DRIVER_JOINT_PRECISION",starting_seed_invocations=len(starts),
        full_lgn_units=int(sizes.sum()),rate_classes=len(sizes),bin_ms=1,lag_ms=lag_values.tolist(),bootstrap_draws=config["bootstrap_draws"],
        bootstrap_seeds=experiment_seeds,seed_results=per_seed,
        pooled_standardized_second_moment=float(scores["pooled_second_moment_scores"].mean()),
        pooled_second_moment_independent_primary_seed_reference=1.0,
        maximum_absolute_total_covariance_reference_deviation=float(np.max(np.abs(scores["total_covariance_scores"].mean(axis=0)-scores["total_covariance_reference"]))),
        maximum_absolute_temporal_covariance_reference_deviation=float(np.max(np.abs(scores["temporal_covariance_scores"].mean(axis=0)-scores["temporal_covariance_reference"]))),
        scientific_equivalence="NOT_EVALUATED_NO_FROZEN_TOLERANCE",intervals_are_simultaneous=False,
        bootstrap_seeds_are_independent_source_invocations=False,
        invocation_independence_established=False,scientific_qualification=False,source_sampler_calls=0,cortical_simulations=0,production_episodes=0,
        source_temporal_array_sha256=sha256_file(path),recorded_utc=datetime.now(timezone.utc).isoformat(),
        population_replay=replay_receipt,
        interpretation=f"Three bootstrap seeds resample the same {len(starts)} LGN starting-seed invocations. Bootstrap repeats are not new source replicates. Pointwise conditional intervals do not establish independence, equivalence or full joint distribution. The shared background generator is outside this comparison.")
    freeze_json(output/"report.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","output"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args(argv);report=run(args.config,args.output)
    print(json.dumps(report))


if __name__=="__main__":main()
