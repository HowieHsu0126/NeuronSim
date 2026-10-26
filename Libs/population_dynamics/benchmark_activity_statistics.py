"""Describe joint activity in accepted mechanistic benchmark episodes.

Covariance uses overlapping time bins from one episode. These bins are not
independent trials. Every unit remains in the denominator, including silence.
Signed mean pair covariance does not characterize the full covariance matrix.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix

from .benchmark_dataset import read_circuit, read_episode
from .circuit_campaign import campaign_config
from .circuit_data import CircuitDataset
from .provenance import freeze_json, runtime_manifest, sha256_file
from .v1_benchmark_readout import CONDITIONS, describe_episode

SCENARIO_UNITS = {"v1_visual": 230924, "ca3_pattern_completion": 168106}
CA3_POST_CUE_STARTS = (540, 770, 1000)
CA3_POST_CUE_STOPS = (750, 980)


def event_statistics(ticks, event_units, populations, *, duration_ticks: int,
                     start_tick: int, stop_tick: int, bin_ticks: int,
                     lag_bins: tuple[int, ...]) -> tuple[dict, dict]:
    """Compute all-unit temporal moments from a sparse unit-by-bin count matrix.

    At lag l, each side is centered on its own overlapping interval. Covariance
    divides by overlapping_bins-1. Population entries use total counts, not
    mean rates. Mean distinct-unit covariance includes all ordered unit pairs.
    """
    ticks, event_units, populations = map(np.asarray, (ticks, event_units, populations))
    n = len(populations)
    if (type(duration_ticks) is not int or duration_ticks < 1 or type(bin_ticks) is not int
            or bin_ticks < 1 or type(start_tick) is not int or type(stop_tick) is not int
            or not 0 <= start_tick < stop_tick <= duration_ticks
            or (stop_tick-start_tick) % bin_ticks or ticks.ndim != 1
            or ticks.dtype.kind not in "iu" or event_units.shape != ticks.shape
            or event_units.dtype.kind not in "iu" or populations.ndim != 1
            or populations.dtype.kind not in "US" or n < 2
            or (len(ticks) and (ticks.min() < 0 or ticks.max() > duration_ticks
                               or event_units.min() < 0 or event_units.max() >= n))
            or any(type(lag) is not int or lag < 0 for lag in lag_bins)
            or len(set(lag_bins)) != len(lag_bins) or not lag_bins):
        raise ValueError("BENCHMARK_ACTIVITY_EVENT_CLOCK_UNIT_OR_LAG_INVALID")
    # The complete source unit inventory defines population order. Sorting
    # labels would assign different indices to the native CA3 populations.
    names = np.asarray(tuple(dict.fromkeys(populations.tolist())))
    if any(not name for name in names):
        raise ValueError("BENCHMARK_ACTIVITY_SOURCE_POPULATION_LABEL_REQUIRED")
    mapping = {name: index for index, name in enumerate(names)}
    codes = np.asarray([mapping[name] for name in populations], dtype=np.int64)
    bins = (stop_tick-start_tick)//bin_ticks
    keep = (ticks >= start_tick) & (ticks < stop_tick)
    units = event_units[keep].astype(np.int64)
    clock = (ticks[keep].astype(np.int64)-start_tick)//bin_ticks
    counts = coo_matrix((np.ones(len(units), dtype=np.int64), (units, clock)),
                        shape=(n, bins)).tocsr()
    counts.sum_duplicates()
    totals = np.asarray(counts.sum(axis=0)).ravel()
    unit_totals = np.asarray(counts.sum(axis=1)).ravel()
    pop_counts = np.zeros((bins, len(names)), dtype=np.int64)
    np.add.at(pop_counts, (clock, codes[units]), 1)
    if (int(unit_totals.sum()) != len(units) or not np.array_equal(totals, pop_counts.sum(axis=1))):
        raise ValueError("BENCHMARK_ACTIVITY_EVENT_COUNT_RECONCILIATION_FAILED")
    unit_covariances = np.full((len(lag_bins), n), np.nan)
    population_covariances = np.full((len(lag_bins), len(names), len(names)), np.nan)
    lag_rows = []
    for index, lag in enumerate(lag_bins):
        overlap = bins-lag
        row = dict(lag_bins=lag, overlapping_bins=max(0, overlap),
                   temporal_count_covariance=None, mean_unit_count_autocovariance=None,
                   mean_distinct_unit_count_covariance=None)
        if overlap >= 2:
            left, right = counts[:, :overlap], counts[:, lag:lag+overlap]
            sum_left = np.asarray(left.sum(axis=1)).ravel().astype(np.float64)
            sum_right = np.asarray(right.sum(axis=1)).ravel().astype(np.float64)
            products = np.asarray(left.multiply(right).sum(axis=1)).ravel()
            unit_covariance = (products-sum_left*sum_right/overlap)/(overlap-1)
            aggregate_left = totals[:overlap].astype(np.float64)
            aggregate_right = totals[lag:lag+overlap].astype(np.float64)
            aggregate = float((aggregate_left @ aggregate_right
                               -aggregate_left.sum()*aggregate_right.sum()/overlap)/(overlap-1))
            diagonal_sum = float(unit_covariance.sum())
            a, b = pop_counts[:overlap].astype(np.float64), pop_counts[lag:lag+overlap].astype(np.float64)
            population_covariance = (a.T @ b-np.outer(a.sum(axis=0), b.sum(axis=0))/overlap)/(overlap-1)
            if not np.isclose(population_covariance.sum(), aggregate, rtol=1e-10, atol=1e-8):
                raise ValueError("BENCHMARK_ACTIVITY_POPULATION_COVARIANCE_RECONCILIATION_FAILED")
            unit_covariances[index] = unit_covariance
            population_covariances[index] = population_covariance
            row.update(temporal_count_covariance=aggregate,
                       mean_unit_count_autocovariance=diagonal_sum/n,
                       mean_distinct_unit_count_covariance=(aggregate-diagonal_sum)/(n*(n-1)))
        lag_rows.append(row)
    arrays = dict(unit_totals=unit_totals, unit_mean_count=unit_totals/bins,
                  population_index=codes, population_names=names,
                  population_unit_counts=np.bincount(codes), population_counts=pop_counts,
                  total_counts=totals, lag_bins=np.asarray(lag_bins),
                  unit_count_autocovariance=unit_covariances,
                  population_count_covariance=population_covariances)
    summary = dict(units=n, bins=bins, start_tick=start_tick, stop_tick_exclusive=stop_tick,
                   bin_ticks=bin_ticks, events=int(unit_totals.sum()),
                   silent_units=int(np.count_nonzero(unit_totals==0)),
                   ordered_distinct_unit_pairs=n*(n-1), lags=lag_rows,
                   covariance_units="spike_count_squared", covariance_ddof=1,
                   population_names=names.tolist(),
                   population_index_semantics="First occurrence in complete source unit inventory",
                   lag_centering="Separate means of the two overlapping intervals",
                   interpretation="Within-episode temporal descriptions; no trial independence, noise covariance, stationarity, significance or full-distribution claim.")
    return summary, arrays


def windows(scenario: str, duration_ticks: int) -> list[tuple[str, int, int]]:
    if scenario == "v1_visual":
        return [("full",0,12000),("pre_command",0,2004),("command_response",2004,12000)]
    if scenario == "ca3_pattern_completion":
        return [("full",0,duration_ticks),("pre_cue",0,520)] + [
            (f"post_cue_{index+1}",start,stop) for index,(start,stop) in enumerate(
                zip(CA3_POST_CUE_STARTS, (*CA3_POST_CUE_STOPS,duration_ticks)))]
    raise ValueError("BENCHMARK_ACTIVITY_SCENARIO_INVALID")


def accepted_episode(entry: dict, project: Path):
    def resolve(value):
        path=Path(value)
        return path if path.is_absolute() else project/path
    acceptance_path, source = resolve(entry["acceptance"]), resolve(entry["source"])
    if sha256_file(acceptance_path) != entry["acceptance_sha256"]:
        raise ValueError("BENCHMARK_ACTIVITY_ACCEPTANCE_CHANGED")
    accepted=json.loads(acceptance_path.read_text())
    if sha256_file(source/"dataset.json") != entry["catalog_sha256"]:
        raise ValueError("BENCHMARK_ACTIVITY_SOURCE_CATALOG_CHANGED")
    dataset=CircuitDataset(source,splits=("development",))
    scenario=entry["scenario_id"]
    if (scenario not in SCENARIO_UNITS or dataset.catalog["complete_episodes"] != 1
            or len(dataset.episodes) != 1 or dataset.catalog["binding"]["scenario_id"] != scenario
            or dataset.catalog["binding"]["scope"] != "source_reproduction"
            or dataset.catalog["source_fidelity_qualified"] is not False):
        raise ValueError("BENCHMARK_ACTIVITY_ONE_ACCEPTED_DEVELOPMENT_EPISODE_REQUIRED")
    episode=read_episode(dataset,next(dataset.iter_episodes()))
    circuit=read_circuit(dataset,episode.circuit_id)
    episode.validate(len(circuit.unit_ids))
    if len(circuit.unit_ids) != SCENARIO_UNITS[scenario]:
        raise ValueError("BENCHMARK_ACTIVITY_COMPLETE_CORE_POPULATION_REQUIRED")
    if scenario == "v1_visual":
        if (accepted["status"] != "PASS_AUTHENTICATED_COMPLETE_V1_DEVELOPMENT_BLOCK"
                or accepted["exit_code"] != 0 or accepted["complete_development_episodes"] != 10
                or accepted["all_unit_native_export_readout_counts_exact"] is not True):
            raise ValueError("BENCHMARK_ACTIVITY_V1_COMPLETE_BLOCK_ACCEPTANCE_REQUIRED")
        index=entry["condition_index"]
        row=accepted["episodes"][index]
        if row["native_job"] != entry["native_job"] or row["export_job"] != entry["export_job"]:
            raise ValueError("BENCHMARK_ACTIVITY_V1_SOURCE_JOB_DIFFERS")
        producer_path=source.parent.parent/"report.json"
        producer=json.loads(producer_path.read_text())
        if (sha256_file(producer_path) != entry["producer_report_sha256"]
                or producer["status"] != "COMPLETE_V1_BENCHMARK_DEVELOPMENT_EPISODE"
                or producer["actual_split"] != "development" or producer["complete_episodes"] != 1
                or producer["dataset_catalog_sha256"] != entry["catalog_sha256"]
                or producer["candidate_identity"]["condition_index"] != index
                or producer["candidate_identity"]["circuit_index"] != accepted["circuit_index"]
                or producer["candidate_identity"]["block_index"] != accepted["block_index"]
                or Path(producer["source_archives"]["native"]).name != "run_"+str(entry["native_job"])
                or dataset.catalog["binding"]["source_bindings"][0]["job_id"] != str(entry["native_job"])):
            raise ValueError("BENCHMARK_ACTIVITY_V1_PRODUCER_BINDING_DIFFERS")
        data=describe_episode(episode,circuit)
        if data["condition_id"] != CONDITIONS[index] or len(episode.spike_ticks) != row["events"]:
            raise ValueError("BENCHMARK_ACTIVITY_V1_CONDITION_OR_COUNT_DIFFERS")
        condition=data["condition_id"]
    else:
        if (accepted["status"] != "COMPLETE_CA3_BENCHMARK_DEVELOPMENT_EPISODE"
                or accepted["complete_episodes"] != 1 or accepted["actual_split"] != "development"
                or accepted["native_terminal_footer_verified"] is not True
                or accepted["all_unit_event_counter_equality"] is not True
                or accepted["unit_millisecond_counts_compared"] != len(circuit.unit_ids)*episode.duration_ticks
                or episode.dt_ms != 1.0 or episode.duration_ticks not in (1001,1210)
                or len(episode.spike_ticks) != accepted["events"]
                or episode.source_record["formation_binding"] != accepted["formed_circuit"]
                or episode.source_record["condition"] != accepted["condition"]
                or accepted["candidate_identity"]["condition_index"] != entry["condition_index"]
                or accepted["production_episodes"] != 0
                or any(episode.seeds[name] != accepted["candidate_identity"][key] for name,key in (
                    ("uniform_seed","background_seed"),("engine_seed","initial_state_seed"),
                    ("cue_seed","stimulus_seed")))):
            raise ValueError("BENCHMARK_ACTIVITY_CA3_COMPLETE_EPISODE_ACCEPTANCE_REQUIRED")
        condition=accepted["condition"]["condition_id"]
    return episode,circuit,condition


def analyze(config_path: Path, inputs_path: Path, project: Path, output: Path,
            *, bin_ms: float, lags_ms: tuple[float,...]) -> dict:
    campaign=campaign_config(config_path)
    if (bin_ms <= 0 or not np.isfinite(bin_ms) or not lags_ms or len(set(lags_ms)) != len(lags_ms)
            or any(not np.isfinite(x) or x < 0 for x in lags_ms)):
        raise ValueError("BENCHMARK_ACTIVITY_BIN_AND_LAGS_REQUIRED")
    inputs=json.loads(inputs_path.read_text())
    if not inputs["episodes"] or output.exists():
        raise ValueError("BENCHMARK_ACTIVITY_FRESH_OUTPUT_AND_ACCEPTED_INPUTS_REQUIRED")
    output.mkdir(parents=True)
    report=dict(status="FAILED",episodes=[],production_episodes=0,new_source_simulations=0,
                scientific_qualification=False,inferential_statistics=False,diagnostic_bin_ms=bin_ms,
                scientific_bins_frozen=False,input_specification_sha256=sha256_file(inputs_path))
    started=time.monotonic()
    try:
        identities=set()
        for entry in inputs["episodes"]:
            episode,circuit,condition=accepted_episode(entry,project)
            identity=(entry["scenario_id"],episode.episode_id)
            if identity in identities:
                raise ValueError("BENCHMARK_ACTIVITY_DUPLICATE_SOURCE_EPISODE")
            identities.add(identity)
            bin_ticks=round(bin_ms/episode.dt_ms)
            lag_bins=tuple(round(lag/bin_ms) for lag in lags_ms)
            if (bin_ticks < 1 or bin_ticks*episode.dt_ms != bin_ms
                    or any(value*bin_ms != lag for value,lag in zip(lag_bins,lags_ms))):
                raise ValueError("BENCHMARK_ACTIVITY_EXACT_NATIVE_BIN_AND_LAGS_REQUIRED")
            case=output/entry["scenario_id"]/episode.episode_id
            case.mkdir(parents=True)
            summaries=[]
            for label,start,stop in windows(entry["scenario_id"],episode.duration_ticks):
                summary,arrays=event_statistics(episode.spike_ticks,episode.spike_unit_indices,circuit.populations,
                    duration_ticks=episode.duration_ticks,start_tick=start,stop_tick=stop,
                    bin_ticks=bin_ticks,lag_bins=lag_bins)
                arrays["unit_ids"]=circuit.unit_ids
                for lag in summary["lags"]:
                    lag["lag_ms"]=lag["lag_bins"]*bin_ms
                artifact=case/(label+".npz")
                np.savez_compressed(artifact,**arrays)
                summary.update(window=label,start_ms=start*episode.dt_ms,stop_ms_exclusive=stop*episode.dt_ms,
                               artifact=artifact.name,artifact_sha256=sha256_file(artifact),
                               artifact_bytes=artifact.stat().st_size)
                summaries.append(summary)
            terminal=int(np.count_nonzero(episode.spike_ticks==episode.duration_ticks))
            if summaries[0]["events"]+terminal != len(episode.spike_ticks):
                raise ValueError("BENCHMARK_ACTIVITY_TERMINAL_EVENT_RECONCILIATION_FAILED")
            record=dict(scenario_id=entry["scenario_id"],condition_id=condition,
                episode_id=episode.episode_id,circuit_id=episode.circuit_id,block_id=episode.block_id,
                source_dt_ms=episode.dt_ms,complete_source_events=len(episode.spike_ticks),
                terminal_events=terminal,units=len(circuit.unit_ids),windows=summaries,
                acceptance=entry["acceptance"],acceptance_sha256=entry["acceptance_sha256"],
                source_catalog_sha256=entry["catalog_sha256"],source_dataset=entry["source"],
                seeds=episode.seeds,task_outcome=episode.task_outcome,
                covariance_artifacts=str(case.relative_to(output)))
            freeze_json(case/"report.json",record)
            report["episodes"].append(record)
        report.update(status="DESCRIBED_ACCEPTED_BENCHMARK_JOINT_ACTIVITY",
            complete_source_episodes_read=len(identities),
            scenario_source_circuits={name:len({x["circuit_id"] for x in report["episodes"] if x["scenario_id"]==name}) for name in SCENARIO_UNITS},
            same_observation_map_for_generated_outputs_required=True,
            interpretation="Descriptive accepted development activity. The mean covariance preserves signs and all units but does not identify pair structure or establish a joint law, biological success, statistical independence, or production qualification.")
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      recorded_utc=datetime.now(timezone.utc).isoformat(),
                      runtime=runtime_manifest(campaign,list(sys.argv),0 if report["status"]=="DESCRIBED_ACCEPTED_BENCHMARK_JOINT_ACTIVITY" else 1))
        freeze_json(output/"report.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--bin-ms",type=float,required=True)
    parser.add_argument("--lags-ms",type=float,nargs="+",required=True)
    args=parser.parse_args()
    result=analyze(args.config,args.inputs,args.project,args.output,bin_ms=args.bin_ms,lags_ms=tuple(args.lags_ms))
    print(json.dumps({k:result[k] for k in ["status","complete_source_episodes_read","elapsed_seconds"]}))


if __name__=="__main__":
    main()
