"""Replay and analyze complete independent native-driver calibration records."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .nest_gpu_poisson import _decode
from .poisson_count_dependence import CountTrace, analyze_traces, independent_record_variance
from .provenance import sha256_file


def source_support(report):
    """Native update at tick1 populates mu[0]; direct delivery starts at d-1.

    This is the pinned poiss_gen::Update/sendDirectSpikeKernel clock. The
    acquisition adapter uses backend d=physical delay+2. Earlier capture
    metadata mistakenly included the preceding empty tick; preserve that
    original declaration and disclose the correction rather than editing it.
    """
    first = report["backend_delay_ticks"]-1
    last = round(report["configuration"]["duration_ms"]/report["dt_ms"])-1
    declared = report["expected_arrival_support"]
    if declared not in ([first,last],[first-1,last]):
        raise ValueError("declared support incompatible with pinned continuous-driver clock")
    return first,last


def analyze(records: Path, output: Path):
    design = json.loads((records/"design.json").read_text())
    execution = json.loads((records/"execution.json").read_text())
    cfg = design["configuration"]; seeds = cfg["independent_record_seeds"]
    if execution["status"] != "COMPLETED" or [r["seed"] for r in execution["runs"]] != seeds:
        raise ValueError("complete prospective independent-record cohort required")
    if any(r["status"] != "COMPLETED" or r["exit_code"] != 0 for r in execution["runs"]):
        raise ValueError("failed records remain ineligible for source analysis")
    output.mkdir(parents=True, exist_ok=False)
    traces = []; checks = []; record_summaries = []; mapping = None; corrections=[]
    for seed in seeds:
        path=records/f"seed_{seed}"; report=json.loads((path/"acquisition.json").read_text())
        if report["status"] != "ACQUIRED_ENGINEERING" or report["configuration"]["seed"] != seed:
            raise ValueError("driver seed/status mismatch")
        if sha256_file(path/"trace.npz") != report["trace_sha256"]:
            raise ValueError("driver trace changed")
        identity={key:report[key] for key in ("graph_manifest_sha256","native_library_sha256","population_names",
                                             "population_sizes","rates_hz","dt_ms","expected_arrival_support")}
        if mapping is None: mapping=identity
        if identity != mapping: raise ValueError("source mapping or clock differs across records")
        with np.load(path/"trace.npz", allow_pickle=False) as arrays:
            trace=CountTrace(arrays["arrival_ticks"],arrays["sampled_counts"],arrays["aggregate_counts"],
                arrays["sampled_ids"],arrays["sampled_groups"],arrays["sampled_means"],tuple(arrays["group_labels"].tolist()),
                arrays["aggregate_means"],tuple(map(int,arrays["group_target_counts"])),int(arrays["complete_target_count"]))
            raw=arrays["selected_native_pre_currents"].astype(np.float64)
            decoded,_=_decode(raw,arrays["selected_decay"].astype(np.float64))
            if not np.array_equal(decoded[trace.arrival_ticks],trace.sampled_counts):
                raise ValueError("selected raw PSC count replay mismatch")
            if not np.array_equal(trace.aggregate_counts[:,:-1].sum(1),trace.aggregate_counts[:,-1]):
                raise ValueError("all-target aggregate differs from eight populations")
            if trace.complete_target_count != sum(report["population_sizes"]) or trace.complete_target_count != 100000:
                raise ValueError("production target coverage changed")
            first,last=source_support(report)
            if [first,last]!=report["expected_arrival_support"]:
                corrections.append(dict(seed=seed,declared=report["expected_arrival_support"],analyzed=[first,last],
                    reason="source clock: first update at tick1 plus physical delay; backend_delay-1"))
            inside=(trace.arrival_ticks>=first)&(trace.arrival_ticks<=last)
            if np.any(trace.aggregate_counts[~inside]): raise ValueError("outside-support counts")
            if not np.array_equal(trace.arrival_ticks[inside],np.arange(first,last+1)):
                raise ValueError("incomplete arrival support")
            trace=replace(trace,arrival_ticks=trace.arrival_ticks[inside],sampled_counts=trace.sampled_counts[inside],
                          aggregate_counts=trace.aggregate_counts[inside])
            traces.append(trace)
            checks.append(dict(seed=seed,raw_psc_values_replayed=raw.size,sampled_count_values_replayed=decoded[1:].size,
                               complete_target_count=trace.complete_target_count,source_trace_sha256=report["trace_sha256"],
                               count_decoding_replayed=True,full_population_sums_replayed_from_raw=False,
                               producer_full_unit_bins_decoded=report["decoded_unit_bins"]))
        detail=analyze_traces(trace,window_ticks=tuple(cfg["native_window_ticks"]),temporal_lags=tuple(cfg["temporal_lags"]),
                              hac_bandwidth_ticks=500,minimum_record_spans=20.,maximum_pairs=256,alpha=cfg["family_alpha"])
        (output/f"seed_{seed}_dependence.json").write_text(json.dumps(detail,indent=2,allow_nan=False)+"\n")
        for row in detail["aggregate_variance"]:
            record_summaries.append(dict(seed=seed,group=row["group"],window_ticks=row["native_window_ticks"],
                                         variance_excess_fraction=row.get("variance_excess_fraction"),
                                         mean_relative_error=row.get("mean_relative_error"),
                                         windows=row["windows"],status=row["status"]))
        print(json.dumps(dict(seed=seed,records_replayed=len(traces))),flush=True)
    result=independent_record_variance(traces,[str(seed) for seed in seeds],
        window_ticks=tuple(cfg["native_window_ticks"]),alpha=cfg["family_alpha"])
    # Independently use np.var on phase-aligned sums rather than the estimator's
    # centering/jackknife algebra. Independent units remain whole records.
    replay_max=0.
    for row in result["aggregate_variance"]:
        width=row["native_window_ticks"]; index=traces[0].group_labels.index(row["group"])
        ticks=traces[0].arrival_ticks; start=int(np.searchsorted(ticks,((int(ticks[0])+width-1)//width)*width))
        windows=(len(ticks)-start)//width
        values=np.stack([t.aggregate_counts[start:start+windows*width,index].reshape(windows,width).sum(1) for t in traces])
        expected=row["expected_count"]
        estimate=float(values.var(0,ddof=1).mean()/expected-1)
        delta=abs(estimate-row["variance_excess_fraction"]);replay_max=max(replay_max,delta)
        if not np.isclose(estimate,row["variance_excess_fraction"],atol=1e-12,rtol=0):
            raise ValueError("independent phase-aligned variance replay differs")
    (output/"independent_record_variance.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for name, rows in [("per_record_aggregate.csv",record_summaries),("capture_replay.csv",checks)]:
        with (output/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result_report=dict(status="CALIBRATION_REPLAYED_NOT_QUALIFIED",independent_records=len(traces),source_mapping=mapping,
                       analyzed_arrival_support=[int(traces[0].arrival_ticks[0]),int(traces[0].arrival_ticks[-1])],
                       source_clock_corrections=corrections,
                       independent_variance_replay_max_absolute_difference=replay_max,
                       sampled_raw_psc_values_replayed=sum(r["raw_psc_values_replayed"] for r in checks),
                       producer_full_unit_bins_decoded=sum(r["producer_full_unit_bins_decoded"] for r in checks),
                       full_population_aggregate_scope="every target reduced during acquisition; raw PSC replay covers fixed64 targets",
                       independent_population_aggregate_replay="separate engineering process only; production whole-record replay pending",
                       scientific_mean_error_margin=cfg["scientific_mean_error_margin"],
                       scientific_variance_error_margin=cfg["scientific_variance_error_margin"],
                       scientific_reference_qualified=False,neural_output_fidelity_qualified=False,
                       interpretation="source-count intervals do not bound neuronal response distortion or prove independence",
                       uncertainty="approximate whole-record jackknife with declared simultaneous Bonferroni family")
    (output/"analysis.json").write_text(json.dumps(result_report,indent=2)+"\n")
    return result_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(analyze(args.records,args.output)))


if __name__=="__main__":
    main()
