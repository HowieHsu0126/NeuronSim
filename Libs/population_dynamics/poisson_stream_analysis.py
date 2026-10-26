"""Compare native-matched XORWOW and prospectively declared Philox streams."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .poisson_count_dependence import CountTrace, complete_window_sums, independent_record_variance
from .provenance import sha256_file


def binary_counts(path, ticks, sample_sizes):
    data=np.fromfile(path,dtype=np.uint64);position=0;totals=[];samples=[]
    for count in sample_sizes:
        totals.append(data[position:position+ticks]);position+=ticks
        samples.append(data[position:position+ticks*count].reshape(ticks,count));position+=ticks*count
    if position!=len(data):raise ValueError("inconsistent complete reference payload")
    groups=np.stack(totals,1).astype(np.int64)
    return np.column_stack((groups,groups.sum(1))),np.concatenate(samples,1).astype(np.int32)


def analyze(streams: Path, records: Path, output: Path):
    protocol=json.loads((streams/"cohort_design.json").read_text())
    execution=json.loads((streams/"cohort_execution.json").read_text())
    if execution["status"]!="COMPLETED":raise ValueError("all declared stream comparisons must finish")
    seeds=protocol["seeds"];widths=tuple(protocol["mean_and_variance_windows"])
    output.mkdir(parents=True,exist_ok=False)
    traces={"xorwow":[],"philox":[]};replay=[]
    for seed in seeds:
        path=streams/f"seed_{seed}";report=json.loads((path/"replay.json").read_text())
        source=records/f"seed_{seed}";receipt=json.loads((source/"acquisition.json").read_text())
        if report["status"]!="EXACT_NATIVE_RANDOM_STREAM_REPLAY" or sha256_file(source/"trace.npz")!=receipt["trace_sha256"]:
            raise ValueError("all exact native source stream replays required")
        with np.load(source/"trace.npz",allow_pickle=False) as native:
            first=receipt["backend_delay_ticks"]-1
            inside=native["arrival_ticks"]>=first;ticks=native["arrival_ticks"][inside]
            sample_sizes=[int(np.sum(native["sampled_groups"]==g)) for g in range(8)]
            for method in traces:
                sums,sample=binary_counts(path/f"{method}.bin",len(ticks),sample_sizes)
                if method=="xorwow":
                    if not np.array_equal(sums,native["aggregate_counts"][inside]) or not np.array_equal(sample,native["sampled_counts"][inside]):
                        raise ValueError("independent native/XORWOW binary equality failed")
                traces[method].append(CountTrace(ticks.copy(),sample,sums,native["sampled_ids"],native["sampled_groups"],
                    native["sampled_means"],tuple(native["group_labels"].tolist()),native["aggregate_means"],
                    tuple(map(int,native["group_target_counts"])),int(native["complete_target_count"])))
            replay.append(dict(seed=seed,population_values=9*len(ticks),sampled_values=len(ticks)*sum(sample_sizes),
                               xorwow_payload_sha256=sha256_file(path/"xorwow.bin"),philox_payload_sha256=sha256_file(path/"philox.bin")))
    summaries={method:independent_record_variance(values,list(map(str,seeds)),window_ticks=widths)
               for method,values in traces.items()}
    table=[]
    for method,summary in summaries.items():
        (output/f"{method}_variance.json").write_text(json.dumps(summary,indent=2)+"\n")
        for row in summary["aggregate_variance"]:
            table.append(dict(method=method,group=row["group"],window_ticks=row["native_window_ticks"],
                              variance_excess=row["variance_excess_fraction"],
                              variance_low=row["variance_interval"][0],variance_high=row["variance_interval"][1],
                              mean_relative_error=row["mean_relative_error"],records=len(seeds),windows_per_record=row["windows_per_record"]))
    comparisons=[];r=len(seeds);family=len(widths)*len(traces["xorwow"][0].group_labels)
    z=NormalDist().inv_cdf(1-.05/(2*family))
    for width in widths:
        values={m:np.stack([complete_window_sums(t.aggregate_counts,t.arrival_ticks,width)[0] for t in ts]).astype(float)
                for m,ts in traces.items()}
        for group,label in enumerate(traces["xorwow"][0].group_labels):
            expected=width*traces["xorwow"][0].aggregate_means[group]
            estimates={m:float(v[:,:,group].var(0,ddof=1).mean()/expected-1) for m,v in values.items()}
            difference=estimates["xorwow"]-estimates["philox"]
            leave_one=[]
            for omitted in range(r):
                keep=np.arange(r)!=omitted
                leave_one.append((values["xorwow"][keep,:,group].var(0,ddof=1).mean()
                                  -values["philox"][keep,:,group].var(0,ddof=1).mean())/expected)
            loo=np.asarray(leave_one);se=float(np.sqrt((r-1)/r*np.sum((loo-loo.mean())**2)))
            comparisons.append(dict(group=label,window_ticks=width,xorwow_minus_philox=difference,
                                    interval_low=difference-z*se,interval_high=difference+z*se,
                                    jackknife_standard_error=se,records=r,interval_family_size=family))
    for name,rows in [("stream_variance_summary.csv",table),("stream_paired_variance.csv",comparisons),("native_replay_summary.csv",replay)]:
        with (output/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(status="ANALYZED_RANDOM_STREAM_DIAGNOSTIC",independent_records=len(seeds),
                native_xorwow_population_values_exact=sum(v["population_values"] for v in replay),
                native_xorwow_sampled_values_exact=sum(v["sampled_values"] for v in replay),
                paired_contrast="same record/derived-seed membership; no claim of matched uniform random draws",
                source_fidelity_qualified=False,philox_source_implementation_qualified=False,
                uncertainty="approximate independent-record jackknife; simultaneous Bonferroni across declared windows/groups",
                scope="isolated continual production-rate drivers; no neuronal inference or native RNG replacement")
    (output/"analysis.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("streams","records","output"):parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.streams,args.records,args.output)))


if __name__=="__main__":main()
