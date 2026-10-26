"""Separate native cuRAND generation from delivery and PSC count decoding.

The preload logs the actual scalar seeds produced during native calibration.
A fresh native run must first reproduce its original record. The standalone
CUDA program then uses those seeds and native per-population stream indices.
Any identity or sample mismatch is retained, never repaired by phase fitting.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

from .provenance import sha256_file


def run(records, seed, graph_manifest, preload, binary, output, capture_duration_ms=None):
    source=records/f"seed_{seed}"; acquired=json.loads((source/"acquisition.json").read_text())
    if acquired["configuration"]["seed"]!=seed or acquired["status"]!="ACQUIRED_ENGINEERING":
        raise ValueError("completed matching original driver record required")
    if sha256_file(source/"trace.npz")!=acquired["trace_sha256"]:
        raise ValueError("source trace changed")
    output.mkdir(parents=True,exist_ok=False)
    capture_config=dict(acquired["configuration"])
    if capture_duration_ms is not None:
        if not 0<capture_duration_ms<=capture_config["duration_ms"]:
            raise ValueError("seed capture must be a positive prefix of the original episode")
        capture_config["duration_ms"]=capture_duration_ms
    config=output/"capture.yaml";config.write_text(yaml.safe_dump(capture_config,sort_keys=False))
    seed_file=output/"native_derived_seeds.txt"
    command=[sys.executable,"-m","Libs.population_dynamics.native_driver_trace","--graph-manifest",str(graph_manifest),
             "--config",str(config),"--expected-library-sha256",acquired["native_library_sha256"],"--output",str(output/"native_replay")]
    environment=dict(os.environ,NEURONSIM_CURAND_SEED_TRACE=str(seed_file),
                     LD_PRELOAD=str(preload)+(":"+os.environ["LD_PRELOAD"] if os.environ.get("LD_PRELOAD") else ""))
    report=dict(status="RUNNING",seed=seed,native_command=command,scientific_reference_qualified=False,
                preload_sha256=sha256_file(preload),reference_binary_sha256=sha256_file(binary),
                original_trace_sha256=acquired["trace_sha256"])
    def save(): (output/"replay.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    save()
    try:
        start=time.perf_counter()
        with (output/"native_replay.log").open("w") as stream:
            result=subprocess.run(command,env=environment,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        report["native_exit_code"]=result.returncode;report["native_seconds"]=time.perf_counter()-start;save()
        if result.returncode: raise RuntimeError("instrumented native capture failed")
        with np.load(source/"trace.npz",allow_pickle=False) as original,np.load(output/"native_replay/trace.npz",allow_pickle=False) as repeated:
            temporal={"arrival_ticks","sampled_counts","aggregate_counts","selected_native_pre_currents","record_ticks"}
            short=capture_config["duration_ms"]!=acquired["configuration"]["duration_ms"]
            exact={key:bool(np.array_equal(original[key][:len(repeated[key])] if key in temporal else original[key],repeated[key]))
                   for key in original.files if key!="terminal_current" or not short}
            report["native_capture_duration_ms"]=capture_config["duration_ms"]
            report["native_comparison_scope"]="complete recorded prefix; different-time terminal current omitted" if short else "complete recorded arrays"
            report["instrumented_native_exact_arrays"]=exact;save()
            if not all(exact.values()):raise ValueError("seed instrumentation changed native results")
            native_seeds=[int(value) for value in seed_file.read_text().splitlines()]
            if len(native_seeds)!=8:raise ValueError("expected exactly eight scalar seed draws; do not guess seed mapping")
            report["native_derived_seeds"]=native_seeds
            first=acquired["backend_delay_ticks"]-1
            inside=original["arrival_ticks"]>=first
            ticks=original["arrival_ticks"][inside];n_ticks=len(ticks)
            offsets=np.r_[0,np.cumsum(acquired["population_sizes"])]
            sampled_ids=original["sampled_ids"];sampled_groups=original["sampled_groups"]
            lines=[];sample_sizes=[]
            for i,(stream_seed,count,rate) in enumerate(zip(native_seeds,acquired["population_sizes"],acquired["rates_hz"])):
                # The native group starts at ID0; record the proposed exact
                # per-group connection map and reject mismatch instead of
                # searching post hoc for a matching permutation.
                selected=sampled_ids[sampled_groups==i]-offsets[i]
                sample_sizes.append(len(selected))
                lines.append(" ".join(map(str,[stream_seed,count,rate,len(selected),*selected])))
            inputs=output/"streams.txt";inputs.write_text("\n".join(lines)+"\n")
            reference=output/"xorwow.bin"
            command=[str(binary),str(inputs),str(reference),str(n_ticks),str(acquired["dt_ms"]),"0","xorwow","8"]
            with (output/"reference.log").open("w") as stream:
                result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            report["reference_command"]=command;report["reference_exit_code"]=result.returncode;save()
            if result.returncode:raise RuntimeError("standalone reference failed")
            values=np.fromfile(reference,dtype=np.uint64);position=0;totals=[];samples=[]
            for k in sample_sizes:
                totals.append(values[position:position+n_ticks]);position+=n_ticks
                samples.append(values[position:position+n_ticks*k].reshape(n_ticks,k));position+=n_ticks*k
            if position!=len(values):raise ValueError("standalone binary payload size mismatch")
            total=np.stack(totals,axis=1);sample=np.concatenate(samples,axis=1)
            total_exact=np.array_equal(total,original["aggregate_counts"][inside,:8])
            sample_exact=np.array_equal(sample,original["sampled_counts"][inside])
            report.update(standalone_population_totals_exact=bool(total_exact),standalone_sampled_targets_exact=bool(sample_exact),
                group_total_values_compared=total.size,sampled_target_values_compared=sample.size,
                population_total_max_abs_difference=int(np.max(np.abs(total.astype(np.int64)-original["aggregate_counts"][inside,:8]))),
                sampled_target_max_abs_difference=int(np.max(np.abs(sample.astype(np.int64)-original["sampled_counts"][inside]))),
                first_active_tick=int(ticks[0]),last_active_tick=int(ticks[-1]),
                reference_scope="native XORWOW per-target stream after startup; no neurons, delays or floating PSC states",
                status="EXACT_NATIVE_RANDOM_STREAM_REPLAY" if total_exact and sample_exact else "REFERENCE_MISMATCH")
            np.savez_compressed(output/"standalone_counts.npz",arrival_ticks=ticks,aggregate_counts=total,sampled_counts=sample,
                                sampled_ids=sampled_ids,native_derived_seeds=native_seeds)
            save()
            if not total_exact or not sample_exact:raise ValueError("standalone source stream mismatch requires localization")
    except Exception as error:
        report["error"]=repr(error)
        if report["status"]=="RUNNING":report["status"]="FAILED"
        save();raise
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("records","graph-manifest","preload","binary","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--capture-duration-ms",type=float)
    args=parser.parse_args()
    print(json.dumps(run(args.records,args.seed,args.graph_manifest,args.preload,args.binary,args.output,args.capture_duration_ms)))


if __name__=="__main__":
    main()
