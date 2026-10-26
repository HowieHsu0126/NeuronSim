"""Predeclared factorial acquisition of source-law pulse-history responses."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import yaml

from .nest_gpu_network import ticks
from .nest_gpu_spatial import Execution
from .provenance import sha256_file


def conditions(config, graph_manifest, library):
    if config["schema"] != "mesocircuit_pulse_history_development_v1":
        raise ValueError("unknown pulse-history acquisition schema")
    initial, background = config["initial_state_seeds"], config["background_seeds"]
    if (len(initial) != len(background) or len(initial) < 2
            or len(set(initial)) != len(initial) or len(set(background)) != len(background)):
        raise ValueError("distinct initial-state/background replicate blocks required")
    net = graph_manifest["resolved_source_network"]
    dt = graph_manifest["source_config"]["simulation"]["sim_resolution"]
    delay = net["th_delay_pulse_generator"]
    times = config["focal_arrivals_ms"]
    offsets = config["conditioning_offsets_ms"]
    if (not times or len(set(times)) != len(times) or not offsets
            or any(x >= 0 for x in offsets) or any(a >= b for a,b in zip(offsets,offsets[1:]))):
        raise ValueError("distinct focal times and increasing negative conditioning offsets required")
    if (config["history_window_ms"] <= 0 or config["response_window_ms"] <= 0
            or config["response_curve_ms"] < config["response_window_ms"]):
        raise ValueError("positive complete response/history windows required")
    rows=[]
    for block, (state_seed, noise_seed) in enumerate(zip(initial,background)):
        for focal, history, present in itertools.product(times,(False,True),(False,True)):
            if focal-delay-config["history_window_ms"] < 0 or focal+config["response_curve_ms"] >= config["duration_ms"]:
                raise ValueError("analysis windows outside complete episode support")
            arrivals = [focal+x for x in offsets] if history else []
            if present:arrivals.append(focal)
            emission = [x-delay for x in arrivals]
            ticks([config["duration_ms"],focal,delay,*emission],dt)
            if any(x <= 0 or x+delay >= config["duration_ms"] for x in emission):
                raise ValueError("invalid source pulse timing")
            settings = dict(config["execution"], duration_ms=config["duration_ms"],
                background_seed=noise_seed, initial_state_seed=state_seed,
                source_rng_library=str(Path(library).resolve()),
                pulse_control={"initial_emission_ms":emission})
            if config.get("capture_focal_state", False):
                settings["state_observation_ticks"] = [int(ticks([focal-delay],dt)[0])-1]
            Execution(**settings).validate()
            name=f"block_{block}_focal_{focal:g}_history_{int(history)}_pulse_{int(present)}"
            rows.append(dict(case_id=name,block=block,initial_state_seed=state_seed,background_seed=noise_seed,
                focal_arrival_ms=focal,history=history,present=present,settings=settings))
    if len(rows)!=config["expected_episodes"]:
        raise ValueError("predeclared factorial denominator mismatch")
    random.Random(config["run_order_seed"]).shuffle(rows)
    return rows


def acquire(graph: Path, library: Path, config_path: Path, output: Path, resume=False):
    config=yaml.safe_load(config_path.read_text())
    source=json.loads((graph/"manifest.json").read_text())
    cases=conditions(config,source,library)
    declaration=dict(schema=config["schema"],config=config,cases=cases,
        graph_manifest_sha256=sha256_file(graph/"manifest.json"),source_rng_sha256=sha256_file(library),
        scientific_reference_qualified=False,neuroscience_finding_qualified=False,
        formal_training_completed=False,formal_training_seeds=[41,42,43],
        independent_replication="initial state and drive-noise block, conditional on the fixed graph")
    return acquire_cases(graph, output, declaration, resume)


def acquire_cases(graph: Path, output: Path, declaration: dict, resume=False):
    """Execute declared continuous episodes with the shared native runner.

    The declaration owns the denominator and order. Both factorial and longer
    trajectory acquisitions use this path and its exact-resume contract.
    """
    cases = declaration["cases"]
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("nonempty unique declared episode identities required")
    native_digest = declaration.get("native_library_sha256")
    if native_digest is not None and sha256_file(Path(os.environ["NESTGPU_LIB"])) != native_digest:
        raise ValueError("declared native runtime changed before acquisition")
    if output.exists():
        if not resume or json.loads((output/"design.json").read_text())!=declaration:
            raise ValueError("resume requires the exact predeclared acquisition")
    else:
        output.mkdir(parents=True)
        (output/"design.json").write_text(json.dumps(declaration,indent=2)+"\n")
    started=time.perf_counter();completed=[]
    for case in cases:
        run=output/case["case_id"];settings=output/(case["case_id"]+".yaml")
        text=yaml.safe_dump(case["settings"],sort_keys=False)
        if settings.exists() and settings.read_text()!=text:
            raise ValueError("existing episode configuration changed")
        settings.write_text(text)
        command=[sys.executable,"-m","Libs.population_dynamics.nest_gpu_spatial","--graph",str(graph),
                 "--config",str(settings),"--output",str(run)]
        if run.exists():
            report=json.loads((run/"report.json").read_text())
            if (not resume or report["status"]!="EXECUTED" or asdict(Execution(**report["config"]))!=asdict(Execution(**case["settings"]))
                    or report["graph_manifest_sha256"]!=declaration["graph_manifest_sha256"]
                    or (native_digest is not None and report.get("native_library_sha256") != native_digest)):
                raise ValueError("cannot reuse incomplete or different episode")
            for name,digest in report["artifacts"].items():
                if sha256_file(run/name)!=digest:raise ValueError("resume episode artifact mismatch")
            exit_code=0;reused=True
        else:
            with (output/(case["case_id"]+".log")).open("w") as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            exit_code=result.returncode;reused=False
        row=dict(case_id=case["case_id"],exit_code=exit_code,reused=reused,command=command)
        completed.append(row)
        status="RUNNING" if exit_code==0 else "FAILED"
        ledger=dict(status=status,expected_episodes=len(cases),processed_episodes=len(completed),
            completed_episodes=sum(r["exit_code"]==0 for r in completed),runs=completed,
            elapsed_seconds=time.perf_counter()-started,scientific_reference_qualified=False)
        (output/"acquisition.json").write_text(json.dumps(ledger,indent=2)+"\n")
        print(json.dumps({k:v for k,v in ledger.items() if k!="runs"}),flush=True)
        if exit_code:return 1
    ledger["status"]="ACQUIRED"
    (output/"acquisition.json").write_text(json.dumps(ledger,indent=2)+"\n")
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph",type=Path,required=True)
    parser.add_argument("--source-rng-library",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    a=parser.parse_args()
    raise SystemExit(acquire(a.graph,a.source_rng_library,a.config,a.output,a.resume))


if __name__=="__main__":main()
