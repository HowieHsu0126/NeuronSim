"""Compare existing 100k PFC gain banks with named source readouts and stored-input checks."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import os
from pathlib import Path

import numpy as np
import yaml

from pfc_coupled_clock_block_diagnostics import Archive, CLOCKS, PAIRS, PHASES, METRICS, moments, require, write_csv
from pfc_coupled_gain_input_identity import check_archives

CONDITIONS=("omission","cue","strong_cue")
BOUNDS=np.array([[0,100],[100,300],[300,500],[1000,2000],[4000,5000]],dtype=float)
SEEDS=tuple(range(95001,95033))
CONTRASTS=(("cue_minus_omission",1,0),("strong_cue_minus_omission",2,0),("strong_cue_minus_cue",2,1))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(data):
    with np.load(io.BytesIO(data),allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def axis_order(saved, expected, name):
    values=list(saved)
    require(len(values)==len(expected) and len(set(values))==len(values) and set(values)==set(expected), "named axis differs: "+name)
    return [values.index(x) for x in expected]


def normalize_response(z, *, candidate_seed=None):
    ci=axis_order(z["condition_names"],CONDITIONS,"condition")
    pi=axis_order(z["phase_names"],PHASES,"phase")
    mi=axis_order(z["metric_names"],METRICS,"metric")
    require(str(z["units"].item())=="Hz", "response units differ")
    if candidate_seed is None:
        si=axis_order(z["source_rng_seeds"],SEEDS,"source block")
        bounds=np.column_stack([z["phase_start_ms"],z["phase_stop_ms"]])
        values=[]
        for clock in CLOCKS:
            a=z[clock+"_values"]
            require(a.shape==(3,32,5,8), "baseline shape differs")
            values.append(a[np.ix_(ci,si,pi,mi)].transpose(1,0,2,3))
        result=np.stack(values,axis=1)  # block,clock,condition,phase,metric
    else:
        require(z["cohort_names"].tolist()==["expanded_100k"] and
                z["source_rng_seeds"].tolist()==[candidate_seed] and
                z["value_axes"].tolist()==["cohort","source_rng_seed","clock","condition","phase","metric"] and
                z["evidence_kind"].item()=="archived_source_execution", "candidate identity differs")
        ki=axis_order(z["clock_names"],CLOCKS,"clock")
        require(np.array_equal(z["clock_dt_ms"][ki],[.05,.025,.0125]), "clock values differ")
        bounds=z["phase_bounds_ms"];a=z["values"]
        require(a.shape==(1,1,3,3,5,8), "candidate shape differs")
        result=a[0,0][np.ix_(ki,ci,pi,mi)][None]
    require(np.array_equal(bounds[pi],BOUNDS), "physical phase bounds differ")
    require(np.isfinite(result).all(), "nonfinite source response")
    return result


def gain_rows(baseline,candidate):
    require(baseline.shape==candidate.shape==(32,3,3,5,8), "complete response bank required")
    rows=[];interactions=[]
    for b,seed in enumerate(SEEDS):
        partition="pilot3" if b<3 else "remaining29"
        for phase,metric in itertools.product(range(5),range(8)):
            for label,positive,negative in [(x,i,None) for i,x in enumerate(CONDITIONS)]+list(CONTRASTS):
                before=baseline[b,:,positive,phase,metric].copy();after=candidate[b,:,positive,phase,metric].copy()
                if negative is not None:
                    before-=baseline[b,:,negative,phase,metric];after-=candidate[b,:,negative,phase,metric]
                effects=after-before
                base=dict(partition=partition,source_rng_seed=seed,response=label,phase=PHASES[phase],metric=METRICS[metric],units="Hz")
                for k,clock in enumerate(CLOCKS):
                    rows.append(dict(base,clock=clock,baseline_value=float(before[k]),candidate_value=float(after[k]),gain_difference=float(effects[k])))
                for left,right in PAIRS:
                    i,j=CLOCKS.index(left),CLOCKS.index(right)
                    interactions.append(dict(base,left_clock=left,right_clock=right,
                        baseline_clock_change=float(before[i]-before[j]),candidate_clock_change=float(after[i]-after[j]),
                        gain_by_clock_difference=float(effects[i]-effects[j])))
    return rows,interactions


def summarize(rows, keys, metrics):
    groups=defaultdict(list)
    for r in rows:
        for partition in (r["partition"],"full32_descriptive"):
            groups[(partition,*(r[k] for k in keys))].append(r)
    result=[]
    for key,group in sorted(groups.items()):
        require(len(group)=={"pilot3":3,"remaining29":29,"full32_descriptive":32}[key[0]] and
                len({r["source_rng_seed"] for r in group})==len(group), "paired block denominator differs")
        row=dict(zip(("partition",*keys),key),blocks=len(group))
        for metric in metrics:
            row.update({metric+"_"+k:v for k,v in moments([r[metric] for r in group]).items()})
        result.append(row)
    return result


def run(config_path,project,output):
    config=yaml.safe_load(Path(config_path).read_text());output=Path(output)
    require(not output.exists(),"output must be new")
    require(config["schema"]=="pfc_gain_clock_existing_bank_comparison_v1" and
            config["source_rng_seeds"]==list(SEEDS) and config["source_simulations"]==0 and
            config["new_independent_blocks"]==0 and not config["confirmation_execution_released"],"existing32-block protocol required")
    output.mkdir(parents=True);bindings={};reports=[]
    reuse=config["reused_identity"];prior=Archive(project/reuse["archive"],bindings,reuse["delivery_sha256"])
    prior_bytes=prior.read("analysis/report.json");require(hashlib.sha256(prior_bytes).hexdigest()==reuse["report_sha256"],"prior identity report changed")
    # Reuse the real one-block check and its tests only for the identical checker.
    require(hashlib.sha256(prior.read("pfc_coupled_gain_input_identity.py")).hexdigest()==digest(Path(__file__).with_name("pfc_coupled_gain_input_identity.py")),"identity checker changed")
    entries=config["blocks"];require([r["source_rng_seed"] for r in entries]==list(SEEDS),"complete ordered block mapping required")
    for entry in entries:
        seed=entry["source_rng_seed"]
        if seed==reuse["source_rng_seed"]:
            report=json.loads(prior_bytes)
            for role in ("baseline","candidate"):
                require(report["archives"][role]["expected_delivery_sha256"]==entry[role]["delivery_sha256"],"reused source identity differs")
            reused=True
        else:
            report=check_archives(project/entry["baseline"]["archive"],project/entry["candidate"]["archive"],
                entry["baseline"]["delivery_sha256"],entry["candidate"]["delivery_sha256"],seed,output/"identity"/str(seed))
            reused=False
        accepted=report["status"]=="PASS_STORED_COUPLED_GAIN_INPUT_IDENTITY" and report["passed_pairs"]==6
        reports.append(dict(source_rng_seed=seed,accepted=accepted,passed_pairs=report.get("passed_pairs",0),reused=reused,
            report_path=str(prior.path/"analysis/report.json") if reused else str(Path("identity")/str(seed)/"report.json")))
        print(json.dumps(reports[-1]),flush=True)
    (output/"identity_summary.json").write_text(json.dumps(dict(blocks=reports,complete=len(reports)==32 and all(r["accepted"] for r in reports),
        source_simulations=0,full_external_streams_compared=False),indent=2)+"\n")
    require(all(r["accepted"] for r in reports),"stored input identity failed; gain analysis withheld")
    sources={k:Archive(project/config[k]["archive"],bindings,config[k]["delivery_sha256"]) for k in
             ("baseline_readout","candidate_readout","candidate_pilot_readout")}
    for archive in sources.values():
        for name,expected in config["observation_definition_modules"].items():
            require(hashlib.sha256(archive.read("Libs/population_dynamics/"+name)).hexdigest()==expected,"source observation definition changed")
    baseline=normalize_response(arrays(sources["baseline_readout"].read("analysis/coupled_responses.npz")))
    pieces=[]
    for seed in SEEDS:
        archive=sources["candidate_pilot_readout" if seed<95004 else "candidate_readout"]
        pieces.append(normalize_response(arrays(archive.read(f"analysis/expanded_100k/{seed}/source_phase_responses.npz")),candidate_seed=seed))
    candidate=np.concatenate(pieces,axis=0);rows,interactions=gain_rows(baseline,candidate)
    summaries=summarize(rows,("response","phase","metric","clock"),("baseline_value","candidate_value","gain_difference"))
    clock_summaries=summarize(interactions,("response","phase","metric","left_clock","right_clock"),
                              ("baseline_clock_change","candidate_clock_change","gain_by_clock_difference"))
    outputs={}
    for name,table in (("gain_response_by_block.csv",rows),("gain_clock_by_block.csv",interactions),
                       ("gain_response_summary.csv",summaries),("gain_clock_summary.csv",clock_summaries)):
        write_csv(output/name,table);outputs[name]=dict(bytes=(output/name).stat().st_size,sha256=digest(output/name))
    report=dict(status="COMPLETE_DESCRIPTIVE_GAIN_BY_CLOCK_COMPARISON",finished_utc=datetime.now(timezone.utc).isoformat(),
        job_id=os.environ.get("SLURM_JOB_ID"),config_sha256=digest(config_path),implementation_sha256=digest(__file__),
        input_sha256=bindings,stored_input_pairs=192,new_stored_input_pairs_checked=186,reused_stored_input_pairs=6,
        independent_source_blocks=32,complete_source_episodes_reused=576,source_simulations=0,new_rng_draws=0,
        source_response_rows=len(rows),gain_clock_rows=len(interactions),summary_rows=len(summaries),
        full_external_streams_compared=False,full_unit_fidelity_compared=False,confidence_intervals=False,p_values=False,
        source_scale_qualified=False,scientific_fidelity_qualified=False,confirmation_execution_released=False,
        production_episodes=0,outputs=outputs,interpretation=config["interpretation"])
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","project","output"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();report=run(args.config,args.project,args.output)
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,dict)}))


if __name__=="__main__":main()
