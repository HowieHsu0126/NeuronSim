"""Development-only timing prediction for conditional-history sampling design.

Use complete coarse archives, holding one original block out of each fit.
No population confidence interval, paired-clock inference or neural forecasting
claim follows from these overlapping cross-validation folds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def require(condition, message):
    if not condition:
        raise ValueError("PFC_HISTORY_TIMING_"+message)


def time_bin(row, dt_ms, edges):
    category=row["category"]
    if category in ("no_first_spike","refractory_covers_remaining_window"):
        require(row["conditional_estimate"]==row["inner_monte_carlo_variance"]==0,"STRUCTURAL_ZERO_SCORE")
        return -1,0.
    require(category=="conditional_integration_required","RESOLVED_REFERENCE_REQUIRED")
    free=(row["window_stop_tick"]-row["first_spike_tick"]-row["refractory_ticks"])*dt_ms
    require(edges[0]<free<=edges[-1],"POSITIVE_POST_REFRACTORY_TIME")
    return int(np.searchsorted(edges[1:],free,side="left")),float(free)


def heldout_predictions(records, heldout, bins):
    train=[row for row in records if row["source_rng_seed"]!=heldout and row["time_bin"]>=0]
    counts={index:sum(row["time_bin"]==index for row in train) for index in range(bins)}
    means={index:float(np.mean([row["estimate"] for row in train if row["time_bin"]==index]))
           for index in counts if counts[index]}
    baseline=float(np.mean([row["estimate"] for row in train])) if train else None
    missing=[index for index in counts if counts[index]==0]
    result=[]
    for row in records:
        if row["source_rng_seed"]!=heldout:continue
        if row["time_bin"]<0:a=b=0.
        else:a=baseline;b=means.get(row["time_bin"])
        va=None if a is None else (row["estimate"]-a)**2-row["inner_variance"]
        vb=None if b is None else (row["estimate"]-b)**2-row["inner_variance"]
        gain=None if a is None or b is None else (row["estimate"]-a)**2-(row["estimate"]-b)**2
        result.append(dict(row,baseline_prediction=a,timing_prediction=b,
            baseline_noise_corrected_loss=va,timing_noise_corrected_loss=vb,paired_loss_reduction=gain))
    complete=not missing and baseline is not None
    diagnostic=dict(source_rng_seed=heldout,heldout_windows=len(result),training_eligible_windows=len(train),
        training_bin_counts=counts,missing_training_bins=missing,complete=complete,
        baseline_noise_corrected_loss=float(np.mean([r["baseline_noise_corrected_loss"] for r in result])) if baseline is not None else None,
        timing_noise_corrected_loss=float(np.mean([r["timing_noise_corrected_loss"] for r in result])) if complete else None,
        paired_loss_reduction=float(np.mean([r["paired_loss_reduction"] for r in result])) if complete else None)
    return result,diagnostic


def decision(folds):
    if not all(row["complete"] for row in folds):
        return dict(status="INSUFFICIENT_TRAINING_BIN_COVERAGE",aggregate_gain=None,
                    gain_after_largest_positive_block_removed=None,prepare_timing_stratum_census=False)
    gains=np.asarray([row["paired_loss_reduction"] for row in folds],dtype=float)
    require(len(gains)>2 and np.isfinite(gains).all(),"COMPLETE_FOLD_GAINS")
    largest=int(np.argmax(gains));positive=gains[largest]>0
    retained=np.delete(gains,largest) if positive else gains
    gain=float(gains.mean());influence_gain=float(retained.mean())
    proceed=gain>0 and influence_gain>0
    return dict(status="PREPARE_TIMING_STRATUM_CENSUS" if proceed else "DO_NOT_PREPARE_TIMING_STRATIFICATION_ON_THIS_RATIONALE",
        aggregate_gain=gain,gain_after_largest_positive_block_removed=influence_gain,
        removed_original_block=folds[largest]["source_rng_seed"] if positive else None,
        positive_block_gains=int(np.count_nonzero(gains>0)),negative_block_gains=int(np.count_nonzero(gains<0)),
        zero_block_gains=int(np.count_nonzero(gains==0)),prepare_timing_stratum_census=bool(proceed))


def bound_json(root,entries,name):
    data=(root/name).read_bytes();entry=entries[name]
    require(len(data)==entry["bytes"] and hashlib.sha256(data).hexdigest()==entry["sha256"],"BOUND_SUMMARY_CHANGED:"+name)
    return json.loads(data)


def write_csv(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)


def run(project,settings,output):
    output.mkdir(parents=True,exist_ok=False)
    seeds=settings["required_source_seeds"];edges=settings["post_refractory_time_bin_edges_ms"]
    require(seeds==list(range(95001,95033)) and edges==[0.,1.,2.,3.],"FROZEN32_BLOCK_TIMING_PLAN")
    origin=project/settings["accepted_coarse_summary"]["archive"]
    payload=(origin/"delivery.json").read_bytes()
    require(hashlib.sha256(payload).hexdigest()==settings["accepted_coarse_summary"]["delivery_sha256"],"ACCEPTED_SUMMARY_CHANGED")
    delivery=json.loads(payload);entries={row["path"]:row for row in delivery["files"]}
    require(delivery["verified"] and delivery["status"]=="COMPLETE_PFC_HISTORY_SUMMARY_COMPLETE","ACCEPTED_COMPLETE_COARSE_SUMMARY")
    wrapper=bound_json(origin,entries,"report.json");summary=bound_json(origin,entries,"analysis/report.json")
    config_name="Libs/configs/neural_population_pfc_first_passage_history_summary.yaml"
    config_bytes=(origin/config_name).read_bytes()
    require(hashlib.sha256(config_bytes).hexdigest()==entries[config_name]["sha256"],"SOURCE_DOMAIN_CONFIG_CHANGED")
    source_settings=yaml.safe_load(config_bytes)
    require(all(settings[key]==source_settings[key] for key in
                ("dt_ms","field_index","phase","window_ms","sample_size","required_source_seeds")),"SOURCE_DOMAIN_IDENTITY")
    require(wrapper["exit_code"]==0 and summary["available_original_blocks"]==32
            and summary["sampled_unit_windows"]==2048,"ALL32_COARSE_INPUTS_REQUIRED")
    csv_path=origin/"analysis/block_diagnostics.csv";raw=csv_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==entries["analysis/block_diagnostics.csv"]["sha256"],"BLOCK_INDEX_CHANGED")
    blocks=list(csv.DictReader(raw.decode().splitlines()))
    require([int(row["source_rng_seed"]) for row in blocks]==seeds,"BLOCK_IDENTITIES")
    accepted_inputs={row["path"]:row["sha256"] for row in wrapper["inputs"]}
    records=[];bindings=[]
    for block in blocks:
        seed=int(block["source_rng_seed"]);relative=block["archive"]+"/validation/integrated_frame.json"
        content=(project/relative).read_bytes();digest=hashlib.sha256(content).hexdigest()
        require(digest==accepted_inputs["project/"+relative],"ACCEPTED_INTEGRATED_FRAME_CHANGED")
        rows=json.loads(content);require(len(rows)==settings["sample_size"]==64,"FULL_WINDOW_DENOMINATOR")
        bindings.append(dict(source_rng_seed=seed,frame=relative,sha256=digest))
        for row in rows:
            index,remaining=time_bin(row,settings["dt_ms"],edges)
            q,v=row["conditional_estimate"],row["inner_monte_carlo_variance"]
            require(np.isfinite(q) and np.isfinite(v) and 0<=q<=4 and v>=0,"SCORE_SUPPORT")
            records.append(dict(source_rng_seed=seed,sample_index=row["sample_index"],frame_index=row["frame_index"],
                unit_index=row["unit_index"],category=row["category"],time_bin=index,
                post_refractory_remaining_ms=remaining,estimate=q,inner_variance=v))
    predictions=[];folds=[]
    for seed in seeds:
        rows,fold=heldout_predictions(records,seed,len(edges)-1);predictions.extend(rows);folds.append(fold)
    choice=decision(folds)
    write_csv(output/"heldout_window_predictions.csv",predictions)
    write_csv(output/"block_prediction_losses.csv",folds)
    counts=Counter(row["time_bin"] for row in records)
    report=dict(status="COMPLETE_PFC_HISTORY_TIMING_DEVELOPMENT_CHECK",original_blocks=32,sampled_windows=len(records),
        field_index=settings["field_index"],phase=settings["phase"],dt_ms=settings["dt_ms"],window_ms=settings["window_ms"],
        positive_time_bin_edges_ms=edges,sampled_bin_counts={str(k):counts.get(k,0) for k in (-1,0,1,2)},
        eligible_zero_estimates=sum(row["time_bin"]>=0 and row["estimate"]==0 for row in records),
        decision=choice,bindings=bindings,
        prediction_scope="Held-out archived conditional-probability scores for sampling design; not neural forecasting.",
        loss_scope="Squared prediction error minus the recorded inner variance. Negative losses retained; this does not remove training-predictor error or establish confidence coverage.",
        fold_scope="Overlapping leave-one-block-out fits; gains are development diagnostics, without a significance test.",
        census_scope="Sampled bin counts are not full-frame stratum weights; future precision requires a complete census and a prospective design.",
        paired_clock_scope="Incomplete fine results were unused. Coarse structural zeros cannot exclude paired physical IDs.",
        population_confidence_interval_computed=False,future_stratified_precision_established=False,
        scientific_fidelity_qualified=False,confirmation_execution_released=False,production_episodes=0,
        new_independent_source_blocks=0,new_neural_trajectories=0)
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("config","project-root","output"):parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();settings=yaml.safe_load(args.config.read_text())
    require(settings["schema"]=="pfc_history_timing_prediction_v1","SCHEMA")
    report=run(args.project_root,settings,args.output)
    print(json.dumps({k:report[k] for k in ("status","original_blocks","sampled_bin_counts","decision")}))


if __name__=="__main__":main()
