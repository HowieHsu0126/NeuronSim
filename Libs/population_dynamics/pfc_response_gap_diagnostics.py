"""Decompose saved PFC numerical-budget gaps over the original paired blocks."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import numpy as np
import yaml
from pfc_coupled_clock_block_diagnostics import Archive, require, write_csv
from pfc_gain_clock_comparison import arrays, normalize_response, digest, SEEDS, CLOCKS, CONDITIONS, PHASES, METRICS, BOUNDS


def decompose(values,margin,seeds):
    x=np.asarray(values,dtype=float);require(x.shape==(32,) and np.isfinite(x).all(),"32 finite paired blocks required")
    require(np.isfinite(margin) and margin>0 and tuple(seeds)==SEEDS,"frozen positive margin and block identities required")
    mean=float(x.mean());residual=x-mean;squared=residual**2;total=float(squared.sum())
    leave=(x.sum()-x)/31
    fractions=squared/total if total else np.zeros_like(x)
    outside=abs(mean)>margin
    rows=[dict(source_rng_seed=int(seed),signed_block_score=float(value),contribution_to_full_mean=float(value/32),
        centered_squared_share=float(frac),mean_if_block_omitted=float(loo),omitted_mean_outside_margin=bool(abs(loo)>margin))
        for seed,value,frac,loo in zip(seeds,x,fractions,leave)]
    summary=dict(estimate=mean,margin=float(margin),point_outside_margin=outside,block_sd=float(x.std(ddof=1)),
        minimum_block_score=float(x.min()),maximum_block_score=float(x.max()),
        exactly_nonzero_blocks=int(np.count_nonzero(x)),positive_blocks=int((x>0).sum()),negative_blocks=int((x<0).sum()),
        maximum_centered_squared_share=float(fractions.max()),
        maximum_share_source_rng_seeds=";".join(str(s) for s,f in zip(seeds,fractions) if f==fractions.max()),
        minimum_mean_if_block_omitted=float(leave.min()),maximum_mean_if_block_omitted=float(leave.max()),
        omitted_point_decision_changes=int(np.count_nonzero((np.abs(leave)>margin)!=outside)))
    return summary,rows


def primitives(identity,family,responses,fields):
    left=CLOCKS.index(identity["left_cohort"]);right=CLOCKS.index(identity["right_cohort"])
    phase=PHASES.index(identity["phase"])
    if family=="field_mean":
        condition=identity["condition"];field=int(identity["field_index"])
        times=fields["time_start_ms"]
        mask=(times>=BOUNDS[phase,0]) & (times<BOUNDS[phase,1])
        require(np.count_nonzero(mask)==int((BOUNDS[phase,1]-BOUNDS[phase,0])/5),"field phase coverage differs")
        a=fields[identity["left_cohort"]+"_"+condition][:,mask,field].mean(axis=1)
        b=fields[identity["right_cohort"]+"_"+condition][:,mask,field].mean(axis=1)
        return a-b,dict(left_positive=a,right_positive=b,left_negative=np.zeros(32),right_negative=np.zeros(32))
    metric=METRICS.index(identity["metric"])
    if family=="source_response":positive=CONDITIONS.index(identity["condition"]);negative=None
    else:
        pos,neg=identity["contrast"].split("_minus_")
        positive=CONDITIONS.index(pos);negative=CONDITIONS.index(neg)
    a=responses[:,left,positive,phase,metric];b=responses[:,right,positive,phase,metric]
    c=np.zeros(32) if negative is None else responses[:,left,negative,phase,metric]
    d=np.zeros(32) if negative is None else responses[:,right,negative,phase,metric]
    return (a-c)-(b-d),dict(left_positive=a,right_positive=b,left_negative=c,right_negative=d)


def run(config_path,project,output):
    config=yaml.safe_load(Path(config_path).read_text());output=Path(output);require(not output.exists(),"output must be new")
    require(config["schema"]=="pfc_response_gap_block_diagnostics_v1" and config["maximum_confirmation_blocks"]==128
        and config["source_simulations"]==config["new_rng_draws"]==0 and not config["confirmation_execution_released"],"unchanged diagnostic scope required")
    bindings={};sources={k:Archive(project/v["archive"],bindings,v["delivery_sha256"]) for k,v in config["inputs"].items()}
    report=sources["gaps"].json("analysis/report.json")
    require(report["frozen_margin_sha256"]==config["frozen_margin_sha256"] and report["planning_blocks"]==128,"frozen budget differs")
    gaps=sources["gaps"].csv("analysis/projected_exceedances.csv")
    require(len(gaps)==config["expected_gaps"] and dict(Counter(r["family"] for r in gaps))==config["expected_family_counts"],"complete33-gap family set required")
    scores=arrays(sources["scores"].read("analysis/coupled_block_scores.npz"))
    require(scores["source_rng_seeds"].tolist()==list(SEEDS) and scores["domain_row_indices"].tolist()==list(range(84330))
        and scores["scores"].shape==(84330,32),"complete frozen score bank required")
    responses=normalize_response(arrays(sources["scores"].read("analysis/coupled_responses.npz")))
    fields=arrays(sources["scores"].read("analysis/coupled_field_rates.npz"))
    require(fields["source_rng_seeds"].tolist()==list(SEEDS) and fields["field_indices"].tolist()==list(range(17))
        and fields["field_unit_counts"].tolist()==[5000]*16+[20000] and fields["units"].item()=="Hz"
        and fields["bin_ms"].item()==5 and np.array_equal(fields["time_start_ms"],np.arange(1000)*5),"physical field axes differ")
    summaries=[];block_rows=[];maximum_error=0.0
    for gap in gaps:
        row_id=int(gap["domain_row_index"]);identity=json.loads(gap["endpoint_identity"]);x=scores["scores"][row_id]
        expected,parts=primitives(identity,gap["family"],responses,fields)
        error=float(np.max(np.abs(x-expected)));maximum_error=max(maximum_error,error)
        require(np.allclose(x,expected,atol=1e-12,rtol=1e-10),"stored scores do not reconstruct from source primitives")
        require(abs(float(x.mean())-float(gap["calibration_estimate"]))<=1e-12,"saved point estimate differs")
        summary,rows=decompose(x,float(gap["margin"]),SEEDS)
        metadata={k:gap[k] for k in ("domain_row_index","family","endpoint_identity","planning_lower","planning_upper","planning_status")}
        summaries.append(dict(metadata,**summary))
        for b,row in enumerate(rows):block_rows.append(dict(metadata,**row,**{k:float(v[b]) for k,v in parts.items()}))
    output.mkdir(parents=True)
    write_csv(output/"gap_block_contributions.csv",block_rows);write_csv(output/"gap_block_summary.csv",summaries)
    result=dict(status="COMPLETE_SAVED_NONCDF_GAP_BLOCK_DIAGNOSTICS",job_id=os.environ.get("SLURM_JOB_ID"),
        finished_utc=datetime.now(timezone.utc).isoformat(),config_sha256=digest(config_path),implementation_sha256=digest(__file__),
        input_sha256=bindings,independent_blocks=32,gap_rows=len(summaries),block_rows=len(block_rows),
        primitive_score_values_verified=len(block_rows),maximum_primitive_score_error=maximum_error,
        family_counts=dict(Counter(r["family"] for r in gaps)),
        point_outside_count=sum(r["point_outside_margin"] for r in summaries),
        source_simulations=0,new_rng_draws=0,bootstrap_recomputed=False,margin_changed=False,
        scientific_fidelity_qualified=False,confirmation_execution_released=False,production_episodes=0,
        interpretation=config["interpretation"],outputs={n:dict(bytes=(output/n).stat().st_size,sha256=digest(output/n)) for n in
            ("gap_block_contributions.csv","gap_block_summary.csv")})
    (output/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("config","project","output"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();r=run(a.config,a.project,a.output);print(json.dumps({k:v for k,v in r.items() if not isinstance(v,dict)}))


if __name__=="__main__":main()
