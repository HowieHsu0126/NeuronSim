"""Independently reconstruct development contrasts from original event arrays.

This replay does not import the acquisition extractor or regression helpers.
It recomputes all response numerators and the six independent interaction values.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def replay(acquisition: Path, analysis: Path, output: Path):
    design=json.loads((acquisition/"design.json").read_text());config=design["config"]
    result=json.loads((analysis/"analysis.json").read_text())
    if result["status"]!="ANALYZED_DEVELOPMENT":raise ValueError("complete development analysis required")
    with (analysis/"paired_responses.csv").open() as f: paired=list(csv.DictReader(f))
    raw={};checks={};features={};windows=config["feature_windows_ms"]
    for case in design["cases"]:
        directory=acquisition/case["case_id"]
        report=json.loads((directory/"report.json").read_text());dt=report["dt_ms"]
        with np.load(directory/"spikes.npz",allow_pickle=False) as x:
            ids=x["unit_ids"];labels=x["populations"];senders=x["spike_unit_ids"];times=x["spike_ticks"]
            labels_by_id=dict(zip(map(int,ids),labels));event_labels=np.asarray([labels_by_id[int(u)] for u in senders])
            populations=sorted(set(labels)-{"TC"})
            left=round(case["focal_arrival_ms"]/dt);right=round((case["focal_arrival_ms"]+config["response_window_ms"])/dt)
            values=[];sizes=[];state=[]
            # Source pulse-control uses the source generator emission, not TC arrival.
            emissions=case["settings"]["pulse_control"]["initial_emission_ms"]
            if case["present"]:cutoff=round(emissions[-1]/dt)
            else:
                other=next(c for c in design["cases"] if c["block"]==case["block"] and c["history"]==case["history"]
                           and c["focal_arrival_ms"]==case["focal_arrival_ms"] and c["present"])
                cutoff=round(other["settings"]["pulse_control"]["initial_emission_ms"][-1]/dt)
            for pop in populations:
                sizes.append(int(np.sum(labels==pop)))
                values.append(int(np.histogram(times[event_labels==pop],bins=[left,right])[0][0])
                              - int(np.sum((event_labels==pop)&(times==right))))
            for width in windows:
                for pop,size in zip(populations,sizes):
                    event_times=times[event_labels==pop]
                    a=np.searchsorted(event_times,cutoff-round(width/dt),side="left")
                    b=np.searchsorted(event_times,cutoff,side="left")
                    state.append((b-a)/size/(width/1000))
            count=dict(zip(*np.unique(senders,return_counts=True)))
            with np.load(directory/"spike_counts.npz") as c:
                checks[case["case_id"]+"_counters"]=all(count.get(int(u),0)==int(v) for u,v in zip(c["unit_ids"],c["native_counts"]))
            raw[(case["block"],case["focal_arrival_ms"],int(case["history"]),int(case["present"]))]=(
                np.asarray([sum(values)/sum(sizes)]+[v/n for v,n in zip(values,sizes)]),populations)
            features[(case["block"],case["focal_arrival_ms"],int(case["history"]),int(case["present"]))]=np.asarray(state)
    contrasts={};matrix=[]
    for row in paired:
        block,timepoint,history=int(row["block"]),float(row["focal_arrival_ms"]),int(row["history"])
        omitted,names=raw[(block,timepoint,history,0)];present,_=raw[(block,timepoint,history,1)]
        delta=present-omitted
        keys=["delta_cortical_spikes_per_neuron"]+["delta_"+str(x) for x in names]
        # Different algebraic reduction order permits only roundoff, not scientific tolerance.
        checks[f"response_{block}_{timepoint}_{history}"]=bool(np.allclose(delta,[float(row[k]) for k in keys],rtol=0,atol=1e-14))
        contrasts[(block,timepoint,history)]=delta
        checks[f"feature_pair_{block}_{timepoint}_{history}"]=bool(np.array_equal(features[(block,timepoint,history,0)],features[(block,timepoint,history,1)]))
        matrix.append(features[(block,timepoint,history,0)])
    with np.load(analysis/"state_probe_arrays.npz") as a:
        checks["features_exact_from_raw_events"]=bool(np.array_equal(np.asarray(matrix),a["features"]))
    block_values=[]
    for block in range(len(config["initial_state_seeds"])):
        effects=[contrasts[(block,t,1)]-contrasts[(block,t,0)] for t in config["focal_arrivals_ms"]]
        block_values.append(np.mean(effects,axis=0))
    block_values=np.asarray(block_values)
    for i,key in enumerate(keys):
        checks[key+"_independent_blocks"]=bool(np.allclose(block_values[:,i],result["effects"][key]["block_values"],rtol=0,atol=1e-14))
    # Independent augmented least-squares formulation of the cell + state fit.
    # It does not reuse the normal-equation solver or the original CV helper.
    with np.load(analysis/"state_probe_arrays.npz") as data:
        x,y,blocks,cells=(data[k] for k in ("features","target","blocks","cells"))
        saved=data["predictions"]
    def predict(train,test,alpha,with_state):
        vocab=np.unique(cells[train]);onehot=np.column_stack([cells[train]==c for c in vocab]).astype(float)
        testhot=np.column_stack([cells[test]==c for c in vocab]).astype(float)
        if not with_state:return testhot@np.linalg.lstsq(onehot,y[train],rcond=None)[0]
        means=np.vstack([x[train&(cells==c)].mean(0) for c in vocab])
        centered=x[train]-onehot@means;scale=centered.std(0);scale[scale==0]=1
        z=centered/scale;zt=(x[test]-testhot@means)/scale
        design_matrix=np.column_stack([onehot,z]);target_matrix=np.column_stack([testhot,zt])
        penalty=np.zeros((x.shape[1],len(vocab)+x.shape[1]))
        penalty[:,len(vocab):]=np.eye(x.shape[1])*np.sqrt(alpha*int(train.sum()))
        coefficients=np.linalg.lstsq(np.vstack([design_matrix,penalty]),np.r_[y[train],np.zeros(x.shape[1])],rcond=None)[0]
        return target_matrix@coefficients
    cv_predictions=np.empty_like(saved);weights=[.01,1.,100.];selected=[]
    for block in np.unique(blocks):
        outer=blocks!=block;test=~outer;scores=[]
        for alpha in weights:
            losses=[]
            for validation in np.unique(blocks[outer]):
                fit=outer&(blocks!=validation);valid=blocks==validation
                estimate=predict(fit,valid,alpha,True);losses.append(float(np.mean((estimate-y[valid])**2)))
            scores.append(float(np.mean(losses)))
        alpha=weights[int(np.argmin(scores))];selected.append(alpha)
        cv_predictions[test,0]=predict(outer,test,alpha,False)
        cv_predictions[test,1]=predict(outer,test,alpha,True)
    checks["nested_cv_predictions_independent_lstsq"]=bool(np.allclose(cv_predictions,saved,rtol=1e-10,atol=1e-12))
    checks["nested_cv_selected_ridge_independent_lstsq"]=selected==[r["selected_ridge"] for r in result["observable_state_probe"]["block_results"]]
    receipt=dict(status="PASS" if all(checks.values()) else "FAILED",checks=checks,
        original_episodes_recomputed=len(design["cases"]),paired_response_rows=len(paired),
        independent_blocks_recomputed=len(block_values),features_recomputed=int(np.asarray(matrix).size),
        nested_cv_prediction_max_absolute_error=float(np.max(np.abs(cv_predictions-saved))),
        scientific_reference_qualified=False,neuroscience_finding_qualified=False)
    output.write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition",type=Path,required=True)
    parser.add_argument("--analysis",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args();result=replay(a.acquisition,a.analysis,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!="checks"}))
    raise SystemExit(int(result["status"]!="PASS"))


if __name__=="__main__":main()
