"""Block-level pulse-history effects and observable-state development probes.

Outcome units are spikes/neuron over the declared post-TC interval. Every block
contains both histories, both focal times, and present/omitted episodes. Training
folds retain whole blocks; cell effects are unpenalized in both predictors.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from .nest_gpu_network import ticks
from .nest_gpu_spatial import Execution
from .provenance import sha256_file
from .spatial_field import PeriodicPopulationGrid

RIDGE_WEIGHTS = (.01, 1., 100.)


def array_digest(*arrays):
    result=hashlib.sha256()
    for array in arrays:
        a=np.ascontiguousarray(array)
        result.update(str(a.dtype).encode());result.update(str(a.shape).encode());result.update(a.tobytes())
    return result.hexdigest()


def _verified_events(episode, case, graph_digest):
    r=json.loads((episode/"report.json").read_text())
    if r["status"]!="EXECUTED" or not all(r["checks"].values()) or r["graph_manifest_sha256"]!=graph_digest:
        raise ValueError("completed checked native episode bound to source graph required")
    if asdict(Execution(**r["config"])) != asdict(Execution(**case["settings"])):
        raise ValueError("executed episode differs from predeclared condition")
    for name in ("spikes.npz","spike_counts.npz","states.npz","initial_parameters.npz"):
        if sha256_file(episode/name)!=r["artifacts"][name]:raise ValueError("episode artifact mismatch: "+name)
    if r["background_seed"]!=case["background_seed"] or r["initial_state_seed"]!=case["initial_state_seed"]:
        raise ValueError("initial-state/background block mismatch")
    with np.load(episode/"spikes.npz",allow_pickle=False) as x: data={k:x[k] for k in x.files}
    ids=data["unit_ids"];labels=data["populations"]
    order=np.argsort(ids);sorted_ids=ids[order]
    indices=np.searchsorted(sorted_ids,data["spike_unit_ids"])
    if np.any(indices>=len(ids)) or not np.array_equal(sorted_ids[indices],data["spike_unit_ids"]):
        raise ValueError("unknown event identity")
    indices=order[indices]
    count=np.bincount(indices,minlength=len(ids))
    with np.load(episode/"spike_counts.npz",allow_pickle=False) as x:
        take=np.searchsorted(sorted_ids,x["unit_ids"])
        if (len(take)!=len(ids) or len(np.unique(take))!=len(ids) or np.any(take>=len(ids))
                or not np.array_equal(sorted_ids[take],x["unit_ids"])
                or not np.array_equal(count[order[take]],x["native_counts"])
                or not np.array_equal(x["native_counts"],x["recorded_counts"])):
            raise ValueError("native unit counters do not match every saved event")
    dt=float(r["dt_ms"]);event_ticks=data["spike_ticks"]
    if (event_ticks.dtype.kind not in "iu" or np.any(np.diff(event_ticks)<0)
            or not np.array_equal(ticks(data["raw_spike_times_ms"],dt,recorded=True),event_ticks)):
        raise ValueError("native tick order/readback mismatch")
    return r,data,indices


def extract(acquisition: Path, graph: Path, output: Path, allow_partial=False):
    design=json.loads((acquisition/"design.json").read_text());config=design["config"]
    source=json.loads((graph/"manifest.json").read_text());digest=sha256_file(graph/"manifest.json")
    if digest!=design["graph_manifest_sha256"]:raise ValueError("acquisition graph changed")
    output.mkdir(parents=True,exist_ok=True)
    source_digest=sha256_file(acquisition/"design.json")
    identity=output/"design_binding.json"
    if identity.exists() and json.loads(identity.read_text())["acquisition_design_sha256"]!=source_digest:
        raise ValueError("feature directory belongs to different acquisition")
    identity.write_text(json.dumps(dict(acquisition_design_sha256=source_digest,
        feature_windows_ms=config["feature_windows_ms"],ridge_weights=RIDGE_WEIGHTS,
        feature_cutoff="strictly before focal source generator emission",formal_qualification=False),indent=2))
    complete=[]
    for case in design["cases"]:
        episode=acquisition/case["case_id"];target=output/(case["case_id"]+".npz");record=output/(case["case_id"]+".json")
        if not (episode/"report.json").exists():
            if allow_partial:continue
            raise ValueError("missing episode: "+case["case_id"])
        r=json.loads((episode/"report.json").read_text())
        if r["status"]!="EXECUTED":
            if allow_partial and r["status"]=="RUNNING":continue
            raise ValueError("episode did not execute: "+case["case_id"])
        if target.exists():
            cached=json.loads(record.read_text())
            if (cached["extractor_sha256"]!=sha256_file(Path(__file__))
                    or cached["episode_report_sha256"]!=sha256_file(episode/"report.json")
                    or cached["features_sha256"]!=sha256_file(target)):
                raise ValueError("cached feature inputs changed")
            complete.append(case["case_id"]);continue
        r,data,event_indices=_verified_events(episode,case,digest)
        ids,labels,xy=(data[k] for k in ("unit_ids","populations","positions_mm"))
        cortical=labels!="TC";event_cortical=cortical[event_indices]
        events=data["spike_ticks"];senders=data["spike_unit_ids"];dt=float(r["dt_ms"])
        focal=int(ticks([case["focal_arrival_ms"]],dt)[0])
        cutoff=focal-int(ticks([source["resolved_source_network"]["th_delay_pulse_generator"]],dt)[0])
        prefix=event_cortical&(events<cutoff)
        with np.load(episode/"states.npz",allow_pickle=False) as x:
            state_prefix=x["ticks"]<cutoff
            state_digest=array_digest(x["unit_ids"],x["ticks"][state_prefix],x["states"][state_prefix])
        with np.load(episode/"initial_parameters.npz",allow_pickle=False) as x:
            initial_digest=array_digest(x["unit_ids"],x["V_m_rel"])
        names=source["resolved_source_network"]["populations"][:-1]
        sizes=np.asarray([np.sum(labels==p) for p in names])
        pop_lookup={name:i for i,name in enumerate(names)}
        event_pop=np.asarray([pop_lookup.get(str(p),-1) for p in labels])[event_indices]
        features=[];feature_names=[]
        for window in config["feature_windows_ms"]:
            width=int(ticks([window],dt)[0]);take=event_cortical&(events>=cutoff-width)&(events<cutoff)
            features.extend(np.bincount(event_pop[take],minlength=8)/sizes/(window/1000))
            feature_names.extend([f"rate_{p}_{window:g}ms" for p in names])
        width=int(ticks([config["response_window_ms"]],dt)[0])
        response=event_cortical&(events>=focal)&(events<focal+width)
        pop_count=np.bincount(event_pop[response],minlength=8)
        curve_width=int(ticks([config["response_curve_ms"]],dt)[0]);bin_ticks=int(ticks([config["bin_ms"]],dt)[0])
        observer=PeriodicPopulationGrid.from_positions(ids[cortical],labels[cortical],xy[cortical],
            extent_mm=source["resolved_source_network"]["extent"],cells_per_axis=config["cells_per_axis"],
            definition="source cortical population and spatial cell")
        edges=np.arange(0,int(ticks([r["duration_ms"]],dt)[0])+1,bin_ticks)
        field=observer.project_events(senders[event_cortical],events[event_cortical],observed_unit_ids=ids[cortical],
            bin_edges_ticks=edges,dt_seconds=dt/1000).field
        if int(field.count_sum.sum())!=int(np.sum(event_cortical&(events<edges[-1]))):
            raise ValueError("spatial projection lost source events")
        curve=field.count_sum[focal//bin_ticks:(focal+curve_width)//bin_ticks]
        np.savez_compressed(target,features=np.asarray(features),feature_names=np.asarray(feature_names),
            population_response_count=pop_count,population_units=sizes,population_names=np.asarray(names),
            cortical_response_count=pop_count.sum(),cortical_units=sizes.sum(),spatial_response_curve=curve,
            field_counts=field.count_sum,field_total_units=field.total_unit_count,field_names=field.field_ids,
            field_populations=observer.population_labels,bin_edges_ticks=edges,dt_ms=dt)
        receipt=dict(status="EXTRACTED",case_id=case["case_id"],extractor_sha256=sha256_file(Path(__file__)),episode_report_sha256=sha256_file(episode/"report.json"),
            features_sha256=sha256_file(target),prefix_spike_sha256=array_digest(senders[prefix],events[prefix]),
            prefix_state_sha256=state_digest,initial_state_sha256=initial_digest,
            cutoff_tick=cutoff,cortical_spikes_before_cutoff=int(prefix.sum()),native_counters_checked=len(ids),
            source_events=len(events),features_use_future=False)
        record.write_text(json.dumps(receipt,indent=2));complete.append(case["case_id"])
    status=dict(status="EXTRACTED" if len(complete)==len(design["cases"]) else "PARTIAL",
        expected_episodes=len(design["cases"]),extracted_episodes=len(complete),cases=complete)
    (output/"extraction.json").write_text(json.dumps(status,indent=2))
    return status


def conditional_ridge(train_x, train_y, train_cell, test_x, test_cell, alpha):
    """Unpenalized four-cell effects; state normalization is training-only."""
    cells=np.unique(train_cell)
    if not set(test_cell).issubset(set(cells)):raise ValueError("unseen factorial cell")
    xm={c:train_x[train_cell==c].mean(0) for c in cells}
    ym={c:train_y[train_cell==c].mean(0) for c in cells}
    xc=train_x-np.asarray([xm[c] for c in train_cell]);yc=train_y-np.asarray([ym[c] for c in train_cell])
    scale=np.std(xc,axis=0);scale[scale==0]=1.
    z=xc/scale;zt=(test_x-np.asarray([xm[c] for c in test_cell]))/scale
    if train_x.shape[1]==0:return np.asarray([ym[c] for c in test_cell])
    weight=np.linalg.solve(z.T@z+alpha*len(z)*np.eye(z.shape[1]),z.T@yc)
    return zt@weight+np.asarray([ym[c] for c in test_cell])


def block_prediction(features, targets, blocks, cells):
    """Nested selection for the global outcome only; no CV-error CI is asserted."""
    unique=np.unique(blocks);predictions=np.empty((len(targets),2));records=[]
    for block in unique:
        train=blocks!=block;test=~train;scores=[]
        for alpha in RIDGE_WEIGHTS:
            losses=[]
            for validation in unique[unique!=block]:
                inner=train&(blocks!=validation);valid=blocks==validation
                pred=conditional_ridge(features[inner],targets[inner],cells[inner],features[valid],cells[valid],alpha)
                losses.append(float(np.mean((pred-targets[valid])**2)))
            scores.append(float(np.mean(losses)))
        alpha=RIDGE_WEIGHTS[int(np.argmin(scores))]
        plain=conditional_ridge(features[train,:0],targets[train],cells[train],features[test,:0],cells[test],1.)
        state=conditional_ridge(features[train],targets[train],cells[train],features[test],cells[test],alpha)
        predictions[test,0]=plain;predictions[test,1]=state
        records.append(dict(block=int(block),selected_ridge=alpha,inner_validation_mse=scores,
            cell_mean_mse=float(np.mean((plain-targets[test])**2)),state_mse=float(np.mean((state-targets[test])**2))))
    return predictions,records


def block_summary(values):
    values=np.asarray(values,dtype=float);n=len(values)
    if n<2 or n>8 or not np.isfinite(values).all():raise ValueError("2..8 finite independent development blocks required")
    draws=np.asarray(list(itertools.product(range(n),repeat=n)),dtype=int)
    bootstrap=values[draws].mean(axis=1)
    signs=np.asarray(list(itertools.product((-1,1),repeat=n)))
    p=float(np.mean(np.abs((signs*values).mean(axis=1)) >= abs(values.mean())-1e-14))
    return dict(block_values=values.tolist(),mean=float(values.mean()),
        empirical_block_bootstrap_percentile_95=np.quantile(bootstrap,[.025,.975]).tolist(),
        sign_symmetry_reference_p=p,independent_blocks=n,
        uncertainty_scope="development conditional on one graph; approximate small-sample percentile interval",
        sign_flip_assumption="independent sign-symmetric null, not a design-randomization or zero-mean-only test")


def analyze(acquisition: Path, extracted: Path, output: Path):
    design=json.loads((acquisition/"design.json").read_text());config=design["config"]
    if json.loads((extracted/"extraction.json").read_text())["status"]!="EXTRACTED":
        raise ValueError("complete predeclared denominator required before analysis")
    output.mkdir(parents=True,exist_ok=False)
    grouped={}
    for c in design["cases"]:grouped[(c["block"],c["focal_arrival_ms"],c["history"],c["present"])]=c
    rows=[];features=[];curves=[];block_initial={}
    for block,timepoint,history in itertools.product(range(len(config["initial_state_seeds"])),config["focal_arrivals_ms"],(False,True)):
        omitted=grouped[(block,timepoint,history,False)];present=grouped[(block,timepoint,history,True)]
        a=json.loads((extracted/(omitted["case_id"]+".json")).read_text());b=json.loads((extracted/(present["case_id"]+".json")).read_text())
        if any(a[k]!=b[k] for k in ("prefix_spike_sha256","prefix_state_sha256","initial_state_sha256","cutoff_tick")):
            raise ValueError("matched pulse omission changed observed pre-emission history")
        if block in block_initial and block_initial[block]!=a["initial_state_sha256"]:
            raise ValueError("one initial-state block contains multiple initial states")
        block_initial[block]=a["initial_state_sha256"]
        with np.load(extracted/(omitted["case_id"]+".npz")) as x,np.load(extracted/(present["case_id"]+".npz")) as y:
            if not np.array_equal(x["features"],y["features"]):raise ValueError("pre-emission features differ between arms")
            if not np.array_equal(x["field_total_units"],y["field_total_units"]):raise ValueError("observation denominator mismatch")
            delta=(y["population_response_count"]-x["population_response_count"])/x["population_units"]
            row=dict(block=block,focal_arrival_ms=timepoint,history=int(history),
                delta_cortical_spikes_per_neuron=float((y["cortical_response_count"]-x["cortical_response_count"])/x["cortical_units"]))
            for name,value in zip(x["population_names"],delta):row["delta_"+str(name)]=float(value)
            rows.append(row);features.append(x["features"].copy());curves.append(y["spatial_response_curve"]-x["spatial_response_curve"])
            feature_names=x["feature_names"].copy();field_names=x["field_names"].copy();population_names=x["population_names"].copy()
    if len(set(block_initial.values()))!=len(block_initial):raise ValueError("source initial states were not independently changed")
    keys=["delta_cortical_spikes_per_neuron"]+["delta_"+str(p) for p in population_names]
    effects={};bytime=[]
    for key in keys:
        block_values=[]
        for block in range(len(config["initial_state_seeds"])):
            interactions=[]
            for timepoint in config["focal_arrivals_ms"]:
                values={r["history"]:r[key] for r in rows if r["block"]==block and r["focal_arrival_ms"]==timepoint}
                interaction=values[1]-values[0];interactions.append(interaction)
                bytime.append(dict(outcome=key,block=block,focal_arrival_ms=timepoint,interaction=interaction))
            block_values.append(float(np.mean(interactions)))
        effects[key]=block_summary(block_values)
    blocks=np.asarray([r["block"] for r in rows]);times=config["focal_arrivals_ms"]
    cells=np.asarray([times.index(r["focal_arrival_ms"])*2+r["history"] for r in rows])
    target=np.asarray([r[keys[0]] for r in rows]);features=np.asarray(features)
    predictions,cv=block_prediction(features,target,blocks,cells)
    for i,row in enumerate(rows):row.update(prediction_cell_mean=float(predictions[i,0]),prediction_state=float(predictions[i,1]))
    for name,values in (("paired_responses.csv",rows),("interactions_by_time.csv",bytime),("prediction_by_block.csv",cv)):
        with (output/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    np.savez_compressed(output/"state_probe_arrays.npz",features=features,feature_names=feature_names,
        target=target,blocks=blocks,cells=cells,predictions=predictions,spatial_effect_curves=np.asarray(curves),field_names=field_names)
    result=dict(status="ANALYZED_DEVELOPMENT",episodes=len(design["cases"]),independent_blocks=len(block_initial),
        matched_pre_emission_pairs=len(rows),changed_initial_states=len(block_initial),effects=effects,
        observable_state_probe=dict(feature_count=features.shape[1],ridge_weights=RIDGE_WEIGHTS,
            cell_mean_mse=float(np.mean([r["cell_mean_mse"] for r in cv])),
            state_mse=float(np.mean([r["state_mse"] for r in cv])),
            block_results=cv,validation="nested leave whole initial/noise block out; unpenalized full history-time cells",
            uncertainty="descriptive CV errors; overlapping training folds do not supply an independent improvement CI"),
        source_future_noise_pairing_verified=False,scientific_reference_qualified=False,neuroscience_finding_qualified=False,
        formal_training_completed=False,acquisition_design_sha256=sha256_file(acquisition/"design.json"))
    result["artifacts"]={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}
    (output/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition",type=Path,required=True)
    parser.add_argument("--graph",type=Path,required=True)
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--allow-partial",action="store_true")
    a=parser.parse_args();status=extract(a.acquisition,a.graph,a.features,a.allow_partial)
    print(json.dumps(status),flush=True)
    if a.output is not None:
        result=analyze(a.acquisition,a.features,a.output)
        print(json.dumps({k:result[k] for k in ("status","episodes","independent_blocks","observable_state_probe")}))


if __name__=="__main__":main()
