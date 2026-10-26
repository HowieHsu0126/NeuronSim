"""Paired coarse/fine conditional-history contrasts on identical physical windows.

The event is N>=2. Its fine-minus-coarse contrast has the opposite sign to
F_fine(1)-F_coarse(1). Third-spike/CDF-at-two contrasts are not estimated here.
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

from .pfc_first_passage_history_summary import read_block, variance_components, between_components

IDENTITY = ("sample_index","frame_index","unit_index","window_start_ms","window_stop_ms")


def require(condition,message):
    if not condition:raise ValueError("PFC_CLOCK_HISTORY_"+message)


def paired_block(coarse, fine, population_size):
    require(len(coarse)==len(fine)>1 and len({r["frame_index"] for r in coarse})==len(coarse), "PAIR_DENOMINATOR")
    rows=[];differences=[];variances=[];categories=Counter()
    for c,f in zip(coarse,fine):
        require(all(c[k]==f[k] for k in IDENTITY), "PHYSICAL_WINDOW_PAIRING")
        delta=f["conditional_estimate"]-c["conditional_estimate"]
        variance=f["inner_monte_carlo_variance"]+c["inner_monte_carlo_variance"]
        differences.append(delta);variances.append(variance)
        categories[c["category"]+" -> "+f["category"]]+=1
        rows.append({**{k:c[k] for k in IDENTITY},
            "coarse_category":c["category"],"fine_category":f["category"],
            "coarse_first_spike_tick":c["first_spike_tick"],"fine_first_spike_tick":f["first_spike_tick"],
            "coarse_estimate":c["conditional_estimate"],"fine_estimate":f["conditional_estimate"],
            "fine_minus_coarse_count_ge2":delta,"fine_minus_coarse_cdf_le1":-delta,
            "inner_variance_of_difference":variance})
    statistics=variance_components(differences,variances,population_size,allow_signed=True)
    statistics["fine_minus_coarse_cdf_le1"]=-statistics["sample_mean"]
    statistics["coarse_mean"]=float(np.mean([r["conditional_estimate"] for r in coarse]))
    statistics["fine_mean"]=float(np.mean([r["conditional_estimate"] for r in fine]))
    tolerance=64*np.finfo(float).eps*(abs(statistics["coarse_mean"])+abs(statistics["fine_mean"]))
    require(abs(statistics["sample_mean"]-(statistics["fine_mean"]-statistics["coarse_mean"]))<=tolerance,
            "PAIRED_MEAN_ARITHMETIC")
    return statistics,rows,dict(categories)


def archive_path(spec,seed):
    return spec["pilot_archives"].get(str(seed),spec["array_archive_prefix"]+str(seed-95001))


def source_registry(project,spec):
    ref=spec["source_registry"];payload=(project/ref["path"]).read_bytes()
    require(hashlib.sha256(payload).hexdigest()==ref["sha256"],"SOURCE_REGISTRY_CHANGED")
    data=yaml.safe_load(payload)
    if ref["kind"]=="coarse_references":
        return {r["source_rng_seed"]:r["coarse_references"]["strong_cue"] for r in data["blocks"]}
    require(ref["kind"]=="fine_source_blocks","SOURCE_REGISTRY_KIND")
    return {r["source_rng_seed"]:r["source"] for r in data["source_blocks"]}


def write_csv(path,rows):
    with path.open("w",newline="") as stream:
        if rows:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(project,settings,output,allow_partial=False):
    output.mkdir(parents=True,exist_ok=False)
    seeds=settings["required_source_seeds"];require(seeds==list(range(95001,95033)),"ALL32_BLOCK_PAIRS_REQUIRED")
    registries={clock:source_registry(project,spec) for clock,spec in settings["clocks"].items()}
    require(set(registries)=={"coarse","fine"},"REGISTERED_CLOCK_PAIR")
    paired=[];windows=[];missing=[];transitions=Counter();inputs=[]
    for seed in seeds:
        paths={clock:project/archive_path(spec,seed) for clock,spec in settings["clocks"].items()}
        absent=[clock for clock,path in paths.items() if not (path/"delivery.json").exists()]
        if absent:missing.append(dict(source_rng_seed=seed,missing_clock_receipts=absent));continue
        measurements={};blocks={}
        for clock,path in paths.items():
            spec=settings["clocks"][clock];options={**settings,**spec}
            digest=hashlib.sha256((path/"delivery.json").read_bytes()).hexdigest()
            if str(seed) in spec["pilot_delivery_sha256"]:
                require(digest==spec["pilot_delivery_sha256"][str(seed)],"PILOT_DELIVERY_CHANGED")
            blocks[clock]=read_block(path,seed,options,registries[clock][seed])
            # Files are already bound by read_block; this local staged copy is immutable.
            measurements[clock]=json.loads((path/"validation/integrated_frame.json").read_text())
            inputs.append(dict(source_rng_seed=seed,clock=clock,archive=archive_path(spec,seed),delivery_sha256=digest))
        fine_report=json.loads((paths["fine"]/"validation/report.json").read_text())
        require(fine_report["paired_physical_frame_equal"]
                and fine_report["paired_coarse"]["archive"]==archive_path(settings["clocks"]["coarse"],seed)
                and fine_report["paired_coarse"]["delivery_sha256"]==blocks["coarse"]["delivery_sha256"],
                "FINE_REFERENCE_COARSE_FRAME_BINDING")
        stats,window_rows,categories=paired_block(measurements["coarse"],measurements["fine"],settings["frame_size"])
        require(stats["coarse_mean"]==blocks["coarse"]["sample_mean"] and stats["fine_mean"]==blocks["fine"]["sample_mean"],
                "MARGINAL_BLOCK_MEANS")
        paired.append(dict(source_rng_seed=seed,**stats))
        windows.extend(dict(source_rng_seed=seed,**row) for row in window_rows);transitions.update(categories)
    write_csv(output/"paired_block_diagnostics.csv",paired);write_csv(output/"paired_unit_windows.csv",windows)
    complete=not missing
    statistics=between_components(paired) if complete else None
    if statistics is not None:statistics["fine_minus_coarse_cdf_le1_mean"]=-statistics["mean"]
    report=dict(status="COMPLETE_PFC_PAIRED_HISTORY_DIAGNOSTICS" if complete else "PARTIAL_PFC_PAIRED_HISTORY_DIAGNOSTICS",
        required_original_block_pairs=32,available_original_block_pairs=len(paired),missing_receipts=missing,
        paired_unit_windows=len(windows),category_transitions=dict(transitions),statistics=statistics,inputs=inputs,
        coarse_dt_ms=settings["clocks"]["coarse"]["dt_ms"],fine_dt_ms=settings["clocks"]["fine"]["dt_ms"],
        field_index=settings["field_index"],phase=settings["phase"],window_ms=settings["window_ms"],
        event_contrast="fine_minus_coarse_probability_count_ge2",
        cdf_contrast="fine_minus_coarse_CDF_at_count1_equals_negative_event_contrast",
        cdf_at_count2_contrast_estimated=False,third_spike_probability_established=False,
        auxiliary_independence_scope="Nominal independent inner randomization across IDs and clocks; distinct seeds do not certify finite-PRNG independence.",
        history_scope="Paired original blocks are the history units; windows/auxiliary trials do not add independent source blocks.",
        population_confidence_interval_computed=False,clock_equivalence_qualified=False,
        scientific_fidelity_qualified=False,confirmation_execution_released=False,production_episodes=0,new_independent_source_blocks=0)
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    require(complete or allow_partial,"INCOMPLETE32_BLOCK_PAIRS")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","project-root","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--allow-partial",action="store_true")
    args=parser.parse_args();settings=yaml.safe_load(args.config.read_text())
    require(settings["schema"]=="pfc_first_passage_clock_comparison_v1","CONFIG_SCHEMA")
    result=run(args.project_root,settings,args.output,args.allow_partial)
    print(json.dumps({k:result[k] for k in ("status","available_original_block_pairs","paired_unit_windows","statistics")}))


if __name__=="__main__":main()
