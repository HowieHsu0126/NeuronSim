"""Read all registered history blocks and separate sampling-noise diagnostics.

No population confidence interval follows from this variance decomposition.
Incomplete receipts keep the full 32-block requirement visibly incomplete.
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
    if not condition: raise ValueError("PFC_HISTORY_SUMMARY_"+message)


def variance_components(estimates, inner_variances, population_size, *, allow_signed=False):
    values, variances = np.asarray(estimates,dtype=float), np.asarray(inner_variances,dtype=float)
    m = len(values)
    require(values.ndim == variances.ndim == 1 and values.shape == variances.shape
            and 1 < m <= population_size and np.isfinite(values).all() and np.isfinite(variances).all()
            and (allow_signed or np.all(values >= 0)) and np.all(variances >= 0), "BLOCK_DENOMINATOR_OR_VARIANCE")
    fraction = m/population_size
    sample_variance = float(np.var(values,ddof=1)); inner = float(np.mean(variances))
    latent = sample_variance-inner
    within = ((1-fraction)*sample_variance+fraction*inner)/m
    return dict(sample_denominator=m, frame_size=population_size, sample_mean=float(values.mean()),
        sampled_estimate_variance=sample_variance, mean_inner_variance=inner,
        latent_unit_window_variance_diagnostic=latent,
        within_archive_mean_variance_diagnostic=within,
        inner_mc_mean_variance_diagnostic=inner/m,
        unit_window_sampling_mean_variance_diagnostic=(1-fraction)*latent/m)


def between_components(blocks):
    require(len(blocks)>1, "MULTIPLE_ORIGINAL_BLOCKS_REQUIRED")
    means=np.array([b["sample_mean"] for b in blocks]); n=len(means)
    observed=float(np.var(means,ddof=1))
    within=float(np.mean([b["within_archive_mean_variance_diagnostic"] for b in blocks]))
    return dict(original_blocks=n,mean=float(means.mean()),observed_between_block_variance=observed,
        mean_within_archive_variance_diagnostic=within,
        latent_archived_block_mean_variance_diagnostic=observed-within,
        empirical_mean_variance_diagnostic=observed/n,
        inner_mc_mean_variance_diagnostic=sum(b["inner_mc_mean_variance_diagnostic"] for b in blocks)/n**2,
        unit_window_sampling_mean_variance_diagnostic=sum(b["unit_window_sampling_mean_variance_diagnostic"] for b in blocks)/n**2,
        negative_variance_diagnostics_preserved=True, population_confidence_interval_computed=False)


def bound_json(root, entries, relative):
    path=Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and relative in entries, "BOUND_RELATIVE_PATH")
    payload=(root/path).read_bytes(); entry=entries[relative]
    require(len(payload)==entry["bytes"] and hashlib.sha256(payload).hexdigest()==entry["sha256"],
            "ARTIFACT_CHANGED:"+relative)
    return json.loads(payload)


def read_block(root, seed, settings, registry):
    delivery=json.loads((root/"delivery.json").read_text())
    require(delivery["verified"] and delivery["status"]==settings.get("wrapper_status", "COMPLETE_PFC_FIRST_PASSAGE_HISTORY_EXECUTE"), "COMPLETE_BLOCK_REQUIRED")
    entries={r["path"]:r for r in delivery["files"]}
    source_module="Libs/population_dynamics/pfc_first_passage_history.py"
    require(entries[source_module]["sha256"]==settings["source_history_implementation_sha256"], "QUALIFIED_HISTORY_IMPLEMENTATION")
    wrapper=bound_json(root,entries,"report.json")
    result=bound_json(root,entries,"validation/report.json")
    selected=bound_json(root,entries,"validation/selected_frame.json")
    rows=bound_json(root,entries,"validation/integrated_frame.json")
    require(wrapper["exit_code"]==0 and result["status"]=="COMPLETE_PFC_FIRST_PASSAGE_HISTORY_PILOT"
            and result["source_rng_seed"]==seed, "SOURCE_BLOCK_IDENTITY")
    frame=selected["frame"]
    require(frame==result["frame"] and frame["sample_size"]==settings["sample_size"]==len(rows)==len(selected["rows"])
            and frame["frame_size"]==settings["frame_size"] and frame["field_index"]==settings["field_index"]
            and frame["phase"]==settings["phase"] and frame["window_ms"]==settings["window_ms"]
            and frame["seed_components"]==[20261009,seed,5,1], "COMPLETE_REGISTERED_SAMPLE_FRAME")
    rng=np.random.RandomState(np.asarray(frame["seed_components"],dtype=np.uint32))
    ids=rng.choice(frame["frame_size"],size=frame["sample_size"],replace=False)
    require(ids.tolist()==[r["frame_index"] for r in selected["rows"]], "REGISTERED_RANDOM_SELECTION")
    values=[]; variances=[]; nonzero=0; eligible_zero=0; inner_trials=0
    for original,row in zip(selected["rows"],rows):
        require(all(row[k]==v for k,v in original.items())
                and row["window_start_tick"]*settings["dt_ms"]==row["window_start_ms"]
                and row["window_stop_tick"]*settings["dt_ms"]==row["window_stop_ms"], "PHYSICAL_WINDOW_IDENTITY")
        category=row["category"]; value=row["conditional_estimate"]; variance=row["inner_monte_carlo_variance"]
        require(0 <= value <= 4 and variance >= 0, "PROBABILITY_SCORE_SUPPORT")
        if category in ("conditional_integration_required","suppressed_network_reference_required"):
            if category=="suppressed_network_reference_required":
                require(row.get("suppressed_reference_resolved") is True and row["observed_count"]>=2,
                        "UNRESOLVED_MULTISPIKE_HISTORY")
                replay=bound_json(root,entries,"validation/"+row["suppressed_reference_report"])
                require(replay["status"]=="COMPLETE_PFC_HISTORY_SUPPRESSED_REFERENCE"
                        and replay["source_rng_seed"]==seed and replay["source_reference"]==registry
                        and replay["dt_ms"]==settings["dt_ms"]
                        and replay["network_feedback_recomputed"] is True and replay["source_law_trajectory"] is False
                        and replay["observed_focal_window_ticks"]==row["observed_window_ticks"]
                        and replay["unit_index"]==row["unit_index"]
                        and replay["first_spike_tick"]==row["first_spike_tick"]
                        and replay["window_stop_tick"]==row["window_stop_tick"], "SUPPRESSED_REFERENCE_BINDING")
            inner=bound_json(root,entries,"validation/"+row["integration_report"])
            require(inner["status"]=="COMPLETE_CONDITIONAL_FIRST_PASSAGE_IMPORTANCE"
                    and inner["samples"]==settings["importance_samples"]
                    and inner["reference_archive"]==registry["archive"]
                    and inner["reference_delivery_sha256"]==registry["delivery_sha256"]
                    and inner["seed_components"]==[20261009,seed,settings["field_index"],row["frame_index"],round(settings["dt_ms"]*1000),2],
                    "INNER_SOURCE_OR_BUDGET")
            score=inner["stopped_prefix"]
            require(value==score["estimate"] and variance==score["empirical_score_variance"]/inner["samples"]
                    and row["proposal_hits"]==inner["proposal_hits"], "INNER_SCORE_ARITHMETIC")
            # Historical sample_frame_row is a pre-integration identity snapshot.
            # Authoritative numerical scores are stopped_prefix and integrated_frame.
            require(all(inner["sample_frame_row"][k]==v for k,v in original.items()), "INNER_SAMPLE_IDENTITY")
            nonzero+=int(value>0); eligible_zero+=int(value==0); inner_trials+=inner["samples"]
        else:
            require(category in ("no_first_spike","refractory_covers_remaining_window")
                    and value==variance==row["proposal_hits"]==0, "STRUCTURAL_ZERO_OR_MISSING_REFERENCE")
            require((category=="no_first_spike" and row["observed_count"]==0 and row["first_spike_tick"] is None)
                    or (category=="refractory_covers_remaining_window" and row["observed_count"]==1
                        and row["first_spike_tick"]+row["refractory_ticks"]>=row["window_stop_tick"]), "STRUCTURAL_ZERO_REASON")
        values.append(value); variances.append(variance)
    stats=variance_components(values,variances,frame["frame_size"])
    require(stats["sample_mean"]==result["sample_mean"] and dict(Counter(r["category"] for r in rows))==result["categories"],
            "BLOCK_SUMMARY_ARITHMETIC")
    require(all(result[k] is False for k in ("confirmation_execution_released","population_history_uncertainty_established","scientific_fidelity_qualified")), "UNRELEASED_SCIENTIFIC_SCOPE")
    return dict(source_rng_seed=seed,**stats,integrated_histories=result["integrated_histories"],
        nonzero_conditional_estimates=nonzero,eligible_zero_estimates=eligible_zero,inner_trials=inner_trials,
        no_first_spike=result["categories"].get("no_first_spike",0),
        refractory_zero=result["categories"].get("refractory_covers_remaining_window",0),
        reconstruction_seconds=result["reconstruction_seconds"],analysis_seconds=result["wall_seconds"],
        delivery_sha256=hashlib.sha256((root/"delivery.json").read_bytes()).hexdigest(),archive=str(root))


def run(project,settings,output,allow_partial=False):
    output.mkdir(parents=True,exist_ok=False)
    registry_path=project/settings["source_registry"]["path"]
    payload=registry_path.read_bytes()
    require(hashlib.sha256(payload).hexdigest()==settings["source_registry"]["sha256"], "SOURCE_REGISTRY_CHANGED")
    registry={b["source_rng_seed"]:b["coarse_references"]["strong_cue"] for b in yaml.safe_load(payload)["blocks"]}
    seeds=settings["required_source_seeds"]
    require(seeds==list(range(95001,95033)), "ALL32_ORIGINAL_BLOCKS_REQUIRED")
    blocks=[];missing=[]
    for seed in seeds:
        relative=settings["pilot_archives"].get(str(seed))
        if relative is None:relative=settings["array_archive_prefix"]+str(seed-95001)
        root=project/relative
        if not (root/"delivery.json").exists():missing.append(seed);continue
        block=read_block(root,seed,settings,registry[seed]);block["archive"]=relative;blocks.append(block)
    with (output/"block_diagnostics.csv").open("w",newline="") as stream:
        if blocks:
            writer=csv.DictWriter(stream,fieldnames=list(blocks[0]));writer.writeheader();writer.writerows(blocks)
    complete=not missing
    report=dict(status="COMPLETE_PFC_HISTORY_BLOCK_DIAGNOSTICS" if complete else "PARTIAL_PFC_HISTORY_BLOCK_DIAGNOSTICS",
        required_original_blocks=32,available_original_blocks=len(blocks),missing_receipt_source_seeds=missing,
        missing_receipt_is_terminal_job_evidence=False,unit_windows_per_block=settings["sample_size"],
        sampled_unit_windows=sum(b["sample_denominator"] for b in blocks),
        integrated_histories=sum(b["integrated_histories"] for b in blocks),inner_trials=sum(b["inner_trials"] for b in blocks),
        nonzero_conditional_estimates=sum(b["nonzero_conditional_estimates"] for b in blocks),
        eligible_zero_estimates=sum(b["eligible_zero_estimates"] for b in blocks),
        statistics=between_components(blocks) if complete else None,
        variance_scope="Nominal independent inner randomization plus simple random sampling without replacement within each fixed archived block. Corrected between-block quantity estimates the sample variance of latent archived-block means; common population-history variance additionally needs a common block law.",
        population_confidence_interval_computed=False,scientific_fidelity_qualified=False,
        confirmation_execution_released=False,production_episodes=0,new_independent_source_blocks=0)
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    require(complete or allow_partial,"INCOMPLETE32_BLOCK_RECEIPTS")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","project-root","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--allow-partial",action="store_true")
    args=parser.parse_args();settings=yaml.safe_load(args.config.read_text())
    require(settings["schema"]=="pfc_first_passage_history_summary_v1","CONFIG_SCHEMA")
    result=run(args.project_root,settings,args.output,args.allow_partial)
    print(json.dumps({k:result[k] for k in ("status","available_original_blocks","sampled_unit_windows","statistics")}))


if __name__=="__main__":main()
