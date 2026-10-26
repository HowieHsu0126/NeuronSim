"""Compare auxiliary PFC refinements within the unchanged 32 original blocks."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import yaml

from . import pfc_count_histograms, pfc_coupled_precision as precision
from .pfc_coupled_clock_comparison import DT, LABELS, cdf_on_support, pilot_case
from .pfc_coupled_clock_execution import (
    REFINEMENT_REPEAT_SCOPE, case_key, select_settings,
)
from .pfc_coupled_precision_comparison import (
    SEEDS, SIZES, UNDEFINED, bound_analysis, output_file, read_rows, export_arrays,
)
from .pfc_delay_analysis import PHASES, require, write_rows
from .pfc_phase_responses import CONDITIONS, METRICS
from .pfc_scale_analysis import _source_manifest, _source_bound_file, _source_pairing
from .provenance import freeze_json, sha256_file

SCHEMA = "pfc_refinement_repeat_comparison_v1"
BLOCK_STATUS = "COMPLETE_PFC_REFINEMENT_REPEAT_BLOCK_READOUT"
IDENTITY_KEYS = ("domain_row_index", "family", "endpoint_identity", "definition_status", "margin", "units")


def configuration(path, root):
    settings = yaml.safe_load(path.read_text())
    require(settings["schema"] == SCHEMA and settings["source_rng_seeds"] == SEEDS
            and settings["new_original_independent_blocks"] == 0
            and settings["maximum_confirmation_blocks"] == 128
            and settings["confirmation_execution_released"] is False,
            "UNCHANGED_REFINEMENT_REPEAT_ANALYSIS_SCOPE_REQUIRED")
    protocol_path = root/settings["protocol"]["path"]
    require(sha256_file(protocol_path) == settings["protocol"]["sha256"], "REPEAT_PROTOCOL_CHANGED")
    protocol = yaml.safe_load(protocol_path.read_text())
    # Reuse the actual source runner's complete registration, including auxiliary seeds.
    blocks = [select_settings(protocol, seed) for seed in SEEDS]
    require([r["source_rng_seed"] for r in settings["source_plan"]] == SEEDS
            and len({r["archive"] for r in settings["source_plan"]}) == 32,
            "COMPLETE_UNIQUE_PLANNED_SOURCE_BLOCKS_REQUIRED")
    return settings, blocks


def original_bank(root, settings):
    path, report = bound_analysis(root, settings["original"])
    require(report["status"] == "COMPLETE_FIXED_BANK_COUPLED_PRECISION_COMPARISON"
            and report["row_count"] == 84330 and report["original_independent_blocks"] == 32
            and report["core_implementation_sha256"] == sha256_file(Path(precision.__file__)),
            "MATCHING_ORIGINAL_STATISTICS_REQUIRED")
    table = read_rows(output_file(path, report, "endpoint_precision_comparison.csv"))
    budget, budget_report = bound_analysis(root, report["settings"]["budget"])
    margin_path = output_file(budget, budget_report, "frozen_numerical_margins.csv")
    require(sha256_file(margin_path) == settings["frozen_margin_sha256"], "FROZEN_MARGINS_CHANGED")
    margins = read_rows(margin_path)
    require(len(table) == len(margins) == 84330, "ALL_FROZEN_ENDPOINTS_REQUIRED")
    for i, (row, margin) in enumerate(zip(table, margins)):
        require(int(row["domain_row_index"]) == i and all(row[k] == v for k, v in margin.items()),
                "ORIGINAL_ENDPOINT_OR_MARGIN_CHANGED")
        margin["physical_maximum_count"] = row["physical_maximum_count"]
    data = {}
    for key, filename in (
        ("fields", "coupled_field_rates.npz"), ("responses", "coupled_responses.npz"),
        ("cdfs", "coupled_count_cdfs.npz"), ("scores", "coupled_block_scores.npz"),
    ):
        with np.load(output_file(path, report, filename), allow_pickle=False) as saved:
            require(np.array_equal(saved["source_rng_seeds"], SEEDS), "ORIGINAL_BLOCK_ORDER_CHANGED")
            data[key] = {k: saved[k].copy() for k in saved.files}
    for key in ("fields", "cdfs"):
        require(np.array_equal(data[key]["field_unit_counts"], SIZES)
                and np.array_equal(data[key]["field_indices"], np.arange(17)), "ORIGINAL_FIELD_AXIS_CHANGED")
    for key in ("fields", "responses", "cdfs"):
        require(np.array_equal(data[key]["condition_names"], CONDITIONS), "ORIGINAL_CONDITION_AXIS_CHANGED")
    require(np.array_equal(data["responses"]["metric_names"], METRICS)
            and np.array_equal(data["responses"]["phase_names"], list(PHASES))
            and np.array_equal(data["cdfs"]["phase_names"], list(PHASES))
            and np.array_equal(data["cdfs"]["window_ms"], [5, 50, 100])
            and np.array_equal(data["fields"]["time_start_ms"], np.arange(1000)*5)
            and data["fields"]["bin_ms"].item() == 5
            and data["fields"]["units"].item() == data["responses"]["units"].item() == "Hz"
            and data["cdfs"]["units"].item() == "probability", "ORIGINAL_OBSERVATION_MAP_CHANGED")
    fields = {l: {c: data["fields"][l+"_"+c] for c in CONDITIONS} for l in LABELS}
    responses = {l: data["responses"][l+"_values"] for l in LABELS}
    cdfs = {l: data["cdfs"][l+"_cdf"] for l in LABELS}
    support = data["cdfs"]["count_support"]
    precision._inputs(fields, responses, cdfs, support, SEEDS)
    scores = data["scores"]["scores"]
    require(scores.shape == (84330, 32)
            and np.array_equal(data["scores"]["domain_row_indices"], np.arange(84330)),
            "COMPLETE_ORIGINAL_SCORE_BANK_REQUIRED")
    for i, row in enumerate(table):
        if row["definition_status"] == UNDEFINED:
            require(np.isnan(scores[i]).all(), "UNDEFINED_ORIGINAL_SCORE_CHANGED")
        else:
            require(np.isfinite(scores[i]).all()
                    and np.isclose(scores[i].mean(), float(row["coupled_signed_estimate"]), rtol=1e-10, atol=1e-12),
                    "ORIGINAL_SCORE_POINT_IDENTITY_FAILED")
    return margins, table, fields, responses, cdfs, support, scores


def check_wrapper(delivery, wrapper, entry):
    require(delivery["status"] == wrapper["status"] == "COMPLETE_COUPLED_REFINEMENT_REPEAT_BLOCK"
            and str(delivery["job_id"]) == str(wrapper["job_id"]) == str(entry["job"])
            and wrapper["exit_code"] == 0 and wrapper["tests_only"] is False
            and int(wrapper["selected_source_rng_seed"]) == entry["source_rng_seed"]
            and wrapper["expected_new_development_episodes"] == 6
            and wrapper["original_independent_blocks"] == wrapper["new_original_independent_blocks"] == 0
            and wrapper["reused_original_independent_blocks"] == 1
            and wrapper["auxiliary_repeat_index"] == 1
            and wrapper["confirmation_execution_released"] is False,
            "COMPLETE_BOUND_REPEAT_WRAPPER_REQUIRED")
    for phase in ("tests_reused", "source_input_validation", "source_execution"):
        matching = [r for r in wrapper["commands"] if r["phase"] == phase]
        require(len(matching) == 1 and matching[0]["exit_code"] == 0, "SUCCESSFUL_SOURCE_PHASE_REQUIRED:"+phase)


def check_case(report, block, settings):
    require(report["scope"] == REFINEMENT_REPEAT_SCOPE
            and report["settings_sha256"] == settings["protocol"]["sha256"]
            and report["source_rng_seed"] == block["source_rng_seed"]
            and report["refinement_seeds"] == block["refinement_seeds"]
            and report["original_refinement_seeds"] == block["original_refinement_seeds"]
            and report["coarse_reference"] == block["coarse_references"][report["condition"]]
            and report["code"] == settings["source_engine_sha256"]
            and report["auxiliary_repeat_index"] == 1
            and report["original_independent_blocks"] == report["new_original_independent_blocks"] == 0
            and report["reused_original_independent_blocks"] == 1
            and report["refinement_seeds_independent_units"] is False
            and report["production_episodes"] == 0
            and report["confirmation_execution_released"] is False
            and report["scientific_fidelity_qualified"] is False,
            "REPEAT_SOURCE_IDENTITY_CHANGED")


def block_readout(root, settings, block, output):
    seed = block["source_rng_seed"]
    entry = settings["source_plan"][SEEDS.index(seed)]
    archive = root/entry["archive"]
    delivery = json.loads((archive/"delivery.json").read_text())
    manifest = _source_manifest(delivery)
    wrapper = json.loads(_source_bound_file(archive, manifest, "report.json").read_text())
    check_wrapper(delivery, wrapper, entry)
    output.mkdir(parents=True, exist_ok=False)
    fields, responses, cdfs, supports, cases, pairing = {}, {}, {}, [], [], []
    for label, dt in zip(LABELS[1:], DT[1:]):
        histogram, pairs, registered = {}, {}, {}
        fields[label], responses[label] = {}, np.empty((3, 5, 8))
        for ci, condition in enumerate(CONDITIONS):
            relative = Path("source_cases")/case_key(dt, condition, seed)
            source = json.loads(_source_bound_file(archive, manifest, str(relative/"report.json")).read_text())
            check_case(source, block, settings)
            rates, values, pair, info = pilot_case(archive, manifest, relative, seed, condition, dt, SIZES, histogram, ci)
            fields[label][condition], responses[label][ci] = rates, values
            case = f"pfc_{seed}_{condition}"
            pairs[case], registered[case] = pair, (seed, condition)
            cases.append(dict(source_rng_seed=seed, **info))
        checks = _source_pairing(pairs, registered, [seed])
        flags = ("exact_pre_cue_events", "exact_initial_voltage", "exact_final_source_rng", "same_source_input_consumption")
        require(all(all(row[k] for k in flags) for row in checks), "REPEAT_CONDITION_PAIRING_FAILED")
        pairing.extend(dict(cohort=label, **row) for row in checks)
        hpath = output/(label+"_histograms.npz")
        pfc_count_histograms.write(hpath, histogram, conditions=CONDITIONS, seeds=[seed], phases=PHASES,
                                  windows_ms=[5,50,100], field_sizes=SIZES)
        with np.load(hpath, allow_pickle=False) as saved:
            coordinates, observations = saved["coordinates"], saved["observations"]
        support = np.arange(int(coordinates[:,5].max())+1, dtype=np.int64)
        supports.append(support)
        cdfs[label] = pfc_count_histograms.cdf(coordinates, observations, support, (3,1,5,3,17))[:,0]
    common = np.arange(max(x[-1] for x in supports)+1, dtype=np.int64)
    np.savez_compressed(output/"block_readout.npz", source_rng_seeds=[seed], field_unit_counts=SIZES,
        condition_names=CONDITIONS, phase_names=list(PHASES), metric_names=METRICS,
        time_start_ms=np.arange(1000)*5, window_ms=[5,50,100], count_support=common,
        **{l+"_"+c: fields[l][c] for l in LABELS[1:] for c in CONDITIONS},
        **{l+"_values": responses[l] for l in LABELS[1:]},
        **{l+"_cdf": cdf_on_support(cdfs[l], s, common) for l,s in zip(LABELS[1:],supports)})
    write_rows(output/"shared_input_checks.csv", pairing)
    return dict(status=BLOCK_STATUS, source_rng_seed=seed, source_case_count=6,
        source_archive=entry["archive"], source_job=entry["job"],
        source_delivery_sha256=sha256_file(archive/"delivery.json"), source_report_sha256=sha256_file(archive/"report.json"),
        cases=cases, original_independent_blocks=1, reused_original_independent_blocks=1,
        cost_scope="Six fine source case wall times, including output serialization/checks; coarse and transfer excluded.",
        source_case_wall_seconds=sum(c["wall_seconds"] for c in cases))



def index_from_plan(root, plan, settings_sha256):
    """Resolve planned readers only after all 32 successful deliveries exist."""
    require(plan["schema"] == "pfc_refinement_repeat_readout_plan_v1"
            and plan["settings_sha256"] == settings_sha256
            and [r["source_rng_seed"] for r in plan["blocks"]] == SEEDS
            and len({r["archive"] for r in plan["blocks"]}) == 32
            and len({str(r["job"]) for r in plan["blocks"]}) == 32,
            "COMPLETE_UNIQUE_READOUT_PLAN_REQUIRED")
    entries=[]
    for planned in plan["blocks"]:
        archive=root/planned["archive"]
        delivery=json.loads((archive/"delivery.json").read_text())
        manifest=_source_manifest(delivery)
        wrapper=json.loads(_source_bound_file(archive,manifest,"report.json").read_text())
        report_path=_source_bound_file(archive,manifest,"analysis/report.json")
        report=json.loads(report_path.read_text())
        require(delivery["status"] == wrapper["status"] == "COMPLETE_PFC_REFINEMENT_REPEAT_BLOCK"
                and str(delivery["job_id"]) == str(wrapper["job_id"]) == str(planned["job"])
                and wrapper["exit_code"] == 0 and wrapper["mode"] == "block"
                and any(r["phase"]=="analysis" and r["exit_code"]==0 for r in wrapper["commands"])
                and all(r["exit_code"]==0 for r in wrapper["commands"])
                and report["status"] == BLOCK_STATUS
                and report["source_rng_seed"] == planned["source_rng_seed"]
                and report["source_case_count"] == 6
                and report["settings_sha256"] == settings_sha256,
                "ALL_COMPLETE_BOUND_READOUTS_REQUIRED")
        entries.append(dict(source_rng_seed=planned["source_rng_seed"],
            path=str(Path(planned["archive"])/"analysis"),report_sha256=sha256_file(report_path)))
    return dict(schema="pfc_refinement_repeat_readout_index_v1",settings_sha256=settings_sha256,blocks=entries)


def assemble_readouts(root, index, settings, old_fields, old_responses, old_cdfs, old_support):
    require(index["schema"] == "pfc_refinement_repeat_readout_index_v1"
            and index["settings_sha256"] == settings["_sha256"]
            and [r["source_rng_seed"] for r in index["blocks"]] == SEEDS
            and len({r["path"] for r in index["blocks"]}) == 32,
            "COMPLETE_UNIQUE_32_BLOCK_READOUT_INDEX_REQUIRED")
    fields = {LABELS[0]: old_fields[LABELS[0]], **{l: {c: [] for c in CONDITIONS} for l in LABELS[1:]}}
    responses = {LABELS[0]: old_responses[LABELS[0]], **{l: [] for l in LABELS[1:]}}
    cdfs = {l: [] for l in LABELS[1:]}
    supports, accepted = [old_support], []
    for i, entry in enumerate(index["blocks"]):
        path, report = bound_analysis(root, entry)
        require(report["status"] == BLOCK_STATUS and report["source_rng_seed"] == SEEDS[i]
                and report["source_case_count"] == 6 and report["settings_sha256"] == settings["_sha256"]
                and report["source_archive"] == settings["source_plan"][i]["archive"]
                and str(report["source_job"]) == str(settings["source_plan"][i]["job"]),
                "COMPLETE_REGISTERED_REPEAT_READOUT_REQUIRED")
        with np.load(output_file(path, report, "block_readout.npz"), allow_pickle=False) as saved:
            for key, expected in (("source_rng_seeds",[SEEDS[i]]), ("field_unit_counts",SIZES),
                                  ("condition_names",CONDITIONS), ("phase_names",list(PHASES)),
                                  ("metric_names",METRICS), ("window_ms",[5,50,100]),
                                  ("time_start_ms",np.arange(1000)*5)):
                require(np.array_equal(saved[key], expected), "REPEAT_READOUT_AXIS_CHANGED:"+key)
            supports.append(saved["count_support"].copy())
            for l in LABELS[1:]:
                for c in CONDITIONS: fields[l][c].append(saved[l+"_"+c].copy())
                responses[l].append(saved[l+"_values"].copy())
                cdfs[l].append(saved[l+"_cdf"].copy())
        accepted.append(dict(entry, source_archive=report["source_archive"], source_job=report["source_job"],
            source_delivery_sha256=report["source_delivery_sha256"], source_case_count=6,
            source_case_wall_seconds=report["source_case_wall_seconds"]))
    common = np.arange(max(s[-1] for s in supports)+1, dtype=np.int64)
    for l in LABELS[1:]:
        fields[l] = {c: np.stack(v) for c,v in fields[l].items()}
        responses[l] = np.stack(responses[l], axis=1)
        cdfs[l] = np.stack([cdf_on_support(x,s,common) for x,s in zip(cdfs[l],supports[1:])], axis=1)
    cdfs[LABELS[0]] = cdf_on_support(old_cdfs[LABELS[0]],old_support,common)
    return fields, responses, cdfs, common, accepted


def compare_scores(original_rows, repeated_rows, old_scores, repeated_scores):
    require(old_scores.shape == repeated_scores.shape == (len(original_rows),32)
            and len(original_rows) == len(repeated_rows), "MATCHED_32_BLOCK_SCORE_BANKS_REQUIRED")
    rows, primitive_indices, midpoints, deltas = [], [], [], []
    for i, (old, new) in enumerate(zip(original_rows, repeated_rows)):
        require(all(str(new[k]) == old[k] for k in IDENTITY_KEYS), "REPEAT_ENDPOINT_IDENTITY_CHANGED")
        row = {k: old[k] for k in IDENTITY_KEYS}
        row.update(original_estimate="", repeated_estimate="", estimate_change="", estimate_midpoint="",
                   original_point_outside_margin="", repeated_point_outside_margin="", same_nonzero_sign="",
                   observed_block_mean_absolute_change="", observed_block_rms_change="",
                   auxiliary_scatter_kind="UNDEFINED", original_independent_blocks=32, auxiliary_realizations=2)
        if old["definition_status"] != UNDEFINED:
            a, b = float(old["coupled_signed_estimate"]), float(new["signed_estimate"])
            require(np.isfinite(old_scores[i]).all() and np.isfinite(repeated_scores[i]).all()
                    and np.isclose(old_scores[i].mean(),a,rtol=1e-10,atol=1e-12)
                    and np.isclose(repeated_scores[i].mean(),b,rtol=1e-10,atol=1e-12),
                    "POINT_AND_SCORE_IDENTITY_FAILED")
            require(new["score_kind"] == old["coupled_score_kind"], "SCORE_KIND_CHANGED")
            row.update(original_estimate=a, repeated_estimate=b, estimate_change=b-a, estimate_midpoint=(a+b)/2,
                original_point_outside_margin=abs(a)>float(old["margin"]),
                repeated_point_outside_margin=abs(b)>float(old["margin"]),
                same_nonzero_sign=bool((a>0 and b>0) or (a<0 and b<0)))
            if new["score_kind"] == "covariance_jackknife_pseudovalue":
                row["auxiliary_scatter_kind"] = "FULL_U_STATISTIC_POINTS_ONLY_NO_BLOCK_SCATTER"
            else:
                delta = repeated_scores[i]-old_scores[i]
                row.update(auxiliary_scatter_kind="OBSERVED_WITHIN_ORIGINAL_BLOCK_CHANGE",
                    observed_block_mean_absolute_change=float(np.abs(delta).mean()),
                    observed_block_rms_change=float(np.sqrt(np.mean(delta*delta))))
                primitive_indices.append(int(old["domain_row_index"]))
                midpoints.append((old_scores[i]+repeated_scores[i])/2)
                deltas.append(delta)
        else:
            require(np.isnan(old_scores[i]).all() and np.isnan(repeated_scores[i]).all(),
                    "UNDEFINED_REPEAT_SCORES_CHANGED")
        rows.append(row)
    return rows, np.asarray(primitive_indices), np.asarray(midpoints).reshape(-1,32), np.asarray(deltas).reshape(-1,32)


def compare(root, settings, index, output):
    margins, old_rows, old_fields, old_responses, old_cdfs, old_support, old_scores = original_bank(root,settings)
    fields, responses, cdfs, support, accepted = assemble_readouts(
        root,index,settings,old_fields,old_responses,old_cdfs,old_support)
    # Compute covariance separately for the new realization. Never average its field with the old one.
    rows, scores = precision.summarize_endpoints(margins,fields,responses,cdfs,support,SEEDS)
    comparisons, ids, midpoints, deltas = compare_scores(old_rows,rows,old_scores,scores)
    output.mkdir(parents=True,exist_ok=False)
    export_arrays(output,fields,responses,cdfs,support,scores,[int(r["domain_row_index"]) for r in rows])
    write_rows(output/"repeated_endpoint_statistics.csv",rows)
    write_rows(output/"auxiliary_repeat_comparison.csv",comparisons)
    np.savez_compressed(output/"primitive_nested_scores.npz",source_rng_seeds=SEEDS,
        domain_row_indices=ids,original_scores=old_scores[ids],repeated_scores=scores[ids],
        within_block_midpoint=midpoints,repeated_minus_original=deltas)
    freeze_json(output/"accepted_source_index.json",dict(schema="pfc_refinement_repeat_accepted_source_index_v1",
        original_independent_blocks=32,new_original_independent_blocks=0,complete_repeated_fine_cases=192,blocks=accepted))
    counts = Counter((r["family"], r["original_point_outside_margin"],r["repeated_point_outside_margin"])
                     for r in comparisons if r["definition_status"] != UNDEFINED)
    write_rows(output/"point_margin_counts.csv",[dict(family=f,original_point_outside_margin=a,
        repeated_point_outside_margin=b,endpoint_count=n) for (f,a,b),n in sorted(counts.items())])
    return dict(status="COMPLETE_DESCRIPTIVE_PFC_REFINEMENT_REPEAT_COMPARISON",row_count=len(rows),
        source_case_count=192,original_independent_blocks=32,auxiliary_realizations=2,
        primitive_endpoint_count=len(ids),covariance_realizations_evaluated_separately=True,
        original_arrays_reused_from=settings["original"],original_raw_spikes_reread=False,
        source_case_wall_seconds=sum(r["source_case_wall_seconds"] for r in accepted))


def run(config_path, root, output, mode, seed=None, index_path=None):
    started=time.perf_counter()
    settings,blocks=configuration(config_path,root);settings["_sha256"]=sha256_file(config_path)
    require(not output.exists(),"NEW_ANALYSIS_OUTPUT_REQUIRED")
    if mode == "validate":
        margins,*_=original_bank(root,settings);output.mkdir(parents=True)
        result=dict(status="PASS_REFINEMENT_REPEAT_ANALYSIS_INPUTS",row_count=len(margins),source_case_count=0,
                    original_independent_blocks=32,original_statistics_recomputed=False)
    elif mode == "block":
        require(seed in SEEDS,"REGISTERED_REPEAT_BLOCK_REQUIRED")
        result=block_readout(root,settings,blocks[SEEDS.index(seed)],output)
    else:
        require(mode == "compare" and index_path is not None,"COMPLETE_READOUT_INDEX_REQUIRED")
        result=compare(root,settings,json.loads(index_path.read_text()),output)
    result.update(settings_sha256=settings["_sha256"],protocol_sha256=settings["protocol"]["sha256"],
        implementation_sha256=sha256_file(Path(__file__)),source_simulations=0,new_original_independent_blocks=0,
        confirmation_execution_released=False,scientific_fidelity_qualified=False,production_episodes=0,
        confidence_intervals=False,uncertainty_coverage_qualified=False,numerical_margins_changed=False,
        wall_seconds=time.perf_counter()-started,
        interpretation="Development description on32 original blocks. Two auxiliary realizations are nested, not64 independent blocks. "
        "Same signed shifts under two refinements do not establish numerical bias. Zero observed scatter does not establish zero population uncertainty. "
        "Covariance points use each complete realization separately; pseudovalues are not independent block responses.",
        outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha256_file(p)) for p in output.iterdir() if p.is_file()})
    freeze_json(output/"report.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("config","project-root","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--mode",choices=("validate","block","compare"),required=True)
    p.add_argument("--source-rng-seed",type=int);p.add_argument("--readout-index",type=Path)
    a=p.parse_args()
    result=run(a.config,a.project_root,a.output,a.mode,a.source_rng_seed,a.readout_index)
    print(json.dumps({k:result[k] for k in ("status","source_case_count","wall_seconds")}))


if __name__=="__main__":main()
