"""Compare delivered LDNS or EAG and population ensembles on every development block."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from .braid_population_comparison import (DeliveredCell, _score_table, _number,
    _training_cost, _population_cost_receipt, _population_method, _population_prior_binding)
from .population_forecast_analysis import _measured_origin_columns, _is_prior_refit, _prior_protocol
from .provenance import sha256_file
from .population_forecast_replay import METRIC_ATOL, METRIC_RTOL

SEEDS = (41, 42, 43)
METRICS = ("unit_bin_mean_mse_corrected", "sampled_expected_spikes_per_unit", "observed_spikes_per_unit",
           "unit_total_coverage_90", "unit_total_interval_width_90", "energy_score",
           "variogram_spatial", "variogram_cross_population", "variogram_temporal_lag_1",
           "variogram_temporal_lag_4", "variogram_temporal_lag_16",
           "refractory_count_violation_1bin", "refractory_count_violation_2bin", "refractory_count_violation_4bin")


def _write(path, rows):
    with path.open("w", newline="") as stream:
        columns=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def _population_budget_label(run):
    """Describe executed training from its recorded ceiling and selection."""
    config = run["binding"]["config"]
    maximum = run.get("effective_maximum_epochs", config.get("maximum_epochs"))
    if maximum is None and run["stop_reason"] == "epoch_budget":
        maximum = run["completed_epochs"]
    base = config.get("maximum_epochs", maximum)
    completed, best = run["completed_epochs"], run["best_epoch"]
    if (any(type(v) is not int for v in (maximum, base, completed, best))
            or not 0 <= best < completed <= maximum or not 0 < base <= maximum):
        raise ValueError("recorded population training ceiling and selection required")
    continuation = run.get("continuation")
    if maximum != base:
        if (not isinstance(continuation, dict)
                or continuation.get("schema") != "neural_population_trajectory_budget_continuation_v1"
                or continuation.get("total_epochs") != maximum
                or continuation.get("base_maximum_epochs") != base):
            raise ValueError("extended population training budget requires bound continuation")
    label=f"{maximum}epochs_maximum;{completed}completed;best_epoch{best};base{base}"
    if _is_prior_refit(run):
        parent=_prior_protocol(run)
        return (_population_method(run, "").removeprefix("population_recurrent_")+";"+label+f";parent{parent['parent_completed_epochs']}completed"
                +f";parent_selected_epoch{parent['parent_selected_epoch']};inherited_training_additional")
    return label


EAG_CONFIG = "Libs/configs/neural_population_latent_energy.yaml"
EAG_TRAINING_COUNTERS = dict(
    autoencoder_completed_epochs=260, energy_completed_passes=4000,
    energy_presentations=672000, energy_optimizer_steps=2625,
    completed_stage_epochs=4000, energy_epoch_rows=4000,
)
EAG_SOURCE_DENOMINATORS = dict(
    fit_complete_episodes=56, validation_complete_episodes=16,
    fit_origin_windows=168, validation_origin_windows=48,
    future_bins=100, units=100000,
)


def _eag_training_binding(cell, evaluation, run, seed):
    """Authenticate the saved EAG protocol and complete committed budget."""
    binding = evaluation["binding"]
    if (evaluation.get("method") != "eag" or binding.get("method") != "eag"
            or evaluation.get("seed") != seed or evaluation.get("source_unit_count") != 100000
            or evaluation.get("observation_law") != "source_softplus_poisson"
            or evaluation.get("native_refractory_qualified") is not False):
        raise ValueError("identified EAG source-softplus-Poisson evaluation required")
    config_path = cell.path(EAG_CONFIG)
    config_digest = sha256_file(config_path)
    if binding.get("configuration_sha256") != config_digest:
        raise ValueError("EAG configuration digest differs")
    config = yaml.safe_load(config_path.read_text())
    protocol = dict(schema="neural_population_latent_energy_training_v1",
        training_seeds=list(SEEDS), fit_blocks=list(range(14)), validation_blocks=list(range(14,18)),
        bin_ms=5., complete_episode_bins=1200, forecast_origins_bins=[200,600,1000], forecast_bins=100,
        expected_poisson_rng_family="per_edge_philox4_32_10")
    if (not isinstance(config,dict) or any(config.get(key)!=value for key,value in protocol.items())
            or any(not isinstance(config.get(key),dict) for key in ("autoencoder","energy","model","sampling"))
            or config["autoencoder"].get("maximum_epochs") != 260
            or config["energy"].get("maximum_epochs") != 4000
            or config["energy"].get("source_batch_size") != 256
            or config["model"].get("bin_seconds") != .005
            or config["model"].get("complete_episode_bins") != 1200
            or config["model"].get("forecast_bins") != 100
            or config["sampling"].get("joint_samples") != 16
            or binding.get("origins_bins") != [200,600,1000]
            or binding.get("horizons_ms") != [20.,80.,500.]):
        raise ValueError("EAG frozen comparison protocol or budget differs")
    if (run.get("status") != "FITTED_DEVELOPMENT" or run.get("stage") != "complete"
            or run.get("training_seed") != seed
            or any(type(run.get(key)) is not int or run[key] != value
                   for key,value in EAG_TRAINING_COUNTERS.items())):
        raise ValueError("complete EAG training budget required")
    denominators = run.get("source_denominators")
    if (denominators != EAG_SOURCE_DENOMINATORS
            or any(type(value) is not int for value in denominators.values())):
        raise ValueError("complete EAG source coverage required")
    if (_number(run["elapsed_seconds"]) < 0 or type(run.get("parameter_count")) is not int
            or run["parameter_count"] < 1):
        raise ValueError("finite nonnegative EAG training cost and parameter count required")
    pointer = cell.read("training/committed.json")
    pointer_digest = sha256_file(cell.path("training/committed.json"))
    if binding.get("training_committed_sha256") != pointer_digest:
        raise ValueError("EAG committed pointer digest differs")
    checkpoint_name = pointer.get("checkpoint")
    if not isinstance(checkpoint_name,str):
        raise ValueError("EAG committed checkpoint path required")
    relative = Path(checkpoint_name)
    if (relative.is_absolute() or ".." in relative.parts or not relative.parts
            or relative.parts[0] != "checkpoints" or relative.suffix != ".pt"):
        raise ValueError("EAG declared training checkpoint path required")
    checkpoint_path = cell.path("training/" + relative.as_posix())
    checkpoint_digest = sha256_file(checkpoint_path)
    if (pointer.get("stage") != "complete" or pointer.get("next_epoch") != 4000
            or pointer.get("sha256") != checkpoint_digest
            or run.get("latest_checkpoint_sha256") != checkpoint_digest):
        raise ValueError("EAG complete pointer or latest checkpoint digest differs")
    return dict(method="eag", observation_law="source_softplus_poisson",
        native_refractory_qualified=False, source_denominators=denominators,
        **{key:run[key] for key in EAG_TRAINING_COUNTERS},
        configuration_sha256=config_digest, training_committed_sha256=pointer_digest,
        latest_checkpoint_sha256=checkpoint_digest,
        committed_checkpoint="training/" + relative.as_posix())


def _match(ldns, population, ldns_run, population_run, *, baseline_kind="ldns"):
    if baseline_kind not in ("ldns","eag"):
        raise ValueError("registered joint-count baseline kind required")
    if (ldns["status"] != population["status"] or ldns["status"] != "EVALUATED_DEVELOPMENT"
            or ldns["score_rows"] != population["score_rows"] or ldns["score_rows"] != 144
            or ldns["complete_origins"] != 48 or ldns["binding"]["draws"] != 16
            or population["configuration"]["joint_samples"] != 16
            or population["configuration"]["origins_ms"] != [1000.,3000.,5000.]
            or population["configuration"]["horizons_ms"] != [20.,80.,500.]
            or population["formal_test_accessed"] is not False):
        raise ValueError("complete matched sixteen-draw development forecasts required")
    source=ldns["binding"]["cohort"];other=population_run["binding"]
    for key in ("acquisition_design_sha256","acquisition_ledger_sha256","dataset_sha256",
                "episode_table_sha256","episode_bindings"):
        if source[key] != other[key]:raise ValueError(f"{baseline_kind.upper()}/population source cohort differs: "+key)
    for key,value in source["native_source_identity"].items():
        if other["native_source_identity"][key] != value:raise ValueError("native source identity differs")
    baseline_units = (ldns_run["units"] if baseline_kind == "ldns" else ldns_run["source_denominators"]["units"])
    if (ldns_run["status"] != population_run["status"] or ldns_run["status"] != "FITTED_DEVELOPMENT"
            or ldns_run["training_seed"] != population_run["training_seed"]
            or ldns_run["stage"] != "complete" or baseline_units != 100000
            or population_run["model_configuration"]["bin_seconds"] != .005):
        raise ValueError("complete full-unit training required")


def _count_cost_rows(cell, evaluation, scores, baseline_kind):
    """Validate measured origins and charge shared episode loading once."""
    label = baseline_kind.upper()
    path=cell.path("forecasts/costs.csv")
    if evaluation["artifacts"].get("costs.csv") != sha256_file(path):
        raise ValueError(f"{label} evaluation cost digest differs")
    with path.open(newline="") as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames)!=len(set(reader.fieldnames)):
            raise ValueError(f"unique {label} cost columns required")
        rows=list(reader)
    expected={(case,origin):block for block,case,origin,horizon in scores}
    result,loading={},{}
    components=("inference_seconds","complete_unit_generation_seconds",
                "projection_and_scoring_seconds","serialization_and_hash_seconds")
    for raw in rows:
        origin=_number(raw["origin_ms"]);key=(raw["case_id"],origin)
        if key not in expected or key in result:
            raise ValueError(f"duplicate or foreign {label} cost origin")
        values={name:_number(raw[name]) for name in components+(
            "episode_loading_seconds","origin_total_seconds","generated_count_values","peak_rss_bytes")}
        if any(value<0 for value in values.values()):
            raise ValueError(f"nonnegative {label} costs required")
        if values["generated_count_values"]!=160000000 or values["peak_rss_bytes"]!=int(values["peak_rss_bytes"]):
            raise ValueError(f"complete {label} count values and integer peak RSS required")
        total=values["origin_total_seconds"]
        if total+1e-9*max(1.,total)<sum(values[name] for name in components):
            raise ValueError(f"{label} origin total omits measured components")
        case=key[0]
        if case in loading and loading[case]!=values["episode_loading_seconds"]:
            raise ValueError(f"{label} shared episode loading differs across origins")
        loading[case]=values["episode_loading_seconds"]
        charged=values["episode_loading_seconds"] if origin==1000. else None
        result[key]=dict(block=expected[key],case_id=case,origin_ms=origin,
            **values,charged_episode_loading_seconds=charged,
            accounted_origin_total_seconds=total+(charged or 0.))
    if len(rows)!=48 or set(result)!=set(expected):
        raise ValueError(f"complete {label} 48-origin costs required")
    elapsed=_number(evaluation["elapsed_seconds"])
    accounted=sum(row["accounted_origin_total_seconds"] for row in result.values())
    if elapsed<0 or elapsed+1e-9*max(1.,elapsed)<accounted:
        raise ValueError(f"{label} loop total omits origin or episode costs")
    return [result[key] for key in sorted(result)]



def _ldns_cost_rows(cell, evaluation, scores):
    """Retain the historical LDNS cost-reader interface."""
    return _count_cost_rows(cell, evaluation, scores, "ldns")


def _validate_observed_total_scores(ldns, population, observed_fields, horizon_bins, unit_count):
    """Check both saved FP32 reductions against the exact common count total.

    Source fields still match exactly. The tolerance is the frozen independent
    replay criterion, including its reported-value denominator.
    """
    counts = np.asarray(observed_fields)
    if (counts.ndim != 2 or not 0 < horizon_bins <= len(counts) or unit_count != 100000
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts != np.floor(counts))):
        raise ValueError("exact common observed field counts required")
    total = float(counts[:horizon_bins].sum(dtype=np.float64))
    if not total < 2**53:
        raise ValueError("exactly representable common observed count total required")
    reference = total / unit_count
    for value in (ldns, population):
        reported = _number(value)
        if abs(reported-reference) > METRIC_ATOL + METRIC_RTOL*abs(reported):
            raise ValueError("matched observed unit count totals required")
    return reference


def compare(ldns_root, population_root, output, *, baseline_kind="ldns"):
    if baseline_kind not in ("ldns","eag"):
        raise ValueError("registered joint-count baseline kind required")
    ldns_root,population_root,output=map(Path,(ldns_root,population_root,output))
    if output.exists():raise FileExistsError("new delivered comparison output required")
    rows,costs,paired_costs,evaluation_costs,training,bindings=[],[],[],[],[],{}
    prior_bindings={}
    baseline_bindings={}
    for seed in SEEDS:
        ldns,population=DeliveredCell(ldns_root/f"seed_{seed}",seed),DeliveredCell(population_root/f"seed_{seed}",seed)
        le,pe=ldns.read("forecasts/evaluation.json"),population.read("forecasts/evaluation.json")
        lr,pr=ldns.read("training/run.json"),population.read("training/run.json")
        if baseline_kind == "eag":
            baseline_bindings[str(seed)] = _eag_training_binding(ldns,le,lr,seed)
        _match(le,pe,lr,pr,baseline_kind=baseline_kind)
        candidate=_population_method(pr,"population")
        prior_binding=_population_prior_binding(population,pr)
        if prior_binding is not None:prior_bindings[str(seed)]=prior_binding
        replay=ldns.read("forecasts/replay.json");pop_replay=population.read("replay.json")
        if (replay["status"] != "PASS" or replay["evaluation_sha256"] != sha256_file(ldns.path("forecasts/evaluation.json"))
                or replay["all_origin_count_digests_fields_latents_metrics_exact"] is not True
                or pop_replay["status"] != "PASS" or pop_replay["verified_origins"] != 48
                or pop_replay["all_origin_fields_and_latents_exact"] is not True
                or pop_replay["all_forecast_unit_metrics_replayed"] is not True
                or pop_replay["evaluation_sha256"] != sha256_file(population.path("forecasts/evaluation.json"))):
            raise ValueError("complete successful independent forecast replay required")
        if baseline_kind == "eag" and (replay.get("source_unit_count") != 100000
                or replay.get("count_values_digest_replayed") != 48*160000000):
            raise ValueError("complete EAG source-unit replay denominator required")
        ls=_score_table(ldns.path("forecasts/scores.csv"),seed,baseline_kind)
        ps=_score_table(population.path("forecasts/scores.csv"),seed,"population")
        population_cost_receipt=_population_cost_receipt(pe,pr,ps)
        if le["artifacts"]["scores.csv"] != sha256_file(ldns.path("forecasts/scores.csv")):
            raise ValueError(f"{baseline_kind.upper()} evaluation score digest differs")
        with np.load(population.path("forecasts/fitting_field_scale.npz"),allow_pickle=False) as scale:
            field_ids,units=scale["field_ids"],scale["unit_count"]
            if le["binding"]["fitting_field_scale_sha256"] != sha256_file(population.path("forecasts/fitting_field_scale.npz")):
                raise ValueError(f"{baseline_kind.upper()} and population fitting scales differ")
        observed_counts = {}
        for key in sorted(ls):
            block,case_id,origin,horizon=key
            if horizon==20.:
                name=f"forecasts/{case_id}_origin_{origin:g}.npz"
                with np.load(ldns.path(name),allow_pickle=False) as a,np.load(population.path(name),allow_pickle=False) as b:
                    if (not np.array_equal(a["observed_field_counts"],b["observed_field_counts"])
                            or a["sampled_field_counts"].shape != b["sampled_field_counts"].shape
                            or not np.array_equal(a["field_ids"],field_ids)
                            or list(map(int,a["unit_ids"])) != pr["model_configuration"]["unit_ids"]):
                        raise ValueError("matched source truth, unit and field maps required")
                    observed_counts[case_id, origin] = b["observed_field_counts"].copy()
            row=dict(seed=seed,block=block,case_id=case_id,origin_ms=origin,horizon_ms=horizon,
                     **({"candidate_method":candidate} if prior_binding is not None else {}))
            for metric in METRICS:
                a,b=ls[key][metric],ps[key][metric]
                if a=="" or b=="":
                    if a!=b:raise ValueError("matched undefined metric required: "+metric)
                    row[baseline_kind+"_"+metric]=row["population_"+metric]=row["population_minus_"+baseline_kind+"_"+metric]=None
                else:
                    a,b=_number(a),_number(b)
                    row[baseline_kind+"_"+metric]=a;row["population_"+metric]=b
                    row["population_minus_"+baseline_kind+"_"+metric]=b-a
            _validate_observed_total_scores(
                row[baseline_kind+"_observed_spikes_per_unit"], row["population_observed_spikes_per_unit"],
                observed_counts[case_id, origin], int(horizon / 5.), len(pr["model_configuration"]["unit_ids"]))
            rows.append(row)
        baseline_training=dict(seed=seed,method=baseline_kind,recorded_training_seconds=lr["elapsed_seconds"],
            parameter_count=lr["parameter_count"],
            training_budget=("260AE+2000diffusionepochs" if baseline_kind=="ldns"
                             else "260AE+4000energypasses;672000presentations;2625updates"),
            device=le["hardware"]["device"],training_cost_scope="committed_wall_time;unreported_interruption_tails_excluded")
        if baseline_kind=="eag":
            baseline_training.update({key:lr[key] for key in EAG_TRAINING_COUNTERS})
        training.append(baseline_training)
        popcost=_training_cost(population,pr,candidate)
        population_training=dict(seed=seed,method=candidate,recorded_training_seconds=popcost["recorded_training_seconds"],
            parameter_count=pr["parameter_count"],training_budget=_population_budget_label(pr),
            device=pr.get("hardware",{}).get("device"),training_cost_scope=popcost["training_cost_scope"])
        if prior_binding is not None:
            population_training.update({key:value for key,value in popcost.items() if key not in population_training})
        training.append(population_training)
        cost_rows=_count_cost_rows(ldns,le,ls,baseline_kind)
        costs.extend(dict(seed=seed,method=baseline_kind,**row) for row in cost_rows)
        for cost in cost_rows:
            key=(cost["block"],cost["case_id"],cost["origin_ms"],500.)
            measured=_measured_origin_columns(population_cost_receipt,cost["case_id"],cost["origin_ms"])
            preparation=measured["episode_preparation_seconds"]
            origin_total=measured["origin_total_seconds"]
            paired_costs.append(dict(seed=seed,block=cost["block"],case_id=cost["case_id"],
                origin_ms=cost["origin_ms"],
                **{baseline_kind+"_"+name:value for name,value in cost.items()
                   if name not in ("block","case_id","origin_ms")},
                **{baseline_kind+"_hardware":json.dumps(le["hardware"],sort_keys=True)},
                population_inference_seconds=_number(ps[key]["inference_seconds"]),
                population_full_unit_generation_seconds=_number(ps[key]["full_unit_generation_seconds"]),
                population_field_projection_seconds=_number(ps[key]["field_projection_seconds"]),
                **{"population_"+name:value for name,value in measured.items()},
                population_accounted_origin_total_seconds=(None if origin_total is None
                    else origin_total+(preparation or 0.)),
                output_scope="both 16 draws, 100000 units, 100 bins; event materialization excluded"))
        measured=population_cost_receipt.get("measured_evaluation_costs")
        evaluation_costs.extend([
            dict(seed=seed,method=baseline_kind,elapsed_seconds=_number(le["elapsed_seconds"]),
                function_total_seconds=None,setup_seconds=None,score_hash_seconds=None,
                peak_process_rss_bytes=None,hardware=json.dumps(le["hardware"],sort_keys=True),
                cost_scope="loop elapsed includes episode loading and origin totals; setup, score hash and delivery unmeasured"),
            dict(seed=seed,method=candidate,
                **{name:measured["function_costs"][name] if measured else None for name in
                    ("elapsed_seconds","function_total_seconds","setup_seconds","score_hash_seconds","peak_process_rss_bytes")},
                hardware=json.dumps(measured["hardware"] if measured else pr.get("hardware",{}),sort_keys=True),
                cost_scope=json.dumps(measured["cost_scope"],sort_keys=True) if measured
                    else "legacy input loading, scoring, serialization and evaluator setup unmeasured")])
        bindings[str(seed)]={baseline_kind:ldns.binding(),"population":population.binding()}
    grouped=defaultdict(list)
    for row in rows:grouped[row["seed"],row["block"],row["horizon_ms"]].append(row)
    block_rows=[]
    for (seed,block,horizon),values in sorted(grouped.items()):
        if len(values)!=12:raise ValueError("all four templates and three origins per block required")
        summary=dict(seed=seed,block=block,horizon_ms=horizon,matched_conditions=12)
        for name in values[0]:
            if name.startswith((baseline_kind+"_","population_")):
                valid=[row[name] for row in values]
                if any(value is None for value in valid):
                    if not all(value is None for value in valid):raise ValueError("incomplete descriptive block metric")
                    summary[name]=None
                else:summary[name]=float(np.mean(valid))
        block_rows.append(summary)
    if len(rows)!=432 or len(block_rows)!=36:raise ValueError("complete three-seed comparison denominator required")
    if prior_bindings and (set(prior_bindings)!={str(seed) for seed in SEEDS}
            or len({binding["training_objective"] for binding in prior_bindings.values()}) != 1):
        raise ValueError("all three population seeds must use the same prior objective")
    output.mkdir(parents=True)
    files={"joint_comparison_all_seeds.csv":rows,"joint_comparison_by_block.csv":block_rows,
           baseline_kind+"_costs_by_origin.csv":costs,"costs_by_origin.csv":paired_costs,
           "evaluation_costs.csv":evaluation_costs,"training_costs.csv":training}
    for name,values in files.items():_write(output/name,values)
    report=dict(status="COMPLETE_DESCRIPTIVE_DEVELOPMENT_COMPARISON",matched_rows=432,block_rows=36,
        training_seeds=list(SEEDS),independent_blocks_per_seed=4,connectivity_instances=1,bindings=bindings,
        same_training_budget=False,same_hardware=False,full_end_to_end_cost_matched=False,
        paired_cost_origins=len(paired_costs),shared_episode_cost_charged_once=True,
        population_measured_evaluation_cost_seeds=[row["seed"] for row in evaluation_costs
            if row["method"]!=baseline_kind and row["function_total_seconds"] is not None],
        speedup_qualified=False,
        observed_total_validation=dict(reference="exact_common_observed_field_count_total_per_100000_units",
            rtol=METRIC_RTOL, atol=METRIC_ATOL, tolerance_source="frozen_population_forecast_replay",
            observed_field_counts_match_exactly=True, saved_scalar_differences_retained=True),
        inferential_statistics_qualified=False,scientific_claim_qualified=False,
        artifacts={name:sha256_file(output/name) for name in files})
    if baseline_kind=="eag":
        report.update(baseline_method="eag",baseline_training_bindings=baseline_bindings,
                      baseline_observation_law="source_softplus_poisson",
                      baseline_native_refractory_qualified=False)
    if prior_bindings:
        report.update(candidate_method=prior_bindings[str(SEEDS[0])]["method"],
                      population_training_bindings=prior_bindings,inherited_population_training_cost_included=True)
    (output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("ldns-root","population-root","output"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(compare(args.ldns_root,args.population_root,args.output)))


if __name__=="__main__":main()
