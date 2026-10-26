"""Source-paired stimulus response comparisons using existing forecast metrics.

The source pairing receipt supplies previously verified native prefixes and
initial/background RNG identity. Model generation receives only the factual
observed prefix and known commands. Both future source windows are scoring inputs.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .pfc_compiled_response import prepare_compiled_response
from .pfc_population_evaluation import _load, _scores, source_controls
from .pfc_population_training_data import PfcPopulationTrainingCohort, _execution_paths, _declared_path
from .pfc_population_forecast_analysis import FrozenInputs
from .circuit_data import CircuitDataset
from .resource_probe import probe
from .population_forecast_replay import _project_units
from .population_trajectory_evaluation import _draw, _synchronize
from .provenance import sha256_file, freeze_json
from .trajectory_sample_metrics import corrected_count_mse


def source_pairing(receipt_path, factual, candidate, origin):
    """Bind admitted episodes to one existing, authenticated native pairing."""
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != "ACCEPTED_PFC_TEN_CONDITION_TASK_READBACK"
            or receipt.get("complete_development_episodes") != 10
            or len(receipt.get("sources", [])) != 10):
        raise ValueError("PFC_RESPONSE_COMPLETE_ACCEPTED_TEN_CONDITION_RECEIPT_REQUIRED")
    archive = Path(receipt["archive"])
    if (sha256_file(archive / "delivery.json") != receipt["delivery_sha256"]
            or sha256_file(archive / "report.json") != receipt["report_sha256"]):
        raise ValueError("PFC_RESPONSE_NATIVE_PAIRING_RECEIPT_CHANGED")
    report = json.loads((archive / "report.json").read_text())
    if (report.get("status") != "PASS_PFC_TEN_CONDITION_SOURCE_READBACK"
            or report.get("exit_code") != 0 or len(report.get("circuits", [])) != 1):
        raise ValueError("PFC_RESPONSE_COMPLETE_NATIVE_PAIRING_REPORT_REQUIRED")
    circuit = report["circuits"][0]
    if (circuit["pairing"] != receipt["pairing"]
            or any(circuit[key] != receipt[key] for key in ("circuit_index", "block_index"))
            or len(circuit["source_bindings"]) != 10):
        raise ValueError("PFC_RESPONSE_NATIVE_PAIRING_BINDINGS_CHANGED")
    if (len({row["episode_id"] for row in receipt["sources"]}) != 10
            or len({row["episode_id"] for row in circuit["source_bindings"]}) != 10):
        raise ValueError("PFC_RESPONSE_UNIQUE_COMPLETE_SOURCE_PAIRING_REQUIRED")
    selected = []
    for data in (factual, candidate):
        matches = [row for row in receipt["sources"] if row["episode_id"] == data.episode_id]
        declarations = [row for row in circuit["source_bindings"] if row["episode_id"] == data.episode_id]
        if len(matches) != 1 or len(declarations) != 1:
            raise ValueError("PFC_RESPONSE_SOURCE_NOT_IN_NATIVE_PAIRING")
        row, declared = matches[0], declarations[0]
        identity = data.source_config["benchmark_episode_candidate"]["identity"]
        if (any(row[key] != declared[key] for key in
                ("episode_id", "condition_id", "source_archive", "delivery_sha256", "complete_units"))
                or row["event_count"] != declared["events"]
                or row["delivery_sha256"] != data.binding["source_delivery_sha256"]
                or row["complete_units"] != len(data.unit_ids)
                or any(identity[key] != receipt[key] for key in ("circuit_index", "block_index"))):
            raise ValueError("PFC_RESPONSE_ADMITTED_SOURCE_AND_PAIRING_DIFFER")
        selected.append(row)
    if (factual.binding["circuit_id"] != candidate.binding["circuit_id"]
            or factual.binding["block_id"] != candidate.binding["block_id"]
            or factual.binding["split"] != candidate.binding["split"]
            or factual.dt_ms != candidate.dt_ms or factual.bin_ticks != candidate.bin_ticks
            or factual.duration_ticks != candidate.duration_ticks
            or factual.field_names != candidate.field_names
            or any(not np.array_equal(getattr(factual, name), getattr(candidate, name)) for name in
                   ("unit_ids", "unit_field_index", "minimum_interval_ticks"))):
        raise ValueError("PFC_RESPONSE_SAME_NATIVE_MAP_CLOCK_BLOCK_AND_SPLIT_REQUIRED")
    if factual.episode_id == candidate.episode_id:
        first = factual.duration_ticks
        reference = selected[0]["condition_id"]
    else:
        matches = [row for row in receipt["pairing"]
                   if row["condition_id"] == selected[1]["condition_id"]
                   and row["reference_condition"] == selected[0]["condition_id"]]
        if (len(matches) != 1 or matches[0].get("initial_and_final_rng_exact") is not True
                or type(matches[0].get("first_changed_current_tick")) is not int):
            raise ValueError("PFC_RESPONSE_PREVIOUSLY_VERIFIED_REFERENCE_PAIR_REQUIRED")
        first, reference = matches[0]["first_changed_current_tick"], matches[0]["reference_condition"]
    if type(origin) is not int or not 0 < origin <= first:
        raise ValueError("PFC_RESPONSE_ORIGIN_EXCEEDS_VERIFIED_NATIVE_PREFIX")
    return dict(receipt=str(receipt_path), receipt_sha256=sha256_file(receipt_path),
        report_sha256=receipt["report_sha256"], delivery_sha256=receipt["delivery_sha256"],
        factual_case=factual.episode_id, candidate_case=candidate.episode_id,
        reference_condition=reference, candidate_condition=selected[1]["condition_id"],
        first_changed_current_tick=first, source_initial_and_background_pairing_verified=True,
        source_future_arrays_reverified=False)


def mean_contrast_scores(candidate, factual, candidate_truth, factual_truth, mapping, fields,
                         sizes, bin_seconds, horizons):
    """Compare conditional mean effects, preserving signed sampling corrections.

    Common model sampling seeds define a numerical coupling between two arms.
    This coupling is not claimed to reproduce the source joint intervention law.
    """
    if (candidate.shape != factual.shape or candidate.ndim != 3 or len(candidate) < 2
            or candidate_truth.shape != candidate.shape[1:] or factual_truth.shape != candidate_truth.shape
            or any(np.asarray(x).dtype.kind not in "iu" or np.any(x < 0)
                   for x in (candidate, factual, candidate_truth, factual_truth))):
        raise ValueError("PFC_RESPONSE_COMPLETE_NONNEGATIVE_INTEGER_ARMS_REQUIRED")
    candidate = np.asarray(candidate, dtype=np.int64)
    factual = np.asarray(factual, dtype=np.int64)
    actual = np.asarray(candidate_truth, dtype=np.int64) - np.asarray(factual_truth, dtype=np.int64)
    difference = candidate - factual
    generated, truth = _project_units(difference, actual, np.asarray(mapping), fields)
    present = np.asarray(sizes) > 0
    rates = generated[..., present] / (np.asarray(sizes)[present] * bin_seconds)
    observed = truth[:, present] / (np.asarray(sizes)[present] * bin_seconds)
    result = []
    for horizon in horizons:
        if type(horizon) is not int or not 1 <= horizon <= candidate.shape[1]:
            raise ValueError("PFC_RESPONSE_VALID_COMPLETE_CONTRAST_HORIZON_REQUIRED")
        residual = rates[:, :horizon] - observed[None, :horizon]
        correction = residual.mean(0)**2 - residual.var(0, ddof=1)/len(residual)
        result.append(dict(horizon_bins=horizon,
            unit_mean_contrast_mse_corrected=corrected_count_mse(
                torch.from_numpy(difference[:, :horizon]), torch.from_numpy(actual[:horizon])),
            field_mean_contrast_mse_hz2_corrected=float(correction.mean()),
            generated_total_contrast_per_unit=float(difference[:, :horizon].sum(1).mean()),
            observed_total_contrast_per_unit=float(actual[:horizon].sum(0).mean()),
            model_coupling="common_explicit_sampling_seeds; conditional_mean_comparison_only",
            contrast_distribution_qualified=False, scientific_intervention_qualified=False))
    return result


@torch.no_grad()
def evaluate_source_response(view, factual, candidate, origin, config, receipt_path, sampling_seed_base):
    """Generate both arms before reading either source's future scoring window."""
    horizons = config["horizons_bins"]
    draws = config["joint_samples"]
    if (not horizons or horizons != sorted(set(horizons))
            or any(type(h) is not int or h < 1 for h in horizons)
            or type(draws) is not int or draws < 2
            or type(sampling_seed_base) is not int or sampling_seed_base != 41):
        raise ValueError("PFC_RESPONSE_DECLARED_HORIZONS_DRAWS_AND_SEEDS_41_42_43_REQUIRED")
    if (type(origin) is not int or origin < 1 or origin % factual.bin_ticks
            or origin + max(horizons)*factual.bin_ticks > factual.duration_ticks):
        raise ValueError("PFC_RESPONSE_COMPLETE_ALIGNED_SOURCE_WINDOW_REQUIRED")
    pairing = source_pairing(receipt_path, factual, candidate, origin)
    device = view.posterior_mean.weight.device
    _synchronize(device)
    started = time.perf_counter()
    target, state, future = prepare_compiled_response(view, factual, origin, max(horizons),
                                                      candidate.source_config)
    reference_state = replace(state, source_metadata_identity=view.source_metadata_identity)
    factual_future = source_controls(factual, origin, origin + max(horizons)*factual.bin_ticks)
    _synchronize(device)
    prepared = time.perf_counter()
    candidate_counts, candidate_latent, _, candidate_ready = _draw(target, "population", state,
        future["known_controls"][None], sampling_seed_base, draws, device)
    _synchronize(device)
    generated_candidate = time.perf_counter()
    factual_counts, factual_latent, _, factual_ready = _draw(view, "population", reference_state,
        factual_future["known_controls"][None], sampling_seed_base, draws, device)
    _synchronize(device)
    generated_factual = time.perf_counter()
    # Candidate outcome data never enters either inferred state or sampled arm.
    stop = origin + max(horizons)*factual.bin_ticks
    target_truth, target_commands = candidate.window(origin, stop)
    reference_truth, reference_commands = factual.window(origin, stop)
    if (not np.array_equal(target_commands, future["known_controls"])
            or not np.array_equal(reference_commands, factual_future["known_controls"])):
        raise ValueError("PFC_RESPONSE_KNOWN_AND_EVALUATED_SOURCE_COMMANDS_DIFFER")
    arrays = dict(candidate_generated_counts=candidate_counts.cpu().numpy(),
                  factual_generated_counts=factual_counts.cpu().numpy(),
                  candidate_observed_counts=target_truth, factual_observed_counts=reference_truth,
                  candidate_latent_states=candidate_latent.cpu().numpy(),
                  factual_latent_states=factual_latent.cpu().numpy(),
                  candidate_terminal_ready=candidate_ready.cpu().numpy(),
                  factual_terminal_ready=factual_ready.cpu().numpy(),
                  unit_ids=np.asarray(view.unit_ids), unit_field_index=view.unit_field_index.cpu().numpy(),
                  field_names=np.asarray(view.field_names))
    projected = time.perf_counter()
    candidate_fields, target_fields = _project_units(arrays["candidate_generated_counts"], target_truth,
        arrays["unit_field_index"], len(view.field_names))
    factual_fields, reference_fields = _project_units(arrays["factual_generated_counts"], reference_truth,
        arrays["unit_field_index"], len(view.field_names))
    candidate_scores = _scores(target, target_truth, candidate_counts, candidate_fields, target_fields, config)
    factual_scores = _scores(view, reference_truth, factual_counts, factual_fields, reference_fields, config)
    contrasts = mean_contrast_scores(arrays["candidate_generated_counts"], arrays["factual_generated_counts"],
        target_truth, reference_truth, arrays["unit_field_index"], len(view.field_names),
        view.field_unit_counts.cpu().numpy(), view.bin_seconds, horizons)
    _synchronize(device)
    costs = dict(preparation_and_factual_inference_seconds=prepared-started,
        candidate_generation_seconds=generated_candidate-prepared,
        factual_generation_seconds=generated_factual-generated_candidate,
        source_future_windows_and_transfer_seconds=projected-generated_factual,
        full_unit_projection_and_scores_seconds=time.perf_counter()-projected,
        cost_scope="this_comparison_only; excludes_complete_source_admission_training_and_mechanistic_execution",
        matched_fidelity_speedup_qualified=False)
    return dict(status="EVALUATED_SOURCE_PAIRED_DEVELOPMENT_RESPONSE", pairing=pairing,
        candidate_scores=candidate_scores, factual_scores=factual_scores, mean_contrasts=contrasts,
        origin_ticks=origin, horizons_bins=horizons, draws=draws, sampling_seed_base=sampling_seed_base,
        units=len(view.unit_ids), fields=len(view.field_names), costs=costs,
        sampling_stream_seeds=dict(initial=41, dynamics=42, observation=43),
        future_outcomes_used_only_after_both_generations=True,
        source_simulations=0, training_updates=0, independent_success_gate_evaluated=False,
        scientific_intervention_qualified=False), arrays

def _response_config(path):
    value = yaml.safe_load(Path(path).read_text())
    fixed = dict(schema="pfc_source_paired_response_development_v1",
        campaign="neural_population_dynamics", training_seeds=[41,42,43],
        sampling_stream_seeds=dict(initial=41,dynamics=42,observation=43),
        evaluation_scope="development_source_response", formal_evaluation=False,
        reference_condition="cue_none", condition_indices=list(range(10)),
        origins_ticks_by_condition=[2000,8000,8000,8000,48000,2000,2000,2000,2000,2000],
        field_units="source_field_mean_unit_rate_hz",
        causal_history="all_observed_bins_before_origin", finite_ensemble_corrections=True,
        candidate_outcomes="scoring_only_after_both_generations",
        contrast_estimand="conditional_mean_candidate_minus_factual",
        model_draw_coupling="common_explicit_sampling_seeds",
        independent_success_gate_evaluated=False, sampling_seed_base=41)
    if (not isinstance(value,dict) or any(value.get(key)!=expected for key,expected in fixed.items())
            or value.get("horizons_bins")!=[4,16,100] or value.get("support_windows")!=[1,2]
            or value.get("joint_samples")!=16):
        raise ValueError("PFC_RESPONSE_REGISTERED_DEVELOPMENT_CONFIG_REQUIRED")
    return value



class PreparedResponseCohort(PfcPopulationTrainingCohort):
    """Reuse complete frozen metadata while admitting requested episode arrays.

    The inherited adapter and model constructors retain source authentication.
    This path does not create, modify or reinterpret any input identity.
    """
    def __init__(self, inputs: Path, preparation: Path):
        frozen=FrozenInputs(preparation)
        if frozen.proof["input_sha256"]!=sha256_file(inputs):
            raise ValueError("PFC_RESPONSE_FROZEN_INPUTS_CHANGED")
        paths=_execution_paths(inputs,frozen.spec,frozen.config)
        self.config=frozen.config
        self.bin_ticks=frozen.config["bin_ticks"]
        entry=_declared_path(frozen.spec["source_entry"],inputs)
        self.source_entry=paths.get(entry,entry)
        self.data={}
        self.fit=[]
        self.validation=[]
        self.active_case=self.active_episode=None
        for row,record in zip(frozen.spec["episodes"],frozen.records):
            dataset=_declared_path(row["dataset"],inputs)
            dataset=paths.get(dataset,dataset)
            source=_declared_path(row["source_run"],inputs,source_run=True)
            source=paths.get(source,source)
            if dataset not in self.data:
                self.data[dataset]=CircuitDataset(dataset,splits=("development",))
            source_binding={key:value for key,value in record.items() if key!="partition"}
            case=dict(case_id=record["circuit_id"]+"__"+row["episode_id"],episode_id=row["episode_id"],
                      circuit_id=record["circuit_id"],dataset=dataset,source_run=source,
                      source_binding=source_binding)
            (self.fit if row["partition"]=="development_fit" else self.validation).append(case)
        self.binding=dict(circuit_inputs_sha256=frozen.proof["input_sha256"],
            pfc_source_entry_sha256=frozen.proof["paths"][frozen.spec["source_entry"]]["sha256"],
            complete_episode_bindings=frozen.records,partition_unit="whole_source_circuit",
            partition_scope="declared_development_only; original_archive_splits_preserved")


def run_source_responses(inputs, training, config_path, receipt_path, output, *, resource_delivery=None, preparation=None):
    """Run the declared ten-condition source block and retain every output."""
    output = Path(output)
    config = _response_config(config_path)
    identity = dict(configuration_sha256=sha256_file(Path(config_path)),
        source_pairing_receipt_sha256=sha256_file(Path(receipt_path)),
        training_run_sha256=sha256_file(Path(training)/"run.json"),
        checkpoint_sha256=sha256_file(Path(training)/"best.pt"),
        inputs_sha256=sha256_file(Path(inputs)),
        resource_delivery_sha256=(sha256_file(Path(resource_delivery)) if resource_delivery else None),
        preparation_sha256=(sha256_file(Path(preparation)/"preparation.json") if preparation else None),
        evaluator_sha256=sha256_file(Path(__file__)),
        compiled_response_sha256=sha256_file(Path(__file__).with_name("pfc_compiled_response.py")))
    if (output/"evaluation.json").exists():
        previous=json.loads((output/"evaluation.json").read_text())
        if previous.get("identity")!=identity or previous.get("status")!="COMPLETE_TEN_CONDITION_SOURCE_RESPONSE_DEVELOPMENT":
            raise ValueError("PFC_RESPONSE_EXISTING_RESULT_IDENTITY_CHANGED")
        for name,binding in previous["outputs"].items():
            path=output/name
            if path.stat().st_size!=binding["bytes"] or sha256_file(path)!=binding["sha256"]:
                raise ValueError("PFC_RESPONSE_EXISTING_RESULT_CHANGED:"+name)
        return previous
    output.mkdir(parents=True,exist_ok=True)
    health=probe(output/"resource_preflight.json",6)
    if health["status"]!="PASS":
        raise RuntimeError("PFC_RESPONSE_RESOURCE_BLOCKED:"+",".join(health["reasons"]))
    cohort=PreparedResponseCohort(Path(inputs),Path(preparation)) if preparation is not None else None
    model,cohort,run=_load(Path(inputs),Path(training),torch.device("cpu"),resource_delivery=resource_delivery,
                           prepared_inputs=cohort)
    receipt=json.loads(Path(receipt_path).read_text())
    declarations=receipt["sources"]
    factual_row=next(row for row in declarations if row["condition_id"]=="cue_none")
    cases={case["episode_id"]:dict(case,partition=partition)
           for partition,items in (("development_fit",cohort.fit),("development_validation",cohort.validation))
           for case in items}
    if (len(declarations)!=10 or any(row["episode_id"] not in cases for row in declarations)
            or len({cases[row["episode_id"]]["partition"] for row in declarations})!=1):
        raise ValueError("PFC_RESPONSE_COMPLETE_SINGLE_PARTITION_SOURCE_BLOCK_REQUIRED")
    factual_case=cases[factual_row["episode_id"]]
    factual=cohort._episode(factual_case)
    cohort.active_case,cohort.active_episode=factual_case,factual
    view=cohort.model_for_case(model,factual_case).eval()
    records=[]
    for index,origin in zip(config["condition_indices"],config["origins_ticks_by_condition"]):
        declaration=next(row for row in declarations
                         if int(row["episode_id"].rsplit("_",1)[1])==index)
        case=cases[declaration["episode_id"]]
        cell=output/f"condition_{index:02d}"
        if (cell/"comparison.json").exists():
            saved=json.loads((cell/"comparison.json").read_text())
            if saved.get("identity")!=identity:
                raise ValueError("PFC_RESPONSE_COMPLETED_CONDITION_BINDING_CHANGED")
            array=cell/"samples.npz"
            if sha256_file(array)!=saved["samples_sha256"]:
                raise ValueError("PFC_RESPONSE_COMPLETED_CONDITION_SAMPLES_CHANGED")
            records.append(saved)
            continue
        candidate=factual if index==0 else cohort._episode(case)
        result,arrays=evaluate_source_response(view,factual,candidate,origin,config,receipt_path,41)
        result.update(identity=identity,condition_index=index,condition_id=declaration["condition_id"],
                      checkpoint_training_seed=run["training_seed"],checkpoint_next_epoch=run["best_epoch"]+1,
                      partition=case["partition"],factual_source_binding=factual.binding,
                      candidate_source_binding=candidate.binding)
        cell.mkdir(parents=True,exist_ok=True)
        array=cell/"samples.npz"
        with array.open("wb") as stream:
            np.savez_compressed(stream,**arrays)
        result["samples_sha256"]=sha256_file(array)
        freeze_json(cell/"comparison.json",result)
        records.append(result)
        del arrays,candidate
    score_rows,contrast_rows,cost_rows=[],[],[]
    for result in records:
        common=dict(condition_index=result["condition_index"],condition_id=result["condition_id"],
                    origin_ticks=result["origin_ticks"],checkpoint_training_seed=run["training_seed"])
        for arm in ("factual","candidate"):
            score_rows += [dict(common,arm=arm,**row) for row in result[arm+"_scores"]]
        contrast_rows += [dict(common,**row) for row in result["mean_contrasts"]]
        cost_rows.append(dict(common,**result["costs"]))
    for name,rows in (("arm_scores.csv",score_rows),("mean_contrasts.csv",contrast_rows),("costs.csv",cost_rows)):
        with (output/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    names=["arm_scores.csv","mean_contrasts.csv","costs.csv"]
    names += [f"condition_{index:02d}/{name}" for index in range(10)
              for name in ("comparison.json","samples.npz")]
    report=dict(status="COMPLETE_TEN_CONDITION_SOURCE_RESPONSE_DEVELOPMENT",identity=identity,
        source_pairing_receipt=str(Path(receipt_path).resolve()),configuration=config,
        checkpoint_training_seed=run["training_seed"],checkpoint_best_epoch=run["best_epoch"],
        checkpoint_status=run["status"],preserved_resource_checkpoint_diagnostic=resource_delivery is not None,
        conditions=10,arm_score_rows=len(score_rows),mean_contrast_rows=len(contrast_rows),
        costs_rows=len(cost_rows),units=len(view.unit_ids),fields=len(view.field_names),
        independent_circuits=1,independent_initial_background_blocks=1,
        candidate_future_is_evaluation_only=True,source_simulations=0,training_updates=0,
        independent_success_gate_evaluated=False,scientific_intervention_qualified=False,
        outputs={name:dict(bytes=(output/name).stat().st_size,sha256=sha256_file(output/name)) for name in names})
    freeze_json(output/"evaluation.json",report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--training",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--source-pairing",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resource-delivery",type=Path)
    parser.add_argument("--preparation",type=Path)
    args=parser.parse_args(argv)
    try:
        result=run_source_responses(args.inputs,args.training,args.config,args.source_pairing,args.output,
                                    resource_delivery=args.resource_delivery,preparation=args.preparation)
    except Exception as error:
        if args.output.exists():
            failure=args.output/"execution_failure.json"
            if not failure.exists():
                freeze_json(failure,dict(status="FAILED_SOURCE_RESPONSE_EVALUATION",
                    error_type=type(error).__name__,error=str(error),
                    completed_conditions_preserved=True,scientific_intervention_qualified=False))
        raise
    print(json.dumps({key:result[key] for key in ("status","conditions","arm_score_rows","mean_contrast_rows",
        "units","scientific_intervention_qualified")}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
