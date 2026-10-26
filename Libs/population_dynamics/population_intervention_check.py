"""Run native pulse candidate ranking with accepted population checkpoints.

This check uses actual source units, observations, controls and reference
episodes. It does not refit a model or execute a new mechanistic simulation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
import yaml

from .mesocircuit_continuation import prepare_pulse_protocol
from .population_trajectory_evaluation import _load_model
from .native_renewal_data import NativeRenewalEpisode
from .population_intervention import PulseCandidate, ReferenceTarget, design_intervention
from .population_state import ControlledPopulationState
from .poisson_trajectory_evaluation import fitting_scale
from .provenance import sha256_file


def validate_config(config):
    expected = dict(schema="neural_population_intervention_engineering_check_v1",
        scope="engineering_only_existing_checkpoint", case_id="train_block_000_familiar_0",
        reference_case_ids=[f"train_block_{b:03d}_familiar_0" for b in range(4)],
        reference_role="development_response_distribution_not_independent_test",
        decision_time_ms=1000, horizon_ms=500,
        candidate_rule="omission_first_future_pulse_and_all_original_future_pulses_in_horizon",
        joint_draws=4, sampling_stream_seeds=[77001, 77002, 77003], pulse_cost_weight=0.,
        scientific_success_tolerance=None, native_intervention_execution=False, repeat_training=False)
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("registered inference-only engineering check required")


def reference_target(native, cases, config, acquisition, fields):
    names = config["reference_case_ids"]
    unit_counts = np.bincount(native.unit_field_index, minlength=len(native.field_ids))
    bin_seconds = native.bin_ms / 1000
    boundary = round(config["decision_time_ms"] / native.dt_ms)
    horizon = round(config["horizon_ms"] / native.bin_ms)
    reference_hz, reference_bindings = [], []
    for name in names:
        source = NativeRenewalEpisode(acquisition / name, fields / name,
            expected_source=native.source_identity, expected_unit_ids=native.unit_ids,
            expected_unit_field_index=native.unit_field_index)
        values, _ = source.window(boundary, boundary + horizon * native.bin_ticks)
        field_counts = np.stack([values[:, source.unit_field_index == f].sum(-1, dtype=np.float64)
                                for f in range(len(source.field_ids))], -1)
        reference_hz.append(field_counts / (unit_counts * bin_seconds))
        reference_bindings.append(dict(case_id=name, block=cases[name]["block"], native=source.binding))
        del values, source
    # Reuse the established fit-only scale calculation on these training cases.
    _, scale, scale_metadata = fitting_scale(fields, [cases[name] for name in names])
    if (not np.array_equal(scale_metadata["field_ids"], native.field_ids)
            or not np.array_equal(scale_metadata["unit_count"], unit_counts)):
        raise ValueError("fitting reference observation map changed")
    target = ReferenceTarget(np.stack(reference_hz), tuple(map(str, native.field_ids)), scale,
                             tuple(str(cases[name]["block"]) for name in names), bin_seconds)
    return target, reference_bindings


def execute_search(model, native, cases, config, graph, acquisition, fields, output, provenance, status, prepared_reference=None):
    names = config["reference_case_ids"]
    target, reference_bindings = (reference_target(native, cases, config, acquisition, fields)
                                 if prepared_reference is None else prepared_reference)
    boundary = round(config["decision_time_ms"] / native.dt_ms)
    y, u = native.window(0, boundary)
    history = dict(counts=y[None], stimulus=u[None], ready_in_ticks=native.ready_at(boundary)[None])
    initial = cases[config["case_id"]]["settings"]["pulse_control"]["initial_emission_ms"]
    delay = graph["resolved_source_network"]["th_delay_pulse_generator"]
    future = tuple(v for v in initial if v > config["decision_time_ms"]
                   and v + delay < config["decision_time_ms"] + config["horizon_ms"])
    if len(future) < 2:
        raise ValueError("source episode must support distinct omission, single and full pulse candidates")
    candidates = (PulseCandidate("omit", ()), PulseCandidate("first_future_pulse", future[:1]),
                  PulseCandidate("original_future_pulses", future))
    sim = dict(graph["source_config"]["simulation"], t_presim=0., t_sim=native.duration_ticks*native.dt_ms)
    started = time.perf_counter()
    report, predictions = design_intervention(model, history, target, native=native,
        net=graph["resolved_source_network"], sim=sim, initial_emission_ms=initial,
        decision_time_ms=config["decision_time_ms"], candidates=candidates,
        stream_seeds=tuple(config["sampling_stream_seeds"]), draws=config["joint_draws"],
        pulse_cost_weight=config["pulse_cost_weight"])
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "reference.npz", field_rate_hz=target.field_rate_hz,
                        scale_hz=target.scale_hz, field_ids=np.asarray(target.field_names),
                        block_ids=np.asarray(target.independent_block_ids))
    for name, prediction in predictions.items():
        np.savez_compressed(output / (name + ".npz"), **{k:v for k,v in prediction.items() if isinstance(v, np.ndarray)},
                            unit_ids=np.asarray(model.unit_ids), field_ids=np.asarray(model.field_names))
    rows = [dict(candidate=r["candidate"], reference_energy_score=r["reference_energy_score"],
                 future_pulse_count=r["future_pulse_count"], objective=r["objective"],
                 generation_and_projection_seconds=r["generation_and_projection_seconds"],
                 generated_unit_count_values=r["generated_unit_count_values"]) for r in report["candidates"]]
    with (output / "candidates.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report.update(status=status, configuration=config, **provenance, native_binding=native.binding,
                  reference_bindings=reference_bindings, reference_scope=config["reference_role"],
                  target_scaling_cases=names, training_performed=False, mechanistic_simulation_performed=False,
                  elapsed_seconds=time.perf_counter()-started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  artifacts={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
    (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def check(acquisition, fields, checkpoint, model_receipt, graph_manifest, config_path, output):
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    receipt = json.loads(model_receipt.read_text())
    if (receipt["status"] != "ENGINEERING_EXECUTED"
            or receipt["core_neurons"] != 100000 or receipt["episode_ms"] != 6000
            or sha256_file(checkpoint) != receipt["artifacts"]["model.pt"]):
        raise ValueError("unchanged accepted full-unit engineering checkpoint required")
    binding = receipt["binding"]
    for name, digest in binding["implementation_sources"].items():
        if sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError("engineering checkpoint numerical implementation changed:" + name)
    if sha256_file(graph_manifest) != binding["native"]["native_source_identity"]["graph_manifest_sha256"]:
        raise ValueError("native graph control declaration changed")
    graph = json.loads(graph_manifest.read_text())
    design = json.loads((acquisition / "design.json").read_text())
    cases = {r["case_id"]: r for r in design["cases"]}
    names = config["reference_case_ids"]
    if len(cases) != len(design["cases"]) or any(name not in cases for name in names):
        raise ValueError("complete identified native reference cases required")
    for name in names:
        case = cases[name]
        if case["split"] != "train" or case["template_id"] != "familiar_0" or case["block"] not in range(4):
            raise ValueError("registered training-only reference blocks required")
    torch.set_num_threads(1)
    model = ControlledPopulationState.load(checkpoint, device="cpu").eval()
    native = NativeRenewalEpisode(acquisition / config["case_id"], fields / config["case_id"],
        expected_source=binding["native"]["native_source_identity"], expected_unit_ids=np.asarray(model.unit_ids),
        expected_unit_field_index=model.unit_field_index.numpy())
    if native.binding != binding["native"]:
        raise ValueError("engineering checkpoint observed source changed")
    provenance = dict(checkpoint_sha256=sha256_file(checkpoint), model_receipt_sha256=sha256_file(model_receipt),
                      graph_manifest_sha256=sha256_file(graph_manifest))
    return execute_search(model, native, cases, config, graph, acquisition, fields, output, provenance,
                          "ENGINEERING_NATIVE_PULSE_SEARCH_EXECUTED")


def validate_development_config(config):
    expected = dict(schema="neural_population_intervention_development_v1", scope="development_only",
        training_seeds=[41, 42, 43], case_id="train_block_014_familiar_0", selection_block=14,
        reference_case_ids=[f"train_block_{b:03d}_familiar_0" for b in range(14)],
        reference_role="fitting_response_distribution_not_independent_intervention_test",
        decision_time_ms=1000, horizon_ms=500, joint_draws=16,
        candidate_rule="omission_first_future_pulse_and_all_original_future_pulses_in_horizon",
        sampling_stream_seeds=[77001, 77002, 77003], pulse_cost_weight=0.,
        expected_units=100000, expected_fields=128, expected_episodes=72,
        scientific_success_tolerance=None, native_intervention_execution=False, repeat_training=False)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("registered bounded three-seed development search required")


def development_cases(design, config):
    cases = {row["case_id"]: row for row in design["cases"]}
    if len(cases) != config["expected_episodes"] or len(cases) != len(design["cases"]):
        raise ValueError("complete unique development episode identities required")
    selected = cases.get(config["case_id"])
    if (selected is None or selected["split"] != "train" or selected["block"] != config["selection_block"]
            or selected["template_id"] != "familiar_0"):
        raise ValueError("registered development selection case required")
    for block, name in enumerate(config["reference_case_ids"]):
        case = cases.get(name)
        if (case is None or case["split"] != "train" or case["block"] != block
                or case["template_id"] != "familiar_0" or block == config["selection_block"]
                or case["settings"]["pulse_control"] != selected["settings"]["pulse_control"]):
            raise ValueError("fitting-only reference blocks and identical known control program required")
    return cases


def load_development_models(campaign, acquisition, fields, native, config):
    models = []
    for seed in config["training_seeds"]:
        training = campaign / f"seed_{seed}/training"
        model, binding, checkpoint = _load_model(training, acquisition, fields, "population", torch.device("cpu"), native)
        if (binding["seed"] != seed or binding["config"]["training_seeds"] != config["training_seeds"]
                or binding["config"]["fit_blocks"] != list(range(14))
                or binding["config"]["validation_blocks"] != list(range(14, 18))):
            raise ValueError("complete registered three-seed training and fitting partition required")
        models.append((seed, model, binding, checkpoint, sha256_file(training / "run.json")))
    return models


def development(acquisition, fields, campaign, graph_manifest, config_path, output, validate_only=False):
    config = yaml.safe_load(config_path.read_text())
    validate_development_config(config)
    cases = development_cases(json.loads((acquisition / "design.json").read_text()), config)
    native = NativeRenewalEpisode(acquisition / config["case_id"], fields / config["case_id"])
    if (len(native.unit_ids) != config["expected_units"] or len(native.field_ids) != config["expected_fields"]
            or native.duration_ticks * native.dt_ms != 6000
            or native.source_identity["native_poisson_rng_family"] != "per_edge_philox4_32_10"
            or sha256_file(graph_manifest) != native.source_identity["graph_manifest_sha256"]):
        raise ValueError("qualified implementation inputs with complete source scale and graph identity required")
    graph = json.loads(graph_manifest.read_text())
    initial = cases[config["case_id"]]["settings"]["pulse_control"]["initial_emission_ms"]
    sim = dict(graph["source_config"]["simulation"], t_presim=0., t_sim=native.duration_ticks*native.dt_ms)
    original = prepare_pulse_protocol(graph["resolved_source_network"], sim, dict(initial_emission_ms=initial))
    if (original.resolution_ms != native.dt_ms or not np.array_equal(
            np.asarray(original.initial_ticks, dtype=np.int64)+original.delay_ticks, native.pulse_ticks)):
        raise ValueError("native development control binding changed")
    boundary, horizon = config["decision_time_ms"] / native.dt_ms, config["horizon_ms"] / native.bin_ms
    if boundary != round(boundary) or round(boundary) % native.bin_ticks or horizon != round(horizon):
        raise ValueError("aligned development decision and forecast horizon required")
    end = round(boundary) + round(horizon) * native.bin_ticks
    future = [v for v in initial if v > config["decision_time_ms"]
              and v + original.delay_ticks * native.dt_ms < end * native.dt_ms]
    if len(future) < 2 or end > native.duration_ticks:
        raise ValueError("distinct development pulse candidates with complete response support required")
    for schedule in ([], future[:1], future):
        prepare_pulse_protocol(graph["resolved_source_network"], sim, dict(initial_emission_ms=initial,
            decision_time_ms=config["decision_time_ms"], future_emission_ms=schedule))
    target, references = reference_target(native, cases, config, acquisition, fields)
    if validate_only:
        report = dict(status="PREPARED_NATIVE_PULSE_SEARCH_DEVELOPMENT_INPUTS", configuration=config,
            native_units=len(native.unit_ids), fields=len(native.field_ids), reference_blocks=len(references),
            native_binding=native.binding, reference_bindings=references,
            graph_manifest_sha256=sha256_file(graph_manifest), fitting_scale_hz=target.scale_hz.tolist(),
            training_performed=False, inference_performed=False, mechanistic_simulation_performed=False,
            scientific_claim_qualified=False, independent_intervention_qualified=False)
        output.mkdir(parents=True, exist_ok=False)
        (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        return report
    # All three completed checkpoints must pass before generating any candidate.
    models = load_development_models(campaign, acquisition, fields, native, config)
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    torch.set_num_threads(1)
    for seed, model, binding, checkpoint, receipt in models:
        provenance = dict(training_seed=seed, training_binding=binding, checkpoint_sha256=checkpoint,
                          training_receipt_sha256=receipt, graph_manifest_sha256=sha256_file(graph_manifest))
        report = execute_search(model, native, cases, config, graph, acquisition, fields, output / f"seed_{seed}", provenance,
                                "DEVELOPMENT_NATIVE_PULSE_SEARCH_EXECUTED", (target, references))
        reports.append(dict(training_seed=seed, selected_candidate=report["selected_candidate"],
            checkpoint_sha256=checkpoint, candidates=report["candidates"],
            analysis_sha256=sha256_file(output / f"seed_{seed}/analysis.json")))
    report = dict(status="THREE_SEED_DEVELOPMENT_NATIVE_PULSE_SEARCH_EXECUTED", configuration=config,
        completed_seeds=[row["training_seed"] for row in reports], seeds=reports,
        reference_blocks=14, selection_blocks=1, candidate_results=9,
        training_performed=False, mechanistic_simulation_performed=False,
        scientific_claim_qualified=False, independent_intervention_qualified=False,
        selection_used_for_checkpoint_selection=True, success_tolerance_qualified=False)
    (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "fields", "graph-manifest", "config", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path)
    mode.add_argument("--campaign", type=Path)
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.campaign:
        if args.model_receipt:
            parser.error("--model-receipt applies to --checkpoint")
        report = development(args.acquisition, args.fields, args.campaign, args.graph_manifest,
                             args.config, args.output, args.validate_only)
    else:
        if args.model_receipt is None or args.validate_only:
            parser.error("--checkpoint requires --model-receipt and supports execution only")
        report = check(args.acquisition, args.fields, args.checkpoint, args.model_receipt,
                       args.graph_manifest, args.config, args.output)
    print(json.dumps({key:value for key,value in report.items()
        if key not in ("candidates", "reference_bindings", "native_binding", "seeds", "fitting_scale_hz")}), flush=True)


if __name__ == "__main__":
    main()
