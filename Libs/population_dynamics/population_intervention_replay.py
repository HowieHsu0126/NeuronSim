"""Replay saved pulse-search arrays through the checkpoint simulator directly.

The replay verifies saved field trajectories, unit totals and terminal clocks.
Per-bin full-unit counts were not saved and are not certified by this replay.
No neural training or mechanistic intervention is performed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .mesocircuit_continuation import prepare_pulse_protocol
from .native_renewal_data import NativeRenewalEpisode
from .poisson_state_space import LatentRandomStreams
from .poisson_trajectory_evaluation import fitting_scale
from .population_intervention_check import validate_config, validate_development_config, development_cases
from .population_trajectory_evaluation import _load_model
from .population_state import ControlledPopulationState
from .provenance import sha256_file
from .trajectory_scores import energy_score


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _name(value):
    _require(isinstance(value, str) and bool(value) and Path(value).name == value
             and value not in (".", ".."), "plain artifact or candidate name required")
    return value


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def compare_arrays(saved, expected, scope):
    """Reject every schema, dtype, identity or numerical array difference."""
    _require(set(saved) == set(expected), "saved array schema differs: " + scope)
    values = 0
    for name, actual in saved.items():
        wanted = np.asarray(expected[name])
        _require(actual.shape == wanted.shape and actual.dtype == wanted.dtype
                 and np.array_equal(actual, wanted), "saved array replay differs: " + scope + "/" + name)
        values += actual.size
    return values


def verify_artifacts(results, report, candidate_names):
    expected = {"reference.npz", "candidates.csv"} | {_name(name) + ".npz" for name in candidate_names}
    _require(set(report["artifacts"]) == expected, "complete declared pulse-search artifacts required")
    for name, digest in report["artifacts"].items():
        _require(sha256_file(results / _name(name)) == digest, "pulse-search artifact changed: " + name)


@torch.no_grad()
def direct_prediction(model, state, controls, stream_seeds, draws):
    """Accumulate verified simulator outputs without using the search sampler."""
    _require(type(draws) is int and draws >= 2 and len(stream_seeds) == 3
             and all(type(seed) is int and seed >= 0 for seed in stream_seeds)
             and len(set(stream_seeds)) == 3, "three distinct streams and at least two draws required")
    _require(model.field_unit_counts.device.type == "cpu", "saved CPU engineering replay requires the CPU model")
    units, fields = len(model.unit_ids), len(model.field_names)
    _require(bool(torch.all(model.field_unit_counts > 0)), "positive source field denominators required")
    controls = np.asarray(controls)
    _require(controls.ndim == 3 and controls.shape[0] == 1 and controls.shape[1] > 0
             and controls.shape[2] == model.input_size and np.isfinite(controls).all(), "complete finite replay controls required")
    rng = LatentRandomStreams.from_seeds(*stream_seeds, device="cpu")
    projected, totals, terminal = [], np.zeros((draws, units), dtype=np.int64), None
    for step in model.iter_simulation(state, controls, rng, draws, record_events=False):
        current = step.counts[:, 0]
        _require(step.counts.shape == (draws, 1, units) and current.dtype == torch.int64
                 and bool(torch.all(current >= 0)) and len(step.events) == 0,
                 "complete nonnegative count-only simulator outputs required")
        # Use the complete source-indexed map. The search implementation instead
        # reduces independently stored per-field index buffers.
        counts = torch.zeros((draws, fields), dtype=torch.int64)
        counts.index_add_(-1, model.unit_field_index.cpu(), current.cpu())
        projected.append(counts.numpy())
        totals += current.cpu().numpy()
        terminal = step.ready_in_ticks[:, 0].cpu().numpy().copy()
    _require(len(projected) == controls.shape[1] and terminal is not None,
             "all replay bins and terminal unit clocks required")
    field_counts = np.stack(projected, axis=1)
    return dict(field_counts=field_counts,
                field_rate_hz=field_counts / (model.field_unit_counts.cpu().numpy() * model.bin_seconds),
                unit_total_counts=totals, terminal_ready_in_ticks=terminal,
                unit_ids=np.asarray(model.unit_ids), field_ids=np.asarray(model.field_names))


def _controls(protocol, model, horizon):
    boundary = protocol.decision_tick
    _require(boundary is not None and boundary % model.bin_ticks == 0
             and protocol.resolution_ms / 1000 == model.tick_seconds and model.input_size == 1,
             "complete source-aligned replay boundary required")
    stop = boundary + horizon * model.bin_ticks
    # Reconstruct arrivals independently of population_intervention.future_controls.
    values = np.zeros((1, horizon, 1), dtype=np.float32)
    for emission in protocol.effective_ticks:
        arrival = emission + protocol.delay_ticks
        if boundary <= arrival < stop:
            values[0, (arrival - boundary) // model.bin_ticks, 0] += 1
    return values


def replay_candidates(model, native, report, reference, *, net, sim, initial_emission_ms,
                      decision_time_ms, results):
    """Replay all candidate payloads and reference-block scores from observations."""
    _require(tuple(map(int, native.unit_ids)) == model.unit_ids
             and tuple(map(str, native.field_ids)) == model.field_names
             and np.array_equal(native.unit_field_index, model.unit_field_index.cpu().numpy())
             and np.array_equal(native.minimum_interval_ticks, model.minimum_interval_ticks.cpu().numpy())
             and native.bin_ms / 1000 == model.bin_seconds and native.dt_ms / 1000 == model.tick_seconds,
             "native replay source identity or observation map differs")
    _require(set(reference) == {"field_rate_hz", "scale_hz", "field_ids", "block_ids"},
             "complete reference array schema required")
    rates, scale = reference["field_rate_hz"], reference["scale_hz"]
    _require(rates.ndim == 3 and len(rates) > 0 and rates.shape[-1] == len(model.field_names)
             and rates.shape[1] > 0 and np.isfinite(rates).all() and np.all(rates >= 0)
             and scale.shape == (len(model.field_names),) and np.isfinite(scale).all() and np.all(scale > 0)
             and np.array_equal(reference["field_ids"], np.asarray(model.field_names))
             and reference["block_ids"].shape == (len(rates),)
             and len(set(map(str, reference["block_ids"]))) == len(rates)
             and report["reference_blocks"] == list(map(str, reference["block_ids"]))
             and np.array_equal(report["target_scale_hz"], scale)
             and report["field_names"] == list(model.field_names)
             and report["source_unit_count"] == len(model.unit_ids)
             and report["horizon_bins"] == rates.shape[1], "reference source, block, scale or temporal binding differs")
    original = prepare_pulse_protocol(net, sim, dict(initial_emission_ms=list(initial_emission_ms)))
    _require(original is not None and original.resolution_ms == native.dt_ms
             and np.array_equal(np.asarray(original.initial_ticks) + original.delay_ticks, native.pulse_ticks)
             and native.duration_ticks * native.dt_ms == sim["t_presim"] + sim["t_sim"],
             "original source TC control or complete clock binding differs")
    rows = report["candidates"]
    names = [_name(row["candidate"]) for row in rows]
    _require(bool(names) and len(set(names)) == len(names), "every distinct saved candidate required")
    horizon, protocols = rates.shape[1], []
    for row in rows:
        saved = row["protocol"]
        _require(saved["initial_emission_ms"] == list(initial_emission_ms)
                 and saved["decision_time_ms"] == decision_time_ms
                 and isinstance(saved["requested_future_emission_ms"], list),
                 "saved candidate causal pulse inputs differ")
        protocol = prepare_pulse_protocol(net, sim, dict(initial_emission_ms=list(initial_emission_ms),
            decision_time_ms=decision_time_ms, future_emission_ms=saved["requested_future_emission_ms"]))
        _require(protocol.record() == saved, "saved candidate protocol differs from native reconstruction")
        end = protocol.decision_tick + horizon * model.bin_ticks
        _require(end <= native.duration_ticks and all(t + protocol.delay_ticks < end for t in protocol.future_ticks),
                 "complete strictly future scored pulse support required")
        protocols.append((row, protocol, _controls(protocol, model, horizon)))
    _require(len({controls.tobytes() for _, _, controls in protocols}) == len(protocols),
             "saved candidates are indistinguishable at model resolution")
    boundary = protocols[0][1].decision_tick
    count, delivered = native.window(0, boundary)
    expected_controls = np.zeros((boundary // model.bin_ticks, 1), dtype=np.float32)
    for emission in original.initial_ticks:
        arrival = emission + original.delay_ticks
        if arrival < boundary:
            expected_controls[arrival // model.bin_ticks, 0] += 1
    _require(count.shape == (boundary // model.bin_ticks, len(model.unit_ids))
             and np.array_equal(delivered, expected_controls), "source causal prefix or past TC bins differ")
    with torch.no_grad():
        state = model.infer_state(dict(counts=count[None], stimulus=delivered[None],
                                      ready_in_ticks=native.ready_at(boundary)[None]))
    seeds, draws = tuple(report["sampling_stream_seeds"]), report["joint_draws"]
    compared_values, replay_rows = 0, []
    for row, protocol, controls in protocols:
        expected = direct_prediction(model, state, controls, seeds, draws)
        compared_values += compare_arrays(_arrays(results / (row["candidate"] + ".npz")), expected, row["candidate"])
        sample = expected["field_rate_hz"] / scale
        block_scores = [energy_score(sample, observed / scale) for observed in rates]
        score = float(np.mean(block_scores))
        cost = len(protocol.future_ticks)
        weight = row["pulse_cost_weight"]
        _require(np.isfinite(weight) and weight >= 0 and row["future_pulse_count"] == cost
                 and len(row["reference_block_scores"]) == len(rates)
                 and np.allclose(row["reference_block_scores"], block_scores, rtol=1e-12, atol=1e-12)
                 and np.isclose(row["reference_energy_score"], score, rtol=1e-12, atol=1e-12)
                 and np.isclose(row["objective"], score + weight * cost, rtol=1e-12, atol=1e-12)
                 and row["generated_unit_count_values"] == draws * horizon * len(model.unit_ids),
                 "saved reference scores, pulse cost or generation coverage differs")
        replay_rows.append(dict(candidate=row["candidate"], objective=score + weight * cost,
                                reference_energy_score=score, future_pulse_count=cost))
    selected = min(replay_rows, key=lambda row: (row["objective"], row["future_pulse_count"], row["candidate"]))
    _require(selected["candidate"] == report["selected_candidate"], "saved selected candidate differs")
    with (results / "candidates.csv").open(newline="", encoding="utf-8") as stream:
        table = list(csv.DictReader(stream))
    _require(len(table) == len(rows) and [row["candidate"] for row in table] == names,
             "complete candidate CSV identity/order required")
    for saved, row in zip(table, rows):
        for key in ("reference_energy_score", "future_pulse_count", "objective",
                    "generation_and_projection_seconds", "generated_unit_count_values"):
            _require(np.isfinite(float(saved[key])) and float(saved[key]) == row[key], "candidate CSV/report differs: " + key)
    return dict(candidate_count=len(rows), saved_candidate_array_values_compared=compared_values,
                reference_blocks_compared=len(rates), selected_candidate=selected["candidate"], candidate_scores=replay_rows,
                saved_field_counts_exact=True, saved_field_rates_exact=True, saved_all_unit_totals_exact=True,
                saved_terminal_unit_clocks_exact=True, source_identities_exact=True,
                per_bin_full_unit_counts_saved=False, per_bin_full_unit_counts_exact_replay_claimed=False)


def native_references(acquisition, fields, native, model, cases, names, boundary, horizon):
    references, source_bindings = [], []
    for name in names:
        source = NativeRenewalEpisode(acquisition / name, fields / name,
            expected_source=native.source_identity, expected_unit_ids=native.unit_ids,
            expected_unit_field_index=native.unit_field_index)
        counts, _ = source.window(boundary, boundary + horizon * native.bin_ticks)
        projected = np.zeros((horizon, len(model.field_names)), dtype=np.float64)
        for f in range(len(model.field_names)):
            projected[:, f] = counts[:, source.unit_field_index == f].sum(axis=-1, dtype=np.float64)
        references.append(projected / (model.field_unit_counts.cpu().numpy() * model.bin_seconds))
        source_bindings.append(dict(case_id=name, block=cases[name]["block"], native=source.binding))
    _, scale, metadata = fitting_scale(fields, [cases[name] for name in names])
    _require(np.array_equal(metadata["field_ids"], native.field_ids)
             and np.array_equal(metadata["unit_count"], model.field_unit_counts.cpu().numpy()), "fitting-scale observation map changed")
    reference = dict(field_rate_hz=np.stack(references), scale_hz=scale,
                     field_ids=np.asarray(model.field_names), block_ids=np.asarray([str(cases[name]["block"]) for name in names]))
    return reference, source_bindings


def replay(acquisition: Path, fields: Path, checkpoint: Path, model_receipt: Path,
           graph_manifest: Path, config_path: Path, results: Path, output: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    report, receipt = _read(results / "analysis.json"), _read(model_receipt)
    _require(report["status"] == "ENGINEERING_NATIVE_PULSE_SEARCH_EXECUTED" and report["configuration"] == config
             and report["training_performed"] is False and report["mechanistic_simulation_performed"] is False
             and report["native_execution_performed"] is False and report["inverse_intervention_qualified"] is False
             and report["scientific_claim_qualified"] is False, "complete engineering-only pulse search required")
    _require(receipt["status"] == "ENGINEERING_EXECUTED" and receipt["core_neurons"] == 100000
             and receipt["episode_ms"] == 6000 and sha256_file(checkpoint) == receipt["artifacts"]["model.pt"]
             == report["checkpoint_sha256"] and sha256_file(model_receipt) == report["model_receipt_sha256"],
             "checkpoint or engineering receipt binding changed")
    binding = receipt["binding"]
    for name, digest in binding["implementation_sources"].items():
        _require(sha256_file(Path(__file__).with_name(_name(name))) == digest,
                 "checkpoint numerical implementation changed: " + name)
    _require(sha256_file(graph_manifest) == report["graph_manifest_sha256"]
             == binding["native"]["native_source_identity"]["graph_manifest_sha256"], "native graph control binding changed")
    verify_artifacts(results, report, [row["candidate"] for row in report["candidates"]])
    design, graph = _read(acquisition / "design.json"), _read(graph_manifest)
    cases = {row["case_id"]: row for row in design["cases"]}
    names = config["reference_case_ids"]
    _require(len(cases) == len(design["cases"]) and all(name in cases for name in names)
             and report["target_scaling_cases"] == names and report["reference_scope"] == config["reference_role"],
             "native reference case or fitting-scale provenance changed")
    for name, block in zip(names, range(4)):
        _require(cases[name]["split"] == "train" and cases[name]["template_id"] == "familiar_0"
                 and cases[name]["block"] == block, "all four ordered training reference blocks required")
    torch.set_num_threads(1)
    rng_before = torch.get_rng_state().clone()
    model = ControlledPopulationState.load(checkpoint, device="cpu").eval()
    native = NativeRenewalEpisode(acquisition / config["case_id"], fields / config["case_id"],
        expected_source=binding["native"]["native_source_identity"], expected_unit_ids=np.asarray(model.unit_ids),
        expected_unit_field_index=model.unit_field_index.cpu().numpy())
    _require(native.binding == binding["native"] == report["native_binding"], "checkpoint/search native episode changed")
    boundary_ratio, horizon_ratio = config["decision_time_ms"] / native.dt_ms, config["horizon_ms"] / native.bin_ms
    _require(boundary_ratio == int(boundary_ratio) and horizon_ratio == int(horizon_ratio), "aligned native replay window required")
    boundary, horizon = int(boundary_ratio), int(horizon_ratio)
    reference, source_bindings = native_references(acquisition, fields, native, model, cases, names, boundary, horizon)
    _require(source_bindings == report["reference_bindings"], "native reference episode bindings changed")
    reference_values = compare_arrays(_arrays(results / "reference.npz"), reference, "reference")
    initial = cases[config["case_id"]]["settings"]["pulse_control"]["initial_emission_ms"]
    delay = graph["resolved_source_network"]["th_delay_pulse_generator"]
    future = [value for value in initial if config["decision_time_ms"] < value
              and value + delay < config["decision_time_ms"] + config["horizon_ms"]]
    expected_schedules = [("omit", []), ("first_future_pulse", future[:1]), ("original_future_pulses", future)]
    _require(len(future) >= 2 and [(r["candidate"], r["protocol"]["requested_future_emission_ms"])
             for r in report["candidates"]] == expected_schedules
             and report["sampling_stream_seeds"] == config["sampling_stream_seeds"]
             and report["joint_draws"] == config["joint_draws"]
             and all(r["pulse_cost_weight"] == config["pulse_cost_weight"] for r in report["candidates"]),
             "registered candidate search, sampling streams or cost changed")
    sim = dict(graph["source_config"]["simulation"], t_presim=0., t_sim=native.duration_ticks * native.dt_ms)
    evidence = replay_candidates(model, native, report, reference, net=graph["resolved_source_network"], sim=sim,
        initial_emission_ms=initial, decision_time_ms=config["decision_time_ms"], results=results)
    _require(torch.equal(rng_before, torch.get_rng_state()), "global RNG changed during replay")
    evidence.update(status="INDEPENDENT_SAVED_ARRAY_PULSE_SEARCH_REPLAYED", reference_array_values_compared=reference_values,
        configuration=config, analysis_sha256=sha256_file(results / "analysis.json"), checkpoint_sha256=sha256_file(checkpoint),
        model_receipt_sha256=sha256_file(model_receipt), graph_manifest_sha256=sha256_file(graph_manifest),
        replay_implementation_sha256=sha256_file(Path(__file__)), global_rng_preserved=True,
        native_intervention_execution_performed=False, scientific_claim_qualified=False,
        inverse_intervention_qualified=False, native_future_noise_pairing_verified=False,
        completed_utc=datetime.now(timezone.utc).isoformat())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return evidence


def replay_development(acquisition, fields, campaign, graph_manifest, config_path, results, output):
    """Verify all trained development seeds through independent saved-array replay."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_development_config(config)
    aggregate = _read(results / "analysis.json")
    _require(aggregate["status"] == "THREE_SEED_DEVELOPMENT_NATIVE_PULSE_SEARCH_EXECUTED"
             and aggregate["configuration"] == config and aggregate["completed_seeds"] == config["training_seeds"]
             and aggregate["training_performed"] is False and aggregate["mechanistic_simulation_performed"] is False
             and aggregate["scientific_claim_qualified"] is False and aggregate["independent_intervention_qualified"] is False
             and aggregate["candidate_results"] == 9 and aggregate["reference_blocks"] == 14
             and aggregate["selection_blocks"] == 1, "complete unqualified three-seed development search required")
    cases = development_cases(_read(acquisition / "design.json"), config)
    graph = _read(graph_manifest)
    native = NativeRenewalEpisode(acquisition / config["case_id"], fields / config["case_id"])
    graph_digest = sha256_file(graph_manifest)
    _require(graph_digest == native.source_identity["graph_manifest_sha256"], "native graph control binding changed")
    sim = dict(graph["source_config"]["simulation"], t_presim=0., t_sim=native.duration_ticks * native.dt_ms)
    initial = cases[config["case_id"]]["settings"]["pulse_control"]["initial_emission_ms"]
    delay = graph["resolved_source_network"]["th_delay_pulse_generator"]
    future = [value for value in initial if config["decision_time_ms"] < value
              and value + delay < config["decision_time_ms"] + config["horizon_ms"]]
    expected_schedules = [("omit", []), ("first_future_pulse", future[:1]), ("original_future_pulses", future)]
    _require(len(future) >= 2, "complete development pulse schedule required")
    names = config["reference_case_ids"]
    cells = aggregate["seeds"]
    _require([row["training_seed"] for row in cells] == config["training_seeds"], "complete ordered development seed payloads required")
    torch.set_num_threads(1)
    rng_before = torch.get_rng_state().clone()
    evidence, reference = [], None
    for seed, declared in zip(config["training_seeds"], cells):
        cell = results / f"seed_{seed}"
        _require(sha256_file(cell / "analysis.json") == declared["analysis_sha256"], "development seed analysis changed")
        report = _read(cell / "analysis.json")
        model, binding, checkpoint = _load_model(campaign / f"seed_{seed}/training", acquisition, fields,
                                                "population", torch.device("cpu"), native)
        _require(report["status"] == "DEVELOPMENT_NATIVE_PULSE_SEARCH_EXECUTED"
                 and report["configuration"] == config and report["training_seed"] == seed == binding["seed"]
                 and report["training_binding"] == binding
                 and report["checkpoint_sha256"] == checkpoint == declared["checkpoint_sha256"]
                 and report["training_receipt_sha256"] == sha256_file(campaign / f"seed_{seed}/training/run.json")
                 and report["graph_manifest_sha256"] == graph_digest and report["native_binding"] == native.binding
                 and report["training_performed"] is False and report["mechanistic_simulation_performed"] is False
                 and report["scientific_claim_qualified"] is False and report["inverse_intervention_qualified"] is False,
                 "trained seed source, checkpoint, receipt or declared scientific scope changed")
        _require([(row["candidate"],row["protocol"]["requested_future_emission_ms"]) for row in report["candidates"]] == expected_schedules
                 and report["sampling_stream_seeds"] == config["sampling_stream_seeds"]
                 and report["joint_draws"] == config["joint_draws"]
                 and all(row["pulse_cost_weight"] == config["pulse_cost_weight"] for row in report["candidates"])
                 and report["target_scaling_cases"] == names and report["reference_scope"] == config["reference_role"]
                 and report["candidates"] == declared["candidates"] and report["selected_candidate"] == declared["selected_candidate"],
                 "registered development candidate, reference, sampling or aggregate values changed")
        verify_artifacts(cell, report, [row["candidate"] for row in report["candidates"]])
        boundary, horizon = round(config["decision_time_ms"] / native.dt_ms), round(config["horizon_ms"] / native.bin_ms)
        if reference is None:
            reference, source_bindings = native_references(acquisition, fields, native, model, cases, names, boundary, horizon)
        _require(source_bindings == report["reference_bindings"], "development native reference bindings changed")
        reference_values = compare_arrays(_arrays(cell / "reference.npz"), reference, f"seed_{seed}/reference")
        result = replay_candidates(model, native, report, reference, net=graph["resolved_source_network"], sim=sim,
            initial_emission_ms=initial, decision_time_ms=config["decision_time_ms"], results=cell)
        result.update(training_seed=seed,checkpoint_sha256=checkpoint,reference_array_values_compared=reference_values,
                      analysis_sha256=sha256_file(cell / "analysis.json"))
        evidence.append(result)
    _require(torch.equal(rng_before,torch.get_rng_state()), "global RNG changed during development replay")
    report = dict(status="INDEPENDENT_THREE_SEED_SAVED_ARRAY_PULSE_SEARCH_REPLAYED", completed_seeds=config["training_seeds"],
        seeds=evidence, configuration=config, analysis_sha256=sha256_file(results / "analysis.json"),
        replay_implementation_sha256=sha256_file(Path(__file__)), global_rng_preserved=True,
        native_intervention_execution_performed=False, scientific_claim_qualified=False, inverse_intervention_qualified=False,
        per_bin_full_unit_counts_saved=False, per_bin_full_unit_counts_exact_replay_claimed=False,
        completed_utc=datetime.now(timezone.utc).isoformat())
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "fields", "graph-manifest", "config", "results", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint",type=Path)
    mode.add_argument("--campaign",type=Path)
    parser.add_argument("--model-receipt",type=Path)
    args = parser.parse_args()
    if args.campaign:
        if args.model_receipt:
            parser.error("--model-receipt applies to --checkpoint")
        report = replay_development(args.acquisition,args.fields,args.campaign,args.graph_manifest,args.config,args.results,args.output)
    else:
        if args.model_receipt is None:
            parser.error("--checkpoint requires --model-receipt")
        report = replay(args.acquisition,args.fields,args.checkpoint,args.model_receipt,args.graph_manifest,args.config,args.results,args.output)
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
