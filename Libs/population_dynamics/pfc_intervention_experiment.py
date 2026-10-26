"""Three-seed PFC stimulus selection and independent retained-array replay.

Source admission may read complete episodes for integrity. Candidate prediction
uses only the factual prefix and compiled commands. No mechanistic intervention
or memory-success qualification is performed. A caller must establish that a
previous writer is terminal before requesting reuse from its output directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import torch
import yaml

from .benchmark_tasks import pfc_working_memory_config, task_protocol
from .braid_population_comparison import DeliveredCell
from .pfc_compiled_response import prepare_compiled_response
from .pfc_intervention_references import load_reference
from .pfc_population_evaluation import _load
from .pfc_population_forecast_analysis import FrozenInputs, ORIGINS
from .pfc_population_intervention import PfcInterventionCandidate, design_pfc_intervention
from .pfc_population_training_data import PfcPopulationTrainingCohort
from .population_intervention_replay import _arrays, compare_arrays, direct_prediction, verify_artifacts
from .provenance import canonical_json, freeze_json, sha256_file
from .trajectory_scores import energy_score

SEEDS = (41, 42, 43)
SEED_CELL_FILES = (
    "training/run.json", "training/best.pt", "forecasts/evaluation.json", "replay.json",
    "phases/fit/receipt.json", "phases/evaluate/receipt.json", "phases/replay/receipt.json",
)
PREDICTION_ARRAYS = ("field_counts", "field_rate_hz", "unit_total_counts", "terminal_ready_in_ticks")
FACTUAL_EPISODES = tuple(f"pfc_working_memory_circuit_074_block_{block:03d}_condition_00"
                        for block in range(3))


def _require(condition, message):
    if not condition:
        raise ValueError("PFC_INTERVENTION_EXPERIMENT_" + message)


def _read(path):
    return json.loads(Path(path).read_text())


def validate_config(config):
    expected = dict(schema="pfc_population_intervention_experiment_v1",
        evaluation_scope="development_validation_stimulus_selection", training_seeds=list(SEEDS),
        factual_episode_ids=list(FACTUAL_EPISODES), candidate_condition_indices=list(range(10)),
        origin_tick=2000, horizon_bins=980, joint_draws=16, sampling_stream_seeds=list(SEEDS),
        control_geometry="actuation_only", formal_evaluation=False, scientific_claim_qualified=False)
    _require(isinstance(config, dict) and set(config) == set(expected) | {"reference_sha256"},
             "EXPLICIT_CONFIGURATION_REQUIRED")
    for name, value in expected.items():
        _require(canonical_json(config[name]) == canonical_json(value), "REGISTERED_CONFIGURATION_REQUIRED:" + name)
    _require(isinstance(config["reference_sha256"], str)
             and re.fullmatch("[0-9a-f]{64}", config["reference_sha256"]) is not None,
             "EXPLICIT_REFERENCE_DIGEST_REQUIRED")
    return config


def admit_seed_cells(seed_cells):
    """Authenticate terminal metadata and checkpoints; return delivered cells.

    A staging consumer needs delivery.json and exactly SEED_CELL_FILES from
    each explicit seed path. Forecast payloads are not consumed by this step.
    """
    _require(set(seed_cells) == set(SEEDS) and all(type(seed) is int for seed in seed_cells),
             "EXACT_THREE_EXPLICIT_SEED_CELLS_REQUIRED")
    cells = {seed: DeliveredCell(Path(seed_cells[seed]), seed) for seed in SEEDS}
    admitted, preparation_hash = {}, None
    for seed, cell in cells.items():
        proof = cell.proof
        _require(proof.get("input_family") == "pfc_source_reproduction"
                 and proof.get("calibration_or_test_accessed") is False
                 and proof.get("formal_evaluation") is False and proof.get("device") == "cpu"
                 and (proof.get("fit_episodes"), proof.get("validation_episodes")) == (80, 30)
                 and proof.get("core_neurons") == 100000, "COMPLETE_PFC_DELIVERY_REQUIRED")
        if preparation_hash is None:
            preparation_hash = proof.get("preparation_sha256")
        _require(isinstance(preparation_hash, str) and proof.get("preparation_sha256") == preparation_hash,
                 "COMMON_FROZEN_PREPARATION_REQUIRED")
        for phase in ("fit", "evaluate", "replay"):
            receipt = cell.read(f"phases/{phase}/receipt.json")
            _require(receipt.get("phase") == phase and receipt.get("status") == "PASS"
                     and receipt.get("command_started") is True and receipt.get("exit_code") == 0
                     and bool(receipt.get("finished_utc")), "TERMINAL_PHASE_REQUIRED:" + phase)
        run = cell.read("training/run.json")
        evaluation, replay = cell.read("forecasts/evaluation.json"), cell.read("replay.json")
        cell.path("training/best.pt")
        config = run["binding"]["config"]
        completed, best = run["completed_epochs"], run["best_epoch"]
        _require(run.get("status") == "FITTED_DEVELOPMENT" and run.get("training_seed") == seed
                 and run.get("input_family") == "pfc_source_reproduction" and run.get("units") == 100000
                 and (run.get("fitting_episodes"), run.get("validation_episodes")) == (80, 30)
                 and run["binding"]["seed"] == seed and config.get("training_seeds") == list(SEEDS)
                 and config.get("control_geometry") == run["model_configuration"].get("control_geometry") == "actuation_only"
                 and run.get("scientific_claim_qualified") is False
                 and run.get("calibration_outcomes_accessed") is False and run.get("test_outcomes_accessed") is False
                 and cell.files["training/best.pt"]["sha256"] == run["best_checkpoint_sha256"],
                 "TERMINAL_ACTUATION_CHECKPOINT_REQUIRED")
        _require(type(completed) is int and type(best) is int
                 and 0 <= best < completed <= config["maximum_epochs"]
                 and run.get("effective_maximum_epochs") == config["maximum_epochs"]
                 and ((run.get("stop_reason") == "epoch_budget" and completed == config["maximum_epochs"])
                      or (run.get("stop_reason") == "validation_patience"
                          and completed - best - 1 >= config["early_stopping_patience"])),
                 "REGISTERED_TERMINAL_TRAINING_STOP_REQUIRED")
        records = run["binding"]["complete_episode_bindings"]
        validation = [row for row in records if row["partition"] == "development_validation"]
        expected = [(row["circuit_id"] + "__" + row["episode_id"], origin)
                    for row in validation for origin in ORIGINS]
        _require(len(records) == 110 and len(validation) == 30
                 and evaluation.get("status") == "EVALUATED_PFC_DEVELOPMENT_CHECKPOINT"
                 and evaluation.get("training_seed") == seed
                 and evaluation.get("checkpoint_sha256") == replay.get("checkpoint_sha256") == run["best_checkpoint_sha256"]
                 and evaluation.get("inputs_sha256") == evaluation.get("original_training_inputs_sha256")
                     == run["binding"]["circuit_inputs_sha256"]
                 and evaluation.get("episodes") == 30 and evaluation.get("score_rows") == 360
                 and evaluation.get("scientific_claim_qualified") is False
                 and evaluation.get("formal_evaluation") is False
                 and replay.get("status") == "PASS_ALL_UNIT_AND_INDEPENDENT_METRIC_REPLAY"
                 and replay.get("scientific_claim_qualified") is False
                 and replay.get("unit_values_consumed") == 16 * 100 * 100000 * 120
                 and [(row["case_id"], row["origin_tick"]) for row in evaluation["origins"]] == expected
                 and [(row["case_id"], row["origin_tick"]) for row in replay["origins"]] == expected
                 and all(row["unit_values"] == 16 * 100 * 100000 for row in replay["origins"]),
                 "COMPLETE_CHECKPOINT_FORECAST_AND_REPLAY_REQUIRED")
        for index, origin in enumerate(evaluation["origins"]):
            source = {key: value for key, value in validation[index // len(ORIGINS)].items() if key != "partition"}
            _require(origin["source_binding"] == source and origin["units"] == 100000 and origin["fields"] == 17,
                     "FORECAST_SOURCE_BINDING_CHANGED")
        admitted[seed] = dict(cell=cell, run=run, evaluation=evaluation, replay=replay,
                              binding=deepcopy(cell.binding()))
    return admitted


def _context(inputs, preparation, config_path, reference_path, seed_cells):
    config = validate_config(yaml.safe_load(config_path.read_text()))
    _require(sha256_file(reference_path) == config["reference_sha256"], "REFERENCE_CHANGED")
    target, reference_report = load_reference(reference_path)
    admitted = admit_seed_cells(seed_cells)  # All terminal cells precede source admission or generation.
    frozen = FrozenInputs(preparation)
    input_hash, proof_hash = sha256_file(inputs), sha256_file(Path(preparation) / "preparation.json")
    _require(frozen.proof["input_sha256"] == input_hash == reference_report["inputs_sha256"]
             and all(row["cell"].proof["preparation_sha256"] == proof_hash for row in admitted.values())
             and frozen.config.get("control_geometry") == "actuation_only",
             "INPUT_REFERENCE_AND_FROZEN_PREPARATION_DIFFER")
    for row in admitted.values():
        run, cell = row["run"], row["cell"]
        expected_config = dict(frozen.config, device=cell.proof["device"], cpu_threads=2,
                               representation=cell.proof["representation"])
        _require(run["binding"]["config"] == expected_config
                 and run["binding"]["complete_episode_bindings"] == frozen.records
                 and row["evaluation"]["configuration"] == frozen.forecast
                 and row["evaluation"]["evaluator_sha256"] ==
                     frozen.files["Libs/population_dynamics/pfc_population_evaluation.py"]["sha256"],
                 "FROZEN_TRAINING_OR_FORECAST_BINDING_CHANGED")
        for name, digest in run["trainer_sources"].items():
            relative = ("Libs/configs/" if name.endswith(".yaml") else "Libs/population_dynamics/") + name
            _require(relative in frozen.files and frozen.files[relative]["sha256"] == digest,
                     "FROZEN_TRAINER_SOURCE_CHANGED:" + name)
    cohort = PfcPopulationTrainingCohort(inputs, admitted[41]["run"]["binding"]["config"])
    models = {}
    for seed, row in admitted.items():
        model, same_cohort, run = _load(inputs, row["cell"].root / "training", torch.device("cpu"),
                                        prepared_inputs=cohort)
        _require(same_cohort is cohort and run == row["run"] and model.control_geometry == "actuation_only"
                 and len(model.unit_ids) == 100000 and len(model.field_names) == 17,
                 "COMPLETE_ACTUATION_MODEL_REQUIRED")
        models[seed] = model
    cases = {}
    for episode in config["factual_episode_ids"]:
        matches = [case for case in cohort.validation if case["episode_id"] == episode]
        _require(len(matches) == 1, "EXACT_FACTUAL_VALIDATION_CASE_REQUIRED")
        cases[episode] = matches[0]
    numerical_sources = ("pfc_intervention_experiment.py", "pfc_population_intervention.py",
                         "pfc_compiled_response.py", "population_intervention.py", "population_intervention_replay.py")
    binding = dict(configuration=config, config_sha256=sha256_file(config_path), inputs_sha256=input_hash,
                   preparation_sha256=proof_hash, reference_sha256=config["reference_sha256"],
                   seed_cells={str(seed): row["binding"] for seed, row in admitted.items()},
                   implementation_sources={name: sha256_file(Path(__file__).with_name(name)) for name in numerical_sources})
    return dict(config=config, target=target, admitted=admitted, cohort=cohort, models=models, cases=cases, binding=binding)


def compile_candidates(data, condition_indices):
    """Compile known programs without reading candidate outcomes."""
    factual = data.source_config["benchmark_episode_candidate"]["identity"]
    protocol = data.source_config.get("benchmark_task_protocol", task_protocol("pfc_working_memory"))
    candidates = []
    for condition in condition_indices:
        identity = deepcopy(factual)
        identity["condition_index"] = condition
        identity["episode_id"] = factual["episode_id"].rsplit("_condition_", 1)[0] + f"_condition_{condition:02d}"
        config = pfc_working_memory_config(data.source_config, protocol, identity)
        candidates.append(PfcInterventionCandidate(config["benchmark_episode_candidate"]["condition_id"], config))
    return tuple(candidates)


def _case_inputs(context, seed, episode):
    cohort, case = context["cohort"], context["cases"][episode]
    cohort.active_case, cohort.active_episode = None, None
    data = cohort._episode(case)
    cohort.active_case, cohort.active_episode = case, data
    view = cohort.model_for_case(context["models"][seed], case).eval()
    _require(len(data.unit_ids) == 100000 and len(data.field_names) == 17 and data.episode_id == episode,
             "FACTUAL_SOURCE_SCALE_CHANGED")
    candidates = compile_candidates(data, context["config"]["candidate_condition_indices"])
    binding = dict(experiment=context["binding"], training_seed=seed, factual_episode_id=episode,
                   checkpoint_sha256=context["admitted"][seed]["run"]["best_checkpoint_sha256"],
                   factual_source_binding=deepcopy(data.binding), model_source_metadata_identity=view.source_metadata_identity)
    return view, data, candidates, binding


def _reference_arrays(target):
    return dict(field_rate_hz=target.field_rate_hz, scale_hz=target.scale_hz,
                field_ids=np.asarray(target.field_names), block_ids=np.asarray(target.independent_block_ids))


def _candidate_rows(report):
    return [{key: row[key] for key in ("candidate", "reference_energy_score", "objective",
                                      "generated_unit_count_values", "generation_and_projection_seconds")}
            for row in report["candidates"]]


def _csv_payload(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_csv(path, rows):
    payload = _csv_payload(rows)
    if path.exists():
        _require(path.read_text() == payload, "EXISTING_CSV_CHANGED")
    else:
        with path.open("x", newline="") as stream:
            stream.write(payload)


def _save_npz(path, arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _scores(arrays, target):
    scores = [energy_score(arrays["field_rate_hz"] / target.scale_hz, reference / target.scale_hz)
              for reference in target.field_rate_hz]
    _require(np.isfinite(scores).all(), "FINITE_REFERENCE_SCORES_REQUIRED")
    return float(np.mean(scores)), scores


def _verify_search_case(path, view, data, target, candidates, config, binding):
    _require((path / "analysis.json").is_file(), "INCOMPLETE_CASE_PRESERVED_USE_NEW_OUTPUT")
    report = _read(path / "analysis.json")
    names = [candidate.name for candidate in candidates]
    _require(report.get("schema") == "pfc_population_intervention_case_v1"
             and report.get("status") == "RANKED_DEVELOPMENT_COMPILED_PFC_CANDIDATES"
             and report.get("experiment_binding") == binding
             and [row["candidate"] for row in report["candidates"]] == names
             and report["candidate_count"] == len(names) and report["joint_draws"] == config["joint_draws"]
             and report["sampling_stream_seeds"] == config["sampling_stream_seeds"]
             and report["origin_tick"] == config["origin_tick"] and report["horizon_bins"] == config["horizon_bins"]
             and report["source_unit_count"] == len(view.unit_ids)
             and report["source_binding"] == data.binding
             and report["model_source_metadata_identity"] == view.source_metadata_identity
             and report["scientific_claim_qualified"] is False and report["native_execution_performed"] is False
             and report["inverse_intervention_qualified"] is False and report["per_bin_unit_counts_retained"] is False
             and report["all_candidate_predictions_retained"] is True,
             "COMPLETE_BOUND_SEARCH_CASE_REQUIRED")
    verify_artifacts(path, report, names)
    compare_arrays(_arrays(path / "reference.npz"), _reference_arrays(target), "reference")
    reconstructed = []
    for candidate, row in zip(candidates, report["candidates"]):
        candidate_view, _, future = prepare_compiled_response(
            view, data, config["origin_tick"], config["horizon_bins"], candidate.source_config)
        _require(row["source_config"] == candidate.source_config
                 and row["candidate_identity"] == candidate.source_config["benchmark_episode_candidate"]["identity"]
                 and row["registered_response_binding"] == future["registered_response_binding"],
                 "SOURCE_COMPILED_CANDIDATE_CHANGED")
        arrays = _arrays(path / (candidate.name + ".npz"))
        _require(set(arrays) == set(PREDICTION_ARRAYS) | {"unit_ids", "field_ids"},
                 "COMPLETE_CANDIDATE_ARRAY_SCHEMA_REQUIRED")
        shapes = dict(field_counts=(config["joint_draws"], config["horizon_bins"], len(view.field_names)),
                      field_rate_hz=(config["joint_draws"], config["horizon_bins"], len(view.field_names)),
                      unit_total_counts=(config["joint_draws"], len(view.unit_ids)),
                      terminal_ready_in_ticks=(config["joint_draws"], len(view.unit_ids)))
        for name, shape in shapes.items():
            value = arrays[name]
            _require(value.shape == shape and np.isfinite(value).all() and np.all(value >= 0)
                     and (name == "field_rate_hz" or value.dtype == np.int64),
                     "COMPLETE_FINITE_SAVED_ARRAY_REQUIRED:" + name)
        compare_arrays({key: arrays[key] for key in ("unit_ids", "field_ids")},
                       dict(unit_ids=np.asarray(view.unit_ids), field_ids=np.asarray(view.field_names)), candidate.name)
        _require(np.array_equal(arrays["field_rate_hz"], arrays["field_counts"] /
                     (candidate_view.field_unit_counts.cpu().numpy() * candidate_view.bin_seconds)),
                 "SAVED_FIELD_RATE_UNITS_CHANGED")
        score, block_scores = _scores(arrays, target)
        _require(np.allclose(row["reference_block_scores"], block_scores, rtol=1e-12, atol=1e-12)
                 and np.isclose(row["reference_energy_score"], score, rtol=1e-12, atol=1e-12)
                 and np.isclose(row["objective"], score, rtol=1e-12, atol=1e-12)
                 and row["generated_unit_count_values"] == config["joint_draws"] * config["horizon_bins"] * len(view.unit_ids),
                 "SAVED_TARGET_SCORES_OR_GENERATION_DENOMINATOR_CHANGED")
        reconstructed.append(dict(candidate=candidate.name, objective=score))
    ranking = sorted(reconstructed, key=lambda row: (row["objective"], row["candidate"]))
    selected = next(row for row in report["candidates"] if row["candidate"] == ranking[0]["candidate"])
    _require(report["candidate_ranking"] == [row["candidate"] for row in ranking]
             and report["selected_candidate"] == selected["candidate"]
             and report["selected_action"]["source_config"] == selected["source_config"]
             and report["selected_action"]["registered_response_binding"] == selected["registered_response_binding"]
             and (path / "candidates.csv").read_text() == _csv_payload(_candidate_rows(report)),
             "SAVED_RANKING_ACTION_OR_CSV_CHANGED")
    return report


def execute_case(path, view, data, target, candidates, config, binding):
    """Create a complete case or reuse its authenticated committed artifacts."""
    path = Path(path)
    if path.exists():
        return _verify_search_case(path, view, data, target, candidates, config, binding)
    report, predictions = design_pfc_intervention(
        view, data, target, origin=config["origin_tick"], candidates=candidates,
        stream_seeds=tuple(config["sampling_stream_seeds"]), draws=config["joint_draws"])
    path.mkdir(parents=True, exist_ok=False)
    _save_npz(path / "reference.npz", _reference_arrays(target))
    for candidate in candidates:
        prediction = predictions[candidate.name]
        arrays = {name: prediction[name] for name in PREDICTION_ARRAYS}
        arrays.update(unit_ids=np.asarray(view.unit_ids), field_ids=np.asarray(view.field_names))
        _save_npz(path / (candidate.name + ".npz"), arrays)
    _write_csv(path / "candidates.csv", _candidate_rows(report))
    names = ["reference.npz", "candidates.csv", *[candidate.name + ".npz" for candidate in candidates]]
    report.update(schema="pfc_population_intervention_case_v1", experiment_binding=deepcopy(binding),
                  generation_job_id=os.environ.get("SLURM_JOB_ID"),
                  artifacts={name: sha256_file(path / name) for name in names})
    freeze_json(path / "analysis.json", report)
    return _verify_search_case(path, view, data, target, candidates, config, binding)


def _verify_replay_case(path, binding, source_sha, candidate_names, generated_values):
    _require((path / "analysis.json").is_file(), "INCOMPLETE_REPLAY_CASE_PRESERVED_USE_NEW_OUTPUT")
    report = _read(path / "analysis.json")
    _require(report.get("status") == "PASS_PFC_INTERVENTION_RETAINED_ARRAY_REPLAY"
             and report.get("experiment_binding") == binding and report.get("search_analysis_sha256") == source_sha
             and [row["candidate"] for row in report["candidates"]] == candidate_names
             and report["regenerated_unit_count_values"] == generated_values
             and report["per_bin_full_unit_counts_exact_replay_claimed"] is False
             and report["scientific_claim_qualified"] is False
             and report["native_execution_performed"] is False
             and report["retained_arrays_exact"] is True
             and set(report["artifacts"]) == {"candidates.csv"},
             "COMPLETE_BOUND_REPLAY_CASE_REQUIRED")
    _require(sha256_file(path / "candidates.csv") == report["artifacts"]["candidates.csv"]
             and (path / "candidates.csv").read_text() == _csv_payload(report["candidates"]),
             "REPLAY_CSV_CHANGED")
    return report


def replay_case(path, source, view, data, target, candidates, config, binding):
    source, path = Path(source), Path(path)
    report = _verify_search_case(source, view, data, target, candidates, config, binding)
    source_sha = sha256_file(source / "analysis.json")
    generated = len(candidates) * config["joint_draws"] * config["horizon_bins"] * len(view.unit_ids)
    names = [candidate.name for candidate in candidates]
    if path.exists():
        return _verify_replay_case(path, binding, source_sha, names, generated)
    rows, compared = [], 0
    for candidate, saved_row in zip(candidates, report["candidates"]):
        candidate_view, boundary, future = prepare_compiled_response(
            view, data, config["origin_tick"], config["horizon_bins"], candidate.source_config)
        expected = direct_prediction(candidate_view, boundary, future["known_controls"][None],
                                     tuple(config["sampling_stream_seeds"]), config["joint_draws"])
        compared += compare_arrays(_arrays(source / (candidate.name + ".npz")), expected, candidate.name)
        score, blocks = _scores(expected, target)
        _require(np.allclose(saved_row["reference_block_scores"], blocks, rtol=1e-12, atol=1e-12)
                 and np.isclose(saved_row["objective"], score, rtol=1e-12, atol=1e-12),
                 "INDEPENDENT_REFERENCE_SCORE_REPLAY_CHANGED")
        rows.append(dict(candidate=candidate.name, objective=score,
                         reference_block_scores=json.dumps(blocks), retained_arrays_exact=True))
    ranking = sorted(rows, key=lambda row: (row["objective"], row["candidate"]))
    _require([row["candidate"] for row in ranking] == report["candidate_ranking"], "INDEPENDENT_RANKING_CHANGED")
    path.mkdir(parents=True, exist_ok=False)
    _write_csv(path / "candidates.csv", rows)
    result = dict(status="PASS_PFC_INTERVENTION_RETAINED_ARRAY_REPLAY",
        experiment_binding=deepcopy(binding), search_analysis_sha256=source_sha, candidates=rows,
        selected_candidate=ranking[0]["candidate"], candidate_ranking=[row["candidate"] for row in ranking],
        reference_blocks_compared=len(target.independent_block_ids),
        saved_candidate_array_values_compared=compared, regenerated_unit_count_values=generated,
        retained_arrays_exact=True, per_bin_full_unit_counts_saved=False,
        per_bin_full_unit_counts_exact_replay_claimed=False,
        native_execution_performed=False, scientific_claim_qualified=False,
        replay_job_id=os.environ.get("SLURM_JOB_ID"),
        artifacts={"candidates.csv": sha256_file(path / "candidates.csv")})
    freeze_json(path / "analysis.json", result)
    return result


def _copy_case(source, destination, report):
    destination.mkdir(parents=True, exist_ok=False)
    for name in [*report["artifacts"], "analysis.json"]:
        _require(Path(name).name == name, "PLAIN_CASE_ARTIFACT_NAME_REQUIRED")
        shutil.copy2(source / name, destination / name)


def _summary(context, reports, output, replay):
    rows, cases = [], []
    for seed in SEEDS:
        for episode in context["config"]["factual_episode_ids"]:
            relative = Path(f"seed_{seed}") / episode
            report = reports[(seed, episode)]
            cases.append(dict(training_seed=seed, factual_episode_id=episode, path=str(relative),
                              analysis_sha256=sha256_file(output / relative / "analysis.json"),
                              selected_candidate=report["selected_candidate"]))
            rows.extend(dict(training_seed=seed, factual_episode_id=episode, **row)
                        for row in (report["candidates"] if replay else _candidate_rows(report)))
    generated = sum(report["regenerated_unit_count_values"] if replay else
                    sum(row["generated_unit_count_values"] for row in report["candidates"])
                    for report in reports.values())
    _require(len(cases) == 9 and len(rows) == 90 and generated == 141120000000,
             "COMPLETE_NINE_CASE_NINETY_CANDIDATE_DENOMINATOR_REQUIRED")
    _write_csv(output / "candidates.csv", rows)
    result = dict(status="PASS_PFC_INTERVENTION_EXPERIMENT_REPLAY" if replay else "COMPLETE_PFC_INTERVENTION_EXPERIMENT",
        experiment_binding=context["binding"], cases=cases, candidate_rows=len(rows), seed_block_cases=len(cases),
        generated_unit_count_values_across_completed_cases=generated,
        retained_candidate_arrays=["field_counts", "field_rate_hz", "unit_total_counts", "terminal_ready_in_ticks",
                                   "unit_ids", "field_ids"],
        per_bin_full_unit_counts_saved=False, per_bin_full_unit_counts_exact_replay_claimed=False,
        source_admission_may_read_full_outcomes_for_integrity=True, candidate_prediction_uses_factual_prefix_only=True,
        training_performed=False, native_execution_performed=False, success_tolerance_qualified=False,
        memory_recovery_qualified=False, scientific_claim_qualified=False,
        artifacts={"candidates.csv": sha256_file(output / "candidates.csv")})
    if replay:
        result["saved_candidate_array_values_compared"] = sum(
            report["saved_candidate_array_values_compared"] for report in reports.values())
    freeze_json(output / "analysis.json", result)
    return result


def run_experiment(inputs, preparation, config_path, reference_path, seed_cells, output, *,
                   replay_from=None, reuse_completed_from=None):
    """Run all nine cases or replay them; preserve incomplete source attempts."""
    inputs, preparation, config_path, reference_path, output = map(
        Path, (inputs, preparation, config_path, reference_path, output))
    replay_from = None if replay_from is None else Path(replay_from)
    reuse_completed_from = None if reuse_completed_from is None else Path(reuse_completed_from)
    _require(replay_from is None or replay_from.resolve() != output.resolve(), "SEPARATE_REPLAY_OUTPUT_REQUIRED")
    _require(reuse_completed_from is None or reuse_completed_from.resolve() != output.resolve(),
             "SEPARATE_REUSE_SOURCE_REQUIRED")
    torch.set_num_threads(1)
    context = _context(inputs, preparation, config_path, reference_path, seed_cells)
    config, target = context["config"], context["target"]
    reports, copies = {}, []
    if (output / "analysis.json").exists():
        previous = _read(output / "analysis.json")
        expected_status = "PASS_PFC_INTERVENTION_EXPERIMENT_REPLAY" if replay_from else "COMPLETE_PFC_INTERVENTION_EXPERIMENT"
        _require(previous.get("experiment_binding") == context["binding"] and previous.get("status") == expected_status,
                 "EXISTING_EXPERIMENT_BINDING_CHANGED")
    # Resolve all existing evidence before any candidate generation or replay.
    for seed in SEEDS:
        for episode in config["factual_episode_ids"]:
            relative = Path(f"seed_{seed}") / episode
            view, data, candidates, binding = _case_inputs(context, seed, episode)
            source_sha = None
            if replay_from is not None:
                _verify_search_case(replay_from / relative, view, data, target, candidates, config, binding)
                source_sha = sha256_file(replay_from / relative / "analysis.json")
            destination = output / relative
            existing = destination if destination.exists() else None
            if existing is None and reuse_completed_from is not None:
                possible = reuse_completed_from / relative
                if (possible / "analysis.json").is_file():
                    existing = possible  # An uncommitted source case is left untouched.
            if existing is not None:
                if replay_from is None:
                    report = _verify_search_case(existing, view, data, target, candidates, config, binding)
                else:
                    report = _verify_replay_case(existing, binding, source_sha, [c.name for c in candidates],
                        len(candidates) * config["joint_draws"] * config["horizon_bins"] * len(view.unit_ids))
                reports[(seed, episode)] = report
                if existing != destination:
                    copies.append((existing, destination, report))
    for source, destination, report in copies:
        _copy_case(source, destination, report)
    output.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for episode in config["factual_episode_ids"]:
            if (seed, episode) in reports:
                continue
            relative = Path(f"seed_{seed}") / episode
            view, data, candidates, binding = _case_inputs(context, seed, episode)
            if replay_from is None:
                report = execute_case(output / relative, view, data, target, candidates, config, binding)
            else:
                report = replay_case(output / relative, replay_from / relative, view, data, target, candidates, config, binding)
            reports[(seed, episode)] = report
    return _summary(context, reports, output, replay_from is not None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "preparation", "config", "reference", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--seed-cell", nargs=2, action="append", required=True, metavar=("SEED", "PATH"))
    parser.add_argument("--replay-from", type=Path)
    parser.add_argument("--reuse-completed-from", type=Path)
    args = parser.parse_args(argv)
    pairs = [(int(seed), Path(path)) for seed, path in args.seed_cell]
    _require(len(pairs) == 3 and len(dict(pairs)) == 3, "EXACT_THREE_EXPLICIT_SEED_CELLS_REQUIRED")
    report = run_experiment(args.inputs, args.preparation, args.config, args.reference, dict(pairs), args.output,
                            replay_from=args.replay_from, reuse_completed_from=args.reuse_completed_from)
    print(json.dumps(dict(status=report["status"], candidate_rows=report["candidate_rows"],
                          output=str(args.output / "analysis.json"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
