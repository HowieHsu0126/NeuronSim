"""Compare existing terminal PFC checkpoints with accepted ELBO-selected forecasts.

Only terminal-latest predictions are generated. All selected-checkpoint forecasts
and native outcomes are reused. Full-unit generation and independent replay use
the existing intervention case implementation; selection is descriptive.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from . import pfc_intervention_experiment as experiment
from .pfc_intervention_references import load_reference
from .pfc_population_forecast_analysis import FrozenInputs
from .pfc_population_training_data import PfcPopulationTrainingCohort
from .population_state import ControlledPopulationState
from .population_intervention_replay import verify_artifacts
from .provenance import canonical_json, freeze_json, sha256_file
from .trajectory_scores import energy_score

SEEDS = experiment.SEEDS
FACTUAL_EPISODES = experiment.FACTUAL_EPISODES
LATEST_CELL_FILES = (*experiment.SEED_CELL_FILES, "training/latest.pt", "training/epochs.json")
GENERATED_VALUES = 141120000000
CASE_VALUES = GENERATED_VALUES // 9
SELECTION = {
    "criterion": "lower_equal_condition_mean_source_forecast_energy_score",
    "cases_per_seed": 30,
    "tie_break": "retain_best",
    "scope": "development_checkpoint_comparison",
    "proper_energy_score_unchanged": True,
}
STATUSES = {
    "generate": "COMPLETE_TERMINAL_LATEST_PFC_FORECASTS",
    "replay": "PASS_TERMINAL_LATEST_PFC_FORECAST_REPLAY",
}


def require(condition, message):
    if not condition:
        raise ValueError("PFC_CHECKPOINT_COMPARISON_" + message)


def read(path):
    return experiment._read(path)


def commit(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, "COMMITTED_METADATA_CHANGED")
    else:
        freeze_json(path, value)


def commit_csv(path, rows):
    path = Path(path)
    if path.exists():
        require(path.read_text() == experiment._csv_payload(rows), "COMMITTED_CSV_CHANGED")
    else:
        experiment._write_csv(path, rows)


def table(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def bound_file(root, index, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts and name in index,
            "SAFE_DECLARED_ARTIFACT_REQUIRED:" + name)
    path, row = Path(root) / relative, index[name]
    require(path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"],
            "DELIVERED_ARTIFACT_CHANGED:" + name)
    return path


def admit_best_execution(execution, acceptance):
    """Read accepted best metadata and scores, without source/native array reads."""
    root, acceptance = Path(execution), Path(acceptance)
    receipt = read(acceptance)
    require(receipt.get("status") == "ACCEPTED_COMPLETE_PFC_INTERVENTION_DEVELOPMENT_DELIVERY"
            and receipt.get("candidate_rows") == 90 and receipt.get("native_source_episodes_reused") == 30
            and receipt.get("scientific_claim_qualified") is False
            and sha256_file(root / "delivery.json") == receipt["delivery_sha256"],
            "ACCEPTED_COMPLETE_BEST_EXECUTION_REQUIRED")
    delivery = read(root / "delivery.json")
    require(delivery.get("status") == "PASS" and delivery.get("exit_code") == 0,
            "SUCCESSFUL_BEST_DELIVERY_REQUIRED")
    index = {row["path"]: row for row in delivery["files"]}
    require(len(index) == len(delivery["files"]), "UNIQUE_DELIVERY_PATHS_REQUIRED")
    def metadata(name):
        return read(bound_file(root, index, name))
    search = metadata("results/analysis.json")
    native = metadata("source_outcomes/analysis.json")
    require(search.get("status") == "COMPLETE_PFC_INTERVENTION_EXPERIMENT"
            and search.get("candidate_rows") == 90 and search.get("seed_block_cases") == 9
            and native.get("status") == "COMPLETE_PFC_INTERVENTION_SOURCE_OUTCOMES"
            and native.get("binding", {}).get("experiment") == search["experiment_binding"],
            "BEST_SEARCH_NATIVE_BINDING_CHANGED")
    config = experiment.validate_config(search["experiment_binding"]["configuration"])
    reference = bound_file(root, index, "reference.json")
    require(sha256_file(reference) == config["reference_sha256"], "FITTING_REFERENCE_CHANGED")
    target, reference_report = load_reference(reference)
    cases = {}
    for entry in search["cases"]:
        key = entry["training_seed"], entry["factual_episode_id"]
        relative = str(Path("results") / entry["path"] / "analysis.json")
        report = metadata(relative)
        require(key not in cases and sha256_file(root / relative) == entry["analysis_sha256"]
                and report["experiment_binding"]["experiment"] == search["experiment_binding"],
                "BEST_CASE_BINDING_CHANGED")
        cases[key] = dict(path=root / "results" / entry["path"], report=report)
    require(set(cases) == {(s, e) for s in SEEDS for e in FACTUAL_EPISODES},
            "COMPLETE_BEST_CASE_GRID_REQUIRED")
    score_path = bound_file(root, index, "source_outcomes/candidates.csv")
    require(sha256_file(score_path) == native["artifacts"]["candidates.csv"],
            "BEST_SOURCE_SCORE_BINDING_CHANGED")
    scores = {}
    for row in table(score_path):
        key = int(row["training_seed"]), row["factual_episode_id"], row["candidate"]
        require(key not in scores, "DUPLICATE_BEST_SCORE")
        scores[key] = row
    expected = {(s, e, c["candidate"]) for (s, e), case in cases.items()
                for c in case["report"]["candidates"]}
    require(len(scores) == 90 and set(scores) == expected, "COMPLETE_BEST_SCORE_GRID_REQUIRED")
    sources = metadata("source_outcomes/sources.json")["episodes"]
    source_index = {(r["factual_episode_id"], r["candidate"]): r for r in sources}
    require(len(sources) == len(source_index) == 30, "THIRTY_DISTINCT_NATIVE_PATHS_REQUIRED")
    binding = dict(acceptance_sha256=sha256_file(acceptance),
        delivery_sha256=receipt["delivery_sha256"],
        search_sha256=sha256_file(root / "results/analysis.json"),
        native_analysis_sha256=sha256_file(root / "source_outcomes/analysis.json"),
        accepted_execution=receipt["execution"])
    return dict(root=root, index=index, search=search, native=native, cases=cases,
                scores=scores, sources=source_index, config=config, target=target,
                reference_report=reference_report, binding=binding)


def validate_latest_checkpoint(checkpoint, run, epochs):
    """Check the terminal payload against its own unchanged training metadata."""
    completed, best = run["completed_epochs"], run["best_epoch"]
    require(run.get("status") == "FITTED_DEVELOPMENT" and bool(run.get("finished_utc"))
            and isinstance(epochs, list) and len(epochs) == completed
            and [row.get("epoch") for row in epochs] == list(range(completed)),
            "COMPLETE_TERMINAL_HISTORY_REQUIRED")
    for name in ("binding", "model_configuration", "trainer_sources", "effective_maximum_epochs",
                 "continuation", "attempt"):
        require(checkpoint.get(name) == run.get(name), "LATEST_RUN_BINDING_CHANGED:" + name)
    require(checkpoint.get("next_epoch") == completed and checkpoint.get("history") == epochs
            and checkpoint.get("best_epoch") == best
            and checkpoint.get("stale") == run["stale_epochs"] == completed - best - 1,
            "LATEST_TERMINAL_POSITION_CHANGED")
    losses = [row["validation"]["loss"] for row in epochs]
    require(all(type(value) in (int, float) and math.isfinite(value) for value in losses)
            and losses[best] == min(losses)
            and checkpoint.get("best") == run["best_validation_negative_elbo"] == losses[best],
            "LATEST_VALIDATION_HISTORY_CHANGED")


def admit_latest_seed_cells(seed_cells):
    """Reuse terminal best admission, then authenticate each actual latest payload."""
    admitted = experiment.admit_seed_cells(seed_cells)
    for seed, row in admitted.items():
        cell, run = row["cell"], row["run"]
        path = cell.path("training/latest.pt")
        require(cell.files["training/latest.pt"]["sha256"] == run.get("latest_checkpoint_sha256"),
                "LATEST_CHECKPOINT_DIGEST_CHANGED")
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        epochs = cell.read("training/epochs.json")
        validate_latest_checkpoint(checkpoint, run, epochs)
        row.update(checkpoint=checkpoint, selected_checkpoint_kind="latest",
                   selected_checkpoint_sha256=run["latest_checkpoint_sha256"],
                   selected_checkpoint_epoch=checkpoint["next_epoch"] - 1,
                   binding=deepcopy(cell.binding()))
    return admitted


def load_latest_model(admitted, cohort):
    """Load latest directly; never edit or alias the training report's best fields."""
    checkpoint, run = admitted["checkpoint"], admitted["run"]
    for name, value in cohort.binding.items():
        if name != "circuit_inputs_sha256":
            require(checkpoint["binding"].get(name) == value, "LATEST_SOURCE_COHORT_CHANGED:" + name)
    for name, digest in checkpoint["trainer_sources"].items():
        require(Path(name).name == name, "PLAIN_TRAINER_SOURCE_NAME_REQUIRED")
        path = (Path(__file__).parents[1] / "configs" / name if name.endswith(".yaml")
                else Path(__file__).with_name(name))
        require(sha256_file(path) == digest, "LATEST_FITTED_SOURCE_CHANGED:" + name)
    constructor = cohort.population_configuration(run["binding"]["config"], run["training_seed"])
    require(constructor == checkpoint["model_configuration"], "LATEST_SOURCE_CONSTRUCTOR_CHANGED")
    model = ControlledPopulationState(**constructor)
    for name, expected in model.named_buffers():
        require(torch.equal(checkpoint["model"][name], expected), "LATEST_SOURCE_BUFFER_CHANGED:" + name)
    model.to(dtype=checkpoint["model"]["posterior_mean.weight"].dtype)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.cpu().eval()


def latest_context(inputs, preparation, config_path, reference_path, seed_cells, best):
    inputs, preparation, config_path, reference_path = map(
        Path, (inputs, preparation, config_path, reference_path))
    config = experiment.validate_config(yaml.safe_load(config_path.read_text()))
    require(config == best["config"] and sha256_file(reference_path) == config["reference_sha256"],
            "BEST_LATEST_FORECAST_PROTOCOL_CHANGED")
    target, reference_report = load_reference(reference_path)
    admitted = admit_latest_seed_cells(seed_cells)
    frozen = FrozenInputs(preparation)
    input_sha, prep_sha = sha256_file(inputs), sha256_file(preparation / "preparation.json")
    original = best["search"]["experiment_binding"]
    require(input_sha == frozen.proof["input_sha256"] == reference_report["inputs_sha256"] == original["inputs_sha256"]
            and prep_sha == original["preparation_sha256"]
            and sha256_file(config_path) == original["config_sha256"],
            "LATEST_FROZEN_INPUT_BINDING_CHANGED")
    for seed, row in admitted.items():
        run, cell = row["run"], row["cell"]
        require(cell.proof["preparation_sha256"] == prep_sha
                and run["binding"]["config"] == dict(frozen.config, device=cell.proof["device"],
                                                     cpu_threads=2, representation=cell.proof["representation"])
                and run["binding"]["complete_episode_bindings"] == frozen.records
                and row["evaluation"]["configuration"] == frozen.forecast
                and row["evaluation"]["evaluator_sha256"] == frozen.files[
                    "Libs/population_dynamics/pfc_population_evaluation.py"]["sha256"]
                and run["best_checkpoint_sha256"] == original["seed_cells"][str(seed)][
                    "consumed_files"]["training/best.pt"], "BEST_LATEST_TRAINING_BINDING_CHANGED")
        for name, digest in run["trainer_sources"].items():
            relative = ("Libs/configs/" if name.endswith(".yaml") else "Libs/population_dynamics/") + name
            require(frozen.files[relative]["sha256"] == digest, "FROZEN_TRAINER_CHANGED:" + name)
    cohort = PfcPopulationTrainingCohort(inputs, admitted[41]["run"]["binding"]["config"])
    models = {seed: load_latest_model(row, cohort) for seed, row in admitted.items()}
    cases = {case["episode_id"]: case for case in cohort.validation if case["episode_id"] in FACTUAL_EPISODES}
    require(set(cases) == set(FACTUAL_EPISODES), "COMPLETE_FACTUAL_SOURCE_CASES_REQUIRED")
    checkpoints = {str(seed): dict(latest_sha256=row["selected_checkpoint_sha256"],
        latest_epoch=row["selected_checkpoint_epoch"], best_sha256=row["run"]["best_checkpoint_sha256"],
        best_epoch=row["run"]["best_epoch"], delivery=row["binding"]) for seed, row in admitted.items()}
    binding = dict(schema="pfc_terminal_checkpoint_prior_comparison_v1", configuration=config,
        selected_checkpoint_kind="latest", latest_checkpoints=checkpoints,
        best_execution=best["binding"], selection=deepcopy(SELECTION),
        implementation_sha256=sha256_file(Path(__file__)))
    return dict(config=config, target=target, admitted=admitted, cohort=cohort,
                models=models, cases=cases, binding=binding)


def latest_case_inputs(context, seed, episode):
    view, data, candidates, binding = experiment._case_inputs(context, seed, episode)
    row = context["admitted"][seed]
    binding.update(checkpoint_sha256=row["selected_checkpoint_sha256"],
        selected_checkpoint_kind="latest", selected_checkpoint_epoch=row["selected_checkpoint_epoch"],
        training_best_checkpoint_sha256=row["run"]["best_checkpoint_sha256"])
    return view, data, candidates, binding


def run_latest_phase(inputs, preparation, config_path, reference_path, seed_cells,
                     best_execution, best_acceptance, output, *, replay_from=None):
    best = admit_best_execution(best_execution, best_acceptance)
    context = latest_context(inputs, preparation, config_path, reference_path, seed_cells, best)
    output = Path(output)
    phase = "replay" if replay_from is not None else "generate"
    output.mkdir(parents=True, exist_ok=True)
    cases, rows, compared = [], [], 0
    for seed in SEEDS:
        for episode in FACTUAL_EPISODES:
            view, data, candidates, binding = latest_case_inputs(context, seed, episode)
            relative = Path(f"seed_{seed}") / episode
            if replay_from is None:
                report = experiment.execute_case(output / relative, view, data, context["target"],
                                                 candidates, context["config"], binding)
                candidate_rows = report["candidates"]
            else:
                report = experiment.replay_case(output / relative, Path(replay_from) / relative,
                    view, data, context["target"], candidates, context["config"], binding)
                candidate_rows = report["candidates"]
                compared += report["saved_candidate_array_values_compared"]
            cases.append(dict(training_seed=seed, factual_episode_id=episode, path=str(relative),
                analysis_sha256=sha256_file(output / relative / "analysis.json"),
                selected_checkpoint_kind="latest", checkpoint_sha256=binding["checkpoint_sha256"]))
            rows.extend(dict(training_seed=seed, factual_episode_id=episode,
                candidate=row["candidate"], reference_energy_score=row["objective"]) for row in candidate_rows)
    require(len(rows) == 90 and len(cases) == 9, "COMPLETE_LATEST_PHASE_REQUIRED")
    commit_csv(output / "candidates.csv", rows)
    summary = dict(status=STATUSES[phase], binding=context["binding"], cases=cases,
        candidate_rows=90, seed_block_cases=9, generated_unit_count_values=GENERATED_VALUES,
        selected_checkpoint_kind="latest", training_performed=False, native_execution_performed=False,
        best_predictions_regenerated=False, per_bin_full_unit_counts_exact_replay_claimed=False,
        scientific_claim_qualified=False, artifacts={"candidates.csv": sha256_file(output / "candidates.csv")})
    if replay_from is not None:
        summary["saved_candidate_array_values_compared"] = compared
    commit(output / "analysis.json", summary)
    return summary


def select_checkpoints(rows):
    """Equal weight for all 30 native conditions; exact score ties retain best."""
    expected = {(seed, episode, condition) for seed in SEEDS for episode in FACTUAL_EPISODES
                for condition in range(10)}
    keys = [(row["training_seed"], row["factual_episode_id"], row["condition_index"]) for row in rows]
    require(len(keys) == len(set(keys)) == 90 and set(keys) == expected, "COMPLETE_PAIRED_SCORE_GRID_REQUIRED")
    result = []
    for seed in SEEDS:
        selected = [row for row in rows if row["training_seed"] == seed]
        values = {kind: [row[kind + "_source_forecast_energy_score"] for row in selected]
                  for kind in ("best", "latest")}
        require(all(math.isfinite(v) for values_for_kind in values.values() for v in values_for_kind),
                "FINITE_PROPER_ENERGY_SCORES_REQUIRED")
        means = {kind: math.fsum(scores) / 30 for kind, scores in values.items()}
        result.append(dict(training_seed=seed, paired_conditions=30,
            best_mean_source_forecast_energy_score=means["best"],
            latest_mean_source_forecast_energy_score=means["latest"],
            latest_minus_best=means["latest"] - means["best"],
            selected_checkpoint_kind="latest" if means["latest"] < means["best"] else "best"))
    return result


def compare_latest(best_execution, best_acceptance, latest, replay, output):
    best = admit_best_execution(best_execution, best_acceptance)
    latest, replay, output = map(Path, (latest, replay, output))
    summaries = [read(root / "analysis.json") for root in (latest, replay)]
    for summary, phase in zip(summaries, ("generate", "replay")):
        require(summary.get("status") == STATUSES[phase] and summary.get("candidate_rows") == 90
                and summary.get("seed_block_cases") == 9
                and summary.get("generated_unit_count_values") == GENERATED_VALUES
                and summary.get("selected_checkpoint_kind") == "latest"
                and summary.get("per_bin_full_unit_counts_exact_replay_claimed") is False,
                "COMPLETE_LATEST_GENERATION_AND_REPLAY_REQUIRED")
    binding = summaries[0]["binding"]
    require(summaries[1]["binding"] == binding and binding["best_execution"] == best["binding"]
            and binding["configuration"] == best["config"] and binding["selection"] == SELECTION,
            "LATEST_COMPARISON_BINDING_CHANGED")
    expected = {(s, e) for s in SEEDS for e in FACTUAL_EPISODES}
    indexes = [{(x["training_seed"], x["factual_episode_id"]): x for x in summary["cases"]}
               for summary in summaries]
    require(all(len(summary["cases"]) == len(index) == 9 and set(index) == expected
                for summary, index in zip(summaries, indexes)), "LATEST_CASE_DENOMINATOR_CHANGED")
    rows, native_cache = [], {}
    for seed, episode in sorted(expected):
        entry, repeat = (index[seed, episode] for index in indexes)
        relative = str(Path(f"seed_{seed}") / episode)
        require(entry["path"] == repeat["path"] == relative, "LATEST_CASE_PATH_CHANGED")
        path = latest / relative
        require(sha256_file(path / "analysis.json") == entry["analysis_sha256"]
                and sha256_file(replay / relative / "analysis.json") == repeat["analysis_sha256"],
                "LATEST_CASE_COMMIT_CHANGED")
        report = read(path / "analysis.json")
        case_binding = report["experiment_binding"]
        require(case_binding["experiment"] == binding and case_binding["training_seed"] == seed
                and case_binding["factual_episode_id"] == episode
                and case_binding.get("selected_checkpoint_kind") == "latest"
                and case_binding["checkpoint_sha256"] == binding["latest_checkpoints"][str(seed)]["latest_sha256"],
                "LATEST_CASE_CHECKPOINT_CHANGED")
        original = best["cases"][seed, episode]["report"]
        candidates = report["candidates"]
        require(len(candidates) == 10 and [r["candidate_identity"]["condition_index"] for r in candidates] == list(range(10)),
                "LATEST_ALL_CONDITIONS_REQUIRED")
        names = [row["candidate"] for row in candidates]
        verify_artifacts(path, report, names)
        experiment._verify_replay_case(replay / relative, case_binding, entry["analysis_sha256"], names, CASE_VALUES)
        for condition, candidate in enumerate(candidates):
            name = candidate["candidate"]
            old = original["candidates"][condition]
            require(name == old["candidate"] and candidate["source_config"] == old["source_config"]
                    and candidate["registered_response_binding"] == old["registered_response_binding"],
                    "BEST_LATEST_CANDIDATE_PROGRAM_CHANGED")
            source = best["sources"][episode, name]
            if (episode, name) not in native_cache:
                filename = "source_outcomes/" + source["source_arrays"]
                source_path = bound_file(best["root"], best["index"], filename)
                with np.load(source_path, allow_pickle=False) as saved:
                    native_cache[episode, name] = {k: saved[k] for k in ("field_rate_hz", "field_ids", "unit_ids")}
            native = native_cache[episode, name]
            with np.load(path / (name + ".npz"), allow_pickle=False) as saved:
                rates = saved["field_rate_hz"]
                require(rates.shape == (16, 980, 17) and native["field_rate_hz"].shape == (980, 17)
                        and np.isfinite(rates).all() and np.isfinite(native["field_rate_hz"]).all()
                        and np.array_equal(saved["field_ids"], native["field_ids"])
                        and tuple(saved["field_ids"]) == tuple(best["target"].field_names)
                        and np.array_equal(saved["unit_ids"], native["unit_ids"]),
                        "LATEST_NATIVE_FIELD_MAP_CHANGED")
                score = float(energy_score(rates / best["target"].scale_hz,
                                          native["field_rate_hz"] / best["target"].scale_hz))
            old_score = best["scores"][seed, episode, name]
            require(int(old_score["condition_index"]) == condition
                    and old_score["source_episode_id"] == source["source_episode_id"],
                    "BEST_NATIVE_SCORE_IDENTITY_CHANGED")
            rows.append(dict(training_seed=seed, factual_episode_id=episode, condition_index=condition,
                candidate=name, source_episode_id=source["source_episode_id"],
                best_source_forecast_energy_score=float(old_score["source_forecast_energy_score"]),
                latest_source_forecast_energy_score=score,
                latest_minus_best=score - float(old_score["source_forecast_energy_score"]),
                best_reference_energy_score=float(old_score["predicted_reference_energy_score"]),
                latest_reference_energy_score=candidate["objective"],
                best_reference_rank=original["candidate_ranking"].index(name) + 1,
                latest_reference_rank=report["candidate_ranking"].index(name) + 1,
                original_best_selected_action=original["selected_candidate"] == name,
                latest_reference_selected_action=report["selected_candidate"] == name,
                best_checkpoint_sha256=binding["latest_checkpoints"][str(seed)]["best_sha256"],
                latest_checkpoint_sha256=binding["latest_checkpoints"][str(seed)]["latest_sha256"]))
    selection = select_checkpoints(rows)
    require(len(native_cache) == 30, "THIRTY_NATIVE_EPISODES_REQUIRED")
    result = dict(status="COMPLETE_DESCRIPTIVE_PFC_CHECKPOINT_PRIOR_COMPARISON", binding=binding,
        candidate_rows=90, native_source_episodes_reused=30, native_source_blocks=3, native_circuits=1,
        comparison_selection=selection, criterion=SELECTION, best_predictions_regenerated=False,
        latest_generated_unit_count_values=GENERATED_VALUES, latest_independently_regenerated_unit_count_values=GENERATED_VALUES,
        native_execution_performed=False, training_performed=False, independent_intervention_qualified=False,
        native_future_noise_pairing_verified=False, scientific_claim_qualified=False,
        inferential_statistics_qualified=False, per_bin_full_unit_counts_exact_replay_claimed=False,
        source_episodes_previously_used_for_checkpoint_selection=True)
    output.mkdir(parents=True, exist_ok=True)
    commit_csv(output / "paired_candidates.csv", rows)
    commit_csv(output / "checkpoint_selection.csv", selection)
    result["artifacts"] = {name: sha256_file(output / name)
                          for name in ("paired_candidates.csv", "checkpoint_selection.csv")}
    commit(output / "analysis.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("generate", "replay", "compare"), required=True)
    for name in ("inputs", "preparation", "config", "reference"):
        parser.add_argument("--" + name, type=Path)
    for name in ("best-execution", "best-acceptance", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed-cell", nargs=2, action="append", default=[])
    parser.add_argument("--latest", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "compare":
        if args.latest is None or args.replay is None:
            parser.error("compare requires --latest and --replay")
        result = compare_latest(args.best_execution, args.best_acceptance, args.latest, args.replay, args.output)
    else:
        if any(getattr(args, name) is None for name in ("inputs", "preparation", "config", "reference")):
            parser.error("generation/replay require --inputs --preparation --config --reference")
        pairs = [(int(seed), Path(path)) for seed, path in args.seed_cell]
        if len(pairs) != 3 or set(dict(pairs)) != set(SEEDS):
            parser.error("exactly three explicit seed cells are required")
        if args.phase == "replay" and args.latest is None:
            parser.error("replay requires --latest")
        result = run_latest_phase(args.inputs, args.preparation, args.config, args.reference,
            dict(pairs), args.best_execution, args.best_acceptance, args.output,
            replay_from=args.latest if args.phase == "replay" else None)
    print(canonical_json({key: result[key] for key in ("status", "candidate_rows")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
