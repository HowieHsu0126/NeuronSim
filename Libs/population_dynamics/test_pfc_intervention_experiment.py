"""Bound PFC search artifacts, terminal admission and independent replay."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from . import pfc_intervention_experiment as experiment
from .provenance import canonical_json, sha256_file
from .test_pfc_population_intervention import problem
from .test_pfc_compiled_response import compiled


@pytest.fixture
def case(problem, tmp_path):
    view, data, target, candidates = problem
    config = dict(origin_tick=8000, horizon_bins=2, joint_draws=3, sampling_stream_seeds=[41, 42, 43])
    binding = dict(reference_sha256="a" * 64, checkpoint_sha256="b" * 64, source=data.binding)
    return dict(path=tmp_path / "search", view=view, data=data, target=target, candidates=candidates,
                config=config, binding=binding)


def execute(case):
    return experiment.execute_case(**case)


def replay(case, output):
    return experiment.replay_case(output, case["path"], case["view"], case["data"], case["target"],
                                  case["candidates"], case["config"], case["binding"])


def test_real_source_case_saves_all_candidates_and_replays_metadata_model_exactly(case, tmp_path):
    before = torch.get_rng_state().clone()
    report = execute(case)
    assert report["candidate_count"] == len(case["candidates"])
    assert not hasattr(case["view"], "unit_loading")
    with patch.object(experiment, "design_pfc_intervention", side_effect=AssertionError("REPEATED_SEARCH")):
        again = execute(case)
    assert again == report
    with patch.object(experiment, "design_pfc_intervention", side_effect=AssertionError("SEARCH_USED_FOR_REPLAY")):
        replayed = replay(case, tmp_path / "replay")
    assert replayed["retained_arrays_exact"]
    assert replayed["selected_candidate"] == report["selected_candidate"]
    assert replayed["regenerated_unit_count_values"] == len(case["candidates"]) * 3 * 2 * len(case["view"].unit_ids)
    assert replayed["reference_blocks_compared"] == len(case["target"].independent_block_ids)
    assert not replayed["per_bin_full_unit_counts_exact_replay_claimed"]
    assert not replayed["scientific_claim_qualified"]
    with patch.object(experiment, "direct_prediction", side_effect=AssertionError("REPEATED_REPLAY")):
        assert replay(case, tmp_path / "replay") == replayed
    assert torch.equal(before, torch.get_rng_state())


def test_all_ten_compiled_programs_keep_original_block_and_do_not_read_outcomes(case):
    data = case["data"]
    original = deepcopy(data.source_config)
    with patch.object(data, "window", side_effect=AssertionError("COMPILE_READ_OUTCOME")):
        candidates = experiment.compile_candidates(data, list(range(10)))
    assert len(candidates) == len({candidate.name for candidate in candidates}) == 10
    factual = data.source_config["benchmark_episode_candidate"]["identity"]
    for index, candidate in enumerate(candidates):
        identity = candidate.source_config["benchmark_episode_candidate"]["identity"]
        assert identity["condition_index"] == index
        for key in factual.keys() - {"episode_id", "condition_index"}:
            assert identity[key] == factual[key]
    assert data.source_config == original


def test_incomplete_case_and_changed_binding_are_preserved(case):
    case["path"].mkdir()
    marker = case["path"] / "partial.txt"
    marker.write_text("preserve")
    with patch.object(experiment, "design_pfc_intervention", side_effect=AssertionError("OVERWRITE_PARTIAL")):
        with pytest.raises(ValueError, match="INCOMPLETE_CASE_PRESERVED"):
            execute(case)
    assert marker.read_text() == "preserve"
    # Use a fresh output; the incomplete original remains unchanged.
    case = dict(case, path=case["path"].parent / "complete")
    execute(case)
    changed = dict(case, binding=dict(case["binding"], reference_sha256="c" * 64))
    with patch.object(experiment, "design_pfc_intervention", side_effect=AssertionError("CHANGED_BINDING_GENERATION")):
        with pytest.raises(ValueError, match="COMPLETE_BOUND_SEARCH_CASE"):
            execute(changed)
    assert marker.read_text() == "preserve"


@pytest.mark.parametrize("change", ["npz", "ranking", "score", "candidate"])
def test_saved_arrays_scores_rankings_and_candidate_denominator_cannot_change(case, change):
    report = execute(case)
    if change == "npz":
        path = case["path"] / (case["candidates"][0].name + ".npz")
        with path.open("ab") as stream:
            stream.write(b"changed")
    else:
        if change == "ranking":
            report["candidate_ranking"].reverse()
        elif change == "score":
            report["candidates"][0]["reference_block_scores"][0] += 1
        else:
            report["candidates"].pop()
        (case["path"] / "analysis.json").write_text(canonical_json(report))
    with pytest.raises(ValueError):
        execute(case)


def test_independent_replay_detects_a_simulator_array_difference(case, tmp_path):
    execute(case)
    original = experiment.direct_prediction
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        result["unit_total_counts"][0, 0] += 1
        return result
    with patch.object(experiment, "direct_prediction", changed):
        with pytest.raises(ValueError, match="saved array replay differs"):
            replay(case, tmp_path / "different_replay")
    assert not (tmp_path / "different_replay/analysis.json").exists()


def test_complete_case_copy_preserves_all_committed_bytes(case, tmp_path):
    report = execute(case)
    destination = tmp_path / "copy"
    experiment._copy_case(case["path"], destination, report)
    for name in [*report["artifacts"], "analysis.json"]:
        assert sha256_file(destination / name) == sha256_file(case["path"] / name)
    copied = dict(case, path=destination)
    with patch.object(experiment, "design_pfc_intervention", side_effect=AssertionError("COPIED_CASE_REGENERATED")):
        assert execute(copied) == report


@pytest.fixture
def delivered(tmp_path):
    roots = {}
    records = [dict(partition="development_fit", circuit_id="fit", episode_id=f"fit_{index}") for index in range(80)]
    records += [dict(partition="development_validation", circuit_id="native_validation",
                     episode_id=f"pfc_working_memory_circuit_074_block_{block:03d}_condition_{condition:02d}")
                for block in range(3) for condition in range(10)]
    validation = records[80:]
    for seed in experiment.SEEDS:
        root = tmp_path / f"seed_{seed}"
        root.mkdir()
        config = dict(training_seeds=[41, 42, 43], control_geometry="actuation_only", maximum_epochs=20,
                      early_stopping_patience=4)
        (root / "training").mkdir()
        (root / "training/best.pt").write_bytes(f"checkpoint_{seed}".encode())
        checkpoint = sha256_file(root / "training/best.pt")
        run = dict(status="FITTED_DEVELOPMENT", training_seed=seed, input_family="pfc_source_reproduction",
            units=100000, fitting_episodes=80, validation_episodes=30, completed_epochs=16, best_epoch=11,
            effective_maximum_epochs=20, stop_reason="validation_patience", best_checkpoint_sha256=checkpoint,
            model_configuration=dict(control_geometry="actuation_only"), scientific_claim_qualified=False,
            calibration_outcomes_accessed=False, test_outcomes_accessed=False,
            binding=dict(seed=seed, config=config, circuit_inputs_sha256="a" * 64, complete_episode_bindings=records))
        origins = [dict(case_id=row["circuit_id"] + "__" + row["episode_id"], origin_tick=origin,
                        units=100000, fields=17,
                        source_binding={key: value for key, value in row.items() if key != "partition"})
                   for row in validation for origin in experiment.ORIGINS]
        evaluation = dict(status="EVALUATED_PFC_DEVELOPMENT_CHECKPOINT", training_seed=seed,
            checkpoint_sha256=checkpoint, inputs_sha256="a" * 64, original_training_inputs_sha256="a" * 64,
            episodes=30, score_rows=360, scientific_claim_qualified=False, formal_evaluation=False, origins=origins)
        replayed = dict(status="PASS_ALL_UNIT_AND_INDEPENDENT_METRIC_REPLAY", checkpoint_sha256=checkpoint,
            scientific_claim_qualified=False, unit_values_consumed=16 * 100 * 100000 * 120,
            origins=[dict(case_id=row["case_id"], origin_tick=row["origin_tick"], unit_values=16 * 100 * 100000)
                     for row in origins])
        contents = {"training/run.json": run, "forecasts/evaluation.json": evaluation, "replay.json": replayed}
        for phase in ("fit", "evaluate", "replay"):
            contents[f"phases/{phase}/receipt.json"] = dict(phase=phase, status="PASS", command_started=True,
                                                         exit_code=0, finished_utc="2026-10-15T00:00:00+00:00")
        for name, content in contents.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(canonical_json(content))
        delivery = dict(status="PASS", exit_code=0, training_seed=seed, input_family="pfc_source_reproduction",
            scientific_claim_qualified=False, calibration_or_test_accessed=False, formal_evaluation=False,
            device="cpu", fit_episodes=80, validation_episodes=30, core_neurons=100000, preparation_sha256="b" * 64,
            files=[dict(path=name, bytes=(root / name).stat().st_size, sha256=sha256_file(root / name))
                   for name in experiment.SEED_CELL_FILES])
        (root / "delivery.json").write_text(canonical_json(delivery))
        roots[seed] = root
    return roots


def test_terminal_admission_consumes_only_explicit_metadata_and_checkpoint(delivered):
    admitted = experiment.admit_seed_cells(delivered)
    assert set(admitted) == {41, 42, 43}
    for value in admitted.values():
        assert set(value["binding"]["consumed_files"]) == set(experiment.SEED_CELL_FILES)
    with pytest.raises(ValueError, match="EXACT_THREE"):
        experiment.admit_seed_cells({41: delivered[41], 42: delivered[42]})


@pytest.mark.parametrize("change", ["delivery", "phase", "checkpointed", "replay"])
def test_unfinished_or_incomplete_last_seed_rejects_before_models_or_generation(delivered, change):
    root = delivered[43]
    delivery = json.loads((root / "delivery.json").read_text())
    if change == "delivery":
        delivery["status"] = "RUNNING"
    else:
        name = {"phase": "phases/replay/receipt.json", "checkpointed": "training/run.json", "replay": "replay.json"}[change]
        record = json.loads((root / name).read_text())
        if change == "phase":
            record["command_started"] = False
        elif change == "checkpointed":
            record["status"] = "CHECKPOINTED_DEVELOPMENT"
        else:
            record["origins"].pop()
        (root / name).write_text(canonical_json(record))
        for row in delivery["files"]:
            if row["path"] == name:
                row.update(bytes=(root / name).stat().st_size, sha256=sha256_file(root / name))
    (root / "delivery.json").write_text(canonical_json(delivery))
    with patch.object(experiment, "_load", side_effect=AssertionError("PREMATURE_MODEL_LOAD")), \
            patch.object(experiment, "design_pfc_intervention", side_effect=AssertionError("PREMATURE_GENERATION")):
        with pytest.raises(ValueError):
            experiment.admit_seed_cells(delivered)


@pytest.mark.parametrize("key,value", [("joint_draws", 4), ("horizon_bins", 100),
                                      ("training_seeds", [41, 42]), ("candidate_condition_indices", [0, 1]),
                                      ("scientific_claim_qualified", True)])
def test_changed_registered_experiment_scope_rejects(key, value):
    path = Path(__file__).resolve().parents[1] / "configs/neural_population_pfc_intervention_development.yaml"
    config = yaml.safe_load(path.read_text())
    config[key] = value
    with pytest.raises(ValueError, match="REGISTERED_CONFIGURATION"):
        experiment.validate_config(config)
