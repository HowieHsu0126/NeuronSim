"""Terminal-checkpoint identity, complete-grid selection and phase dispatch."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from . import pfc_checkpoint_comparison as comparison
from .population_state import ControlledPopulationState
from .provenance import sha256_file


@pytest.fixture
def terminal():
    epochs = [dict(epoch=i, validation={"loss": value}) for i, value in enumerate((4., 2., 3.))]
    run = dict(status="FITTED_DEVELOPMENT", finished_utc="2026-10-16T00:00:00Z", completed_epochs=3,
        best_epoch=1, stale_epochs=1, best_validation_negative_elbo=2., binding={"seed": 41, "config": {}},
        model_configuration={"source": "unchanged"}, trainer_sources={}, effective_maximum_epochs=20,
        continuation=None, attempt=0, training_seed=41, best_checkpoint_sha256="best")
    checkpoint = {k: deepcopy(run[k]) for k in
                  ("binding", "model_configuration", "trainer_sources", "effective_maximum_epochs", "continuation", "attempt")}
    checkpoint.update(next_epoch=3, history=deepcopy(epochs), best_epoch=1, stale=1, best=2.)
    return checkpoint, run, epochs


def test_terminal_latest_is_validated_without_rewriting_best(terminal):
    checkpoint, run, epochs = terminal
    before = deepcopy(run)
    comparison.validate_latest_checkpoint(checkpoint, run, epochs)
    assert run == before
    assert checkpoint["next_epoch"] - 1 != run["best_epoch"]


@pytest.mark.parametrize("name,value", [
    ("next_epoch", 2), ("best_epoch", 2), ("stale", 0), ("best", 3.),
    ("binding", {"seed": 42}), ("model_configuration", {}), ("trainer_sources", {"changed": "x"}),
    ("effective_maximum_epochs", 80), ("attempt", 1), ("continuation", {"unexpected": True}),
    ("history", []),
])
def test_latest_rejects_mismatched_terminal_payload(terminal, name, value):
    checkpoint, run, epochs = terminal
    checkpoint[name] = value
    with pytest.raises(ValueError):
        comparison.validate_latest_checkpoint(checkpoint, run, epochs)


@pytest.mark.parametrize("mutation", ("running", "missing_epoch", "nonfinite_validation", "false_best"))
def test_latest_rejects_uncommitted_or_invalid_history(terminal, mutation):
    checkpoint, run, epochs = terminal
    if mutation == "running":
        run["status"] = "RUNNING"
    elif mutation == "missing_epoch":
        epochs[1]["epoch"] = 9
        checkpoint["history"] = deepcopy(epochs)
    elif mutation == "nonfinite_validation":
        epochs[0]["validation"]["loss"] = float("nan")
        checkpoint["history"] = deepcopy(epochs)
    else:
        run["best_validation_negative_elbo"] = checkpoint["best"] = 7.
    with pytest.raises(ValueError):
        comparison.validate_latest_checkpoint(checkpoint, run, epochs)


def test_load_latest_uses_actual_payload_and_preserves_source_buffers():
    constructor = dict(unit_ids=[0, 1, 2, 3], unit_field_index=[0, 0, 1, 1], field_names=["E", "I"],
        minimum_interval_ticks=[2, 2, 1, 1], tick_seconds=.001, bin_seconds=.005,
        input_size=2, latent_channels=2, encoder_size=3, initialization_seed=41)
    original = ControlledPopulationState(**constructor)
    with torch.no_grad():
        original.posterior_mean.bias.fill_(3.)
    payload = dict(binding={"source_id": "fixture"}, model_configuration=constructor,
                   trainer_sources={}, model=deepcopy(original.state_dict()))
    run = dict(training_seed=41, binding={"config": {}}, best_checkpoint_sha256="unchanged-best")
    row = dict(checkpoint=payload, run=run)
    cohort = SimpleNamespace(binding={"source_id": "fixture"},
        population_configuration=lambda config, seed: constructor)
    before = deepcopy(run)
    loaded = comparison.load_latest_model(row, cohort)
    assert torch.equal(loaded.posterior_mean.bias, torch.full((2,), 3.))
    assert run == before
    payload["model"]["minimum_interval_ticks"] = torch.ones(4, dtype=torch.long)
    with pytest.raises(ValueError, match="SOURCE_BUFFER"):
        comparison.load_latest_model(row, cohort)


def complete_rows():
    return [dict(training_seed=seed, factual_episode_id=episode, condition_index=condition,
        best_source_forecast_energy_score=1.,
        latest_source_forecast_energy_score={41: .5, 42: 1., 43: 2.}[seed])
        for seed in comparison.SEEDS for episode in comparison.FACTUAL_EPISODES for condition in range(10)]


def test_equal_condition_selection_keeps_all_seeds_and_retains_best_on_ties():
    rows = complete_rows()
    selected = comparison.select_checkpoints(rows)
    assert [r["selected_checkpoint_kind"] for r in selected] == ["latest", "best", "best"]
    assert [r["paired_conditions"] for r in selected] == [30, 30, 30]
    # Order cannot change the result, and a large single-condition error is retained.
    assert comparison.select_checkpoints(rows[::-1]) == selected
    rows[0]["latest_source_forecast_energy_score"] = 100.
    assert comparison.select_checkpoints(rows)[0]["selected_checkpoint_kind"] == "best"


@pytest.mark.parametrize("mutation", ("drop", "duplicate", "foreign_seed", "foreign_condition", "nonfinite"))
def test_selection_rejects_incomplete_or_changed_denominator(mutation):
    rows = complete_rows()
    if mutation == "drop":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif mutation == "foreign_seed":
        rows[-1]["training_seed"] = 44
    elif mutation == "foreign_condition":
        rows[-1]["condition_index"] = 10
    else:
        rows[-1]["latest_source_forecast_energy_score"] = float("inf")
    with pytest.raises(ValueError):
        comparison.select_checkpoints(rows)


def test_latest_case_binding_does_not_relabel_training_best():
    config = dict(latest_sha256="latest", epoch=16)
    context = dict(admitted={43: dict(selected_checkpoint_sha256="latest", selected_checkpoint_epoch=16,
                                     run={"best_checkpoint_sha256": "original-best"})})
    original_binding = dict(checkpoint_sha256="original-best", source="source")
    with patch.object(comparison.experiment, "_case_inputs",
                      return_value=("view", "data", ["candidate"], deepcopy(original_binding))):
        _, _, _, bound = comparison.latest_case_inputs(context, 43, "episode")
    assert bound["selected_checkpoint_kind"] == "latest"
    assert bound["checkpoint_sha256"] == "latest"
    assert bound["training_best_checkpoint_sha256"] == "original-best"
    assert context["admitted"][43]["run"]["best_checkpoint_sha256"] == "original-best"


def test_generation_and_replay_dispatch_all_nine_existing_cases(tmp_path):
    context = dict(binding={"source": "same"}, target="target", config={"unchanged": True})
    calls = []
    def inputs(context, seed, episode):
        return "view", "data", ["c0", "c1"], {"seed": seed, "checkpoint_sha256": f"latest-{seed}"}
    def execute(path, view, data, target, candidates, config, binding):
        calls.append(("generate", binding["seed"], path.name))
        path.mkdir(parents=True, exist_ok=True)
        report = dict(candidates=[dict(candidate=f"condition_{i}", objective=float(i)) for i in range(10)])
        (path / "analysis.json").write_text(json.dumps(report))
        return report
    def replay(path, source, view, data, target, candidates, config, binding):
        report = execute(path, view, data, target, candidates, config, binding)
        calls[-1] = ("replay", binding["seed"], path.name)
        report["saved_candidate_array_values_compared"] = 38331370
        return report
    arguments = ("inputs", "prepared", "config", "reference", {}, "best", "acceptance")
    with patch.object(comparison, "admit_best_execution", return_value={}), \
         patch.object(comparison, "latest_context", return_value=context), \
         patch.object(comparison, "latest_case_inputs", side_effect=inputs), \
         patch.object(comparison.experiment, "execute_case", side_effect=execute), \
         patch.object(comparison.experiment, "replay_case", side_effect=replay):
        generated = comparison.run_latest_phase(*arguments, tmp_path / "latest")
        repeated = comparison.run_latest_phase(*arguments, tmp_path / "replay", replay_from=tmp_path / "latest")
        # Reuse commits byte-for-byte at the caller level; actual case reuse is the existing execute_case contract.
        assert comparison.run_latest_phase(*arguments, tmp_path / "latest") == generated
    assert generated["candidate_rows"] == repeated["candidate_rows"] == 90
    assert generated["generated_unit_count_values"] == 141120000000
    assert not generated["best_predictions_regenerated"]
    assert repeated["saved_candidate_array_values_compared"] == 344982330
    assert len([r for r in calls if r[0] == "replay"]) == 9


def test_cli_requires_three_exact_seeds_and_explicit_replay_source():
    common = ["--phase", "generate", "--inputs", "x", "--preparation", "x",
              "--config", "x", "--reference", "x", "--best-execution", "x",
              "--best-acceptance", "x", "--output", "x"]
    with pytest.raises(SystemExit):
        comparison.main([*common, "--seed-cell", "41", "x", "--seed-cell", "42", "x"])
    with pytest.raises(SystemExit):
        comparison.main(["--phase", "compare", "--best-execution", "x", "--best-acceptance", "x", "--output", "x"])


def test_bound_artifact_rejects_corruption_and_unsafe_paths(tmp_path):
    path = tmp_path / "small.json"
    path.write_text("{}")
    index = {"small.json": dict(bytes=2, sha256=sha256_file(path))}
    assert comparison.bound_file(tmp_path, index, "small.json") == path
    with pytest.raises(ValueError):
        comparison.bound_file(tmp_path, index, "../small.json")
    path.write_text("[]")
    with pytest.raises(ValueError):
        comparison.bound_file(tmp_path, index, "small.json")


def test_complete_compare_cli_uses_saved_best_scores_and_proper_energy_score(tmp_path):
    best_root, latest_root, replay_root = (tmp_path / name for name in ("best", "latest", "replay"))
    latest_root.mkdir()
    replay_root.mkdir()
    field_ids = np.asarray([f"field_{i}" for i in range(17)])
    ids = np.arange(100000)
    native_arrays = dict(field_rate_hz=np.ones((980, 17)), field_ids=field_ids, unit_ids=ids)
    model_arrays = dict(field_rate_hz=np.full((16, 980, 17), 2.), field_ids=field_ids, unit_ids=ids)
    class ArrayFile(dict):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    best = dict(root=best_root, index={}, binding={"accepted": "best"}, config={"fixed": True},
                target=SimpleNamespace(scale_hz=np.full(17, 2.), field_names=tuple(field_ids)),
                cases={}, scores={}, sources={})
    binding = dict(best_execution=best["binding"], configuration=best["config"],
        selection=deepcopy(comparison.SELECTION),
        latest_checkpoints={str(s): dict(latest_sha256=f"latest-{s}", best_sha256=f"best-{s}")
                            for s in comparison.SEEDS})
    entries, repeats = [], []
    for seed in comparison.SEEDS:
        for episode in comparison.FACTUAL_EPISODES:
            relative = Path(f"seed_{seed}") / episode
            candidates = [dict(candidate=f"c{i}", candidate_identity={"condition_index": i},
                source_config={"condition": i}, registered_response_binding={"source": i}, objective=float(i))
                for i in range(10)]
            old = dict(candidates=deepcopy(candidates), candidate_ranking=[f"c{i}" for i in range(10)],
                       selected_candidate="c0")
            best["cases"][seed, episode] = dict(report=old)
            case_binding = dict(experiment=binding, training_seed=seed, factual_episode_id=episode,
                                selected_checkpoint_kind="latest", checkpoint_sha256=f"latest-{seed}")
            report = dict(experiment_binding=case_binding, candidates=candidates,
                          candidate_ranking=old["candidate_ranking"], selected_candidate="c0")
            for root, value in ((latest_root, report), (replay_root, {"replayed": True})):
                (root / relative).mkdir(parents=True)
                (root / relative / "analysis.json").write_text(json.dumps(value))
            entries.append(dict(training_seed=seed, factual_episode_id=episode, path=str(relative),
                                analysis_sha256=sha256_file(latest_root / relative / "analysis.json")))
            repeats.append(dict(training_seed=seed, factual_episode_id=episode, path=str(relative),
                                analysis_sha256=sha256_file(replay_root / relative / "analysis.json")))
            for i in range(10):
                source_id = episode.rsplit("_condition_", 1)[0] + f"_condition_{i:02d}"
                best["scores"][seed, episode, f"c{i}"] = dict(condition_index=str(i), source_episode_id=source_id,
                    source_forecast_energy_score="0.5", predicted_reference_energy_score=str(i))
                best["sources"][episode, f"c{i}"] = dict(source_episode_id=source_id, source_arrays=source_id+".npz")
    for root, phase, cases in ((latest_root, "generate", entries), (replay_root, "replay", repeats)):
        (root / "analysis.json").write_text(json.dumps(dict(status=comparison.STATUSES[phase],
            candidate_rows=90, seed_block_cases=9, generated_unit_count_values=comparison.GENERATED_VALUES,
            selected_checkpoint_kind="latest", per_bin_full_unit_counts_exact_replay_claimed=False,
            binding=binding, cases=cases)))
    score_calls = []
    def proper_score(model, native):
        assert model.shape == (16, 980, 17) and native.shape == (980, 17)
        assert model[0, 0, 0] == 1. and native[0, 0] == .5
        score_calls.append(True)
        return .25
    def arrays(path, **kwargs):
        return ArrayFile(native_arrays if str(path).startswith(str(best_root)) else model_arrays)
    with patch.object(comparison, "admit_best_execution", return_value=best), \
         patch.object(comparison, "bound_file", side_effect=lambda root, index, name: Path(root) / name), \
         patch.object(comparison, "verify_artifacts"), \
         patch.object(comparison.experiment, "_verify_replay_case"), \
         patch.object(comparison.np, "load", side_effect=arrays), \
         patch.object(comparison, "energy_score", side_effect=proper_score):
        exit_code = comparison.main(["--phase", "compare", "--best-execution", str(best_root),
            "--best-acceptance", "accepted", "--latest", str(latest_root), "--replay", str(replay_root),
            "--output", str(tmp_path / "comparison")])
    assert exit_code == 0 and len(score_calls) == 90
    output = comparison.read(tmp_path / "comparison/analysis.json")
    assert output["candidate_rows"] == 90 and output["native_source_episodes_reused"] == 30
    assert [row["selected_checkpoint_kind"] for row in output["comparison_selection"]] == ["latest"] * 3
    assert len(comparison.table(tmp_path / "comparison/paired_candidates.csv")) == 90
    assert not output["best_predictions_regenerated"]
    assert not output["independent_intervention_qualified"]
