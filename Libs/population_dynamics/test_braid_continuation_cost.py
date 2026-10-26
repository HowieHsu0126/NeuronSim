"""Use the population continuation cost contract in direct baseline consumers."""
from __future__ import annotations

import itertools
import json

import pytest

from Libs.scripts.run_braid_population_comparison import _stage_comparison_cell

from .braid_population_comparison import DeliveredCell, ORIGINS, _training_cost
from .provenance import sha256_file
from .test_population_continuation_cost import continuation_fixture, row, write_json


def test_braid_cost_consumer_preserves_parent_and_incremental_accounting(tmp_path):
    _, child, _, run, _, proof = continuation_fixture(tmp_path, 42, 1)
    proof.update(status="PASS", exit_code=0, training_seed=42, scientific_claim_qualified=False)
    write_json(child / "delivery.json", proof)
    run.update(parameter_count=520664, best_epoch=79, stop_reason="epoch_budget",
               peak_process_rss_bytes=1000000)
    result = _training_cost(DeliveredCell(child, 42), run, "population")
    assert result["recorded_training_seconds"] == 125.
    assert result["training_attempts"] == 3
    assert result["completed_epochs"] == 80
    assert result["best_epoch"] == 79
    assert "each once" in result["training_cost_scope"]
    parent = json.loads((child / "prerequisites/budget_parent/training/run.json").read_text())
    parent["elapsed_seconds"] = 1.
    write_json(child / "prerequisites/budget_parent/training/run.json", parent)
    with pytest.raises(ValueError, match="binding changed"):
        _training_cost(DeliveredCell(child, 42), run, "population")


def _staging_fixture(root, seed=42, parent_attempt=1, child_attempt=3):
    parent, child, _, run, _, proof = continuation_fixture(root, seed, parent_attempt, child_attempt)
    proof.update(status="PASS", exit_code=0, training_seed=seed, scientific_claim_qualified=False)
    run.update(parameter_count=520664, best_epoch=79, stop_reason="epoch_budget",
               peak_process_rss_bytes=1000000)
    write_json(child / "training/run.json", run)
    # Staging copies saved forecasts; the cost consumer does not interpret them.
    names = ["forecasts/evaluation.json", "forecasts/scores.csv", "replay.json",
             "forecasts/fitting_field_scale.npz"]
    names += [f"forecasts/train_block_{block:03d}_familiar_{template}_origin_{origin:g}.npz"
              for block, template, origin in itertools.product(range(14, 18), range(4), ORIGINS)]
    for name in names:
        path = child / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"saved forecast fixture")
    proof["files"] += [row(child, name) for name in ["training/run.json", *names]]
    write_json(child / "delivery.json", proof)
    return parent, child, run


@pytest.mark.parametrize("seed,parent_attempt,child_attempt", [(41, 0, 1), (42, 1, 3), (43, 1, 2)])
def test_wrapper_stage_can_consume_parent_and_incremental_cost_once(
        tmp_path, seed, parent_attempt, child_attempt):
    parent, child, run = _staging_fixture(tmp_path, seed, parent_attempt, child_attempt)
    destination = tmp_path / "staged" / f"seed_{seed}"
    _stage_comparison_cell(DeliveredCell(child, seed), "population", destination, ORIGINS, sha256_file)
    staged = DeliveredCell(destination, seed)
    result = _training_cost(staged, staged.read("training/run.json"), "population")
    assert result["recorded_training_seconds"] == 100. + 20. * parent_attempt + 5. + 7. * (
        child_attempt - parent_attempt - 1)
    assert result["training_attempts"] == child_attempt + 1
    assert result["completed_epochs"] == run["completed_epochs"] == 80
    assert "each once" in result["training_cost_scope"]
    assert staged.proof["optimization_budget_continuation"]["parent"] == str(parent)
    for name in ("prerequisites/budget_parent/delivery.json", "prerequisites/budget_parent/training/run.json"):
        assert (destination / name).read_bytes() == (child / name).read_bytes()
    for attempt in range(parent_attempt + 1):
        assert not (destination / f"training/attempts/attempt_{attempt:03d}.json").exists()
    assert not (destination / f"training/attempts/attempt_{child_attempt:03d}.json").exists()
    if child_attempt > parent_attempt + 1:
        assert (destination / f"training/attempts/attempt_{parent_attempt + 1:03d}.json").is_file()


@pytest.mark.parametrize("name", ["prerequisites/budget_parent/delivery.json",
                                  "prerequisites/budget_parent/training/run.json"])
@pytest.mark.parametrize("location", ["source", "staged"])
def test_wrapper_stage_rejects_changed_required_parent_files(tmp_path, name, location):
    _, child, _ = _staging_fixture(tmp_path)
    destination = tmp_path / "staged" / "seed_42"
    if location == "source":
        (child / name).write_text("{}")
        with pytest.raises(ValueError, match="delivered comparison input changed"):
            _stage_comparison_cell(DeliveredCell(child, 42), "population", destination, ORIGINS, sha256_file)
    else:
        _stage_comparison_cell(DeliveredCell(child, 42), "population", destination, ORIGINS, sha256_file)
        (destination / name).write_text("{}")
        staged = DeliveredCell(destination, 42)
        with pytest.raises(ValueError, match="continuation training cost binding changed"):
            _training_cost(staged, staged.read("training/run.json"), "population")

def test_prior_staging_supplies_selected_parent_and_counts_all_training_once(tmp_path):
    from .test_braid_population_comparison import fixture, prior_population_fixture
    from .braid_population_comparison import _population_prior_binding
    fixture(tmp_path)
    population = prior_population_fixture(tmp_path)
    source = DeliveredCell(population / "seed_41", 41)
    target = tmp_path / "staged/seed_41"
    _stage_comparison_cell(source, "population", target, ORIGINS, sha256_file)
    staged = DeliveredCell(target, 41)
    run = staged.read("training/run.json")
    binding = _population_prior_binding(staged, run)
    cost = _training_cost(staged, run, "population")
    assert binding["method"] == cost["method"] == "population_recurrent_prior_refit"
    assert binding["prior_refit_lineage"]["parent_completed_epochs"] == 80
    assert cost["recorded_training_seconds"] == 190. and cost["training_attempts"] == 4
    for name in ("training/best.pt", "parent_training/run.json", "parent_training/best.pt",
                 "prerequisites/prior_parent.json", "training/attempts/attempt_000.json"):
        assert (target / name).read_bytes() == (source.root / name).read_bytes()
    assert not (target / "training/latest.pt").exists()
    assert not (target / "training/attempts/attempt_001.json").exists()
    # Identical observation implementations do not require an unread code copy.
    assert not (target / "Libs/population_dynamics/refractory_observation.py").exists()
    (target / "parent_training/run.json").write_text("{}")
    with pytest.raises(ValueError, match="staged prior parent binding changed"):
        _population_prior_binding(DeliveredCell(target, 41), run)


@pytest.mark.parametrize("node,partition,job_gpu,step_gpu,accepted", [
    ("cpu04", "cpu", "", "", True), ("cpu03", "gpu", "", "", True),
    ("cpu03", "cpu", "", "", False), ("cpu05", "gpu", "", "", False),
    ("mgmt2", "cpu", "", "", False), ("cpu04", "cpu", "0", "", False),
    ("cpu03", "gpu", "", "0", False)])
def test_braid_comparison_cpu_placement_keeps_allowed_nodes_and_rejects_gpus(
        monkeypatch, node, partition, job_gpu, step_gpu, accepted):
    from Libs.scripts.run_braid_population_comparison import cpu_placement
    for key, value in dict(SLURM_JOB_ID="fixture", SLURMD_NODENAME=node, SLURM_JOB_PARTITION=partition,
                           SLURM_JOB_GPUS=job_gpu, SLURM_STEP_GPUS=step_gpu).items():
        monkeypatch.setenv(key, value)
    if accepted:
        cpu_placement()
    else:
        with pytest.raises(ValueError, match="without a GPU"):
            cpu_placement()
