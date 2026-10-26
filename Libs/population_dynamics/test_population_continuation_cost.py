"""Account for delivered parent and continuation training time once each."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from . import population_forecast_analysis as analysis
from .provenance import sha256_file


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def row(root, name):
    path = root / name
    return dict(path=name, bytes=path.stat().st_size, sha256=sha256_file(path))


def continuation_fixture(root, seed, parent_attempt, child_attempt=None):
    parent, child = root / "parent" / f"seed_{seed}", root / "child" / f"seed_{seed}"
    binding = dict(seed=seed, config=dict(maximum_epochs=20, early_stopping_patience=4,
        representation="recurrent"), dataset_sha256="a" * 64, model_implementation_sha256="b" * 64)
    sources = {"population_state.py": "b" * 64, "renewal_trajectory_training.py": "c" * 64}
    (parent / "training").mkdir(parents=True)
    (parent / "training/latest.pt").write_bytes(b"fixture committed checkpoint")
    parent_run = dict(status="FITTED_DEVELOPMENT", attempt=parent_attempt, training_seed=seed,
        binding=binding, trainer_sources=sources, elapsed_seconds=100., completed_epochs=20,
        latest_checkpoint_sha256=sha256_file(parent / "training/latest.pt"), continuation=None)
    write_json(parent / "training/run.json", parent_run)
    parent_names = ["training/run.json", "training/latest.pt"]
    for attempt in range(parent_attempt):
        name = f"training/attempts/attempt_{attempt:03d}.json"
        write_json(parent / name, dict(parent_run, attempt=attempt, elapsed_seconds=20. + attempt))
        parent_names.append(name)
    # A current-attempt copy is an archive receipt, not a second cost charge.
    current_name = f"training/attempts/attempt_{parent_attempt:03d}.json"
    write_json(parent / current_name, parent_run)
    parent_names.append(current_name)
    parent_delivery = dict(status="PASS", exit_code=0, training_seed=seed,
        scientific_claim_qualified=False, calibration_or_test_accessed=False,
        files=[row(parent, name) for name in parent_names])
    write_json(parent / "delivery.json", parent_delivery)
    local = child / "prerequisites/budget_parent"
    (local / "training").mkdir(parents=True)
    shutil.copy2(parent / "delivery.json", local / "delivery.json")
    shutil.copy2(parent / "training/run.json", local / "training/run.json")
    continuation = dict(schema="neural_population_trajectory_budget_continuation_v1",
        parent_binding=copy.deepcopy(binding), parent_source_hashes=sources, current_source_hashes=sources,
        source_migration={}, parent_artifact_sha256={"run.json": sha256_file(parent / "training/run.json")},
        parent_checkpoint_sha256=parent_run["latest_checkpoint_sha256"], parent_completed_epochs=20,
        base_maximum_epochs=20, total_epochs=80, parent_run="/tmp/removed-budget-work/prerequisites/budget_parent/training")
    child_attempt = parent_attempt + 1 if child_attempt is None else child_attempt
    run = dict(parent_run, attempt=child_attempt, elapsed_seconds=5., completed_epochs=80,
               effective_maximum_epochs=80, continuation=continuation)
    child_names = ["prerequisites/budget_parent/delivery.json", "prerequisites/budget_parent/training/run.json"]
    for attempt in range(parent_attempt + 1, child_attempt):
        name = f"training/attempts/attempt_{attempt:03d}.json"
        write_json(child / name, dict(run, attempt=attempt, elapsed_seconds=7.))
        child_names.append(name)
    current_name = f"training/attempts/attempt_{child_attempt:03d}.json"
    write_json(child / current_name, run)
    child_names.append(current_name)
    delivery = dict(optimization_budget_continuation=dict(parent=str(parent),
        parent_delivery_sha256=sha256_file(parent / "delivery.json"),
        parent_checkpoint_sha256=parent_run["latest_checkpoint_sha256"], parent_completed_epochs=20, total_epochs=80),
        files=[row(child, name) for name in child_names])
    write_json(child / "delivery.json", delivery)
    return parent, child, parent_run, run, parent_delivery, delivery


@pytest.mark.parametrize("seed,parent_attempt", [(41, 0), (42, 1), (43, 1)])
def test_actual_parent_manifest_and_incremental_attempt_are_counted_once(tmp_path, seed, parent_attempt):
    parent, child, _, run, _, _ = continuation_fixture(tmp_path, seed, parent_attempt)
    result = analysis._training_attempt_cost(child, run)
    assert result["training_elapsed_seconds"] == 105. + 20. * parent_attempt
    assert result["parent_training_elapsed_seconds"] == 100. + 20. * parent_attempt
    assert result["incremental_training_elapsed_seconds"] == 5.
    assert result["training_attempts"] == parent_attempt + 2
    assert result["parent_training_attempts"] == parent_attempt + 1
    assert result["incremental_training_attempts"] == 1
    assert "unmeasured" in result["training_cost_scope"]
    assert not (child / "prerequisites/budget_parent/training/attempts").exists()


@pytest.mark.parametrize("seed", [42, 43])
def test_incremental_resource_attempts_are_added_without_readding_parent(tmp_path, seed):
    _, child, _, run, _, _ = continuation_fixture(tmp_path, seed, 1, child_attempt=3)
    view = tmp_path / "analysis_campaign"
    view.mkdir()
    (view / f"seed_{seed}").symlink_to(child, target_is_directory=True)
    result = analysis._training_attempt_cost(view / f"seed_{seed}", run)
    assert result["parent_training_elapsed_seconds"] == 120.
    assert result["incremental_training_elapsed_seconds"] == 12.
    assert result["training_elapsed_seconds"] == 132.
    assert result["training_attempts"] == 4


@pytest.mark.parametrize("failure", ["parent_payload", "parent_delivery", "staged_run", "wrong_seed",
    "wrong_binding", "wrong_sources", "wrong_parent_report", "missing_prior_metadata", "missing_prior_elapsed",
    "wrong_incremental_attempt"])
def test_changed_or_missing_parent_and_incremental_cost_evidence_is_rejected(tmp_path, failure):
    parent, child, parent_run, run, parent_delivery, delivery = continuation_fixture(tmp_path, 42, 1, child_attempt=3)
    if failure == "parent_payload":
        (parent / "training/attempts/attempt_000.json").write_text("{}")
    elif failure == "parent_delivery":
        (parent / "delivery.json").write_text("{}")
    elif failure == "staged_run":
        (child / "prerequisites/budget_parent/training/run.json").write_text("{}")
    elif failure == "wrong_seed":
        run["training_seed"] = 41
    elif failure == "wrong_binding":
        run["binding"] = dict(run["binding"], dataset_sha256="foreign")
    elif failure == "wrong_sources":
        run["trainer_sources"] = {"population_state.py": "foreign"}
    elif failure == "wrong_parent_report":
        run["continuation"]["parent_artifact_sha256"]["run.json"] = "e" * 64
    elif failure in ("missing_prior_metadata", "missing_prior_elapsed"):
        if failure == "missing_prior_metadata":
            parent_delivery["files"] = [r for r in parent_delivery["files"] if r["path"] != "training/attempts/attempt_000.json"]
        else:
            name = "training/attempts/attempt_000.json"
            record = json.loads((parent / name).read_text())
            record["elapsed_seconds"] = None
            write_json(parent / name, record)
            parent_delivery["files"] = [row(parent, r["path"]) for r in parent_delivery["files"]]
        write_json(parent / "delivery.json", parent_delivery)
        shutil.copy2(parent / "delivery.json", child / "prerequisites/budget_parent/delivery.json")
        delivery["optimization_budget_continuation"]["parent_delivery_sha256"] = sha256_file(parent / "delivery.json")
        delivery["files"] = [row(child, r["path"]) for r in delivery["files"]]
        write_json(child / "delivery.json", delivery)
    else:
        run["attempt"] = 5
    with pytest.raises(ValueError):
        analysis._training_attempt_cost(child, run)


def test_current_unmeasured_time_remains_null(tmp_path):
    _, child, _, run, _, _ = continuation_fixture(tmp_path, 41, 0)
    run["elapsed_seconds"] = None
    result = analysis._training_attempt_cost(child, run)
    assert result["training_elapsed_seconds"] is None
    assert result["parent_training_elapsed_seconds"] == 100.
    assert result["incremental_training_elapsed_seconds"] is None
