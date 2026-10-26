"""Prior budget parent staging and mode contracts."""
import importlib.util
import json
from pathlib import Path

import pytest


def _module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parent(tmp_path, *, mixture=False):
    helper = _module("run_population_trajectory_campaign.py")
    caller = _module("run_population_prior_campaign.py")
    configuration = caller.MIXTURE_CONFIG if mixture else caller.CONFIG
    schema, objective, selection = caller.PRIOR_PROFILES[configuration]
    parent, work = tmp_path / "parent", tmp_path / "work"
    (parent / "training").mkdir(parents=True)
    work.mkdir()
    report = dict(status="FITTED_DEVELOPMENT", training_seed=41, completed_epochs=80, best_epoch=79,
                  training_objective=objective, selection_metric=selection,
                  effective_maximum_epochs=80, continuation=None, calibration_outcomes_accessed=False,
                  test_outcomes_accessed=False, frozen_parameters_unchanged_in_attempt=True,
                  binding=dict(config=dict(schema=schema, observation_objective=objective, maximum_epochs=80,
                                           fit_blocks=list(range(14)), validation_blocks=list(range(14, 18)))))
    (parent / "training/run.json").write_text(json.dumps(report))
    for name in ("latest.pt", "best.pt", "epochs.json", "optimization.csv", "validation.csv"):
        (parent / "training" / name).write_text(name)
    for block in range(18):
        for template in range(4):
            path = parent / f"training/posterior_cache/train_block_{block:03d}_familiar_{template}.pt"
            path.parent.mkdir(exist_ok=True)
            path.write_text(f"{block}:{template}")
    rows = [dict(path=str(path.relative_to(parent)), bytes=path.stat().st_size, sha256=helper.digest(path))
            for path in sorted(parent.rglob("*")) if path.is_file()]
    proof = dict(status="PASS", exit_code=0, training_seed=41,
                 parent_delivery_sha256="original-parent", files=rows)
    (parent / "delivery.json").write_text(json.dumps(proof))
    return helper, caller, parent, work


@pytest.mark.parametrize("mixture", [False, True])
def test_prior_budget_stages_cached_parent_without_creating_child_output(tmp_path, mixture):
    helper, caller, parent, work = _parent(tmp_path, mixture=mixture)
    configuration = caller.MIXTURE_CONFIG if mixture else caller.CONFIG
    before = helper.digest(parent / "delivery.json")
    lineage = caller.stage_budget_parent(helper, parent, work, 41, "original-parent", 320,
                                         configuration=configuration)
    assert lineage["total_epochs"] == 320 and lineage["parent_completed_epochs"] == 80
    assert lineage["reused_posterior_cache_files"] == 72
    assert len(list((work / "budget_training/posterior_cache").glob("*.pt"))) == 72
    assert not (work / "training").exists()
    assert helper.digest(parent / "delivery.json") == before == lineage["parent_delivery_sha256"]
    assert (work / "prerequisites/budget_parent/training/run.json").read_bytes() == (parent / "training/run.json").read_bytes()


@pytest.mark.parametrize("mixture", [False, True])
@pytest.mark.parametrize("failure", ["seed", "original_parent", "budget", "cache"])
def test_prior_budget_rejects_foreign_parent_budget_or_changed_cache(tmp_path, failure, mixture):
    helper, caller, parent, work = _parent(tmp_path, mixture=mixture)
    configuration = caller.MIXTURE_CONFIG if mixture else caller.CONFIG
    seed, digest, total = 41, "original-parent", 320
    if failure == "seed": seed = 42
    elif failure == "original_parent": digest = "foreign"
    elif failure == "budget": total = 80
    else: (parent / "training/posterior_cache/train_block_000_familiar_0.pt").write_text("changed")
    with pytest.raises((ValueError, RuntimeError)):
        caller.stage_budget_parent(helper, parent, work, seed, digest, total, configuration=configuration)


def test_prior_budget_and_recovery_modes_are_explicit(tmp_path):
    caller = _module("run_population_prior_campaign.py")
    for options in (dict(budget_parent=tmp_path), dict(total_epochs=320),
                    dict(budget_parent=tmp_path, total_epochs=320, resume_from=tmp_path)):
        with pytest.raises(ValueError, match="explicit budget"):
            caller.run(tmp_path, tmp_path, tmp_path, 41, tmp_path, **options)


@pytest.mark.parametrize("mixture", [False, True])
def test_prior_budget_rejects_a_parent_from_the_other_objective(tmp_path, mixture):
    helper, caller, parent, work = _parent(tmp_path, mixture=mixture)
    requested = caller.CONFIG if mixture else caller.MIXTURE_CONFIG
    with pytest.raises(ValueError, match="same-seed prior parent"):
        caller.stage_budget_parent(helper, parent, work, 41, "original-parent", 320,
                                   configuration=requested)
    assert not (work / "training").exists()


@pytest.mark.parametrize("field", ["training_objective", "selection_metric"])
def test_mixture_budget_rejects_report_objective_substitution(tmp_path, field):
    helper, caller, parent, work = _parent(tmp_path, mixture=True)
    report_path = parent / "training/run.json"
    report = json.loads(report_path.read_text())
    report[field] = "analytic_kl_substitution"
    report_path.write_text(json.dumps(report))
    delivery_path = parent / "delivery.json"
    proof = json.loads(delivery_path.read_text())
    row = next(row for row in proof["files"] if row["path"] == "training/run.json")
    row.update(bytes=report_path.stat().st_size, sha256=helper.digest(report_path))
    delivery_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="same-seed prior parent"):
        caller.stage_budget_parent(helper, parent, work, 41, "original-parent", 320,
                                   configuration=caller.MIXTURE_CONFIG)
