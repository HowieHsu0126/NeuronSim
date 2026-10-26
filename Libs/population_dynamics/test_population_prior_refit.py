"""Refit lifecycle tests replace only the complete native cohort boundary."""
from pathlib import Path
import copy
import json

import pytest
import torch
import yaml

from . import renewal_trajectory_training as training
from .population_prior_refit import OVERRIDES, initialize_from_parent
from .population_prior_objective import PRIOR_PARAMETER_NAMES
from .population_state import ControlledPopulationState
from .provenance import sha256_file
from .test_renewal_trajectory_training import TinyCohort, _same


class CompleteClockFixture(TinyCohort):
    reads = []
    def __init__(self, acquisition, fields, config):
        super().__init__(acquisition, fields, config)
        self.duration_ticks = 60000

    def window(self, case):
        self.reads.append(case["case_id"])
        return super().window(case)


def make_parent(tmp_path):
    config = dict(training.POPULATION_FIXED_BUDGET, training_variational_seed_base=61001,
                  validation_seed_base=64001, cpu_threads=1, device="cpu")
    cohort = CompleteClockFixture(None, None, config)
    constructor = training.model_configuration(cohort, config, 41, "population")
    model = ControlledPopulationState(**constructor)
    binding = dict(config=config, seed=41, **cohort.binding)
    parent = tmp_path / "parent"
    parent.mkdir()
    checkpoint = dict(binding=binding, model_configuration=constructor, model=model.state_dict(),
                      next_epoch=80, initialization={"fixture": "synthetic_parent"})
    torch.save(checkpoint, parent / "best.pt")
    digest = sha256_file(parent / "best.pt")
    (parent / "run.json").write_text(json.dumps(dict(status="FITTED_DEVELOPMENT", binding=binding,
        model_configuration=constructor, best_checkpoint_sha256=digest, completed_epochs=80)))
    config.update(OVERRIDES)
    config["parent_checkpoint_sha256"] = {str(seed): digest for seed in (41, 42, 43)}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return parent, config, path, checkpoint, cohort, constructor


def test_refit_resume_preserves_optimizer_both_streams_cache_and_frozen_weights(tmp_path, monkeypatch):
    parent, config, path, original, _, _ = make_parent(tmp_path)
    monkeypatch.setattr(training, "TrainingCohort", CompleteClockFixture)
    CompleteClockFixture.reads = []
    full, split = tmp_path / "full", tmp_path / "split"
    train = lambda out, **kwargs: training.train(tmp_path, tmp_path, path, out, 41,
                                                backend="population", **kwargs)
    report = train(full, prior_parent=parent, stop_after_epochs=2)
    assert len(CompleteClockFixture.reads) == 3
    assert report["frozen_parameters_unchanged_in_attempt"]
    assert "best_validation_prior_loss" in report
    assert "best_validation_negative_elbo" not in report
    assert not report["observation_likelihood_evaluated_during_refit"]
    train(split, prior_parent=parent, stop_after_epochs=1)
    assert len(CompleteClockFixture.reads) == 6
    resumed = train(split, resume=True, stop_after_epochs=1)
    assert len(CompleteClockFixture.reads) == 6
    assert resumed["resumed"] and resumed["completed_epochs"] == 2
    a = torch.load(full / "latest.pt", weights_only=True)
    b = torch.load(split / "latest.pt", weights_only=True)
    for key in ("model", "optimizer", "variational_rng", "prior_dynamics_rng", "history", "best", "best_epoch"):
        _same(a[key], b[key])
    for name, value in original["model"].items():
        if name not in PRIOR_PARAMETER_NAMES:
            assert torch.equal(value, b["model"][name]), name
    assert any(not torch.equal(original["model"][name], b["model"][name]) for name in PRIOR_PARAMETER_NAMES)
    assert a["binding"]["config"]["parent_checkpoint_sha256"] == config["parent_checkpoint_sha256"]


def test_new_objective_requires_same_seed_cohort_and_constructor_parent(tmp_path):
    parent, config, _, _, cohort, constructor = make_parent(tmp_path)
    model = ControlledPopulationState(**constructor)
    changed = copy.deepcopy(cohort)
    changed.binding["dataset_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="cohort differs"):
        initialize_from_parent(model, changed, config, 41, parent, constructor)
    wrong = dict(constructor, unit_ids=[7, 3, 12])
    with pytest.raises(ValueError, match="constructor"):
        initialize_from_parent(model, cohort, config, 41, parent, wrong)
    with pytest.raises(ValueError, match="seed"):
        initialize_from_parent(model, cohort, config, 42, parent, constructor)


def _prior_budget_parent(tmp_path, monkeypatch):
    original, config, path, original_checkpoint, _, _ = make_parent(tmp_path)
    monkeypatch.setattr(training, "TrainingCohort", CompleteClockFixture)
    CompleteClockFixture.reads = []
    parent = tmp_path / "prior_parent"
    training.train(tmp_path, tmp_path, path, parent, 41, backend="population",
                   prior_parent=original, stop_after_epochs=2)
    return parent, path, original_checkpoint


def _replace_parent_sources(parent, replacements):
    """Build an authenticated synthetic old-runtime parent for migration tests."""
    report_path = parent / "run.json"
    report = json.loads(report_path.read_text())
    aliases = {"renewal_trajectory_training.py": "trainer_implementation_sha256",
               "poisson_trajectory_training.py": "training_utilities_implementation_sha256"}
    for name in ("latest.pt", "best.pt"):
        path = parent / name
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["trainer_sources"].update(replacements)
        for source, digest in replacements.items():
            if source in aliases:
                checkpoint["binding"][aliases[source]] = digest
        torch.save(checkpoint, path)
    latest = torch.load(parent / "latest.pt", weights_only=True)
    report.update(binding=latest["binding"], trainer_sources=latest["trainer_sources"],
                  latest_checkpoint_sha256=sha256_file(parent / "latest.pt"),
                  best_checkpoint_sha256=sha256_file(parent / "best.pt"))
    report_path.write_text(json.dumps(report))


def test_prior_budget_continuation_preserves_state_cache_and_original_parent(tmp_path, monkeypatch):
    from .poisson_trajectory_training import PRIOR_BUDGET_TRAINER_SOURCES

    parent, path, original = _prior_budget_parent(tmp_path, monkeypatch)
    _replace_parent_sources(parent, PRIOR_BUDGET_TRAINER_SOURCES)
    committed = torch.load(parent / "latest.pt", weights_only=True)
    parent_files = {str(p.relative_to(parent)): sha256_file(p) for p in parent.rglob("*") if p.is_file()}
    assert len(CompleteClockFixture.reads) == 3

    def no_encoding(self, case):
        raise AssertionError("prior budget continuation re-encoded a cached episode")

    monkeypatch.setattr(CompleteClockFixture, "window", no_encoding)
    boundary_states = []
    save = training._save_checkpoint

    def capture_boundary(payload, destination):
        if destination.name == "latest.pt" and payload["next_epoch"] == committed["next_epoch"]:
            boundary_states.append(copy.deepcopy(payload))
        save(payload, destination)

    monkeypatch.setattr(training, "_save_checkpoint", capture_boundary)
    whole, split = tmp_path / "continued_whole", tmp_path / "continued_split"
    invoke = lambda out, **kw: training.train(tmp_path, tmp_path, path, out, 41, backend="population", **kw)
    whole_report = invoke(whole, resume_from=parent, total_epochs=320, stop_after_epochs=2)
    invoke(split, resume_from=parent, total_epochs=320, stop_after_epochs=1)
    split_report = invoke(split, resume=True, stop_after_epochs=1)
    assert len(boundary_states) == 2
    state_keys = ("model", "optimizer", "variational_rng", "prior_dynamics_rng", "history",
                  "best", "stale", "best_epoch", "initialization")
    for boundary in boundary_states:
        for key in state_keys:
            _same(boundary[key], committed[key])
    a, b = [torch.load(out / "latest.pt", weights_only=True) for out in (whole, split)]
    for key in state_keys:
        _same(a[key], b[key])
    assert a["next_epoch"] == b["next_epoch"] == committed["next_epoch"] + 2
    assert a["binding"]["config"]["maximum_epochs"] == 80
    assert a["effective_maximum_epochs"] == b["effective_maximum_epochs"] == 320
    assert whole_report["refit_initialization"] == split_report["refit_initialization"] == committed["initialization"]
    migration = a["continuation"]["source_migration"]
    assert set(migration) == set(PRIOR_BUDGET_TRAINER_SOURCES)
    for source, old_digest in PRIOR_BUDGET_TRAINER_SOURCES.items():
        assert migration[source] == dict(parent=old_digest, current=a["trainer_sources"][source])
    assert a["continuation"]["inherited_stale_epochs"] == committed["stale"]
    assert a["continuation"]["prior_dynamics_rng"] == "explicit_checkpointed_generator"
    for name, value in original["model"].items():
        if name not in PRIOR_PARAMETER_NAMES:
            assert torch.equal(value, a["model"][name]), name
    cache = {name: digest for name, digest in parent_files.items() if name.startswith("posterior_cache/")}
    assert len(cache) == 3
    for out in (whole, split):
        assert {name: sha256_file(out / name) for name in cache} == cache
    assert {str(p.relative_to(parent)): sha256_file(p) for p in parent.rglob("*") if p.is_file()} == parent_files


@pytest.mark.parametrize("corruption", ["changed", "elbo_substitution"])
def test_prior_budget_continuation_rejects_wrong_selection_metric(tmp_path, monkeypatch, corruption):
    parent, path, _ = _prior_budget_parent(tmp_path, monkeypatch)
    report_path = parent / "run.json"
    report = json.loads(report_path.read_text())
    if corruption == "changed":
        report["best_validation_prior_loss"] += 1.
    else:
        report["best_validation_negative_elbo"] = report.pop("best_validation_prior_loss")
    report_path.write_text(json.dumps(report))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="best/stale"):
        training.train(tmp_path, tmp_path, path, output, 41, backend="population",
                       resume_from=parent, total_epochs=320, stop_after_epochs=1)
    assert not output.exists()


@pytest.mark.parametrize("source", [
    "poisson_trajectory_training.py", "renewal_trajectory_training.py", "population_state.py",
    "refractory_observation.py", "native_renewal_data.py",
    "population_prior_refit.py", "population_prior_objective.py",
])
def test_prior_budget_continuation_rejects_unknown_or_changed_numerical_sources(tmp_path, monkeypatch, source):
    parent, path, _ = _prior_budget_parent(tmp_path, monkeypatch)
    _replace_parent_sources(parent, {source: "0" * 64})
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="source"):
        training.train(tmp_path, tmp_path, path, output, 41, backend="population",
                       resume_from=parent, total_epochs=320, stop_after_epochs=1)
    assert not output.exists()


@pytest.mark.parametrize("missing", ["variational_rng", "prior_dynamics_rng", "posterior_cache"])
def test_prior_budget_continuation_requires_both_streams_and_complete_cache(tmp_path, monkeypatch, missing):
    parent, path, _ = _prior_budget_parent(tmp_path, monkeypatch)
    if missing == "posterior_cache":
        next((parent / "posterior_cache").glob("*.pt")).unlink()
    else:
        checkpoint = torch.load(parent / "latest.pt", weights_only=True)
        del checkpoint[missing]
        torch.save(checkpoint, parent / "latest.pt")
        report_path = parent / "run.json"
        report = json.loads(report_path.read_text())
        report["latest_checkpoint_sha256"] = sha256_file(parent / "latest.pt")
        report_path.write_text(json.dumps(report))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="RNG|posterior cache"):
        training.train(tmp_path, tmp_path, path, output, 41, backend="population",
                       resume_from=parent, total_epochs=320, stop_after_epochs=1)
    assert not output.exists()


def test_resumed_prior_budget_continuation_rejects_changed_cache(tmp_path, monkeypatch):
    parent, path, _ = _prior_budget_parent(tmp_path, monkeypatch)
    output = tmp_path / "continued"
    training.train(tmp_path, tmp_path, path, output, 41, backend="population",
                   resume_from=parent, total_epochs=320, stop_after_epochs=1)
    cache = next((output / "posterior_cache").glob("*.pt"))
    saved = torch.load(cache, weights_only=True)
    saved["mean"] += .125
    torch.save(saved, cache)
    with pytest.raises(ValueError, match="posterior cache changed"):
        training.train(tmp_path, tmp_path, path, output, 41, backend="population",
                       resume=True, stop_after_epochs=1)
