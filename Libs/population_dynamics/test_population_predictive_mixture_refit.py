"""Predictive-mixture refit wiring, exact recovery and fixed-cache reuse."""
import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from . import population_trajectory_training as population_training
from . import renewal_trajectory_training as training
from .population_prior_refit import MIXTURE_OVERRIDES, reuse_posterior_cache
from .population_prior_objective import PRIOR_PARAMETER_NAMES
from .provenance import sha256_file
from .test_population_prior_refit import CompleteClockFixture, make_parent, _replace_parent_sources
from .test_population_prior_campaign import _module
from .test_renewal_trajectory_training import _same


def mixture_parent(tmp_path):
    parent, config, path, original, cohort, constructor = make_parent(tmp_path)
    config.update(MIXTURE_OVERRIDES)
    path.write_text(yaml.safe_dump(config))
    return parent, config, path, original, cohort, constructor


@pytest.mark.parametrize("key,value", [("prediction_particles", 4), ("posterior_target_draws", 2),
                                        ("prior_selection", "training_loss")])
def test_mixture_profile_keeps_registered_counts_and_selection(tmp_path, key, value):
    _, config, _, _, _, _ = mixture_parent(tmp_path)
    training.validate_config(config, 41, backend="population")
    changed = dict(config, **{key: value})
    with pytest.raises(ValueError, match="configuration differs"):
        training.validate_config(changed, 41, backend="population")


def test_mixture_resume_preserves_three_streams_optimizer_and_frozen_parameters(tmp_path, monkeypatch):
    parent, config, path, original, _, _ = mixture_parent(tmp_path)
    monkeypatch.setattr(training, "TrainingCohort", CompleteClockFixture)
    CompleteClockFixture.reads = []
    full, split = tmp_path / "full", tmp_path / "split"
    invoke = lambda out, **kw: training.train(tmp_path, tmp_path, path, out, 41, backend="population", **kw)
    report = invoke(full, prior_parent=parent, stop_after_epochs=2)
    invoke(split, prior_parent=parent, stop_after_epochs=1)
    reads = list(CompleteClockFixture.reads)
    recovered = invoke(split, resume=True, stop_after_epochs=1)
    assert CompleteClockFixture.reads == reads
    assert report["training_objective"] == "fixed_posterior_joint_predictive_mixture_kl"
    assert report["selection_metric"] == "validation_joint_predictive_mixture_kl"
    assert report["prediction_particles"] == 8 and report["posterior_target_draws"] == 4
    assert recovered["resumed"] and recovered["frozen_parameters_unchanged_in_attempt"]
    a, b = [torch.load(out / "latest.pt", weights_only=True) for out in (full, split)]
    for key in ("model", "optimizer", "variational_rng", "prior_dynamics_rng", "prior_target_rng", "history", "best", "best_epoch"):
        _same(a[key], b[key])
    assert "prior_target_seed" in (split / "validation.csv").read_text().splitlines()[0]
    assert all(key.startswith("prior_mixture_kl_h") or key == "loss" for key in a["history"][0]["validation"])
    for name, value in original["model"].items():
        if name not in PRIOR_PARAMETER_NAMES:
            assert torch.equal(value, b["model"][name]), name
    assert any(not torch.equal(original["model"][name], b["model"][name]) for name in PRIOR_PARAMETER_NAMES)


def test_mixture_resume_requires_checkpointed_target_stream(tmp_path, monkeypatch):
    parent, _, path, _, _, _ = mixture_parent(tmp_path)
    monkeypatch.setattr(training, "TrainingCohort", CompleteClockFixture)
    output = tmp_path / "training"
    training.train(tmp_path, tmp_path, path, output, 41, backend="population", prior_parent=parent, stop_after_epochs=1)
    checkpoint = torch.load(output / "latest.pt", weights_only=True)
    del checkpoint["prior_target_rng"]
    torch.save(checkpoint, output / "latest.pt")
    report = json.loads((output / "run.json").read_text())
    report["latest_checkpoint_sha256"] = sha256_file(output / "latest.pt")
    (output / "run.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="target RNG"):
        training.train(tmp_path, tmp_path, path, output, 41, backend="population", resume=True, stop_after_epochs=1)


def test_real_trainer_reuses_authenticated_caches_without_encoding(tmp_path, monkeypatch):
    parent, config, path, _, _, _ = make_parent(tmp_path)
    monkeypatch.setattr(training, "TrainingCohort", CompleteClockFixture)
    source = tmp_path / "accepted_control"
    source.mkdir()
    output = source / "training"
    training.train(tmp_path, tmp_path, path, output, 41, backend="population", prior_parent=parent, stop_after_epochs=1)
    # A synthetic accepted receipt isolates cache intake from the production budget gate.
    report = json.loads((output / "run.json").read_text())
    report["status"] = "FITTED_DEVELOPMENT"
    report["refit_initialization"]["parent_training"] = "/retained/original/scratch/parent_training"
    (output / "run.json").write_text(json.dumps(report))
    names = ["training/run.json", *["training/posterior_cache/" + p.name for p in sorted((output / "posterior_cache").iterdir())]]
    files = [dict(path=name, bytes=(source / name).stat().st_size, sha256=sha256_file(source / name)) for name in names]
    (source / "delivery.json").write_text(json.dumps(dict(status="PASS", exit_code=0, training_seed=41, files=files)))
    config.update(MIXTURE_OVERRIDES)
    path.write_text(yaml.safe_dump(config))
    def no_encoding(self, case):
        raise AssertionError("accepted posterior cache was re-encoded")
    monkeypatch.setattr(CompleteClockFixture, "window", no_encoding)
    child = tmp_path / "mixture"
    result = training.train(tmp_path, tmp_path, path, child, 41, backend="population", prior_parent=parent,
                            prior_cache_source=source, stop_after_epochs=1)
    assert result["posterior_cache_reuse"]["reused_episodes"] == 3
    assert not result["posterior_cache_reuse"]["optimizer_or_prior_weights_reused"]
    assert result["frozen_parameters_unchanged_in_attempt"]
    for row in files[1:]:
        assert sha256_file(child / "posterior_cache" / Path(row["path"]).name) == row["sha256"]
    recovered = training.train(tmp_path, tmp_path, path, child, 41, backend="population", resume=True, stop_after_epochs=1)
    assert recovered["posterior_cache_reuse"] == result["posterior_cache_reuse"]
    continued = training.train(tmp_path, tmp_path, path, tmp_path / "continued_mixture", 41,
        backend="population", resume_from=child, total_epochs=320, stop_after_epochs=1)
    assert continued["posterior_cache_reuse"] == result["posterior_cache_reuse"]
    assert continued["refit_initialization"] == recovered["refit_initialization"]


def cache_delivery(tmp_path):
    source = tmp_path / "source"
    (source / "training/posterior_cache").mkdir(parents=True)
    initial = dict(mode="selected_parent_prior_refit", parent_checkpoint_sha256="a" * 64,
                   parent_training="/new/parent")
    report = dict(status="FITTED_DEVELOPMENT", binding=dict(seed=41, config=dict(schema="neural_population_prior_refit_v1")),
                  posterior_targets_fixed=True, frozen_parameters_unchanged_in_attempt=True,
                  refit_initialization=dict(initial, parent_training="/old/parent"))
    (source / "training/run.json").write_text(json.dumps(report))
    (source / "training/posterior_cache/case.pt").write_bytes(b"cache-payload")
    files = [dict(path=name, bytes=(source / name).stat().st_size, sha256=sha256_file(source / name))
             for name in ["training/run.json", "training/posterior_cache/case.pt"]]
    proof = dict(status="PASS", exit_code=0, training_seed=41, files=files)
    (source / "delivery.json").write_text(json.dumps(proof))
    output = tmp_path / "output"; output.mkdir()
    return source, output, initial, proof


@pytest.mark.parametrize("failure", ["payload", "case_set", "seed", "parent"])
def test_cache_reuse_rejects_changed_or_foreign_inputs(tmp_path, failure):
    source, output, initial, _ = cache_delivery(tmp_path)
    seed, cases = 41, ("case",)
    if failure == "payload": (source / "training/posterior_cache/case.pt").write_bytes(b"wrong-payload")
    elif failure == "case_set": cases = ("other",)
    elif failure == "seed": seed = 42
    else: initial["parent_checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="cache|parent"):
        reuse_posterior_cache(source, output, initial, seed, cases)


def test_population_wrapper_forwards_cache_source(tmp_path, monkeypatch):
    seen = {}
    def shared(*args, **kwargs):
        seen.update(kwargs); return {"status": "fixture"}
    monkeypatch.setattr(population_training.shared, "train", shared)
    population_training.train(tmp_path, tmp_path, tmp_path, tmp_path, 41,
                              prior_parent=tmp_path / "parent", prior_cache_source=tmp_path / "cache")
    assert seen["backend"] == "population"
    assert seen["prior_cache_source"] == tmp_path / "cache"


def test_preparation_uses_mixture_config_and_transitive_tests(tmp_path, monkeypatch):
    caller = _module("run_population_prior_campaign.py")
    repo = Path(__file__).resolve().parents[2]
    destination = tmp_path / "prepared"
    proof = caller.prepare(repo, destination, configuration=caller.MIXTURE_CONFIG)
    names = {row["path"] for row in proof["files"]}
    assert proof["configuration_path"] == caller.MIXTURE_CONFIG
    assert caller.MIXTURE_CONFIG in names
    assert "Libs/population_dynamics/test_population_state.py" in names
    assert "Libs/population_dynamics/test_population_predictive_mixture.py" in names
    assert "Libs/population_dynamics/population_predictive_mixture.py" in names
    monkeypatch.setenv("SLURMD_NODENAME", "cpu04")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    with pytest.raises(ValueError, match="accepted posterior caches"):
        caller.run(repo, destination, tmp_path / "never_started", 41, tmp_path / "parent")
    assert not (tmp_path / "never_started").exists()


def _mixture_budget_parent(tmp_path, monkeypatch):
    original, config, path, checkpoint, _, _ = mixture_parent(tmp_path)
    monkeypatch.setattr(training, "TrainingCohort", CompleteClockFixture)
    parent = tmp_path / "mixture_parent"
    training.train(tmp_path, tmp_path, path, parent, 41, backend="population",
                   prior_parent=original, stop_after_epochs=2)
    return parent, path, original, checkpoint


def test_mixture_budget_preserves_next_updates_three_streams_cache_and_original_parent(tmp_path, monkeypatch):
    from .poisson_trajectory_training import MIXTURE_BUDGET_TRAINER_SOURCES

    parent, path, original_path, original = _mixture_budget_parent(tmp_path, monkeypatch)
    # The bounded numerical reference keeps the same objective across four epochs.
    # The separate runner fixture checks admission of an 80-epoch parent to 320.
    full = tmp_path / "uninterrupted"
    invoke = lambda out, **kw: training.train(tmp_path, tmp_path, path, out, 41, backend="population", **kw)
    invoke(full, prior_parent=original_path, stop_after_epochs=4)
    reference = torch.load(full / "latest.pt", weights_only=True)
    _replace_parent_sources(parent, MIXTURE_BUDGET_TRAINER_SOURCES)
    committed = torch.load(parent / "latest.pt", weights_only=True)
    parent_files = {str(p.relative_to(parent)): sha256_file(p) for p in parent.rglob("*") if p.is_file()}
    boundaries = []
    save = training._save_checkpoint

    def capture_boundary(payload, destination):
        if destination.name == "latest.pt" and payload["next_epoch"] == committed["next_epoch"]:
            boundaries.append(copy.deepcopy(payload))
        save(payload, destination)

    def no_encoding(self, case):
        raise AssertionError("mixture continuation re-encoded a fixed posterior cache")

    monkeypatch.setattr(training, "_save_checkpoint", capture_boundary)
    monkeypatch.setattr(CompleteClockFixture, "window", no_encoding)
    whole, split = tmp_path / "whole", tmp_path / "split"
    report = invoke(whole, resume_from=parent, total_epochs=320, stop_after_epochs=2)
    first = invoke(split, resume_from=parent, total_epochs=320, stop_after_epochs=1)
    recovered = invoke(split, resume=True, stop_after_epochs=1)
    state_keys = ("model", "optimizer", "variational_rng", "prior_dynamics_rng", "prior_target_rng",
                  "history", "best", "stale", "best_epoch", "initialization")
    assert len(boundaries) == 2
    for boundary in boundaries:
        for key in state_keys:
            _same(boundary[key], committed[key])
    assert first["completed_epochs"] == committed["next_epoch"] + 1
    for output in (whole, split):
        current = torch.load(output / "latest.pt", weights_only=True)
        for key in state_keys:
            _same(current[key], reference[key])
        assert current["next_epoch"] == committed["next_epoch"] + 2
        assert current["binding"]["config"]["maximum_epochs"] == 80
        assert current["binding"]["config"]["early_stopping_patience"] == 12
        assert current["effective_maximum_epochs"] == 320
        assert current["continuation"]["inherited_stale_epochs"] == committed["stale"]
        assert current["continuation"]["history_prefix_epochs"] == len(committed["history"])
        for key in ("prior_dynamics_rng", "prior_target_rng"):
            assert current["continuation"][key] == "explicit_checkpointed_generator"
        migration = current["continuation"]["source_migration"]
        assert set(migration) == set(MIXTURE_BUDGET_TRAINER_SOURCES)
        for source, digest in MIXTURE_BUDGET_TRAINER_SOURCES.items():
            assert migration[source] == dict(parent=digest, current=current["trainer_sources"][source])
        for name, digest in parent_files.items():
            if name.startswith("posterior_cache/"):
                assert sha256_file(output / name) == digest
        for name, value in original["model"].items():
            if name not in PRIOR_PARAMETER_NAMES:
                assert torch.equal(value, current["model"][name]), name
    assert report["refit_initialization"] == recovered["refit_initialization"] == committed["initialization"]
    assert report["training_objective"] == "fixed_posterior_joint_predictive_mixture_kl"
    assert recovered["selection_metric"] == "validation_joint_predictive_mixture_kl"
    assert report["prediction_particles"] == 8 and report["posterior_target_draws"] == 4
    assert {str(p.relative_to(parent)): sha256_file(p) for p in parent.rglob("*") if p.is_file()} == parent_files


@pytest.mark.parametrize("missing", ["variational_rng", "prior_dynamics_rng", "prior_target_rng"])
def test_mixture_budget_requires_all_three_checkpointed_streams(tmp_path, monkeypatch, missing):
    parent, path, _, _ = _mixture_budget_parent(tmp_path, monkeypatch)
    checkpoint = torch.load(parent / "latest.pt", weights_only=True)
    del checkpoint[missing]
    torch.save(checkpoint, parent / "latest.pt")
    report_path = parent / "run.json"
    report = json.loads(report_path.read_text())
    report["latest_checkpoint_sha256"] = sha256_file(parent / "latest.pt")
    report_path.write_text(json.dumps(report))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="RNG"):
        training.train(tmp_path, tmp_path, path, output, 41, backend="population",
                       resume_from=parent, total_epochs=320, stop_after_epochs=1)
    assert not output.exists()


@pytest.mark.parametrize("source", ["population_predictive_mixture.py", "population_prior_objective.py",
                                    "renewal_trajectory_training.py"])
def test_mixture_budget_rejects_changed_numerical_or_unregistered_trainer_sources(tmp_path, monkeypatch, source):
    parent, path, _, _ = _mixture_budget_parent(tmp_path, monkeypatch)
    _replace_parent_sources(parent, {source: "0" * 64})
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="source"):
        training.train(tmp_path, tmp_path, path, output, 41, backend="population",
                       resume_from=parent, total_epochs=320, stop_after_epochs=1)
    assert not output.exists()


def test_mixture_budget_requires_its_numerical_source_and_rejects_analytic_ancestor(tmp_path, monkeypatch):
    from .poisson_trajectory_training import PRIOR_BUDGET_TRAINER_SOURCES, validate_prior_budget_binding

    parent, path, _, _ = _mixture_budget_parent(tmp_path, monkeypatch)
    checkpoint = torch.load(parent / "latest.pt", weights_only=True)
    sources = dict(checkpoint["trainer_sources"])
    del sources["population_predictive_mixture.py"]
    with pytest.raises(ValueError, match="numerical source set"):
        validate_prior_budget_binding(checkpoint["binding"], checkpoint["binding"], sources, sources)
    _replace_parent_sources(parent, PRIOR_BUDGET_TRAINER_SOURCES)
    with pytest.raises(ValueError, match="source"):
        training.train(tmp_path, tmp_path, path, tmp_path / "rejected", 41, backend="population",
                       resume_from=parent, total_epochs=320, stop_after_epochs=1)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("failure", ["objective", "selection", "particles", "target_draws", "budget"])
def test_mixture_budget_rejects_changed_profile_and_nonincreasing_budget(tmp_path, monkeypatch, failure):
    parent, path, _, _ = _mixture_budget_parent(tmp_path, monkeypatch)
    total_epochs = 320
    if failure == "budget":
        total_epochs = 80
    elif failure in ("particles", "target_draws"):
        config = yaml.safe_load(path.read_text())
        config["prediction_particles" if failure == "particles" else "posterior_target_draws"] = 2
        path.write_text(yaml.safe_dump(config))
    else:
        report_path = parent / "run.json"
        report = json.loads(report_path.read_text())
        report["training_objective" if failure == "objective" else "selection_metric"] = "analytic_kl_substitution"
        report_path.write_text(json.dumps(report))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="objective|configuration differs|total-epochs"):
        training.train(tmp_path, tmp_path, path, output, 41, backend="population",
                       resume_from=parent, total_epochs=total_epochs, stop_after_epochs=1)
    assert not output.exists()


def test_mixture_budget_at_epoch80_keeps_exhausted_patience_and_selected_history(tmp_path, monkeypatch):
    parent, path, _, _ = _mixture_budget_parent(tmp_path, monkeypatch)
    latest = torch.load(parent / "latest.pt", weights_only=True)
    selected = torch.load(parent / "best.pt", weights_only=True)
    # This synthetic history tests bookkeeping at epoch 80; it is not training evidence.
    history = []
    for epoch in range(80):
        row = copy.deepcopy(latest["history"][-1])
        row["epoch"] = epoch
        row["validation"]["loss"] = latest["best"] + (0.0 if epoch == 67 else 1.0)
        history.append(row)
    latest.update(next_epoch=80, history=history, best_epoch=67, stale=12)
    selected.update(next_epoch=68, history=history[:68], best=latest["best"], best_epoch=67, stale=0)
    torch.save(latest, parent / "latest.pt")
    torch.save(selected, parent / "best.pt")
    (parent / "epochs.json").write_text(json.dumps(history))
    report_path = parent / "run.json"
    report = json.loads(report_path.read_text())
    report.update(status="FITTED_DEVELOPMENT", completed_epochs=80, best_epoch=67, stale_epochs=12,
                  best_validation_prior_loss=latest["best"], latest_checkpoint_sha256=sha256_file(parent / "latest.pt"),
                  best_checkpoint_sha256=sha256_file(parent / "best.pt"))
    report_path.write_text(json.dumps(report))
    output = tmp_path / "continued"
    result = training.train(tmp_path, tmp_path, path, output, 41, backend="population",
                            resume_from=parent, total_epochs=320, stop_after_epochs=1)
    current = torch.load(output / "latest.pt", weights_only=True)
    assert result["status"] == "FITTED_DEVELOPMENT" and result["stop_reason"] == "validation_patience"
    assert result["completed_epochs"] == 80 and result["effective_maximum_epochs"] == 320
    assert result["continuation"]["inherited_early_stopping_patience"] == 12
    assert result["continuation"]["inherited_stale_epochs"] == 12
    for key in ("model", "optimizer", "variational_rng", "prior_dynamics_rng", "prior_target_rng", "history"):
        _same(current[key], latest[key])
    saved_best = torch.load(output / "best.pt", weights_only=True)
    assert saved_best["next_epoch"] == 68 and saved_best["history"] == history[:68]
