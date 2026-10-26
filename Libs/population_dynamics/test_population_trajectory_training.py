"""Exercise the population backend through the actual complete-episode trainer.

Only the production cohort boundary is replaced by the existing tiny fixture.
The optimizer, loss, checkpoint, validation and continuation code are unchanged.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from . import population_trajectory_training as training
from . import renewal_trajectory_training as shared
from .population_state import ControlledPopulationState
from .poisson_trajectory_training import _trainer_sources
from .provenance import sha256_file
from .test_poisson_trajectory_training import snapshot
from .test_renewal_trajectory_training import TinyCohort, _checkpoint, _same


@pytest.fixture
def population_fixture(tmp_path, monkeypatch):
    config = copy.deepcopy(training.FIXED_BUDGET)
    config.update(training_variational_seed_base=61001, validation_seed_base=64001,
                  device="cpu", cpu_threads=1)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(shared, "TrainingCohort", TinyCohort)
    return config, path


def restored_loss(checkpoint):
    model = ControlledPopulationState(**checkpoint["model_configuration"])
    model.load_state_dict(checkpoint["model"])
    generator = torch.Generator().set_state(checkpoint["variational_rng"])
    cohort = TinyCohort(None, None, None)
    inputs = shared._inputs(cohort.window(cohort.fit[0]), shared.model_device(model, "population"))
    return shared._numbers(model.sequence_loss(*inputs, generator))


@pytest.mark.parametrize("representation", ["fixed", "recurrent"])
def test_population_fit_logs_source_map_and_fit_only_hazard(tmp_path, population_fixture, monkeypatch, representation):
    values, config = population_fixture
    values["representation"] = representation
    config.write_text(yaml.safe_dump(values))
    captured = {}
    original = shared.initialize_observation
    def capture(model, cohort, pseudocount, *, backend):
        result = original(model, cohort, pseudocount, backend=backend)
        captured["bias"] = shared.hazard_bias(model, backend).detach().numpy().copy()
        captured["report"] = result
        captured["initial"] = {key: value.clone() for key, value in model.state_dict().items()}
        return result
    monkeypatch.setattr(shared, "initialize_observation", capture)
    output = tmp_path / "fit"
    report = training.train(tmp_path, tmp_path, config, output, 41, stop_after_epochs=1)
    assert report["model_family"] == "population" and report["representation"] == representation
    assert report["status"] == "CHECKPOINTED_DEVELOPMENT" and report["completed_epochs"] == 1
    assert report["complete_episode_updates"] and not report["scientific_claim_qualified"]
    assert not report["calibration_outcomes_accessed"] and not report["test_outcomes_accessed"]
    checkpoint = _checkpoint(output)
    constructor = checkpoint["model_configuration"]
    assert constructor["unit_ids"] == [3, 7, 12]
    assert constructor["unit_field_index"] == [0, 1, 0]
    assert constructor["field_names"] == ["E_0", "I_0"]
    assert constructor["minimum_interval_ticks"] == [21, 21, 21]
    assert constructor["latent_channels"] == 4 and constructor["encoder_size"] == 16
    assert constructor["unit_chunk_size"] == 4096 and constructor["tick_seconds"] == .0001
    assert any(not torch.equal(captured["initial"][key], value) for key, value in checkpoint["model"].items())
    cohort = TinyCohort(None, None, values)
    windows = [cohort.window(case) for case in cohort.fit]
    successes = sum(window.counts.sum(0) for window in windows)
    eligible = sum(window.eligible_ticks.sum(0) for window in windows)
    expected = np.log(-np.log1p(-(successes + .5) / (eligible + 1)) / .0001)
    np.testing.assert_allclose(captured["bias"], expected, rtol=1e-6)
    assert captured["report"]["fitting_episodes"] == 2
    for name, expected_rows in (("optimization.csv", 2), ("validation.csv", 4)):
        with (output / name).open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == expected_rows
        assert all(np.isfinite(float(row[key])) for row in rows for key in shared.LOSS_TERMS)


def test_population_split_resume_restores_optimizer_rng_and_next_loss(tmp_path, population_fixture):
    _, config = population_fixture
    continuous, split = tmp_path / "continuous", tmp_path / "split"
    training.train(tmp_path, tmp_path, config, continuous, 42, stop_after_epochs=3)
    training.train(tmp_path, tmp_path, config, split, 42, stop_after_epochs=1)
    report = training.train(tmp_path, tmp_path, config, split, 42, resume=True, stop_after_epochs=2)
    left, right = _checkpoint(continuous), _checkpoint(split)
    for key in ("model", "optimizer", "variational_rng", "history", "next_epoch", "best", "stale", "best_epoch"):
        _same(left[key], right[key])
    assert restored_loss(left) == restored_loss(right)
    assert report["resumed"] and right["attempt"] == 1
    assert report["fitting_episodes"] == 2 and report["validation_episodes"] == 1


def test_population_validation_rng_is_fixed_and_isolated(population_fixture):
    config, _ = population_fixture
    cohort = TinyCohort(None, None, config)
    model = ControlledPopulationState(**shared.model_configuration(cohort, config, 41, "population"))
    training_rng = torch.Generator().manual_seed(992)
    train_state, global_state = training_rng.get_state().clone(), torch.get_rng_state().clone()
    first = shared.evaluate_validation(model, cohort, config, 41)
    second = shared.evaluate_validation(model, cohort, config, 41)
    assert first == second
    assert torch.equal(training_rng.get_state(), train_state)
    assert torch.equal(torch.get_rng_state(), global_state)


@pytest.mark.parametrize("changed", ["config", "field_map", "field_names"])
def test_population_resume_changed_configuration_or_map_has_no_output_writes(tmp_path, population_fixture, monkeypatch, changed):
    values, config = population_fixture
    output = tmp_path / "fit"
    training.train(tmp_path, tmp_path, config, output, 41, stop_after_epochs=1)
    before = snapshot(output)
    if changed == "config":
        values["validation_seed_base"] += 1
        config.write_text(yaml.safe_dump(values))
    else:
        class DifferentMap(TinyCohort):
            def __init__(self, *args):
                super().__init__(*args)
                if changed == "field_map":
                    self.unit_field_index = np.array([1, 0, 1])
                else:
                    self.field_ids = np.array(["renamed", "I_0"])
        monkeypatch.setattr(shared, "TrainingCohort", DifferentMap)
    with pytest.raises(ValueError, match="binding changed|constructor changed"):
        training.train(tmp_path, tmp_path, config, output, 41, resume=True)
    assert snapshot(output) == before


def test_population_continuation_preserves_parent_state_and_explicit_rng(tmp_path, population_fixture):
    _, config = population_fixture
    parent = tmp_path / "parent"
    training.train(tmp_path, tmp_path, config, parent, 43, stop_after_epochs=2)
    before = snapshot(parent)
    whole, divided = tmp_path / "whole", tmp_path / "divided"
    training.train(tmp_path, tmp_path, config, whole, 43, resume_from=parent, total_epochs=21, stop_after_epochs=2)
    training.train(tmp_path, tmp_path, config, divided, 43, resume_from=parent, total_epochs=21, stop_after_epochs=1)
    report = training.train(tmp_path, tmp_path, config, divided, 43, resume=True, stop_after_epochs=1)
    left, right = _checkpoint(whole), _checkpoint(divided)
    for key in ("model", "optimizer", "variational_rng", "history", "next_epoch", "best", "stale", "best_epoch"):
        _same(left[key], right[key])
    assert restored_loss(left) == restored_loss(right)
    assert report["continuation"]["training_rng"] == "explicit_checkpointed_variational_generator"
    assert report["effective_maximum_epochs"] == 21 and report["binding"]["config"]["maximum_epochs"] == 20
    assert left["history"][:2] == _checkpoint(parent)["history"]
    assert snapshot(parent) == before


@pytest.mark.parametrize("missing", ["variational_rng", "trainer_sources"])
def test_population_continuation_requires_rng_and_recorded_sources(tmp_path, population_fixture, missing):
    _, config = population_fixture
    parent = tmp_path / "parent"
    training.train(tmp_path, tmp_path, config, parent, 41, stop_after_epochs=1)
    checkpoint = _checkpoint(parent)
    checkpoint.pop(missing)
    torch.save(checkpoint, parent / "latest.pt")
    report = json.loads((parent / "run.json").read_text())
    report["latest_checkpoint_sha256"] = sha256_file(parent / "latest.pt")
    (parent / "run.json").write_text(json.dumps(report))
    before = snapshot(parent)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="RNG state unavailable|checkpoint-recorded"):
        training.train(tmp_path, tmp_path, config, output, 41, resume_from=parent, total_epochs=21)
    assert not output.exists() and snapshot(parent) == before


def test_population_source_set_and_frozen_protocol_remain_explicit(population_fixture):
    config, _ = population_fixture
    sources = _trainer_sources("population")
    assert {"population_state.py", "population_trajectory_training.py", "renewal_trajectory_training.py",
            "poisson_trajectory_training.py", "native_renewal_data.py", "native_count_windows.py",
            "refractory_observation.py"} <= sources.keys()
    for key, bad in (("maximum_epochs", 1), ("latent_channels", 2), ("encoder_size", 32),
                     ("training_seeds", [41]), ("model_family", "renewal"),
                     ("expected_poisson_rng_family", "per_edge_xorwow")):
        with pytest.raises(ValueError):
            training.validate_config(dict(config, **{key: bad}), 41)
    with pytest.raises(ValueError, match="backend"):
        shared.validate_config(config, 41, backend="unknown")
    with pytest.raises(ValueError, match="model family"):
        shared.validate_config(config, 41)


def test_population_cli_has_existing_resume_options_and_rejects_unregistered_seed():
    command = [sys.executable, "-m", "Libs.population_dynamics.population_trajectory_training"]
    result = subprocess.run(command + ["--help"], text=True, capture_output=True)
    assert result.returncode == 0 and "population" in result.stdout
    assert "--resume-from" in result.stdout and "--stop-after-epochs" in result.stdout
    assert subprocess.run(command + ["--seed", "44"], text=True, capture_output=True).returncode != 0
