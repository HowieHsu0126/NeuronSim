"""Committed BRAID epoch and complete native count-cohort contracts."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf
import yaml

from . import braid_trajectory_training as training
from .braid import BRAIDMeanPredictor


class TinyCohort:
    def __init__(self, acquisition, fields, config):
        self.fit = [dict(case_id=f"fit_{i}", block=i) for i in range(2)]
        self.validation = [dict(case_id="validation", block=2)]
        self.unit_ids = np.array([1, 2])
        self.binding = dict(fixture="identified complete tiny episodes")

    def window(self, case):
        counts = (np.arange(22).reshape(11, 2) + case["block"]) % 5
        controls = np.linspace(-.2, .3, 11, dtype=np.float32)[:, None]
        return counts, controls


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    config = copy.deepcopy(training.FIXED_PROTOCOL)
    config.update(device="cpu", history_bins=5, horizon_bins=6, maximum_epochs=3,
                  model=dict(latent_dim=3, hidden_units=5, steps_ahead=[1, 2, 4],
                             learning_rate=.001, gradient_clip_norm=1.))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(training, "validate_config", lambda config, seed: None)
    monkeypatch.setattr(training, "TrajectoryCohort", TinyCohort)
    return config, path


def test_committed_resume_matches_uninterrupted_model_adam_and_epoch_history(tmp_path, setup):
    config, path = setup
    whole, split = tmp_path / "whole", tmp_path / "split"
    training.train(tmp_path, tmp_path, path, whole, 41)
    training.train(tmp_path, tmp_path, path, split, 41, stop_after_epochs=1)
    training.train(tmp_path, tmp_path, path, split, 41, resume=True)
    a = json.loads((whole / "committed.json").read_text())
    b = json.loads((split / "committed.json").read_text())
    x = BRAIDMeanPredictor.load(whole / a["checkpoint"])
    y = BRAIDMeanPredictor.load(split / b["checkpoint"])
    assert x.training_steps == y.training_steps == 6
    for left, right in zip(x.native.model.weights, y.native.model.weights):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    for left, right in zip(x.native.model.optimizer.weights, y.native.model.optimizer.weights):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    assert json.loads((whole / "epochs.json").read_text()) == json.loads((split / "epochs.json").read_text())


def test_validation_leaves_model_and_optimizer_unchanged(setup, tmp_path):
    config, _ = setup
    cohort = TinyCohort(tmp_path, tmp_path, config)
    model = BRAIDMeanPredictor((1, 2), 1, 5, 6, .005, config["model"], initialization_seed=42)
    counts, controls = cohort.window(cohort.fit[0])
    training._losses(model, counts, controls, training=True)
    weights = [v.numpy().copy() for v in model.native.model.weights]
    adam = [v.numpy().copy() for v in model.native.model.optimizer.weights]
    training._evaluate(model, cohort, cohort.validation, training=False)
    assert model.training_steps == 1
    for old, new in zip(weights, model.native.model.weights): np.testing.assert_array_equal(old, new.numpy())
    for old, new in zip(adam, model.native.model.optimizer.weights): np.testing.assert_array_equal(old, new.numpy())


def test_partial_epoch_rolls_back_to_initial_commit(tmp_path, setup, monkeypatch):
    config, path = setup
    root = tmp_path / "interrupted"
    original = TinyCohort.window
    calls = 0
    def interrupted(self, case):
        nonlocal calls
        calls += 1
        if calls == 2: raise RuntimeError("fixture interruption after one update")
        return original(self, case)
    monkeypatch.setattr(TinyCohort, "window", interrupted)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        training.train(tmp_path, tmp_path, path, root, 43)
    pointer = json.loads((root / "committed.json").read_text())
    model = BRAIDMeanPredictor.load(root / pointer["checkpoint"])
    assert model.training_steps == 0
    assert (root / "failure.json").exists()
    monkeypatch.setattr(TinyCohort, "window", original)
    training.train(tmp_path, tmp_path, path, root, 43, resume=True, stop_after_epochs=1)
    assert json.loads((root / "run.json").read_text())["training_steps"] == 2


def test_changed_data_binding_rejected_before_update(tmp_path, setup, monkeypatch):
    config, path = setup
    root = tmp_path / "binding"
    training.train(tmp_path, tmp_path, path, root, 41, stop_after_epochs=1)
    original = TinyCohort.__init__
    def changed(self, *args):
        original(self, *args)
        self.binding["fixture"] = "changed"
    monkeypatch.setattr(TinyCohort, "__init__", changed)
    before = (root / "committed.json").read_bytes()
    with pytest.raises(ValueError, match="binding changed"):
        training.train(tmp_path, tmp_path, path, root, 41, resume=True)
    assert (root / "committed.json").read_bytes() == before


def test_registered_protocol_rejects_seed_units_clock_or_budget(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    config = copy.deepcopy(training.FIXED_PROTOCOL); config["device"] = "cpu"
    training.validate_config(config, 41)
    for key, value in [("expected_units", 2), ("duration_ms", 100.), ("maximum_epochs", 1), ("training_seeds", [41])]:
        with pytest.raises(ValueError): training.validate_config({**config, key: value}, 41)
    with pytest.raises(ValueError): training.validate_config(config, 44)


def test_invalid_cli_seed_returns_nonzero_without_output(tmp_path):
    command = [sys.executable, "-B", "-m", "Libs.population_dynamics.braid_trajectory_training",
               "--acquisition", str(tmp_path), "--fields", str(tmp_path), "--config", str(tmp_path / "missing.yaml"),
               "--output", str(tmp_path / "output"), "--seed", "44"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert not (tmp_path / "output").exists()
