"""MemNO observation identities, transition timing and epoch continuation."""
import copy
import json

import numpy as np
import pytest
import torch
import yaml

from . import memory_field_training as training
from .memory_field_cohort import fitting_scale, grid_contract
from .memory_field_evaluation import forecast
from .memory_neural_operator import MemoryNeuralOperator


def test_source_order_and_fit_only_scale():
    ids = [json.dumps([p, y, x]) for p in ("E", "I") for y in range(2) for x in range(2)]
    assert grid_contract(ids) == (["E", "I"], (2, 2))
    with pytest.raises(ValueError, match="source population-major"):
        grid_contract(ids[::-1])
    a = np.arange(32, dtype=np.float32).reshape(4, 2, 2, 2)
    b = 2 * a
    expected = ((a.mean(axis=0) + b.mean(axis=0)) / 2).astype(np.float32)
    np.testing.assert_array_equal(fitting_scale([a, b]), expected)
    np.testing.assert_array_equal(fitting_scale([np.zeros_like(a)]), np.ones_like(expected))


class RecordingModel:
    def __init__(self):
        self.calls = []
    def eval(self):
        return self
    def reset_state(self, batch):
        self.calls = []
        return None
    def one_step(self, field, control, state):
        self.calls.append((field.clone(), control.clone()))
        return field + 100, state


def test_target_bin_controls_prefix_boundary_and_free_rollout():
    field = torch.arange(8.).reshape(1, 8, 1, 1, 1)
    control = torch.arange(8.).reshape(1, 8, 1)
    inputs, paired, target = training.teacher_pairs(field, control)
    torch.testing.assert_close(inputs, field[:, :-1])
    torch.testing.assert_close(paired, control[:, 1:])
    torch.testing.assert_close(target, field[:, 1:])
    model = RecordingModel()
    result = forecast(model, field[:, :3], control[:, :6], 3)
    assert [float(row[1].item()) for row in model.calls] == [1., 2., 3., 4., 5.]
    assert [float(row[0].item()) for row in model.calls] == [0., 1., 2., 102., 202.]
    torch.testing.assert_close(result.flatten(), torch.tensor([102., 202., 302.]))


def test_forecast_prefix_streaming_equals_batch_first_unseen_bin():
    torch.manual_seed(41)
    torch.set_num_threads(1)
    model = MemoryNeuralOperator(2, (2, 2), 1, .005)
    field, control = torch.rand(1, 8, 2, 2, 2), torch.rand(1, 8, 1)
    with torch.no_grad():
        expected = model(field[:, :4], control[:, 1:5])[:, -1:]
        actual = forecast(model, field[:, :4], control[:, :5], 1)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


class TinyCohort:
    def __init__(self, acquisition, fields, config):
        self.fit = [dict(case_id="fit", block=0)]
        self.validation = [dict(case_id="selection", block=14)]
        self.populations, self.grid_shape, self.bin_seconds = ["E"], (2, 2), .005
        self.scale = np.ones((1, 2, 2), dtype=np.float32)
    def field_binding(self):
        return dict(fixture="complete short source-grid episodes")
    def window(self, case):
        values = np.arange(32, dtype=np.float32).reshape(8, 1, 2, 2) / 32
        return values, np.zeros((8, 1), dtype=np.float32)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = copy.deepcopy(training.FIXED_PROTOCOL)
    config.update(device="cpu", maximum_epochs=2, cpu_threads=1, batch_size=1)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(training, "validate_config", lambda config, seed: None)
    monkeypatch.setattr(training, "MemoryFieldCohort", TinyCohort)
    return path


def test_committed_continuation_matches_uninterrupted_training(tmp_path, setup):
    training.train(tmp_path, tmp_path, setup, tmp_path / "whole", 41)
    training.train(tmp_path, tmp_path, setup, tmp_path / "split", 41, stop_after_epochs=1)
    training.train(tmp_path, tmp_path, setup, tmp_path / "split", 41, resume=True)
    states = []
    for name in ("whole", "split"):
        root = tmp_path / name
        pointer = json.loads((root / "committed.json").read_text())
        states.append(torch.load(root / pointer["checkpoint"], weights_only=False))
    assert states[0]["history"] == states[1]["history"]
    for name, left in states[0]["model"].items():
        right = states[1]["model"][name]
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert left == right
    assert states[0]["scheduler"] == states[1]["scheduler"]
    for key in states[0]["optimizer"]["state"]:
        for name, left in states[0]["optimizer"]["state"][key].items():
            torch.testing.assert_close(left, states[1]["optimizer"]["state"][key][name], rtol=0, atol=0)


def test_corrupt_checkpoint_and_changed_binding_fail(tmp_path, setup):
    root = tmp_path / "run"
    training.train(tmp_path, tmp_path, setup, root, 42, stop_after_epochs=1)
    run = json.loads((root / "run.json").read_text())
    changed = copy.deepcopy(run["binding"])
    changed["training_seed"] = 43
    with pytest.raises(ValueError, match="source, config or data changed"):
        training.load_committed(root, changed, "cpu")
    with (root / run["latest_checkpoint"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        training.load_committed(root, run["binding"], "cpu")
