"""Causal target, fitting-only scale and complete two-stage resume checks."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from . import latent_spike_diffusion_training as training


class TinyCohort:
    def __init__(self, acquisition, fields, config):
        self.fit = [dict(case_id="fit_a", offset=0), dict(case_id="fit_b", offset=1)]
        self.validation = [dict(case_id="selection", offset=9)]
        self.unit_ids = np.arange(8)
        self.binding = dict(fixture="complete identified tiny episodes")

    def window(self, case):
        counts = (np.arange(32 * 8).reshape(32, 8) + case["offset"]) % 3
        control = np.linspace(.1, .5, 32, dtype=np.float32)[:, None]
        return counts, control


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = copy.deepcopy(training.FIXED_PROTOCOL)
    config.update(device="cpu", cpu_threads=1, complete_episode_bins=32, forecast_origins_bins=[8, 16, 24], forecast_bins=8,
        maximum_autoencoder_epochs=2, maximum_diffusion_epochs=2,
        autoencoder_validation_every=1, diffusion_validation_every=1, validation_draws=2,
        autoencoder_warmup_epochs=0, diffusion_warmup_epochs=0,
        model=dict(latent_channels=8, hidden_channels=32, encoder_blocks=1, denoiser_blocks=1, diffusion_steps=8))
    path = tmp_path / "config.yaml"; path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(training, "validate_config", lambda config, seed: None)
    monkeypatch.setattr(training, "TrajectoryCohort", TinyCohort)
    return config, path


def _payload(path):
    pointer = json.loads((path / "committed.json").read_text())
    return torch.load(path / pointer["checkpoint"], weights_only=False)


def _equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first: _equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second): _equal(a, b)
    else:
        assert first == second


def test_resume_across_both_stages_matches_model_optimizer_scheduler_ema_rng_history(tmp_path, setup):
    _, config_path = setup
    whole, split = tmp_path / "whole", tmp_path / "split"
    training.train(tmp_path, tmp_path, config_path, whole, 41)
    training.train(tmp_path, tmp_path, config_path, split, 41, stop_after_epochs=1)
    training.train(tmp_path, tmp_path, config_path, split, 41, resume=True, stop_after_epochs=2)
    training.train(tmp_path, tmp_path, config_path, split, 41, resume=True)
    a, b = _payload(whole), _payload(split)
    for key in ("model", "optimizer", "scheduler", "ema", "rng", "global_rng"):
        _equal(a[key], b[key])
    assert a["state"]["stage"] == b["state"]["stage"] == "complete"
    _equal(a["state"]["history"], b["state"]["history"])
    assert len(a["state"]["history"]) == 4


def test_fitting_scale_and_cache_match_separate_target_and_prefix_transforms(setup, tmp_path):
    config, _ = setup
    cohort = TinyCohort(tmp_path, tmp_path, config)
    torch.manual_seed(41); model = training._model(cohort, config)
    cache = training.build_latent_cache(model, cohort, config)
    raw = []
    for case in cohort.fit:
        counts, controls = cohort.window(case)
        for _, _, target, _ in training._window_pairs(torch.tensor(counts)[None].float(),
                                                     torch.tensor(controls)[None], config):
            raw.append(model.autoencoder.encode(model._counts(target)).detach())
    all_targets = torch.cat(raw, dim=2)
    torch.testing.assert_close(model.latent_mean, all_targets.mean((0, 2), keepdim=True))
    torch.testing.assert_close(model.latent_sd, all_targets.std((0, 2), correction=0, keepdim=True))
    assert cache["moments"]["fit_cases"] == ["fit_a", "fit_b"]
    assert cache["moments"]["fit_windows"] == 6
    for row in cache["rows"]["validation"]:
        counts, _ = cohort.window(cohort.validation[0]); counts = torch.tensor(counts)[None].float()
        torch.testing.assert_close(row["prefix"], model.infer_state(counts[:, :row["origin"]]).latent)
        torch.testing.assert_close(row["target"], model.encode_target(counts[:, row["origin"]:row["origin"]+8]))


def test_validation_is_repeatable_and_preserves_training_stream_and_parameters(setup, tmp_path):
    config, _ = setup; cohort = TinyCohort(tmp_path, tmp_path, config)
    model = training._model(cohort, config); rng = torch.Generator().manual_seed(123)
    cache = training.build_latent_cache(model, cohort, config)
    before = rng.get_state().clone(); weights = copy.deepcopy(model.state_dict())
    ae1 = training._ae_epoch(model, cohort, config, 41, 0, rng)
    ae2 = training._ae_epoch(model, cohort, config, 41, 1, rng)
    assert ae1 == ae2
    first, _ = training._diffusion_epoch(model, cache, config, 41, rng)
    second, _ = training._diffusion_epoch(model, cache, config, 41, rng)
    assert first == second; assert torch.equal(before, rng.get_state())
    _equal(weights, model.state_dict())


def test_interrupted_epoch_preserves_commit_and_resumes_without_losing_failure(setup, tmp_path, monkeypatch):
    _, config = setup; root = tmp_path / "interrupted"
    original = TinyCohort.window
    def failed(self, case):
        if case["case_id"] == "fit_b": raise RuntimeError("fixture interrupted epoch")
        return original(self, case)
    monkeypatch.setattr(TinyCohort, "window", failed)
    with pytest.raises(RuntimeError, match="interrupted epoch"):
        training.train(tmp_path, tmp_path, config, root, 43)
    initial = _payload(root); assert initial["state"]["next_epoch"] == 0
    assert list(root.glob("failure_*.json"))
    monkeypatch.setattr(TinyCohort, "window", original)
    training.train(tmp_path, tmp_path, config, root, 43, resume=True)
    assert _payload(root)["state"]["stage"] == "complete"
    assert list(root.glob("failure_*.json"))


def test_resume_rejects_changed_binding_and_checkpoint(setup, tmp_path):
    _, config = setup; root = tmp_path / "run"
    training.train(tmp_path, tmp_path, config, root, 42, stop_after_epochs=1)
    with pytest.raises(ValueError, match="resume source/data/protocol"):
        training.train(tmp_path, tmp_path, config, root, 43, resume=True)
    pointer = json.loads((root / "committed.json").read_text())
    (root / pointer["checkpoint"]).write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError, match="checkpoint changed"):
        training.train(tmp_path, tmp_path, config, root, 42, resume=True)
