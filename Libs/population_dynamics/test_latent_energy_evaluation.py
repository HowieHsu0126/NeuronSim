"""Full EAG score denominator, observed-only prefix and shared replay checks."""
import csv
import json

import numpy as np
import pytest
import torch

from . import latent_energy_evaluation as evaluation
from .test_latent_spike_diffusion_evaluation import Cohort, Native, Model as BaseModel


class Model(BaseModel):
    def infer_state(self, counts, controls):
        assert counts.shape[:2] == controls.shape[:2]
        assert counts.shape[1] in (200, 600, 1000)
        return type("State", (), dict(latent=counts[..., :1].clone()))()

    def simulate(self, state, stimulus, *, steps, guidance, temperature, **streams):
        assert (steps, guidance, temperature) == (50, 4.0, 0.7)
        assert stimulus.shape == (1, 100, 1)
        return super().simulate(state, stimulus, **streams)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    for name in ("config.yaml", "scale.npz"):
        (tmp_path / name).write_bytes(b"bound fixture")
    (tmp_path / "training").mkdir()
    (tmp_path / "training/committed.json").write_text("{}")
    cohort = Cohort()
    config = dict(device="cpu", forecast_origins_bins=[200, 600, 1000], forecast_bins=100, bin_ms=5.,
                  sampling=dict(steps=50, guidance=4.0, temperature=.7))
    monkeypatch.setattr(evaluation, "_load", lambda *args: (
        Model(), cohort, config, np.zeros(2), np.ones(2), np.ones(2), dict(spatial=np.array([[0, 1]]))))
    monkeypatch.setattr(evaluation, "NativeRenewalEpisode", Native)
    torch.set_num_threads(1)
    return (tmp_path, tmp_path, tmp_path / "training", tmp_path / "config.yaml",
            tmp_path / "scale.npz", tmp_path / "evaluation", 41)


def test_eag_all_origins_observed_prefix_and_independent_replay(inputs):
    result = evaluation.evaluate(*inputs)
    assert result["score_rows"] == 144 and result["complete_origins"] == 48
    assert result["source_unit_count"] == 2 and result["method"] == "eag"
    replay = evaluation.evaluate(*inputs, replay=True)
    assert replay["count_values_digest_replayed"] == 48 * 16 * 100 * 2
    with (inputs[5] / "scores.csv").open() as stream:
        assert {row["method"] for row in csv.DictReader(stream)} == {"eag"}
    with np.load(inputs[5] / "case_0_origin_5000.npz", allow_pickle=False) as saved:
        assert saved["prefix_condition"].shape == (1, 1000, 1)
        assert saved["observed_controls"].shape == (1, 1000, 1)


def test_eag_replay_rejects_corrupt_outputs_and_repeated_generation(inputs):
    evaluation.evaluate(*inputs)
    with pytest.raises(FileExistsError):
        evaluation.evaluate(*inputs)
    (inputs[5] / "full_unit_samples.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="saved output changed"):
        evaluation.evaluate(*inputs, replay=True)
