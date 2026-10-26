"""Complete matched LDNS denominator, saved-array replay and rejection checks."""
import json

import numpy as np
import pytest
import torch

from . import latent_spike_diffusion_evaluation as evaluation


class Cohort:
    def __init__(self):
        self.validation = [dict(case_id=f"case_{i}", block=14+i//4, template_id=f"familiar_{i%4}") for i in range(16)]
        self.unit_ids=np.array([11,12]); self.field_ids=np.array(["a","b"])
        self.unit_field_index=np.array([0,1]); self.bin_seconds=.005; self.binding=dict(fixture="complete cohort")
    def window(self, case):
        return np.ones((1200,2),dtype=np.int64),np.zeros((1200,1),dtype=np.float32)


class Model:
    def infer_state(self, counts):
        return type("State",(),dict(latent=counts[:,-1:,].transpose(1,2)))()
    def simulate(self,state,stimulus,*,latent_rng,observation_rng):
        latent=torch.randn((1,2,100),generator=latent_rng)
        rates=torch.ones((1,100,2))
        return dict(counts=torch.poisson(rates,generator=observation_rng).long(),latent=latent,expected_counts=rates)


class Native:
    def __init__(self,*args,**kwargs):
        self.minimum_interval_ticks=np.array([21,21]);self.bin_ticks=50


@pytest.fixture
def inputs(tmp_path,monkeypatch):
    for name in ("config.yaml","scale.npz"): (tmp_path/name).write_bytes(b"bound fixture")
    (tmp_path/"training").mkdir();(tmp_path/"training/committed.json").write_text("{}")
    cohort=Cohort();config=dict(device="cpu",forecast_origins_bins=[200,600,1000],forecast_bins=100,bin_ms=5.)
    monkeypatch.setattr(evaluation,"_load",lambda *args:(Model(),cohort,config,np.zeros(2),np.ones(2),np.ones(2),dict(spatial=np.array([[0,1]]))))
    monkeypatch.setattr(evaluation,"NativeRenewalEpisode",Native)
    torch.set_num_threads(1)
    return (tmp_path,tmp_path,tmp_path/"training",tmp_path/"config.yaml",tmp_path/"scale.npz",tmp_path/"evaluation",41)


def test_all_origins_samples_scores_and_fresh_replay(inputs):
    result=evaluation.evaluate(*inputs)
    assert result["score_rows"]==144;assert result["complete_origins"]==48
    replay=evaluation.evaluate(*inputs,replay=True)
    assert replay["status"]=="PASS"
    with np.load(inputs[5]/"full_unit_samples.npz",allow_pickle=False) as arrays:
        assert arrays["sampled_unit_counts"].shape==(16,100,2)
    assert replay["all_origin_count_digests_fields_latents_metrics_exact"]


def test_replay_rejects_corrupt_saved_outputs_and_existing_output(inputs):
    evaluation.evaluate(*inputs)
    with pytest.raises(FileExistsError):evaluation.evaluate(*inputs)
    (inputs[5]/"full_unit_samples.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="saved output changed"):
        evaluation.evaluate(*inputs,replay=True)


def test_corrected_scores_reuse_finite_ensemble_estimands():
    counts=torch.zeros((16,100,2),dtype=torch.int64);counts[:8]=2
    truth=torch.ones((100,2),dtype=torch.int64)
    rows,fields,actual=evaluation.sample_scores(counts,truth,np.array([0,1]),np.ones(2),
        np.zeros(2),np.ones(2),np.array([21,21]),50,.005,dict(spatial=np.array([[0,1]])))
    assert rows[0]["unit_bin_mean_mse_corrected"]<0
    assert rows[0]["fitting_standardized_field_rate_mse"]<0
    assert np.array_equal(fields.sum(2),counts.sum(2).numpy())
    assert np.array_equal(actual.sum(1),truth.sum(1).numpy())
