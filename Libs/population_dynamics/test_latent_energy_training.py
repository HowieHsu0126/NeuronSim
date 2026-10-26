"""Fit-only conditioning, complete logical batches and exact EAG restart tests."""
import copy
import json

import numpy as np
import pytest
import torch
import yaml

from . import latent_energy_training as training


class TinyCohort:
    def __init__(self, acquisition, fields, config):
        self.fit = [dict(case_id="fit_a", offset=0), dict(case_id="fit_b", offset=1)]
        self.validation = [dict(case_id="validation", offset=9)]
        self.unit_ids = np.arange(8)
        self.binding = dict(fixture="complete identified tiny EAG episodes")

    def window(self, case):
        counts = (np.arange(32 * 8).reshape(32, 8) + case["offset"]) % 3
        controls = np.linspace(.1, .5, 32, dtype=np.float32)[:, None]
        return counts, controls


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = copy.deepcopy(training.FIXED_PROTOCOL)
    config.update(device="cpu", cpu_threads=1, complete_episode_bins=32,
                  forecast_origins_bins=[8, 16, 24], forecast_bins=8)
    config["model"].update(complete_episode_bins=32, maximum_prefix_bins=24, forecast_bins=8,
                           latent_channels=4, hidden_channels=16, encoder_blocks=1,
                           decoder_blocks=1, linear_layers_per_block=2)
    config["model"]["generator_options"].update(encoder_width=8, encoder_depth=1, encoder_heads=2,
        decoder_width=8, decoder_depth=1, decoder_heads=2, score_mlp_width=16, score_mlp_depth=2, noise_channels=4)
    config["autoencoder"].update(maximum_epochs=2, validation_every=1, validation_draws=2, warmup_epochs=0)
    config["energy"].update(maximum_epochs=4, source_batch_size=4, microbatch_size=2,
                            validation_every=1, validation_draws=2, warmup_epochs=1)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(training, "validate_config", lambda config, seed: None)
    monkeypatch.setattr(training, "TrajectoryCohort", TinyCohort)
    return config, path


def _payload(output):
    pointer = json.loads((output / "committed.json").read_text())
    return torch.load(output / pointer["checkpoint"], weights_only=False)


def _equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            _equal(a, b)
    else:
        assert first == second


def test_protocol_keeps_three_seeds_budget_and_allows_measured_microbatch(monkeypatch):
    config = copy.deepcopy(training.FIXED_PROTOCOL)
    config.update(device="cpu", cpu_threads=1)
    config["energy"]["microbatch_size"] = 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for seed in (41, 42, 43):
        training.validate_config(config, seed)
    config["energy"]["microbatch_size"] = 8
    training.validate_config(config, 41)
    config["energy"]["maximum_epochs"] = 3999
    with pytest.raises(ValueError, match="protocol differs"):
        training.validate_config(config, 41)
    config["energy"]["maximum_epochs"] = 4000
    with pytest.raises(ValueError, match="seed41/42/43"):
        training.validate_config(config, True)


def test_fit_only_scale_whole_prefix_and_independent_target_cache(setup, tmp_path):
    config, _ = setup
    cohort = TinyCohort(tmp_path, tmp_path, config)
    torch.manual_seed(41)
    model = training._model(cohort, config)
    cache = training.build_latent_cache(model, cohort, config)
    targets = []
    for case in cohort.fit:
        counts, _ = cohort.window(case)
        counts = torch.tensor(counts)[None].float()
        for origin in config["forecast_origins_bins"]:
            targets.append(model.autoencoder.encode(model._counts(counts[:, origin:origin + 8])).detach())
    values = torch.cat(targets, dim=2)
    torch.testing.assert_close(model.latent_mean, values.mean((0, 2), keepdim=True))
    torch.testing.assert_close(model.latent_sd, values.std((0, 2), correction=0, keepdim=True))
    assert cache["moments"]["fit_cases"] == ["fit_a", "fit_b"]
    assert cache["source_denominators"]["fit_origin_windows"] == 6
    for row in cache["rows"]["validation"]:
        counts, controls = cohort.window(cohort.validation[0])
        counts, controls = torch.tensor(counts)[None].float(), torch.tensor(controls)[None]
        state = model.infer_state(counts[:, :row["origin"]], controls[:, :row["origin"]])
        assert row["prefix"].shape == (1, row["origin"], 4)
        torch.testing.assert_close(row["prefix"], state.latent)
        torch.testing.assert_close(row["observed_controls"], state.controls)
        torch.testing.assert_close(row["target"], model.encode_target(counts[:, row["origin"]:row["origin"] + 8]))
    original = cohort.window
    def altered_validation(case):
        counts, controls = original(case)
        return (counts + 20 if case["case_id"] == "validation" else counts), controls
    cohort.window = altered_validation
    before = model.latent_mean.clone(), model.latent_sd.clone()
    training.build_latent_cache(model, cohort, config)
    assert torch.equal(model.latent_mean, before[0]) and torch.equal(model.latent_sd, before[1])


def test_future_counts_do_not_enter_prefix_encoding(setup, tmp_path):
    config, _ = setup
    cohort = TinyCohort(tmp_path, tmp_path, config)
    torch.manual_seed(42)
    model = training._model(cohort, config)
    first = training.build_latent_cache(model, cohort, config)
    mean, sd = model.latent_mean.transpose(1, 2).clone(), model.latent_sd.transpose(1, 2).clone()
    original = cohort.window
    def altered_future(case):
        counts, controls = original(case)
        counts = counts.copy()
        if case["case_id"] == "fit_a":
            counts[8:] += 10
        return counts, controls
    cohort.window = altered_future
    second = training.build_latent_cache(model, cohort, config)
    a = first["rows"]["fit"][0]["prefix"] * sd + mean
    b = second["rows"]["fit"][0]["prefix"] * model.latent_sd.transpose(1, 2) + model.latent_mean.transpose(1, 2)
    torch.testing.assert_close(a, b)
    assert not torch.equal(first["rows"]["fit"][0]["target"], second["rows"]["fit"][0]["target"])


def test_source_batch256_spans_all4000_complete168_window_passes():
    sampler = training.new_sampler()
    rng = torch.Generator().manual_seed(training._seed(41, "energy_shuffle"))
    seen, updates = {}, 0
    while True:
        batch = training.take_logical_batch(sampler, 168, 4000, 256, rng)
        if not batch:
            break
        assert len(batch) == 256
        updates += 1
        for row in batch:
            seen.setdefault(row["pass_index"], []).append(row["index"])
    assert updates == 2625 and sampler["presentations"] == 672000
    assert sampler["completed_passes"] == 4000 and sampler["cursor"] == 0 and sampler["order"] == []
    assert set(seen) == set(range(4000))
    assert all(sorted(indices) == list(range(168)) for indices in seen.values())


@pytest.mark.parametrize("passes", [3, 4])
def test_interrupted_training_matches_complete_model_optimizer_rng_and_losses(tmp_path, setup, passes):
    values, config = setup
    values["energy"]["maximum_epochs"] = passes
    config.write_text(yaml.safe_dump(values))
    whole, split = tmp_path / "whole", tmp_path / "split"
    training.train(tmp_path, tmp_path, config, whole, 41)
    training.train(tmp_path, tmp_path, config, split, 41, stop_after_commits=1)
    training.train(tmp_path, tmp_path, config, split, 41, resume=True, stop_after_commits=5 if passes == 3 else 2)
    assert _payload(split)["state"]["stage"] == "energy"
    training.train(tmp_path, tmp_path, config, split, 41, resume=True)
    a, b = _payload(whole), _payload(split)
    for key in ("model", "optimizer", "scheduler", "rng", "shuffle_rng", "global_rng", "cuda_rng"):
        _equal(a[key], b[key])
    for key in ("history", "energy_epochs", "energy_sampler", "energy_pass_accumulators"):
        _equal(a["state"][key], b["state"][key])
    assert a["state"]["stage"] == b["state"]["stage"] == "complete"
    assert a["state"]["steps"] == (6 * passes + 3) // 4
    assert a["state"]["energy_sampler"]["presentations"] == 6 * passes
    assert len(a["state"]["energy_epochs"]) == passes
    selected = torch.load(whole / a["state"]["selected_path"], weights_only=False)
    _equal(a["model"], selected["model"])


def test_validation_keeps_training_stream_weights_and_fixed_draws(setup, tmp_path):
    config, _ = setup
    cohort = TinyCohort(tmp_path, tmp_path, config)
    torch.manual_seed(43)
    model = training._model(cohort, config)
    cache = training.build_latent_cache(model, cohort, config)
    before, weights = torch.get_rng_state().clone(), copy.deepcopy(model.state_dict())
    first = training._energy_validation(model, cache, config, 43)
    second = training._energy_validation(model, cache, config, 43)
    _equal(first, second)
    assert torch.equal(before, torch.get_rng_state())
    _equal(weights, model.state_dict())


@pytest.mark.parametrize("passes", [3, 4])
def test_uncommitted_update_is_replayed_and_changed_resume_binding_is_rejected(setup, tmp_path, monkeypatch, passes):
    values, config = setup
    values["energy"]["maximum_epochs"] = passes
    config.write_text(yaml.safe_dump(values))
    output = tmp_path / "interrupted"
    training.train(tmp_path, tmp_path, config, output, 42, stop_after_commits=6 if passes == 3 else 2)
    pointer = (output / "committed.json").read_text()
    original = training._energy_update
    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("fixture interrupted before commit")
    monkeypatch.setattr(training, "_energy_update", interrupted)
    with pytest.raises(RuntimeError, match="before commit"):
        training.train(tmp_path, tmp_path, config, output, 42, resume=True)
    assert (output / "committed.json").read_text() == pointer
    assert list(output.glob("failure_*.json"))
    monkeypatch.setattr(training, "_energy_update", original)
    with pytest.raises(ValueError, match="resume source/data/protocol"):
        training.train(tmp_path, tmp_path, config, output, 43, resume=True)
    training.train(tmp_path, tmp_path, config, output, 42, resume=True)
    whole = tmp_path / "whole"
    training.train(tmp_path, tmp_path, config, whole, 42)
    a, b = _payload(output), _payload(whole)
    for key in ("model", "optimizer", "scheduler", "rng", "shuffle_rng", "global_rng"):
        _equal(a[key], b[key])
    _equal(a["state"]["history"], b["state"]["history"])
    assert list(output.glob("failure_*.json"))


def test_terminal_partial_batch_preserves_each_complete_pass():
    sampler = training.new_sampler()
    rng = torch.Generator().manual_seed(41)
    batches = []
    while True:
        batch = training.take_logical_batch(sampler, 6, 3, 4, rng)
        if not batch:
            break
        batches.append(batch)
    assert list(map(len, batches)) == [4,4,4,4,2]
    assert sampler == dict(completed_passes=3, order=[], cursor=0, presentations=18)
    for current in range(3):
        assert sorted(row["index"] for batch in batches for row in batch if row["pass_index"] == current) == list(range(6))


def test_partial_gradient_uses_actual_example_count_across_microbatches():
    class Energy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))
        def loss(self,target,prefix,controls,**kwargs):
            return dict(loss=(self.weight-target).square().mean())
    class Model:
        autoencoder = torch.nn.Identity()
        energy = Energy()
    model = Model()
    config = dict(device="cpu",energy=dict(source_batch_size=4,microbatch_size=2,gradient_limit=100.),
                  model=dict(generator_options=dict(mask_ratio_min=.7,mask_ratio_max=1.)))
    rows=[dict(origin=2+i%2,target=torch.tensor([[[value]]]),prefix=torch.zeros(1,2+i%2,1),
               observed_controls=torch.zeros(1,2+i%2,1),stimulus=torch.zeros(1,1,1))
          for i,value in enumerate((1.,2.,5.))]
    batch=[dict(index=i,pass_index=0) for i in range(3)]
    optimizer=torch.optim.SGD(model.energy.parameters(),lr=.1)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    result=training._energy_update(model,dict(rows=dict(fit=rows)),config,batch,
                                   torch.Generator().manual_seed(43),optimizer,scheduler)
    torch.testing.assert_close(model.energy.weight,torch.tensor(.1 * 2 * 8 / 3))
    assert result["pass_contributions"]["0"]["examples"]==3
    assert result["mean"]["loss"]==pytest.approx(10.)


def test_terminal_loader_accepts_exact_partial_batch_accounting(setup,tmp_path,monkeypatch):
    from . import latent_energy_evaluation as evaluation
    values,path=setup
    values["energy"]["maximum_epochs"]=3
    path.write_text(yaml.safe_dump(values))
    output=tmp_path/"loadable"
    training.train(tmp_path,tmp_path,path,output,41)
    monkeypatch.setattr(evaluation,"validate_config",lambda *args:None)
    monkeypatch.setattr(evaluation,"TrajectoryCohort",TinyCohort)
    monkeypatch.setattr(evaluation,"fitting_scale",lambda *args:(None,None,None,{}))
    model,*_=evaluation._load(tmp_path,tmp_path,output,path,tmp_path/"scale.npz",41)
    _equal(model.state_dict(),_payload(output)["model"])
