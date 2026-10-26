"""Native event-label prior fitting through the shared trainer and recovery."""
import json

import pytest
import torch
import yaml

from . import renewal_trajectory_training as training
from .population_prior_refit import OBSERVATION_OVERRIDES
from .population_prior_objective import PRIOR_PARAMETER_NAMES
from .population_state import ControlledPopulationState
from .provenance import sha256_file
from .test_population_prior_refit import CompleteClockFixture, make_parent
from .test_population_prior_campaign import _module
from .test_renewal_trajectory_training import _same


def observation_parent(tmp_path):
    parent,config,path,original,cohort,constructor=make_parent(tmp_path)
    config.update(OBSERVATION_OVERRIDES)
    path.write_text(yaml.safe_dump(config))
    return parent,config,path,original,cohort,constructor


@pytest.mark.parametrize("key,value",[("prediction_particles",4),("prior_selection","training_loss")])
def test_observation_profile_remains_registered(tmp_path,key,value):
    _,config,_,_,_,_=observation_parent(tmp_path)
    training.validate_config(config,41,backend="population")
    with pytest.raises(ValueError,match="configuration differs"):
        training.validate_config(dict(config,**{key:value}),41,backend="population")


def test_observation_refit_resume_preserves_optimizer_streams_and_readout(tmp_path,monkeypatch):
    parent,_,path,original,_,_=observation_parent(tmp_path)
    monkeypatch.setattr(training,"TrainingCohort",CompleteClockFixture)
    invoke=lambda out,**kw:training.train(tmp_path,tmp_path,path,out,41,backend="population",**kw)
    whole,split=tmp_path/"whole",tmp_path/"split"
    report=invoke(whole,prior_parent=parent,stop_after_epochs=2)
    invoke(split,prior_parent=parent,stop_after_epochs=1)
    resumed=invoke(split,resume=True,stop_after_epochs=1)
    assert report["training_objective"]=="fixed_readout_native_event_path_mixture_nll"
    assert report["selection_metric"]=="validation_native_event_path_mixture_nll"
    assert report["observation_likelihood_evaluated_during_refit"]
    assert report["posterior_cache_role"]=="causal_boundary_only"
    assert not report["posterior_targets_used"] and not report["future_exposure_used_in_prior_dynamics"]
    assert report["prediction_particles"]==8 and not report["marginal_likelihood_exact"]
    assert resumed["resumed"] and resumed["frozen_parameters_unchanged_in_attempt"]
    a,b=[torch.load(out/"latest.pt",weights_only=True) for out in (whole,split)]
    for key in ("model","optimizer","variational_rng","prior_dynamics_rng","history","best","best_epoch"):
        _same(a[key],b[key])
    assert "prior_target_rng" not in a
    assert "population_observation_prior.py" in a["trainer_sources"]
    assert len(a["history"][0]["validation"])==9
    for name,value in original["model"].items():
        if name not in PRIOR_PARAMETER_NAMES:assert torch.equal(value,b["model"][name]),name
    assert any(not torch.equal(original["model"][name],b["model"][name]) for name in PRIOR_PARAMETER_NAMES)


def test_native_labels_reuse_frozen_cache_without_reencoding(tmp_path,monkeypatch):
    parent,config,path,_,_,_=make_parent(tmp_path)
    monkeypatch.setattr(training,"TrainingCohort",CompleteClockFixture)
    source=tmp_path/"accepted_control";source.mkdir();out=source/"training"
    training.train(tmp_path,tmp_path,path,out,41,backend="population",prior_parent=parent,stop_after_epochs=1)
    report=json.loads((out/"run.json").read_text());report["status"]="FITTED_DEVELOPMENT"
    (out/"run.json").write_text(json.dumps(report))
    names=["training/run.json",*["training/posterior_cache/"+p.name for p in sorted((out/"posterior_cache").iterdir())]]
    files=[dict(path=name,bytes=(source/name).stat().st_size,sha256=sha256_file(source/name)) for name in names]
    (source/"delivery.json").write_text(json.dumps(dict(status="PASS",exit_code=0,training_seed=41,files=files)))
    config.update(OBSERVATION_OVERRIDES);path.write_text(yaml.safe_dump(config))
    def no_encoding(self,*args,**kwargs):raise AssertionError("frozen posterior cache was re-encoded")
    monkeypatch.setattr(ControlledPopulationState,"encode",no_encoding)
    child=tmp_path/"native_observation"
    result=training.train(tmp_path,tmp_path,path,child,41,backend="population",prior_parent=parent,
                          prior_cache_source=source,stop_after_epochs=1)
    assert result["posterior_cache_reuse"]["reused_episodes"]==3
    assert result["frozen_parameters_unchanged_in_attempt"]
    for row in files[1:]:assert sha256_file(child/"posterior_cache"/row["path"].split("/")[-1])==row["sha256"]


def test_launcher_registers_observation_config_and_tests():
    module=_module("run_population_prior_campaign.py")
    assert module.OBSERVATION_CONFIG in module.CONFIGURATIONS
    assert module.PRIOR_PROFILES[module.OBSERVATION_CONFIG]==(
        "neural_population_native_observation_prior_v1","fixed_readout_native_event_path_mixture_nll",
        "validation_native_event_path_mixture_nll")
    assert {"test_population_observation_prior.py","test_population_observation_prior_refit.py"}<=set(module.test_names(module.OBSERVATION_CONFIG))
