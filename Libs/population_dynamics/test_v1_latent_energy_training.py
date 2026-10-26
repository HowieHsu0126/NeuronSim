"""V1 EAG protocol, whole-circuit admission and shared training dispatch."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from . import latent_energy_training as shared
from . import v1_latent_energy_training as source
from . import v1_latent_energy_data as data
from .test_latent_energy_training import TinyCohort, _payload, setup


def config():
    value=source.fixed_protocol()
    value.update(device="cpu",cpu_threads=1,fit_circuit_ids=["fit"],validation_circuit_ids=["validation"])
    value["energy"]["microbatch_size"]=2
    return value


def test_source_protocol_uses_full_clock_controls_and_original_partitions(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    value=config()
    for seed in (41,42,43):
        shared.validate_config(value,seed)
    assert value["complete_episode_bins"]==600
    assert value["model"]["input_size"]==6
    assert value["forecast_origins_bins"]==[100,300,500]
    for change in (dict(validation_circuit_ids=["fit"]),dict(fit_circuit_ids=[]),
                   dict(fit_circuit_ids=["fit","fit"]),dict(bin_ticks=10),dict(complete_episode_bins=1200)):
        with pytest.raises(ValueError):
            source.validate_config(dict(value,**change),41)
    for seed in (40,44,True):
        with pytest.raises(ValueError,match="seed41/42/43"):
            source.validate_config(value,seed)


def test_actual_seed_and_original_admission_precede_model_creation(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    path=tmp_path/"config.yaml";path.write_text(yaml.safe_dump(config()))
    calls=[]
    def reject(inputs,values,*,config_validator):
        calls.append(inputs)
        raise ValueError("original validation missing")
    monkeypatch.setattr(data,"V1LatentEnergyCohort",reject)
    monkeypatch.setattr(shared,"_model",lambda *args:pytest.fail("model created before admission"))
    with pytest.raises(ValueError,match="seed41/42/43"):
        source.train(tmp_path/"inputs.json",path,tmp_path/"training",40)
    assert calls==[]
    with pytest.raises(ValueError,match="original validation missing"):
        source.train(tmp_path/"inputs.json",path,tmp_path/"training",42)
    assert len(calls)==1 and not (tmp_path/"training").exists()


def test_source_dispatch_completes_partial_batch_and_records_binding(setup,tmp_path,monkeypatch):
    value,path=setup
    value["schema"]=shared.V1_SCHEMA
    value["model"]["input_size"]=6
    value["energy"]["maximum_epochs"]=3
    path.write_text(yaml.safe_dump(value))
    class SourceCohort(TinyCohort):
        def __init__(self,inputs,config,*,config_validator):
            super().__init__(inputs,None,config)
            self.binding.update(count_adapter="case_specific_native_source_order",input_manifest="fixture")
        def window(self,case):
            counts,controls=super().window(case)
            return counts,np.tile(controls,(1,6))
    monkeypatch.setattr(data,"V1LatentEnergyCohort",SourceCohort)
    output=tmp_path/"source_training"
    source.train(tmp_path/"inputs.json",path,output,43,stop_after_commits=6)
    assert _payload(output)["state"]["energy_sampler"]["presentations"]==16
    source.train(tmp_path/"inputs.json",path,output,43,resume=True)
    state=_payload(output)["state"]
    assert state["stage"]=="complete" and state["steps"]==5
    assert state["energy_sampler"]["presentations"]==18
    assert state["binding"]["cohort"]["count_adapter"]=="case_specific_native_source_order"
    assert "v1_latent_energy_data.py" in state["binding"]["sources"]
    assert state["source_denominators"]["fit_complete_episodes"]==2
