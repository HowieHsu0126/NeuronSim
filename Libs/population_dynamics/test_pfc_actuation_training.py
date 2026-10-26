"""PFC actuation fitting uses registered geometry and only three random seeds."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from .pfc_population_training_data import PfcPopulationTrainingCohort, validate_config
from . import renewal_trajectory_training as trainer
from .population_state import ControlledPopulationState
from .test_population_control_geometry import source_geometry, model


def configured(data):
    value=yaml.safe_load((Path(__file__).resolve().parents[1]/
                         "configs/neural_population_pfc_actuation_training.yaml").read_text())
    value["fit_circuit_ids"]=[data.binding["circuit_id"]]
    value["validation_circuit_ids"]=["other_source_circuit"]
    return value


@pytest.mark.parametrize("seed",[41,42,43])
def test_registered_training_roles_and_constructor_propagate(source_geometry,seed):
    data=source_geometry["episode"]
    config=configured(data)
    validate_config(config,seed)
    assert {seed,trainer.registered_variational_seed(config,seed,"training"),
                 trainer.registered_variational_seed(config,seed,"validation")}=={41,42,43}
    cohort=PfcPopulationTrainingCohort.__new__(PfcPopulationTrainingCohort)
    cohort.fit=[dict(case_id="fixture")]
    cohort.bin_ticks=config["bin_ticks"]
    with patch.object(cohort,"_episode",return_value=data):
        constructor=cohort.population_configuration(config,seed)
    assert constructor["control_geometry"]=="actuation_only"
    assert ControlledPopulationState(**constructor).control_geometry=="actuation_only"


@pytest.mark.parametrize("name,value",[
    ("control_geometry","unknown"),("random_seed_policy","hashed"),
    ("training_variational_seed_base",61001),("validation_seed_base",64001),
])
def test_unregistered_new_mode_or_seed_contract_rejects(source_geometry,name,value):
    config=configured(source_geometry["episode"]);config[name]=value
    with pytest.raises(ValueError,match="CONTROL_GEOMETRY|ACTUATION_SEEDS"):
        validate_config(config,41)


def test_validation_restarts_one_complete_sequence_without_training_rng(source_geometry):
    data=source_geometry["episode"]
    config=configured(data);config["validation_draws"]=2
    canonical=model(source_geometry,41)
    window=SimpleNamespace(counts=np.zeros((2,len(data.unit_ids)),dtype=np.int64),
                           stimulus=np.zeros((2,4)),eligible_ticks=np.full((2,len(data.unit_ids)),100))
    class Cohort:
        validation=[dict(case_id="first"),dict(case_id="second")]
        def window(self,case):return window
        def model_for_case(self,model,case):return model
    before=torch.get_rng_state().clone()
    rows=[]
    left=trainer.evaluate_validation(canonical,Cohort(),config,41,on_draw=rows.append)
    right=trainer.evaluate_validation(canonical,Cohort(),config,41)
    assert left==right and torch.equal(before,torch.get_rng_state())
    assert {row["variational_seed"] for row in rows}=={43}
    assert len(rows)==4


def test_source_geometry_cannot_enter_fit_only_readout_initialization(source_geometry):
    data=source_geometry["episode"]
    config=configured(data)
    canonical=model(source_geometry,41)
    cohort=PfcPopulationTrainingCohort.__new__(PfcPopulationTrainingCohort)
    cohort.fit=[dict(case_id="first",circuit_id=data.binding["circuit_id"],source_target_geometry=(1,)),
                dict(case_id="second",circuit_id=data.binding["circuit_id"],source_target_geometry=(2,))]
    cohort.window=lambda case:SimpleNamespace(counts=np.zeros((2,len(data.unit_ids)),dtype=np.int64),
                                              eligible_ticks=np.full((2,len(data.unit_ids)),100))
    other=canonical.training_for_circuit(unit_ids=canonical.unit_ids,
        unit_field_index=canonical.unit_field_index,field_names=canonical.field_names,
        minimum_interval_ticks=canonical.minimum_interval_ticks,metadata=source_geometry["candidate"])
    cohort.model_for_case=lambda model,case:canonical if case["case_id"]=="first" else other
    result=cohort.initialize_observation(canonical,.5)
    target_columns=canonical._unit_target_columns
    assert np.all(canonical.shared_unit_readout.weight[-1,list(target_columns)].detach().numpy()==0)
    assert not result["validation_observations_used"]
