"""V1 source-clock, reset, causal boundary and population-input contracts."""
from dataclasses import replace
import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from .circuit_data import SCHEMA, CircuitArchiveWriter, CircuitDataset, CircuitEpisode, CircuitUnits
from .population_state import ControlledPopulationState
from .provenance import canonical_json
from .v1_renewal_data import CONTROL_FEATURES, RESET_POLICY, V1RenewalEpisode


def fixture(root, *, ticks=None, refractory=1.5500000000000003,
            circuit_id="v1_fixture", episode_id="fixture", split="train"):
    parameters = {"111": dict(model="glif_psc", all_units_exact=True, units=2,
                            parameters=dict(t_ref=refractory, C_m=100., g=10., V_th=-50., V_reset=-70.)),
                  "222": dict(model="glif_psc", all_units_exact=True, units=1,
                            parameters=dict(t_ref=.25, C_m=80., g=5., V_th=-48., V_reset=-65.))}
    native = root / "parameters.json"; native.write_text(json.dumps(parameters))
    identity = hashlib.sha256(canonical_json({key:dict(model=v["model"],parameters=v["parameters"])
                                            for key,v in parameters.items()}).encode()).hexdigest()
    provenance = dict(parameter_identity=identity, connectivity_identity="fixture-connectivity")
    units = CircuitUnits("v1_fixture", np.array([10,20,30]), np.array(["e23","i1","e23"]),
        np.array([0,1,0]), ("e23","i1"), "source-sampled model coordinates", "pointnet-fixture",
        np.array([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]]), "um", provenance,
        node_type_ids=np.array([111,222,111]), source_tuning_angle_deg=np.array([0.,90.,180.]))
    def document(value):
        payload=json.dumps(value); return dict(payload=payload,sha256=hashlib.sha256(payload.encode()).hexdigest())
    reference = document(dict(execution_clock=dict(duration_ms=10.,resolution_ms=.25,tics_per_ms=1000.,tics_per_step=250),
                              native_parameters_and_initial_state_exact=True))
    execution = document(dict(status="PASS_SOURCE_INPUT_GLIF_CUDA_EXECUTION",source_reset_policy=RESET_POLICY,units=3))
    qualification = "a"*64
    record = dict(complete_unit_recording=True, simulation_completed=True, recorded_until_tick=40,
        expected_duration_ticks=40, core_unit_count=3, source_revision="pointnet-fixture",circuit_provenance=provenance,
        production_declaration=dict(production_eligible=True,intended_split="train",circuit_id="v1_fixture",
                                    source_qualification_sha256=qualification),
        production_source_admission=dict(source_qualification_sha256=qualification,
            cohort_identity=dict(episode_id="fixture",circuit_id="v1_fixture",block_id="block",split="train"),
            source_documents=dict(circuit_reference=reference,execution=execution)),
        circuit_reference_report_sha256=reference["sha256"],execution_report_sha256=execution["sha256"])
    episode = CircuitEpisode("fixture","v1_fixture","block","train",40,.25,
        np.array([1,2,4,9,18,25,40] if ticks is None else ticks),np.array([0,1,1,0,1,0,0]),
        np.array([0,20,40]),np.array([[0.,.04,2.,0.,0.,0.],[.8,.04,2.,0.,0.,.005],[.8,.04,2.,0.,0.,.01]]),
        CONTROL_FEATURES,source_record=record,privileged={"future_voltage":np.array([99999.])})
    binding=dict(schema=SCHEMA, sample_unit="complete_mechanistic_episode",scenario_id="v1_visual",scope="source_reproduction",
                 source_revision="pointnet-fixture",source_qualification_sha256=qualification)
    units = replace(units, circuit_id=circuit_id)
    record["production_declaration"].update(circuit_id=circuit_id, intended_split=split)
    record["production_source_admission"]["cohort_identity"].update(
        circuit_id=circuit_id, episode_id=episode_id, split=split)
    episode = replace(episode, circuit_id=circuit_id, episode_id=episode_id, split=split, source_record=record)
    with CircuitArchiveWriter(root/"dataset",binding) as writer:
        writer.register_circuit(units); writer.add(episode)
    data = CircuitDataset(root/"dataset",splits=("train",))
    # Supply an admitted-catalog fixture; production qualification belongs to the archive tests.
    data.catalog["binding"]["scope"] = "production"
    return data,native


def load(root, **kwargs):
    data,native=fixture(root,**kwargs)
    return V1RenewalEpisode(data,"fixture",bin_ticks=5,native_parameters=native)


def dense_oracle(episode,start,stop):
    counts=np.zeros(((stop-start)//5,3),dtype=np.int64); exposure=np.zeros_like(counts)
    ready=np.ones(3,dtype=np.int64); boundary=None
    for tick in range(stop):
        if tick==start: boundary=np.maximum(ready-start,0)
        if tick>=start: exposure[(tick-start)//5] += tick>=ready
        emitted=episode.event_units[episode.event_ticks==tick]
        if len(emitted):
            assert np.all(tick>=ready[emitted])
            ready[emitted]=tick+episode.minimum_interval_ticks[emitted]
            if tick>=start: np.add.at(counts[(tick-start)//5],emitted,1)
    return counts,exposure,boundary,np.maximum(ready-stop,0)


@pytest.mark.parametrize("start,stop",[(0,40),(5,20),(35,40)])
def test_exposure_and_ready_match_independent_tick_oracle(tmp_path,start,stop):
    episode=load(tmp_path)
    np.testing.assert_array_equal(episode.minimum_interval_ticks,[8,2,8])
    value=episode.renewal_window(start,stop)
    expected=dense_oracle(episode,start,stop)
    for actual,wanted in zip((value.counts,value.eligible_ticks,value.initial_ready_in_ticks,value.terminal_ready_in_ticks),expected):
        np.testing.assert_array_equal(actual,wanted)
    np.testing.assert_array_equal(episode.ready_at(start),expected[2])
    np.testing.assert_array_equal(episode.ready_at(stop),expected[3])
    assert episode.event_ticks[-1]==40
    if start==0: assert value.counts.sum()==6 and value.eligible_ticks[0,2]==4


def test_future_events_do_not_change_prefix_exposure_or_ready(tmp_path):
    episode=load(tmp_path); before=episode.renewal_window(5,10); ready=episode.ready_at(10)
    episode.event_ticks[-3]=19
    after=episode.renewal_window(5,10)
    np.testing.assert_array_equal(before.eligible_ticks,after.eligible_ticks)
    np.testing.assert_array_equal(ready,episode.ready_at(10))


def test_parameter_identity_is_checked_before_use(tmp_path):
    data,native=fixture(tmp_path); p=json.loads(native.read_text());p["111"]["parameters"]["t_ref"]=2.
    native.write_text(json.dumps(p))
    with pytest.raises(ValueError,match="PARAMETER_IDENTITY_CHANGED"):
        V1RenewalEpisode(data,"fixture",bin_ticks=5,native_parameters=native)


def test_unselected_split_rejects_before_source_read(tmp_path):
    data,native=fixture(tmp_path); selected=CircuitDataset(data.root,splits=("validation",))
    with patch("pathlib.Path.read_text",side_effect=AssertionError("source accessed")):
        with pytest.raises(KeyError): V1RenewalEpisode(selected,"fixture",bin_ticks=5,native_parameters=native)


def test_tick_zero_and_source_deadtime_violations_reject(tmp_path):
    with pytest.raises(ValueError,match="EVENT_SUPPORT_INVALID"):
        load(tmp_path,ticks=[0,2,4,9,18,25,40])


def test_source_deadtime_violation_rejects_likelihood(tmp_path):
    episode=load(tmp_path,ticks=[1,2,4,8,18,25,40])
    with pytest.raises(ValueError,match="minimum interval"): episode.renewal_window(0,40)


def test_zero_refractory_has_explicit_unsupported_reset_error(tmp_path):
    with pytest.raises(ValueError,match="ZERO_REFRACTORY_RESET_NOT_SUPPORTED"):
        load(tmp_path,refractory=0.)


def test_visual_controls_and_static_features_exclude_privileged_values(tmp_path):
    episode=load(tmp_path); metadata=episode.population_metadata()
    assert len(metadata["control_features"])==6
    assert np.asarray(metadata["unit_features"]).shape==(3,9)
    assert np.asarray(metadata["field_features"]).shape==(2,10)
    assert "99999" not in json.dumps(metadata)
    np.testing.assert_allclose(np.asarray(metadata["unit_features"])[:,0],[0.,.003,.006])
    assert episode.renewal_window(0,40).stimulus.shape==(8,6)


@pytest.mark.parametrize("seed",[41,42,43])
def test_population_model_receives_native_events_and_completes_gradient_step(tmp_path,seed):
    episode=load(tmp_path); window=episode.renewal_window(0,40)
    model=ControlledPopulationState(unit_ids=list(map(int,episode.unit_ids)),
        unit_field_index=episode.unit_field_index.tolist(),field_names=episode.field_names,
        minimum_interval_ticks=episode.minimum_interval_ticks.tolist(),tick_seconds=.00025,bin_seconds=.00125,
        input_size=6,latent_channels=2,encoder_size=3,unit_chunk_size=2,initialization_seed=seed,
        metadata=episode.population_metadata())
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    values=[torch.as_tensor(x,dtype=torch.float32).unsqueeze(0) for x in (window.counts,window.stimulus,window.eligible_ticks)]
    loss=model.sequence_loss(*values,torch.Generator().manual_seed(seed))["loss"]
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    boundary=model.infer_state(dict(counts=values[0],stimulus=values[1],ready_in_ticks=episode.ready_at(40)[None]))
    assert torch.isfinite(boundary.mean).all()
