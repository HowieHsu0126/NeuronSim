"""Bound source metadata checks only; no simulation or array processing."""
import copy
import hashlib
import json
import pytest
from . import pfc_source_refractory_support as support


def encoded(value):return json.dumps(value,sort_keys=True).encode()
def binding(path,data):return dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def fixture():
    identity=dict(episode_id="episode-1",initial_state_seed=210063006,background_seed=220063006)
    source=dict(scope="source_reproduction",excitatory_neurons=80000,inhibitory_neurons=20000,dt_ms=0.0125,duration_ms=5000,
        sigma_weight_profile=20,Jpos_excit2excit=2.3)
    case_part=dict(random_stream_policy="independent_initial_external",stimulus_strength_nA=0.5,distractor_strength_nA=0)
    expected=dict(identity=identity,config_sha256="c"*64,source_signature=dict(source_config=source,case=case_part))
    proof=dict(status="PASS_NATIVE_STREAM_SOURCE_EXECUTION",backend="numpy",validated_dt_ms=[0.0125],numpy_version="2.4.2",
        executor_sha256=support.EXECUTOR_SHA256,source_entry_sha256=support.SOURCE_ENTRY_SHA256,drive_sha256="d"*64,
        production_episodes=0,scientific_fidelity_qualified=False)
    native=dict(status="PFC_SOURCE_EXECUTION_COMPLETE",source_execution_complete=True,case_id=identity["episode_id"],
        config_sha256=expected["config_sha256"],resolved_config=dict(source,cases=[],benchmark_episode_candidate=dict(identity=identity)),
        selected_case=dict(case_part,case_id=identity["episode_id"],initial_rng_seed=identity["initial_state_seed"],external_noise_seed=identity["background_seed"]),
        registered_case_ids=[identity["episode_id"]],registered_case_count=1,selected_case_count=1,completed_case_count=1,
        completed_duration_ticks=400000,dimensions=dict(n_e=80000,n_i=20000,coupled_core_units=100000,duration_ticks=400000,dt_seconds=0.0000125,duration_seconds=5),
        backend="numpy",source_entry_sha256=support.SOURCE_ENTRY_SHA256,code=dict(executor_sha256=support.EXECUTOR_SHA256,drive_sha256="d"*64),
        driver=dict(n_e=80000,n_i=20000,total_ticks=400000,ticks_generated=400000,dt_seconds=0.0000125,
            random_stream_policy="independent_initial_external",seeds=dict(initial_rng_seed=identity["initial_state_seed"],external_noise_seed=identity["background_seed"]),
            source_entry_sha256=support.SOURCE_ENTRY_SHA256,numpy_version="2.4.2",external_event_totals=dict(E=12,I=3)),
        inputs_used=dict(ticks_consumed=400000,E=dict(values=32000000000,events=12),I=dict(values=8000000000,events=3)),
        source_external_event_totals=dict(E=12,I=3),production_episodes=0,scientific_fidelity_qualified=False,
        validations=dict(complete_source_input_consumption=dict(passed=True,expected_ticks=400000,generated_ticks=400000,
            expected_values=dict(E=32000000000,I=8000000000),event_totals=dict(E=12,I=3)),
            refractory_spacing_E=dict(passed=True,minimum_required_ticks=160,minimum_observed_ticks=170,violation_count=0,within_unit_event_pairs=3),
            refractory_spacing_I=dict(passed=True,minimum_required_ticks=80,minimum_observed_ticks=None,violation_count=0,within_unit_event_pairs=0)))
    return expected,native,proof


def arguments(expected,native,proof):
    q=encoded(proof)
    qdelivery=dict(status="ARCHIVED_PFC_STREAM_VALIDATION",result_status="PASS_NATIVE_STREAM_SOURCE_EXECUTION",files=[binding("comparison/qualification.json",q)],production_episodes=0,scientific_fidelity_qualified=False)
    qd=encoded(qdelivery)
    native=copy.deepcopy(native);native["backend_qualification"]=dict(delivery_sha256=hashlib.sha256(qd).hexdigest(),receipt_sha256=hashlib.sha256(q).hexdigest(),receipt_file="comparison/qualification.json",qualification=proof)
    n=encoded(native)
    delivery=dict(status="ARCHIVED_PFC_SOURCE_EXECUTION",result_status="PFC_SOURCE_EXECUTION_COMPLETE",source_execution_complete=True,case_id=expected["identity"]["episode_id"],exit_code=0,
        files=[binding("source_run/report.json",n)],production_episodes=0,scientific_fidelity_qualified=False)
    d=encoded(delivery)
    readout=dict(identity=expected["identity"],config_sha256=expected["config_sha256"],source_delivery_sha256=hashlib.sha256(d).hexdigest(),source_report_sha256=hashlib.sha256(n).hexdigest(),source_archive="original-bound-archive")
    return (expected,encoded(readout),d,n,qd,q)


def test_nonzero_and_null_pair_support_remain_distinct():
    expected,native,proof=fixture();original=copy.deepcopy(expected)
    result=support.validate_episode(*arguments(expected,native,proof))
    assert result["populations"]["E"]["minimum_observed_ticks"]==170
    assert result["populations"]["I"]["minimum_observed_ticks"] is None
    assert result["populations"]["I"]["within_unit_event_pairs"]==0
    assert result["source_qualification_granted"] is result["scientific_parameter_domain_qualified"] is False
    assert expected==original and result["production_episodes"]==0


@pytest.mark.parametrize("change",["required","minimum","null_with_pairs","gap_without_pairs","violations","passed","units","clock","ticks","config","identity","case","driver","input_values","scope"])
def test_resealed_invalid_source_metadata_is_rejected(change):
    expected,native,proof=fixture()
    if change=="required":native["validations"]["refractory_spacing_E"]["minimum_required_ticks"]=159
    elif change=="minimum":native["validations"]["refractory_spacing_E"]["minimum_observed_ticks"]=159
    elif change=="null_with_pairs":native["validations"]["refractory_spacing_E"]["minimum_observed_ticks"]=None
    elif change=="gap_without_pairs":native["validations"]["refractory_spacing_I"]["minimum_observed_ticks"]=80
    elif change=="violations":native["validations"]["refractory_spacing_E"]["violation_count"]=1
    elif change=="passed":native["validations"]["refractory_spacing_E"]["passed"]=False
    elif change=="units":native["dimensions"]["n_e"]=79999
    elif change=="clock":native["dimensions"]["dt_seconds"]=0.00005
    elif change=="ticks":native["driver"]["ticks_generated"]=399999
    elif change=="config":native["resolved_config"]["Jpos_excit2excit"]=1.6
    elif change=="identity":expected["identity"]["background_seed"]+=1
    elif change=="case":native["selected_case"]["stimulus_strength_nA"]=0
    elif change=="driver":native["code"]["drive_sha256"]="e"*64
    elif change=="input_values":native["inputs_used"]["I"]["values"]-=1
    else:native["production_episodes"]=1
    with pytest.raises(ValueError):support.validate_episode(*arguments(expected,native,proof))


def test_altered_bytes_break_original_source_binding():
    values=list(arguments(*fixture()));values[3]+=b" "
    with pytest.raises(ValueError,match="READOUT_SOURCE_BINDING_CHANGED"):support.validate_episode(*values)


def test_valid_nested_source_config_preserves_parameter_meaning():
    expected,native,proof=fixture();native["resolved_config"]=dict(source_config=native["resolved_config"])
    assert support.validate_episode(*arguments(expected,native,proof))["source_signature"]==expected["source_signature"]
