import numpy as np
import pytest
from .v1_glif_network import positions, connection_csr, external_events


def test_native_identity_order_is_preserved():
    assert positions(np.array([9,3,7]),np.array([7,9])).tolist()==[2,0]
    with pytest.raises(ValueError,match="UNKNOWN_IDENTITY"):
        positions(np.array([9,3,7]),np.array([4]))


def test_duplicate_native_identity_rejects():
    with pytest.raises(ValueError,match="IDENTITY_DOMAIN"):
        positions(np.array([2,2]),np.array([2]))


def test_csr_retains_zero_weights_silent_sources_and_receptors():
    raw=dict(source=np.array([20,10,20]),target=np.array([11,10,10]),
             weight=np.array([0.,-2.,3.]),delay=np.array([.25,.75,.5]),
             receptor_type=np.array([1,4,2]))
    csr=connection_csr(raw,np.array([10,11]),np.array([20,21]),.25,4)
    assert csr["offsets"].tolist()==[0,1,1,3,3]
    assert csr["weight"].tolist()==[-2.,0.,3.]
    assert csr["receptor"].tolist()==[3,0,1]
    raw["delay"][0]=.125
    with pytest.raises(ValueError,match="CONNECTION_DOMAIN"):
        connection_csr(raw,np.array([10,11]),np.array([20,21]),.25,4)


def test_input_encoding_retains_same_stamp_multiplicity_and_silent_units():
    raw=dict(source_event_counts=np.array([2,0,1]),encoded_steps=np.array([1,1,4]),
             source_event_unit=np.array([0,0,2]),native_times_ms=np.array([.125,.25,1.]),
             encoded_offsets_ms=np.array([.125,0.,0.]),precise_times=np.ones(3,dtype=bool))
    packed=external_events(raw,3,4,.25)
    assert packed["offsets"].tolist()==[0,2,2,2,3]
    assert packed["source"].tolist()==[3,3,5]
    raw["source_event_counts"][0]=1
    with pytest.raises(ValueError,match="COUNTS_DIFFER"):
        external_events(raw,3,4,.25)


def test_snapshot_is_bound_to_the_actual_native_reference(tmp_path):
    import json
    from .v1_glif_network import snapshot_connections
    original={"connections":{"source_batch_sha256":"expected"}}
    (tmp_path/"report.json").write_text(json.dumps({"status":"PASS_COMPLETE_NATIVE_CONNECTION_SNAPSHOT",
        "native_report_sha256":"other"}))
    with pytest.raises(ValueError,match="SNAPSHOT_REFERENCE_DIFFERS"):
        snapshot_connections(tmp_path,original,"requested")


def test_snapshot_consumer_uses_native_delay_and_preserves_zero_weights(tmp_path):
    import json
    from .test_v1_connection_snapshot import make_file
    from .v1_glif_network import snapshot_connections
    path,digest=make_file(tmp_path)
    original={"connections":{"source_batch_sha256":digest,"native_connections_to_cortex":2,
                               "full_native_tuple_readback":False}}
    (tmp_path/"report.json").write_text(json.dumps(dict(status="PASS_COMPLETE_NATIVE_CONNECTION_SNAPSHOT",
        native_report_sha256="native",expected_source_batch_sha256=digest,
        parameters_and_initial_state_exact=True,input_events_exact=True,native_biological_time_ms=0,
        connections_file=path.name)))
    rows,evidence=snapshot_connections(tmp_path,original,"native")
    assert rows["delay"].tolist()==[.75,.75]
    assert rows["weight"].tolist()==[0.,-2.5]
    assert evidence["provenance"]=="source_call_digest_snapshot"
    assert not evidence["full_native_tuple_readback"]


def cache_fixture():
    import copy
    from .v1_glif_network import CircuitIdentity
    state=dict(source_id=np.array([0,1]),native_id=np.array([8,3]),
        node_type_id=np.array([10,11]),inverse_source_id=np.array([0,1]),
        population=np.array(["v1","v1"]),initial_vm_mV=np.array([-70.,-60.]),
        terminal_vm_mV=np.array([-61.,-62.]),spike_count=np.array([1,2]))
    inputs=dict(population=np.array(["lgn","bkg"]),source_id=np.array([0,0]),
        native_id=np.array([9,10]),inverse_source_id=np.array([0,0]),
        precise_times=np.array([True,True]),encoded_steps=np.array([1,2]),
        source_event_counts=np.array([1,1]))
    report=dict(connections=dict(source_batch_sha256="matrix",native_connections_to_cortex=3),
        execution_clock=dict(resolution_ms=.25,duration_ms=3000.,tics_per_ms=1000.))
    data=dict(cortical_state=state,input_events=inputs)
    binding={"files":{"primary/native/parameters.json":{"sha256":"parameters"}}}
    return CircuitIdentity.capture(report,data,binding),copy.deepcopy(report),copy.deepcopy(data),copy.deepcopy(binding)


def test_circuit_reuse_allows_changed_inputs_and_outcomes():
    identity,report,data,binding=cache_fixture()
    data["input_events"]["encoded_steps"]=np.array([3,5,7])
    data["input_events"]["source_event_counts"]=np.array([2,1])
    data["cortical_state"]["terminal_vm_mV"]+=1
    data["cortical_state"]["spike_count"]+=4
    identity.validate(report,data,binding)


@pytest.mark.parametrize("field",["source_id","native_id","node_type_id","initial_vm_mV"])
def test_circuit_reuse_rejects_changed_cell_mapping_or_reset(field):
    identity,report,data,binding=cache_fixture()
    data["cortical_state"][field][0]+=1
    with pytest.raises(ValueError,match="CELL_IDENTITY_DIFFERS"):
        identity.validate(report,data,binding)


def test_circuit_reuse_rejects_changed_input_mapping():
    identity,report,data,binding=cache_fixture()
    data["input_events"]["native_id"][0]+=1
    with pytest.raises(ValueError,match="INPUT_IDENTITY_DIFFERS"):
        identity.validate(report,data,binding)


@pytest.mark.parametrize("change",["matrix","parameters","clock"])
def test_circuit_reuse_rejects_changed_dynamics(change):
    identity,report,data,binding=cache_fixture()
    if change=="matrix":report["connections"]["source_batch_sha256"]="other"
    if change=="parameters":binding["files"]["primary/native/parameters.json"]["sha256"]="other"
    if change=="clock":report["execution_clock"]["resolution_ms"]=.125
    with pytest.raises(ValueError,match="DIFFERS"):
        identity.validate(report,data,binding)


def test_batch_reference_rejects_swapped_native_case():
    from .v1_glif_network import validate_expected_reference
    expected=dict(native_job_id=10,spike_inputs_reference="inputs",native_report_sha256="native")
    binding=dict(native_job_id="11",spike_inputs_reference="inputs",
                 files={"primary/report.json":{"sha256":"native"}})
    with pytest.raises(ValueError,match="REFERENCE_BINDING_DIFFERS"):
        validate_expected_reference(binding,expected)

def test_source_inputs_sort_per_unit_and_preserve_empty_and_duplicate_events():
    from .v1_glif_network import source_input_events
    identity, _, _, _ = cache_fixture()
    plan = dict(population=np.array([0,1]),source_id=np.array([0,0]),
        source_counts=np.array([3,0]),trains=[np.array([.5,.125,.125]),np.array([])])
    raw = source_input_events(plan, identity)
    assert raw["native_id"].tolist() == [9,10]
    assert raw["source_event_counts"].tolist() == [3,0]
    assert raw["requested_times_ms"].tolist() == [.125,.125,.5]
    assert raw["encoded_steps"].tolist() == [1,1,2]
    assert raw["source_event_unit"].tolist() == [0,0,0]
    packed = external_events(raw,2,12000,.25)
    assert packed["source"].tolist() == [2,2,2]


def test_source_inputs_reject_changed_population_identity():
    from .v1_glif_network import source_input_events
    identity, _, _, _ = cache_fixture()
    plan = dict(population=np.array([0,1]),source_id=np.array([1,0]),
        source_counts=np.array([1,1]),trains=[np.array([.25]),np.array([.5])])
    with pytest.raises(ValueError,match="SOURCE_IDS_DIFFER"):
        source_input_events(plan, identity)


def test_source_reader_never_requires_cortical_outcomes(tmp_path, monkeypatch):
    import json
    from . import v1_glif_network as network
    _, _, data, binding = cache_fixture()
    target=tmp_path/"primary/native";target.mkdir(parents=True)
    # Keep only static source/reset arrays. No cortical spike file exists.
    np.savez(target/"cortical_state.npz", **{key:data["cortical_state"][key]
                                         for key in network.CELL_IDENTITY_FIELDS})
    np.savez(target/"input_events.npz", **data["input_events"])
    (tmp_path/"primary/report.json").write_text(json.dumps(dict(exit_code=0,
        native_nest_simulation=True,native_parameters_and_initial_state_exact=True)))
    required=[]
    def supplied(archive, relatives):
        required.extend(relatives)
        return dict(exit_code=0,job_id=1), binding
    monkeypatch.setattr(network,"bound_files",supplied)
    loaded, _, _ = network.read_reference_data(tmp_path, require_tuples=False, include_outcomes=False)
    assert "primary/native/cortical_spikes.npz" not in required
    assert "cortical_spikes" not in loaded
    assert set(loaded["cortical_state"]) == set(network.CELL_IDENTITY_FIELDS)


def test_source_cli_dispatches_source_inputs(monkeypatch):
    import sys
    from pathlib import Path
    from . import v1_glif_network as network
    called=[]
    def execute(*args):
        called.append(args)
        return {"status":"executed"}
    monkeypatch.setattr(network,"execute_source",execute)
    monkeypatch.setattr(sys,"argv",["v1_glif_network","--archive","circuit","--snapshot","snapshot",
        "--source-inputs","inputs","--cuda-library","library","--output","output"])
    network.main()
    assert called == [(Path("circuit"),Path("inputs"),Path("library"),Path("output"),Path("snapshot"))]

def source_block_fixture(tmp_path):
    import json
    cases=[]
    for index in range(10):
        identity=dict(scenario_id="v1_visual",circuit_id="circuit0",circuit_index=0,
            block_id="block2",block_index=2,condition_index=index,episode_id="episode"+str(index),
            connectivity_seed=100,initial_state_seed=110,background_seed=120,stimulus_seed=130,
            split="train",sample_unit="complete_mechanistic_episode")
        config=dict(background=dict(starting_noise_seed=120),
                    spike_generation=dict(starting_noise_seed=130))
        cases.append(dict(identity=identity,input_archive=str(tmp_path/("inputs"+str(index))),
            input_report_sha256="input"+str(index),input_config=config))
    config=dict(schema="v1_source_input_block_v1",scenario_id="v1_visual",
                circuit_index=0,block_index=2,cases=cases)
    path=tmp_path/"cases.json";path.write_text(json.dumps(config))
    return path,config


def test_source_cache_load_excludes_outcomes_and_freezes_arrays(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from . import v1_glif_network as network
    _,report,data,binding=cache_fixture()
    binding["connection_snapshot"]={"verification":"bound"}
    plan=SimpleNamespace(native_ids=np.array([8,3]),initial_voltage=np.array([-70.,-60.]))
    edges={"weight":np.array([1.,2.])}
    calls=[]
    def read(archive,snapshot,*,include_outcomes):
        calls.append(include_outcomes)
        return plan,edges,{},data,binding,report
    monkeypatch.setattr(network,"read_native",read)
    cache=network.NativeCircuitCache.load(tmp_path,tmp_path/"snapshot",include_outcomes=False)
    assert calls==[False] and cache.first_reference is None and not cache.includes_outcomes
    assert not cache.cells.native_ids.flags.writeable and not cache.edges["weight"].flags.writeable
    assert not cache.identity.cells["initial_vm_mV"].flags.writeable
    with pytest.raises(ValueError,match="REFERENCE_OUTCOMES_DISABLED"):
        cache.read(tmp_path)


def test_source_case_rejects_wrong_noise_identity(tmp_path):
    from copy import deepcopy
    from pathlib import Path
    from .v1_glif_network import validate_source_case
    _,config=source_block_fixture(tmp_path)
    case=config["cases"][0]
    binding=dict(config=deepcopy(case["input_config"]),bundle_report_sha256=case["input_report_sha256"])
    validate_source_case(Path(case["input_archive"]),binding,case)
    case["identity"]["background_seed"]+=1
    with pytest.raises(ValueError,match="NOISE_IDENTITY_DIFFERS"):
        validate_source_case(Path(case["input_archive"]),binding,case)


@pytest.mark.parametrize("change",["missing","duplicate","different_block"])
def test_source_block_rejects_incomplete_or_mixed_cases(tmp_path,change):
    import json
    from .v1_glif_network import source_block_cases
    path,config=source_block_fixture(tmp_path)
    if change=="missing":config["cases"].pop()
    elif change=="duplicate":config["cases"][1]["input_archive"]=config["cases"][0]["input_archive"]
    else:config["cases"][1]["identity"]["block_index"]+=1
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match="GLIF_SOURCE_"):
        source_block_cases(path)


def test_source_block_retains_failed_condition_and_continues(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from . import v1_glif_network as network
    path,_=source_block_fixture(tmp_path)
    cache=SimpleNamespace(cells=SimpleNamespace(count=2),identity=SimpleNamespace(connection_rows=3),
                          source_inputs_prepared=0)
    loads=[];calls=[]
    def load(*args,**kwargs):
        loads.append(kwargs)
        return cache
    monkeypatch.setattr(network.NativeCircuitCache,"load",load)
    monkeypatch.setattr(network,"device_library",lambda path:(None,{}))
    monkeypatch.setenv("SLURM_JOB_ID","test")
    def execute(archive,inputs,library,output,snapshot,**kwargs):
        assert kwargs["cache"] is cache
        index=kwargs["expected_source"]["identity"]["condition_index"]
        calls.append(index);cache.source_inputs_prepared+=1
        output.mkdir()
        result=dict(status="PASS_SOURCE_INPUT_GLIF_CUDA_EXECUTION",candidate_spikes=index,elapsed_seconds=1.)
        if index==4:result.update(status="FAILED_SOURCE_INPUT_GLIF_CUDA_EXECUTION",error="retained failure")
        (output/"report.json").write_text(json.dumps(result))
        if index==4:raise ValueError("retained failure")
        return result
    monkeypatch.setattr(network,"execute_source",execute)
    destination=tmp_path/"output"
    with pytest.raises(ValueError,match="BLOCK_EXECUTION_FAILED"):
        network.execute_source_block(tmp_path,path,tmp_path/"library",destination,tmp_path/"snapshot")
    report=json.loads((destination/"report.json").read_text())
    assert loads==[{"include_outcomes":False}] and calls==list(range(10))
    assert report["passed_conditions"]==9 and report["failed_conditions"]==1
    assert report["circuit_matrix_loads"]==1 and report["circuit_cache_hits"]==9
    assert report["cases"][4]["status"]=="FAILED_SOURCE_INPUT_GLIF_CUDA_EXECUTION"
    assert (destination/report["cases"][4]["report"]).is_file()


def test_source_block_cli_dispatches(monkeypatch):
    import sys
    from pathlib import Path
    from . import v1_glif_network as network
    calls=[]
    def execute(*args):
        calls.append(args);return {"status":"executed"}
    monkeypatch.setattr(network,"execute_source_block",execute)
    monkeypatch.setattr(sys,"argv",["v1_glif_network","--archive","circuit","--snapshot","snapshot",
        "--source-cases","cases","--cuda-library","library","--output","output"])
    network.main()
    assert calls==[(Path("circuit"),Path("cases"),Path("library"),Path("output"),Path("snapshot"))]

def test_source_npz_recording_preserves_readonly_noncontiguous_and_unicode_arrays(tmp_path):
    from .v1_glif_network import save_source_arrays
    ids=np.arange(12,dtype=np.uint64)[::2]
    ids.setflags(write=False)
    arrays=dict(source_id=ids,voltage=np.array([-70.,-60.],dtype=np.float64),
                population=np.array(["lgn","bkg"]),flags=np.array([True,False]))
    path=tmp_path/"source.npz"
    save_source_arrays(path,**arrays)
    with np.load(path,allow_pickle=False) as read:
        assert set(read.files)==set(arrays)
        for name,value in arrays.items():
            assert read[name].dtype==value.dtype
            assert np.array_equal(read[name],value)
