"""Tiny structural fixtures test observation evidence, not CA3 physiology."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from . import ca3_native_observation as observer
from . import ca3_native_observation_replay as replay
from . import ca3_retrieval_readouts as readouts
from .carlsim_records import (Group, GROUP_HEADER, NETWORK_HEADER, NETWORK_SIGNATURE,
                             SPIKE_HEADER, SPIKE_SIGNATURE)
from .provenance import sha256_file


GROUPS = (Group(0,"B",2,3,(1,1,1)),Group(1,"A",0,2,(2,1,1)))
EVENTS = {"A":[(0,0),(1,1)],"B":[(0,0),(0,0)]}


@pytest.mark.parametrize("mutation", ["path", "sha256", "resolved", "extra", "missing_hash", "missing_resolved"])
def test_coupled_configuration_compares_content_and_preserves_unknown_fields(mutation):
    left = dict(path="/build/source/config.yaml", sha256="a"*64, resolved={"population_multiplier":2})
    right = deepcopy(left)
    if mutation == "path": right["path"] = "/build/observed/config.yaml"
    elif mutation == "sha256": right["sha256"] = "b"*64
    elif mutation == "resolved": right["resolved"]["population_multiplier"] = 3
    elif mutation == "extra": right["extra"] = "different scientific option"
    elif mutation == "missing_hash": del right["sha256"]
    else: del right["resolved"]
    assert replay.same_coupled_configuration(left, right) is (mutation == "path")


def write_json(path, value):
    path.write_text(json.dumps(value))


def record(kind,start,stop,gid,values):
    dtype = "<i4" if kind==b"C" else "<f4"
    return observer.RECORD.pack(kind,start,stop,gid,len(values))+np.asarray(values,dtype=dtype).tobytes()


def trace_bytes(events, counts=None):
    data = observer.HEADER.pack(observer.MAGIC,2,3)
    for group in GROUPS:
        name = group.name.encode()
        data += observer.GROUP.pack(group.group_id,group.start,group.stop,0,group.group_id,
                                    group.start,group.stop,1,len(name))+name
    for group in GROUPS:
        data += record(b"R",0,0,group.group_id,np.arange(group.start,group.stop)+1)
    for t in range(2):
        for group in GROUPS:
            data += record(b"A",t,t+1,group.group_id,np.arange(group.start,group.stop)+1)
        for group in GROUPS:
            values = ([sum(a==t and b==i for a,b in events[group.name]) for i in range(group.units)]
                      if counts is None else counts[t][group.start:group.stop])
            data += record(b"C",t,t+1,group.group_id,values)
    return data+observer.RECORD.pack(b"E",2,2,-1,0)


def refresh_bindings(root):
    manifest = json.loads((root/"manifest.json").read_text())
    for row in manifest["files"]:
        row["sha256"] = sha256_file(root/row["path"])
    write_json(root/"manifest.json",manifest)
    native = json.loads((root/"native_report.json").read_text())
    native["runtime_manifest_sha256"] = sha256_file(root/"manifest.json")
    if (root/"native_observation.bin").exists():
        native["native_observation_sha256"] = sha256_file(root/"native_observation.bin")
        native["native_observation_bytes"] = (root/"native_observation.bin").stat().st_size
    write_json(root/"native_report.json",native)
    recording = json.loads((root/"observations/recording.json").read_text())
    recording["native_report_sha256"] = sha256_file(root/"native_report.json")
    write_json(root/"observations/recording.json",recording)


def add_memory_overlay(root, build, paths, *, observed, originals=None):
    """Use the real two overlay APIs; no native compiler or simulator is called."""
    from . import ca3_conductance_memory as memory

    originals = originals or {name:(replay.PINNED_SOURCE_ROOT/name).read_bytes()
                              for name in observer.PINNED_FILES}
    original_root = root.parent/(root.name+"_memory_original")
    staging_root = root.parent/(root.name+"_memory_staging")
    for directory in (original_root, staging_root):
        for name, data in originals.items():
            path = directory/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data)
    preceding = observer.apply_overlay(original_root,staging_root) if observed else None
    build["conductance_memory_overlay"] = memory.apply_overlay(
        original_root,staging_root,preceding_overlay=preceding)
    implementation = Path(memory.__file__).read_bytes()
    build["conductance_memory_implementation_sha256"] = sha256_file(Path(memory.__file__))
    path = root/replay.MEMORY_ADAPTER_PATH
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(implementation)
    paths.append(replay.MEMORY_ADAPTER_PATH)
    if observed:
        name = "observer_source/"+memory.GPU_FILE
        target = root/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((staging_root/memory.GPU_FILE).read_bytes())
        if name not in paths:
            paths.append(name)


@pytest.fixture
def pair_factory(tmp_path,monkeypatch):
    # Only dimensions, duration and the pattern digest are scaled for fixtures;
    # production constants and source/overlay cryptographic checks stay fixed.
    monkeypatch.setattr(readouts,"CORE_UNITS",3)
    monkeypatch.setattr(readouts,"POPULATIONS",{"A":2,"B":1})
    monkeypatch.setattr(readouts,"DURATION_MS",2)
    monkeypatch.setattr(replay,"DURATION_MS",2)
    pinned = {p:(replay.PINNED_SOURCE_ROOT/p).read_bytes() for p in observer.PINNED_FILES}
    overlays = observer.overlay_bytes(pinned)
    overlay_hashes = replay._expected_overlay()
    pattern = "0\n"*1100

    def make(root,observed,events,counts=None,truncate=False,memory_overlay=False):
        (root/"native_run/results").mkdir(parents=True)
        (root/"observations").mkdir()
        for path in (*replay.REBUILT_ARTIFACTS,replay.SHARED_RUNTIME):
            target=root/path;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((path+(" observed" if observed and path!=replay.SHARED_RUNTIME else " original")).encode())
        (root/"trainPat.csv").write_text(pattern)
        (root/"native_run/trainPat.csv").write_text(pattern)
        pattern_sha=sha256_file(root/"trainPat.csv")
        monkeypatch.setattr(readouts,"PATTERN_SHA256",pattern_sha)
        payload=NETWORK_HEADER.pack(NETWORK_SIGNATURE,.3,45.849998474121094,1.,3,2)
        for group in GROUPS:
            payload+=GROUP_HEADER.pack(group.start,group.stop-1,*group.grid,group.name.encode())
        (root/"native_run"/readouts.INPUT_NETWORK).write_bytes(payload)
        proofs=[]
        for path,preserved,binding,old,new in [
            (readouts.MAIN_PATH,False,True,readouts.AUTHOR_MAIN_SHA256,readouts.OVERLAY_SHA256),
            ("generateCONFIGStateSTP.h",True,False,readouts.CONFIG_SHA256,readouts.CONFIG_SHA256),
            ("generateSETUPStateSTP.h",True,False,readouts.SETUP_SHA256,readouts.SETUP_SHA256)]:
            proofs.append(dict(path=path,source_bytes_preserved=preserved,source_path_binding_only=binding,
                               source_sha256=old,staged_sha256=new))
        build=dict(status="NATIVE_RETRIEVAL_SOURCE_BUILD_PASSED_NORMALIZED65P_EXECUTION_PENDING",
            scope="native_source_retrieval",expected_duration_ms=2,framework_revision=observer.REVISION,
            author_revision=readouts.AUTHOR_REVISION,core_neurons=3,populations={"A":2,"B":1},
            pattern_sha256=pattern_sha,pattern_integer_count=1100,integration="RUNGE_KUTTA4_5",
            delay_ms=1,source_rng={"source_seed":10},
            original_author_main_sha256=readouts.AUTHOR_MAIN_SHA256,overlay_source_sha256=readouts.OVERLAY_SHA256,
            source_file_byte_checks=proofs,retrieval_program_changed_only_in_input_path=True,
            binary_sha256=sha256_file(root/"bin/ca3_retrieval"),
            retrieval_input_contract=dict(effective_input_binding=readouts.INPUT_NETWORK,
                source_path_binding_only=True,source_carlsim_seed=10,source_host_srand_seed=3,source_estdp_enabled=True))
        paths=["build_report.json","trainPat.csv",*replay.REBUILT_ARTIFACTS,replay.SHARED_RUNTIME]
        if observed:
            build["observer_implementation_sha256"]=sha256_file(Path(observer.__file__))
            build["native_observation_overlay"]=dict(framework_revision=observer.REVISION,
                enabled_by=observer.ENVIRONMENT_VARIABLE,original_sha256=observer.PINNED_FILES.copy(),
                overlay_sha256=overlay_hashes,current_units=observer.CURRENT_UNIT.copy(),
                counter_source="native nSpikeCnt",count_interval_ms=1,accumulated_count_dtype="int64",
                current_source="runtime extCurrent; GPU device-to-host or CPU direct",
                native_build_verified=True,native_execution_verified=False)
            (root/replay.OBSERVER_ADAPTER_PATH).parent.mkdir(parents=True,exist_ok=True)
            (root/replay.OBSERVER_ADAPTER_PATH).write_bytes(Path(observer.__file__).read_bytes())
            paths.append(replay.OBSERVER_ADAPTER_PATH)
            for relative,data in overlays.items():
                path="observer_source/"+relative
                (root/path).parent.mkdir(parents=True,exist_ok=True)
                (root/path).write_bytes(data);paths.append(path)
        else:
            (root/"unused_normalization_helper.py").write_text("# historical helper\n")
            paths.append("unused_normalization_helper.py")
        if memory_overlay:
            add_memory_overlay(root,build,paths,observed=observed,originals=pinned)
        write_json(root/"build_report.json",build)
        write_json(root/"manifest.json",dict(files=[dict(path=p,sha256=sha256_file(root/p)) for p in paths]))
        native=deepcopy(build)
        native.update(status="NATIVE_RETRIEVAL_EXIT_ZERO",exit_code=0,native_run_executed=True,
            retrieval_executed=True,normalized_input_qualified=True,
            normalized_input_sha256=sha256_file(root/"native_run"/readouts.INPUT_NETWORK),
            runtime_manifest_sha256=sha256_file(root/"manifest.json"),
            native_pid=2 if observed else 1,slurm_job_id="test_observed" if observed else "test_original",
            started_utc="2026-09-29T01:01:00Z" if observed else "2026-09-29T01:00:00Z",
            finished_utc="2026-09-29T01:01:01Z" if observed else "2026-09-29T01:00:01Z")
        for field in replay.INPUT_FIELDS:
            if field!="normalized_input_sha256":native[field]="a"*64
        if observed:
            data=trace_bytes(events,counts)
            (root/"native_observation.bin").write_bytes(data[:-observer.RECORD.size] if truncate else data)
            native.update(native_observation_stream_validated=True,
                native_observation_sha256=sha256_file(root/"native_observation.bin"),
                native_observation_bytes=(root/"native_observation.bin").stat().st_size,
                observed_native_duration_ms=2,host_request_runtime_current_equal=True)
            native["native_observation_overlay"]["native_execution_verified"]=True
        write_json(root/"native_report.json",native)
        unit=np.zeros(3,dtype=np.int64);population=np.zeros((2,2),dtype=np.int64);files=[]
        for group in GROUPS:
            relative="native_run/results/spk_"+group.name+".dat"
            rows=np.asarray(events[group.name],dtype="<i4").reshape(-1,2)
            (root/relative).write_bytes(SPIKE_HEADER.pack(SPIKE_SIGNATURE,.2,*group.grid)+rows.tobytes())
            if len(rows):
                np.add.at(unit,rows[:,1]+group.start,1);np.add.at(population[:,group.group_id],rows[:,0],1)
            files.append(dict(group=group.name,relative_path=relative,sha256=sha256_file(root/relative),
                records=len(rows),first_event_ms=int(rows[:,0].min()) if len(rows) else None,
                last_event_ms=int(rows[:,0].max()) if len(rows) else None))
        np.savez_compressed(root/"observations/population_counts.npz",unit_ids=np.arange(3),
            unit_counts=unit,population_counts_per_ms=population,group_names=np.asarray([g.name for g in GROUPS]))
        write_json(root/"observations/recording.json",dict(status="DECODED_NATIVE_RETRIEVAL_FILES",
            duration_ms=2,time_resolution_ms=1,units=3,retrieval_executed=True,
            groups=[dict(asdict(g),grid=list(g.grid)) for g in GROUPS],files=files,records=int(unit.sum()),
            native_report_sha256=sha256_file(root/"native_report.json"),
            observation_sha256=sha256_file(root/"observations/population_counts.npz"),
            implementation_sha256=sha256_file(Path(replay.__file__).with_name("carlsim_records.py"))))
        return root

    def pair(*,observed_events=None,counts=None,truncate=False,memory_overlay=False):
        return (make(tmp_path/"original",False,EVENTS,memory_overlay=memory_overlay),
                make(tmp_path/"observed",True,observed_events or EVENTS,counts,truncate,memory_overlay))
    return pair


def run_rejected(tmp_path,left,right,match):
    output=tmp_path/"failure.json"
    with pytest.raises(ValueError,match=match):
        replay.compare(left,right,output)
    result=json.loads(output.read_text())
    assert result["status"]=="FAILED_NATIVE_CA3_OBSERVATION_REPLAY"
    assert not result["native_internal_spike_counter_verified"]
    assert not result["actual_current_trace_verified"]
    return result


def test_full_binding_multisets_multiplicity_and_distinct_file_order(pair_factory,tmp_path):
    events=deepcopy(EVENTS);events["A"].reverse()
    left,right=pair_factory(observed_events=events)
    result=replay.compare(left,right,tmp_path/"pass.json")
    assert result["native_internal_spike_counter_verified"] and result["actual_current_trace_verified"]
    assert result["unit_millisecond_counts_compared"]==6 and result["native_intervals_compared"]==2
    assert result["events_compared"]==4 and result["event_multisets_equal"]
    assert result["original_recording_sha256"]==sha256_file(left/"observations/recording.json")
    assert not result["event_order_equal"] and not result["event_file_bytes_equal"]
    assert result["exact_binary_fresh_replay_contract_unchanged"]
    assert not result["pattern_completion_qualified"] and result["production_episodes"]==0


def test_memory_corrected_reference_and_observer_bind_exact_source_layers(pair_factory,tmp_path):
    left,right=pair_factory(memory_overlay=True)
    result=replay.compare(left,right,tmp_path/"memory_pass.json")
    original,observed=(result["source_overlay_verification"][role] for role in ("original","observed"))
    assert original["memory_receipt_and_implementation_verified"]
    assert not original["gpu_source_snapshot_verified"]  # Existing reference archives omit this snapshot.
    assert observed["memory_receipt_and_implementation_verified"] and observed["gpu_source_snapshot_verified"]
    metadata=json.loads((right/"native_report.json").read_text())
    memory=metadata["conductance_memory_overlay"]
    assert original["original_gpu_sha256"]==memory["original_sha256"][replay.GPU_SOURCE_FILE]
    assert observed["overlay_input_gpu_sha256"]==memory["input_sha256"][replay.GPU_SOURCE_FILE]
    assert observed["final_gpu_sha256"]==memory["overlay_sha256"][replay.GPU_SOURCE_FILE]
    assert result["bound_overlay_sha256"][replay.GPU_SOURCE_FILE]==observed["final_gpu_sha256"]
    assert result["observer_intermediate_sha256"]==metadata["native_observation_overlay"]["overlay_sha256"]
    assert result["helper_implementation_sha256"]["ca3_conductance_memory"]==metadata["conductance_memory_implementation_sha256"]
    assert result["unit_millisecond_counts_compared"]==6 and result["event_multisets_equal"]
    assert not result["scientific_fidelity_qualified"]


@pytest.mark.parametrize("field",["original_sha256","input_sha256","overlay_sha256",
                                  "initialization_correction","rise_buffer_allocation"])
def test_memory_metadata_is_reconstructed_not_trusted(pair_factory,tmp_path,field):
    left,right=pair_factory(memory_overlay=True)
    for name in ("build_report.json","native_report.json"):
        value=json.loads((right/name).read_text());memory=value["conductance_memory_overlay"]
        if field=="initialization_correction":
            memory[field]["corrected_extent"]="syn_gPitch * syn_gLength"
        elif field=="rise_buffer_allocation":
            del memory[field]["GABAb_r"]
        else:
            memory[field][replay.GPU_SOURCE_FILE]="0"*64
        write_json(right/name,value)
    refresh_bindings(right)
    run_rejected(tmp_path,left,right,"EXACT_MEMORY_OVERLAY_RECEIPT")


@pytest.mark.parametrize("change,match",[
    ("gpu_changed","MEMORY_GPU_SOURCE_IDENTITY"),
    ("gpu_missing","MEMORY_GPU_SOURCE_IDENTITY"),
    ("helper_changed","MEMORY_MANIFEST_ADAPTER_IDENTITY"),
    ("helper_missing","MEMORY_MANIFEST_ADAPTER_IDENTITY"),
    ("implementation_receipt","MEMORY_IMPLEMENTATION_IDENTITY"),
])
def test_memory_final_source_and_implementation_are_manifest_bound(pair_factory,tmp_path,change,match):
    left,right=pair_factory(memory_overlay=True)
    name=("observer_source/"+replay.GPU_SOURCE_FILE if change.startswith("gpu") else replay.MEMORY_ADAPTER_PATH)
    if change.endswith("changed"):
        path=right/name;path.write_bytes(path.read_bytes()+b"\n// rebound but unexpected source\n")
    elif change.endswith("missing"):
        manifest=json.loads((right/"manifest.json").read_text())
        manifest["files"]=[row for row in manifest["files"] if row["path"]!=name]
        write_json(right/"manifest.json",manifest)
    else:
        for filename in ("build_report.json","native_report.json"):
            value=json.loads((right/filename).read_text())
            value["conductance_memory_implementation_sha256"]="0"*64
            write_json(right/filename,value)
    refresh_bindings(right)
    run_rejected(tmp_path,left,right,match)


def test_memory_final_hash_cannot_replace_observer_intermediate_receipt(pair_factory,tmp_path):
    left,right=pair_factory(memory_overlay=True)
    for name in ("build_report.json","native_report.json"):
        value=json.loads((right/name).read_text())
        value["native_observation_overlay"]["overlay_sha256"].update(value["conductance_memory_overlay"]["overlay_sha256"])
        write_json(right/name,value)
    refresh_bindings(right)
    run_rejected(tmp_path,left,right,"EXACT_OVERLAY_RECEIPT")


def test_memory_build_and_native_metadata_must_match(pair_factory,tmp_path):
    left,right=pair_factory(memory_overlay=True)
    value=json.loads((right/"native_report.json").read_text())
    value["conductance_memory_overlay"]["native_qualification"]=True
    write_json(right/"native_report.json",value);refresh_bindings(right)
    run_rejected(tmp_path,left,right,"MEMORY_BUILD_NATIVE_RECEIPTS_DIFFER")


def test_memory_corrected_and_uncorrected_references_cannot_be_mixed(pair_factory,tmp_path):
    left,right=pair_factory(memory_overlay=True)
    for name in ("build_report.json","native_report.json"):
        value=json.loads((left/name).read_text())
        del value["conductance_memory_overlay"]
        del value["conductance_memory_implementation_sha256"]
        write_json(left/name,value)
    refresh_bindings(left)
    run_rejected(tmp_path,left,right,"MEMORY_OVERLAY_PRESENCE_DIFFERS")


def test_supplied_reference_gpu_snapshot_must_match_memory_receipt(pair_factory,tmp_path):
    left,right=pair_factory(memory_overlay=True)
    name="observer_source/"+replay.GPU_SOURCE_FILE;path=left/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((replay.PINNED_SOURCE_ROOT/replay.GPU_SOURCE_FILE).read_bytes())
    manifest=json.loads((left/"manifest.json").read_text())
    manifest["files"].append(dict(path=name,sha256=sha256_file(path)))
    write_json(left/"manifest.json",manifest);refresh_bindings(left)
    run_rejected(tmp_path,left,right,"MEMORY_GPU_SOURCE_IDENTITY")


def test_memory_source_reconstruction_rejects_changed_pinned_original(pair_factory,tmp_path,monkeypatch):
    left,right=pair_factory(memory_overlay=True)
    copied=tmp_path/"changed_pinned"
    for name in observer.PINNED_FILES:
        path=copied/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((replay.PINNED_SOURCE_ROOT/name).read_bytes())
    path=copied/replay.GPU_SOURCE_FILE;path.write_bytes(path.read_bytes()+b"\n")
    monkeypatch.setattr(replay,"PINNED_SOURCE_ROOT",copied)
    run_rejected(tmp_path,left,right,"PINNED_SOURCE_CHANGED")


def test_time_unit_reassignment_with_identical_marginals_is_rejected(pair_factory,tmp_path):
    # Both neuron totals and population-per-ms totals remain exactly unchanged.
    left,right=pair_factory(counts=np.array([[0,1,2],[1,0,0]],dtype=np.int64))
    result=run_rejected(tmp_path,left,right,"PER_UNIT_MILLISECOND_COUNTER_MISMATCH")
    assert result["first_counter_mismatch"]["start_ms"]==0
    assert result["first_counter_mismatch"]["differing_units"]==2


@pytest.mark.parametrize("change",["missing","extra"])
def test_event_multiplicity_change_is_rejected(pair_factory,tmp_path,change):
    events=deepcopy(EVENTS)
    if change=="missing":events["B"].pop()
    else:events["B"].append((0,0))
    left,right=pair_factory(observed_events=events)
    run_rejected(tmp_path,left,right,"REFERENCE_EVENT_MULTISET_MISMATCH")


def test_incomplete_trace_with_rebound_digest_is_rejected(pair_factory,tmp_path):
    left,right=pair_factory(truncate=True)
    run_rejected(tmp_path,left,right,"OBSERVATION_TRUNCATED")


@pytest.mark.parametrize("change,match",[
    ("author","BOUND_SOURCE_PROGRAM"),
    ("adapter","ADAPTER_IDENTITY"),
    ("overlay","EXACT_OVERLAY"),
    ("parameters","REFERENCE_SOURCE_OR_INPUT"),
    ("curand","CURAND_RUNTIME"),
    ("patched_file","PATCHED_SOURCE_IDENTITY"),
])
def test_source_adapter_overlay_and_runtime_mismatches(pair_factory,tmp_path,change,match):
    left,right=pair_factory()
    for name in ["build_report.json","native_report.json"]:
        value=json.loads((right/name).read_text())
        if change=="author":value["source_file_byte_checks"][1]["staged_sha256"]="0"*64
        elif change=="adapter":value["observer_implementation_sha256"]="0"*64
        elif change=="overlay":value["native_observation_overlay"]["original_sha256"]={}
        elif change=="parameters":value["source_rng"]={"source_seed":11}
        write_json(right/name,value)
    if change=="curand":(right/replay.SHARED_RUNTIME).write_bytes(b"changed runtime")
    if change=="patched_file":
        manifest=json.loads((right/"manifest.json").read_text())
        manifest["files"]=[r for r in manifest["files"] if r["path"]!="observer_source/"+observer.GENERATED_HEADER]
        write_json(right/"manifest.json",manifest)
    refresh_bindings(right)
    run_rejected(tmp_path,left,right,match)


def test_trace_digest_mismatch_is_rejected(pair_factory,tmp_path):
    left,right=pair_factory()
    path=right/"native_observation.bin";path.write_bytes(path.read_bytes()+b"changed")
    run_rejected(tmp_path,left,right,"TRACE_DIGEST_OR_SIZE")


def test_cli_failure_keeps_receipt_and_returns_nonzero(tmp_path):
    output=tmp_path/"cli_failure.json"
    cmd=[sys.executable,"-m","population_dynamics.ca3_native_observation_replay",
         "--original",str(tmp_path/"missing"),"--observed",str(tmp_path/"other"),"--output",str(output)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode==2
    assert json.loads(output.read_text())["status"]=="FAILED_NATIVE_CA3_OBSERVATION_REPLAY"
