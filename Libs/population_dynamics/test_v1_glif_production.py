"""Disposable GPU production routing fixtures; no scientific qualification."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from . import benchmark_production as production
from . import benchmark_production_block as blocks
from . import v1_production_execution as execution
from . import v1_glif_production as glif
from .provenance import sha256_file
from .test_benchmark_production import fixture_configuration
from .test_v1_production_execution import resources, archive, Scheduler, prepare
from .test_v1_glif_network import cache_fixture


@pytest.fixture(scope="module")
def gpu_cohort(tmp_path_factory):
    root=tmp_path_factory.mktemp("glif_cohort")
    config,qualification,table=fixture_configuration(root,"v1_visual")
    software=root/"software";software.mkdir()
    names=("v1_glif_network.py","v1_glif_component_emulator.py","v1_glif_cuda.py","v1_nest_time_semantics.py")
    for name in names:
        (software/name).write_text("# synthetic routing fixture "+name+"\n")
    library=root/"library.so";library.write_bytes(b"synthetic routing library")
    detector=root/"resources.py";detector.write_text("# synthetic detector\n")
    proof=json.loads(qualification.read_text())
    proof["source_recipes"]["glif_cuda"]=dict(
        runtime_code={name:sha256_file(software/name) for name in names},
        library_sha256=sha256_file(library),software_directory=str(software),
        component_rtol=1e-12,component_atol=1e-12)
    qualification.write_text(json.dumps(proof))
    manifest=table.with_suffix(".json");frozen=json.loads(manifest.read_text())
    frozen["source_qualification_sha256"]=sha256_file(qualification)
    manifest.write_text(json.dumps(frozen))
    return production.ProductionCohort.load(config,"v1_visual",qualification,table)


def gpu_resources(root,cohort,*,reuse):
    path,document=resources(root,cohort,reuse=reuse)
    identity=cohort.identity(0,1,0)
    recurrent=json.loads((Path(document["recurrent_source"])/"primary/report.json").read_text())
    inputs=json.loads((Path(document["input_source"])/"primary/report.json").read_text())
    native=dict(original_config_sha256=recurrent["original_config_sha256"],cortical_units=230924,
        native_nest_simulation=True,native_parameters_and_initial_state_exact=True,
        execution_clock=dict(resolution_ms=.25,duration_ms=3000.,tics_per_ms=1000.,tics_per_step=250),
        connections=dict(source_batch_sha256="synthetic-matrix",native_connections_to_cortex=3),
        bindings={role:dict(files={"primary/report.json":dict(sha256=sha256_file(Path(document[source])/"primary/report.json"))})
                  for role,source in (("recurrent","recurrent_source"),("inputs","input_source"))})
    reference=archive(root/"circuit_reference","synthetic",{
        "primary/report.json":native,"recurrent/primary/report.json":recurrent,"inputs_reference/primary/report.json":inputs})
    snapshot=root/"snapshot";snapshot.mkdir()
    (snapshot/"report.json").write_text(json.dumps(dict(status="PASS_COMPLETE_NATIVE_CONNECTION_SNAPSHOT",
        native_report_sha256=sha256_file(reference/"primary/report.json"),
        snapshot=dict(source_batch_sha256="synthetic-matrix",rows=3),native_simulation_calls=0)))
    backend=cohort.qualification["source_recipes"]["glif_cuda"]
    validation=root/"circuit_validation.json"
    validation.write_text(json.dumps(dict(status="PASS_NATIVE_GLIF_CUDA_COUPLED_DIAGNOSTIC",
        all_spike_pairs_exact=True,all_unit_counts_exact=True,terminal_voltage_within_component_tolerance=True,
        source_bindings=dict(files={"primary/report.json":dict(sha256=sha256_file(reference/"primary/report.json"))}),
        runtime_code=backend["runtime_code"],cuda_execution=dict(library_sha256=backend["library_sha256"]),
        component_rtol=1e-12,component_atol=1e-12)))
    document["backend"]="glif_cuda"
    document["memory_gib"]=dict(rates=4,spikes=4,gpu=40,export=8)
    document["gpu"]=dict(node="cpu06",partition="cpu",
        library=str(cohort.qualification_path.parent/"library.so"),
        detector=str(cohort.qualification_path.parent/"resources.py"),
        software=backend["software_directory"],circuit_reference=str(reference),
        snapshot=str(snapshot),circuit_validation=str(validation))
    path.write_text(yaml.safe_dump(document,sort_keys=False))
    return path,document


class GpuScheduler(Scheduler):
    def __init__(self,available=True):
        super().__init__();self.available=available;self.preflights=0

    def __call__(self,command,**kwargs):
        if command[0]=="sinfo":
            self.preflights+=1
            return subprocess.CompletedProcess(command,0,"cpu06|cpu|mix\n" if self.available else "cpu06|cpu|drng\n","")
        return super().__call__(command,**kwargs)


@pytest.mark.parametrize("reuse,expected_jobs",[(False,24),(True,14)])
@pytest.mark.parametrize("hold_gpu",[False,True])
def test_gpu_source_chain_and_independent_replay(gpu_cohort,tmp_path,monkeypatch,reuse,expected_jobs,hold_gpu):
    path,document=gpu_resources(tmp_path/"resources",gpu_cohort,reuse=reuse)
    output=tmp_path/"execution";plan=prepare(gpu_cohort,path,output)
    scheduler=GpuScheduler()
    monkeypatch.setattr(execution.subprocess,"run",scheduler)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","must_not_reach_job")
    result=execution.submit(gpu_cohort,plan,output,hold_gpu=hold_gpu)
    assert len(scheduler.calls)==expected_jobs and scheduler.preflights==1
    phases=[row["phase"] for row in result["jobs"]]
    assert phases[-4:]==["source_plan","gpu_primary","gpu_replay","export"]
    assert "native" not in phases
    for (command,options),row in zip(scheduler.calls,result["jobs"]):
        if row["phase"].startswith("gpu_"):
            assert {"--gres=gpu:1","--nodelist=cpu06","--partition=cpu"}<=set(command)
            assert command[-3:]==["source_batch",str(output/"glif_cases.json"),document["gpu"]["software"]]
            assert ("--hold" in command)==hold_gpu
        else:
            assert "--gres=none" in command and "--nodelist=cpu03" in command
            assert "--hold" not in command
        assert "CUDA_VISIBLE_DEVICES" not in options["env"]
    entries=blocks.load_block(gpu_cohort,output/"sources.json",0,1)
    assert len(entries)==10 and all(entry.backend=="glif_cuda" for entry in entries)
    assert len({entry.sources["gpu_primary"] for entry in entries})==1
    assert entries[0].sources["gpu_primary"]!=entries[0].sources["gpu_replay"]
    for job in scheduler.states:scheduler.states[job]="COMPLETED"
    assert result.get("gpu_jobs_submitted_held",False)==hold_gpu
    assert execution.submit(gpu_cohort,plan,output,resume=True,hold_gpu=hold_gpu)==result
    assert len(scheduler.calls)==expected_jobs and scheduler.preflights==1


def test_cli_can_hold_gpu_phases_while_submitting_cpu_work(gpu_cohort,tmp_path,monkeypatch,capsys):
    path,_=gpu_resources(tmp_path/"resources",gpu_cohort,reuse=True)
    scheduler=GpuScheduler();monkeypatch.setattr(execution.subprocess,"run",scheduler)
    root=gpu_cohort.qualification_path.parent
    arguments=["--repo",str(Path(__file__).resolve().parents[2]),
               "--config",str(root/"campaign.yaml"),"--qualification",str(gpu_cohort.qualification_path),
               "--identity-table",str(gpu_cohort.identity_table),"--resources",str(path),
               "--output",str(tmp_path/"execution"),"--submit","--hold-gpu","--start-after-job","620000"]
    assert execution.main(arguments)==0
    result=json.loads(capsys.readouterr().out)
    assert result["gpu_jobs_submitted_held"] is True
    assert result["start_after_job_id"]==620000
    assert "--dependency=afterok:620000" in scheduler.calls[0][0]
    assert len(scheduler.calls)==14
    held=[row["phase"] for (command,_),row in zip(scheduler.calls,result["jobs"]) if "--hold" in command]
    assert held==["gpu_primary","gpu_replay"]


@pytest.mark.parametrize("reuse",[False,True])
def test_start_dependency_preserves_internal_chain_and_resume(gpu_cohort,tmp_path,monkeypatch,reuse):
    path,_=gpu_resources(tmp_path/"resources",gpu_cohort,reuse=reuse)
    output=tmp_path/"execution";plan=prepare(gpu_cohort,path,output)
    scheduler=GpuScheduler();monkeypatch.setattr(execution.subprocess,"run",scheduler)
    result=execution.submit(gpu_cohort,plan,output,start_after_job=620000)
    previous=620000
    for (command,_),row in zip(scheduler.calls,result["jobs"]):
        assert [value for value in command if value.startswith("--dependency=")]==[
            "--dependency=afterok:"+str(previous)]
        previous=row["job_id"]
    submitted=len(scheduler.calls)
    assert execution.submit(gpu_cohort,plan,output,resume=True,start_after_job=620000)==result
    assert len(scheduler.calls)==submitted
    with pytest.raises(ValueError,match="SUBMISSION_REQUEST_CHANGED"):
        execution.submit(gpu_cohort,plan,output,resume=True,start_after_job=620001)
    assert len(scheduler.calls)==submitted


@pytest.mark.parametrize("job",[0,-1,True])
def test_invalid_start_dependency_rejects_before_submission(gpu_cohort,tmp_path,monkeypatch,job):
    path,_=gpu_resources(tmp_path/"resources",gpu_cohort,reuse=True)
    output=tmp_path/"execution";plan=prepare(gpu_cohort,path,output)
    scheduler=GpuScheduler();monkeypatch.setattr(execution.subprocess,"run",scheduler)
    with pytest.raises(ValueError,match="START_DEPENDENCY_INVALID"):
        execution.submit(gpu_cohort,plan,output,start_after_job=job)
    assert not scheduler.calls and not (output/"jobs").exists()


def test_unavailable_gpu_rejects_before_any_submission(gpu_cohort,tmp_path,monkeypatch):
    path,_=gpu_resources(tmp_path/"resources",gpu_cohort,reuse=True)
    output=tmp_path/"execution";plan=prepare(gpu_cohort,path,output)
    scheduler=GpuScheduler(available=False);monkeypatch.setattr(execution.subprocess,"run",scheduler)
    with pytest.raises(ValueError,match="GPU_NODE_NOT_AVAILABLE"):
        execution.submit(gpu_cohort,plan,output)
    assert not scheduler.calls and not (output/"jobs").exists()


def test_missing_circuit_validation_rejects_preparation(gpu_cohort,tmp_path):
    path,document=gpu_resources(tmp_path/"resources",gpu_cohort,reuse=True)
    proof=Path(document["gpu"]["circuit_validation"])
    value=json.loads(proof.read_text());value["all_unit_counts_exact"]=False
    proof.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="CIRCUIT_VALIDATION_REQUIRED"):
        prepare(gpu_cohort,path,tmp_path/"execution")
    assert not (tmp_path/"execution").exists()


def test_glif_input_binding_uses_all_production_identities(gpu_cohort,tmp_path):
    conditions=[]
    for index in range(10):
        identity=gpu_cohort.identity(0,1,index)
        root=archive(tmp_path/f"inputs{index}","synthetic",{"visual_inputs/report.json":dict(
            exit_code=0,complete_lgn_population=True,config=gpu_cohort.v1_visual_config(identity))})
        conditions.append(dict(condition_index=index,spike_source=str(root)))
    path=tmp_path/"inputs.json";path.write_text(json.dumps(dict(schema=blocks.GLIF_INPUT_SCHEMA,
        circuit_index=0,block_index=1,conditions=conditions)))
    output=tmp_path/"cases.json"
    result=blocks.prepare_glif_inputs(gpu_cohort,path,0,1,output)
    cases=json.loads(output.read_text())["cases"]
    assert result["cases"]==10 and len({row["input_archive"] for row in cases})==10
    assert [row["identity"] for row in cases]==[gpu_cohort.identity(0,1,index) for index in range(10)]
    wrong=Path(conditions[-1]["spike_source"])
    value=json.loads((wrong/"visual_inputs/report.json").read_text());value["config"]["background"]["starting_noise_seed"]+=1
    archive(wrong,"synthetic",{"visual_inputs/report.json":value})
    with pytest.raises(ValueError,match="INPUT_RECIPE_DIFFERS"):
        blocks.prepare_glif_inputs(gpu_cohort,path,0,1,tmp_path/"rejected.json")
    assert not (tmp_path/"rejected.json").exists()

def glif_admission_fixture(cohort, identity, native_admission):
    import hashlib
    admission=deepcopy(native_admission)
    backend=cohort.qualification["source_recipes"]["glif_cuda"]
    def document(value):
        payload=json.dumps(value)
        return dict(payload=payload,sha256=hashlib.sha256(payload.encode()).hexdigest())
    construction=json.loads(admission["source_documents"]["recurrent"]["payload"])
    reference=dict(original_config_sha256=construction["original_config_sha256"],cortical_units=230924,
        execution_clock=dict(duration_ms=3000.,resolution_ms=.25,tics_per_ms=1000.,tics_per_step=250),
        connections=dict(native_connections_to_cortex=3))
    reference_doc=document(reference)
    validation=dict(status="PASS_NATIVE_GLIF_CUDA_COUPLED_DIAGNOSTIC",
        all_spike_pairs_exact=True,all_unit_counts_exact=True,terminal_voltage_within_component_tolerance=True,
        source_bindings=dict(files={"primary/report.json":dict(sha256=reference_doc["sha256"])}),
        runtime_code=deepcopy(backend["runtime_code"]),cuda_execution=dict(library_sha256=backend["library_sha256"]),
        component_rtol=1e-12,component_atol=1e-12)
    validation_doc=document(validation)
    executions=[];replays=[];rows=[]
    for index in range(10):
        case=cohort.identity(identity["circuit_index"],identity["block_index"],index)
        executed=dict(status="PASS_SOURCE_INPUT_GLIF_CUDA_EXECUTION",job_id="primary",
            runtime_code=deepcopy(backend["runtime_code"]),
            cuda_execution=dict(library_sha256=backend["library_sha256"]),
            execution_clock=reference["execution_clock"],units=230924,connection_rows=3,
            input_bindings=dict(config=cohort.v1_visual_config(case)),
            circuit_bindings=dict(files={"primary/report.json":dict(sha256=reference_doc["sha256"])}))
        first=document(executed);second=document(dict(executed,job_id="replay"))
        executions.append(first);replays.append(second)
        rows.append(dict(identity=case,status="PASS",primary_report_sha256=first["sha256"],
            replay_report_sha256=second["sha256"],all_spike_pairs_exact=True,
            all_exact_checks={"input_and_state":True},terminal_voltage_within_component_tolerance=True,
            library_sha256=backend["library_sha256"]))
    comparison=document(dict(status="PASS_GLIF_SOURCE_BLOCK_REPLAY",passed_conditions=10,failed_conditions=0,
        primary_job_id="primary",replay_job_id="replay",runtime_code=deepcopy(backend["runtime_code"]),
        component_rtol=1e-12,component_atol=1e-12,conditions=rows))
    first,second=executions[identity["condition_index"]],replays[identity["condition_index"]]
    admission.update(execution_backend="glif_cuda",construction_config_sha256=reference["original_config_sha256"],
        circuit_reference_report_sha256=reference_doc["sha256"],circuit_validation_sha256=validation_doc["sha256"],
        execution_report_sha256=first["sha256"],replay_report_sha256=second["sha256"],
        replay_comparison_sha256=comparison["sha256"])
    admission["source_documents"].update(circuit_reference=reference_doc,circuit_validation=validation_doc,
        execution=first,replay=second,replay_comparison=comparison)
    return admission


def replace_admission_document(admission,role,digest_key,update):
    import hashlib
    value=json.loads(admission["source_documents"][role]["payload"])
    update(value)
    payload=json.dumps(value);digest=hashlib.sha256(payload.encode()).hexdigest()
    admission["source_documents"][role]=dict(payload=payload,sha256=digest)
    admission[digest_key]=digest


@pytest.mark.parametrize("failure",["runtime","job_identity","failed_condition"])
def test_glif_admission_checks_runtime_and_independent_complete_replay(gpu_cohort,tmp_path,failure):
    from .test_benchmark_production import source_fixture
    identity=gpu_cohort.identity(0,1,1)
    _,_,native=source_fixture(tmp_path/"source",gpu_cohort,identity)
    admission=glif_admission_fixture(gpu_cohort,identity,native)
    production.verify_source_admission(gpu_cohort,identity,admission)
    if failure=="runtime":
        replace_admission_document(admission,"execution","execution_report_sha256",
            lambda doc:doc["runtime_code"].update({"v1_glif_network.py":"f"*64}))
        expected="GLIF_EXECUTION_DIFFERS"
    elif failure=="job_identity":
        replace_admission_document(admission,"replay_comparison","replay_comparison_sha256",
            lambda doc:doc.update(replay_job_id="unrelated"))
        expected="GLIF_COMPLETE_REPLAY_REQUIRED"
    else:
        replace_admission_document(admission,"replay_comparison","replay_comparison_sha256",
            lambda doc:doc["conditions"][1].update(terminal_voltage_within_component_tolerance=False))
        expected="GLIF_REPLAY_CASE_DIFFERS"
    with pytest.raises(ValueError,match=expected):
        production.verify_source_admission(gpu_cohort,identity,admission)


def test_glif_writer_selects_requested_episode_from_block(gpu_cohort,tmp_path):
    from dataclasses import replace
    from .test_benchmark_production import source_fixture
    from .benchmark_dataset import read_circuit,read_episode
    from .circuit_data import CircuitArchiveWriter,CircuitDataset,SCHEMA
    identities=[gpu_cohort.identity(0,1,index) for index in range(2)]
    episodes=[];admissions=[];units=None
    for index,identity in enumerate(identities):
        source,episode,native=source_fixture(tmp_path/("native"+str(index)),gpu_cohort,identity)
        units=read_circuit(source,episode.circuit_id)
        admission=glif_admission_fixture(gpu_cohort,identity,native)
        record=dict(episode.source_record,execution_backend="source_derived_float64_cuda",
            native_nest_simulation=False,execution_report_sha256=admission["execution_report_sha256"],
            circuit_reference_report_sha256=admission["circuit_reference_report_sha256"])
        episodes.append(replace(episode,episode_id="source"+str(index),source_record=record))
        admissions.append(admission)
    block=tmp_path/"block"
    with CircuitArchiveWriter(block,dict(schema=SCHEMA,scope="source_reproduction",scenario_id="v1_visual",
        source_revision=units.source_revision,sample_unit="complete_mechanistic_episode",
        execution_backend="source_derived_float64_cuda")) as writer:
        writer.register_circuit(units)
        for episode in episodes:writer.add(episode)
    source=CircuitDataset(block,splits=("development",))
    destination=tmp_path/"production"
    with pytest.raises(ValueError,match="ONE_FRESH_SOURCE_EPISODE_REQUIRED"):
        production.write_episode(gpu_cohort,identities[1],source,destination,admission=admissions[1])
    assert not destination.exists()
    result=production.write_episode(gpu_cohort,identities[1],source,destination,
        admission=admissions[1],source_episode_id="source1")
    assert result["identity"]==identities[1]
    dataset=CircuitDataset(destination,splits=("train",))
    stored=read_episode(dataset,dataset.episodes[identities[1]["episode_id"]])
    assert stored.source_record["native_identity"]["episode_id"]=="source1"
    assert stored.source_record["execution_backend"]=="source_derived_float64_cuda"
    assert stored.task_outcome=="failed_synthetic_target"
    assert stored.control_values.tolist()==episodes[1].control_values.tolist()
    assert production.completed_episode(gpu_cohort,identities[1],destination,resume=True)["status"]=="REUSED_COMMITTED_PRODUCTION_EPISODE"


def saved_reference_fixture(root):
    identity,original,data,_=cache_fixture()
    data["cortical_spikes"]=dict(native_id=np.array([8,3,3]),time_ms=np.array([.25,.5,.75]))
    original.update(status="synthetic_native",exit_code=0,native_nest_simulation=True,
        native_parameters_and_initial_state_exact=True,cortical_units=2,original_config_sha256="c"*64,
        bindings={role:dict(files={"primary/report.json":dict(sha256=digest*64)})
                  for role,digest in (("recurrent","d"),("inputs","e"))})
    native=archive(root/"native","synthetic",{"primary/report.json":original,
                                                  "primary/native/parameters.json":{}})
    candidate=root/"execution";case=candidate/"coupled/condition_00";case.mkdir(parents=True)
    report=dict(status=glif.CASE_PASS,job_id="synthetic_gpu",units=2,connection_rows=3,
        execution_clock=identity.clock,cortical_outcomes_used_for_execution=False,circuit_reference=str(native),
        circuit_bindings=dict(files={"primary/report.json":dict(sha256=sha256_file(native/"primary/report.json"))}),
        runtime_code={"synthetic.py":"a"*64},cuda_execution=dict(library_sha256="b"*64))
    (case/"report.json").write_text(json.dumps(report))
    for name,arrays in data.items():
        np.savez(native/f"primary/native/{name}.npz",**arrays)
        changed={key:value.copy() for key,value in arrays.items()}
        if name=="cortical_state": changed["terminal_vm_mV"][0]+=1e-14
        if name=="cortical_spikes": changed={key:value[::-1] for key,value in changed.items()}
        np.savez(case/(name+".npz"),**changed)
    native_names=("primary/report.json","primary/native/parameters.json",
                  *("primary/native/"+name for name in glif.SOURCE_ARRAYS))
    candidate_names=tuple("coupled/condition_00/"+name for name in ("report.json",*glif.SOURCE_ARRAYS))
    write_saved_delivery(native,native_names,"synthetic_native")
    write_saved_delivery(candidate,candidate_names,"synthetic_gpu")
    return native,candidate,candidate_names


def write_saved_delivery(root,names,job):
    (root/"delivery.json").write_text(json.dumps(dict(exit_code=0,job_id=job,
        files=[dict(path=name,bytes=(root/name).stat().st_size,sha256=sha256_file(root/name)) for name in names])))


def test_saved_native_reference_uses_existing_arrays_and_tolerance(tmp_path,monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID","synthetic_reader")
    native,candidate,_=saved_reference_fixture(tmp_path)
    report=glif.validate_saved_reference(native,candidate,0,tmp_path/"comparison")
    assert report["status"] in production.GLIF_CIRCUIT_REFERENCE_STATUSES
    assert report["all_spike_pairs_exact"] and report["all_unit_counts_exact"]
    assert not report["terminal_voltage_bitwise_equal"]
    assert report["terminal_voltage_within_component_tolerance"]
    assert report["simulator_calls"]==0 and report["candidate_spikes"]==3
    assert report["runtime_code"]=={"synthetic.py":"a"*64}


@pytest.mark.parametrize("failure",["input","count","spike","voltage","nonfinite","delivery","reference"])
def test_saved_reference_rejects_changed_values_or_binding(tmp_path,monkeypatch,failure):
    monkeypatch.setenv("SLURM_JOB_ID","synthetic_reader")
    native,candidate,names=saved_reference_fixture(tmp_path)
    case=candidate/"coupled/condition_00"
    if failure=="reference":
        path=case/"report.json";report=json.loads(path.read_text())
        report["circuit_bindings"]["files"]["primary/report.json"]["sha256"]="c"*64
        path.write_text(json.dumps(report))
        expected="EXECUTION_BINDING_DIFFERS"
    else:
        file,key={"input":("input_events","encoded_steps"),"count":("cortical_state","spike_count"),
            "spike":("cortical_spikes","time_ms"),"voltage":("cortical_state","terminal_vm_mV"),
            "nonfinite":("cortical_state","terminal_vm_mV"),"delivery":("input_events","encoded_steps")}[failure]
        path=case/(file+".npz")
        with np.load(path,allow_pickle=False) as source: arrays={k:source[k] for k in source.files}
        arrays[key][0]=np.nan if failure=="nonfinite" else arrays[key][0]+1
        np.savez(path,**arrays)
        expected="DELIVERED_FILE_CHANGED" if failure=="delivery" else "VALUES_DIFFER"
    if failure!="delivery":write_saved_delivery(candidate,names,"synthetic_gpu")
    output=tmp_path/"comparison"
    with pytest.raises(ValueError,match=expected):glif.validate_saved_reference(native,candidate,0,output)
    assert json.loads((output/"report.json").read_text())["status"]=="FAILED_SAVED_NATIVE_GLIF_CUDA_REFERENCE"


def test_production_preparation_accepts_saved_native_reference_receipt(gpu_cohort,tmp_path):
    path,document=gpu_resources(tmp_path/"resources",gpu_cohort,reuse=True)
    validation=Path(document["gpu"]["circuit_validation"])
    proof=json.loads(validation.read_text());proof["status"]="PASS_SAVED_NATIVE_GLIF_CUDA_REFERENCE"
    validation.write_text(json.dumps(proof))
    plan=prepare(gpu_cohort,path,tmp_path/"execution")
    assert plan["backend"]=="glif_cuda"


@pytest.mark.parametrize("same_circuit",[True,False])
def test_saved_condition_reference_must_use_same_constructed_circuit(tmp_path,monkeypatch,same_circuit):
    import shutil
    monkeypatch.setenv("SLURM_JOB_ID","synthetic_reader")
    native,candidate,_=saved_reference_fixture(tmp_path)
    other=tmp_path/"other_condition";shutil.copytree(native,other)
    path=other/"primary/report.json";value=json.loads(path.read_text())
    value["condition_reference"]="another_condition"
    if not same_circuit:value["bindings"]["recurrent"]["files"]["primary/report.json"]["sha256"]="f"*64
    path.write_text(json.dumps(value))
    names=[row["path"] for row in json.loads((other/"delivery.json").read_text())["files"]]
    write_saved_delivery(other,names,"synthetic_other_native")
    if same_circuit:
        assert glif.validate_saved_reference(other,candidate,0,tmp_path/"comparison")["all_spike_pairs_exact"]
    else:
        with pytest.raises(ValueError,match="EXECUTION_BINDING_DIFFERS"):
            glif.validate_saved_reference(other,candidate,0,tmp_path/"comparison")


@pytest.mark.parametrize("mode,pinned",[(None,False),("compare",False),("compare",True),
                                        ("source",False),("source",True),("source_batch",True)])
@pytest.mark.parametrize("admitted",[True,False])
def test_glif_launcher_selects_the_declared_numerical_software(tmp_path,mode,pinned,admitted):
    """Enforce admission before staging while retaining every software route."""
    repo=Path(__file__).resolve().parents[2]
    capture=tmp_path/"arguments.txt"
    preflight_capture=tmp_path/"preflight_arguments.txt"
    interpreter=tmp_path/"capture-python"
    interpreter.write_text('#!/bin/bash\n'
        'if [[ "$2" == */resource_probe.py ]]; then\n'
        '  printf \'%s\\n\' "$@" > "$GLIF_TEST_PREFLIGHT_ARGUMENTS"\n'
        '  exit "$GLIF_TEST_PREFLIGHT_EXIT"\nfi\n'
        'printf \'%s\\n\' "$@" > "$GLIF_TEST_ARGUMENTS"\nexit 73\n')
    interpreter.chmod(0o700)
    bin_directory=tmp_path/"bin";bin_directory.mkdir()
    srun=bin_directory/"srun"
    srun.write_text('#!/bin/bash\n[[ "$1" == --cpu-bind=none ]] || exit 91\nshift\nexec "$@"\n')
    srun.chmod(0o700)
    arguments=[str(repo),str(tmp_path/"reference"),str(tmp_path/"snapshot"),str(tmp_path/"output"),
               str(interpreter),str(tmp_path/"library"),str(tmp_path/"detector")]
    if mode is not None:arguments.append(mode)
    if mode in ("source","source_batch"):arguments.append(str(tmp_path/"inputs"))
    frozen=tmp_path/"frozen_software"
    if pinned:arguments.append(str(frozen))
    result=subprocess.run(["bash",str(repo/"Libs/scripts/run_v1_glif_network.sbatch"),*arguments],
        env=dict(os.environ,SLURM_JOB_ID="software-selection",GLIF_TEST_ARGUMENTS=str(capture),
                 GLIF_TEST_PREFLIGHT_ARGUMENTS=str(preflight_capture),
                 GLIF_TEST_PREFLIGHT_EXIT="0" if admitted else "2",
                 PATH=str(bin_directory)+os.pathsep+os.environ["PATH"]),
        capture_output=True,text=True)
    output=tmp_path/"output/run_software-selection"
    assert preflight_capture.read_text().splitlines()==[
        "-B",str(output/"resource_probe.py"),"--output",str(output/"resource_preflight.json")]
    assert (output/"resource_probe.py").read_bytes()==(repo/"Libs/population_dynamics/resource_probe.py").read_bytes()
    if not admitted:
        assert result.returncode==2,result.stderr
        assert not capture.exists() and not (output/"software").exists()
        return
    assert result.returncode==73,result.stderr
    observed=capture.read_text().splitlines()
    assert observed[:2]==["-B","-"]
    assert observed[2]==str(frozen if pinned else repo/"Libs/population_dynamics")


def test_gpu_walltime_budgets_cover_source_plan_and_both_executions(gpu_cohort,tmp_path,monkeypatch):
    path,document=gpu_resources(tmp_path/"resources",gpu_cohort,reuse=False)
    budgets=dict(rates=240,spikes=120,gpu=60,export=90)
    document["walltime_minutes"]=budgets
    path.write_text(yaml.safe_dump(document,sort_keys=False))
    output=tmp_path/"execution";plan=prepare(gpu_cohort,path,output)
    scheduler=GpuScheduler();monkeypatch.setattr(execution.subprocess,"run",scheduler)
    result=execution.submit(gpu_cohort,plan,output)
    for (command,_),row in zip(scheduler.calls,result["jobs"]):
        phase=row["phase"]
        key="gpu" if phase.startswith("gpu_") else "export" if phase=="source_plan" else phase
        assert [value for value in command if value.startswith("--time=")]==["--time="+str(budgets[key])]
    assert plan["walltime_minutes"]==budgets


@pytest.mark.parametrize("budget",[None,{"gpu":60},dict(rates=240,spikes=120,gpu=True,export=90),
                                    dict(rates=240,spikes=120,gpu=-1,export=90)])
def test_invalid_walltime_budget_rejects_before_plan_write(gpu_cohort,tmp_path,budget):
    path,document=gpu_resources(tmp_path/"resources",gpu_cohort,reuse=True)
    document["walltime_minutes"]=budget;path.write_text(yaml.safe_dump(document,sort_keys=False))
    output=tmp_path/"execution"
    with pytest.raises(ValueError,match="WALLTIME_MINUTES_INVALID"):
        prepare(gpu_cohort,path,output)
    assert not output.exists()


def test_resource_barrier_preserves_internal_success_dependencies(gpu_cohort, tmp_path, monkeypatch):
    path, _ = gpu_resources(tmp_path / "resources", gpu_cohort, reuse=True)
    output = tmp_path / "execution"
    plan = prepare(gpu_cohort, path, output)
    scheduler = GpuScheduler()
    monkeypatch.setattr(execution.subprocess, "run", scheduler)
    result = execution.submit(gpu_cohort, plan, output, start_after_any_jobs=(620000, 620001))
    assert "--dependency=afterany:620000:620001" in scheduler.calls[0][0]
    for index in range(1, len(scheduler.calls)):
        assert "--dependency=afterok:" + str(result["jobs"][index - 1]["job_id"]) in scheduler.calls[index][0]
    calls = len(scheduler.calls)
    assert execution.submit(gpu_cohort, plan, output, resume=True,
                            start_after_any_jobs=(620000, 620001)) == result
    assert len(scheduler.calls) == calls


@pytest.mark.parametrize("failure", [None, "active_original", "wrong_archive", "unknown_submission"])
def test_source_recovery_reuses_completed_prefix(gpu_cohort, tmp_path, monkeypatch, failure):
    path, document = gpu_resources(tmp_path / "resources", gpu_cohort, reuse=True)
    original = tmp_path / "original"
    plan = prepare(gpu_cohort, path, original)
    # Keep synthetic archives within the disposable fixture.
    plan["repo"] = str(tmp_path)
    (original / "plan.json").write_text(json.dumps(plan))
    scheduler = GpuScheduler()
    monkeypatch.setattr(execution.subprocess, "run", scheduler)
    first = execution.submit(gpu_cohort, plan, original)
    original_bytes = (original / "submission.json").read_bytes()
    for index, row in enumerate(first["jobs"]):
        scheduler.states[row["job_id"]] = "COMPLETED" if index < 3 else ("FAILED" if index == 3 else "CANCELLED")
    for index in range(3):
        row = first["jobs"][index]
        root = tmp_path / execution.FAMILY / "visual_inputs" / ("spikes_" + str(row["job_id"]))
        recipe = gpu_cohort.v1_visual_config(gpu_cohort.identity(0, 1, index))
        if failure == "wrong_archive" and index == 2:
            recipe["background"]["starting_noise_seed"] += 1
        archive(root, "PASS_CONSTRUCTED_VISUAL_INPUT_FULL", {"visual_inputs/report.json": dict(
            config=recipe, exit_code=0, complete_lgn_population=True,
            input_construction_binding=dict(delivery_sha256=sha256_file(Path(document["input_source"]) / "delivery.json")))})
        delivery = json.loads((root / "delivery.json").read_text())
        delivery.update(rate_archive=document["conditions"][index]["rate_source"],
                        input_construction_archive=document["input_source"])
        (root / "delivery.json").write_text(json.dumps(delivery))
    output = tmp_path / "recovery"
    calls = len(scheduler.calls)
    if failure == "active_original":
        scheduler.states[first["jobs"][4]["job_id"]] = "RUNNING"
        with pytest.raises(ValueError, match="ORIGINAL_WORK_NOT_TERMINAL"):
            execution.recover_source_block(gpu_cohort, original, output)
        assert not output.exists() and len(scheduler.calls) == calls
        return
    if failure == "wrong_archive":
        with pytest.raises(ValueError, match="RETAINED_SPIKES_BINDING_DIFFERS"):
            execution.recover_source_block(gpu_cohort, original, output)
        assert len(scheduler.calls) == calls
        return
    if failure == "unknown_submission":
        def disconnect(command, **kwargs):
            if command[0] == "sbatch":
                raise OSError("synthetic unknown submission outcome")
            return scheduler(command, **kwargs)
        monkeypatch.setattr(execution.subprocess, "run", disconnect)
        with pytest.raises(OSError, match="unknown submission"):
            execution.recover_source_block(gpu_cohort, original, output)
        monkeypatch.setattr(execution.subprocess, "run", scheduler)
        with pytest.raises(ValueError, match="SUBMISSION_OUTCOME_UNKNOWN"):
            execution.recover_source_block(gpu_cohort, original, output, resume=True)
        assert len(scheduler.calls) == calls
        return
    result = execution.recover_source_block(gpu_cohort, original, output, start_after_any_jobs=(620000,))
    assert len(scheduler.calls) - calls == 11
    assert [row["job_id"] for row in result["jobs"][:3]] == [row["job_id"] for row in first["jobs"][:3]]
    assert all(row["reused"] for row in result["jobs"][:3])
    assert "--dependency=afterany:620000" in scheduler.calls[calls][0]
    sources = json.loads((output / "sources.json").read_text())
    assert sources["conditions"][2]["sources"]["spike_source"].endswith("spikes_700003")
    assert (original / "submission.json").read_bytes() == original_bytes
    assert execution.recover_source_block(gpu_cohort, original, output,
        start_after_any_jobs=(620000,), resume=True) == result
    assert len(scheduler.calls) - calls == 11


_CPU_ENVIRONMENT_FIELDS = (
    "RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "ARNOLD_MONITOR_ROLE", "ARNOLD_APPLIED_GPU_NUM",
    "ARNOLD_HEAD_CPU", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURMD_NODENAME",
    "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_GPUS",
)


@pytest.fixture
def cpu_block_fixture(tmp_path, monkeypatch):
    """Exercise platform routing with disposable data; no source qualification."""
    events = []
    cohort = SimpleNamespace(
        require_current=lambda: events.append("current"),
        qualification={"source_recipes": {"glif_cuda": {
            "runtime_code": {name: "1" * 64 for name in (
                "v1_glif_network.py", "v1_glif_component_emulator.py",
                "v1_glif_cuda.py", "v1_nest_time_semantics.py")},
            "library_sha256": "2" * 64, "component_rtol": 1e-12, "component_atol": 1e-12}}})
    entries = [SimpleNamespace(identity={"condition_index": index}, sources={
        role: tmp_path / role for role in (
            "gpu_primary", "gpu_replay", "rate_source", "spike_source", "authority_source")})
        for index in range(10)]

    def bind(cohort, entry, comparison_path):
        events.append(("bind", entry.identity["condition_index"]))
        return {"source_documents": {}}

    def compare(primary, replay, output, *, execution_context=None):
        events.append("compare")
        assert os.environ["SLURM_JOB_ID"]
        report = {"status": "SYNTHETIC_PLATFORM_FIXTURE", "execution_context": execution_context}
        glif.freeze_json(output / "report.json", report)
        return report

    def verify(cohort, identity, admission):
        events.append(("admit", identity["condition_index"]))
        assert "replay_comparison" in admission["source_documents"]

    def inspect(rate_source, output):
        events.append("inspect")
        assert len([event for event in events if isinstance(event, tuple) and event[0] == "admit"]) == 10
        return {"status": "SOURCE_MOVIE_CONTROL_WITNESS_PASS"}

    def export(primary, controls, authority, output):
        assert len(json.loads(controls.read_text())["cases"]) == 10
        return {"status": "EXPORTED_VERIFIED_GLIF_SOURCE_BLOCK", "complete_episodes": 10,
                "episode_ids": ["synthetic_" + str(index) for index in range(10)]}

    monkeypatch.setattr(glif, "bind_sources", bind)
    monkeypatch.setattr(glif, "compare_executions", compare)
    monkeypatch.setattr(production, "verify_source_admission", verify)
    monkeypatch.setattr(glif, "inspect_source_movie", inspect)
    monkeypatch.setattr(glif, "local_delivery", lambda *args: events.append("delivery"))
    monkeypatch.setattr(glif, "export_source_controls", lambda *args: events.append("controls"))
    monkeypatch.setattr(glif, "export_block", export)
    monkeypatch.setattr(production, "CircuitDataset", lambda path, **kwargs: path)
    return cohort, entries, tmp_path / "attempt", events


@pytest.fixture
def merlin_cpu_environment(monkeypatch):
    for field in _CPU_ENVIRONMENT_FIELDS:
        monkeypatch.delenv(field, raising=False)
    values = {"RH2_JOB_RUN_ID": "synthetic-merlin-run", "ARNOLD_TRIAL_ID": "123456",
              "ARNOLD_MONITOR_ROLE": "head", "ARNOLD_APPLIED_GPU_NUM": "0", "ARNOLD_HEAD_CPU": "4"}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


@pytest.mark.parametrize("previous_job_id", [None, "", "saved-compatibility-id"])
def test_prepared_source_block_merlin_context_is_scoped(
        cpu_block_fixture, merlin_cpu_environment, monkeypatch, previous_job_id):
    cohort, entries, attempt, events = cpu_block_fixture
    if previous_job_id is not None:
        monkeypatch.setenv("SLURM_JOB_ID", previous_job_id)
    with glif.prepared_source_block(cohort, entries, attempt) as source:
        context = source.comparison["execution_context"]
        assert context["backend"] == "merlin"
        assert context["run_id"] == merlin_cpu_environment["RH2_JOB_RUN_ID"]
        assert context["trial_id"] == merlin_cpu_environment["ARNOLD_TRIAL_ID"]
        assert context["allocated_cpus"] == 4 and context["allocated_gpus"] == 0
        assert context["legacy_job_id_is_compatibility"] is True
        assert os.environ["SLURM_JOB_ID"] == context["legacy_job_id"] == "merlin-trial-123456"
        assert "SLURMD_NODENAME" not in os.environ
        assert len(source.episode_ids) == 10
    assert os.environ.get("SLURM_JOB_ID") == previous_job_id
    assert events[:2] == ["current", ("bind", 0)]
    assert len([event for event in events if isinstance(event, tuple) and event[0] == "bind"]) == 10
    assert json.loads((attempt / "source_replay/report.json").read_text())["execution_context"] == context


@pytest.mark.parametrize("failure_point", ["admission", "consumer"])
def test_prepared_source_block_restores_merlin_context_after_failure(
        cpu_block_fixture, merlin_cpu_environment, monkeypatch, failure_point):
    cohort, entries, attempt, events = cpu_block_fixture
    monkeypatch.setenv("SLURM_JOB_ID", "original-value")

    def reject(*args):
        raise ValueError("SYNTHETIC_ADMISSION_REJECTED")

    if failure_point == "admission":
        monkeypatch.setattr(production, "verify_source_admission", reject)
    with pytest.raises(ValueError, match="SYNTHETIC_.*REJECTED"):
        with glif.prepared_source_block(cohort, entries, attempt):
            raise ValueError("SYNTHETIC_CONSUMER_REJECTED")
    assert os.environ["SLURM_JOB_ID"] == "original-value"
    assert "SLURMD_NODENAME" not in os.environ
    if failure_point == "admission":
        assert "inspect" not in events


@pytest.mark.parametrize("field,value", [
    ("ARNOLD_TRIAL_ID", None), ("RH2_JOB_RUN_ID", " "), ("ARNOLD_HEAD_CPU", "0"),
    ("ARNOLD_MONITOR_ROLE", "worker"), ("ARNOLD_APPLIED_GPU_NUM", "1"),
    ("CUDA_VISIBLE_DEVICES", "0"),
])
def test_prepared_source_block_rejects_invalid_merlin_allocation_without_slurm_fallback(
        cpu_block_fixture, merlin_cpu_environment, monkeypatch, field, value):
    cohort, entries, attempt, events = cpu_block_fixture
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURMD_NODENAME", "cpu05")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    if value is None:
        monkeypatch.delenv(field)
    else:
        monkeypatch.setenv(field, value)
    with pytest.raises(ValueError, match="CPU_EXECUTION_INVALID_FIELD:" + field):
        with glif.prepared_source_block(cohort, entries, attempt):
            pytest.fail("invalid allocation reached export")
    assert events == ["current"]
    assert os.environ["SLURM_JOB_ID"] == "123"
    assert not attempt.exists()


@pytest.mark.parametrize("node,gpus,accepted", [
    ("cpu03", "", True), ("cpu05", "", True), ("cpu06", "", True),
    ("cpu04", "", False), ("cpu05", "0", False),
])
def test_prepared_source_block_preserves_legacy_cpu_scope(
        cpu_block_fixture, monkeypatch, node, gpus, accepted):
    cohort, entries, attempt, events = cpu_block_fixture
    for field in _CPU_ENVIRONMENT_FIELDS:
        monkeypatch.delenv(field, raising=False)
    for key, value in {"SLURM_JOB_ID": "123", "SLURMD_NODENAME": node,
                       "SLURM_CPUS_PER_TASK": "4", "SLURM_JOB_GPUS": gpus}.items():
        monkeypatch.setenv(key, value)
    if accepted:
        with glif.prepared_source_block(cohort, entries, attempt) as source:
            context = source.comparison["execution_context"]
            assert context["backend"] == "slurm" and context["slurm_node"] == node
            assert context["slurm_job_id"] == os.environ["SLURM_JOB_ID"] == "123"
            assert "legacy_job_id_is_compatibility" not in context
    else:
        with pytest.raises(ValueError, match="GLIF_PRODUCTION_ALLOCATED_CPU_EXPORT_REQUIRED"):
            with glif.prepared_source_block(cohort, entries, attempt):
                pytest.fail("disallowed legacy allocation reached export")
        assert events == ["current"]
    assert os.environ["SLURMD_NODENAME"] == node


def test_prepared_source_block_still_requires_qualified_backend(
        cpu_block_fixture, merlin_cpu_environment):
    cohort, entries, attempt, events = cpu_block_fixture
    cohort.qualification["source_recipes"]["glif_cuda"]["component_rtol"] = 1e-6
    with pytest.raises(ValueError, match="BENCHMARK_PRODUCTION_GLIF_BACKEND_NOT_QUALIFIED"):
        with glif.prepared_source_block(cohort, entries, attempt):
            pytest.fail("unqualified backend reached export")
    assert events == ["current"] and not attempt.exists()


@pytest.mark.parametrize("with_context", [False, True])
def test_compare_executions_persists_optional_context_before_failure(tmp_path, monkeypatch, with_context):
    for field in _CPU_ENVIRONMENT_FIELDS:
        monkeypatch.delenv(field, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "standalone-reader")

    def reject(*args):
        raise ValueError("SYNTHETIC_BOUND_FILES_REJECTED")

    monkeypatch.setattr(glif, "bound_files", reject)
    context = {"backend": "merlin", "run_id": "synthetic-run", "trial_id": "123456",
               "legacy_job_id": "standalone-reader", "legacy_job_id_is_compatibility": True}
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="SYNTHETIC_BOUND_FILES_REJECTED"):
        glif.compare_executions(tmp_path / "primary", tmp_path / "replay", output,
                                **({"execution_context": context} if with_context else {}))
    saved = json.loads((output / "report.json").read_text())
    assert saved["status"] == "FAILED_GLIF_SOURCE_BLOCK_REPLAY"
    assert saved["error"] == "SYNTHETIC_BOUND_FILES_REJECTED"
    if with_context:
        assert saved["execution_context"] == context
    else:
        assert "execution_context" not in saved


@pytest.mark.skipif(os.environ.get("NEURONSIM_VALIDATE_MERLIN_CPU") != "1",
                    reason="requires an actual Merlin CPU validation allocation")
def test_prepared_source_block_uses_actual_merlin_cpu_allocation(cpu_block_fixture):
    cohort, entries, attempt, events = cpu_block_fixture
    original = {key: os.environ.get(key) for key in _CPU_ENVIRONMENT_FIELDS}
    with glif.prepared_source_block(cohort, entries, attempt) as source:
        context = source.comparison["execution_context"]
        assert context["backend"] == "merlin"
        assert context["run_id"] == original["RH2_JOB_RUN_ID"]
        assert context["trial_id"] == original["ARNOLD_TRIAL_ID"]
        assert context["allocated_cpus"] == 4 and context["allocated_gpus"] == 0
        assert os.environ.get("SLURMD_NODENAME") == original["SLURMD_NODENAME"]
        assert len(source.episode_ids) == 10
        print("ACTUAL_MERLIN_CPU_CONTEXT=" + json.dumps(context, sort_keys=True))
    assert {key: os.environ.get(key) for key in _CPU_ENVIRONMENT_FIELDS} == original
