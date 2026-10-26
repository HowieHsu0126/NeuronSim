"""No-simulation contracts for remaining95 reference-kernel preparation."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
import pytest
import numpy as np

from scripts import run_pfc_reference_parameter_qualification as runner
from scripts import resume_pfc_native_validation as native_resume
from . import pfc_source_execution as source


def executable_phase_fixture(root):
    row=dict(circuit_index=5,status=runner.PREPARED_STATUS,parameters=dict(sigma_weight_profile=20.5,Jpos_excit2excit=1.605),
             independent="independent.yaml",global_config="global.yaml")
    root.mkdir()
    staged=[]
    for name in ("templates/native.sbatch","templates/pairing.sbatch","templates/stream.sbatch"):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("#!/bin/bash\n")
        staged.append(runner.file_row(root,path))
    return dict(rows=[row],qualification_inputs_sha256="fixture-input",staged_files=staged,
                templates={phase:dict(path=f"templates/{phase}.sbatch",python="/qualified/python",
                    test_archive="test_archive",test_label="tests",test_command=[])
                    for phase in runner.PHASES})


def test_execute_failure_retains_completed_local_native_evidence(tmp_path,monkeypatch):
    root=tmp_path / "prepared";prepared=executable_phase_fixture(root)
    monkeypatch.setattr(runner,"validated_preparation",lambda path:prepared)
    monkeypatch.setattr(runner,"admission",lambda *args:dict(scope="fixture"))
    monkeypatch.setattr(runner.os,"access",lambda *args:True)
    monkeypatch.setenv("SLURM_JOB_ID","991001");monkeypatch.setenv("SLURMD_NODENAME","cpu05")
    monkeypatch.delenv("SLURM_JOB_GPUS",raising=False);monkeypatch.delenv("SLURM_STEP_GPUS",raising=False)
    retained=[]
    def failed(command,*,env,**kwargs):
        local=Path(env["NEURONSIM_PFC_SCRATCH"])/"neuronsim-pfc-input-validation-991001"
        local.mkdir();(local / "plain").mkdir();(local / "plain/report.json").write_text('{"status":"COMPLETED_NATIVE_FIXTURE"}')
        execution=dict(status="ARCHIVE_RESOURCE_BLOCKED",job_id="991001",
            commands=[dict(label=name,status="PASS",exit_code=0) for name in ("plain","global_plain","audit","capture","compare")])
        (local / "execution.json").write_text(json.dumps(execution));retained.append(local)
        return subprocess.CompletedProcess(command,returncode=1)
    monkeypatch.setattr(runner.subprocess,"run",failed)
    try:
        with pytest.raises(RuntimeError,match="native launcher failed"):
            runner.execute(root,5,tmp_path / "assessment")
        result=runner.load_json(root / "results/circuit_005/runner_execution.json")
        assert result["status"]=="FAILED" and result["failed_phase"]=="native"
        assert result["producer_job"]=="991001" and result["node"]=="cpu05"
        assert result["runner_implementation_sha256"]==runner.digest(Path(runner.__file__))
        assert result["runtime_state"]=="RETAINED_AFTER_FAILURE"
        assert Path(result["retained_runtime"]).is_dir()
        native=result["phases"]["native"]
        assert native["status"]=="FAILED_LAUNCHER" and native["launcher_exit_code"]==1
        assert native["local_execution"]["status"]=="ARCHIVE_RESOURCE_BLOCKED"
        assert all(row["status"]=="PASS" for row in native["local_execution"]["commands"])
        assert (retained[0] / "plain/report.json").is_file()
        assert result["source_qualification_granted"] is False
    finally:
        if retained:shutil.rmtree(retained[0].parents[1])


def test_execute_success_archives_three_phases_then_cleans_runtime(tmp_path,monkeypatch):
    root=tmp_path / "prepared";prepared=executable_phase_fixture(root)
    monkeypatch.setattr(runner,"validated_preparation",lambda path:prepared)
    monkeypatch.setattr(runner,"admission",lambda *args:dict(scope="fixture"))
    monkeypatch.setattr(runner.os,"access",lambda *args:True)
    monkeypatch.setenv("SLURM_JOB_ID","991002");monkeypatch.setenv("SLURMD_NODENAME","cpu05")
    monkeypatch.delenv("SLURM_JOB_GPUS",raising=False);monkeypatch.delenv("SLURM_STEP_GPUS",raising=False)
    locals=[]
    def success(command,*,env,**kwargs):
        phase=Path(command[1]).stem;local=Path(env["NEURONSIM_PFC_SCRATCH"]).parent;locals.append(local)
        archive=Path(env["NEURONSIM_PFC_OUTPUT_BASE"])/(runner.PHASES[phase][3]+"991002")
        archive.mkdir();path=archive / "execution.json"
        path.write_text(json.dumps(dict(status=runner.PHASES[phase][2],job_id="991002")))
        (archive / "delivery.json").write_text(json.dumps(dict(files=[runner.file_row(archive,path)])))
        return subprocess.CompletedProcess(command,returncode=0)
    monkeypatch.setattr(runner.subprocess,"run",success)
    result=runner.execute(root,5,tmp_path / "assessment")
    assert result["status"]=="PASS_EXACT_TUPLE_REFERENCE_KERNEL_CHECKS"
    assert set(result["phases"])==set(runner.PHASES)
    assert result["runtime_state"]=="REMOVED_AFTER_SUCCESS" and result["retained_runtime"] is None
    assert len(locals)==3 and len(set(locals))==1 and not locals[0].exists()
    assert all(Path(row["archive"]).is_dir() for row in result["phases"].values())
    assert result["source_qualification_granted"] is False and result["production_episodes"]==0


def retained_native_fixture(root):
    root.mkdir();(root/"Libs/configs").mkdir(parents=True)
    independent=root/"Libs/configs/independent.yaml";independent.write_text("scope: source_reproduction\n")
    global_config=root/"Libs/configs/global.yaml";global_config.write_text("scope: source_reproduction\n")
    (root/"source").mkdir();(root/"source/source.json").write_text('{"files":[]}')
    (root/"plain").mkdir();(root/"plain/report.json").write_text('{"status":"fixture_complete"}')
    (root/"global_reference/plain").mkdir(parents=True)
    parent=dict(status="FAILED",job_id="991100",commands=[dict(label="tests",command=["python","tests"],status="REUSED_PASS",exit_code=0)])
    (root/"execution.json").write_text(json.dumps(parent))
    return independent,global_config


@pytest.mark.parametrize("global_complete",[False,True])
def test_native_resume_reuses_plain_and_resumes_partial_global_once(tmp_path,monkeypatch,global_complete):
    retained=tmp_path/"retained";independent,global_config=retained_native_fixture(retained)
    if global_complete:(retained/"global_reference/plain/report.json").write_text('{"status":"fixture_complete"}')
    calls=[]
    monkeypatch.setattr(native_resume,"validate_native_output",lambda output,*args,**kwargs:["all8plain"] if output.name=="plain" and output.parent==retained else ["global1"])
    from . import resource_probe
    monkeypatch.setattr(resource_probe,"probe",lambda *args,**kwargs:dict(status="PASS",reasons=[]))
    monkeypatch.setenv("SLURM_JOB_ID","991101");monkeypatch.setenv("SLURMD_NODENAME","cpu04")
    monkeypatch.setenv("NEURONSIM_PFC_SCRATCH",str(tmp_path/"scratch"));monkeypatch.delenv("SLURM_JOB_GPUS",raising=False)
    class Child:
        returncode=0
        def poll(self):return 0
    def launch(command,**kwargs):
        calls.append(command);destination=Path(command[command.index("--output")+1]);destination.mkdir(parents=True,exist_ok=True)
        if "population_dynamics.pfc_reference" in command:
            (destination/"report.json").write_text('{"status":"fixture_complete_global"}')
        elif "compare" in command:
            (destination/"comparison.json").write_text(json.dumps(dict(status="PASS_NATIVE_INPUT_OBSERVATION_PARITY",plain_source_execution_compared=True)))
        return Child()
    monkeypatch.setattr(native_resume.subprocess,"Popen",launch)
    output=tmp_path/"newnative"
    original=(retained/"execution.json").read_bytes()
    result=native_resume.resume_native(retained,independent,global_config,output)
    assert result["status"]=="PASS_NATIVE_INPUT_OBSERVATION_PARITY"
    assert result["parent_job_id"]=="991100" and result["job_id"]=="991101"
    assert result["parent_execution"]["status"]=="FAILED"
    references=[command for command in calls if "population_dynamics.pfc_reference" in command]
    if global_complete:
        assert references==[] and len(calls)==3
    else:
        assert len(references)==1 and "--resume" in references[0]
        assert Path(references[0][references[0].index("--output")+1]).parent.name=="global_reference"
        assert len(calls)==4  # global remainder, audit, capture, comparison.
    assert (retained/"execution.json").read_bytes()==original
    assert not Path(result["runtime"]).exists()
    delivery=runner.load_json(output/"delivery.json")
    assert delivery["status"]=="ARCHIVED_NATIVE_INPUT_OBSERVATION_PARITY"
    assert delivery["comparison"]==runner.load_json(output/"comparison/comparison.json")
    with pytest.raises(ValueError,match="FRESH_CPU_ATTEMPT_REQUIRED"):
        native_resume.resume_native(retained,independent,global_config,output)


def test_native_resume_rejects_changed_config_or_existing_partial_observer(tmp_path,monkeypatch):
    retained=tmp_path/"retained";independent,global_config=retained_native_fixture(retained)
    monkeypatch.setenv("SLURM_JOB_ID","991102");monkeypatch.setenv("SLURMD_NODENAME","cpu04")
    monkeypatch.delenv("SLURM_JOB_GPUS",raising=False)
    changed=tmp_path/"independent.yaml";changed.write_text("scope: changed\n")
    monkeypatch.setattr(native_resume,"validate_native_output",lambda *args,**kwargs:pytest.fail("changed config read payload"))
    with pytest.raises(ValueError,match="CONFIG_CHANGED"):
        native_resume.resume_native(retained,changed,global_config,tmp_path/"output")
    assert not (tmp_path/"output").exists()


def test_native_resume_stages_unstarted_global_config(tmp_path,monkeypatch):
    retained=tmp_path/"retained";independent,global_config=retained_native_fixture(retained)
    canonical=tmp_path/"global.yaml";canonical.write_bytes(global_config.read_bytes())
    global_config.unlink();shutil.rmtree(retained/"global_reference")
    monkeypatch.setattr(native_resume,"validate_native_output",lambda *args,**kwargs:["plain"])
    from . import resource_probe
    monkeypatch.setattr(resource_probe,"probe",lambda *args,**kwargs:dict(status="PASS",reasons=[]))
    monkeypatch.setenv("SLURM_JOB_ID","991107");monkeypatch.setenv("SLURMD_NODENAME","cpu04")
    monkeypatch.setenv("NEURONSIM_PFC_SCRATCH",str(tmp_path/"scratch"));monkeypatch.delenv("SLURM_JOB_GPUS",raising=False)
    class Child:
        returncode=1
        def poll(self):return 1
    def launch(command,**kwargs):
        assert "population_dynamics.pfc_reference" in command and "--resume" not in command
        config=Path(command[command.index("--config")+1]);assert config.read_bytes()==canonical.read_bytes()
        return Child()
    monkeypatch.setattr(native_resume.subprocess,"Popen",launch)
    with pytest.raises(RuntimeError,match="COMMAND_FAILED:global_plain"):
        native_resume.resume_native(retained,independent,canonical,tmp_path/"output")
    assert not global_config.exists()


@pytest.mark.parametrize("changed_capture",[False,True])
@pytest.mark.parametrize("archive_resource_failure",[False,True])
def test_native_resume_archived_observers_reused_without_simulation(tmp_path,monkeypatch,changed_capture,archive_resource_failure):
    retained=tmp_path/"retained";independent,global_config=retained_native_fixture(retained)
    (retained/"global_reference/plain/report.json").write_text('{"status":"fixture_complete"}')
    from .provenance import sha256_file
    for mode in ("audit","capture"):
        root=retained/mode;root.mkdir()
        (root/"execution.json").write_text(json.dumps(dict(config_sha256=sha256_file(independent))))
    comparison=dict(status="PASS_NATIVE_INPUT_OBSERVATION_PARITY",plain_source_execution_compared=True,
        audit_execution_sha256=sha256_file(retained/"audit/execution.json"),
        capture_execution_sha256=sha256_file(retained/"capture/execution.json"))
    (retained/"comparison").mkdir();(retained/"comparison/comparison.json").write_text(json.dumps(comparison))
    parent=runner.load_json(retained/"execution.json");parent.update(status="PASS_NATIVE_INPUT_OBSERVATION_PARITY",comparison=comparison)
    if archive_resource_failure:parent.update(status="FAILED",error="PFC_INPUT_ARCHIVE_RESOURCE_BLOCKED:PERSISTENT_UNINTERRUPTIBLE_IO")
    parent["commands"].extend(dict(label=label,status="PASS",exit_code=0) for label in ("audit","capture","compare"))
    (retained/"execution.json").write_text(json.dumps(parent))
    paths=[p for p in retained.rglob("*") if p.is_file()]
    (retained/"delivery.json").write_text(json.dumps(dict(status="ARCHIVED_NATIVE_INPUT_OBSERVATION_PARITY",
        files=[runner.file_row(retained,p) for p in paths])))
    original=(retained/"execution.json").read_bytes()
    if changed_capture:(retained/"capture/execution.json").write_text("changed capture")
    monkeypatch.setenv("SLURM_JOB_ID","991109");monkeypatch.setenv("SLURMD_NODENAME","cpu04")
    monkeypatch.setenv("NEURONSIM_PFC_SCRATCH",str(tmp_path/"scratch"));monkeypatch.delenv("SLURM_JOB_GPUS",raising=False)
    from . import resource_probe,pfc_input_capture
    monkeypatch.setattr(resource_probe,"probe",lambda *args,**kwargs:dict(status="PASS",reasons=[]))
    monkeypatch.setattr(native_resume,"validate_native_output",lambda *args,**kwargs:["already_complete"])
    monkeypatch.setattr(pfc_input_capture,"_load",lambda root,mode:(runner.load_json(root/"execution.json"),{}))
    monkeypatch.setattr(native_resume.subprocess,"Popen",lambda *args,**kwargs:pytest.fail("repeated completed simulation or comparison"))
    output=tmp_path/"newnative"
    if changed_capture:
        with pytest.raises(ValueError,match="ARCHIVE_CHANGED"):
            native_resume.resume_native(retained,independent,global_config,output)
        assert not output.exists()
    else:
        result=native_resume.resume_native(retained,independent,global_config,output)
        assert result["status"]=="PASS_NATIVE_INPUT_OBSERVATION_PARITY"
        assert all(row["status"]=="REUSED_PASS" and row["command_started"] is False for row in result["commands"])
        delivery=runner.load_json(output/"delivery.json")
        assert delivery["comparison"]==comparison
        assert "delivery.json" not in {row["path"] for row in delivery["files"]}
        for row in delivery["files"]:runner.checked(output,row)
    assert (retained/"execution.json").read_bytes()==original


@pytest.mark.parametrize("complete_modes",[("audit",),("audit","capture")])
def test_native_resume_reuses_completed_observer_before_missing_comparison(tmp_path,monkeypatch,complete_modes):
    retained=tmp_path/"retained";independent,global_config=retained_native_fixture(retained)
    (retained/"global_reference/plain/report.json").write_text('{"status":"fixture_complete"}')
    parent=runner.load_json(retained/"execution.json")
    from .provenance import sha256_file
    for mode in complete_modes:
        root=retained/mode;root.mkdir();(root/"native").mkdir()
        (root/"execution.json").write_text(json.dumps(dict(config_sha256=sha256_file(independent))))
        parent["commands"].append(dict(label=mode,status="PASS",exit_code=0))
    (retained/"execution.json").write_text(json.dumps(parent))
    (retained/"comparison").mkdir();(retained/"comparison/partial.json").write_text('{}')
    monkeypatch.setenv("SLURM_JOB_ID","991110");monkeypatch.setenv("SLURMD_NODENAME","cpu04")
    monkeypatch.setenv("NEURONSIM_PFC_SCRATCH",str(tmp_path/"scratch"));monkeypatch.delenv("SLURM_JOB_GPUS",raising=False)
    from . import resource_probe,pfc_input_capture
    monkeypatch.setattr(resource_probe,"probe",lambda *args,**kwargs:dict(status="PASS",reasons=[]))
    monkeypatch.setattr(native_resume,"validate_native_output",lambda *args,**kwargs:["existing"])
    monkeypatch.setattr(pfc_input_capture,"_load",lambda root,mode:(runner.load_json(root/"execution.json"),{}))
    calls=[]
    class Child:
        returncode=0
        def poll(self):return 0
    def launch(command,**kwargs):
        calls.append(command)
        assert "population_dynamics.pfc_reference" not in command
        if "record" in command:assert command[-1] not in complete_modes
        destination=Path(command[command.index("--output")+1]);destination.mkdir(parents=True)
        if "compare" in command:
            (destination/"comparison.json").write_text(json.dumps(dict(status="PASS_NATIVE_INPUT_OBSERVATION_PARITY",plain_source_execution_compared=True)))
        return Child()
    monkeypatch.setattr(native_resume.subprocess,"Popen",launch)
    result=native_resume.resume_native(retained,independent,global_config,tmp_path/"output")
    assert result["status"]=="PASS_NATIVE_INPUT_OBSERVATION_PARITY"
    assert len(calls)==3-len(complete_modes)
    assert (tmp_path/"output/parent_incomplete_comparison/partial.json").exists()
    for mode in complete_modes:
        row=next(row for row in result["commands"] if row["label"]==mode)
        assert row["status"]=="REUSED_PASS" and row["command_started"] is False


def test_native_resume_partial_plain_uses_existing_case_resume(tmp_path,monkeypatch):
    retained=tmp_path/"retained";independent,global_config=retained_native_fixture(retained)
    (retained/"plain/report.json").unlink()
    monkeypatch.setenv("SLURM_JOB_ID","991111");monkeypatch.setenv("SLURMD_NODENAME","cpu05")
    monkeypatch.setenv("NEURONSIM_PFC_SCRATCH",str(tmp_path/"scratch"));monkeypatch.delenv("SLURM_JOB_GPUS",raising=False)
    from . import resource_probe
    monkeypatch.setattr(resource_probe,"probe",lambda *args,**kwargs:dict(status="PASS",reasons=[]))
    checked=[]
    def validate(path,*args,**kwargs):
        checked.append((path,kwargs["complete"]));return ["five_existing"]
    monkeypatch.setattr(native_resume,"validate_native_output",validate)
    calls=[]
    class Child:
        returncode=0
        def poll(self):return 0
    def launch(command,**kwargs):
        calls.append(command);destination=Path(command[command.index("--output")+1]);destination.mkdir(parents=True,exist_ok=True)
        if "population_dynamics.pfc_reference" in command:
            (destination/"report.json").write_text('{"status":"fixture_complete"}')
        elif "compare" in command:
            (destination/"comparison.json").write_text(json.dumps(dict(status="PASS_NATIVE_INPUT_OBSERVATION_PARITY",plain_source_execution_compared=True)))
        return Child()
    monkeypatch.setattr(native_resume.subprocess,"Popen",launch)
    result=native_resume.resume_native(retained,independent,global_config,tmp_path/"output")
    plain=calls[0];assert "--resume" in plain and Path(plain[plain.index("--output")+1]).name=="plain"
    assert checked[0][1] is False and any(complete for _,complete in checked)
    assert result["reused_plain_case_ids"]==["five_existing"]
    assert result["status"]=="PASS_NATIVE_INPUT_OBSERVATION_PARITY"


def test_native_resume_missing_global_config_rejects_started_global(tmp_path,monkeypatch):
    retained=tmp_path/"retained";independent,global_config=retained_native_fixture(retained)
    canonical=tmp_path/"global.yaml";canonical.write_bytes(global_config.read_bytes());global_config.unlink()
    monkeypatch.setenv("SLURM_JOB_ID","991108");monkeypatch.setenv("SLURMD_NODENAME","cpu04")
    monkeypatch.delenv("SLURM_JOB_GPUS",raising=False)
    monkeypatch.setattr(native_resume,"validate_native_output",lambda *args,**kwargs:pytest.fail("read invalid payload"))
    with pytest.raises(ValueError,match="CONFIG_CHANGED:global_plain"):
        native_resume.resume_native(retained,independent,canonical,tmp_path/"output")


def test_native_resume_failure_keeps_local_work_and_original_failure(tmp_path,monkeypatch):
    retained=tmp_path/"retained";independent,global_config=retained_native_fixture(retained)
    monkeypatch.setattr(native_resume,"validate_native_output",lambda *args,**kwargs:["existing"])
    from . import resource_probe
    monkeypatch.setattr(resource_probe,"probe",lambda *args,**kwargs:dict(status="PASS",reasons=[]))
    monkeypatch.setenv("SLURM_JOB_ID","991103");monkeypatch.setenv("SLURMD_NODENAME","cpu04")
    monkeypatch.setenv("NEURONSIM_PFC_SCRATCH",str(tmp_path/"scratch"));monkeypatch.delenv("SLURM_JOB_GPUS",raising=False)
    class Child:
        returncode=1
        def poll(self):return 1
    monkeypatch.setattr(native_resume.subprocess,"Popen",lambda *args,**kwargs:Child())
    with pytest.raises(RuntimeError,match="COMMAND_FAILED:global_plain"):
        native_resume.resume_native(retained,independent,global_config,tmp_path/"output")
    local=tmp_path/"scratch/neuronsim-pfc-input-validation-991103"
    result=runner.load_json(local/"execution.json")
    assert result["status"]=="FAILED" and result["parent_execution"]["status"]=="FAILED"
    assert (local/"plain/report.json").is_file()
    assert runner.load_json(retained/"execution.json")["job_id"]=="991100"


def test_native_resume_runner_dispatch_preserves_new_attempt_and_original_failure(tmp_path,monkeypatch):
    root=tmp_path/"prepared";prepared=executable_phase_fixture(root)
    retained=tmp_path/"retained";retained_native_fixture(retained)
    original=(retained/"execution.json").read_bytes()
    original_result=root/"results/circuit_005/runner_execution.json"
    original_result.parent.mkdir(parents=True);original_result.write_text('{"status":"FAILED"}')
    monkeypatch.setattr(runner,"validated_preparation",lambda path:prepared)
    monkeypatch.setattr(runner,"admission",lambda *args:dict(scope="fixture"))
    monkeypatch.setattr(runner.os,"access",lambda *args:True)
    monkeypatch.setenv("SLURM_JOB_ID","991104");monkeypatch.setenv("SLURMD_NODENAME","cpu04")
    monkeypatch.delenv("SLURM_JOB_GPUS",raising=False);monkeypatch.delenv("SLURM_STEP_GPUS",raising=False)
    output=tmp_path/"newattempt";calls=[]
    def run(command,*,env,**kwargs):
        calls.append(command)
        if "--retained" in command:
            phase="native";archive=Path(command[command.index("--output")+1])
        else:
            phase=Path(command[1]).stem;archive=output/(runner.PHASES[phase][3]+"991104")
        archive.mkdir();path=archive/"execution.json"
        path.write_text(json.dumps(dict(status=runner.PHASES[phase][2],job_id="991104")))
        (archive/"delivery.json").write_text(json.dumps(dict(files=[runner.file_row(archive,path)])))
        return subprocess.CompletedProcess(command,returncode=0)
    monkeypatch.setattr(runner.subprocess,"run",run)
    result=runner.execute(root,5,tmp_path/"assessment",resume_native_from=retained,attempt_output=output)
    assert result["status"]=="PASS_EXACT_TUPLE_REFERENCE_KERNEL_CHECKS"
    assert result["native_resume"]["parent_job_id"]=="991100"
    assert result["native_resume"]["parent_status"]=="FAILED"
    assert len(calls)==3 and "--retained" in calls[0] and calls[1][0]==calls[2][0]=="bash"
    assert (retained/"execution.json").read_bytes()==original
    assert original_result.read_text()=='{"status":"FAILED"}'
    assert all(Path(item["archive"]).parent==output for item in result["phases"].values())


def test_native_resume_rejects_source_file_change_before_payload(tmp_path,monkeypatch):
    retained=tmp_path/"retained";independent,global_config=retained_native_fixture(retained)
    source_file=retained/"source/neurodynex3/working_memory_network/wm_model.py"
    source_file.parent.mkdir(parents=True);source_file.write_text("changed source\n")
    (retained/"source/source.json").write_text(json.dumps(dict(files=[dict(
        path="neurodynex3/working_memory_network/wm_model.py",sha256="wrong")])) )
    monkeypatch.setenv("SLURM_JOB_ID","991105");monkeypatch.setenv("SLURMD_NODENAME","cpu04")
    monkeypatch.delenv("SLURM_JOB_GPUS",raising=False)
    monkeypatch.setattr(native_resume,"validate_native_output",lambda *args,**kwargs:pytest.fail("changed source read native payload"))
    with pytest.raises(ValueError,match="SOURCE_CHANGED"):
        native_resume.resume_native(retained,independent,global_config,tmp_path/"output")
    assert not (tmp_path/"output").exists()


def intake_fixture(root):
    """Metadata-only completed100 fixture; no native simulation or arrays."""
    prepared = dict(schema=runner.SCHEMA, dependent_on_fixed150_quality_assessment=True,
                    qualification_inputs_sha256="fixture-input-sha", rows=[])
    summary = root / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(dict(status="COMPLETE_REFERENCE_QUALITY_DESCRIPTIVE_COHORT",
        complete_episodes=150, circuits=list(range(5)), blocks=[3,4,5], conditions=list(range(10)),
        core_units=100000, duration_ticks=400000, dt_ms=.0125,
        bindings=[dict(episode_id=f"episode_{index}") for index in range(150)])))
    assessment = root / "assessment.json"
    assessment.write_text(json.dumps(dict(approved_remaining_reference_kernel_checks=True,
        scope="remaining95_reference_kernel_checks", rationale="Fixture complete cohort assessed.",
        qualification_inputs_sha256=prepared["qualification_inputs_sha256"],
        fixed150_quality_summary=dict(path=str(summary), sha256=runner.digest(summary)))))

    def delivery(directory, status, extra=None):
        files = [runner.file_row(directory, path) for path in directory.rglob("*")
                 if path.is_file() and path != directory / "delivery.json"]
        value = dict(status=status, files=files, **(extra or {}))
        (directory / "delivery.json").write_text(json.dumps(value))
        return value

    def stream(directory, parameters, job, native=None, pairing=None):
        (directory / "comparison").mkdir(parents=True)
        receipt = dict(status="PASS_NATIVE_STREAM_SOURCE_EXECUTION", backend="numpy",
            source_entry_sha256=source.native_drive.SOURCE_ENTRY_SHA256, numpy_version=np.__version__,
            executor_sha256=runner.digest(Path(source.executor.__file__)),
            drive_sha256=runner.digest(Path(source.native_drive.__file__)),
            kernel_parameterization_sha256=runner.digest(Path(source.kernel_parameters.__file__)),
            validated_kernel_parameters=[parameters], validated_dt_ms=[.0125],
            input_replay_status="PASS_NATIVE_NUMPY_DRIVE_REPLAY", input_replay_passed_comparisons=24,
            global_cases_passed=2, global_arrays_passed=10,
            full_scale_qualified=False, scientific_fidelity_qualified=False, production_episodes=0)
        if native is not None:
            receipt.update(global_reference_delivery_sha256=runner.digest(native / "global_reference/delivery.json"),
                           global_reference_report_sha256=runner.digest(native / "global_reference/plain/report.json"))
        (directory / "comparison/qualification.json").write_text(json.dumps(receipt))
        (directory / "execution.json").write_text(json.dumps(dict(status=runner.PHASES["stream"][2],
            job_id=job, scientific_fidelity_qualified=False, production_episodes=0)))
        extra = dict(result_status=runner.PHASES["stream"][2], backend="numpy", job_id=job)
        if native is not None:
            extra.update(source_validation=str(native), pairing_source=str(pairing),
                         global_reference=str(native / "global_reference"))
        delivery(directory, "ARCHIVED_PFC_STREAM_VALIDATION", extra)

    for index in range(100):
        parameters = dict(sigma_weight_profile=20.+index/10, Jpos_excit2excit=1.6+index/1000)
        row = dict(circuit_index=index, parameters=parameters,
                   status=runner.REUSE_STATUS if index<5 else runner.PREPARED_STATUS)
        prepared["rows"].append(row)
        if index<5:
            directory=root / f"reused/circuit_{index:03d}"
            row["qualification"]=str(directory)
            stream(directory,parameters,str(80000+index))
            continue
        row.update(independent=f"proof_inputs/circuit_{index:03d}/independent.yaml",
                   global_config=f"proof_inputs/circuit_{index:03d}/global.yaml")
        archive=root / "results" / f"circuit_{index:03d}"
        job=str(90000+index)
        native=archive / (runner.PHASES["native"][3]+job)
        pairing=archive / (runner.PHASES["pairing"][3]+job)
        qualification=archive / (runner.PHASES["stream"][3]+job)
        for key,name in (("independent","independent.yaml"),("global_config","global.yaml")):
            path=root / "inputs" / row[key]
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(yaml.safe_dump(dict(parameters,dt_ms=.0125)))
            target=native / "Libs/configs" / name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes())
        global_root=native / "global_reference"
        (global_root / "plain").mkdir(parents=True)
        (global_root / "plain/report.json").write_text('{"scope":"fixture native global"}')
        delivery(global_root,"ARCHIVED_NATIVE_GLOBAL_REFERENCE")
        (native / "execution.json").write_text(json.dumps(dict(status=runner.PHASES["native"][2],job_id=job,
            scientific_fidelity_qualified=False,production_episodes=0)))
        delivery(native,"ARCHIVED_NATIVE_INPUT_OBSERVATION_PARITY")
        pairing.mkdir()
        (pairing / "execution.json").write_text(json.dumps(dict(status=runner.PHASES["pairing"][2],job_id=job,
            source_validation=str(native),scientific_fidelity_qualified=False,production_episodes=0)))
        delivery(pairing,"ARCHIVED_NATIVE_INPUT_PAIRING",dict(source_validation=str(native),
            source_delivery_sha256=runner.digest(native / "delivery.json")))
        stream(qualification,parameters,job,native,pairing)
        phases={name:dict(status=runner.PHASES[name][2],archive=str(path),delivery_sha256=runner.digest(path / "delivery.json"))
                for name,path in (("native",native),("pairing",pairing),("stream",qualification))}
        (root / "execution.json").write_text(json.dumps(prepared))
        result=dict(status="PASS_EXACT_TUPLE_REFERENCE_KERNEL_CHECKS",circuit_index=index,parameters=parameters,
            qualification_inputs_sha256=prepared["qualification_inputs_sha256"],assessment={},phases=phases,
            source_qualification_granted=False,scientific_parameter_domain_qualified=False,production_episodes=0)
        (archive / "runner_execution.json").write_text(json.dumps(result))
    (root / "execution.json").write_text(json.dumps(prepared))
    chain=runner.admission(assessment,prepared)
    for index in range(5,100):
        path=root / "results" / f"circuit_{index:03d}" / "runner_execution.json"
        value=json.loads(path.read_text());value["assessment"]=chain;path.write_text(json.dumps(value))
    return prepared


def test_collect_complete100_uses_real_tuple_verifier_without_scientific_pass(tmp_path,monkeypatch):
    prepared=intake_fixture(tmp_path / "prepared")
    monkeypatch.setattr(runner,"validated_preparation",lambda root:prepared)
    output=tmp_path / "collected"
    with patch.object(runner.subprocess,"run") as run:
        result=runner.collect(tmp_path / "prepared",output)
    run.assert_not_called()
    assert set(result["qualifications"])=={str(index) for index in range(100)}
    assert result["complete_circuits"]==100 and len(result["bindings"])==100
    assert result["source_quality_qualified"] is result["source_qualification_granted"] is False
    assert result["scientific_parameter_domain_qualified"] is result["scientific_claim_qualified"] is False
    assert result["production_episodes"]==result["new_source_simulations"]==0
    assert json.loads((output / "qualifications.json").read_text())==result["qualifications"]
    assert all(binding["receipt_sha256"] and binding["delivery_sha256"] for binding in result["bindings"])


@pytest.mark.parametrize("damage",["missing","running","failed","identity","input_hash","assessment","delivery_hash","kernel","phase_job","native_config","chain"])
def test_collect_rejects_partial_or_conflicting_results_before_output(tmp_path,monkeypatch,damage):
    root=tmp_path / "prepared";prepared=intake_fixture(root)
    monkeypatch.setattr(runner,"validated_preparation",lambda path:prepared)
    path=root / "results/circuit_005/runner_execution.json"
    result=json.loads(path.read_text())
    if damage=="missing":path.unlink()
    elif damage in ("running","failed"):
        result["status"]="RUNNING_STREAM" if damage=="running" else "FAILED";path.write_text(json.dumps(result))
    elif damage in ("identity","input_hash","assessment","delivery_hash","chain"):
        if damage=="identity":result["circuit_index"]=6
        elif damage=="input_hash":result["qualification_inputs_sha256"]="wrong"
        elif damage=="assessment":result["assessment"]["sha256"]="wrong"
        elif damage=="delivery_hash":result["phases"]["stream"]["delivery_sha256"]="wrong"
        else:
            native=Path(result["phases"]["native"]["archive"])
            pairing=Path(result["phases"]["pairing"]["archive"])
            receipt=runner.load_json(pairing / "delivery.json");receipt["source_validation"]="/wrong"
            (pairing / "delivery.json").write_text(json.dumps(receipt))
            result["phases"]["pairing"]["delivery_sha256"]=runner.digest(pairing / "delivery.json")
        path.write_text(json.dumps(result))
    elif damage=="native_config":
        native=Path(result["phases"]["native"]["archive"])
        (native / "Libs/configs/independent.yaml").write_text("dt_ms: 0.05\n")
    elif damage=="phase_job":
        native=Path(result["phases"]["native"]["archive"])
        execution=runner.load_json(native / "execution.json");execution["job_id"]="wrong"
        (native / "execution.json").write_text(json.dumps(execution))
    else:
        qualification=Path(result["phases"]["stream"]["archive"])
        receipt=runner.load_json(qualification / "comparison/qualification.json")
        receipt["validated_kernel_parameters"]=[dict(sigma_weight_profile=20.,Jpos_excit2excit=1.6)]
        (qualification / "comparison/qualification.json").write_text(json.dumps(receipt))
        delivery=runner.load_json(qualification / "delivery.json")
        for binding in delivery["files"]:
            if binding["path"]=="comparison/qualification.json":binding.update(runner.file_row(qualification,qualification / binding["path"]))
        (qualification / "delivery.json").write_text(json.dumps(delivery))
        result["phases"]["stream"]["delivery_sha256"]=runner.digest(qualification / "delivery.json")
        path.write_text(json.dumps(result))
    output=tmp_path / "collected"
    with pytest.raises((ValueError,FileNotFoundError)):
        runner.collect(root,output)
    assert not output.exists()


@pytest.mark.parametrize("wrong_target",[None,"native","pairing","global_reference"])
def test_collect_recovered_phase_aliases_keep_semantic_reference_chain(tmp_path,monkeypatch,wrong_target):
    root=tmp_path/"prepared";prepared=intake_fixture(root)
    monkeypatch.setattr(runner,"validated_preparation",lambda path:prepared)
    original=root/"results/circuit_005"
    result=runner.load_json(original/"runner_execution.json")
    recovered=root/"recovered/run_91005"
    recovered.mkdir(parents=True)
    for phase in runner.PHASES:
        old=Path(result["phases"][phase]["archive"])
        alias=recovered/old.name
        alias.symlink_to(old,target_is_directory=True)
        result["phases"][phase]["archive"]=str(alias)
    result.update(producer_job="91005",native_resume=dict(parent_job_id="90005"))
    if wrong_target:
        if wrong_target=="global_reference":
            stream=Path(result["phases"]["stream"]["archive"])
            delivery=runner.load_json(stream/"delivery.json")
            delivery["global_reference"]=str(root/"results/circuit_006/native_validation_90006/global_reference")
            (stream/"delivery.json").write_text(json.dumps(delivery))
            result["phases"]["stream"]["delivery_sha256"]=runner.digest(stream/"delivery.json")
        else:
            phase="native" if wrong_target=="native" else "pairing"
            alias=Path(result["phases"][phase]["archive"])
            alias.unlink()
            alias.symlink_to(root/"results/circuit_006"/(runner.PHASES[phase][3]+"90006"),target_is_directory=True)
    (recovered/"runner_execution.json").write_text(json.dumps(result))
    # The result view selects a distinct verified wrapper; source phase producers stay90005.
    selected=root/"view"
    selected.mkdir()
    (selected/"execution.json").write_bytes((root/"execution.json").read_bytes())
    (selected/"inputs").symlink_to(root/"inputs",target_is_directory=True)
    (selected/"results").mkdir()
    for index in range(5,100):
        (selected/"results"/f"circuit_{index:03d}").symlink_to(
            recovered if index==5 else root/"results"/f"circuit_{index:03d}",target_is_directory=True)
    output=tmp_path/"collected"
    if wrong_target:
        with pytest.raises((ValueError,FileNotFoundError)):
            runner.collect(selected,output)
        assert not output.exists()
    else:
        collected=runner.collect(selected,output)
        assert collected["complete_circuits"]==100
        retained=collected["bindings"][5]
        assert retained["terminal_metadata"]["producer_job"]=="90005"
        assert retained["runner_execution"]==str((recovered/"runner_execution.json").resolve())
        assert collected["source_qualification_granted"] is False


def test_reference_alias_rejects_missing_and_wrong_target(tmp_path):
    actual=tmp_path/"actual";actual.mkdir()
    alias=tmp_path/"alias";alias.symlink_to(actual,target_is_directory=True)
    wrong=tmp_path/"wrong";wrong.mkdir()
    assert runner.same_reference(str(alias),actual)
    assert not runner.same_reference(str(wrong),actual)
    assert not runner.same_reference(str(tmp_path/"missing"),actual)
    assert not runner.same_reference(None,actual)


def compact_reuse_fixture(directory):
    """Preserve a failed old run beside its later bound PASS qualification."""
    delivery=runner.load_json(directory / "delivery.json")
    receipt=runner.load_json(directory / "comparison/qualification.json")
    (directory / "execution.json").unlink()
    comparison={key:receipt[key] for key in ("backend","numpy_version","source_entry_sha256",
        "executor_sha256","drive_sha256","kernel_parameterization_sha256")}
    comparison.update(status="FAILED_NATIVE_STREAM_SOURCE_EXECUTION",global_cases=[],
        global_cases_passed=0,global_arrays_passed=0,input_replay_status="NOT_RUN",
        input_replay_passed_comparisons=0,scientific_fidelity_qualified=False,
        full_scale_qualified=False,production_episodes=0)
    path=directory / "comparison/run.json";path.write_text(json.dumps(comparison))
    delivery["files"]=[row for row in delivery["files"] if row["path"]!="execution.json"]
    delivery["files"].append(runner.file_row(directory,path))
    delivery.update(scientific_fidelity_qualified=False,production_episodes=0)
    (directory / "delivery.json").write_text(json.dumps(delivery))
    return comparison


def test_collect_compact_reused_proof_retains_failed_comparison(tmp_path,monkeypatch):
    root=tmp_path / "prepared";prepared=intake_fixture(root)
    directory=Path(prepared["rows"][0]["qualification"])
    comparison=compact_reuse_fixture(directory)
    monkeypatch.setattr(runner,"validated_preparation",lambda path:prepared)
    result=runner.collect(root,tmp_path / "collected")
    retained=result["bindings"][0]["terminal_metadata"]
    assert retained["kind"]=="compact_reused_qualification_delivery"
    assert retained["qualification_status"]=="PASS_NATIVE_STREAM_SOURCE_EXECUTION"
    assert retained["archived_comparison"]==comparison
    assert retained["archived_comparison"]["status"]=="FAILED_NATIVE_STREAM_SOURCE_EXECUTION"
    assert result["source_qualification_granted"] is False
    with pytest.raises(ValueError,match="Complete bounded stream execution"):
        runner.collected_qualification(directory,prepared["rows"][0]["parameters"])


@pytest.mark.parametrize("damage",["unbound_run","run_hash","run_code","delivery_job"])
def test_collect_compact_reused_proof_rejects_conflicting_metadata(tmp_path,damage):
    root=tmp_path / "prepared";prepared=intake_fixture(root)
    directory=Path(prepared["rows"][0]["qualification"])
    compact_reuse_fixture(directory)
    delivery=runner.load_json(directory / "delivery.json")
    path=directory / "comparison/run.json"
    if damage=="unbound_run":delivery["files"]=[row for row in delivery["files"] if row["path"]!="comparison/run.json"]
    elif damage=="run_hash":path.write_text(path.read_text()+"\n")
    elif damage=="run_code":
        comparison=runner.load_json(path);comparison["drive_sha256"]="wrong";path.write_text(json.dumps(comparison))
        delivery["files"]=[runner.file_row(directory,path) if row["path"]=="comparison/run.json" else row for row in delivery["files"]]
    else:delivery["job_id"]="not-a-job"
    (directory / "delivery.json").write_text(json.dumps(delivery))
    with pytest.raises(ValueError):
        runner.collected_qualification(directory,prepared["rows"][0]["parameters"],allow_compact_reuse=True)


class ParameterQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pfc-parameter-test-", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input_root = self.root / "parameters"
        self.input_root.mkdir()
        rows = []
        for index in range(100):
            parameters = {"sigma_weight_profile": 20.0 + index / 10, "Jpos_excit2excit": 1.6 + index / 1000}
            row = dict(circuit_index=index, parameters=parameters)
            if index < 5:
                row.update(status=runner.REUSE_STATUS, qualification=f"completed/circuit_{index:03d}")
            else:
                row.update(status=runner.PREPARED_STATUS, independent=f"proof_inputs/circuit_{index:03d}/independent.yaml",
                           global_config=f"proof_inputs/circuit_{index:03d}/global.yaml")
                for key, case_count in (("independent", 8), ("global_config", 2)):
                    config = dict(parameters, dt_ms=0.0125, synapse_execution="factorized_uniform_projections",
                                  cases=[dict(case_id=f"case_{i}") for i in range(case_count)])
                    if key == "independent":
                        config["random_stream_policy"] = "independent_initial_external"
                    else:
                        config["source_rng_seed"] = 127001
                    path = self.input_root / row[key]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(yaml.safe_dump(config))
            rows.append(row)
        self.inputs = self.input_root / "qualification_inputs.json"
        self.inputs.write_text(json.dumps(dict(rows=rows, dependent_on_fixed150_quality_assessment=True)))
        (self.input_root / "report.json").write_text(json.dumps(dict(candidate_circuits=100, distinct_effective_kernels=100)))
        self.software = self.root / "software"
        self.software.mkdir()
        self.templates = self.root / "templates"
        for phase, (directory, label, status, _) in runner.PHASES.items():
            archive = self.templates / directory
            archive.mkdir(parents=True)
            launcher = archive / "launcher.sbatch"
            launcher.write_text('#!/bin/bash\nset -euo pipefail\nsrun "$NEURONSIM_PFC_PYTHON" -B -m verified_' + phase + '\n')
            command = ["/qualified/python", "-m", "verified_" + phase + "_tests"]
            (archive / "execution.json").write_text(json.dumps(dict(status=status, commands=[dict(label=label, command=command, status="PASS", exit_code=0)])))
            (archive / (label + ".log")).write_text("verified test output\n")
            module = archive / "Libs/population_dynamics/verified_module.py"
            module.parent.mkdir(parents=True)
            module.write_text("VALUE = 1\n")
            self.write_delivery(archive)
            path = self.software / runner.LAUNCHER_PATHS[phase]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(launcher.read_bytes())
        module = self.software / "Libs/population_dynamics/verified_module.py"
        module.parent.mkdir(parents=True)
        module.write_text("VALUE = 1\n")
        self.write_delivery(self.software)
        self.output = self.root / "prepared"

    def write_delivery(self, root):
        paths = [path for path in root.rglob("*") if path.is_file() and path.name != "delivery.json"]
        (root / "delivery.json").write_text(json.dumps(dict(files=[runner.file_row(root, path) for path in paths])))

    def prepare(self):
        runner.prepare(self.inputs, self.software / "delivery.json", self.templates, self.output)
        return runner.load_json(self.output / "execution.json")

    def assessment(self, prepared):
        summary = dict(status="COMPLETE_REFERENCE_QUALITY_DESCRIPTIVE_COHORT", complete_episodes=150,
                       circuits=list(range(5)), blocks=[3, 4, 5], conditions=[f"condition_{i}" for i in range(10)],
                       core_units=100000, duration_ticks=400000, dt_ms=0.0125,
                       bindings=[dict(episode_id=f"episode_{i}") for i in range(150)])
        summary_path = self.root / "quality.json"
        summary_path.write_text(json.dumps(summary))
        assessment = dict(approved_remaining_reference_kernel_checks=True, scope="remaining95_reference_kernel_checks",
                          rationale="Complete development quality cohort assessed for exact kernel checks.",
                          qualification_inputs_sha256=prepared["qualification_inputs_sha256"],
                          fixed150_quality_summary=dict(path=summary_path.name, sha256=runner.digest(summary_path)))
        path = self.root / "assessment.json"
        path.write_text(json.dumps(assessment))
        return path, summary_path

    def test_prepare_preserves_exact_inputs_and_unchanged_templates(self):
        with patch.object(runner.subprocess, "run") as run:
            prepared = self.prepare()
        run.assert_not_called()
        self.assertEqual(prepared["new_source_simulations"], 0)
        self.assertEqual(sum(row["status"] == runner.PREPARED_STATUS for row in prepared["rows"]), 95)
        self.assertEqual(sum(row["status"] == runner.REUSE_STATUS for row in prepared["rows"]), 5)
        runner.validated_preparation(self.output)
        for phase, (directory, _, _, _) in runner.PHASES.items():
            self.assertEqual((self.output / f"templates/{phase}.sbatch").read_bytes(), (self.templates / directory / "launcher.sbatch").read_bytes())
        self.assertEqual((self.output / "inputs/proof_inputs/circuit_005/independent.yaml").read_bytes(),
                         (self.input_root / "proof_inputs/circuit_005/independent.yaml").read_bytes())
        self.assertFalse(prepared["source_qualification_granted"])
        launcher = (self.output / "launcher.sbatch").read_text()
        self.assertIn("--execute --prepared", launcher)
        self.assertIn("exec " + prepared["templates"]["stream"]["python"] + " -B ", launcher)
        durable_runner = self.output / "runner/run_pfc_reference_parameter_qualification.py"
        self.assertIn(str(durable_runner), launcher)
        self.assertNotIn(str(Path(runner.__file__).resolve()) + " --execute", launcher)
        self.assertEqual(durable_runner.read_bytes(), Path(runner.__file__).read_bytes())
        self.assertIn("runner/run_pfc_reference_parameter_qualification.py",
                      {row["path"] for row in prepared["staged_files"]})

    def test_exact_tuple_mismatch_and_duplicate_kernels_rejected(self):
        path = self.input_root / "proof_inputs/circuit_005/global.yaml"
        config = yaml.safe_load(path.read_text())
        config["sigma_weight_profile"] += 0.1
        path.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "exact parameter mismatch"):
            runner.validate_inputs(self.inputs)
        inputs = runner.load_json(self.inputs)
        inputs["rows"][6]["parameters"] = inputs["rows"][5]["parameters"]
        self.inputs.write_text(json.dumps(inputs))
        with self.assertRaisesRegex(ValueError, "Distinct exact"):
            runner.validate_inputs(self.inputs)

    def test_changed_archived_software_and_prepared_input_rejected(self):
        module = self.software / "Libs/population_dynamics/verified_module.py"
        module.write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "Changed archived binding"):
            self.prepare()
        module.write_text("VALUE = 1\n")
        shutil.rmtree(self.output)
        self.prepare()
        path = self.output / "inputs/proof_inputs/circuit_005/independent.yaml"
        path.write_text(path.read_text() + "purpose: changed\n")
        with self.assertRaisesRegex(ValueError, "Changed archived binding"):
            runner.validated_preparation(self.output)

    def test_changed_archived_launcher_rejected(self):
        phase = self.templates / runner.PHASES["native"][0] / "launcher.sbatch"
        phase.write_text("#!/bin/bash\ntrue\n")
        with self.assertRaisesRegex(ValueError, "Changed archived binding"):
            self.prepare()

    def test_summary_alone_and_wrong_input_binding_rejected(self):
        prepared = self.prepare()
        with self.assertRaisesRegex(ValueError, "Explicit fixed150"):
            runner.admission(None, prepared)
        path, summary = self.assessment(prepared)
        with self.assertRaisesRegex(ValueError, "Explicit remaining"):
            runner.admission(summary, prepared)
        assessment = runner.load_json(path)
        assessment["qualification_inputs_sha256"] = "wrong"
        path.write_text(json.dumps(assessment))
        with self.assertRaisesRegex(ValueError, "input binding differs"):
            runner.admission(path, prepared)

    def test_explicit_admission_requires_complete_unchanged_cohort(self):
        prepared = self.prepare()
        path, summary_path = self.assessment(prepared)
        self.assertEqual(runner.admission(path, prepared)["scope"], "remaining95_reference_kernel_checks")
        summary = runner.load_json(summary_path)
        summary["bindings"].pop()
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "summary binding differs"):
            runner.admission(path, prepared)
        assessment = runner.load_json(path)
        assessment["fixed150_quality_summary"]["sha256"] = runner.digest(summary_path)
        path.write_text(json.dumps(assessment))
        with self.assertRaisesRegex(ValueError, "150 distinct"):
            runner.admission(path, prepared)

    def test_phase_environment_preserves_job_identity_and_chain(self):
        prepared = self.prepare()
        row = prepared["rows"][5]
        base = dict(SLURM_JOB_ID="1234", SLURMD_NODENAME="cpu05")
        for phase in runner.PHASES:
            env = runner.phase_environment(base, Path("/tmp/staged"), Path("/archive/circuit_005"), prepared, row, phase)
            self.assertEqual(env["SLURM_JOB_ID"], "1234")
            self.assertEqual(env["NEURONSIM_PFC_PLAIN_REFERENCE"], "execute")
            self.assertEqual(env["NEURONSIM_PFC_REPLAY_BACKEND"], "numpy")
            self.assertEqual(env["NEURONSIM_PFC_PYTHON"], prepared["templates"][phase]["python"])
            self.assertTrue(env["NEURONSIM_PFC_CONFIG"].endswith("circuit_005/independent.yaml"))
            self.assertTrue(env["NEURONSIM_PFC_GLOBAL_CONFIG"].endswith("circuit_005/global.yaml"))
            if phase in ("pairing", "stream"):
                self.assertEqual(env["NEURONSIM_PFC_VALIDATION"], "/archive/circuit_005/native_validation_1234")
            if phase == "stream":
                self.assertEqual(env["NEURONSIM_PFC_PAIRING"], "/archive/circuit_005/pairing_1234")
                self.assertEqual(env["NEURONSIM_PFC_GLOBAL_REFERENCE"], "/archive/circuit_005/native_validation_1234/global_reference")

    def test_reuse_missing_interpreter_and_existing_attempt_start_no_simulation(self):
        prepared = self.prepare()
        path, _ = self.assessment(prepared)
        with patch.object(runner.subprocess, "run") as run:
            self.assertEqual(runner.execute(self.output, 0, None)["status"], runner.REUSE_STATUS)
            with patch.dict(os.environ, dict(SLURM_JOB_ID="1234", SLURMD_NODENAME="cpu05"), clear=True):
                with patch.object(runner.os, "access", return_value=False):
                    with self.assertRaisesRegex(ValueError, "interpreter unavailable"):
                        runner.execute(self.output, 5, path)
                archive = self.output / "results/circuit_005"
                archive.mkdir(parents=True)
                (archive / "runner_execution.json").write_text('{"status":"RUNNING_NATIVE"}')
                with patch.object(runner.os, "access", return_value=True):
                    with self.assertRaises(FileExistsError):
                        runner.execute(self.output, 5, path)
        run.assert_not_called()
        self.assertEqual((archive / "runner_execution.json").read_text(), '{"status":"RUNNING_NATIVE"}')


if __name__ == "__main__":
    unittest.main()
