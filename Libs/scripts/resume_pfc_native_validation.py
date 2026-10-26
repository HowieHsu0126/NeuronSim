#!/usr/bin/env python3
"""Resume retained native validation without repeating committed plain cases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import yaml


def validate_native_output(output: Path, config_path: Path, source: Path, *, complete: bool) -> list[str]:
    """Use the existing complete-case validator before any resumed execution."""
    from population_dynamics import pfc_reference as reference
    from population_dynamics import pfc_kernel_parameters as kernels
    from population_dynamics.circuit_data import CircuitUnits
    from population_dynamics.provenance import sha256_file

    config = yaml.safe_load(config_path.read_text())
    manifest = json.loads((source / "source.json").read_text())
    entry = next(row for row in manifest["files"] if row["path"] == reference.SOURCE_ENTRY)
    if (config.get("scope") != "source_reproduction"
            or config.get("expected_source_revision") != manifest.get("revision")
            or manifest.get("repository") != "https://github.com/EPFL-LCN/neuronaldynamics-exercises"
            or sha256_file(source / entry["path"]) != entry["sha256"]):
        raise ValueError("PFC_NATIVE_RESUME_SOURCE_CHANGED")
    provenance = kernels.circuit_provenance(config, entry["sha256"])
    n_e, n_i, fields = config["excitatory_neurons"], config["inhibitory_neurons"], config["excitatory_preference_bins"]
    units = CircuitUnits("pfc_source_reference", np.arange(n_e+n_i), np.r_[np.full(n_e,"E"),np.full(n_i,"I")],
        np.r_[np.arange(n_e)*fields//n_e,np.full(n_i,fields)],
        tuple([f"E_preference_{i*360/fields:g}_{(i+1)*360/fields:g}_deg" for i in range(fields)]+["I_population"]),
        "E angle=360*source_index/N_E degrees; I has no preference coordinate",manifest["revision"],circuit_provenance=provenance)
    binding = dict(schema=reference.SCHEMA,scope="source_reproduction",scenario_id="pfc_working_memory",
        source_revision=manifest["revision"],sample_unit="complete_mechanistic_episode",config_sha256=sha256_file(config_path),
        source_entry_sha256=entry["sha256"],resolved_source_config=config,synapse_execution=config["synapse_execution"],
        resolved_kernel_parameters=kernels.resolve_kernel_parameters(config),
        kernel_parameterization_sha256=sha256_file(Path(kernels.__file__)),
        execution_implementation_sha256=sha256_file(Path(reference.__file__).with_name("pfc_factorized_source.py")))
    cases=reference.case_settings(config)
    if cases[0]["random_stream_policy"]=="independent_initial_external":
        binding["initial_voltage_implementation_sha256"]=sha256_file(Path(reference.__file__).with_name("pfc_initial_rng.py"))
    recovered=reference.resume_cases(output,binding,units,cases,round(config["duration_ms"]/config["dt_ms"]))
    if complete:
        report=json.loads((output/"report.json").read_text())
        if (set(recovered)!={row["case_id"] for row in cases}
                or report.get("status")!="NATIVE_SOURCE_AND_ARCHIVE_REPLAY_PASS"
                or report.get("resolved_config")!=config
                or report.get("source_entry_sha256")!=entry["sha256"]
                or report.get("circuit_provenance")!=provenance
                or report.get("kernel_parameterization_sha256")!=binding["kernel_parameterization_sha256"]
                or report.get("execution_implementation_sha256")!=binding["execution_implementation_sha256"]
                or report.get("initial_voltage_implementation_sha256")!=binding.get("initial_voltage_implementation_sha256")):
            raise ValueError("PFC_NATIVE_RESUME_COMPLETE_OUTPUT_CHANGED")
    return list(recovered)


def resume_native(retained: Path, independent: Path, global_config: Path, output: Path) -> dict:
    from population_dynamics.resource_probe import probe
    from population_dynamics.provenance import sha256_file

    parent=json.loads((retained/"execution.json").read_text())
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURMD_NODENAME") not in ("cpu03","cpu04","cpu05","cpu06")
            or os.environ.get("SLURM_JOB_GPUS") or output.exists()):
        raise ValueError("PFC_NATIVE_RESUME_FRESH_CPU_ATTEMPT_REQUIRED")
    if (not isinstance(parent.get("job_id"),str) or not parent["job_id"].isdecimal()
            or parent.get("scientific_fidelity_qualified") not in (None,False)
            or parent.get("production_episodes") not in (None,0)):
        raise ValueError("PFC_NATIVE_RESUME_PARENT_IDENTITY_REQUIRED")
    for config,label in ((independent,"plain"),(global_config,"global_plain")):
        existing=retained/"Libs/configs"/config.name
        # The source launcher can stop after plain before staging global.yaml.
        # A missing global config is admissible only if global execution did not start.
        unstarted_global=(label=="global_plain" and not existing.exists()
            and not (retained/"global_reference").exists()
            and not any(row.get("label")==label for row in parent.get("commands",[])))
        if not unstarted_global and (not existing.is_file() or existing.read_bytes()!=config.read_bytes()):
            raise ValueError("PFC_NATIVE_RESUME_CONFIG_CHANGED:"+label)
    source=retained/"source"
    manifest=json.loads((source/"source.json").read_text())
    for row in manifest["files"]:
        if row["path"] in ("neurodynex3/__init__.py","neurodynex3/tools/__init__.py",
            "neurodynex3/tools/plot_tools.py","neurodynex3/tools/spike_tools.py",
            "neurodynex3/working_memory_network/wm_model.py"):
            if sha256_file(source/row["path"])!=row["sha256"]:
                raise ValueError("PFC_NATIVE_RESUME_SOURCE_CHANGED")
    tests=[row for row in parent.get("commands",[]) if row.get("label")=="tests"
           and row.get("status") in ("PASS","REUSED_PASS") and row.get("exit_code")==0]
    if len(tests)!=1:
        raise ValueError("PFC_NATIVE_RESUME_COMPLETED_TEST_BINDING_REQUIRED")
    delivery_path=retained/"delivery.json"
    if delivery_path.is_file():
        delivery=json.loads(delivery_path.read_text())
        if delivery.get("status")!="ARCHIVED_NATIVE_INPUT_OBSERVATION_PARITY":
            raise ValueError("PFC_NATIVE_RESUME_ARCHIVE_INCOMPLETE")
        names=set()
        for row in delivery["files"]:
            name=Path(row["path"])
            if name.is_absolute() or ".." in name.parts or str(name) in names:
                raise ValueError("PFC_NATIVE_RESUME_ARCHIVE_PATH_CHANGED")
            names.add(str(name));path=retained/name
            if path.stat().st_size!=row["bytes"] or sha256_file(path)!=row["sha256"]:
                raise ValueError("PFC_NATIVE_RESUME_ARCHIVE_CHANGED:"+str(name))
        if "execution.json" not in names:
            raise ValueError("PFC_NATIVE_RESUME_ARCHIVE_EXECUTION_REQUIRED")
    plain_complete=(retained/"plain/report.json").is_file()
    reused=validate_native_output(retained/"plain",independent,source,complete=plain_complete)
    global_path=retained/"global_reference/plain"
    global_complete=(global_path/"report.json").is_file()
    global_reused=(validate_native_output(global_path,global_config,source,complete=global_complete)
                   if global_path.exists() else [])
    scratch=Path(os.environ.get("NEURONSIM_PFC_SCRATCH","/tmp"))
    local=scratch/("neuronsim-pfc-input-validation-"+os.environ["SLURM_JOB_ID"])
    local.mkdir(parents=True,exist_ok=False)
    record=dict(status="PREPARING",job_id=os.environ["SLURM_JOB_ID"],runtime=str(local),
        parent_runtime=str(retained),parent_job_id=parent.get("job_id"),parent_execution_sha256=sha256_file(retained/"execution.json"),
        parent_execution=parent,reused_plain_case_ids=reused,reused_global_case_ids=global_reused,commands=[],
        scientific_fidelity_qualified=False,production_episodes=0)
    def save():
        (local/"execution.json").write_text(json.dumps(record,indent=2)+"\n")
    def health():
        h=probe(local/"resource.json",required_memory_gib=2)
        if h["status"]!="PASS":raise RuntimeError("PFC_NATIVE_RESUME_RESOURCE_BLOCKED:"+",".join(h["reasons"]))
    try:
        health()
        shutil.copytree(retained,local,dirs_exist_ok=True,ignore=shutil.ignore_patterns("mpl_cache","cache","__pycache__"))
        for config in (independent,global_config):
            destination=local/"Libs/configs"/config.name
            if not destination.exists():shutil.copy2(config,destination)
        save()
        def run(command,label):
            health();row=dict(label=label,command=command,status="RUNNING");record["commands"].append(row);save()
            with (local/(label+"_resume.log")).open("x") as log:
                child=subprocess.Popen(command,cwd=local,stdout=log,stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:health()
                except BaseException:
                    if child.poll() is None:child.terminate();child.wait()
                    row.update(status="STOPPED",exit_code=child.returncode);save();raise
            row.update(status="PASS" if child.returncode==0 else "FAILED",exit_code=child.returncode);save()
            if child.returncode:raise RuntimeError("PFC_NATIVE_RESUME_COMMAND_FAILED:"+label)
        plain=local/"plain";global_plain=local/"global_reference/plain"
        record["commands"].append(dict(tests[0],status="REUSED_PASS",command_started=False,
            parent_job_id=parent.get("job_id")))
        if plain_complete:
            record["commands"].append(dict(label="plain",status="REUSED_PASS",command_started=False,exit_code=0,
                source_archive=str(retained/"plain"),source_report_sha256=sha256_file(retained/"plain/report.json")))
        else:
            run([os.sys.executable,"-B","-m","population_dynamics.pfc_reference","--source",str(local/"source"),
                "--config",str(local/"Libs/configs"/independent.name),"--output",str(plain),"--resume"],"plain")
            validate_native_output(plain,independent,source,complete=True)
        if global_complete:
            record["commands"].append(dict(label="global_plain",status="REUSED_PASS",command_started=False,exit_code=0))
        else:
            command=[os.sys.executable,"-B","-m","population_dynamics.pfc_reference","--source",str(local/"source"),
                "--config",str(local/"Libs/configs"/global_config.name),"--output",str(global_plain)]
            if global_plain.exists():command.append("--resume")
            run(command,"global_plain")
        prior_comparison=retained/"comparison/comparison.json"
        completed_observers=((parent.get("status")=="PASS_NATIVE_INPUT_OBSERVATION_PARITY"
            or (parent.get("status")=="FAILED" and str(parent.get("error", "")).startswith("PFC_INPUT_ARCHIVE_RESOURCE_BLOCKED:")))
            and global_complete and prior_comparison.is_file()
            and all(any(row.get("label")==label and row.get("status") in ("PASS","REUSED_PASS")
                        and row.get("exit_code")==0 for row in parent.get("commands",[]))
                    for label in ("audit","capture","compare")))
        if completed_observers:
            from population_dynamics.pfc_input_capture import _load
            comparison=json.loads(prior_comparison.read_text())
            if comparison!=parent.get("comparison"):
                raise ValueError("PFC_NATIVE_RESUME_COMPARISON_CHANGED")
            for mode in ("audit","capture"):
                execution,_=_load(local/mode,mode)
                if (execution["config_sha256"]!=sha256_file(independent)
                        or comparison[mode+"_execution_sha256"]!=sha256_file(local/mode/"execution.json")):
                    raise ValueError("PFC_NATIVE_RESUME_OBSERVER_BINDING_CHANGED")
            if not delivery_path.is_file():
                # A retained complete phase has no manifest. Recompute only the
                # comparison from existing measurements, never the simulations.
                run([os.sys.executable,"-B","-m","population_dynamics.pfc_input_capture","compare",
                    "--audit",str(local/"audit"),"--capture",str(local/"capture"),"--plain",str(plain),
                    "--output",str(local/"resumed_comparison")],"compare_existing")
                if json.loads((local/"resumed_comparison/comparison.json").read_text())!=comparison:
                    raise ValueError("PFC_NATIVE_RESUME_COMPARISON_CHANGED")
            for label in ("audit","capture","compare"):
                record["commands"].append(dict(label=label,status="REUSED_PASS",command_started=False,
                    exit_code=0,parent_job_id=parent["job_id"]))
        else:
            for mode in ("audit","capture"):
                destination=local/mode
                if destination.exists():
                    from population_dynamics.pfc_input_capture import _load
                    if not any(row.get("label")==mode and row.get("status") in ("PASS","REUSED_PASS")
                            and row.get("exit_code")==0 for row in parent.get("commands",[])):
                        raise ValueError("PFC_NATIVE_RESUME_UNRESOLVED_OBSERVER_OUTPUT:"+mode)
                    execution,_=_load(destination,mode)
                    if execution["config_sha256"]!=sha256_file(independent):
                        raise ValueError("PFC_NATIVE_RESUME_OBSERVER_BINDING_CHANGED")
                    validate_native_output(destination/"native",independent,source,complete=True)
                    record["commands"].append(dict(label=mode,status="REUSED_PASS",command_started=False,
                        exit_code=0,parent_job_id=parent["job_id"]))
                else:
                    run([os.sys.executable,"-B","-m","population_dynamics.pfc_input_capture","record","--source",str(local/"source"),
                        "--config",str(local/"Libs/configs"/independent.name),"--output",str(destination),"--mode",mode],mode)
            # Keep partial parent comparisons for diagnosis; write a fresh comparison.
            if (local/"comparison").exists():
                (local/"comparison").rename(local/"parent_incomplete_comparison")
            run([os.sys.executable,"-B","-m","population_dynamics.pfc_input_capture","compare","--audit",str(local/"audit"),
                 "--capture",str(local/"capture"),"--plain",str(plain),"--output",str(local/"comparison")],"compare")
            comparison=json.loads((local/"comparison/comparison.json").read_text())
        if comparison.get("status")!="PASS_NATIVE_INPUT_OBSERVATION_PARITY" or comparison.get("plain_source_execution_compared") is not True:
            raise ValueError("PFC_NATIVE_RESUME_COMPLETE_COMPARISON_REQUIRED")
        global_root=local/"global_reference"
        files=[dict(path=str(p.relative_to(global_root)),bytes=p.stat().st_size,sha256=sha256_file(p))
               for p in (global_root/"plain").rglob("*") if p.is_file()]
        (global_root/"delivery.json").write_text(json.dumps(dict(status="ARCHIVED_NATIVE_GLOBAL_REFERENCE",files=files,
            scientific_fidelity_qualified=False,production_episodes=0),indent=2)+"\n")
        record.update(status="PASS_NATIVE_INPUT_OBSERVATION_PARITY",comparison=comparison,completed_utc=datetime.now(timezone.utc).isoformat());save()
        health();output.mkdir(parents=True,exist_ok=False);files=[];last=time.monotonic()
        for path in local.rglob("*"):
            if path==local/"delivery.json" or not path.is_file() or set(path.relative_to(local).parts)&{"mpl_cache","cache","__pycache__"}:continue
            if time.monotonic()-last>=30:health();last=time.monotonic()
            destination=output/path.relative_to(local);destination.parent.mkdir(parents=True,exist_ok=True)
            checksum=sha256_file(path);shutil.copy2(path,destination)
            if sha256_file(destination)!=checksum:raise ValueError("PFC_NATIVE_RESUME_ARCHIVE_CHANGED")
            files.append(dict(path=str(path.relative_to(local)),bytes=destination.stat().st_size,sha256=checksum))
        (output/"delivery.json").write_text(json.dumps(dict(status="ARCHIVED_NATIVE_INPUT_OBSERVATION_PARITY",files=files,
            scientific_fidelity_qualified=False,production_episodes=0,parent_job_id=parent.get("job_id"),
            comparison=comparison),indent=2)+"\n")
        shutil.rmtree(local)
        return record
    except BaseException as error:
        record.update(status="FAILED",error=str(error),retained_runtime=str(local));save()
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("retained","independent-config","global-config","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(resume_native(args.retained,args.independent_config,args.global_config,args.output),indent=2))


if __name__=="__main__":main()
