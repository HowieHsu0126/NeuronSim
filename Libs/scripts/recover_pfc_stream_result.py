"""Recover a complete retained stream comparison after a failed resource wrapper."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


def recover(parent_path: Path, output: Path) -> dict:
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURMD_NODENAME") not in ("cpu03", "cpu04", "cpu05", "cpu06"):
        raise ValueError("PFC_STREAM_RECOVERY_ALLOCATED_CPU_REQUIRED")
    if os.environ.get("SLURM_JOB_GPUS") or output.exists():
        raise ValueError("PFC_STREAM_RECOVERY_FRESH_CPU_OUTPUT_REQUIRED")
    parent=json.loads(parent_path.read_text())
    phase=parent.get("phases", {}).get("stream", {})
    if parent.get("status")!="FAILED" or parent.get("failed_phase")!="stream" or phase.get("status")!="FAILED_LAUNCHER":
        raise ValueError("PFC_STREAM_RECOVERY_FAILED_STREAM_REQUIRED")
    stream=Path(phase["retained_phase_runtime"])
    original=json.loads((stream/"execution.json").read_text())
    if (original.get("status")!="FAILED" or not str(original.get("error", "")).startswith("PFC_EQUATIONS_RESOURCE_BLOCKED:")):
        raise ValueError("PFC_STREAM_RECOVERY_TERMINAL_RESOURCE_FAILURE_REQUIRED")
    sys.path.insert(0,str(stream/"Libs"))
    from population_dynamics.provenance import sha256_file
    from population_dynamics.pfc_deterministic_replay import _array_comparison
    import numpy as np
    # The retained execution's modules remain the source identity. The current
    # validator supplies only its already tested complete-phase reuse consumer.
    package=Path(__file__).resolve().parents[1]/"population_dynamics"
    spec=importlib.util.spec_from_file_location("population_dynamics.stream_recovery_validator",package/"pfc_stream_validation.py")
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    resource_spec=importlib.util.spec_from_file_location("pfc_recovery_resource_probe",package/"resource_probe.py")
    resource=importlib.util.module_from_spec(resource_spec);resource_spec.loader.exec_module(resource)
    work=Path(os.environ.get("TMPDIR", "/tmp"))/f"pfc-stream-recovery-{os.environ['SLURM_JOB_ID']}-{parent['circuit_index']}"
    work.mkdir(exist_ok=False)
    try:
        health=resource.probe(work/"resource.json",4)
        if health["status"]!="PASS":raise ValueError("PFC_STREAM_RECOVERY_RESOURCE_BLOCKED:"+",".join(health["reasons"]))
        inputs,recorded,reuse=validator.reuse_completed_phases(stream/"validation",stream/"pairing.json",stream/"comparison",work,"numpy")
        proof=json.loads((stream/"comparison/qualification.json").read_text())
        if (proof.get("status")!="PASS_NATIVE_STREAM_SOURCE_EXECUTION" or proof.get("input_replay_passed_comparisons")!=24
                or proof.get("global_cases_passed")!=2 or proof.get("global_arrays_passed")!=10):
            raise ValueError("PFC_STREAM_RECOVERY_COMPLETE_PROOF_REQUIRED")
        if (proof["input_replay_sha256"]!=sha256_file(stream/"comparison"/proof["input_replay_file"])
                or proof["global_reference_delivery_sha256"]!=sha256_file(stream/"reference/delivery.json")
                or proof["global_reference_report_sha256"]!=sha256_file(stream/"reference/plain/report.json")):
            raise ValueError("PFC_STREAM_RECOVERY_PROOF_BINDING_CHANGED")
        reference=json.loads((stream/"reference/plain/report.json").read_text())
        raw_checks=[]
        for case in proof["global_cases"]:
            directory=stream/"comparison"/case["case_id"]
            if sha256_file(directory/"raw.npz")!=case["raw_sha256"] or json.loads((directory/"comparison.json").read_text())!=case:
                raise ValueError("PFC_STREAM_RECOVERY_GLOBAL_OUTPUT_CHANGED")
            target=next(row for row in reference["cases"] if row["case_id"]==case["case_id"])
            expected=stream/"reference/plain"/target["raw_file"]
            if sha256_file(expected)!=target["raw_sha256"]:raise ValueError("PFC_STREAM_RECOVERY_GLOBAL_REFERENCE_CHANGED")
            with np.load(directory/"raw.npz",allow_pickle=False) as actual,np.load(expected,allow_pickle=False) as truth:
                checks=[_array_comparison(case["case_id"],"raw",key,actual[key],truth[key]) for key in truth.files]
            if not case["passed"] or not all(row["passed"] for row in checks):raise ValueError("PFC_STREAM_RECOVERY_GLOBAL_COMPARISON_FAILED")
            raw_checks.extend(checks)
        output.mkdir(parents=True,exist_ok=False)
        for name in ("native","pairing"):
            source=Path(parent["phases"][name]["archive"])
            (output/source.name).symlink_to(source,target_is_directory=True)
        archive=output/f"stream_numpy_{original['job_id']}";archive.mkdir()
        files=[]
        for source in stream.rglob("*"):
            name=source.relative_to(stream)
            if not source.is_file() or set(name.parts)&{"validation","reference","__pycache__"} or str(name)=="execution.json":continue
            target=archive/name;target.parent.mkdir(parents=True,exist_ok=True)
            checksum=sha256_file(source);shutil.copy2(source,target)
            if sha256_file(target)!=checksum:raise ValueError("PFC_STREAM_RECOVERY_COPY_CHANGED")
            files.append(dict(path=str(name),bytes=target.stat().st_size,sha256=checksum))
        shutil.copy2(stream/"execution.json",archive/"original_failed_execution.json")
        recovery=dict(recovery_job=os.environ["SLURM_JOB_ID"],scope="validate_and_archive_completed_stream_outputs",
            original_wrapper_status="FAILED",original_wrapper_error=original["error"],
            original_execution_sha256=sha256_file(stream/"execution.json"),
            original_phase_process_exit_code=original["commands"][-1]["exit_code"],
            source_simulations=0,completed_phase_reuse=reuse,raw_checks=raw_checks)
        execution=dict(status="PASS_NATIVE_STREAM_SOURCE_EXECUTION",backend="numpy",job_id=original["job_id"],
            commands=original["commands"],completed_cases=2,passed_cases=2,comparison_exit_code=0,
            scientific_fidelity_qualified=False,production_episodes=0,recovery=recovery)
        (archive/"execution.json").write_text(json.dumps(execution,indent=2)+"\n")
        for name in ("original_failed_execution.json","execution.json"):
            path=archive/name;files.append(dict(path=name,bytes=path.stat().st_size,sha256=sha256_file(path)))
        delivery=dict(status="ARCHIVED_PFC_STREAM_VALIDATION",result_status=execution["status"],backend="numpy",job_id=original["job_id"],
            files=files,source_validation=parent["phases"]["native"]["archive"],pairing_source=parent["phases"]["pairing"]["archive"],
            global_reference=parent["phases"]["native"]["archive"]+"/global_reference",scientific_fidelity_qualified=False,
            production_episodes=0,recovery_job_id=os.environ["SLURM_JOB_ID"])
        (archive/"delivery.json").write_text(json.dumps(delivery,indent=2)+"\n")
        phases={name:dict(parent["phases"][name],archive=str(output/Path(parent["phases"][name]["archive"]).name)) for name in ("native","pairing")}
        phases["stream"]=dict(status=execution["status"],archive=str(archive),launcher_exit_code=0,delivery_sha256=sha256_file(archive/"delivery.json"))
        result=dict(status="PASS_EXACT_TUPLE_REFERENCE_KERNEL_CHECKS",circuit_index=parent["circuit_index"],parameters=parent["parameters"],
            qualification_inputs_sha256=parent["qualification_inputs_sha256"],assessment=parent["assessment"],
            producer_job=os.environ["SLURM_JOB_ID"],node=os.environ["SLURMD_NODENAME"],phases=phases,
            source_qualification_granted=False,scientific_parameter_domain_qualified=False,production_episodes=0,source_simulations=0,
            native_resume=dict(parent_job_id=original["job_id"],parent_status="FAILED",parent_execution_sha256=sha256_file(parent_path)),
            recovery_scope=recovery)
        (output/"runner_execution.json").write_text(json.dumps(result,indent=2)+"\n")
        return result
    finally:
        shutil.rmtree(work)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-result",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();print(json.dumps(recover(args.parent_result,args.output)))


if __name__=="__main__":main()
