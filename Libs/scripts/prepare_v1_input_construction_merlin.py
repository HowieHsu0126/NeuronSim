#!/usr/bin/env python3
"""Prepare a CPU request for the two remaining V1 input constructions.

Run on the existing transfer host. Reuse the verified 853-file upload and add
only declared source/runtime files. This command prepares; it never submits.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import re
import socket
import sys
import zlib

from v1_input_construction_merlin import input_dependencies


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(repo: Path, operation: Path, name: str, payload_upload: Path | None = None) -> dict:
    if socket.gethostname().split(".")[0] == "mgmt2":
        raise ValueError("V1_INPUT_PREPARATION_REQUIRES_TRANSFER_OR_COMPUTE_HOST")
    output = operation / "v1_input_construction"
    output.mkdir(exist_ok=True)
    qualification = repo / "Output/neural_population_dynamics/benchmark/v1_source_io_recovery_20261018/input_implementation_qualification"
    cohort = repo / "Output/neural_population_dynamics/benchmark/v1_source_qualification_20261016"
    base = read_json(operation / "v1_construction_intake/input_files.json")
    upload = read_json(operation / "v1_construction_intake/upload_4mib.json")
    if not upload["complete"] or upload["sha256"] != "66b3089b67c63d596b7b25bf27572591ad987d9763920ce56f6d0993adfd54c4":
        raise ValueError("V1_INPUT_EXISTING_VERIFIED_BASE_UPLOAD_REQUIRED")
    dependencies, tasks = set(), []
    for circuit in (98, 99):
        config = cohort / f"circuit_{circuit:03d}/circuit.yaml"
        record = cohort / f"construction_expansion_20261017/circuit_{circuit:03d}/recurrent/report.json"
        dependencies.update(input_dependencies(repo, config, record, qualification))
        tasks.append(dict(circuit_index=circuit, config=str(config), recurrent_record=str(record),
                          source_slurm_job=f"624319_{circuit}"))
    supplemental, contents, overrides, reused = {}, {}, {}, []
    for path in sorted(dependencies):
        name_key = str(path)
        # Large recurrent files are already in the verified base. Remote intake
        # authenticates them against that manifest; no local rehash or reupload.
        if name_key in base and path.suffix not in (".py", ".json", ".yaml", ".sbatch"):
            reused.append(name_key)
            continue
        data = path.read_bytes()
        identity = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        if base.get(name_key) == identity:
            reused.append(name_key)
            continue
        if name_key in base:
            allowed = {repo / "Libs/population_dynamics/v1_circuit_inputs.py",
                       repo / "Libs/population_dynamics/provenance.py"}
            if path not in allowed:
                raise ValueError("V1_INPUT_UNDECLARED_BASE_OVERRIDE:" + name_key)
            overrides[name_key] = dict(base_sha256=base[name_key]["sha256"],
                                       executed_sha256=identity["sha256"],
                                       reason="Merlin CPU execution identity; frozen numerical source retained")
        supplemental[name_key] = identity
        contents[name_key] = base64.b64encode(data).decode("ascii")
    template = read_json(operation / "v1_reference_preparation/request.json")
    script = template["job_config"]["job_template_config"]["entrypoint_full_script"]
    original_payload = json.loads(zlib.decompress(base64.b64decode(re.search(r"b64decode\('([^']+)'\)", script)[1])))
    packages = list(original_payload["job"]["python_packages"])
    packages.extend(["pytest==8.4.2", "mpi4py==4.1.1", "mpich==4.3.2"])
    resource_shape = dict(name="head", cpu=4, gpu=0, memory_mb=81920)
    job = dict(repo=str(repo), qualification=str(qualification), tasks=tasks,
               input_files=base, upload=upload, supplemental_files=supplemental,
               supplemental_contents=contents, runtime_overrides=overrides,
               resource_shape=resource_shape, python_version="3.13.9", uv_version="0.9.4",
               python_packages=packages, tests=[str(repo / "Libs/population_dynamics" / name)
                   for name in ("test_v1_circuit_inputs.py", "test_v1_input_construction_merlin.py")])
    runtime = repo / "Libs/scripts/v1_input_construction_merlin_remote.py"
    transport = repo / "Libs/scripts/ca3_merlin_remote.py"
    files = {"run_inputs.py": runtime.read_text(), "ca3_merlin_remote.py": transport.read_text(),
             "pack_results.py": original_payload["files"]["pack_results.py"]}
    for filename, body in files.items():
        ast.parse(body, filename)
    payload = zlib.compress(json.dumps(dict(job=job, files=files)).encode())
    (output / "payload.zlib").write_bytes(payload)
    payload_identity = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    uploaded = read_json(payload_upload) if payload_upload else None
    if uploaded and (uploaded["bytes"] != payload_identity["bytes"]
                     or uploaded["sha256"] != payload_identity["sha256"]):
        raise ValueError("V1_INPUT_UPLOADED_EXECUTION_PAYLOAD_CHANGED")
    payload_reference = json.dumps(uploaded or {}, separators=(",", ":"))
    entrypoint = """#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
python3 - <<'MRLN_INPUT_CONSTRUCTION'
import hashlib,json,os,pathlib,subprocess,sys,zlib
work=pathlib.Path('/tmp')/('neuronsim-v1-input-construction-'+os.environ['ARNOLD_TRIAL_ID'])
work.mkdir(exist_ok=False)
reference=json.loads('""" + payload_reference + """')
path=work/'payload.zlib'
subprocess.run(['hdfs','dfs','-get',reference['hdfs_path'],str(path)],check=True)
data=path.read_bytes()
if len(data)!=reference['bytes'] or hashlib.sha256(data).hexdigest()!=reference['sha256']:
 raise ValueError('V1_INPUT_EXECUTION_PAYLOAD_IDENTITY_DIFFERS')
payload=json.loads(zlib.decompress(data))
for name,text in payload['files'].items():(work/name).write_text(text)
(work/'job.json').write_text(json.dumps(payload['job']))
raise SystemExit(subprocess.call([sys.executable,'-B',str(work/'run_inputs.py')]))
MRLN_INPUT_CONSTRUCTION
"""
    request = copy.deepcopy(template)
    request["name"] = name
    request["description"] = "mrln_ C98/C99 complete V1 input construction and independent replay; source namespaces preserved; no episodes or admission granted."
    request["job_config"]["job_template_config"]["entrypoint_full_script"] = entrypoint
    role = request["resource_config"]["arnold_resource_config"]["roles"][0]
    role.update(resource_shape)
    request.pop("request_context", None)
    plan = dict(status="PREPARED_NOT_SUBMITTED" if uploaded else "PAYLOAD_UPLOAD_REQUIRED",
                payload=payload_identity, tasks=tasks, resource_shape=resource_shape,
                base_input_files=len(base), reused_dependency_files=len(reused),
                supplemental_files=len(supplemental), supplemental_bytes=sum(x["bytes"] for x in supplemental.values()),
                runtime_overrides=overrides, source_simulations=0, production_episodes=0,
                python_version=job["python_version"], python_packages=packages,
                source_qualification=str(qualification))
    if uploaded:
        write_json(output / "request.json", request)
    write_json(output / "supplemental_files.json", supplemental)
    write_json(output / "staging_plan.json", plan)
    (output / "runtime.py").write_text(files["run_inputs.py"])
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--operation", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--payload-upload", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.repo.resolve(), args.operation.resolve(), args.name, args.payload_upload), indent=2))


if __name__ == "__main__":
    main()
