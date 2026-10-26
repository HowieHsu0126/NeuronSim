#!/usr/bin/env python3
"""Prepare a CPU-only V1 replay/readout request from real completed source receipts."""
from __future__ import annotations

import argparse
import ast
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zlib

from prepare_ca3_merlin_replay import publish, read_json, validate_upload
from v1_merlin_readout import RUNTIME_TEST_RUNNER, native_reference_match, platform_identity, validate_receipt

BOOTSTRAP = """import base64,json,os,pathlib,re,subprocess,sys,zlib
payload=json.loads(zlib.decompress(base64.b64decode(PAYLOAD_LITERAL)))
run=os.environ.get('RH2_JOB_RUN_ID')
trial=os.environ.get('ARNOLD_TRIAL_ID')
if not run or not trial or not re.fullmatch(r'[A-Za-z0-9_.-]+',trial):
    raise ValueError('ACTUAL_MERLIN_CPU_RUN_AND_TRIAL_REQUIRED')
work=pathlib.Path('/tmp')/('neuronsim-v1-readout-'+trial)
work.mkdir(exist_ok=False)
for name,text in payload['files'].items():
    (work/name).write_text(text,encoding='utf-8')
(work/'job.json').write_text(json.dumps(payload['job'],indent=2)+'\\n')
raise SystemExit(subprocess.call([sys.executable,str(work/'v1_merlin_readout.py'),
                                 '--job-json',str(work/'job.json')]))
"""


def cpu_shape(request: dict) -> dict:
    configuration = request["resource_config"]
    if configuration.get("resource_backend") != "arnold":
        raise ValueError("An explicit Arnold CPU_ONLY resource template is required")
    roles = configuration["arnold_resource_config"]["roles"]
    if len(roles) != 1:
        raise ValueError("V1 readout requires one CPU worker role")
    role = roles[0]
    if (role.get("num") != 1 or type(role.get("gpu")) is not int or role["gpu"] != 0
            or role.get("gpu_type") != "CPU_ONLY" or type(role.get("cpu")) is not int or role["cpu"] < 1
            or type(role.get("memory_mb")) is not int or role["memory_mb"] < 16 * 1024):
        raise ValueError("Use a CPU_ONLY template with zero GPUs and at least 16 GiB requested memory")
    image = request["job_config"]["job_template_config"]["image_meta"]
    if image.get("need_build") is not False or not any(image.get(key) for key in ("image_url", "image_vid")):
        raise ValueError("An existing pinned image is required")
    role.setdefault("advanced", {}).update(
        is_complete_trial_on_task_no_restart=True,
        is_fail_trial_on_pod_failure=True, pod_no_restart_state="any")
    return deepcopy(role)


def prepare(args: argparse.Namespace) -> dict:
    if re.fullmatch(r"[0-9a-f]{64}", args.expected_library_sha256) is None:
        raise ValueError("The verified sm80 library SHA-256 is required")
    if not re.fullmatch(r"mrln_[A-Za-z0-9_-]{1,85}", args.name):
        raise ValueError("Use a Merlin name beginning mrln_ with at most 90 characters")
    source_job = read_json(args.source_job)
    sources = read_json(args.sources)
    inputs = read_json(args.input_files)
    upload = read_json(args.upload_manifest)
    validate_upload(upload)
    if (sources.get("scenario_id") != "v1_visual" or sources.get("execution_backend") != "glif_cuda"
            or sources.get("circuit_index") != 0 or sources.get("block_index") != 0
            or [row["condition_index"] for row in sources.get("conditions", [])] != list(range(10))):
        raise ValueError("The frozen C0/B0 ten-condition source map is required")
    if source_job.get("mode") != "source_batch" or source_job["cases"] not in inputs:
        raise ValueError("The frozen source-batch case identity is required")
    authority = {row["sources"]["authority_source"] for row in sources["conditions"]}
    references = {row["sources"]["circuit_reference"] for row in sources["conditions"]}
    if len(authority) != 1 or references != {source_job["reference"]}:
        raise ValueError("The source map contains mixed circuit/coordinate bindings")
    public = Path(source_job["repo"]) / "Libs/population_dynamics"
    required = ("__init__.py", "v1_glif_production.py", "v1_glif_dataset.py", "v1_benchmark.py",
                "v1_source_dataset.py", "v1_lgn_runtime.py", "v1_visual_controls.py",
                "circuit_data.py", "benchmark_production.py", "provenance.py")
    if any(str(public / name) not in inputs for name in required):
        raise ValueError("The archived public reader module closure is incomplete")
    native_match = native_reference_match(source_job, inputs)
    evidence, receipts = {}, {}
    for label in ("primary", "replay"):
        receipt = read_json(getattr(args, label + "_receipt"))
        validate_receipt(receipt, args.expected_library_sha256)
        records = {}
        for kind in ("run", "trial"):
            path = getattr(args, label + "_" + kind + "_response")
            raw = path.read_bytes()
            document = json.loads(raw)
            expected = receipt["merlin_job_run_id" if kind == "run" else "merlin_trial_id"]
            record = platform_identity(document, expected, kind, receipt["merlin_job_run_id"])
            record.update(response_path=str(path.resolve()), response_sha256=hashlib.sha256(raw).hexdigest())
            records[kind] = record
        receipts[label], evidence[label] = receipt, records
    if (receipts["primary"]["merlin_job_run_id"] == receipts["replay"]["merlin_job_run_id"]
            or receipts["primary"]["merlin_trial_id"] == receipts["replay"]["merlin_trial_id"]):
        raise ValueError("Primary and replay must have different actual Merlin run and trial IDs")
    request = deepcopy(read_json(args.resource_template))
    shape = cpu_shape(request)
    directory = Path(__file__).resolve().parent
    scripts = {
        "v1_merlin_readout.py": (directory / "v1_merlin_readout.py").read_text(),
        "ca3_merlin_remote.py": (directory / "ca3_merlin_remote.py").read_text(),
        "test_merlin_runtime_memory.py": (
            directory.parent / "population_dynamics/test_merlin_runtime_memory.py").read_text(),
        "pack_results.py": args.pack_results.read_text()}
    for name, text in scripts.items():
        ast.parse(text, filename=name)
    ast.parse(RUNTIME_TEST_RUNNER, filename="v1_runtime_memory_test_runner.py")
    runtime_tests = dict(
        module="test_merlin_runtime_memory", expected_tests=13,
        files={name: dict(bytes=len(scripts[name].encode("utf-8")),
                         sha256=hashlib.sha256(scripts[name].encode("utf-8")).hexdigest())
               for name in ("ca3_merlin_remote.py", "v1_merlin_readout.py",
                            "test_merlin_runtime_memory.py")})
    payload = dict(job=dict(
        source_job=source_job, sources=sources, input_files=inputs, upload=upload,
        primary=receipts["primary"], replay=receipts["replay"], platform_evidence=evidence,
        source_map_binding=dict(path=str(args.sources.resolve()),
                                sha256=hashlib.sha256(args.sources.read_bytes()).hexdigest()),
        public_modules=str(public), resource_shape=shape, native_reference_match=native_match,
        runtime_tests=runtime_tests, expected_library_sha256=args.expected_library_sha256), files=scripts)
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode("ascii")
    bootstrap = BOOTSTRAP.replace("PAYLOAD_LITERAL", repr(encoded))
    ast.parse(bootstrap, filename="v1_merlin_readout_bootstrap.py")
    script = ("#!/bin/bash\nset -euo pipefail\n"
              "export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "
              "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg\n"
              "python3 -m pip install numpy==2.4.2 pandas==2.3.3 scipy==1.16.2 "
              "h5py==3.14.0 PyYAML==6.0.3 psutil==7.1.0 matplotlib==3.10.7 scikit-image==0.25.2 sympy==1.14.0\n"
              "python3 - <<'MRLN_V1_READOUT_BOOTSTRAP'\n" + bootstrap + "MRLN_V1_READOUT_BOOTSTRAP\n")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    template = request["job_config"]["job_template_config"]
    template["entrypoint_full_script"] = script
    template.setdefault("env_map", {}).update(
        CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    request["name"] = args.name
    request["description"] = (
        "mrln_ CPU-only V1 verification of two distinct completed Merlin sm80 ten-condition runs. "
        "Bound native-reference comparison, frozen replay tolerance, and full public archive readers. No simulation, training seeds, "
        "production admission, or new production episodes.")
    if "request_context" in request:
        request["request_context"].pop("precheck_sid", None)
        if not request["request_context"]:
            del request["request_context"]
    args.output.mkdir(parents=True, exist_ok=True)
    request_path = args.output / (args.name + ".request.json")
    script_path = args.output / (args.name + ".sh")
    publish(script_path, script)
    publish(request_path, json.dumps(request, indent=2) + "\n")
    summary = dict(
        status="PREPARED_NOT_SUBMITTED", request=str(request_path), entrypoint=str(script_path),
        resource_shape=shape, required_available_memory_gib=8, recommended_requested_memory_gib=16,
        source_runs={key: value["merlin_job_run_id"] for key, value in receipts.items()},
        source_trials={key: value["merlin_trial_id"] for key, value in receipts.items()},
        public_module_root=str(public), public_module_bindings={name: inputs[str(public / name)] for name in required},
        expected_library_sha256=args.expected_library_sha256, native_reference_match=native_match,
        shared_transport="ca3_merlin_remote.download_inputs/stage_inputs/persist; no CA3 resource or science calls",
        transport_job_keys=["upload", "input_files"], production_episodes=0, source_simulations=0,
        production_gate="NOT_MODIFIED_OR_GRANTED", current_precheck_required=True,
        runtime_unverified=True,
        runtime_tests=dict(runtime_tests, status="NOT_RUN", scope="isolated_child_process"),
        validation=dict(wrapper_ast=True, transport_ast=True, packer_ast=True, bootstrap_ast=True,
                        runtime_tests_ast=True, runtime_test_runner_ast=True, shell_syntax=True))
    publish(args.output / (args.name + ".prepared.json"), json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-job", "sources", "input-files", "upload-manifest", "resource-template",
                 "pack-results", "output", "primary-receipt", "replay-receipt",
                 "primary-run-response", "primary-trial-response",
                 "replay-run-response", "replay-trial-response"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-library-sha256", required=True)
    parser.add_argument("--name", required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
