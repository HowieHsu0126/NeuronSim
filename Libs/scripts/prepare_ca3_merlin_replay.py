#!/usr/bin/env python3
"""Prepare a CA3 Merlin request from complete staged inputs; never submit it.

The caller supplies a currently available resource/image template. Preserve its
resource shape, require at least the original 160 GiB request, and use the saved
sm89 runtime or an explicitly tested sm80 candidate composition. The available
memory admission floor remains 128 GiB in both modes.
"""
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

from ca3_merlin_remote import (persistence_batch_fingerprint, validate_persistence_batch,
                               validate_completion75_recovery)

BOOTSTRAP = """import base64,json,os,pathlib,re,subprocess,sys,zlib
payload=json.loads(zlib.decompress(base64.b64decode(PAYLOAD_LITERAL)))
run=os.environ.get('RH2_JOB_RUN_ID')
trial=os.environ.get('ARNOLD_TRIAL_ID')
if not run or not trial or not re.fullmatch(r'[A-Za-z0-9_.-]+',trial):
    raise ValueError('ACTUAL_MERLIN_RUN_AND_TRIAL_REQUIRED')
work=pathlib.Path('/tmp')/('neuronsim-ca3-'+trial)
work.mkdir(exist_ok=False)
for name,text in payload['files'].items():
    (work/name).write_text(text)
(work/'job.json').write_text(json.dumps(payload['job'],indent=2)+'\\n')
raise SystemExit(subprocess.call([sys.executable,str(work/'ca3_merlin_remote.py'),
                                 '--job-json',str(work/'job.json')]))
"""


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("JSON object required: " + str(path))
    return value


def validate_upload(upload: dict) -> None:
    if (upload.get("complete") is not True or type(upload.get("bytes")) is not int
            or upload["bytes"] <= 0 or type(upload.get("chunk_bytes")) is not int
            or upload["chunk_bytes"] <= 0
            or not isinstance(upload.get("chunks"), list) or not upload["chunks"]
            or re.fullmatch(r"[0-9a-f]{64}", str(upload.get("sha256"))) is None):
        raise ValueError("A complete, hashed CA3 upload manifest is required")
    total = 0
    count = len(upload["chunks"])
    for index, row in enumerate(upload["chunks"]):
        if (row.get("index") != index or type(row.get("bytes")) is not int
                or not 0 < row["bytes"] <= upload["chunk_bytes"]
                or (index < count - 1 and row["bytes"] != upload["chunk_bytes"])
                or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None
                or not isinstance(row.get("hdfs_path"), str) or not row["hdfs_path"].startswith("hdfs://")):
            raise ValueError("CA3 upload chunk identity/order is invalid")
        total += row["bytes"]
    if total != upload["bytes"]:
        raise ValueError("CA3 uploaded bytes do not match the complete archive")


def validate_inputs(identity: dict, files: dict) -> None:
    expected = dict(runtime_duration_ms=1210, reference_duration_ms=1001, units=168106,
                    prefix_tolerance=0, required_memory_gib=128, requested_memory_gib=160,
                    production_episodes=0)
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("CA3 execution identity differs from the frozen replay")
    for name, row in files.items():
        path = Path(name)
        if (not path.is_absolute() or ".." in path.parts or type(row.get("bytes")) is not int
                or row["bytes"] < 0 or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None):
            raise ValueError("CA3 input file identity is invalid")
    bindings = dict(identity["source_file_bindings"])
    runtime = Path(identity["prepared"]) / "ca3_runtime"
    bindings[str(runtime / "build_report.json")] = identity["runtime_build_sha256"]
    bindings[str(runtime / "bin/ca3_benchmark")] = identity["runtime_binary_sha256"]
    bindings.update({str(runtime / "lib" / name): value for name, value in identity["runtime_libraries"].items()})
    if any(name not in files or files[name]["sha256"] != value for name, value in bindings.items()):
        raise ValueError("CA3 input manifest differs from the frozen source/runtime bindings")
    required = [identity["config"], identity["prior_preparation"], identity["current_replay_engine"],
                str(Path(identity["current_replay_engine"]).with_name("host_current_replay_build.json"))]
    required.extend(str(Path(identity["reference"]) / name) for name in
                    ("report.json", "native_observation.bin", "dataset/dataset.json",
                     "ordered_cue_commands.npz", "native_run/trainPat.csv"))
    if any(name not in files for name in required):
        raise ValueError("CA3 runtime/prefix evidence is absent from the input manifest")


def resource_shape(request: dict) -> dict:
    configuration = request["resource_config"]
    if configuration.get("resource_backend") != "arnold":
        raise ValueError("An explicit Arnold resource template is required")
    roles = configuration["arnold_resource_config"]["roles"]
    if len(roles) != 1:
        raise ValueError("The CA3 replay requires one explicit worker role")
    role = roles[0]
    if (role.get("num") != 1 or type(role.get("gpu")) is not int or role["gpu"] < 1
            or type(role.get("cpu")) is not int or role["cpu"] < 1
            or type(role.get("memory_mb")) is not int or role["memory_mb"] < 160 * 1024):
        raise ValueError("The supplied resource shape must preserve the >=160 GiB request")
    image = request["job_config"]["job_template_config"]["image_meta"]
    if image.get("need_build") is not False or not any(image.get(key) for key in ("image_url", "image_vid")):
        raise ValueError("An existing pinned image identity is required")
    role.setdefault("advanced", {}).update(
        is_complete_trial_on_task_no_restart=True,
        is_fail_trial_on_pod_failure=True, pod_no_restart_state="any")
    return deepcopy(role)


def candidate_consumer(frozen: str, current: str) -> str:
    """Add only the composition interface to the frozen scientific consumer."""
    tree = ast.parse(current)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "validate_runtime_composition")
    validator = ast.get_source_segment(current, function)

    def replace(old: str, new: str) -> None:
        nonlocal frozen
        if frozen.count(old) != 1:
            raise ValueError("CA3_FROZEN_CONSUMER_PATCH_CONTEXT_DIFFERS:" + old[:80])
        frozen = frozen.replace(old, new)

    replace("def run(packaged:", validator + "\n\n\ndef run(packaged:")
    replace("current_replay_engine: Path | None = None, prefix_reference: Path | None = None) -> dict:",
            "current_replay_engine: Path | None = None, prefix_reference: Path | None = None,\n"
            "        runtime_composition: Path | None = None) -> dict:")
    replace('    for name, digest in built["backend_libraries"].items():\n'
            '        if sha256_file(packaged / "lib" / name) != digest:\n'
            '            raise ValueError("CA3_BENCHMARK_RUNTIME_LIBRARY_CHANGED:" + name)',
            '    effective_libraries, composition = validate_runtime_composition(packaged, built, runtime_composition)')
    replace("    started = time.perf_counter()\n    try:",
            "    if composition is not None:\n"
            "        report.update(runtime_composition=composition, effective_runtime_libraries=effective_libraries)\n"
            "    started = time.perf_counter()\n    try:")
    replace('    runner.add_argument("--prefix-reference", type=Path)',
            '    runner.add_argument("--prefix-reference", type=Path)\n'
            '    runner.add_argument("--runtime-composition", type=Path)')
    replace("current_replay_engine=args.current_replay_engine, prefix_reference=args.prefix_reference)",
            "current_replay_engine=args.current_replay_engine, prefix_reference=args.prefix_reference,\n"
            "            runtime_composition=args.runtime_composition)")
    ast.parse(frozen)
    return frozen


def validate_candidate(candidate: dict, original_files: dict) -> None:
    validate_upload(candidate["upload"])
    files = candidate["input_files"]
    if set(files) & set(original_files):
        raise ValueError("CA3_CANDIDATE_MUST_NOT_REPLACE_FROZEN_INPUTS")
    for role in ("library", "build_report", "receipt", "consumer"):
        binding = candidate[role]
        path = Path(binding["path"])
        if (not path.is_absolute() or ".." in path.parts
                or type(binding.get("bytes")) is not int or binding["bytes"] <= 0
                or re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256"))) is None
                or files.get(str(path)) != {key: binding[key] for key in ("bytes", "sha256")}):
            raise ValueError("CA3_CANDIDATE_INPUT_BINDING_DIFFERS:" + role)
    if set(files) != {candidate[role]["path"] for role in ("library", "build_report", "receipt", "consumer")}:
        raise ValueError("CA3_CANDIDATE_INPUT_SCOPE_DIFFERS")


def validate_candidate_preflight(candidate: dict, proof: dict, identity: dict,
                                 image: dict, remote: str, persistence_batch: dict | None = None,
                                 completion75_recovery: dict | None = None) -> None:
    if persistence_batch is not None and completion75_recovery is not None:
        raise ValueError("CA3_CONTINUATION_MODES_ARE_EXCLUSIVE")
    expected_status = ("PASS_CA3_COMPLETION75_RECOVERY_PREFLIGHT" if completion75_recovery is not None
                       else "PASS_CA3_PERSISTENCE_CONDITION_PREFLIGHT" if persistence_batch is not None
                       else "PASS_CA3_FROZEN_CANDIDATE_PREFLIGHT")
    if (proof.get("status") != expected_status
            or proof.get("exit_code") != 0 or proof.get("persisted") is not True
            or proof.get("candidate_input_files") != candidate["input_files"]
            or proof.get("source_file_bindings") != identity["source_file_bindings"]
            or proof.get("runtime_build_sha256") != identity["runtime_build_sha256"]
            or proof.get("remote_sha256") != hashlib.sha256(remote.encode()).hexdigest()
            or proof.get("image_meta") != image
            or (persistence_batch is not None and proof.get("persistence_batch_sha256")
                != persistence_batch_fingerprint(persistence_batch))
            or (completion75_recovery is not None and proof.get("completion75_recovery_sha256")
                != persistence_batch_fingerprint(completion75_recovery))):
        raise ValueError("CA3_MATCHING_CPU_CANDIDATE_PREFLIGHT_REQUIRED")


def publish(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise ValueError("Prepared output already exists with different content: " + str(path))
        return
    with path.open("x") as stream:
        stream.write(text)


def prepare(args: argparse.Namespace) -> dict:
    identity = read_json(args.execution_inputs)
    files = read_json(args.input_files)
    upload = read_json(args.upload_manifest)
    validate_inputs(identity, files)
    validate_upload(upload)
    request = deepcopy(read_json(args.resource_template))
    shape = resource_shape(request)
    if not re.fullmatch(r"mrln_[A-Za-z0-9_-]{1,85}", args.name):
        raise ValueError("The Merlin job name must start with mrln_ and fit 90 characters")
    remote_path = Path(__file__).with_name("ca3_merlin_remote.py")
    remote = remote_path.read_text()
    packer = args.pack_results.read_text()
    ast.parse(remote, filename=str(remote_path))
    ast.parse(packer, filename=str(args.pack_results))
    payload = dict(
        job=dict(execution_inputs=identity, input_files=files, upload=upload,
                 resource_shape=shape, expected_compute_capability=[8, 9],
                 archive_identity_source=str(args.upload_manifest.resolve())),
        files={"ca3_merlin_remote.py": remote, "pack_results.py": packer})
    batch_path = getattr(args, "persistence_batch", None)
    batch = read_json(batch_path) if batch_path is not None else None
    recovery_path = getattr(args, "completion75_recovery", None)
    recovery = read_json(recovery_path) if recovery_path is not None else None
    if batch is not None and recovery is not None:
        raise ValueError("CA3_CONTINUATION_MODES_ARE_EXCLUSIVE")
    candidate_path = getattr(args, "candidate_runtime", None)
    if (batch is not None or recovery is not None) and candidate_path is None:
        raise ValueError("CA3_PERSISTENCE_REQUIRES_PREFIX_CANDIDATE_RUNTIME")
    if candidate_path is not None:
        candidate = read_json(candidate_path)
        validate_candidate(candidate, files)
        if getattr(args, "candidate_preflight", None) is None:
            raise ValueError("CA3_MATCHING_CPU_CANDIDATE_PREFLIGHT_REQUIRED")
        proof = read_json(args.candidate_preflight)
        validate_candidate_preflight(candidate, proof, identity,
                                     request["job_config"]["job_template_config"]["image_meta"], remote,
                                      persistence_batch=batch, completion75_recovery=recovery)
        payload["job"].update(candidate_runtime=candidate, candidate_preflight=proof,
                              expected_compute_capability=[8, 0])
    if batch is not None:
        payload["job"].update(execution_mode="persistence_conditions", persistence_batch=batch)
        validate_persistence_batch(payload["job"])
    if recovery is not None:
        payload["job"].update(execution_mode="completion75_recovery", condition_index=4,
                              completion75_recovery=recovery)
        validate_completion75_recovery(payload["job"])
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode("ascii")
    bootstrap = BOOTSTRAP.replace("PAYLOAD_LITERAL", repr(encoded))
    ast.parse(bootstrap, filename="ca3_merlin_bootstrap.py")
    script = ("#!/bin/bash\nset -euo pipefail\n"
              "export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
              "python3 -m pip install numpy==2.4.2 pandas==2.3.3 scipy==1.16.2 h5py==3.14.0 PyYAML==6.0.3 psutil==7.1.0\n"
              "python3 - <<'MRLN_CA3_BOOTSTRAP'\n" + bootstrap + "MRLN_CA3_BOOTSTRAP\n")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    template = request["job_config"]["job_template_config"]
    template["entrypoint_full_script"] = script
    template.setdefault("env_map", {}).update(
        PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    request["name"] = args.name
    request["description"] = (
        "mrln_ CA3 frozen completion_50 replay: 168106 units; 1210 ms execution; "
        "1001 ms reference prefix with exact counts/current bits; saved sm89 runtime; "
        "160 GiB original request and 128 GiB available-memory floor. "
        "One source GPU; extra flavor GPUs are declared. Development evidence only; production remains closed.")
    if candidate_path is not None:
        request["description"] = request["description"].replace(
            "saved sm89 runtime", "explicit unqualified sm80 candidate with frozen numerical software")
    if batch is not None:
        request["description"] = (
            "mrln_ CA3 frozen persistence-condition batch after a bound real Merlin prefix. "
            "168106 units; 1210 ms per condition; full 203408260 count/current checks; "
            "independent condition results and failures; >=160 GiB request and 128 GiB memory floor. "
            "Development evidence only; production remains closed. Conditions: "
            + ",".join(str(row["condition_index"]) for row in batch["conditions"]))
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
        input_archive_sha256=upload["sha256"], input_archive_bytes=upload["bytes"],
        input_files=len(files), resource_shape=shape, source_execution_gpus=1,
        extra_allocated_gpus=shape["gpu"] - 1,
        expected_compute_capability=payload["job"]["expected_compute_capability"],
        original_requested_memory_gib=160, required_available_memory_gib=128,
        runtime_duration_ms=1210, reference_duration_ms=1001, prefix_tolerance=0,
        production_episodes=0, requires_current_resource_precheck=True,
        image_dependencies="Existing image must provide torch, numpy, pandas, scipy, h5py, PyYAML, psutil, hdfs, readelf, vmstat, nvidia-smi",
        checkpoint_limit="Frozen runtime has no full-state restart; periodic HDFS progress is not a scientific checkpoint",
        validation=dict(remote_python_ast=True, packer_python_ast=True, bootstrap_python_ast=True, shell_syntax=True))
    if batch is not None:
        summary.update(execution_mode="persistence_conditions",
                       condition_indices=[row["condition_index"] for row in batch["conditions"]],
                       source_jobs=[row["source_job_id"] for row in batch["conditions"]],
                       persistence_batch_sha256=persistence_batch_fingerprint(batch),
                       source_prefix_comparison=False,
                       prerequisite= batch["prerequisite"]["origin"])
    if recovery is not None:
        summary.update(execution_mode="completion75_recovery", condition_index=4,
                       completion75_recovery_sha256=persistence_batch_fingerprint(recovery),
                       source_prefix_comparison=False,
                       retry_branch="relevant_cluster_condition_changed",
                       administrative_cancellation_cause_resolved=False,
                       historical_cancelled_jobs=[609703, 609907])
    publish(args.output / (args.name + ".prepared.json"), json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("execution-inputs", "input-files", "upload-manifest", "resource-template",
                 "pack-results", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--candidate-runtime", type=Path)
    parser.add_argument("--candidate-preflight", type=Path)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--persistence-batch", type=Path)
    continuation.add_argument("--completion75-recovery", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
