#!/usr/bin/env python3
"""Prepare and run a Merlin CPU job that consolidates verified input chunks.

The output is the original archive, unchanged, plus a single-chunk upload
manifest accepted by prepare_v1_merlin_block. This script runs no simulator.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zlib

READ_BYTES = 8 * 1024 * 1024
PROGRESS_CHUNKS = 16
PROGRESS_MEMBERS = 256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def binding(data: bytes) -> dict:
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def valid_binding(value: dict, *, nonempty: bool = False) -> None:
    require(isinstance(value, dict) and type(value.get("bytes")) is int
            and value["bytes"] >= int(nonempty)
            and re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256"))) is not None,
            "INPUT_FILE_BINDING_INVALID")


def validate_upload(upload: dict) -> None:
    valid_binding(upload, nonempty=True)
    require(upload.get("complete") is True and isinstance(upload.get("chunks"), list)
            and upload["chunks"], "COMPLETE_INPUT_UPLOAD_REQUIRED")
    for index, row in enumerate(upload["chunks"]):
        valid_binding(row, nonempty=True)
        require(type(row.get("index")) is int and row["index"] == index
                and isinstance(row.get("hdfs_path"), str)
                and row["hdfs_path"].startswith("hdfs://")
                and not any(char.isspace() for char in row["hdfs_path"]),
                "INPUT_CHUNK_ORDER_OR_PATH_INVALID")
    require(sum(row["bytes"] for row in upload["chunks"]) == upload["bytes"],
            "INPUT_CHUNK_COVERAGE_DIFFERS")


def member_bindings(files: dict) -> dict:
    require(isinstance(files, dict) and files, "DECLARED_INPUT_FILES_REQUIRED")
    expected = {}
    for name, value in files.items():
        valid_binding(value)
        require(isinstance(name, str) and name and ".." not in PurePosixPath(name).parts,
                "DECLARED_INPUT_PATH_INVALID")
        key = str(PurePosixPath(name)).lstrip("/")
        require(key not in ("", ".") and key not in expected, "DECLARED_INPUT_PATH_COLLISION")
        expected[key] = value
    return expected


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def hdfs(*arguments: str) -> None:
    subprocess.run(["hdfs", "dfs", *arguments], check=True)


def prepare(args) -> dict:
    require(re.fullmatch(r"mrln_[A-Za-z0-9_-]+", args.name) is not None and len(args.name) <= 90,
            "MRLN_JOB_NAME_REQUIRED")
    raw = {"source_upload.json": args.upload_manifest.read_bytes(),
           "input_files.json": args.input_files.read_bytes(),
           "source_job.json": args.source_job.read_bytes()}
    upload = json.loads(raw["source_upload.json"])
    files = json.loads(raw["input_files.json"])
    require(isinstance(json.loads(raw["source_job.json"]), dict), "SOURCE_JOB_OBJECT_REQUIRED")
    validate_upload(upload)
    member_bindings(files)
    template_raw = args.template.read_bytes()
    template = json.loads(template_raw)
    resource = template["resource_config"]
    roles = resource["arnold_resource_config"]["roles"]
    require(resource["resource_backend"] == "arnold" and len(roles) == 1, "ONE_CPU_ROLE_REQUIRED")
    role = roles[0]
    require(role["name"] == "head" and role["num"] == 1 and role["gpu"] == 0
            and role["gpu_type"] == "CPU_ONLY" and type(role["cpu"]) is int
            and 0 < role["cpu"] <= 4 and type(role["memory_mb"]) is int
            and 0 < role["memory_mb"] <= 16384, "CPU_TEMPLATE_EXCEEDS_ALLOWED_SHAPE")
    require(template["job_config"]["job_template_config"]["image_meta"].get("image_url"),
            "EXPLICIT_CPU_IMAGE_REQUIRED")
    provenance = Path(__file__).resolve().parents[1] / "population_dynamics/provenance.py"
    raw["provenance.py"] = provenance.read_bytes()
    raw["prepare_merlin_input_archive.py"] = Path(__file__).read_bytes()
    job = dict(name=args.name, source_paths={
        "source_upload.json": str(args.upload_manifest.resolve()),
        "input_files.json": str(args.input_files.resolve()), "source_job.json": str(args.source_job.resolve())},
        input_bindings={name: binding(data) for name, data in raw.items()},
        cpu=role["cpu"], memory_mb=role["memory_mb"],
        template_path=str(args.template.resolve()), template_sha256=hashlib.sha256(template_raw).hexdigest(),
        source_simulations=0, production_episodes=0)
    raw["merge_job.json"] = (json.dumps(job, sort_keys=True, indent=2) + "\n").encode()
    envelope = {name: dict(binding(data), text=data.decode("utf-8")) for name, data in raw.items()}
    payload = base64.b64encode(zlib.compress(json.dumps(envelope).encode(), level=9)).decode()
    remote = "/tmp/" + args.name
    bootstrap = """#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 - <<'MRLN_ARCHIVE_STAGE'
import base64,hashlib,json,pathlib,zlib
root=pathlib.Path(@ROOT@)
root.mkdir(exist_ok=False)
files=json.loads(zlib.decompress(base64.b64decode(@PAYLOAD@)))
for name,item in files.items():
    if pathlib.PurePosixPath(name).name!=name:raise ValueError('STAGED_INPUT_PATH_INVALID')
    data=item['text'].encode('utf-8')
    if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('STAGED_INPUT_DIFFERS')
    (root/name).write_bytes(data)
MRLN_ARCHIVE_STAGE
exec python3 -u @SCRIPT@ run --job-json @JOB@
""".replace("@ROOT@", repr(remote)).replace(
        "@SCRIPT@", shlex.quote(remote + "/prepare_merlin_input_archive.py")).replace(
        "@JOB@", shlex.quote(remote + "/merge_job.json")).replace("@PAYLOAD@", repr(payload))
    request = deepcopy(template)
    request.pop("request_context", None)
    request.update(name=args.name, description=(
        "CPU-only consolidation of verified input chunks using one HDFS source path per read. "
        "Verify archive bytes and every declared tar member. Preserve source identities; run no science."))
    request["job_config"]["job_template_config"]["entrypoint_full_script"] = bootstrap
    request_bytes = (json.dumps(request, sort_keys=True, indent=2) + "\n").encode()
    require(len(request_bytes) < 1048576, "MERLIN_REQUEST_BODY_TOO_LARGE")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.output_dir / (args.name + ".request.json")
    script_path = args.output_dir / (args.name + ".sh")
    prepared_path = args.output_dir / (args.name + ".prepared.json")
    require(not any(path.exists() for path in (request_path, script_path, prepared_path)),
            "FRESH_PREPARATION_OUTPUTS_REQUIRED")
    request_path.write_bytes(request_bytes)
    script_path.write_text(bootstrap, encoding="utf-8")
    result = dict(status="PREPARED_NOT_SUBMITTED", name=args.name, request=str(request_path),
                  request_bytes=len(request_bytes), request_sha256=hashlib.sha256(request_bytes).hexdigest(),
                  script=str(script_path), source_chunks=len(upload["chunks"]), archive_bytes=upload["bytes"],
                  archive_sha256=upload["sha256"], declared_files=len(files),
                  declared_file_bytes=sum(row["bytes"] for row in files.values()),
                  source_paths=job["source_paths"], input_bindings=job["input_bindings"],
                  resource_config=resource, output_manifest_relative="input_archive/single_chunk_upload.json",
                  source_simulations=0, production_episodes=0)
    write_json(prepared_path, result)
    return result


def run(args) -> dict:
    root = args.job_json.resolve().parent
    job = json.loads(args.job_json.read_text())
    for name, expected in job["input_bindings"].items():
        require(PurePosixPath(name).name == name and binding((root / name).read_bytes()) == expected,
                "MERGE_JOB_INPUT_DIFFERS:" + name)
    specification = importlib.util.spec_from_file_location("merlin_archive_provenance", root / "provenance.py")
    provenance = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(provenance)
    execution = provenance.cpu_execution_provenance()
    require(execution["backend"] == "merlin" and execution["allocated_cpus"] == job["cpu"]
            and execution["allocated_gpus"] == 0, "ACTUAL_MERLIN_CPU_ALLOCATION_REQUIRED")
    checkpoint = os.environ.get("RH2_CHECKPOINT_HDFS", "")
    require(checkpoint.startswith("hdfs://") and not any(char.isspace() for char in checkpoint),
            "ACTUAL_MERLIN_CHECKPOINT_REQUIRED")
    upload = json.loads((root / "source_upload.json").read_text())
    validate_upload(upload)
    expected = member_bindings(json.loads((root / "input_files.json").read_text()))
    require(shutil.disk_usage(root).free > upload["bytes"] + max(row["bytes"] for row in upload["chunks"]),
            "INSUFFICIENT_LOCAL_ARCHIVE_SCRATCH")
    target = checkpoint.rstrip("/") + "/input_archive"
    started = time.monotonic()
    state = dict(status="PREPARING_INPUT_ARCHIVE", execution_context=execution, name=job["name"],
                 source_paths=job["source_paths"], source_input_bindings=job["input_bindings"],
                 source_chunks=len(upload["chunks"]), verified_chunks=0, verified_archive_bytes=0,
                 expected_archive_bytes=upload["bytes"], expected_archive_sha256=upload["sha256"],
                 declared_files=len(expected), verified_files=0, verified_file_bytes=0,
                 source_simulations=0, production_episodes=0, checkpoint_path=target)

    def progress(status: str) -> None:
        state.update(status=status, elapsed_seconds=time.monotonic() - started,
                     recorded_utc=datetime.now(timezone.utc).isoformat())
        write_json(root / "progress.json", state)
        print(json.dumps(state, sort_keys=True), flush=True)
        hdfs("-put", "-f", str(root / "progress.json"), target + "/progress.json")

    hdfs("-mkdir", "-p", target)
    try:
        for name in ("source_upload.json", "input_files.json", "source_job.json", "merge_job.json"):
            hdfs("-put", str(root / name), target + "/" + name)
        progress("MERGING_INPUT_CHUNKS")
        with tempfile.TemporaryDirectory(prefix="verified-archive-", dir=root) as temporary:
            work = Path(temporary)
            archive_path, part = work / "inputs.tar.gz", work / "input.part"
            archive_hash = hashlib.sha256()
            with archive_path.open("xb") as archive:
                for row in upload["chunks"]:
                    # Exactly one HDFS source path per process; never pass the full list.
                    hdfs("-get", row["hdfs_path"], str(part))
                    digest, size = hashlib.sha256(), 0
                    with part.open("rb") as source:
                        while data := source.read(READ_BYTES):
                            digest.update(data)
                            archive_hash.update(data)
                            archive.write(data)
                            size += len(data)
                    require(size == row["bytes"] and digest.hexdigest() == row["sha256"],
                            "INPUT_CHUNK_DIFFERS:" + str(row["index"]))
                    part.unlink()
                    state["verified_chunks"] += 1
                    state["verified_archive_bytes"] += size
                    if state["verified_chunks"] % PROGRESS_CHUNKS == 0:
                        progress("MERGING_INPUT_CHUNKS")
            require(state["verified_archive_bytes"] == upload["bytes"]
                    and archive_hash.hexdigest() == upload["sha256"], "COMPLETE_ARCHIVE_DIFFERS")
            progress("VERIFYING_ARCHIVE_MEMBERS")
            seen = set()
            with tarfile.open(archive_path, mode="r|gz") as archive:
                for member in archive:
                    relative = PurePosixPath(member.name)
                    key = str(relative)
                    require(member.isfile() and not relative.is_absolute() and ".." not in relative.parts
                            and key in expected and key not in seen, "ARCHIVE_MEMBER_INVALID:" + member.name)
                    require(member.size == expected[key]["bytes"], "ARCHIVE_MEMBER_SIZE_DIFFERS:" + key)
                    digest, size = hashlib.sha256(), 0
                    with archive.extractfile(member) as source:
                        while data := source.read(READ_BYTES):
                            digest.update(data)
                            size += len(data)
                    require(size == expected[key]["bytes"] and digest.hexdigest() == expected[key]["sha256"],
                            "ARCHIVE_MEMBER_HASH_DIFFERS:" + key)
                    seen.add(key)
                    state["verified_files"] += 1
                    state["verified_file_bytes"] += size
                    if state["verified_files"] % PROGRESS_MEMBERS == 0:
                        progress("VERIFYING_ARCHIVE_MEMBERS")
            require(seen == set(expected), "ARCHIVE_DECLARED_FILES_MISSING")
            progress("UPLOADING_VERIFIED_ARCHIVE")
            archive_uri = target + "/inputs.tar.gz"
            hdfs("-put", str(archive_path), archive_uri)
            stored_size = subprocess.check_output(["hdfs", "dfs", "-stat", "%b", archive_uri], text=True).strip()
            require(stored_size.isdecimal() and int(stored_size) == upload["bytes"], "PUBLISHED_ARCHIVE_SIZE_DIFFERS")
            result = dict(upload, complete=True, chunk_bytes=upload["bytes"], chunks=[
                dict(index=0, bytes=upload["bytes"], sha256=upload["sha256"], hdfs_path=archive_uri)],
                status="VERIFIED_SINGLE_CHUNK_INPUT_ARCHIVE", execution_context=execution,
                original_chunk_count=len(upload["chunks"]), source_manifests={name: dict(
                    job["input_bindings"][name], original_path=job["source_paths"][name],
                    hdfs_path=target + "/" + name)
                    for name in ("source_upload.json", "input_files.json", "source_job.json")},
                verified_files=state["verified_files"], verified_file_bytes=state["verified_file_bytes"],
                source_simulations=0, production_episodes=0, exit_code=0,
                recorded_utc=datetime.now(timezone.utc).isoformat())
            write_json(root / "single_chunk_upload.json", result)
            progress("ARCHIVE_UPLOADED_MANIFEST_PENDING")
            # Readiness is published last. Consumers still verify the pinned archive hash.
            hdfs("-put", str(root / "single_chunk_upload.json"), target + "/single_chunk_upload.json")
            return result
    except BaseException as error:
        state.update(error_type=type(error).__name__, error=str(error))
        try:
            progress("FAILED_INPUT_ARCHIVE_PREPARATION")
        except Exception as publication_error:
            print("FAILED_TO_PERSIST_FAILURE_PROGRESS: " + str(publication_error), file=sys.stderr, flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    preparation = commands.add_parser("prepare")
    for name in ("upload-manifest", "input-files", "source-job", "template", "output-dir"):
        preparation.add_argument("--" + name, type=Path, required=True)
    preparation.add_argument("--name", required=True)
    execution = commands.add_parser("run")
    execution.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args) if args.action == "prepare" else run(args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
