#!/usr/bin/env python3
"""Package the pinned CA3 backend and prepare a Merlin CPU build request."""
from __future__ import annotations

import argparse
import ast
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import zlib

from prepare_ca3_merlin_replay import validate_upload
from prepare_v1_merlin_readout import BOOTSTRAP, cpu_shape
from prepare_v1_native_comparison_inputs import pack
from stage_v1_merlin_block_inputs import module_files

FRAMEWORK_REVISION = "c4955998b950b9c561b3373a71b611aedc705be9"
SAMPLES_REVISION = "81992093d2b8c33cab22dbf6852c070c330f1715"
COMPONENTS = {"cuda_nvcc", "cuda_cudart", "cuda_cccl", "libcurand", "cuda_cuobjdump"}
TEMPLATES = ("cmake/carlsim-config.cmake.in", "carlsim/interface/inc/carlsim_conf.h.in")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def identity(path: Path) -> dict:
    value = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            value.update(data)
            size += len(data)
    return dict(bytes=size, sha256=value.hexdigest())


def package(args: argparse.Namespace) -> None:
    if socket.gethostname().split(".")[0] == "mgmt2":
        raise ValueError("CA3_PACKAGE_ON_RECEIVE_HOST")
    repo, record = args.repo.resolve(), args.record.resolve()
    framework = repo / "Libs/external/carlsim_ca3"
    samples = args.samples.resolve()
    report_path = args.build_report.resolve()
    original = read(report_path)
    backend = original["backend_build_report"]
    if (backend["framework_revision"] != FRAMEWORK_REVISION
            or backend["cuda_samples_revision"] != SAMPLES_REVISION
            or "V11.8.89" not in backend["nvcc_version"]
            or not isinstance(original["backend_libraries"], dict)
            or set(original["backend_libraries"]) != {
                "libcarlsim.so.6.0.0", "libcarlsim-stopwatch.so", "libcurand.so.10"}):
        raise ValueError("CA3_ORIGINAL_BACKEND_PROVENANCE_DIFFERS")
    compiler = record / "compiler_locations.json"
    compiler_record = read(compiler)
    if any("11.4.1" not in compiler_record[key].splitlines()[0]
           for key in ("gcc_version", "gxx_version")):
        raise ValueError("CA3_RECORDED_GCC_11_4_1_REQUIRED")
    components = read(record / "downloaded_components.json")
    redistribution_path = record.parent / "cuda_11p8_redistribution.json"
    redistribution = read(redistribution_path)
    if set(components) != COMPONENTS:
        raise ValueError("CA3_FIVE_CUDA_COMPONENTS_REQUIRED")
    paths, expected = {}, {}

    def add(path: Path, declared: dict | None = None) -> None:
        path = path.absolute()
        if path.is_symlink() or not path.is_file():
            raise ValueError("CA3_REGULAR_BUILD_INPUT_REQUIRED:" + str(path))
        paths[str(path)] = path
        if declared is not None:
            expected[str(path)] = {key: declared[key] for key in ("bytes", "sha256")}

    for name, row in components.items():
        official = redistribution[name]["linux-x86_64"]
        if (row["version"] != redistribution[name]["version"]
                or row["sha256"] != official["sha256"]
                or row["bytes"] != int(official["size"])
                or row["relative_path"] != official["relative_path"]):
            raise ValueError("CA3_CUDA_REDISTRIBUTION_IDENTITY_DIFFERS:" + name)
        add(Path(row["path"]), row)
    for source, revision in ((framework, FRAMEWORK_REVISION), (samples, SAMPLES_REVISION)):
        manifest = read(source / "source.json")
        if manifest["revision"] != revision:
            raise ValueError("CA3_PINNED_SOURCE_REVISION_DIFFERS:" + str(source))
        add(source / "source.json")
        for row in manifest["files"]:
            add(source / row["path"], row)
    deferred = {row["path"]: row for row in read(framework / "source.json")["deferred_blobs"]}
    for name in TEMPLATES:
        path, row = framework / name, deferred[name]
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if len(raw) != row["size"] or blob != row["sha"]:
            raise ValueError("CA3_DEFERRED_BUILD_TEMPLATE_DIFFERS:" + name)
        add(path, dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    public = repo / "Libs/population_dynamics"
    helpers = module_files(public, ["ca3_reference", "ca3_native_observation", "ca3_conductance_memory"])
    for path in helpers:
        add(path)
    for name, key in (("ca3_native_observation.py", "observer_implementation_sha256"),
                      ("ca3_conductance_memory.py", "conductance_memory_implementation_sha256")):
        observed = identity(public / name)
        if observed["sha256"] != backend[key]:
            raise ValueError("CA3_ORIGINAL_OVERLAY_IMPLEMENTATION_CHANGED:" + name)
        expected[str(public / name)] = observed
    for path in (report_path, compiler, redistribution_path, record / "downloaded_components.json",
                 record.parent / "candidate_build_provenance.json"):
        add(path)
    runner_files = {name: identity(repo / "Libs/scripts" / name) for name in
                    ("ca3_merlin_backend_build.py", "ca3_merlin_remote.py")}
    for name in runner_files:
        ast.parse((repo / "Libs/scripts" / name).read_text(), filename=name)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = pack(args.archive, paths, expected)
    write(record / "backend_build_inputs.json", manifest)
    result = dict(
        status="PACKAGED_NOT_UPLOADED", archive=str(args.archive.resolve()),
        archive_identity=identity(args.archive), files=len(manifest),
        file_bytes=sum(row["bytes"] for row in manifest.values()),
        prepared_utc=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(),
        build_inputs=dict(repo=str(repo), framework=str(framework), samples=str(samples),
                          components=components, deferred_templates=list(TEMPLATES),
                          original_build_report=str(report_path), compiler_locations=str(compiler)),
        runner_files=runner_files, host_compiler_packaged=False,
        compiler_requirement="Actual /usr/bin/gcc and g++ version lines must match compiler_locations.json",
        target="carlsim", cuda_architecture="sm_80", candidate_qualified=False,
        builds_run=0, simulations_run=0, production_episodes=0)
    write(record / "backend_build_package.json", result)
    print(json.dumps(result), flush=True)


def request(args: argparse.Namespace) -> None:
    record = args.record.resolve()
    prepared = read(record / "backend_build_package.json")
    upload = read(args.upload)
    validate_upload(upload)
    if {key: upload[key] for key in ("bytes", "sha256")} != prepared["archive_identity"]:
        raise ValueError("CA3_PREPARED_BUILD_UPLOAD_DIFFERS")
    if not re.fullmatch(r"mrln_[A-Za-z0-9_-]{1,85}", args.name):
        raise ValueError("CA3_MERLIN_NAME_PREFIX_REQUIRED")
    template = deepcopy(read(args.template))
    shape = cpu_shape(template)
    if shape["cpu"] < 2:
        raise ValueError("CA3_ORIGINAL_TWO_COMPILER_WORKERS_REQUIRED")
    scripts = {}
    candidate = read(args.candidate_compiler) if args.candidate_compiler else None
    completed = read(args.completed_build) if args.completed_build else None
    if candidate is not None and completed is not None:
        raise ValueError("CA3_BUILD_AND_REPORT_RECOVERY_ARE_SEPARATE_MODES")
    runner_files = prepared["runner_files"]
    binding = candidate if candidate is not None else completed
    if binding is not None:
        validate_upload(binding["upload"])
        if (binding["profile"] != "debian_bookworm_gcc11"
                or set(binding["runner_files"]) != set(runner_files)
                or binding["runner_files"]["ca3_merlin_remote.py"] != runner_files["ca3_merlin_remote.py"]):
            raise ValueError("CA3_CANDIDATE_RUNNER_BINDING_DIFFERS")
        runner_files = binding["runner_files"]
    for name, expected in runner_files.items():
        path = args.repo / "Libs/scripts" / name
        if identity(path) != expected:
            raise ValueError("CA3_PREPARED_RUNNER_CHANGED:" + name)
        scripts[name] = path.read_text()
    scripts["pack_results.py"] = args.pack_results.read_text()
    for name, source in scripts.items():
        ast.parse(source, filename=name)
    job = dict(upload=upload, input_files=read(record / "backend_build_inputs.json"),
               build_inputs=prepared["build_inputs"], resource_shape=shape,
               runner_files=runner_files, package_identity=prepared["archive_identity"])
    if candidate is not None:
        job["candidate_compiler"] = candidate
        job["original_runner_files"] = prepared["runner_files"]
    if completed is not None:
        job["completed_build"] = completed
        job["original_runner_files"] = prepared["runner_files"]
    encoded = base64.b64encode(zlib.compress(json.dumps(dict(job=job, files=scripts)).encode(), 9)).decode()
    bootstrap = BOOTSTRAP.replace("PAYLOAD_LITERAL", repr(encoded)).replace(
        "neuronsim-v1-readout-", "neuronsim-ca3-backend-build-").replace(
        "str(work/'v1_merlin_readout.py')", "str(work/'ca3_merlin_backend_build.py')")
    ast.parse(bootstrap)
    entry = template["job_config"]["job_template_config"]
    prefix = entry["entrypoint_full_script"].split("python3 - <<", 1)[0]
    script = prefix + "python3 - <<'MRLN_CA3_BACKEND_BOOTSTRAP'\n" + bootstrap + "MRLN_CA3_BACKEND_BOOTSTRAP\n"
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    template.update(name=args.name, description="CPU-only build of pinned CA3 libcarlsim for sm80; qualification remains closed.")
    template.pop("request_context", None)
    entry["entrypoint_full_script"] = script
    with (record / (args.name + ".sh")).open("x") as stream:
        stream.write(script)
    write(record / (args.name + ".job.json"), job)
    write(record / (args.name + ".request.json"), template)
    print(json.dumps(dict(status="PREPARED_NOT_SUBMITTED", name=args.name,
                         input_files=len(job["input_files"]), resource_shape=shape)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("package", "request"):
        command = actions.add_parser(name)
        command.add_argument("--repo", type=Path, required=True)
        command.add_argument("--record", type=Path, required=True)
    command = actions.choices["package"]
    for name in ("archive", "samples", "build-report"):
        command.add_argument("--" + name, type=Path, required=True)
    command = actions.choices["request"]
    for name in ("upload", "template", "pack-results"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--name", required=True)
    command.add_argument("--candidate-compiler", type=Path,
                         help="Explicit uploaded candidate compiler and current runner binding")
    command.add_argument("--completed-build", type=Path,
                         help="Verified completed library from a preserved report-only failure")
    args = parser.parse_args()
    (package if args.action == "package" else request)(args)


if __name__ == "__main__":
    main()
