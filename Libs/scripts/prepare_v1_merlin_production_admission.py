#!/usr/bin/env python3
"""Package isolated V1 admission software and prepare a CPU request; never submit."""
from __future__ import annotations

import argparse
import ast
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
import tarfile
import zlib

from prepare_v1_merlin_readout import BOOTSTRAP, cpu_shape
from v1_merlin_readout import validate_receipt
from stage_v1_merlin_block_inputs import module_files

PUBLIC_ROOT = Path("/tmp/neuronsim-v1-admission-public")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def identity(path):
    raw = Path(path).read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def package(args):
    if socket.gethostname().split(".")[0] == "mgmt2":
        raise ValueError("PACKAGE_ON_RECEIVE_HOST")
    plan = read(args.record / "cpu_admission_input_plan.json")
    files = module_files(args.repo / "Libs/population_dynamics", ["benchmark_production_block"])
    expected = plan["current_public_modules"]
    if {str(p) for p in files} != set(expected):
        raise ValueError("PUBLIC_CLOSURE_CHANGED")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with args.archive.open("xb") as target, tarfile.open(fileobj=target, mode="w:gz", compresslevel=1) as archive:
        for source in sorted(files):
            raw = source.read_bytes()
            found = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            if found != expected[str(source)]:
                raise ValueError("TESTED_PUBLIC_MODULE_CHANGED:" + str(source))
            destination = PUBLIC_ROOT / "population_dynamics" / source.name
            member = tarfile.TarInfo(str(destination).lstrip("/"))
            member.size = len(raw)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(raw))
            manifest[str(destination)] = found
    write(args.record / "cpu_software_files.json", manifest)
    result = dict(status="PACKAGED_NOT_UPLOADED", archive=str(args.archive),
                  archive_identity=identity(args.archive), files=len(manifest),
                  file_bytes=sum(row["bytes"] for row in manifest.values()),
                  public_root=str(PUBLIC_ROOT), original_public_paths_unchanged=True)
    write(args.record / "cpu_software_preparation.json", result)
    print(json.dumps(result))


def request(args):
    op, record = args.operation, args.record
    plan = read(record / "cpu_admission_input_plan.json")
    software_record = args.software_record or record
    software = read(software_record / "cpu_software_files.json")
    if args.base_input_layer:
        layers = [read(path) for path in args.base_input_layer]
    else:
        layers = [
            dict(upload=read(op / "v1_single_chunk_upload.json"), input_files=read(op / "input_files.json")),
            dict(upload=read(record / "upload.json"), input_files=read(record / "input_files.json"))]
    data_layer = dict(upload=read(args.data_upload),
                      input_files=read(record / "cpu_admission_data_files.json"))
    software_layer = dict(upload=read(args.software_upload), input_files=software)
    layers.extend((data_layer, software_layer))
    if args.extra_input_layer is not None:
        layers.insert(-1, read(args.extra_input_layer))
    seen = set()
    for layer in layers:
        upload = layer["upload"]
        if (upload.get("complete") is not True or sum(row["bytes"] for row in upload["chunks"]) != upload["bytes"]
                or seen & set(layer["input_files"])):
            raise ValueError("COMPLETE_DISJOINT_INPUT_LAYERS_REQUIRED")
        seen.update(layer["input_files"])
    for layer, metadata in ((data_layer, record / "cpu_admission_data_preparation.json"),
                            (software_layer, software_record / "cpu_software_preparation.json")):
        if {k: layer["upload"][k] for k in ("bytes", "sha256")} != read(metadata)["archive_identity"]:
            raise ValueError("PREPARED_UPLOAD_IDENTITY_CHANGED:" + str(metadata))
    source_job = read(record / "source_job.json")
    block = Path(source_job["cases"]).parent
    qualification = args.qualification_root or Path(plan.get("qualification_root", block.parent.parent))
    required = {str(qualification / name) for name in
                ("frozen_campaign.yaml", "qualification.json", "identities.csv")}
    required.update((str(block / "sources.json"), source_job["cases"],
                     plan["runtime_qualification"]["path"], plan["native_reference_validation"]))
    if missing := required - seen:
        raise ValueError("ADMISSION_INPUTS_NOT_STAGED:" + ",".join(sorted(missing)))
    # Saved GPU inputs omit some rate witnesses that the CPU admission consumes.
    source_directories = {row["sources"][key]
                          for row in read(block / "sources.json")["conditions"]
                          for key in ("rate_source", "spike_source", "authority_source", "circuit_reference")}
    for directory in sorted(source_directories):
        prefix = directory.rstrip("/") + "/"
        if not any(path.startswith(prefix) for path in seen):
            raise ValueError("ADMISSION_SOURCE_DIRECTORY_NOT_STAGED:" + directory)
    artifact = read(plan["runtime_qualification"]["path"])
    if identity(plan["runtime_qualification"]["path"]) != {k: plan["runtime_qualification"][k] for k in ("bytes", "sha256")}:
        raise ValueError("VERIFIED_RUNTIME_ARTIFACT_CHANGED")
    primary, replay = read(record / "primary_receipt.json"), read(record / "replay_receipt.json")
    for value in (primary, replay):
        validate_receipt(value, artifact["backend"]["library_sha256"])
    if primary["merlin_job_run_id"] == replay["merlin_job_run_id"] or primary["merlin_trial_id"] == replay["merlin_trial_id"]:
        raise ValueError("DISTINCT_SOURCE_EXECUTIONS_REQUIRED")
    template = deepcopy(read(args.template))
    shape = cpu_shape(template)
    cases = read(source_job["cases"])
    identities = [row["identity"] for row in cases["cases"]]
    if len(identities) != 10 or [r["condition_index"] for r in identities] != list(range(10)):
        raise ValueError("COMPLETE_ORDERED_BLOCK_REQUIRED")
    job = dict(input_layers=layers, software_files=software, public_root=str(PUBLIC_ROOT),
               primary=primary, replay=replay, expected_library_sha256=artifact["backend"]["library_sha256"],
               config=str(qualification / "frozen_campaign.yaml"),
               qualification=str(qualification / "qualification.json"),
               identity_table=str(qualification / "identities.csv"),
               runtime_qualification=plan["runtime_qualification"]["path"],
               native_validation=plan["native_reference_validation"],
               sources=str(block / "sources.json"), cases=source_job["cases"],
               circuit_index=cases["circuit_index"], block_index=cases["block_index"],
               identities=identities, block_root=str(block), output_root=str(block / "merlin_admission"),
               resource_shape=shape)
    scripts = {name: (args.repo / "Libs/scripts" / name).read_text()
               for name in ("v1_merlin_production_admission.py", "v1_merlin_readout.py", "ca3_merlin_remote.py")}
    scripts["pack_results.py"] = args.pack_results.read_text()
    for name, content in scripts.items():
        ast.parse(content, filename=name)
    encoded = base64.b64encode(zlib.compress(json.dumps(dict(job=job, files=scripts)).encode(), 9)).decode()
    bootstrap = BOOTSTRAP.replace("PAYLOAD_LITERAL", repr(encoded)).replace(
        "str(work/'v1_merlin_readout.py')", "str(work/'v1_merlin_production_admission.py')")
    ast.parse(bootstrap)
    old_script = template["job_config"]["job_template_config"]["entrypoint_full_script"]
    prefix = old_script.split("python3 - <<", 1)[0]
    script = prefix + "python3 - <<'MRLN_V1_ADMISSION_BOOTSTRAP'\n" + bootstrap + "MRLN_V1_ADMISSION_BOOTSTRAP\n"
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    if not args.name.startswith("mrln_"):
        raise ValueError("MERLIN_NAME_PREFIX_REQUIRED")
    template["name"] = args.name
    template["description"] = "CPU admission and public reuse of one saved ten-condition V1 block; no simulator execution."
    template.pop("request_context", None)
    template["job_config"]["job_template_config"]["entrypoint_full_script"] = script
    with (record / (args.name + ".sh")).open("x") as stream:
        stream.write(script)
    write(record / (args.name + ".request.json"), template)
    write(record / (args.name + ".job.json"), job)
    print(json.dumps(dict(status="PREPARED_NOT_SUBMITTED", name=args.name, files=len(seen))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("package", "request"):
        command = actions.add_parser(name)
        command.add_argument("--repo", type=Path, required=True)
        command.add_argument("--record", type=Path, required=True)
    actions.choices["package"].add_argument("--archive", type=Path, required=True)
    command = actions.choices["request"]
    for name in ("operation", "data-upload", "software-upload", "template", "pack-results"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--base-input-layer", type=Path, action="append",
                         help="Prepared upload/input_files layer; repeat to replace the legacy two bases")
    command.add_argument("--qualification-root", type=Path,
                         help="Explicit frozen campaign root, including for gpu_admission block layouts")
    command.add_argument("--extra-input-layer", type=Path)
    command.add_argument("--software-record", type=Path)
    command.add_argument("--name", required=True)
    args = parser.parse_args()
    (package if args.action == "package" else request)(args)


if __name__ == "__main__":
    main()
