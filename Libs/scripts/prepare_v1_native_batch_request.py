#!/usr/bin/env python3
"""Prepare the frozen native-family Merlin request without platform actions."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identity(path: Path) -> dict:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def encoded_json(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def save(path: Path, data: bytes) -> None:
    if path.exists():
        require(path.read_bytes() == data, "EXISTING_PREPARATION_DIFFERS:" + str(path))
        return
    with path.open("xb") as stream:
        stream.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-request", type=Path, required=True)
    parser.add_argument("--compiler-job", type=Path, required=True)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--execution-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    require(args.name.startswith("mrln_") and "/" not in args.name, "MERLIN_NAME_REQUIRED")
    template = read_json(args.template_request)
    original_script = template["job_config"]["job_template_config"]["entrypoint_full_script"]
    opener, closer = "python3 - <<'MRLN_V1_STAGE'\n", "\nMRLN_V1_STAGE\n"
    prefix, body = original_script.split(opener, 1)
    stage, _ = body.split(closer, 1)
    assignment = next(node for node in ast.parse(stage).body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "request" for t in node.targets))
    original_assignment = ast.get_source_segment(stage, assignment)
    payload = json.loads(ast.literal_eval(assignment.value.args[0]))
    compiler_job = read_json(args.compiler_job)
    compiler_report_bytes = base64.b64decode(payload["files"]["compiler_report.json"])
    compiler = json.loads(compiler_report_bytes)
    require(compiler["publication_status"] == "PASS" and compiler["exit_code"] == 0
            and compiler["target_arch"] == "sm_80" and "V12.8.61" in compiler["nvcc_version"]
            and compiler["library_sha256"] == compiler_job["expected_library_sha256"]
            == payload["candidate_library"]["sha256"]
            and compiler["source_sha256"] == compiler_job["expected_cuda_source_sha256"],
            "COMPILER_CANDIDATE_IDENTITY_DIFFERS")
    for name, expected in payload["auxiliary_sha256"].items():
        require(hashlib.sha256(base64.b64decode(payload["files"][name])).hexdigest() == expected,
                "TEMPLATE_AUXILIARY_IDENTITY_DIFFERS:" + name)
    preparation = read_json(args.input_directory / "preparation.json")
    upload = read_json(args.input_directory / "upload.json")
    file_manifest = read_json(args.input_directory / "input_manifest.json")
    job = read_json(args.input_directory / "job.json")
    require(preparation["complete"] and preparation["files"] == 114
            and preparation["declared_file_hashes_verified"] == 113
            and preparation["conditions"] == 10 and len(file_manifest) == 114
            and upload["complete"] and len(upload["chunks"]) == 58
            and [row["index"] for row in upload["chunks"]] == list(range(58))
            and sum(row["bytes"] for row in upload["chunks"]) == upload["bytes"]
            and {"bytes": upload["bytes"], "sha256": upload["sha256"]}
            == preparation["archive"]["identity"], "COMPLETE_UPLOADED_FAMILY_REQUIRED")
    require(identity(args.input_directory / "input_manifest.json") ==
            {key: preparation["file_manifest"][key] for key in ("bytes", "sha256")}
            and identity(args.input_directory / "job.json") ==
            {key: preparation["job"][key] for key in ("bytes", "sha256")},
            "PREPARED_CONTROL_IDENTITY_DIFFERS")
    runner_source = args.runner.read_text()
    constants = {}
    for node in ast.parse(runner_source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                        "CASES_SHA256", "SOURCE_BATCH_SHA256", "RUNTIME_CODE"):
                    constants[target.id] = ast.literal_eval(node.value)
    control = preparation["source_control"]
    require(control["cases"]["sha256"] == constants["CASES_SHA256"]
            and file_manifest[job["cases"]]["sha256"] == constants["CASES_SHA256"]
            and control["source_batch_sha256"] == constants["SOURCE_BATCH_SHA256"]
            and control["cortical_units"] == 230924 and control["connection_rows"] == 82246780
            and control["dataset_streams"]["connectivity_seed"] == 100000000
            and [row["identity"]["condition_index"] for row in control["family"]] == list(range(10))
            and all(row["identity"]["circuit_index"] == 0 and row["identity"]["block_index"] == 2
                    for row in control["family"]), "FROZEN_FAMILY_CONTROL_DIFFERS")
    for name, expected in constants["RUNTIME_CODE"].items():
        path = Path(job["code"]) / name
        require(file_manifest[str(path)]["sha256"] == expected
                and identity(path)["sha256"] == expected, "FROZEN_RUNTIME_CODE_DIFFERS:" + name)
    root = Path("/tmp") / args.name
    old_root = str(Path(compiler_job["library"]).parent)
    job.update(scope="cuda_architecture_qualification", mode="batch_check",
               output=str(args.execution_output), library=str(root / "libv1_glif_cuda_sm80.so"),
               build_report=str(root / "compiler_report.json"),
               expected_library_sha256=compiler_job["expected_library_sha256"],
               expected_cuda_source_sha256=compiler_job["expected_cuda_source_sha256"])
    files = {}
    for name in ("v1_cuda_architecture_qualification.py", "pack_results.py"):
        data = (args.template_request.parent / name).read_bytes()
        require(data == base64.b64decode(payload["files"][name]),
                "VALIDATED_HELPER_CHANGED:" + name)
        files[name] = data
    files.update({"compiler_report.json": compiler_report_bytes,
                  args.runner.name: runner_source.encode(), "job.json": encoded_json(job)})
    for name, data in files.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
    payload.update(manifest=upload, file_manifest=file_manifest,
                   files={name: base64.b64encode(data).decode() for name, data in files.items()},
                   auxiliary_sha256={name: hashlib.sha256(data).hexdigest()
                                     for name, data in files.items()})
    stage = stage.replace(original_assignment, "request=json.loads(" + repr(json.dumps(payload)) + ")", 1)
    prefix = prefix.replace(old_root, str(root)).replace("/v1_sm80/circuit_016",
                                                        "/v1_sm80/frozen_native_family")
    stage = stage.replace(old_root, str(root)).replace("INPUTS_VERIFIED_SM80_C16",
                                                       "INPUTS_VERIFIED_SM80_FROZEN_NATIVE_FAMILY")
    compile(stage, "native_family_stage.py", "exec")
    probe = prefix.split("MRLN_RESOURCE_PREFLIGHT", 2)[1].split("\n", 1)[1]
    compile(probe, "native_family_resource_preflight.py", "exec")
    entrypoint = (prefix + opener + stage + closer + "exec python3 -u " + str(root / args.runner.name)
                  + " --job-file " + str(root / "job.json") + "\n")
    require("numpy==2.4.2 pandas==2.3.3 scipy==1.16.2 h5py==3.14.0 PyYAML==6.0.3 psutil==7.1.0"
            in entrypoint, "QUALIFIED_ENVIRONMENT_DIFFERS")
    image = template["job_config"]["job_template_config"]["image_meta"]["image_url"]
    require(image.endswith("devbox.pytorch2:d4e66f435a261cef842a651afee3e09a"),
            "QUALIFIED_IMAGE_DIFFERS")
    template.update(name=args.name, description="Frozen C0/B2 ten-condition native family: "
                    "sm80 comparison and independent replay; development qualification only.",
                    request_context={})
    template["job_config"]["job_template_config"]["entrypoint_full_script"] = entrypoint
    request_bytes = encoded_json(template)
    check = {"check_point": "create", "node_type": "merlin_job_run", "job_run_context": template}
    args.output.mkdir(parents=True, exist_ok=True)
    request_path = args.output / (args.name + ".request.json")
    check_path = args.output / (args.name + ".check.json")
    entrypoint_path = args.output / "entrypoint.sh"
    save(args.output / "job.json", files["job.json"])
    save(entrypoint_path, entrypoint.encode())
    save(request_path, request_bytes)
    save(check_path, encoded_json(check))
    subprocess.run(["bash", "-n", str(entrypoint_path)], check=True)
    manifest = {
        "schema": "v1_frozen_native_batch_request_preparation_v1",
        "status": "PASS_PREPARED_REQUEST_STATIC_CHECKS", "name": args.name,
        "source_preparation": {"path": str(args.input_directory / "preparation.json"),
                               **identity(args.input_directory / "preparation.json")},
        "input_archive": preparation["archive"], "uploaded_chunks": 58,
        "remote_input_manifest_hash_checks": 114, "original_declared_file_checks": 113,
        "runtime_code": constants["RUNTIME_CODE"], "cases_sha256": constants["CASES_SHA256"],
        "source_batch_sha256": constants["SOURCE_BATCH_SHA256"],
        "candidate_library": payload["candidate_library"], "compiler_run_sid": compiler["run_sid"],
        "staged_auxiliary_sha256": payload["auxiliary_sha256"],
        "resource_config": template["resource_config"], "image": image,
        "request": {"path": str(request_path), **identity(request_path)},
        "check_payload": {"path": str(check_path), **identity(check_path)},
        "entrypoint": {"path": str(entrypoint_path), **identity(entrypoint_path)},
        "job": {"path": str(args.output / "job.json"), **identity(args.output / "job.json")},
        "bash_syntax": "PASS", "python_static_compile": "PASS", "precheck_performed": False,
        "submitted": False, "scientific_qualification": False, "production_qualified": False,
        "consumer": "Root uses this request and check payload after refreshing resource "
                    "availability and obtaining a new platform precheck.",
    }
    save(args.output / "manifest.json", encoded_json(manifest))
    print(json.dumps({"status": manifest["status"], "request": manifest["request"],
                      "manifest": str(args.output / "manifest.json"),
                      "resources": template["resource_config"]}), flush=True)


if __name__ == "__main__":
    main()

