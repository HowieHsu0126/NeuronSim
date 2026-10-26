#!/usr/bin/env python3
"""Adapt the saved Merlin comparison request to a verified native input package."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(template_path: Path, prepared: Path, migration_tools: Path,
          name: str, gpu_type: str) -> tuple[dict, str]:
    if not name.startswith("mrln_") or len(name) > 90:
        raise ValueError("MERLIN_MIGRATION_NAME_REQUIRED")
    preparation = load(prepared / "preparation.json")
    upload = load(prepared / "upload.json")
    identity = preparation["archive"]["identity"]
    if (not preparation["complete"] or not upload.get("complete")
            or upload["sha256"] != identity["sha256"]
            or upload["bytes"] != identity["bytes"]
            or sum(row["bytes"] for row in upload["chunks"]) != identity["bytes"]):
        raise ValueError("COMPLETE_MATCHED_INPUT_UPLOAD_REQUIRED")
    for filename, key in (("job.json", "job"), ("input_manifest.json", "file_manifest")):
        if digest(prepared / filename) != preparation[key]["sha256"]:
            raise ValueError("PREPARED_CONTROL_CHANGED:" + filename)
    job = load(prepared / "job.json")
    if job["mode"] != "compare" or job["cases"] is not None:
        raise ValueError("NATIVE_COMPARISON_ONLY")
    if job["expected_compute_capability"] != [8, 9]:
        raise ValueError("ORIGINAL_SM89_COMPARISON_REQUIRED")

    payload = load(template_path)
    script = payload["job_config"]["job_template_config"]["entrypoint_full_script"]
    marker = "python3 - <<'MRLN_V1_STAGE'\n"
    before, stage_tail = script.split(marker)
    stage, after = stage_tail.split("\nMRLN_V1_STAGE\n")
    environment_request = load(migration_tools / "mrln_v1_sm80_native_env_20261021.request.json")
    environment = environment_request["job_config"]["job_template_config"]
    if environment["image_meta"] != payload["job_config"]["job_template_config"]["image_meta"]:
        raise ValueError("VALIDATED_IMAGE_IDENTITY_DIFFERS")
    installs = [line for line in environment["entrypoint_full_script"].splitlines()
                if line.startswith("python3 -m pip install ")]
    original_install = "python3 -m pip install numpy==2.4.2 scipy h5py PyYAML psutil"
    if len(installs) != 1 or before.count(original_install) != 1:
        raise ValueError("VALIDATED_ENVIRONMENT_INSTALL_ANCHOR_DIFFERS")
    # Reuse the environment that passed the C16 native comparison and replay.
    # Its pandas pin also fixes the observed missing-import startup failure.
    before = before.replace(original_install, installs[0], 1)
    statements = ast.parse(stage).body
    assignments = [node for node in statements if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "request"]
    if len(assignments) != 1:
        raise ValueError("STAGING_REQUEST_ASSIGNMENT_DIFFERS")
    assignment = assignments[0]
    original_stage = json.loads(ast.literal_eval(assignment.value.args[0]))
    old_jobs = [key for key in original_stage["files"] if key.startswith("v1_job_")]
    if len(old_jobs) != 1:
        raise ValueError("TEMPLATE_JOB_IDENTITY_DIFFERS")
    old_job = json.loads(base64.b64decode(original_stage["files"][old_jobs[0]]))

    # Restore the detector invocation present in the original Slurm entrypoint.
    # Numerical modules and the native CUDA library remain in the frozen archive.
    runner = (migration_tools / "v1_remote.py").read_text()
    anchor = "    try:\n        library = Path(job['library'])"
    replacement = (
        "    try:\n"
        "        subprocess.run([sys.executable, '-B', job['detector'],\n"
        "            '-o', str(output / 'resources.json')], check=True)\n"
        "        library = Path(job['library'])"
    )
    if runner.count(anchor) != 1:
        raise ValueError("COMPARISON_RUNNER_ANCHOR_DIFFERS")
    runner = runner.replace(anchor, replacement)
    runner = runner.replace("import argparse\n", "import argparse\nfrom importlib.metadata import version\n", 1)
    version_anchor = "        gpu_name=torch.cuda.get_device_name(0),"
    if runner.count(version_anchor) != 1:
        raise ValueError("COMPARISON_RECEIPT_ANCHOR_DIFFERS")
    runner = runner.replace(version_anchor,
        "        package_versions={package: version(package) for package in\n"
        "            ('numpy', 'pandas', 'scipy', 'h5py', 'PyYAML', 'psutil', 'torch')},\n"
        + version_anchor)
    chunk_anchor = "while data := stream.read(1024 * 1024):"
    if runner.count(chunk_anchor) != 1:
        raise ValueError("COMPARISON_RETURN_CHUNK_ANCHOR_DIFFERS")
    # The existing return route truncated 1-MiB previews. C98/C99 use this
    # verified transport size; the whole binary bundle and its hash are retained.
    runner = runner.replace(chunk_anchor, "while data := stream.read(256 * 1024):")
    compile(runner, "v1_remote.py", "exec")
    files = {
        "v1_remote.py": base64.b64encode(runner.encode()).decode(),
        "pack_results.py": base64.b64encode((migration_tools / "pack_results.py").read_bytes()).decode(),
        f"v1_job_{job['job']}.json": base64.b64encode(json.dumps(job).encode()).decode(),
    }
    staged = dict(manifest=upload, files=files, file_manifest=load(prepared / "input_manifest.json"))
    lines = stage.splitlines()
    lines[assignment.lineno - 1:assignment.end_lineno] = ["request=json.loads(" + repr(json.dumps(staged)) + ")"]
    stage = "\n".join(lines)
    # These tokens occur only in orchestration paths and compatibility identifiers.
    # Encoded scientific inputs are replaced above, never rewritten in place.
    before = before.replace(old_job["job"], job["job"])
    stage = stage.replace("/tmp/neuronsim-v1-" + old_job["job"] + "-20261021",
                          "/tmp/neuronsim-v1-" + job["job"] + "-20261021")
    after = after.replace(old_job["job"], job["job"])
    compile(stage, "v1_stage.py", "exec")
    payload["job_config"]["job_template_config"]["entrypoint_full_script"] = (
        before + marker + stage + "\nMRLN_V1_STAGE\n" + after)
    payload["name"] = name
    payload["description"] = (
        f"{name}: native comparison for original Xie job {job['job']}; "
        "verified complete native inputs, unchanged sm89 library and numerical modules; "
        "original detector restored; development qualification only."
    )
    payload["request_context"] = {}
    resource = payload["resource_config"]["arnold_resource_config"]
    if (resource["cluster_ids"] != [61] or resource["group_id"] != 2088
            or len(resource["roles"]) != 1):
        raise ValueError("SAVED_RESOURCE_CONTEXT_DIFFERS")
    role = resource["roles"][0]
    if (role["gpu"] != 1 or role["cpu"] != 22 or role["memory_mb"] != 117760):
        raise ValueError("SAVED_SINGLE_GPU_BUDGET_DIFFERS")
    role["gpu_type"] = gpu_type
    return payload, runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--migration-tools", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpu-type", choices=("NVIDIA_L20", "NVIDIA_L40"), required=True)
    args = parser.parse_args()
    payload, runner = build(args.template, args.prepared, args.migration_tools,
                            args.name, args.gpu_type)
    artifacts = {
        "request.json": json.dumps(payload, indent=2) + "\n",
        "check.json": json.dumps(dict(check_point="create", node_type="merlin_job_run",
                                      job_run_context=payload), indent=2) + "\n",
        "entrypoint.sh": payload["job_config"]["job_template_config"]["entrypoint_full_script"],
        "v1_remote.py": runner,
    }
    for name, text in artifacts.items():
        target = args.prepared / name
        if target.exists() and target.read_text() != text:
            raise FileExistsError("EXISTING_REQUEST_DIFFERS:" + str(target))
    for name, text in artifacts.items():
        target = args.prepared / name
        if not target.exists():
            with target.open("x") as stream:
                stream.write(text)
    print(json.dumps(dict(status="PREPARED_VERIFIED_NATIVE_COMPARISON_REQUEST",
                          request=str(args.prepared / "request.json"), name=args.name)))


if __name__ == "__main__":
    main()
