#!/usr/bin/env python3
"""Prepare Merlin source-block or native-comparison jobs from the verified sm80 launcher.

The generated jobs retain frozen numerical modules, inputs and CUDA build.
They run backend validation and add no production episode identities.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib


def parallel_source_command(root: str, job_names: list[str], workers: int) -> str:
    """Run isolated workers and publish their dispatch logs before pod exit."""
    program = f'''import json, os, shutil, subprocess, sys
from pathlib import Path
root = Path({root!r})
workspace = root / "source_workspaces"
logs = root / "source_batch_logs"
logs.mkdir(exist_ok=False)
command = [sys.executable, str(root / "v1_merlin_source_batch.py"),
           "--runtime-dir", str(root), "--workspace-root", str(workspace),
           "--workers", {str(workers)!r}]
for name in {job_names!r}:
    command.extend(["--job-file", str(root / name)])
with (logs / "dispatcher.log").open("wb") as log:
    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
report_path = workspace / "batch_results.json"
report = json.loads(report_path.read_text()) if report_path.exists() else {{
    "result_kind": "process_dispatch", "status": "FAILED", "jobs": [],
    "error": "dispatcher did not produce batch_results.json"}}
if code == 0 and not report_path.exists():
    code = 1
for job in report["jobs"]:
    path = Path(job["log"])
    if path.is_file():
        shutil.copyfile(path, logs / (job["job_id"] + ".log"))
target = (os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") +
          "/v1/source_batch/" + os.environ["ARNOLD_TRIAL_ID"])
subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
subprocess.run(["hdfs", "dfs", "-put", str(logs), target + "/logs"], check=True)
report.update(exit_code=code, persisted=True, checkpoint_path=target,
              merlin_job_run_id=os.environ["RH2_JOB_RUN_ID"],
              merlin_trial_id=os.environ["ARNOLD_TRIAL_ID"],
              new_production_episodes=0, scientific_admission_inferred=False)
receipt = root / "source_batch_receipt.json"
receipt.write_text(json.dumps(report, indent=2) + "\\n")
subprocess.run(["hdfs", "dfs", "-put", str(receipt), target + "/receipt.json"], check=True)
print(json.dumps({{"status": report["status"], "exit_code": code,
                  "workers": {workers}, "checkpoint_path": target}}), flush=True)
raise SystemExit(code)
'''
    ast.parse(program)
    return "exec python3 - <<'MRLN_V1_SOURCE_BATCH'\n" + program + "MRLN_V1_SOURCE_BATCH\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migration-tools", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--upload-manifest", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-upload-manifest", type=Path, action="append", default=[])
    parser.add_argument("--base-file-manifest", type=Path, action="append", default=[])
    parser.add_argument("--source-job", type=Path,
                        help="Saved source command; defaults to record/source_job.json")
    parser.add_argument("--following-source-job", type=Path, action="append", default=[],
                        help="Include another saved complete source block")
    parser.add_argument("--source-workers", type=int, default=1,
                        help="Independent source processes; default keeps sequential execution")
    args = parser.parse_args()
    if not args.name.startswith("mrln_") or len(args.name) > 90:
        raise ValueError("Merlin name must start with mrln_ and fit the platform limit")
    upload = json.loads(args.upload_manifest.read_text())
    if upload.get("complete") is not True:
        raise ValueError("Complete verified input upload required before job preparation")
    if sum(row["bytes"] for row in upload["chunks"]) != upload["bytes"]:
        raise ValueError("Input chunk coverage differs")
    if len(args.base_upload_manifest) != len(args.base_file_manifest):
        raise ValueError("Base archive upload and file manifests must be supplied together")
    base_uploads = [json.loads(path.read_text()) for path in args.base_upload_manifest]
    for base_upload in base_uploads:
        if (base_upload.get("complete") is not True
                or sum(row["bytes"] for row in base_upload["chunks"]) != base_upload["bytes"]):
            raise ValueError("Complete verified base upload required")
    tools = args.migration_tools
    base = "mrln_v1_sm80_native_env_20261021"
    script = (tools / (base + ".sh")).read_text()
    start = "python3 - <<'MRLN_V1_STAGE'\n"
    stop = "\nMRLN_V1_STAGE\n"
    prefix, tail = script.split(start)
    if "set -euo pipefail" not in prefix:
        raise ValueError("SEQUENTIAL_SOURCE_JOBS_REQUIRE_FAIL_FAST_SHELL")
    stage, _ = tail.split(stop)
    assignments = [node for node in ast.parse(stage).body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "request"]
    if len(assignments) != 1:
        raise ValueError("Verified launcher request anchor differs")
    assignment = assignments[0]
    payload = json.loads(ast.literal_eval(assignment.value.args[0]))
    prototype = json.loads(base64.b64decode(payload["files"]["v1_sm80_qualification_job.json"]))
    job = json.loads((args.source_job or args.record / "source_job.json").read_text())
    if job["mode"] not in {"source_batch", "compare"}:
        raise ValueError("A complete source-block or saved native-comparison command is required")
    jobs = [job, *(json.loads(path.read_text()) for path in args.following_source_job)]
    if args.source_workers < 1 or (args.source_workers > 1 and any(
            item["mode"] != "source_batch" for item in jobs)):
        raise ValueError("POSITIVE_WORKER_COUNT_AND_COMPLETE_SOURCE_BLOCKS_REQUIRED")
    if any(item["mode"] != "source_batch" for item in jobs[1:]):
        raise ValueError("FOLLOWING_JOBS_MUST_BE_COMPLETE_SOURCE_BLOCKS")
    if (len({item["job"] for item in jobs}) != len(jobs)
            or any(any(item[key] != job[key] for key in ("reference", "snapshot", "code")) for item in jobs)):
        raise ValueError("DISTINCT_JOBS_WITH_COMMON_CIRCUIT_REQUIRED")
    cases = [item["cases"] for item in jobs if item["mode"] == "source_batch"]
    if len(set(cases)) != len(cases):
        raise ValueError("DUPLICATE_SOURCE_BLOCK_IN_BATCH")
    old_root = "/tmp/neuronsim-v1-sm80-qualification-20261021"
    new_root = "/tmp/neuronsim-v1-sm80-block-validation"
    for item in jobs:
        item.update(scope="cuda_architecture_qualification", expected_compute_capability=[8, 0],
                    output="/tmp/neuronsim-v1-sm80-block-results", return_chunk_bytes=256 * 1024)
        for key in ("library", "build_report", "expected_library_sha256", "expected_cuda_source_sha256"):
            item[key] = prototype[key]
        for key in ("library", "build_report"):
            item[key] = item[key].replace(old_root, new_root)
    auxiliary = {
        "v1_remote.py": (tools / "v1_remote.py").read_bytes(),
        "pack_results.py": (tools / "pack_results.py").read_bytes(),
        "compiler_report.json": base64.b64decode(payload["files"]["compiler_report.json"]),
    }
    if args.source_workers > 1:
        auxiliary["v1_merlin_source_batch.py"] = (
            Path(__file__).with_name("v1_merlin_source_batch.py").read_bytes())
    job_names = ["source_job.json", *(f"source_job_{index}.json" for index in range(1, len(jobs)))]
    auxiliary.update({name: (json.dumps(item, indent=2) + "\n").encode()
                      for name, item in zip(job_names, jobs)})
    payload["manifest"] = upload
    payload["file_manifest"] = json.loads((args.record / "input_files.json").read_text())
    for path in args.base_file_manifest:
        base_files = json.loads(path.read_text())
        if set(base_files) & set(payload["file_manifest"]):
            raise ValueError("Delta must not overwrite immutable base members")
        payload["file_manifest"] = dict(base_files, **payload["file_manifest"])
    payload["base_manifests"] = base_uploads
    if any(path not in payload["file_manifest"] for path in cases):
        raise ValueError("COMPLETE_SOURCE_CASE_FILES_REQUIRED")
    payload["files"] = {name: base64.b64encode(value).decode() for name, value in auxiliary.items()}
    payload["auxiliary_sha256"] = {name: hashlib.sha256(value).hexdigest()
                                    for name, value in auxiliary.items()}
    compressed = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    lines = stage.splitlines()
    lines[assignment.lineno - 1:assignment.end_lineno] = [
        "import zlib", "request=json.loads(zlib.decompress(base64.b64decode(" + repr(compressed) + ")))"
    ]
    stage = "\n".join(lines).replace(old_root, new_root)
    download_start = "manifest=request['manifest'];h=hashlib.sha256()\n"
    download_end = "subprocess.run(['sudo','-n','mkdir','-p','/workspace'],check=True)"
    head, remainder = stage.split(download_start)
    _, tail = remainder.split(download_end)
    download = """manifests=request['base_manifests']+[request['manifest']]
input_archives=[]
for archive_index,manifest in enumerate(manifests):
 if [row['index'] for row in manifest['chunks']]!=list(range(len(manifest['chunks']))):raise ValueError('V1_INPUT_CHUNK_ORDER_DIFFERS')
 h=hashlib.sha256();total=0
 archive_path=root/('inputs_'+str(archive_index)+'.tar.gz')
 with archive_path.open('xb') as target:
  for row in manifest['chunks']:
   part=root/'input_chunk.part'
   subprocess.run(['hdfs','dfs','-get',row['hdfs_path'],str(part)],check=True)
   chunk_hash=hashlib.sha256();chunk_bytes=0
   with part.open('rb') as source:
    while data:=source.read(8388608):
     chunk_hash.update(data);chunk_bytes+=len(data);h.update(data);target.write(data)
   if chunk_bytes!=row['bytes'] or chunk_hash.hexdigest()!=row['sha256']:raise ValueError('V1_INPUT_CHUNK_DIFFERS')
   total+=chunk_bytes;part.unlink()
 if total!=manifest['bytes'] or h.hexdigest()!=manifest['sha256']:raise ValueError('V1_INPUT_ARCHIVE_DIFFERS')
 input_archives.append(archive_path)
"""
    ast.parse(download)
    extraction_start = "with tarfile.open(root/'inputs.tar.gz') as archive:\n"
    extraction_end = "for name,row in request['file_manifest'].items():\n"
    before, remainder = tail.split(extraction_start)
    extraction, after = remainder.split(extraction_end)
    extraction = "for archive_path in input_archives:\n with tarfile.open(archive_path) as archive:\n" + "\n".join(
        " " + line for line in extraction.rstrip().splitlines()) + "\n"
    stage = head + download + download_end + before + extraction + extraction_end + after
    stage = stage.replace("INPUTS_VERIFIED_SM80_C16", "INPUTS_VERIFIED_SM80_COMPLETE_BLOCK")
    prefix = prefix.replace(old_root, new_root).replace("/v1_sm80/circuit_016", "/v1/job_" + job["job"])
    command = "".join(("exec " if index == len(job_names) - 1 else "")
                      + "python3 -u " + new_root + "/v1_remote.py --job-file " + new_root + "/" + name + "\n"
                      for index, name in enumerate(job_names))
    if args.source_workers > 1:
        command = parallel_source_command(new_root, job_names, args.source_workers)
    ast.parse(stage)
    generated = prefix + start + stage + stop + command
    request = json.loads((tools / (base + ".request.json")).read_text())
    request["name"] = args.name
    request["description"] = (
        "Complete ten-condition sm80 source execution with the saved frozen block inputs. "
        "A separate Merlin run supplies the independent replay. "
        "Preserve all outcomes and readout gates; add zero production episodes."
        if job["mode"] == "source_batch" else
        "Compare the sm80 candidate against this circuit's saved native reference. "
        "Preserve frozen inputs and numerical tolerances; add zero production episodes.")
    if args.following_source_job:
        request["description"] += (" Run " + str(len(cases)) + " saved complete blocks with "
                                   + str(args.source_workers) + " independent source workers. "
                                   "Stop dispatch on failure, retain running producers and persist each job.")
    request.pop("request_context", None)
    request["job_config"]["job_template_config"]["entrypoint_full_script"] = generated
    for suffix, value in ((".sh", generated), (".request.json", json.dumps(request, indent=2) + "\n")):
        with (args.record / (args.name + suffix)).open("x") as stream:
            stream.write(value)
    evidence = dict(name=args.name, source_job=job["job"], input_bytes=upload["bytes"],
                    input_sha256=upload["sha256"], frozen_input_files=len(payload["file_manifest"]),
                    base_archive_sha256=base_uploads[0]["sha256"] if len(base_uploads) == 1 else None,
                    base_archive_sha256s=[item["sha256"] for item in base_uploads],
                    source_jobs=[item["job"] for item in jobs],
                    library_sha256=job["expected_library_sha256"],
                    auxiliary_sha256=payload["auxiliary_sha256"],
                    source_workers=args.source_workers,
                    image=request["job_config"]["job_template_config"]["image_meta"],
                    new_production_episodes=0, production_qualified=False)
    with (args.record / (args.name + ".prepared.json")).open("x") as stream:
        json.dump(evidence, stream, indent=2)
        stream.write("\n")
    print(json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
