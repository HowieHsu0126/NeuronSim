"""Verify an AppWorld result archive on Merlin CPU and return its files.

The management host receives already-verified bytes without opening the archive.
This program never runs a model, task, evaluator, or archived executable.
Failed diagnostics require an explicit source_outcome and retain the source exit.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess
import sys
import tarfile
import traceback


def identity(content):
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def check(content, reference):
    if identity(content) != {key: reference[key] for key in ("bytes", "sha256")}:
        raise ValueError("Artifact identity differs")


def validate_source(inputs):
    source = inputs["source_receipt"]
    outcome = inputs.get("source_outcome", "completed")
    if source.get("persisted") is not True:
        raise ValueError("Persisted source required")
    exit_code = source.get("exit_code")
    if type(exit_code) is not int:
        raise ValueError("Integer source exit code required")
    if outcome == "completed":
        if exit_code != 0:
            raise ValueError("Completed persisted source required")
    elif outcome == "failed_diagnostics":
        if exit_code == 0 or source.get("status") not in ("FAIL", "FAILED"):
            raise ValueError("Failed persisted source required for diagnostics")
    else:
        raise ValueError("Unsupported source outcome")
    return outcome


def main():
    inputs = json.loads(Path(sys.argv[1]).read_text())
    if os.environ.get("ARNOLD_APPLIED_GPU_NUM") != "0":
        raise ValueError("CPU allocation required")
    work = Path("/tmp") / ("appworld-result-return-" + os.environ["RH2_JOB_RUN_ID"])
    work.mkdir(exist_ok=False)
    prefix = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/appworld_result_return"
    def hdfs(*args):
        subprocess.run(["hdfs", "dfs", *args], check=True)
    report = {"schema": "appworld_merlin_result_return_v1", "run_id": os.environ["RH2_JOB_RUN_ID"],
              "trial_id": os.environ["ARNOLD_TRIAL_ID"], "hostname": socket.gethostname(),
              "source_run_id": inputs["source_receipt"]["run_id"], "status": "RUNNING",
              "native_tasks_executed": 0, "model_generations": 0, "formal_eligible": False,
              "source_trial_id": inputs["source_receipt"]["trial_id"],
              "source_exit_code": inputs["source_receipt"].get("exit_code"),
              "source_status": inputs["source_receipt"].get("status"),
              "canonical_integrated": False,
              "persisted": False, "exit_code": None}
    try:
        source = inputs["source_receipt"]
        outcome = validate_source(inputs)
        report["source_outcome"] = outcome
        reference = source["output_bundle"]
        archive = work / "outputs.tar.gz"
        hdfs("-get", reference["uri"], str(archive))
        raw = archive.read_bytes(); check(raw, reference)
        manifest_bytes = base64.b64decode(inputs["manifest_content_base64"], validate=True)
        check(manifest_bytes, source["artifact_manifest"])
        manifest = json.loads(manifest_bytes)
        if manifest["run_id"] != source["run_id"] or manifest["trial_id"] != source["trial_id"]:
            raise ValueError("Manifest source identity differs")
        expected = {"artifacts/" + name for name in manifest["files"]} | {"output_manifest.json"}
        files = {}
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError("Archive membership differs from manifest")
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not member.isfile():
                    raise ValueError("Unsafe or nonregular archive member")
                stream = handle.extractfile(member)
                if stream is None:
                    raise ValueError("Archive member unreadable")
                content = stream.read()
                if member.name == "output_manifest.json":
                    if content != manifest_bytes:
                        raise ValueError("Embedded manifest differs")
                else:
                    check(content, manifest["files"][member.name.removeprefix("artifacts/")])
                files[member.name] = {**identity(content), "content_base64": base64.b64encode(content).decode()}
        payload = {"source_run_id": source["run_id"], "source_trial_id": source["trial_id"],
                   "bundle": {**reference, "content_base64": base64.b64encode(raw).decode()}, "files": files}
        payload_path = work / "files.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
        hdfs("-mkdir", "-p", prefix)
        hdfs("-put", str(payload_path), prefix + "/files.json")
        payload_bytes = payload_path.read_bytes()
        # HDFS preview rejects files over 5 MiB. Use the existing merlin_io.collect
        # protocol for large returns; each 3 MiB part encodes below that limit.
        chunk_bytes = 3 * 1024 * 1024
        if len(payload_bytes) > chunk_bytes:
            chunks = []
            for index, offset in enumerate(range(0, len(payload_bytes), chunk_bytes)):
                content = payload_bytes[offset:offset + chunk_bytes]
                name = f"chunk_{index:05d}.json"
                path = work / name
                path.write_text(json.dumps({"content_base64": base64.b64encode(content).decode()}) + "\n")
                hdfs("-put", str(path), prefix + "/" + name)
                chunks.append({"name": name, **identity(content)})
            report.update(checkpoint_path=prefix, result_bytes=len(payload_bytes),
                          result_sha256=hashlib.sha256(payload_bytes).hexdigest(), chunks=chunks)
        status = "PASS_FAILED_SOURCE_DIAGNOSTICS" if outcome == "failed_diagnostics" else "PASS"
        report.update(status=status, exit_code=0, persisted=True, output_files=len(files)-1,
                      bundle=reference, returned_files={"uri": prefix + "/files.json", **identity(payload_bytes)})
    except Exception as error:
        report.update(status="FAIL", exit_code=1, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    receipt = work / "receipt.json"; receipt.write_text(json.dumps(report, indent=2) + "\n")
    hdfs("-mkdir", "-p", prefix); hdfs("-put", str(receipt), prefix + "/receipt.json")
    print(json.dumps(report), flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
