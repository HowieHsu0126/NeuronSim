"""Republish a verified diagnostic return with bounded HDFS preview objects."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

# This block size already returned the native Planner archive successfully.
RETURN_CHUNK_BYTES = 32 * 1024
UPLOAD_BATCH_FILES = 64


def validate_source(source):
    status = source.get("status")
    source_code = source.get("source_exit_code")
    valid_outcome = (
        status == "PASS_FAILED_SOURCE_DIAGNOSTICS"
        and type(source_code) is int and source_code != 0
        and source.get("source_outcome") == "failed_diagnostics"
        and source.get("source_status") in {"FAIL", "FAILED"}
    ) or (
        status == "PASS" and type(source_code) is int and source_code == 0
        and source.get("source_outcome") == "completed"
        and source.get("source_status") not in {None, "FAIL", "FAILED"}
    )
    if (not valid_outcome
            or source.get("exit_code") != 0 or source.get("persisted") is not True
            or not source.get("source_run_id") or not source.get("source_trial_id")):
        raise ValueError("Verified source return and consistent outcome required")
    reference = source["returned_files"]
    if reference["bytes"] != source["result_bytes"] or reference["sha256"] != source["result_sha256"]:
        raise ValueError("Verified payload identity differs")
    return reference


def verify_content(content, reference):
    if len(content) != reference["bytes"] or hashlib.sha256(content).hexdigest() != reference["sha256"]:
        raise ValueError("Returned payload bytes differ")


def transport_reference(source, artifact):
    payload = validate_source(source)
    if artifact == "payload":
        return payload
    if artifact == "source_archive":
        return source["bundle"]
    raise ValueError("Unsupported transport artifact")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_receipt")
    parser.add_argument("--artifact", choices=("payload", "source_archive"), default="payload")
    arguments = parser.parse_args()
    source = json.loads(Path(arguments.source_receipt).read_text())
    reference = transport_reference(source, arguments.artifact)
    if os.environ.get("ARNOLD_APPLIED_GPU_NUM") != "0":
        raise ValueError("Actual Merlin CPU allocation required")
    run_id = os.environ["RH2_JOB_RUN_ID"]
    trial_id = os.environ["ARNOLD_TRIAL_ID"]
    work = Path("/tmp") / ("appworld-return-fragments-" + run_id)
    work.mkdir(exist_ok=False)
    prefix = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/appworld_return_fragments"

    def hdfs(*arguments):
        subprocess.run(["hdfs", "dfs", *arguments], check=True, timeout=300)

    payload = work / "source_artifact"
    hdfs("-get", reference["uri"], str(payload))
    content = payload.read_bytes()
    verify_content(content, reference)
    if arguments.artifact == "payload":
        document = json.loads(content)
        if (document["source_run_id"] != source["source_run_id"]
                or document["source_trial_id"] != source["source_trial_id"]):
            raise ValueError("Original execution identity differs")
    hdfs("-mkdir", "-p", prefix)
    chunks = []
    pending = []

    def publish_pending():
        if pending:
            hdfs("-put", *[str(path) for path in pending], prefix + "/")
            pending.clear()

    for index, offset in enumerate(range(0, len(content), RETURN_CHUNK_BYTES)):
        block = content[offset:offset + RETURN_CHUNK_BYTES]
        name = f"chunk_{index:05d}.json"
        path = work / name
        path.write_text(json.dumps({"content_base64": base64.b64encode(block).decode()}) + "\n")
        chunks.append({"name": name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
        pending.append(path)
        if len(pending) == UPLOAD_BATCH_FILES:
            publish_pending()
    publish_pending()
    receipt = {
        "status": "PASS_VERIFIED_SOURCE_RETURN_FRAGMENTS", "exit_code": 0, "persisted": True,
        "run_id": run_id, "trial_id": trial_id, "hostname": socket.gethostname(),
        "source_run_id": source["source_run_id"], "source_trial_id": source["source_trial_id"],
        "source_status": source["source_status"], "source_exit_code": source["source_exit_code"],
        "source_outcome": source["source_outcome"],
        "original_return_run_id": source["run_id"], "original_return_trial_id": source["trial_id"],
        "checkpoint_path": prefix, "result_bytes": reference["bytes"],
        "result_sha256": reference["sha256"], "chunks": chunks,
        "output_files": source["output_files"], "original_bundle": source["bundle"],
        "artifact": arguments.artifact,
        "native_tasks_executed": 0, "model_generations": 0, "formal_eligible": False,
        "canonical_integrated": False, "source_science_repeated": False,
    }
    path = work / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    hdfs("-put", str(path), prefix + "/receipt.json")
    print(json.dumps({key: receipt[key] for key in ("status", "run_id", "trial_id", "result_bytes", "source_exit_code")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
