#!/usr/bin/env python3
"""Republish completed Merlin bundles as bounded, byte-identical return chunks."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

CHUNK_BYTES = 256 * 1024


def hdfs(*arguments: str) -> None:
    subprocess.run(["hdfs", "dfs", *arguments], check=True)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def republish(origin: dict, label: str, work: Path, checkpoint: str,
              *, chunk_bytes: int = CHUNK_BYTES) -> dict:
    # A 2 MiB binary chunk stays below the receiver's 4 MiB JSON response limit.
    if type(chunk_bytes) is not int or chunk_bytes not in (
            256 * 1024, 512 * 1024, 1024 * 1024, 2048 * 1024):
        raise ValueError("UNSUPPORTED_RETURN_CHUNK_BYTES")
    if origin.get("persisted") is not True or origin.get("exit_code") != 0:
        raise ValueError("SUCCESSFUL_PERSISTED_SOURCE_REQUIRED")
    target = checkpoint.rstrip("/") + "/result_transport/" + label
    work.mkdir(exist_ok=False)
    receipt = dict(status="FAILED_RESULT_TRANSPORT_RECHUNK", exit_code=1,
                   persisted=False, merlin_job_run_id=os.environ["RH2_JOB_RUN_ID"],
                   merlin_trial_id=os.environ["ARNOLD_TRIAL_ID"], hostname=socket.gethostname(),
                   source_simulations=0, scientific_outputs_changed=False,
                   origin_receipt=origin, started_utc=datetime.now(timezone.utc).isoformat())
    hdfs("-mkdir", "-p", target)
    try:
        bundle = work / "result.tar.gz"
        hdfs("-get", origin["checkpoint_path"].rstrip("/") + "/result.tar.gz", str(bundle))
        if bundle.stat().st_size != origin["result_bytes"] or digest(bundle) != origin["result_sha256"]:
            raise ValueError("ORIGINAL_BUNDLE_IDENTITY_DIFFERS")
        chunks = work / "chunks"
        chunks.mkdir()
        rows = []
        with bundle.open("rb") as stream:
            while block := stream.read(chunk_bytes):
                name = f"{len(rows):05d}.json"
                (chunks / name).write_text(json.dumps(dict(
                    content_base64=base64.b64encode(block).decode())) + "\n")
                rows.append(dict(name="chunks/" + name, bytes=len(block),
                                 sha256=hashlib.sha256(block).hexdigest()))
        hdfs("-put", str(bundle), target + "/result.tar.gz")
        hdfs("-put", str(chunks), target + "/chunks")
        receipt.update(status="PASS_BYTE_IDENTICAL_RESULT_TRANSPORT_RECHUNK", exit_code=0,
                       persisted=True, checkpoint_path=target, result_bytes=origin["result_bytes"],
                       result_sha256=origin["result_sha256"], chunks=rows, chunk_bytes=chunk_bytes)
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        path = work / "receipt.json"
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        hdfs("-put", str(path), target + "/receipt.json")
        print(json.dumps({key: receipt.get(key) for key in (
            "status", "exit_code", "persisted", "merlin_job_run_id", "merlin_trial_id",
            "result_bytes", "result_sha256", "error")}), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--chunk-kib", type=int, choices=(256, 512, 1024, 2048), default=256)
    args = parser.parse_args()
    if (socket.gethostname().split(".")[0] == "mgmt2"
            or os.environ.get("ARNOLD_HEAD_GPU") != "0"
            or os.environ.get("ARNOLD_APPLIED_GPU_NUM") != "0"):
        raise ValueError("MERLIN_CPU_ALLOCATION_REQUIRED")
    origins = json.loads(args.receipts.read_text())
    work = Path("/tmp") / ("neuronsim-result-transport-" + os.environ["ARNOLD_TRIAL_ID"])
    work.mkdir(exist_ok=False)
    for index, origin in enumerate(origins):
        label = str(index)
        republish(origin, label, work / label, os.environ["RH2_CHECKPOINT_HDFS"],
                  chunk_bytes=args.chunk_kib * 1024)


if __name__ == "__main__":
    main()
