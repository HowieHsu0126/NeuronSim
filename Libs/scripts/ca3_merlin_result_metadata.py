#!/usr/bin/env python3
"""Verify an already persisted CA3 result bundle and return its small metadata."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess
import tarfile

from ca3_merlin_remote import digest, persist, write_json
from v1_merlin_readout import cpu_preflight


def execute(job: dict, work: Path) -> int:
    output = work / "output"
    output.mkdir(exist_ok=False)
    trial = os.environ["ARNOLD_TRIAL_ID"]
    target = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/ca3/result_metadata/" + trial
    receipt = dict(status="FAILED_CA3_RESULT_METADATA_VERIFICATION", exit_code=1, persisted=False,
        merlin_job_run_id=os.environ["RH2_JOB_RUN_ID"], merlin_trial_id=trial,
        hostname=socket.gethostname(), execution_backend="merlin", output=str(output),
        started_utc=datetime.now(timezone.utc).isoformat(), simulations_run=0, production_episodes=0,
        scientific_fidelity_qualified=False, original_run_sid=job["original_run_sid"],
        original_trial_sid=job["original_trial_sid"])
    try:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
        receipt["resource_preflight"] = cpu_preflight(work, job["resource_shape"])
        bundle = work / "source_result.tar.gz"
        subprocess.run(["hdfs", "dfs", "-get", job["source_bundle_uri"], str(bundle)], check=True)
        if bundle.stat().st_size != job["source_bundle_bytes"]:
            raise ValueError("CA3_SOURCE_RESULT_BUNDLE_SIZE_DIFFERS")
        source_sha = digest(bundle)
        prefix = "tmp/neuronsim-ca3-" + job["original_trial_sid"] + "/output/"
        observed, expected = {}, None
        metadata_root = output / "original_metadata"
        metadata_root.mkdir()
        # The packer writes its manifest last. Verify each file in one streaming
        # decompression pass, then compare all observed identities to that manifest.
        with tarfile.open(bundle, "r|gz") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if not member.isfile() or name.is_absolute() or ".." in name.parts:
                    raise ValueError("CA3_RESULT_ARCHIVE_MEMBER_UNSAFE")
                stream = archive.extractfile(member)
                if member.name == "merlin_return_manifest.json":
                    if expected is not None or member.size > 1024 * 1024:
                        raise ValueError("CA3_RESULT_MANIFEST_INVALID")
                    expected = json.load(stream)
                    continue
                if expected is not None or member.name in observed or not member.name.startswith(prefix):
                    raise ValueError("CA3_RESULT_ARCHIVE_SCOPE_DIFFERS")
                metadata = name.suffix in (".json", ".log")
                if metadata and member.size > 8 * 1024 * 1024:
                    raise ValueError("CA3_RESULT_METADATA_EXCEEDS_BOUND:" + member.name)
                chunks = []
                hashed, size = hashlib.sha256(), 0
                while data := stream.read(8 * 1024 * 1024):
                    hashed.update(data)
                    size += len(data)
                    if metadata:
                        chunks.append(data)
                if size != member.size:
                    raise ValueError("CA3_RESULT_MEMBER_SIZE_DIFFERS")
                observed[member.name] = dict(bytes=size, sha256=hashed.hexdigest())
                if metadata:
                    path = metadata_root / member.name[len(prefix):]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"".join(chunks))
        if expected != observed or len(observed) != job["source_result_files"]:
            raise ValueError("CA3_RESULT_FILE_MANIFEST_DIFFERS")
        terminal = json.loads((metadata_root / "merlin_terminal.json").read_text())
        if (terminal["merlin_job_run_id"] != job["original_run_sid"]
                or terminal["merlin_trial_id"] != job["original_trial_sid"]
                or terminal["scope"] != "ca3_sm80_candidate_1210ms_completion50_prefix_replay"
                or type(terminal.get("exit_code")) is not int
                or terminal.get("production_episodes") != 0):
            raise ValueError("CA3_ORIGINAL_TERMINAL_IDENTITY_DIFFERS")
        write_json(output / "source_result_manifest.json", observed)
        write_json(output / "execution_inputs.json", job)
        receipt.update(status="PASS_CA3_RESULT_METADATA_VERIFICATION", exit_code=0,
            source_bundle_uri=job["source_bundle_uri"], source_bundle_bytes=bundle.stat().st_size,
            source_bundle_sha256=source_sha, source_result_files_verified=len(observed),
            original_source_status=terminal["status"], original_source_exit_code=terminal["exit_code"],
            original_source_error=terminal.get("error"),
            metadata_only_return=True, full_source_bundle_retained_in_hdfs=True)
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "merlin_terminal.json", receipt)
        try:
            persist(output, work, target, receipt, chunk_bytes=256 * 1024)
        except BaseException as error:
            receipt.update(persisted=False, exit_code=receipt["exit_code"] or 1,
                           persistence_error_type=type(error).__name__)
        print(json.dumps(receipt), flush=True)
    return receipt["exit_code"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", type=Path, required=True)
    args = parser.parse_args()
    return execute(json.loads(args.job_json.read_text()), args.job_json.resolve().parent)


if __name__ == "__main__":
    raise SystemExit(main())
