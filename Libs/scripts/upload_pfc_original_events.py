"""Transfer exactly the existing original35 PFC raw event files from cpu05.

This is a bounded data transfer. It starts no Slurm jobs or simulations. Unknown
upload outcomes stop before another create; confirmed chunks support resumption.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

import requests

sys.path.insert(0, "/opt/cluster-tools/scripts")
from merlin_io import atomic_json, http_session, call


class UploadRequestNotSentError(RuntimeError):
    """The proxy or TLS connection failed before the upload request."""


def upload_chunk_once_connected(name: str, data: bytes):
    """Retry only failed proxy connections, which did not send the upload."""
    for attempt in range(3):
        try:
            response = http_session().post(
                "https://cluster-api.example.invalid/api/v4/seed/data/datacard-upload-files/create",
                files={"file": (name, io.BytesIO(data), "application/octet-stream")},
                timeout=(20, 60), allow_redirects=False)
            if response.status_code == 401 and attempt < 2:
                # A401 is a definite rejection. The existing CLI refreshes the
                # shared authentication during this read-only run query.
                call(["job-v2", "runs", "get"], {"sid": "413de649f13ddf8f"}, timeout=45)
                continue
            return response
        except requests.exceptions.RequestException as error:
            chain, inspected, before_request = error, set(), isinstance(error, requests.exceptions.ProxyError)
            while chain is not None and id(chain) not in inspected:
                inspected.add(id(chain))
                trace = chain.__traceback__
                while trace is not None:
                    if trace.tb_frame.f_code.co_name in ("do_handshake", "_prepare_proxy", "_tunnel"):
                        before_request = True
                    trace = trace.tb_next
                chain = chain.__context__ or chain.__cause__
            if not before_request:
                raise
            if attempt == 2:
                raise UploadRequestNotSentError(
                    "Proxy/TLS connection failed before upload: " + str(error)) from error
            time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--chunk-mib", type=int, default=4)
    parser.add_argument("--compression", choices=("none", "gzip"), default="none")
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "cpu05" or not 0 <= args.start < args.stop <= 35:
        raise ValueError("PFC_ORIGINAL_EVENTS_BOUNDED_CPU05_TRANSFER_REQUIRED")
    if not 1 <= args.chunk_mib <= 16:
        raise ValueError("PFC_ORIGINAL_EVENT_CHUNK_SIZE_INVALID")
    rows = json.loads(args.inputs.read_bytes())
    if [r["index"] for r in rows] != list(range(35)):
        raise ValueError("PFC_ORIGINAL35_COMPLETE_INPUT_IDENTITIES_REQUIRED")
    args.output.mkdir(exist_ok=True)
    scratch = Path("/tmp/neuronsim-pfc-original35-transfer")
    scratch.mkdir(exist_ok=True)
    if shutil.disk_usage(scratch).free < 2 * 1024**3:
        raise RuntimeError("PFC_ORIGINAL_EVENTS_SCRATCH_SPACE_REQUIRED")
    for row in rows[args.start:args.stop]:
        index = row["index"]
        manifest = args.output / f"task_{index:04d}_upload.json"
        inflight = args.output / f"task_{index:04d}_inflight.json"
        if inflight.exists():
            raise RuntimeError("PFC_ORIGINAL_EVENT_UPLOAD_OUTCOME_UNKNOWN:" + str(index))
        state = json.loads(manifest.read_bytes()) if manifest.exists() else dict(
            index=index, source_path=row["path"], bytes=row["bytes"], sha256=row["sha256"],
            identity=row["identity"], source_report=row["source_report"],
            chunk_bytes=args.chunk_mib * 1024**2, chunks=[], complete=False,
            transfer_host=socket.gethostname(), source_simulations=0)
        if state["sha256"] != row["sha256"] or state["bytes"] != row["bytes"]:
            raise ValueError("PFC_ORIGINAL_EVENT_SOURCE_BINDING_CHANGED")
        if state["complete"]:
            print(json.dumps(dict(index=index, status="REUSED_COMPLETE_TRANSFER", chunks=len(state["chunks"]))), flush=True)
            continue
        local = scratch / f"task_{index:04d}_raw.npz"
        digest = hashlib.sha256()
        with Path(row["path"]).open("rb") as source, local.open("wb") as target:
            for data in iter(lambda: source.read(4 * 1024**2), b""):
                digest.update(data)
                target.write(data)
        if local.stat().st_size != row["bytes"] or digest.hexdigest() != row["sha256"]:
            raise ValueError("PFC_ORIGINAL_EVENT_INPUT_CHANGED:" + str(index))
        if args.compression == "gzip":
            if state.get("encoding", "gzip") != "gzip" or (state["chunks"] and "encoding" not in state):
                raise ValueError("PFC_ORIGINAL_EVENT_TRANSFER_ENCODING_CHANGED")
            encoded = scratch / f"task_{index:04d}_raw.npz.gz"
            with local.open("rb") as source, encoded.open("wb") as target:
                with gzip.GzipFile(fileobj=target, filename="", mode="wb", compresslevel=1, mtime=0) as stream:
                    shutil.copyfileobj(source, stream, 4 * 1024**2)
            local.unlink()
            local = encoded
            with local.open("rb") as stream:
                encoded_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            encoded_bytes = local.stat().st_size
            if "encoding" in state and (state["encoded_bytes"] != encoded_bytes
                    or state["encoded_sha256"] != encoded_sha256):
                raise ValueError("PFC_ORIGINAL_EVENT_ENCODED_RESUME_CHANGED")
            state.update(encoding="gzip", encoded_bytes=encoded_bytes, encoded_sha256=encoded_sha256)
        with local.open("rb") as stream:
            for chunk in state["chunks"]:
                data = stream.read(chunk["bytes"])
                if len(data) != chunk["bytes"] or hashlib.sha256(data).hexdigest() != chunk["sha256"]:
                    raise ValueError("PFC_ORIGINAL_EVENT_RESUME_CHUNK_CHANGED")
            state["chunk_bytes"] = args.chunk_mib * 1024**2
            while data := stream.read(state["chunk_bytes"]):
                number = len(state["chunks"])
                name = f"pfc_original_event_{index:04d}_{args.compression}_{number:04d}_20261023.part"
                pending = dict(index=index, chunk_index=number, file_name=name, bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(), status="CREATE_DISPATCHED",
                    recorded_utc=datetime.now(timezone.utc).isoformat())
                atomic_json(inflight, pending)
                try:
                    response = upload_chunk_once_connected(name, data)
                    if 300 <= response.status_code < 400:
                        raise RuntimeError("PFC_ORIGINAL_EVENT_UPLOAD_REDIRECT_OUTCOME_UNKNOWN")
                    response.raise_for_status()
                    value = response.json()
                    if value.get("ResponseBase", {}).get("code", 0) != 0:
                        raise RuntimeError("PFC_ORIGINAL_EVENT_UPLOAD_API_FAILED")
                    accepted = value["data"]
                    if accepted["size_bytes"] != len(data):
                        raise ValueError("PFC_ORIGINAL_EVENT_UPLOADED_SIZE_CHANGED")
                except Exception as error:
                    if isinstance(error, (requests.exceptions.ProxyError, UploadRequestNotSentError)):
                        pending.update(status="NOT_SENT_PROXY_CONNECTION_FAILED",
                            error_type=type(error).__name__, error=str(error))
                        atomic_json(args.output / f"task_{index:04d}_proxy_failure.json", pending)
                        inflight.unlink()
                        atomic_json(manifest, state)
                        raise
                    if isinstance(error, requests.exceptions.HTTPError) and error.response is not None:
                        if error.response.status_code == 401:
                            pending.update(status="CONFIRMED_AUTHORIZATION_REJECTION_401",
                                error_type=type(error).__name__)
                            atomic_json(args.output / f"task_{index:04d}_auth_rejection.json", pending)
                            inflight.unlink()
                            atomic_json(manifest, state)
                            raise
                    pending.update(status="CREATE_OUTCOME_UNKNOWN", error_type=type(error).__name__, error=str(error))
                    atomic_json(inflight, pending)
                    atomic_json(manifest, state)
                    raise
                state["chunks"].append(dict(index=number, bytes=len(data), sha256=pending["sha256"],
                    hdfs_path=accepted["hdfs_path"]))
                atomic_json(manifest, state)
                inflight.unlink()
                print(json.dumps(dict(index=index, chunks=len(state["chunks"]),
                    uploaded_bytes=sum(c["bytes"] for c in state["chunks"]))), flush=True)
        state.update(complete=True, completed_utc=datetime.now(timezone.utc).isoformat())
        if sum(c["bytes"] for c in state["chunks"]) != state.get("encoded_bytes", row["bytes"]):
            raise ValueError("PFC_ORIGINAL_EVENT_TRANSFER_DENOMINATOR_CHANGED")
        atomic_json(manifest, state)
        local.unlink()
    if not any(scratch.iterdir()):
        scratch.rmdir()


if __name__ == "__main__":
    main()
