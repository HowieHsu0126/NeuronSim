"""Transfer retained readout inputs from Xie to HDFS without numerical work.

Run this bitwise transfer on a permitted source-side I/O host. Scientific
execution uses Merlin. No Slurm job or model evaluation is started here.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

import requests


def multipart_chunk(data, name, helper):
    """Use the data control plane and its matching authentication cache."""
    plane = os.environ.get("MERLIN_DATA_CONTROL_PLANE", "cn-seed")
    hosts = {"cn": "cluster-api.example.invalid", "cn-seed": "cluster-seed.example.invalid"}
    if plane not in hosts:
        raise ValueError("unsupported multipart data control plane: " + plane)
    auth_path = Path(os.environ.get("MERLIN_AUTH_FILE") or
                     str(Path.home() / ".merlin-cli" / "auth" / (plane + ".json")))
    auth = json.loads(auth_path.read_text())
    session = helper.http_session()
    session.headers["x-jwt-token"] = auth["jwt_token"]
    session.cookies.clear()
    session.cookies.update(auth.get("cookies", {}))
    endpoint = "https://" + hosts[plane] + "/api/v4/seed/data/datacard-upload-files/create"
    for attempt in range(3):
        try:
            response = session.post(endpoint, files={"file": (name, io.BytesIO(data),
                                    "application/octet-stream")}, timeout=(20, 90))
            response.raise_for_status()
            value = response.json()
            if value.get("ResponseBase", {}).get("code", 0) != 0:
                raise ValueError("Multipart API failed: " + json.dumps(value["ResponseBase"]))
            return value["data"]
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as error:
            status = error.response.status_code if getattr(error, "response", None) is not None else None
            if status is not None and status < 500 and status != 429:
                raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def upload_archive(archive, manifest, chunk_bytes, helper, transport="multipart"):
    """Resume by recorded byte lengths so smaller chunks retain the old prefix."""
    state = (json.loads(manifest.read_text()) if manifest.exists() else
             dict(file=str(archive.resolve()), bytes=archive.stat().st_size, chunks=[], complete=False))
    if state["file"] != str(archive.resolve()) or state["bytes"] != archive.stat().st_size:
        raise ValueError("upload manifest belongs to a different source archive")
    state.update(chunk_bytes=chunk_bytes, chunk_layout="per_entry_byte_lengths")
    digest = hashlib.sha256()
    consumed = 0
    with archive.open("rb") as stream:
        for index, row in enumerate(state["chunks"]):
            if row["index"] != index or not 0 < row["bytes"] <= 32 * 1024**2:
                raise ValueError("invalid uploaded prefix index or byte count")
            data = stream.read(row["bytes"])
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("retained uploaded prefix differs from the source archive")
            digest.update(data)
            consumed += len(data)
        print(json.dumps(dict(reused_chunks=len(state["chunks"]), reused_bytes=consumed,
                              remaining_chunk_bytes=chunk_bytes)), flush=True)
        while data := stream.read(chunk_bytes):
            index = len(state["chunks"])
            # Single uploads worked while reused proxy sessions failed in this
            # campaign. Keep each multipart call in its own connection pool.
            session = getattr(helper.HTTP_LOCAL, "session", None)
            if session is not None:
                session.close()
                del helper.HTTP_LOCAL.session
            name = f"{archive.stem}_{index:05d}.part"
            if transport == "json":
                for attempt in range(3):
                    try:
                        response = helper.call(["data", "datacard-upload-files", "create", "--control-plane",
                            os.environ.get("MERLIN_DATA_CONTROL_PLANE", "cn-seed")],
                            dict(file_name=name, content_base64=base64.b64encode(data).decode()))["data"]
                        break
                    except (RuntimeError, subprocess.TimeoutExpired) as error:
                        message = str(error)
                        authentication_error = any(marker in message.lower() for marker in (
                            "token expired", "refresh failed", "unauthorized", "forbidden", "status code: 401",
                            "status code: 403", '"code":401', '"code":403'))
                        transient = not authentication_error and (isinstance(error, subprocess.TimeoutExpired) or any(
                            marker in message for marker in ("unexpected EOF", "context deadline exceeded")))
                        event = dict(chunk=index, attempt=attempt + 1, transient=transient,
                                     error=str(error)[-1600:])
                        state.setdefault("upload_errors", []).append(event)
                        helper.atomic_json(manifest, state)
                        print(json.dumps(dict(upload_error=event)), flush=True)
                        if not transient or attempt == 2:
                            raise
                        # An interrupted upload has no returned URI to consume.
                        # Re-send only this chunk; saved prefix receipts are reused.
                        time.sleep(2 ** attempt)
            else:
                response = multipart_chunk(data, name, helper)
            if response["size_bytes"] != len(data):
                raise ValueError("uploaded chunk byte count differs")
            sha = hashlib.sha256(data).hexdigest()
            digest.update(data)
            consumed += len(data)
            state["chunks"].append(dict(index=index, bytes=len(data), sha256=sha,
                                        hdfs_path=response["hdfs_path"]))
            helper.atomic_json(manifest, state)
            print(json.dumps(dict(chunk=index, chunks_saved=len(state["chunks"]),
                                  bytes_processed=consumed, archive_bytes=state["bytes"])), flush=True)
    if consumed != state["bytes"]:
        raise ValueError("uploaded chunks do not cover the complete archive")
    state.update(complete=True, sha256=digest.hexdigest())
    helper.atomic_json(manifest, state)
    return state


def inputs(campaign, corrections):
    for seed in (41, 42, 43):
        cases = [f"train_block_{block:03d}_familiar_{template}"
                 for block in range(14, 18) for template in range(4)]
        names = ["training/best.pt", "replay.json", "forecasts/evaluation.json", "forecasts/scores.csv"]
        names += [f"training/posterior_cache/{case}.pt" for case in cases]
        names += [f"forecasts/{case}_origin_{origin}.npz" for case in cases for origin in (1000, 3000, 5000)]
        corrected = ["provenance/fit.json"]
        for mode in ("current", "history"):
            corrected += [f"results/{mode}/replay.json", f"results/{mode}/forecasts/evaluation.json",
                          f"results/{mode}/forecasts/scores.csv"]
            corrected += [f"results/{mode}/forecasts/{case}_origin_{origin}.npz"
                          for case in cases for origin in (1000, 3000, 5000)]
        for label, root, selected in (("parent", campaign, names), ("corrected", corrections, corrected)):
            cell = root / f"seed_{seed}"
            delivery = json.loads((cell / "delivery.json").read_text())
            if delivery.get("status") != "PASS" or delivery.get("exit_code") != 0 or delivery.get("training_seed") != seed:
                raise ValueError("accepted same-seed delivery required")
            index = {row["path"]: row for row in delivery["files"]}
            if len(index) != len(delivery["files"]):
                raise ValueError("duplicate delivery paths")
            yield cell / "delivery.json", f"{label}/seed_{seed}/delivery.json", None
            for name in selected:
                yield cell / name, f"{label}/seed_{seed}/{name}", index[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "corrections", "qualification", "comparison", "output", "helper-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp"))
    parser.add_argument("--chunk-mib", type=int, choices=range(1, 33), default=16)
    parser.add_argument("--transport", choices=("multipart", "json"), default="multipart")
    args = parser.parse_args()
    for name in ("campaign", "corrections", "qualification", "comparison", "output", "helper_dir", "scratch"):
        setattr(args, name, getattr(args, name).resolve())
    if platform.node().split(".")[0] not in ("cpu03", "cpu04"):
        raise RuntimeError("source transfer requires an allowed source-side I/O host")
    fs = subprocess.check_output(["findmnt", "-T", str(args.scratch), "-n", "-o", "FSTYPE"], text=True).strip()
    if fs not in ("xfs", "ext4", "tmpfs") or shutil.disk_usage(args.scratch).free < 20 * 1024**3:
        raise RuntimeError("local scratch with at least20GiB available required")
    sys.path.insert(0, str(args.helper_dir))
    import merlin_io
    atomic_json = merlin_io.atomic_json
    args.output.mkdir(parents=True, exist_ok=True)
    upload_path = args.output / "upload.json"
    if upload_path.exists():
        previous = json.loads(upload_path.read_text())
        if previous.get("complete") is True:
            print(json.dumps(dict(status="INPUT_UPLOAD_ALREADY_COMPLETE", upload=str(upload_path))), flush=True)
            return
        archive = Path(previous["file"])
        if not archive.is_file():
            raise FileNotFoundError("partial upload archive unavailable: " + str(archive))
    else:
        work = None
        try:
            work = Path(tempfile.mkdtemp(prefix="neuronsim-readout-transfer-", dir=args.scratch))
            archive = work / "inputs.tar"
            selected = list(inputs(args.campaign, args.corrections))
            if len(selected) != 519:
                raise ValueError("complete519 delivery-tree inputs required")
            selected += [(args.qualification, "qualification.json", None), (args.comparison, "native_comparison.json", None)]
            inventory = []
            with tarfile.open(archive, "w") as target:
                for source, relative, expected in selected:
                    data = source.read_bytes()
                    digest = hashlib.sha256(data).hexdigest()
                    if expected is not None and (len(data) != expected.get("bytes", expected.get("destination_bytes"))
                                                 or digest != expected["sha256"]):
                        raise ValueError("source input differs from delivery: " + str(source))
                    info = tarfile.TarInfo(relative)
                    info.size = len(data)
                    info.mode = 0o600
                    target.addfile(info, io.BytesIO(data))
                    inventory.append(dict(path=relative, bytes=len(data), sha256=digest))
                data = (json.dumps(inventory, indent=2) + "\n").encode()
                info = tarfile.TarInfo("input_inventory.json")
                info.size = len(data)
                info.mode = 0o600
                target.addfile(info, io.BytesIO(data))
            atomic_json(args.output / "input_inventory.json", inventory)
            atomic_json(args.output / "transfer.json", dict(status="STAGED_SOURCE_BYTES", hostname=platform.node(),
                source_files=len(inventory), source_bytes=sum(row["bytes"] for row in inventory), archive=str(archive),
                source_execution="bitwise_transfer_only", scientific_backend="merlin", slurm_job_submitted=False))
            print(json.dumps(dict(staged_files=len(inventory), source_bytes=sum(row["bytes"] for row in inventory))), flush=True)
        except BaseException:
            if work is not None:
                shutil.rmtree(work)
            raise
    # CLI requests contain temporary base64 JSON files; keep them on local scratch.
    os.chdir(archive.parent)
    result = upload_archive(archive, upload_path, args.chunk_mib * 1024**2, merlin_io, args.transport)
    if result.get("complete") is not True:
        raise RuntimeError("incomplete source upload")
    print(json.dumps(dict(status="INPUT_UPLOAD_COMPLETE", bytes=result["bytes"], chunks=len(result["chunks"]),
                          sha256=result["sha256"], upload=str(upload_path))), flush=True)
    if archive.parent.name.startswith("neuronsim-readout-transfer-") and archive.parent.parent == args.scratch:
        shutil.rmtree(archive.parent)


if __name__ == "__main__":
    main()
