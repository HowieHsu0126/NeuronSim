"""Restore transport publication from complete Merlin PFC source bundles.

Run only in a Merlin CPU allocation. Source data, execution identity, and failed
publication evidence remain unchanged. Scientific admission is a separate step.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import subprocess
import tarfile
import tempfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return digest(stream)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_identity(record, task):
    expected = {key: task[key] for key in ("index", "xie_job", "run_id", "trial_id")}
    expected.update(
        execution_backend="merlin",
        compatibility_execution_id=f"merlin_{task['trial_id']}_pfc_{task['index']:04d}",
    )
    for key, value in expected.items():
        require(type(record.get(key)) is type(value) and record[key] == value,
                "SOURCE_IDENTITY_MISMATCH:" + key)
    require(isinstance(record.get("hostname"), str) and bool(record["hostname"]),
            "SOURCE_HOST_REQUIRED")


def verify_bundle(bundle, manifest, terminal, task):
    """Check every archive member and bind the external terminal to its copy."""
    check_identity(terminal, task)
    require(terminal.get("status") == "COMPLETE"
            and type(terminal.get("exit_code")) is int and terminal["exit_code"] == 0,
            "SOURCE_EXECUTION_NOT_COMPLETE")
    require(isinstance(manifest, dict) and bool(manifest), "EMPTY_SOURCE_MANIFEST")
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)), "DUPLICATE_ARCHIVE_MEMBER")
        require(set(names) == set(manifest) | {"merlin_return_manifest.json"},
                "ARCHIVE_MEMBERSHIP_MISMATCH")
        embedded = None
        embedded_terminal = None
        request = None
        for member in members:
            name = member.name
            path = PurePosixPath(name)
            require(member.isfile() and not path.is_absolute()
                    and ".." not in path.parts and str(path) == name,
                    "UNSAFE_ARCHIVE_MEMBER:" + name)
            with archive.extractfile(member) as stream:
                if name == "merlin_return_manifest.json":
                    embedded = json.load(stream)
                    continue
                row = manifest[name]
                require(member.size == row["bytes"] and digest(stream) == row["sha256"],
                        "ARCHIVE_FILE_CHANGED:" + name)
            if name.endswith(f"/tasks/{task['index']}/terminal.json"):
                require(embedded_terminal is None, "DUPLICATE_TASK_TERMINAL")
                with archive.extractfile(member) as stream:
                    embedded_terminal = json.load(stream)
            if name.endswith(f"/tasks/{task['index']}/request.json"):
                require(request is None, "DUPLICATE_TASK_REQUEST")
                with archive.extractfile(member) as stream:
                    request = json.load(stream)
        require(embedded == manifest, "EMBEDDED_MANIFEST_MISMATCH")
        require(embedded_terminal == terminal, "SOURCE_TERMINAL_BINDING_MISMATCH")
        require(request is not None, "SOURCE_REQUEST_MISSING")
        check_identity(request, task)
        require(all(terminal.get(key) == value for key, value in request.items()
                    if key != "status"), "SOURCE_REQUEST_BINDING_MISMATCH")
    # TAR can stop at its end marker before reading the gzip footer.
    with gzip.open(bundle, "rb") as stream:
        for _ in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            pass
    return {"bytes": bundle.stat().st_size, "sha256": file_digest(bundle),
            "verified_files": len(manifest)}


def hdfs(*arguments):
    return subprocess.run(["hdfs", "dfs", *map(str, arguments)],
                          check=True, text=True, capture_output=True)


def get_optional(uri, destination):
    result = subprocess.run(["hdfs", "dfs", "-get", uri, str(destination)],
                            text=True, capture_output=True)
    if result.returncode == 0:
        return json.loads(destination.read_text())
    # Only an explicit missing-file result permits recovery without a receipt.
    if not destination.exists() and any(text in result.stderr for text in
            ("No such file or directory", "File does not exist", "FileNotFoundException")):
        return None
    raise RuntimeError("HDFS_RECEIPT_READ_FAILED:" + result.stderr[-2000:])


def embedded_metadata(bundle, task):
    """Read original metadata only from safe, unique archive members."""
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)), "DUPLICATE_ARCHIVE_MEMBER")
        for member in members:
            path = PurePosixPath(member.name)
            require(member.isfile() and not path.is_absolute()
                    and ".." not in path.parts and str(path) == member.name,
                    "UNSAFE_ARCHIVE_MEMBER:" + member.name)
        result = {}
        for role, suffix in (("manifest", "merlin_return_manifest.json"),
                             ("terminal", f"/tasks/{task['index']}/terminal.json"),
                             ("request", f"/tasks/{task['index']}/request.json")):
            selected = [member for member in members if
                        (member.name == suffix if role == "manifest" else member.name.endswith(suffix))]
            require(bool(selected), "SOURCE_" + role.upper() + "_MISSING")
            require(len(selected) == 1, "DUPLICATE_TASK_" + role.upper())
            with archive.extractfile(selected[0]) as stream:
                result[role] = json.load(stream)
        return result


def verify_existing_transport(receipt, work):
    """Authenticate remote chunks before reusing a publication receipt.

    Explicitly missing or corrupt objects permit publication from the already
    verified source bundle. An uncertain HDFS read remains an observation error.
    """
    source = receipt["checkpoint_path"].rstrip("/")
    total = hashlib.sha256()
    names = set()
    failures = []
    for index, row in enumerate(receipt["chunks"]):
        name = row.get("name")
        require(isinstance(name, str) and PurePosixPath(name).name == name
                and name not in (".", "..") and name not in names,
                "EXISTING_RECEIPT_UNSAFE_CHUNK_NAME")
        require(type(row.get("bytes")) is int and row["bytes"] > 0
                and isinstance(row.get("sha256"), str)
                and len(row["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in row["sha256"]),
                "EXISTING_RECEIPT_CHUNK_BINDING_REQUIRED")
        names.add(name)
        path = work / f"existing_transport_{index:05d}.json"
        try:
            try:
                value = get_optional(source + "/" + name, path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                failures.append(dict(name=name, reason="INVALID_CHUNK_JSON"))
                continue
            if value is None:
                failures.append(dict(name=name, reason="MISSING_CHUNK"))
                continue
            try:
                data = base64.b64decode(value["content_base64"], validate=True)
            except (KeyError, TypeError, ValueError):
                failures.append(dict(name=name, reason="INVALID_CHUNK_ENCODING"))
                continue
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                failures.append(dict(name=name, reason="CHUNK_BYTES_OR_SHA256_MISMATCH"))
                continue
            total.update(data)
        finally:
            path.unlink(missing_ok=True)
    if not failures and total.hexdigest() != receipt["result_sha256"]:
        failures.append(dict(reason="ASSEMBLED_BUNDLE_SHA256_MISMATCH"))
    return dict(status="PASS" if not failures else "FAILED", failures=failures,
                chunks_checked=len(names))


def recover_task(task, work, publisher):
    source = task["checkpoint_path"].rstrip("/")
    source_bundle_uri = task.get("source_bundle_uri", source + "/result.tar.gz")
    bundle = work / "result.tar.gz"
    hdfs("-get", source_bundle_uri, bundle)
    manifest = get_optional(source + "/manifest.json", work / "manifest.json")
    terminal = get_optional(source + "/terminal.json", work / "terminal.json")
    origin = dict(manifest="external_hdfs", terminal="external_hdfs", request="embedded_bundle")
    missing = {role for role, value in (("manifest", manifest), ("terminal", terminal)) if value is None}
    if missing:
        embedded = embedded_metadata(bundle, task)
        request_path = work / "source_request.json"
        hdfs("-get", source + "/request.json", request_path)
        request = json.loads(request_path.read_text())
        check_identity(request, task)
        require(all(embedded["request"].get(key) == value for key, value in request.items()
                    if key != "status"), "SOURCE_EXTERNAL_REQUEST_BINDING_MISMATCH")
        if manifest is None:
            manifest = embedded["manifest"]
            origin["manifest"] = "embedded_bundle"
        if terminal is None:
            terminal = embedded["terminal"]
            origin["terminal"] = "embedded_bundle"
        require(all(terminal.get(key) == value for key, value in request.items()
                    if key != "status"), "SOURCE_EXTERNAL_REQUEST_BINDING_MISMATCH")
        origin["request"] = "external_hdfs"
    verified = verify_bundle(bundle, manifest, terminal, task)
    for role in missing:
        write_json(work / (role + ".json"), manifest if role == "manifest" else terminal)
    original = get_optional(source + "/receipt.json", work / "source_receipt.json")
    transport_validation = None
    if original is not None:
        check_identity(original, task)
        require(original.get("status") == "COMPLETE"
                and type(original.get("exit_code")) is int and original["exit_code"] == 0
                and original.get("persisted") is True
                and original.get("result_sha256") == verified["sha256"]
                and original.get("result_bytes") == verified["bytes"],
                "EXISTING_RECEIPT_BINDING_MISMATCH")
        require(original.get("checkpoint_path") == source
                and bool(original.get("chunks"))
                and sum(row["bytes"] for row in original["chunks"]) == verified["bytes"],
                "EXISTING_RECEIPT_TRANSPORT_MISMATCH")
        transport_validation = verify_existing_transport(original, work)
        if transport_validation["status"] == "PASS":
            return dict(index=task["index"], status="EXISTING_PUBLICATION_REUSED",
                        receipt_uri=source + "/receipt.json", validation=verified,
                        source_metadata_origin=origin,
                        source_transport_validation=transport_validation)

    target = publisher["checkpoint_path"] + f"/pfc_publication/task_{task['index']:04d}"
    hdfs("-mkdir", target)
    # Existing readout consumers retrieve checkpoint_path/result.tar.gz.
    hdfs("-put", bundle, target + "/result.tar.gz")
    chunks = []
    with bundle.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            name = f"chunk_{len(chunks):05d}.json"
            path = work / name
            write_json(path, dict(content_base64=base64.b64encode(data).decode()))
            hdfs("-put", path, target + "/" + name)
            chunks.append(dict(name=name, bytes=len(data),
                               sha256=hashlib.sha256(data).hexdigest()))
            path.unlink()
    for name in ("manifest.json", "terminal.json"):
        hdfs("-put", work / name, target + "/" + name)
    publication = dict(publisher, source_checkpoint_path=source,
        source_bundle_uri=source_bundle_uri,
        published_bundle_uri=target + "/result.tar.gz",
        source_terminal_uri=(target if "terminal" in missing else source) + "/terminal.json",
        source_manifest_uri=(target if "manifest" in missing else source) + "/manifest.json",
        source_metadata_origin=origin,
        source_request_uri=source + "/request.json" if missing else None,
        original_receipt_present=original is not None, bundle_validation=verified,
        source_transport_validation=transport_validation,
        source_simulations=0, scientific_admission_granted=False)
    write_json(work / "publication.json", publication)
    hdfs("-put", work / "publication.json", target + "/publication.json")
    receipt = dict(terminal, checkpoint_path=target,
        result_sha256=verified["sha256"], result_bytes=verified["bytes"],
        chunks=chunks, persisted=True, publication_recovery=publication)
    write_json(work / "receipt.json", receipt)
    hdfs("-put", work / "receipt.json", target + "/receipt.json")
    return dict(index=task["index"], status="PUBLICATION_RECOVERED",
                receipt_uri=target + "/receipt.json", validation=verified,
                source_metadata_origin=origin)


def run(config, work):
    run_id, trial_id = os.environ.get("RH2_JOB_RUN_ID"), os.environ.get("ARNOLD_TRIAL_ID")
    require(bool(run_id) and bool(trial_id)
            and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0"
            and os.environ.get("ARNOLD_MONITOR_ROLE") == "head"
            and not os.environ.get("CUDA_VISIBLE_DEVICES"), "MERLIN_CPU_ALLOCATION_REQUIRED")
    checkpoint = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/")
    require(all(not task["checkpoint_path"].startswith(checkpoint + "/")
                for task in config["tasks"]), "SOURCE_AND_PUBLICATION_ROOTS_OVERLAP")
    require(len({task["index"] for task in config["tasks"]}) == len(config["tasks"]),
            "DUPLICATE_TASK_INDEX")
    filesystem = subprocess.check_output(["stat", "-f", "-c", "%T", str(work)], text=True).strip()
    require(filesystem not in {"nfs", "nfs4", "cifs", "smb2"}, "LOCAL_SCRATCH_REQUIRED")
    require(shutil.disk_usage(work).free >= 5 * 1024**3, "LOCAL_SCRATCH_SPACE_REQUIRED")
    publisher = dict(run_id=run_id, trial_id=trial_id, hostname=socket.gethostname(),
        checkpoint_path=checkpoint, recorded_utc=datetime.now(timezone.utc).isoformat(),
        implementation_sha256=file_digest(Path(__file__)), scratch_filesystem=filesystem,
        available_scratch_bytes=shutil.disk_usage(work).free,
        allocated_cpus=os.environ.get("ARNOLD_HEAD_CPU"), allocated_gpus=0)
    hdfs("-mkdir", "-p", checkpoint + "/pfc_publication")
    report = dict(status="RECOVERING", publisher=publisher, tasks=[],
                  source_simulations=0, production_episodes=0,
                  scientific_admission_granted=False)
    for task in config["tasks"]:
        with tempfile.TemporaryDirectory(prefix=f"task_{task['index']:04d}_", dir=work) as directory:
            try:
                result = recover_task(task, Path(directory), publisher)
            except Exception as error:
                result = dict(index=task["index"], status="FAILED_PUBLICATION_RECOVERY",
                              error_type=type(error).__name__, error=str(error))
                if isinstance(error, subprocess.CalledProcessError):
                    result["stderr"] = error.stderr[-4000:]
        report["tasks"].append(result)
        print(json.dumps(result), flush=True)
    report["status"] = ("PASS" if all(row["status"] != "FAILED_PUBLICATION_RECOVERY"
                                    for row in report["tasks"]) else "FAILED")
    write_json(work / "publication_report.json", report)
    hdfs("-put", work / "publication_report.json", checkpoint + "/pfc_publication/report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    report = run(json.loads(args.config.read_text()), args.work)
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
