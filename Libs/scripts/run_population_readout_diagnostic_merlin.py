"""Stage authenticated Merlin inputs, run the frozen readout diagnostic, return evidence."""
from __future__ import annotations

import argparse
import base64
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import traceback

CHUNK_BYTES = 1024 * 1024
RUNTIME_KEYS = ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "RH2_CHECKPOINT_HDFS")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fingerprint(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def verify(path, expected):
    actual = fingerprint(path)
    if actual != {key: expected[key] for key in actual}:
        raise ValueError(f"Content identity mismatch: {path}")


def local_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe staged path: {name}")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Staged path escapes destination: {name}")
    return target


def hdfs(*args, stdout=subprocess.PIPE):
    result = subprocess.run(["hdfs", "dfs", *map(str, args)], stdout=stdout, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"hdfs dfs {args}: {result.stderr.decode(errors='replace')}")


def fetch(uri, path):
    with path.open("wb") as stream:
        hdfs("-cat", uri, stdout=stream)


def stage(source_upload, input_upload, work):
    for spec in (source_upload, input_upload):
        if spec.get("complete") is not True:
            raise ValueError("Staging requires complete upload manifests")
    source, inputs = work / "source", work / "inputs"
    source.mkdir()
    inputs.mkdir()
    payload = work / "source.json"
    fetch(source_upload["hdfs_path"], payload)
    verify(payload, source_upload)
    source_map = json.loads(payload.read_text())
    for name, entry in source_map.items():
        target = local_path(source, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(base64.b64decode(entry["content"], validate=True))
        verify(target, entry)
    archive, chunk = work / "inputs.tar", work / "input.chunk"
    with archive.open("xb") as assembled:
        for index, entry in enumerate(input_upload["chunks"]):
            if entry["index"] != index:
                raise ValueError("Input chunks must have contiguous ordered indices")
            fetch(entry["hdfs_path"], chunk)
            verify(chunk, entry)
            with chunk.open("rb") as stream:
                shutil.copyfileobj(stream, assembled, CHUNK_BYTES)
    verify(archive, input_upload)
    chunk.unlink(missing_ok=True)
    seen = set()
    with tarfile.open(archive, "r:*") as packaged:
        for member in packaged:
            if not member.isfile() or member.name in seen:
                raise ValueError(f"Nonregular or duplicate archive member: {member.name}")
            target = local_path(inputs, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with packaged.extractfile(member) as stream, target.open("xb") as destination:
                shutil.copyfileobj(stream, destination, CHUNK_BYTES)
            seen.add(member.name)
    inventory = json.loads((inputs / "input_inventory.json").read_text())
    expected = {entry["path"] for entry in inventory}
    if len(expected) != len(inventory) or seen != expected | {"input_inventory.json"}:
        raise ValueError("Archive members do not match the unique input inventory")
    for entry in inventory:
        verify(local_path(inputs, entry["path"]), entry)
    return {"source_files": len(source_map), "input_files": len(inventory),
            "input_bytes": sum(entry["bytes"] for entry in inventory)}


def persist(work, output, checkpoint, exit_code):
    destination = checkpoint.rstrip("/") + "/readout_diagnostic"
    hdfs("-mkdir", "-p", destination)
    archive = work / "results.tar.gz"
    with tarfile.open(archive, "w:gz") as packaged:
        packaged.add(output, arcname="results")
    hdfs("-put", archive, destination + "/results.tar.gz")
    for name in ("analysis.json", "execution.json"):
        if (output / name).is_file():
            hdfs("-put", output / name, destination + "/" + name)
    chunks, encoded = [], work / "result_chunk.json"
    with archive.open("rb") as stream:
        while block := stream.read(CHUNK_BYTES):
            name = f"chunk_{len(chunks):06d}.json"
            write_json(encoded, {"content_base64": base64.b64encode(block).decode("ascii")})
            hdfs("-put", encoded, destination + "/" + name)
            chunks.append({"name": name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
    identity = fingerprint(archive)
    receipt = {"persisted": True, "exit_code": exit_code, "checkpoint_path": destination,
               "result_bytes": identity["bytes"], "result_sha256": identity["sha256"], "chunks": chunks}
    write_json(work / "receipt.json", receipt)
    hdfs("-put", work / "receipt.json", destination + "/receipt.json")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-upload", type=Path, required=True)
    parser.add_argument("--input-upload", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output = work / "results"
    if output.exists():
        raise FileExistsError("Merlin diagnostic requires a fresh results directory")
    execution = {"started_at": datetime.now(timezone.utc).isoformat(), "exit_code": 1}
    with (work / "trace.log").open("w") as trace, contextlib.redirect_stdout(trace), contextlib.redirect_stderr(trace):
        try:
            environment = {key: os.environ.get(key, "") for key in RUNTIME_KEYS}
            cgroup_paths = ("cpu.max", "memory.max", "cpuset.cpus.effective", "cpu/cpu.cfs_quota_us",
                            "cpu/cpu.cfs_period_us", "memory/memory.limit_in_bytes")
            runtime = {"execution_backend": "merlin", "environment": environment, "hostname": platform.node(),
                       "cpu_affinity": sorted(os.sched_getaffinity(0)), "cpu_threads": args.cpu_threads,
                       "cgroup": {name: (Path("/sys/fs/cgroup") / name).read_text().strip()
                                  for name in cgroup_paths if (Path("/sys/fs/cgroup") / name).is_file()},
                       "scratch": {"path": str(work), **shutil.disk_usage(work)._asdict(),
                                   "filesystem": subprocess.check_output(["stat", "-f", "-c", "%T", str(work)], text=True).strip()}}
            write_json(work / "runtime.json", runtime)
            execution["runtime"] = runtime
            if not all(environment.values()) or args.cpu_threads < 1:
                raise ValueError("Actual Merlin trial identity and positive CPU threads are required")
            source_upload, input_upload = (json.loads(path.read_text()) for path in (args.source_upload, args.input_upload))
            required_free = 2 * int(input_upload["bytes"]) + 1024**3
            runtime["scratch"]["required_free_bytes"] = required_free
            if (platform.node().split(".")[0] == "mgmt2"
                    or runtime["scratch"]["filesystem"] not in ("overlayfs", "overlay", "xfs", "ext2/ext3", "ext4", "tmpfs")
                    or runtime["scratch"]["free"] < required_free):
                raise RuntimeError("Local scratch must hold the input archive, extracted files and result buffer")
            write_json(work / "runtime.json", runtime)
            execution.update(source_upload=source_upload, input_upload=input_upload, phase="staging")
            execution["staging"] = stage(source_upload, input_upload, work)
            manifest = {"parent_campaign": "inputs/parent", "corrections": "inputs/corrected", "cpu_threads": args.cpu_threads}
            for role, path in (("qualification", work / "inputs/qualification.json"),
                               ("native_comparison", work / "inputs/native_comparison.json"), ("runtime_receipt", work / "runtime.json")):
                manifest[role] = {"path": str(path.relative_to(work)), **fingerprint(path)}
            write_json(work / "manifest.json", manifest)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ[name] = str(args.cpu_threads)
            sys.path.insert(0, str(work / "source"))
            from population_dynamics.population_readout_diagnostic import analyze
            execution["phase"] = "diagnostic"
            analyze(work / "manifest.json", output)
            execution.update(exit_code=0, phase="complete")
        except Exception:
            execution["error"] = traceback.format_exc()
            traceback.print_exc()
    output.mkdir(exist_ok=True)
    execution["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "execution.json", execution)
    for name in ("runtime.json", "manifest.json", "trace.log"):
        if (work / name).is_file():
            shutil.copy2(work / name, output / name)
    try:
        receipt = persist(work, output, os.environ["RH2_CHECKPOINT_HDFS"], execution["exit_code"])
        print(json.dumps(receipt), flush=True)
    except Exception:
        with (output / "trace.log").open("a") as trace:
            traceback.print_exc(file=trace)
        raise
    return execution["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
