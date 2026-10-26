#!/usr/bin/env python3
"""Package an exact native V1 comparison input list for Merlin staging."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import tarfile
from typing import BinaryIO

CHUNK_BYTES = 8 * 1024 * 1024
INPUT_FILES = 48
NATIVE_FILES = (
    "delivery.json", "primary/report.json", "primary/native/parameters.json",
    "primary/native/cortical_state.npz", "primary/native/cortical_spikes.npz",
    "primary/native/input_events.npz",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fingerprint(path: Path) -> dict:
    require(stat.S_ISREG(path.lstat().st_mode), f"REGULAR_FILE_REQUIRED: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(CHUNK_BYTES):
            h.update(data)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def derive_job(template: dict, args: argparse.Namespace) -> dict:
    require(template["mode"] == "compare" and template["cases"] is None,
            "NATIVE_COMPARISON_TEMPLATE_REQUIRED")
    native = str(Path(template["reference"]).parent / ("run_" + args.native_job_id))
    reference_root = Path(template["output"]).parent
    circuit_root = reference_root.parent.parent / f"circuit_{args.circuit_index:03d}"
    reference_root = circuit_root / "reference_execution"
    replacements = {
        template["reference"]: native,
        template["snapshot"]: str(reference_root / "connection_snapshot" /
                                  ("run_" + args.snapshot_job_id) / "snapshot"),
        template["output"]: str(reference_root / "gpu_reference"),
        template["detector"]: str(reference_root / "detect_resources.py"),
    }
    job = dict(template)
    for key in ("reference", "snapshot", "output", "detector"):
        job[key] = replacements[template[key]]
    job["job"] = args.job_id
    job["command"] = [replacements.get(value, value) for value in template["command"]]
    job["expected_compute_capability"] = [8, 9]
    return job


def input_plan(template: dict, frozen: dict, job: dict) -> tuple[dict, dict]:
    require(len(frozen) == INPUT_FILES - 1, "FROZEN_INPUT_LIST_SIZE_DIFFERS")
    paths, expected = {}, {}
    for old, identity in frozen.items():
        if old.startswith(template["reference"] + "/"):
            new = job["reference"] + old[len(template["reference"]):]
        elif old.startswith(template["snapshot"] + "/"):
            new = job["snapshot"] + old[len(template["snapshot"]):]
        else:
            require(old == template["library"] or
                    old.startswith(template["code"] + "/"), "UNEXPECTED_SHARED_INPUT")
            new = old
            expected[new] = identity
        require(new not in paths, "DUPLICATE_INPUT_PATH")
        paths[new] = Path(new)
    paths[job["detector"]] = Path(job["detector"])
    require(len(paths) == INPUT_FILES and len(expected) == 39, "INPUT_LIST_SIZE_DIFFERS")
    require({str(Path(job["reference"]) / name) for name in NATIVE_FILES}.issubset(paths),
            "NATIVE_INPUT_LIST_DIFFERS")
    return paths, expected


def declared_identities(job: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    native = Path(job["reference"])
    delivery = load_json(native / "delivery.json")
    report = load_json(native / "primary/report.json")
    snapshot = load_json(Path(job["snapshot"]) / "report.json")
    require(delivery["status"] == "PASS_POINTNET_NATIVE_VISUAL_INPUTS_DEVELOPMENT"
            and delivery["exit_code"] == 0 and delivery["mode"] == "full"
            and delivery["job_id"] == args.native_job_id, "NATIVE_DELIVERY_DIFFERS")
    require(report["status"] == "POINTNET_NATIVE_VISUAL_INPUTS_COMPLETE_DEVELOPMENT"
            and report["exit_code"] == 0 and report["native_nest_simulation"]
            and report["native_parameters_and_initial_state_exact"],
            "COMPLETE_NATIVE_REFERENCE_REQUIRED")
    require(report["training_seeds_used"] == [], "TRAINING_SEEDS_IN_DATASET")
    require(snapshot["status"] == "PASS_COMPLETE_NATIVE_CONNECTION_SNAPSHOT"
            and snapshot["job_id"] == args.snapshot_job_id
            and snapshot["native_reference"] == str(native)
            and snapshot["parameters_and_initial_state_exact"]
            and snapshot["input_events_exact"]
            and snapshot["native_biological_time_ms"] == 0, "SNAPSHOT_REFERENCE_DIFFERS")
    batch = report["connections"]["source_batch_sha256"]
    require(all(value == batch for value in (
        snapshot["expected_source_batch_sha256"],
        snapshot["connections"]["source_batch_sha256"],
        snapshot["snapshot"]["source_batch_sha256"])), "SOURCE_BATCH_IDENTITY_DIFFERS")
    require(snapshot["units"] == report["cortical_units"] == 230924
            and snapshot["snapshot"]["rows"] ==
            report["connections"]["native_connections_to_cortex"]
            and snapshot["execution_clock"] == report["execution_clock"],
            "SNAPSHOT_NUMERICAL_IDENTITY_DIFFERS")
    declared = {str(native / row["path"]): {"bytes": row["bytes"], "sha256": row["sha256"]}
                for row in delivery["files"]}
    binding = snapshot["reference_binding"]
    require(fingerprint(native / "delivery.json")["sha256"] == binding["delivery_sha256"],
            "SNAPSHOT_DELIVERY_BINDING_DIFFERS")
    require(snapshot["native_report_sha256"] ==
            declared[str(native / "primary/report.json")]["sha256"],
            "SNAPSHOT_REPORT_BINDING_DIFFERS")
    for name, identity in binding["files"].items():
        require(declared[str(native / name)] == identity, "SNAPSHOT_FILE_BINDING_DIFFERS")
    config = native / "original_config.yaml"
    config_identity = fingerprint(config)
    require(config_identity == declared[str(config)] and
            config_identity["sha256"] == report["original_config_sha256"],
            "ORIGINAL_CONTROL_IDENTITY_DIFFERS")
    require((Path(job["snapshot"]) / "connections.h5").stat().st_size ==
            snapshot["file_bytes"], "SNAPSHOT_FILE_SIZE_DIFFERS")
    control = {
        "circuit_index": args.circuit_index,
        "training_seeds_used": [],
        "dataset_streams_are_independent_of_training_seeds": True,
        "native_kernel_rng_seed": report["kernel_rng_seed"],
        "source_batch_sha256": batch,
        "native_report_sha256": snapshot["native_report_sha256"],
        "native_delivery_sha256": binding["delivery_sha256"],
        "cortical_units": report["cortical_units"],
        "connection_rows": snapshot["snapshot"]["rows"],
        "execution_clock": report["execution_clock"],
        "original_config": {"path": str(config), **config_identity, "text": config.read_text()},
        "native_source_commands": delivery["commands"],
        "snapshot_tuple_digest_recomputed": False,
        "snapshot_tuple_digest_validation": "Declared native and snapshot identities match; "
            "the unchanged comparison consumer verifies tuple contents.",
    }
    return declared, control


class HashingReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size: int = -1) -> bytes:
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes += len(data)
        return data


def pack(archive: Path, paths: dict, expected: dict) -> dict:
    partial = archive.with_name(archive.name + ".part")
    manifest = {}
    owned_partial = False
    try:
        with partial.open("xb") as raw:
            owned_partial = True
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0,
                               compresslevel=1) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                    for name, path in sorted(paths.items()):
                        before = path.lstat()
                        require(stat.S_ISREG(before.st_mode), f"REGULAR_FILE_REQUIRED: {path}")
                        member = str(path.relative_to("/"))
                        require(".." not in Path(member).parts, "UNSAFE_ARCHIVE_PATH")
                        info = tarfile.TarInfo(member)
                        info.size = before.st_size
                        info.mode = stat.S_IMODE(before.st_mode)
                        info.mtime = 0
                        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                        with os.fdopen(fd, "rb") as stream:
                            opened = os.fstat(stream.fileno())
                            require((opened.st_dev, opened.st_ino, opened.st_size,
                                     opened.st_mtime_ns) ==
                                    (before.st_dev, before.st_ino, before.st_size,
                                     before.st_mtime_ns), "SOURCE_CHANGED_BEFORE_PACKING")
                            hashed = HashingReader(stream)
                            tar.addfile(info, hashed)
                            after = os.fstat(stream.fileno())
                        require((after.st_size, after.st_mtime_ns) ==
                                (before.st_size, before.st_mtime_ns) and
                                hashed.bytes == before.st_size, "SOURCE_CHANGED_DURING_PACKING")
                        identity = {"bytes": hashed.bytes, "sha256": hashed.digest.hexdigest()}
                        if name in expected:
                            require(identity == expected[name], f"DECLARED_INPUT_DIFFERS: {name}")
                        manifest[name] = identity
                        print(json.dumps({"packed": len(manifest), "files": len(paths),
                                          "path": name, **identity}), flush=True)
        os.link(partial, archive)
        partial.unlink()
    except BaseException:
        if owned_partial and partial.exists():
            partial.unlink()
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-job", type=Path, required=True)
    parser.add_argument("--template-manifest", type=Path, required=True)
    parser.add_argument("--circuit-index", type=int, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--native-job-id", required=True)
    parser.add_argument("--snapshot-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    require(args.archive.is_absolute() and args.archive.is_relative_to("/tmp"),
            "NODE_LOCAL_ARCHIVE_REQUIRED")
    template, frozen = load_json(args.template_job), load_json(args.template_manifest)
    job = derive_job(template, args)
    paths, expected = input_plan(template, frozen, job)
    declared, control = declared_identities(job, args)
    expected.update({name: declared[name] for name in paths if name in declared})
    targets = {name: args.output / name for name in
               ("job.json", "input_manifest.json", "preparation.json")}
    if any(path.exists() for path in targets.values()) or args.archive.exists():
        require(all(path.is_file() for path in targets.values()) and args.archive.is_file(),
                "INCOMPLETE_EXISTING_PREPARATION")
        previous = load_json(targets["preparation.json"])
        require(load_json(targets["job.json"]) == job and previous["complete"] and
                previous["source_control"] == control, "EXISTING_PREPARATION_DIFFERS")
        actual = {name: fingerprint(path) for name, path in paths.items()}
        require(load_json(targets["input_manifest.json"]) == actual and
                fingerprint(args.archive) == previous["archive"]["identity"],
                "EXISTING_INPUT_PACKAGE_DIFFERS")
        for name, identity in expected.items():
            require(actual[name] == identity, f"EXISTING_DECLARED_INPUT_DIFFERS: {name}")
        print(json.dumps({"status": "REUSED_COMPLETE_PREPARATION",
                          "preparation": str(targets["preparation.json"])}), flush=True)
        return
    total = sum(path.lstat().st_size for path in paths.values())
    require(shutil.disk_usage("/tmp").free > total + 1024**3, "INSUFFICIENT_LOCAL_DISK")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = pack(args.archive, paths, expected)
    archive_identity = fingerprint(args.archive)
    write_json(targets["job.json"], job)
    write_json(targets["input_manifest.json"], manifest)
    preparation = {
        "schema": "v1_native_comparison_input_preparation_v1",
        "status": "PASS_V1_NATIVE_COMPARISON_INPUT_PREPARATION",
        "complete": True, "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "files": len(manifest),
        "uncompressed_bytes": sum(row["bytes"] for row in manifest.values()),
        "shared_frozen_files_verified": 39,
        "source_job": args.job_id, "source_control": control,
        "original_comparison_command": job["command"],
        "template_job": {"path": str(args.template_job), **fingerprint(args.template_job)},
        "template_manifest": {"path": str(args.template_manifest),
                              **fingerprint(args.template_manifest)},
        "job": {"path": str(targets["job.json"]), **fingerprint(targets["job.json"])},
        "file_manifest": {"path": str(targets["input_manifest.json"]),
                          **fingerprint(targets["input_manifest.json"])},
        "archive": {"path": str(args.archive), "host": socket.gethostname(),
                    "identity": archive_identity, "member_paths_relative_to": "/",
                    "regular_files_only": True},
        "consumer": "Merlin input upload and execution-request preparation must consume "
                    "this archive, job and file manifest and preserve their identities.",
        "uploaded": False, "submitted": False, "scientific_qualification": False,
        "new_benchmark_episodes": 0,
    }
    write_json(targets["preparation.json"], preparation)
    print(json.dumps({"status": preparation["status"],
                      "preparation": str(targets["preparation.json"]),
                      "archive": preparation["archive"], "files": len(manifest)}), flush=True)


if __name__ == "__main__":
    main()

