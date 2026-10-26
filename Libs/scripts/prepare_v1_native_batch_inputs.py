#!/usr/bin/env python3
"""Package the frozen C0/B2 native family for a separate CUDA qualification."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socket

from prepare_v1_native_comparison_inputs import fingerprint, pack, require, write_json

SOURCE_BATCH_SHA256 = "09b8dd29bd6c30117c2ed22fdfd0b56d0b45444e405334bf4ca178911f25db17"
NATIVE_FILES = (
    "delivery.json", "original_config.yaml", "primary/report.json",
    "primary/native/parameters.json", "primary/native/cortical_state.npz",
    "primary/native/cortical_spikes.npz", "primary/native/input_events.npz",
)
DATASET_STREAMS = {
    "connectivity_seed": 100000000, "initial_state_seed": 110000002,
    "background_seed": 120000002, "stimulus_seed": 130000002,
}
PACKAGE_FILES = 114


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def prepare_plan(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    submission = read_json(args.submission_request)
    command = submission["command"]
    start = next(i for i, value in enumerate(command)
                 if value.endswith("/run_v1_glif_network.sbatch"))
    original_command = command[start:]
    require(len(original_command) == 9 and original_command[-1] == "batch_check",
            "ORIGINAL_NATIVE_BATCH_COMMAND_DIFFERS")
    script, repo, cases_name, snapshot_name, output, python, library, detector, mode = original_command
    cases_path, snapshot_path = Path(cases_name), Path(snapshot_name)
    cases_identity = fingerprint(cases_path)
    require(cases_identity["sha256"] == args.expected_cases_sha256,
            "FROZEN_CASES_HASH_DIFFERS")
    cases = read_json(cases_path)
    require(cases["schema"] == "v1_native_block_comparison_v1"
            and cases["scenario_id"] == "v1_visual"
            and cases["circuit_index"] == 0 and cases["block_index"] == 2
            and len(cases["cases"]) == 10
            and submission["cases"] == 10
            and submission["circuit_matrix_loads_expected"] == 1
            and submission["comparison_tolerances_unchanged"] is True,
            "FROZEN_NATIVE_FAMILY_DIFFERS")
    frozen = read_json(args.frozen_manifest)
    shared = {name: identity for name, identity in frozen.items()
              if name == library or "/software/population_dynamics/" in name}
    require(len(shared) == 39 and library in shared, "FROZEN_NUMERICAL_PACKAGE_DIFFERS")
    code_roots = {str(Path(name).parent) for name in shared if name.endswith(".py")}
    require(len(code_roots) == 1, "FROZEN_NUMERICAL_CODE_ROOT_DIFFERS")
    paths = {name: Path(name) for name in shared}
    expected = dict(shared)

    def add(path: Path, identity: dict | None = None) -> None:
        name = str(path)
        require(name not in paths, f"DUPLICATE_PACKAGE_INPUT: {name}")
        paths[name] = path
        if identity is not None:
            expected[name] = identity

    add(cases_path, cases_identity)
    add(args.submission_request, fingerprint(args.submission_request))
    add(Path(detector), fingerprint(Path(detector)))
    snapshot_report_path = snapshot_path / "report.json"
    snapshot_identity = fingerprint(snapshot_report_path)
    snapshot = read_json(snapshot_report_path)
    require(snapshot["status"] == "PASS_COMPLETE_NATIVE_CONNECTION_SNAPSHOT"
            and str(snapshot["job_id"]) == "622914"
            and snapshot["parameters_and_initial_state_exact"] is True
            and snapshot["input_events_exact"] is True
            and snapshot["native_biological_time_ms"] == 0
            and snapshot["native_simulation_calls"] == 0
            and snapshot["units"] == 230924
            and snapshot["snapshot"]["rows"] == 82246780,
            "FROZEN_SNAPSHOT_METADATA_DIFFERS")
    require(all(value == SOURCE_BATCH_SHA256 for value in (
        snapshot["expected_source_batch_sha256"],
        snapshot["connections"]["source_batch_sha256"],
        snapshot["snapshot"]["source_batch_sha256"])), "SNAPSHOT_SOURCE_BATCH_DIFFERS")
    require(snapshot["connections_file"] == "connections.h5", "SNAPSHOT_FILENAME_DIFFERS")
    connections = snapshot_path / snapshot["connections_file"]
    require(connections.stat().st_size == snapshot["file_bytes"], "SNAPSHOT_BYTES_DIFFER")
    add(snapshot_report_path, snapshot_identity)
    add(connections)
    family = []
    native_directories = set()
    parameter_hashes, config_hashes = set(), set()
    for index, case in enumerate(cases["cases"]):
        identity = case["identity"]
        require(identity["condition_index"] == index
                and identity["scenario_id"] == "v1_visual"
                and identity["circuit_index"] == 0 and identity["block_index"] == 2
                and identity["split"] == "train"
                and identity["sample_unit"] == "complete_mechanistic_episode"
                and identity["episode_id"] ==
                f"v1_visual_circuit_000_block_002_condition_{index:02d}"
                and all(identity[name] == value for name, value in DATASET_STREAMS.items()),
                "FROZEN_CASE_IDENTITY_DIFFERS")
        native = Path(case["native_archive"])
        require(native not in native_directories, "DUPLICATE_NATIVE_ARCHIVE")
        native_directories.add(native)
        delivery_path, report_path = native / "delivery.json", native / "primary/report.json"
        delivery_identity = fingerprint(delivery_path)
        delivery, report = read_json(delivery_path), read_json(report_path)
        require(delivery["status"] == "PASS_POINTNET_NATIVE_VISUAL_INPUTS_DEVELOPMENT"
                and delivery["exit_code"] == 0 and delivery["mode"] == "full"
                and str(delivery["job_id"]) == str(case["native_job_id"])
                and delivery["spike_inputs_reference"] == case["spike_inputs_reference"],
                "NATIVE_DELIVERY_IDENTITY_DIFFERS")
        require(report["status"] == "POINTNET_NATIVE_VISUAL_INPUTS_COMPLETE_DEVELOPMENT"
                and report["exit_code"] == 0 and report["native_nest_simulation"] is True
                and report["native_parameters_and_initial_state_exact"] is True
                and report["training_seeds_used"] == []
                and report["cortical_units"] == snapshot["units"]
                and report["execution_clock"] == snapshot["execution_clock"]
                and report["connections"]["source_batch_sha256"] == SOURCE_BATCH_SHA256
                and report["connections"]["native_connections_to_cortex"] ==
                snapshot["snapshot"]["rows"], "NATIVE_FAMILY_NUMERICAL_IDENTITY_DIFFERS")
        declared = {row["path"]: {"bytes": row["bytes"], "sha256": row["sha256"]}
                    for row in delivery["files"]}
        require(declared["primary/report.json"]["sha256"] == case["native_report_sha256"]
                and declared["original_config.yaml"]["sha256"] ==
                report["original_config_sha256"], "NATIVE_REPORT_OR_CONTROL_IDENTITY_DIFFERS")
        if index == 0:
            require(snapshot["native_reference"] == str(native)
                    and snapshot["native_report_sha256"] == case["native_report_sha256"]
                    and snapshot["reference_binding"]["delivery_sha256"] ==
                    delivery_identity["sha256"], "SNAPSHOT_NATIVE_BINDING_DIFFERS")
            for relative, binding in snapshot["reference_binding"]["files"].items():
                require(declared[relative] == binding, "SNAPSHOT_FILE_BINDING_DIFFERS")
        for relative in NATIVE_FILES:
            add(native / relative, delivery_identity if relative == "delivery.json"
                else declared[relative])
        parameter_hashes.add(declared["primary/native/parameters.json"]["sha256"])
        config_hashes.add(declared["original_config.yaml"]["sha256"])
        family.append({
            "identity": identity, "native_archive": str(native),
            "native_job_id": case["native_job_id"],
            "spike_inputs_reference": case["spike_inputs_reference"],
            "native_report_sha256": case["native_report_sha256"],
            "native_delivery_sha256": delivery_identity["sha256"],
            "original_config_sha256": report["original_config_sha256"],
            "training_seeds_used": [],
        })
    require(len(parameter_hashes) == len(config_hashes) == 1,
            "FROZEN_FAMILY_PARAMETER_OR_CONFIG_DIFFERS")
    require(len(paths) == PACKAGE_FILES, "NATIVE_BATCH_PACKAGE_SIZE_DIFFERS")
    job = {
        "job": "623080", "command": original_command, "repo": repo,
        "reference": cases_name, "cases": cases_name, "snapshot": snapshot_name,
        "output": output, "python": python, "library": library, "detector": detector,
        "mode": mode, "code": next(iter(code_roots)),
    }
    control = {
        "cases": {"path": cases_name, **cases_identity},
        "original_submission": {"path": str(args.submission_request),
                                **expected[str(args.submission_request)], "command": command},
        "source_batch_sha256": SOURCE_BATCH_SHA256,
        "snapshot_report": {"path": str(snapshot_report_path), **snapshot_identity},
        "snapshot_tuple_file": str(connections),
        "snapshot_tuple_digest_recomputed": False,
        "snapshot_tuple_validation": "Declared family and snapshot identities match; "
            "the unchanged native comparison validates tuple contents.",
        "cortical_units": snapshot["units"], "connection_rows": snapshot["snapshot"]["rows"],
        "execution_clock": snapshot["execution_clock"], "family": family,
        "dataset_streams": DATASET_STREAMS, "training_seeds_used": [],
        "dataset_streams_are_independent_of_training_seeds": True,
        "shared_frozen_files": shared,
    }
    return job, paths, expected, control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-request", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--expected-cases-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    require(args.archive.is_absolute() and args.archive.is_relative_to("/tmp"),
            "NODE_LOCAL_ARCHIVE_REQUIRED")
    require(len(args.expected_cases_sha256) == 64, "FULL_CASES_HASH_REQUIRED")
    job, paths, expected, control = prepare_plan(args)
    targets = {name: args.output / name for name in
               ("job.json", "input_manifest.json", "preparation.json")}
    if any(path.exists() for path in targets.values()) or args.archive.exists():
        require(all(path.is_file() for path in targets.values()) and args.archive.is_file(),
                "INCOMPLETE_EXISTING_PREPARATION")
        previous = read_json(targets["preparation.json"])
        require(previous["complete"] and previous["source_control"] == control
                and read_json(targets["job.json"]) == job, "EXISTING_PREPARATION_DIFFERS")
        manifest = {name: fingerprint(path) for name, path in paths.items()}
        require(manifest == read_json(targets["input_manifest.json"])
                and fingerprint(args.archive) == previous["archive"]["identity"],
                "EXISTING_PACKAGE_DIFFERS")
        require(all(manifest[name] == value for name, value in expected.items()),
                "EXISTING_DECLARED_HASH_DIFFERS")
        print(json.dumps({"status": "REUSED_COMPLETE_PREPARATION",
                          "preparation": str(targets["preparation.json"])}), flush=True)
        return
    require(shutil.disk_usage("/tmp").free >
            sum(path.stat().st_size for path in paths.values()) + 1024**3,
            "INSUFFICIENT_NODE_LOCAL_DISK")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = pack(args.archive, paths, expected)
    archive_identity = fingerprint(args.archive)
    write_json(targets["job.json"], job)
    write_json(targets["input_manifest.json"], manifest)
    preparation = {
        "schema": "v1_frozen_native_batch_input_preparation_v1",
        "status": "PASS_V1_FROZEN_NATIVE_BATCH_INPUT_PREPARATION",
        "complete": True, "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "source_job": "623080",
        "files": len(manifest), "conditions": len(control["family"]),
        "uncompressed_bytes": sum(value["bytes"] for value in manifest.values()),
        "shared_frozen_files_verified": 39,
        "declared_file_hashes_verified": len(expected), "source_control": control,
        "job": {"path": str(targets["job.json"]), **fingerprint(targets["job.json"])},
        "file_manifest": {"path": str(targets["input_manifest.json"]),
                          **fingerprint(targets["input_manifest.json"])},
        "archive": {"path": str(args.archive), "host": socket.gethostname(),
                    "identity": archive_identity, "member_paths_relative_to": "/",
                    "regular_files_only": True},
        "consumer": "The native-batch upload and Merlin request must consume this exact "
                    "archive, input manifest and original job control. The candidate sm80 "
                    "library and qualification request are prepared separately.",
        "uploaded": False, "submitted": False, "scientific_qualification": False,
        "new_benchmark_episodes": 0, "native_source_execution_repeated": False,
    }
    write_json(targets["preparation.json"], preparation)
    print(json.dumps({"status": preparation["status"],
                      "preparation": str(targets["preparation.json"]),
                      "archive": preparation["archive"], "files": len(manifest),
                      "conditions": preparation["conditions"]}), flush=True)


if __name__ == "__main__":
    main()

