#!/usr/bin/env python3
"""Stage the existing CA3 persistence runtime, source inputs and prefix evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile

import yaml

from stage_v1_merlin_block_inputs import curl_upload_chunk, module_at


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--merlin-skill", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--pack-only", action="store_true")
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] == "mgmt2":
        raise SystemExit("CA3 input staging requires the compute or transfer host")
    args.work.mkdir(parents=True, exist_ok=True)
    args.record.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(["stat", "-f", "-c", "%T", str(args.work)], text=True).strip()
    if filesystem in {"nfs", "nfs4", "cifs", "smb2"}:
        raise ValueError("Transfer scratch must use a local filesystem")
    benchmark = args.repo / "Output/neural_population_dynamics/benchmark"
    prior_path = benchmark / "ca3_cpu06_full_replay_20261017/preparation.json"
    prior = json.loads(prior_path.read_text())
    prepared = benchmark / "ca3_persistence_preparation_20261011"
    runtime = prepared / "ca3_runtime"
    build_path = runtime / "build_report.json"
    built = json.loads(build_path.read_text())
    old_build_path = Path(prior["prepared"]) / "ca3_runtime/build_report.json"
    old_built = json.loads(old_build_path.read_text())
    config = args.repo / "Libs/configs/neural_population_ca3_persistence_development.yaml"
    old = yaml.safe_load(Path(prior["config"]).read_text())["simulation_campaign"]["scenarios"]["ca3_pattern_completion"]
    current = yaml.safe_load(config.read_text())["simulation_campaign"]["scenarios"]["ca3_pattern_completion"]
    expected = json.loads(json.dumps(old))
    expected["task_protocol"]["retrieval_duration_ms"] = 1210
    expected["task_protocol"]["condition_duration_ticks"] = [1210] * 10
    if expected != current:
        raise ValueError("CA3 configuration differs beyond the authorized duration extension")
    if not (old_built["control_program"]["duration_ms"] == 1001
            and built["status"] == "BUILT_CA3_BENCHMARK_NATIVE_CONTROLS"
            and built["control_program"]["duration_ms"] == 1210
            and built["backend_libraries"] == old_built["backend_libraries"]
            and built["source_headers"] == old_built["source_headers"]
            and built["control_program"]["base_main_sha256"] == old_built["control_program"]["base_main_sha256"]):
        raise ValueError("Existing CA3 persistence runtime identity differs")
    source = Path(prior["source_repo"])
    formation = args.repo / "Output/neural_population_dynamics/multicircuit/ca3_source_reproduction/coupled_extension/formation_589360"
    analysis = formation.with_name("formation_analysis_589369")
    reference = Path(prior["reference"])
    files = {Path(name) for name in prior["source_file_bindings"]}
    files.update([prior_path, prior_path.parent / "run_624517/report.json", config,
                  source / "Libs/__init__.py", source / "Libs/population_dynamics/__init__.py",
                  build_path, old_build_path, runtime / "bin/ca3_benchmark", runtime / "trainPat.csv",
                  prepared / "host_current_replay", prepared / "host_current_replay_build.json",
                  prepared / "tests.xml"])
    files.update(runtime / "lib" / name for name in built["backend_libraries"])
    files.update(formation / name for name in ("native_report.json", "observations/recording.json", "native_run/network65p.dat"))
    files.update(analysis / name for name in ("formation_replay.json", "normalization.json", "normalization_replay.json", "network65p_assem550_divnorm.dat"))
    files.update(reference / name for name in ("report.json", "native_observation.bin", "native_run/trainPat.csv", "ordered_cue_commands.npz", "dataset/dataset.json"))
    catalog = json.loads((reference / "dataset/dataset.json").read_text())
    for row in [*catalog["circuits"].values(), *catalog["shards"]]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid CA3 dataset member path")
        files.add(reference / "dataset" / relative)
    if any(not path.is_file() for path in files):
        raise FileNotFoundError([str(path) for path in files if not path.is_file()])
    identity = dict(prior_preparation=str(prior_path), source_repo=str(source), prepared=str(prepared),
                    config=str(config), formation=str(formation), analysis=str(analysis),
                    reference=str(reference), current_replay_engine=str(prepared / "host_current_replay"),
                    original_pending_head_job="625044", required_memory_gib=128, requested_memory_gib=160,
                    runtime_build_sha256=digest(build_path), runtime_binary_sha256=built["binary_sha256"],
                    runtime_libraries=built["backend_libraries"], runtime_duration_ms=1210,
                    reference_duration_ms=1001, units=168106, prefix_tolerance=0,
                    source_file_bindings=prior["source_file_bindings"], production_episodes=0)
    (args.record / "execution_inputs.json").write_text(json.dumps(identity, indent=2) + "\n")
    total = sum(path.stat().st_size for path in files)
    inventory = dict(status="INPUTS_LOCATED", hostname=socket.gethostname(),
                     observed_utc=datetime.now(timezone.utc).isoformat(), files=len(files), bytes=total,
                     scratch_filesystem=filesystem, scratch_free_bytes=shutil.disk_usage(args.work).free,
                     original_pending_runtime_duration_ms=1001, selected_existing_runtime_duration_ms=1210,
                     new_compilation=False, new_source_simulation=False, production_episodes=0)
    (args.record / "input_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory), flush=True)
    if args.inventory_only:
        return
    if inventory["scratch_free_bytes"] < total * 2 + 2**30:
        raise ValueError("Insufficient local CA3 transfer storage")
    manifest_path = args.record / "input_files.json"
    archive_path = args.work / "ca3_inputs.tar.gz"
    if not (manifest_path.exists() and archive_path.exists()):
        manifest = {}
        expected_hashes = {Path(name): value for name, value in prior["source_file_bindings"].items()}
        expected_hashes[runtime / "bin/ca3_benchmark"] = built["binary_sha256"]
        expected_hashes.update({runtime / "lib" / name: value for name, value in built["backend_libraries"].items()})
        partial = archive_path.with_suffix(".partial")
        with tarfile.open(partial, "w:gz", compresslevel=1, dereference=True) as bundle:
            for path in sorted(files):
                row = dict(bytes=path.stat().st_size, sha256=digest(path))
                if path in expected_hashes and row["sha256"] != expected_hashes[path]:
                    raise ValueError("CA3 frozen input changed: " + str(path))
                manifest[str(path)] = row
                bundle.add(path, arcname=str(path).lstrip("/"), recursive=False)
        partial.replace(archive_path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    elif set(json.loads(manifest_path.read_text())) != {str(path) for path in files}:
        raise ValueError("Saved CA3 input selection differs")
    print(json.dumps(dict(status="CA3_INPUT_BUNDLE_READY", archive=str(archive_path),
                          archive_bytes=archive_path.stat().st_size, input_files=len(files))), flush=True)
    if args.pack_only:
        return
    api = module_at("merlin_ca3_transfer", args.merlin_skill / "scripts/merlin_io.py")
    api.upload_chunk = lambda data, name, timeout=180: curl_upload_chunk(api, args.work, data, name, timeout)
    result = api.upload(archive_path, args.record / "upload.json", 4)
    print(json.dumps({key: result[key] for key in ("complete", "bytes", "sha256")}), flush=True)


if __name__ == "__main__":
    main()
