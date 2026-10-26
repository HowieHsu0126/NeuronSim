#!/usr/bin/env python3
"""Stage a saved V1 block for Merlin without generating scientific samples.

Run on the existing transfer host. Reuse the migration bundle selector and
resumable uploader. Preserve source paths because the frozen readers bind them.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time


def module_at(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_files(directory: Path, names: list[str]) -> set[Path]:
    files = {directory / "__init__.py"}
    pending = list(names)
    while pending:
        path = directory / (pending.pop() + ".py")
        if path in files:
            continue
        files.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                imports = [node.module.split(".")[0]] if node.module else [v.name for v in node.names]
                pending.extend(name for name in imports if (directory / (name + ".py")).is_file())
    return files


def curl_upload_chunk(api, work: Path, data: bytes, name: str, timeout: int = 180) -> dict:
    """Adapt the existing uploader to the verified curl path through the relay.

    Authentication stays in stdin. The existing uploader owns chunk identities,
    response byte checks and the resumable manifest.
    """
    with tempfile.NamedTemporaryFile(dir=work, suffix=".part") as source:
        source.write(data)
        source.flush()
        command = ["curl", "--silent", "--show-error", "--fail-with-body",
                   "--connect-timeout", "15", "--max-time", str(timeout),
                   "--proxy", os.environ.get("MERLIN_HTTP_PROXY", "http://127.0.0.1:18084"),
                   "--config", "-", "--form", "file=@" + source.name + ";filename=" + name,
                   "https://cluster-api.example.invalid/api/v4/seed/data/datacard-upload-files/create"]
        for attempt in range(5):
            # A long transport retry can cross a normal credential refresh.
            # Re-read current credentials for each request, without logging them.
            session = api.http_session()
            headers = ["x-jwt-token: " + session.headers["x-jwt-token"], "Accept-Encoding: identity"]
            cookies = "; ".join(key + "=" + value for key, value in session.cookies.get_dict().items())
            if cookies:
                headers.append("Cookie: " + cookies)
            configuration = "\n".join("header = " + json.dumps(value) for value in headers) + "\n"
            result = subprocess.run(command, input=configuration, text=True, capture_output=True)
            if result.returncode == 0 or result.returncode not in {28, 35, 52, 55, 56} or attempt == 4:
                break
            # Observed relay restarts interrupt CONNECT/TLS. Allow the relay to
            # recover, with five attempts total. HTTP rejection is not retried.
            # Keep the same bytes; the uploader records only an acknowledged receipt.
            time.sleep(min(30, 5 * 2 ** attempt))
    if result.returncode:
        raise RuntimeError("Curl upload failed: " + result.stderr[-1000:])
    response = json.loads(result.stdout)
    if response.get("ResponseBase", {}).get("code") != 0:
        raise RuntimeError("Curl upload API rejected the chunk: " + str(response.get("ResponseBase")))
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=Path, required=True)
    parser.add_argument("--migration-tools", type=Path, required=True)
    parser.add_argument("--merlin-skill", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--gpu-only", action="store_true",
                        help="Exclude rate movies and CPU export dependencies")
    parser.add_argument("--include-native-comparison", action="store_true",
                        help="Include saved native spikes for circuit-specific backend comparison")
    parser.add_argument("--base-file-manifest", type=Path,
                        help="Reuse immutable files from an already verified base archive")
    parser.add_argument("--transport", choices=("multipart", "curl"), default="multipart")
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--upload-manifest", type=Path)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] == "mgmt2":
        raise SystemExit("Bulk staging requires a compute or transfer host")
    args.work.mkdir(parents=True, exist_ok=True)
    args.record.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.check_output(["stat", "-f", "-c", "%T", str(args.work)], text=True).strip()
    if filesystem in {"nfs", "nfs4", "cifs", "smb2"}:
        raise ValueError("Transfer scratch must use a local filesystem")
    builder = module_at("v1_bundle_builder", args.migration_tools / "build_v1_bundle.py")
    request = json.loads((args.block / "jobs/block_gpu_primary/request.json").read_text())
    command = request["command"]
    launcher = next(i for i, value in enumerate(command) if value.endswith("/run_v1_glif_network.sbatch"))
    values = command[launcher + 1:]
    if len(values) != 10 or values[7] != "source_batch":
        raise ValueError("Saved complete source-block command required")
    response = json.loads((args.block / "jobs/block_gpu_primary/response.json").read_text())
    if response.get("returncode") != 0 or not response.get("job_id"):
        raise ValueError("Accepted source submission receipt required")
    job = dict(job=str(response["job_id"]), command=command[launcher:], repo=values[0], reference=values[1],
               snapshot=values[2], output=values[3], python=values[4], library=values[5],
               detector=values[6], mode=values[7], cases=values[8], code=values[9])
    job["source_submission_response"] = response
    files = builder.files_for(job)
    reference = Path(job["reference"])
    if args.include_native_comparison:
        files.add(reference / "primary/native/cortical_spikes.npz")
    if not args.gpu_only:
        files.update(reference / name for name in (
        "primary/native/cortical_spikes.npz", "primary/network/v1_nodes.h5",
        "primary/network/v1_node_types.csv", "recurrent/primary/biophysical/v1_nodes.h5",
        "recurrent/primary/report.json", "inputs_reference/primary/report.json"))
    expected = {}
    sources = json.loads((args.block / "sources.json").read_text())
    archives = {Path(row["sources"][key]) for row in sources["conditions"]
                for key in (("spike_source",) if args.gpu_only else
                            ("rate_source", "spike_source", "authority_source"))}
    # The public readers consume rate witnesses, runtime sources and geometry.
    # Their existing deliveries define the required input set and identities.
    for archive in sorted(archives):
        delivery_path = archive / "delivery.json"
        delivery = json.loads(delivery_path.read_text())
        if delivery.get("exit_code") != 0:
            raise ValueError("Input delivery failed: " + str(archive))
        files.add(delivery_path)
        for row in delivery["files"]:
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid declared input path")
            source = archive / relative
            if source in expected:
                raise ValueError("Duplicate delivery path: " + str(source))
            expected[source] = row
            if not args.gpu_only or source in files:
                files.add(source)
    repo = Path(job["repo"])
    if not args.gpu_only:
        files.update(module_files(repo / "Libs/population_dynamics", [
            "v1_glif_production", "v1_glif_dataset", "v1_benchmark", "circuit_data"]))
    files.update([args.block / "sources.json", args.block / "campaign.yaml", Path(__file__)])
    base_files = (json.loads(args.base_file_manifest.read_text())
                  if args.base_file_manifest else {})
    required_base = {str(path): base_files[str(path)] for path in files if str(path) in base_files}
    for path, identity in required_base.items():
        declared = expected.get(Path(path))
        if declared and any(identity[key] != declared[key] for key in ("bytes", "sha256")):
            raise ValueError("Base input differs from current delivery: " + path)
    files = {path for path in files if str(path) not in required_base}
    if not files:
        raise ValueError("No additional input files to package")
    if any(not path.is_file() for path in files):
        raise FileNotFoundError([str(path) for path in files if not path.is_file()])
    total = sum(path.stat().st_size for path in files)
    inventory = dict(status="INPUTS_LOCATED", hostname=socket.gethostname(),
                     observed_utc=datetime.now(timezone.utc).isoformat(),
                     source_block=str(args.block), files=len(files), bytes=total,
                     input_scope="gpu_execution" if args.gpu_only else "execution_and_cpu_readout",
                     reused_base_files=len(required_base),
                     base_file_manifest=str(args.base_file_manifest) if args.base_file_manifest else None,
                     local_scratch=str(args.work), scratch_filesystem=filesystem,
                     scratch_free_bytes=shutil.disk_usage(args.work).free,
                     scientific_samples_generated=0, new_production_episodes=0)
    (args.record / "input_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory), flush=True)
    if args.base_file_manifest:
        (args.record / "required_base_files.json").write_text(json.dumps(required_base, indent=2) + "\n")
    if args.inventory_only:
        return
    if inventory["scratch_free_bytes"] < total * 2 + 2**30:
        raise ValueError("Insufficient local transfer storage")
    manifest_path = args.record / "input_files.json"
    archive_path = args.work / "inputs.tar.gz"
    if manifest_path.exists() and archive_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if set(manifest) != {str(path) for path in files}:
            raise ValueError("Saved input selection differs")
    else:
        manifest = {}
        partial = archive_path.with_suffix(".partial")
        with tarfile.open(partial, "w:gz", compresslevel=1, dereference=True) as bundle:
            for source in sorted(files):
                row = dict(bytes=source.stat().st_size, sha256=builder.digest(source))
                if source in expected and any(row[key] != expected[source][key] for key in row):
                    raise ValueError("Source differs from delivery: " + str(source))
                manifest[str(source)] = row
                bundle.add(source, arcname=str(source).lstrip("/"), recursive=False)
        partial.replace(archive_path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (args.record / "source_job.json").write_text(json.dumps(job, indent=2) + "\n")
    if args.package_only:
        print(json.dumps(dict(status="PACKAGED_NOT_UPLOADED", archive=str(archive_path),
                              bytes=archive_path.stat().st_size,
                              sha256=builder.digest(archive_path), files=len(manifest))), flush=True)
        return
    # The existing uploader verifies and resumes the exact saved archive.
    upload_manifest = args.upload_manifest or args.record / "upload.json"
    if upload_manifest.exists() and json.loads(upload_manifest.read_text())["chunk_bytes"] != args.chunk_mib * 1024**2:
        raise ValueError("Use a separate upload manifest when changing chunk size")
    if args.transport == "curl":
        api = module_at("merlin_block_transfer", args.merlin_skill / "scripts/merlin_io.py")
        # This isolated module instance leaves the shared installed helper intact.
        api.upload_chunk = lambda data, name, timeout=180: curl_upload_chunk(
            api, args.work, data, name, timeout)
        stalled_attempts = 0
        while True:
            before = len(json.loads(upload_manifest.read_text())["chunks"]) if upload_manifest.exists() else 0
            try:
                result = api.upload(archive_path, upload_manifest, args.chunk_mib)
                break
            except RuntimeError as error:
                message = str(error)
                retryable = any(message.startswith(f"Curl upload failed: curl: ({code})")
                                for code in (28, 35, 52, 55, 56))
                if not retryable:
                    raise
                after = len(json.loads(upload_manifest.read_text())["chunks"]) if upload_manifest.exists() else 0
                stalled_attempts = 0 if after > before else stalled_attempts + 1
                if stalled_attempts >= 3:
                    raise
                print(json.dumps(dict(status="RESUMING_INTERRUPTED_UPLOAD", chunks_saved=after,
                                      consecutive_attempts_without_progress=stalled_attempts)), flush=True)
                time.sleep(30)
        print(json.dumps({key: result[key] for key in ("complete", "bytes", "sha256")}), flush=True)
    else:
        subprocess.run([sys.executable, str(args.merlin_skill / "scripts/merlin_io.py"),
                        "upload", str(archive_path), str(upload_manifest),
                        "--chunk-mib", str(args.chunk_mib)], check=True)


if __name__ == "__main__":
    main()
