#!/usr/bin/env python3
"""Return validated Merlin V1 input deliveries on the existing transfer host."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--max-new-chunks", type=int, default=50)
    parser.add_argument("--merlin-skill", type=Path, required=True)
    parser.add_argument("--migration-tools", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] == "mgmt2":
        raise ValueError("V1_DELIVERY_RETURN_REQUIRES_TRANSFER_HOST")
    sys.path.insert(0, str(args.merlin_skill / "scripts"))
    sys.path.insert(0, str(args.migration_tools))
    from merlin_io import collect, preview
    from return_results import integrate
    request = json.loads(args.request.read_text())
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    receipts, stages = {}, {}
    for role in ("consumer", "source"):
        bundle = args.work / role / "result.tar.gz"
        uri = request[role + "_receipt"]
        cached_receipt = bundle.parent / "remote_receipt.json"
        receipt = json.loads(cached_receipt.read_text()) if cached_receipt.exists() else preview(uri)
        if (receipt.get("persisted") is not True or receipt.get("exit_code") != 0
                or receipt["checkpoint_path"].rstrip("/") + "/receipt.json" != uri):
            raise ValueError("V1_RETURN_PERSISTED_RECEIPT_REQUIRED")
        bundle.parent.mkdir(parents=True, exist_ok=True)
        cached_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        cache = bundle.with_name(bundle.name + ".chunks")
        cache.mkdir(exist_ok=True)
        if args.max_new_chunks <= 0:
            raise ValueError("V1_RETURN_POSITIVE_CHUNK_BUDGET_REQUIRED")
        missing = []
        for index, row in enumerate(receipt["chunks"]):
            chunk = cache / f"{index:05d}.part"
            if chunk.exists():
                if chunk.stat().st_size != row["bytes"] or digest(chunk) != row["sha256"]:
                    raise ValueError("V1_RETURN_CACHED_CHUNK_HASH_DIFFERS")
            else:
                missing.append((index, row))
        def fetch(item):
            index, row = item
            value = preview(receipt["checkpoint_path"].rstrip("/") + "/" + row["name"])
            data = base64.b64decode(value["content_base64"], validate=True)
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("V1_RETURN_REMOTE_CHUNK_HASH_DIFFERS")
            chunk = cache / f"{index:05d}.part"
            temporary = chunk.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(chunk)
            return index
        # Limit queued requests so one broken connection cannot leave the full
        # delivery queued behind a failed future in the shared collector.
        selected = missing[:args.max_new_chunks]
        pool = ThreadPoolExecutor(max_workers=args.workers)
        futures = [pool.submit(fetch, item) for item in selected]
        downloaded = 0
        try:
            for future in as_completed(futures):
                index = future.result()
                downloaded += 1
                if downloaded == 1 or downloaded % 10 == 0:
                    print(json.dumps(dict(role=role, new_chunks=downloaded, index=index)), flush=True)
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        if len(missing) > len(selected):
            progress = dict(status="PARTIAL_CHUNK_RETURN", role=role,
                            new_chunks=downloaded, total_chunks=len(receipt["chunks"]),
                            cached_chunks=len(receipt["chunks"]) - len(missing) + downloaded,
                            observed_utc=datetime.now(timezone.utc).isoformat())
            (args.output / "transfer_progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(progress), flush=True)
            return 0
        receipt = collect(uri, bundle, receipt=receipt, workers=args.workers)
        if (receipt["merlin_job_run_id"] != request["run_sid"]
                or receipt["merlin_trial_id"] != request["trial_sid"]
                or receipt["exit_code"] != 0):
            raise ValueError("V1_RETURN_EXECUTION_IDENTITY_DIFFERS")
        verified = integrate(bundle)
        stages[role] = bundle.with_suffix(".intake")
        receipts[role] = dict(receipt=receipt, verification=verified)
        dest = args.output / role
        dest.mkdir(exist_ok=True)
        shutil.copy2(bundle, dest / bundle.name)
        (dest / "remote_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(dict(role=role, verified_files=verified["files"], result_bytes=receipt["result_bytes"])), flush=True)
    source_remote = Path(request["source_remote_root"])
    consumer_remote = Path(request["consumer_remote_root"])
    source = stages["source"] / str(source_remote).lstrip("/")
    consumer = stages["consumer"] / str(consumer_remote).lstrip("/")
    terminal = json.loads((consumer / "merlin_terminal.json").read_text())
    intake = json.loads((consumer / "intake/report.json").read_text())
    reference = json.loads((consumer / "reference/report.json").read_text())
    manifest = json.loads((consumer / "source_manifest.json").read_text())
    circuit = request["circuit"]
    if (terminal["status"] != "PASS_V1_INPUT_INTAKE_AND_REFERENCE"
            or terminal["circuits"] != [circuit] or intake["validated_input_acceptances"] != 1
            or reference["prepared_conditions"] != 10 or len(reference["artifacts"]) != 12
            or [row["circuit_index"] for row in manifest["constructions"]] != [circuit]):
        raise ValueError("V1_RETURN_ONE_COMPLETE_CIRCUIT_REQUIRED")
    row = manifest["constructions"][0]
    if digest(source / "delivery.json") != row["delivery_sha256"]:
        raise ValueError("V1_RETURN_SOURCE_MANIFEST_DIFFERS")
    delivery = json.loads((source / "delivery.json").read_text())
    if delivery["status"] != "PASS_SOURCE_INPUT_BUILD_REPEATABILITY" or delivery["exit_code"]:
        raise ValueError("V1_RETURN_INPUT_SOURCE_NOT_PASSED")
    canonical_source = Path(row["archive"])
    allowed = Path(request["qualification_root"]).parent.parent
    changes = []
    source_files = [source / item["path"] for item in delivery["files"]]
    source_files.append(source / "delivery.json")
    for item in source_files:
        changes.append((item, canonical_source / item.relative_to(source), digest(item)))
    def returned(name: str) -> Path:
        return consumer / Path(name).relative_to(consumer_remote)
    for item in intake["acceptances"]:
        changes.append((returned(item["acceptance"]), Path(item["canonical_target"]), item["acceptance_identity"]["sha256"]))
        changes.append((returned(item["construction_record"]), Path(item["canonical_construction_record"]), item["construction_record_identity"]["sha256"]))
        receipt_file = returned(item["acceptance"]).with_name("inputs_intake_execution.json")
        changes.append((receipt_file, Path(item["canonical_target"]).with_name(receipt_file.name), digest(receipt_file)))
    for item in reference["artifacts"]:
        changes.append((returned(item["path"]), Path(item["canonical_target"]), item["sha256"]))
    changes.append((consumer / "source_manifest.json", Path(terminal["source_manifest_target"]), digest(consumer / "source_manifest.json")))
    destinations = set()
    for source_file, destination, sha in changes:
        if (not destination.is_absolute() or ".." in destination.parts
                or not destination.resolve().is_relative_to(allowed) or destination in destinations):
            raise ValueError("V1_RETURN_CANONICAL_DESTINATION_INVALID:" + str(destination))
        destinations.add(destination)
        if digest(source_file) != sha:
            raise ValueError("V1_RETURN_CANDIDATE_HASH_DIFFERS")
        if destination.exists() and digest(destination) != sha:
            raise FileExistsError("V1_RETURN_CANONICAL_CONFLICT:" + str(destination))
    added = []
    for source_file, destination, sha in changes:
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source_file.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        if digest(destination) != sha:
            raise ValueError("V1_RETURN_COPY_HASH_DIFFERS:" + str(destination))
        added.append(str(destination))
    # Retain small execution and consumer reports beside the returned bundles.
    for relative in ("merlin_terminal.json", "intake/report.json", "reference/report.json", "reference/execution_receipt.json"):
        target = args.output / "reports" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(consumer / relative, target)
    q = Path(request["qualification_root"])
    report = dict(status="PASS_V1_MERLIN_INPUT_DELIVERY_INTEGRATED", circuit=circuit,
                  run_sid=request["run_sid"], trial_sid=request["trial_sid"],
                  observed_utc=datetime.now(timezone.utc).isoformat(), source_delivery_sha256=row["delivery_sha256"],
                  verified_source_files=len(source_files), canonical_files_added=added,
                  validated_input_acceptances=1, prepared_conditions=10, source_simulations=0, production_episodes=0,
                  canonical_input_acceptances=sum((q / f"circuit_{c:03d}/input_acceptance.json").is_file() for c in range(100)),
                  canonical_reference_controls=sum((q / f"circuit_{c:03d}/reference_execution/prepared_control.json").is_file() for c in range(100)),
                  returns=receipts)
    (args.output / "integration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "circuit", "verified_source_files", "canonical_input_acceptances", "canonical_reference_controls")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
