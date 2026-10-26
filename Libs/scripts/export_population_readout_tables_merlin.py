"""Return existing readout tables in small JSON files when large previews truncate.

Run on Merlin. The original archive and CSV tables remain authoritative. This
transport changes neither metric values nor their aggregation or denominators.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import tempfile


def hdfs_read(uri: str) -> bytes:
    return subprocess.check_output(["hdfs", "dfs", "-cat", uri])


def publish(directory: Path, destination: str, name: str, value: dict) -> dict:
    data = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path = directory / name
    path.write_bytes(data)
    subprocess.run(["hdfs", "dfs", "-put", str(path), destination + "/" + name], check=True)
    return {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-uri", required=True)
    parser.add_argument("--archive-sha256", required=True)
    args = parser.parse_args()
    if platform.node().split(".")[0] == "mgmt2":
        raise RuntimeError("Result archive processing must run on Merlin")
    run, trial, checkpoint = (os.environ[key] for key in (
        "RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "RH2_CHECKPOINT_HDFS"))
    receipt = json.loads(hdfs_read(args.receipt_uri))
    if (receipt.get("persisted") is not True or receipt.get("exit_code") != 0
            or receipt["checkpoint_path"] + "/receipt.json" != args.receipt_uri
            or receipt["result_sha256"] != args.archive_sha256):
        raise ValueError("A bound successful original receipt is required")
    data = hdfs_read(receipt["checkpoint_path"] + "/results.tar.gz")
    if len(data) != receipt["result_bytes"] or hashlib.sha256(data).hexdigest() != args.archive_sha256:
        raise ValueError("Original result archive identity differs")
    destination = checkpoint.rstrip("/") + "/readout_tables"
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", destination], check=True)
    with tempfile.TemporaryDirectory(prefix="readout-table-return-") as temporary, \
            tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = {member.name: member for member in archive}
        if len(members) != len(archive.getmembers()):
            raise ValueError("Duplicate result archive names")

        def read(name: str) -> bytes:
            member = members["results/" + name]
            if not member.isfile():
                raise ValueError("Expected a regular result file")
            return archive.extractfile(member).read()

        analysis_bytes = read("analysis.json")
        analysis = json.loads(analysis_bytes)
        execution_bytes = read("execution.json")
        execution = json.loads(execution_bytes)
        if (analysis["status"] != "COMPLETE_POSTHOC_NATIVE_READOUT_DIAGNOSTIC"
                or analysis["training_seeds"] != [41, 42, 43]
                or execution["exit_code"] != 0):
            raise ValueError("The completed three-seed diagnostic is required")
        published, verified = [], {}
        for name, identity in analysis["artifacts"].items():
            table = read(name)
            if hashlib.sha256(table).hexdigest() != identity["sha256"]:
                raise ValueError("Result table checksum differs: " + name)
            reader = csv.DictReader(io.StringIO(table.decode(), newline=""))
            rows = list(reader)
            if len(rows) != identity["rows"]:
                raise ValueError("Result table row count differs: " + name)
            verified[name] = {**identity, "bytes": len(table)}
            if name not in ("seed_horizon_summary.csv", "source_block_summary.csv"):
                continue
            # Factor repeated metric definitions and identities. Every value and
            # denominator remains the original CSV string, including empty values.
            blocks = [None] if name.startswith("seed_") else analysis["source_blocks"]
            for block in blocks:
                selected = rows if block is None else [row for row in rows if row["block"] == str(block)]
                definitions, definition_index, groups = [], {}, {}
                for row in selected:
                    definition = (row["metric"], row["statistic"], row["metric_scope"])
                    if definition not in definition_index:
                        definition_index[definition] = len(definitions)
                        definitions.append(definition)
                    key = (row["model"], row["seed"], row["horizon_ms"])
                    groups.setdefault(key, []).append([definition_index[definition], row["value"],
                        row["available_origins"], row["structural_na_origins"], row["total_origins"]])
                compact = {"source_table": name, "source_identity": verified[name], "block": block,
                    "definition_columns": ["metric", "statistic", "metric_scope"], "definitions": definitions,
                    "identity_columns": ["model", "seed", "horizon_ms"],
                    "value_columns": ["definition_index", "value", "available_origins", "structural_na_origins", "total_origins"],
                    "groups": [{"identity": key, "rows": value} for key, value in groups.items()],
                    "rows": len(selected)}
                prefix = "seed_summary" if block is None else f"block_{block}"
                # Keep each response small as the diagnostic adds metrics. This
                # partitions transport only; all model/seed/horizon groups remain.
                all_groups = compact["groups"]
                for start in range(0, len(all_groups), 9):
                    part = {**compact, "groups": all_groups[start:start + 9]}
                    part["rows"] = sum(len(group["rows"]) for group in part["groups"])
                    output_name = f"{prefix}_part_{start // 9:04d}.json"
                    published.append(publish(Path(temporary), destination, output_name, part))
        result = {"status": "PASS", "exit_code": 0, "run_id": run, "trial_id": trial,
            "checkpoint_path": destination, "original_receipt": args.receipt_uri,
            "original_archive_sha256": args.archive_sha256, "original_execution": execution["runtime"]["environment"],
            "analysis_sha256": hashlib.sha256(analysis_bytes).hexdigest(),
            "execution_sha256": hashlib.sha256(execution_bytes).hexdigest(),
            "verified_tables": verified, "files": published,
            "source_simulations": 0, "model_training": 0, "diagnostics_recomputed": 0}
        publish(Path(temporary), destination, "receipt.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
