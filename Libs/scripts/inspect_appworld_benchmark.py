"""Inspect AppWorld release and split membership without loading any task."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import socket

from Libs.population_dynamics.resource_probe import probe

SPLITS = ("train", "dev", "test_normal", "test_challenge")
TASK_ID = re.compile(r"[a-f0-9]+_[1-9][0-9]*\Z")


def inspect_splits(data_root: Path, load_task_ids) -> dict:
    result = {}
    seen_tasks: dict[str, str] = {}
    seen_scenarios: dict[str, str] = {}
    for split in SPLITS:
        path = data_root / "data" / "datasets" / (split + ".txt")
        raw = path.read_bytes()
        task_ids = load_task_ids(split)  # Official tag semantics; no difficulty/ground-truth filtering.
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("APPWORLD_SPLIT_EMPTY_OR_DUPLICATE:" + split)
        scenarios: dict[str, list[str]] = {}
        for task_id in task_ids:
            if TASK_ID.fullmatch(task_id) is None:
                raise ValueError("APPWORLD_INVALID_TASK_ID:" + split)
            scenario = task_id.rsplit("_", 1)[0]
            if task_id in seen_tasks:
                raise ValueError("APPWORLD_TASK_SPLIT_OVERLAP:" + split)
            if scenario in seen_scenarios and seen_scenarios[scenario] != split:
                raise ValueError("APPWORLD_SCENARIO_SPLIT_OVERLAP:" + split)
            seen_tasks[task_id] = split
            seen_scenarios[scenario] = split
            scenarios.setdefault(scenario, []).append(task_id)
        incomplete = [key for key, ids in scenarios.items()
                      if {x.rsplit("_", 1)[1] for x in ids} != {"1", "2", "3"}]
        if incomplete:
            raise ValueError("APPWORLD_INCOMPLETE_SCENARIO:" + split)
        result[split] = {
            "membership_file": str(path),
            "membership_sha256": hashlib.sha256(raw).hexdigest(),
            "task_count": len(task_ids),
            "scenario_count": len(scenarios),
            "task_ids": task_ids,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("APPWORLD_INTAKE_REQUIRES_SLURM")
    if socket.gethostname().split(".")[0] not in {"cpu03", "cpu04"}:
        raise ValueError("APPWORLD_CPU_INTAKE_NODE_NOT_AUTHORIZED")
    args.output.mkdir(parents=True, exist_ok=False)
    health = probe(args.output / "resource_admission.json", required_memory_gib=2)
    if health["status"] != "PASS":
        raise RuntimeError("APPWORLD_RESOURCE_ADMISSION_FAILED")
    # Native imports run inside the existing isolated compute-node runtime.
    from appworld import AppWorld, update_root
    from appworld.task import load_task_ids

    update_root(str(args.data_root.resolve()))
    splits = inspect_splits(args.data_root.resolve(), load_task_ids)
    distribution = importlib.metadata.distribution("appworld")
    source = {}
    for relative in ("appworld/environment.py", "appworld/task.py", "appworld/evaluator.py"):
        path = Path(distribution.locate_file(relative))
        raw = path.read_bytes()
        source[relative] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = {
        "schema": "appworld_split_membership_v1",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "appworld_version": distribution.version,
        "data_root": str(args.data_root.resolve()),
        "splits": splits,
        "task_contents_read": False,
        "task_worlds_created": 0,
        "model_generations": 0,
        "native_source": source,
    }
    (args.output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "PASS_SPLIT_MEMBERSHIP_AND_RUNTIME_IMPORT",
        "job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(),
        "appworld_version": distribution.version,
        "python_version": os.sys.version,
        "splits": {name: {k: row[k] for k in ("task_count", "scenario_count")}
                   for name, row in splits.items()},
        "native_interfaces": {
            "AppWorld": str(inspect.signature(AppWorld)),
            "load_task_ids": str(inspect.signature(load_task_ids)),
        },
        "task_contents_read": False,
        "task_worlds_created": 0,
        "model_generations": 0,
        "formal_evaluation_released": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
