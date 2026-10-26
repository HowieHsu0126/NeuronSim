#!/usr/bin/env python3
"""Observe an existing CA3 Merlin trial; never submit, restart, or cancel work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", type=Path, required=True)
    parser.add_argument("--merlin-tools", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--persistence-batch", type=Path,
                      help="Observe the declared conditions from an existing batch binding.")
    mode.add_argument("--completion75-recovery", action="store_true",
                      help="Observe the existing single-condition completion75 recovery trial.")
    args = parser.parse_args()
    sys.path.insert(0, str(args.merlin_tools))
    io = importlib.import_module("merlin_io")
    initial = json.loads((args.operation / "latest_trial.json").read_text())
    namespace = "completion75_recovery" if args.completion75_recovery else "sm80_candidate_prefix"
    base = initial["checkpoint"] + "/ca3/" + namespace + "/" + initial["sid"]
    targets = [("", base)]
    if args.persistence_batch:
        batch = json.loads(args.persistence_batch.read_text())
        base = initial["checkpoint"] + "/ca3/persistence_conditions/" + initial["sid"]
        targets = [(f"condition_{row['condition_index']:02d}_",
                    base + f"/condition_{row['condition_index']:02d}")
                   for row in batch["conditions"]]
    progress_cursors = {}
    for prefix, _ in targets:
        cursor = 0
        while (args.operation / f"{prefix}progress_{cursor:05d}.json").exists():
            cursor += 1
        progress_cursors[prefix] = cursor
    while True:
        started = time.monotonic()
        observation = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
                           run_sid=initial["job_run_sid"], trial_sid=initial["sid"])
        receipt = None
        try:
            trial = io.call(["job-v2", "run-trials", "get"], dict(sid=initial["sid"]), timeout=90)["data"]
            observation["trial"] = {key: trial.get(key) for key in
                                     ("sid", "status", "start_at", "completed_at")}
            try:
                receipt = io.preview(base + "/receipt.json")
                io.atomic_json(args.operation / "receipt.json", receipt)
                observation["receipt"] = {key: receipt.get(key) for key in
                    ("status", "exit_code", "persisted", "result_bytes", "error", "source_exit_code")}
            except Exception as error:
                observation["receipt_read_error"] = str(error)[:500]
            for prefix, target in targets:
                condition_receipt = None
                if prefix:
                    saved = args.operation / (prefix + "receipt.json")
                    if saved.exists():
                        condition_receipt = json.loads(saved.read_text())
                    else:
                        try:
                            condition_receipt = io.preview(target + "/receipt.json")
                            io.atomic_json(saved, condition_receipt)
                        except Exception as error:
                            observation["condition_receipt_read_error"] = str(error)[:500]
                    if condition_receipt is not None and condition_receipt.get("persisted") is True:
                        observation.setdefault("conditions", {})[prefix.removesuffix("_")] = {
                            key: condition_receipt.get(key) for key in
                            ("status", "exit_code", "persisted", "result_bytes", "finished_utc")}
                        continue
                    observation["active_condition"] = prefix.removesuffix("_")
                for _ in range(3):
                    cursor = progress_cursors[prefix]
                    try:
                        progress = io.preview(target + f"/progress_{cursor:05d}.json")
                    except Exception as error:
                        observation["progress_read_error"] = str(error)[:500]
                        break
                    io.atomic_json(args.operation / f"{prefix}progress_{cursor:05d}.json", progress)
                    observation["last_source_progress"] = progress
                    progress_cursors[prefix] += 1
                # Source conditions execute sequentially. Observe the first
                # unfinished condition without polling later nonexistent files.
                if prefix:
                    break
            observation["observed_utc"] = datetime.now(timezone.utc).isoformat()
            observation["next_check_seconds"] = 300
            io.atomic_json(args.operation / "monitor.json", observation)
            print(json.dumps(observation), flush=True)
            if receipt is not None and receipt.get("persisted") is True:
                return 0 if receipt.get("exit_code") == 0 else 1
            if trial.get("status") in (4, 5):
                # A missing receipt never authorizes a repeated scientific run.
                return 2
        except Exception as error:
            observation["observation_error"] = str(error)[:500]
            io.atomic_json(args.operation / "monitor.json", observation)
            print(json.dumps(observation), flush=True)
        while time.monotonic() - started < 300:
            time.sleep(min(30, 300 - (time.monotonic() - started)))


if __name__ == "__main__":
    raise SystemExit(main())
