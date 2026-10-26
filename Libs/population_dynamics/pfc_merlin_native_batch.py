"""Execute the remaining PFC circuit23/block6 native conditions after canary PASS.

Each child is a process within the same real Merlin allocation. Candidate
outputs already exist. Native results and failures retain separate task paths.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from .pfc_merlin_native_correspondence import hdfs, read_json, require
from .provenance import sha256_file

TASK_INDICES = tuple(range(691, 700))
CANARY_RUN = "58d9076a85680612"
CANARY_TRIAL = "437239519"


def validate_request(request):
    require(request.get("schema") == "pfc_merlin_native_conditions_request_v1"
            and [t["index"] for t in request.get("tasks", [])] == list(TASK_INDICES)
            and request.get("candidate_source_run") == "6de397acc866aadd"
            and request.get("candidate_source_trial") == "435906314",
            "PFC_NATIVE_EXACT_REMAINING_NINE_CONDITIONS_REQUIRED")
    for task in request["tasks"]:
        identity = task["identity"]
        require(identity.get("circuit_index") == 23 and identity.get("block_index") == 6
                and identity.get("condition_index") == task["index"] - 690
                and identity.get("episode_id") == f"pfc_working_memory_circuit_023_block_006_condition_{task['index'] - 690:02d}"
                and identity.get("split") == "train",
                "PFC_NATIVE_REMAINING_CONDITION_IDENTITY_CHANGED")
    require(request.get("canary_run") == CANARY_RUN and request.get("canary_trial") == CANARY_TRIAL
            and request["canary_checkpoint"].endswith("/" + CANARY_RUN + "/pfc_native_correspondence"),
            "PFC_NATIVE_CANARY_BINDING_CHANGED")


def validate_canary(comparison, receipt, validation, request, receipt_sha256):
    require(comparison.get("status") == "PASS_NATIVE_CAMPAIGN_CORRESPONDENCE"
            and comparison.get("identity", {}).get("episode_id") == "pfc_working_memory_circuit_023_block_006_condition_00"
            and comparison.get("execution", {}).get("run_id") == CANARY_RUN
            and comparison.get("execution", {}).get("trial_id") == CANARY_TRIAL
            and comparison.get("native_receipt_sha256") == receipt_sha256
            and comparison.get("candidate_source_report_sha256") == request["canary_candidate_source_report_sha256"]
            and comparison.get("full_scale_source_case_complete") is True
            and comparison.get("candidate_source_simulations") == 0
            and comparison.get("new_native_source_simulations") == 1
            and comparison.get("production_episodes") == 0
            and comparison.get("source_qualification_granted") is False
            and comparison.get("scaling_qualified") is False,
            "PFC_NATIVE_CANARY_COMPLETE_COMPARISON_PASS_REQUIRED")
    arrays = comparison.get("arrays", [])
    require(bool(arrays) and all(r.get("passed") is True for r in arrays)
            and comparison.get("array_comparisons") == comparison.get("passed_arrays") == len(arrays)
            and any(r.get("array") == "spike_ticks" and r.get("exact") is True for r in arrays)
            and any(r.get("array") == "terminal_voltage_mV" and r.get("atol") == 1e-10
                    and r.get("rtol") == 0. for r in arrays), "PFC_NATIVE_CANARY_ARRAY_CRITERIA_CHANGED")
    require(receipt.get("status") == "PERSISTED_COMPLETE_NATIVE_EXECUTION" and receipt.get("exit_code") == 0
            and receipt.get("run_id") == CANARY_RUN and receipt.get("trial_id") == CANARY_TRIAL
            and receipt.get("bundle_uri") == comparison["native_bundle_uri"],
            "PFC_NATIVE_CANARY_DURABLE_OUTPUT_REQUIRED")
    require(validation.get("exit_code") == 0 and validation.get("run_id") == CANARY_RUN
            and validation.get("trial_id") == CANARY_TRIAL
            and validation.get("files", {}).get("Libs/population_dynamics/pfc_merlin_native_correspondence.py")
                == request["canary_consumer_sha256"], "PFC_NATIVE_CANARY_SOFTWARE_BINDING_CHANGED")


def run(request_path, source, work, workers):
    request = read_json(request_path)
    validate_request(request)
    require(bool(os.environ.get("RH2_JOB_RUN_ID")) and bool(os.environ.get("ARNOLD_TRIAL_ID"))
            and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0"
            and type(workers) is int and 1 <= workers <= 9
            and workers < int(os.environ.get("ARNOLD_HEAD_CPU", "0")) and not work.exists(),
            "PFC_NATIVE_BATCH_ALLOCATED_CPU_REQUIRED")
    work.mkdir()
    canary = work / "canary"
    canary.mkdir()
    for name in ("comparison.json", "native_receipt.json", "validation.json"):
        hdfs("-get", request["canary_checkpoint"] + "/" + name, canary / name)
    validate_canary(read_json(canary / "comparison.json"), read_json(canary / "native_receipt.json"),
                    read_json(canary / "validation.json"), request, sha256_file(canary / "native_receipt.json"))
    checkpoint = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/")
    hdfs("-mkdir", "-p", checkpoint + "/pfc_native_batch")
    state = dict(schema="pfc_native_conditions_batch_v1", status="RUNNING", run_id=os.environ["RH2_JOB_RUN_ID"],
        trial_id=os.environ["ARNOLD_TRIAL_ID"], task_indices=list(TASK_INDICES), workers=workers,
        canary_run=CANARY_RUN, canary_trial=CANARY_TRIAL,
        canary_comparison_sha256=sha256_file(canary / "comparison.json"),
        completed=[], failures=[], candidate_source_simulations=0, production_episodes=0,
        source_qualification_granted=False, scaling_qualified=False)

    def publish():
        state["recorded_utc"] = datetime.now(timezone.utc).isoformat()
        path = work / "progress.json"
        path.write_text(json.dumps(state, indent=2) + "\n")
        hdfs("-put", "-f", path, checkpoint + "/pfc_native_batch/progress.json")

    def single(task):
        index = task["index"]
        root = work / f"task_{index:04d}"
        root.mkdir()
        request_file = root / "request.json"
        request_file.write_text(json.dumps(dict(schema="pfc_merlin_native_correspondence_request_v1", tasks=[task]), indent=2) + "\n")
        env = dict(os.environ, MPLCONFIGDIR=str(root / "mpl"), TMPDIR=str(root),
                   PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        suffix = f"native_tasks/task_{index:04d}"
        command = [sys.executable, "-B", "-u", "-m", "population_dynamics.pfc_merlin_native_correspondence",
                   "--request", str(request_file), "--source", str(source), "--work", str(root / "work"),
                   "--checkpoint-subdirectory", suffix]
        with (root / "execution.log").open("x") as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        comparison = root / "work/comparison.json"
        native_receipt = root / "work/native_receipt.json"
        result = dict(index=index, exit_code=process.returncode,
            native_persisted=native_receipt.is_file(), comparison_returned=comparison.is_file(),
            output_prefix=checkpoint + "/" + suffix + "/pfc_native_correspondence",
            run_id=state["run_id"], trial_id=state["trial_id"], command=command)
        if comparison.is_file():
            value = read_json(comparison)
            require(value["identity"] == task["identity"]
                    and value["execution"]["run_id"] == state["run_id"]
                    and value["execution"]["trial_id"] == state["trial_id"], "PFC_NATIVE_CHILD_RESULT_IDENTITY_CHANGED")
            result.update(comparison_status=value["status"], comparison_sha256=sha256_file(comparison),
                          comparison_passed=value["status"] == "PASS_NATIVE_CAMPAIGN_CORRESPONDENCE")
        else:
            result["comparison_passed"] = False
        result["passed"] = process.returncode == 0 and result["native_persisted"] and result["comparison_passed"]
        (root / "terminal.json").write_text(json.dumps(result, indent=2) + "\n")
        hdfs("-put", root / "terminal.json", result["output_prefix"] + "/batch_terminal.json")
        hdfs("-put", root / "execution.log", result["output_prefix"] + "/batch_execution.log")
        return result

    publish()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(single, task): task["index"] for task in request["tasks"]}
        for future in as_completed(futures):
            try:
                result = future.result()
                state["completed" if result["passed"] else "failures"].append(result)
            except Exception as error:
                state["failures"].append(dict(index=futures[future], error_type=type(error).__name__, error=str(error)))
            publish()
    state["status"] = "PASS" if len(state["completed"]) == 9 and not state["failures"] else "FAILED"
    publish()
    path = work / "report.json"
    path.write_text(json.dumps(state, indent=2) + "\n")
    hdfs("-put", path, checkpoint + "/pfc_native_batch/report.json")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    state = run(args.request, args.source, args.work, args.workers)
    raise SystemExit(0 if state["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
