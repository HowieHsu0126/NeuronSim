"""Build the declared uniform closure dataset from a verified Merlin return."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

from population_dynamics.appworld_benchmark_closure_data import build_closure_dataset


def identity(content: bytes) -> dict:
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def read_json(path: Path) -> tuple[dict | list, dict]:
    content = path.read_bytes()
    return json.loads(content), {"path": str(path), **identity(content)}


def verify_return_source(intake: dict, returned: dict, receipt: dict, returned_id: dict) -> dict:
    """Admit a verified return while preserving the original source outcome."""
    source = intake["source_receipt"]
    outcome = intake.get("source_outcome", "completed")
    code = source.get("exit_code")
    valid_source = type(code) is int and source.get("persisted") is True
    if outcome == "completed":
        expected_status = "PASS"
        valid_source = valid_source and code == 0
    elif outcome == "failed_diagnostics":
        expected_status = "PASS_FAILED_SOURCE_DIAGNOSTICS"
        valid_source = (
            valid_source and code != 0 and source.get("status") in {"FAIL", "FAILED"}
            and receipt.get("source_outcome") == outcome
            and receipt.get("source_exit_code") == code
            and receipt.get("source_status") == source.get("status")
            and receipt.get("source_trial_id") == source["trial_id"]
        )
    else:
        raise ValueError("Unsupported source outcome")
    if (
        not valid_source or receipt.get("status") != expected_status
        or receipt.get("exit_code") != 0 or receipt.get("persisted") is not True
        or receipt.get("source_run_id") != source["run_id"]
        or returned.get("source_run_id") != source["run_id"]
        or returned.get("source_trial_id") != source["trial_id"]
        or any(returned_id[key] != receipt["returned_files"][key] for key in ("bytes", "sha256"))
    ):
        raise ValueError("Verified complete return differs from its source")
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("return-inputs", "returned-files", "return-receipt", "split-manifest", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("ARNOLD_APPLIED_GPU_NUM") != "0":
        raise ValueError("Merlin CPU allocation required")
    if args.output.exists():
        raise FileExistsError(args.output)
    intake, intake_id = read_json(args.return_inputs)
    returned, returned_id = read_json(args.returned_files)
    receipt, receipt_id = read_json(args.return_receipt)
    source = verify_return_source(intake, returned, receipt, returned_id)
    manifest = json.loads(base64.b64decode(intake["manifest_content_base64"], validate=True))
    files = returned["files"]

    def decoded(relative: str) -> tuple[dict, dict]:
        row = files["artifacts/" + relative]
        content = base64.b64decode(row["content_base64"], validate=True)
        actual = identity(content)
        if actual != manifest["files"][relative] or any(actual[key] != row[key] for key in actual):
            raise ValueError("Returned artifact differs: " + relative)
        return json.loads(content), {
            "uri": source["checkpoint_path"].rstrip("/") + "/stages/" + relative, **actual,
        }

    split, split_id = read_json(args.split_manifest)
    plan, plan_id = read_json(args.plan)
    batch, batch_id = decoded("development_batch.json")
    admission, admission_id = decoded("input_admission.json")
    fields = ("split", "task_id", "method", "seed")
    key = lambda row: tuple(row[field] for field in fields)
    uniform = {key(row): row for row in batch["attempts"] if row["agent"] == "collaboration_uniform"}
    if len(uniform) != len(plan) or set(uniform) != {key(row) for row in plan}:
        raise ValueError("Plan must retain the complete declared uniform roster")
    records = []
    for planned in plan:
        row = uniform[key(planned)]
        if row.get("state") not in {"SCORED_SUCCESS", "SCORED_NEGATIVE"} or row.get("closed_outputs_persisted") is not True:
            raise ValueError("Every declared uniform attempt must be scored and persisted")
        directory = row["output_directory"] + "/attempt/"
        execution, execution_id = decoded(directory + "execution_receipt.json")
        score, score_id = decoded(directory + "development_score.json")
        records.append({
            "execution_receipt": execution, "score_receipt": score, "input_admission": admission,
            "input_files": {"execution_receipt": execution_id, "score_receipt": score_id,
                            "input_admission": admission_id},
        })
    result = build_closure_dataset(
        split, manifest_identity={name: split_id[name] for name in ("bytes", "sha256")},
        plan=plan, attempts=records,
    )
    result["source_run_id"] = source["run_id"]
    result["source_trial_id"] = source["trial_id"]
    result["source_status"] = source["status"]
    result["source_exit_code"] = source["exit_code"]
    result["source_outcome"] = intake.get("source_outcome", "completed")
    result["source_batch_coverage"] = batch["coverage"]
    result["input_files"] = {
        "return_inputs": intake_id, "returned_files": returned_id, "return_receipt": receipt_id,
        "split_manifest": split_id, "plan": plan_id, "development_batch": batch_id,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    prefix = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/appworld_closure_data"
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", prefix], check=True)
    subprocess.run(["hdfs", "dfs", "-put", str(args.output), prefix + "/dataset.json"], check=True)
    report = {
        "schema": "appworld_merlin_closure_intake_v1", "status": "DATASET_BUILT",
        "run_id": os.environ["RH2_JOB_RUN_ID"], "trial_id": os.environ["ARNOLD_TRIAL_ID"],
        "hostname": socket.gethostname(), "source_run_id": source["run_id"],
        "source_trial_id": source["trial_id"], "source_status": source["status"],
        "source_exit_code": source["exit_code"], "source_outcome": result["source_outcome"],
        "source_batch_coverage": batch["coverage"],
        "coverage": result["coverage"], "fitted_model": False, "formal_eligible": False,
        "native_tasks_executed": 0, "model_generations": 0,
        "dataset": {"uri": prefix + "/dataset.json", **identity(args.output.read_bytes())},
    }
    path = args.output.with_name("closure_intake.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    subprocess.run(["hdfs", "dfs", "-put", str(path), prefix + "/receipt.json"], check=True)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
