#!/usr/bin/env python3
"""Run a manifest-bound tokenizer diagnostic on Merlin CPU.

Restore the pinned model runtime and tokenizer assets. Do not restore model
weights, native task data, or a model server.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from run_appworld_benchmark_merlin import (
    HdfsStore, MODEL_PREFIX, declared_content, model_download_uris,
    restore_archive, verify_bytes, write_json,
)

TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "chat_template.jinja", "merges.txt", "config.json",
)


def main() -> int:
    inputs = json.loads(Path(sys.argv[1]).read_text())
    if os.environ.get("ARNOLD_APPLIED_GPU_NUM") != "0":
        raise ValueError("Merlin CPU allocation required")
    run_id = os.environ["RH2_JOB_RUN_ID"]
    work = Path("/tmp") / ("appworld-prompt-diagnostic-" + run_id)
    work.mkdir(exist_ok=False)
    artifacts = work / "artifacts"
    artifacts.mkdir()
    prefix = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/appworld_prompt_diagnostic"
    store = HdfsStore(prefix, work, artifacts)
    receipt = {
        "schema": "appworld_prompt_diagnostic_merlin_v1",
        "run_id": run_id, "trial_id": os.environ["ARNOLD_TRIAL_ID"],
        "hostname": socket.gethostname(), "checkpoint_path": prefix,
        "status": "RUNNING", "stage": "stage_sources", "started_at": time.time(),
        "native_tasks_executed": 0, "model_generations": 0,
        "model_weights_restored": False, "formal_eligible": False,
        "source_input_identity": {key: inputs["source_inputs"][key]
                                  for key in ("bytes", "sha256")},
        "diagnostic_source_identity": {
            name: {key: row[key] for key in ("bytes", "sha256")}
            for name, row in inputs["files"].items()
        },
    }
    def progress(stage: str) -> None:
        receipt["stage"] = stage
        write_json(artifacts / "progress.json", receipt)
        store.publish(artifacts / "progress.json", "progress.json", replace=True)

    try:
        for name, row in inputs["files"].items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "Libs":
                raise ValueError("Unsafe declared diagnostic source path")
            target = work / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(declared_content(row))
        original = work / "source_inputs.json"
        original.write_bytes(declared_content(inputs["source_inputs"]))
        source = json.loads(original.read_text())
        progress("restore_pinned_model_runtime")
        runtime = source["model_runtime"]
        restore_archive(
            store, work, "model_runtime", runtime["runtime_bundle_uri"],
            {"bytes": runtime["runtime_bundle_bytes"], "sha256": runtime["runtime_bundle_sha256"]},
            (MODEL_PREFIX,),
            (MODEL_PREFIX, Path("/opt/python"), Path("/opt/environments/neuronsim-mechanism-learning")),
        )
        receipt["runtime_identity"] = {
            "source_run": runtime["run_id"],
            "sha256": runtime["runtime_bundle_sha256"],
            "packages": runtime["packages"],
        }
        progress("restore_tokenizer_assets")
        tokenizer = work / "tokenizer"
        tokenizer.mkdir()
        locations = model_download_uris(source["model_inputs"])
        receipt["tokenizer_assets"] = {}
        for name in TOKENIZER_FILES:
            target = tokenizer / name
            store.hdfs("-get", locations[name], str(target))
            actual = verify_bytes(target, source["model_inputs"]["files"][name])
            receipt["tokenizer_assets"][name] = {**actual, "uri": locations[name]}
        progress("retrieve_verified_training_receipts")
        returned = work / "returned_files.json"
        store.hdfs("-get", inputs["returned_files"]["uri"], str(returned))
        receipt["returned_files_identity"] = verify_bytes(returned, inputs["returned_files"])
        progress("measure_prompt_tokens")
        environment = dict(os.environ, PYTHONPATH=str(work / "Libs"),
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
        output = artifacts / "prompt_tokens.json"
        command = [
            runtime["environment"]["python"],
            str(work / "Libs/scripts/diagnose_appworld_prompt_tokens.py"),
            "--source-inputs", str(original), "--returned-files", str(returned),
            "--tokenizer-dir", str(tokenizer), "--output", str(output),
        ]
        receipt["command"] = command
        with (artifacts / "diagnostic.log").open("w") as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        receipt["diagnostic_exit_code"] = result.returncode
        if output.exists():
            receipt["diagnostic_output"] = store.publish(output, "prompt_tokens.json")
        receipt["diagnostic_log"] = store.publish(artifacts / "diagnostic.log", "diagnostic.log")
        if result.returncode:
            raise RuntimeError("Prompt diagnostic exited " + str(result.returncode))
        report = json.loads(output.read_text())
        receipt.update(status=report["status"], exit_code=0)
    except Exception as error:
        receipt.update(status="FAILED", exit_code=1, error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc())
    receipt["finished_at"] = time.time()
    receipt["elapsed_seconds"] = receipt["finished_at"] - receipt["started_at"]
    receipt["persisted"] = True
    write_json(artifacts / "receipt.json", receipt)
    store.publish(artifacts / "receipt.json", "receipt.json")
    print(json.dumps(receipt), flush=True)
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
