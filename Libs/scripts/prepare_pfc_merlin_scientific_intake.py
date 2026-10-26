"""Prepare a small Merlin intake request from a verified staged entrypoint."""
from __future__ import annotations

import argparse
import ast
import base64
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex

FINAL_EXEC = ('exec python3 -u -m population_dynamics.pfc_merlin_scientific_intake '
              '--config "$NRNSIM_MERLIN_INTAKE_WORK/config.json" '
              '--work "$NRNSIM_MERLIN_INTAKE_WORK/results"')
ARRAY_JOBS = ("643651", "643652", "643653")
MAX_REQUEST_BYTES = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(path.read_bytes())


def task_indices(values, label):
    require(isinstance(values, list), label + "_LIST_REQUIRED")
    require(all(type(value) is int and 0 <= value < 3000 for value in values),
            label + "_INDEX_OUT_OF_RANGE")
    require(len(set(values)) == len(values), label + "_DUPLICATE_INDEX")
    return set(values)


def parse_indices(value):
    require(isinstance(value, str) and bool(value), "SELECTED_INDICES_REQUIRED")
    fields = value.split(",")
    require(all(re.fullmatch(r"[0-9]+", field) for field in fields), "SELECTED_INVALID_INDEX")
    result = [int(field) for field in fields]
    task_indices(result, "SELECTED")
    return result


def source_xie_job(index):
    return ARRAY_JOBS[index // 1000] + "_" + str(index % 1000)


def runtime_preparation(payload):
    """Generate input restoration after the unchanged base preparation body."""
    return ('import base64, hashlib, json, os, pathlib, subprocess\n'
            + 'payload = ' + repr(payload) + '\n'
            + 'final_exec = ' + repr(FINAL_EXEC) + '\n'
            + """
root = pathlib.Path(os.environ["NRNSIM_MERLIN_INTAKE_WORK"])
bootstrap = pathlib.Path(os.environ["NRNSIM_MERLIN_INTAKE_BOOTSTRAP"])
data = (bootstrap / "base_entrypoint.sh").read_bytes()
if hashlib.sha256(data).hexdigest() != payload["base_sha256"]:
    raise ValueError("BASE_ENTRYPOINT_CHANGED")
text = data.decode("utf-8")
if text.count(final_exec) != 1:
    raise ValueError("BASE_FINAL_EXEC_NOT_UNIQUE")
body, tail = text.split(final_exec)
if tail.strip():
    raise ValueError("BASE_TRAILING_COMMANDS")
preparation = bootstrap / "base_preparation.sh"
preparation.write_text(body, encoding="utf-8")
subprocess.run(["bash", str(preparation)], check=True)
for role in ("prior", "config"):
    data = base64.b64decode(payload[role + "_base64"], validate=True)
    if hashlib.sha256(data).hexdigest() != payload[role + "_sha256"]:
        raise ValueError("PREPARATION_PAYLOAD_CHANGED:" + role)
    target = pathlib.Path(payload["prior_path"]) if role == "prior" else root / "config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if role == "prior" and target.exists() and target.read_bytes() != data:
        raise ValueError("PRIOR_CANONICAL_PATH_CHANGED")
    target.write_bytes(data)
    if hashlib.sha256(target.read_bytes()).hexdigest() != payload[role + "_sha256"]:
        raise ValueError("EFFECTIVE_INPUT_CHANGED:" + role)
effective = json.loads((root / "config.json").read_bytes())
if effective["prior_acceptance"] != payload["prior_path"]:
    raise ValueError("EFFECTIVE_PRIOR_BINDING_CHANGED")
if [row["index"] for row in effective["return_tasks"]] != payload["selected_indices"]:
    raise ValueError("EFFECTIVE_TASK_SELECTION_CHANGED")
""")


def prepare(template, prior_acceptance, completion_index, indices, name, output):
    template, prior_acceptance = Path(template).resolve(), Path(prior_acceptance).resolve()
    completion_index, output = Path(completion_index).resolve(), Path(output).resolve()
    require(isinstance(name, str) and len(name) <= 90
            and re.fullmatch(r"mrln_[A-Za-z0-9_]+", name) is not None, "MRLN_JOB_NAME_REQUIRED")
    selected = parse_indices(indices) if isinstance(indices, str) else list(indices)
    require(bool(selected), "SELECTED_INDICES_REQUIRED")
    selected_set = task_indices(selected, "SELECTED")
    prior_data = prior_acceptance.read_bytes()
    prior = json.loads(prior_data)
    require(prior.get("status") == "PASS", "PRIOR_ACCEPTANCE_NOT_PASS")
    require(prior.get("schema") == "pfc_merlin_scientific_intake_aggregation_v1",
            "PRIOR_ACCEPTANCE_SCHEMA_REQUIRED")
    accepted = task_indices(prior.get("accepted_task_indices"), "PRIOR")
    require(type(prior.get("accepted_episode_count")) is int
            and prior["accepted_episode_count"] == len(accepted), "PRIOR_ACCEPTANCE_COUNT_MISMATCH")
    require(type(prior.get("production_episodes")) is int and prior["production_episodes"] == 0,
            "PRIOR_PRODUCTION_EPISODES_NOT_ZERO")
    require(not selected_set.intersection(accepted), "SELECTED_TASK_ALREADY_ACCEPTED")
    config = read_json(template / "config.json")
    require(prior_acceptance != Path(config["prior_acceptance"]).resolve(),
            "PRIOR_CANONICAL_PATH_MUST_BE_DISTINCT")
    tasks = config.get("return_tasks")
    require(isinstance(tasks, list) and bool(tasks), "TEMPLATE_TASKS_REQUIRED")
    task_indices([row["index"] for row in tasks], "TEMPLATE")
    canonical = tasks[0]
    run_id, trial_id = canonical["run_id"], canonical["trial_id"]
    require(isinstance(run_id, str) and bool(run_id) and isinstance(trial_id, str) and bool(trial_id),
            "TEMPLATE_SOURCE_IDENTITY_REQUIRED")
    suffix = f"/pfc/task_{canonical['index']:04d}/receipt.json"
    require(canonical["receipt_uri"].endswith(suffix), "TEMPLATE_SOURCE_CHECKPOINT_INVALID")
    source = canonical["receipt_uri"][:-len(suffix)]
    require(source.startswith("hdfs://") and source.endswith("/" + run_id),
            "TEMPLATE_SOURCE_CHECKPOINT_INVALID")
    require(canonical["source_root"] == config["domain_base"].rstrip("/") + "/source_results",
            "TEMPLATE_SOURCE_ROOT_INVALID")
    for row in tasks:
        index = row["index"]
        require(row["run_id"] == run_id and row["trial_id"] == trial_id,
                "TEMPLATE_SOURCE_IDENTITY_MISMATCH")
        require(row["source_root"] == canonical["source_root"], "TEMPLATE_SOURCE_ROOT_MISMATCH")
        require(row["xie_job"] == source_xie_job(index), "TEMPLATE_XIE_IDENTITY_MISMATCH")
        checkpoint = source + f"/pfc/task_{index:04d}"
        require(row["receipt_uri"] == checkpoint + "/receipt.json"
                and row["bundle_uri"] == checkpoint + "/result.tar.gz",
                "TEMPLATE_SOURCE_CHECKPOINT_MISMATCH")
    readiness_data = completion_index.read_bytes()
    ready = json.loads(readiness_data)
    require(ready.get("run_id") == run_id, "READY_SOURCE_RUN_MISMATCH")
    require(ready.get("trial_id") == trial_id, "READY_SOURCE_TRIAL_MISMATCH")
    require(ready.get("original_progress_uri") == source + "/pfc/progress.json",
            "READY_PROGRESS_CHECKPOINT_MISMATCH")
    rows = ready.get("completed")
    require(isinstance(rows, list), "READY_COMPLETED_LIST_REQUIRED")
    ready_indices = task_indices([row["index"] for row in rows], "READY")
    for row in rows:
        require(row.get("persisted") is True and type(row.get("exit_code")) is int
                and row["exit_code"] == 0, "READY_TASK_NOT_PERSISTED_COMPLETE")
        require(row.get("run_id") == run_id, "READY_TASK_RUN_MISMATCH")
        require(row.get("trial_id") == trial_id, "READY_TASK_TRIAL_MISMATCH")
        require(row.get("checkpoint_path") == source + f"/pfc/task_{row['index']:04d}",
                "READY_TASK_CHECKPOINT_MISMATCH")
    require(selected_set <= ready_indices, "SELECTED_TASK_NOT_READY")
    upload = read_json(template / "entrypoint_upload.json")
    base_data = (template / "entrypoint.sh").read_bytes()
    require(type(upload.get("bytes")) is int and len(base_data) == upload["bytes"]
            and sha256(base_data) == upload.get("sha256"), "TEMPLATE_ENTRYPOINT_CHANGED")
    require(isinstance(upload.get("hdfs_path"), str) and upload["hdfs_path"].startswith("hdfs://"),
            "TEMPLATE_ENTRYPOINT_HDFS_URI_REQUIRED")
    base = base_data.decode("utf-8")
    require(base.count(FINAL_EXEC) == 1, "TEMPLATE_FINAL_EXEC_NOT_UNIQUE")
    require(not base.split(FINAL_EXEC)[1].strip(), "TEMPLATE_TRAILING_COMMANDS")
    request = read_json(template / "request_template.json")
    roles = request["resource_config"]["arnold_resource_config"]["roles"]
    require(len(roles) == 1 and roles[0].get("name") == "head" and roles[0].get("num") == 1
            and roles[0].get("cpu") == 2 and roles[0].get("gpu") == 0
            and roles[0].get("memory_mb") == 4096 and roles[0].get("gpu_type") == "CPU_ONLY",
            "TEMPLATE_INTAKE_RESOURCE_MISMATCH")
    image = request["job_config"]["job_template_config"]["image_meta"]
    require(isinstance(image.get("image_url"), str) and bool(image["image_url"])
            and image.get("image_source") == "url", "TEMPLATE_IMAGE_REQUIRED")
    effective = copy.deepcopy(config)
    effective["prior_acceptance"] = str(prior_acceptance)
    effective["return_tasks"] = []
    for index in selected:
        row = copy.deepcopy(canonical)
        row.update(index=index, xie_job=source_xie_job(index),
                   receipt_uri=source + f"/pfc/task_{index:04d}/receipt.json",
                   bundle_uri=source + f"/pfc/task_{index:04d}/result.tar.gz")
        for field, suffix in (("output_bundle", ".tar.gz"), ("ready_evidence", ".json")):
            if field in row:
                row[field] = str(Path(row[field]).with_name(f"pfc_{index:04d}_{run_id}{suffix}"))
        effective["return_tasks"].append(row)
    effective["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    effective["validation_scope"] = "Disjoint persisted main-source episodes; reuse verified staging; no simulation"
    config_data = json_bytes(effective)
    payload = dict(base_sha256=upload["sha256"], prior_path=str(prior_acceptance), selected_indices=selected,
                   prior_base64=base64.b64encode(prior_data).decode(), prior_sha256=sha256(prior_data),
                   config_base64=base64.b64encode(config_data).decode(), config_sha256=sha256(config_data))
    runtime = runtime_preparation(payload)
    ast.parse(runtime, filename="runtime_preparation.py")
    entrypoint = ('#!/bin/bash\nset -euo pipefail\n'
        "export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
        'export NRNSIM_MERLIN_INTAKE_WORK=/tmp/neuronsim-pfc-scientific-intake-$RH2_JOB_RUN_ID\n'
        'export NRNSIM_MERLIN_INTAKE_BOOTSTRAP=/tmp/neuronsim-pfc-intake-bootstrap-$RH2_JOB_RUN_ID\n'
        'export PYTHONPATH="$NRNSIM_MERLIN_INTAKE_WORK:$NRNSIM_MERLIN_INTAKE_WORK/Libs"\n'
        'mkdir "$NRNSIM_MERLIN_INTAKE_BOOTSTRAP"\n'
        + 'hdfs dfs -get ' + shlex.quote(upload["hdfs_path"])
        + ' "$NRNSIM_MERLIN_INTAKE_BOOTSTRAP/base_entrypoint.sh"\n'
        + "python3 - <<'PREPARE_INTAKE'\n" + runtime + 'PREPARE_INTAKE\n' + FINAL_EXEC + '\n')
    request["name"] = name
    request["description"] = (f"mrln_ Validate {len(selected)} disjoint persisted PFC episodes; "
                              f"prior {len(accepted)}; no source simulations.")
    request["job_config"]["job_template_config"]["entrypoint_full_script"] = entrypoint
    request_data = json_bytes(request)
    require(len(entrypoint.encode()) < MAX_REQUEST_BYTES and len(request_data) < MAX_REQUEST_BYTES,
            "MERLIN_REQUEST_ENVELOPE_TOO_LARGE")
    execution = dict(status="PREPARED_NOT_SUBMITTED", name=name, selected_indices=selected,
                 task_indices=selected, selected_episode_count=len(selected),
                 prior_accepted_episode_count=len(accepted), prior_acceptance=str(prior_acceptance),
                 prior_acceptance_sha256=sha256(prior_data), completion_index=str(completion_index),
                 completion_index_sha256=sha256(readiness_data), source_run_id=run_id, source_trial_id=trial_id,
                 template=str(template), template_entrypoint_hdfs_uri=upload["hdfs_path"],
                 template_entrypoint_sha256=upload["sha256"], effective_config_sha256=sha256(config_data),
                 request_bytes=len(request_data), entrypoint_bytes=len(entrypoint.encode()),
                 source_simulations=0, production_episodes=0)
    check = dict(check_point="create", node_type="merlin_job_run", job_run_context=request)
    require(not output.exists(), "PREPARATION_OUTPUT_EXISTS")
    output.mkdir(parents=True)
    for name_, data in (("config.json", config_data), ("prior_acceptance.json", prior_data),
                        ("entrypoint.sh", entrypoint.encode()), ("runtime_preparation.py", runtime.encode()),
                        ("request.json", request_data), ("check.json", json_bytes(check)),
                        ("execution.json", json_bytes(execution))):
        with (output / name_).open("xb") as stream:
            stream.write(data)
    return {"output": str(output), "request": str(output / "request.json"),
            "check": str(output / "check.json"), "selected_episode_count": len(selected)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--prior-acceptance", type=Path, required=True)
    parser.add_argument("--completion-index", type=Path, required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.template, args.prior_acceptance, args.completion_index,
                     args.indices, args.name, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
