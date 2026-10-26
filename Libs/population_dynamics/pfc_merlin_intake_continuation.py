"""Continue scientific intake of persisted PFC episodes on one Merlin CPU job."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from . import pfc_merlin_scientific_intake as intake

SOURCE_RUN_ID = "6de397acc866aadd"
SOURCE_TRIAL_ID = "435906314"
INDEX_RUN_ID = "c295fedae001740e"
ARRAY_JOBS = ("643651", "643652", "643653")
TARGET = 3000
SCHEMA = "pfc_merlin_scientific_intake_aggregation_v1"
NETWORK_FILESYSTEMS = {"nfs", "nfs4", "cifs", "smb2"}


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_bytes())


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
    os.replace(temporary, path)


def indices(value, field):
    require(isinstance(value, list), field + "_LIST_REQUIRED")
    require(all(type(index) is int and 0 <= index < TARGET for index in value),
            field + "_INDEX_INVALID")
    require(len(value) == len(set(value)), field + "_DUPLICATE_INDEX")
    return set(value)


def validate_prior(prior):
    require(prior.get("schema") == SCHEMA and prior.get("status") == "PASS", "PRIOR_NOT_PASS")
    accepted = indices(prior.get("accepted_task_indices"), "PRIOR")
    require(type(prior.get("accepted_episode_count")) is int
            and prior["accepted_episode_count"] == len(accepted), "PRIOR_COUNT_MISMATCH")
    require(prior.get("production_episodes") == 0 and prior.get("source_simulations") == 0,
            "PRIOR_SCIENTIFIC_SCOPE_CHANGED")
    require(prior.get("failures", []) == [], "PRIOR_CONTAINS_FAILURES")
    require(all(prior.get(field) is False for field in
                ("source_qualification_granted", "scaling_qualified", "scientific_parameter_domain_qualified")),
            "PRIOR_QUALIFICATION_SCOPE_CHANGED")
    return accepted


def template_binding(config, template):
    require(config.get("expected_source_run_id") == SOURCE_RUN_ID
            and config.get("expected_source_trial_id") == SOURCE_TRIAL_ID, "SOURCE_IDENTITY_NOT_DECLARED")
    require(type(config.get("batch_size")) is int and 1 <= config["batch_size"] <= 48,
            "BATCH_SIZE_INVALID")
    require(type(config.get("poll_seconds")) is int and config["poll_seconds"] > 0,
            "POLL_SECONDS_INVALID")
    require(config.get("target") == TARGET, "CONTINUATION_TARGET_CHANGED")
    tasks = template.get("return_tasks")
    require(isinstance(tasks, list) and bool(tasks), "TEMPLATE_TASKS_REQUIRED")
    indices([task["index"] for task in tasks], "TEMPLATE")
    canonical = tasks[0]
    suffix = f"/pfc/task_{canonical['index']:04d}/receipt.json"
    require(canonical["receipt_uri"].endswith(suffix), "TEMPLATE_RECEIPT_PATH_INVALID")
    source = canonical["receipt_uri"][:-len(suffix)]
    require(source.startswith("hdfs://") and source.endswith("/" + SOURCE_RUN_ID),
            "TEMPLATE_SOURCE_PATH_INVALID")
    compact = source.rsplit("/", 1)[0] + "/" + INDEX_RUN_ID + "/pfc_compact/progress.json"
    require(config.get("completion_index_uri") in (source + "/pfc/progress.json", compact),
            "COMPLETION_INDEX_PATH_INVALID")
    for task in tasks:
        index = task["index"]
        require(task.get("run_id") == SOURCE_RUN_ID and task.get("trial_id") == SOURCE_TRIAL_ID,
                "TEMPLATE_SOURCE_IDENTITY_CHANGED")
        require(task["xie_job"] == ARRAY_JOBS[index // 1000] + "_" + str(index % 1000),
                "TEMPLATE_XIE_MAPPING_CHANGED")
        require(task["source_root"] == template["domain_base"].rstrip("/") + "/source_results",
                "TEMPLATE_SOURCE_ROOT_CHANGED")
        require(task["receipt_uri"] == source + f"/pfc/task_{index:04d}/receipt.json"
                and task["bundle_uri"] == source + f"/pfc/task_{index:04d}/result.tar.gz",
                "TEMPLATE_TASK_PATH_CHANGED")
    return source, canonical


def select_ready(index, accepted, source, batch_size):
    require(index.get("run_id") == SOURCE_RUN_ID and index.get("trial_id") == SOURCE_TRIAL_ID,
            "COMPLETION_SOURCE_IDENTITY_CHANGED")
    if "original_progress_uri" in index:
        require(index["original_progress_uri"] == source + "/pfc/progress.json",
                "COMPLETION_ORIGINAL_PROGRESS_PATH_CHANGED")
    rows = index.get("completed")
    require(isinstance(rows, list), "COMPLETION_ROWS_REQUIRED")
    seen = indices([row["index"] for row in rows], "COMPLETION")
    for row in rows:
        require(row.get("run_id") == SOURCE_RUN_ID and row.get("trial_id") == SOURCE_TRIAL_ID,
                "COMPLETION_TASK_IDENTITY_CHANGED")
        require(row.get("persisted") is True and type(row.get("exit_code")) is int and row["exit_code"] == 0,
                "COMPLETION_TASK_NOT_PERSISTED_SUCCESS")
        require(row.get("checkpoint_path") == source + f"/pfc/task_{row['index']:04d}",
                "COMPLETION_TASK_PATH_CHANGED")
    if "status" in index:
        require(index["status"] in ("RUNNING", "COMPLETE", "PASS"), "COMPLETION_STATUS_INVALID")
    return sorted(seen - accepted)[:batch_size]


def batch_config(template, canonical, source, selected, prior_path):
    value = copy.deepcopy(template)
    value["prior_acceptance"] = str(prior_path)
    value["return_tasks"] = []
    for index in selected:
        task = copy.deepcopy(canonical)
        task.update(index=index, xie_job=ARRAY_JOBS[index // 1000] + "_" + str(index % 1000),
                    receipt_uri=source + f"/pfc/task_{index:04d}/receipt.json",
                    bundle_uri=source + f"/pfc/task_{index:04d}/result.tar.gz")
        value["return_tasks"].append(task)
    return value


def filesystem(path):
    return subprocess.check_output(["stat", "-f", "-c", "%T", str(path)], text=True).strip()


def owned_paths(template, tasks):
    domain = Path(template["domain_base"])
    require(domain.is_absolute(), "CLEANUP_DOMAIN_NOT_ABSOLUTE")
    allowed = (domain / "source_results", domain / "observations")
    for root in allowed:
        require(root.is_dir() and root.resolve() == root and not root.is_symlink(),
                "CLEANUP_ROOT_NOT_CANONICAL")
        require(filesystem(root) not in NETWORK_FILESYSTEMS, "CLEANUP_ROOT_NOT_PRIVATE_LOCAL")
    result = []
    for task in tasks:
        index = task["index"]
        require(Path(task["source_root"]) == allowed[0], "CLEANUP_SOURCE_ROOT_CHANGED")
        for root in allowed:
            path = root / f"task_{index:04d}"
            require(not path.exists() and not path.is_symlink(), "CLEANUP_TASK_PATH_PREEXISTS:" + str(path))
            result.append(path)
    return result


def cleanup_owned(paths, allowed_roots):
    cleaned = []
    for path in paths:
        require(path.parent in allowed_roots and path.name.startswith("task_")
                and path.name[5:].isdigit() and len(path.name) == 9, "CLEANUP_PATH_OUTSIDE_ALLOWLIST")
        require(path.parent.resolve() == path.parent and not path.is_symlink(), "CLEANUP_PATH_NOT_CANONICAL")
        require(filesystem(path.parent) not in NETWORK_FILESYSTEMS, "CLEANUP_PATH_NOT_PRIVATE_LOCAL")
        if path.exists():
            require(path.is_dir(), "CLEANUP_TASK_NOT_DIRECTORY")
            shutil.rmtree(path)
            cleaned.append(str(path))
    return cleaned


def merge_batch(state, batch, selected, batch_path, batch_checkpoint, run_id, trial_id):
    execution = batch.get("execution", {})
    require(execution.get("run_id") == run_id and execution.get("trial_id") == trial_id,
            "BATCH_MERLIN_EXECUTION_IDENTITY_CHANGED")
    old = indices(state["accepted_task_indices"], "CUMULATIVE")
    completed = batch.get("completed")
    require(isinstance(completed, list), "BATCH_COMPLETION_ROWS_REQUIRED")
    new = indices([row["index"] for row in completed], "BATCH_COMPLETED")
    require(new <= set(selected) and not new.intersection(old), "BATCH_COMPLETED_SELECTION_CHANGED")
    require(batch.get("accepted_episode_count") == len(old | new)
            and indices(batch.get("accepted_task_indices"), "BATCH_ACCEPTED") == old | new,
            "BATCH_ACCEPTANCE_COUNT_CHANGED")
    require(batch.get("new_accepted_episode_count") == len(new), "BATCH_NEW_COUNT_CHANGED")
    require(batch.get("source_simulations") == 0 and batch.get("production_episodes") == 0,
            "BATCH_SCIENTIFIC_SCOPE_CHANGED")
    require(all(batch.get(field) is False for field in
                ("source_qualification_granted", "scaling_qualified", "scientific_parameter_domain_qualified")),
            "BATCH_QUALIFICATION_SCOPE_CHANGED")
    failures = batch.get("failures")
    require(isinstance(failures, list), "BATCH_FAILURES_REQUIRED")
    failed = indices([row["index"] for row in failures], "BATCH_FAILED")
    require(not failed.intersection(new) and failed <= set(selected), "BATCH_FAILURE_SELECTION_CHANGED")
    require(new | failed == set(selected), "BATCH_UNRESOLVED_TASKS")
    require(batch.get("status") == ("FAILED" if failures or batch.get("publication_error") else "PASS"),
            "BATCH_STATUS_CHANGED")
    for row in completed:
        require(row.get("acceptance_uri") == batch_checkpoint + f"/pfc_scientific_intake/task_{row['index']:04d}_acceptance.json",
                "BATCH_ACCEPTANCE_URI_CHANGED")
        require(isinstance(row.get("acceptance_sha256"), str) and len(row["acceptance_sha256"]) == 64,
                "BATCH_ACCEPTANCE_HASH_REQUIRED")
    state["accepted_task_indices"] = sorted(old | new)
    state["accepted_episode_count"] = len(old | new)
    state["new_accepted_episode_count"] = len(old | new) - state["prior_accepted_episode_count"]
    state["failures"].extend(failures)
    if batch.get("publication_error"):
        state["failures"].append(dict(batch["publication_error"], context="batch_publication"))
    state["batches"].append(dict(acceptance=str(batch_path), acceptance_sha256=sha256_file(batch_path),
                                acceptance_uri=batch_checkpoint + "/pfc_scientific_intake/"
                                + ("acceptance_recovery.json" if batch.get("recovered_publication") else "acceptance.json"),
                                completed=completed, failures=failures, run_id=run_id, trial_id=trial_id,
                                batch_checkpoint_path=batch_checkpoint, new_accepted_episode_count=len(new)))
    state["status"] = "FAILED" if state["failures"] else ("PASS" if len(old | new) == TARGET else "RUNNING")
    return state


def publish(work, checkpoint, name, value, mutable=False):
    path = work / name
    atomic_json(path, value)
    intake.hdfs("-put", *(["-f"] if mutable else []), path, checkpoint + "/" + name)
    return path


def run(config_path, work):
    work = Path(work)
    run_id, trial_id = os.environ.get("RH2_JOB_RUN_ID"), os.environ.get("ARNOLD_TRIAL_ID")
    require(bool(run_id) and bool(trial_id) and os.environ.get("ARNOLD_MONITOR_ROLE") == "head"
            and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0" and not os.environ.get("CUDA_VISIBLE_DEVICES"),
            "MERLIN_CPU_ALLOCATION_REQUIRED")
    require(work.is_absolute() and work.resolve() == work and not work.is_symlink(), "WORK_NOT_PRIVATE_CANONICAL")
    require(filesystem(work) not in NETWORK_FILESYSTEMS, "WORK_NOT_LOCAL")
    config = read_json(config_path)
    template = read_json(config["template_config"])
    source, canonical = template_binding(config, template)
    initial_path = Path(config["prior_acceptance"])
    initial = read_json(initial_path)
    accepted = validate_prior(initial)
    require(initial_path != Path(template["prior_acceptance"]), "INITIAL_PRIOR_PATH_MUST_BE_DISTINCT")
    checkpoint_root = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/")
    require(checkpoint_root.startswith("hdfs://"), "REAL_CHECKPOINT_REQUIRED")
    checkpoint = checkpoint_root + "/pfc_intake_continuation"
    intake.hdfs("-mkdir", "-p", checkpoint)
    state = dict(schema=SCHEMA, status="RUNNING", accepted_task_indices=sorted(accepted),
        accepted_episode_count=len(accepted), prior_accepted_episode_count=len(accepted),
        prior_acceptance=str(initial_path), prior_acceptance_sha256=sha256_file(initial_path),
        new_accepted_episode_count=0, batches=[], failures=copy.deepcopy(initial.get("failures", [])),
        source_simulations=0, production_episodes=0,
        source_qualification_granted=False, scaling_qualified=False, scientific_parameter_domain_qualified=False,
        execution=dict(backend="merlin", run_id=run_id, trial_id=trial_id, hostname=socket.gethostname(),
                       allocated_cpus=os.environ.get("ARNOLD_HEAD_CPU"), allocated_gpus=0),
        checkpoint_path=checkpoint, implementation_sha256=sha256_file(Path(__file__)), target=TARGET,
        template_config_sha256=sha256_file(Path(config["template_config"])))
    prior_path, counter = initial_path, 0
    allowed = (Path(template["domain_base"]) / "source_results", Path(template["domain_base"]) / "observations")
    while len(accepted) < TARGET:
        poll_path = work / "completion_index.json"
        poll_path.unlink(missing_ok=True)
        try:
            index = intake.download_json(config["completion_index_uri"], poll_path)
            selected = select_ready(index, accepted, source, config["batch_size"])
            state["completion_index_sha256"] = sha256_file(poll_path)
            state["recorded_utc"] = datetime.now(timezone.utc).isoformat()
            if not selected:
                state["status"] = "WAITING_FOR_PERSISTED_SOURCE"
                publish(work, checkpoint, "progress.json", state, mutable=True)
                time.sleep(config["poll_seconds"])
                continue
            counter += 1
            batch_root = work / f"batch_{counter:04d}"
            batch_root.mkdir()
            value = batch_config(template, canonical, source, selected, prior_path)
            owned = owned_paths(template, value["return_tasks"])
            config_file = batch_root / "config.json"
            atomic_json(config_file, value)
            batch_work = batch_root / "results"
            batch_work.mkdir()
            batch_checkpoint = checkpoint + f"/batch_{counter:04d}"
            os.environ["RH2_CHECKPOINT_HDFS"] = batch_checkpoint
            try:
                batch = intake.run(config_file, batch_work)
            except Exception as error:
                progress_path = batch_work / "progress.json"
                batch = read_json(progress_path) if progress_path.exists() else dict(
                    completed=[], failures=[], accepted_task_indices=sorted(accepted),
                    accepted_episode_count=len(accepted), new_accepted_episode_count=0,
                    source_simulations=0, production_episodes=0, source_qualification_granted=False,
                    scaling_qualified=False, scientific_parameter_domain_qualified=False)
                resolved = {row["index"] for row in batch["completed"] + batch["failures"]}
                batch["failures"].extend(dict(index=index, error_type=type(error).__name__,
                                              error=str(error), context="batch_publication")
                                         for index in selected if index not in resolved)
                batch["status"] = "FAILED"
                batch["recovered_publication"] = True
                batch.setdefault("execution", dict(run_id=run_id, trial_id=trial_id))
                if not batch["failures"]:
                    batch["publication_error"] = dict(error_type=type(error).__name__, error=str(error))
                atomic_json(batch_work / "acceptance.json", batch)
                intake.hdfs("-put", batch_work / "acceptance.json",
                            batch_checkpoint + "/pfc_scientific_intake/acceptance_recovery.json")
            finally:
                os.environ["RH2_CHECKPOINT_HDFS"] = checkpoint_root
            state = merge_batch(state, batch, selected, batch_work / "acceptance.json",
                                batch_checkpoint, run_id, trial_id)
            accepted = set(state["accepted_task_indices"])
            publish(work, checkpoint, f"batch_{counter:04d}_cumulative.json", state)
            publish(work, checkpoint, "progress.json", state, mutable=True)
            completed_paths = [path for path in owned
                               if int(path.name[5:]) in {row["index"] for row in batch["completed"]}]
            state["batches"][-1]["cleaned_private_paths"] = cleanup_owned(completed_paths, allowed)
            if state["status"] == "FAILED":
                publish(work, checkpoint, "acceptance.json", state)
                return state
            prior = copy.deepcopy(state)
            prior["status"] = "PASS"
            prior_path = work / f"accepted_prior_{counter:04d}.json"
            atomic_json(prior_path, prior)
        except Exception as error:
            state["status"] = "FAILED"
            state["failures"].append(dict(error_type=type(error).__name__, error=str(error),
                                          context="continuation", batch_number=counter))
            publish(work, checkpoint, "progress.json", state, mutable=True)
            publish(work, checkpoint, "acceptance.json", state)
            return state
    state["status"] = "PASS"
    publish(work, checkpoint, "progress.json", state, mutable=True)
    publish(work, checkpoint, "acceptance.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=False)
    report = run(args.config, args.work.resolve())
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
