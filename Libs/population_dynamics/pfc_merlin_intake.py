"""Receive declared Merlin PFC results and apply the existing scientific consumers.

The receipt watched by the migration controller is published only after acceptance.
No simulation, dataset identity, threshold, or source numerical code is changed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import traceback

from merlin_io import collect, preview
from return_results import integrate
from population_dynamics.pfc_parameter_domain_quality import load_task
from population_dynamics.pfc_parameter_domain_summary import load_observation
from population_dynamics import pfc_parameter_task_dynamics as dynamics
from population_dynamics.provenance import sha256_file
from population_dynamics.resource_probe import probe

NUMERICAL_FILES = ("pfc_native_drive.py", "pfc_deterministic_replay.py", "pfc_kernel_parameters.py", "pfc_input_pairing.py")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        require(path.read_bytes() == data, "EXISTING_JSON_DIFFERS:" + str(path))
        return
    with path.open("xb") as stream:
        stream.write(data)

def check_expected(receipt, expected):
    for key, value in expected.items():
        require(key in receipt and type(receipt[key]) is type(value) and receipt[key] == value, "MERLIN_RECEIPT_MISMATCH:" + key)

def bound_source_file(archive, delivery, relative):
    rows = [row for row in delivery["files"] if row["path"] == relative]
    require(len(rows) == 1, "SOURCE_MANIFEST_ENTRY_REQUIRED:" + relative)
    path = archive / relative
    require(path.stat().st_size == rows[0]["bytes"] and sha256_file(path) == rows[0]["sha256"], "SOURCE_MANIFEST_FILE_CHANGED:" + relative)
    return path

def validate_episode(config, task, receipt):
    expected = {key: task[key] for key in ("index", "run_id", "trial_id", "xie_job")}
    expected.update(status="COMPLETE", exit_code=0, persisted=True,
                    compatibility_execution_id=f"merlin_{task['trial_id']}_pfc_{task['index']:04d}")
    check_expected(receipt, expected)
    require(isinstance(receipt.get("hostname"), str) and bool(receipt["hostname"]),
            "MERLIN_HOST_RECEIPT_REQUIRED")
    domain = Path(config["domain_base"])
    plan, row, recipe = load_task(domain / "prepared", task["index"])
    observation = domain / "observations" / f"task_{task['index']:04d}"
    entry = dict(path=str(observation), development_report_sha256=sha256_file(observation / "development5ms/report.json"), observation_report_sha256=sha256_file(observation / "observation1ms/report.json"))
    development, _, _, frozen, _ = load_observation(entry, row)
    require(frozen["observation_map"]["controls"]["commands"] == dynamics.CONTROL_COMMANDS, "PFC_CONTROL_COMMAND_SEMANTICS_MISMATCH")
    transitions = frozen["source_control_transitions"]
    ticks = [item["transition_tick"] for item in transitions]
    require(all(item["boundary_aligned"] is True for item in transitions) and ticks == list(dynamics.expected_control_transition_ticks(plan["task_protocol"], row["condition"])), "PFC_CONDITION_CONTROL_SCHEDULE_MISMATCH")
    archive = Path(development["source_archive"]).resolve()
    compatibility = receipt["compatibility_execution_id"]
    require(archive == (Path(task["source_root"]) / f"task_{task['index']:04d}" / ("run_" + compatibility)).resolve(), "MERLIN_SOURCE_ARCHIVE_MISMATCH")
    delivery_path = archive / "delivery.json"
    require(sha256_file(delivery_path) == development["source_delivery_sha256"], "MERLIN_SOURCE_DELIVERY_BINDING_MISMATCH")
    delivery = read_json(delivery_path)
    episode = row["identity"]["episode_id"]
    require(delivery["status"] == "ARCHIVED_PFC_SOURCE_EXECUTION" and delivery["result_status"] == "PFC_SOURCE_EXECUTION_COMPLETE" and delivery["source_execution_complete"] is True and delivery["exit_code"] == 0 and delivery["job_id"] == compatibility and delivery["case_id"] == episode, "MERLIN_SOURCE_DELIVERY_NOT_COMPLETE")
    execution_path = bound_source_file(archive, delivery, "execution.json")
    execution = read_json(execution_path)
    require(execution["status"] == "PFC_SOURCE_EXECUTION_COMPLETE" and execution["source_execution_complete"] is True and execution["exit_code"] == 0 and execution["job_id"] == compatibility and execution["case_id"] == episode and execution.get("node") in (None, receipt["hostname"]) and execution["config_sha256"] == row["config_sha256"], "MERLIN_SOURCE_EXECUTION_BINDING_MISMATCH")
    numerical = {}
    for name in NUMERICAL_FILES:
        returned = bound_source_file(archive, delivery, "Libs/population_dynamics/" + name)
        reference = Path(config["numerical_source"]) / name
        actual, original = sha256_file(returned), sha256_file(reference)
        require(actual == original, "PFC_NUMERICAL_SOURCE_CHANGED:" + name)
        numerical[name] = dict(returned_sha256=actual, frozen_sha256=original, frozen_path=str(reference))
    return dict(status="ACCEPTED_MERLIN_DEVELOPMENT_EPISODE", index=task["index"],
                run_id=receipt["run_id"], trial_id=receipt["trial_id"],
                xie_job=task["xie_job"], hostname=receipt["hostname"],
                identity=row["identity"], config_sha256=row["config_sha256"],
                observation_entry=entry, source_archive=str(archive),
                source_delivery_sha256=sha256_file(delivery_path),
                source_execution_sha256=sha256_file(execution_path),
                execution_host_provenance={"source_execution_node": execution.get("node"),
                    "actual_host_from_remote_receipt": receipt["hostname"],
                    "compatibility_id_is_not_slurm_job_id": True},
                source_control_transition_ticks=ticks, numerical_source=numerical,
                production_episodes=0, source_simulations=0, source_qualification_granted=False,
                scaling_qualified=False, scientific_parameter_domain_qualified=False)

def publish(source, target):
    """Publish one complete file without overwriting different existing evidence."""
    target.parent.mkdir(parents=True, exist_ok=True)
    size, digest = source.stat().st_size, sha256_file(source)
    if target.exists():
        require(target.stat().st_size == size and sha256_file(target) == digest,
                "EXISTING_PUBLICATION_DIFFERS:" + str(target))
        return
    descriptor, name = tempfile.mkstemp(prefix="." + target.name + "-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as destination, source.open("rb") as origin:
            shutil.copyfileobj(origin, destination, 8 * 1024 * 1024)
        require(temporary.stat().st_size == size and sha256_file(temporary) == digest,
                "PUBLICATION_COPY_CHANGED:" + str(target))
        try:
            os.link(temporary, target)
        except FileExistsError:
            require(target.stat().st_size == size and sha256_file(target) == digest,
                    "CONCURRENT_PUBLICATION_DIFFERS:" + str(target))
    finally:
        temporary.unlink(missing_ok=True)


def copy_chunks(receipt, origin, destination, *, keep_valid=False):
    count, errors = 0, []
    for index, row in enumerate(receipt["chunks"]):
        source = origin / f"{index:05d}.part"
        if not source.exists():
            continue
        try:
            require(source.stat().st_size == row["bytes"] and sha256_file(source) == row["sha256"],
                    "CACHED_CHUNK_CHANGED:" + str(source))
            publish(source, destination / source.name)
            count += 1
        except Exception as error:
            if not keep_valid:
                raise
            errors.append(dict(index=index, error_type=type(error).__name__, error=str(error)))
    return count, errors


def receive(config, task, output, scratch):
    target = Path(task["output_bundle"])
    watched_receipt = target.with_suffix(".receipt.json")
    receipt = None
    local = scratch / target.name
    try:
        if watched_receipt.exists():
            integration = read_json(watched_receipt)
            require(integration["status"] == "VERIFIED_AND_INTEGRATED"
                    and integration["bundle"] == str(target), "MERLIN_RETURN_NOT_INTEGRATED")
            receipt = read_json(target.with_suffix(".remote_receipt.json"))
            require(target.stat().st_size == receipt["result_bytes"]
                    and sha256_file(target) == receipt["result_sha256"], "RETURNED_BUNDLE_CHANGED")
            transferred = False
        else:
            receipt = preview(task["receipt_uri"])
            for key in ("index", "run_id", "trial_id", "xie_job"):
                require(receipt[key] == task[key], "RETURN_OWNERSHIP_MISMATCH:" + key)
            copy_chunks(receipt, target.with_name(target.name + ".chunks"),
                        local.with_name(local.name + ".chunks"))
            require(collect(task["receipt_uri"], local) == receipt,
                    "MERLIN_RECEIPT_CHANGED_DURING_TRANSFER")
            manifest = None
            if receipt.get("external_manifest_uri"):
                manifest = local.with_suffix(".manifest.json")
                write_json(manifest, preview(receipt["external_manifest_uri"]))
            require(probe(output / "resource_before_integrate.json", 12)["status"] == "PASS",
                    "PFC_MERLIN_INTAKE_RESOURCE_BLOCKED")
            # The helper writes its file-level receipt beside the node-local bundle.
            # The controller cannot see that receipt before scientific acceptance.
            integration = integrate(local, manifest)
            transferred = True
        accepted = validate_episode(config, task, receipt)
        accepted.update(recorded_utc=datetime.now(timezone.utc).isoformat(),
                        remote_receipt=receipt, receipt_uri=task["receipt_uri"],
                        returned_bundle=str(target), transferred_in_this_execution=transferred,
                        intake_job_id=os.environ["SLURM_JOB_ID"],
                        intake_implementation_sha256=sha256_file(Path(__file__)))
        write_json(output / "acceptance.json", accepted)
        if transferred:
            publish(local, target)
            publish(local.with_suffix(".remote_receipt.json"), target.with_suffix(".remote_receipt.json"))
            if manifest is not None:
                publish(manifest, target.with_suffix(".manifest.json"))
            gate = dict(integration, bundle=str(target),
                        scientific_acceptance=str((output / "acceptance.json").resolve()),
                        scientific_acceptance_sha256=sha256_file(output / "acceptance.json"))
            write_json(output / "controller_receipt.json", gate)
            publish(output / "controller_receipt.json", watched_receipt)
        return accepted
    except Exception as error:
        failure = dict(status="FAILED_MERLIN_INTAKE", index=task["index"],
                       error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if receipt is not None and local.with_name(local.name + ".chunks").exists():
            try:
                count, errors = copy_chunks(receipt, local.with_name(local.name + ".chunks"),
                    target.with_name(target.name + ".chunks"), keep_valid=True)
                failure.update(preserved_chunks=count, chunk_preservation_errors=errors)
            except Exception as preservation_error:
                failure["chunk_preservation_error"] = str(preservation_error)
        write_json(output / "failure.json", failure)
        raise


def run(config_path, output, scratch):
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(config_path)
    control = Path(config["control_validation"])
    require(sha256_file(Path(dynamics.__file__)) == read_json(control)["implementation_sha256"],
            "VALIDATED_CONTROL_IMPLEMENTATION_CHANGED")
    prior = read_json(config["prior_acceptance"])
    accepted = set(prior["accepted_task_indices"])
    require(len(accepted) == prior["accepted_episode_count"], "PRIOR_ACCEPTANCE_COUNT_MISMATCH")
    tasks = config["return_tasks"]
    require(len({t["index"] for t in tasks}) == len(tasks), "DUPLICATE_RETURN_TASK")
    require(not accepted.intersection(t["index"] for t in tasks), "TASK_ALREADY_ACCEPTED")
    require(probe(output / "resource.json", 12)["status"] == "PASS", "PFC_MERLIN_INTAKE_RESOURCE_BLOCKED")
    scratch.mkdir(parents=True, exist_ok=True)
    completed = []
    for task in tasks:
        task_output = output / f"task_{task['index']:04d}"
        task_output.mkdir()
        with tempfile.TemporaryDirectory(prefix=f"task_{task['index']:04d}-", dir=scratch) as temporary:
            result = receive(config, task, task_output, Path(temporary))
        accepted.add(result["index"])
        completed.append(dict(index=result["index"], acceptance=str(task_output / "acceptance.json")))
        progress = dict(status="ACCEPTED_RETURN_BATCH" if len(completed) == len(tasks) else "ACCEPTING_RETURN_BATCH",
                        accepted_episode_count=len(accepted), accepted_task_indices=sorted(accepted),
                        prior_acceptance=config["prior_acceptance"], new_accepted_episode_count=len(completed),
                        requested_return_count=len(tasks), completed=completed,
                        config=str(config_path), config_sha256=sha256_file(config_path),
                        control_validation=str(control), control_validation_sha256=sha256_file(control),
                        production_episodes=0, source_simulations=0, source_qualification_granted=False,
                        scaling_qualified=False, scientific_parameter_domain_qualified=False)
        path = output / "acceptance.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(path)
        print(json.dumps(dict(index=result["index"], accepted_episode_count=len(accepted))), flush=True)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output, args.scratch)


if __name__ == "__main__":
    main()
