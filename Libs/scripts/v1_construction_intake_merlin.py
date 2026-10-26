#!/usr/bin/env python3
"""Validate the nine completed V1 constructions in an actual Merlin CPU trial.

The staging command lists explicit files for the existing transfer tools. The
run command writes acceptance copies only under a new isolated output directory.
It never writes canonical acceptances or prepares subsequent reference jobs.
Existing acceptance_job/acceptance_step_id fields use merlin:<run>/<trial>
identifiers. Legacy source job IDs retain their original Slurm meaning. The explicit
Merlin-input commands preserve run/trial namespaces for C98/C99.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import yaml

from ca3_merlin_remote import memory_evidence
from stage_v1_merlin_block_inputs import module_files
from population_dynamics.benchmark_production import ProductionCohort
from population_dynamics.provenance import cpu_execution_provenance, freeze_json, sha256_file
from population_dynamics.v1_circuit_inputs import NATIVE_FILES, REFERENCE_FILES, bind_reference
from population_dynamics.v1_nest_input_replay import bound_json


TASKS = (
    ("recurrent", 95, 628376), ("recurrent", 96, 628377),
    ("recurrent", 97, 628378), ("recurrent", 98, 628379),
    ("recurrent", 99, 624318), ("inputs", 94, 626824),
    ("inputs", 95, 626825), ("inputs", 96, 626826), ("inputs", 97, 626827),
)
ARRAYS = {"recurrent": "624318", "inputs": "624319"}
RECURRENT_PAYLOADS = ("primary/report.json", "replay/report.json", "comparison/report.json",
                      *("primary/" + name for name in REFERENCE_FILES))
EXECUTION_KEYS = {"acceptance_job", "acceptance_host", "acceptance_step_id", "recorded_utc"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_identity(path: Path) -> dict:
    return dict(bytes=path.stat().st_size, sha256=sha256_file(path))


def checked_manifest(path: Path, expected_sha256: str) -> dict:
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
            "V1_INTAKE_BOUND_MANIFEST_SHA256_REQUIRED")
    require(sha256_file(path) == expected_sha256, "V1_INTAKE_MANIFEST_IDENTITY_DIFFERS:" + str(path))
    return read_json(path)


def accounting_row(accounting: dict, stage: str, circuit: int, source_job: int) -> dict:
    require((stage, circuit, source_job) in TASKS, "V1_INTAKE_TASK_OUTSIDE_NINE_CONSTRUCTIONS")
    require(accounting.get("returncode") == 0, "V1_INTAKE_SOURCE_ACCOUNTING_COMMAND_FAILED")
    array_member = ARRAYS[stage] + "_" + str(circuit)
    matches = []
    for line in accounting["stdout"].splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 5 and fields[:2] == [str(source_job), array_member]:
            matches.append(fields[:5])
    require(len(matches) == 1 and matches[0][2:4] == ["COMPLETED", "0:0"],
            "V1_INTAKE_SOURCE_ACCOUNTING_NOT_COMPLETE:" + array_member)
    elapsed = int(matches[0][4])
    require(elapsed >= 0, "V1_INTAKE_SOURCE_ELAPSED_INVALID")
    expected = dict(stage=stage, circuit=circuit, raw_job_id=str(source_job),
                    array_member=array_member, state="COMPLETED", exit_code="0:0",
                    elapsed_seconds=str(elapsed))
    declared = [row for row in accounting["matched_rows"]
                if row.get("stage") == stage and row.get("circuit") == circuit]
    require(declared == [expected], "V1_INTAKE_SOURCE_ACCOUNTING_RECEIPT_DIFFERS:" + array_member)
    return expected


@dataclass(frozen=True)
class BoundInputs:
    qualification_root: Path
    construction_root: Path
    source_root: Path
    cohort_manifest: Path
    cohort_manifest_sha256: str
    construction_manifest: Path
    construction_manifest_sha256: str
    accounting: dict

    def circuit(self, index: int) -> Path:
        return self.qualification_root / f"circuit_{index:03d}"

    def record_path(self, index: int, stage: str) -> Path:
        return self.construction_root / f"circuit_{index:03d}" / stage / "report.json"

    def record(self, index: int, stage: str) -> dict:
        value = read_json(self.record_path(index, stage))
        source = Path(value["archive"])
        require(source.is_absolute() and ".." not in source.parts
                and source.resolve().is_relative_to(self.source_root),
                "V1_INTAKE_SOURCE_ARCHIVE_OUTSIDE_DECLARED_ROOT")
        return value


def load_bound_inputs(args: argparse.Namespace) -> BoundInputs:
    q = args.qualification_root.resolve()
    cohort_path = args.cohort_manifest.resolve()
    require(cohort_path == q / "identities.json", "V1_INTAKE_CANONICAL_COHORT_MANIFEST_REQUIRED")
    checked_manifest(cohort_path, args.cohort_manifest_sha256)
    construction_path = args.construction_manifest.resolve()
    accounting = checked_manifest(construction_path, args.construction_manifest_sha256)
    rows = accounting.get("matched_rows", [])
    expected = {(stage, circuit, str(job)) for stage, circuit, job in TASKS}
    actual = {(row.get("stage"), row.get("circuit"), row.get("raw_job_id")) for row in rows}
    require(len(rows) == 9 and accounting.get("expected_rows") == 9 and actual == expected,
            "V1_INTAKE_EXACT_NINE_SOURCE_CONSTRUCTIONS_REQUIRED")
    for stage, circuit, source_job in TASKS:
        accounting_row(accounting, stage, circuit, source_job)
    return BoundInputs(q, args.construction_root.resolve(), args.source_root.resolve(),
                       cohort_path, args.cohort_manifest_sha256, construction_path,
                       args.construction_manifest_sha256, accounting)


def source_identity(record: dict) -> dict:
    """Keep legacy Slurm integers and preserve explicit Merlin source identities."""
    namespace = record.get("job_id_namespace", "slurm")
    if namespace == "slurm":
        value = str(record["job_id"])
        require(value.isascii() and value.isdecimal() and int(value) > 0,
                "V1_INTAKE_SOURCE_SLURM_IDENTITY_REQUIRED")
        return dict(source_job_id=int(value))
    require(namespace == "merlin_run_trial", "V1_INTAKE_SOURCE_IDENTITY_NAMESPACE_INVALID")
    execution = record.get("execution_context", {})
    require(execution.get("backend") == "merlin"
            and all(isinstance(execution.get(key), str) and execution[key]
                    and execution[key] == execution[key].strip() for key in ("run_id", "trial_id", "hostname"))
            and type(execution.get("allocated_cpus")) is int and execution["allocated_cpus"] > 0
            and type(execution.get("allocated_gpus")) is int and execution["allocated_gpus"] == 0,
            "V1_INTAKE_SOURCE_MERLIN_CPU_IDENTITY_REQUIRED")
    expected = "merlin:" + execution["run_id"] + ":" + execution["trial_id"]
    require(record.get("job_id") == expected, "V1_INTAKE_SOURCE_MERLIN_JOB_ID_DIFFERS")
    return dict(source_job_id=expected, source_job_id_namespace=namespace,
                source_execution_context=execution)


def construction_circuit_index(record: dict) -> int:
    identity = source_identity(record)
    key = "circuit_index" if "source_job_id_namespace" in identity else "array_task_id"
    value = record.get(key)
    require(type(value) is int, "V1_INTAKE_SOURCE_CIRCUIT_INDEX_REQUIRED")
    return value


def load_merlin_bound_inputs(args: argparse.Namespace) -> BoundInputs:
    q = args.qualification_root.resolve()
    cohort_path = args.cohort_manifest.resolve()
    require(cohort_path == q / "identities.json", "V1_INTAKE_CANONICAL_COHORT_MANIFEST_REQUIRED")
    checked_manifest(cohort_path, args.cohort_manifest_sha256)
    manifest_path = args.construction_manifest.resolve()
    manifest = checked_manifest(manifest_path, args.construction_manifest_sha256)
    rows = manifest.get("constructions")
    require(manifest.get("schema") == "v1_merlin_input_deliveries_v1"
            and isinstance(rows, list) and 1 <= len(rows) <= 2
            and all(isinstance(row, dict) and type(row.get("circuit_index")) is int
                    and row["circuit_index"] in (98, 99) for row in rows)
            and len({row["circuit_index"] for row in rows}) == len(rows),
            "V1_INTAKE_MERLIN_C98_C99_MANIFEST_REQUIRED")
    return BoundInputs(q, args.construction_root.resolve(), args.source_root.resolve(),
                       cohort_path, args.cohort_manifest_sha256, manifest_path,
                       args.construction_manifest_sha256, manifest)


def merlin_input_record(context: BoundInputs, row: dict) -> dict:
    circuit = row["circuit_index"]
    source = Path(row["archive"])
    require(source.is_absolute() and ".." not in source.parts,
            "V1_INTAKE_MERLIN_SOURCE_ARCHIVE_PATH_INVALID")
    delivery = checked_manifest(source / "delivery.json", row["delivery_sha256"])
    identity = source_identity(delivery)
    require(identity.get("source_job_id_namespace") == "merlin_run_trial"
            and identity["source_execution_context"] == row["execution_context"]
            and delivery.get("node") == row["execution_context"]["hostname"],
            "V1_INTAKE_MERLIN_SOURCE_MANIFEST_DIFFERS")
    parent = context.record(circuit, "recurrent")
    require(delivery.get("circuit") == parent["circuit"]
            and delivery.get("parent_job_id") == parent["job_id"]
            and delivery.get("parent_job_id_namespace") == "slurm"
            and Path(delivery["recurrent_reference"]) == Path(parent["archive"]),
            "V1_INTAKE_MERLIN_CIRCUIT_BINDING_DIFFERS")
    require(delivery["status"] == "PASS_SOURCE_INPUT_BUILD_REPEATABILITY"
            and delivery["exit_code"] == 0 and delivery["mode"] == "full"
            and delivery["production_episodes"] == 0 and delivery["source_simulations"] == 0,
            "V1_INTAKE_MERLIN_INPUT_DELIVERY_NOT_COMPLETE")
    return dict(status="COMPLETE_SOURCE_CONSTRUCTION", exit_code=0, stage="inputs",
                circuit_index=circuit, circuit=parent["circuit"], archive=str(source),
                job_id=identity["source_job_id"], job_id_namespace="merlin_run_trial",
                execution_context=identity["source_execution_context"], node=delivery["node"],
                parent_job_id=parent["job_id"], parent_job_id_namespace="slurm",
                delivery_sha256=row["delivery_sha256"], production_episodes=0,
                recorded_utc=delivery["recorded_utc"])


def delivery_rows(delivery: dict) -> dict[str, dict]:
    rows = {}
    for row in delivery["files"]:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and relative.parts
                and row["path"] not in rows, "V1_INTAKE_DELIVERY_PATH_INVALID_OR_DUPLICATE")
        require(type(row.get("bytes")) is int and row["bytes"] >= 0
                and re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is not None,
                "V1_INTAKE_DELIVERY_FILE_IDENTITY_INVALID")
        rows[row["path"]] = dict(bytes=row["bytes"], sha256=row["sha256"])
    return rows


def staging_files(context: BoundInputs) -> dict[str, dict]:
    """Read metadata; retain producer identities for payload transfer verification."""
    files = {}

    def add_actual(path: Path) -> None:
        identity = file_identity(path)
        if str(path) in files:
            require(files[str(path)] == identity, "V1_INTAKE_STAGING_IDENTITY_CONFLICT")
        files[str(path)] = identity

    for name in ("frozen_campaign.yaml", "qualification.json", "identities.csv", "identities.json"):
        add_actual(context.qualification_root / name)
    add_actual(context.construction_manifest)
    for circuit in range(94, 100):
        add_actual(context.circuit(circuit) / "circuit.yaml")
        add_actual(context.record_path(circuit, "recurrent"))
    add_actual(context.circuit(94) / "recurrent_acceptance.json")
    for stage, circuit, source_job in TASKS:
        add_actual(context.record_path(circuit, stage))
        record = context.record(circuit, stage)
        source = Path(record["archive"])
        require(record["job_id"] == str(source_job), "V1_INTAKE_SOURCE_RECORD_JOB_DIFFERS")
        add_actual(source / "delivery.json")
        require(files[str(source / "delivery.json")]["sha256"] == record["delivery_sha256"],
                "V1_INTAKE_SOURCE_DELIVERY_DIFFERS")
        rows = delivery_rows(read_json(source / "delivery.json"))
        selected = RECURRENT_PAYLOADS if stage == "recurrent" else tuple(rows)
        for relative in selected:
            require(relative in rows, "V1_INTAKE_REQUIRED_DELIVERY_PAYLOAD_MISSING:" + relative)
            path = str(source / relative)
            if path in files:
                require(files[path] == rows[relative], "V1_INTAKE_STAGING_IDENTITY_CONFLICT")
            files[path] = rows[relative]
        acceptance = context.circuit(circuit) / ("recurrent_acceptance.json" if stage == "recurrent"
                                                 else "input_acceptance.json")
        if acceptance.exists():
            add_actual(acceptance)
        if stage == "inputs":
            parent = context.record(circuit, "recurrent")
            add_actual(Path(parent["archive"]) / "delivery.json")
            require(files[str(Path(parent["archive"]) / "delivery.json")]["sha256"] == parent["delivery_sha256"],
                    "V1_INTAKE_PARENT_DELIVERY_DIFFERS")
    return files


def staging_plan(context: BoundInputs, resource_shape: Path | None = None) -> dict:
    files = staging_files(context)
    # Reuse the established relative-import closure, including lazy imports.
    # Unrelated actively edited modules must not invalidate this intake snapshot.
    package = Path(sys.modules[ProductionCohort.__module__].__file__).resolve().parent
    runtime = [Path(__file__).resolve(), Path(sys.modules[memory_evidence.__module__].__file__).resolve(),
               Path(sys.modules[module_files.__module__].__file__).resolve()]
    runtime.extend(sorted(module_files(package, ["benchmark_production", "provenance",
                                               "v1_circuit_inputs", "v1_nest_input_replay"])))
    runtime.append(package / "test_v1_construction_intake_merlin.py")
    if resource_shape is not None:
        runtime.append(resource_shape.resolve())
    for path in runtime:
        files[str(path)] = file_identity(path)
    return dict(status="STAGING_PLAN_ONLY", files=files, file_count=len(files),
                declared_bytes=sum(row["bytes"] for row in files.values()),
                tasks=[dict(stage=stage, circuit=circuit, source_job_id=job) for stage, circuit, job in TASKS],
                recurrent_payloads_per_archive=list(RECURRENT_PAYLOADS),
                input_payload_selection="all_delivery_manifest_entries", source_simulations=0,
                production_episodes=0, future_reference_preparations=0)


def acceptance_execution(execution: dict) -> dict:
    require(execution.get("backend") == "merlin" and execution.get("run_id")
            and execution.get("trial_id") and execution.get("hostname"),
            "V1_INTAKE_ACTUAL_MERLIN_EXECUTION_REQUIRED")
    return dict(acceptance_job="merlin:" + execution["run_id"],
                acceptance_host=execution["hostname"],
                acceptance_step_id="merlin:" + execution["trial_id"])


def validate_recurrent(context: BoundInputs, circuit: int, cohort: ProductionCohort,
                       execution: dict, scratch: Path) -> dict:
    job = next(job for stage, index, job in TASKS if stage == "recurrent" and index == circuit)
    row = accounting_row(context.accounting, "recurrent", circuit, job)
    record = context.record(circuit, "recurrent")
    require(record["status"] == "COMPLETE_SOURCE_CONSTRUCTION" and record["exit_code"] == 0
            and record["array_job_id"] == "624318" and record["array_task_id"] == circuit
            and record["job_id"] == str(job), "V1_INTAKE_RECURRENT_SOURCE_RECORD_DIFFERS")
    source = Path(record["archive"])
    config_path = context.circuit(circuit) / "circuit.yaml"
    config = yaml.safe_load(config_path.read_text())
    config_sha = sha256_file(config_path)
    identity = cohort.identity(circuit, 0, 0)
    cohort.verify_declaration(config, identity)
    require(config_sha == record["circuit"]["config_sha256"], "V1_INTAKE_RECURRENT_CONFIG_DIFFERS")
    primary, delivery = bound_json(source, "primary/report.json")
    replay, replay_delivery = bound_json(source, "replay/report.json")
    comparison, comparison_delivery = bound_json(source, "comparison/report.json")
    require(delivery == replay_delivery == comparison_delivery
            and sha256_file(source / "delivery.json") == record["delivery_sha256"]
            and delivery["exit_code"] == 0 and delivery["mode"] == "full"
            and delivery["job_id"] == record["job_id"]
            and delivery["status"].startswith("PASS_SOURCE_RECURRENT_"),
            "V1_INTAKE_RECURRENT_DELIVERY_DIFFERS")
    for value in (primary, replay):
        require(value["status"] == "ORIGINAL_RECURRENT_FULL_BUILD_COMPLETE" and value["exit_code"] == 0
                and value["mode"] == "full" and value["original_config"] == config
                and value["original_config_sha256"] == config_sha and value["training_seeds_used"] == [],
                "V1_INTAKE_RECURRENT_SOURCE_QUALIFICATION_FAILED")
        require(value["rng"]["builder_seed"] == identity["connectivity_seed"]
                and value["rng"]["glif_assignment_seed"] == identity["connectivity_seed"] + 1000000,
                "V1_INTAKE_RECURRENT_DATASET_RNG_DIFFERS")
        valid = value["validation"]
        require(valid["nodes"] == valid["connectivity"]["nodes"]
                == valid["connectivity"]["largest_strong_component"] == 230924
                and valid["connectivity"]["strong_components"] == 1,
                "V1_INTAKE_FULL_COUPLED_RECURRENT_GRAPH_REQUIRED")
        require(valid["all_edge_rows_verified"] == valid["edge_rows"] == valid["connectivity"]["source_edge_rows"]
                and all(valid[key] for key in ("endpoints_and_nsyns_preserved", "full_scale_gate_applied",
                    "node_assignment_membership_verified", "source_and_target_predicates_verified")),
                "V1_INTAKE_RECURRENT_SOURCE_EDGE_EVIDENCE_FAILED")
    require(primary["rng"] == replay["rng"]
            and comparison["status"] == "ORIGINAL_RECURRENT_REPLAY_EQUAL" and comparison["exit_code"] == 0
            and comparison["all_native_datasets_and_tables_equal"] and len(comparison["files"]) == 8
            and all(row["equal"] for row in comparison["files"])
            and comparison["input_report_sha256"] == [sha256_file(source / name)
                  for name in ("primary/report.json", "replay/report.json")],
            "V1_INTAKE_RECURRENT_REPLAY_EVIDENCE_FAILED")
    consumer_primary, binding = bind_reference(source, scratch / "consumer")
    require(consumer_primary == primary and binding["delivery_sha256"] == record["delivery_sha256"],
            "V1_INTAKE_RECURRENT_EXISTING_CONSUMER_BINDING_DIFFERS")
    cohort.require_current()
    return dict(status="PASS_RECURRENT_SOURCE_DELIVERY_AND_REPLAY", source=str(source),
        source_job_id=job, array_job_id="624318", array_task_id=circuit,
        **acceptance_execution(execution),
        acceptance_scope="bounded_metadata_and_node_file_authentication_using_the_existing_input_consumer",
        consumer_binding=binding, validation=primary["validation"],
        connectivity_seed=identity["connectivity_seed"], glif_assignment_seed=identity["connectivity_seed"] + 1000000,
        rng_states_equal=True, edge_archives_rehashed_during_this_intake=False,
        source_replay_comparison=dict(status=comparison["status"], datasets_and_tables_equal=True,
            files_compared=8, comparison_sha256=sha256_file(source / "comparison/report.json")),
        primary_elapsed_seconds=primary["elapsed_seconds"], replay_elapsed_seconds=replay["elapsed_seconds"],
        child_peak_rss_bytes=delivery["child_peak_rss_bytes"], scheduler_elapsed_seconds=int(row["elapsed_seconds"]),
        completed_recurrent_realizations=1, completed_full_circuits=0,
        source_simulations=0, production_episodes=0, scope="recurrent_graph_and_node_inputs_only",
        recorded_utc=datetime.now(timezone.utc).isoformat())


def validate_inputs(context: BoundInputs, circuit: int, cohort: ProductionCohort,
                    execution: dict, accepted_parent: dict) -> dict:
    job = next(job for stage, index, job in TASKS if stage == "inputs" and index == circuit)
    accounting_row(context.accounting, "inputs", circuit, job)
    recurrent, inputs = context.record(circuit, "recurrent"), context.record(circuit, "inputs")
    require(inputs["job_id"] == str(job) and inputs["array_job_id"] == "624319"
            and inputs["parent_job_id"] == recurrent["job_id"], "V1_INTAKE_INPUT_SOURCE_PARENT_DIFFERS")
    return validate_input_delivery(context, circuit, cohort, execution, accepted_parent, recurrent, inputs)


def validate_input_delivery(context: BoundInputs, circuit: int, cohort: ProductionCohort,
                            execution: dict, accepted_parent: dict, recurrent: dict, inputs: dict) -> dict:
    config_path = context.circuit(circuit) / "circuit.yaml"
    config = yaml.safe_load(config_path.read_text())
    config_sha = sha256_file(config_path)
    cohort.verify_declaration(config, cohort.identity(circuit, 0, 0))
    for item in (recurrent, inputs):
        require(item["status"] == "COMPLETE_SOURCE_CONSTRUCTION" and item["exit_code"] == 0
                and item["circuit"]["config_sha256"] == config_sha
                and construction_circuit_index(item) == circuit,
                "V1_INTAKE_INPUT_CONSTRUCTION_RECORD_DIFFERS")
    require(inputs["parent_job_id"] == recurrent["job_id"], "V1_INTAKE_INPUT_SOURCE_PARENT_DIFFERS")
    source_fields = source_identity(inputs)
    source, parent = Path(inputs["archive"]), Path(recurrent["archive"])
    parent_sha = sha256_file(parent / "delivery.json")
    require(accepted_parent["status"] == "PASS_RECURRENT_SOURCE_DELIVERY_AND_REPLAY"
            and accepted_parent["source_job_id"] == int(recurrent["job_id"])
            and parent_sha == recurrent["delivery_sha256"] == accepted_parent["consumer_binding"]["delivery_sha256"],
            "V1_INTAKE_ACCEPTED_RECURRENT_PARENT_REQUIRED")
    primary, delivery = bound_json(source, "primary/report.json")
    replay, replay_delivery = bound_json(source, "replay/report.json")
    comparison, comparison_delivery = bound_json(source, "comparison/report.json")
    delivery_sha = sha256_file(source / "delivery.json")
    require(delivery_sha == inputs["delivery_sha256"] and delivery == replay_delivery == comparison_delivery
            and delivery["exit_code"] == 0 and delivery["status"] == "PASS_SOURCE_INPUT_BUILD_REPEATABILITY"
            and delivery["job_id"] == inputs["job_id"] and delivery["mode"] == "full"
            and Path(delivery["recurrent_reference"]) == parent,
            "V1_INTAKE_INPUT_DELIVERY_DIFFERS")
    if "source_job_id_namespace" in source_fields:
        require(source_identity(delivery) == source_fields and delivery["circuit"] == inputs["circuit"],
                "V1_INTAKE_MERLIN_EXECUTION_CONTEXT_DIFFERS")
    expected_files = (*NATIVE_FILES, *REFERENCE_FILES)
    require(len(expected_files) == 20 and comparison["status"] == "ORIGINAL_CIRCUIT_INPUT_REPLAY_EQUAL"
            and comparison["exit_code"] == 0 and comparison["all_native_files_equal"]
            and comparison["bound_reference_and_rng_states_equal"] and comparison["binding_differences"] == []
            and len(comparison["files"]) == len(expected_files)
            and {row["path"] for row in comparison["files"]} == set(expected_files)
            and all(row["equal"] for row in comparison["files"]),
            "V1_INTAKE_COMPLETE_TWENTY_FILE_INPUT_REPLAY_REQUIRED")
    for item in (primary, replay):
        if "source_job_id_namespace" in source_fields:
            require(item.get("execution_context") == source_fields["source_execution_context"],
                    "V1_INTAKE_MERLIN_EXECUTION_CONTEXT_DIFFERS")
        require(item["status"] == "ORIGINAL_CIRCUIT_INPUTS_COMPLETE" and item["exit_code"] == 0
                and item["original_config"] == config and item["original_config_sha256"] == config_sha
                and item["mode"] == "full" and item["stored_v1_files_preserved"]
                and item["imported_v1"]["nodes"] == 230924 and item["reference"]["delivery_sha256"] == parent_sha
                and item["rng"]["lgn_selection_python_seed"] == config["builder_seed"]
                and item["training_seeds_used"] == [], "V1_INTAKE_INPUT_SOURCE_QUALIFICATION_FAILED")
        for population, units in (("lgn", 17400), ("bkg", 1)):
            valid = item["validation"][population]
            require(valid["source_units"] == units and valid["all_edge_rows_verified"] == valid["edge_rows"]
                    and valid["identities_nsyns_and_target_predicates_verified"],
                    "V1_INTAKE_INPUT_SOURCE_EDGE_EVIDENCE_FAILED:" + population)
    require(primary["rng"] == replay["rng"] and primary["reference"]["files"] == replay["reference"]["files"],
            "V1_INTAKE_INPUT_RNG_OR_REFERENCE_BINDING_DIFFERS")
    rows = delivery_rows(delivery)
    verified_bytes = 0
    for relative, expected in rows.items():
        path = source / relative
        require(file_identity(path) == expected, "V1_INTAKE_INPUT_DELIVERY_PAYLOAD_CHANGED:" + relative)
        verified_bytes += expected["bytes"]
    require(sha256_file(source / "delivery.json") == delivery_sha, "V1_INTAKE_INPUT_DELIVERY_CHANGED_DURING_READ")
    # The original combined consumer checked freshness after reference preparation.
    # This independent acceptance preserves that gate before any acceptance write.
    cohort.require_current()
    identity = acceptance_execution(execution)
    result = dict(status="PASS_SOURCE_INPUT_DELIVERY_AND_REPLAY", **source_fields,
        recurrent_job_id=int(recurrent["job_id"]), recorded_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source), config_sha256=config_sha, delivery_sha256=delivery_sha,
        primary_report_sha256=sha256_file(source / "primary/report.json"),
        replay_report_sha256=sha256_file(source / "replay/report.json"),
        comparison_sha256=sha256_file(source / "comparison/report.json"), files_compared=20,
        all_native_files_equal=True, bound_reference_and_rng_states_equal=True, validation=primary["validation"],
        verified_files=len(rows), verified_bytes=verified_bytes,
        primary_elapsed_seconds=primary["elapsed_seconds"], replay_elapsed_seconds=replay["elapsed_seconds"],
        child_peak_rss_bytes=delivery["child_peak_rss_bytes"], acceptance_job=identity["acceptance_job"],
        acceptance_step_id=identity["acceptance_step_id"], source_simulations=0, production_episodes=0,
        scope="complete_source_input_delivery_authentication_and_replay_intake")

    if "source_job_id_namespace" in source_fields:
        result.update(circuit_index=circuit, source_manifest=dict(path=str(context.construction_manifest),
                      sha256=context.construction_manifest_sha256))
    return result

def save_acceptance(canonical: Path, destination: Path, acceptance: dict, stage: str) -> tuple[dict, bool]:
    if canonical.exists():
        prior = read_json(canonical)
        if stage == "recurrent":
            require(set(prior) - EXECUTION_KEYS == set(acceptance) - EXECUTION_KEYS,
                    "V1_INTAKE_EXISTING_RECURRENT_ACCEPTANCE_FIELDS_DIFFER")
            keys = set(acceptance) - EXECUTION_KEYS
        else:
            keys = ("source_job_id", "recurrent_job_id", "config_sha256", "delivery_sha256", "status")
        if acceptance.get("source_job_id_namespace") == "merlin_run_trial":
            keys = tuple(keys) + ("source_job_id_namespace", "source_execution_context", "circuit_index")
        for key in keys:
            require(key in prior and prior[key] == acceptance[key], "V1_INTAKE_EXISTING_ACCEPTANCE_DIFFERS:" + key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as target:
            target.write(canonical.read_bytes())
        return prior, True
    freeze_json(destination, acceptance)
    return acceptance, False


def run_intake(args: argparse.Namespace) -> dict:
    context = load_bound_inputs(args)
    execution = cpu_execution_provenance()
    acceptance_execution(execution)
    require(execution["allocated_gpus"] == 0, "V1_INTAKE_CPU_ONLY_ALLOCATION_REQUIRED")
    shape = read_json(args.resource_shape)
    require(shape.get("name") == "head" and shape.get("gpu") == 0
            and type(shape.get("cpu")) is int and 0 < shape["cpu"] <= execution["allocated_cpus"],
            "V1_INTAKE_RESOURCE_SHAPE_DIFFERS")
    memory = memory_evidence(shape, required_available_gib=4)
    output = args.output.resolve()
    for protected in (context.qualification_root, context.construction_root, context.source_root):
        require(not output.is_relative_to(protected), "V1_INTAKE_ISOLATED_OUTPUT_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    bindings = dict(cohort_manifest=str(context.cohort_manifest),
                    cohort_manifest_sha256=context.cohort_manifest_sha256,
                    construction_manifest=str(context.construction_manifest),
                    construction_manifest_sha256=context.construction_manifest_sha256)
    report = dict(status="FAILED_V1_CONSTRUCTION_INTAKE", exit_code=1, execution=execution,
                  memory_evidence=memory, manifest_bindings=bindings, acceptances=[],
                  source_simulations=0, production_episodes=0, canonical_acceptances_written=0,
                  future_reference_preparations=0, started_utc=datetime.now(timezone.utc).isoformat())
    parents = {}
    try:
        cohort = ProductionCohort.load(context.qualification_root / "frozen_campaign.yaml", "v1_visual",
            context.qualification_root / "qualification.json", context.qualification_root / "identities.csv")
        for stage, circuit, source_job in TASKS:
            report["current_task"] = dict(stage=stage, circuit=circuit, source_job_id=source_job)
            if stage == "recurrent":
                with tempfile.TemporaryDirectory(prefix="recurrent_consumer_", dir=output) as scratch:
                    acceptance = validate_recurrent(context, circuit, cohort, execution, Path(scratch))
                name = "recurrent_acceptance.json"
            else:
                parent = parents.get(circuit)
                if parent is None:
                    parent = read_json(context.circuit(circuit) / "recurrent_acceptance.json")
                acceptance = validate_inputs(context, circuit, cohort, execution, parent)
                name = "input_acceptance.json"
            cohort.require_current()
            destination = output / f"circuit_{circuit:03d}" / name
            canonical = context.circuit(circuit) / name
            saved, reused = save_acceptance(canonical, destination, acceptance, stage)
            if stage == "recurrent":
                parents[circuit] = saved
            artifact = dict(stage=stage, circuit=circuit, source_job_id=source_job,
                acceptance=str(destination), canonical_target=str(canonical),
                acceptance_identity=file_identity(destination), canonical_record_reused=reused)
            receipt = dict(status="PASS_MERLIN_CONSTRUCTION_INTAKE", execution=execution,
                consumer_execution_backend="merlin", manifest_bindings=bindings,
                source_accounting=accounting_row(context.accounting, stage, circuit, source_job),
                source_accounting_observed_utc=context.accounting["observed_utc"],
                artifact=artifact, source_simulations=0, production_episodes=0,
                future_reference_preparations=0, recorded_utc=datetime.now(timezone.utc).isoformat())
            freeze_json(destination.with_name(stage + "_intake_execution.json"), receipt)
            report["acceptances"].append(artifact)
        report.update(status="PASS_NINE_V1_CONSTRUCTION_INTAKES", exit_code=0,
                      validated_recurrent_acceptances=5, validated_input_acceptances=4)
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error),
                      exit_code=130 if isinstance(error, KeyboardInterrupt) else 1)
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        freeze_json(output / "report.json", report)
    return report


def staging_merlin_inputs(context: BoundInputs, resource_shape: Path | None = None) -> dict:
    files = {}

    def add(path: Path) -> None:
        value = file_identity(path)
        require(str(path) not in files or files[str(path)] == value, "V1_INTAKE_STAGING_IDENTITY_CONFLICT")
        files[str(path)] = value

    for name in ("frozen_campaign.yaml", "qualification.json", "identities.csv", "identities.json"):
        add(context.qualification_root / name)
    add(context.construction_manifest)
    for row in context.accounting["constructions"]:
        circuit = row["circuit_index"]
        record = merlin_input_record(context, row)
        for path in (context.circuit(circuit) / "circuit.yaml",
                     context.circuit(circuit) / "recurrent_acceptance.json",
                     context.record_path(circuit, "recurrent")):
            add(path)
        add(Path(context.record(circuit, "recurrent")["archive"]) / "delivery.json")
        source = Path(record["archive"])
        add(source / "delivery.json")
        for relative, identity in delivery_rows(read_json(source / "delivery.json")).items():
            path = str(source / relative)
            require(path not in files or files[path] == identity, "V1_INTAKE_STAGING_IDENTITY_CONFLICT")
            files[path] = identity
        canonical = context.circuit(circuit) / "input_acceptance.json"
        if canonical.exists():
            add(canonical)
        canonical_record = context.record_path(circuit, "inputs")
        if canonical_record.exists():
            add(canonical_record)
    package = Path(sys.modules[ProductionCohort.__module__].__file__).resolve().parent
    runtime = [Path(__file__).resolve(), Path(sys.modules[memory_evidence.__module__].__file__).resolve(),
               Path(sys.modules[module_files.__module__].__file__).resolve(),
               package / "test_v1_construction_intake_merlin.py"]
    runtime.extend(sorted(module_files(package, ["benchmark_production", "provenance",
                                               "v1_circuit_inputs", "v1_nest_input_replay"])))
    if resource_shape is not None:
        runtime.append(resource_shape.resolve())
    for path in runtime:
        add(path)
    return dict(status="STAGING_PLAN_ONLY", files=files, file_count=len(files),
                declared_bytes=sum(row["bytes"] for row in files.values()),
                circuits=[row["circuit_index"] for row in context.accounting["constructions"]],
                source_simulations=0, production_episodes=0, future_reference_preparations=0)


def run_merlin_intake(args: argparse.Namespace) -> dict:
    context = load_merlin_bound_inputs(args)
    execution = cpu_execution_provenance()
    acceptance_execution(execution)
    require(execution["allocated_gpus"] == 0, "V1_INTAKE_CPU_ONLY_ALLOCATION_REQUIRED")
    shape = read_json(args.resource_shape)
    require(shape.get("name") == "head" and shape.get("gpu") == 0
            and type(shape.get("cpu")) is int and 0 < shape["cpu"] <= execution["allocated_cpus"],
            "V1_INTAKE_RESOURCE_SHAPE_DIFFERS")
    memory = memory_evidence(shape, required_available_gib=4)
    output = args.output.resolve()
    protected = [context.qualification_root, context.construction_root, context.source_root]
    protected.extend(Path(row["archive"]) for row in context.accounting["constructions"])
    require(all(not output.is_relative_to(path) for path in protected), "V1_INTAKE_ISOLATED_OUTPUT_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_MERLIN_V1_INPUT_INTAKES", exit_code=1, execution=execution,
                  memory_evidence=memory, acceptances=[], validated_input_acceptances=0,
                  source_manifest=dict(path=str(context.construction_manifest), sha256=context.construction_manifest_sha256),
                  cohort_manifest=dict(path=str(context.cohort_manifest), sha256=context.cohort_manifest_sha256),
                  source_simulations=0, production_episodes=0, canonical_acceptances_written=0,
                  canonical_records_written=0, future_reference_preparations=0,
                  started_utc=datetime.now(timezone.utc).isoformat())
    try:
        cohort = ProductionCohort.load(context.qualification_root / "frozen_campaign.yaml", "v1_visual",
            context.qualification_root / "qualification.json", context.qualification_root / "identities.csv")
        for row in context.accounting["constructions"]:
            circuit = row["circuit_index"]
            report["current_circuit"] = circuit
            record = merlin_input_record(context, row)
            parent = read_json(context.circuit(circuit) / "recurrent_acceptance.json")
            acceptance = validate_input_delivery(context, circuit, cohort, execution, parent,
                                                context.record(circuit, "recurrent"), record)
            cohort.require_current()
            destination = output / f"circuit_{circuit:03d}" / "input_acceptance.json"
            canonical = context.circuit(circuit) / "input_acceptance.json"
            _, reused = save_acceptance(canonical, destination, acceptance, "inputs")
            canonical_record = context.record_path(circuit, "inputs")
            local_record = destination.parent / "inputs/report.json"
            if canonical_record.exists():
                prior = read_json(canonical_record)
                require(prior == record, "V1_INTAKE_EXISTING_MERLIN_CONSTRUCTION_RECORD_DIFFERS")
                local_record.parent.mkdir(parents=True, exist_ok=True)
                with local_record.open("xb") as target:
                    target.write(canonical_record.read_bytes())
            else:
                freeze_json(local_record, record)
            artifact = dict(circuit_index=circuit, **source_identity(record), acceptance=str(destination),
                            canonical_target=str(canonical), acceptance_identity=file_identity(destination),
                            canonical_record_reused=reused, construction_record=str(local_record),
                            canonical_construction_record=str(canonical_record),
                            construction_record_identity=file_identity(local_record))
            freeze_json(destination.with_name("inputs_intake_execution.json"),
                        dict(status="PASS_MERLIN_CONSTRUCTION_INTAKE", execution=execution,
                             source_manifest=report["source_manifest"], source_delivery=row,
                             artifact=artifact, source_simulations=0, production_episodes=0))
            report["acceptances"].append(artifact)
            report["validated_input_acceptances"] += 1
        report.update(status="PASS_MERLIN_V1_INPUT_INTAKES", exit_code=0)
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error),
                      exit_code=130 if isinstance(error, KeyboardInterrupt) else 1)
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        freeze_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("staging", "run", "staging-merlin-inputs", "run-merlin-inputs"))
    for name in ("qualification-root", "construction-root", "source-root", "cohort-manifest", "construction-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("cohort-manifest-sha256", "construction-manifest-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--resource-shape", type=Path, help="JSON object for the actual requested CPU role")
    parser.add_argument("--output", type=Path, help="New isolated directory; never a canonical output root")
    args = parser.parse_args()
    if args.command in ("run", "run-merlin-inputs"):
        if args.output is None or args.resource_shape is None:
            parser.error("run commands require --output and --resource-shape")
        result = run_intake(args) if args.command == "run" else run_merlin_intake(args)
    elif args.command == "staging-merlin-inputs":
        result = staging_merlin_inputs(load_merlin_bound_inputs(args), args.resource_shape)
    else:
        result = staging_plan(load_bound_inputs(args), args.resource_shape)
    print(json.dumps(result, indent=2))
    return result.get("exit_code", 0)


if __name__ == "__main__":
    raise SystemExit(main())
