#!/usr/bin/env python3
"""Prepare the forty existing V1 reference conditions on an actual Merlin CPU job.

The default consumes accepted C94-C97 constructions. Explicit --circuits
98 99 can consume isolated, validated Merlin input-intake candidates. It never performs
construction intake, simulation, GPU execution, or scheduler submission. Returned
files are isolated integration candidates. The gpu/node/native_python fields in
prepared controls retain their original reference-execution plan semantics;
actual Merlin execution is recorded separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml

from population_dynamics.benchmark_production import ProductionCohort
from population_dynamics.provenance import cpu_execution_provenance, freeze_json, sha256_file
from population_dynamics.v1_execution_recipe import require_same_recipe
from population_dynamics.v1_nest_input_replay import bound_json
from stage_v1_merlin_block_inputs import module_files
from v1_construction_intake_merlin import source_identity, construction_circuit_index


CIRCUITS = (94, 95, 96, 97)
CONDITIONS = tuple(range(10))
# These are the actual arguments' YAML values for legacy jobs
# 643964/643966/643968/643970. They are not a Merlin resource allocation.
LEGACY_GPU_PLAN = dict(node="cpu04", partition="cpu")
LEGACY_GPU_MEMORY_GIB = 16


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "V1_REFERENCE_JSON_OBJECT_REQUIRED:" + str(path))
    return value


def file_identity(path: Path) -> dict:
    return dict(bytes=path.stat().st_size, sha256=sha256_file(path))


def legacy_execution_plan(path: Path) -> tuple[dict, int]:
    value = yaml.safe_load(path.read_text())
    gpu = {key: value["gpu"][key] for key in ("node", "partition")}
    memory = value["memory_gib"]["gpu"]
    require(gpu == LEGACY_GPU_PLAN and type(memory) is int and memory == LEGACY_GPU_MEMORY_GIB,
            "V1_REFERENCE_ORIGINAL_FOUR_JOB_RESOURCE_PLAN_REQUIRED")
    return gpu, memory


def source_control(path: Path, qualification_root: Path) -> dict:
    control = read_json(path)
    require(control["circuit"] == str(qualification_root / "circuit_000"),
            "V1_REFERENCE_C0_CONTROL_REQUIRED")
    require([row["condition_index"] for row in control["records"]] == list(CONDITIONS),
            "V1_REFERENCE_ALL_TEN_SOURCE_CONDITIONS_REQUIRED")
    require(Path(control["detector"]) == path.parent / "detect_resources.py",
            "V1_REFERENCE_SOURCE_DETECTOR_PATH_DIFFERS")
    return control


def selected_circuits(args: argparse.Namespace) -> tuple[int, ...]:
    values = tuple(getattr(args, "circuits", None) or CIRCUITS)
    require(len(values) == len(set(values)) and all(type(value) is int and 94 <= value <= 99 for value in values),
            "V1_REFERENCE_DISTINCT_SUPPORTED_CIRCUITS_REQUIRED")
    return values


def input_paths(qualification_root: Path, circuit: int, intake_output: Path | None = None) -> tuple[Path, Path]:
    if intake_output is not None and circuit in (98, 99):
        root = intake_output / f"circuit_{circuit:03d}"
        return root / "inputs/report.json", root / "input_acceptance.json"
    root = qualification_root / "construction_expansion_20261017" / f"circuit_{circuit:03d}"
    return root / "inputs/report.json", qualification_root / f"circuit_{circuit:03d}" / "input_acceptance.json"


def construction_records(qualification_root: Path, circuit: int,
                         intake_output: Path | None = None) -> tuple[dict, dict]:
    root = qualification_root / "construction_expansion_20261017" / f"circuit_{circuit:03d}"
    record_path, _ = input_paths(qualification_root, circuit, intake_output)
    return read_json(root / "recurrent/report.json"), read_json(record_path)

def accepted_construction(qualification_root: Path, circuit: int, cohort: ProductionCohort,
                          intake_output: Path | None = None) -> tuple[dict, dict, dict]:
    base = qualification_root / f"circuit_{circuit:03d}"
    config_path = base / "circuit.yaml"
    config = yaml.safe_load(config_path.read_text())
    config_sha = sha256_file(config_path)
    cohort.verify_declaration(config, cohort.identity(circuit, 0, 0))
    recurrent, inputs = construction_records(qualification_root, circuit, intake_output)
    for item in (recurrent, inputs):
        require(item["status"] == "COMPLETE_SOURCE_CONSTRUCTION" and item["exit_code"] == 0
                and item["circuit"]["config_sha256"] == config_sha and construction_circuit_index(item) == circuit,
                "V1_REFERENCE_CONSTRUCTION_SOURCE_IDENTITY_DIFFERS")
    require(inputs["parent_job_id"] == recurrent["job_id"], "V1_REFERENCE_INPUT_PARENT_DIFFERS")
    recurrent_path = base / "recurrent_acceptance.json"
    _, input_path = input_paths(qualification_root, circuit, intake_output)
    input_identity = source_identity(inputs)
    accepted_recurrent, accepted_input = read_json(recurrent_path), read_json(input_path)
    require(accepted_recurrent["status"] == "PASS_RECURRENT_SOURCE_DELIVERY_AND_REPLAY"
            and accepted_recurrent["source_job_id"] == int(recurrent["job_id"])
            and accepted_recurrent["source"] == recurrent["archive"]
            and accepted_recurrent["consumer_binding"]["delivery_sha256"] == recurrent["delivery_sha256"],
            "V1_REFERENCE_EXISTING_RECURRENT_ACCEPTANCE_REQUIRED")
    require(accepted_input["status"] == "PASS_SOURCE_INPUT_DELIVERY_AND_REPLAY"
            and all(accepted_input.get(key) == value for key, value in input_identity.items())
            and accepted_input["recurrent_job_id"] == int(recurrent["job_id"])
            and accepted_input["source"] == inputs["archive"]
            and accepted_input["config_sha256"] == config_sha
            and accepted_input["delivery_sha256"] == inputs["delivery_sha256"],
            "V1_REFERENCE_EXISTING_INPUT_ACCEPTANCE_REQUIRED")
    for acceptance in (accepted_recurrent, accepted_input):
        require(acceptance["source_simulations"] == 0 and acceptance["production_episodes"] == 0,
                "V1_REFERENCE_ACCEPTANCE_SCOPE_DIFFERS")
    # Authenticate the accepted source identities through their small receipts.
    # Payload authentication and source replay were completed by the intake job.
    for item in (recurrent, inputs):
        require(sha256_file(Path(item["archive"]) / "delivery.json") == item["delivery_sha256"],
                "V1_REFERENCE_ACCEPTED_SOURCE_DELIVERY_CHANGED")
    if "source_job_id_namespace" in input_identity:
        require(accepted_input.get("circuit_index") == circuit
                and source_identity(read_json(Path(inputs["archive"]) / "delivery.json")) == input_identity,
                "V1_REFERENCE_MERLIN_INPUT_SOURCE_IDENTITY_DIFFERS")
    proof = dict(circuit_index=circuit, config_identity=file_identity(config_path),
        recurrent_acceptance=dict(path=str(recurrent_path), **file_identity(recurrent_path)),
        input_acceptance=dict(path=str(input_path), **file_identity(input_path)),
        source_recurrent_job_id=int(recurrent["job_id"]), source_input_job_id=input_identity["source_job_id"])
    if "source_job_id_namespace" in input_identity:
        proof.update(source_input_job_id_namespace=input_identity["source_job_id_namespace"],
                     source_input_execution_context=input_identity["source_execution_context"])
    return recurrent, inputs, proof


def write_text_candidate(destination: Path, canonical: Path, text: str) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    reused = canonical.exists()
    if reused:
        require(canonical.read_text() == text, "V1_REFERENCE_EXISTING_RECIPE_DIFFERS:" + str(canonical))
        data = canonical.read_bytes()
    else:
        data = text.encode("utf-8")
    with destination.open("xb") as stream:
        stream.write(data)
    return dict(path=str(destination), canonical_target=str(canonical),
                reused_canonical=reused, **file_identity(destination))


def prepare_circuit(qualification_root: Path, output: Path, source: dict,
                    source_detector: Path, gpu: dict, memory_gib: int,
                    circuit: int, cohort: ProductionCohort, execution: dict,
                    intake_output: Path | None = None) -> tuple[list[dict], dict]:
    recurrent, inputs, accepted = accepted_construction(qualification_root, circuit, cohort, intake_output)
    base = qualification_root / f"circuit_{circuit:03d}"
    target, recipes = base / "reference_execution", base / "reference_preparation"
    local = output / base.name
    control = {key: value for key, value in source.items()
               if key not in ("records", "preparation_job", "preparation_step", "prior_preparation_failure")}
    control.update(circuit=str(base), output=str(target),
        parent=dict(job_id=int(recurrent["job_id"]), expected_archive=recurrent["archive"]),
        inputs=dict(job_id=accepted["source_input_job_id"], expected_archive=inputs["archive"],
                    parent_job_id=int(recurrent["job_id"])),
        detector=str(target / "detect_resources.py"), gpu=dict(gpu), gpu_memory_gib=memory_gib,
        records=[], preparation_job="merlin:" + execution["run_id"],
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        status="PREPARED_PRODUCTION_CIRCUIT_REFERENCE_INPUTS", production_episodes=0)
    if "source_input_job_id_namespace" in accepted:
        control["inputs"].update(job_id_namespace=accepted["source_input_job_id_namespace"],
                                 execution_context=accepted["source_input_execution_context"])
    artifacts = []
    for row in source["records"]:
        index = row["condition_index"]
        identity = cohort.identity(circuit, 0, index)
        recipe = cohort.v1_visual_config(identity)
        rate_root, spike_root = Path(row["rate_code_reference"]), Path(row["spike_canary"])
        rate, rate_delivery = bound_json(rate_root, "probe/probe.json")
        spike, spike_delivery = bound_json(spike_root, "visual_inputs/report.json")
        require(rate_delivery["exit_code"] == spike_delivery["exit_code"] == 0
                and spike_delivery["status"] == "PASS_CONSTRUCTED_VISUAL_INPUT_CANARY",
                "V1_REFERENCE_SOURCE_RATE_OR_SPIKE_PROOF_FAILED")
        require(sha256_file(rate_root / "delivery.json") == row["rate"]["delivery_sha256"]
                and sha256_file(spike_root / "delivery.json") == row["spike"]["delivery_sha256"],
                "V1_REFERENCE_SOURCE_RATE_OR_SPIKE_DELIVERY_CHANGED")
        require_same_recipe(recipe, rate["config"], family="visual")
        require_same_recipe(recipe, spike["config"], family="visual")
        name = f"condition_{index:02d}.yaml"
        artifacts.append(write_text_candidate(local / "reference_preparation" / name,
                                               recipes / name, yaml.safe_dump(recipe, sort_keys=False)))
        control["records"].append(dict(row, identity=identity, errors=[]))
    detector = local / "reference_execution/detect_resources.py"
    canonical_detector = target / "detect_resources.py"
    detector.parent.mkdir(parents=True, exist_ok=True)
    detector_identity = file_identity(source_detector)
    if canonical_detector.exists():
        require(file_identity(canonical_detector) == detector_identity, "V1_REFERENCE_EXISTING_DETECTOR_DIFFERS")
    with detector.open("xb") as stream:
        stream.write(source_detector.read_bytes())
    artifacts.append(dict(path=str(detector), canonical_target=str(canonical_detector),
                          reused_canonical=canonical_detector.exists(), **file_identity(detector)))
    # Preserve the original freshness gate immediately before freezing control.
    cohort.require_current()
    canonical_control = target / "prepared_control.json"
    local_control = local / "reference_execution/prepared_control.json"
    if canonical_control.exists():
        saved = read_json(canonical_control)
        for key in ("circuit", "parent", "inputs", "records", "detector", "gpu"):
            require(saved[key] == control[key], "V1_REFERENCE_EXISTING_CONTROL_DIFFERS:" + key)
        with local_control.open("xb") as stream:
            stream.write(canonical_control.read_bytes())
    else:
        freeze_json(local_control, control)
    artifacts.append(dict(path=str(local_control), canonical_target=str(canonical_control),
                          reused_canonical=canonical_control.exists(), **file_identity(local_control)))
    accepted.update(condition_indices=list(CONDITIONS), prepared_conditions=10,
                    prepared_control=str(local_control), canonical_control=str(canonical_control))
    return artifacts, accepted


def run(args: argparse.Namespace) -> dict:
    q = args.qualification_root.resolve()
    output = args.output.resolve()
    source_path = args.source_control.resolve()
    resource_path = args.execution_resources.resolve()
    circuits = selected_circuits(args)
    intake_output = args.intake_output.resolve() if getattr(args, "intake_output", None) else None
    require(not output.is_relative_to(q) and not output.is_relative_to(source_path.parent)
            and not output.is_relative_to(resource_path.parent), "V1_REFERENCE_ISOLATED_OUTPUT_REQUIRED")
    execution = cpu_execution_provenance()
    require(execution.get("backend") == "merlin" and execution.get("allocated_gpus") == 0,
            "V1_REFERENCE_ACTUAL_MERLIN_CPU_EXECUTION_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    report = dict(status="FAILED_V1_REFERENCE_PREPARATION", exit_code=1, artifacts=[], circuits=[],
        prepared_conditions=0, prepared_circuits=0, source_simulations=0, production_episodes=0,
        intake_executions=0, canonical_files_written=0, gpu_execution_performed=False, jobs_submitted=0)
    receipt = dict(execution=execution, started_utc=started,
        source_simulations=0, production_episodes=0, gpu_execution_performed=False, jobs_submitted=0,
        plan_field_semantics=dict(gpu="retained legacy reference-execution plan; not Merlin hardware or a new budget",
            native_python="retained source interpreter reference; not executed by this CPU preparation"))
    try:
        old = source_control(source_path, q)
        gpu, memory = legacy_execution_plan(resource_path)
        receipt.update(source_control=dict(path=str(source_path), **file_identity(source_path)),
            legacy_reference_plan=dict(source=str(resource_path), source_identity=file_identity(resource_path),
                                       gpu=gpu, gpu_memory_gib=memory))
        cohort = ProductionCohort.load(q / "frozen_campaign.yaml", "v1_visual",
                                      q / "qualification.json", q / "identities.csv")
        for circuit in circuits:
            report["current_circuit"] = circuit
            artifacts, accepted = prepare_circuit(q, output, old, source_path.parent / "detect_resources.py",
                                                  gpu, memory, circuit, cohort, execution, intake_output)
            report["artifacts"].extend(artifacts)
            report["circuits"].append(accepted)
            report["prepared_conditions"] += accepted["prepared_conditions"]
            report["prepared_circuits"] += 1
        require(report["prepared_conditions"] == 10 * len(circuits) and report["prepared_circuits"] == len(circuits)
                and len(report["artifacts"]) == 12 * len(circuits)
                and len({row["canonical_target"] for row in report["artifacts"]}) == 12 * len(circuits),
                "V1_REFERENCE_COMPLETE_SELECTED_TEN_CONDITION_FAMILIES_REQUIRED")
        report.update(status="PASS_V1_REFERENCE_PREPARATION", exit_code=0)
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error),
                      exit_code=130 if isinstance(error, KeyboardInterrupt) else 1)
    finally:
        receipt.update(status=report["status"], exit_code=report["exit_code"],
                       finished_utc=datetime.now(timezone.utc).isoformat(),
                       prepared_circuits=report["prepared_circuits"], prepared_conditions=report["prepared_conditions"])
        receipt_path = output / "execution_receipt.json"
        freeze_json(receipt_path, receipt)
        report["execution_receipt"] = dict(path=str(receipt_path), **file_identity(receipt_path))
        freeze_json(output / "report.json", report)
    return report


def staging(args: argparse.Namespace) -> dict:
    q = args.qualification_root.resolve()
    source_path = args.source_control.resolve()
    resources = args.execution_resources.resolve()
    circuits = selected_circuits(args)
    intake_output = args.intake_output.resolve() if getattr(args, "intake_output", None) else None
    old = source_control(source_path, q)
    legacy_execution_plan(resources)
    base = read_json(args.base_input_files)
    additions, reused = {}, {}

    def add_metadata(path: Path) -> None:
        name = str(path)
        actual = file_identity(path)
        if name in base:
            require(base[name] == actual, "V1_REFERENCE_BASE_METADATA_CHANGED:" + name)
            reused[name] = base[name]
        else:
            additions[name] = actual

    for path in (source_path, source_path.parent / "detect_resources.py", resources):
        add_metadata(path)
    for name in ("frozen_campaign.yaml", "qualification.json", "identities.csv", "identities.json"):
        add_metadata(q / name)
    for row in old["records"]:
        for directory, names in ((Path(row["rate_code_reference"]), ("delivery.json", "probe/probe.json")),
                                 (Path(row["spike_canary"]), ("delivery.json", "visual_inputs/report.json"))):
            for name in names:
                add_metadata(directory / name)
    for circuit in circuits:
        directory = q / f"circuit_{circuit:03d}"
        for name in ("circuit.yaml", "recurrent_acceptance.json"):
            add_metadata(directory / name)
        record_path, acceptance_path = input_paths(q, circuit, intake_output)
        add_metadata(acceptance_path)
        records = q / "construction_expansion_20261017" / directory.name
        for path in (records / "recurrent/report.json", record_path):
            add_metadata(path)
            add_metadata(Path(read_json(path)["archive"]) / "delivery.json")
        # Preserve any existing preparation artifacts without changing their bytes.
        candidates = [directory / "reference_preparation" / f"condition_{index:02d}.yaml" for index in CONDITIONS]
        candidates.extend(directory / "reference_execution" / name for name in ("prepared_control.json", "detect_resources.py"))
        for path in candidates:
            if path.exists():
                add_metadata(path)
    package = Path(sys.modules[ProductionCohort.__module__].__file__).resolve().parent
    code = module_files(package, ["benchmark_production", "provenance", "v1_execution_recipe",
                                 "v1_nest_input_replay", "v1_circuit_inputs"])
    owned_code = {Path(__file__).resolve(), Path(sys.modules[source_identity.__module__].__file__).resolve(),
                  package / "test_v1_construction_intake_merlin.py", package / "test_v1_reference_preparation_merlin.py"}
    code.update(owned_code)
    code.update((Path(sys.modules[module_files.__module__].__file__).resolve(),
                 Path(sys.modules["ca3_merlin_remote"].__file__).resolve()))
    runtime_overrides = {}
    for path in sorted(code):
        name = str(path)
        if name in base and path in owned_code and file_identity(path) != base[name]:
            additions[name] = file_identity(path)
            runtime_overrides[name] = dict(base_identity=base[name], executed_identity=additions[name])
        elif name in base:
            # Numerical/runtime dependencies retain the uploaded snapshot.
            reused[name] = base[name]
        else:
            additions[name] = file_identity(path)
    return dict(status="STAGING_PLAN_ONLY", files=additions, supplemental_files=len(additions),
        supplemental_bytes=sum(row["bytes"] for row in additions.values()), base_files_used=reused,
        base_input_manifest=dict(path=str(args.base_input_files.resolve()), **file_identity(args.base_input_files)),
        circuits=list(circuits), conditions_per_circuit=10, expected_conditions=10 * len(circuits),
        expected_canonical_targets=12 * len(circuits), runtime_overrides=runtime_overrides,
        source_simulations=0, production_episodes=0, intake_executions=0, jobs_submitted=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "staging"):
        command = commands.add_parser(name)
        for flag in ("qualification-root", "source-control", "execution-resources"):
            command.add_argument("--" + flag, type=Path, required=True)
        command.add_argument("--circuits", nargs="+", type=int, choices=range(94, 100))
        command.add_argument("--intake-output", type=Path, help="Isolated validated C98/C99 intake candidates")
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--base-input-files", type=Path, required=True)
    args = parser.parse_args()
    result = run(args) if args.command == "run" else staging(args)
    print(json.dumps(result, indent=2))
    return result.get("exit_code", 0)


if __name__ == "__main__":
    raise SystemExit(main())
