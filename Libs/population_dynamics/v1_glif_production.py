"""Production admission for source-derived GLIF CUDA block executions."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import tempfile

import numpy as np

from . import benchmark_production as production
from .provenance import cpu_execution_provenance, freeze_json, sha256_file
from .v1_glif_component_emulator import COMPONENT_RTOL, COMPONENT_ATOL
from .v1_glif_dataset import BLOCK_PASS, CASE_PASS, SOURCE_ARRAYS, export_block
from .v1_glif_network import read_reference_data
from .v1_pointnet_native import bound_files
from .v1_nest_input_replay import bound_json
from .v1_benchmark import export_source_controls, inspect_source_movie, local_delivery


def compare_executions(primary: Path, replay: Path, output: Path, *,
                       execution_context: dict | None = None) -> dict:
    """Verify all declared conditions from two independent GPU processes."""
    production.require(bool(os.environ.get("SLURM_JOB_ID")) and not output.exists(),
                       "GLIF_REPLAY_FRESH_COMPUTE_OUTPUT_REQUIRED")
    output.mkdir(parents=True)
    report = dict(status="FAILED_GLIF_SOURCE_BLOCK_REPLAY", job_id=os.environ["SLURM_JOB_ID"],
        primary=str(primary), replay=str(replay), component_rtol=COMPONENT_RTOL,
        component_atol=COMPONENT_ATOL, conditions=[], production_episodes=0)
    if execution_context is not None:
        report["execution_context"] = dict(execution_context)
    started = time.monotonic()
    try:
        headers, deliveries, bindings = [], [], []
        for root in (primary, replay):
            delivery, binding = bound_files(root, ("coupled/report.json", "coupled/cases.json"))
            header = json.loads((root/"coupled/report.json").read_text())
            production.require(str(header["job_id"]) == str(delivery["job_id"])
                and header["status"] == BLOCK_PASS and header["passed_conditions"] == 10
                and header["failed_conditions"] == 0 and len(header["cases"]) == 10,
                "GLIF_REPLAY_COMPLETE_BLOCK_REQUIRED")
            headers.append(header); deliveries.append(delivery); bindings.append(binding)
        production.require(str(deliveries[0]["job_id"]) != str(deliveries[1]["job_id"]),
                           "GLIF_REPLAY_INDEPENDENT_JOBS_REQUIRED")
        for name in ("circuit_index", "block_index", "cases_sha256", "runtime_code",
                     "coupled_units", "source_connection_rows"):
            production.require(headers[0][name] == headers[1][name], "GLIF_REPLAY_BLOCK_BINDING_DIFFERS:"+name)
        report.update(primary_binding=bindings[0], replay_binding=bindings[1],
            primary_job_id=deliveries[0]["job_id"], replay_job_id=deliveries[1]["job_id"],
            runtime_code=headers[0]["runtime_code"], circuit_index=headers[0]["circuit_index"],
            block_index=headers[0]["block_index"])
        for index, (a, b) in enumerate(zip(headers[0]["cases"], headers[1]["cases"])):
            production.require(a["condition_index"] == b["condition_index"] == index
                and a["identity"] == b["identity"] and a["input_archive"] == b["input_archive"]
                and a["report"] == b["report"] == f"condition_{index:02d}/report.json",
                "GLIF_REPLAY_CONDITION_IDENTITY_DIFFERS")
            prefix = f"coupled/condition_{index:02d}/"
            documents, case_bindings, arrays = [], [], []
            for root in (primary, replay):
                _, binding = bound_files(root, tuple(prefix+name for name in ("report.json", *SOURCE_ARRAYS)))
                document = json.loads((root/(prefix+"report.json")).read_text())
                production.require(document["status"] == CASE_PASS and str(document["job_id"]) == str(
                    deliveries[0 if root == primary else 1]["job_id"]), "GLIF_REPLAY_PASSED_CASE_REQUIRED")
                data = {}
                for name in SOURCE_ARRAYS:
                    with np.load(root/(prefix+name), allow_pickle=False) as file:
                        data[name] = {key:file[key] for key in file.files}
                documents.append(document); case_bindings.append(binding); arrays.append(data)
            for name in ("runtime_code", "circuit_reference", "input_archive", "input_bindings",
                         "units", "connection_rows", "execution_clock", "candidate_spikes"):
                production.require(documents[0][name] == documents[1][name],
                                   "GLIF_REPLAY_EXECUTION_BINDING_DIFFERS:"+name)
            library = documents[0]["cuda_execution"]["library_sha256"]
            production.require(library == documents[1]["cuda_execution"]["library_sha256"],
                               "GLIF_REPLAY_LIBRARY_DIFFERS")
            exact = {}
            for name in ("input_events.npz", "cortical_state.npz"):
                left, right = arrays[0][name], arrays[1][name]
                production.require(set(left) == set(right), "GLIF_REPLAY_ARRAY_SCHEMA_DIFFERS")
                for key in left:
                    if name == "cortical_state.npz" and key == "terminal_vm_mV":
                        continue
                    exact[name+":"+key] = bool(left[key].dtype == right[key].dtype and np.array_equal(left[key], right[key]))
            left, right = [data["cortical_spikes.npz"] for data in arrays]
            order = np.lexsort((left["native_id"],left["time_ms"]))
            other = np.lexsort((right["native_id"],right["time_ms"]))
            spike_pairs = all(np.array_equal(left[key][order],right[key][other]) for key in ("native_id","time_ms"))
            left, right = [data["cortical_state.npz"]["terminal_vm_mV"] for data in arrays]
            production.require(left.shape == right.shape, "GLIF_REPLAY_TERMINAL_SHAPE_DIFFERS")
            finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
            voltage = bool(finite and np.isclose(left,right,rtol=COMPONENT_RTOL,atol=COMPONENT_ATOL).all())
            passed = all(exact.values()) and spike_pairs and voltage
            report["conditions"].append(dict(condition_index=index, identity=a["identity"],
                status="PASS" if passed else "FAILED", all_exact_checks=exact,
                all_spike_pairs_exact=spike_pairs, terminal_voltage_within_component_tolerance=voltage,
                terminal_voltage_bitwise_equal=bool(np.array_equal(left,right)),
                maximum_terminal_voltage_error_mV=float(np.max(np.abs(left-right))) if finite else None,
                primary_report_sha256=case_bindings[0]["files"][prefix+"report.json"]["sha256"],
                replay_report_sha256=case_bindings[1]["files"][prefix+"report.json"]["sha256"],
                library_sha256=library, input_archive=documents[0]["input_archive"],
                input_report_sha256=documents[0]["input_bindings"]["bundle_report_sha256"],
                spike_pairs=documents[0]["candidate_spikes"]))
        report.update(passed_conditions=sum(row["status"] == "PASS" for row in report["conditions"]),
            failed_conditions=sum(row["status"] != "PASS" for row in report["conditions"]),
            total_spike_pairs=sum(row["spike_pairs"] for row in report["conditions"]))
        production.require(report["passed_conditions"] == 10, "GLIF_REPLAY_CONDITION_FAILED")
        report["status"] = "PASS_GLIF_SOURCE_BLOCK_REPLAY"
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,recorded_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output/"report.json",report)
    return report




def validate_saved_reference(reference: Path, execution: Path, condition_index: int,
                             output: Path) -> dict:
    """Compare the circuit's native reference with an authenticated saved CUDA case.

    This reader executes no simulator. Runtime identities come from the saved
    execution, so adding a reader does not invalidate a verified numerical build.
    """
    production.require(bool(os.environ.get("SLURM_JOB_ID")) and not output.exists(),
                       "GLIF_SAVED_REFERENCE_FRESH_COMPUTE_OUTPUT_REQUIRED")
    production.require(type(condition_index) is int and 0 <= condition_index < 10,
                       "GLIF_SAVED_REFERENCE_CONDITION_INVALID")
    output.mkdir(parents=True)
    report = dict(status="FAILED_SAVED_NATIVE_GLIF_CUDA_REFERENCE",
        job_id=os.environ["SLURM_JOB_ID"], native_archive=str(reference),
        execution_archive=str(execution), condition_index=condition_index,
        source_execution_repeated=False, simulator_calls=0, production_episodes=0,
        scientific_qualification=False, component_rtol=COMPONENT_RTOL,
        component_atol=COMPONENT_ATOL, validator_sha256=sha256_file(Path(__file__)))
    started = time.monotonic()
    try:
        native, binding, original = read_reference_data(reference, require_tuples=False)
        prefix = f"coupled/condition_{condition_index:02d}/"
        delivery, executed_binding = bound_files(execution,
            tuple(prefix+name for name in ("report.json", *SOURCE_ARRAYS)))
        executed = json.loads((execution/(prefix+"report.json")).read_text())
        circuit_root = Path(executed["circuit_reference"])
        _, circuit_binding = bound_files(circuit_root, ("primary/report.json",))
        circuit = json.loads((circuit_root/"primary/report.json").read_text())
        production.require(delivery["exit_code"] == 0 and executed["status"] == CASE_PASS
            and str(executed["job_id"]) == str(delivery["job_id"])
            and executed["execution_clock"] == original["execution_clock"]
            and executed["units"] == original["cortical_units"]
            and executed["connection_rows"] == original["connections"]["native_connections_to_cortex"]
            and executed["circuit_bindings"]["files"]["primary/report.json"]["sha256"]
                == circuit_binding["files"]["primary/report.json"]["sha256"]
            and circuit["original_config_sha256"] == original["original_config_sha256"]
            and all(circuit["bindings"][role]["files"]["primary/report.json"]["sha256"]
                    == original["bindings"][role]["files"]["primary/report.json"]["sha256"]
                    for role in ("recurrent", "inputs"))
            and not executed["cortical_outcomes_used_for_execution"],
            "GLIF_SAVED_REFERENCE_EXECUTION_BINDING_DIFFERS")
        candidate = {}
        for name in SOURCE_ARRAYS:
            with np.load(execution/(prefix+name), allow_pickle=False) as archive:
                candidate[name.removesuffix(".npz")] = {key: archive[key] for key in archive.files}
        exact = {}
        for name in ("input_events", "cortical_state"):
            left, right = native[name], candidate[name]
            production.require(set(left) == set(right), "GLIF_SAVED_REFERENCE_ARRAY_SCHEMA_DIFFERS:"+name)
            for key in left:
                if name == "cortical_state" and key == "terminal_vm_mV":
                    continue
                exact[name+":"+key] = bool(left[key].dtype == right[key].dtype
                                                   and np.array_equal(left[key], right[key]))
        left, right = native["cortical_spikes"], candidate["cortical_spikes"]
        order = np.lexsort((left["native_id"], left["time_ms"]))
        other = np.lexsort((right["native_id"], right["time_ms"]))
        pairs = all(np.array_equal(left[key][order], right[key][other]) for key in ("native_id", "time_ms"))
        state, saved = native["cortical_state"], candidate["cortical_state"]
        left, right = state["terminal_vm_mV"], saved["terminal_vm_mV"]
        production.require(left.shape == right.shape == (original["cortical_units"],),
                           "GLIF_SAVED_REFERENCE_TERMINAL_SHAPE_DIFFERS")
        finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
        voltage = bool(finite and np.isclose(left, right, rtol=COMPONENT_RTOL, atol=COMPONENT_ATOL).all())
        counts = bool(np.array_equal(state["spike_count"], saved["spike_count"]))
        report.update(source_bindings=binding, execution_binding=executed_binding,
            circuit_reference_binding=circuit_binding,
            native_reference_status=original["status"], execution_job_id=delivery["job_id"],
            runtime_code=executed["runtime_code"], cuda_execution=executed["cuda_execution"],
            units=original["cortical_units"], duration_ms=original["execution_clock"]["duration_ms"],
            all_exact_checks=exact, all_spike_pairs_exact=pairs, all_unit_counts_exact=counts,
            candidate_spikes=len(candidate["cortical_spikes"]["time_ms"]),
            native_spikes=len(native["cortical_spikes"]["time_ms"]),
            terminal_voltage_within_component_tolerance=voltage,
            terminal_voltage_bitwise_equal=bool(np.array_equal(left, right)),
            terminal_voltage_maximum_absolute_error_mV=float(np.max(np.abs(left-right))) if finite else None)
        production.require(all(exact.values()) and pairs and counts and voltage,
                           "GLIF_SAVED_REFERENCE_VALUES_DIFFER")
        report["status"] = "PASS_SAVED_NATIVE_GLIF_CUDA_REFERENCE"
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      recorded_utc=datetime.now(timezone.utc).isoformat())
        freeze_json(output/"report.json", report)
    return report


def verify_circuit_resources(cohort, identity: dict, reference: Path,
                             snapshot: Path, validation: Path) -> dict:
    backend = production.glif_backend_qualification(cohort)
    relatives = ("primary/report.json", "recurrent/primary/report.json", "inputs_reference/primary/report.json")
    _, binding = bound_files(reference, relatives)
    native = json.loads((reference/"primary/report.json").read_text())
    for role, relative in (("recurrent", relatives[1]), ("inputs", relatives[2])):
        report = json.loads((reference/relative).read_text())
        cohort.verify_declaration(report["original_config"], identity)
        production.require(report["original_config"] == cohort.v1_construction_config(identity)
            and report["original_config_sha256"] == native["original_config_sha256"]
            and native["bindings"][role]["files"]["primary/report.json"]["sha256"] == binding["files"][relative]["sha256"],
            "GLIF_PRODUCTION_CIRCUIT_CONSTRUCTION_DIFFERS")
    proof = json.loads(validation.read_text())
    prepared = json.loads((snapshot/"report.json").read_text())
    reference_sha = binding["files"]["primary/report.json"]["sha256"]
    production.require(native["cortical_units"] == 230924 and native["native_nest_simulation"]
        and native["native_parameters_and_initial_state_exact"]
        and proof["status"] in production.GLIF_CIRCUIT_REFERENCE_STATUSES
        and proof["all_spike_pairs_exact"] and proof["all_unit_counts_exact"]
        and proof["terminal_voltage_within_component_tolerance"]
        and proof["source_bindings"]["files"]["primary/report.json"]["sha256"] == reference_sha
        and proof["runtime_code"] == backend["runtime_code"]
        and proof["cuda_execution"]["library_sha256"] == backend["library_sha256"]
        and proof["component_rtol"] == backend["component_rtol"]
        and proof["component_atol"] == backend["component_atol"],
        "GLIF_PRODUCTION_CIRCUIT_VALIDATION_REQUIRED")
    production.require(prepared["status"] == "PASS_COMPLETE_NATIVE_CONNECTION_SNAPSHOT"
        and prepared["native_report_sha256"] == reference_sha
        and prepared["snapshot"]["source_batch_sha256"] == native["connections"]["source_batch_sha256"]
        and prepared["snapshot"]["rows"] == native["connections"]["native_connections_to_cortex"]
        and prepared["native_simulation_calls"] == 0,
        "GLIF_PRODUCTION_CIRCUIT_SNAPSHOT_DIFFERS")
    paths = [reference/"delivery.json", *(reference/name for name in relatives),
             snapshot/"report.json", validation]
    return {str(path): sha256_file(path) for path in paths}


def bind_sources(cohort, entry, comparison_path: Path) -> dict:
    identity, sources = entry.identity, entry.sources
    reference = sources["circuit_reference"]
    native, _ = bound_json(reference, "primary/report.json")
    recurrent, _ = bound_json(reference, "recurrent/primary/report.json")
    inputs, _ = bound_json(reference, "inputs_reference/primary/report.json")
    rates, rate_delivery = bound_json(sources["rate_source"], "probe/probe.json")
    spikes, spike_delivery = bound_json(sources["spike_source"], "visual_inputs/report.json")
    production.require(rate_delivery["exit_code"] == spike_delivery["exit_code"] == 0,
                       "GLIF_PRODUCTION_PASSED_INPUTS_REQUIRED")
    relative = f"coupled/condition_{identity['condition_index']:02d}/report.json"
    execution, _ = bound_json(sources["gpu_primary"], relative)
    replay, _ = bound_json(sources["gpu_replay"], relative)
    # bind_v1 consumes construction and input bindings; it does not require or
    # fabricate a native-execution receipt for the current GPU condition.
    context = dict(original_config_sha256=native["original_config_sha256"],
        bindings=dict(recurrent=native["bindings"]["recurrent"], inputs=native["bindings"]["inputs"],
                      spike_inputs=execution["input_bindings"]))
    admission = production.bind_v1(cohort, identity, recurrent, inputs, context, rates, spikes)
    production.require(execution["input_archive"] == replay["input_archive"] == str(sources["spike_source"])
        and execution["circuit_reference"] == replay["circuit_reference"] == str(reference),
        "GLIF_PRODUCTION_EXECUTION_SOURCE_PATHS_DIFFER")
    paths = dict(recurrent=reference/"recurrent/primary/report.json",
        inputs=reference/"inputs_reference/primary/report.json", circuit_reference=reference/"primary/report.json",
        circuit_validation=sources["circuit_validation"], execution=sources["gpu_primary"]/relative,
        replay=sources["gpu_replay"]/relative)
    admission.update(execution_backend="glif_cuda", construction_config_sha256=native["original_config_sha256"],
        execution_report_sha256=sha256_file(paths["execution"]), replay_report_sha256=sha256_file(paths["replay"]),
        circuit_reference_report_sha256=sha256_file(paths["circuit_reference"]),
        circuit_validation_sha256=sha256_file(paths["circuit_validation"]),
        source_documents={role: production.source_document(path) for role,path in paths.items()})
    if getattr(cohort, "runtime_document", None) is not None:
        admission["runtime_qualification_sha256"] = cohort.runtime_document["sha256"]
    if comparison_path.exists():
        admission["replay_comparison_sha256"] = sha256_file(comparison_path)
        admission["source_documents"]["replay_comparison"] = production.source_document(comparison_path)
    return admission


@dataclass
class GlifBlockSource:
    dataset: production.CircuitDataset
    episode_ids: list[str]
    admissions: list[dict]
    comparison: dict


@contextmanager
def prepared_source_block(cohort, entries, attempt: Path):
    cohort.require_current()
    production.glif_backend_qualification(cohort)
    execution = cpu_execution_provenance()
    production.require(not os.environ.get("SLURM_JOB_GPUS")
                       and (execution["backend"] == "merlin"
                            or execution["slurm_node"] in ("cpu03", "cpu05", "cpu06")),
                       "GLIF_PRODUCTION_ALLOCATED_CPU_EXPORT_REQUIRED")
    previous_job_id = os.environ.get("SLURM_JOB_ID")
    if execution["backend"] == "merlin":
        # Existing delivery readers consume job_id. This value is a labeled
        # compatibility identity, never a Slurm job or a fabricated Xie node.
        compatibility_id = "merlin-trial-" + execution["trial_id"]
        execution.update(legacy_job_id=compatibility_id, legacy_job_id_is_compatibility=True)
        os.environ["SLURM_JOB_ID"] = compatibility_id
    try:
        comparison_root = attempt/"source_replay"
        comparison_path = comparison_root/"report.json"
        # Validate every declared circuit/control/noise identity before processing data.
        admissions = [bind_sources(cohort, entry, comparison_path) for entry in entries]
        first = entries[0].sources
        comparison = compare_executions(first["gpu_primary"], first["gpu_replay"], comparison_root,
                                        execution_context=execution)
        for entry, admission in zip(entries, admissions):
            admission["replay_comparison_sha256"] = sha256_file(comparison_path)
            admission["source_documents"]["replay_comparison"] = production.source_document(comparison_path)
            production.verify_source_admission(cohort, entry.identity, admission)
        with tempfile.TemporaryDirectory(prefix="neuronsim-glif-production-", dir="/tmp") as temporary:
            work = Path(temporary)
            controls = []
            for entry in entries:
                index, sources = entry.identity["condition_index"], entry.sources
                movie = work/f"movie_{index:02d}"
                commands = work/f"controls_{index:02d}"
                result = inspect_source_movie(sources["rate_source"], movie/"movie")
                production.require(result["status"] == "SOURCE_MOVIE_CONTROL_WITNESS_PASS",
                                   "GLIF_PRODUCTION_MOVIE_WITNESS_FAILED")
                local_delivery(movie, "movie")
                export_source_controls(sources["rate_source"], movie, sources["spike_source"], commands/"controls")
                local_delivery(commands, "controls")
                controls.append(dict(condition_index=index, control_archive=str(commands)))
            controls_file = work/"controls.json"
            freeze_json(controls_file, dict(cases=controls))
            exported = export_block(first["gpu_primary"], controls_file, first["authority_source"], work/"source")
            production.require(exported["status"] == "EXPORTED_VERIFIED_GLIF_SOURCE_BLOCK"
                               and exported["complete_episodes"] == 10, "GLIF_PRODUCTION_COMPLETE_EXPORT_REQUIRED")
            yield GlifBlockSource(production.CircuitDataset(work/"source/dataset", splits=("development",)),
                                  exported["episode_ids"], admissions, comparison)
    finally:
        if execution["backend"] == "merlin":
            if previous_job_id is None:
                os.environ.pop("SLURM_JOB_ID", None)
            else:
                os.environ["SLURM_JOB_ID"] = previous_job_id


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_subparsers(dest="action",required=True)
    comparison=actions.add_parser("compare")
    for name in ("primary","replay","output"):
        comparison.add_argument("--"+name,type=Path,required=True)
    reference=actions.add_parser("validate-reference")
    for name in ("reference","execution","output"):
        reference.add_argument("--"+name,type=Path,required=True)
    reference.add_argument("--condition-index",type=int,required=True)
    args=parser.parse_args()
    if args.action == "compare":
        result=compare_executions(args.primary,args.replay,args.output)
    else:
        result=validate_saved_reference(args.reference,args.execution,args.condition_index,args.output)
    print(json.dumps(result))


if __name__=="__main__":
    main()
