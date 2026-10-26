"""Export fresh qualified neural-circuit execution into the frozen production cohort.

Source validators retain their source-reproduction format. A production circuit
must be declared before construction. Existing development archives cannot enter
this interface. Native stream names remain in provenance; cohort seed identifiers
stay separate from microscopic initial-state draws and training seeds.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from dataclasses import dataclass, replace
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
import tempfile
import time

import numpy as np
import yaml

from . import v1_runtime_qualification as runtime
from .benchmark_dataset import read_circuit, read_episode
from .benchmark_tasks import condition_for_episode, v1_visual_config, pfc_working_memory_config
from . import ca3_benchmark, pfc_source_dataset, pfc_source_execution, pfc_kernel_parameters
from .ca3_benchmark_readout import bind_episode as bind_ca3_episode
from .ca3_control_trace import check_currents
from .ca3_event_counters import compare_event_counters
from .ca3_host_current_replay import compare_currents
from .ca3_native_observation import iter_intervals
from .ca3_retrieval_execution import qualify_input
from .ca3_realization import SourceRealization
from .circuit_campaign import campaign_config, episode_identities, production_prerequisites
from .circuit_data import SCHEMA, CircuitArchiveWriter, CircuitDataset, CircuitEpisode, CircuitUnits
from .provenance import freeze_json, sha256_file
from .v1_benchmark import local_delivery
from .v1_benchmark_readout import describe_episode
from .v1_circuit_stimulus import rate_config_without_noise
from .v1_lgn_runtime import inspect_stimulus
from .v1_source_dataset import export_pointnet
from .v1_visual_controls import export_constructed_visual_controls
from .v1_nest_input_replay import bound_json

# Preserve the established V1/CA3 enumeration for existing callers and fixtures.
LEGACY_SCENARIOS = ("v1_visual", "ca3_pattern_completion")
SCENARIOS = ("v1_visual", "pfc_working_memory", "ca3_pattern_completion")
COHORT_SEEDS = ("connectivity_seed", "initial_state_seed", "background_seed", "stimulus_seed")
DECLARATION_SCHEMA = "neural_benchmark_production_circuit_v1"
GLIF_CIRCUIT_REFERENCE_STATUSES = (
    "PASS_NATIVE_GLIF_CUDA_COUPLED_DIAGNOSTIC",
    "PASS_SAVED_NATIVE_GLIF_CUDA_REFERENCE",
)


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def validate_pfc_recipes(campaign: dict, qualification: dict) -> None:
    """Bind the frozen circuit mapping to 100 distinct source coupling kernels."""
    recipes = qualification.get("source_recipes", {})
    configs, native = recipes.get("circuit_configs"), recipes.get("native_qualifications")
    names = {f"pfc_working_memory_circuit_{index:03d}" for index in range(100)}
    require(isinstance(configs, dict) and set(configs) == names
            and isinstance(native, dict) and set(native) == names,
            "BENCHMARK_PRODUCTION_PFC_100_FROZEN_SOURCE_RECIPES_REQUIRED")
    require(campaign["sampling"] == dict(circuit_instances=100,
                initial_noise_blocks_per_circuit=100, conditions_per_block=10)
            and qualification["source_revision"] == pfc_source_dataset.SOURCE_REVISION,
            "BENCHMARK_PRODUCTION_PFC_SOURCE_OR_ENVELOPE_DIFFERS")
    protocol = campaign["scenarios"]["pfc_working_memory"]["task_protocol"]
    seen = set()
    for name in sorted(names):
        config, receipt = configs[name], native[name]
        require(isinstance(config, dict) and not any(key.startswith("benchmark_") for key in config)
                and config.get("excitatory_neurons") == 80000 and config.get("inhibitory_neurons") == 20000
                and config.get("excitatory_preference_bins") == 16 and config.get("readout_bin_ms") == 5.0
                and config.get("dt_ms") == protocol["dt_ms"]
                and config.get("duration_ms") == protocol["duration_ms"],
                "BENCHMARK_PRODUCTION_PFC_FULL_SOURCE_RECIPE_REQUIRED:" + name)
        pfc_source_execution._case_config(config, config["cases"][0]["case_id"])
        digest = hashlib.sha256(pfc_kernel_parameters.source_kernel(80000, config).tobytes()).hexdigest()
        require(digest not in seen, "BENCHMARK_PRODUCTION_PFC_DUPLICATE_NUMERICAL_KERNEL:" + name)
        seen.add(digest)
        require(isinstance(receipt, dict) and receipt.get("backend") in ("numpy", "cuda")
                and all(isinstance(receipt.get(key), str) and len(receipt[key]) == 64
                        for key in ("receipt_sha256", "delivery_sha256")),
                "BENCHMARK_PRODUCTION_PFC_NATIVE_QUALIFICATION_BINDING_REQUIRED:" + name)


def qualify_pfc_backend(cohort, identity, directory: Path, backend: str) -> dict:
    """Use the existing exact-clock, exact-tuple native qualification guard."""
    config = cohort.pfc_episode_config(identity)
    bound = pfc_source_execution._qualification(directory, backend, config["dt_ms"], config)
    expected = cohort.qualification["source_recipes"]["native_qualifications"][identity["circuit_id"]]
    require(backend == expected["backend"]
            and all(bound[key] == expected[key] for key in ("receipt_sha256", "delivery_sha256")),
            "BENCHMARK_PRODUCTION_PFC_FROZEN_NATIVE_QUALIFICATION_DIFFERS")
    return bound


@dataclass(frozen=True)
class ProductionCohort:
    campaign: dict
    scenario: str
    qualification_path: Path
    identity_table: Path
    qualification: dict
    qualification_sha256: str
    identity_table_sha256: str
    identities: dict[str, dict]
    runtime_document: dict | None = None
    runtime_path: Path | None = None

    @classmethod
    def load(cls, config: Path, scenario: str, qualification: Path, identity_table: Path,
             *, runtime_qualification: Path | None = None):
        require(scenario in SCENARIOS, "BENCHMARK_PRODUCTION_SCENARIO_REQUIRED")
        campaign = campaign_config(config)
        proof = json.loads(qualification.read_text())
        missing = production_prerequisites(campaign, scenario, proof)
        for name in ("parameter_domain", "task_protocol", "observation_map"):
            value = campaign["scenarios"][scenario].get(name)
            if not isinstance(value, dict) or value.get("frozen") is not True:
                missing.append(name + "_not_explicitly_frozen")
        if scenario == "v1_visual" and not all(isinstance(proof.get("source_recipes", {}).get(name), dict)
                and proof["source_recipes"][name] for name in ("construction_config", "visual_config")):
            missing.append("qualified_resolved_source_recipes")
        require(not missing, "BENCHMARK_PRODUCTION_NOT_QUALIFIED:" + ",".join(missing))
        if scenario == "pfc_working_memory":
            validate_pfc_recipes(campaign, proof)
        frozen = json.loads(identity_table.with_suffix(".json").read_text())
        qsha, isha = sha256_file(qualification), sha256_file(identity_table)
        require(frozen["status"] == "FROZEN_IDENTITIES_ONLY" and frozen["scenario_id"] == scenario
                and frozen["resolved_campaign"] == campaign
                and frozen["source_qualification_sha256"] == qsha
                and frozen["identity_table_sha256"] == isha,
                "BENCHMARK_PRODUCTION_FROZEN_COHORT_BINDING_DIFFERS")
        expected = {}
        with identity_table.open(newline="") as stream:
            actual = csv.DictReader(stream)
            for row in episode_identities(campaign, scenario, production=True):
                require(next(actual, None) == {k: str(v) for k, v in row.items()},
                        "BENCHMARK_PRODUCTION_IDENTITY_ROW_DIFFERS")
                expected[row["episode_id"]] = row
            require(next(actual, None) is None, "BENCHMARK_PRODUCTION_EXTRA_IDENTITY_ROWS")
        cohort = cls(campaign, scenario, qualification, identity_table, proof, qsha, isha, expected)
        if runtime_qualification is not None:
            require(scenario == "v1_visual", "GLIF_RUNTIME_ONLY_FOR_V1")
            original = glif_backend_qualification(cohort)
            runtime_document, _ = runtime.load(runtime_qualification, qsha, original)
            cohort = replace(cohort, runtime_document=runtime_document,
                             runtime_path=runtime_qualification)
        return cohort

    def require_current(self) -> None:
        if self.runtime_path is not None:
            require(runtime.source_document(self.runtime_path) == self.runtime_document,
                    "BENCHMARK_PRODUCTION_RUNTIME_QUALIFICATION_CHANGED")
        require(sha256_file(self.qualification_path) == self.qualification_sha256
                and sha256_file(self.identity_table) == self.identity_table_sha256,
                "BENCHMARK_PRODUCTION_QUALIFICATION_OR_COHORT_CHANGED")
        frozen = json.loads(self.identity_table.with_suffix(".json").read_text())
        require(frozen["resolved_campaign"] == self.campaign
                and frozen["source_qualification_sha256"] == self.qualification_sha256
                and frozen["identity_table_sha256"] == self.identity_table_sha256
                and json.loads(self.qualification_path.read_text()) == self.qualification,
                "BENCHMARK_PRODUCTION_FROZEN_SNAPSHOT_CHANGED")
        require(not production_prerequisites(self.campaign, self.scenario, self.qualification)
                and all(isinstance(self.campaign["scenarios"][self.scenario].get(name), dict)
                        and self.campaign["scenarios"][self.scenario][name].get("frozen") is True
                        for name in ("parameter_domain", "task_protocol", "observation_map")),
                "BENCHMARK_PRODUCTION_QUALIFICATION_CHANGED")

    def require_identity(self, identity: dict) -> None:
        require(identity == self.identities.get(identity.get("episode_id"))
                and identity.get("scenario_id") == self.scenario,
                "BENCHMARK_PRODUCTION_IDENTITY_NOT_FROZEN")

    def identity(self, circuit_index: int, block_index: int, condition_index: int) -> dict:
        for name, value, bound in (("circuit", circuit_index, 100), ("block", block_index, 100),
                                   ("condition", condition_index, 10)):
            require(type(value) is int and 0 <= value < bound,
                    "BENCHMARK_PRODUCTION_INDEX_INVALID:" + name)
        key = f"{self.scenario}_circuit_{circuit_index:03d}_block_{block_index:03d}_condition_{condition_index:02d}"
        require(key in self.identities, "BENCHMARK_PRODUCTION_IDENTITY_NOT_FROZEN")
        return copy.deepcopy(self.identities[key])

    def declaration(self, identity: dict) -> dict:
        declaration = dict(schema=DECLARATION_SCHEMA, status="PREPARED_QUALIFIED_PRODUCTION_CIRCUIT",
                    scenario_id=self.scenario, circuit_id=identity["circuit_id"],
                    circuit_index=identity["circuit_index"], intended_split=identity["split"],
                    construction_scope="production", connectivity_seed=identity["connectivity_seed"],
                    complete_episodes=0, production_eligible=True,
                    source_qualification_sha256=self.qualification_sha256,
                    identity_table_sha256=self.identity_table_sha256)
        if self.scenario == "pfc_working_memory":
            config = self.qualification["source_recipes"]["circuit_configs"][identity["circuit_id"]]
            declaration.update(parameters=pfc_kernel_parameters.resolve_kernel_parameters(config),
                effective_kernel_sha256=hashlib.sha256(pfc_kernel_parameters.source_kernel(
                    config["excitatory_neurons"], config).tobytes()).hexdigest(),
                **pfc_kernel_parameters.circuit_provenance(config, pfc_source_dataset.SOURCE_ENTRY_SHA256))
        return declaration

    def pfc_circuit_config(self, identity: dict) -> dict:
        config = copy.deepcopy(self.qualification["source_recipes"]["circuit_configs"][identity["circuit_id"]])
        config["benchmark_circuit_candidate"] = self.declaration(identity)
        return config

    def pfc_episode_config(self, identity: dict) -> dict:
        return pfc_working_memory_config(self.pfc_circuit_config(identity),
            self.campaign["scenarios"][self.scenario]["task_protocol"], identity)

    def verify_declaration(self, config: dict, identity: dict) -> None:
        self.require_current()
        self.require_identity(identity)
        require(config.get("benchmark_circuit_candidate") == self.declaration(identity),
                "BENCHMARK_PRODUCTION_FRESH_DECLARED_CONSTRUCTION_REQUIRED")

    def v1_construction_config(self, identity: dict) -> dict:
        config = copy.deepcopy(self.qualification["source_recipes"]["construction_config"])
        config.update(builder_seed=identity["connectivity_seed"],
                      glif_assignment_seed=identity["connectivity_seed"] + 1000000,
                      benchmark_circuit_candidate=self.declaration(identity))
        return config

    def v1_visual_config(self, identity: dict) -> dict:
        return v1_visual_config(self.qualification["source_recipes"]["visual_config"],
            self.campaign["scenarios"][self.scenario]["task_protocol"], identity)

    def writer(self, output: Path, source_revision: str, *, resume: bool = False):
        self.require_current()
        require(source_revision == self.qualification["source_revision"],
                "BENCHMARK_PRODUCTION_SOURCE_REVISION_DIFFERS")
        return CircuitArchiveWriter(output, dict(schema=SCHEMA, scenario_id=self.scenario,
            sample_unit="complete_mechanistic_episode", scope="production", source_revision=source_revision),
            production_config=self.campaign, qualification_path=self.qualification_path,
            identity_table=self.identity_table, runtime_qualification=self.runtime_document, resume=resume)


def prepare_circuit(cohort: ProductionCohort, base_config: Path, circuit_index: int,
                    output: Path, *, native_qualification: Path | None = None,
                    backend: str | None = None) -> dict:
    """Bind production intent to the config hashed by construction validators."""
    cohort.require_current()
    require(not output.exists(), "BENCHMARK_PRODUCTION_PREPARATION_EXISTS")
    identity = cohort.identity(circuit_index, 0, 0)
    config = copy.deepcopy(yaml.safe_load(base_config.read_text()))
    if cohort.scenario == "v1_visual":
        require(config["expected_full_nodes"] == 230924 and config["full_fraction"] == 1.0
                and config["canary_fraction"] == .01, "BENCHMARK_PRODUCTION_FULL_V1_REQUIRED")
        require(config == cohort.qualification["source_recipes"]["construction_config"],
                "BENCHMARK_PRODUCTION_V1_QUALIFIED_CONSTRUCTION_RECIPE_DIFFERS")
        config = cohort.v1_construction_config(identity)
    elif cohort.scenario == "pfc_working_memory":
        require(config == cohort.qualification["source_recipes"]["circuit_configs"][identity["circuit_id"]],
                "BENCHMARK_PRODUCTION_PFC_QUALIFIED_CIRCUIT_RECIPE_DIFFERS")
        require(native_qualification is not None and backend is not None,
                "BENCHMARK_PRODUCTION_PFC_NATIVE_QUALIFICATION_REQUIRED")
        qualify_pfc_backend(cohort, identity, native_qualification, backend)
        config = cohort.pfc_circuit_config(identity)
    else:
        require(campaign_config(base_config) == cohort.campaign,
                "BENCHMARK_PRODUCTION_CA3_CAMPAIGN_DIFFERS")
        # This is the existing source validator's internal execution scope.
        # The production declaration and final archive hold cohort membership.
        config["ca3_source_realization"] = SourceRealization(
            identity["circuit_id"], identity["circuit_id"] + "_formation", "formation",
            "development", identity["connectivity_seed"], identity["connectivity_seed"], 1).record()
    config["benchmark_circuit_candidate"] = cohort.declaration(identity)
    output.mkdir(parents=True)
    (output / "circuit.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    report = dict(status="PREPARED_QUALIFIED_PRODUCTION_CIRCUIT_INPUT",
                  declaration=cohort.declaration(identity), config_sha256=sha256_file(output / "circuit.yaml"),
                  source_simulations=0, complete_episodes=0, production_episodes=0)
    freeze_json(output / "report.json", report)
    return report


def bind_v1(cohort: ProductionCohort, identity: dict, recurrent: dict, inputs: dict,
            native: dict, rates: dict, spikes: dict) -> dict:
    for report in (recurrent, inputs):
        cohort.verify_declaration(report["original_config"], identity)
        config = report["original_config"]
        require(config == cohort.v1_construction_config(identity),
                "BENCHMARK_PRODUCTION_V1_QUALIFIED_CONSTRUCTION_RECIPE_DIFFERS")
        require(config["builder_seed"] == identity["connectivity_seed"]
                and config["glif_assignment_seed"] == identity["connectivity_seed"] + 1000000
                and report["original_config_sha256"] == native["original_config_sha256"],
                "BENCHMARK_PRODUCTION_V1_CONSTRUCTION_SEED_OR_BINDING_DIFFERS")
    require(recurrent["rng"]["builder_seed"] == identity["connectivity_seed"]
            and recurrent["rng"]["glif_assignment_seed"] == identity["connectivity_seed"] + 1000000
            and inputs["rng"]["lgn_selection_python_seed"] == identity["connectivity_seed"],
            "BENCHMARK_PRODUCTION_V1_ACTUAL_RANDOM_STREAM_DIFFERS")
    expected = cohort.v1_visual_config(identity)
    require(spikes["config"] == expected
            and rate_config_without_noise(rates["config"]) == rate_config_without_noise(expected)
            and native["bindings"]["spike_inputs"]["config"] == expected,
            "BENCHMARK_PRODUCTION_V1_ACTUAL_CONTROL_OR_NOISE_DIFFERS")
    return dict(scenario_id=cohort.scenario, cohort_identity=identity,
                source_qualification_sha256=cohort.qualification_sha256,
                identity_table_sha256=cohort.identity_table_sha256,
                construction_report_sha256=native["bindings"]["recurrent"]["files"]["primary/report.json"]["sha256"],
                input_report_sha256=native["bindings"]["inputs"]["files"]["primary/report.json"]["sha256"],
                construction_config=copy.deepcopy(recurrent["original_config"]))


def _bind_ca3_config(cohort: ProductionCohort, identity: dict, config: dict,
                     config_sha256: str, formation: dict) -> dict:
    cohort.verify_declaration(config, identity)
    require(config.get("simulation_campaign") == cohort.campaign,
            "BENCHMARK_PRODUCTION_CA3_CONFIG_DIFFERS")
    binding = formation.get("realization_circuit_binding", {})
    require(binding.get("source_realization_config_sha256") == config_sha256
            and binding.get("circuit_id") == identity["circuit_id"]
            and binding.get("circuit_seed") == identity["connectivity_seed"]
            and binding.get("fresh_native_graph_constructed") is True,
            "BENCHMARK_PRODUCTION_CA3_FRESH_BOUND_FORMATION_REQUIRED")
    return dict(scenario_id=cohort.scenario, cohort_identity=identity,
                source_qualification_sha256=cohort.qualification_sha256,
                identity_table_sha256=cohort.identity_table_sha256,
                source_config_sha256=config_sha256,
                # Source YAML may contain integer mapping keys. The archive
                # stores JSON. Preserve the raw YAML document separately.
                construction_config=json.loads(json.dumps(config, allow_nan=False)))


def bind_ca3(cohort: ProductionCohort, identity: dict, prepared_config: Path,
             formation: dict) -> dict:
    return _bind_ca3_config(cohort, identity, yaml.safe_load(prepared_config.read_text()),
                            sha256_file(prepared_config), formation)


def source_document(path: Path) -> dict:
    """Retain the bytes consumed by the archive admission check."""
    payload = path.read_bytes()
    return dict(payload=payload.decode("utf-8"), sha256=hashlib.sha256(payload).hexdigest())


def authenticated_document(admission: dict, role: str, expected_sha256: str):
    document = admission.get("source_documents", {}).get(role, {})
    payload = document.get("payload")
    require(isinstance(payload, str) and isinstance(expected_sha256, str)
            and hashlib.sha256(payload.encode("utf-8")).hexdigest() == expected_sha256
            and document.get("sha256") == expected_sha256,
            "BENCHMARK_PRODUCTION_SOURCE_DOCUMENT_DIFFERS:" + role)
    return payload



def glif_backend_qualification(cohort: ProductionCohort) -> dict:
    backend = cohort.qualification.get("source_recipes", {}).get("glif_cuda", {})
    code = backend.get("runtime_code", {})
    required = {"v1_glif_network.py", "v1_glif_component_emulator.py",
                "v1_glif_cuda.py", "v1_nest_time_semantics.py"}
    require(set(code) == required and all(isinstance(value, str) and len(value) == 64 for value in code.values())
            and isinstance(backend.get("library_sha256"), str) and len(backend["library_sha256"]) == 64
            and backend.get("component_rtol") == 1e-12 and backend.get("component_atol") == 1e-12,
            "BENCHMARK_PRODUCTION_GLIF_BACKEND_NOT_QUALIFIED")
    document = getattr(cohort, "runtime_document", None)
    if document is not None:
        return runtime.validate_document(document, cohort.qualification_sha256, backend)
    return backend


def verify_glif_admission(cohort: ProductionCohort, identity: dict, admission: dict) -> None:
    selected = getattr(cohort, "runtime_document", None)
    require(admission.get("runtime_qualification_sha256") ==
            (selected["sha256"] if selected is not None else None),
            "BENCHMARK_PRODUCTION_EXPLICIT_RUNTIME_BINDING_REQUIRED")
    backend = glif_backend_qualification(cohort)
    documents = {role: json.loads(authenticated_document(admission, role, admission.get(digest)))
        for role, digest in (("execution", "execution_report_sha256"), ("replay", "replay_report_sha256"),
                             ("replay_comparison", "replay_comparison_sha256"),
                             ("circuit_reference", "circuit_reference_report_sha256"),
                             ("circuit_validation", "circuit_validation_sha256"))}
    reference, validation = documents["circuit_reference"], documents["circuit_validation"]
    require(reference["original_config_sha256"] == admission["construction_config_sha256"]
            and reference["cortical_units"] == 230924
            and reference["execution_clock"]["resolution_ms"] == cohort.campaign["scenarios"]["v1_visual"]["task_protocol"]["dt_ms"],
            "BENCHMARK_PRODUCTION_GLIF_CIRCUIT_REFERENCE_DIFFERS")
    require(validation["status"] in GLIF_CIRCUIT_REFERENCE_STATUSES
            and validation["all_spike_pairs_exact"] and validation["all_unit_counts_exact"]
            and validation["terminal_voltage_within_component_tolerance"]
            and validation["source_bindings"]["files"]["primary/report.json"]["sha256"]
                == admission["circuit_reference_report_sha256"]
            and validation["runtime_code"] == backend["runtime_code"]
            and validation["cuda_execution"]["library_sha256"] == backend["library_sha256"]
            and validation["component_rtol"] == backend["component_rtol"]
            and validation["component_atol"] == backend["component_atol"],
            "BENCHMARK_PRODUCTION_GLIF_CIRCUIT_VALIDATION_DIFFERS")
    for role in ("execution", "replay"):
        report = documents[role]
        require(report["status"] == "PASS_SOURCE_INPUT_GLIF_CUDA_EXECUTION"
                and report["runtime_code"] == backend["runtime_code"]
                and report["cuda_execution"]["library_sha256"] == backend["library_sha256"]
                and report["execution_clock"] == reference["execution_clock"]
                and report["units"] == reference["cortical_units"]
                and report["connection_rows"] == reference["connections"]["native_connections_to_cortex"]
                and report["input_bindings"]["config"] == cohort.v1_visual_config(identity)
                and report["circuit_bindings"]["files"]["primary/report.json"]["sha256"]
                    == admission["circuit_reference_report_sha256"],
                "BENCHMARK_PRODUCTION_GLIF_EXECUTION_DIFFERS:" + role)
    comparison = documents["replay_comparison"]
    require(comparison["status"] == "PASS_GLIF_SOURCE_BLOCK_REPLAY"
            and comparison["passed_conditions"] == 10 and comparison["failed_conditions"] == 0
            and str(comparison["primary_job_id"]) != str(comparison["replay_job_id"])
            and str(comparison["primary_job_id"]) == str(documents["execution"]["job_id"])
            and str(comparison["replay_job_id"]) == str(documents["replay"]["job_id"])
            and comparison["runtime_code"] == backend["runtime_code"]
            and comparison["component_rtol"] == backend["component_rtol"]
            and comparison["component_atol"] == backend["component_atol"]
            and len(comparison["conditions"]) == 10,
            "BENCHMARK_PRODUCTION_GLIF_COMPLETE_REPLAY_REQUIRED")
    row = comparison["conditions"][identity["condition_index"]]
    require(row["identity"] == identity and row["status"] == "PASS"
            and row["primary_report_sha256"] == admission["execution_report_sha256"]
            and row["replay_report_sha256"] == admission["replay_report_sha256"]
            and row["all_spike_pairs_exact"] and row["all_exact_checks"]
            and all(row["all_exact_checks"].values()) and row["terminal_voltage_within_component_tolerance"]
            and row["library_sha256"] == backend["library_sha256"],
            "BENCHMARK_PRODUCTION_GLIF_REPLAY_CASE_DIFFERS")


def verify_source_admission(cohort: ProductionCohort, identity: dict, admission: dict) -> None:
    """Authenticate construction evidence at the archive-write boundary."""
    require(admission.get("scenario_id") == cohort.scenario
            and admission.get("cohort_identity") == identity
            and admission.get("source_qualification_sha256") == cohort.qualification_sha256
            and admission.get("identity_table_sha256") == cohort.identity_table_sha256,
            "BENCHMARK_PRODUCTION_SOURCE_ADMISSION_REQUIRED")
    cohort.verify_declaration(admission.get("construction_config", {}), identity)
    if cohort.scenario == "v1_visual":
        reports = {}
        for role, digest in (("recurrent", "construction_report_sha256"),
                             ("inputs", "input_report_sha256")):
            reports[role] = json.loads(authenticated_document(admission, role, admission.get(digest)))
            require(reports[role].get("original_config") == admission["construction_config"]
                    == cohort.v1_construction_config(identity),
                    "BENCHMARK_PRODUCTION_AUTHENTICATED_CONSTRUCTION_CONFIG_DIFFERS")
        require(reports["recurrent"]["original_config_sha256"] == reports["inputs"]["original_config_sha256"],
                "BENCHMARK_PRODUCTION_AUTHENTICATED_CONSTRUCTION_BINDING_DIFFERS")
        if admission.get("execution_backend") == "glif_cuda":
            require(admission.get("construction_config_sha256") == reports["recurrent"]["original_config_sha256"],
                    "BENCHMARK_PRODUCTION_GLIF_CONSTRUCTION_HASH_DIFFERS")
            verify_glif_admission(cohort, identity, admission)
        else:
            require(admission.get("execution_backend", "native") == "native",
                    "BENCHMARK_PRODUCTION_UNKNOWN_V1_BACKEND")
    elif cohort.scenario == "pfc_working_memory":
        config = yaml.safe_load(authenticated_document(admission, "prepared_config", admission.get("source_config_sha256")))
        report = json.loads(authenticated_document(admission, "source_report", admission.get("source_report_sha256")))
        request = json.loads(authenticated_document(admission, "source_request", admission.get("source_request_sha256")))
        require(config == admission["construction_config"] == cohort.pfc_episode_config(identity),
                "BENCHMARK_PRODUCTION_PFC_AUTHENTICATED_SOURCE_CONFIG_DIFFERS")
        require(report.get("status") == "PFC_SOURCE_EXECUTION_COMPLETE"
                and report.get("source_execution_complete") is True
                and report.get("resolved_config") == config and report.get("selected_case") == config["cases"][0]
                and report.get("case_id") == identity["episode_id"]
                and report.get("config_sha256") == admission["source_config_sha256"]
                and report.get("completed_case_count") == 1
                and report.get("random_stream_policy") == "independent_initial_external"
                and report.get("seeds") == dict(initial_rng_seed=identity["initial_state_seed"],
                                               external_noise_seed=identity["background_seed"])
                and report.get("validations") and all(row.get("passed") is True
                                                      for row in report["validations"].values()),
                "BENCHMARK_PRODUCTION_PFC_AUTHENTICATED_EXECUTION_DIFFERS")
        require(request.get("status") == "PFC_SOURCE_EXECUTION_REQUESTED"
                and all(request.get(key) == report.get(key) for key in ("case_id", "selected_case",
                        "config_sha256", "backend", "backend_qualification", "code", "dimensions")),
                "BENCHMARK_PRODUCTION_PFC_AUTHENTICATED_REQUEST_DIFFERS")
        qualification = report["backend_qualification"]
        expected = cohort.qualification["source_recipes"]["native_qualifications"][identity["circuit_id"]]
        receipt = json.loads(authenticated_document(admission, "native_qualification", qualification.get("receipt_sha256")))
        require(report["backend"] == expected["backend"] and qualification.get("qualification") == receipt
                and all(qualification.get(key) == expected[key] for key in ("receipt_sha256", "delivery_sha256"))
                and receipt.get("validated_dt_ms") == [config["dt_ms"]],
                "BENCHMARK_PRODUCTION_PFC_AUTHENTICATED_NATIVE_QUALIFICATION_DIFFERS")
        pfc_kernel_parameters.require_kernel_coverage(receipt, config,
            report["code"]["kernel_parameterization_sha256"])
    else:
        config = yaml.safe_load(authenticated_document(admission, "prepared_config", admission.get("source_config_sha256")))
        formation = json.loads(authenticated_document(admission, "formation", admission.get("formation_report_sha256")))
        bound = _bind_ca3_config(cohort, identity, config, admission["source_config_sha256"], formation)
        require(all(admission.get(key) == value for key, value in bound.items()),
                "BENCHMARK_PRODUCTION_AUTHENTICATED_CONSTRUCTION_CONFIG_DIFFERS")



def pfc_completion_costs(cohort: ProductionCohort, identity: dict, output: Path,
                         dataset: CircuitDataset, episode: CircuitEpisode) -> dict:
    """Read original completion costs without changing committed episode payloads."""
    source_root = output / "source_execution" / identity["episode_id"]
    path = source_root / "report.json"
    if not path.exists():
        return dict(production_cost_status="UNAVAILABLE_ORIGINAL_COMPLETION_RECEIPT",
                    production_costs=None)
    report = json.loads(path.read_text())
    admission = episode.source_record["production_source_admission"]
    row = dataset.episodes[identity["episode_id"]]
    require(report.get("status") == "EXPORTED_QUALIFIED_PRODUCTION_EPISODE"
            and report.get("identity") == identity
            and report.get("source_qualification_sha256") == cohort.qualification_sha256
            and report.get("identity_table_sha256") == cohort.identity_table_sha256
            and report.get("source_report_sha256") == admission["source_report_sha256"]
            and report.get("source_delivery_sha256") == admission["source_delivery_sha256"]
            and report.get("dataset_episode_binding") == dict(
                shard=row["shard"], sha256=dataset.shards[row["shard"]]["sha256"]),
            "BENCHMARK_PRODUCTION_PFC_COMPLETION_RECEIPT_DIFFERS")
    costs = report.get("production_costs", {})
    require(isinstance(costs, dict) and isinstance(costs.get("scopes"), dict)
            and all(type(costs.get(name)) in (int, float) and np.isfinite(costs[name]) and costs[name] >= 0
                    for name in ("source_exporter_elapsed_seconds", "source_exporter_wall_seconds",
                                 "cohort_writer_wall_seconds", "export_wall_seconds")),
            "BENCHMARK_PRODUCTION_PFC_COMPLETION_COST_INVALID")
    job = report.get("source_producer_job")
    if job is not None:
        intent = json.loads((source_root / "execution.json").read_text())
        launcher_path = source_root / "launcher_exit.json"
        launcher = json.loads(launcher_path.read_text()) if launcher_path.is_file() else {}
        require(isinstance(job, str) and job.isdecimal() and intent.get("identity") == identity
                and intent.get("producer_job") == job
                and (not launcher or launcher.get("producer_job") == job)
                and intent.get("source_archive") == report.get("source_archive")
                and intent.get("batch_delivery") == report.get("source_batch_delivery")
                and costs.get("source_launcher_wall_seconds") == launcher.get("wall_seconds")
                and costs.get("scopes", {}).get("source_launcher") == launcher.get("cost_scope"),
                "BENCHMARK_PRODUCTION_PFC_COMPLETION_PRODUCER_DIFFERS")
    launcher_seconds = costs.get("source_launcher_wall_seconds")
    require(launcher_seconds is None or (type(launcher_seconds) in (int, float)
            and np.isfinite(launcher_seconds) and launcher_seconds >= 0),
            "BENCHMARK_PRODUCTION_PFC_COMPLETION_COST_INVALID")
    return dict(production_cost_status="RECORDED_ORIGINAL_COMPLETION",
                production_costs=costs, production_completion_report=str(path.resolve()),
                source_producer_job=job, source_archive=report["source_archive"],
                source_batch_delivery=report.get("source_batch_delivery"))


def completed_episode(cohort: ProductionCohort, identity: dict, output: Path,
                      *, resume: bool) -> dict | None:
    cohort.require_current()
    cohort.require_identity(identity)
    if not output.exists():
        return None
    require(resume, "BENCHMARK_PRODUCTION_EXPLICIT_RESUME_REQUIRED")
    if not (output / "dataset.json").exists():
        # A retained CA3 attempt is reconciled by run_ca3 before repeating work.
        require(all(path.name == "source_execution" and path.is_dir() and not path.is_symlink()
                    for path in output.iterdir()), "BENCHMARK_PRODUCTION_UNRESOLVED_EXECUTION_OUTPUT")
        return None
    catalog = json.loads((output / "dataset.json").read_text())
    next_closed = output / f"episodes_{len(catalog['shards']):06d}.h5"
    # An unpublished closed shard may need the explicitly selected proof.
    # Normal reuse resolves each episode's recorded proof without adding one.
    recovery_cohort = (cohort if next_closed.is_file() else
                       replace(cohort, runtime_document=None, runtime_path=None))
    with recovery_cohort.writer(output, cohort.qualification["source_revision"], resume=True):
        pass
    dataset = CircuitDataset(output, splits=(identity["split"],), allow_test=identity["split"] == "test")
    if identity["episode_id"] not in dataset.episodes:
        return None
    episode = read_episode(dataset, dataset.episodes[identity["episode_id"]])
    units = read_circuit(dataset, episode.circuit_id)
    episode.validate(len(units.unit_ids))
    require(episode.source_record.get("cohort_identity") == identity
            and episode.source_record.get("production_declaration") == cohort.declaration(identity)
            and episode.seeds == {k: identity[k] for k in COHORT_SEEDS},
            "BENCHMARK_PRODUCTION_COMMITTED_IDENTITY_DIFFERS")
    admission = episode.source_record.get("production_source_admission", {})
    require(admission.get("cohort_identity") == identity
            and admission.get("source_qualification_sha256") == cohort.qualification_sha256
            and admission.get("identity_table_sha256") == cohort.identity_table_sha256,
            "BENCHMARK_PRODUCTION_COMMITTED_ADMISSION_DIFFERS")
    selected_sha = admission.get("runtime_qualification_sha256")
    selected = dataset.catalog.get("runtime_qualifications", {}).get(selected_sha)
    require(selected_sha is None or selected is not None, "COMMITTED_RUNTIME_DOCUMENT_MISSING")
    readback_cohort = replace(cohort, runtime_document=selected, runtime_path=None)
    verify_source_admission(readback_cohort, identity, admission)
    costs = (pfc_completion_costs(cohort, identity, output, dataset, episode)
             if cohort.scenario == "pfc_working_memory" else {})
    return dict(status="REUSED_COMMITTED_PRODUCTION_EPISODE", identity=identity,
                complete_episodes=1, new_source_simulations=0,
                source_events=len(episode.spike_ticks),
                native_seed_streams=episode.source_record["native_seed_streams"],
                dataset_catalog_sha256=sha256_file(output / "dataset.json"), **costs)


def write_episode(cohort: ProductionCohort, identity: dict, source: CircuitDataset,
                  output: Path, *, admission: dict, resume: bool = False,
                  source_episode_id: str | None = None) -> dict:
    """Archive a just-verified source result; preserve native identities and seeds."""
    cohort.require_current()
    cohort.require_identity(identity)
    verify_source_admission(cohort, identity, admission)
    require(source.catalog["binding"]["scenario_id"] == cohort.scenario
            and source.catalog["binding"]["scope"] == "source_reproduction",
            "BENCHMARK_PRODUCTION_SOURCE_FORMAT_DIFFERS")
    require(len(source.episodes) == source.catalog["complete_episodes"],
            "BENCHMARK_PRODUCTION_SOURCE_CATALOG_COUNT_DIFFERS")
    if source_episode_id is None:
        require(len(source.episodes) == 1, "BENCHMARK_PRODUCTION_ONE_FRESH_SOURCE_EPISODE_REQUIRED")
        row = next(source.iter_episodes())
    else:
        require(source_episode_id in source.episodes, "BENCHMARK_PRODUCTION_SOURCE_EPISODE_MISSING")
        row = source.episodes[source_episode_id]
    episode = read_episode(source, row)
    units = read_circuit(source, episode.circuit_id)
    required_units = {"v1_visual": 230924, "pfc_working_memory": 100000, "ca3_pattern_completion": 168106}[cohort.scenario]
    require(len(units.unit_ids) == required_units and np.array_equal(units.unit_ids, np.arange(required_units)),
            "BENCHMARK_PRODUCTION_ALL_ORDERED_CORE_UNITS_REQUIRED")
    if cohort.scenario == "v1_visual":
        bindings = episode.source_record["upstream_bindings"]
        if admission.get("execution_backend") == "glif_cuda":
            require(source.catalog["binding"].get("execution_backend") == "source_derived_float64_cuda"
                    and episode.source_record.get("execution_backend") == "source_derived_float64_cuda"
                    and episode.source_record.get("execution_report_sha256") == admission.get("execution_report_sha256")
                    and episode.source_record.get("circuit_reference_report_sha256") == admission.get("circuit_reference_report_sha256")
                    and episode.source_record.get("native_nest_simulation") is False,
                    "BENCHMARK_PRODUCTION_GLIF_SOURCE_ADMISSION_DIFFERS")
        else:
            require(episode.source_record["native_report_sha256"] == admission.get("native_report_sha256"),
                    "BENCHMARK_PRODUCTION_V1_SOURCE_ADMISSION_DIFFERS")
        require(bindings["recurrent"]["files"]["primary/report.json"]["sha256"] == admission.get("construction_report_sha256")
                and bindings["inputs"]["files"]["primary/report.json"]["sha256"] == admission.get("input_report_sha256"),
                "BENCHMARK_PRODUCTION_V1_SOURCE_ADMISSION_DIFFERS")
        require(bindings["spike_inputs"]["config"] == cohort.v1_visual_config(identity),
                "BENCHMARK_PRODUCTION_V1_QUALIFIED_DRIVER_RECIPE_DIFFERS")
        data = describe_episode(episode, units)
        expected = condition_for_episode(cohort.campaign["scenarios"][cohort.scenario]["task_protocol"],
                                         cohort.scenario, identity)
        require(data["direction_deg"] == expected["direction_deg"]
                and data["contrast_percent"] == expected["contrast_percent"]
                and episode.seeds["lgn_starting_noise_seed"] == identity["stimulus_seed"]
                and episode.seeds["background_starting_noise_seed"] == identity["background_seed"],
                "BENCHMARK_PRODUCTION_V1_SOURCE_EPISODE_DIFFERS")
        semantics = dict(initial_state_seed="unused; deterministic source reset then stochastic gray warmup",
                         stimulus_seed="LGN sampler starting seed", background_seed="background sampler starting seed")
    elif cohort.scenario == "pfc_working_memory":
        config = cohort.pfc_episode_config(identity)
        expected = pfc_kernel_parameters.circuit_provenance(config, pfc_source_dataset.SOURCE_ENTRY_SHA256)
        require(units.circuit_provenance == expected
                and episode.source_record.get("source_report_sha256") == admission.get("source_report_sha256")
                and episode.source_record.get("source_delivery_sha256") == admission.get("source_delivery_sha256")
                and source.catalog["binding"].get("config_sha256") == admission.get("source_config_sha256")
                and source.catalog["binding"].get("resolved_source_config") == config
                and episode.source_record.get("benchmark_episode_candidate") == config["benchmark_episode_candidate"]
                and episode.seeds == dict(initial_rng_seed=identity["initial_state_seed"],
                                          external_noise_seed=identity["background_seed"]),
                "BENCHMARK_PRODUCTION_PFC_SOURCE_ADMISSION_DIFFERS")
        n_e, n_i = 80000, 20000
        fields = np.r_[np.arange(n_e) * 16 // n_e, np.full(n_i, 16)]
        require(np.array_equal(units.populations, np.r_[np.full(n_e, "E"), np.full(n_i, "I")])
                and np.array_equal(units.field_index, fields) and len(units.field_names) == 17
                and units.coordinates is None and units.source_tuning_angle_deg is None,
                "BENCHMARK_PRODUCTION_PFC_SOURCE_NATIVE_FIELD_MAP_DIFFERS")
        required_private = {"initial_voltage_mV"}
        required_private.update("terminal__" + population + "__" + name
            for population in ("E", "I") for name in pfc_source_execution._STATE_FIELDS[population]
            + pfc_source_execution._IDENTITY_FIELDS)
        required_private.update("delivered_current__" + name for name in pfc_source_execution._CURRENT_ARRAY_DTYPES)
        require(set(episode.privileged) == required_private
                and episode.source_record.get("applied_current_observation", {}).get("status") == "VERIFIED_ACTUAL_CURRENT"
                and episode.source_record.get("source_unit_metadata", {}).get("terminal_state_units")
                   == dict(v="V", I_stim="A", lastspike="s", synaptic_state_variables="dimensionless", not_refractory="boolean"),
                "BENCHMARK_PRODUCTION_PFC_COMPLETE_PRIVATE_STATES_REQUIRED")
        observed = {name: episode.privileged["delivered_current__" + name]
                    for name in pfc_source_execution._CURRENT_ARRAY_DTYPES}
        ticks, values, features = pfc_source_dataset._controls(config, config["cases"][0], observed)
        require(np.array_equal(episode.control_ticks, ticks) and np.array_equal(episode.control_values, values)
                and episode.control_features == features,
                "BENCHMARK_PRODUCTION_PFC_NATIVE_CONTROL_SEMANTICS_DIFFERS")
        semantics = dict(initial_state_seed="source uniform voltage draws, I then E; archived E then I",
                         background_seed="native per-tick binomial input stream, I then E",
                         connectivity_seed="unused as RNG; circuit indexes a frozen distinct numerical kernel",
                         stimulus_seed="unused; canonical cue and distractor commands are deterministic")
    else:
        require(episode.source_record["formation_binding"]["formation_report_sha256"] == admission.get("formation_report_sha256")
                and admission.get("source_config_sha256") is not None,
                "BENCHMARK_PRODUCTION_CA3_SOURCE_ADMISSION_DIFFERS")
        require(episode.seeds == dict(circuit_seed=identity["connectivity_seed"],
            uniform_seed=identity["background_seed"], engine_seed=identity["initial_state_seed"],
            cue_seed=identity["stimulus_seed"]), "BENCHMARK_PRODUCTION_CA3_SOURCE_STREAM_DIFFERS")
        expected_condition = condition_for_episode(cohort.campaign["scenarios"][cohort.scenario]["task_protocol"],
                                                  cohort.scenario, identity)
        require(episode.source_record.get("condition") == expected_condition,
                "BENCHMARK_PRODUCTION_CA3_SOURCE_CONDITION_DIFFERS")
        semantics = dict(initial_state_seed="common seed of nine host engines; not a microscopic initial-state draw",
                         stimulus_seed="cue membership permutation seed", background_seed="shared uniform-current stream seed")
    native_identity = {name: getattr(episode, name) for name in ("episode_id", "circuit_id", "block_id", "split")}
    record = dict(episode.source_record, native_identity=native_identity,
                  native_seed_streams=dict(episode.seeds), cohort_seed_semantics=semantics,
                  production_declaration=cohort.declaration(identity), cohort_identity=identity,
                  production_source_admission=admission)
    units = replace(units, circuit_id=identity["circuit_id"])
    episode = replace(episode, episode_id=identity["episode_id"], circuit_id=identity["circuit_id"],
                      block_id=identity["block_id"], split=identity["split"],
                      seeds={k: identity[k] for k in COHORT_SEEDS}, source_record=record)
    with cohort.writer(output, units.source_revision, resume=resume) as writer:
        writer.register_circuit(units)
        writer.add(episode)
    # Packaging readback may inspect the newly written test shard. It is not
    # test access for fitting, model selection, or threshold calibration.
    actual = CircuitDataset(output, splits=(identity["split"],), allow_test=identity["split"] == "test")
    reloaded = read_episode(actual, actual.episodes[episode.episode_id])
    for name in ("spike_ticks", "spike_unit_indices", "control_ticks", "control_values"):
        require(np.array_equal(getattr(reloaded, name), getattr(episode, name)),
                "BENCHMARK_PRODUCTION_ARCHIVE_VALUES_DIFFER:" + name)
    require(reloaded.seeds == episode.seeds and reloaded.source_record == record
            and reloaded.task_outcome == episode.task_outcome
            and set(reloaded.privileged) == set(episode.privileged)
            and all(np.array_equal(reloaded.privileged[k], v) for k, v in episode.privileged.items()),
            "BENCHMARK_PRODUCTION_ARCHIVE_SOURCE_RECORD_DIFFERS")
    return dict(status="EXPORTED_QUALIFIED_PRODUCTION_EPISODE", identity=identity,
                complete_episodes=1, source_events=len(episode.spike_ticks),
                native_seed_streams=record["native_seed_streams"],
                dataset_catalog_sha256=sha256_file(output / "dataset.json"))


def prepare_pfc_episode(cohort: ProductionCohort, identity: dict, qualification: Path,
                        backend: str, output: Path, *, production_output: Path | None = None,
                        resume: bool = False) -> dict:
    """Declare exact episode controls and RNGs before invoking the source wrapper."""
    cohort.require_current(); cohort.require_identity(identity)
    require(cohort.scenario == "pfc_working_memory", "BENCHMARK_PRODUCTION_PFC_SCENARIO_REQUIRED")
    if production_output is not None:
        existing = completed_episode(cohort, identity, production_output, resume=resume)
        if existing is not None:
            return existing
    retained_preparation = output.exists()
    require(resume or not retained_preparation, "BENCHMARK_PRODUCTION_PREPARATION_EXISTS")
    native = qualify_pfc_backend(cohort, identity, qualification, backend)
    config = cohort.pfc_episode_config(identity)
    pfc_source_execution._case_config(config, identity["episode_id"])
    output.mkdir(parents=True, exist_ok=resume)
    path = output / "episode.yaml"
    if retained_preparation:
        require(all(item.name in ("episode.yaml", "report.json") and item.is_file()
                    and not item.is_symlink() for item in output.iterdir()),
                "BENCHMARK_PRODUCTION_PFC_RETAINED_PREPARATION_CHANGED")
        if path.exists():
            require(yaml.safe_load(path.read_text()) == config,
                    "BENCHMARK_PRODUCTION_PFC_RETAINED_PREPARATION_CHANGED")
    if not path.exists():
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    report = dict(status="PREPARED_QUALIFIED_PFC_PRODUCTION_EPISODE", identity=identity,
        config_sha256=sha256_file(path), native_qualification=native,
        source_runner="Libs/scripts/run_pfc_source_cases.py",
        source_environment=dict(NEURONSIM_PFC_SOURCE_CONFIG=str(path.resolve()),
            NEURONSIM_PFC_SOURCE_CASE_ID=identity["episode_id"],
            NEURONSIM_PFC_SOURCE_QUALIFICATION=str(qualification.resolve()),
            NEURONSIM_PFC_SOURCE_BACKEND=backend),
        source_simulations=0, complete_episodes=0, production_episodes=0)
    freeze_json(output / "report.json", report)
    return report


def export_pfc(cohort: ProductionCohort, identity: dict, prepared_config: Path,
               source_archive: Path, output: Path, *, resume: bool = False) -> dict:
    """Authenticate the existing native archive before production packaging."""
    existing = completed_episode(cohort, identity, output, resume=resume)
    if existing is not None:
        return existing
    export_started = time.perf_counter()
    require(cohort.scenario == "pfc_working_memory", "BENCHMARK_PRODUCTION_PFC_SCENARIO_REQUIRED")
    config = yaml.safe_load(prepared_config.read_text())
    cohort.verify_declaration(config, identity)
    require(config == cohort.pfc_episode_config(identity),
            "BENCHMARK_PRODUCTION_PFC_PREPARED_EPISODE_DIFFERS")
    config_sha = sha256_file(prepared_config)
    _, registered, _, _ = pfc_source_dataset._benchmark_registration(config)
    status, header = pfc_source_dataset._source_header(source_archive, config, registered, config_sha)
    require(header is not None and status["status"] == "COMPLETE",
            "BENCHMARK_PRODUCTION_PFC_COMPLETE_NATIVE_ARCHIVE_REQUIRED")
    archive = header["archive"]
    report = header["report"]
    simulator_cost = report.get("simulator_cost")
    require(isinstance(simulator_cost, dict)
            and type(simulator_cost.get("wall_seconds")) in (int, float)
            and np.isfinite(simulator_cost["wall_seconds"]) and simulator_cost["wall_seconds"] >= 0
            and isinstance(simulator_cost.get("scope"), str) and bool(simulator_cost["scope"]),
            "BENCHMARK_PRODUCTION_PFC_SOURCE_SIMULATOR_COST_REQUIRED")
    qualify_pfc_backend(cohort, identity, archive / "qualification", report["backend"])
    admission = dict(scenario_id=cohort.scenario, cohort_identity=identity,
        source_qualification_sha256=cohort.qualification_sha256,
        identity_table_sha256=cohort.identity_table_sha256, construction_config=config,
        source_config_sha256=config_sha, source_report_sha256=status["report_sha256"],
        source_delivery_sha256=status["delivery_sha256"],
        source_request_sha256=report["outputs"]["request.json"]["sha256"],
        source_documents=dict(prepared_config=source_document(archive / "source_run/config.yaml"),
            source_report=source_document(archive / "source_run/report.json"),
            source_request=source_document(archive / "source_run/request.json"),
            native_qualification=source_document(archive / "qualification/comparison/qualification.json")))
    verify_source_admission(cohort, identity, admission)
    require(os.environ.get("SLURMD_NODENAME") in ("cpu03", "cpu04", "cpu05", "cpu06") and os.environ.get("SLURM_JOB_ID")
            and not os.environ.get("SLURM_JOB_GPUS"), "BENCHMARK_PRODUCTION_PFC_ALLOCATED_CPU_REQUIRED")
    # The source exporter authenticates all code, units, inputs, events and actual
    # currents against the native delivery before producing the writer input.
    with tempfile.TemporaryDirectory(prefix="neuronsim-production-pfc-", dir="/tmp") as temporary:
        source_export_started = time.perf_counter()
        result = pfc_source_dataset.export([archive], prepared_config, Path(temporary) / "source")
        source_export_wall = time.perf_counter() - source_export_started
        require(result["status"] == "EXPORTED_VERIFIED_PFC_DEVELOPMENT_DATASET",
                "BENCHMARK_PRODUCTION_PFC_SOURCE_EXPORT_FAILED")
        source = CircuitDataset(Path(temporary) / "source/dataset", splits=("development",))
        writer_started = time.perf_counter()
        committed = write_episode(cohort, identity, source, output, admission=admission, resume=resume)
        writer_wall = time.perf_counter() - writer_started
    export_wall = time.perf_counter() - export_started
    source_root = output / "source_execution" / identity["episode_id"]
    producer, batch_delivery, launcher_wall, launcher_scope = None, None, None, None
    if (source_root / "execution.json").is_file():
        intent = json.loads((source_root / "execution.json").read_text())
        require(intent.get("identity") == identity and intent.get("source_archive") == str(archive.resolve()),
                "BENCHMARK_PRODUCTION_PFC_COMPLETION_SOURCE_DIFFERS")
        producer, batch_delivery = intent["producer_job"], intent["batch_delivery"]
        launcher_path = source_root / "launcher_exit.json"
        launcher = json.loads(launcher_path.read_text()) if launcher_path.is_file() else {}
        require(not launcher or launcher.get("producer_job") == producer,
                "BENCHMARK_PRODUCTION_PFC_COMPLETION_PRODUCER_DIFFERS")
        launcher_wall, launcher_scope = launcher.get("wall_seconds"), launcher.get("cost_scope")
    dataset = CircuitDataset(output, splits=(identity["split"],), allow_test=identity["split"] == "test")
    shard = dataset.episodes[identity["episode_id"]]["shard"]
    costs = dict(source_launcher_wall_seconds=launcher_wall,
        source_simulator_cost=simulator_cost,
        source_exporter_elapsed_seconds=result["elapsed_seconds"],
        source_exporter_wall_seconds=source_export_wall,
        cohort_writer_wall_seconds=writer_wall, export_wall_seconds=export_wall,
        scopes=dict(source_launcher=launcher_scope,
            source_simulator="native report costs; preserves its original scope; not included in export wall",
            source_exporter_elapsed="existing exporter timer: dataset construction and full-unit verification; excludes its initial header admission",
            source_exporter_wall="entire source exporter call, including source authentication, all-unit temporary dataset and verification",
            cohort_writer="entire cohort writer call: admission checks, all-unit commit and complete readback",
            export="prepared/source admission, source export, cohort write/readback and local cleanup; excludes cohort loading, source launch and completion receipt write"))
    completion = dict(committed, production_cost_status="RECORDED_ORIGINAL_COMPLETION",
        production_costs=costs, source_producer_job=producer,
        source_archive=str(archive.resolve()), source_batch_delivery=batch_delivery,
        source_qualification_sha256=cohort.qualification_sha256,
        identity_table_sha256=cohort.identity_table_sha256,
        source_report_sha256=status["report_sha256"], source_delivery_sha256=status["delivery_sha256"],
        dataset_episode_binding=dict(shard=shard, sha256=dataset.shards[shard]["sha256"]))
    freeze_json(source_root / "report.json", completion)
    return dict(completion, production_completion_report=str((source_root / "report.json").resolve()))


def run_pfc(cohort: ProductionCohort, identity: dict, qualification: Path,
            source_python: Path, project: Path, output: Path, *, resume: bool = False) -> dict:
    """Execute one qualified reference episode or export its retained complete source.

    The existing source wrapper owns simulation, resource monitoring and native
    archival. An unresolved attempt never starts another source execution.
    """
    require(cohort.scenario == "pfc_working_memory", "BENCHMARK_PRODUCTION_PFC_SCENARIO_REQUIRED")
    existing = completed_episode(cohort, identity, output, resume=resume)
    if existing is not None:
        return existing
    source_root = output / "source_execution" / identity["episode_id"]
    prepared = source_root / "prepared"
    intent_path = source_root / "execution.json"
    retained_source = intent_path.exists()
    if source_root.exists() and not retained_source:
        require(resume and all(item.name == "prepared" and item.is_dir() and not item.is_symlink()
                               for item in source_root.iterdir()),
                "BENCHMARK_PRODUCTION_PFC_UNRESOLVED_SOURCE_EXECUTION")
    if retained_source:
        require(resume and intent_path.is_file() and (prepared / "report.json").is_file(),
                "BENCHMARK_PRODUCTION_PFC_UNRESOLVED_SOURCE_EXECUTION")
        intent = json.loads(intent_path.read_text())
        preparation = json.loads((prepared / "report.json").read_text())
        require(intent.get("identity") == preparation.get("identity") == identity
                and intent.get("config_sha256") == preparation.get("config_sha256")
                == sha256_file(prepared / "episode.yaml")
                and yaml.safe_load((prepared / "episode.yaml").read_text()) == cohort.pfc_episode_config(identity),
                "BENCHMARK_PRODUCTION_PFC_RETAINED_PREPARATION_CHANGED")
    else:
        require(os.environ.get("SLURMD_NODENAME") in ("cpu03", "cpu04", "cpu05", "cpu06")
                and os.environ.get("SLURM_JOB_ID")
                and not os.environ.get("SLURM_JOB_GPUS") and not os.environ.get("SLURM_STEP_GPUS"),
                "BENCHMARK_PRODUCTION_PFC_ALLOCATED_CPU_REQUIRED")
        source_python, project = source_python.resolve(), project.resolve()
        launcher = project / "Libs/scripts/run_pfc_source_cases.sbatch"
        require(source_python.is_file() and os.access(source_python, os.X_OK)
                and launcher.is_file() and (project / "Libs/scripts/run_pfc_source_cases.py").is_file(),
                "BENCHMARK_PRODUCTION_PFC_SOURCE_RUNTIME_REQUIRED")
        preparation = prepare_pfc_episode(cohort, identity, qualification, "numpy", prepared,
                                          resume=resume)
        job = os.environ["SLURM_JOB_ID"]
        native = source_root / "native"
        intent = dict(status="STARTED_PFC_SOURCE_ATTEMPT", identity=identity,
            config_sha256=preparation["config_sha256"], producer_job=job,
            source_archive=str((native / ("run_" + job)).resolve()),
            batch_delivery=str((native / "batches" / ("run_" + job) / "delivery.json").resolve()),
            source_python=str(source_python), source_project=str(project),
            command=["bash", str(launcher)], native_qualification=preparation["native_qualification"])
        freeze_json(intent_path, intent)
        env = dict(os.environ, **preparation["source_environment"],
            NEURONSIM_PFC_PYTHON=str(source_python), NEURONSIM_PROJECT_ROOT=str(project),
            NEURONSIM_PFC_OUTPUT_BASE=str(native.resolve()), NEURONSIM_PFC_SCRATCH="/tmp",
            SLURM_NTASKS="1", SLURM_CPUS_PER_TASK="1", CUDA_VISIBLE_DEVICES="")
        for name in ("SLURM_ARRAY_TASK_ID", "NEURONSIM_PFC_SOURCE_CASES_JSON",
                     "NEURONSIM_PFC_SOURCE_CONFIG_SHA256", "NEURONSIM_PFC_SOURCE_CASES_SHA256",
                     "NEURONSIM_PFC_SOURCE_CODE_SHA256"):
            env.pop(name, None)
        launcher_started = time.perf_counter()
        with (source_root / "source_launcher.log").open("x") as log:
            child = subprocess.run(intent["command"], cwd="/tmp", env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        freeze_json(source_root / "launcher_exit.json", dict(producer_job=job, exit_code=child.returncode,
            wall_seconds=time.perf_counter() - launcher_started,
            cost_scope="source wrapper subprocess: launch/step wait, source setup/input/simulation, archive/hash and scratch cleanup; excludes production preparation and export"))
    job = intent.get("producer_job")
    require(isinstance(job, str) and job.isdecimal()
            and intent.get("source_archive") == str((source_root / "native" / ("run_" + job)).resolve())
            and intent.get("batch_delivery") == str((source_root / "native" / "batches" / ("run_" + job)
                                                      / "delivery.json").resolve()),
            "BENCHMARK_PRODUCTION_PFC_RETAINED_SOURCE_PATH_CHANGED")
    batch_path = Path(intent["batch_delivery"])
    require(batch_path.is_file(), "BENCHMARK_PRODUCTION_PFC_UNRESOLVED_SOURCE_EXECUTION")
    batch = json.loads(batch_path.read_text())
    cases = batch.get("cases", [])
    require(batch.get("status") == "COMPLETE" and batch.get("job_id") == job
            and batch.get("exit_code") == batch.get("srun_exit_code") == 0
            and batch.get("case_ids") == [identity["episode_id"]] and len(cases) == 1
            and cases[0].get("case_id") == identity["episode_id"]
            and cases[0].get("job_id") == job and cases[0].get("exit_code") == 0
            and cases[0].get("source_execution_complete") is True
            and cases[0].get("archive_status") == "COMPLETE"
            and cases[0].get("archive") == intent["source_archive"],
            "BENCHMARK_PRODUCTION_PFC_COMPLETE_RETAINED_SOURCE_REQUIRED")
    # The existing exporter authenticates the retained source against this
    # episode and the frozen scientific admission before committing any data.
    result = export_pfc(cohort, identity, prepared / "episode.yaml", Path(intent["source_archive"]),
                        output, resume=True)
    return dict(result, new_source_simulations=0 if retained_source else 1,
                source_producer_job=job, source_archive=intent["source_archive"],
                source_batch_delivery=intent["batch_delivery"])


def run_pfc_blocks(cohort: ProductionCohort, circuit_index: int, block_start: int, block_count: int,
                   qualification: Path, source_python: Path, project: Path, output: Path,
                   *, resume: bool = False) -> dict:
    """Run complete ten-condition blocks in order, preserving committed episodes."""
    cohort.require_current()
    require(cohort.scenario == "pfc_working_memory", "BENCHMARK_PRODUCTION_PFC_SCENARIO_REQUIRED")
    require(type(block_start) is int and type(block_count) is int
            and 0 <= block_start < 100 and block_count > 0 and block_start + block_count <= 100,
            "BENCHMARK_PRODUCTION_PFC_BLOCK_RANGE_INVALID")
    cohort.identity(circuit_index, block_start, 0)
    require(resume or not output.exists(), "BENCHMARK_PRODUCTION_EXPLICIT_RESUME_REQUIRED")
    rows = []
    for block in range(block_start, block_start + block_count):
        for condition in range(10):
            identity = cohort.identity(circuit_index, block, condition)
            rows.append(run_pfc(cohort, identity, qualification, source_python, project, output, resume=True))
    return dict(status="COMPLETE_QUALIFIED_PFC_PRODUCTION_BLOCKS", circuit_index=circuit_index,
                block_start=block_start, block_count=block_count, complete_episodes=len(rows),
                new_source_simulations=sum(row.get("new_source_simulations", 0) for row in rows), episodes=rows)


def export_v1(cohort: ProductionCohort, identity: dict, rate_source: Path, spike_source: Path,
              native_source: Path, authority_source: Path, output: Path, *, resume: bool = False) -> dict:
    existing = completed_episode(cohort, identity, output, resume=resume)
    if existing is not None:
        return existing
    native, native_delivery = bound_json(native_source, "primary/report.json")
    recurrent, _ = bound_json(native_source, "recurrent/primary/report.json")
    inputs, _ = bound_json(native_source, "inputs_reference/primary/report.json")
    rates, rate_delivery = bound_json(rate_source, "probe/probe.json")
    spikes, spike_delivery = bound_json(spike_source, "visual_inputs/report.json")
    require(all(x.get("exit_code") == 0 for x in (native_delivery, rate_delivery, spike_delivery)),
            "BENCHMARK_PRODUCTION_SUCCESSFUL_SOURCE_DELIVERY_REQUIRED")
    for role, report_file in (("recurrent", "recurrent/primary/report.json"),
                              ("inputs", "inputs_reference/primary/report.json")):
        bound = native["bindings"][role]["files"]["primary/report.json"]
        require((native_source / report_file).stat().st_size == bound["bytes"]
                and sha256_file(native_source / report_file) == bound["sha256"],
                "BENCHMARK_PRODUCTION_CONSTRUCTION_REPORT_BINDING_DIFFERS:" + role)
    admission = bind_v1(cohort, identity, recurrent, inputs, native, rates, spikes)
    admission["native_report_sha256"] = sha256_file(native_source / "primary/report.json")
    admission["source_documents"] = dict(recurrent=source_document(native_source / "recurrent/primary/report.json"),
                                         inputs=source_document(native_source / "inputs_reference/primary/report.json"))
    require(os.environ.get("SLURMD_NODENAME") == "cpu03" and os.environ.get("SLURM_JOB_ID")
            and not os.environ.get("SLURM_JOB_GPUS"), "BENCHMARK_PRODUCTION_V1_CPU03_REQUIRED")
    # The declaration is checked before generating any intermediate archive.
    # Existing exporters then authenticate all consumed payloads and replay.
    with tempfile.TemporaryDirectory(prefix="neuronsim-production-v1-", dir="/tmp") as temporary:
        work = Path(temporary)
        movie = inspect_stimulus(rate_source, work / "movie/movie")
        require(movie["status"] == "SOURCE_MOVIE_CONTROL_WITNESS_PASS",
                "BENCHMARK_PRODUCTION_V1_MOVIE_WITNESS_FAILED")
        local_delivery(work / "movie", "movie")
        export_constructed_visual_controls(rate_source, work / "movie", spike_source, work / "controls/controls")
        local_delivery(work / "controls", "controls")
        result = export_pointnet(native_source, authority_source, work / "source", work / "controls")
        require(result["status"] == "EXPORTED_VERIFIED_POINTNET_SOURCE_OBSERVATION",
                "BENCHMARK_PRODUCTION_V1_SOURCE_EXPORT_FAILED")
        source = CircuitDataset(work / "source/dataset", splits=("development",))
        return write_episode(cohort, identity, source, output, admission=admission, resume=resume)


def verify_retained_ca3_trace(bound: dict, engine: Path, realization: SourceRealization | None) -> dict:
    """Reconcile retained native data before an interrupted archive commit."""
    build = json.loads(engine.with_name("host_current_replay_build.json").read_text())
    require(build.get("status") == "BUILT_AND_VALIDATED" and sha256_file(engine) == build["binary_sha256"],
            "BENCHMARK_PRODUCTION_CA3_REPLAY_ENGINE_CHANGED")
    episode, native, trace = bound["episode"], bound["native"], bound["path"] / "native_observation.bin"
    duration = episode.duration_ticks
    _, controls = check_currents(bound["groups"], iter_intervals(trace, expected_stop_ms=duration),
                                 bound["commands"], expected_stop_ms=duration)
    counts = {}
    compare_event_counters(trace, np.column_stack((episode.spike_ticks, episode.spike_unit_indices)),
                           len(bound["circuit"].unit_ids), counts, expected_stop_ms=duration)
    if realization is None:
        realization = SourceRealization(episode.circuit_id,
            "block_" + str(native["candidate_identity"]["block_index"]).zfill(3), "retrieval", "development",
            episode.seeds["circuit_seed"], episode.seeds["uniform_seed"], episode.seeds["engine_seed"])
    replay = compare_currents(engine, bound["groups"], bound["commands"],
        iter_intervals(trace, expected_stop_ms=duration), realization=realization, expected_stop_ms=duration)
    require(replay["exact_float32_bits_equal"] is True
            and counts["unit_millisecond_counts_compared"] == native["unit_millisecond_counts_compared"],
            "BENCHMARK_PRODUCTION_CA3_RETAINED_TRACE_RECONCILIATION_FAILED")
    return dict(native_trace_sha256=sha256_file(trace), replay_engine_sha256=build["binary_sha256"],
                control_delivery=controls, event_counter_readback=counts, independent_current_replay=replay)


def run_ca3(cohort: ProductionCohort, identity: dict, prepared_config: Path, packaged: Path,
            formation: Path, analysis: Path, output: Path, current_replay_engine: Path,
            *, resume: bool = False) -> dict:
    existing = completed_episode(cohort, identity, output, resume=resume)
    if existing is not None:
        return existing
    formation_path = formation / "native_report.json"
    native_formation = json.loads(formation_path.read_text())
    admission = bind_ca3(cohort, identity, prepared_config, native_formation)
    admission["formation_report_sha256"] = sha256_file(formation_path)
    admission["source_documents"] = dict(prepared_config=source_document(prepared_config),
                                         formation=source_document(formation_path))
    realization = ca3_benchmark.retrieval_realization(native_formation, identity)
    formed = qualify_input(formation, analysis / "formation_replay.json",
        analysis / "network65p_assem550_divnorm.dat", analysis / "normalization.json",
        analysis / "normalization_replay.json", coupled_extension=True, source_realization=realization)
    source_root = output / "source_execution" / identity["episode_id"]
    retained = source_root / "result"
    retained_source = source_root.exists()
    if retained_source:
        require(resume and (retained / "report.json").exists(),
                "BENCHMARK_PRODUCTION_CA3_UNRESOLVED_SOURCE_EXECUTION")
    else:
        require(os.environ.get("SLURMD_NODENAME") == "gpu03"
                and os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_GPUS"),
                "BENCHMARK_PRODUCTION_CA3_ALLOCATED_GPU03_REQUIRED")
        source_root.mkdir(parents=True)
        scratch = Path(tempfile.mkdtemp(prefix="neuronsim-production-ca3-", dir="/tmp"))
        attempt = scratch / "attempt"
        freeze_json(source_root / "execution.json", dict(status="STARTED_NATIVE_SOURCE_ATTEMPT",
            node=socket.gethostname(), job_id=os.environ["SLURM_JOB_ID"], scratch=str(attempt), identity=identity))
        try:
            ca3_benchmark.run(packaged, formation, analysis, prepared_config, attempt,
                circuit_index=identity["circuit_index"], block_index=identity["block_index"],
                condition_index=identity["condition_index"], current_replay_engine=current_replay_engine,
                production_identity=identity)
        except BaseException as error:
            freeze_json(source_root / "execution_error.json", dict(error_type=type(error).__name__, error=str(error)))
            raise
        finally:
            # Copy completed and failed native outcomes before deleting scratch.
            # If copy-out fails or the job is killed, execution.json identifies
            # the unresolved attempt. Resume then refuses to launch it again.
            if attempt.exists():
                shutil.copytree(attempt, retained, symlinks=True)
            shutil.rmtree(scratch)
    bound = bind_ca3_episode(retained)
    result = bound["native"]
    if retained_source:
        verify_retained_ca3_trace(bound, current_replay_engine, realization)
    expected_condition = condition_for_episode(cohort.campaign["scenarios"][cohort.scenario]["task_protocol"], cohort.scenario, identity)
    require(result["candidate_identity"] == identity and result["condition"] == expected_condition
            and result["formed_circuit"] == formed
            and result.get("independent_current_replay", {}).get("exact_float32_bits_equal") is True,
            "BENCHMARK_PRODUCTION_CA3_COMPLETE_MATCHED_SOURCE_EXECUTION_REQUIRED")
    return write_episode(cohort, identity, CircuitDataset(retained / "dataset", splits=("development",)),
                         output, admission=admission, resume=resume)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--identity-table", type=Path, required=True)
    parser.add_argument("--runtime-qualification", type=Path)
    parser.add_argument("--circuit-index", type=int, required=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare-circuit")
    prepare.add_argument("--base-config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--native-qualification", type=Path)
    prepare.add_argument("--backend", choices=("numpy", "cuda"))
    pfc_prepare = sub.add_parser("prepare-pfc-episode")
    pfc_prepare.add_argument("--native-qualification", type=Path, required=True)
    pfc_prepare.add_argument("--backend", choices=("numpy", "cuda"), required=True)
    pfc_prepare.add_argument("--production-output", type=Path)
    pfc_prepare.add_argument("--output", type=Path, required=True)
    pfc = sub.add_parser("export-pfc")
    for name in ("prepared-config", "source-archive", "output"):
        pfc.add_argument("--" + name, type=Path, required=True)
    pfc_run = sub.add_parser("run-pfc-episode")
    pfc_blocks = sub.add_parser("run-pfc-blocks")
    for command in (pfc_run, pfc_blocks):
        for name in ("native-qualification", "source-python", "project", "output"):
            command.add_argument("--" + name, type=Path, required=True)
    pfc_blocks.add_argument("--block-start", type=int, required=True)
    pfc_blocks.add_argument("--block-count", type=int, required=True)
    pfc_blocks.add_argument("--resume", action="store_true")
    v1 = sub.add_parser("export-v1")
    for name in ("rate-source", "spike-source", "native-source", "authority-source", "output"):
        v1.add_argument("--" + name, type=Path, required=True)
    ca3 = sub.add_parser("run-ca3")
    for name in ("prepared-config", "packaged", "formation", "analysis", "current-replay-engine", "output"):
        ca3.add_argument("--" + name, type=Path, required=True)
    for command in (v1, ca3, pfc, pfc_prepare, pfc_run):
        command.add_argument("--block-index", type=int, required=True)
        command.add_argument("--condition-index", type=int, required=True)
        command.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cohort = ProductionCohort.load(args.config, args.scenario, args.qualification, args.identity_table,
                                  runtime_qualification=args.runtime_qualification)
    if args.mode == "prepare-circuit":
        result = prepare_circuit(cohort, args.base_config, args.circuit_index, args.output,
                                 native_qualification=args.native_qualification, backend=args.backend)
    elif args.mode == "run-pfc-blocks":
        result = run_pfc_blocks(cohort, args.circuit_index, args.block_start, args.block_count,
                                args.native_qualification, args.source_python, args.project, args.output,
                                resume=args.resume)
    else:
        identity = cohort.identity(args.circuit_index, args.block_index, args.condition_index)
        expected_scenario = {"export-v1": "v1_visual", "run-ca3": "ca3_pattern_completion",
                             "prepare-pfc-episode": "pfc_working_memory", "export-pfc": "pfc_working_memory",
                             "run-pfc-episode": "pfc_working_memory"}[args.mode]
        require(args.scenario == expected_scenario,
                "BENCHMARK_PRODUCTION_ENTRY_SCENARIO_DIFFERS")
        if args.mode == "export-v1":
            result = export_v1(cohort, identity, args.rate_source, args.spike_source, args.native_source,
                               args.authority_source, args.output, resume=args.resume)
        elif args.mode == "prepare-pfc-episode":
            result = prepare_pfc_episode(cohort, identity, args.native_qualification, args.backend, args.output,
                                        production_output=args.production_output, resume=args.resume)
        elif args.mode == "export-pfc":
            result = export_pfc(cohort, identity, args.prepared_config, args.source_archive, args.output,
                                resume=args.resume)
        elif args.mode == "run-pfc-episode":
            result = run_pfc(cohort, identity, args.native_qualification, args.source_python,
                             args.project, args.output, resume=args.resume)
        else:
            result = run_ca3(cohort, identity, args.prepared_config, args.packaged, args.formation,
                             args.analysis, args.output, args.current_replay_engine, resume=args.resume)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
