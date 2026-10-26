"""Validate saved development search programs before native execution.

The caller verifies artifact hashes and source manifests before passing parsed
reports. This helper keeps every declared candidate, regardless of its score.
It does not establish a success tolerance or qualify an intervention claim.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .mesocircuit_continuation import prepare_pulse_protocol

TRAINING_SEEDS = (41, 42, 43)
SOURCE_CASE_ID = "train_block_014_familiar_0"
CANDIDATE_NAMES = ("omit", "first_future_pulse", "original_future_pulses")
DT_MS = 0.1
DECISION_TIME_MS = 1000.0
HORIZON_MS = 500.0
EPISODE_MS = 6000.0
HORIZON_BINS = 100
SOURCE_UNITS = 100000
SOURCE_FIELDS = 128
SOURCE_REVISION = "74fc021cad2b4bd5c011c390b3f95a302b663483"
PROTOCOL_TIME_FIELDS = (
    "initial_emission_ms", "requested_future_emission_ms", "committed_emission_ms",
    "inflight_tc_arrival_ms", "effective_emission_ms", "effective_tc_arrival_ms",
)


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}: a parsed object mapping is required")
    return value


def _digest(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{name}: an explicit SHA-256 digest is required")
    return value


def _report_protocol(value, expected, candidate, seed):
    record = _mapping(value, f"seed {seed} candidate {candidate} protocol")
    for key in PROTOCOL_TIME_FIELDS:
        if not isinstance(record.get(key), list) or record[key] != expected[key]:
            raise ValueError(f"seed {seed} candidate {candidate}: {key} differs from the source program")
    for key in ("schema", "time_units", "time_origin", "control_semantics", "resolution_ms",
                "generator_delay_ms", "decision_time_ms", "initial_schedule_overridden",
                "observed_history_interval"):
        if key not in record or record[key] != expected[key]:
            raise ValueError(f"seed {seed} candidate {candidate}: {key} differs from the source protocol")
    for key in ("full_microscopic_state_qualified", "paired_stochastic_future_noise_verified",
                "inverse_intervention_qualified"):
        if record.get(key) is not False:
            raise ValueError(f"seed {seed} candidate {candidate}: unsupported {key}")


def native_candidate_protocols(
    search_reports: Mapping[int, Mapping[str, Any]], source_case: Mapping[str, Any],
    source_graph: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[int, str]]:
    """Return all three native pulse programs and each seed's selected name.

    Inputs are the per-seed saved analysis objects, the registered 6-second
    development case and its source graph. ``source_graph`` must also contain
    ``graph_manifest_sha256``, supplied by the caller after hashing the actual
    graph manifest. Training seeds identify checkpoints, not independent native
    simulation units. Return values contain no inferred scientific qualification.
    """
    reports = _mapping(search_reports, "search_reports")
    if (set(reports) != set(TRAINING_SEEDS)
            or any(type(seed) is not int for seed in reports)):
        raise ValueError("exactly the completed training seeds 41, 42 and 43 are required")
    case = _mapping(source_case, "source_case")
    if (case.get("case_id") != SOURCE_CASE_ID or case.get("split") != "train"
            or type(case.get("block")) is not int or case["block"] != 14
            or case.get("template_id") != "familiar_0"):
        raise ValueError("the registered train_block_014_familiar_0 development case is required")
    settings = _mapping(case.get("settings"), "source_case settings")
    source_control = _mapping(settings.get("pulse_control"), "source_case pulse_control")
    if set(source_control) != {"initial_emission_ms"}:
        raise ValueError("the source case must declare only the original initial emission program")
    graph = _mapping(source_graph, "source_graph")
    graph_digest = _digest(graph.get("graph_manifest_sha256"), "source_graph graph_manifest_sha256")
    net = _mapping(graph.get("resolved_source_network"), "resolved_source_network")
    sim = dict(sim_resolution=DT_MS, t_presim=0.0, t_sim=EPISODE_MS)
    initial = deepcopy(source_control["initial_emission_ms"])
    original = prepare_pulse_protocol(net, sim, dict(initial_emission_ms=initial))
    if original is None or original.resolution_ms != DT_MS:
        raise ValueError("the source must define native thalamic pulse emissions on the 0.1 ms clock")
    boundary_tick = round(DECISION_TIME_MS / DT_MS)
    response_stop_tick = round((DECISION_TIME_MS + HORIZON_MS) / DT_MS)
    future_ticks = tuple(tick for tick in original.initial_ticks
                         if tick > boundary_tick and tick + original.delay_ticks < response_stop_tick)
    if len(future_ticks) < 2:
        raise ValueError("the original source program must support three distinct declared pulse candidates")
    future_schedules = ([], original.milliseconds(future_ticks[:1]).tolist(),
                        original.milliseconds(future_ticks).tolist())
    programs = {}
    expected_protocols = {}
    for name, future in zip(CANDIDATE_NAMES, future_schedules):
        program = dict(initial_emission_ms=deepcopy(initial), decision_time_ms=DECISION_TIME_MS,
                       future_emission_ms=future)
        prepared = prepare_pulse_protocol(net, sim, program)
        programs[name] = program
        expected_protocols[name] = prepared.record()
    selected = {}
    common_identity = None
    common_binding = None
    common_fields = None
    for seed in TRAINING_SEEDS:
        report = _mapping(reports[seed], f"seed {seed} report")
        if report.get("status") != "DEVELOPMENT_NATIVE_PULSE_SEARCH_EXECUTED":
            raise ValueError(f"seed {seed}: a completed development search analysis is required")
        if type(report.get("training_seed")) is not int or report["training_seed"] != seed:
            raise ValueError(f"seed {seed}: saved training seed identity differs")
        configuration = _mapping(report.get("configuration"), f"seed {seed} configuration")
        required_config = dict(case_id=SOURCE_CASE_ID, training_seeds=list(TRAINING_SEEDS),
                               decision_time_ms=DECISION_TIME_MS, horizon_ms=HORIZON_MS,
                               native_intervention_execution=False, repeat_training=False,
                               scientific_success_tolerance=None)
        if any(key not in configuration or configuration[key] != value
               for key, value in required_config.items()):
            raise ValueError(f"seed {seed}: the saved search case, boundary or inference-only scope differs")
        if (type(report.get("source_unit_count")) is not int or report["source_unit_count"] != SOURCE_UNITS
                or type(report.get("horizon_bins")) is not int or report["horizon_bins"] != HORIZON_BINS):
            raise ValueError(f"seed {seed}: complete 100000-unit and 100-bin search output is required")
        for key in ("training_performed", "mechanistic_simulation_performed", "inverse_intervention_qualified",
                    "scientific_claim_qualified", "success_tolerance_qualified", "native_execution_performed",
                    "native_future_noise_pairing_verified"):
            if report.get(key) is not False:
                raise ValueError(f"seed {seed}: unsupported or missing inference-only declaration {key}")
        if report.get("all_candidate_outcomes_retained") is not True:
            raise ValueError(f"seed {seed}: every declared candidate outcome must be retained")
        binding = _mapping(report.get("native_binding"), f"seed {seed} native_binding")
        identity = _mapping(binding.get("native_source_identity"), f"seed {seed} native_source_identity")
        if (identity.get("graph_manifest_sha256") != graph_digest or identity.get("dt_ms") != DT_MS
                or identity.get("gpu_source_revision") != SOURCE_REVISION
                or identity.get("native_poisson_rng_family") != "per_edge_philox4_32_10"):
            raise ValueError(f"seed {seed}: native graph, clock or stochastic source identity differs")
        _digest(identity.get("native_library_sha256"), f"seed {seed} native_library_sha256")
        if report.get("graph_manifest_sha256") != graph_digest:
            raise ValueError(f"seed {seed}: report graph identity differs from its native binding")
        fields = report.get("field_names")
        if (not isinstance(fields, list) or len(fields) != SOURCE_FIELDS
                or any(not isinstance(field, str) or not field for field in fields)
                or len(set(fields)) != len(fields)):
            raise ValueError(f"seed {seed}: all 128 identified source fields in their original order are required")
        if common_identity is None:
            common_identity, common_binding, common_fields = deepcopy(identity), deepcopy(binding), list(fields)
        elif identity != common_identity or binding != common_binding or fields != common_fields:
            raise ValueError(f"seed {seed}: source episode, source identity or observation field order changed")
        rows = report.get("candidates")
        if (not isinstance(rows, list) or len(rows) != len(CANDIDATE_NAMES)
                or any(not isinstance(row, Mapping) for row in rows)
                or [row.get("candidate") for row in rows] != list(CANDIDATE_NAMES)):
            raise ValueError(f"seed {seed}: all three declared candidate rows in registered order are required")
        for row in rows:
            name = row["candidate"]
            _report_protocol(row.get("protocol"), expected_protocols[name], name, seed)
        choice = report.get("selected_candidate")
        if choice not in programs:
            raise ValueError(f"seed {seed}: selected_candidate must name one of the three declared programs")
        selected[seed] = choice
    return programs, selected
