"""Replay two archived PFC prefixes and observe their first threshold branch.

This bounded witness tests a single frozen spike schedule against two legal
paths. It supplies neither an auxiliary-event enclosure nor a probability bound.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time

import numpy as np
import yaml

from . import pfc_clock_coupling, pfc_clock_temporal_control, pfc_coupled_drive
from . import pfc_deterministic_replay as executor
from . import pfc_native_drive
from .provenance import freeze_json, sha256_file

SCHEMA = "pfc_threshold_branch_probe_v1"
PREPARED = "PREPARED_PFC_THRESHOLD_BRANCH_PROBE"
PASSED = "PASS_PFC_THRESHOLD_BRANCH_TWO_PATH_WITNESS"
FAILED = "FAILED_PFC_THRESHOLD_BRANCH_PROBE"
TRANSFORM = "within_coarse_tick_tree_temporal_reversal"
PAIRED_INVARIANTS = (
    "initial_voltage_equal", "all_coarse_and_auxiliary_final_rng_equal",
    "full_fine_clock_applied_current_equal", "population_external_event_totals_equal",
)
STATE_FIELDS = ("v", "s_AMPA", "s_GABA", "s_NMDA_total", "lastspike", "not_refractory")
E_FIELDS = ("I_stim", "s_NMDA", "x")
MODULES = (executor, pfc_native_drive, pfc_clock_coupling,
           pfc_coupled_drive, pfc_clock_temporal_control)
STABLE_MODULES = (executor, pfc_native_drive, pfc_clock_coupling)
ANCHORS = {
    "pre_rk2": '_rk2(e, constants["E"], t, dt, arithmetic, "E")',
    "pre_threshold": "spikes = {}",
}
THRESHOLD_STATEMENT = ('selected = arithmetic.indices((state["v"] > '
                       'constants[population]["threshold"]) & state["not_refractory"])')


def require(condition, message):
    if not condition:
        raise ValueError("PFC_THRESHOLD_PROBE_" + message)


def evidence_scope():
    return dict(two_path_witness_only=True, auxiliary_event=None,
                conditional_failure_probability=None, auxiliary_enclosure_qualified=False,
                cdf_qualified=False, population_bound_established=False,
                new_independent_blocks=0, new_auxiliary_streams=0,
                new_full_episodes=0, confirmation_episodes=0,
                confirmation_execution_released=False, production_episodes=0)


def validate_settings(settings):
    expected = dict(schema=SCHEMA, case_relative_path="source_cases/dt_0p0125/pfc_95001_omission",
                    source_rng_seed=95001, refinement_seeds=[95411, 95412],
                    coarse_dt_ms=.05, dt_ms=.0125, expected_units=100000,
                    prefix_window_ms=5.0, maximum_prefix_ms=100.0,
                    confirmation_execution_released=False, production_episodes=0)
    require(isinstance(settings, dict), "SETTINGS_MAPPING_REQUIRED")
    for key, value in expected.items():
        require(settings.get(key) == value and not (
            isinstance(settings.get(key), bool) and not isinstance(value, bool)),
            "FROZEN_SETTING_CHANGED:" + key)
    require(settings["confirmation_execution_released"] is False,
            "CONFIRMATION_MUST_REMAIN_CLOSED")
    require(all(type(settings[name]) is int for name in ("source_rng_seed", "expected_units", "production_episodes"))
            and all(type(seed) is int for seed in settings["refinement_seeds"]), "INTEGER_SEEDS_AND_COUNTS_REQUIRED")
    for variant in ("original", "reversed"):
        source = settings.get(variant + "_source", {})
        digest = source.get("delivery_sha256", "")
        require(isinstance(source.get("archive"), str) and bool(source["archive"])
                and isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), "SOURCE_BINDING_REQUIRED:" + variant)
    return settings


def canonical_events(ticks, units, expected_units, total_ticks):
    """Sort event identities; duplicate unit/tick events are invalid source data."""
    ticks, units = np.asarray(ticks), np.asarray(units)
    require(ticks.ndim == units.ndim == 1 and ticks.shape == units.shape
            and ticks.dtype.kind in "iu" and units.dtype.kind in "iu",
            "INTEGER_EVENT_VECTORS_REQUIRED")
    require(np.all(ticks < total_ticks) and np.all(ticks >= 0)
            and np.all(units < expected_units) and np.all(units >= 0), "EVENT_RANGE_INVALID")
    order = np.lexsort((units, ticks))
    events = np.column_stack((ticks[order], units[order])).astype(np.int64, copy=False)
    require(not np.any(np.all(events[1:] == events[:-1], axis=1)), "DUPLICATE_UNIT_TICK_EVENT")
    return events


def first_spike_difference(original, reversed_events):
    """Return the earliest set difference, including an unmatched sequence tail."""
    length = min(len(original), len(reversed_events))
    different = np.flatnonzero(np.any(original[:length] != reversed_events[:length], axis=1))
    index = int(different[0]) if different.size else length
    if index == len(original) == len(reversed_events):
        return None
    tick = min(int(events[index, 0]) for events in (original, reversed_events) if index < len(events))
    a = original[original[:, 0] == tick, 1]
    b = reversed_events[reversed_events[:, 0] == tick, 1]
    return dict(tick=tick, original_only_unit_indices=np.setdiff1d(a, b).tolist(),
                reversed_only_unit_indices=np.setdiff1d(b, a).tolist(),
                original_tick_unit_indices=a.tolist(), reversed_tick_unit_indices=b.tolist(),
                common_prior_event_count=int(np.count_nonzero(original[:, 0] < tick)))


def prefix_stop_tick(tick, dt_ms, window_ms, maximum_ms):
    require(type(tick) is int and tick >= 0, "NONNEGATIVE_MISMATCH_TICK_REQUIRED")
    require(all(type(value) in (int, float) and np.isfinite(value) and value > 0
                for value in (dt_ms, window_ms, maximum_ms)), "PREFIX_CLOCK_INVALID")
    window_ticks = round(window_ms / dt_ms)
    require(window_ticks > 0 and np.isclose(window_ticks * dt_ms, window_ms, rtol=0, atol=1e-12),
            "PREFIX_WINDOW_OFF_CLOCK")
    stop = (tick // window_ticks + 1) * window_ticks
    require(stop * dt_ms <= maximum_ms, "FIRST_DIVERGENCE_EXCEEDS_PREFIX_CAP")
    return stop


def _file_record(path):
    return dict(bytes=path.stat().st_size, sha256=sha256_file(path))


def _bound_file(root, name, record):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "UNSAFE_MANIFEST_PATH")
    path = root / relative
    require(path.stat().st_size == record["bytes"] and sha256_file(path) == record["sha256"],
            "BOUND_FILE_CHANGED:" + name)
    return path


def _code_bindings():
    return {module.__name__.split(".")[-1]: sha256_file(Path(module.__file__)) for module in MODULES}


def observer_anchors():
    """Resolve source statements through AST, rejecting missing/ambiguous hooks."""
    lines, start = inspect.getsourcelines(executor.simulate)
    source = textwrap.dedent("".join(lines))
    tree = ast.parse(source)
    result = {}
    for name, statement in dict(ANCHORS, threshold_selection=THRESHOLD_STATEMENT).items():
        target = ast.dump(ast.parse(statement).body[0], include_attributes=False)
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.stmt)
                   and ast.dump(node, include_attributes=False) == target]
        require(len(matches) == 1, "SOURCE_OBSERVER_ANCHOR_CHANGED:" + name)
        result[name] = dict(line=start + matches[0].lineno - 1, statement=statement)
    require(result["pre_rk2"]["line"] < result["pre_threshold"]["line"]
            < result["threshold_selection"]["line"], "SOURCE_OBSERVER_ORDER_CHANGED")
    return result


class ThresholdObserver:
    """Copy only the target tick's full state; never alter engine or drive state."""

    def __init__(self, tick, config):
        self.tick = tick
        self.config = config
        self.anchors = observer_anchors()
        self.snapshots = {}
        self.capture_seconds = 0.0

    def capture(self, stage, local):
        started = time.perf_counter()
        require(stage not in self.snapshots, "OBSERVER_STAGE_REPEATED")
        snapshot = {}
        for population, local_name, size_key in (("E", "e", "excitatory_neurons"),
                                                  ("I", "i", "inhibitory_neurons")):
            state = local[local_name]
            for field in STATE_FIELDS + (E_FIELDS if population == "E" else ()):
                value = state[field]
                require(isinstance(value, np.ndarray) and value.shape == (self.config[size_key],)
                        and value.dtype == (np.dtype(bool) if field == "not_refractory" else np.dtype("float64")),
                        "OBSERVER_STATE_SHAPE_OR_DTYPE_CHANGED:" + population + "__" + field)
                require(np.isfinite(value).all(), "OBSERVER_NONFINITE_STATE")
                snapshot[population + "__" + field] = value.copy()
        self.snapshots[stage] = snapshot
        self.capture_seconds += time.perf_counter() - started

    @contextmanager
    def observing(self):
        previous = sys.gettrace()
        target = executor.simulate.__code__
        stages = {self.anchors[stage]["line"]: stage for stage in ANCHORS}

        def trace(frame, event, arg):
            if frame.f_code is not target:
                return None
            if event == "line" and frame.f_lineno in stages and frame.f_locals.get("tick") == self.tick:
                self.capture(stages[frame.f_lineno], frame.f_locals)
            return trace

        try:
            sys.settrace(trace)
            yield self
        finally:
            sys.settrace(previous)


def _load_source(settings, project_root, variant):
    binding = settings[variant + "_source"]
    root = project_root / binding["archive"]
    delivery_path = root / "delivery.json"
    require(sha256_file(delivery_path) == binding["delivery_sha256"], "DELIVERY_CHANGED:" + variant)
    delivery = json.loads(delivery_path.read_text())
    status = ("COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT" if variant == "original"
              else "COMPLETE_COUPLED_CLOCK_TEMPORAL_CONTROL")
    require(delivery.get("verified") is True and delivery.get("status") == status,
            "COMPLETE_VERIFIED_SOURCE_REQUIRED:" + variant)
    manifest = {row["path"]: row for row in delivery["files"]}
    require(len(manifest) == len(delivery["files"]), "DUPLICATE_MANIFEST_PATH")
    prefix = settings["case_relative_path"]
    paths = {name: _bound_file(root, prefix + "/" + name, manifest[prefix + "/" + name])
             for name in ("report.json", "raw.npz", "initial_voltage.npy")}
    report = json.loads(paths["report.json"].read_text())
    for name in ("raw.npz", "initial_voltage.npy"):
        require(all(report["outputs"][name][field] == manifest[prefix + "/" + name][field]
                    for field in ("sha256", "bytes")),
                "CASE_OUTPUT_BINDING_CHANGED:" + name)
    expected_status = ("COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE" if variant == "original"
                       else "COMPLETE_COUPLED_CLOCK_TEMPORAL_CONTROL_CASE")
    require(report.get("status") == expected_status and report.get("source_execution_complete") is True
            and report.get("case_id") == "pfc_95001_omission" and report.get("condition") == "omission"
            and report.get("source_rng_seed") == settings["source_rng_seed"]
            and report.get("refinement_seeds") == settings["refinement_seeds"]
            and report.get("dt_ms") == settings["dt_ms"] and report.get("coarse_dt_ms") == settings["coarse_dt_ms"]
            and report.get("duration_ms") == 5000.0 and report.get("coupled_core_units") == settings["expected_units"]
            and report.get("confirmation_execution_released") is False and report.get("production_episodes") == 0
            and bool(report.get("validations"))
            and all(row.get("passed") is True for row in report["validations"].values()),
            "FROZEN_COMPLETE_CASE_REQUIRED:" + variant)
    config, case = report["resolved_config"], report["selected_case"]
    require(config.get("duration_ms") == 5000.0 and config.get("dt_ms") == settings["dt_ms"]
            and config.get("excitatory_neurons") == 80000 and config.get("inhibitory_neurons") == 20000
            and config.get("readout_bin_ms") == 5.0 and config.get("excitatory_preference_bins") == 16
            and config.get("random_stream_policy", "source_global") == "source_global"
            and config.get("synapse_execution") == "factorized_uniform_projections"
            and case.get("case_id") == "pfc_95001_omission"
            and case.get("source_rng_seed", config.get("source_rng_seed")) == settings["source_rng_seed"]
            and case.get("stimulus_strength_nA") == 0 and case.get("distractor_strength_nA") == 0,
            "SOURCE_PROTOCOL_CHANGED:" + variant)
    current_code = _code_bindings()
    for module in (STABLE_MODULES if variant == "original" else MODULES):
        name = module.__name__.split(".")[-1]
        require(report["code"].get(name) == current_code[name], "SOURCE_CODE_CHANGED:" + name)
    order = report.get("drive", {}).get("coupling", {}).get("input_order_control")
    if variant == "original":
        require(not order, "ORIGINAL_INPUT_ORDER_CHANGED")
    else:
        agreement = report.get("original_fine_input_agreement", {})
        require(set(agreement) == set(PAIRED_INVARIANTS)
                and all(agreement[name] is True for name in PAIRED_INVARIANTS),
                "FOUR_PAIRED_INPUT_INVARIANTS_REQUIRED")
        require(report.get("temporal_transform") == TRANSFORM and isinstance(order, dict)
                and order.get("name") == TRANSFORM and order.get("middle_row_order") == [1, 0]
                and order.get("fine_row_order") == [3, 2, 1, 0]
                and order.get("unit_axis_permuted") is False and order.get("rng_draw_order_changed") is False,
                "EXACT_TEMPORAL_TRANSFORM_REQUIRED")
    initial = np.load(paths["initial_voltage.npy"], allow_pickle=False)
    require(initial.shape == (settings["expected_units"],) and initial.dtype == np.float64
            and np.isfinite(initial).all() and np.all(initial >= -60) and np.all(initial < -50),
            "INITIAL_VOLTAGE_INVALID")
    with np.load(paths["raw.npz"], allow_pickle=False) as raw:
        events = canonical_events(raw["spike_ticks"], raw["spike_unit_indices"], settings["expected_units"],
                                  round(config["duration_ms"] / settings["dt_ms"]))
    evidence = dict(binding, case_relative_path=prefix,
                    files={name: dict(manifest[prefix + "/" + name]) for name in paths},
                    source_status=report["status"], source_code=report["code"],
                    original_fine_input_agreement=report.get("original_fine_input_agreement"),
                    cost=report.get("cost"), wall_seconds=report.get("wall_seconds"),
                    original_fine_reference=report.get("original_fine_reference"),
                    original_fine_report_sha256=report.get("original_fine_report_sha256"),
                    full_archived_event_count=len(events))
    return report, initial, events, evidence


def prepare(config_path, project_root, output):
    config_path, project_root, output = Path(config_path), Path(project_root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status=FAILED, stage="prepare", **evidence_scope())
    names = []
    started = time.perf_counter()
    try:
        settings = validate_settings(yaml.safe_load(config_path.read_text()))
        report["settings_sha256"] = sha256_file(config_path)
        original, initial, a, original_binding = _load_source(settings, project_root, "original")
        reversed_report, other_initial, b, reversed_binding = _load_source(settings, project_root, "reversed")
        require(original["resolved_config"] == reversed_report["resolved_config"]
                and original["selected_case"] == reversed_report["selected_case"], "PAIRED_CONFIG_OR_CASE_DRIFT")
        require(np.array_equal(initial, other_initial), "PAIRED_INITIAL_VOLTAGES_DIFFER")
        require(reversed_report["original_fine_reference"]["delivery_sha256"]
                == settings["original_source"]["delivery_sha256"]
                and reversed_report["original_fine_report_sha256"]
                == original_binding["files"]["report.json"]["sha256"], "REVERSED_ORIGINAL_LINEAGE_CHANGED")
        difference = first_spike_difference(a, b)
        report["first_mismatch"] = difference
        require(difference is not None, "NO_ARCHIVED_SPIKE_DIFFERENCE")
        stop = prefix_stop_tick(difference["tick"], settings["dt_ms"],
                                settings["prefix_window_ms"], settings["maximum_prefix_ms"])
        np.save(output / "initial_voltage.npy", initial, allow_pickle=False)
        names.append("initial_voltage.npy")
        for variant, events in (("original", a), ("reversed", b)):
            events = events[events[:, 0] < stop]
            name = variant + "_events.npz"
            np.savez(output / name, spike_ticks=events[:, 0], spike_unit_indices=events[:, 1])
            names.append(name)
        plan = dict(schema=SCHEMA, status=PREPARED, settings=settings,
                    full_config=original["resolved_config"], selected_case=original["selected_case"],
                    prefix_stop_tick=stop, prefix_duration_ms=stop * settings["dt_ms"],
                    first_mismatch=difference, source_bindings=dict(original=original_binding, reversed=reversed_binding),
                    current_code=_code_bindings(), probe_sha256=sha256_file(Path(__file__)),
                    observer_anchors=observer_anchors(),
                    observation=dict(unit_order="source_E_then_I", voltage="V", initial_voltage="mV",
                                     time="source_tick", threshold="strict v > -0.050 V and not_refractory",
                                     current_tick_external_events="affect the next RK2 update"),
                    artifacts={name: _file_record(output / name) for name in names}, **evidence_scope())
        freeze_json(output / "plan.json", plan)
        names.append("plan.json")
        report.update(status=PREPARED, prefix_stop_tick=stop, prefix_duration_ms=plan["prefix_duration_ms"],
                      source_bindings=plan["source_bindings"])
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    report.update(outputs={name: _file_record(output / name) for name in names},
                  wall_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
    freeze_json(output / "report.json", report)
    return report


def _load_prepared(prepared):
    root = Path(prepared)
    if root.name == "plan.json":
        root = root.parent
    receipt = json.loads((root / "report.json").read_text())
    require(receipt.get("status") == PREPARED, "SUCCESSFUL_PREPARATION_REQUIRED")
    plan_path = _bound_file(root, "plan.json", receipt["outputs"]["plan.json"])
    plan = json.loads(plan_path.read_text())
    validate_settings(plan["settings"])
    require(plan.get("status") == PREPARED and plan.get("schema") == SCHEMA
            and plan["current_code"] == _code_bindings()
            and plan["probe_sha256"] == sha256_file(Path(__file__))
            and plan["observer_anchors"] == observer_anchors(), "PREPARED_CODE_BINDING_CHANGED")
    expected_names = {"initial_voltage.npy", "original_events.npz", "reversed_events.npz"}
    require(set(plan["artifacts"]) == expected_names, "PREPARED_ARTIFACT_SET_CHANGED")
    paths = {name: _bound_file(root, name, record) for name, record in plan["artifacts"].items()}
    require(plan["prefix_stop_tick"] == prefix_stop_tick(plan["first_mismatch"]["tick"],
                plan["settings"]["dt_ms"], plan["settings"]["prefix_window_ms"], plan["settings"]["maximum_prefix_ms"])
            and plan["prefix_duration_ms"] == plan["prefix_stop_tick"] * plan["settings"]["dt_ms"],
            "PREPARED_PREFIX_BOUND_CHANGED")
    return plan, paths


def _branch_analysis(snapshots, original, reversed_events, difference, config):
    tick, n_e = difference["tick"], config["excitatory_neurons"]
    require(np.array_equal(original[original[:, 0] < tick], reversed_events[reversed_events[:, 0] < tick]),
            "PRIOR_SPIKE_HISTORIES_DIFFER")
    recurrent = {}
    for stage in ANCHORS:
        for population in ("E", "I"):
            fields = ("s_GABA", "s_NMDA_total", "lastspike", "not_refractory")
            if population == "E":
                fields += ("s_NMDA", "x", "I_stim")
            for field in fields:
                key = population + "__" + field
                recurrent[stage + "__" + key] = bool(np.array_equal(
                    snapshots["original"][stage][key], snapshots["reversed"][stage][key]))
    # Availability is recomputed by the original RK2 from identical lastspike histories.
    require(all(recurrent.values()), "RECURRENT_OR_REFRACTORY_HISTORY_DIFFERS_BEFORE_BRANCH")
    decisions = {}
    for variant, events in (("original", original), ("reversed", reversed_events)):
        post = snapshots[variant]["pre_threshold"]
        ids = []
        for population, offset in (("E", 0), ("I", n_e)):
            threshold = executor._constants(config, population)["threshold"]
            require(threshold == -.050, "SOURCE_THRESHOLD_CHANGED")
            selected = np.flatnonzero((post[population + "__v"] > threshold)
                                      & post[population + "__not_refractory"]) + offset
            ids.extend(selected.tolist())
        decisions[variant] = ids
        require(ids == events[events[:, 0] == tick, 1].tolist(), "OBSERVED_THRESHOLD_SPIKES_DIFFER:" + variant)
    units = sorted(set(difference["original_only_unit_indices"] + difference["reversed_only_unit_indices"]))
    require(bool(units), "NONEMPTY_BRANCH_WITNESS_REQUIRED")
    rows = []
    for unit in units:
        population, index = ("E", unit) if unit < n_e else ("I", unit - n_e)
        row = dict(archive_unit_index=unit, source_population=population, source_unit_index=index)
        voltages, margins, branches = [], [], []
        for variant in ("original", "reversed"):
            pre, post = snapshots[variant]["pre_rk2"], snapshots[variant]["pre_threshold"]
            voltage = float(post[population + "__v"][index])
            margin = voltage - (-.050)
            eligible = bool(post[population + "__not_refractory"][index])
            branch = bool(voltage > -.050 and eligible)
            row[variant] = dict(pre_rk2_voltage_V=float(pre[population + "__v"][index]),
                                pre_threshold_voltage_V=voltage, threshold_margin_V=margin,
                                refractory_eligible=eligible, spike_branch=branch,
                                pre_rk2_s_AMPA=float(pre[population + "__s_AMPA"][index]))
            voltages.append(voltage)
            margins.append(margin)
            branches.append(branch)
        require(branches[0] != branches[1]
                and all(row[v]["refractory_eligible"] for v in ("original", "reversed"))
                and min(margins) <= 0 < max(margins), "STRICT_ELIGIBLE_THRESHOLD_BRANCH_NOT_WITNESSED")
        row.update(two_path_voltage_span_V=max(voltages) - min(voltages),
                   minimum_margin_V=min(margins), maximum_margin_V=max(margins))
        rows.append(row)
    return dict(prior_spike_histories_identical=True, recurrent_and_refractory_state_equal=recurrent,
                history_basis="identical initial state and every prior spike event under unchanged deterministic recurrent updates",
                threshold_decisions=decisions, differing_units=rows,
                one_frozen_spike_schedule_encloses_both_observed_paths=False)


def execute(prepared, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status=FAILED, stage="execute", paths={}, **evidence_scope())
    names, snapshots, observed_events = [], {}, {}
    started = time.perf_counter()
    try:
        plan, paths = _load_prepared(prepared)
        settings, config, case = plan["settings"], plan["full_config"], plan["selected_case"]
        report.update(first_mismatch=plan["first_mismatch"], prefix_duration_ms=plan["prefix_duration_ms"],
                      source_bindings=plan["source_bindings"], observer_anchors=plan["observer_anchors"],
                      prepared_plan_sha256=sha256_file(Path(prepared) / "plan.json" if Path(prepared).is_dir() else Path(prepared)))
        initial = np.load(paths["initial_voltage.npy"], allow_pickle=False)
        prefix_config = deepcopy(config)
        prefix_config["duration_ms"] = plan["prefix_duration_ms"]
        for variant, factory in (("original", pfc_coupled_drive.create_coupled_drive),
                                 ("reversed", pfc_clock_temporal_control.create_temporal_reversal_drive)):
            path_report = dict(status="FAILED", archived_prefix_equal=False)
            report["paths"][variant] = path_report
            observer = ThresholdObserver(plan["first_mismatch"]["tick"], config)
            try:
                generated_initial, drive = factory(config, case, settings["coarse_dt_ms"], settings["refinement_seeds"])
                require(np.array_equal(generated_initial, initial), "GENERATED_INITIAL_DIFFERS:" + variant)
                require(drive.total_ticks == round(config["duration_ms"] / config["dt_ms"]),
                        "DRIVE_FULL_DURATION_CHANGED")
                with observer.observing():
                    result = executor.simulate(prefix_config, case, generated_initial, backend="numpy", drive=drive)
                name = variant + "_raw_prefix.npz"
                np.savez(output / name, **result["raw"])
                names.append(name)
                events = canonical_events(result["raw"]["spike_ticks"], result["raw"]["spike_unit_indices"],
                                          settings["expected_units"], plan["prefix_stop_tick"])
                observed_events[variant] = events
                with np.load(paths[variant + "_events.npz"], allow_pickle=False) as expected:
                    archive = canonical_events(expected["spike_ticks"], expected["spike_unit_indices"],
                                               settings["expected_units"], plan["prefix_stop_tick"])
                matches = np.array_equal(events, archive)
                path_report.update(status="REPLAYED", archived_prefix_equal=bool(matches),
                                   replay_archive_first_difference=first_spike_difference(archive, events),
                                   initial_voltage_equal=True, cost=result["cost"],
                                   inputs_used=result["inputs_used"], drive=drive.metadata(),
                                   prefix_event_count=len(events))
            except Exception as exc:
                path_report["error"] = dict(type=type(exc).__name__, message=str(exc))
            finally:
                snapshots[variant] = observer.snapshots
                for stage, arrays in observer.snapshots.items():
                    name = variant + "_" + stage + ".npz"
                    np.savez(output / name, **arrays)
                    names.append(name)
                path_report.update(observer_capture_seconds=observer.capture_seconds,
                                   captured_stages=list(observer.snapshots))
        require(all(row.get("archived_prefix_equal") is True for row in report["paths"].values()),
                "BOTH_ARCHIVED_EVENT_PREFIXES_MUST_MATCH_EXACTLY")
        require(all(set(snapshots[variant]) == set(ANCHORS) for variant in snapshots),
                "BOTH_OBSERVER_STAGES_REQUIRED")
        actual_difference = first_spike_difference(observed_events["original"], observed_events["reversed"])
        require(actual_difference == plan["first_mismatch"], "REPLAY_FIRST_DIFFERENCE_CHANGED")
        analysis = _branch_analysis(snapshots, observed_events["original"], observed_events["reversed"],
                                    actual_difference, config)
        report["branch_analysis"] = analysis
        counts = {variant: np.bincount(events[:, 1], minlength=settings["expected_units"])
                  for variant, events in observed_events.items()}
        delta = counts["reversed"] - counts["original"]
        name = "count_differences.npz"
        np.savez(output / name, archive_unit_indices=np.arange(settings["expected_units"]),
                 original_counts=counts["original"], reversed_counts=counts["reversed"],
                 reversed_minus_original=delta)
        names.append(name)
        report.update(status=PASSED, completed_prefix_replays=2,
                      prefix_count_differences=dict(units_with_different_counts=int(np.count_nonzero(delta)),
                                                   absolute_count_difference_sum=int(np.abs(delta).sum()),
                                                   reversed_minus_original_total=int(delta.sum())),
                      conclusion="The two exactly replayed legal paths take different eligible strict threshold branches; a single frozen spike schedule does not contain both paths.")
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    report.update(outputs={name: _file_record(output / name) for name in names},
                  wall_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
    freeze_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "execute"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.config is None or args.project_root is None or args.prepared is not None:
            parser.error("prepare requires --config and --project-root only")
        report = prepare(args.config, args.project_root, args.output)
    else:
        if args.prepared is None or args.config is not None or args.project_root is not None:
            parser.error("execute requires --prepared only")
        report = execute(args.prepared, args.output)
    print(json.dumps({key: report[key] for key in ("status", "error", "prefix_duration_ms") if key in report}, indent=2))
    return 0 if report["status"] in (PREPARED, PASSED) else 1


if __name__ == "__main__":
    raise SystemExit(main())
