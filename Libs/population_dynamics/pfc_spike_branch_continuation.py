"""One imposed PFC spike followed by the original future external input stream.

An instantaneous voltage trigger is a counterfactual event intervention. It is
not a native auxiliary realization, statistical draw, or admissible stimulus.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import yaml

from . import pfc_threshold_branch_probe as branch
from . import pfc_count_histograms, pfc_fluctuation_analysis
from . import pfc_coupled_drive
from . import pfc_deterministic_replay as executor
from .provenance import freeze_json, sha256_file

SCHEMA = "pfc_spike_branch_continuation_v1"
PREPARED = "PREPARED_PFC_SPIKE_BRANCH_CONTINUATION"
COMPLETE = "COMPLETE_PFC_SPIKE_BRANCH_CONTINUATION"
FAILED = "FAILED_PFC_SPIKE_BRANCH_CONTINUATION"
ACCEPTED_DELIVERY = "0028178795cca3f0255fcacc45adf6a21baf9f295e22aadc0d79fa8e3839ef1b"
ACCEPTED_REPORT = "2969d86954b9a119495e4acecf76ee5958b41d6f3ae0b3f90ffb143b4c7f4103"
ACCEPTED_SNAPSHOT = "a19efee6e1601f481023a4f6e8148ac0d07b002717f021129b6c94995cfe0de2"
ORIGINAL_DELIVERY = "d5d1de242753a17b5f5833e1777740938328a4e75f7f27d634b0f4e6389cc261"
ACCEPTED_FILES = ("report.json", "execution/report.json", "prepared/plan.json",
                  "prepared/initial_voltage.npy", "execution/original_pre_threshold.npz")


def require(condition, message):
    if not condition:
        raise ValueError("PFC_SPIKE_CONTINUATION_" + message)


def scope():
    return dict(counterfactual_event_sensitivity_diagnostic=True,
                legal_native_auxiliary_realization=False, statistical_source_draw=False,
                admissible_inverse_stimulus=False, scientific_fidelity_qualified=False,
                auxiliary_enclosure_qualified=False, cdf_qualified=False,
                population_bound_established=False, all_scientific_gates_released=False,
                auxiliary_event=None, conditional_failure_probability=None,
                source_numerical_engine_changed=False, new_independent_blocks=0,
                new_auxiliary_streams=0, new_full_episodes=0, confirmation_episodes=0,
                confirmation_execution_released=False, production_episodes=0)


def validate_settings(settings):
    frozen = dict(schema=SCHEMA, source_rng_seed=95001, refinement_seeds=[95411, 95412],
                  coarse_dt_ms=.05, dt_ms=.0125, expected_units=100000,
                  forced_unit_index=6423, forced_tick=89, stop_ms=100., windows_ms=[5., 50., 100.],
                  case_relative_path="source_cases/dt_0p0125/pfc_95001_omission",
                  confirmation_execution_released=False, production_episodes=0)
    require(isinstance(settings, dict), "SETTINGS_REQUIRED")
    for name, value in frozen.items():
        require(settings.get(name) == value and not (isinstance(settings.get(name), bool)
                and not isinstance(value, bool)), "FROZEN_SETTING_CHANGED:" + name)
    for name in ("source_rng_seed", "expected_units", "forced_unit_index", "forced_tick", "production_episodes"):
        require(type(settings[name]) is int, "INTEGER_SETTING_REQUIRED:" + name)
    require(settings["confirmation_execution_released"] is False
            and all(type(x) is int for x in settings["refinement_seeds"]), "FROZEN_SEEDS_AND_CLOSED_CONFIRMATION_REQUIRED")
    for name, digest in (("accepted_probe", ACCEPTED_DELIVERY), ("original_source", ORIGINAL_DELIVERY)):
        binding = settings.get(name, {})
        require(isinstance(binding, dict) and isinstance(binding.get("archive"), str)
                and bool(binding["archive"]) and binding.get("delivery_sha256") == digest,
                "FROZEN_ARCHIVE_BINDING_REQUIRED:" + name)
    require(settings["accepted_probe"].get("execution_report_sha256") == ACCEPTED_REPORT
            and settings["accepted_probe"].get("snapshot_sha256") == ACCEPTED_SNAPSHOT,
            "ACCEPTED_OBSERVATION_BINDING_CHANGED")
    return settings


def stop_tick(stop_ms, dt_ms, full_duration_ms):
    require(all(type(x) in (int, float) and np.isfinite(x) and x > 0
                for x in (stop_ms, dt_ms, full_duration_ms)), "VALID_CLOCK_REQUIRED")
    ticks = round(stop_ms / dt_ms)
    require(0 < stop_ms <= min(100., full_duration_ms)
            and np.isclose(ticks * dt_ms, stop_ms, atol=1e-12, rtol=0), "EXACT_BOUNDED_STOP_CLOCK_REQUIRED")
    return ticks


def clip_events(raw, config, absolute_stop_tick):
    events = branch.canonical_events(raw["spike_ticks"], raw["spike_unit_indices"],
        config["excitatory_neurons"] + config["inhibitory_neurons"], round(config["duration_ms"] / config["dt_ms"]))
    return events[events[:, 0] < absolute_stop_tick]


def _bindings():
    extra = (branch, pfc_count_histograms, pfc_fluctuation_analysis)
    return dict(source=branch._code_bindings(), numpy_version=np.__version__, helpers={module.__name__.split(".")[-1]:
                sha256_file(Path(module.__file__)) for module in extra}, implementation=sha256_file(Path(__file__)))


def _manifest(root, digest, expected_status):
    require(sha256_file(root / "delivery.json") == digest, "DELIVERY_BINDING_CHANGED")
    delivery = json.loads((root / "delivery.json").read_text())
    require(delivery.get("verified") is True and delivery.get("status") == expected_status,
            "VERIFIED_COMPLETE_ARCHIVE_REQUIRED")
    manifest = {row["path"]: row for row in delivery["files"]}
    require(len(manifest) == len(delivery["files"]), "DUPLICATE_MANIFEST_PATH")
    return manifest


def _snapshot_arrays(path, config):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    expected = {population + "__" + field for population in ("E", "I")
                for field in branch.STATE_FIELDS + (branch.E_FIELDS if population == "E" else ())}
    require(set(arrays) == expected, "COMPLETE_REFERENCE_STATE_REQUIRED")
    for key, value in arrays.items():
        size = config["excitatory_neurons" if key.startswith("E__") else "inhibitory_neurons"]
        require(value.shape == (size,) and value.dtype == (bool if key.endswith("not_refractory") else np.float64)
                and np.isfinite(value).all(), "REFERENCE_STATE_SCHEMA_CHANGED:" + key)
    return arrays


def prepare(config_path, project_root, output):
    config_path, project_root, output = Path(config_path), Path(project_root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report, names = dict(status=FAILED, stage="prepare", **scope()), []
    started = time.perf_counter()
    try:
        settings = validate_settings(yaml.safe_load(config_path.read_text()))
        accepted = settings["accepted_probe"]
        root = project_root / accepted["archive"]
        manifest = _manifest(root, accepted["delivery_sha256"], "COMPLETE_PFC_THRESHOLD_BRANCH_TWO_PATH_WITNESS")
        paths = {name: branch._bound_file(root, name, manifest[name]) for name in ACCEPTED_FILES}
        require(manifest["execution/report.json"]["sha256"] == ACCEPTED_REPORT
                and manifest["execution/original_pre_threshold.npz"]["sha256"] == ACCEPTED_SNAPSHOT,
                "ACCEPTED_REPORT_OR_SNAPSHOT_CHANGED")
        wrapper = json.loads(paths["report.json"].read_text())
        prior = json.loads(paths["execution/report.json"].read_text())
        accepted_plan = json.loads(paths["prepared/plan.json"].read_text())
        require(wrapper.get("exit_code") == 0 and prior.get("status") == branch.PASSED
                and prior["paths"]["original"]["archived_prefix_equal"] is True
                and prior["paths"]["original"]["prefix_event_count"] == 183
                and prior["first_mismatch"]["tick"] == settings["forced_tick"]
                and prior["first_mismatch"]["reversed_only_unit_indices"] == [settings["forced_unit_index"]]
                and prior["first_mismatch"]["common_prior_event_count"] == 0,
                "ACCEPTED_ORIGINAL_PREFIX_REQUIRED")
        bindings = _bindings()
        require(accepted_plan["current_code"] == bindings["source"]
                and accepted_plan["probe_sha256"] == bindings["helpers"]["pfc_threshold_branch_probe"],
                "UNCHANGED_ACCEPTED_SOURCE_REQUIRED")
        config, case = accepted_plan["full_config"], accepted_plan["selected_case"]
        require(config["duration_ms"] == 5000. and config["dt_ms"] == settings["dt_ms"]
                and config["excitatory_neurons"] == 80000 and config["inhibitory_neurons"] == 20000
                and config["stimulus_start_ms"] == settings["stop_ms"]
                and case == dict(case_id="pfc_95001_omission", source_rng_seed=95001,
                                 stimulus_strength_nA=0., distractor_strength_nA=0.), "FROZEN_BASELINE_PROTOCOL_REQUIRED")
        end = stop_tick(settings["stop_ms"], settings["dt_ms"], config["duration_ms"])
        original_binding = settings["original_source"]
        source_root = project_root / original_binding["archive"]
        source_manifest = _manifest(source_root, original_binding["delivery_sha256"],
                                    "COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT")
        prefix = settings["case_relative_path"]
        source_paths = {name: branch._bound_file(source_root, prefix + "/" + name,
                                                source_manifest[prefix + "/" + name])
                        for name in ("report.json", "raw.npz")}
        original = json.loads(source_paths["report.json"].read_text())
        for name in source_paths:
            require(source_manifest[prefix + "/" + name]["sha256"]
                    == accepted_plan["source_bindings"]["original"]["files"][name]["sha256"],
                    "ACCEPTED_ORIGINAL_CASE_BINDING_CHANGED:" + name)
        require(original["resolved_config"] == config and original["selected_case"] == case
                and original["status"] == "COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE"
                and original["source_execution_complete"] is True
                and original["cost"]["arithmetic_path"]["numpy_version"] == np.__version__,
                "ORIGINAL_CONFIG_CASE_OR_NUMPY_CHANGED")
        with np.load(source_paths["raw.npz"], allow_pickle=False) as raw:
            events = clip_events(raw, config, end)
        require(not np.any(events[:, 0] <= settings["forced_tick"])
                and int(np.count_nonzero(events[:, 0] < 400)) == 183,
                "ORIGINAL_PREINTERVENTION_OR_SHORT_PREFIX_CHANGED")
        initial_path = paths["prepared/initial_voltage.npy"]
        require(sha256_file(initial_path) == accepted_plan["artifacts"]["initial_voltage.npy"]["sha256"],
                "ACCEPTED_INITIAL_BINDING_CHANGED")
        _snapshot_arrays(paths["execution/original_pre_threshold.npz"], config)
        for source, name in ((initial_path, "initial_voltage.npy"),
                             (paths["execution/original_pre_threshold.npz"], "original_pre_threshold.npz")):
            shutil.copyfile(source, output / name)
            names.append(name)
        np.savez(output / "original_events.npz", spike_ticks=events[:, 0], spike_unit_indices=events[:, 1])
        names.append("original_events.npz")
        first_e = events[events[:, 1] < config["excitatory_neurons"], 0]
        plan = dict(schema=SCHEMA, status=PREPARED, settings=settings, full_config=config, selected_case=case,
                    stop_tick=end, bindings=bindings, observer_anchors=branch.observer_anchors(),
                    original_event_count=len(events), reused_original_short_prefix_count=183,
                    first_original_event_tick=int(events[0, 0]) if len(events) else None,
                    first_original_E_event_tick=int(first_e[0]) if len(first_e) else None,
                    accepted_probe=accepted, original_source=original_binding,
                    accepted_inputs={name: manifest[name] for name in ACCEPTED_FILES},
                    original_inputs={name: source_manifest[prefix + "/" + name] for name in source_paths},
                    artifacts={name: branch._file_record(output / name) for name in names}, **scope())
        freeze_json(output / "plan.json", plan)
        names.append("plan.json")
        report.update(status=PREPARED, original_event_count=len(events), stop_tick=end,
                      first_original_event_tick=plan["first_original_event_tick"])
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    report.update(outputs={name: branch._file_record(output / name) for name in names},
                  wall_seconds=time.perf_counter() - started)
    freeze_json(output / "report.json", report)
    return report


def load_prepared(prepared):
    root = Path(prepared)
    if root.name == "plan.json":
        root = root.parent
    receipt = json.loads((root / "report.json").read_text())
    require(receipt.get("status") == PREPARED, "SUCCESSFUL_PREPARATION_REQUIRED")
    plan_path = branch._bound_file(root, "plan.json", receipt["outputs"]["plan.json"])
    plan = json.loads(plan_path.read_text())
    validate_settings(plan["settings"])
    require(plan["bindings"] == _bindings() and plan["observer_anchors"] == branch.observer_anchors(),
            "PREPARED_SOURCE_OR_HELPER_CHANGED")
    require(set(plan["artifacts"]) == {"initial_voltage.npy", "original_events.npz", "original_pre_threshold.npz"},
            "PREPARED_ARTIFACT_SET_CHANGED")
    paths = {name: branch._bound_file(root, name, record) for name, record in plan["artifacts"].items()}
    require(plan["stop_tick"] == stop_tick(plan["settings"]["stop_ms"], plan["full_config"]["dt_ms"],
                                           plan["full_config"]["duration_ms"]), "PREPARED_STOP_CHANGED")
    return plan, paths


def _bitwise_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class SpikeBranchObserver(branch.ThresholdObserver):
    """Observe the unchanged engine; optionally impose exactly one voltage trigger."""

    def __init__(self, tick, unit, config, reference=None, *, intervene=True):
        super().__init__(tick, config)
        require(type(unit) is int and 0 <= unit < config["excitatory_neurons"], "EXCITATORY_TRIGGER_UNIT_REQUIRED")
        self.unit, self.reference, self.intervene = unit, reference, intervene
        self.applied = 0
        self.receipt = dict(intervention_applied=False, counterfactual=intervene)
        self.recurrence = {}
        self.followup = {}
        dt = config["dt_ms"] * 1e-3
        refractory = executor._constants(config, "E")["refractory"]
        self.recovery_tick = tick + int((refractory + 1e-3 * dt) / dt)

    def _trigger(self, local):
        super().capture("pre_threshold", local)
        actual = self.snapshots["pre_threshold"]
        if not self.intervene:
            return
        require(isinstance(self.reference, dict) and set(actual) == set(self.reference), "FULL_ARCHIVED_REFERENCE_REQUIRED")
        matches = {key: _bitwise_equal(value, self.reference[key]) for key, value in actual.items()}
        self.receipt["preintervention_bitwise_match"] = matches
        require(all(matches.values()), "PREINTERVENTION_STATE_MISMATCH")
        require(not local["event_ticks"] and not local["event_ids"], "UNEXPECTED_PRIOR_EVENTS")
        native = []
        for population, state, offset in (("E", local["e"], 0),
                                           ("I", local["i"], self.config["excitatory_neurons"])):
            threshold = local["constants"][population]["threshold"]
            require(threshold == -.050, "NATIVE_THRESHOLD_CHANGED")
            native.extend((np.flatnonzero((state["v"] > threshold) & state["not_refractory"]) + offset).tolist())
        self.receipt["native_threshold_events_at_trigger"] = native
        require(not native and self.applied == 0 and bool(local["e"]["not_refractory"][self.unit]),
                "ONE_ELIGIBLE_NON_NATIVE_EVENT_REQUIRED")
        before = float(local["e"]["v"][self.unit])
        trigger = np.nextafter(np.float64(-.050), np.float64(np.inf))
        local["e"]["v"][self.unit] = trigger
        self.applied += 1
        require(np.count_nonzero(local["e"]["v"] != actual["E__v"]) == 1,
                "EXACTLY_ONE_VOLTAGE_CHANGED_REQUIRED")
        self.receipt.update(intervention_applied=True, applied_event_count=self.applied,
            stage="after both source RK2 updates, immediately before source threshold selection",
            tick=self.tick, population="E", source_unit_index=self.unit, archive_unit_index=self.unit,
            before_voltage_V=before, imposed_voltage_V=float(trigger),
            intervention="instantaneous voltage trigger imposing one spike; ordinary source threshold/reset transition follows",
            changed_state_locations=["E__v[" + str(self.unit) + "]"])

    def _follow(self, local, stage):
        tick = int(local["tick"])
        state = local["e"]
        self.followup[str(tick) + "_" + stage] = {field: (bool(state[field][self.unit])
            if field == "not_refractory" else float(state[field][self.unit]))
            for field in branch.STATE_FIELDS + branch.E_FIELDS}
        if stage == "pre_rk2":
            arrays = {population + "__s_NMDA_total": local[name]["s_NMDA_total"].copy()
                      for population, name in (("E", "e"), ("I", "i"))}
            self.recurrence[tick] = arrays
            if self.intervene and tick == self.tick + 1:
                require(state["v"][self.unit] == -.060 and state["x"][self.unit] == 1.
                        and state["lastspike"][self.unit] == self.tick * (self.config["dt_ms"] * 1e-3)
                        and not state["not_refractory"][self.unit], "ORDINARY_SOURCE_RESET_TRANSITION_FAILED")
                require(sum(len(x) for x in local["event_ids"]) == 1
                        and int(local["event_ids"][0][0]) == self.unit
                        and int(local["event_ticks"][0][0]) == self.tick, "IMPOSED_NATIVE_EVENT_IDENTITY_FAILED")
                require(all(np.count_nonzero(value) == 0 for value in arrays.values()),
                        "RECURRENT_NMDA_APPEARED_BEFORE_SOURCE_DELAY")
        elif self.intervene:
            require(bool(state["not_refractory"][self.unit]), "INITIATING_UNIT_REFRACTORY_RECOVERY_FAILED")

    @contextmanager
    def observing(self):
        previous, target = sys.gettrace(), executor.simulate.__code__
        pre, post = self.anchors["pre_rk2"]["line"], self.anchors["pre_threshold"]["line"]

        def trace(frame, event, arg):
            if frame.f_code is not target:
                return None
            if event == "line":
                tick = frame.f_locals.get("tick")
                if frame.f_lineno == post and tick == self.tick:
                    self._trigger(frame.f_locals)
                elif frame.f_lineno == pre and tick in (self.tick + 1, self.tick + 2):
                    self._follow(frame.f_locals, "pre_rk2")
                elif frame.f_lineno == post and tick == self.recovery_tick:
                    self._follow(frame.f_locals, "pre_threshold")
            return trace
        try:
            sys.settrace(trace)
            yield self
        finally:
            sys.settrace(previous)


def event_comparison(original, counterfactual, units, tick, forced_unit, refractory_ticks, *, window_ticks=400, total_ticks=8000):
    def codes(events):
        return events[:, 0] * units + events[:, 1]
    def unpack(values):
        return np.column_stack((values // units, values % units)).astype(np.int64)
    original_only = unpack(np.setdiff1d(codes(original), codes(counterfactual)))
    counterfactual_only = unpack(np.setdiff1d(codes(counterfactual), codes(original)))
    imposed = (counterfactual_only[:, 0] == tick) & (counterfactual_only[:, 1] == forced_unit)
    require(np.count_nonzero(imposed) == 1, "EXACT_IMPOSED_EVENT_REQUIRED")
    remaining = np.concatenate((original_only, counterfactual_only[~imposed]))
    if len(remaining):
        remaining = remaining[np.lexsort((remaining[:, 1], remaining[:, 0]))]
    other = remaining[remaining[:, 1] != forced_unit]
    require(not np.any(other[:, 0] < tick + 2), "NON_TARGET_DIFFERENCE_PRECEDES_NMDA_CAUSAL_DELAY")
    target = counterfactual[counterfactual[:, 1] == forced_unit, 0]
    require(not np.any((target > tick) & (target < tick + refractory_ticks)), "INITIATING_UNIT_REFRACTORY_VIOLATION")
    result = dict(imposed_event=dict(tick=tick, archive_unit_index=forced_unit),
                  earliest_new_event_difference=None if not len(remaining) else dict(tick=int(remaining[0, 0]), archive_unit_index=int(remaining[0, 1])),
                  earliest_non_target_event_difference=None if not len(other) else dict(tick=int(other[0, 0]), archive_unit_index=int(other[0, 1])),
                  other_units_with_event_discrepancies=int(len(np.unique(other[:, 1]))),
                  non_target_event_identity_discrepancies=len(other),
                  event_schedule_locality_observed=not len(other),
                  original_only_events=len(original_only), counterfactual_only_events=len(counterfactual_only),
                  imposed_event_reported_separately=True)
    result["five_ms_event_discrepancies"] = [dict(start_tick=start, stop_tick=start + window_ticks,
        original_only_events=int(np.count_nonzero((original_only[:, 0] >= start) & (original_only[:, 0] < start + window_ticks))),
        counterfactual_only_events=int(np.count_nonzero((counterfactual_only[:, 0] >= start) & (counterfactual_only[:, 0] < start + window_ticks))),
        non_target_event_discrepancies=int(np.count_nonzero((other[:, 0] >= start) & (other[:, 0] < start + window_ticks))))
        for start in range(0, total_ticks, window_ticks)]
    return result, dict(original_only=original_only, counterfactual_only=counterfactual_only,
                        counterfactual_only_is_imposed=imposed)


def count_comparison(original, counterfactual, config, settings, output):
    ne, ni, dt = config["excitatory_neurons"], config["inhibitory_neurons"], config["dt_ms"]
    units, total_ticks = ne + ni, round(settings["stop_ms"] / dt)
    fields, sizes = pfc_fluctuation_analysis.unit_mapping(ne, ni, config["excitatory_preference_bins"])
    five_ticks = round(5. / dt)
    counts = {}
    for name, events in (("original", original), ("counterfactual", counterfactual)):
        flat = (events[:, 0] // five_ticks) * units + events[:, 1]
        counts[name] = np.bincount(flat, minlength=(total_ticks // five_ticks) * units).reshape(-1, units)
    histogram, arrays, rows, bound_rows = {}, {}, [], []
    others = np.arange(units) != settings["forced_unit_index"]
    for w, width in enumerate(settings["windows_ms"]):
        bins, ticks = round(width / 5.), round(width / dt)
        grouped = {name: values.reshape(-1, bins, units).sum(axis=1, dtype=np.int64) for name, values in counts.items()}
        # Source availability uses the integer-timestep refractory comparison.
        refr = {p: int((executor._constants(config, p)["refractory"] + 1e-3 * dt * 1e-3) / (dt * 1e-3)) for p in ("E", "I")}
        bounds = np.r_[np.full(ne, (ticks - 1) // refr["E"] + 1), np.full(ni, (ticks - 1) // refr["I"] + 1)]
        for condition, name in enumerate(("original", "counterfactual")):
            require(np.all(grouped[name] <= bounds), "COUNT_EXCEEDS_SOURCE_REFRACTORY_BOUND")
            pfc_count_histograms.accumulate(histogram, counts[name][None], fields, bins,
                                            condition=condition, phase=0, window=w)
            arrays[name + "_" + str(int(width)) + "ms"] = grouped[name]
        delta = grouped["counterfactual"] - grouped["original"]
        arrays["delta_" + str(int(width)) + "ms"] = delta
        field_delta = np.stack([delta[:, fields == field].sum(axis=1) for field in range(len(sizes))], axis=1)
        arrays["field_delta_" + str(int(width)) + "ms"] = field_delta
        for index, values in enumerate(delta):
            rows.append(dict(window_ms=width, start_ms=index * width, stop_ms=(index + 1) * width,
                signed_total_delta=int(values.sum()), absolute_unit_count_delta=int(np.abs(values).sum()),
                units_with_count_difference=int(np.count_nonzero(values)),
                non_target_signed_delta=int(values[others].sum()),
                non_target_absolute_count_delta=int(np.abs(values[others]).sum()),
                non_target_units_with_count_difference=int(np.count_nonzero(values[others]))))
        bound_rows.append(dict(window_ms=width, E_max_count=int(bounds[0]), I_max_count=int(bounds[-1])))
    coordinates = np.asarray(sorted(histogram), dtype=np.int64)
    mass = np.asarray([histogram[tuple(key)] for key in coordinates], dtype=np.int64)
    support = np.arange(int(coordinates[:, 5].max()) + 1, dtype=np.int64)
    cdfs = pfc_count_histograms.cdf(coordinates, mass, support, (2, 1, 1, 3, len(sizes)))
    require(np.all(cdfs[..., -1] == 1.), "CDF_ALL_MASS_REQUIRED")
    arrays.update(archive_unit_indices=np.arange(units), source_field_indices=fields, field_unit_counts=sizes)
    np.savez_compressed(output / "unit_window_count_differences.npz", **arrays)
    np.savez_compressed(output / "descriptive_count_cdfs.npz", count_support=support, cdf=cdfs,
                        sparse_coordinates=coordinates, observations=mass, windows_ms=settings["windows_ms"],
                        field_unit_counts=sizes, condition_names=np.asarray(["original", "counterfactual"]))
    return dict(windows=rows, physical_count_bounds=bound_rows, field_count=len(sizes),
                count_locality_observed=all(row["non_target_units_with_count_difference"] == 0 for row in rows),
                full_100ms_count_delta_signed=int(arrays["delta_100ms"].sum()),
                full_100ms_count_delta_absolute=int(np.abs(arrays["delta_100ms"]).sum()),
                max_absolute_field_cdf_difference=float(np.abs(cdfs[1] - cdfs[0]).max()),
                cdf_support_min=0, cdf_support_max=int(support[-1]),
                cdf_denominator="all source units within each field times complete nonoverlapping baseline windows",
                cdf_is_descriptive_only=True, unit_windows_are_independent_replicates=False,
                field_coordinates="source preferred-angle bins; not anatomical soma positions")


def execute(prepared, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report, names = dict(status=FAILED, stage="execute", **scope()), []
    observer = None
    started = time.perf_counter()
    try:
        plan, paths = load_prepared(prepared)
        settings, config, case = plan["settings"], plan["full_config"], plan["selected_case"]
        report.update(settings=settings, bindings=plan["bindings"], source_inputs=plan["original_inputs"],
                      accepted_inputs=plan["accepted_inputs"], stop_ms=settings["stop_ms"], stop_tick=plan["stop_tick"],
                      reused_original_short_prefix_count=183, baseline_simulations_run=0,
                      terminal_E_I_state_max_difference=None,
                      terminal_state_comparison="original 100ms state unavailable; no state contraction inferred")
        reference = _snapshot_arrays(paths["original_pre_threshold.npz"], config)
        with np.load(paths["original_events.npz"], allow_pickle=False) as raw:
            original = clip_events(raw, dict(config, duration_ms=settings["stop_ms"]), plan["stop_tick"])
        initial = np.load(paths["initial_voltage.npy"], allow_pickle=False)
        generated, drive = pfc_coupled_drive.create_coupled_drive(config, case, settings["coarse_dt_ms"], settings["refinement_seeds"])
        require(_bitwise_equal(initial, generated) and drive.total_ticks == round(5000. / settings["dt_ms"]),
                "FULL_ORIGINAL_DRIVE_OR_INITIAL_CHANGED")
        short_config = deepcopy(config)
        short_config["duration_ms"] = settings["stop_ms"]
        observer = SpikeBranchObserver(settings["forced_tick"], settings["forced_unit_index"], config, reference)
        with observer.observing():
            result = executor.simulate(short_config, case, generated, backend="numpy", drive=drive)
        np.savez(output / "counterfactual_raw.npz", **result["raw"])
        np.savez(output / "counterfactual_terminal_state.npz", **result["terminal_state"])
        names.extend(["counterfactual_raw.npz", "counterfactual_terminal_state.npz"])
        report["counterfactual_terminal_recurrent_nmda"] = {
            "scope": "source terminal arrays; summed NMDA last computed before the final RK2; no original terminal-state comparison",
            "populations": {p: dict(
                l1=float(np.abs(result["terminal_state"][p + "__s_NMDA_total"]).sum()),
                linf=float(np.abs(result["terminal_state"][p + "__s_NMDA_total"]).max()))
                for p in ("E", "I")}}

        counterfactual = clip_events(result["raw"], short_config, plan["stop_tick"])
        require(observer.applied == 1 and drive.ticks == plan["stop_tick"], "ONE_INTERVENTION_COMPLETE_PREFIX_REQUIRED")
        report.update(cost=result["cost"], drive=drive.metadata(), completed_counterfactual_prefixes=1,
                      counterfactual_event_count=len(counterfactual), original_event_count=len(original))
        comparison, arrays = event_comparison(original, counterfactual, settings["expected_units"],
                    settings["forced_tick"], settings["forced_unit_index"], observer.recovery_tick - observer.tick)
        np.savez(output / "event_identity_differences.npz", **arrays)
        names.append("event_identity_differences.npz")
        report["event_comparison"] = comparison
        report["count_comparison"] = count_comparison(original, counterfactual, config, settings, output)
        names.extend(["unit_window_count_differences.npz", "descriptive_count_cdfs.npz"])
        require(set(observer.recurrence) == {observer.tick + 1, observer.tick + 2}
                and str(observer.recovery_tick) + "_pre_threshold" in observer.followup,
                "SOURCE_DELAY_AND_RECOVERY_OBSERVATIONS_REQUIRED")
        recurrence = {}
        for tick, arrays in observer.recurrence.items():
            first_e = plan["first_original_E_event_tick"]
            zero_reference = first_e is None or tick < first_e + 2
            recurrence[str(tick)] = dict(stage="preRK2 after source NMDA sums and current update",
                original_recurrent_NMDA_known_zero_from_spike_history=zero_reference,
                populations={p: dict(l1=float(np.abs(arrays[p + "__s_NMDA_total"]).sum()),
                    l2=float(np.linalg.norm(arrays[p + "__s_NMDA_total"])),
                    linf=float(np.abs(arrays[p + "__s_NMDA_total"]).max())) for p in ("E", "I")})
        require(all(np.any(observer.recurrence[observer.tick + 2][p + "__s_NMDA_total"] != 0) for p in ("E", "I")),
                "EXPECTED_DELAYED_RECURRENT_NMDA_NOT_OBSERVED")
        report.update(status=COMPLETE, recurrence=recurrence,
                      conclusion="Descriptive single-event sensitivity under the common original future stream; event and count locality are assessed separately.")
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    finally:
        if observer is not None:
            report["intervention"] = observer.receipt
            report["source_stage_followup"] = observer.followup
            for stage, arrays in observer.snapshots.items():
                name = "counterfactual_before_intervention_" + stage + ".npz"
                np.savez(output / name, **arrays)
                names.append(name)
            for tick, arrays in observer.recurrence.items():
                name = "pre_rk2_recurrent_nmda_tick_" + str(tick) + ".npz"
                np.savez(output / name, **arrays)
                names.append(name)
    report.update(outputs={name: branch._file_record(output / name) for name in names},
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
            parser.error("prepare requires --config and --project-root")
        report = prepare(args.config, args.project_root, args.output)
    else:
        if args.prepared is None or args.config is not None or args.project_root is not None:
            parser.error("execute requires --prepared")
        report = execute(args.prepared, args.output)
    print(json.dumps({k: report[k] for k in ("status", "error") if k in report}, indent=2))
    return 0 if report["status"] in (PREPARED, COMPLETE) else 1


if __name__ == "__main__":
    raise SystemExit(main())
