"""A two-coordinate ideal fine-input response table on a fixed PFC background.

The replay seed reconstructs the fixed other inputs; it is not conditioned on
in the probability calculation. Omitted input counts retain their full mass.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
import yaml

from . import pfc_threshold_branch_probe as branch
from . import pfc_spike_branch_continuation as continuation
from . import pfc_conditional_spike_probability as conditional
from . import pfc_coupled_drive, pfc_clock_temporal_control
from . import pfc_deterministic_replay as executor
from .provenance import freeze_json, sha256_file

SCHEMA = "pfc_innovation_response_v1"
PREPARED = "PREPARED_PFC_INNOVATION_RESPONSE"
COMPLETE = "COMPLETE_PFC_INNOVATION_RESPONSE"
FAILED = "FAILED_PFC_INNOVATION_RESPONSE"
SOURCE_ARCHIVES = {
    "kernel_source": ("b2aeacffdd7d39cc12662556fbec63717eb0112f8e852a7ebff4fbfb951ff0a1", "COMPLETE_PFC_CONDITIONAL_SPIKE_KERNEL"),
    "threshold_source": ("0028178795cca3f0255fcacc45adf6a21baf9f295e22aadc0d79fa8e3839ef1b", "COMPLETE_PFC_THRESHOLD_BRANCH_TWO_PATH_WITNESS"),
    "original_reference": ("70577629a08efc031d6576be2b612a2464f99ded124b65f9407520f71d1f7ab0", "COMPLETE_PFC_SPIKE_BRANCH_CONTINUATION"),
    "reversed_reference": ("648b874eb79480b5d3bffb0589883fa1794f4b26a59c65c30500d9a28cef23ff", "COMPLETE_PFC_COUNT_BOUNDARY_DECOMPOSITION"),
}
INPUT_FILES = {
    "kernel_source": ("analysis/report.json", "analysis/conditional_thresholds.npz", "analysis/binomial_tail_bounds.csv"),
    "threshold_source": ("execution/report.json", "prepared/plan.json", "prepared/initial_voltage.npy",
                         "execution/original_pre_threshold.npz", "execution/reversed_pre_threshold.npz"),
    "original_reference": ("execution/report.json", "prepared/plan.json", "prepared/original_events.npz"),
    "reversed_reference": ("analysis/report.json", "analysis/prefix_events.npz"),
}
VARIANTS = ("original", "reversed")
CELLS = tuple((x, y) for x in range(3) for y in range(3))
REALIZED_CELL = (0, 1)
CONDITIONING = "all_other_fine_input_coordinates_including_future_and_initial_state_and_controls"


def require(condition, message):
    if not condition:
        raise ValueError("PFC_INNOVATION_RESPONSE_" + message)


def scope():
    return dict(two_free_ideal_fine_coordinates=True, fixed_background_pointwise_response=True,
                source_seed_used_only_for_computational_replay=True, computational_rng_calls_replayed=True,
                finite_prng_distribution_certified=False, selection_conditioned_probability=False,
                all_background_histories_qualified=False, full_84330_endpoint_bound_qualified=False,
                full_trajectory_bound_established=False, scientific_fidelity_qualified=False,
                confirmation_execution_released=False, production_episodes=0,
                new_full_episodes=0, new_independent_blocks=0, new_auxiliary_streams=0)


def validate_settings(settings):
    expected = dict(schema=SCHEMA, source_rng_seed=95001, refinement_seeds=[95411, 95412],
        expected_units=100000, unit_index=6423, input_ticks=[88, 91], values=[0, 1, 2],
        coarse_dt_ms=.05, dt_ms=.0125, stop_ms=100., windows_ms=[5., 50., 100.],
        conditioning=CONDITIONING, condition_on_outputs=False, condition_on_coarse_parents=False,
        condition_on_selection=False, condition_on_rng_or_seed=False,
        omitted_mass_ceiling="0.000002", confirmation_execution_released=False, production_episodes=0)
    require(isinstance(settings, dict), "SETTINGS_MAPPING_REQUIRED")
    for key, value in expected.items():
        require(settings.get(key) == value and (type(settings.get(key)) is bool if type(value) is bool else
                not isinstance(settings.get(key), bool)), "FROZEN_SETTING_CHANGED:" + key)
    require(all(type(settings[k]) is int for k in ("source_rng_seed", "expected_units", "unit_index", "production_episodes"))
            and all(type(v) is int for key in ("refinement_seeds", "input_ticks", "values") for v in settings[key]),
            "INTEGER_COORDINATES_AND_SEEDS_REQUIRED")
    for key, (digest, _) in SOURCE_ARCHIVES.items():
        value = settings.get(key, {})
        require(isinstance(value, dict) and isinstance(value.get("archive"), str) and value["archive"]
                and value.get("delivery_sha256") == digest, "FROZEN_SOURCE_BINDING_REQUIRED:" + key)
    return settings


def code_bindings():
    return dict(source=branch._code_bindings(), numpy_version=np.__version__,
                helpers={module.__name__.split(".")[-1]: sha256_file(Path(module.__file__))
                         for module in (branch, continuation, conditional)},
                implementation=sha256_file(Path(__file__)))


def directed_masses(tails, digits):
    """Difference directed tails, then multiply independent coordinate bounds."""
    require(type(digits) is int and digits > 0, "DECIMAL_SCALE_REQUIRED")
    scale = 10 ** digits
    require(len(tails) >= 4, "TAILS_ZERO_THROUGH_THREE_REQUIRED")
    bounds = []
    for index, row in enumerate(tails):
        lower, upper = int(row["lower_scaled"]), int(row["upper_scaled"])
        require(int(row["minimum_events"]) == index and 0 <= lower <= upper <= scale,
                "ORDERED_DIRECTED_TAILS_REQUIRED")
        bounds.append((lower, upper))
    require(bounds[0] == (scale, scale)
            and all(a[0] >= b[0] and a[1] >= b[1] for a, b in zip(bounds, bounds[1:])),
            "MONOTONE_FULL_TAIL_REQUIRED")
    marginal = [(max(0, bounds[k][0] - bounds[k + 1][1]),
                 min(scale, bounds[k][1] - bounds[k + 1][0])) for k in range(3)]
    require(all(0 <= lo <= hi <= scale for lo, hi in marginal), "VALID_PMF_DIFFERENCES_REQUIRED")
    cells = []
    for x, y in CELLS:
        lo = marginal[x][0] * marginal[y][0] // scale
        numerator = marginal[x][1] * marginal[y][1]
        hi = (numerator + scale - 1) // scale
        cells.append(dict(x=x, y=y, lower_scaled=str(lo), upper_scaled=str(hi)))
    tail_lower, tail_upper = bounds[3]
    lower = (2 * scale * tail_lower - tail_lower * tail_lower) // scale
    numerator = 2 * scale * tail_upper - tail_upper * tail_upper
    upper = (numerator + scale - 1) // scale
    return dict(decimal_scale_digits=digits, scale=str(scale),
                marginal=[dict(value=k, lower_scaled=str(a), upper_scaled=str(b)) for k, (a, b) in enumerate(marginal)],
                cells=cells, omitted_mass=dict(lower_scaled=str(lower), upper_scaled=str(upper)),
                retained_mass=dict(lower_scaled=str(scale - upper), upper_scaled=str(scale - lower)),
                normalized_after_truncation=False, omitted_formula="2*P(X>=3)-P(X>=3)^2")


def immediate_xor_bounds(tails, scale):
    a = [int(tails[7][name]) for name in ("lower_scaled", "upper_scaled")]
    b = [int(tails[1][name]) for name in ("lower_scaled", "upper_scaled")]
    values = [scale * (x + y) - 2 * x * y for x in a for y in b]
    return dict(lower_scaled=str(min(values) // scale),
                upper_scaled=str((max(values) + scale - 1) // scale))


def weighted_bounds(numerators, denominator, masses, *, signed):
    """Directed scaled expectations using exact outcome numerators/denominators."""
    values, denominator = np.asarray(numerators), np.asarray(denominator)
    require(values.dtype.kind in "iu" and values.shape[0] == len(masses["cells"])
            and denominator.dtype.kind in "iu" and np.all(denominator > 0), "INTEGER_OUTCOMES_AND_DENOMINATOR_REQUIRED")
    denominator = np.broadcast_to(denominator, values.shape[1:]).astype(object)
    values = values.astype(object)
    require(np.all(values <= denominator) and np.all(values >= (-denominator if signed else 0)), "BOUNDED_OUTCOME_REQUIRED")
    lower = np.zeros(values.shape[1:], dtype=object)
    upper = np.zeros(values.shape[1:], dtype=object)
    for value, mass in zip(values, masses["cells"]):
        lo, hi = int(mass["lower_scaled"]), int(mass["upper_scaled"])
        # Scalar slices are Python ints; explicit object arrays also prevent
        # np.where from trying to infer a C-long dtype for their products.
        low_product = np.asarray(value * lo, dtype=object)
        high_product = np.asarray(value * hi, dtype=object)
        lower += np.where(value >= 0, low_product, high_product)
        upper += np.where(value >= 0, high_product, low_product)
    retained_lower, retained_upper = lower // denominator, -((-upper) // denominator)
    epsilon, scale = int(masses["omitted_mass"]["upper_scaled"]), int(masses["scale"])
    # Object arithmetic on a scalar-shaped array can return a Python int.
    # Keep both ufunc operands explicit objects so NumPy never promotes 10**40
    # to a platform C long while clipping a scalar probability interval.
    lower = np.asarray(retained_lower - (epsilon if signed else 0), dtype=object)
    upper = np.asarray(retained_upper + epsilon, dtype=object)
    lower = np.maximum(np.full(lower.shape, -scale if signed else 0, dtype=object), lower)
    upper = np.minimum(np.full(upper.shape, scale, dtype=object), upper)
    return {key: np.asarray(value, dtype=str) for key, value in dict(
        lower_scaled=lower, upper_scaled=upper, retained_lower_scaled=retained_lower,
        retained_upper_scaled=retained_upper).items()}


class InnovationDrive:
    """Replace two returned fine-count coordinates without altering RNG advancement."""

    def __init__(self, source, variant, unit, x, y, *, input_ticks=(88, 91), expected_original=None):
        require(variant in VARIANTS and source.ticks == 0 and source.factor == 4,
                "FRESH_QUARTER_CLOCK_SOURCE_REQUIRED")
        require(type(unit) is int and 0 <= unit < source.n_e
                and all(type(value) is int and 0 <= value <= 1000 for value in (x, y)), "BINOMIAL_COUNT_SUPPORT_REQUIRED")
        a, b = input_ticks
        require(type(a) is int and type(b) is int and a >= 0 and a % 4 == 0 and b == a + 3
                and b < source.total_ticks, "REVERSAL_ENDPOINT_COORDINATES_REQUIRED")
        self.source, self.variant, self.unit = source, variant, unit
        self.assignments = {a: x, b: y} if variant == "original" else {a: y, b: x}
        self.expected_original = expected_original
        self.observations = []
        self.event_delta = 0
        self.n_e, self.n_i = source.n_e, source.n_i
        self.dt_seconds, self.total_ticks, self.max_rows = source.dt_seconds, source.total_ticks, source.max_rows

    @property
    def ticks(self):
        return self.source.ticks

    def take(self, rows):
        start = self.ticks
        values = self.source.take(rows)
        # The source returns independent packed storage. Copy before substitution
        # so this wrapper never mutates a source-returned view held by a caller.
        result = {key: value.copy() for key, value in values.items()}
        for tick, requested in self.assignments.items():
            if start <= tick < self.ticks:
                old = int(result["E"][tick - start, self.unit])
                if self.expected_original is not None:
                    require(old == self.expected_original[tick], "ARCHIVED_REALIZED_COORDINATE_CHANGED")
                result["E"][tick - start, self.unit] = requested
                self.event_delta += requested - old
                self.observations.append(dict(input_tick=tick, population="E", source_unit_index=self.unit,
                                              original_value=old, substituted_value=requested))
        return result

    def metadata(self):
        return dict(source=self.source.metadata(), input_coordinate_overrides=self.observations,
                    E_external_event_delta=self.event_delta, I_external_event_delta=0,
                    same_rng_advancement_outside_substitution=True,
                    conditioning_on_seed_or_rng=False, conditioned_coarse_parent_counts=False,
                    original_coordinate_order=[88, 91], temporal_reversal_swaps_coordinate_values=self.variant == "reversed")


def _source_inputs(settings, project_root):
    paths, records = {}, {}
    for key, (_, status) in SOURCE_ARCHIVES.items():
        root = project_root / settings[key]["archive"]
        manifest = continuation._manifest(root, settings[key]["delivery_sha256"], status)
        paths[key] = {name: branch._bound_file(root, name, manifest[name]) for name in INPUT_FILES[key]}
        records[key] = {name: manifest[name] for name in INPUT_FILES[key]}
    return paths, records


def prepare(config_path, project_root, output):
    config_path, project_root, output = Path(config_path), Path(project_root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report, names = dict(status=FAILED, stage="prepare", **scope()), []
    started = time.perf_counter()
    try:
        settings = validate_settings(yaml.safe_load(config_path.read_text()))
        inputs, records = _source_inputs(settings, project_root)
        kernel = json.loads(inputs["kernel_source"]["analysis/report.json"].read_text())
        source_plan = json.loads(inputs["threshold_source"]["prepared/plan.json"].read_text())
        original_plan = json.loads(inputs["original_reference"]["prepared/plan.json"].read_text())
        original_report = json.loads(inputs["original_reference"]["execution/report.json"].read_text())
        reverse_report = json.loads(inputs["reversed_reference"]["analysis/report.json"].read_text())
        accepted = json.loads(inputs["threshold_source"]["execution/report.json"].read_text())
        config, case = source_plan["full_config"], source_plan["selected_case"]
        bindings = code_bindings()
        require(source_plan["current_code"] == bindings["source"]
                and source_plan["probe_sha256"] == bindings["helpers"]["pfc_threshold_branch_probe"], "UNCHANGED_NUMERICAL_SOURCE_REQUIRED")
        require(config == original_plan["full_config"] == reverse_report["config"]
                and case == original_plan["selected_case"] == reverse_report["selected_case"]
                and config["duration_ms"] == 5000. and config["dt_ms"] == .0125
                and config["excitatory_neurons"] == 80000 and config["inhibitory_neurons"] == 20000
                and case == dict(case_id="pfc_95001_omission", source_rng_seed=95001,
                                 stimulus_strength_nA=0., distractor_strength_nA=0.), "FROZEN_SOURCE_PROTOCOL_REQUIRED")
        require(accepted["status"] == branch.PASSED and original_report["status"] == continuation.COMPLETE
                and reverse_report["status"] == "COMPLETE_PFC_COUNT_BOUNDARY_DECOMPOSITION"
                and reverse_report["stop_tick"] == 8000 and original_plan["stop_tick"] == 8000,
                "ACCEPTED_100MS_REFERENCES_REQUIRED")
        for name in ("prepared/plan.json", "prepared/initial_voltage.npy", "execution/original_pre_threshold.npz", "execution/reversed_pre_threshold.npz"):
            require(kernel["source_bindings"][name]["sha256"] == records["threshold_source"][name]["sha256"], "KERNEL_WITNESS_LINEAGE_CHANGED:" + name)
        law = kernel["input_law"]
        probability = float.fromhex(law["binary64_probability_hex"])
        require(kernel["status"] == "COMPLETE_PFC_CONDITIONAL_SPIKE_KERNEL"
                and kernel["tick"] == 89 and kernel["input_tick"] == 88 and kernel["decimal_scale_digits"] == 40
                and law["n"] == 1000 and probability.as_integer_ratio() == (int(law["exact_numerator"]), int(law["exact_denominator"])),
                "EXACT_BINARY64_IDEAL_INPUT_LAW_REQUIRED")
        require(kernel["variants"]["original"]["drive_metadata"]["coupling"]["coarse_source"]["numpy_version"] == np.__version__,
                "SOURCE_NUMPY_VERSION_CHANGED")
        with np.load(inputs["kernel_source"]["analysis/conditional_thresholds.npz"], allow_pickle=False) as data:
            thresholds = {variant: int(data[variant + "_minimum_input_for_spike"][6423]) for variant in VARIANTS}
            actual = {variant: int(data[variant + "_E_actual_input"][6423]) for variant in VARIANTS}
        require(thresholds == dict(original=7, reversed=1) and actual == dict(original=0, reversed=1),
                "ACCEPTED_FOCUS_INPUT_AND_THRESHOLD_CHANGED")
        with inputs["kernel_source"]["analysis/binomial_tail_bounds.csv"].open(newline="") as stream:
            tails = list(csv.DictReader(stream))
        require(len(tails) == 1002, "COMPLETE_BINOMIAL_TAIL_TABLE_REQUIRED")
        masses = directed_masses(tails, 40)
        require(int(masses["omitted_mass"]["upper_scaled"]) <= 2 * 10 ** 34, "PREDECLARED_OMITTED_MASS_CEILING_EXCEEDED")
        short = dict(config, duration_ms=100.)
        with np.load(inputs["original_reference"]["prepared/original_events.npz"], allow_pickle=False) as raw:
            original = continuation.clip_events(raw, short, 8000)
        with np.load(inputs["reversed_reference"]["analysis/prefix_events.npz"], allow_pickle=False) as raw:
            reversed_events = branch.canonical_events(raw["reversal_ticks"], raw["reversal_units"], 100000, 8000)
            duplicate_original = branch.canonical_events(raw["original_ticks"], raw["original_units"], 100000, 8000)
        require(np.array_equal(original, duplicate_original) and len(original) == original_report["original_event_count"], "ORIGINAL_REFERENCE_PREFIX_CHANGED")
        for variant, events in (("original", original), ("reversed", reversed_events)):
            require(int(np.count_nonzero(events[:, 0] < 400)) == accepted["paths"][variant]["prefix_event_count"], "ACCEPTED_SHORT_PREFIX_COUNT_CHANGED")
            name = variant + "_events.npz"
            np.savez(output / name, spike_ticks=events[:, 0], spike_unit_indices=events[:, 1])
            names.append(name)
            snapshot = inputs["threshold_source"]["execution/" + variant + "_pre_threshold.npz"]
            continuation._snapshot_arrays(snapshot, config)
            name = variant + "_pre_threshold.npz"
            shutil.copyfile(snapshot, output / name)
            names.append(name)
        shutil.copyfile(inputs["threshold_source"]["prepared/initial_voltage.npy"], output / "initial_voltage.npy")
        names.append("initial_voltage.npy")
        plan = dict(schema=SCHEMA, status=PREPARED, settings=settings, full_config=config, selected_case=case,
            stop_tick=continuation.stop_tick(100., .0125, 5000.), bindings=bindings, inputs=records,
            probability_law=law, probability_bounds=masses, immediate_thresholds=thresholds,
            immediate_kernel_xor=immediate_xor_bounds(tails, 10 ** 40),
            realized_canonical_cell=[0, 1], realized_physical_coordinates=dict(original=[0, 1], reversed=[1, 0]),
            source_costs={variant: source_plan["source_bindings"][variant]["cost"] for variant in VARIANTS},
            artifacts={name: branch._file_record(output / name) for name in names}, **scope())
        freeze_json(output / "plan.json", plan)
        names.append("plan.json")
        report.update(status=PREPARED, new_prefixes_planned=16, reused_prefixes=2, probability_bounds=masses)
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    report.update(outputs={name: branch._file_record(output / name) for name in names}, wall_seconds=time.perf_counter() - started)
    freeze_json(output / "report.json", report)
    return report


def load_prepared(prepared):
    root = Path(prepared)
    if root.name == "plan.json":
        root = root.parent
    receipt = json.loads((root / "report.json").read_text())
    require(receipt["status"] == PREPARED, "SUCCESSFUL_PREPARATION_REQUIRED")
    plan = json.loads(branch._bound_file(root, "plan.json", receipt["outputs"]["plan.json"]).read_text())
    validate_settings(plan["settings"])
    require(plan["bindings"] == code_bindings(), "PREPARED_CODE_OR_NUMPY_CHANGED")
    require(set(plan["artifacts"]) == {"initial_voltage.npy", "original_events.npz", "reversed_events.npz",
                                       "original_pre_threshold.npz", "reversed_pre_threshold.npz"}, "PREPARED_ARTIFACT_SET_CHANGED")
    return plan, {name: branch._bound_file(root, name, record) for name, record in plan["artifacts"].items()}


def _events(path, config):
    with np.load(path, allow_pickle=False) as raw:
        return continuation.clip_events(raw, dict(config, duration_ms=100.), 8000)


def _run_cell(plan, paths, output, x, y, variant):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED", x=x, y=y, variant=variant, **scope())
    started = time.perf_counter()
    try:
        if (x, y) == REALIZED_CELL:
            shutil.copyfile(paths[variant + "_events.npz"], output / "raw.npz")
            report.update(status="REUSED", source_prefix_sha256=sha256_file(paths[variant + "_events.npz"]),
                          new_simulations=0, source_full_episode_cost=plan["source_costs"][variant],
                          reused_cost_is_not_a_100ms_runtime=True)
        else:
            config, case, settings = plan["full_config"], plan["selected_case"], plan["settings"]
            factory = pfc_coupled_drive.create_coupled_drive if variant == "original" else pfc_clock_temporal_control.create_temporal_reversal_drive
            initial, source = factory(config, case, .05, [95411, 95412])
            require(conditional.exact_arrays_equal(initial, np.load(paths["initial_voltage.npy"], allow_pickle=False)), "ARCHIVED_INITIAL_CHANGED")
            require(source.total_ticks == 400000
                    and (source._rows.coarse.probability / source.factor).hex() == plan["probability_law"]["binary64_probability_hex"], "FULL_SOURCE_DRIVE_OR_PROBABILITY_CHANGED")
            expected = {88: 0, 91: 1} if variant == "original" else {88: 1, 91: 0}
            drive = InnovationDrive(source, variant, 6423, x, y, expected_original=expected)
            anchored = (variant == "original" and x == 0) or (variant == "reversed" and y == 1)
            observer = branch.ThresholdObserver(89, config) if anchored else None
            short = deepcopy(config)
            short["duration_ms"] = 100.
            with observer.observing() if observer is not None else nullcontext():
                result = executor.simulate(short, case, initial, backend="numpy", drive=drive)
            np.savez(output / "raw.npz", **result["raw"])
            # The unchanged source's terminal arrays are preserved only for newly computed cells.
            np.savez_compressed(output / "terminal_state.npz", **result["terminal_state"])
            report.update(cost=result["cost"], inputs_used=result["inputs_used"], drive=drive.metadata(), new_simulations=1,
                          source_anchor_applies=anchored)
            if observer is not None:
                reference = continuation._snapshot_arrays(paths[variant + "_pre_threshold.npz"], config)
                snapshot = observer.snapshots["pre_threshold"]
                matches = {name: conditional.exact_arrays_equal(value, reference[name]) for name, value in snapshot.items()}
                np.savez(output / "anchored_pre_threshold.npz", **snapshot)
                report["archived_prethreshold_matches"] = matches
                require(all(matches.values()), "ARCHIVED_REALIZED_STATE_ANCHOR_CHANGED")
            require(drive.ticks == 8000 and len(drive.observations) == 2
                    and {row["input_tick"] for row in drive.observations} == {88, 91}, "COMPLETE_TWO_COORDINATE_PREFIX_REQUIRED")
            report["status"] = "COMPUTED"
        events = _events(output / "raw.npz", plan["full_config"])
        spike = bool(np.any((events[:, 0] == 89) & (events[:, 1] == 6423)))
        expected_spike = x >= 7 if variant == "original" else y >= 1
        report.update(event_count=len(events), immediate_E6423_spike=spike, immediate_threshold_prediction=expected_spike)
        require(spike == expected_spike, "ACTUAL_COORDINATE_RESPONSE_DIFFERS_FROM_ACCEPTED_THRESHOLD_MAP")
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    report.update(wall_seconds=time.perf_counter() - started,
        outputs={name: branch._file_record(output / name) for name in ("raw.npz", "terminal_state.npz", "anchored_pre_threshold.npz") if (output / name).is_file()})
    freeze_json(output / "report.json", report)
    return report


def count_features(events, config):
    """Complete unit-window counts and integer source-field CDF numerators."""
    ne, ni = config["excitatory_neurons"], config["inhibitory_neurons"]
    units = ne + ni
    fields, sizes = continuation.pfc_fluctuation_analysis.unit_mapping(ne, ni, config["excitatory_preference_bins"])
    counts5 = np.bincount((events[:, 0] // 400) * units + events[:, 1], minlength=20 * units).reshape(20, units)
    grouped, numerators, denominators = [], [], []
    for bins in (1, 10, 20):
        counts = counts5.reshape(-1, bins, units).sum(axis=1, dtype=np.int64)
        grouped.append(counts)
        field_numerators, field_denominators = [], []
        for field, size in enumerate(sizes):
            values = counts[:, fields == field]
            maximum = ((400 * bins - 1) // (160 if field < len(sizes) - 1 else 80)) + 1
            require(int(values.max()) <= maximum, "SOURCE_REFRACTORY_COUNT_SUPPORT_EXCEEDED")
            histogram = np.bincount(values.ravel(), minlength=101)
            field_numerators.append(np.cumsum(histogram, dtype=np.int64)[:101])
            field_denominators.append(len(counts) * int(size))
        numerators.append(field_numerators)
        denominators.append(field_denominators)
    return grouped, np.asarray(numerators, dtype=np.int64), np.asarray(denominators, dtype=np.int64)


def response_features(original, reversed_path, zero_original, zero_reversed):
    """Paired discrepancy and path-specific effects use distinct zero references."""
    pairs = ((original, reversed_path), (zero_original, original), (zero_reversed, reversed_path))
    indicators = np.asarray([np.concatenate([np.any(b != a, axis=1).astype(np.int64)
                                             for a, b in zip(left[0], right[0])]) for left, right in pairs])
    signed = np.stack([right[1] - left[1] for left, right in pairs])
    require(all(np.array_equal(left[2], right[2]) for left, right in pairs), "COMMON_CDF_DENOMINATORS_REQUIRED")
    return indicators, signed, original[2]


def integrate(plan, output, reports):
    config = plan["full_config"]
    def features(x, y, variant):
        return count_features(_events(output / "cells" / ("x_" + str(x) + "_y_" + str(y)) / variant / "raw.npz", config), config)
    zero = {variant: features(0, 0, variant) for variant in VARIANTS}
    indicators, signed, immediate, rows = [], [], [], []
    for x, y in CELLS:
        left, right = (features(x, y, variant) for variant in VARIANTS)
        h, g, denominators = response_features(left, right, zero["original"], zero["reversed"])
        indicators.append(h)
        signed.append(g)
        a, b = (reports[(x, y, variant)]["immediate_E6423_spike"] for variant in VARIANTS)
        xor = int(a != b)
        require(xor == int((x >= 7) != (y >= 1)), "IMMEDIATE_XOR_CROSSCHECK_FAILED")
        immediate.append(xor)
        rows.append(dict(x=x, y=y, original_immediate_spike=a, reversed_immediate_spike=b, immediate_xor=bool(xor),
                         paired_terminal_any_count_difference=bool(h[0, -1]),
                         original_terminal_change_from_00=bool(h[1, -1]),
                         reversed_terminal_change_from_00=bool(h[2, -1])))
    indicators, signed = np.asarray(indicators, dtype=np.int64), np.asarray(signed, dtype=np.int64)
    masses = plan["probability_bounds"]
    indicator_bounds = weighted_bounds(indicators, np.asarray(1), masses, signed=False)
    cdf_bounds = weighted_bounds(signed, denominators[None, :, :, None], masses, signed=True)
    xor_bounds = weighted_bounds(np.asarray(immediate, dtype=np.int64), np.asarray(1), masses, signed=False)
    require(int(xor_bounds["lower_scaled"].item()) <= int(plan["immediate_kernel_xor"]["lower_scaled"])
            and int(xor_bounds["upper_scaled"].item()) >= int(plan["immediate_kernel_xor"]["upper_scaled"]),
            "WEIGHTED_IMMEDIATE_XOR_DOES_NOT_ENCLOSE_ACCEPTED_KERNEL")
    np.savez_compressed(output / "integer_response_table.npz", cells=np.asarray(CELLS),
        any_count_difference_indicators=indicators, signed_cdf_numerators=signed, cdf_denominators=denominators,
        cdf_count_thresholds=np.arange(101), immediate_E6423_xor=np.asarray(immediate),
        comparison_names=np.asarray(["paired_reversed_minus_original", "original_minus_original_00", "reversed_minus_reversed_00"]),
        windows_ms=np.asarray([5., 50., 100.]), indicator_window_width_ms=np.asarray([5.] * 20 + [50.] * 2 + [100.]),
        indicator_start_ms=np.asarray(list(np.arange(20) * 5.) + [0., 50., 0.]))
    np.savez_compressed(output / "directed_response_bounds.npz", scale=np.asarray(masses["scale"]),
        **{"count_indicator_" + key: value for key, value in indicator_bounds.items()},
        **{"signed_cdf_" + key: value for key, value in cdf_bounds.items()},
        **{"immediate_xor_" + key: value for key, value in xor_bounds.items()})
    summary = dict(cells=rows, probability_bounds=masses,
        immediate_E6423_xor={key: str(value.item()) for key, value in xor_bounds.items()},
        accepted_immediate_kernel_xor=plan["immediate_kernel_xor"],
        terminal_any_count_difference={name: {key: str(value[index, -1]) for key, value in indicator_bounds.items()}
            for index, name in enumerate(("paired", "original_vs_00", "reversed_vs_00"))},
        cdf_method="exact integer CDF numerator difference divided by common field-size times complete-window denominator; directed probability weights",
        signed_omitted_outcome_interval=[-1, 1], indicator_omitted_outcome_interval=[0, 1],
        all_omitted_mass_retained=True, renormalized=False,
        conditioning=CONDITIONING, conditioning_excludes=["outputs", "coarse_parents", "outcome_selection", "PRNG_state", "seed"],
        interpretation="Pointwise ideal-law expectation over two free input coordinates on one fixed remaining input path; not a bound over other histories or selected events.")
    freeze_json(output / "integration.json", summary)
    return summary


def execute(prepared, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status=FAILED, stage="execute", cells=[], **scope())
    started = time.perf_counter()
    try:
        plan, paths = load_prepared(prepared)
        reports = {}
        for x, y in CELLS:
            for variant in VARIANTS:
                directory = output / "cells" / ("x_" + str(x) + "_y_" + str(y)) / variant
                cell = _run_cell(plan, paths, directory, x, y, variant)
                reports[(x, y, variant)] = cell
                report["cells"].append(dict(x=x, y=y, variant=variant, status=cell["status"],
                    report_path=str((directory / "report.json").relative_to(output)),
                    report_sha256=sha256_file(directory / "report.json")))
                print(json.dumps(report["cells"][-1]), flush=True)
        report.update(computed_prefixes=sum(row["status"] == "COMPUTED" for row in report["cells"]),
                      reused_prefixes=sum(row["status"] == "REUSED" for row in report["cells"]),
                      failed_paths=[row for row in report["cells"] if row["status"] == "FAILED"])
        require(report["computed_prefixes"] == 16 and report["reused_prefixes"] == 2, "ALL_18_PATH_OUTPUTS_REQUIRED")
        summary = integrate(plan, output, reports)
        report.update(status=COMPLETE, integration_summary=summary,
                      outputs={name: branch._file_record(output / name) for name in
                               ("integer_response_table.npz", "directed_response_bounds.npz", "integration.json")})
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    report["wall_seconds"] = time.perf_counter() - started
    freeze_json(output / "report.json", report)
    return report



def integrate_saved(archive, delivery_digest, output):
    """Integrate completed cell outputs, including a run whose integration failed."""
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status=FAILED, stage="integrate", new_source_simulations=0, **scope())
    try:
        require(sha256_file(archive / "delivery.json") == delivery_digest, "EXECUTED_DELIVERY_BINDING_CHANGED")
        delivery = json.loads((archive / "delivery.json").read_text())
        require(delivery["verified"] and delivery["status"] in
                ("FAILED_PFC_INNOVATION_EXECUTE", COMPLETE), "EXECUTED_SOURCE_ARCHIVE_REQUIRED")
        manifest = {row["path"]: row for row in delivery["files"]}
        def bound(name):
            return branch._bound_file(archive, name, manifest[name])
        prior = json.loads(bound("execution/report.json").read_text())
        plan = json.loads(bound("prepared/plan.json").read_text())
        prepared = json.loads(bound("prepared/report.json").read_text())
        require(prepared["status"] == PREPARED and prepared["outputs"]["plan.json"]["sha256"]
                == manifest["prepared/plan.json"]["sha256"], "BOUND_PREPARATION_REQUIRED")
        validate_settings(plan["settings"])
        current = code_bindings()
        require(all(current[key] == plan["bindings"][key] for key in ("source", "helpers", "numpy_version"))
                and sha256_file(bound("Libs/population_dynamics/pfc_innovation_response.py"))
                == plan["bindings"]["implementation"], "EXECUTED_SOURCE_BINDINGS_CHANGED")
        expected = {(x, y, variant) for x, y in CELLS for variant in VARIANTS}
        require(prior["computed_prefixes"] == 16 and prior["reused_prefixes"] == 2
                and prior["failed_paths"] == [] and len(prior["cells"]) == 18
                and {(row["x"], row["y"], row["variant"]) for row in prior["cells"]} == expected,
                "ALL_SOURCE_CELLS_COMPLETED_REQUIRED")
        reports = {}
        for row in prior["cells"]:
            x, y, variant = row["x"], row["y"], row["variant"]
            relative = Path("cells") / ("x_" + str(x) + "_y_" + str(y)) / variant
            report_name = "execution/" + str(relative / "report.json")
            require(row["report_path"] == str(relative / "report.json")
                    and row["report_sha256"] == manifest[report_name]["sha256"], "CELL_REPORT_BINDING_CHANGED")
            cell_path = bound(report_name)
            cell = json.loads(cell_path.read_text())
            require((cell["x"], cell["y"], cell["variant"]) == (x, y, variant)
                    and cell["status"] == ("REUSED" if (x, y) == REALIZED_CELL else "COMPUTED"),
                    "COMPLETE_CELL_IDENTITY_REQUIRED")
            raw_name = "execution/" + str(relative / "raw.npz")
            require(cell["outputs"]["raw.npz"]["sha256"] == manifest[raw_name]["sha256"], "CELL_EVENTS_BINDING_CHANGED")
            target = output / relative
            target.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(bound(raw_name), target / "raw.npz")
            shutil.copyfile(cell_path, target / "report.json")
            reports[(x, y, variant)] = cell
        summary = integrate(plan, output, reports)
        report.update(status="COMPLETE_PFC_INNOVATION_RESPONSE_INTEGRATION", integration_summary=summary,
            reused_completed_prefixes=18, original_computed_prefixes=16, original_reused_prefixes=2,
            source_archive=str(archive), source_delivery_sha256=delivery_digest,
            source_scheduler_failure_preserved=delivery["status"] != COMPLETE,
            previous_integration_error=prior.get("error"),
            previous_implementation_sha256=plan["bindings"]["implementation"],
            integration_implementation_sha256=sha256_file(Path(__file__)),
            source_costs=[dict(x=x, y=y, variant=v, cost=cell.get("cost"),
                               reused_full_episode_cost=cell.get("source_full_episode_cost"))
                          for (x, y, v), cell in reports.items()],
            outputs={name: branch._file_record(output / name) for name in
                     ("integer_response_table.npz", "directed_response_bounds.npz", "integration.json")})
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    report["wall_seconds"] = time.perf_counter() - started
    freeze_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "execute", "integrate"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--delivery-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.config is None or args.project_root is None or args.prepared is not None:
            parser.error("prepare requires --config and --project-root")
        report = prepare(args.config, args.project_root, args.output)
    elif args.stage == "execute":
        if args.prepared is None or args.config is not None or args.project_root is not None:
            parser.error("execute requires --prepared")
        report = execute(args.prepared, args.output)
    else:
        if args.archive is None or args.delivery_sha256 is None or any(
                value is not None for value in (args.config, args.project_root, args.prepared)):
            parser.error("integrate requires --archive and --delivery-sha256")
        report = integrate_saved(args.archive, args.delivery_sha256, args.output)
    print(json.dumps({key: report[key] for key in ("status", "error") if key in report}, indent=2))
    return 0 if report["status"] in (PREPARED, COMPLETE, "COMPLETE_PFC_INNOVATION_RESPONSE_INTEGRATION") else 1


if __name__ == "__main__":
    raise SystemExit(main())
