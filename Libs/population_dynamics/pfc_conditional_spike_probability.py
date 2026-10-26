"""One-step native spike kernel at an outcome-selected historical PFC state."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from . import pfc_threshold_branch_probe as branch
from . import pfc_coupled_drive, pfc_clock_temporal_control
from . import pfc_deterministic_replay as executor
from .pfc_fluctuation_analysis import unit_mapping
from .provenance import freeze_json, sha256_file

DECIMAL_DIGITS = 40


def require(condition, message):
    if not condition:
        raise ValueError("PFC_CONDITIONAL_SPIKE_" + message)


def binomial_tail_bounds(n, numerator, denominator, digits=DECIMAL_DIGITS):
    """Directed integer bounds for an explicitly specified ideal binomial law."""
    require(type(n) is int and n >= 0 and type(numerator) is int
            and type(denominator) is int and 0 < numerator < denominator
            and type(digits) is int and digits > 0, "BINOMIAL_PARAMETERS")
    common = denominator ** n
    mass = (denominator - numerator) ** n
    masses = [mass]
    for k in range(n):
        mass, remainder = divmod(mass * (n - k) * numerator,
                                 (k + 1) * (denominator - numerator))
        require(remainder == 0, "EXACT_MASS_RECURRENCE")
        masses.append(mass)
    require(sum(masses) == common, "EXACT_TOTAL_PROBABILITY")
    scale, tail = 10 ** digits, 0
    rows = [None] * (n + 2)
    rows[n + 1] = dict(minimum_events=n + 1, lower_scaled="0", upper_scaled="0")
    for k in range(n, -1, -1):
        tail += masses[k]
        lower, remainder = divmod(tail * scale, common)
        rows[k] = dict(minimum_events=k, lower_scaled=str(lower),
                       upper_scaled=str(lower + int(remainder != 0)))
    require(rows[0]["lower_scaled"] == rows[0]["upper_scaled"] == str(scale),
            "EXACT_FULL_TAIL")
    return rows


def reconstruct_ampa(config, case, factory, coarse_dt_ms, seeds, tick):
    """Rebuild only input/AMPA history; do not rerun membrane or network dynamics."""
    require(type(tick) is int and tick > 0, "POSITIVE_TARGET_TICK")
    initial, drive = factory(config, case, coarse_dt_ms, seeds)
    require(tick <= drive.max_rows, "BOUNDED_INPUT_PREFIX")
    rows = drive.take(tick)
    decay = -(config["dt_ms"] * 1e-3) / executor._constants(config, "E")["tau_ampa"]
    before, actual, combined = {}, {}, {}
    for population, size in (("E", config["excitatory_neurons"]), ("I", config["inhibitory_neurons"])):
        a = np.zeros(size, dtype=np.float64)
        for counts in rows[population]:
            midpoint = (.5 * (decay * a)) + a
            base = (decay * midpoint) + a
            a = base + counts.astype(np.float64)
        before[population], actual[population], combined[population] = base, counts.copy(), a
    probability = drive._rows.coarse.probability / drive.factor
    return initial, before, actual, combined, probability, drive.metadata()


def exact_arrays_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def threshold_map(pre, post, bases, actual, config, tick):
    """Enumerate all native input counts with the unchanged source RK2 routine."""
    arithmetic, result, agreements = executor._Arithmetic("numpy"), {}, {}
    dt = config["dt_ms"] * 1e-3
    for population, size in (("E", config["excitatory_neurons"]), ("I", config["inhibitory_neurons"])):
        keys = branch.STATE_FIELDS + (branch.E_FIELDS if population == "E" else ())
        original = {key: pre[population + "__" + key] for key in keys}
        constants = executor._constants(config, population)
        first = np.full(size, 1001, dtype=np.int32)
        previous = np.zeros(size, dtype=bool)
        reconstructed = {key: np.empty_like(value) for key, value in original.items()}
        covered = np.zeros(size, dtype=bool)
        for count in range(1001):
            state = {key: value.copy() for key, value in original.items()}
            state["s_AMPA"] = bases[population] + float(count)
            executor._rk2(state, constants, tick * dt, dt, arithmetic, population)
            selected = (state["v"] > constants["threshold"]) & state["not_refractory"]
            require(not np.any(previous & ~selected), "NONMONOTONE_SPIKE_DECISION:" + population + ":" + str(count))
            first[(first == 1001) & selected] = count
            previous = selected
            mask = actual[population] == count
            covered |= mask
            for key in keys:
                reconstructed[key][mask] = state[key][mask]
        require(covered.all(), "ACTUAL_INPUT_OUTSIDE_ENUMERATED_SUPPORT")
        for key, value in reconstructed.items():
            name = population + "__" + key
            agreements[name] = exact_arrays_equal(value, post[name])
        require(all(agreements.values()), "OBSERVED_PRETHRESHOLD_TRANSITION_MISMATCH")
        result[population] = first
    return result, agreements


def summarize_thresholds(thresholds, fields, sizes, bounds, digits=DECIMAL_DIGITS):
    scale = 10 ** digits
    rows = []
    for field in range(len(sizes)):
        values = thresholds[fields == field]
        histogram = np.bincount(values, minlength=len(bounds))
        lower = sum(int(n) * int(bounds[k]["lower_scaled"]) for k, n in enumerate(histogram))
        upper = sum(int(n) * int(bounds[k]["upper_scaled"]) for k, n in enumerate(histogram))
        rows.append(dict(field=field, units=int(sizes[field]), always_spikes=int(histogram[0]),
            never_spikes=int(histogram[-1]), expected_spikes_lower_scaled=str(lower),
            expected_spikes_upper_scaled=str(upper), any_spike_union_upper_scaled=str(min(scale, upper))))
    total_lower = sum(int(row["expected_spikes_lower_scaled"]) for row in rows)
    total_upper = sum(int(row["expected_spikes_upper_scaled"]) for row in rows)
    return rows, dict(expected_spikes_lower_scaled=str(total_lower),
        expected_spikes_upper_scaled=str(total_upper), any_spike_union_upper_scaled=str(min(scale, total_upper)))


def analyze(archive, delivery_digest, output):
    output.mkdir(parents=True, exist_ok=False)
    started, names = time.perf_counter(), []
    report = dict(schema="pfc_conditional_spike_kernel_v1", status="FAILED_PFC_CONDITIONAL_SPIKE_KERNEL",
        one_step_kernel_evaluation_complete=False, source_simulations=0,
        full_network_simulations=0, new_full_episodes=0, new_independent_blocks=0,
        probability_kernel_evaluated_at_outcome_selected_history=True,
        selected_event_probability_established=False, paired_disagreement_probability_established=False,
        fixed_seed_probability_established=False, full_trajectory_bound_established=False,
        scientific_fidelity_qualified=False, confirmation_execution_released=False, production_episodes=0)
    try:
        require(sha256_file(archive / "delivery.json") == delivery_digest, "ACCEPTED_DELIVERY_BINDING")
        delivery = json.loads((archive / "delivery.json").read_text())
        require(delivery["verified"] and delivery["status"] == "COMPLETE_PFC_THRESHOLD_BRANCH_TWO_PATH_WITNESS",
                "ACCEPTED_THRESHOLD_WITNESS")
        manifest = {row["path"]: row for row in delivery["files"]}
        input_names = ["execution/report.json", "prepared/plan.json", "prepared/initial_voltage.npy"]
        input_names += ["execution/" + variant + "_" + stage + ".npz"
                        for variant in ("original", "reversed") for stage in ("pre_rk2", "pre_threshold")]
        paths = {name: branch._bound_file(archive, name, manifest[name]) for name in input_names}
        plan = json.loads(paths["prepared/plan.json"].read_text())
        accepted = json.loads(paths["execution/report.json"].read_text())
        require(plan["current_code"] == branch._code_bindings()
                and plan["probe_sha256"] == sha256_file(Path(branch.__file__)), "SOURCE_CODE_BINDING")
        config, case, settings = plan["full_config"], plan["selected_case"], plan["settings"]
        tick = plan["first_mismatch"]["tick"]
        require(tick == 89 and config["dt_ms"] == .0125 and config["duration_ms"] == 5000.
                and config["excitatory_neurons"] == 80000 and config["inhibitory_neurons"] == 20000
                and accepted["status"] == branch.PASSED, "FROZEN_WITNESS_SCOPE")
        initial = np.load(paths["prepared/initial_voltage.npy"], allow_pickle=False)
        fields, sizes = unit_mapping(config["excitatory_neurons"], config["inhibitory_neurons"],
                                     config["excitatory_preference_bins"])
        variants, bounds, probability_ratio = {}, None, None
        report["variants"] = variants
        saved = dict(archive_unit_indices=np.arange(len(initial), dtype=np.int64), source_field_indices=fields)
        for variant, factory in (("original", pfc_coupled_drive.create_coupled_drive),
                                  ("reversed", pfc_clock_temporal_control.create_temporal_reversal_drive)):
            pre = load_snapshot(paths["execution/" + variant + "_pre_rk2.npz"])
            post = load_snapshot(paths["execution/" + variant + "_pre_threshold.npz"])
            generated, bases, actual, combined, probability, metadata = reconstruct_ampa(
                config, case, factory, settings["coarse_dt_ms"], settings["refinement_seeds"], tick)
            require(exact_arrays_equal(generated, initial), "SOURCE_INITIAL_STATE")
            require(all(exact_arrays_equal(combined[p], pre[p + "__s_AMPA"]) for p in ("E", "I")),
                    "EXACT_PREINPUT_AMPA_RECONSTRUCTION")
            ratio = probability.as_integer_ratio()
            require(probability == float(7 / 400000), "NATIVE_FINE_PROBABILITY")
            if bounds is None:
                probability_ratio = ratio
                bounds = binomial_tail_bounds(1000, *ratio)
            require(ratio == probability_ratio, "PATH_PROBABILITY_MISMATCH")
            mapping, agreements = threshold_map(pre, post, bases, actual, config, tick)
            joined = np.concatenate((mapping["E"], mapping["I"]))
            field_rows, totals = summarize_thresholds(joined, fields, sizes, bounds)
            for population in ("E", "I"):
                saved[variant + "_" + population + "_preinput_ampa"] = bases[population]
                saved[variant + "_" + population + "_actual_input"] = actual[population]
            saved[variant + "_minimum_input_for_spike"] = joined
            variants[variant] = dict(prethreshold_state_bitwise_match=agreements,
                ampa_preinput_history_exact=True, full_input_support_enumerated=True,
                all_spike_decisions_monotone=True, source_input_rows_reconstructed=tick,
                source_input_values_reconstructed=tick * len(initial),
                field_bounds=field_rows, global_bounds=totals,
                focus_unit_6423=dict(minimum_input_for_spike=int(joined[6423]),
                    actual_input=int(actual["E"][6423]), probability_bounds=bounds[int(joined[6423])]),
                drive_metadata=metadata)
        np.savez_compressed(output / "conditional_thresholds.npz", **saved)
        names.append("conditional_thresholds.npz")
        with (output / "binomial_tail_bounds.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["minimum_events", "lower_scaled", "upper_scaled"])
            writer.writeheader()
            writer.writerows(bounds)
        names.append("binomial_tail_bounds.csv")
        report.update(status="COMPLETE_PFC_CONDITIONAL_SPIKE_KERNEL", variants=variants,
            one_step_kernel_evaluation_complete=True, neuron_transition_evaluations=2 * 1001 * len(initial),
            tick=tick, input_tick=tick - 1, dt_ms=config["dt_ms"], decimal_scale_digits=DECIMAL_DIGITS,
            input_law=dict(n=1000, binary64_probability_hex=probability.hex(),
                exact_numerator=str(probability_ratio[0]), exact_denominator=str(probability_ratio[1]),
                nominal_probability="7/400000",
                interpretation="ideal iid Binomial law with the recorded binary64 probability; finite PRNG distribution and floating sampler implementation are not certified by the integer probability bounds"),
            conditioning="Each fixed fine-clock path's own complete initial state, controls, and all-unit inputs through tick87. Excludes X88, complete observed AMPA89, coarse parents, prefetched inputs, RNG/seed, cap event G, other-path history, and the event that this witness was selected.",
            selection_scope="The kernel is evaluated pointwise at an outcome-selected historical state; it is not the selection-conditioned probability of the first divergence.",
            source_bindings={name: manifest[name] for name in input_names})
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    report.update(wall_seconds=time.perf_counter() - started,
                  outputs={name: branch._file_record(output / name) for name in names})
    freeze_json(output / "report.json", report)
    return report



def reversal_joint_information(input_tick, factor):
    """Which current inputs are fresh under the two-path chronological past?"""
    require(type(input_tick) is int and input_tick >= 0 and type(factor) is int
            and factor in (2, 4), "REVERSAL_CLOCK")
    residue = input_tick % factor
    revealed = sorted(set(range(residue)) | set(range(factor - residue, factor)))
    current = [residue, factor - 1 - residue]
    fresh = all(index not in revealed for index in current)
    return dict(refinement_factor=factor, input_residue=residue, revealed_coordinates=revealed,
        current_coordinates=current, both_current_inputs_fresh=fresh,
        individual_history_sigma_fields_equal=residue == 0,
        probability_conditioning="joint chronological fine-input past; excludes coarse parents, prefetched values, RNG/seed and outcome selection")


def independent_pair_bounds(a_lower, a_upper, b_lower, b_upper, scale):
    require(type(scale) is int and scale > 0
            and all(type(x) is int for x in (a_lower, a_upper, b_lower, b_upper))
            and 0 <= a_lower <= a_upper <= scale and 0 <= b_lower <= b_upper <= scale,
            "PROBABILITY_INTERVAL")
    corners = [(a, b) for a in (a_lower, a_upper) for b in (b_lower, b_upper)]
    numerators = dict(
        neither=[(scale - a) * (scale - b) for a, b in corners],
        original_only=[a * (scale - b) for a, b in corners],
        reversed_only=[(scale - a) * b for a, b in corners],
        both=[a * b for a, b in corners],
        disagreement=[scale * (a + b) - 2 * a * b for a, b in corners])
    result = {}
    for name, values in numerators.items():
        lo = min(values) // scale
        hi, remainder = divmod(max(values), scale)
        result[name + "_lower_scaled"] = str(lo)
        result[name + "_upper_scaled"] = str(hi + int(remainder != 0))
    result["signed_mean_reversed_minus_original_lower_scaled"] = str(b_lower - a_upper)
    result["signed_mean_reversed_minus_original_upper_scaled"] = str(b_upper - a_lower)
    return result


def analyze_pair(archive, delivery_digest, output):
    """Reuse accepted one-step thresholds; no transition enumeration or RNG draws."""
    output.mkdir(parents=True, exist_ok=False)
    started, names = time.perf_counter(), []
    report = dict(schema="pfc_paired_spike_kernel_v1", status="FAILED_PFC_PAIRED_SPIKE_KERNEL",
        full_network_simulations=0, source_simulations=0, neuron_transition_evaluations=0,
        new_rng_draws=0, new_full_episodes=0, new_independent_blocks=0,
        selected_event_probability_established=False, full_trajectory_bound_established=False,
        scientific_fidelity_qualified=False, confirmation_execution_released=False, production_episodes=0)
    try:
        require(sha256_file(archive / "delivery.json") == delivery_digest, "KERNEL_DELIVERY_BINDING")
        delivery = json.loads((archive / "delivery.json").read_text())
        require(delivery["verified"] and delivery["status"] == "COMPLETE_PFC_CONDITIONAL_SPIKE_KERNEL",
                "ACCEPTED_ONE_STEP_KERNEL")
        manifest = {row["path"]: row for row in delivery["files"]}
        inputs = {name: branch._bound_file(archive, name, manifest[name]) for name in
                  ("analysis/report.json", "analysis/conditional_thresholds.npz", "analysis/binomial_tail_bounds.csv")}
        parent = json.loads(inputs["analysis/report.json"].read_text())
        require(parent["one_step_kernel_evaluation_complete"], "COMPLETE_ONE_STEP_KERNEL_REQUIRED")
        original, reversed_path = (parent["variants"][name] for name in ("original", "reversed"))
        factor = original["drive_metadata"]["refinement_factor"]
        require(reversed_path["drive_metadata"]["refinement_factor"] == factor, "PAIRED_REFINEMENT_CLOCK")
        transform = reversed_path["drive_metadata"]["coupling"]["input_order_control"]
        expected_order = list(range(factor - 1, -1, -1))
        require(transform["name"] == "within_coarse_tick_tree_temporal_reversal"
                and transform["fine_row_order"] == expected_order, "FROZEN_REVERSAL_REQUIRED")
        information = reversal_joint_information(parent["input_tick"], factor)
        report["joint_information"] = information
        require(information["both_current_inputs_fresh"], "JOINT_HISTORY_REVEALS_CURRENT_INPUT")
        scale = 10 ** parent["decimal_scale_digits"]
        with inputs["analysis/binomial_tail_bounds.csv"].open(newline="") as stream:
            table = list(csv.DictReader(stream))
        require(len(table) == 1002 and all(int(row["minimum_events"]) == index
                and 0 <= int(row["lower_scaled"]) <= int(row["upper_scaled"]) <= scale
                for index, row in enumerate(table)), "FULL_DIRECTED_TAIL_TABLE")
        with np.load(inputs["analysis/conditional_thresholds.npz"], allow_pickle=False) as data:
            units, fields = data["archive_unit_indices"], data["source_field_indices"]
            a, b = data["original_minimum_input_for_spike"], data["reversed_minimum_input_for_spike"]
            actual_a = np.concatenate((data["original_E_actual_input"], data["original_I_actual_input"]))
            actual_b = np.concatenate((data["reversed_E_actual_input"], data["reversed_I_actual_input"]))
        require(np.array_equal(units, np.arange(100000)) and a.shape == b.shape == fields.shape == (100000,)
                and a.dtype.kind in "iu" and b.dtype.kind in "iu" and np.all(a <= 1001) and np.all(b <= 1001),
                "FULL_UNIT_THRESHOLD_MAPS")
        pairs, assignment, multiplicities = np.unique(np.column_stack((a, b)), axis=0, return_inverse=True, return_counts=True)
        rows = []
        for group, ((ka, kb), count) in enumerate(zip(pairs, multiplicities)):
            left, right = table[int(ka)], table[int(kb)]
            row = dict(group=group, original_minimum_input=int(ka), reversed_minimum_input=int(kb), units=int(count))
            row.update(independent_pair_bounds(int(left["lower_scaled"]), int(left["upper_scaled"]),
                                               int(right["lower_scaled"]), int(right["upper_scaled"]), scale))
            rows.append(row)
        with (output / "paired_threshold_groups.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        names.append("paired_threshold_groups.csv")
        np.savez_compressed(output / "paired_unit_groups.npz", archive_unit_indices=units,
                           source_field_indices=fields, threshold_group=assignment)
        names.append("paired_unit_groups.npz")
        field_rows = []
        for field in range(17):
            mask = fields == field
            frequencies = np.bincount(assignment[mask], minlength=len(rows))
            lower = sum(int(count) * int(row["disagreement_lower_scaled"]) for count, row in zip(frequencies, rows))
            upper = sum(int(count) * int(row["disagreement_upper_scaled"]) for count, row in zip(frequencies, rows))
            field_rows.append(dict(field=field, units=int(mask.sum()),
                expected_disagreeing_units_lower_scaled=str(lower), expected_disagreeing_units_upper_scaled=str(upper),
                any_disagreement_union_upper_scaled=str(min(scale, upper))))
        total_lower = sum(int(row["expected_disagreeing_units_lower_scaled"]) for row in field_rows)
        total_upper = sum(int(row["expected_disagreeing_units_upper_scaled"]) for row in field_rows)
        observed = np.flatnonzero((actual_a >= a) != (actual_b >= b)).tolist()
        require(observed == [6423], "OBSERVED_PAIRED_BRANCH_BINDING")
        report.update(status="COMPLETE_PFC_PAIRED_SPIKE_KERNEL", conditional_paired_kernel_complete=True,
            one_step_kernel_archive=str(archive), one_step_kernel_delivery_sha256=delivery_digest,
            source_bindings={name: manifest[name] for name in inputs},
            input_law=parent["input_law"], input_tick=parent["input_tick"], target_tick=parent["tick"],
            decimal_scale_digits=parent["decimal_scale_digits"], unique_threshold_pairs=len(rows),
            focus_unit_6423=rows[int(assignment[6423])], field_bounds=field_rows,
            global_bounds=dict(expected_disagreeing_units_lower_scaled=str(total_lower),
                expected_disagreeing_units_upper_scaled=str(total_upper),
                any_disagreement_union_upper_scaled=str(min(scale, total_upper))),
            observed_disagreeing_unit_indices=observed,
            selection_scope="Pointwise joint-history kernel at an outcome-selected record; does not condition on being selected as the first divergence.",
            interval_semantics="Separate directed joint-outcome intervals need not sum to one; no renormalization.",
            temporal_scope="Fresh within-parent input coordinates only; later reversed coordinates already revealed in the opposite history require a different conditional kernel.")
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    report.update(wall_seconds=time.perf_counter() - started,
                  outputs={name: branch._file_record(output / name) for name in names})
    freeze_json(output / "report.json", report)
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--archive", type=Path)
    inputs.add_argument("--kernel-archive", type=Path)
    parser.add_argument("--delivery-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = (analyze(args.archive, args.delivery_sha256, args.output) if args.archive is not None
              else analyze_pair(args.kernel_archive, args.delivery_sha256, args.output))
    print(json.dumps({key: report[key] for key in ("status", "error") if key in report}, indent=2))
    return 0 if report["status"] in ("COMPLETE_PFC_CONDITIONAL_SPIKE_KERNEL", "COMPLETE_PFC_PAIRED_SPIKE_KERNEL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
