"""Exact ideal clock-tree probabilities conditional on a coarse input count."""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import product
import csv
import json
from pathlib import Path
import time

import numpy as np

from . import pfc_coupled_drive
from . import pfc_threshold_branch_probe as branch
from .provenance import freeze_json, sha256_file


def require(ok, message):
    if not ok:
        raise ValueError("PFC_CLOCK_TREE_" + message)


def categories(p):
    require(isinstance(p, Fraction) and 0 < p <= 1, "RATIONAL_PROBABILITY_REQUIRED")
    return {(0, 0): p / 4, (1, 0): Fraction(1, 2) - p / 4,
            (0, 1): Fraction(1, 2) - p / 4, (1, 1): p / 4}


def joint_tree(parent_count, p):
    """Coefficients of [G_p(U G_(p/2)(z0,z1), V G_(p/2)(z2,z3))]^c."""
    require(type(parent_count) is int and 0 <= parent_count <= 8, "BOUNDED_PARENT_ENUMERATION_REQUIRED")
    atom = {}
    zero = {(0, 0): Fraction(1)}
    for middle, weight in categories(p).items():
        left = categories(p / 2) if middle[0] else zero
        right = categories(p / 2) if middle[1] else zero
        for a, wa in left.items():
            for b, wb in right.items():
                atom[middle + a + b] = weight * wa * wb
    require(len(atom) == 25 and sum(atom.values()) == 1, "SINGLE_PARENT_JOINT_MASS")
    law = {(0,) * 6: Fraction(1)}
    for _ in range(parent_count):
        updated = defaultdict(Fraction)
        for state, weight in law.items():
            for child, probability in atom.items():
                updated[tuple(a + b for a, b in zip(state, child))] += weight * probability
        law = dict(updated)
    require(sum(law.values()) == 1, "JOINT_MASS")
    return law


def fine_projection(law):
    fine = defaultdict(Fraction)
    for state, probability in law.items():
        fine[state[2:]] += probability
    return dict(fine)


def single_fine_closed_form(p):
    law = {}
    for bits in product((0, 1), repeat=4):
        k = sum(bits)
        probability = (p / 4) ** k * (1 - p / 4) ** (4 - k)
        law[bits] = (probability - (1 - p) * int(k == 0)) / p
    return law


def fraction_record(value):
    return dict(numerator=str(value.numerator), denominator=str(value.denominator))


def directed(value, digits=40):
    scale = 10 ** digits
    lower, remainder = divmod(value.numerator * scale, value.denominator)
    return dict(lower_scaled=str(lower), upper_scaled=str(lower + int(remainder != 0)))



def response_slice(projected, observed_inner, cells, indicators):
    """Condition additionally on unchanged inner fine coordinates; use saved responses."""
    selected = {state: value for state, value in projected.items() if state[1:3] == tuple(observed_inner)}
    mass = sum(selected.values())
    require(mass > 0, "POSITIVE_CONDITIONAL_SLICE_REQUIRED")
    mapping = {tuple(int(v) for v in row): i for i, row in enumerate(cells)}
    require(len(mapping) == len(cells), "UNIQUE_RESPONSE_CELLS_REQUIRED")
    uncovered, known_change = Fraction(0), Fraction(0)
    covered = []
    for state, value in selected.items():
        key = (state[0], state[3])
        if key not in mapping:
            uncovered += value
        else:
            index = mapping[key]
            covered.append(list(state))
            if np.any(indicators[index, 1:, :] != 0):
                known_change += value
    return dict(conditional_slice_mass=fraction_record(mass), covered_fine_states=sorted(covered),
        complete_slice_response_coverage=uncovered == 0,
        uncomputed_conditional_mass=fraction_record(uncovered / mass),
        any_own00_count_change_lower=fraction_record(known_change / mass),
        any_own00_count_change_upper=fraction_record((known_change + uncovered) / mass),
        pointwise_count_invariant_on_complete_slice=uncovered == known_change == 0,
        conditioning="complete coarse inputs, all other fine coordinates including observed inner coordinates, initial state and controls; not outputs or selection",
        scope="Both fine-clock original/reversal paths relative to their own00, all registered100ms-prefix unit-window counts. Middle-clock responses are not evaluated here.")


def rounded_parameter_difference(p):
    # A comparison of two ideal Binomial-draw models, not a finite-PRNG theorem.
    pf = float(p)
    a = Fraction.from_float(pf / 4)
    b = Fraction.from_float((pf / 4) / (1 - pf / 4))
    delta = (1 - a) * b - p / 4
    require(a == p / 4, "POWER_OF_TWO_PROBABILITY_SCALE_CHANGED")
    return dict(rounded_second_parameter_hex=float(b).hex(),
                double_category_signed_difference=fraction_record(delta),
                single_split_total_variation=abs(delta))


def bound_inputs(root, digest, names):
    require(sha256_file(root / "delivery.json") == digest, "DELIVERY_BINDING_CHANGED")
    delivery = json.loads((root / "delivery.json").read_text())
    require(delivery["verified"], "VERIFIED_DELIVERY_REQUIRED")
    manifest = {row["path"]: row for row in delivery["files"]}
    return {name: branch._bound_file(root, name, manifest[name]) for name in names}, delivery


def run(preparation, preparation_digest, kernel, kernel_digest, response, response_digest, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status="FAILED_PFC_CONDITIONAL_CLOCK_TREE", source_simulations=0,
        new_full_episodes=0, new_independent_blocks=0, new_auxiliary_streams=0,
        computational_rng_prefix_replayed=True, neural_response_bound_established=False,
        all_background_histories_qualified=False, confirmation_execution_released=False,
        production_episodes=0, scientific_fidelity_qualified=False)
    try:
        paths, delivery = bound_inputs(preparation, preparation_digest,
            ("prepared/plan.json", "prepared/initial_voltage.npy"))
        require(delivery["status"] == "COMPLETE_PFC_INNOVATION_PREPARATION", "ACCEPTED_PREPARATION_REQUIRED")
        inputs, source_delivery = bound_inputs(kernel, kernel_digest,
            ("analysis/report.json", "analysis/conditional_thresholds.npz"))
        require(source_delivery["status"] == "COMPLETE_PFC_CONDITIONAL_SPIKE_KERNEL", "ACCEPTED_KERNEL_REQUIRED")
        plan = json.loads(paths["prepared/plan.json"].read_text())
        prior = json.loads(inputs["analysis/report.json"].read_text())
        require(plan["bindings"]["source"] == branch._code_bindings(), "UNCHANGED_SOURCE_CODE_REQUIRED")
        config, case, settings = plan["full_config"], plan["selected_case"], plan["settings"]
        require(settings["input_ticks"] == [88, 91] and settings["unit_index"] == 6423
                and config["excitatory_neurons"] == 80000 and config["inhibitory_neurons"] == 20000,
                "ACCEPTED_INPUT_COORDINATES_REQUIRED")
        initial, drive = pfc_coupled_drive.create_coupled_drive(config, case,
            settings["coarse_dt_ms"], settings["refinement_seeds"])
        expected_initial = np.load(paths["prepared/initial_voltage.npy"], allow_pickle=False)
        require(initial.dtype == expected_initial.dtype and initial.tobytes() == expected_initial.tobytes(),
                "INITIAL_STATE_CHANGED")
        coarse_tick = settings["input_ticks"][0] // drive.factor
        for _ in range(coarse_tick + 1):
            coarse, middle, fine = drive._rows.take_row()
        packed_unit = config["inhibitory_neurons"] + settings["unit_index"]
        observed = tuple(int(x) for x in np.concatenate((middle[:, packed_unit], fine[:, packed_unit])))
        c = int(coarse[0, packed_unit])
        with np.load(inputs["analysis/conditional_thresholds.npz"], allow_pickle=False) as saved:
            require(observed[2] == int(saved["original_E_actual_input"][6423])
                    and observed[5] == int(saved["reversed_E_actual_input"][6423]), "ACCEPTED_FINE_COORDINATES_CHANGED")
        p = Fraction.from_float(drive._rows.coarse.probability)
        require(p / 4 == Fraction(int(prior["input_law"]["exact_numerator"]),
                                 int(prior["input_law"]["exact_denominator"])), "ACCEPTED_PROBABILITY_CHANGED")
        law = joint_tree(c, p)
        projected = fine_projection(law)
        require(observed in law and sum(projected.values()) == 1, "REALIZED_TREE_OUTSIDE_SUPPORT")
        for j in range(4):
            mean = sum(state[j] * value for state, value in projected.items())
            require(mean == Fraction(c, 4), "CONDITIONAL_FINE_MEAN")
            for k in range(j):
                covariance = sum(state[j] * state[k] * value for state, value in projected.items()) - mean * mean
                require(covariance == -c * (1 - p) / 16, "CONDITIONAL_FINE_COVARIANCE")
        require(all(projected[state] == projected[tuple(sorted(state))] for state in projected), "FINE_EXCHANGEABILITY")
        for name, table, columns in (("joint_tree.csv", law, ["middle_0", "middle_1", "fine_0", "fine_1", "fine_2", "fine_3"]),
                                      ("fine_projection.csv", projected, ["fine_0", "fine_1", "fine_2", "fine_3"])):
            with (output / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns + ["numerator", "denominator", "lower_scaled", "upper_scaled"])
                writer.writeheader()
                for state, value in sorted(table.items()):
                    writer.writerow(dict(zip(columns, state)) | fraction_record(value) | directed(value))
        original_slice = {state: value for state, value in projected.items()
                          if state[1:3] == observed[3:5] and state[0] <= 2 and state[3] <= 2}
        response_inputs, response_delivery = bound_inputs(response, response_digest,
            ("report.json", "analysis/report.json", "analysis/integer_response_table.npz"))
        require(response_delivery["status"] == "COMPLETE_PFC_INNOVATION_RESPONSE_INTEGRATION", "ACCEPTED_RESPONSE_REQUIRED")
        wrapper = json.loads(response_inputs["report.json"].read_text())
        require(any(row["path"].endswith("/prepared/plan.json") and row["sha256"] == sha256_file(paths["prepared/plan.json"])
                    for row in wrapper["inputs"]), "RESPONSE_PREPARATION_LINEAGE_CHANGED")
        with np.load(response_inputs["analysis/integer_response_table.npz"], allow_pickle=False) as data:
            require(data["any_count_difference_indicators"].shape == (9, 3, 23), "COMPLETE_RESPONSE_WINDOW_TABLE_REQUIRED")
            conditional_response = response_slice(projected, observed[3:5], data["cells"], data["any_count_difference_indicators"])
        rounded_coarse, rounded_middle = rounded_parameter_difference(p), rounded_parameter_difference(p / 2)
        conditional_tv = min(Fraction(1), c * (rounded_coarse.pop("single_split_total_variation")
                                              + 2 * rounded_middle.pop("single_split_total_variation")))
        report.update(status="COMPLETE_PFC_CONDITIONAL_CLOCK_TREE", ideal_law=True,
            probability_binary64_hex=float(p).hex(), probability=fraction_record(p),
            input_coarse_tick=coarse_tick, packed_unit_index=packed_unit, source_unit_index=settings["unit_index"],
            reconstructed_coarse_rows=coarse_tick + 1, reconstructed_unit_count=len(initial),
            observed_coarse_count=c, observed_middle_fine=list(observed), observed_joint_mass=fraction_record(law[observed]),
            joint_support=len(law), fine_support=len(projected), conditional_mass_exact=True,
            conditional_fine_component_law="Binomial(c,1/4); components dependent",
            conditional_fine_mean=fraction_record(Fraction(c, 4)),
            conditional_distinct_fine_covariance=fraction_record(-c * (1 - p) / 16),
            latent_all_parents_exactly_one_fine_mass=fraction_record((1 - p / 4) ** (3 * c)),
            latent_component_is_total_count_conditioning=False,
            reusable_previous_original_fine_vectors=[list(x) for x in sorted(original_slice)],
            previous_slice_conditional_mass=fraction_record(sum(original_slice.values())),
            saved_response_delivery_sha256=response_digest, fixed_coarse_response_slice=conditional_response,
            previous_slice_scope="Existing innovation response has only coordinates0/3 varied; coordinates1/2 stay fixed. This slice is not the full fixed-coarse conditional law.",
            rounded_parameter_comparison=dict(coarse=rounded_coarse, middle=rounded_middle,
                conditional_joint_total_variation_upper=fraction_record(conditional_tv),
                scope="Ideal categorical tree versus ideal Binomial draws using literal rounded second parameters; finite PRNG and sampling algorithms are not certified."),
            decimal_scale_digits=40, preparation_delivery_sha256=preparation_digest, kernel_delivery_sha256=kernel_digest,
            interpretation="Exact conditional input tree only. Does not make neural units independent or bound the remaining input histories or full endpoint scores.",
            outputs={name: branch._file_record(output / name) for name in ("joint_tree.csv", "fine_projection.csv")})
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    report["wall_seconds"] = time.perf_counter() - started
    freeze_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-archive", type=Path, required=True)
    parser.add_argument("--preparation-sha256", required=True)
    parser.add_argument("--kernel-archive", type=Path, required=True)
    parser.add_argument("--kernel-sha256", required=True)
    parser.add_argument("--response-archive", type=Path, required=True)
    parser.add_argument("--response-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.preparation_archive, args.preparation_sha256, args.kernel_archive,
                 args.kernel_sha256, args.response_archive, args.response_sha256, args.output)
    print(json.dumps({key: result[key] for key in ("status", "error") if key in result}))
    return 0 if result["status"] == "COMPLETE_PFC_CONDITIONAL_CLOCK_TREE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
