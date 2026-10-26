"""Exact uncertainty for complete PFC episode direction-state descriptors.

The source cue support defines geometric alignment. The four states retain
silence, cancellation and outside-support activity. They are not memory PASS
labels. Conditions are paired within independently sampled source histories.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import time

from scipy.stats import binomtest

from .pfc_vector_uncertainty import (
    BLOCKS, CIRCUITS, CONDITIONS, SIZES, SUMMARY_SHA256, require, sha256,
    write_json, write_rows,
)

STATES = ("silent", "cancelled_non_silent", "aligned_with_cue_support", "resolved_outside_cue_support")
PAIRS = tuple((k + 5, k, "cue_minus_omission") for k in range(5)) + tuple(
    (0, k, "cue_distractor_minus_none") for k in range(1, 5)) + tuple(
    (5, k + 5, "omission_distractor_minus_none") for k in range(1, 5))
SCHEMA = "pfc_episode_direction_state_probabilities_v1"


def state_for_episode(row):
    """Use existing numerical eligibility and source-derived cue support."""
    total = int(row["excitatory_spikes"])
    require(total >= 0, "PFC_STATE_NEGATIVE_COUNTS")
    require(row["silent"] in ("True", "False") and row["numerical_direction_defined"] in ("True", "False"),
            "PFC_STATE_BOOLEAN_INVALID")
    silent, defined = row["silent"] == "True", row["numerical_direction_defined"] == "True"
    require(silent == (total == 0), "PFC_STATE_SILENCE_COUNT_MISMATCH")
    require(not silent or not defined, "PFC_STATE_SILENT_DIRECTION_DEFINED")
    angle = row["circular_cue_error_deg"]
    if not defined:
        require(angle == "" and row["numerical_angle_inside_cue_support"] == "", "PFC_STATE_UNDEFINED_ANGLE")
        return "silent" if silent else "cancelled_non_silent"
    value = float(angle)
    require(math.isfinite(value) and -180 <= value < 180, "PFC_STATE_CIRCULAR_ERROR_INVALID")
    aligned = abs(value) <= 15.
    require(row["numerical_angle_inside_cue_support"] == str(aligned), "PFC_STATE_CUE_SUPPORT_CHANGED")
    return "aligned_with_cue_support" if aligned else "resolved_outside_cue_support"


def exact_interval(successes, trials, alpha):
    """Clopper-Pearson interval; retain positive uncertainty for zero counts."""
    require(type(successes) is int and type(trials) is int and trials > 0
            and 0 <= successes <= trials and 0 < alpha < 1, "PFC_STATE_BINOMIAL_DESIGN_INVALID")
    interval = binomtest(successes, trials).proportion_ci(confidence_level=1 - alpha, method="exact")
    return float(interval.low), float(interval.high)


def paired_state_effect(reference, comparison, state, alpha):
    """Unconditional gains/losses use all histories, including concordances."""
    require(len(reference) == len(comparison) > 0
            and all(s in STATES for s in (*reference, *comparison)) and state in STATES,
            "PFC_STATE_COMPLETE_PAIRED_HISTORIES_REQUIRED")
    counts = Counter((a == state, b == state) for a, b in zip(reference, comparison))
    n00, n01, n10, n11 = (counts[False, False], counts[False, True], counts[True, False], counts[True, True])
    n = len(reference)
    gain, loss = exact_interval(n01, n, alpha), exact_interval(n10, n, alpha)
    return dict(n00=n00, n01=n01, n10=n10, n11=n11, independent_histories=n,
        mean_probability_change=(n01 - n10) / n,
        gain_probability_lower=gain[0], gain_probability_upper=gain[1],
        loss_probability_lower=loss[0], loss_probability_upper=loss[1],
        probability_change_lower=gain[0] - loss[1], probability_change_upper=gain[1] - loss[0])


def load_vector_rows(source):
    report_path = source / "report.json"
    report = json.loads(report_path.read_bytes())
    require(report.get("schema") == "pfc_joint_vector_uncertainty_v1"
            and report.get("status") == "COMPLETE_CALIBRATION_VECTOR_UNCERTAINTY"
            and report["summary_sha256"] == SUMMARY_SHA256
            and report["circuits"] == list(CIRCUITS) and report["blocks"] == list(BLOCKS)
            and report["source_sizes"] == list(SIZES) and report["conditions"] == list(CONDITIONS)
            and report["complete_source_episodes"] == 240
            and report["source_simulations"] == report["production_episodes"] == 0
            and all(report[k] is False for k in ("protected_circuits_used", "source_qualification_granted",
                "scaling_qualified", "memory_success_qualified", "scientific_direction_qualified")),
            "PFC_STATE_BOUND_COMPLETE_VECTOR_CALIBRATION_REQUIRED")
    path = source / "episode_vector_moments.csv"
    require(sha256(path) == report["outputs"][path.name]["sha256"], "PFC_STATE_VECTOR_TABLE_CHANGED")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(c, size, k, phase, b) for c in CIRCUITS for size in SIZES
                for k in range(10) for phase in range(8) for b in BLOCKS}
    indexed = {}
    for row in rows:
        require(row["condition_id"] in CONDITIONS, "PFC_STATE_CONDITION_INVALID")
        key = (int(row["circuit_index"]), int(row["source_size"]), CONDITIONS.index(row["condition_id"]),
               int(row["phase_index"]), int(row["block_index"]))
        require(key in expected and key not in indexed, "PFC_STATE_DUPLICATE_OR_PROTECTED_CELL")
        require(row["split"] == ("calibration" if key[0] == 86 else "train")
                and row["scientific_direction_qualified"] == "False", "PFC_STATE_INFERENCE_SCOPE_CHANGED")
        indexed[key] = dict(row, direction_state=state_for_episode(row))
    require(set(indexed) == expected and len(rows) == 1920, "PFC_STATE_COMPLETE_HISTORY_FAMILY_REQUIRED")
    return report, indexed


def analyze(source, output, alpha):
    started = time.perf_counter()
    report, episodes = load_vector_rows(source)
    require(not output.exists() and 0 < alpha < 1, "PFC_STATE_OUTPUT_OR_ALPHA_INVALID")
    strata = len(CIRCUITS) * len(SIZES) * 8
    primitive_endpoints = strata * len(STATES) * (len(CONDITIONS) + 2 * len(PAIRS))
    elementary_alpha = alpha / primitive_endpoints
    marginal, effects, transitions, size_effects, bounds = [], [], [], [], {}
    for c in CIRCUITS:
        for size in SIZES:
            for phase in range(8):
                for k, condition in enumerate(CONDITIONS):
                    states = [episodes[c, size, k, phase, b]["direction_state"] for b in BLOCKS]
                    for state in STATES:
                        successes = states.count(state)
                        lo, hi = exact_interval(successes, len(BLOCKS), elementary_alpha)
                        marginal.append(dict(circuit_index=c, source_size=size, phase_index=phase,
                            condition_id=condition, direction_state=state, occurrences=successes,
                            independent_histories=len(BLOCKS), probability=successes / len(BLOCKS),
                            lower=lo, upper=hi, elementary_alpha=elementary_alpha,
                            descriptor_is_memory_success=False))
                        bounds[c, size, phase, k, state] = (successes / len(BLOCKS), lo, hi)
                for reference, comparison, contrast in PAIRS:
                    first = [episodes[c, size, reference, phase, b]["direction_state"] for b in BLOCKS]
                    second = [episodes[c, size, comparison, phase, b]["direction_state"] for b in BLOCKS]
                    base = dict(circuit_index=c, source_size=size, phase_index=phase,
                        reference_condition=CONDITIONS[reference], comparison_condition=CONDITIONS[comparison],
                        contrast=contrast)
                    table = Counter(zip(first, second))
                    for before in STATES:
                        for after in STATES:
                            transitions.append(dict(base, reference_state=before, comparison_state=after,
                                histories=table[before, after], independent_histories=len(BLOCKS)))
                    for state in STATES:
                        effects.append(dict(base, direction_state=state,
                            **paired_state_effect(first, second, state, elementary_alpha),
                            elementary_alpha=elementary_alpha, contrast_is_memory_effect=False))
        for phase in range(8):
            for k, condition in enumerate(CONDITIONS):
                for state in STATES:
                    small, large = bounds[c, 2560, phase, k, state], bounds[c, 100000, phase, k, state]
                    size_effects.append(dict(circuit_index=c, phase_index=phase, condition_id=condition,
                        direction_state=state, expanded_minus_original=large[0] - small[0],
                        lower=large[1] - small[2], upper=large[2] - small[1],
                        cross_size_noise_pairing=False, scaling_qualified=False))
    output.mkdir(parents=True)
    outputs = dict(episode_direction_states=list(episodes.values()), state_probability_intervals=marginal,
        paired_state_transitions=transitions, paired_state_probability_changes=effects,
        state_probability_size_effects=size_effects)
    for name, rows in outputs.items():
        write_rows(output / (name + ".csv"), rows)
    result = dict(schema=SCHEMA, status="COMPLETE_CALIBRATION_EPISODE_STATE_PROBABILITIES",
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        execution=dict(backend="merlin", run_id=os.environ.get("RH2_JOB_RUN_ID"), trial_id=os.environ.get("ARNOLD_TRIAL_ID")),
        source_vector_report_sha256=sha256(source / "report.json"), source_summary_sha256=SUMMARY_SHA256,
        source_bindings=report["source_bindings"], task_definition_sha256=report["task_definition_sha256"],
        circuits=list(CIRCUITS), source_sizes=list(SIZES), blocks=list(BLOCKS), conditions=list(CONDITIONS),
        complete_source_episodes=240, direction_states=list(STATES), phases=8,
        independent_histories_per_circuit_and_size=3, family_alpha=alpha,
        primitive_probability_endpoints=primitive_endpoints, elementary_alpha=elementary_alpha,
        confidence_method="exact_two_sided_Clopper_Pearson_with_Bonferroni_over_all9216_marginal_and_transition_probability_endpoints",
        paired_interval_method="probability_gain_minus_loss_from_same_complete_history_pairs; both exact intervals use all histories",
        size_interval_method="signed propagation of same marginal probability intervals; unpaired across sizes",
        coverage_basis="ideal independent identically distributed joint source histories within each fixed circuit and size; arbitrary dependence between paired conditions, phases and state indicators",
        uncertainty_scope="complete development calibration descriptor family only; formal source/coarse-engine error allocation unchanged",
        descriptor_semantics="numerically resolved phase vector within source cue support; silence means zero E counts; near-quiet nonzero activity remains visible; omission alignment does not imply memory",
        zero_occurrence_semantics="zero observed state count is not zero probability; no zero-width physical-range fallback",
        original_vector_mean_inference_reused=True, protected_circuits_used=False,
        source_simulations=0, production_episodes=0, source_qualification_granted=False,
        scaling_qualified=False, scientific_direction_qualified=False, memory_success_qualified=False,
        scientific_parameter_domain_qualified=False, formal_confirmation_released=False,
        elapsed_seconds=time.perf_counter() - started, implementation_sha256=sha256(Path(__file__)),
        outputs={name + ".csv": dict(rows=len(rows), sha256=sha256(output / (name + ".csv")))
                 for name, rows in outputs.items()})
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    require(bool(os.environ.get("RH2_JOB_RUN_ID")) and bool(os.environ.get("ARNOLD_TRIAL_ID"))
            and os.environ.get("ARNOLD_APPLIED_GPU_NUM") == "0", "PFC_STATE_MERLIN_CPU_REQUIRED")
    report = analyze(args.source, args.output, args.alpha)
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_bindings", "outputs")}))


if __name__ == "__main__":
    main()
