"""Pure synthetic mathematics/cohort tests; no native or formal outcomes used.

The deliberately small bootstrap override is software-test scope only. It does
not replace the default resampling budget or a frozen scientific protocol.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from itertools import product
import math

import numpy as np
import pytest

from population_dynamics import appworld_benchmark_inference as inference


_SYNTHETIC_RESAMPLES_PER_SEED = 31
_SEEDS = (41, 42, 43)


def pattern_with_success_count(count):
    """Place a known number of synthetic successes into three complete seeds."""
    assert 0 <= count <= 9
    return tuple(tuple(3 * seed_index + variant_index < count for variant_index in range(3)) for seed_index in range(3))


def synthetic_cohort(scenarios_by_split):
    """Construct booleans directly, independently of production attempt indexing."""
    first_scenarios = next(iter(scenarios_by_split.values()))
    methods = list(first_scenarios[0])
    manifest = {
        "schema": "synthetic_appworld_membership_fixture_v1", "appworld_version": "synthetic-not-native",
        "data_root": "/synthetic/not-read", "splits": {},
    }
    outcomes = []
    for split, scenarios in scenarios_by_split.items():
        task_ids = [f"{split}-scenario-{index}_{variant}" for index in range(len(scenarios)) for variant in (1, 2, 3)]
        manifest["splits"][split] = {
            "task_ids": task_ids, "task_count": len(task_ids), "scenario_count": len(scenarios),
            "membership_sha256": "a" * 64,
        }
        for scenario_index, patterns in enumerate(scenarios):
            assert set(patterns) == set(methods)
            for variant_index in range(3):
                for seed_index, seed in enumerate(_SEEDS):
                    for method in methods:
                        score = patterns[method][seed_index][variant_index]
                        assert type(score) is bool
                        outcomes.append({
                            "split": split, "task_id": f"{split}-scenario-{scenario_index}_{variant_index + 1}",
                            "method": method, "seed": seed, "status": "completed", "native_tgc": score,
                        })
    return manifest, methods, list(scenarios_by_split), outcomes


def analyze(cohort, **kwargs):
    manifest, methods, splits, outcomes = cohort
    return inference.analyze_native_results(
        manifest, methods, splits, outcomes,
        bootstrap_resamples_per_seed=kwargs.pop("bootstrap_resamples_per_seed", _SYNTHETIC_RESAMPLES_PER_SEED),
        **kwargs,
    )


def paired_row(report, metric, *, split=None, reference="baseline"):
    rows = [row for row in report["paired_comparisons"] if row["metric"] == metric and row["reference_method"] == reference and (split is None or row["split"] == split)]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("differences", [
    [0], [0, 0, 0], [1], [-1], [1, 1], [1, -1], [2, -2, 0],
    [1, 2, -3], [2, 2, -1, 0], [-2, -1, -3, 0], [3, -1, -1, 0, 2],
])
def test_exact_sign_flip_matches_independent_brute_force_including_ties(differences):
    observed = sum(differences)
    signed_sums = [
        sum(sign * difference for sign, difference in zip(signs, differences))
        for signs in product((-1, 1), repeat=len(differences))
    ]
    brute_tail_count = sum(abs(value) >= abs(observed) for value in signed_sums)
    expected = Fraction(brute_tail_count, 2 ** len(differences))
    result = inference.exact_scenario_sign_flip(differences)
    zeros = differences.count(0)
    assert result["p_value"] == float(expected)
    assert result["observed_integer_sum"] == observed
    assert result["independent_scenario_count"] == len(differences)
    assert result["zero_scenario_count"] == zeros
    assert result["nonzero_scenario_count"] == len(differences) - zeros
    assert int(result["effective_sign_assignment_count"]) == 2 ** (len(differences) - zeros)
    assert int(result["tail_sign_assignment_count"]) == brute_tail_count // (2 ** zeros)


def test_139_positive_scenarios_have_exact_extreme_tail_without_seed_inflation():
    result = inference.exact_scenario_sign_flip([1] * 139)
    assert result["p_value"] == math.ldexp(1.0, -138)
    assert result["independent_scenario_count"] == result["nonzero_scenario_count"] == 139
    assert result["effective_sign_assignment_count"] == str(2 ** 139)
    assert result["tail_sign_assignment_count"] == "2"
    assert result["p_value_method"] == "exact_two_sided_scenario_label_swap"


def test_sgc_conjoins_variants_within_seed_before_seed_averaging():
    full_then_partial = ((True, True, True), (True, False, True), (False, True, True))
    disjoint_seed_successes = ((True, False, False), (False, True, False), (False, False, True))
    cohort = synthetic_cohort({"synthetic-a": [
        {"neuronsim": full_then_partial, "baseline": pattern_with_success_count(9)},
        {"neuronsim": disjoint_seed_successes, "baseline": pattern_with_success_count(9)},
    ]})
    report = analyze(cohort)
    target_scenarios = [row for row in report["scenario_statistics"] if row["method"] == "neuronsim"]
    assert [(row["tgc_success_count"], row["sgc_success_seed_count"]) for row in target_scenarios] == [(7, 1), (3, 0)]
    assert all(row["tgc_denominator"] == 9 and row["sgc_denominator"] == 3 for row in target_scenarios)
    assert [row["sgc"] for row in target_scenarios] == [1 / 3, 0]
    estimates = {row["metric"]: row for row in report["method_estimates"] if row["method"] == "neuronsim"}
    assert estimates["tgc"]["point_estimate"] == pytest.approx(5 / 9)
    assert estimates["sgc"]["point_estimate"] == pytest.approx(1 / 6)
    for row in report["paired_comparisons"]:
        assert row["independent_scenario_count"] == 2
        assert row["effective_sign_assignment_count"] == "4"
        assert row["p_value"] == 0.5
    assert all(row["independent_scenario_count"] == 2 for row in report["method_estimates"])


def test_joint_bootstrap_preserves_scenario_relations_across_methods_and_metrics(monkeypatch):
    # Each nonzero scenario has target=(2/3,2/3), baseline=(1/3,1/3).
    # Whole-scenario joint sampling must preserve both relations in every draw;
    # independently sampling seeds, methods or endpoints can break them.
    cohort = synthetic_cohort({"synthetic-a": [
        {"neuronsim": pattern_with_success_count(target), "baseline": pattern_with_success_count(target // 2)}
        for target in (0, 6, 6, 0)
    ]})
    original_bootstrap = inference._joint_bootstrap
    observed = []

    def observe(values, resamples_per_seed):
        distribution = original_bootstrap(values, resamples_per_seed)
        observed.append((values.copy(), distribution.copy(), resamples_per_seed))
        return distribution

    monkeypatch.setattr(inference, "_joint_bootstrap", observe)
    report = analyze(cohort)
    assert len(observed) == 1
    values, draws, resamples = observed[0]
    assert values.shape == (4, 2, 2)  # Four scenarios; neither 12 seeds nor 36 attempts.
    assert draws.shape == (3 * _SYNTHETIC_RESAMPLES_PER_SEED, 2, 2)
    assert resamples == _SYNTHETIC_RESAMPLES_PER_SEED
    np.testing.assert_allclose(draws[:, 0, 0], 2 * draws[:, 1, 0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(draws[:, :, 0], draws[:, :, 1], rtol=0, atol=1e-15)
    assert report["bootstrap_seeds"] == report["seeds"] == [41, 42, 43]
    assert report["bootstrap_resample_count"] == 3 * _SYNTHETIC_RESAMPLES_PER_SEED
    for metric in ("tgc", "sgc"):
        row = paired_row(report, metric)
        assert row["interval_method"] == "paired_scenario_percentile_bootstrap"
        lower, upper = row["confidence_interval"]
        assert lower >= -1e-15 and upper <= 1 / 3 + 1e-15


def test_constant_integer_paired_difference_uses_nonzero_hoeffding_interval():
    # Subtracting separately scaled counts is not bitwise constant in float64.
    marginal_float_differences = [(count + 1) / 9 - count / 9 for count in range(9)]
    assert len(set(marginal_float_differences)) > 1
    cohort = synthetic_cohort({"synthetic-a": [
        {"neuronsim": pattern_with_success_count(count + 1), "baseline": pattern_with_success_count(count)}
        for count in range(9)
    ]})
    report = analyze(cohort)
    row = paired_row(report, "tgc")
    assert row["effect"] == pytest.approx(1 / 9)
    assert row["observed_constant"] is True
    assert row["interval_method"] == "hoeffding_bounded_mean_for_constant_sample"
    radius = 2 * math.sqrt(math.log(40) / (2 * 9))
    assert row["confidence_interval"] == pytest.approx([max(-1, 1 / 9 - radius), min(1, 1 / 9 + radius)])
    lower, upper = row["confidence_interval"]
    assert lower < row["effect"] < upper
    assert row["independent_scenario_count"] == 9 and row["p_value"] == 2 ** -8


@pytest.mark.parametrize("defect", ["missing", "all_missing", "completed", "agent_failure", "evaluator_failure", "infrastructure_failure", "unknown"])
def test_any_incomplete_family_withholds_all_inferential_rows(defect):
    scenarios = [{method: pattern_with_success_count(9) for method in ("neuronsim", "baseline_a", "baseline_b")} for _ in range(2)]
    manifest, methods, splits, outcomes = synthetic_cohort({"synthetic-a": deepcopy(scenarios), "synthetic-b": deepcopy(scenarios)})
    if defect == "missing":
        outcomes.pop()
    elif defect == "all_missing":
        outcomes = []
    else:
        outcomes[-1].update(status=defect, native_tgc=None)
    report = analyze((manifest, methods, splits, outcomes))
    assert report["status"] == "INCOMPLETE_COHORT" and report["complete"] is False
    assert report["planned_comparison_count"] == 8
    assert report["scenario_statistics"] == []
    assert report["method_estimates"] == [] and report["paired_comparisons"] == []
    assert not report["aggregation"]["complete"]


def test_scored_agent_failure_preserves_native_success_for_inference():
    cohort = synthetic_cohort({"synthetic-a": [
        {"neuronsim": pattern_with_success_count(9), "baseline": pattern_with_success_count(3)},
        {"neuronsim": pattern_with_success_count(6), "baseline": pattern_with_success_count(0)},
    ]})
    ordinary = analyze(cohort)
    failed_execution = deepcopy(cohort)
    failed_execution[3][0]["status"] = "agent_failure"
    assert failed_execution[3][0]["native_tgc"] is True
    observed = analyze(failed_execution)
    assert observed["complete"] and observed["status"] == "STATISTICS_COMPUTED"
    assert observed["aggregation"]["agent_failures"][0]["native_tgc"] is True
    for table in ("scenario_statistics", "method_estimates", "paired_comparisons"):
        assert observed[table] == ordinary[table]


def test_reversing_outcome_order_preserves_every_statistic_exactly():
    cohort = synthetic_cohort({"synthetic-a": [
        {"neuronsim": pattern_with_success_count(target), "baseline": pattern_with_success_count(reference)}
        for target, reference in ((2, 1), (7, 5), (4, 8))
    ]})
    forward = analyze(cohort, bootstrap_resamples_per_seed=37)
    manifest, methods, splits, outcomes = cohort
    backward = analyze((manifest, methods, splits, list(reversed(outcomes))), bootstrap_resamples_per_seed=37)
    assert backward == forward
    assert forward["bootstrap_resamples_per_seed"] == 37 and forward["bootstrap_resample_count"] == 111
    assert inference.BOOTSTRAP_RESAMPLES_PER_SEED == 10000


def test_bh_uses_one_eight_comparison_family_with_analytic_adjusted_values(monkeypatch):
    positive_split = [{
        "neuronsim": pattern_with_success_count(9), "baseline_a": pattern_with_success_count(0),
        "baseline_b": pattern_with_success_count(9),
    } for _ in range(6)]
    null_split = [{method: pattern_with_success_count(9) for method in ("neuronsim", "baseline_a", "baseline_b")} for _ in range(6)]
    cohort = synthetic_cohort({"synthetic-effect": positive_split, "synthetic-null": null_split})
    fdr_calls = []
    original_fdr = inference.false_discovery_control

    def observe_family(p_values, *, method):
        fdr_calls.append((list(p_values), method))
        return original_fdr(p_values, method=method)

    monkeypatch.setattr(inference, "false_discovery_control", observe_family)
    report = analyze(cohort)
    assert report["planned_comparison_count"] == len(report["paired_comparisons"]) == 8
    assert len(fdr_calls) == 1 and fdr_calls[0][1] == "bh"
    assert sorted(fdr_calls[0][0]) == [0.03125, 0.03125, 1, 1, 1, 1, 1, 1]
    effects = [row for row in report["paired_comparisons"] if row["split"] == "synthetic-effect" and row["reference_method"] == "baseline_a"]
    nulls = [row for row in report["paired_comparisons"] if row not in effects]
    assert len(effects) == 2 and len(nulls) == 6
    for row in effects:
        assert row["effect"] == 1 and row["independent_scenario_count"] == 6
        assert row["p_value"] == 0.03125 and row["p_value_bh"] == 0.125
        assert row["fdr_reject"] is False
        assert row["confidence_interval"][0] < row["confidence_interval"][1]
    assert all(row["p_value"] == row["p_value_bh"] == 1 and not row["fdr_reject"] for row in nulls)
    for row in report["method_estimates"]:
        if row["method"] == "neuronsim":
            assert row["observed_constant"] is True
            assert row["interval_method"] == "hoeffding_bounded_mean_for_constant_sample"
            assert row["confidence_interval"][0] < 1 and row["confidence_interval"][1] == 1
