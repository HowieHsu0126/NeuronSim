"""Scenario-level inference for complete native AppWorld outcome tables.

This module performs no task or file access and does not admit sources or open
formal evaluation. Callers own source admission and protocol freeze. Seeds and
variants stay inside each scenario. All planned comparisons must be scoreable
before any inferential table is returned.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
import scipy
from scipy.stats import false_discovery_control

from .appworld_benchmark_protocol import EXPERIMENT_SEEDS, aggregate_results, build_attempt_index


PROCEDURE = "scenario_bootstrap_exact_paired_sign_flip_v1"
CONFIDENCE_LEVEL = 0.95
FDR_Q = 0.05
BOOTSTRAP_RESAMPLES_PER_SEED = 10000
_BOOTSTRAP_BATCH_SIZE = 256
_METRICS = ("tgc", "sgc")
_VARIANT_COUNT = 3


def exact_scenario_sign_flip(differences: Sequence[int]) -> dict[str, Any]:
    """Exact two-sided label-swap test for integer scenario score differences.

    Under the null, exchanging the paired method labels within any independent
    scenario preserves the joint distribution. All variants and seeds move
    together. Dynamic programming counts every attainable signed sum, including
    ties. Zero differences remain in the scenario count; their duplicate sign
    assignments cancel in the probability ratio. No Monte Carlo p-value is used.
    """
    values = list(differences)
    if not values or any(type(value) is not int for value in values):
        raise ValueError("nonempty integer scenario differences are required")
    observed = sum(values)
    active = [abs(value) for value in values if value != 0]
    counts = {0: 1}
    for difference in active:
        updated: dict[int, int] = defaultdict(int)
        for total, count in counts.items():
            updated[total - difference] += count
            updated[total + difference] += count
        counts = dict(updated)
    tail = sum(count for total, count in counts.items() if abs(total) >= abs(observed))
    assignments = 1 << len(active)
    return {
        "p_value": tail / assignments, "observed_integer_sum": observed,
        "independent_scenario_count": len(values), "nonzero_scenario_count": len(active),
        "zero_scenario_count": len(values) - len(active),
        "effective_sign_assignment_count": str(assignments), "tail_sign_assignment_count": str(tail),
        "p_value_method": "exact_two_sided_scenario_label_swap",
    }


def _joint_bootstrap(values: np.ndarray, resamples_per_seed: int) -> np.ndarray:
    # values axes are scenario, method, metric. One index matrix resamples all
    # methods and both endpoints together, retaining their paired dependence.
    n_scenarios = values.shape[0]
    distributions = []
    for seed in EXPERIMENT_SEEDS:
        generator = np.random.Generator(np.random.PCG64(seed))
        blocks = []
        for first in range(0, resamples_per_seed, _BOOTSTRAP_BATCH_SIZE):
            count = min(_BOOTSTRAP_BATCH_SIZE, resamples_per_seed - first)
            indices = generator.integers(0, n_scenarios, size=(count, n_scenarios))
            blocks.append(values[indices].mean(axis=1))
        distributions.append(np.concatenate(blocks, axis=0))
    return np.concatenate(distributions, axis=0)


def _interval(values: np.ndarray, bootstrap_values: np.ndarray, bounds: tuple[float, float]) -> dict[str, Any]:
    point = float(values.mean())
    constant = bool(np.all(values == values[0]))
    if constant:
        # A constant empirical sample gives a zero-width ordinary bootstrap.
        # Use the predeclared independent bounded-mean inequality instead of
        # interpreting that empirical degeneracy as population certainty.
        radius = (bounds[1] - bounds[0]) * math.sqrt(math.log(2 / (1 - CONFIDENCE_LEVEL)) / (2 * len(values)))
        lower, upper = max(bounds[0], point - radius), min(bounds[1], point + radius)
        method = "hoeffding_bounded_mean_for_constant_sample"
    else:
        alpha = 1 - CONFIDENCE_LEVEL
        lower, upper = np.quantile(bootstrap_values, [alpha / 2, 1 - alpha / 2], method="linear")
        method = "paired_scenario_percentile_bootstrap"
    return {
        "confidence_interval": [float(lower), float(upper)], "interval_method": method,
        "nominal_confidence_level": CONFIDENCE_LEVEL, "observed_constant": constant,
    }


def analyze_native_results(
    manifest: Mapping[str, Any], methods: Sequence[str] | Mapping[str, Any],
    splits: Sequence[str], outcomes: Sequence[Mapping[str, Any]], *,
    target_method: str = "neuronsim", bootstrap_resamples_per_seed: int = BOOTSTRAP_RESAMPLES_PER_SEED,
) -> dict[str, Any]:
    """Return complete-cohort estimates, paired intervals and one BH family.

    The estimand averages pass@1 over the declared three seeds, conditional on
    those fitted models/generation seeds. Confidence intervals resample scenarios,
    not training seeds. Bootstrap percentile intervals are approximate and are
    not simultaneous family intervals. A constant observed sample uses the
    explicitly named bounded-mean interval. BH covers every target/reference x
    split x TGC/SGC comparison supplied in this call.

    The resample override supports bounded software validation and development.
    Formal callers must use the complete pre-test frozen protocol. This pure
    computation does not assert that its supplied booleans came from a native
    evaluator or that the methods and protocol were formally admitted.
    """
    if type(bootstrap_resamples_per_seed) is not int or bootstrap_resamples_per_seed < 1:
        raise ValueError("positive integer bootstrap_resamples_per_seed is required")
    aggregation = aggregate_results(manifest, methods, splits, outcomes)
    method_names, split_names = aggregation["methods"], aggregation["splits"]
    if target_method not in method_names or len(method_names) < 2:
        raise ValueError("the target and at least one reference method must be in the cohort")
    report: dict[str, Any] = {
        "schema": "appworld_benchmark_inference_v1", "procedure": PROCEDURE,
        "complete": aggregation["complete"], "status": "INCOMPLETE_COHORT",
        "source_and_freeze_admission": "caller_responsibility",
        "independent_unit": "scenario", "metric_unit": "fraction", "target_method": target_method,
        "seeds": list(EXPERIMENT_SEEDS), "bootstrap_seeds": list(EXPERIMENT_SEEDS),
        "bootstrap_resamples_per_seed": bootstrap_resamples_per_seed,
        "bootstrap_resample_count": bootstrap_resamples_per_seed * len(EXPERIMENT_SEEDS),
        "bootstrap_rng": "numpy.Generator(PCG64)", "quantile_method": "linear",
        "confidence_level": CONFIDENCE_LEVEL, "fdr_method": "BH", "fdr_q": FDR_Q,
        "comparison_family": "all_supplied_splits_and_endpoints_and_target_references",
        "planned_comparison_count": len(split_names) * len(_METRICS) * (len(method_names) - 1),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "aggregation": aggregation, "scenario_statistics": [], "method_estimates": [], "paired_comparisons": [],
    }
    if not aggregation["complete"]:
        return report
    attempts = build_attempt_index(manifest, method_names, split_names)
    score_lookup = {(row["split"], row["task_id"], row["method"], row["seed"]): int(row["native_tgc"]) for row in outcomes}
    method_index = {method: index for index, method in enumerate(method_names)}
    seed_index = {seed: index for index, seed in enumerate(EXPERIMENT_SEEDS)}
    point_lookup = {(row["split"], row["method"]): row for row in aggregation["seed_means"]}
    target_index = method_index[target_method]
    denominators = np.asarray([len(EXPERIMENT_SEEDS) * _VARIANT_COUNT, len(EXPERIMENT_SEEDS)], dtype=np.int64)
    for split in split_names:
        selected = [row for row in attempts if row["split"] == split]
        scenario_ids = list(dict.fromkeys(row["scenario_id"] for row in selected))
        scenario_index = {scenario: index for index, scenario in enumerate(scenario_ids)}
        raw = np.empty((len(scenario_ids), len(method_names), len(EXPERIMENT_SEEDS), _VARIANT_COUNT), dtype=np.int64)
        for row in selected:
            raw[scenario_index[row["scenario_id"]], method_index[row["method"]], seed_index[row["seed"]], row["variant"] - 1] = score_lookup[(split, row["task_id"], row["method"], row["seed"])]
        # SGC conjunction occurs within a seed, before the arithmetic seed mean.
        counts = np.stack((raw.sum(axis=(2, 3)), raw.all(axis=3).sum(axis=2)), axis=-1)
        values = counts / denominators
        bootstrap = _joint_bootstrap(values, bootstrap_resamples_per_seed)
        for scenario, scenario_id in enumerate(scenario_ids):
            for method, index in method_index.items():
                report["scenario_statistics"].append({
                    "split": split, "scenario_id": scenario_id, "method": method,
                    "tgc_success_count": int(counts[scenario, index, 0]), "tgc_denominator": int(denominators[0]),
                    "sgc_success_seed_count": int(counts[scenario, index, 1]), "sgc_denominator": int(denominators[1]),
                    "tgc": float(values[scenario, index, 0]), "sgc": float(values[scenario, index, 1]),
                })
        for metric_index, metric in enumerate(_METRICS):
            for method, index in method_index.items():
                report["method_estimates"].append({
                    "split": split, "method": method, "metric": metric,
                    "independent_scenario_count": len(scenario_ids),
                    "point_estimate": point_lookup[(split, method)][metric + "_mean"],
                    **_interval(values[:, index, metric_index], bootstrap[:, index, metric_index], (0.0, 1.0)),
                })
                if method == target_method:
                    continue
                integer_differences = (counts[:, target_index, metric_index] - counts[:, index, metric_index]).tolist()
                differences = np.asarray(integer_differences, dtype=float) / int(denominators[metric_index])
                bootstrap_differences = bootstrap[:, target_index, metric_index] - bootstrap[:, index, metric_index]
                report["paired_comparisons"].append({
                    "split": split, "metric": metric, "target_method": target_method, "reference_method": method,
                    "effect": sum(integer_differences) / (len(scenario_ids) * int(denominators[metric_index])),
                    "effect_direction": "target_minus_reference",
                    **_interval(differences, bootstrap_differences, (-1.0, 1.0)),
                    **exact_scenario_sign_flip(integer_differences),
                })
    adjusted = false_discovery_control([row["p_value"] for row in report["paired_comparisons"]], method="bh")
    for row, q_value in zip(report["paired_comparisons"], adjusted):
        row.update(p_value_bh=float(q_value), fdr_reject=bool(q_value <= FDR_Q))
    report["status"] = "STATISTICS_COMPUTED"
    return report
