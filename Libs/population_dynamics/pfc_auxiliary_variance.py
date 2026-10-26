"""Describe two auxiliary realizations nested within 32 unchanged PFC blocks.

For fixed conditional block means mu_i and variances v_i, equal-law independent
auxiliary A/B draws and independent streams across blocks imply
E[mean((B-A)**2)/2] = mean(v_i) and
E[sample_var((A+B)/2) - mean((B-A)**2)/4] = sample_var(mu_i).
The latter is a fixed-block estimand; a population interpretation additionally
requires representative IID original blocks. This module makes no inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .pfc_coupled_clock_comparison import LABELS
from .pfc_coupled_precision_comparison import (
    SEEDS, UNDEFINED, bound_analysis, output_file, read_rows,
)
from .pfc_delay_analysis import require, write_rows
from .provenance import freeze_json, sha256_file

ADJACENT_CLOCK_PAIRS = tuple(zip(LABELS[:-1], LABELS[1:]))
SCHEMA = "pfc_auxiliary_variance_diagnostic_v1"
STATUS = "COMPLETE_DESCRIPTIVE_PFC_AUXILIARY_VARIANCE_DIAGNOSTIC"
PRIMITIVE_KIND = "OBSERVED_WITHIN_ORIGINAL_BLOCK_CHANGE"
COVARIANCE_KIND = "FULL_U_STATISTIC_POINTS_ONLY_NO_BLOCK_SCATTER"
PRIMITIVE_COUNTS = {"count_cdf": 48660, "field_mean": 510,
                    "paired_response": 240, "source_response": 240}
FULL_ENDPOINT_COUNT = 84330
DEFINED_COVARIANCE_COUNT = 32946
UNDEFINED_COVARIANCE_COUNT = 1734
IDENTITY_RTOL, IDENTITY_ATOL = 1e-10, 1e-12
SAVED_KEYS = {
    "source_rng_seeds", "domain_row_indices", "original_scores", "repeated_scores",
    "within_block_midpoint", "repeated_minus_original",
}
POINT_KEYS = ("original_estimate", "repeated_estimate", "estimate_change", "estimate_midpoint")
POINT_FLAG_KEYS = ("original_point_outside_margin", "repeated_point_outside_margin", "same_nonzero_sign")
SCATTER_KEYS = ("observed_block_mean_absolute_change", "observed_block_rms_change")
Row = dict[str, Any]


def configuration(path: Path) -> Row:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(settings, dict) and settings.get("schema") == SCHEMA,
            "AUXILIARY_VARIANCE_SCHEMA_REQUIRED")
    require(settings.get("maximum_confirmation_blocks") == 128
            and settings.get("source_simulations") == settings.get("new_rng_draws") == 0
            and all(settings.get(key) is False for key in (
                "confidence_intervals", "confirmation_execution_released", "scientific_fidelity_qualified")),
            "UNCHANGED_DATA_ONLY_AUXILIARY_VARIANCE_SCOPE_REQUIRED")
    reference = settings.get("comparison")
    require(isinstance(reference, dict) and isinstance(reference.get("path"), str)
            and isinstance(reference.get("report_sha256"), str), "BOUND_REPEAT_COMPARISON_REQUIRED")
    require(isinstance(settings.get("interpretation"), str) and settings["interpretation"].strip()
            and isinstance(settings.get("assumptions"), list) and settings["assumptions"]
            and all(isinstance(item, str) and item.strip() for item in settings["assumptions"]),
            "EXPLICIT_CONDITIONAL_INTERPRETATION_AND_ASSUMPTIONS_REQUIRED")
    return settings


def decompose_auxiliary_pairs(original: np.ndarray, repeated: np.ndarray) -> dict[str, np.ndarray]:
    """Return raw endpoint-wise moments; columns are fixed original blocks.

    No clipping, noise fractions, confidence intervals, or IID interpretation is
    applied. The observed squared changes are descriptive even if the conditional
    independence/equal-law assumptions have not been established.
    """
    a, b = np.asarray(original), np.asarray(repeated)
    require(a.ndim == b.ndim == 2 and a.shape == b.shape and a.shape[1] >= 2,
            "MATCHED_ENDPOINT_BY_BLOCK_AUXILIARY_PAIRS_REQUIRED")
    require(a.dtype.kind in "fiu" and b.dtype.kind in "fiu"
            and np.isfinite(a).all() and np.isfinite(b).all(), "FINITE_REAL_AUXILIARY_PAIRS_REQUIRED")
    a, b = a.astype(np.float64, copy=False), b.astype(np.float64, copy=False)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        midpoint = (a+b)/2
        delta = b-a
        squared_change = delta*delta
        within = squared_change.mean(axis=1)/2
        midpoint_variance = midpoint.var(axis=1, ddof=1)
        between = midpoint_variance-within/2
        total = squared_change.sum(axis=1)
        concentration = np.full(a.shape[0], np.nan)
        defined = total > 0
        concentration[defined] = squared_change[defined].max(axis=1)/total[defined]
    require(all(np.isfinite(x).all() for x in (
        midpoint, delta, squared_change, within, midpoint_variance, between, total)),
        "NONFINITE_AUXILIARY_DECOMPOSITION")
    return dict(original_scores=a, repeated_scores=b, within_block_midpoint=midpoint,
        repeated_minus_original=delta, paired_auxiliary_squared_change=squared_change/2,
        auxiliary_within_variance_estimate=within, midpoint_sample_variance=midpoint_variance,
        between_block_variance_estimate=between, between_block_estimate_negative=between < 0,
        observed_zero_within_variance=within == 0,
        exact_nonzero_auxiliary_change_blocks=np.count_nonzero(delta, axis=1),
        squared_change_total=total, maximum_squared_change_share=concentration)


def _boolean(value: Any, name: str) -> bool:
    require(value in ("True", "False"), "CANONICAL_POINT_FLAG_REQUIRED:"+name)
    return value == "True"


def _clock_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    identity = json.loads(row["endpoint_identity"])
    require(isinstance(identity, dict)
            and (identity.get("left_cohort"), identity.get("right_cohort")) in ADJACENT_CLOCK_PAIRS,
            "CANONICAL_ADJACENT_CLOCK_IDENTITY_REQUIRED")
    return identity


def _inclusion(row: Mapping[str, Any]) -> str:
    if row["family"] in PRIMITIVE_COUNTS:
        return "INCLUDED_PRIMITIVE"
    if row["definition_status"] == UNDEFINED:
        return "EXCLUDED_FROZEN_UNDEFINED_COVARIANCE"
    return "EXCLUDED_DEFINED_COVARIANCE"


def validate_primitive_bank(comparison_rows: Sequence[Row], saved: Mapping[str, np.ndarray]
                            ) -> tuple[list[Row], dict[str, np.ndarray], dict[str, float]]:
    """Validate the pair/axis boundary while excluding covariance pseudovalues.

    The canonical 32-block axis is required here. Full endpoint counts are a run
    gate, so small endpoint fixtures exercise this same boundary in tests.
    """
    require(set(saved) == SAVED_KEYS, "EXACT_PRIMITIVE_NESTED_SCORE_ARRAYS_REQUIRED")
    seeds = np.asarray(saved["source_rng_seeds"])
    require(seeds.ndim == 1 and seeds.dtype.kind in "iu" and np.array_equal(seeds, SEEDS),
            "CANONICAL_SOURCE_RNG_SEED_ORDER_REQUIRED")
    primitives, expected_ids, seen = [], [], set()
    for index, row in enumerate(comparison_rows):
        require(str(row["domain_row_index"]) == str(index), "CANONICAL_DOMAIN_ROW_ORDER_REQUIRED")
        require(str(row["original_independent_blocks"]) == "32" and str(row["auxiliary_realizations"]) == "2",
                "TWO_AUXILIARIES_WITHIN_32_ORIGINAL_BLOCKS_REQUIRED")
        family, status, kind = row["family"], row["definition_status"], row["auxiliary_scatter_kind"]
        identity = _clock_identity(row)
        key = family, json.dumps(identity, sort_keys=True)
        require(key not in seen, "DUPLICATE_ENDPOINT_IDENTITY")
        seen.add(key)
        require(isinstance(status, str) and bool(status), "EXPLICIT_ENDPOINT_DEFINITION_STATUS_REQUIRED")
        if family in PRIMITIVE_COUNTS:
            require(status != UNDEFINED and kind == PRIMITIVE_KIND,
                    "ONLY_DEFINED_PRIMITIVE_BLOCK_RESPONSES_ALLOWED")
            primitives.append(row)
            expected_ids.append(index)
        else:
            require(family == "field_lag_covariance", "UNKNOWN_ENDPOINT_FAMILY")
            require(kind == ("UNDEFINED" if status == UNDEFINED else COVARIANCE_KIND),
                    "COVARIANCE_BLOCK_SCATTER_FORBIDDEN")
            require(all(row[name] == "" for name in SCATTER_KEYS), "COVARIANCE_BLOCK_SCATTER_FORBIDDEN")
        if status == UNDEFINED:
            require(row["margin"] == "" and all(row[name] == "" for name in POINT_KEYS+POINT_FLAG_KEYS),
                    "FROZEN_UNDEFINED_COVARIANCE_CHANGED")
        else:
            margin = float(row["margin"])
            points = np.asarray([float(row[name]) for name in POINT_KEYS])
            require(np.isfinite(margin) and margin >= 0 and np.isfinite(points).all(),
                    "FINITE_POINT_AND_UNCHANGED_NONNEGATIVE_MARGIN_REQUIRED")
            a, b, change, midpoint = points
            require(np.isclose(change, b-a, rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL)
                    and np.isclose(midpoint, (a+b)/2, rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL),
                    "SAVED_POINT_PAIR_IDENTITY_CHANGED")
            for name, expected in zip(POINT_FLAG_KEYS, (abs(a) > margin, abs(b) > margin,
                                                       (a > 0 and b > 0) or (a < 0 and b < 0))):
                require(_boolean(row[name], name) == expected, "SAVED_MARGIN_POINT_FLAG_CHANGED:"+name)
        expected_units = "probability" if family == "count_cdf" else "Hz^2" if family == "field_lag_covariance" else "Hz"
        require(row["units"] == expected_units, "SOURCE_ENDPOINT_UNITS_CHANGED")
    ids = np.asarray(saved["domain_row_indices"])
    require(ids.ndim == 1 and ids.dtype.kind in "iu" and np.array_equal(ids, expected_ids),
            "EXACT_PRIMITIVE_DOMAIN_AXIS_REQUIRED_COVARIANCE_AND_UNDEFINED_EXCLUDED")
    require(bool(primitives), "NONEMPTY_PRIMITIVE_BANK_REQUIRED")
    shape = (len(primitives), len(SEEDS))
    for name in SAVED_KEYS-{"source_rng_seeds", "domain_row_indices"}:
        values = np.asarray(saved[name])
        require(values.shape == shape and values.dtype.kind in "fiu" and np.isfinite(values).all(),
                "FINITE_MATCHED_PRIMITIVE_SCORE_SHAPE_REQUIRED:"+name)
    stats = decompose_auxiliary_pairs(saved["original_scores"], saved["repeated_scores"])
    errors = {}
    for name in ("within_block_midpoint", "repeated_minus_original"):
        require(np.allclose(saved[name], stats[name], rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL),
                "SAVED_AUXILIARY_PAIR_IDENTITY_CHANGED:"+name)
        errors[name] = float(np.max(np.abs(saved[name]-stats[name])))
    delta = stats["repeated_minus_original"]
    checks = dict(original_estimate=stats["original_scores"].mean(axis=1),
        repeated_estimate=stats["repeated_scores"].mean(axis=1),
        estimate_midpoint=stats["within_block_midpoint"].mean(axis=1), estimate_change=delta.mean(axis=1),
        observed_block_mean_absolute_change=np.abs(delta).mean(axis=1),
        observed_block_rms_change=np.sqrt(stats["paired_auxiliary_squared_change"].mean(axis=1)*2))
    for name, actual in checks.items():
        expected = np.asarray([float(row[name]) for row in primitives])
        require(np.isfinite(expected).all()
                and np.allclose(actual, expected, rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL),
                "SAVED_PRIMITIVE_POINT_OR_SCATTER_IDENTITY_CHANGED:"+name)
        errors[name] = float(np.max(np.abs(actual-expected)))
    return primitives, stats, errors


def _point_position(value: float, margin: float) -> str:
    if value > margin:
        return "ABOVE_POSITIVE_MARGIN"
    if value < -margin:
        return "BELOW_NEGATIVE_MARGIN"
    return "WITHIN_OR_ON_MARGIN"


def endpoint_rows(primitives: Sequence[Row], stats: Mapping[str, np.ndarray]) -> list[Row]:
    """Keep every primitive and the exact source metadata, margins and point flags."""
    rows = []
    seeds = np.asarray(SEEDS)
    for index, original in enumerate(primitives):
        row = dict(original)
        margin = float(row["margin"])
        for prefix, point_key in (("original", "original_estimate"), ("repeated", "repeated_estimate"),
                                  ("midpoint", "estimate_midpoint")):
            row[prefix+"_point_margin_position"] = _point_position(float(row[point_key]), margin)
        row["midpoint_point_outside_margin"] = abs(float(row["estimate_midpoint"])) > margin
        row["variance_units"] = row["units"]+"^2"
        for name in ("auxiliary_within_variance_estimate", "midpoint_sample_variance", "between_block_variance_estimate"):
            row[name] = float(stats[name][index])
        row["between_block_estimate_negative"] = bool(stats["between_block_estimate_negative"][index])
        row["observed_zero_within_variance"] = bool(stats["observed_zero_within_variance"][index])
        row["exact_nonzero_auxiliary_change_blocks"] = int(stats["exact_nonzero_auxiliary_change_blocks"][index])
        row["observed_zero_auxiliary_change"] = row["exact_nonzero_auxiliary_change_blocks"] == 0
        half_squared = stats["paired_auxiliary_squared_change"][index]
        row["contributing_source_rng_seeds"] = json.dumps(seeds[half_squared > 0].tolist())
        if stats["squared_change_total"][index] > 0:
            row["concentration_status"] = "DEFINED_OBSERVED_SQUARED_CHANGE_CONCENTRATION"
            row["maximum_squared_change_share"] = float(stats["maximum_squared_change_share"][index])
            row["maximum_share_source_rng_seeds"] = json.dumps(seeds[half_squared == half_squared.max()].tolist())
        else:
            row["concentration_status"] = "UNDEFINED_ZERO_OBSERVED_SQUARED_CHANGE"
            row["maximum_squared_change_share"] = None
            row["maximum_share_source_rng_seeds"] = None
        rows.append(row)
    return rows


def group_count_rows(comparison_rows: Sequence[Row], endpoints: Sequence[Row]) -> list[Row]:
    """Account for the complete input denominator, leaving excluded statistics blank."""
    summaries = {int(row["domain_row_index"]): row for row in endpoints}
    groups = defaultdict(list)
    for row in comparison_rows:
        identity = _clock_identity(row)
        key = (_inclusion(row), row["family"], identity["left_cohort"], identity["right_cohort"],
               str(identity.get("phase", "")), row["definition_status"])
        groups[key].append(row)
    keys = ("inclusion", "family", "left_cohort", "right_cohort", "phase", "definition_status")
    counts = ("original_point_outside_count", "repeated_point_outside_count", "midpoint_point_outside_count",
              "raw_negative_between_count", "observed_zero_within_variance_count",
              "observed_zero_auxiliary_change_count", "defined_concentration_count")
    result = []
    for key, members in sorted(groups.items()):
        summary = dict(zip(keys, key), endpoint_count=len(members), **dict.fromkeys(counts, ""))
        if key[0] == "INCLUDED_PRIMITIVE":
            values = [summaries[int(row["domain_row_index"])] for row in members]
            summary.update(original_point_outside_count=sum(_boolean(r["original_point_outside_margin"], "original") for r in values),
                repeated_point_outside_count=sum(_boolean(r["repeated_point_outside_margin"], "repeated") for r in values),
                midpoint_point_outside_count=sum(r["midpoint_point_outside_margin"] for r in values),
                raw_negative_between_count=sum(r["between_block_estimate_negative"] for r in values),
                observed_zero_within_variance_count=sum(r["observed_zero_within_variance"] for r in values),
                observed_zero_auxiliary_change_count=sum(r["observed_zero_auxiliary_change"] for r in values),
                defined_concentration_count=sum(r["maximum_squared_change_share"] is not None for r in values))
        result.append(summary)
    return result


def focus_endpoints(endpoints: Sequence[Row]) -> tuple[list[int], dict[str, Any]]:
    """Include all out-of-margin primitives and both adjacent pairs for their identity."""
    matched = defaultdict(list)
    direct = set()
    for index, row in enumerate(endpoints):
        identity = _clock_identity(row)
        identity.pop("left_cohort")
        identity.pop("right_cohort")
        key = row["family"], json.dumps(identity, sort_keys=True)
        matched[key].append(index)
        if (_boolean(row["original_point_outside_margin"], "original")
                or _boolean(row["repeated_point_outside_margin"], "repeated")
                or row["midpoint_point_outside_margin"]):
            direct.add(index)
    selected = set(direct)
    for members in matched.values():
        if direct.intersection(members):
            pairs = [tuple(_clock_identity(endpoints[index])[k] for k in ("left_cohort", "right_cohort"))
                     for index in members]
            require(len(pairs) == len(ADJACENT_CLOCK_PAIRS) and set(pairs) == set(ADJACENT_CLOCK_PAIRS),
                    "COMPLETE_MATCHING_ADJACENT_CLOCK_FOCUS_REQUIRED")
            selected.update(members)
    ordered = sorted(selected)
    return ordered, dict(point_flagged_endpoint_count=len(direct), focus_endpoint_count=len(ordered),
        added_adjacent_clock_match_count=len(selected-direct),
        point_flagged_domain_row_indices=[int(endpoints[i]["domain_row_index"]) for i in sorted(direct)],
        focus_domain_row_indices=[int(endpoints[i]["domain_row_index"]) for i in ordered],
        family_counts=dict(Counter(endpoints[i]["family"] for i in ordered)),
        selection="Any original, repeated or midpoint point outside its frozen margin; matching identity at both adjacent clock pairs.",
        selection_is_inferential=False, outliers_removed=False)


def write_focus_blocks(path: Path, endpoints: Sequence[Row], stats: Mapping[str, np.ndarray],
                       selected: Sequence[int], direct_ids: Sequence[int]) -> int:
    """Write all 32 blocks for each selected endpoint; an empty focus retains its header."""
    metadata = ("domain_row_index", "family", "endpoint_identity", "definition_status", "margin", "units",
                "original_estimate", "repeated_estimate", "estimate_midpoint",
                "original_point_outside_margin", "repeated_point_outside_margin", "midpoint_point_outside_margin")
    fields = metadata+("focus_reason", "source_rng_seed", "original_score", "repeated_score", "within_block_midpoint",
                       "repeated_minus_original", "paired_auxiliary_squared_change", "squared_change_share")
    direct = set(direct_ids)
    count = 0
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in selected:
            endpoint = endpoints[index]
            base = {key: endpoint[key] for key in metadata}
            base["focus_reason"] = ("POINT_OUTSIDE_FROZEN_MARGIN" if int(endpoint["domain_row_index"]) in direct
                                    else "ADJACENT_CLOCK_MATCH_TO_POINT_OUTSIDE_FROZEN_MARGIN")
            total = stats["squared_change_total"][index]
            for block, seed in enumerate(SEEDS):
                half_squared = float(stats["paired_auxiliary_squared_change"][index, block])
                writer.writerow(dict(base, source_rng_seed=seed,
                    original_score=float(stats["original_scores"][index, block]),
                    repeated_score=float(stats["repeated_scores"][index, block]),
                    within_block_midpoint=float(stats["within_block_midpoint"][index, block]),
                    repeated_minus_original=float(stats["repeated_minus_original"][index, block]),
                    paired_auxiliary_squared_change=half_squared,
                    squared_change_share=half_squared*2/total if total > 0 else None))
                count += 1
    return count


def scientific_limits() -> dict[str, bool]:
    return dict(confidence_intervals=False, uncertainty_coverage_qualified=False, inference_qualified=False,
        power_qualified=False, sample_size_qualified=False, scientific_fidelity_qualified=False,
        confirmation_execution_released=False, numerical_margins_changed=False,
        numerical_bias_established=False, zero_population_uncertainty_established=False,
        negative_between_estimates_clipped=False, noise_fraction_computed=False, covariance_pseudovalues_used=False,
        conditional_equal_law_verified=False, conditional_independence_verified=False,
        representative_iid_original_blocks_verified=False, confirmation_planning_changed=False)


def run(config_path: Path, project_root: Path, output: Path) -> Row:
    started = time.perf_counter()
    config_path, project_root, output = Path(config_path), Path(project_root), Path(output)
    require(not output.exists(), "NEW_AUXILIARY_VARIANCE_OUTPUT_REQUIRED")
    settings = configuration(config_path)
    source, source_report = bound_analysis(project_root, settings["comparison"])
    require(source_report["status"] == "COMPLETE_DESCRIPTIVE_PFC_REFINEMENT_REPEAT_COMPARISON"
            and source_report["row_count"] == FULL_ENDPOINT_COUNT
            and source_report["primitive_endpoint_count"] == sum(PRIMITIVE_COUNTS.values())
            and source_report["source_case_count"] == 192
            and source_report["original_independent_blocks"] == 32 and source_report["auxiliary_realizations"] == 2
            and source_report["new_original_independent_blocks"] == source_report["production_episodes"] == 0
            and all(source_report[key] is False for key in (
                "confidence_intervals", "confirmation_execution_released", "scientific_fidelity_qualified",
                "numerical_margins_changed", "uncertainty_coverage_qualified")),
            "COMPLETE_UNCHANGED_DEVELOPMENT_REPEAT_COMPARISON_REQUIRED")
    csv_path = output_file(source, source_report, "auxiliary_repeat_comparison.csv")
    array_path = output_file(source, source_report, "primitive_nested_scores.npz")
    comparison_rows = read_rows(csv_path)
    with np.load(array_path, allow_pickle=False) as stored:
        saved = {name: stored[name].copy() for name in stored.files}
    primitives, stats, errors = validate_primitive_bank(comparison_rows, saved)
    included_counts = Counter(row["family"] for row in primitives)
    inclusion_counts = Counter(_inclusion(row) for row in comparison_rows)
    require(len(comparison_rows) == FULL_ENDPOINT_COUNT and dict(included_counts) == PRIMITIVE_COUNTS
            and inclusion_counts == {"INCLUDED_PRIMITIVE": sum(PRIMITIVE_COUNTS.values()),
                "EXCLUDED_DEFINED_COVARIANCE": DEFINED_COVARIANCE_COUNT,
                "EXCLUDED_FROZEN_UNDEFINED_COVARIANCE": UNDEFINED_COVARIANCE_COUNT},
            "COMPLETE_CANONICAL_PRIMITIVE_AND_EXCLUDED_DENOMINATORS_REQUIRED")
    endpoints = endpoint_rows(primitives, stats)
    groups = group_count_rows(comparison_rows, endpoints)
    require(sum(row["endpoint_count"] for row in groups) == FULL_ENDPOINT_COUNT,
            "COMPLETE_GROUP_COUNT_DENOMINATOR_REQUIRED")
    selected, focus = focus_endpoints(endpoints)
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output/"auxiliary_variance_endpoints.csv", endpoints)
    write_rows(output/"auxiliary_variance_group_counts.csv", groups)
    np.savez_compressed(output/"paired_auxiliary_squared_change.npz", source_rng_seeds=saved["source_rng_seeds"],
        domain_row_indices=saved["domain_row_indices"],
        paired_auxiliary_squared_change=stats["paired_auxiliary_squared_change"])
    focus["block_rows"] = write_focus_blocks(output/"focus_endpoint_blocks.csv", endpoints, stats,
                                              selected, focus["point_flagged_domain_row_indices"])
    require(focus["block_rows"] == len(selected)*len(SEEDS), "COMPLETE_FOCUS_BLOCKS_REQUIRED")
    names = ("auxiliary_variance_endpoints.csv", "auxiliary_variance_group_counts.csv",
             "paired_auxiliary_squared_change.npz", "focus_endpoint_blocks.csv")
    report = dict(schema=SCHEMA, status=STATUS, config_path=str(config_path), config_sha256=sha256_file(config_path),
        settings=settings, implementation_sha256=sha256_file(Path(__file__)),
        implementation_dependencies={name: sha256_file(Path(__file__).with_name(name)) for name in (
            "pfc_coupled_clock_comparison.py", "pfc_coupled_precision_comparison.py", "pfc_delay_analysis.py", "provenance.py")},
        comparison=settings["comparison"],
        input_outputs={name: source_report["outputs"][name] for name in (
            "auxiliary_repeat_comparison.csv", "primitive_nested_scores.npz")},
        original_independent_blocks=len(SEEDS), source_rng_seeds=SEEDS, auxiliary_realizations=2,
        new_original_independent_blocks=0, source_simulations=0, new_rng_draws=0, production_episodes=0,
        bound_repeat_source_case_count=source_report["source_case_count"], maximum_confirmation_blocks=128,
        row_count=len(endpoints), full_input_endpoint_count=len(comparison_rows),
        included_primitive_family_counts=dict(included_counts), inclusion_counts=dict(inclusion_counts),
        input_definition_status_counts=dict(Counter(row["definition_status"] for row in comparison_rows)),
        raw_negative_between_count=sum(row["between_block_estimate_negative"] for row in endpoints),
        observed_zero_within_variance_count=sum(row["observed_zero_within_variance"] for row in endpoints),
        observed_zero_auxiliary_change_count=sum(row["observed_zero_auxiliary_change"] for row in endpoints),
        undefined_concentration_count=sum(row["maximum_squared_change_share"] is None for row in endpoints),
        original_point_outside_count=sum(_boolean(row["original_point_outside_margin"], "original") for row in endpoints),
        repeated_point_outside_count=sum(_boolean(row["repeated_point_outside_margin"], "repeated") for row in endpoints),
        midpoint_point_outside_count=sum(row["midpoint_point_outside_margin"] for row in endpoints),
        focus=focus, numeric_identity_check=dict(rtol=IDENTITY_RTOL, atol=IDENTITY_ATOL,
            maximum_absolute_errors=errors, tolerance_is_scientific_margin=False),
        formulas=dict(midpoint="M=(A+B)/2", paired_change="D=B-A", within="W_hat=mean(D**2)/2",
            midpoint_variance="s_M2=sample_var(M,ddof=1)", between="between_hat=s_M2-W_hat/2",
            concentration="max(D_i**2)/sum(D_i**2); undefined when the observed denominator is zero"),
        fixed_block_interpretation="Under equal-law conditionally independent A/B and independent auxiliary streams across fixed original blocks, E[W_hat]=mean_i(v_i) and E[between_hat]=sample_var(mu_i,ddof=1), allowing heterogeneous v_i.",
        population_interpretation="A population variance interpretation additionally requires representative IID original blocks; this diagnostic does not establish that assumption.",
        interpretation=settings["interpretation"], assumptions=settings["assumptions"],
        limitations=["Two auxiliary realizations remain nested in 32 original blocks; no independent block count is increased.",
            "Negative between estimates are retained raw; no bounded noise fraction is defined.",
            "Zero observed squared change does not rule out rare events or establish zero population uncertainty.",
            "Exact nonzero-change counts include floating-point residuals and are not event-prevalence estimates.",
            "All point flags use saved canonical A/B/M points and unchanged margins after numerical identity checks.",
            "Focus rows are a descriptive view; every primitive remains in the complete endpoint table and no block is deleted.",
            "Covariance pseudovalues and frozen undefined covariance endpoints are excluded from this decomposition.",
            "No confidence interval, power, sample-size, convergence, bias, fidelity qualification or confirmation release follows from these moments."],
        saved_pair_array_reference=dict(file="primitive_nested_scores.npz",
            sha256=source_report["outputs"]["primitive_nested_scores.npz"]["sha256"],
            keys=["original_scores", "repeated_scores", "within_block_midpoint", "repeated_minus_original"],
            new_array="paired_auxiliary_squared_change.npz", new_array_value="(B-A)**2/2",
            variance_units="Squared original endpoint units, specified per row in auxiliary_variance_endpoints.csv"),
        outputs={name: dict(bytes=(output/name).stat().st_size, sha256=sha256_file(output/name)) for name in names},
        wall_seconds=time.perf_counter()-started, **scientific_limits())
    freeze_json(output/"report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "project-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config, args.project_root, args.output)
    print(json.dumps({key: report[key] for key in ("status", "row_count", "raw_negative_between_count", "wall_seconds")}))


if __name__ == "__main__":
    main()
