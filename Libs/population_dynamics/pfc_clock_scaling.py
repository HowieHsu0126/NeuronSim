"""Describe joint-clock first-order curvature on existing nested PFC draws.

R = d_coarse_medium - 2*d_medium_fine is a first-order weak-error diagnostic,
not a fitted truth or a convergence test. Adjacent clocks share blocks and
auxiliary realizations, so their observed covariance is retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from . import pfc_auxiliary_variance as auxiliary
from .pfc_coupled_precision_comparison import SEEDS, UNDEFINED, bound_analysis, output_file, read_rows
from .pfc_delay_analysis import require, write_rows
from .provenance import freeze_json, sha256_file

SCHEMA = "pfc_clock_scaling_diagnostic_v1"
STATUS = "COMPLETE_DESCRIPTIVE_PFC_JOINT_CLOCK_SCALING_DIAGNOSTIC"
FIRST_ORDER_MULTIPLIER = 2
SCORE_KEYS = ("original_scores", "repeated_scores", "within_block_midpoint", "repeated_minus_original")
MOMENT_KEYS = ("auxiliary_within_variance_estimate", "midpoint_sample_variance", "between_block_variance_estimate")
SOURCE_KEYS = ("domain_row_index", "endpoint_identity", "definition_status", "margin",
    "original_estimate", "repeated_estimate", "estimate_change", "estimate_midpoint",
    "original_point_outside_margin", "repeated_point_outside_margin", "midpoint_point_outside_margin",
    "original_point_margin_position", "repeated_point_margin_position", "midpoint_point_margin_position")+MOMENT_KEYS
PREFIXES = ("coarse_medium", "medium_fine")
Row = dict[str, Any]


def configuration(path: Path) -> Row:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(settings, dict) and settings.get("schema") == SCHEMA, "CLOCK_SCALING_SCHEMA_REQUIRED")
    require(settings.get("weak_order_hypothesis") == 1 and settings.get("maximum_confirmation_blocks") == 128
            and settings.get("source_simulations") == settings.get("new_rng_draws") == 0
            and all(settings.get(name) is False for name in (
                "confidence_intervals", "confirmation_execution_released", "scientific_fidelity_qualified")),
            "UNCHANGED_DATA_ONLY_FIRST_ORDER_CLOCK_SCALING_SCOPE_REQUIRED")
    for name in ("comparison", "auxiliary_variance"):
        reference = settings.get(name)
        require(isinstance(reference, dict) and isinstance(reference.get("path"), str)
                and isinstance(reference.get("report_sha256"), str), "BOUND_CLOCK_SCALING_INPUT_REQUIRED:"+name)
    require(isinstance(settings.get("interpretation"), str) and settings["interpretation"].strip()
            and isinstance(settings.get("assumptions"), list) and settings["assumptions"]
            and all(isinstance(item, str) and item.strip() for item in settings["assumptions"]),
            "EXPLICIT_CLOCK_SCALING_ASSUMPTIONS_REQUIRED")
    return settings


def check_bound_reports(settings: Mapping[str, Any], comparison: Mapping[str, Any],
                        variance: Mapping[str, Any], helper_sha256: str) -> None:
    """Reuse accepted denominator checks and bind their exact input and helper."""
    inclusion = {"INCLUDED_PRIMITIVE": sum(auxiliary.PRIMITIVE_COUNTS.values()),
        "EXCLUDED_DEFINED_COVARIANCE": auxiliary.DEFINED_COVARIANCE_COUNT,
        "EXCLUDED_FROZEN_UNDEFINED_COVARIANCE": auxiliary.UNDEFINED_COVARIANCE_COUNT}
    require(comparison.get("status") == "COMPLETE_DESCRIPTIVE_PFC_REFINEMENT_REPEAT_COMPARISON"
            and comparison.get("row_count") == auxiliary.FULL_ENDPOINT_COUNT
            and comparison.get("primitive_endpoint_count") == inclusion["INCLUDED_PRIMITIVE"]
            and comparison.get("original_independent_blocks") == 32 and comparison.get("auxiliary_realizations") == 2
            and comparison.get("source_case_count") == 192, "COMPLETE_BOUND_REPEAT_COMPARISON_REQUIRED")
    require(variance.get("schema") == auxiliary.SCHEMA and variance.get("status") == auxiliary.STATUS
            and variance.get("row_count") == inclusion["INCLUDED_PRIMITIVE"]
            and variance.get("full_input_endpoint_count") == auxiliary.FULL_ENDPOINT_COUNT
            and variance.get("inclusion_counts") == inclusion
            and variance.get("included_primitive_family_counts") == auxiliary.PRIMITIVE_COUNTS
            and variance.get("original_independent_blocks") == 32 and variance.get("source_rng_seeds") == SEEDS
            and variance.get("auxiliary_realizations") == 2 and variance.get("maximum_confirmation_blocks") == 128,
            "COMPLETE_BOUND_AUXILIARY_VARIANCE_DENOMINATOR_REQUIRED")
    require(variance.get("comparison") == settings["comparison"]
            and variance["input_outputs"]["primitive_nested_scores.npz"] == comparison["outputs"]["primitive_nested_scores.npz"]
            and variance.get("implementation_sha256") == helper_sha256,
            "ACCEPTED_VARIANCE_SCORE_OR_IMPLEMENTATION_BINDING_CHANGED")
    for report in (comparison, variance):
        require(report.get("new_original_independent_blocks") == report.get("production_episodes") == 0
                and all(report.get(name) is False for name in (
                    "confidence_intervals", "confirmation_execution_released", "scientific_fidelity_qualified",
                    "numerical_margins_changed", "uncertainty_coverage_qualified")),
                "UNCHANGED_BOUND_DEVELOPMENT_EVIDENCE_REQUIRED")
    require(variance.get("source_simulations") == variance.get("new_rng_draws") == 0
            and variance.get("covariance_pseudovalues_used") is False,
            "PRIMITIVE_ONLY_EXISTING_DATA_DIAGNOSTIC_REQUIRED")


def pair_endpoint_rows(rows: Sequence[Row], saved: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return position pairs matching family and the complete non-clock identity."""
    require(set(saved) == auxiliary.SAVED_KEYS, "EXACT_ACCEPTED_NESTED_SCORE_ARRAYS_REQUIRED")
    seeds, ids = np.asarray(saved["source_rng_seeds"]), np.asarray(saved["domain_row_indices"])
    require(seeds.ndim == 1 and seeds.dtype.kind in "iu" and np.array_equal(seeds, SEEDS),
            "CANONICAL_SOURCE_RNG_SEED_ORDER_REQUIRED")
    require(ids.ndim == 1 and ids.dtype.kind in "iu" and len(ids) == len(rows) and len(rows) > 0
            and np.all(ids >= 0) and np.all(ids < auxiliary.FULL_ENDPOINT_COUNT) and np.all(ids[1:] > ids[:-1]),
            "ORDERED_UNIQUE_PRIMITIVE_DOMAIN_IDS_REQUIRED")
    for name in SCORE_KEYS:
        values = np.asarray(saved[name])
        require(values.shape == (len(rows), len(SEEDS)) and values.dtype.kind in "fiu" and np.isfinite(values).all(),
                "FINITE_MATCHED_ACCEPTED_SCORE_ARRAY_REQUIRED:"+name)
    paired = {}
    for index, row in enumerate(rows):
        require(str(row["domain_row_index"]) == str(ids[index]), "ACCEPTED_ENDPOINT_SCORE_AXIS_MISMATCH")
        require(row["family"] in auxiliary.PRIMITIVE_COUNTS and row["definition_status"] != UNDEFINED
                and row["auxiliary_scatter_kind"] == auxiliary.PRIMITIVE_KIND,
                "COVARIANCE_AND_UNDEFINED_ENDPOINTS_EXCLUDED")
        require(str(row["original_independent_blocks"]) == "32" and str(row["auxiliary_realizations"]) == "2",
                "UNCHANGED_NESTED_ORIGINAL_BLOCKS_REQUIRED")
        identity = auxiliary._clock_identity(row)
        clock_pair = (identity.pop("left_cohort"), identity.pop("right_cohort"))
        pair_index = auxiliary.ADJACENT_CLOCK_PAIRS.index(clock_pair)
        key = row["family"], json.dumps(identity, sort_keys=True)
        members = paired.setdefault(key, {})
        require(pair_index not in members, "DUPLICATE_CLOCK_ENDPOINT_IDENTITY")
        members[pair_index] = index
        expected_units = "probability" if row["family"] == "count_cdf" else "Hz"
        require(row["units"] == expected_units and row["variance_units"] == expected_units+"^2",
                "MATCHED_SOURCE_UNITS_REQUIRED")
        margin = float(row["margin"])
        require(np.isfinite(margin) and margin >= 0, "UNCHANGED_FINITE_SOURCE_MARGIN_REQUIRED")
        for name in ("original_point_outside_margin", "repeated_point_outside_margin", "midpoint_point_outside_margin"):
            auxiliary._boolean(row[name], name)
    require(all(set(members) == {0, 1} for members in paired.values()), "COMPLETE_MATCHING_ADJACENT_CLOCK_PAIRS_REQUIRED")
    pairs = np.asarray(sorted((members[0], members[1]) for members in paired.values()), dtype=np.int64)
    require(pairs.shape == (len(rows)//2, 2) and pairs.size == len(rows), "COMPLETE_PAIRED_PRIMITIVE_DENOMINATOR_REQUIRED")
    return pairs


def _identity_check(actual: np.ndarray, expected: np.ndarray, name: str) -> float:
    require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
            and np.allclose(actual, expected, rtol=auxiliary.IDENTITY_RTOL, atol=auxiliary.IDENTITY_ATOL),
            "JOINT_CLOCK_MOMENT_IDENTITY_CHANGED:"+name)
    return float(np.max(np.abs(actual-expected)))


def joint_clock_moments(coarse_medium: Mapping[str, np.ndarray], medium_fine: Mapping[str, np.ndarray]
                        ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Compute new curvature and cross moments from accepted marginal moments.

    Equal-law independent auxiliary vectors A/B may have arbitrary dependence
    between adjacent clocks within either vector.
    """
    cm, mf = coarse_medium, medium_fine
    first = np.asarray(cm["original_scores"])
    require(first.ndim == 2 and first.shape[0] > 0 and first.shape[1] >= 2,
            "JOINT_CLOCK_ENDPOINT_BY_BLOCK_ARRAYS_REQUIRED")
    for clock in (cm, mf):
        for name in SCORE_KEYS+MOMENT_KEYS:
            values = np.asarray(clock[name])
            expected_shape = first.shape if name in SCORE_KEYS else (len(first),)
            require(values.shape == expected_shape and values.dtype.kind in "fiu" and np.isfinite(values).all(),
                    "FINITE_MATCHED_JOINT_CLOCK_ARRAY_REQUIRED:"+name)
    with np.errstate(over="ignore", invalid="ignore"):
        r_a = cm["original_scores"]-FIRST_ORDER_MULTIPLIER*mf["original_scores"]
        r_b = cm["repeated_scores"]-FIRST_ORDER_MULTIPLIER*mf["repeated_scores"]
    stats = auxiliary.decompose_auxiliary_pairs(r_a, r_b)
    cm_m, mf_m = cm["within_block_midpoint"], mf["within_block_midpoint"]
    cm_d, mf_d = cm["repeated_minus_original"], mf["repeated_minus_original"]
    with np.errstate(over="ignore", invalid="ignore"):
        cross_within = (cm_d*mf_d).mean(axis=1)/2
        midpoint_covariance = ((cm_m-cm_m.mean(axis=1, keepdims=True))
                               *(mf_m-mf_m.mean(axis=1, keepdims=True))).sum(axis=1)/(first.shape[1]-1)
        between_covariance = midpoint_covariance-cross_within/2
    require(all(np.isfinite(value).all() for value in (cross_within, midpoint_covariance, between_covariance)),
            "FINITE_JOINT_CLOCK_CROSS_MOMENTS_REQUIRED")
    multiplier = FIRST_ORDER_MULTIPLIER
    expectations = {
        "within_block_midpoint": cm_m-multiplier*mf_m,
        "repeated_minus_original": cm_d-multiplier*mf_d,
        "auxiliary_within_variance_estimate": cm[MOMENT_KEYS[0]]+multiplier**2*mf[MOMENT_KEYS[0]]-2*multiplier*cross_within,
        "midpoint_sample_variance": cm[MOMENT_KEYS[1]]+multiplier**2*mf[MOMENT_KEYS[1]]-2*multiplier*midpoint_covariance,
        "between_block_variance_estimate": cm[MOMENT_KEYS[2]]+multiplier**2*mf[MOMENT_KEYS[2]]-2*multiplier*between_covariance,
    }
    errors = {name: _identity_check(stats[name], expected, name) for name, expected in expectations.items()}
    stats.update(cross_auxiliary_moment=cross_within, paired_midpoint_sample_covariance=midpoint_covariance,
                 raw_between_clock_covariance=between_covariance)
    return stats, errors


def analyse_pairs(rows: Sequence[Row], saved: Mapping[str, np.ndarray]
                  ) -> tuple[list[Row], dict[str, np.ndarray], np.ndarray, dict[str, float]]:
    pairs = pair_endpoint_rows(rows, saved)
    inputs = []
    for clock_index in range(2):
        positions = pairs[:, clock_index]
        clock = {name: saved[name][positions] for name in SCORE_KEYS}
        clock.update({name: np.asarray([float(rows[i][name]) for i in positions]) for name in MOMENT_KEYS})
        inputs.append(clock)
    stats, errors = joint_clock_moments(*inputs)
    for name, source_key in (("original_scores", "original_estimate"), ("repeated_scores", "repeated_estimate"),
                             ("within_block_midpoint", "estimate_midpoint"), ("repeated_minus_original", "estimate_change")):
        expected = np.asarray([float(rows[a][source_key])-FIRST_ORDER_MULTIPLIER*float(rows[b][source_key]) for a, b in pairs])
        errors[source_key] = _identity_check(stats[name].mean(axis=1), expected, source_key)
    seeds = np.asarray(SEEDS)
    endpoints = []
    for joint_index, (left, right) in enumerate(pairs):
        cm, mf = rows[left], rows[right]
        identity = auxiliary._clock_identity(cm)
        identity.pop("left_cohort")
        identity.pop("right_cohort")
        row = dict(joint_endpoint_index=joint_index, family=cm["family"],
            endpoint_identity=json.dumps(identity, sort_keys=True), units=cm["units"], variance_units=cm["variance_units"],
            original_independent_blocks=32, auxiliary_realizations=2)
        for prefix, source in zip(PREFIXES, (cm, mf)):
            row.update({prefix+"_"+name: source[name] for name in SOURCE_KEYS})
        for prefix, name in (("original", "original_scores"), ("repeated", "repeated_scores"),
                             ("midpoint", "within_block_midpoint"), ("change", "repeated_minus_original")):
            row["curvature_"+prefix+"_estimate"] = float(stats[name][joint_index].mean())
        for name in MOMENT_KEYS:
            row["curvature_"+name] = float(stats[name][joint_index])
        for name in ("cross_auxiliary_moment", "paired_midpoint_sample_covariance", "raw_between_clock_covariance"):
            row[name] = float(stats[name][joint_index])
        row["curvature_between_block_estimate_negative"] = bool(stats["between_block_estimate_negative"][joint_index])
        row["curvature_observed_zero_within_variance"] = bool(stats["observed_zero_within_variance"][joint_index])
        row["curvature_exact_nonzero_auxiliary_change_blocks"] = int(stats["exact_nonzero_auxiliary_change_blocks"][joint_index])
        row["curvature_observed_zero_auxiliary_change"] = row["curvature_exact_nonzero_auxiliary_change_blocks"] == 0
        half_squared = stats["paired_auxiliary_squared_change"][joint_index]
        row["curvature_contributing_source_rng_seeds"] = json.dumps(seeds[half_squared > 0].tolist())
        if stats["squared_change_total"][joint_index] > 0:
            row["curvature_concentration_status"] = "DEFINED_OBSERVED_SQUARED_CHANGE_CONCENTRATION"
            row["curvature_maximum_squared_change_share"] = float(stats["maximum_squared_change_share"][joint_index])
            row["curvature_maximum_share_source_rng_seeds"] = json.dumps(seeds[half_squared == half_squared.max()].tolist())
        else:
            row["curvature_concentration_status"] = "UNDEFINED_ZERO_OBSERVED_SQUARED_CHANGE"
            row["curvature_maximum_squared_change_share"] = None
            row["curvature_maximum_share_source_rng_seeds"] = None
        row["focus_source_point_outside_margin"] = any(auxiliary._boolean(source[name], name)
            for source in (cm, mf) for name in (
                "original_point_outside_margin", "repeated_point_outside_margin", "midpoint_point_outside_margin"))
        endpoints.append(row)
    return endpoints, stats, pairs, errors


def family_count_rows(endpoints: Sequence[Row]) -> list[Row]:
    result = []
    for family in sorted({row["family"] for row in endpoints}):
        members = [row for row in endpoints if row["family"] == family]
        result.append(dict(family=family, joint_endpoint_count=len(members), source_primitive_endpoint_count=2*len(members),
            focus_endpoint_count=sum(row["focus_source_point_outside_margin"] for row in members),
            raw_negative_between_count=sum(row["curvature_between_block_estimate_negative"] for row in members),
            observed_zero_within_variance_count=sum(row["curvature_observed_zero_within_variance"] for row in members),
            observed_zero_auxiliary_change_count=sum(row["curvature_observed_zero_auxiliary_change"] for row in members),
            undefined_concentration_count=sum(row["curvature_maximum_squared_change_share"] is None for row in members)))
    return result


def write_focus_blocks(path: Path, endpoints: Sequence[Row], saved: Mapping[str, np.ndarray],
                       stats: Mapping[str, np.ndarray], pairs: np.ndarray) -> tuple[list[Row], int]:
    focus = [row for row in endpoints if row["focus_source_point_outside_margin"]]
    metadata = ("joint_endpoint_index", "family", "endpoint_identity", "units",
        "coarse_medium_domain_row_index", "medium_fine_domain_row_index",
        "coarse_medium_margin", "medium_fine_margin")
    flags = tuple(prefix+"_"+name for prefix in PREFIXES for name in (
        "original_point_outside_margin", "repeated_point_outside_margin", "midpoint_point_outside_margin"))
    fields = metadata+flags+("source_rng_seed",)+tuple(prefix+"_"+name for prefix in PREFIXES for name in SCORE_KEYS)+("R_A", "R_B", "M_R", "D_R")
    count = 0
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for endpoint in focus:
            index = endpoint["joint_endpoint_index"]
            base = {name: endpoint[name] for name in metadata+flags}
            for block, seed in enumerate(SEEDS):
                row = dict(base, source_rng_seed=seed)
                for prefix, position in zip(PREFIXES, pairs[index]):
                    row.update({prefix+"_"+name: float(saved[name][position, block]) for name in SCORE_KEYS})
                row.update({label: float(stats[name][index, block]) for label, name in zip(("R_A", "R_B", "M_R", "D_R"), SCORE_KEYS)})
                writer.writerow(row)
                count += 1
    return focus, count


def scientific_limits() -> dict[str, bool]:
    return dict(auxiliary.scientific_limits(), weak_order_qualified=False, asymptotic_regime_qualified=False,
        curvature_threshold_defined=False, curvature_ratio_computed=False, cross_clock_independence_assumed=False,
        fitted_truth_generated=False, extrapolated_truth_generated=False)


def run(config_path: Path, project_root: Path, output: Path) -> Row:
    started = time.perf_counter()
    config_path, project_root, output = Path(config_path), Path(project_root), Path(output)
    require(not output.exists(), "NEW_CLOCK_SCALING_OUTPUT_REQUIRED")
    settings = configuration(config_path)
    comparison_path, comparison = bound_analysis(project_root, settings["comparison"])
    variance_path, variance = bound_analysis(project_root, settings["auxiliary_variance"])
    helper_sha256 = sha256_file(Path(auxiliary.__file__))
    check_bound_reports(settings, comparison, variance, helper_sha256)
    table_path = output_file(variance_path, variance, "auxiliary_variance_endpoints.csv")
    array_path = output_file(comparison_path, comparison, "primitive_nested_scores.npz")
    rows = read_rows(table_path)
    require(len(rows) == sum(auxiliary.PRIMITIVE_COUNTS.values())
            and dict(Counter(row["family"] for row in rows)) == auxiliary.PRIMITIVE_COUNTS,
            "COMPLETE_ACCEPTED_PRIMITIVE_DENOMINATOR_REQUIRED")
    with np.load(array_path, allow_pickle=False) as stored:
        saved = {name: stored[name].copy() for name in stored.files}
    endpoints, stats, pairs, errors = analyse_pairs(rows, saved)
    expected = {family: count//2 for family, count in auxiliary.PRIMITIVE_COUNTS.items()}
    require(dict(Counter(row["family"] for row in endpoints)) == expected and len(endpoints)*2 == len(rows),
            "COMPLETE_JOINT_CLOCK_IDENTITY_DENOMINATOR_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output/"clock_scaling_endpoints.csv", endpoints)
    write_rows(output/"clock_scaling_family_counts.csv", family_count_rows(endpoints))
    np.savez_compressed(output/"joint_curvature_blocks.npz", source_rng_seeds=saved["source_rng_seeds"],
        domain_row_index_pairs=saved["domain_row_indices"][pairs],
        **{label: stats[name] for label, name in zip(("R_A", "R_B", "M_R", "D_R"), SCORE_KEYS)})
    focus, block_count = write_focus_blocks(output/"focus_clock_scaling_blocks.csv", endpoints, saved, stats, pairs)
    require(block_count == len(focus)*len(SEEDS), "COMPLETE_FOCUS_ORIGINAL_BLOCKS_REQUIRED")
    names = ("clock_scaling_endpoints.csv", "clock_scaling_family_counts.csv", "joint_curvature_blocks.npz",
             "focus_clock_scaling_blocks.csv")
    report = dict(schema=SCHEMA, status=STATUS, config_path=str(config_path), config_sha256=sha256_file(config_path),
        settings=settings, comparison=settings["comparison"], auxiliary_variance=settings["auxiliary_variance"],
        implementation_sha256=sha256_file(Path(__file__)), auxiliary_variance_implementation_sha256=helper_sha256,
        implementation_dependencies={name: sha256_file(Path(__file__).with_name(name)) for name in (
            "pfc_coupled_precision_comparison.py", "pfc_delay_analysis.py", "provenance.py")},
        input_outputs=dict(auxiliary_variance_endpoints=variance["outputs"]["auxiliary_variance_endpoints.csv"],
            primitive_nested_scores=comparison["outputs"]["primitive_nested_scores.npz"]),
        prior_denominator_and_source_checks_reused=True, source_marginal_moments_recomputed=False,
        full_input_endpoint_count=variance["full_input_endpoint_count"], inclusion_counts=variance["inclusion_counts"],
        source_primitive_endpoint_count=len(rows), row_count=len(endpoints), joint_endpoint_family_counts=expected,
        unpaired_endpoint_count=0, duplicate_endpoint_count=0, original_independent_blocks=32, source_rng_seeds=SEEDS,
        auxiliary_realizations=2, new_original_independent_blocks=0, source_simulations=0, new_rng_draws=0,
        production_episodes=0, maximum_confirmation_blocks=128, weak_order_hypothesis=1,
        first_order_multiplier=FIRST_ORDER_MULTIPLIER, adjacent_clock_pairs=auxiliary.ADJACENT_CLOCK_PAIRS,
        raw_negative_between_count=sum(row["curvature_between_block_estimate_negative"] for row in endpoints),
        observed_zero_within_variance_count=sum(row["curvature_observed_zero_within_variance"] for row in endpoints),
        observed_zero_auxiliary_change_count=sum(row["curvature_observed_zero_auxiliary_change"] for row in endpoints),
        undefined_concentration_count=sum(row["curvature_maximum_squared_change_share"] is None for row in endpoints),
        focus=dict(endpoint_count=len(focus), block_rows=block_count,
            family_counts=dict(Counter(row["family"] for row in focus)), endpoint_summaries=focus,
            selection="Any original, repeated or midpoint source point outside its unchanged clock-specific margin.",
            curvature_used_for_selection=False, selection_is_inferential=False, outliers_removed=False),
        numeric_identity_check=dict(rtol=auxiliary.IDENTITY_RTOL, atol=auxiliary.IDENTITY_ATOL,
            maximum_absolute_errors=errors, tolerance_is_scientific_margin=False),
        formulas=dict(curvature="R=d_coarse_medium-2*d_medium_fine for each auxiliary realization",
            cross_auxiliary_moment="crossW=mean(D_cm*D_mf)/2",
            midpoint_covariance="sampleCov(M_cm,M_mf), denominator n-1",
            between_covariance="betweenCov=sampleCov(M_cm,M_mf)-crossW/2",
            within_variance="W_R=W_cm+4*W_mf-4*crossW",
            midpoint_variance="s_MR2=s_Mcm2+4*s_Mmf2-4*sampleCov(M_cm,M_mf)",
            between_variance="between_R=between_cm+4*between_mf-4*betweenCov"),
        conditional_interpretation="Under equal-law conditionally independent auxiliary vectors A/B and independent streams across the fixed original blocks, the raw between-curvature moment targets the sample variance of the fixed conditional curvature means. Adjacent clocks within each vector may be dependent.",
        interpretation=settings["interpretation"], assumptions=settings["assumptions"],
        limitations=["A first-order weak-error hypothesis motivates the multiplier; these observations do not establish that order or an asymptotic regime.",
            "The diagnostic does not fit, extrapolate or identify simulator truth, and does not define a curvature margin or pass/fail rule.",
            "Cross-clock auxiliary and midpoint covariance are retained; assuming independence would generally change curvature variance.",
            "The two auxiliary realizations remain nested in the same 32 original blocks. Population interpretation additionally requires representative IID original blocks.",
            "Raw negative between-curvature estimates are retained. Zero observed squared change does not rule out rare events or establish zero population uncertainty.",
            "Exact nonzero counts include floating residuals and are not event prevalence. No confidence, power, sample-size or qualification conclusion is made.",
            "All 24825 matched primitive identities remain included; the 32946 defined and 1734 frozen undefined covariance endpoints remain excluded.",
            "Focus is selected by pre-existing source point flags with separate original margins; no source block is deleted."],
        joint_array_axes=dict(rows="joint_endpoint_index in clock_scaling_endpoints.csv",
            columns="source_rng_seeds", domain_row_index_pairs=list(PREFIXES),
            R_A="original curvature", R_B="repeated curvature", M_R="within-block curvature midpoint", D_R="R_B-R_A"),
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
    print(json.dumps({name: report[name] for name in ("status", "row_count", "raw_negative_between_count", "wall_seconds")}))


if __name__ == "__main__":
    main()
