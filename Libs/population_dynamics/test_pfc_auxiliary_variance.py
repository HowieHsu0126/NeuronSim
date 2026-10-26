"""Raw two-auxiliary moments, fixed-block estimands and primitive-only boundaries."""
from itertools import product
import json

import numpy as np
import pytest
import yaml

from . import pfc_auxiliary_variance as diagnostic


def comparison_row(index, original, repeated, *, pair=0, family="source_response", undefined=False):
    a, b = float(original.mean()), float(repeated.mean())
    identity = dict(left_cohort=diagnostic.ADJACENT_CLOCK_PAIRS[pair][0],
        right_cohort=diagnostic.ADJACENT_CLOCK_PAIRS[pair][1], condition="cue", phase="cue")
    if family == "field_lag_covariance":
        identity.update(leading_field="0", trailing_field="0", lag_ms="100" if undefined else "0")
    else:
        identity["metric"] = "all_unit_rate_hz"
    covariance = family == "field_lag_covariance"
    row = dict(domain_row_index=str(index), family=family, endpoint_identity=json.dumps(identity, sort_keys=True),
        definition_status=diagnostic.UNDEFINED if undefined else "ESTIMATED_EMPIRICAL_SPREAD",
        margin="" if undefined else "0.5", units="Hz^2" if covariance else "Hz",
        original_estimate=str(a), repeated_estimate=str(b), estimate_change=str(b-a), estimate_midpoint=str((a+b)/2),
        original_point_outside_margin=str(abs(a) > .5), repeated_point_outside_margin=str(abs(b) > .5),
        same_nonzero_sign=str((a > 0 and b > 0) or (a < 0 and b < 0)),
        observed_block_mean_absolute_change=str(float(np.abs(repeated-original).mean())),
        observed_block_rms_change=str(float(np.sqrt(np.mean((repeated-original)**2)))),
        auxiliary_scatter_kind=diagnostic.COVARIANCE_KIND if covariance else diagnostic.PRIMITIVE_KIND,
        original_independent_blocks="32", auxiliary_realizations="2")
    if covariance:
        row.update({key: "" for key in diagnostic.SCATTER_KEYS})
    if undefined:
        row.update({key: "" for key in diagnostic.POINT_KEYS+diagnostic.POINT_FLAG_KEYS})
        row["auxiliary_scatter_kind"] = "UNDEFINED"
    return row


def fixture(original=None, repeated=None):
    if original is None:
        original = np.linspace(-.2, .2, 32)[None, :]
    if repeated is None:
        repeated = original+.01
    rows = [comparison_row(i, a, b, pair=i) for i, (a, b) in enumerate(zip(original, repeated))]
    saved = dict(source_rng_seeds=np.asarray(diagnostic.SEEDS), domain_row_indices=np.arange(len(rows)),
        original_scores=original.copy(), repeated_scores=repeated.copy(),
        within_block_midpoint=(original+repeated)/2, repeated_minus_original=repeated-original)
    return rows, saved


def test_exhaustive_heteroskedastic_draws_are_unbiased_for_fixed_block_means():
    # Each row is one of all 64 equally likely independent A/B two-point draws.
    # Unequal conditional variances are intentional; no IID-block assumption is used.
    means = np.array([-2., .5, 3.])
    scales = np.array([.25, 1.5, 2.])
    signs = np.asarray(list(product((-1., 1.), repeat=6)))
    a = means+signs[:, :3]*scales
    b = means+signs[:, 3:]*scales
    stats = diagnostic.decompose_auxiliary_pairs(a, b)
    mean_conditional_variance = np.mean(scales**2)
    fixed_block_variance = means.var(ddof=1)
    assert stats["auxiliary_within_variance_estimate"].mean() == pytest.approx(mean_conditional_variance)
    assert stats["midpoint_sample_variance"].mean() == pytest.approx(fixed_block_variance+mean_conditional_variance/2)
    assert stats["between_block_variance_estimate"].mean() == pytest.approx(fixed_block_variance)


def test_cancelling_changes_keep_raw_negative_between_estimate_and_all_blocks():
    a = np.tile([-1., 1.], 16)[None, :]
    rows, saved = fixture(a, -a)
    primitives, stats, errors = diagnostic.validate_primitive_bank(rows, saved)
    endpoint = diagnostic.endpoint_rows(primitives, stats)[0]
    assert float(endpoint["estimate_change"]) == 0
    assert endpoint["auxiliary_within_variance_estimate"] == 2
    assert endpoint["midpoint_sample_variance"] == 0
    assert endpoint["between_block_variance_estimate"] == -1
    assert endpoint["between_block_estimate_negative"] is True
    assert endpoint["exact_nonzero_auxiliary_change_blocks"] == 32
    assert endpoint["maximum_squared_change_share"] == pytest.approx(1/32)
    assert json.loads(endpoint["contributing_source_rng_seeds"]) == diagnostic.SEEDS
    assert all(error == 0 for error in errors.values())
    assert endpoint["original_independent_blocks"] == "32"
    assert endpoint["auxiliary_realizations"] == "2"


def test_zero_change_concentration_is_undefined_without_qualification():
    a = np.linspace(-1., 1., 32)[None, :]
    rows, saved = fixture(a, a)
    primitives, stats, _ = diagnostic.validate_primitive_bank(rows, saved)
    endpoint = diagnostic.endpoint_rows(primitives, stats)[0]
    assert endpoint["observed_zero_auxiliary_change"] is True
    assert endpoint["observed_zero_within_variance"] is True
    assert endpoint["between_block_variance_estimate"] > 0
    assert endpoint["maximum_squared_change_share"] is None
    assert endpoint["maximum_share_source_rng_seeds"] is None
    assert endpoint["concentration_status"] == "UNDEFINED_ZERO_OBSERVED_SQUARED_CHANGE"
    assert endpoint["contributing_source_rng_seeds"] == "[]"
    assert not any(diagnostic.scientific_limits().values())


def test_concentration_names_the_contributing_block_and_counts_floating_residuals():
    a = np.zeros((1, 32))
    b = a.copy()
    b[0, 2] = np.finfo(np.float64).eps
    rows, saved = fixture(a, b)
    primitives, stats, _ = diagnostic.validate_primitive_bank(rows, saved)
    endpoint = diagnostic.endpoint_rows(primitives, stats)[0]
    assert endpoint["exact_nonzero_auxiliary_change_blocks"] == 1
    assert endpoint["observed_zero_auxiliary_change"] is False
    assert endpoint["maximum_squared_change_share"] == 1
    assert json.loads(endpoint["contributing_source_rng_seeds"]) == [95003]
    assert json.loads(endpoint["maximum_share_source_rng_seeds"]) == [95003]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "reordered", "noninteger"])
def test_source_seed_axis_cannot_change(defect):
    rows, saved = fixture()
    if defect == "missing":
        saved["source_rng_seeds"] = saved["source_rng_seeds"][:-1]
    elif defect == "duplicate":
        saved["source_rng_seeds"][-1] = saved["source_rng_seeds"][0]
    elif defect == "reordered":
        saved["source_rng_seeds"] = saved["source_rng_seeds"][::-1]
    else:
        saved["source_rng_seeds"] = saved["source_rng_seeds"].astype(float)
    with pytest.raises(ValueError, match="CANONICAL_SOURCE_RNG_SEED_ORDER"):
        diagnostic.validate_primitive_bank(rows, saved)


@pytest.mark.parametrize("defect", ["wrong_shape", "nonfinite", "wrong_ids", "wrong_row_order", "missing_array"])
def test_malformed_primitive_bank_rejected(defect):
    rows, saved = fixture()
    if defect == "wrong_shape":
        saved["repeated_scores"] = saved["repeated_scores"][:, :-1]
    elif defect == "nonfinite":
        saved["original_scores"][0, 0] = np.nan
    elif defect == "wrong_ids":
        saved["domain_row_indices"][0] = 1
    elif defect == "wrong_row_order":
        rows[0]["domain_row_index"] = "1"
    else:
        del saved["within_block_midpoint"]
    with pytest.raises(ValueError):
        diagnostic.validate_primitive_bank(rows, saved)


@pytest.mark.parametrize("key", ["within_block_midpoint", "repeated_minus_original", "original_estimate",
                                  "estimate_midpoint", "observed_block_rms_change", "original_point_outside_margin"])
def test_saved_pair_points_and_flags_are_checked(key):
    rows, saved = fixture()
    if key in saved:
        saved[key][0, 4] += 1
    elif key == "original_point_outside_margin":
        rows[0][key] = "True"
    else:
        rows[0][key] = str(float(rows[0][key])+1)
    with pytest.raises(ValueError, match="IDENTITY_CHANGED|FLAG_CHANGED"):
        diagnostic.validate_primitive_bank(rows, saved)


@pytest.mark.parametrize("undefined", [False, True])
def test_covariance_and_undefined_rows_cannot_enter_primitive_array(undefined):
    rows, saved = fixture()
    a, b = saved["original_scores"][0], saved["repeated_scores"][0]
    rows.append(comparison_row(1, a, b, family="field_lag_covariance", undefined=undefined))
    # An otherwise valid covariance row remains excluded even if its numeric
    # pseudovalues are packaged with the primitive scores.
    saved["domain_row_indices"] = np.array([0, 1])
    for key in diagnostic.SAVED_KEYS-{"source_rng_seeds", "domain_row_indices"}:
        saved[key] = np.repeat(saved[key], 2, axis=0)
    with pytest.raises(ValueError, match="EXACT_PRIMITIVE_DOMAIN_AXIS"):
        diagnostic.validate_primitive_bank(rows, saved)


def test_covariance_cannot_be_relabelled_as_primitive_scatter():
    rows, saved = fixture()
    rows[0]["family"] = "field_lag_covariance"
    with pytest.raises(ValueError, match="COVARIANCE_BLOCK_SCATTER_FORBIDDEN"):
        diagnostic.validate_primitive_bank(rows, saved)


@pytest.mark.parametrize("a,b", [([1., 2.], [1., 2.]), ([[1.]], [[1.]]),
                                  ([[1., 2.]], [[1., 2., 3.]]), ([[1., np.inf]], [[1., 2.]])])
def test_math_helper_rejects_invalid_pairs(a, b):
    with pytest.raises(ValueError):
        diagnostic.decompose_auxiliary_pairs(np.asarray(a), np.asarray(b))


def test_focus_contains_adjacent_pair_and_every_original_block(tmp_path):
    a = np.vstack([np.full(32, .4), np.full(32, .05)])
    b = np.vstack([np.full(32, .6), np.full(32, .05)])
    rows, saved = fixture(a, b)
    primitives, stats, _ = diagnostic.validate_primitive_bank(rows, saved)
    endpoints = diagnostic.endpoint_rows(primitives, stats)
    selected, summary = diagnostic.focus_endpoints(endpoints)
    assert selected == [0, 1]
    assert summary["point_flagged_domain_row_indices"] == [0]
    assert summary["added_adjacent_clock_match_count"] == 1
    assert endpoints[0]["original_point_outside_margin"] == "False"
    assert endpoints[0]["repeated_point_outside_margin"] == "True"
    assert endpoints[0]["margin"] == "0.5"
    path = tmp_path/"focus.csv"
    count = diagnostic.write_focus_blocks(path, endpoints, stats, selected, summary["point_flagged_domain_row_indices"])
    assert count == 64
    written = diagnostic.read_rows(path)
    assert [int(row["source_rng_seed"]) for row in written[:32]] == diagnostic.SEEDS
    assert [int(row["source_rng_seed"]) for row in written[32:]] == diagnostic.SEEDS
    assert all(row["squared_change_share"] == "" for row in written[32:])
    assert float(written[0]["paired_auxiliary_squared_change"]) == pytest.approx(.02)
    assert all(row["focus_reason"] == "ADJACENT_CLOCK_MATCH_TO_POINT_OUTSIDE_FROZEN_MARGIN" for row in written[32:])


def test_missing_matching_adjacent_focus_is_not_silently_dropped():
    rows, saved = fixture(np.full((1, 32), .6), np.full((1, 32), .6))
    primitives, stats, _ = diagnostic.validate_primitive_bank(rows, saved)
    with pytest.raises(ValueError, match="COMPLETE_MATCHING_ADJACENT_CLOCK_FOCUS"):
        diagnostic.focus_endpoints(diagnostic.endpoint_rows(primitives, stats))


def test_empty_focus_has_valid_csv_header(tmp_path):
    rows, saved = fixture()
    primitives, stats, _ = diagnostic.validate_primitive_bank(rows, saved)
    endpoints = diagnostic.endpoint_rows(primitives, stats)
    selected, summary = diagnostic.focus_endpoints(endpoints)
    assert selected == [] and summary["focus_endpoint_count"] == 0
    path = tmp_path/"empty.csv"
    assert diagnostic.write_focus_blocks(path, endpoints, stats, selected, []) == 0
    assert diagnostic.read_rows(path) == []
    assert "source_rng_seed" in path.read_text().splitlines()[0]


def test_group_counts_include_excluded_rows_without_covariance_statistics():
    rows, saved = fixture()
    a, b = saved["original_scores"][0], saved["repeated_scores"][0]
    rows.append(comparison_row(1, a, b, family="field_lag_covariance"))
    rows.append(comparison_row(2, a, b, family="field_lag_covariance", undefined=True))
    primitives, stats, _ = diagnostic.validate_primitive_bank(rows, saved)
    groups = diagnostic.group_count_rows(rows, diagnostic.endpoint_rows(primitives, stats))
    assert sum(row["endpoint_count"] for row in groups) == 3
    excluded = [row for row in groups if row["inclusion"].startswith("EXCLUDED")]
    assert len(excluded) == 2
    assert all(row["raw_negative_between_count"] == row["observed_zero_within_variance_count"] == "" for row in excluded)


def config_fixture(comparison):
    return dict(schema=diagnostic.SCHEMA, comparison=comparison, maximum_confirmation_blocks=128,
        source_simulations=0, new_rng_draws=0, confidence_intervals=False,
        confirmation_execution_released=False, scientific_fidelity_qualified=False,
        interpretation="Fixed original blocks, descriptive moments only.",
        assumptions=["Equal conditional A/B laws and independent auxiliary streams; population interpretation also requires representative IID blocks."])


@pytest.mark.parametrize("key,value", [("maximum_confirmation_blocks", 129), ("source_simulations", 1),
    ("new_rng_draws", 1), ("confidence_intervals", True), ("confirmation_execution_released", True),
    ("scientific_fidelity_qualified", True)])
def test_config_cannot_promote_diagnostic_scope(tmp_path, key, value):
    settings = config_fixture(dict(path="source", report_sha256="fixture"))
    settings[key] = value
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="UNCHANGED_DATA_ONLY"):
        diagnostic.configuration(path)


def test_run_rejects_partial_denominator_before_creating_output(tmp_path):
    rows, saved = fixture()
    source = tmp_path/"source"
    source.mkdir()
    diagnostic.write_rows(source/"auxiliary_repeat_comparison.csv", rows)
    np.savez_compressed(source/"primitive_nested_scores.npz", **saved)
    report = dict(status="COMPLETE_DESCRIPTIVE_PFC_REFINEMENT_REPEAT_COMPARISON", row_count=84330,
        primitive_endpoint_count=49650, source_case_count=192, original_independent_blocks=32, auxiliary_realizations=2,
        new_original_independent_blocks=0, production_episodes=0, confidence_intervals=False,
        confirmation_execution_released=False, scientific_fidelity_qualified=False,
        numerical_margins_changed=False, uncertainty_coverage_qualified=False,
        outputs={name: dict(sha256=diagnostic.sha256_file(source/name)) for name in (
            "auxiliary_repeat_comparison.csv", "primitive_nested_scores.npz")})
    diagnostic.freeze_json(source/"report.json", report)
    config = tmp_path/"config.yaml"
    config.write_text(yaml.safe_dump(config_fixture(dict(path="source", report_sha256=diagnostic.sha256_file(source/"report.json")))))
    output = tmp_path/"analysis"
    with pytest.raises(ValueError, match="COMPLETE_CANONICAL_PRIMITIVE_AND_EXCLUDED_DENOMINATORS"):
        diagnostic.run(config, tmp_path, output)
    assert not output.exists()


def test_existing_output_is_not_overwritten(tmp_path):
    output = tmp_path/"existing"
    output.mkdir()
    marker = output/"preserve.txt"
    marker.write_text("unchanged")
    with pytest.raises(ValueError, match="NEW_AUXILIARY_VARIANCE_OUTPUT_REQUIRED"):
        diagnostic.run(tmp_path/"missing-config.yaml", tmp_path, output)
    assert marker.read_text() == "unchanged"
