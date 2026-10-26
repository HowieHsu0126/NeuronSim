"""Joint-clock covariance, complete pairing and existing-data scope checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from . import pfc_clock_scaling as scaling


def fixture(original=None, repeated=None):
    if original is None:
        original = np.vstack([np.full(32, .4), np.full(32, .05)])
    if repeated is None:
        repeated = np.vstack([np.full(32, .6), np.full(32, .05)])
    stats = scaling.auxiliary.decompose_auxiliary_pairs(original, repeated)
    rows = []
    for index, (a, b) in enumerate(zip(original, repeated)):
        point_a, point_b = float(a.mean()), float(b.mean())
        identity = dict(left_cohort=scaling.auxiliary.ADJACENT_CLOCK_PAIRS[index][0],
            right_cohort=scaling.auxiliary.ADJACENT_CLOCK_PAIRS[index][1], condition="cue", phase="cue",
            metric="all_unit_rate_hz")
        rows.append(dict(domain_row_index=str([0, 5][index]), family="source_response",
            endpoint_identity=json.dumps(identity, sort_keys=True), definition_status="ESTIMATED_EMPIRICAL_SPREAD",
            margin="0.5", units="Hz", original_independent_blocks="32", auxiliary_realizations="2",
            auxiliary_scatter_kind=scaling.auxiliary.PRIMITIVE_KIND,
            original_estimate=point_a, repeated_estimate=point_b, estimate_midpoint=(point_a+point_b)/2,
            estimate_change=point_b-point_a, original_point_outside_margin=str(abs(point_a) > .5),
            repeated_point_outside_margin=str(abs(point_b) > .5)))
    rows = scaling.auxiliary.endpoint_rows(rows, stats)
    # Reproduce the serialized accepted CSV boundary, including exact flag strings.
    rows = [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]
    saved = dict(source_rng_seeds=np.asarray(scaling.SEEDS), domain_row_indices=np.array([0, 5]),
        **{name: stats[name].copy() for name in scaling.SCORE_KEYS})
    return rows, saved


def test_exact_first_order_relation_has_zero_curvature_with_shared_auxiliary_noise():
    a = np.array([[-2., 1., 3., -1.]])
    b = np.array([[1., -3., 2., 4.]])
    mf = scaling.auxiliary.decompose_auxiliary_pairs(a, b)
    cm = scaling.auxiliary.decompose_auxiliary_pairs(2*a, 2*b)
    stats, errors = scaling.joint_clock_moments(cm, mf)
    np.testing.assert_array_equal(stats["original_scores"], np.zeros_like(a))
    np.testing.assert_array_equal(stats["repeated_scores"], np.zeros_like(a))
    assert stats["auxiliary_within_variance_estimate"][0] == 0
    assert stats["cross_auxiliary_moment"][0] > 0
    assert stats["exact_nonzero_auxiliary_change_blocks"][0] == 0
    assert np.isnan(stats["maximum_squared_change_share"][0])
    assert max(errors.values()) < 1e-12
    assert not any(scaling.scientific_limits().values())


@pytest.mark.parametrize("sign,expected_within", [(1, 2.), (-1, 18.)])
def test_same_and_opposite_clock_changes_retain_cross_covariance(sign, expected_within):
    a = np.zeros((1, 4))
    change = np.array([[-2., 2., -2., 2.]])
    cm = scaling.auxiliary.decompose_auxiliary_pairs(a, change)
    mf = scaling.auxiliary.decompose_auxiliary_pairs(a, sign*change)
    stats, errors = scaling.joint_clock_moments(cm, mf)
    assert stats["cross_auxiliary_moment"][0] == sign*2
    assert stats["paired_midpoint_sample_covariance"][0] == pytest.approx(sign*4/3)
    assert stats["raw_between_clock_covariance"][0] == pytest.approx(sign/3)
    assert stats["auxiliary_within_variance_estimate"][0] == expected_within
    incorrect_independence = cm[scaling.MOMENT_KEYS[0]]+4*mf[scaling.MOMENT_KEYS[0]]
    assert incorrect_independence[0] == 10
    assert not np.isclose(incorrect_independence[0], expected_within)
    assert max(errors.values()) < 1e-12


def test_negative_between_curvature_is_retained_raw():
    a = np.vstack([np.tile([-1., 1.], 16), np.zeros(32)])
    rows, saved = fixture(a, -a)
    endpoints, stats, _, _ = scaling.analyse_pairs(rows, saved)
    row = endpoints[0]
    assert row["curvature_original_estimate"] == row["curvature_repeated_estimate"] == 0
    assert row["curvature_auxiliary_within_variance_estimate"] == 2
    assert row["curvature_midpoint_sample_variance"] == 0
    assert row["curvature_between_block_variance_estimate"] == -1
    assert row["curvature_between_block_estimate_negative"] is True
    assert row["curvature_exact_nonzero_auxiliary_change_blocks"] == 32
    assert stats["between_block_variance_estimate"][0] < 0


def test_zero_changes_remain_undefined_without_curvature_pass_or_ratio():
    a = np.vstack([np.linspace(-1., 1., 32), np.zeros(32)])
    rows, saved = fixture(a, a)
    endpoints, _, _, _ = scaling.analyse_pairs(rows, saved)
    row = endpoints[0]
    assert row["curvature_observed_zero_auxiliary_change"] is True
    assert row["curvature_observed_zero_within_variance"] is True
    assert row["curvature_maximum_squared_change_share"] is None
    assert row["curvature_maximum_share_source_rng_seeds"] is None
    assert row["curvature_contributing_source_rng_seeds"] == "[]"
    assert row["curvature_between_block_variance_estimate"] > 0
    assert not any(key.startswith(("curvature_ratio", "curvature_margin", "curvature_pass")) for key in row)


def test_floating_residual_is_not_thresholded_out():
    a = np.zeros((2, 32))
    b = a.copy()
    b[0, 3] = np.finfo(float).eps
    rows, saved = fixture(a, b)
    endpoints, _, _, _ = scaling.analyse_pairs(rows, saved)
    row = endpoints[0]
    assert row["curvature_exact_nonzero_auxiliary_change_blocks"] == 1
    assert row["curvature_observed_zero_auxiliary_change"] is False
    assert row["curvature_maximum_squared_change_share"] == 1
    assert json.loads(row["curvature_contributing_source_rng_seeds"]) == [95004]


@pytest.mark.parametrize("defect", ["missing_pair", "duplicate_pair", "mismatched_identity", "covariance", "undefined"])
def test_incomplete_or_ineligible_endpoint_pair_rejected(defect):
    rows, saved = fixture()
    if defect == "missing_pair":
        rows = rows[:1]
        saved = {key: values if key == "source_rng_seeds" else values[:1] for key, values in saved.items()}
    elif defect == "duplicate_pair":
        rows = rows+copy.deepcopy(rows)
        for i, row in enumerate(rows):
            row["domain_row_index"] = str(i)
        saved["domain_row_indices"] = np.arange(4)
        for name in scaling.SCORE_KEYS:
            saved[name] = np.tile(saved[name], (2, 1))
    elif defect == "mismatched_identity":
        identity = json.loads(rows[1]["endpoint_identity"])
        identity["metric"] = "different_metric"
        rows[1]["endpoint_identity"] = json.dumps(identity)
    elif defect == "covariance":
        rows[0]["family"] = "field_lag_covariance"
    else:
        rows[0]["definition_status"] = scaling.UNDEFINED
    with pytest.raises(ValueError):
        scaling.pair_endpoint_rows(rows, saved)


@pytest.mark.parametrize("defect", ["missing_seed", "duplicate_seed", "reordered_seeds", "duplicate_ids",
    "wrong_row_id", "wrong_shape", "nonfinite", "changed_midpoint", "changed_delta", "changed_point", "changed_cached_variance"])
def test_seed_axis_and_joint_array_or_moment_mismatch_rejected(defect):
    rows, saved = fixture()
    if defect == "missing_seed":
        saved["source_rng_seeds"] = saved["source_rng_seeds"][:-1]
    elif defect == "duplicate_seed":
        saved["source_rng_seeds"][-1] = saved["source_rng_seeds"][0]
    elif defect == "reordered_seeds":
        saved["source_rng_seeds"] = saved["source_rng_seeds"][::-1]
    elif defect == "duplicate_ids":
        saved["domain_row_indices"][:] = 0
    elif defect == "wrong_row_id":
        rows[1]["domain_row_index"] = "7"
    elif defect == "wrong_shape":
        saved["original_scores"] = saved["original_scores"][:, :-1]
    elif defect == "nonfinite":
        saved["repeated_scores"][0, 0] = np.inf
    elif defect == "changed_midpoint":
        saved["within_block_midpoint"][0, 0] += 1
    elif defect == "changed_delta":
        saved["repeated_minus_original"][0, 0] += 1
    elif defect == "changed_point":
        rows[0]["original_estimate"] = "99"
    else:
        rows[0]["auxiliary_within_variance_estimate"] = "99"
    with pytest.raises(ValueError):
        scaling.analyse_pairs(rows, saved)


def test_focus_preserves_both_source_margins_flags_and_all_blocks(tmp_path):
    rows, saved = fixture()
    rows[1]["margin"] = "0.2"
    endpoints, stats, pairs, errors = scaling.analyse_pairs(rows, saved)
    row = endpoints[0]
    assert row["coarse_medium_margin"] == "0.5" and row["medium_fine_margin"] == "0.2"
    assert row["coarse_medium_original_point_outside_margin"] == "False"
    assert row["coarse_medium_repeated_point_outside_margin"] == "True"
    assert row["focus_source_point_outside_margin"] is True
    path = tmp_path/"focus.csv"
    focus, count = scaling.write_focus_blocks(path, endpoints, saved, stats, pairs)
    written = scaling.read_rows(path)
    assert focus == endpoints and count == len(written) == 32
    assert [int(row["source_rng_seed"]) for row in written] == scaling.SEEDS
    assert float(written[0]["coarse_medium_original_scores"]) == .4
    assert float(written[0]["medium_fine_original_scores"]) == .05
    assert float(written[0]["R_A"]) == pytest.approx(.3)
    assert float(written[0]["R_B"]) == pytest.approx(.5)
    assert max(errors.values()) < 1e-12
    counts = scaling.family_count_rows(endpoints)
    assert counts[0]["joint_endpoint_count"] == 1
    assert counts[0]["source_primitive_endpoint_count"] == 2


def test_empty_focus_preserves_csv_header(tmp_path):
    rows, saved = fixture(np.zeros((2, 32)), np.zeros((2, 32)))
    endpoints, stats, pairs, _ = scaling.analyse_pairs(rows, saved)
    path = tmp_path/"focus.csv"
    focus, count = scaling.write_focus_blocks(path, endpoints, saved, stats, pairs)
    assert focus == [] and count == 0
    assert scaling.read_rows(path) == []
    assert "R_A" in path.read_text().splitlines()[0]


def report_fixture():
    reference = dict(path="comparison", report_sha256="bound-comparison")
    score = dict(bytes=100, sha256="bound-score")
    common = dict(original_independent_blocks=32, auxiliary_realizations=2, new_original_independent_blocks=0,
        production_episodes=0, confidence_intervals=False, confirmation_execution_released=False,
        scientific_fidelity_qualified=False, numerical_margins_changed=False, uncertainty_coverage_qualified=False)
    comparison = dict(common, status="COMPLETE_DESCRIPTIVE_PFC_REFINEMENT_REPEAT_COMPARISON",
        row_count=84330, primitive_endpoint_count=49650, source_case_count=192,
        outputs={"primitive_nested_scores.npz": copy.deepcopy(score)})
    variance = dict(common, schema=scaling.auxiliary.SCHEMA, status=scaling.auxiliary.STATUS, row_count=49650,
        full_input_endpoint_count=84330, inclusion_counts=dict(INCLUDED_PRIMITIVE=49650,
            EXCLUDED_DEFINED_COVARIANCE=32946, EXCLUDED_FROZEN_UNDEFINED_COVARIANCE=1734),
        included_primitive_family_counts=dict(scaling.auxiliary.PRIMITIVE_COUNTS), source_rng_seeds=list(scaling.SEEDS),
        maximum_confirmation_blocks=128, comparison=copy.deepcopy(reference),
        input_outputs={"primitive_nested_scores.npz": copy.deepcopy(score)}, implementation_sha256="helper",
        source_simulations=0, new_rng_draws=0, covariance_pseudovalues_used=False)
    return dict(comparison=reference), comparison, variance


def test_accepted_report_bindings_preserve_complete_excluded_denominators():
    settings, comparison, variance = report_fixture()
    scaling.check_bound_reports(settings, comparison, variance, "helper")


@pytest.mark.parametrize("defect", ["excluded_count", "primitive_count", "input_binding", "reference_binding", "helper_binding", "promoted_scope"])
def test_changed_accepted_count_or_binding_rejected(defect):
    settings, comparison, variance = report_fixture()
    if defect == "excluded_count":
        variance["inclusion_counts"]["EXCLUDED_DEFINED_COVARIANCE"] -= 1
    elif defect == "primitive_count":
        variance["included_primitive_family_counts"]["count_cdf"] -= 1
    elif defect == "input_binding":
        variance["input_outputs"]["primitive_nested_scores.npz"]["sha256"] = "changed"
    elif defect == "reference_binding":
        variance["comparison"]["report_sha256"] = "changed"
    elif defect == "helper_binding":
        variance["implementation_sha256"] = "changed"
    else:
        variance["confirmation_execution_released"] = True
    with pytest.raises(ValueError):
        scaling.check_bound_reports(settings, comparison, variance, "helper")


def config_fixture():
    return dict(schema=scaling.SCHEMA, comparison=dict(path="comparison", report_sha256="bound"),
        auxiliary_variance=dict(path="variance", report_sha256="bound"), weak_order_hypothesis=1,
        maximum_confirmation_blocks=128, source_simulations=0, new_rng_draws=0, confidence_intervals=False,
        confirmation_execution_released=False, scientific_fidelity_qualified=False,
        assumptions=["Equal-law independent auxiliary vectors across realizations, with shared clocks within each vector."],
        interpretation="Existing-data curvature description only.")


@pytest.mark.parametrize("key,value", [("weak_order_hypothesis", 2), ("maximum_confirmation_blocks", 129),
    ("source_simulations", 1), ("new_rng_draws", 1), ("confidence_intervals", True),
    ("confirmation_execution_released", True), ("scientific_fidelity_qualified", True)])
def test_scope_changes_are_rejected(tmp_path, key, value):
    settings = config_fixture()
    settings[key] = value
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="UNCHANGED_DATA_ONLY_FIRST_ORDER"):
        scaling.configuration(path)


def test_cli_scope_error_exits_nonzero_and_creates_no_output(tmp_path):
    settings = config_fixture()
    settings["weak_order_hypothesis"] = 2
    config = tmp_path/"config.yaml"
    config.write_text(yaml.safe_dump(settings))
    output = tmp_path/"analysis"
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run([sys.executable, "-m", "Libs.population_dynamics.pfc_clock_scaling",
        "--config", str(config), "--project-root", str(tmp_path), "--output", str(output)],
        cwd=root, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "UNCHANGED_DATA_ONLY_FIRST_ORDER" in completed.stderr
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path/"existing"
    output.mkdir()
    with pytest.raises(ValueError, match="NEW_CLOCK_SCALING_OUTPUT_REQUIRED"):
        scaling.run(tmp_path/"absent-config.yaml", tmp_path, output)
