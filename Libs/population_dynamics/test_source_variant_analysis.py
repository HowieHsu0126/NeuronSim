"""Contracts for matched-phase whole-block source comparisons."""
import copy
from dataclasses import asdict

import numpy as np
import pytest

from Libs.population_dynamics.nest_gpu_spatial import Execution
from Libs.population_dynamics.source_variant_analysis import (
    SCHEMA, aggregate_aligned, covariance_from_gram, difference_interval,
    paired_bootstrap_multiplicities, pooled_ensemble_fano, require_acquired,
    time_inner_product_gram, validate_cases, verify_runtime_pair,
)


def direct_covariance(values, counts):
    indices = np.repeat(np.arange(len(values)), counts)
    sample = values[indices]
    centered = sample - sample.mean(0, keepdims=True)
    return np.einsum("btp,btq->pq", centered, centered) / ((len(sample) - 1) * sample.shape[1])


def test_gram_matches_brute_whole_block_resampling():
    rng = np.random.default_rng(13)
    values = rng.poisson(4, size=(5, 11, 3)).astype(float)
    values += np.arange(11)[None, :, None] * 1000
    multiplicities = np.vstack((np.ones(5, dtype=int), [5, 0, 0, 0, 0],
                               [0, 2, 0, 1, 2], paired_bootstrap_multiplicities(5, 20, 87100)))
    got = covariance_from_gram(time_inner_product_gram(values), multiplicities)
    expected = np.stack([direct_covariance(values, c) for c in multiplicities])
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-11)


def test_phase_mean_is_removed_before_covariance():
    trajectory = np.arange(36, dtype=float).reshape(12, 3)
    values = np.repeat(trajectory[None], 4, axis=0)
    got = covariance_from_gram(time_inner_product_gram(values), np.ones((1, 4), dtype=int))
    np.testing.assert_array_equal(got, np.zeros((1, 3, 3)))
    assert np.var(trajectory[:, 0]) > 0


def test_paired_bootstrap_has_declared_full_block_denominator():
    a = paired_bootstrap_multiplicities(18, 5000, 87100)
    b = paired_bootstrap_multiplicities(18, 5000, 87100)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a.sum(1), np.full(5000, 18))
    assert a.shape == (5000, 18)


@pytest.mark.parametrize("counts", [np.ones((2, 4), dtype=float), np.array([[4, 0, 0, 1]]),
                                    np.array([[5, -1, 0, 0]])])
def test_invalid_bootstrap_multiplicities_rejected(counts):
    with pytest.raises(ValueError, match="multiplicities"):
        covariance_from_gram(time_inner_product_gram(np.ones((4, 5, 2))), counts)


def test_half_open_complete_time_aggregation():
    counts = np.arange(36, dtype=np.int64).reshape(2, 6, 3)
    edges = np.arange(0, 301, 50, dtype=np.int64)
    got, output_edges = aggregate_aligned(counts, edges, 50, 250, 100)
    expected = np.stack((counts[:, 1:3].sum(1), counts[:, 3:5].sum(1)), axis=1)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(output_edges, [50, 150, 250])


@pytest.mark.parametrize("start,stop,width", [(25, 225, 100), (50, 300, 100), (0, 300, 75)])
def test_misaligned_or_partial_bins_rejected(start, stop, width):
    with pytest.raises(ValueError, match="align"):
        aggregate_aligned(np.ones((3, 6, 2), dtype=np.int64),
                          np.arange(0, 301, 50, dtype=np.int64), start, stop, width)


def test_zero_count_fano_is_undefined_without_epsilon():
    covariance = np.zeros((2, 3, 3))
    fano = pooled_ensemble_fano(covariance, np.zeros((2, 3)))
    assert np.isnan(fano).all()
    row = difference_interval(float("nan"), 1., np.array([np.nan] * 5), .95)
    assert row["reference"] is None and row["difference"] is None
    assert row["ci_low"] is None and row["interval_status"] == "UNDEFINED_POINT_ESTIMAND"


def test_degenerate_interval_and_zero_reference_are_retained():
    row = difference_interval(0., 2., np.full(12, 2.), .95)
    assert row["difference"] == 2.
    assert row["ci_low"] == row["ci_high"] == 2.
    assert row["interval_status"] == "DEGENERATE_INTERVAL"
    np.testing.assert_array_equal(pooled_ensemble_fano(np.zeros((3, 3)), np.ones(3)), np.zeros(3))


def test_invalid_draw_is_not_silently_dropped():
    row = difference_interval(1., 2., np.array([1., np.nan, 1.]), .95)
    assert row["invalid_bootstrap_replicates"] == 1
    assert row["bootstrap_replicates"] == 3
    assert row["ci_low"] is None
    assert row["interval_status"] == "UNDEFINED_BOOTSTRAP_REPLICATES"


def test_incomplete_failed_or_duplicate_acquisition_rejected():
    good = {"status": "ACQUIRED", "expected_episodes": 2, "completed_episodes": 2,
            "runs": [{"case_id": "a", "exit_code": 0}, {"case_id": "b", "exit_code": 0}]}
    require_acquired(good, ["a", "b"])
    for mutation in ("running", "failed", "duplicate", "missing"):
        data = copy.deepcopy(good)
        if mutation == "running": data["status"] = "RUNNING"
        elif mutation == "failed": data["runs"][1]["exit_code"] = 2
        elif mutation == "duplicate": data["runs"][1]["case_id"] = "a"
        else: data["runs"].pop()
        with pytest.raises(ValueError, match="ACQUIRED"):
            require_acquired(data, ["a", "b"])


def pair_fixture():
    reference_case = {"initial_state_seed": 1, "background_seed": 2,
                      "settings": {"duration_ms": 6000., "initial_state_seed": 1, "background_seed": 2}}
    candidate_case = copy.deepcopy(reference_case)
    candidate_case["settings"]["expected_poisson_rng_family"] = "per_edge_philox4_32_10"
    declaration = {"graph_manifest_sha256": "graph", "source_rng_sha256": "initial_rng",
                   "reference_native_library_sha256": "xorwow_library",
                   "native_library_sha256": "philox_library",
                   "config": {"reference_rng_family": "per_edge_xorwow",
                              "candidate_rng_family": "per_edge_philox4_32_10"}}
    common = {"status": "EXECUTED", "checks": {"complete": True},
              "graph_manifest_sha256": "graph", "initial_state_rng_library_sha256": "initial_rng",
              "initial_state_seed": 1, "background_seed": 2, "dt_ms": .1, "duration_ms": 6000.,
              "gpu_source_revision": "revision", "cortical_neurons": 100000,
              "thalamic_relays": 1000, "recurrent_tc_edges": 10000, "background_edges": 100000,
              "native_connections": 110000, "poisson_native_delay_offset_ticks": 2}
    reference = {**common, "config": asdict(Execution(**reference_case["settings"])),
                 "native_library_sha256": "xorwow_library"}
    candidate = {**common, "config": asdict(Execution(**candidate_case["settings"])),
                 "native_library_sha256": "philox_library", "native_poisson_rng_family": "per_edge_philox4_32_10"}
    return reference, candidate, reference_case, candidate_case, declaration


def test_legacy_reference_is_bound_by_exact_declared_library():
    identity = verify_runtime_pair(*pair_fixture())
    assert identity["reference"]["rng_family_evidence"] == "pinned_reference_library_and_frozen_declaration"
    assert identity["candidate"]["rng_family_evidence"] == "runtime_export"


@pytest.mark.parametrize("field,value", [("native_library_sha256", "other"),
    ("native_poisson_rng_family", "per_edge_xorwow"), ("graph_manifest_sha256", "other"),
    ("initial_state_rng_library_sha256", "other"), ("dt_ms", .2), ("background_seed", 99)])
def test_runtime_mismatch_rejected(field, value):
    arguments = list(pair_fixture())
    arguments[1][field] = value
    with pytest.raises(ValueError):
        verify_runtime_pair(*arguments)


def declaration_fixture():
    config = {"schema": SCHEMA, "split": "train", "template_id": "familiar_0",
              "formal_training_seeds": [41, 42, 43], "expected_blocks": 3,
              "candidate_rng_family": "per_edge_philox4_32_10",
              "previously_observed_candidate_cases": ["case0"]}
    reference = [{"case_id": f"case{i}", "split": "train", "template_id": "familiar_0",
                  "block": i, "initial_state_seed": i + 1, "background_seed": i + 11,
                  "settings": {"initial_state_seed": i + 1}} for i in range(3)]
    templates = [{"template_id": "familiar_0", "motifs": [{"position": i} for i in range(9)]}]
    original = {"cases": reference, "templates": templates}
    candidate = copy.deepcopy(reference)
    for case in candidate: case["settings"]["expected_poisson_rng_family"] = "per_edge_philox4_32_10"
    declaration = {"schema": SCHEMA, "config": config, "cases": candidate, "templates": templates,
                   "common_noise_across_rng_families": False, "common_initial_states": True}
    return declaration, original, {"cases": reference}, config


def test_only_declared_family_changes_are_admissible():
    args = declaration_fixture()
    cases, _, _ = validate_cases(*args)
    assert len(cases) == 3
    args[0]["cases"][1]["settings"]["duration_ms"] = 9999.
    with pytest.raises(ValueError, match="changes more"):
        validate_cases(*args)


def test_missing_case_or_shared_noise_claim_rejected():
    args = declaration_fixture()
    args[0]["cases"].pop()
    with pytest.raises(ValueError, match="every distinct"):
        validate_cases(*args)
    args = declaration_fixture()
    args[0]["common_noise_across_rng_families"] = True
    with pytest.raises(ValueError, match="declaration"):
        validate_cases(*args)
