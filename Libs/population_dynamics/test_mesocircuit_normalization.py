"""Independent radial quadrature and adapter-normalization provenance contracts."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from .mesocircuit_normalization import normalize_connectivity, normalization_record


def native(ns=100.4, k=8.7, radius=0.5, beta=0.2, extent=1.0):
    f = radius / beta
    integral = 1.0 - math.exp(-f) * (1.0 + f)
    potential = ns * math.pi * radius**2 / extent**2
    raw = k / potential * 0.5 * f**2 * integral
    repeats = max(1, math.ceil(raw))
    return {"connect_method": "distr_indegree_exp", "extent": extent,
            "mask_radius": np.array([[radius]]), "beta": np.array([[beta]]),
            "p0_raw": np.array([[raw]]), "p0": np.array([[raw / repeats]]),
            "repeat_connect": np.array([[repeats]], dtype=np.int64),
            "indegrees": np.array([[round(k)]], dtype=np.int64),
            "num_neurons": np.array([round(ns)], dtype=np.int64),
            "weight_matrix_mean": np.array([[87.0]]), "delay_offset_matrix": np.array([[0.5]]),
            "DC_amp": np.array([12.0]), "thalamic_input": True}


def radial_probability_mass(radius, beta, extent):
    # Independent Gauss-Legendre integration of the annular area element.
    nodes, weights = np.polynomial.legendre.leggauss(96)
    r = (nodes + 1.0) * radius / 2.0
    return float(np.sum(weights * 2.0 * np.pi * r * np.exp(-r / beta)) * radius / (2.0 * extent**2))


@pytest.mark.parametrize("ns,k,radius,beta,extent,diagonal", [
    (100.4, 8.7, 0.5, 0.2, 1.0, False),
    (15.2, 2.7, 0.5, 0.3, 1.0, True),
    (42.7, 3.1, 0.25, 1.0, 1.0, False),
    (101.3, 9.2, 1.0, 0.2, 2.0, True),
])
def test_independent_radial_quadrature_preserves_raw_continuum_target(ns, k, radius, beta, extent, diagonal):
    original = native(ns, k, radius, beta, extent)
    corrected = normalize_connectivity(original, "kernel_integral")
    eligible = round(ns) - int(diagonal)
    mass = radial_probability_mass(radius, beta, extent)
    expectation = eligible * corrected["repeat_connect"][0, 0] * corrected["p0"][0, 0] * mass
    np.testing.assert_allclose(expectation, k * eligible / ns, rtol=2e-13, atol=0)
    f = radius / beta
    a = 1.0 - math.exp(-f) * (1.0 + f)
    native_expectation = eligible * original["repeat_connect"][0, 0] * original["p0"][0, 0] * mass
    np.testing.assert_allclose(native_expectation / expectation, a**2, rtol=2e-13, atol=0)


def test_noninteger_counts_and_autapse_exclusion_are_not_rounded_indegree_retargeting():
    ns, k = 10.4, 2.4
    result = normalize_connectivity(native(ns, k), "kernel_integral")
    expectation = (round(ns) - 1) * result["p0_raw"][0, 0] * radial_probability_mass(0.5, 0.2, 1.0)
    np.testing.assert_allclose(expectation, k * (round(ns) - 1) / ns, rtol=2e-13, atol=0)
    assert not np.isclose(expectation, round(k))
    assert result["indegrees"][0, 0] == round(k)


def test_multiple_repetitions_and_zero_links_remain_valid_probabilities():
    original = native(ns=10.0, k=100.0, radius=0.1, beta=0.5)
    result = normalize_connectivity(original, "kernel_integral")
    raw = result["p0_raw"][0, 0]
    assert result["repeat_connect"][0, 0] == math.ceil(raw) > 1
    assert 0.0 < result["p0"][0, 0] <= 1.0
    np.testing.assert_allclose(result["repeat_connect"] * result["p0"], result["p0_raw"], rtol=2e-16, atol=0)
    zero = normalize_connectivity(native(k=0.0), "kernel_integral")
    np.testing.assert_array_equal(zero["p0_raw"], [[0.0]])
    np.testing.assert_array_equal(zero["p0"], [[0.0]])
    np.testing.assert_array_equal(zero["repeat_connect"], [[1]])


def test_native_policy_is_exact_noop_including_object_identity():
    original = native()
    before = deepcopy(original)
    assert normalize_connectivity(original, "native_source") is original
    for name in original:
        np.testing.assert_array_equal(original[name], before[name])
    unrelated_profile = {"connect_method": "fixedindegree_exp", "sentinel": object()}
    assert normalize_connectivity(unrelated_profile, "native_source") is unrelated_profile


def test_variant_preserves_input_arrays_and_every_other_parameter():
    original = native()
    before = deepcopy(original)
    result = normalize_connectivity(original, "kernel_integral")
    assert result is not original
    for name in original:
        np.testing.assert_array_equal(original[name], before[name])
        if name in {"p0_raw", "p0", "repeat_connect"}:
            np.testing.assert_array_equal(result["source_" + name], before[name])
            assert not np.shares_memory(result["source_" + name], original[name])
            assert not np.shares_memory(result[name], original[name])
        else:
            assert result[name] is original[name]
    assert not any(name.startswith("source_") for name in original)


@pytest.mark.parametrize("policy", ["unknown", "rounded_indegree", ""])
def test_unknown_policies_rejected(policy):
    with pytest.raises(ValueError, match="POLICY"):
        normalize_connectivity(native(), policy)


@pytest.mark.parametrize("profile", ["fixedindegree_exp", "distr_indegree_gauss", "fixedtotalnumber"])
def test_variant_requires_native_exponential_bernoulli_profile(profile):
    net = native()
    net["connect_method"] = profile
    with pytest.raises(ValueError, match="DISTR_INDEGREE_EXP"):
        normalize_connectivity(net, "kernel_integral")


@pytest.mark.parametrize("key,value", [
    ("beta", [[0.0]]), ("beta", [[-1.0]]), ("mask_radius", [[0.0]]),
    ("mask_radius", [[np.nan]]), ("p0_raw", [[np.inf]]), ("p0_raw", [[-1.0]]),
    ("p0", [[1.2]]), ("repeat_connect", [[0]]), ("repeat_connect", [[1.5]]),
    ("beta", [[0.1, 0.2]]), ("p0_raw", [[True]]),
])
def test_invalid_source_kernel_or_probability_arrays_rejected(key, value):
    net = native()
    net[key] = np.asarray(value)
    with pytest.raises(ValueError):
        normalize_connectivity(net, "kernel_integral")


def test_unrepresentable_effective_repetition_count_rejected():
    net = native()
    net["p0_raw"] = np.array([[1e25]])
    with pytest.raises(ValueError, match="REPEAT_NOT_REPRESENTABLE"):
        normalize_connectivity(net, "kernel_integral")


def test_nonfinite_effective_probability_rejected_without_clipping():
    net = native(radius=0.1, beta=0.5)
    net["p0_raw"] = np.array([[1e308]])
    with pytest.raises(ValueError, match="NOT_REPRESENTABLE"):
        normalize_connectivity(net, "kernel_integral")


@pytest.mark.parametrize("policy", ["native_source", "kernel_integral"])
def test_record_serializes_source_and_effective_arrays_with_caveats(policy):
    original = native()
    result = normalize_connectivity(original, policy)
    record = normalization_record(result, policy)
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert record["policy"] == policy
    assert record["source_code_modified"] is False
    assert record["scientific_reference_qualified"] is False
    assert record["native_source_probability_arrays_preserved"] is (policy == "native_source")
    for name in ("p0_raw", "p0", "repeat_connect"):
        assert isinstance(record["source"][name], list)
        assert isinstance(record["effective"][name], list)
        np.testing.assert_array_equal(record["source"][name], original[name])
        np.testing.assert_array_equal(record["effective"][name], result[name])
    assert "rounding" in record["target_caveat"] and "autapse" in record["target_caveat"]
    assert "variance" in record["multiplicity_caveat"]


def test_fixed_indegree_record_does_not_require_or_invent_amplitude_arrays():
    record = normalization_record({"connect_method": "fixedindegree_exp"}, "native_source")
    assert record["probability_arrays_applicable"] is False
    assert record["source"] == record["effective"] == {}
    assert record["native_source_probability_arrays_preserved"] is True


@pytest.mark.parametrize("policy", [None, True, ["kernel_integral"]])
def test_nonstring_policy_has_a_meaningful_failure(policy):
    with pytest.raises(ValueError, match="POLICY"):
        normalize_connectivity(native(), policy)
    with pytest.raises(ValueError, match="POLICY"):
        normalization_record(native(), policy)
