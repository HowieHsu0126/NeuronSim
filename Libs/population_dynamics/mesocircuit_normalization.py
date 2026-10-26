"""Labeled adapter normalization of native exponential connection probabilities.

This changes no vendored source bytes and establishes no published-model or
scientific qualification. The native_source policy is an exact no-op.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammainc

PROBABILITY_KEYS = ("p0_raw", "p0", "repeat_connect")
POLICIES = frozenset({"native_source", "kernel_integral"})
INT64_MAX = np.iinfo(np.int64).max


def _matrix(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or not array.size or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"FINITE_NUMERIC_CONNECTION_MATRIX_REQUIRED:{name}")
    return array


def _probability_arrays(net, prefix=""):
    try:
        arrays = {name: _matrix(net[prefix + name], prefix + name) for name in PROBABILITY_KEYS}
    except KeyError as exc:
        raise ValueError(f"CONNECTION_PROBABILITY_ARRAY_REQUIRED:{exc.args[0]}") from exc
    raw, probability, repeats = (arrays[key] for key in PROBABILITY_KEYS)
    if raw.shape != probability.shape or raw.shape != repeats.shape:
        raise ValueError("CONNECTION_PROBABILITY_SHAPES_DIFFER")
    if np.any(raw < 0) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("CONNECTION_PROBABILITY_OUT_OF_RANGE")
    if np.any(repeats < 1) or np.any(repeats != np.floor(repeats)):
        raise ValueError("POSITIVE_INTEGER_CONNECTION_REPEATS_REQUIRED")
    if (np.any(repeats >= 2**63) if repeats.dtype.kind == "f" else np.any(repeats > INT64_MAX)):
        raise ValueError("CONNECTION_REPEAT_NOT_REPRESENTABLE")
    return arrays


def normalize_connectivity(net: dict, policy: str) -> dict:
    """Apply only the kernel-integral algebra, retaining raw native count conventions.

    A = gammainc(2, R/beta) = 1 - exp(-R/beta)*(1 + R/beta).
    The stable special-function evaluation avoids subtracting nearly equal values.
    Other parameters retain their existing values and objects in the copied dict.
    """
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError("UNKNOWN_CONNECTIVITY_NORMALIZATION_POLICY")
    if policy == "native_source":
        return net
    if net.get("connect_method") != "distr_indegree_exp":
        raise ValueError("KERNEL_INTEGRAL_REQUIRES_DISTR_INDEGREE_EXP")
    original = _probability_arrays(net)
    try:
        radius = _matrix(net["mask_radius"], "mask_radius")
        beta = _matrix(net["beta"], "beta")
    except KeyError as exc:
        raise ValueError(f"EXPONENTIAL_KERNEL_PARAMETER_REQUIRED:{exc.args[0]}") from exc
    if radius.shape != beta.shape or radius.shape != original["p0_raw"].shape:
        raise ValueError("EXPONENTIAL_KERNEL_SHAPES_DIFFER")
    if np.any(radius <= 0) or np.any(beta <= 0):
        raise ValueError("POSITIVE_EXPONENTIAL_KERNEL_PARAMETERS_REQUIRED")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = radius / beta
        integral = gammainc(2.0, ratio)
        # Sequential division avoids underflow from explicitly squaring small A.
        effective_raw = original["p0_raw"] / integral / integral
    if not np.isfinite(ratio).all() or not np.isfinite(integral).all() or np.any(integral <= 0):
        raise ValueError("POSITIVE_FINITE_KERNEL_INTEGRAL_REQUIRED")
    if not np.isfinite(effective_raw).all() or np.any(effective_raw < 0):
        raise ValueError("EFFECTIVE_RAW_PROBABILITY_NOT_REPRESENTABLE")
    repeats_float = np.maximum(1.0, np.ceil(effective_raw))
    if np.any(repeats_float >= 2**63):
        raise ValueError("EFFECTIVE_CONNECTION_REPEAT_NOT_REPRESENTABLE")
    repeats = repeats_float.astype(np.int64)
    probability = effective_raw / repeats
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("EFFECTIVE_CONNECTION_PROBABILITY_OUT_OF_RANGE")
    result = dict(net)
    result.update({"source_" + name: value.copy() for name, value in original.items()})
    result.update(p0_raw=effective_raw, p0=probability, repeat_connect=repeats)
    return result


def normalization_record(net: dict, policy: str) -> dict:
    """Record actual probability matrices and the limited target of the variant."""
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError("UNKNOWN_CONNECTIVITY_NORMALIZATION_POLICY")
    if policy == "native_source" and net.get("connect_method") == "fixedindegree_exp":
        return {
            "policy": policy, "connect_method": "fixedindegree_exp",
            "probability_arrays_applicable": False,
            "native_source_probability_arrays_preserved": True,
            "source": {}, "effective": {},
            "formula": "Native fixed-indegree sampling uses distance weights without a p0 amplitude.",
            "target_caveat": "Native fixed indegrees and all model parameters are unchanged.",
            "source_code_modified": False, "scientific_reference_qualified": False,
        }
    effective = _probability_arrays(net)
    source = _probability_arrays(net, "source_") if policy == "kernel_integral" else effective
    preserved = all(np.array_equal(source[name], effective[name]) for name in PROBABILITY_KEYS)
    return {
        "policy": policy,
        "native_source_probability_arrays_preserved": bool(preserved),
        "source": {name: value.tolist() for name, value in source.items()},
        "effective": {name: value.tolist() for name, value in effective.items()},
        "formula": ("p0_raw_effective = p0_raw_native / A**2; "
                    "A = 1-exp(-R/beta)*(1+R/beta); "
                    "repeat_connect = max(1,ceil(p0_raw_effective)); p0 = p0_raw_effective/repeat_connect"
                    if policy == "kernel_integral" else "Native probability arrays are unchanged."),
        "target_caveat": "kernel_integral targets the raw continuum indegree: under uniform positions, "
                         "expected synaptic indegree is K_raw*(n_s-delta)/N_s_raw. Native neuron-count "
                         "rounding and autapse exclusion remain; rounded net['indegrees'] is not retargeted. "
                         "native_source performs no normalization adjustment.",
        "multiplicity_caveat": "Repeated Bernoulli connections retain synapse multiplicity. Recomputed "
                               "repeats change conditional variance and unique-partner statistics. "
                               "Other parameter values remain unchanged, which does not establish "
                               "unchanged realized edge weights/delays or a biological operating point.",
        "source_code_modified": False,
        "scientific_reference_qualified": False,
    }
