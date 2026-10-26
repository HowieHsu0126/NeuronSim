"""Bounded native NumPy witnesses for the frozen source NMDA arithmetic.

Integer nearest-even rounding supplies the comparison oracle. These finite
vectors can contradict a backend premise. Passing vectors do not certify all
binary64 inputs, other hosts, emitted spikes, or closed-network fidelity.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from fractions import Fraction
import hashlib
from pathlib import Path
import platform
import sys

import numpy as np

from . import pfc_deterministic_replay as executor
from .pfc_synaptic_roundoff import (
    Interval, nmda_local_roundoff, propagated_error_envelope, same_event_history_bound,
)
from .pfc_synaptic_trace_bounds import NmdaParameters


def nearest_even_bits(value: Fraction) -> int:
    """Round an exact rational to finite binary64 using only integer arithmetic."""
    if type(value) is not Fraction:
        raise TypeError("PFC_ARITHMETIC_ORACLE_REQUIRES_FRACTION")
    sign = (1 << 63) if value < 0 else 0
    n, d = abs(value.numerator), value.denominator
    if n == 0:
        return 0
    exponent = n.bit_length() - d.bit_length()
    below_power = n < (d << exponent) if exponent >= 0 else (n << -exponent) < d
    if below_power:
        exponent -= 1
    quantum = max(exponent - 52, -1074)
    numerator, denominator = (n << -quantum, d) if quantum < 0 else (n, d << quantum)
    mantissa, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and mantissa & 1):
        mantissa += 1
    if mantissa < 1 << 52:
        return sign | mantissa
    if mantissa == 1 << 53:
        mantissa >>= 1
        quantum += 1
    exponent = quantum + 52
    if exponent > 1023:
        raise ValueError("PFC_ARITHMETIC_ORACLE_NONFINITE")
    return sign | ((exponent + 1023) << 52) | (mantissa - (1 << 52))


def fraction_from_bits(bits: int) -> Fraction:
    exponent, mantissa = (bits >> 52) & 2047, bits & ((1 << 52) - 1)
    if exponent == 2047:
        raise ValueError("PFC_ARITHMETIC_ORACLE_NONFINITE")
    if exponent:
        mantissa += 1 << 52
    power = exponent - 1075 if exponent else -1074
    value = Fraction(mantissa << power) if power >= 0 else Fraction(mantissa, 1 << -power)
    return -value if bits >> 63 else value


def _rounded(value: Fraction) -> Fraction:
    return fraction_from_bits(nearest_even_bits(value))


def unfused_nmda_bits(s: Fraction, x: Fraction, p: NmdaParameters) -> tuple[int, int, int]:
    """Operation-by-operation oracle for _rk2 and the subsequent E x+=1."""
    r, h, half = _rounded, p.dt_seconds, Fraction(1, 2)
    inverse_tau, lx = r(1 / p.tau_s_seconds), r(-h / p.tau_x_seconds)
    ks = r(h * r(r(p.alpha_per_second * r(x * r(1 - s))) - r(inverse_tau * s)))
    kx = r(lx * x)
    midpoint_x = r(r(half * kx) + x)
    complement = r(r(1 + r(half * -ks)) - s)
    midpoint_s = r(r(half * ks) + s)
    next_s = r(r(h * r(r(p.alpha_per_second * r(midpoint_x * complement))
                          - r(inverse_tau * midpoint_s))) + s)
    next_x = r(r(lx * r(r(half * kx) + x)) + x)
    return nearest_even_bits(next_s), nearest_even_bits(next_x), nearest_even_bits(r(next_x + 1))


def _array(values: list[Fraction]) -> np.ndarray:
    return np.asarray([nearest_even_bits(v) for v in values], dtype=np.uint64).view(np.float64)


def primitive_witness() -> dict:
    tiny, normal = Fraction(1, 2**1074), Fraction(1, 2**1022)
    half_ulp = Fraction(1, 2**53)
    cases = {
        "add": [(Fraction(1), half_ulp), (1 + 2*half_ulp, half_ulp),
                (Fraction(-1), -half_ulp), (-1 - 2*half_ulp, -half_ulp),
                (tiny, tiny), (normal, -tiny)],
        "subtract": [(Fraction(1), -half_ulp), (1 + 2*half_ulp, -half_ulp),
                     (normal + tiny, normal), (normal, normal - tiny)],
        "multiply": [(tiny, Fraction(1, 2)), (3*tiny, Fraction(1, 2)),
                     (-tiny, Fraction(1, 2)), (-3*tiny, Fraction(1, 2)),
                     (normal, Fraction(1, 2)), (normal+tiny, Fraction(1,2)),
                     (tiny, Fraction(1)), (tiny, Fraction(2**1022))],
        "divide": [(tiny, Fraction(2)), (3*tiny, Fraction(2)),
                   (normal, Fraction(2)), (Fraction(1), Fraction(3)),
                   (Fraction(-1), Fraction(3))],
    }
    exact_ops = {"add": lambda a,b:a+b, "subtract": lambda a,b:a-b,
                 "multiply": lambda a,b:a*b, "divide": lambda a,b:a/b}
    rows = []
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for name, pairs in cases.items():
            expected = [nearest_even_bits(exact_ops[name](a,b)) for a,b in pairs]
            for size in (1, 17, 257, 80000):
                for offset in (0, 1):
                    a = np.resize(_array([p[0] for p in pairs]), size+offset)[offset:]
                    b = np.resize(_array([p[1] for p in pairs]), size+offset)[offset:]
                    actual = getattr(np, name)(a,b).view(np.uint64)
                    oracle = np.resize(np.asarray(expected, dtype=np.uint64), size+offset)[offset:]
                    rows.append(dict(operation=name, size=size, offset=offset,
                                     mismatch_count=int(np.count_nonzero(actual != oracle))))
        # Separate multiply/add must produce zero; a fused expression gives -2^-54.
        a, b = _array([1 + Fraction(1,2**27)]), _array([1 - Fraction(1,2**27)])
        fused_discriminator = int(np.subtract(np.multiply(a,b), 1.0).view(np.uint64)[0])
    return dict(cases=rows, vector_cases=sum(len(v) for v in cases.values()),
                tested_elements=sum(row["size"] for row in rows),
                mismatch_count=sum(row["mismatch_count"] for row in rows),
                separate_multiply_subtract_bits=f"{fused_discriminator:016x}",
                separate_multiply_subtract_pass=fused_discriminator == 0)


def _fraction_record(value: Fraction) -> dict:
    return dict(numerator_hex=hex(value.numerator), denominator_hex=hex(value.denominator),
                approximation=float(value))


def _rounding_mode() -> int | None:
    library = ctypes.CDLL(ctypes.util.find_library("m") or None)
    fegetround = getattr(library, "fegetround", None)
    if fegetround is None:
        return None
    fegetround.argtypes, fegetround.restype = [], ctypes.c_int
    return fegetround()


def source_witness(parameters: NmdaParameters, *, source_sha256: str,
                   refractory_ticks: int) -> dict:
    source = Path(executor.__file__)
    observed_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if observed_sha != source_sha256:
        raise ValueError("PFC_ARITHMETIC_FROZEN_EXECUTOR_MISMATCH")
    constants = executor._constants({"weight_scaling_factor": 1.0}, "E")
    for key, name in (("tau_nmda_x", "tau_x_seconds"), ("tau_nmda_s", "tau_s_seconds"),
                      ("alpha_nmda", "alpha_per_second")):
        if Fraction.from_float(constants[key]) != getattr(parameters, name):
            raise ValueError("PFC_ARITHMETIC_SOURCE_CONSTANT_MISMATCH: " + key)
    h = float(parameters.dt_seconds)
    if Fraction.from_float(h) != parameters.dt_seconds:
        raise ValueError("PFC_ARITHMETIC_CLOCK_NOT_BINARY64")
    required = int((constants["refractory"] + 1e-3*h) / h)
    if type(refractory_ticks) is not int or refractory_ticks != required:
        raise ValueError("PFC_ARITHMETIC_SOURCE_REFRACTORY_MISMATCH")
    rounding_before = _rounding_mode()
    bound = same_event_history_bound(parameters, refractory_ticks=refractory_ticks,
                                    x_radius=Fraction(1,2**40), s_radius=Fraction(1,2**30))
    e_x, e_s = propagated_error_envelope(bound)
    primitive = primitive_witness()
    # Source-state witnesses include expanded proof corners and subnormal inputs.
    s_values = [Fraction(0), Fraction(1,2), Fraction(1), Fraction(1,2**1074),
                Fraction(1,2**1025), bound.derivatives.s_box.lower, bound.derivatives.s_box.upper]
    x_values = [Fraction(0), Fraction(3,10), Fraction(1), Fraction(1,2**1074),
                Fraction(1,2**1025), bound.derivatives.x_box.lower, bound.derivatives.x_box.upper]
    # Choose the representable upper endpoint inside the exact proof rectangle.
    upper_x_bits = nearest_even_bits(bound.derivatives.x_box.upper)
    if fraction_from_bits(upper_x_bits) > bound.derivatives.x_box.upper:
        upper_x_bits -= 1
    x_values[-1] = fraction_from_bits(upper_x_bits)
    pairs = [(fraction_from_bits(nearest_even_bits(s)), fraction_from_bits(nearest_even_bits(x)))
             for s in s_values for x in x_values]
    oracle = np.asarray([unfused_nmda_bits(s,x,parameters) for s,x in pairs], dtype=np.uint64)
    # Expanded comparison boxes include negative states. Use the exact DAG
    # extension already defined by the roundoff proof, not the physical-state
    # simulator helper, which correctly rejects negative x.
    local_references = [nmda_local_roundoff(parameters, s_box=Interval(s,s), x_box=Interval(x,x))
                        for s,x in pairs]
    references = [(local.s.nominal.lower, local.x_before_spike.nominal.lower)
                  for local in local_references]
    source_rows, maximum_s_error, maximum_x_error = [], Fraction(0), Fraction(0)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for size in (49, 80000):
            s = np.resize(_array([p[0] for p in pairs]), size)
            x = np.resize(_array([p[1] for p in pairs]), size)
            state = {key:np.zeros(size, dtype=np.float64) for key in
                     ("s_AMPA", "s_GABA", "s_NMDA_total", "I_stim")}
            state.update(v=np.full(size, -0.06), lastspike=np.full(size, -1e4), s_NMDA=s, x=x)
            executor._rk2(state, constants, 0.0, h, executor._Arithmetic("numpy"), "E")
            before_x = state["x"].copy()
            state["x"] += 1.0
            actual = np.column_stack((state["s_NMDA"].view(np.uint64), before_x.view(np.uint64),
                                      state["x"].view(np.uint64)))
            expected = oracle[np.arange(size) % len(pairs)]
            source_rows.append(dict(size=size, output_values=size*3,
                                    mismatch_count=int(np.count_nonzero(actual != expected)),
                                    all_outputs_finite=bool(np.isfinite(actual.view(np.float64)).all())))
            # Inspect each distinct input at full source width; repeated rows share the oracle.
            for i, (reference_s, reference_x) in enumerate(references):
                ds = abs(Fraction.from_float(float(state["s_NMDA"][i])) - reference_s)
                dx = max(abs(Fraction.from_float(float(before_x[i])) - reference_x),
                         abs(Fraction.from_float(float(state["x"][i])) - (reference_x+1)))
                maximum_s_error, maximum_x_error = max(maximum_s_error,ds), max(maximum_x_error,dx)
    # fegetround reads the current process mode; it does not change floating-point controls.
    rounding_mode = _rounding_mode()
    numpy_core = getattr(np, "_core", None)
    numpy_native = getattr(numpy_core, "_multiarray_umath", None)
    native_path = Path(numpy_native.__file__) if numpy_native is not None else None
    cpu_features = getattr(numpy_native, "__cpu_features__", {})
    mode_supported = platform.system() == "Linux" and platform.machine() in ("x86_64", "aarch64")
    passed = (mode_supported and rounding_before == rounding_mode == 0 and primitive["mismatch_count"] == 0
              and primitive["separate_multiply_subtract_pass"]
              and all(row["mismatch_count"] == 0 and row["all_outputs_finite"] for row in source_rows)
              and maximum_s_error <= bound.local.s.error
              and maximum_x_error <= max(bound.local.x_before_spike.error, bound.local.x_after_spike.error))
    return dict(schema="pfc_numpy_arithmetic_witness_v1", status="PASS_BOUNDED_WITNESS" if passed else "FAILED_BOUNDED_WITNESS",
                source_sha256=observed_sha, numpy_version=np.__version__, python_version=sys.version,
                machine=platform.machine(), fegetround_before=rounding_before,
                fegetround_after=rounding_mode, primitive=primitive,
                numpy_native_library_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest() if native_path else None,
                numpy_cpu_features={key:bool(value) for key,value in cpu_features.items()},
                source_cases=source_rows, distinct_source_state_pairs=len(pairs),
                source_refractory_ticks=required, parameter_values={name:_fraction_record(getattr(parameters,name))
                    for name in ("dt_seconds", "tau_x_seconds", "tau_s_seconds", "alpha_per_second")},
                maximum_observed_s_local_error=_fraction_record(maximum_s_error),
                maximum_observed_x_local_error=_fraction_record(maximum_x_error),
                conditional_s_local_error_bound=_fraction_record(bound.local.s.error),
                conditional_x_local_error_bound=_fraction_record(max(bound.local.x_before_spike.error,bound.local.x_after_spike.error)),
                conditional_x_error_envelope=_fraction_record(e_x), conditional_s_error_envelope=_fraction_record(e_s),
                conditional_initial_states=dict(x=0, s=0), native_backend_certified=False,
                native_binary64_roundoff_qualified=False, closed_network_fidelity_qualified=False,
                source_qualification_granted=False, production_episodes=0,
                scope="Finite same-process primitive/source-DAG witnesses; uniform envelopes retain IEEE and same-event premises")
