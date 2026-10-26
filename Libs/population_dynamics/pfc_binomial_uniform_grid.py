"""Audit PFC legacy binomial inversion on a finite uniform grid.

Grid laws require fresh independent uniform words and the published monotone
legacy inversion algorithm. They do not certify MT19937 conditional laws or
source confidence. Controlled uniforms use the installed RandomState entry point;
no experiment seed, PRNG state, or neural simulation is used.
"""
from __future__ import annotations

import argparse
import ctypes
from fractions import Fraction
from functools import lru_cache
import json
import math
from pathlib import Path
import platform
import threading

import numpy as np
import yaml

from . import pfc_clock_coupling, pfc_native_drive
from .pfc_refinement_input_law import _rational, analyze, routing_law
from .pfc_refinement_source_binding import _Reader
from .provenance import freeze_json

_CAPSULE_NAME = b"BitGenerator"
_U64 = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)
_U32 = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p)
_F64 = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p)


class _BitGeneratorABI(ctypes.Structure):
    _fields_ = [("state", ctypes.c_void_p), ("next_uint64", _U64),
                ("next_uint32", _U32), ("next_double", _F64), ("next_raw", _U64)]


def require(condition, reason):
    if not condition:
        raise ValueError("PFC_BINOMIAL_GRID_" + reason)


class _ControlledUniforms:
    """One specified uniform, then zero to terminate a restart for observation.

    The second word is a sentinel. The oracle reports restart rather than the
    final zero-valued sample. Unexpected callback use withholds the result.
    Callback references and the capsule name remain alive with this object.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.value = 0.
        self.draws = self.unexpected_calls = 0

        def next_double(_state):
            self.draws += 1
            return self.value if self.draws == 1 else 0.

        def unexpected(_state):
            self.unexpected_calls += 1
            return 0

        self.callbacks = (_U64(unexpected), _U32(unexpected), _F64(next_double), _U64(unexpected))
        self.struct = _BitGeneratorABI(None, *self.callbacks)
        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        capsule_new.restype = ctypes.py_object
        self.capsule = capsule_new(ctypes.addressof(self.struct), _CAPSULE_NAME, None)


class InversionOracle:
    """Single-attempt labels from the installed NumPy legacy scalar sampler."""

    def __init__(self, trials: int, probability: float, bits: int = 53):
        require(np.__version__ == "2.4.2" and np.dtype(np.long).itemsize == 8,
                "NUMPY_2_4_2_64_BIT_LONG_REQUIRED")
        require(type(trials) is int and 0 <= trials <= 1000
                and type(probability) in (int, float) and math.isfinite(probability)
                and 0 < probability <= .5 and trials * probability <= 30,
                "LEGACY_INVERSION_DOMAIN_REQUIRED")
        require(type(bits) is int and 1 <= bits <= 53, "UNIFORM_GRID_BITS_REQUIRED")
        self.trials, self.probability, self.bits = trials, float(probability), bits
        self.grid_size = 2 ** bits
        q = 1. - self.probability
        np_value = trials * self.probability
        self.bound = int(min(trials, np_value + 10. * math.sqrt(np_value * q + 1.)))
        self.masses = [math.exp(trials * math.log(q))]
        for k in range(1, self.bound + 1):
            self.masses.append(((trials - k + 1) * self.probability * self.masses[-1]) / (k * q))
        require(all(math.isfinite(mass) and mass > 0 for mass in self.masses),
                "FINITE_POSITIVE_INVERSION_MASSES_REQUIRED")
        self.driver = _ControlledUniforms()
        self.rng = np.random.RandomState(self.driver)
        self.queries = 0

    @lru_cache(maxsize=2048)
    def __call__(self, word: int) -> int:
        require(type(word) is int and 0 <= word < self.grid_size, "IN_RANGE_INTEGER_UNIFORM_REQUIRED")
        self.driver.value = word / self.grid_size
        self.driver.draws = self.driver.unexpected_calls = 0
        result = int(self.rng.binomial(self.trials, self.probability))
        self.queries += 1
        require(self.driver.unexpected_calls == 0 and self.driver.draws in (1, 2),
                "UNEXPECTED_PRIMITIVE_OR_RESTART_SEQUENCE")
        if self.driver.draws == 2:
            require(result == 0, "ZERO_SENTINEL_MUST_TERMINATE_RESTART")
            return self.bound + 1
        require(0 <= result <= self.bound, "INVERSION_SUPPORT_CHANGED")
        return result

    def published_model(self, word: int) -> int:
        """Source-checked binary64 residual inversion for one attempt."""
        u = word / self.grid_size
        for k, mass in enumerate(self.masses):
            if u <= mass:
                return k
            u -= mass
        return self.bound + 1


def finite_law(oracle: InversionOracle) -> dict:
    """Integer interval census conditional on whole-grid monotonicity.

    Positive fixed masses and rounded subtraction are monotone in the input
    word. Restart is the largest single-attempt label, not final returned zero.
    Boundary comparisons test the installed entry point against this algorithm;
    they do not prove source-to-binary equivalence outside those comparisons.
    """
    ends, counts = [], []
    previous = 0
    boundary_words = {0, oracle.grid_size - 1}
    for k in range(oracle.bound + 1):
        lo, hi = previous, oracle.grid_size
        while lo < hi:
            mid = (lo + hi) // 2
            if oracle(mid) <= k:
                lo = mid + 1
            else:
                hi = mid
        require(previous <= lo <= oracle.grid_size, "ORDERED_ACCEPTANCE_THRESHOLDS_REQUIRED")
        counts.append(lo - previous)
        ends.append(lo)
        for word in (lo - 1, lo, lo + 1):
            if 0 <= word < oracle.grid_size:
                boundary_words.add(word)
        previous = lo
    restarts = oracle.grid_size - previous
    require(sum(counts) + restarts == oracle.grid_size and previous > 0,
            "COMPLETE_GRID_PARTITION_REQUIRED")
    for word in sorted(boundary_words):
        require(oracle(word) == oracle.published_model(word), "NATIVE_MODEL_BOUNDARY_MISMATCH")
    mean = Fraction(sum(k * count for k, count in enumerate(counts)), previous)
    nominal = oracle.trials * Fraction.from_float(oracle.probability)
    return dict(trials=oracle.trials, probability=_rational(Fraction.from_float(oracle.probability)),
                grid_bits=oracle.bits, grid_size=oracle.grid_size, inversion_bound=oracle.bound,
                acceptance_prefix_end_by_k=ends, acceptance_grid_counts_by_k=counts,
                restart_grid_points=restarts, accepted_grid_points=previous,
                normalized_mass_by_k=[_rational(Fraction(count, previous)) for count in counts],
                mean=_rational(mean), nominal_binomial_mean=_rational(nominal),
                mean_minus_nominal=_rational(mean - nominal), mean_at_most_nominal=mean <= nominal,
                restart_probability=_rational(Fraction(restarts, oracle.grid_size)),
                boundary_words_checked=len(boundary_words), installed_oracle_queries=oracle.queries,
                monotonicity_basis="published positive-mass residual inversion with monotone binary64 subtraction",
                whole_grid_exhaustively_enumerated=False,
                actual_mt19937_law_qualified=False, compiled_source_equivalence_proven=False)


def binomial_tv(law: dict) -> Fraction:
    """Exact TV against Binomial(N,p), including its complete unobserved tail."""
    p = Fraction(int(law["probability"]["numerator"]), int(law["probability"]["denominator"]))
    n = law["trials"]
    denominator = p.denominator ** n
    masses = [math.comb(n, k) * p.numerator ** k * (p.denominator - p.numerator) ** (n - k)
              for k in range(len(law["acceptance_grid_counts_by_k"]))]
    accepted = law["accepted_grid_points"]
    tail = denominator - sum(masses)
    require(tail >= 0, "COMPLETE_BINOMIAL_SUPPORT_REQUIRED")
    distance = sum(abs(count * denominator - mass * accepted)
                   for count, mass in zip(law["acceptance_grid_counts_by_k"], masses, strict=True))
    return Fraction(distance + accepted * tail, 2 * accepted * denominator)


def _hex_fraction(value: Fraction) -> dict:
    # A binary64 binomial at N=1000 has integer witnesses above Python's decimal
    # string limit. Hex preserves the exact value without changing global limits.
    return dict(numerator_hex=hex(value.numerator), denominator_hex=hex(value.denominator),
                display_float=float(value), display_float_for_display_only=True)


def _small_model_matches(oracle: InversionOracle, law: dict) -> int:
    if oracle.trials > pfc_clock_coupling._SMALL_COUNT_LIMIT:
        return 0
    masses = pfc_clock_coupling._small_binomial_masses(oracle.probability)
    words = {0, oracle.grid_size - 1}
    for end in law["acceptance_prefix_end_by_k"]:
        words.update(word for word in (end - 1, end, end + 1) if 0 <= word < oracle.grid_size)
    for word in sorted(words):
        result = pfc_clock_coupling._small_binomial_from_uniforms(
            np.array([oracle.trials]), np.array([word / oracle.grid_size]), masses)
        label = oracle.bound + 1 if result is None else int(result[0])
        require(label == oracle(word), "REFINEMENT_HELPER_BOUNDARY_MISMATCH")
    return len(words)


def _mass(law: dict) -> list[Fraction]:
    return [Fraction(count, law["accepted_grid_points"]) for count in law["acceptance_grid_counts_by_k"]]


def refinement_pair_law(kernels: dict, parents: int, level: int, probability: float) -> dict:
    """Compose every zero/double/single branch; compare against ideal marks.

    This uses conditional IID uniform words after any rejection stopping time.
    The exact-right ratio r* is represented by the ideal four-category law,
    so the comparison includes finite inversion and binary64 r perturbation.
    """
    require(type(parents) is int and parents >= 0 and type(level) is int and level in (0, 1),
            "VALID_PAIR_KERNEL_IDENTITY_REQUIRED")
    joint, reference = {}, {}
    source = routing_law(probability)
    q, side = source["q"], (1 - 2 * source["q"]) / 2
    factorial = math.factorial
    for zero, p_zero in enumerate(_mass(kernels[f"zero_level_{level}", parents])):
        remaining = parents - zero
        for double, p_double in enumerate(_mass(kernels[f"double_level_{level}", remaining])):
            singles = remaining - double
            for left, p_left in enumerate(_mass(kernels["single_child_allocation", singles])):
                pair = double + left, double + singles - left
                joint[pair] = joint.get(pair, Fraction(0)) + p_zero * p_double * p_left
    for zero in range(parents + 1):
        for double in range(parents - zero + 1):
            singles = parents - zero - double
            for left in range(singles + 1):
                right = singles - left
                coefficient = factorial(parents) // (factorial(zero) * factorial(double) * factorial(left) * factorial(right))
                pair = double + left, double + right
                reference[pair] = reference.get(pair, Fraction(0)) + coefficient * q ** (zero + double) * side ** singles
    require(sum(joint.values()) == sum(reference.values()) == 1
            and all(0 <= left <= parents and 0 <= right <= parents for left, right in joint),
            "COMPLETE_CONDITIONAL_PAIR_MASS_REQUIRED")
    tv = sum(abs(joint.get(pair, 0) - reference.get(pair, 0)) for pair in set(joint) | set(reference)) / 2
    left_mean = sum(pair[0] * value for pair, value in joint.items())
    right_mean = sum(pair[1] * value for pair, value in joint.items())
    return dict(joint=joint, tv=tv, left_mean=left_mean, right_mean=right_mean)


def complete_input_envelope(native: dict, pairs: dict, units: int, parent_ticks: int) -> dict:
    """Union of first mismatches, weighted by actual finite-grid parent laws.

    The native root is common to A and B. Their refinement costs each occur
    once. Native/root mismatches are counted once, not once per auxiliary tree.
    Correct marginals continue after any mismatch; no trajectories are selected.
    """
    require(type(units) is int and units > 0 and type(parent_ticks) is int and parent_ticks > 0,
            "POSITIVE_INPUT_ENVELOPE_DIMENSIONS_REQUIRED")
    root_mass = _mass(native)
    root_tv = binomial_tv(native)
    first = sum(mass * pairs[0, k]["tv"] for k, mass in enumerate(root_mass))
    second = sum(mass * probability * (pairs[1, left]["tv"] + pairs[1, right]["tv"])
                 for k, mass in enumerate(root_mass)
                 for (left, right), probability in pairs[0, k]["joint"].items())
    tree_cost = first + second
    exposure = units * parent_ticks
    one = min(Fraction(1), exposure * (root_tv + tree_cost))
    two = min(Fraction(1), exposure * (root_tv + 2 * tree_cost))
    return dict(native_root_tv=_hex_fraction(root_tv),
                first_level_expected_pair_tv=_hex_fraction(first),
                second_level_expected_two_child_pair_tv=_hex_fraction(second),
                source_root_exposure=exposure,
                one_tree_complete_input_tv_upper=_hex_fraction(one),
                shared_root_two_tree_complete_input_tv_upper=_hex_fraction(two),
                indexed_32_block_family_tv_upper=_hex_fraction(min(Fraction(1), 32 * two)),
                native_root_counted_once_for_A_B=True,
                no_native_parent_count_mean_assumption_used=True,
                parent_laws_derived_from_conditional_finite_grid=True,
                actual_mt19937_joint_law_qualified=False,
                source_history_confidence_certified=False, error_budget_allocated=False)


def finite_seed_support_obstruction(native: dict, root_count: int, seed_bits: int = 32) -> dict:
    """A finite seeded source cannot have small full-stream TV from IID counts.

    For a fixed simulator configuration the coarse input stream is a
    deterministic function of its bounded integer seed. Its support is <=2**b,
    regardless of the distribution of that seed. An IID count law puts at most
    pmax**M on any stream. The event 'stream belongs to seeded-source support'
    therefore supplies TV >= 1-2**b*pmax**M. A small integer witness avoids
    constructing integers with hundreds of millions of bits.
    """
    require(type(root_count) is int and root_count > 0 and type(seed_bits) is int and seed_bits >= 0,
            "VALID_SEEDED_SUPPORT_DIMENSIONS_REQUIRED")
    maximum = max(_mass(native))
    require(maximum <= Fraction(15, 16) and 2 * 15 ** 16 <= 16 ** 16,
            "IID_MAXIMUM_POINT_MASS_BOUND_REQUIRED")
    exponent = seed_bits - root_count // 16
    return dict(seed_bits_upper=seed_bits, native_root_count=root_count,
        iid_count_maximum_mass=_rational(maximum), iid_maximum_mass_upper=_rational(Fraction(15, 16)),
        integer_witness=dict(block_length=16, twice_numerator_power=2 * 15 ** 16, denominator_power=16 ** 16),
        iid_mass_on_seeded_stream_support_upper=dict(base=2, exponent=exponent),
        full_stream_tv_lower=dict(expression="max(0, 1-2**exponent)", exponent=exponent),
        uniform_seed_distribution_required=False,
        scope="fixed-configuration deterministic integer-seeded coarse stream versus IID finite-grid counts",
        iid_source_model_equivalent_to_actual_seeded_stream=False,
        low_dimensional_observable_law_error_lower_bound_implied=False,
        neural_fidelity_failure_implied=False)


def run(config_path: Path, source_root: Path, output: Path) -> dict:
    require(not output.exists(), "FRESH_GRID_ANALYSIS_OUTPUT_REQUIRED")
    reader = _Reader()
    source_receipt = json.loads(reader.read(source_root / "retrieval.json"))
    require(source_receipt["status"] == "RETRIEVED_PRIMARY_NUMPY_2_4_2_SAMPLER_SOURCES",
            "PINNED_PRIMARY_SAMPLER_SOURCES_REQUIRED")
    expected_names = {"legacy-distributions.c", "distributions.c", "mt19937.h", "mtrand.pyx", "LICENSE.txt"}
    require({row["name"] for row in source_receipt["records"]} == expected_names,
            "COMPLETE_DECLARED_SOURCE_RECEIPT_REQUIRED")
    for row in source_receipt["records"]:
        require(row["url"].startswith("https://raw.githubusercontent.com/numpy/numpy/v2.4.2/"),
                "OFFICIAL_VERSIONED_SOURCE_REQUIRED")
        reader.read(source_root / row["name"], row)
    config = yaml.safe_load(reader.read(config_path))
    parameter = analyze(config)
    require(parameter["source_core_units"] == 100000 and parameter["coarse_dt_ms"] == .05
            and parameter["duration_ms"] == 5000, "UNCHANGED_PFC_SOURCE_DIMENSIONS_REQUIRED")
    p0 = pfc_native_drive._RATE_HZ * (config["dt_ms"] * 1e-3)
    native_oracle = InversionOracle(pfc_native_drive._BINOMIAL_N, p0)
    native = finite_law(native_oracle)
    native["tv_from_ideal_binomial"] = _hex_fraction(binomial_tv(native))
    conditional = []
    probabilities = {"single_child_allocation": .5}
    for level in range(2):
        law = routing_law(p0 / 2 ** level)
        probabilities[f"zero_level_{level}"] = float(law["q"])
        probabilities[f"double_level_{level}"] = float(law["r"])
    # In the published native inversion each parent is <=bound; each child is
    # <=its parent. This census includes native fallback counts above limit8.
    for name, probability in probabilities.items():
        for trials in range(native_oracle.bound + 1):
            oracle = InversionOracle(trials, probability)
            law = finite_law(oracle)
            law["kernel_name"] = name
            law["tv_from_ideal_binomial"] = _hex_fraction(binomial_tv(law))
            law["refinement_helper_boundary_words_checked"] = _small_model_matches(oracle, law)
            conditional.append(law)
    kernel_lookup = {(law["kernel_name"], law["trials"]): law for law in conditional}
    pair_lookup, pair_rows = {}, []
    for level in range(2):
        for parents in range(native_oracle.bound + 1):
            pair = refinement_pair_law(kernel_lookup, parents, level, p0 / 2 ** level)
            pair_lookup[level, parents] = pair
            pair_rows.append(dict(level=level, parent_count=parents,
                pair_tv_from_ideal_marks=_hex_fraction(pair["tv"]),
                left_count_mean=_rational(pair["left_mean"]), right_count_mean=_rational(pair["right_mean"]),
                left_right_mean_difference=_rational(pair["left_mean"] - pair["right_mean"]),
                joint_child_counts=[dict(left=left, right=right, probability=_rational(value))
                                   for (left, right), value in sorted(pair["joint"].items())]))
    envelope = complete_input_envelope(native, pair_lookup, parameter["source_core_units"], parameter["coarse_ticks"])
    support_obstruction = finite_seed_support_obstruction(
        native, parameter["source_core_units"] * parameter["coarse_ticks"])
    for path in (Path(__file__), Path(pfc_clock_coupling.__file__), Path(pfc_native_drive.__file__),
                 Path(np.random.mtrand.__file__), Path(np.random._mt19937.__file__)):
        reader.read(path)
    report = dict(status="COMPLETE_CONDITIONAL_LEGACY_INVERSION_UNIFORM_GRID_CENSUS",
        scope="published_inversion_model_and_installed_boundary_queries_under_iid_uniform_grid",
        native_parent=native, conditional_kernels=conditional,
        complete_refinement_pair_kernels=pair_rows, complete_input_law_envelope=envelope,
        full_stream_iid_approximation_obstruction=support_obstruction,
        conditional_kernel_count=len(conditional), source_core_units=parameter["source_core_units"],
        coarse_dt_ms=parameter["coarse_dt_ms"], duration_ms=parameter["duration_ms"],
        assumptions=["Each attempt receives a fresh independent uniform integer word.",
                     "Restart restores identical attempt parameters and residual state; no retry ceiling.",
                     "The installed sampler implements the published monotone legacy inversion branch.",
                     "Binary64 comparisons/subtraction preserve order and all masses stay finite and positive."],
        zero_count_uniform_consumption_verified=True,
        parent_bound_derived_from_published_inversion_not_ideal_binomial=True,
        native_parent_mean_assumption_certified_for_actual_mt19937=False,
        finite_prng_law_certified=False, finite_sampler_law_certified=False,
        compiled_source_equivalence_proven=False, source_history_confidence_certified=False,
        confirmation_execution_released=False, scientific_fidelity_qualified=False,
        numerical_margins_changed=False, error_budget_allocated=False,
        model_or_experiment_seed_used=False, prng_seed_or_state_used=False,
        source_simulations=0, new_trials=0, new_episodes=0, production_episodes=0,
        numpy_version=np.__version__, platform=platform.platform(),
        provenance=dict(explicit_paths_read=len(reader.snapshots), bytes_read=reader.total_bytes,
                        consumed_artifacts=reader.records()))
    freeze_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config, args.source_root, args.output)
    print(json.dumps({key: report[key] for key in ("status", "conditional_kernel_count", "source_simulations", "production_episodes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
