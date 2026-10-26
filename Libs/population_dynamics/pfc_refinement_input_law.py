"""Exact input-law perturbation from binary64 PFC refinement parameters.

The analysis changes only the conditional double-child probability to its exact
real value at the unchanged binary64 zero-child probability. It uses ideal
conditional binomial draws, not the finite PRNG or sampler implementation.
No source, neural simulation, sampling trial or production episode is run.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

from . import pfc_clock_coupling, pfc_deterministic_replay, pfc_native_drive
from .pfc_deterministic_replay import _dimensions
from .pfc_native_drive import _BINOMIAL_N, _RATE_HZ, SOURCE_ENTRY_SHA256
from .provenance import freeze_json, sha256_file

CATEGORY_NAMES = ("none", "both", "left_only", "right_only")


def _rational(value: Fraction) -> dict:
    """Keep proof values exact; the float is for display only."""
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        "display_float": float(value),
        "display_float_for_display_only": True,
    }


def routing_law(probability: float) -> dict:
    """Return exact category laws for the actual binary64 q and r arguments.

    Each active parent has four possible child outcomes. Independent ideal
    conditional binomial draws are equivalent to independent category marks
    over the active parents. The comparison retains q and replaces only r.
    """
    if (type(probability) not in (int, float)
            or not math.isfinite(probability) or not 0 < probability <= 1):
        raise ValueError("PFC_REFINEMENT_VALID_BINARY64_PROBABILITY_REQUIRED")
    if (sys.float_info.radix, sys.float_info.mant_dig, sys.float_info.max_exp) != (2, 53, 1024):
        raise ValueError("PFC_REFINEMENT_BINARY64_ARITHMETIC_REQUIRED")
    probability = float(probability)
    q_float = probability / 4.0
    if q_float == 0:
        raise ValueError("PFC_REFINEMENT_BINARY64_QUARTER_UNDERFLOW")
    # Reproduce both floating operations passed to split_counts' second draw.
    r_float = q_float / (1.0 - q_float)
    p = Fraction.from_float(probability)
    q = Fraction.from_float(q_float)
    r = Fraction.from_float(r_float)
    r_star = q / (1 - q)
    actual = (q, (1 - q) * r, (1 - q) * (1 - r) / 2, (1 - q) * (1 - r) / 2)
    ideal = (q, q, (1 - 2 * q) / 2, (1 - 2 * q) / 2)
    rho = q - (1 - q) * r
    return {
        "parent_probability": p,
        "q": q,
        "q_exact_parent_quarter": p / 4,
        "q_rounding_error": q - p / 4,
        "r": r,
        "r_star": r_star,
        "rho": rho,
        "category_total_variation": abs(rho),
        "actual_categories": dict(zip(CATEGORY_NAMES, actual)),
        "ideal_categories": dict(zip(CATEGORY_NAMES, ideal)),
    }


def mean_parent_envelope(probability: float, *, trials: int, units: int,
                         parent_ticks: int, levels: int) -> dict:
    """Conditional refinement TV bound using an assumed native count mean.

    Require E[K_0,u,t] <= N*p0 for every root. Ideal binomial root sampling
    supplies that moment, but this function does not verify the finite sampler.
    A child has conditional mean K*(P(both)+P(left_only)). Before the first
    coupling mismatch, K is common; E[K*1(no prior mismatch)] <= E[K]. Summing
    the resulting unconditional first-mismatch probabilities needs no root
    independence. Empirical event averages cannot establish the root moment.
    """
    if any(type(value) is not int or value <= 0 for value in (trials, units, parent_ticks, levels)):
        raise ValueError("PFC_REFINEMENT_POSITIVE_MOMENT_DIMENSIONS_REQUIRED")
    native = routing_law(probability)
    mean = trials * native["parent_probability"]
    total, rows = Fraction(0), []
    for level in range(levels):
        law = routing_law(float(probability) / 2 ** level)
        left = law["actual_categories"]["both"] + law["actual_categories"]["left_only"]
        right = law["actual_categories"]["both"] + law["actual_categories"]["right_only"]
        if left != right or left != (1 - law["rho"]) / 2:
            raise ValueError("PFC_REFINEMENT_CHILD_MOMENT_IDENTITY_REQUIRED")
        ticks = parent_ticks * 2 ** level
        contribution = units * ticks * mean * law["category_total_variation"]
        rows.append(dict(level=level, parent_ticks=ticks,
                         parent_count_mean_upper=_rational(mean),
                         conditional_child_mean_per_parent=_rational(left),
                         first_mismatch_union_contribution=_rational(contribution)))
        total += contribution
        mean *= left
    return dict(scope="conditional_native_parent_moment_input_law_bound",
                native_parent_count_mean_upper=_rational(trials * native["parent_probability"]),
                levels=rows, unclamped_union=_rational(total),
                full_history_tv_upper=_rational(min(Fraction(1), total)),
                native_parent_mean_assumption="For each source unit and coarse tick, E[K] <= N*p0.",
                native_parent_mean_verified_for_finite_sampler=False,
                native_parent_mean_estimated_from_observed_counts=False,
                root_or_block_independence_required_for_input_bound=False,
                ideal_conditional_routing_required=True,
                finite_prng_law_certified=False, finite_sampler_law_certified=False)


def analyze(config: dict, *, levels: int = 2) -> dict:
    """Bound the joint routing-history perturbation without drawing inputs.

    At every level and source neuron, the parent count is at most N. Successive
    maximal category couplings give the deterministic-parent envelope below.
    Parent-count independence and a parent Binomial marginal are not needed.
    """
    if type(levels) is not int or levels <= 0:
        raise ValueError("PFC_REFINEMENT_POSITIVE_INTEGER_LEVELS_REQUIRED")
    required = {"excitatory_neurons", "inhibitory_neurons", "dt_ms", "duration_ms"}
    if not isinstance(config, dict) or not required <= config.keys():
        raise ValueError("PFC_REFINEMENT_SOURCE_DIMENSIONS_REQUIRED")
    dt_ms, duration_ms = config["dt_ms"], config["duration_ms"]
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (dt_ms, duration_ms))
            or not math.isfinite(duration_ms / dt_ms)):
        raise ValueError("PFC_REFINEMENT_FINITE_SOURCE_CLOCK_REQUIRED")
    n_e, n_i, dt_seconds, ticks, _ = _dimensions(config)
    try:
        final_dt_ms = math.ldexp(float(dt_ms), -levels)
    except OverflowError as error:
        raise ValueError("PFC_REFINEMENT_REPRESENTABLE_CLOCK_REQUIRED") from error
    if final_dt_ms <= 0 or not math.isfinite(duration_ms / final_dt_ms):
        raise ValueError("PFC_REFINEMENT_REPRESENTABLE_CLOCK_REQUIRED")
    final_config = dict(config, dt_ms=final_dt_ms)
    _, _, _, final_ticks, _ = _dimensions(final_config)
    if final_ticks != ticks * 2 ** levels:
        raise ValueError("PFC_REFINEMENT_EXACT_TICK_ENVELOPE_REQUIRED")
    p0 = _RATE_HZ * dt_seconds
    units = n_e + n_i
    rows, union = [], Fraction(0)
    for level in range(levels):
        parent_dt_ms = math.ldexp(float(dt_ms), -level)
        parent_ticks = ticks * 2 ** level
        parent_config = dict(config, dt_ms=parent_dt_ms)
        if _dimensions(parent_config)[3] != parent_ticks:
            raise ValueError("PFC_REFINEMENT_EXACT_TICK_ENVELOPE_REQUIRED")
        law = routing_law(math.ldexp(p0, -level))
        contribution = _BINOMIAL_N * units * parent_ticks * law["category_total_variation"]
        union += contribution
        rows.append({
            "level": level,
            "parent_dt_ms": parent_dt_ms,
            "child_dt_ms": parent_dt_ms / 2,
            "parent_ticks": parent_ticks,
            "child_ticks": 2 * parent_ticks,
            "parent_count_support": [0, _BINOMIAL_N],
            "parent_probability": _rational(law["parent_probability"]),
            "q": _rational(law["q"]),
            "q_exact_parent_quarter": _rational(law["q_exact_parent_quarter"]),
            "q_rounding_error": _rational(law["q_rounding_error"]),
            "r": _rational(law["r"]),
            "r_star": _rational(law["r_star"]),
            "rho": _rational(law["rho"]),
            "actual_category_probabilities": {
                name: _rational(value) for name, value in law["actual_categories"].items()
            },
            "ideal_category_probabilities": {
                name: _rational(value) for name, value in law["ideal_categories"].items()
            },
            "per_active_parent_tv": _rational(law["category_total_variation"]),
            "parent_count_mismatch_bound_rule": "min(1, K * abs(rho)) for every integer 0 <= K <= N",
            "maximum_parent_count_mismatch_bound": _rational(
                min(Fraction(1), _BINOMIAL_N * law["category_total_variation"])
            ),
            "history_union_contribution": _rational(contribution),
        })
    full_bound = min(Fraction(1), union)
    return {
        "status": "COMPLETE_EXACT_PFC_REFINEMENT_PARAMETER_INPUT_LAW_ANALYSIS",
        "scope": "ideal_conditional_input_law_parameter_perturbation_only",
        "comparison_changes_only": "binary64_r_to_exact_real_r_star_at_unchanged_binary64_q",
        "arithmetic": "Fraction.from_float parameters; exact rational category probabilities and union bounds",
        "decision_arithmetic": "exact_numerator_denominator; display_float_is_not_a_bound",
        "binomial_n": _BINOMIAL_N,
        "external_rate_hz": _RATE_HZ,
        "external_rate_hz_binary64": _rational(Fraction.from_float(_RATE_HZ)),
        "source_core_units": units,
        "excitatory_neurons": n_e,
        "inhibitory_neurons": n_i,
        "coarse_dt_ms": dt_ms,
        "duration_ms": duration_ms,
        "coarse_ticks": ticks,
        "final_ticks": final_ticks,
        "refinement_levels": levels,
        "category_order": list(CATEGORY_NAMES),
        "levels": rows,
        "deterministic_parent_envelope": {
            "rule": "min(1, N * U * sum_l(T_l * abs(rho_l)))",
            "unclamped_union": _rational(union),
            "full_history_tv_bound": _rational(full_bound),
            "parent_marginal_or_parent_independence_required": False,
            "assumptions": [
                "Every parent count is an integer in [0,N] at every neuron and parent tick.",
                "Conditional routing uses ideal independent binomial draws at the reported exact binary64 q and r parameters.",
                "The comparison uses the same parent law and conditional history, retaining q and replacing only r by q/(1-q) in exact real arithmetic.",
                "Before the first coupling mismatch, parent counts and routing histories are identical; each active parent can be maximally coupled with mismatch abs(rho).",
            ],
        },
        "data_processing": {
            "observable_law_perturbation_tv_bound": _rational(full_bound),
            "assumptions": [
                "The complete input streams are the only changed random inputs.",
                "Initial states, control histories and any other simulator randomness are identically coupled.",
                "Both compared laws use the same complete simulator at each fixed clock.",
            ],
            "conclusion": "Identical complete inputs imply identical trajectories; measurable observation maps cannot increase total variation.",
            "coarse_fine_simulator_equality_implied": False,
        },
        "ideal_iid_parent_bound_reported": False,
        "finite_prng_law_certified": False,
        "finite_sampler_law_certified": False,
        "coarse_fine_neuronal_equality_certified": False,
        "source_history_confidence_certified": False,
        "confirmation_0p01_128_certified": False,
        "confirmation_execution_released": False,
        "confirmation_sampling_design_changed": False,
        "numerical_margins_changed": False,
        "error_budget_allocated": False,
        "protocol_amendment_adopted": False,
        "production_qualified": False,
        "scientific_fidelity_qualified": False,
        "immutable_archived_source_bundle_transfer_verified": False,
        "new_episodes": 0,
        "new_trials": 0,
        "new_sampling": 0,
        "source_simulations": 0,
        "production_episodes": 0,
    }


def run(config_path: Path, output: Path, *, levels: int = 2) -> dict:
    """Write one fresh declared analysis artifact using the project JSON writer."""
    config_path, output = Path(config_path), Path(output)
    if output.exists():
        raise FileExistsError(f"PFC_REFINEMENT_FRESH_ANALYSIS_OUTPUT_REQUIRED:{output}")
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    report = analyze(config, levels=levels)
    report["provenance"] = {
        "source_config_path": str(config_path.resolve()),
        "source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "resolved_source_config": config,
        "source_entry_sha256_declared": SOURCE_ENTRY_SHA256,
        "parameter_operations": [
            "dt_seconds = dt_ms * 1e-3",
            "p0 = _RATE_HZ * dt_seconds",
            "p_l = p0 / 2**l",
            "q_float = p_l / 4.0",
            "r_float = q_float / (1.0 - q_float)",
        ],
        "current_code_sha256": {
            "pfc_refinement_input_law.py": sha256_file(Path(__file__)),
            "pfc_native_drive.py": sha256_file(Path(pfc_native_drive.__file__)),
            "pfc_clock_coupling.py": sha256_file(Path(pfc_clock_coupling.__file__)),
            "pfc_deterministic_replay.py": sha256_file(Path(pfc_deterministic_replay.__file__)),
        },
        "binding_scope": "Current source parameter and routing code only; immutable source bundles require their own matching code verification.",
        "immutable_source_bundles_verified": False,
    }
    freeze_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=2)
    args = parser.parse_args()
    report = run(args.config, args.output, levels=args.levels)
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "source_simulations": report["source_simulations"],
        "new_trials": report["new_trials"],
        "full_history_tv_bound": report["deterministic_parent_envelope"]["full_history_tv_bound"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
