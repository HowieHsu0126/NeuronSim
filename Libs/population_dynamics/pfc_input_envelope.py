"""Exact input-tail bounds under the declared ideal PFC binomial law.

These bounds do not certify the finite PRNG law, neural clock error, threshold
robustness or production qualification. No input cutoff or error allocation is
selected by this diagnostic.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

from .pfc_native_drive import _BINOMIAL_N, _RATE_HZ

CLOCKS_MS = (.05, .025, .0125)
CORE_UNITS = 100000
DURATION_MS = 5000


def binomial_tail(n: int, probability: Fraction, maximum_count: int) -> Fraction:
    """Return P(X > maximum_count) by exact integer arithmetic.

    A common denominator avoids loss when the upper tail is much smaller than
    one. Only terms through the cutoff are needed. No sampling is performed.
    """
    if (type(n) is not int or n <= 0 or not isinstance(probability, Fraction)
            or not 0 <= probability <= 1 or type(maximum_count) is not int
            or not 0 <= maximum_count <= n):
        raise ValueError("PFC_INPUT_ENVELOPE_BINOMIAL_ARGUMENTS")
    if maximum_count == n or probability == 0:
        return Fraction(0)
    if probability == 1:
        return Fraction(1)
    a, d = probability.numerator, probability.denominator
    q = d - a
    denominator = d ** n
    term = q ** n
    cumulative = term
    for j in range(maximum_count):
        numerator = term * (n - j) * a
        divisor = (j + 1) * q
        term, remainder = divmod(numerator, divisor)
        if remainder:
            raise ArithmeticError("PFC_INPUT_ENVELOPE_INTEGER_RECURRENCE")
        cumulative += term
    return Fraction(denominator - cumulative, denominator)


def outward_probability(value: Fraction, digits: int = 24) -> dict:
    """Report a rational upper enclosure on a decimal grid, without floats."""
    if (not isinstance(value, Fraction) or not 0 <= value <= 1
            or type(digits) is not int or not 1 <= digits <= 100):
        raise ValueError("PFC_INPUT_ENVELOPE_PROBABILITY_ENCLOSURE")
    denominator = 10 ** digits
    numerator = (value.numerator * denominator + value.denominator - 1) // value.denominator
    if not value <= Fraction(numerator, denominator):
        raise ArithmeticError("PFC_INPUT_ENVELOPE_UPPER_ENCLOSURE")
    whole, fractional = divmod(numerator, denominator)
    return dict(numerator=numerator, denominator=denominator,
                upper_decimal=f"{whole}.{fractional:0{digits}d}",
                rounding="exact_integer_ceiling")


def analyze(maximum_counts: list[int], *, conditions: int, blocks: int, circuits: int) -> dict:
    if (not maximum_counts or len(maximum_counts) != len(set(maximum_counts))
            or any(type(k) is not int or not 0 <= k <= _BINOMIAL_N for k in maximum_counts)
            or any(type(v) is not int or v <= 0 for v in (conditions, blocks, circuits))):
        raise ValueError("PFC_INPUT_ENVELOPE_EXPLICIT_SCOPE")
    # Preserve the source float probability, then the coupling's exact halvings.
    parent = _RATE_HZ * (CLOCKS_MS[0] * 1e-3)
    probabilities = [Fraction.from_float(parent / factor) for factor in (1, 2, 4)]
    rows = []
    for cap in maximum_counts:
        clock_rows, union = [], Fraction(0)
        for dt_ms, probability in zip(CLOCKS_MS, probabilities):
            ticks = round(DURATION_MS / dt_ms)
            slots = CORE_UNITS * ticks
            tail = binomial_tail(_BINOMIAL_N, probability, cap)
            contribution = slots * tail
            union += contribution
            clock_rows.append(dict(dt_ms=dt_ms, ticks=ticks, input_slots=slots,
                probability=dict(numerator=probability.numerator, denominator=probability.denominator),
                single_draw_tail=outward_probability(tail),
                episode_clock_union=outward_probability(min(Fraction(1), contribution))))
        rows.append(dict(maximum_input_count=cap, clocks=clock_rows,
            one_block_one_condition_three_clock_union=outward_probability(min(Fraction(1), union)),
            declared_scope_union=outward_probability(min(Fraction(1), union * conditions * blocks * circuits))))
    return dict(status="COMPLETE_EXACT_IDEAL_PFC_INPUT_TAIL_ENCLOSURE",
        law="Ideal binomial marginal with exact rational source float probability; coupled half and quarter marginals.",
        binomial_n=_BINOMIAL_N, external_rate_hz=_RATE_HZ, core_units=CORE_UNITS,
        duration_ms=DURATION_MS, conditions=conditions, blocks=blocks, circuits=circuits,
        union_rule="Sum all unit/tick/clock tail probabilities, then multiply by declared conditions, blocks and circuits. Independence among union events is not required.",
        arithmetic="Exact integer binomial CDF complement; rational union; outward integer ceiling for reported bounds.",
        candidate_cutoffs=rows, cutoff_selected=False, error_budget_allocated=False,
        finite_prng_law_certified=False, synaptic_state_bounds_certified=False,
        neural_clock_error_qualified=False, confirmation_execution_released=False,
        production_episodes=0, source_simulations=0, new_sampling=0,
        scientific_fidelity_qualified=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-counts", type=int, nargs="+", required=True)
    parser.add_argument("--conditions", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--circuits", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.maximum_counts, conditions=args.conditions, blocks=args.blocks, circuits=args.circuits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], output=str(args.output))))


if __name__ == "__main__":
    main()
