"""Exact enumeration and outward enclosure tests for ideal input tails."""
from fractions import Fraction
from itertools import product

import pytest

from .pfc_input_envelope import analyze, binomial_tail, outward_probability


@pytest.mark.parametrize("n,p", [(1,Fraction(1,2)), (4,Fraction(1,3)),
                                 (6,Fraction(2,5)), (3,Fraction(0)), (3,Fraction(1))])
def test_tail_matches_full_bernoulli_enumeration(n, p):
    for cap in range(n + 1):
        exact = sum((p ** sum(bits) * (1-p) ** (n-sum(bits))
                     for bits in product((0,1), repeat=n) if sum(bits)>cap), Fraction(0))
        assert binomial_tail(n,p,cap) == exact


@pytest.mark.parametrize("value", [Fraction(0),Fraction(1),Fraction(1,3),
                                   Fraction(1,10**30),Fraction(9,10**24)])
def test_outward_display_never_understates_probability(value):
    row = outward_probability(value)
    upper = Fraction(row["numerator"],row["denominator"])
    assert value <= upper < value + Fraction(1,10**24)
    assert Fraction(row["upper_decimal"]) == upper


def test_scope_union_uses_full_units_clocks_conditions_blocks_and_circuits():
    report = analyze([9],conditions=10,blocks=128,circuits=100)
    row = report["candidate_cutoffs"][0]
    clock_rows = row["clocks"]
    assert [x["input_slots"] for x in clock_rows] == [10**10,2*10**10,4*10**10]
    exact = sum((x["input_slots"] * binomial_tail(1000,Fraction(**x["probability"]),9)
                 for x in clock_rows), Fraction(0)) * 10 * 128 * 100
    assert Fraction(**{k:row["declared_scope_union"][k] for k in ("numerator","denominator")}) >= exact
    assert report["cutoff_selected"] is report["error_budget_allocated"] is False
    assert report["finite_prng_law_certified"] is False


@pytest.mark.parametrize("n,p,cap", [(True,Fraction(1,2),0),(0,Fraction(1,2),0),
                                    (1,.5,0),(1,Fraction(2),0),(1,Fraction(1,2),-1),
                                    (1,Fraction(1,2),2)])
def test_invalid_binomial_contract_rejects(n,p,cap):
    with pytest.raises(ValueError):
        binomial_tail(n,p,cap)


@pytest.mark.parametrize("counts,conditions,blocks,circuits", [([],3,128,1),([9,9],3,128,1),
    ([True],3,128,1),([9],0,128,1),([9],3,0,1),([9],3,128,False)])
def test_missing_or_invalid_declared_scope_rejects(counts,conditions,blocks,circuits):
    with pytest.raises(ValueError):
        analyze(counts,conditions=conditions,blocks=blocks,circuits=circuits)
