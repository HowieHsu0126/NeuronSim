"""Check the usable conditional law against the existing finite-tree oracle."""
from collections import defaultdict
from fractions import Fraction
from itertools import product
from math import comb

import numpy as np
import pytest

from . import pfc_clock_tree_conditioning as tree
from .test_pfc_clock_temporal_control import _conditional_tree


@pytest.mark.parametrize("p", [Fraction.from_float(7e-5), Fraction(1, 2)])
@pytest.mark.parametrize("c", [0, 1, 2])
def test_joint_coefficients_match_existing_nested_exact_kernel(c, p):
    assert tree.joint_tree(c, p) == {state[1:]: value for state, value in _conditional_tree(c, p).items()}


@pytest.mark.parametrize("p", [Fraction.from_float(7e-5), Fraction(1, 2), Fraction(1)])
def test_single_parent_projection_equals_closed_form_and_has_negative_dependence(p):
    fine = tree.fine_projection(tree.joint_tree(1, p))
    assert fine == tree.single_fine_closed_form(p)
    assert sum(v for state, v in fine.items() if sum(state) == 1) == (1 - p / 4) ** 3
    covariance = sum(state[0] * state[1] * v for state, v in fine.items()) - Fraction(1, 16)
    assert covariance == -(1 - p) / 16


def test_mixing_over_coarse_restores_independent_fine_binomials():
    n, p = 2, Fraction(1, 2)
    mixed = defaultdict(Fraction)
    for c in range(n + 1):
        weight = comb(n, c) * p ** c * (1 - p) ** (n - c)
        for state, value in tree.fine_projection(tree.joint_tree(c, p)).items():
            mixed[state] += weight * value
    for state in product(range(n + 1), repeat=4):
        expected = Fraction(1)
        for value in state:
            expected *= comb(n, value) * (p / 4) ** value * (1 - p / 4) ** (n - value)
        assert mixed[state] == expected


def test_latent_single_event_component_does_not_equal_total_count_event():
    p, c = Fraction(1, 2), 2
    law = tree.fine_projection(tree.joint_tree(c, p))
    conserved = sum(weight for state, weight in law.items() if sum(state) == c)
    assert conserved > (1 - p / 4) ** (3 * c)


def test_rounding_comparison_is_nonzero_and_is_not_a_prng_certificate():
    row = tree.rounded_parameter_difference(Fraction(1, 2))
    delta = Fraction(int(row["double_category_signed_difference"]["numerator"]),
                     int(row["double_category_signed_difference"]["denominator"]))
    assert delta != 0 and row["single_split_total_variation"] == abs(delta)


def test_directed_signed_bounds_keep_40_digit_integer_scale():
    value = Fraction(-1, 3)
    result = tree.directed(value)
    assert Fraction(int(result["lower_scaled"]), 10 ** 40) <= value
    assert value <= Fraction(int(result["upper_scaled"]), 10 ** 40)


@pytest.mark.parametrize("c,complete", [(1, True), (3, False)])
def test_conditioned_slice_only_claims_zero_when_all_admissible_states_are_covered(c, complete):
    projected = tree.fine_projection(tree.joint_tree(c, Fraction(1, 2)))
    cells = np.asarray(list(product(range(3), repeat=2)))
    record = tree.response_slice(projected, (0, 0), cells, np.zeros((9, 3, 23), dtype=np.int64))
    assert record["complete_slice_response_coverage"] == complete
    assert record["pointwise_count_invariant_on_complete_slice"] == complete
    upper = record["any_own00_count_change_upper"]
    assert (int(upper["numerator"]) == 0) == complete
