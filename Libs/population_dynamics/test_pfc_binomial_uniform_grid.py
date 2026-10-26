"""Deterministic finite-grid tests; no PRNG seed or scientific episode is used."""
from fractions import Fraction
import json

import numpy as np
import pytest

from population_dynamics import pfc_binomial_uniform_grid as grid


def exact(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


@pytest.mark.parametrize("trials,probability", [(0, .5), (1, .5), (3, .1), (8, 1.75e-5), (1000, 7e-5)])
def test_integer_census_matches_exhaustive_small_grid(trials, probability):
    oracle = grid.InversionOracle(trials, probability, bits=8)
    law = grid.finite_law(oracle)
    labels = [oracle(word) for word in range(oracle.grid_size)]
    assert all(a <= b for a, b in zip(labels[:-1], labels[1:]))
    assert law["acceptance_grid_counts_by_k"] == [labels.count(k) for k in range(oracle.bound + 1)]
    assert law["restart_grid_points"] == labels.count(oracle.bound + 1)
    assert sum(exact(mass) for mass in law["normalized_mass_by_k"]) == 1
    assert law["compiled_source_equivalence_proven"] is False


def test_strict_boundary_at_half_and_grid_quantization():
    oracle = grid.InversionOracle(1, .5)
    assert oracle(2 ** 52) == 0
    assert oracle(2 ** 52 + 1) == 1
    law = grid.finite_law(oracle)
    assert law["acceptance_grid_counts_by_k"] == [2 ** 52 + 1, 2 ** 52 - 1]
    assert exact(law["mean"]) == Fraction(1, 2) - Fraction(1, 2 ** 53)
    assert grid.binomial_tv(law) == Fraction(1, 2 ** 53)


def test_native_restart_observed_as_largest_attempt_label():
    oracle = grid.InversionOracle(1000, 7e-5)
    label = oracle(2 ** 53 - 1)
    assert oracle.driver.draws == 2
    assert label == oracle.bound + 1
    law = grid.finite_law(oracle)
    assert law["restart_grid_points"] > 0
    assert law["accepted_grid_points"] + law["restart_grid_points"] == 2 ** 53
    assert not law["whole_grid_exhaustively_enumerated"]
    assert not law["actual_mt19937_law_qualified"]


def test_zero_counts_still_consume_one_uniform():
    oracle = grid.InversionOracle(0, 1.75e-5)
    assert oracle(2 ** 53 - 1) == 0
    assert oracle.driver.draws == 1
    law = grid.finite_law(oracle)
    assert exact(law["mean"]) == grid.binomial_tv(law) == 0


def test_binomial_tv_keeps_outside_support_tail():
    # A degenerate law at zero is distance 1-P(Binomial(3,1/2)=0)=7/8.
    law = dict(trials=3, probability=dict(numerator="1", denominator="2"),
               accepted_grid_points=16, acceptance_grid_counts_by_k=[16])
    assert grid.binomial_tv(law) == Fraction(7, 8)


def test_restart_normalization_not_uniform_grid_denominator():
    # Twelve accepted words, four restarts, eight accepted zeros and four ones.
    law = dict(trials=1, probability=dict(numerator="1", denominator="2"),
               accepted_grid_points=12, acceptance_grid_counts_by_k=[8, 4])
    assert grid.binomial_tv(law) == Fraction(1, 6)


@pytest.mark.parametrize("trials,probability,bits", [(-1, .1, 53), (1001, .01, 53), (1000, .5, 53),
    (1, 0., 53), (1, .6, 53), (1, float('nan'), 53), (True, .1, 53), (1, .1, 54), (1, .1, True)])
def test_unsupported_domain_rejected(trials, probability, bits):
    with pytest.raises(ValueError, match="PFC_BINOMIAL_GRID"):
        grid.InversionOracle(trials, probability, bits)


@pytest.mark.parametrize("word", [-1, 2 ** 53, 1., True])
def test_invalid_grid_word_rejected(word):
    oracle = grid.InversionOracle(1, .5)
    with pytest.raises(ValueError, match="IN_RANGE_INTEGER_UNIFORM"):
        oracle(word)


@pytest.mark.parametrize("trials", [0, 1, 8])
@pytest.mark.parametrize("probability", [1.75e-5, .5])
def test_existing_refinement_helper_agrees_at_all_acceptance_boundaries(trials, probability):
    oracle = grid.InversionOracle(trials, probability)
    law = grid.finite_law(oracle)
    assert grid._small_model_matches(oracle, law) >= 2


def test_hex_witness_roundtrips_without_changing_decimal_limits():
    value = Fraction(2 ** 67000 - 1, 2 ** 67001)
    encoded = json.loads(json.dumps(grid._hex_fraction(value)))
    assert Fraction(int(encoded["numerator_hex"], 16), int(encoded["denominator_hex"], 16)) == value


def pair_fixture():
    p0 = 7e-5
    laws = {}
    for level in range(2):
        route = grid.routing_law(p0 / 2 ** level)
        for name, probability in [(f"zero_level_{level}", float(route["q"])),
                                  (f"double_level_{level}", float(route["r"]))]:
            for n in (0, 1):
                laws[name, n] = grid.finite_law(grid.InversionOracle(n, probability, bits=4))
    for n in (0, 1):
        laws["single_child_allocation", n] = grid.finite_law(grid.InversionOracle(n, .5, bits=4))
    return p0, laws


def test_pair_composition_retains_zero_parent_identity():
    p0, laws = pair_fixture()
    pair = grid.refinement_pair_law(laws, 0, 0, p0)
    assert pair["joint"] == {(0, 0): Fraction(1)}
    assert pair["tv"] == pair["left_mean"] == pair["right_mean"] == 0


def test_finite_grid_pair_keeps_asymmetry_and_full_ideal_support():
    p0, laws = pair_fixture()
    pair = grid.refinement_pair_law(laws, 1, 0, p0)
    assert pair["joint"].get((0, 1)) == Fraction(9, 16)
    assert pair["joint"].get((1, 0)) == Fraction(7, 16)
    assert pair["left_mean"] - pair["right_mean"] == Fraction(-1, 8)
    assert pair["tv"] == Fraction(1, 16) + grid.routing_law(p0)["q"]


def test_shared_root_counted_once_in_two_tree_envelope():
    p0, laws = pair_fixture()
    pairs = {(level, n): grid.refinement_pair_law(laws, n, level, p0 / 2 ** level)
             for level in range(2) for n in (0, 1)}
    native = grid.finite_law(grid.InversionOracle(1, .5, bits=4))
    result = grid.complete_input_envelope(native, pairs, units=1, parent_ticks=1)
    decode = lambda obj: Fraction(int(obj["numerator_hex"], 16), int(obj["denominator_hex"], 16))
    one = decode(result["one_tree_complete_input_tv_upper"])
    two = decode(result["shared_root_two_tree_complete_input_tv_upper"])
    root = grid.binomial_tv(native)
    assert two == 2 * one - root
    assert result["no_native_parent_count_mean_assumption_used"]
    assert not result["actual_mt19937_joint_law_qualified"]


@pytest.mark.parametrize("units,ticks", [(0, 1), (1, 0), (True, 1)])
def test_invalid_envelope_dimensions_rejected(units, ticks):
    with pytest.raises(ValueError, match="POSITIVE_INPUT_ENVELOPE_DIMENSIONS"):
        grid.complete_input_envelope({}, {}, units, ticks)


def test_seeded_full_stream_support_obstruction_is_not_neural_failure():
    native = grid.finite_law(grid.InversionOracle(1000, 7e-5))
    result = grid.finite_seed_support_obstruction(native, 100000 * 100000)
    assert result["full_stream_tv_lower"]["exponent"] == -624999968
    assert result["integer_witness"]["twice_numerator_power"] <= result["integer_witness"]["denominator_power"]
    assert not result["uniform_seed_distribution_required"]
    assert not result["low_dimensional_observable_law_error_lower_bound_implied"]
    assert not result["neural_fidelity_failure_implied"]


def test_support_obstruction_clips_noninformative_lower_bound():
    native = grid.finite_law(grid.InversionOracle(1000, 7e-5))
    result = grid.finite_seed_support_obstruction(native, 16, seed_bits=32)
    assert result["full_stream_tv_lower"] == dict(expression="max(0, 1-2**exponent)", exponent=31)


def test_support_obstruction_requires_a_valid_point_mass_bound():
    native = dict(accepted_grid_points=16, acceptance_grid_counts_by_k=[16])
    with pytest.raises(ValueError, match="IID_MAXIMUM_POINT_MASS_BOUND"):
        grid.finite_seed_support_obstruction(native, 1000)
