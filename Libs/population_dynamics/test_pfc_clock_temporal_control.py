"""Tiny software and exact finite-law checks; no neural qualification claim."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import cycle
from math import comb
from types import SimpleNamespace

import numpy as np
import pytest

from . import pfc_coupled_drive as drive_module
from .pfc_clock_coupling import CoupledClockRows, split_counts
from .pfc_clock_temporal_control import TemporalReversalRows, create_temporal_reversal_drive
from .pfc_coupled_drive import create_coupled_drive
from .pfc_native_drive import _state_arrays, create_drive

COARSE_DT_MS = 0.05
REFINEMENT_SEEDS = (95411, 95412)
PROBABILITIES = (Fraction(7, 100000), Fraction(1, 2), Fraction(1))


def _config(factor=2, policy="source_global", duration_ms=3.2):
    config = dict(excitatory_neurons=13, inhibitory_neurons=3,
                  dt_ms=COARSE_DT_MS / factor, duration_ms=duration_ms,
                  random_stream_policy=policy)
    if policy == "source_global":
        config["source_rng_seed"] = 95001
    else:
        config.update(initial_rng_seed=95001, external_noise_seed=95002)
    return config


def _case():
    return dict(case_id="software_temporal_control")


def _native_rows(config):
    initial, coarse = create_drive(dict(config, dt_ms=COARSE_DT_MS), _case())
    return initial, CoupledClockRows(coarse, REFINEMENT_SEEDS)


def _equal_arrays(actual, expected):
    assert set(actual) == set(expected)
    for name in actual:
        assert actual[name].shape == expected[name].shape, name
        assert actual[name].dtype == expected[name].dtype, name
        assert actual[name].tobytes() == expected[name].tobytes(), name


def _consume(drive, sizes):
    parts = {"E": [], "I": []}
    for size in cycle(sizes):
        if not drive.remaining:
            break
        block = drive.take(min(size, drive.remaining))
        for population in parts:
            parts[population].append(block[population])
    return {key: np.concatenate(value) for key, value in parts.items()}


def _binomial_probability(n, k, p):
    return comb(n, k) * p**k * (1 - p)**(n - k)


class _PrescribedBinomials:
    """Visit a specified supported path through the real three-draw kernel."""

    def __init__(self, calls):
        self.calls = list(calls)

    def binomial(self, n, probability):
        expected_n, expected_probability, value = self.calls.pop(0)
        np.testing.assert_array_equal(n, [expected_n])
        assert probability == expected_probability
        return np.array([value], dtype=np.int64)


@lru_cache(maxsize=None)
def _exact_kernel(parent, probability):
    # Weights use rational arithmetic; outputs use the production splitter.
    # Enumerating paths also detects a changed number or order of native draws.
    p_zero = probability / 4
    p_double = p_zero / (1 - p_zero)
    floating_zero = float(probability) / 4
    law = defaultdict(Fraction)
    for zeros in range(parent + 1):
        for doubles in range(parent - zeros + 1):
            singles = parent - zeros - doubles
            for left in range(singles + 1):
                weight = (_binomial_probability(parent, zeros, p_zero)
                          * _binomial_probability(parent - zeros, doubles, p_double)
                          * _binomial_probability(singles, left, Fraction(1, 2)))
                if not weight:
                    continue
                rng = _PrescribedBinomials([
                    (parent, floating_zero, zeros),
                    (parent - zeros, floating_zero / (1 - floating_zero), doubles),
                    (singles, 0.5, left),
                ])
                pair = split_counts(np.array([parent], dtype=np.uint16), float(probability), rng)
                assert not rng.calls
                law[tuple(int(value) for value in pair[:, 0])] += weight
    assert sum(law.values()) == 1
    return dict(law)


def _conditional_tree(parent, probability):
    law = defaultdict(Fraction)
    for (middle_a, middle_b), parent_weight in _exact_kernel(parent, probability).items():
        for (fine_a, fine_b), left_weight in _exact_kernel(middle_a, probability / 2).items():
            for (fine_c, fine_d), right_weight in _exact_kernel(middle_b, probability / 2).items():
                state = (parent, middle_a, middle_b, fine_a, fine_b, fine_c, fine_d)
                law[state] += parent_weight * left_weight * right_weight
    assert sum(law.values()) == 1
    return dict(law)


class _FixedTripletRows(CoupledClockRows):
    """An explicit finite-law fixture, with no random draws or neural state."""

    def __init__(self, triplet):
        self.coarse = None
        self.triplet = triplet
        self.calls = 0

    def take_row(self):
        self.calls += 1
        return self.triplet


def _adapter_transformed_law(law):
    transformed = defaultdict(Fraction)
    for state, probability in law.items():
        triplet = (np.array(state[:1], dtype=np.uint16).reshape(1, 1),
                   np.array(state[1:3], dtype=np.uint16).reshape(2, 1),
                   np.array(state[3:], dtype=np.uint16).reshape(4, 1))
        native = _FixedTripletRows(triplet)
        output = TemporalReversalRows(native).take_row()
        assert native.calls == 1
        assert output[0] is triplet[0]
        transformed[tuple(int(value) for rows in output for value in rows[:, 0])] += probability
    return dict(transformed)


@pytest.mark.parametrize("probability", PROBABILITIES)
@pytest.mark.parametrize("parent", [0, 1, 2])
def test_exact_conditional_tree_law_is_invariant_under_adapter(parent, probability):
    law = _conditional_tree(parent, probability)
    assert _adapter_transformed_law(law) == law


@pytest.mark.parametrize("probability", PROBABILITIES)
@pytest.mark.parametrize("trials", [1, 2])
def test_exact_full_coarse_middle_fine_joint_law_is_invariant(trials, probability):
    joint = defaultdict(Fraction)
    for parent in range(trials + 1):
        coarse_weight = _binomial_probability(trials, parent, probability)
        if coarse_weight:
            for state, weight in _conditional_tree(parent, probability).items():
                joint[state] += coarse_weight * weight
    assert sum(joint.values()) == 1
    assert _adapter_transformed_law(joint) == dict(joint)
    # The complete fine marginal is four independent Binomial(N, p/4) counts.
    fine = defaultdict(Fraction)
    for state, weight in joint.items():
        fine[state[3:]] += weight
    for state, weight in fine.items():
        expected = Fraction(1)
        for count in state:
            expected *= _binomial_probability(trials, count, probability / 4)
        assert weight == expected


@pytest.mark.parametrize("probability", [0.0, -0.01, 1.01, float("nan"), True])
def test_unsupported_probability_rejected_before_auxiliary_draws(probability):
    # The zero-event parent is included above; p=0 is outside the real splitter
    # contract, even though its unconditional zero-input law has a limit.
    with pytest.raises(ValueError, match="VALID_PARENT_BINOMIAL"):
        split_counts(np.array([0], dtype=np.uint16), probability, _PrescribedBinomials([]))


def test_only_time_axis_reverses_and_native_child_arrays_do_not_alias():
    coarse = np.array([[5, 17, 83]], dtype=np.uint16)
    middle = np.array([[2, 7, 29], [3, 11, 31]], dtype=np.uint16)
    fine = np.array([[0, 2, 13], [1, 3, 17], [2, 5, 19], [3, 7, 23]], dtype=np.uint16)
    originals = tuple(array.copy() for array in (coarse, middle, fine))
    rows = _FixedTripletRows((coarse, middle, fine))
    reversed_rows = TemporalReversalRows(rows).take_row()
    assert reversed_rows[0] is coarse
    for observed, expected in zip(reversed_rows, (coarse, middle[[1, 0]], fine[[3, 2, 1, 0]])):
        np.testing.assert_array_equal(observed, expected)
        assert observed.dtype == np.uint16
    for level in (1, 2):
        assert not np.shares_memory(reversed_rows[level], rows.triplet[level])
        np.testing.assert_array_equal(reversed_rows[level].sum(axis=0), originals[level].sum(axis=0))
    np.testing.assert_array_equal(
        reversed_rows[2].reshape(2, 2, 3).sum(axis=1),
        originals[2].reshape(2, 2, 3).sum(axis=1)[[1, 0]],
    )
    reversed_rows[1][:] = 65535
    reversed_rows[2][:] = 65535
    for observed, expected in zip(rows.triplet, originals):
        np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("policy", ["source_global", "independent_initial_external"])
def test_each_row_preserves_coarse_initial_all_rng_states_and_parent_totals(policy):
    config = _config(policy=policy)
    expected_initial, original = _native_rows(config)
    actual_initial, source = _native_rows(config)
    control = TemporalReversalRows(source)
    assert control.coarse is source.coarse
    np.testing.assert_array_equal(actual_initial, expected_initial)
    _equal_arrays(control.state_arrays(), original.state_arrays())
    while original.coarse.remaining:
        expected = original.take_row()
        actual = control.take_row()
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1][[1, 0]])
        np.testing.assert_array_equal(actual[2], expected[2][[3, 2, 1, 0]])
        for level in (1, 2):
            np.testing.assert_array_equal(actual[level].sum(axis=0), expected[level].sum(axis=0))
        np.testing.assert_array_equal(
            actual[2].reshape(2, 2, -1).sum(axis=1),
            expected[2].reshape(2, 2, -1).sum(axis=1)[[1, 0]],
        )
        _equal_arrays(control.state_arrays(), original.state_arrays())
    native_initial, native = create_drive(dict(config, dt_ms=COARSE_DT_MS), _case())
    native.take(native.total_ticks)
    np.testing.assert_array_equal(actual_initial, native_initial)
    coarse_state = {key.removeprefix("coarse__"): value for key, value in control.state_arrays().items()
                    if key.startswith("coarse__")}
    _equal_arrays(coarse_state, _state_arrays(native.get_state()))


@pytest.mark.parametrize("factor", [2, 4])
@pytest.mark.parametrize("sizes", [(1,), (1, 7, 2, 11, 3, 5), (100000,)])
def test_factory_consumer_matches_original_triplet_reversal_across_parent_boundaries(factor, sizes):
    config = _config(factor)
    expected_initial, original = _native_rows(config)
    level = 1 if factor == 2 else 2
    packed = np.concatenate([original.take_row()[level][::-1].copy()
                             for _ in range(original.coarse.total_ticks)])
    expected = {"I": packed[:, :original.coarse.n_i], "E": packed[:, original.coarse.n_i:]}
    initial, drive = create_temporal_reversal_drive(config, _case(), COARSE_DT_MS, REFINEMENT_SEEDS)
    np.testing.assert_array_equal(initial, expected_initial)
    _equal_arrays(_consume(drive, sizes), expected)
    _equal_arrays(drive.final_state_arrays(), original.state_arrays())
    assert drive.event_totals == {key: int(value.sum(dtype=np.uint64)) for key, value in expected.items()}
    assert drive.metadata()["buffered_fine_rows"] == 0


@pytest.mark.parametrize("factor", [2, 4])
def test_control_consumer_returns_independent_blocks_and_rejects_incomplete_final_state(factor):
    config = _config(factor, duration_ms=0.15)
    _, reference = create_temporal_reversal_drive(config, _case(), COARSE_DT_MS, REFINEMENT_SEEDS)
    expected = reference.take(reference.total_ticks)
    _, drive = create_temporal_reversal_drive(config, _case(), COARSE_DT_MS, REFINEMENT_SEEDS)
    with pytest.raises(ValueError, match="COMPLETE_CONSUMPTION"):
        drive.final_state_arrays()
    first = drive.take(1)
    second = drive.take(1)
    for key in first:
        assert not np.shares_memory(first[key], second[key])
        first[key][:] = 65535
        np.testing.assert_array_equal(second[key], expected[key][1:2])
    rest = drive.take(drive.remaining - 1)
    for key in rest:
        np.testing.assert_array_equal(rest[key], expected[key][2:-1])
    with pytest.raises(ValueError, match="COMPLETE_CONSUMPTION"):
        drive.final_state_arrays()
    last = drive.take(1)
    for key in last:
        np.testing.assert_array_equal(last[key], expected[key][-1:])
    _equal_arrays(drive.final_state_arrays(), reference.final_state_arrays())
    with pytest.raises(ValueError, match="REMAINING_TICKS"):
        drive.take(1)
    final = drive.final_state_arrays()
    final["coarse__key"][:] = 0
    _equal_arrays(drive.final_state_arrays(), reference.final_state_arrays())


@pytest.mark.parametrize("factor", [2, 4])
def test_default_factory_retains_original_stream_and_metadata(factor):
    config = _config(factor)
    expected_initial, original = _native_rows(config)
    packed = np.concatenate([original.take_row()[1 if factor == 2 else 2]
                             for _ in range(original.coarse.total_ticks)])
    expected = {"I": packed[:, :original.coarse.n_i], "E": packed[:, original.coarse.n_i:]}
    default_metadata = None
    for kwargs in ({}, {"row_adapter": None}, {"row_adapter": lambda rows: rows}):
        initial, drive = create_coupled_drive(config, _case(), COARSE_DT_MS, REFINEMENT_SEEDS, **kwargs)
        np.testing.assert_array_equal(initial, expected_initial)
        _equal_arrays(_consume(drive, (1, 5, 2, 7)), expected)
        _equal_arrays(drive.final_state_arrays(), original.state_arrays())
        assert "input_order_control" not in drive.metadata()["coupling"]
        if default_metadata is None:
            default_metadata = drive.metadata()
        else:
            assert drive.metadata() == default_metadata


def test_metadata_preserves_native_fields_and_identifies_dependent_control():
    _, source = _native_rows(_config())
    native = source.metadata()
    control = TemporalReversalRows(source)
    metadata = control.metadata()
    assert {key: metadata[key] for key in native} == native
    details = metadata["input_order_control"]
    assert details["same_original_auxiliary_realization"] is True
    assert details["new_independent_blocks"] == 0
    assert details["independent_repeat"] is False
    assert details["middle_row_order"] == [1, 0]
    assert details["fine_row_order"] == [3, 2, 1, 0]
    assert details["per_coarse_tick_per_neuron_totals_preserved"] is True
    assert details["fine_child_totals_follow_reversed_middle_parents"] is True
    assert details["cross_level_event_conservation_claimed"] is False
    assert details["joint_input_law_preserved_by_tree_exchangeability"] is True
    for key in ("unit_axis_permuted", "rng_draw_order_changed", "neural_error_reduction_established",
                "variance_reduction_established", "coverage_established"):
        assert details[key] is False
    details["middle_row_order"][0] = 999
    assert control.metadata()["input_order_control"]["middle_row_order"] == [1, 0]


@pytest.mark.parametrize("bad_adapter", [False, 1, "reverse", object()])
def test_noncallable_adapter_rejected_before_initial_source_draws(monkeypatch, bad_adapter):
    def forbidden_draws(*args, **kwargs):
        raise AssertionError("invalid adapter reached source initial draws")
    monkeypatch.setattr(drive_module, "create_drive", forbidden_draws)
    with pytest.raises(ValueError, match="ROW_ADAPTER_MUST_BE_CALLABLE"):
        create_coupled_drive(_config(), _case(), COARSE_DT_MS, REFINEMENT_SEEDS,
                             row_adapter=bad_adapter)


@pytest.mark.parametrize("malformation", ["none", "coarse", "take_row", "state_arrays", "metadata"])
def test_adapter_must_retain_original_coarse_identity_and_row_interface(malformation):
    def malformed(rows):
        if malformation == "none":
            return None
        attributes = dict(coarse=rows.coarse, take_row=rows.take_row,
                          state_arrays=rows.state_arrays, metadata=rows.metadata)
        attributes[malformation] = object()
        return SimpleNamespace(**attributes)
    with pytest.raises(ValueError, match="ROW_ADAPTER_INTERFACE_OR_COARSE_CHANGED"):
        create_coupled_drive(_config(), _case(), COARSE_DT_MS, REFINEMENT_SEEDS,
                             row_adapter=malformed)


def test_adapter_cannot_consume_before_fine_consumer_construction():
    def consumed(rows):
        rows.take_row()
        return TemporalReversalRows(rows)
    with pytest.raises(ValueError, match="FRESH_COARSE_STREAM"):
        create_coupled_drive(_config(), _case(), COARSE_DT_MS, REFINEMENT_SEEDS,
                             row_adapter=consumed)


def test_temporal_adapter_requires_original_coupled_rows():
    with pytest.raises(ValueError, match="ORIGINAL_COUPLED_ROWS_REQUIRED"):
        TemporalReversalRows(object())


@pytest.mark.parametrize("change", ["clock", "duration", "duplicate_aux", "source_aux"])
def test_temporal_factory_keeps_original_validation_before_source_draws(monkeypatch, change):
    def forbidden_draws(*args, **kwargs):
        raise AssertionError("invalid request reached source initial draws")
    monkeypatch.setattr(drive_module, "create_drive", forbidden_draws)
    config, seeds = _config(), REFINEMENT_SEEDS
    if change == "clock":
        config["dt_ms"] = 0.02
    elif change == "duration":
        config["duration_ms"] = 0.075
    elif change == "duplicate_aux":
        seeds = (95411, 95411)
    else:
        seeds = (95001, 95411)
    with pytest.raises(ValueError):
        create_temporal_reversal_drive(config, _case(), COARSE_DT_MS, seeds)
