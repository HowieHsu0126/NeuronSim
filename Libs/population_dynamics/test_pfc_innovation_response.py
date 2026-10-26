"""Small real-source and exact arithmetic tests for the local response table."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import pytest
import yaml

from . import pfc_innovation_response as response
from . import pfc_conditional_spike_probability as conditional
from . import pfc_deterministic_replay as executor
from .pfc_coupled_drive import create_coupled_drive
from .pfc_clock_temporal_control import create_temporal_reversal_drive


def config(duration=2.):
    return dict(excitatory_neurons=32, inhibitory_neurons=8, dt_ms=.0125, duration_ms=duration,
        random_stream_policy="source_global", source_rng_seed=71,
        synapse_execution="factorized_uniform_projections", weight_scaling_factor=64.,
        stimulus_start_ms=100., stimulus_duration_ms=200., stimulus_center_deg=180., stimulus_width_deg=40.,
        distractor_start_ms=400., distractor_duration_ms=50., distractor_center_deg=90., distractor_width_deg=40.,
        excitatory_preference_bins=16)


def case():
    return dict(case_id="software_input_response", stimulus_strength_nA=0., distractor_strength_nA=0.)


def equal_arrays(a, b):
    assert set(a) == set(b)
    for key in a:
        assert conditional.exact_arrays_equal(a[key], b[key]), key


@pytest.mark.parametrize("variant,factory", [("original", create_coupled_drive), ("reversed", create_temporal_reversal_drive)])
def test_only_declared_coordinates_change_and_reversal_swaps_values(variant, factory):
    initial, reference = factory(config(), case(), .05, [101, 102])
    expected = reference.take(reference.total_ticks)
    initial2, source = factory(config(), case(), .05, [101, 102])
    drive = response.InnovationDrive(source, variant, 3, 2, 1)
    parts = {"E": [], "I": []}
    for count in (37, 53, 11, source.total_ticks - 101):
        block = drive.take(count)
        for key in parts:
            parts[key].append(block[key])
    observed = {key: np.concatenate(value) for key, value in parts.items()}
    expected["E"][88, 3], expected["E"][91, 3] = (2, 1) if variant == "original" else (1, 2)
    equal_arrays(observed, expected)
    equal_arrays(source.final_state_arrays(), reference.final_state_arrays())
    assert conditional.exact_arrays_equal(initial, initial2)
    assert [row["input_tick"] for row in drive.observations] == [88, 91]
    assert drive.metadata()["conditioning_on_seed_or_rng"] is False


@pytest.mark.parametrize("values", [(-1, 0), (1001, 0), (True, 1), (.5, 0)])
def test_input_count_support_rejects_before_rng_advancement(values):
    _, source = create_coupled_drive(config(), case(), .05, [101, 102])
    with pytest.raises(ValueError, match="COUNT_SUPPORT"):
        response.InnovationDrive(source, "original", 0, *values)
    assert source.ticks == 0


def test_unmatched_realized_input_refuses_with_evidence():
    _, source = create_coupled_drive(config(), case(), .05, [101, 102])
    drive = response.InnovationDrive(source, "original", 0, 1, 2, expected_original={88: 1000, 91: 1000})
    with pytest.raises(ValueError, match="REALIZED_COORDINATE_CHANGED"):
        drive.take(source.total_ticks)


def test_small_real_source_wrapped_and_materialized_inputs_are_exact():
    full, short = config(5000.), config(4.)
    initial, reference = create_coupled_drive(full, case(), .05, [101, 102])
    counts = reference.take(320)
    counts["E"][0, 0], counts["E"][3, 0] = 1000, 0
    recorded = executor.simulate(short, case(), initial, counts["E"], counts["I"], backend="numpy")
    initial2, source = create_coupled_drive(full, case(), .05, [101, 102])
    drive = response.InnovationDrive(source, "original", 0, 1000, 0, input_ticks=(0, 3))
    wrapped = executor.simulate(short, case(), initial2, backend="numpy", drive=drive)
    for family in ("raw", "terminal_state"):
        equal_arrays(recorded[family], wrapped[family])
    assert np.any(wrapped["raw"]["spike_unit_indices"] == 0)
    assert source.total_ticks == 400000 and source.ticks == 320


def exact_masses():
    return [Fraction(comb(3, k), 8) for k in range(4)]


def test_directed_cell_products_and_omitted_mass_are_not_renormalized():
    tails = conditional.binomial_tail_bounds(3, 1, 2, digits=4)
    result = response.directed_masses(tails, 4)
    scale, pmf = 10000, exact_masses()
    for row in result["cells"]:
        exact = pmf[row["x"]] * pmf[row["y"]] * scale
        assert int(row["lower_scaled"]) <= exact <= int(row["upper_scaled"])
    omitted = (1 - sum(pmf[:3]) ** 2) * scale
    assert int(result["omitted_mass"]["lower_scaled"]) <= omitted <= int(result["omitted_mass"]["upper_scaled"])
    assert result["normalized_after_truncation"] is False
    assert sum(int(row["upper_scaled"]) for row in result["cells"]) < scale


def test_signed_weights_reverse_direction_for_negative_numerators():
    masses = response.directed_masses(conditional.binomial_tail_bounds(3, 1, 2, digits=4), 4)
    numerators = np.asarray([[-2, 1], [1, -3], [0, 2], [2, -1], [-1, 0], [0, 0], [1, 3], [-2, -3], [2, 2]], dtype=np.int64)
    denominator = np.asarray([2, 3])
    result = response.weighted_bounds(numerators, denominator, masses, signed=True)
    pmf, epsilon = exact_masses(), 1 - sum(exact_masses()[:3]) ** 2
    for column in range(2):
        value = sum(pmf[x] * pmf[y] * Fraction(int(numerators[index, column]), int(denominator[column]))
                    for index, (x, y) in enumerate(response.CELLS))
        assert int(result["lower_scaled"][column]) <= (value - epsilon) * 10000
        assert int(result["upper_scaled"][column]) >= (value + epsilon) * 10000


@pytest.mark.parametrize("digits", [4, 40])
def test_indicator_omitted_mass_is_one_sided(digits):
    masses = response.directed_masses(conditional.binomial_tail_bounds(3, 1, 2, digits=digits), digits)
    result = response.weighted_bounds(np.zeros(9, dtype=np.int64), np.asarray(1), masses, signed=False)
    assert int(result["lower_scaled"].item()) == 0
    assert result["upper_scaled"].item() == masses["omitted_mass"]["upper_scaled"]
    with pytest.raises(ValueError, match="BOUNDED_OUTCOME"):
        response.weighted_bounds(np.full(9, -1, dtype=np.int64), np.asarray(1), masses, signed=False)


def test_own_zero_cell_comparators_remove_fixed_reversal_difference():
    cfg = config()
    a = response.count_features(np.empty((0, 2), dtype=np.int64), cfg)
    b = response.count_features(np.asarray([[100, 1]], dtype=np.int64), cfg)
    h, g, denominator = response.response_features(a, b, a, b)
    assert np.any(h[0]) and not np.any(h[1:])
    assert np.any(g[0]) and not np.any(g[1:])
    assert denominator.shape == (3, 17)


@pytest.mark.parametrize("variant,events", [("original", []), ("reversed", [(89, 6423)])])
def test_realized_paired_cell_is_reused_without_any_simulation(tmp_path, monkeypatch, variant, events):
    values = np.asarray(events, dtype=np.int64).reshape(-1, 2)
    source = tmp_path / "reference.npz"
    np.savez(source, spike_ticks=values[:, 0], spike_unit_indices=values[:, 1])
    def forbidden(*args, **kwargs):
        raise AssertionError("realized cell must not run again")
    monkeypatch.setattr(executor, "simulate", forbidden)
    monkeypatch.setattr(response.pfc_coupled_drive, "create_coupled_drive", forbidden)
    monkeypatch.setattr(response.pfc_clock_temporal_control, "create_temporal_reversal_drive", forbidden)
    plan = dict(full_config=dict(config(), excitatory_neurons=80000, inhibitory_neurons=20000),
                source_costs={variant: {}})
    result = response._run_cell(plan, {variant + "_events.npz": source}, tmp_path / "cell", 0, 1, variant)
    assert result["status"] == "REUSED", result
    assert result["new_simulations"] == 0


def test_conditioning_cannot_include_coarse_outputs_selection_or_rng():
    path = Path(__file__).parents[1] / "configs/neural_population_pfc_innovation_response.yaml"
    settings = yaml.safe_load(path.read_text())
    response.validate_settings(settings)
    for name in ("condition_on_outputs", "condition_on_coarse_parents", "condition_on_selection", "condition_on_rng_or_seed"):
        bad = deepcopy(settings)
        bad[name] = True
        with pytest.raises(ValueError, match="FROZEN_SETTING_CHANGED"):
            response.validate_settings(bad)
    bad = deepcopy(settings)
    bad["kernel_source"]["delivery_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SOURCE_BINDING"):
        response.validate_settings(bad)


@pytest.mark.parametrize("sign", [-1, 1])
def test_signed_scalar_40_digit_bounds_preserve_python_integer_precision(sign):
    masses = response.directed_masses(conditional.binomial_tail_bounds(3, 1, 2, digits=40), 40)
    scale = 10 ** 40
    result = response.weighted_bounds(np.full(9, sign, dtype=np.int64), np.asarray(1), masses, signed=True)
    expected = (-scale, -17 * scale // 32) if sign == -1 else (17 * scale // 32, scale)
    assert int(result["lower_scaled"].item()) == expected[0]
    assert int(result["upper_scaled"].item()) == expected[1]
