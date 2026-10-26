"""Outcome-blind PFC target selection on authenticated source fixtures."""
from copy import deepcopy
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from . import pfc_population_intervention as intervention
from .pfc_compiled_response import prepare_compiled_response
from .pfc_source_execution import _case_config
from .poisson_state_space import LatentRandomStreams
from .population_state import ControlledPopulationState
from .test_pfc_compiled_response import compiled, condition
from .test_pfc_factual_response import fitted_view


@pytest.fixture
def problem(compiled):
    configuration = fitted_view(compiled).configuration()
    configuration["control_geometry"] = "actuation_only"
    view = ControlledPopulationState(**configuration).double().eval()
    with torch.no_grad():
        view.shared_unit_readout.bias[-1] = 7.
    fields = len(view.field_names)
    target = intervention.ReferenceTarget(
        field_rate_hz=np.stack([np.full((2, fields), value) for value in (0., 40., 90.)]),
        field_names=view.field_names, scale_hz=np.arange(fields, dtype=float) + 10.,
        independent_block_ids=("reference_block_a", "reference_block_b", "reference_block_c"),
        bin_seconds=view.bin_seconds,
    )
    candidates = tuple(intervention.PfcInterventionCandidate(f"condition_{index:02d}", condition(compiled, index))
                       for index in (4, 1))
    return view, compiled, target, candidates


def design(problem, **overrides):
    view, data, target, candidates = problem
    options = dict(origin=8000, candidates=candidates, draws=3)
    options.update(overrides)
    return intervention.design_pfc_intervention(view, data, target, **options)


def independent_energy(samples, reference, scale):
    x = (samples / scale).reshape(len(samples), -1)
    y = (reference / scale).reshape(-1)
    cross = np.sqrt(((x - y) ** 2).sum(axis=1)).mean()
    distances = np.sqrt(((x[:, None] - x[None]) ** 2).sum(axis=-1))
    return (cross - distances.sum() / (2 * len(x) * (len(x) - 1))) / np.sqrt(y.size)


def assert_same_predictions(left, right):
    assert left.keys() == right.keys()
    for name in left:
        for key in ("field_counts", "field_rate_hz", "unit_total_counts", "terminal_ready_in_ticks"):
            np.testing.assert_array_equal(left[name][key], right[name][key])


def test_all_candidates_reference_blocks_units_and_selected_program_are_reproducible(problem):
    view, data, target, candidates = problem
    before = torch.get_rng_state().clone()
    report, predictions = design(problem)
    assert torch.equal(before, torch.get_rng_state())
    assert [row["candidate"] for row in report["candidates"]] == [row.name for row in candidates]
    assert report["candidate_count"] == len(predictions) == len(candidates)
    assert report["reference_target"]["independent_block_ids"] == list(target.independent_block_ids)
    np.testing.assert_array_equal(report["reference_target"]["scale_hz"], target.scale_hz)
    for candidate, row in zip(candidates, report["candidates"]):
        prediction = predictions[candidate.name]
        candidate_view, boundary, future = prepare_compiled_response(
            view, data, 8000, 2, candidate.source_config)
        steps = list(candidate_view.iter_simulation(boundary, future["known_controls"][None],
                     LatentRandomStreams.from_seeds(41, 42, 43), 3, record_events=False))
        counts = torch.stack([step.counts[:, 0] for step in steps], dim=1).numpy()
        expected_fields = np.stack([counts[:, :, data.unit_field_index == field].sum(-1)
                                    for field in range(len(view.field_names))], axis=-1)
        np.testing.assert_array_equal(prediction["field_counts"], expected_fields)
        np.testing.assert_array_equal(prediction["unit_total_counts"], counts.sum(axis=1))
        np.testing.assert_array_equal(prediction["terminal_ready_in_ticks"], steps[-1].ready_in_ticks[:, 0])
        np.testing.assert_array_equal(prediction["field_rate_hz"],
                                      expected_fields / (view.field_unit_counts.numpy() * view.bin_seconds))
        assert row["generated_unit_count_values"] == prediction["generated_unit_count_values"] == counts.size
        expected = [independent_energy(prediction["field_rate_hz"], reference, target.scale_hz)
                    for reference in target.field_rate_hz]
        np.testing.assert_allclose(row["reference_block_scores"], expected, rtol=0, atol=1e-12)
        assert row["objective"] == pytest.approx(np.mean(expected), abs=1e-12)
        assert row["candidate_identity"] == candidate.source_config["benchmark_episode_candidate"]["identity"]
    selected = min(report["candidates"], key=lambda row: (row["objective"], row["candidate"]))
    assert report["selected_candidate"] == selected["candidate"]
    action = report["selected_action"]
    case, _, dimensions = _case_config(action["source_config"], action["candidate_case_id"])
    assert case["case_id"] == action["candidate_case_id"]
    assert dimensions["duration_ticks"] == data.duration_ticks
    assert action["source_config"] == selected["source_config"]
    assert action["registered_response_binding"] == selected["registered_response_binding"]
    assert action["source_binding"] == data.binding
    assert report["full_unit_outputs_generated"] and report["all_candidate_predictions_retained"]
    for key in ("per_bin_unit_counts_retained", "candidate_future_outcomes_read", "native_execution_performed",
                "native_execution_qualified", "success_tolerance_qualified", "inverse_intervention_qualified",
                "scientific_claim_qualified", "native_future_noise_pairing_verified"):
        assert report[key] is False
    repeated, replay = design(problem)
    assert_same_predictions(predictions, replay)
    assert report["candidate_ranking"] == repeated["candidate_ranking"]
    assert torch.equal(before, torch.get_rng_state())


def test_selection_reads_only_the_factual_prefix_and_known_commands(problem):
    _, data, _, candidates = problem
    window, ready, getitem = data.window, data.ready_at, h5py.Group.__getitem__
    calls = []
    prefix_access = 0
    def prefix(start, stop):
        nonlocal prefix_access
        calls.append((start, stop))
        assert (start, stop) == (0, 8000)
        prefix_access += 1
        try:
            return window(start, stop)
        finally:
            prefix_access -= 1
    def boundary(tick):
        nonlocal prefix_access
        assert tick == 8000
        prefix_access += 1
        try:
            return ready(tick)
        finally:
            prefix_access -= 1
    def command_only(group, name):
        assert not str(name).startswith("privileged")
        if str(name).startswith("spike"):
            assert prefix_access > 0
        return getitem(group, name)
    with patch.object(data, "window", prefix), patch.object(data, "ready_at", boundary), \
            patch.object(h5py.Group, "__getitem__", command_only):
        report, predictions = design(problem)
    assert calls == [(0, 8000)] * len(candidates)
    original = data.event_ticks.copy()
    try:
        data.event_ticks[data.event_ticks >= 8000] += 1
        repeated, changed = design(problem)
    finally:
        data.event_ticks[:] = original
    assert_same_predictions(predictions, changed)
    assert report["candidate_ranking"] == repeated["candidate_ranking"]


@pytest.mark.parametrize("forged_mode", [False, True])
def test_static_or_relabelled_model_rejects_before_source_access(problem, forged_mode):
    _, data, target, candidates = problem
    view = fitted_view(data)
    if forged_mode:
        view.control_geometry = "actuation_only"
    with patch.object(intervention, "prepare_compiled_response", side_effect=AssertionError("SOURCE_READ")):
        with pytest.raises(ValueError, match="SOURCE_BOUND_ACTUATION_ONLY"):
            intervention.design_pfc_intervention(view, data, target, origin=8000, candidates=candidates, draws=3)


@pytest.mark.parametrize("failure", ["physics", "prefix"])
def test_invalid_late_candidate_rejects_entire_set_before_sampling(problem, failure):
    _, data, _, candidates = problem
    config = condition(data, 1 if failure == "physics" else 5)
    if failure == "physics":
        config["excitatory_preference_bins"] = 2
    changed = (candidates[0], intervention.PfcInterventionCandidate("invalid_late", config))
    reason = "NONCONTROL_SOURCE_PARAMETER_CHANGED" if failure == "physics" else "SOURCE_UNIT_COMMAND_PREFIX_CHANGED"
    with patch.object(intervention, "sample_candidate", side_effect=AssertionError("EARLY_SAMPLING")):
        with pytest.raises(ValueError, match=reason):
            design(problem, candidates=changed)


def test_inactive_geometry_is_indistinguishable_at_model_input_resolution(problem):
    _, data, _, _ = problem
    candidates = tuple(intervention.PfcInterventionCandidate(str(index), condition(data, index)) for index in (0, 1))
    with patch.object(intervention, "sample_candidate", side_effect=AssertionError("DUPLICATE_INPUT_SAMPLING")):
        with pytest.raises(ValueError, match="INDISTINGUISHABLE_AT_MODEL_INPUT_RESOLUTION"):
            design(problem, origin=2000, candidates=candidates)


def test_zero_learned_response_retains_distinct_inputs_and_uses_deterministic_tie(problem):
    view, _, _, candidates = problem
    with torch.no_grad():
        for parameter in view.shared_field_parameters.parameters():
            parameter.zero_()
    report, predictions = design(problem)
    assert report["candidate_count"] == len(candidates)
    assert report["candidates"][0]["objective"] == report["candidates"][1]["objective"]
    assert report["selected_candidate"] == min(candidate.name for candidate in candidates)
    reverse, reversed_predictions = design(problem, candidates=tuple(reversed(candidates)))
    assert reverse["selected_candidate"] == report["selected_candidate"]
    assert_same_predictions(predictions, reversed_predictions)


@pytest.mark.parametrize("seeds", [(41, 42, 44), (41, 41, 43), (41, 42), (41., 42, 43)])
def test_unregistered_stream_seeds_reject_before_preparation(problem, seeds):
    with patch.object(intervention, "prepare_compiled_response", side_effect=AssertionError("UNREGISTERED_READ")):
        with pytest.raises(ValueError, match="SEEDS_41_42_43"):
            design(problem, stream_seeds=seeds)


@pytest.mark.parametrize("name", ["field_rate_hz", "unit_total_counts", "terminal_ready_in_ticks",
                                 "generated_unit_count_values"])
def test_incomplete_or_nonfinite_prediction_cannot_return_a_partial_selection(problem, name):
    sample = intervention.sample_candidate
    def corrupted(*args, **kwargs):
        result = sample(*args, **kwargs)
        if name == "generated_unit_count_values":
            result[name] -= 1
        elif name == "unit_total_counts":
            result[name] = result[name][:, :-1]
        else:
            result[name] = result[name].astype(float)
            result[name].flat[0] = np.nan
        return result
    with patch.object(intervention, "sample_candidate", corrupted):
        with pytest.raises(ValueError, match="FINITE_COMPLETE_PREDICTION|COMPLETE_UNIT_GENERATION_DENOMINATOR"):
            design(problem)


def test_nonfinite_target_score_rejects_before_selection(problem):
    with patch.object(intervention, "reference_energy_score", return_value=(float("nan"), [0., 0., 0.])):
        with pytest.raises(FloatingPointError, match="FINITE_COMPLETE_REFERENCE_SCORES"):
            design(problem)


def test_duplicate_names_reject_and_caller_configs_remain_unchanged(problem):
    _, _, _, candidates = problem
    original = deepcopy([candidate.source_config for candidate in candidates])
    with patch.object(intervention, "prepare_compiled_response", side_effect=AssertionError("DUPLICATE_READ")):
        with pytest.raises(ValueError, match="DISTINCT_NAMED_COMPILED_CANDIDATES"):
            design(problem, candidates=(candidates[0], candidates[0]))
    report, _ = design(problem)
    report["selected_action"]["source_config"]["duration_ms"] = -1
    assert [candidate.source_config for candidate in candidates] == original
