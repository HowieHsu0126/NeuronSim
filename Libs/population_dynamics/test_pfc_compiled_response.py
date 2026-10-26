"""Compiled task programs and causal source-bound population responses."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from .benchmark_tasks import pfc_working_memory_config, task_protocol
from .circuit_data import CircuitDataset
from .pfc_compiled_response import (prepare_compiled_response, _program, _same_prefix,
                                    candidate_population_metadata)
from .pfc_population_evaluation import prepare_factual_response
from .poisson_state_space import LatentRandomStreams
from .pfc_renewal_data import PfcRenewalEpisode
from .test_pfc_factual_response import fitted_view
from .test_pfc_source_dataset import PfcSourceDatasetTests
from . import pfc_source_dataset as exporter


@pytest.fixture
def compiled():
    fixture = PfcSourceDatasetTests()
    fixture.setUp()
    try:
        source, identity = fixture.benchmark_source()
        exporter.export([source], fixture.config, fixture.root / "compiled_export")
        dataset = CircuitDataset(fixture.root / "compiled_export/dataset", splits=("development",))
        entry = Path(__file__).resolve().parents[1] / (
            "external/pfc_working_memory/neurodynex3/working_memory_network/wm_model.py")
        yield PfcRenewalEpisode(dataset, identity["episode_id"], bin_ticks=100,
                               source_run=source, source_entry=entry)
    finally:
        fixture.doCleanups()


def condition(data, index):
    identity = deepcopy(data.source_config["benchmark_episode_candidate"]["identity"])
    identity["condition_index"] = index
    identity["episode_id"] = identity["episode_id"][:-2] + f"{index:02d}"
    return pfc_working_memory_config(data.source_config, task_protocol("pfc_working_memory"), identity)


def same_tensors(left, right):
    for name in ("mean", "variance", "ready_in_ticks"):
        assert torch.equal(getattr(left, name), getattr(right, name))


def test_same_compiled_condition_matches_factual(compiled):
    view = fitted_view(compiled)
    factual, commands = prepare_factual_response(view, compiled, 2000, 2)
    rng = torch.get_rng_state().clone()
    candidate, boundary, future = prepare_compiled_response(view, compiled, 2000, 2, condition(compiled, 4))
    assert torch.equal(rng, torch.get_rng_state())
    assert candidate.source_metadata_identity == view.source_metadata_identity == boundary.source_metadata_identity
    same_tensors(boundary, factual)
    for name in ("known_controls", "segment_start_ticks", "segment_stop_ticks", "segment_values"):
        np.testing.assert_array_equal(future[name], commands[name])


@pytest.mark.parametrize("index", list(range(10)))
def test_all_registered_conditions_use_factual_prefix_and_shared_parameters(compiled, index):
    view = fitted_view(compiled)
    factual, _ = prepare_factual_response(view, compiled, 2000, 2)
    original, calls = compiled.window, []
    def prefix(start, stop):
        calls.append((start, stop))
        assert (start, stop) == (0, 2000)
        return original(start, stop)
    rng = torch.get_rng_state().clone()
    with patch.object(compiled, "window", prefix):
        candidate, boundary, future = prepare_compiled_response(view, compiled, 2000, 2, condition(compiled, index))
    assert calls == [(0, 2000)]
    assert torch.equal(rng, torch.get_rng_state())
    same_tensors(boundary, factual)
    assert boundary.source_metadata_identity == candidate.source_metadata_identity
    assert all(parameter is dict(candidate.named_parameters())[name]
               for name, parameter in view.named_parameters())
    assert torch.equal(candidate.unit_field_index, view.unit_field_index)
    assert not future["registered_response_binding"]["candidate_outcomes_read"]
    assert not future["registered_response_binding"]["scientific_intervention_qualified"]
    assert future["known_controls"].shape == (2, 4)
    assert np.all(future["known_controls"][:, 0] == (0.5 if index < 5 else 0))
    assert np.all(future["known_controls"][:, 2] == 1)


def test_distractor_geometry_transfer_preserves_factual_boundary(compiled):
    view = fitted_view(compiled)
    factual, _ = prepare_factual_response(view, compiled, 8000, 2)
    with patch.object(view, "infer_state", wraps=view.infer_state) as inference:
        candidate, boundary, future = prepare_compiled_response(view, compiled, 8000, 2, condition(compiled, 1))
    assert inference.call_count == 1
    same_tensors(boundary, factual)
    assert boundary.source_metadata_identity != factual.source_metadata_identity
    assert not torch.equal(candidate.unit_features, view.unit_features)
    assert not torch.equal(candidate.field_metadata, view.field_metadata)
    assert np.all(future["known_controls"][:, 1] == 0.5)
    assert np.all(future["known_controls"][:, 3] == 1)
    with pytest.raises(ValueError, match="source|boundary"):
        next(candidate.iter_simulation(factual, future["known_controls"][None],
             LatentRandomStreams.from_seeds(41, 42, 43), 2))
    steps = list(candidate.iter_simulation(boundary, future["known_controls"][None],
                 LatentRandomStreams.from_seeds(41, 42, 43), 2))
    assert len(steps) == 2 and steps[0].counts.shape == (2, 1, len(compiled.unit_ids))


@pytest.mark.parametrize("index,origin", [(5, 6000), (1, 8100), (2, 8100), (0, 48100)])
def test_changed_source_unit_prefix_rejects_before_outcomes(compiled, index, origin):
    with patch.object(compiled, "window", side_effect=AssertionError("CHANGED_PREFIX_READ")):
        with pytest.raises(ValueError, match="SOURCE_UNIT_COMMAND_PREFIX_CHANGED"):
            prepare_compiled_response(fitted_view(compiled), compiled, origin, 2, condition(compiled, index))


@pytest.mark.parametrize("name,value,reason", [
    ("excitatory_preference_bins", 2, "NONCONTROL_SOURCE_PARAMETER"),
    ("weight_scaling_factor", 2., "NORMALIZATION"),
    ("distractor_center_deg", 1., "COMPILER_BINDING"),
    ("initial_rng_seed", 10, "COMPILER_BINDING"),
    ("duration_ms", 5001., "COMPILER_BINDING"),
])
def test_unregistered_changes_reject_before_outcomes(compiled, name, value, reason):
    config = condition(compiled, 1)
    config[name] = value
    with patch.object(compiled, "window", side_effect=AssertionError("UNQUALIFIED_READ")):
        with pytest.raises(ValueError, match=reason):
            prepare_compiled_response(fitted_view(compiled), compiled, 8000, 2, config)


def test_foreign_block_split_and_noise_reject(compiled):
    identity = deepcopy(compiled.source_config["benchmark_episode_candidate"]["identity"])
    for name, value in (("block_index", 1), ("split", "test"), ("background_seed", 123)):
        changed = dict(identity, **{name: value})
        config = pfc_working_memory_config(compiled.source_config, task_protocol("pfc_working_memory"), changed)
        with patch.object(compiled, "window", side_effect=AssertionError("FOREIGN_READ")):
            with pytest.raises(ValueError, match="SAME_CIRCUIT_BLOCK_AND_SPLIT"):
                prepare_compiled_response(fitted_view(compiled), compiled, 2000, 2, config)


def test_wrong_factual_geometry_rejects_before_outcomes(compiled):
    view, changed = fitted_view(compiled), condition(compiled, 1)
    metadata = candidate_population_metadata(compiled, changed, view.metadata["control_features"])
    foreign = view.training_for_circuit(unit_ids=list(map(int, compiled.unit_ids)),
        unit_field_index=compiled.unit_field_index, field_names=compiled.field_names,
        minimum_interval_ticks=compiled.minimum_interval_ticks, metadata=metadata)
    with patch.object(compiled, "window", side_effect=AssertionError("WRONG_VIEW_READ")):
        with pytest.raises(ValueError, match="FACTUAL_SOURCE_COMMAND_BINDING"):
            prepare_compiled_response(foreign, compiled, 2000, 2, changed)


def test_future_events_never_change_preparation(compiled):
    view = fitted_view(compiled)
    _, boundary, future = prepare_compiled_response(view, compiled, 8000, 2, condition(compiled, 2))
    compiled.event_ticks[compiled.event_ticks >= 8000] += 1
    _, again, commands = prepare_compiled_response(view, compiled, 8000, 2, condition(compiled, 2))
    same_tensors(boundary, again)
    np.testing.assert_array_equal(future["known_controls"], commands["known_controls"])


def test_exact_prefix_retains_active_zero_flags(compiled):
    from .pfc_source_execution import _case_config
    case, _, dimensions = _case_config(compiled.source_config, compiled.episode_id)
    factual = _program(compiled.source_config, case, dimensions)
    other = deepcopy(factual)
    other["native"]["stimulus"]["amplitude"] = 0.
    with pytest.raises(ValueError, match="PREFIX_CHANGED"):
        _same_prefix(factual, other, 2100, dimensions["n_e"])
    factual["native"]["stimulus"]["amplitude"] = 0.
    other["native"]["stimulus"]["end"] = 0.
    with pytest.raises(ValueError, match="PREFIX_CHANGED"):
        _same_prefix(factual, other, 2100, dimensions["n_e"])


def test_metadata_extraction_preserves_authenticated_descriptors(compiled):
    original = compiled.population_metadata()
    actual = candidate_population_metadata(compiled, compiled.source_config,
                                           compiled.model_control_features)
    assert actual == original


def test_same_scalar_prefix_with_changed_target_indices_rejects(compiled):
    from .pfc_source_execution import _case_config
    case, _, dimensions = _case_config(compiled.source_config, compiled.episode_id)
    factual = _program(compiled.source_config, case, dimensions)
    candidate = deepcopy(factual)
    candidate["native"]["stimulus"]["indices"] = np.array([0, 1, 2])
    np.testing.assert_array_equal(candidate["values"], factual["values"])
    with pytest.raises(ValueError, match="SOURCE_UNIT_COMMAND_PREFIX_CHANGED"):
        _same_prefix(factual, candidate, 2100, dimensions["n_e"])


def test_held_noncue_current_is_retained_until_native_reset(compiled):
    from .pfc_source_execution import _case_config
    from .pfc_deterministic_replay import _held_current
    case, _, dimensions = _case_config(compiled.source_config, compiled.episode_id)
    config = dict(compiled.source_config, stimulus_center_deg=0., stimulus_width_deg=0.,
                  distractor_start_ms=150., distractor_duration_ms=10.,
                  distractor_center_deg=22.5, distractor_width_deg=0.)
    program = _program(config, case, dimensions)
    candidate = _program(dict(config, distractor_center_deg=45.), case, dimensions)
    current = np.zeros(dimensions["n_e"])
    _held_current(current, program["native"], .15)
    _held_current(current, program["native"], .16)
    assert current[1] == .5e-9  # The active cue does not clear noncue targets.
    with pytest.raises(ValueError, match="SOURCE_UNIT_COMMAND_PREFIX_CHANGED"):
        _same_prefix(program, candidate, 3300, dimensions["n_e"])
    _held_current(current, program["native"], .301)
    assert np.all(current == 0)
