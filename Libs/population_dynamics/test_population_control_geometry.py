"""PFC target geometry acts only through present source control channels."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from .benchmark_tasks import pfc_working_memory_config, task_protocol
from .circuit_data import CircuitDataset
from .pfc_renewal_data import PfcRenewalEpisode
from .pfc_compiled_response import candidate_population_metadata
from .pfc_source_dataset import _controls
from .population_state import ControlledPopulationState
from .poisson_state_space import LatentRandomStreams
from . import test_pfc_source_dataset as source_fixtures
from . import pfc_source_dataset as exporter


@pytest.fixture
def source_geometry():
    fixture = source_fixtures.PfcSourceDatasetTests()
    fixture.setUp()
    try:
        source, identity = fixture.benchmark_source()
        exporter.export([source], fixture.config, fixture.root / "geometry_export")
        data = CircuitDataset(fixture.root / "geometry_export/dataset", splits=("development",))
        entry = __import__("pathlib").Path(__file__).resolve().parents[1] / (
            "external/pfc_working_memory/neurodynex3/working_memory_network/wm_model.py")
        episode = PfcRenewalEpisode(data, identity["episode_id"], bin_ticks=100,
                                   source_run=source, source_entry=entry)
        factual = episode.population_metadata()
        other_identity = deepcopy(identity)
        other_identity.update(condition_index=8,
                              episode_id=identity["episode_id"][:-2] + "08")
        config = pfc_working_memory_config(episode.source_config, task_protocol("pfc_working_memory"), other_identity)
        candidate = candidate_population_metadata(episode, config, episode.model_control_features)
        yield dict(episode=episode, factual=factual, candidate=candidate, config=config)
    finally:
        fixture.doCleanups()


def model(fixture, seed, mode="actuation_only", metadata=None):
    episode = fixture["episode"]
    return ControlledPopulationState(unit_ids=list(map(int, episode.unit_ids)),
        unit_field_index=episode.unit_field_index, field_names=episode.field_names,
        minimum_interval_ticks=episode.minimum_interval_ticks,
        tick_seconds=episode.dt_ms / 1000, bin_seconds=episode.dt_ms * episode.bin_ticks / 1000,
        input_size=4, latent_channels=4, encoder_size=16, representation="recurrent",
        initialization_seed=seed, unit_chunk_size=7,
        metadata=fixture["factual"] if metadata is None else metadata,
        control_geometry=mode)


def transferred(value, metadata, *, shared=True):
    method = value.training_for_circuit if shared else value.for_circuit
    return method(unit_ids=list(value.unit_ids), unit_field_index=value.unit_field_index,
        field_names=value.field_names, minimum_interval_ticks=value.minimum_interval_ticks,
        metadata=metadata)


def state(value, counts, controls):
    return value.infer_state(dict(counts=counts, stimulus=controls,
                                 ready_in_ticks=torch.zeros(1, len(value.unit_ids), dtype=torch.long)))


@pytest.mark.parametrize("seed", [41, 42, 43])
def test_inactive_distractor_geometry_cannot_change_autonomous_dynamics_or_samples(source_geometry, seed):
    canonical = model(source_geometry, seed)
    candidate = transferred(canonical, source_geometry["candidate"])
    assert canonical.control_geometry == candidate.control_geometry == "actuation_only"
    assert canonical.source_metadata_identity != candidate.source_metadata_identity
    assert not torch.equal(canonical.unit_features, candidate.unit_features)
    assert not torch.equal(canonical.field_metadata, candidate.field_metadata)
    assert all(parameter is dict(candidate.named_parameters())[name]
               for name, parameter in canonical.named_parameters())
    counts = torch.zeros(1, 4, len(canonical.unit_ids))
    counts[:, 1, ::3] = 1
    controls = torch.zeros(1, 4, 4)
    # Active zero cue is distinct source semantics, but common to both arms.
    controls[:, :, 2] = 1
    left, right = state(canonical, counts, controls), state(candidate, counts, controls)
    for name in ("mean", "variance", "ready_in_ticks"):
        assert torch.equal(getattr(left, name), getattr(right, name))
    for left_parameter, right_parameter in zip(canonical._field_parameters()[1:],
                                                candidate._field_parameters()[1:]):
        assert torch.equal(left_parameter, right_parameter)
    assert torch.equal(canonical.transition_mean(left.mean, controls[:, 0]),
                       candidate.transition_mean(right.mean, controls[:, 0]))
    assert torch.equal(canonical.log_hazard(left.mean), candidate.log_hazard(right.mean))
    future = controls[:, :3]
    first = list(canonical.iter_simulation(left, future, LatentRandomStreams.from_seeds(41, 42, 43), 2))
    second = list(candidate.iter_simulation(right, future, LatentRandomStreams.from_seeds(41, 42, 43), 2))
    for a, b in zip(first, second):
        for name in ("counts", "events", "latent", "ready_in_ticks", "initial_latent"):
            assert torch.equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("seed", [41, 42, 43])
def test_active_registered_distractor_can_change_control_drive(source_geometry, seed):
    factual = model(source_geometry, seed)
    candidate = transferred(factual, source_geometry["candidate"])
    zero = torch.zeros(1, len(factual.field_names), factual.latent_channels)
    inactive = torch.zeros(1, 4)
    assert torch.equal(factual.transition_mean(zero, inactive), candidate.transition_mean(zero, inactive))
    active = inactive.clone()
    active[:, 1] = .5
    active[:, 3] = 1
    assert not torch.equal(factual.transition_mean(zero, active), candidate.transition_mean(zero, active))
    # Autonomous offsets remain invariant even when controls will later differ.
    for a, b in zip(factual._field_parameters()[1:], candidate._field_parameters()[1:]):
        assert torch.equal(a, b)


def test_static_default_configuration_buffers_and_parameter_shapes_remain_exact(source_geometry):
    explicit = model(source_geometry, 41, mode="static")
    configuration = explicit.configuration()
    assert "control_geometry" not in configuration
    implicit = ControlledPopulationState(**configuration)
    assert implicit.configuration() == explicit.configuration()
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for name, value in explicit.state_dict().items():
        assert torch.equal(value, implicit.state_dict()[name])
    active = model(source_geometry, 41)
    assert {name: tuple(value.shape) for name, value in active.named_parameters()} == {
        name: tuple(value.shape) for name, value in explicit.named_parameters()}
    assert dict(active.named_buffers()).keys() == dict(explicit.named_buffers()).keys()


def test_mode_propagates_through_copy_and_shared_view_and_save_load(source_geometry, tmp_path):
    value = model(source_geometry, 42)
    copied = transferred(value, source_geometry["candidate"], shared=False)
    assert copied.control_geometry == "actuation_only"
    assert copied.shared_unit_readout.weight is not value.shared_unit_readout.weight
    configuration = value.configuration()
    assert configuration["control_geometry"] == "actuation_only"
    path = tmp_path / "actuation.pt"
    value.save(path)
    restored = ControlledPopulationState.load(path)
    assert restored.configuration() == configuration
    for key, original in value.state_dict().items():
        assert torch.equal(original, restored.state_dict()[key])
    static = model(source_geometry, 42, mode="static")
    old = tmp_path / "old.pt"
    static.save(old)
    payload = torch.load(old, weights_only=True)
    assert payload["schema"] == "controlled_population_state_v1"
    payload["configuration"]["control_geometry"] = "actuation_only"
    forged = tmp_path / "forged.pt"
    torch.save(payload, forged)
    with pytest.raises(ValueError, match="checkpoint schema"):
        ControlledPopulationState.load(forged)


@pytest.mark.parametrize("change", ["missing_target", "foreign_control", "fraction", "mask", "mode"])
def test_actuation_only_requires_explicit_source_contract(source_geometry, change):
    metadata = deepcopy(source_geometry["factual"])
    mode = "actuation_only"
    if change == "missing_target":
        metadata["unit_feature_schema"][5]["name"] = "foreign_target"
    elif change == "foreign_control":
        metadata["control_features"][1]["name"] = "foreign_current"
    elif change == "fraction":
        column = next(index for index, row in enumerate(metadata["field_feature_schema"])
                      if row["name"] == "distractor_target_fraction")
        metadata["field_features"][0][column] = .125
    elif change == "mask":
        column = next(index for index, row in enumerate(metadata["unit_feature_schema"])
                      if row["name"] == "cue_target")
        metadata["unit_features"][0][column] = .5
    else:
        mode = "unregistered"
    with pytest.raises(ValueError):
        model(source_geometry, 41, mode, metadata)


def test_generated_events_keep_native_source_support(source_geometry):
    value = model(source_geometry, 43)
    with torch.no_grad():
        value.shared_unit_readout.bias[-1] = 7
    counts = torch.zeros(1, 3, len(value.unit_ids))
    control = torch.zeros(1, 3, 4)
    boundary = state(value, counts, control)
    steps = list(value.iter_simulation(boundary, control, LatentRandomStreams.from_seeds(41, 42, 43), 2))
    events = torch.cat([step.events for step in steps])
    for draw in range(2):
        for unit in range(len(value.unit_ids)):
            ticks = events[(events[:, 0] == draw) & (events[:, 2] == unit), 3].sort().values
            assert torch.all(torch.diff(ticks) >= value.minimum_interval_ticks[unit])


@pytest.mark.parametrize("seed",[41,42,43])
def test_observed_active_control_geometry_enters_only_current_encoder_channels(source_geometry,seed):
    factual=model(source_geometry,seed)
    candidate=transferred(factual,source_geometry["candidate"])
    counts=torch.zeros(1,2,len(factual.unit_ids))
    inactive=torch.zeros(1,2,4)
    assert torch.equal(factual.field_features(counts,inactive),candidate.field_features(counts,inactive))
    active=inactive.clone();active[:,:,1]=.5;active[:,:,3]=1
    left=factual.field_features(counts,active);right=candidate.field_features(counts,active)
    assert not torch.equal(left,right)
    assert torch.equal(left[:,:,:,:3],right[:,:,:,:3])
    assert torch.equal(left[:,:,:,7:],right[:,:,:,7:])
