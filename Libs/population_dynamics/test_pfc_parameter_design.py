"""Prospective operator identities and circuit binding for PFC qualification."""
import copy
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import yaml

from .benchmark_sampling import prepare_episode
from .benchmark_tasks import pfc_working_memory_config
from .circuit_campaign import campaign_config, episode_identities, production_prerequisites
from .pfc_parameter_design import describe_kernels, parameter_grid, prepare

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCENARIO = "pfc_working_memory"


def settings():
    return yaml.safe_load((CONFIGS / "neural_population_pfc_parameter_design.yaml").read_text())


def base():
    return yaml.safe_load((CONFIGS / "neural_population_pfc_100k_source.yaml").read_text())


def test_design_is_reproducible_complete_and_unsorted():
    design = settings()
    original = copy.deepcopy(design)
    first = parameter_grid(design, 100)
    assert first == parameter_grid(design, 100)
    assert len({tuple(row.values()) for row in first}) == 100
    assert design == original
    assert first != sorted(first, key=lambda row: tuple(row.values()))
    changed = dict(design, circuit_assignment_seed=design["circuit_assignment_seed"] + 1)
    second = parameter_grid(changed, 100)
    assert first != second
    assert {tuple(row.values()) for row in first} == {tuple(row.values()) for row in second}


@pytest.mark.parametrize("change,expected", [
    ({"frozen": True}, "CANDIDATE_REQUIRED"),
    ({"scope": "production"}, "CANDIDATE_REQUIRED"),
    ({"circuit_assignment_seed": True}, "ASSIGNMENT_SEED_INVALID"),
    ({"sigma_weight_profile": {"lower": 20., "upper": 20., "count": 10}}, "AXIS_INVALID"),
    ({"sigma_weight_profile": {"lower": float("nan"), "upper": 30., "count": 10}}, "AXIS_INVALID"),
])
def test_invalid_design_is_rejected(change, expected):
    with pytest.raises(ValueError, match=expected):
        parameter_grid(dict(settings(), **change), 100)


def test_design_rejects_wrong_count_and_effective_uniform_duplicates():
    with pytest.raises(ValueError, match="CIRCUIT_COUNT_MISMATCH"):
        parameter_grid(settings(), 99)
    with pytest.raises(ValueError, match="DUPLICATE_EFFECTIVE_KERNEL"):
        describe_kernels(base(), [
            dict(sigma_weight_profile=20., Jpos_excit2excit=1.),
            dict(sigma_weight_profile=30., Jpos_excit2excit=1.)])


def test_nondefault_operators_have_distinct_effective_weights():
    rows = describe_kernels(base(), [
        dict(sigma_weight_profile=20., Jpos_excit2excit=1.6),
        dict(sigma_weight_profile=20., Jpos_excit2excit=2.3),
        dict(sigma_weight_profile=30., Jpos_excit2excit=2.3)])
    for name in ("effective_kernel_sha256", "parameter_identity", "connectivity_identity"):
        assert len({row[name] for row in rows}) == 3
    assert all(row["kernel_units"] == 80000 and row["kernel_min"] >= 0. for row in rows)


def test_actual_preparation_reaches_episode_entry_and_keeps_production_closed(tmp_path):
    campaign_path = CONFIGS / "neural_population_dynamics.yaml"
    output = tmp_path / "circuits"
    report = prepare(campaign_path, CONFIGS / "neural_population_pfc_100k_source.yaml",
                     CONFIGS / "neural_population_pfc_parameter_design.yaml", output)
    assert report["candidate_circuits"] == report["distinct_effective_kernels"] == 100
    assert report["compiled_condition_configs"] == 1000
    assert report["complete_episodes"] == 0 and report["production_eligible"] is False
    campaign = campaign_config(campaign_path)
    assert "parameter_domain" in production_prerequisites(campaign, SCENARIO, None)
    identity = next(islice(episode_identities(campaign, SCENARIO), 71000, 71001))
    circuit = output / identity["circuit_id"] / "circuit.yaml"
    prepared = prepare_episode(campaign_path, circuit, SCENARIO, 71004, tmp_path / "episode")
    config = yaml.safe_load((tmp_path / "episode" / "episode.yaml").read_text())
    binding = config["benchmark_circuit_candidate"]
    assert binding["intended_split"] == "validation"
    assert binding["parameters"]["sigma_weight_profile"] == config["sigma_weight_profile"]
    assert config["duration_ms"] == 5000. and config["dt_ms"] == .0125
    assert prepared["condition_id"] == "cue_late_off_target"
    assert prepared["complete_episodes"] == 0 and prepared["production_eligible"] is False
    other = dict(identity, circuit_index=72, circuit_id="pfc_working_memory_circuit_072")
    with pytest.raises(ValueError, match="PARAMETER_CIRCUIT_MISMATCH"):
        pfc_working_memory_config(config, campaign["scenarios"][SCENARIO]["task_protocol"], other)
    changed_scale = dict(config, excitatory_neurons=40000, inhibitory_neurons=10000)
    with pytest.raises(ValueError, match="PARAMETER_CIRCUIT_PROVENANCE_MISMATCH"):
        pfc_working_memory_config(changed_scale, campaign["scenarios"][SCENARIO]["task_protocol"], identity)
    changed = copy.deepcopy(config)
    changed["Jpos_excit2excit"] += .001
    with pytest.raises(ValueError, match="PARAMETER_CIRCUIT_MISMATCH"):
        pfc_working_memory_config(changed, campaign["scenarios"][SCENARIO]["task_protocol"], identity)
    with pytest.raises(FileExistsError, match="OUTPUT_EXISTS"):
        prepare(campaign_path, CONFIGS / "neural_population_pfc_100k_source.yaml",
                CONFIGS / "neural_population_pfc_parameter_design.yaml", output)
