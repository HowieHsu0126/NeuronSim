"""Authorized reference-source routing and unchanged seven-check production gates."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from .benchmark_tasks import task_protocol, validate_protocol, pfc_working_memory_config
from .circuit_campaign import campaign_config, production_prerequisites
from . import benchmark_production as production
from .pfc_source_execution import _case_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCENARIO = "pfc_working_memory"
SOURCE_CHECKS = ("source_reproduction", "numerical_integrity", "driver_statistics",
                 "control_delivery", "full_unit_recording", "coupling", "scaling_fidelity")


def qualified_fixture():
    campaign = campaign_config(CONFIGS / "neural_population_dynamics.yaml")
    spec = campaign["scenarios"][SCENARIO]
    spec["parameter_domain"] = dict(frozen=True, source="disposable engineering fixture")
    spec["task_protocol"]["frozen"] = True
    spec["observation_map"]["frozen"] = True
    proof = dict(status="PASS", scenario_id=SCENARIO, coupled_core_neurons=100000,
                 source_revision="fixture", evidence=["disposable fixture"],
                 source_admission_route="independent_reference_source",
                 source_clock=dict(backend="numpy", dt_ms=0.0125),
                 source_definition="discrete_adapted_EPFL",
                 checks={name: True for name in SOURCE_CHECKS})
    return campaign, proof


def test_authorized_reference_retains_full_task_and_freezes_closed():
    campaign = campaign_config(CONFIGS / "neural_population_dynamics.yaml")
    spec = campaign["scenarios"][SCENARIO]
    policy = spec["source_admission"]
    assert policy["status"] == "AUTHORIZED"
    assert policy["coarse_engine_cross_clock_acceptance_required"] is True
    assert spec["task_protocol"] == task_protocol(SCENARIO, pfc_dt_ms=0.0125)
    assert spec["task_protocol"]["condition_duration_ticks"] == [400000] * 10
    assert spec["observation_map"]["temporal_resolution"]["candidate_dt_ms"] == 0.0125
    assert spec["parameter_domain"] is None
    assert set(production_prerequisites(campaign, SCENARIO, None)) >= {
        "parameter_domain", "task_protocol_not_frozen", "observation_map_not_frozen", "source_qualification"}


def test_reference_does_not_consume_coarse_acceptance_but_requires_all_source_checks():
    campaign, proof = qualified_fixture()
    proof["cross_clock_acceptance"] = dict(status="UNQUALIFIED")
    assert production_prerequisites(campaign, SCENARIO, proof) == []
    for name in SOURCE_CHECKS:
        changed = deepcopy(proof)
        changed["checks"][name] = False
        assert name in production_prerequisites(campaign, SCENARIO, changed)
    changed = deepcopy(proof)
    changed["status"] = "PASS_COMPLETE_NATIVE_CORRESPONDENCE_INTAKE"
    assert "source_qualification" in production_prerequisites(campaign, SCENARIO, changed)


@pytest.mark.parametrize("field,value,error", [
    ("source_admission_route", "", "pfc_source_admission_route"),
    ("source_clock", dict(backend="numpy", dt_ms=0.05), "pfc_source_clock_binding"),
    ("source_clock", dict(backend="cuda", dt_ms=0.0125), "pfc_reference_source_binding"),
    ("source_definition", "continuous_time_exact", "pfc_reference_source_binding"),
])
def test_reference_rejects_wrong_route_clock_backend_or_claim(field, value, error):
    campaign, proof = qualified_fixture()
    proof[field] = value
    assert error in production_prerequisites(campaign, SCENARIO, proof)


def test_coarse_engine_requires_complete_cross_clock_acceptance():
    campaign, proof = qualified_fixture()
    campaign["scenarios"][SCENARIO]["task_protocol"] = task_protocol(SCENARIO)
    campaign["scenarios"][SCENARIO]["task_protocol"]["frozen"] = True
    proof.update(source_admission_route="coarse_engine", source_clock=dict(backend="numpy", dt_ms=0.05))
    error = "pfc_coarse_engine_cross_clock_acceptance"
    assert error in production_prerequisites(campaign, SCENARIO, proof)
    proof["cross_clock_acceptance"] = dict(status="PASS", complete_frozen_domain=True,
        combined_error_budget_accepted=True, reference_cohort="numpy_dt_0p0125", evidence=["fixture"])
    assert production_prerequisites(campaign, SCENARIO, proof) == []
    for field in ("complete_frozen_domain", "combined_error_budget_accepted"):
        changed = deepcopy(proof)
        changed["cross_clock_acceptance"][field] = False
        assert error in production_prerequisites(campaign, SCENARIO, changed)


def test_reference_protocol_compiles_every_condition_with_real_source_entry():
    campaign = campaign_config(CONFIGS / "neural_population_dynamics.yaml")
    protocol = campaign["scenarios"][SCENARIO]["task_protocol"]
    base = yaml.safe_load((CONFIGS / "neural_population_pfc_100k_source_dt_0p0125ms.yaml").read_text())
    for index in range(10):
        identity = dict(condition_index=index, block_index=3, circuit_index=23, circuit_id="fixture",
                        episode_id=f"reference_condition_{index}", split="train",
                        initial_state_seed=210023003, background_seed=220023003)
        config = pfc_working_memory_config(base, protocol, identity)
        dims = _case_config(config, identity["episode_id"])[2]
        registered = production.pfc_source_dataset._benchmark_registration(config)
        assert registered[1] == {identity["episode_id"]: (
            identity["background_seed"], protocol["conditions"][index]["condition_id"])}
        assert registered[3]["duration_ticks"] == 400000
        assert dims["coupled_core_units"] == 100000 and dims["duration_ticks"] == 400000
        assert config["benchmark_episode_candidate"]["production_eligible"] is False
    invalid = deepcopy(protocol)
    invalid["condition_duration_ticks"][0] = 100000
    with pytest.raises(ValueError, match="PROTOCOL_DIFFERS"):
        validate_protocol(invalid, SCENARIO)


def test_production_entry_rejects_authorization_only_before_output(tmp_path):
    proof = tmp_path / "qualification.json"
    proof.write_text("{}")
    output = tmp_path / "production"
    with pytest.raises(ValueError, match="BENCHMARK_PRODUCTION_NOT_QUALIFIED"):
        production.main(["--config", str(CONFIGS / "neural_population_dynamics.yaml"),
            "--scenario", SCENARIO, "--qualification", str(proof),
            "--identity-table", str(tmp_path / "missing.csv"), "--circuit-index", "0",
            "prepare-pfc-episode", "--native-qualification", str(tmp_path / "missing-native"),
            "--backend", "numpy", "--block-index", "0", "--condition-index", "0", "--output", str(output)])
    assert not output.exists()
