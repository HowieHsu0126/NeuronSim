"""PFC prospective controls, realized noise pairing and production boundaries."""
import copy
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import yaml

from .benchmark_sampling import prepare_episode, prepare_pfc_block, resolve_episode
from .benchmark_tasks import pfc_working_memory_config, task_protocol, validate_protocol
from .circuit_campaign import campaign_config, episode_identities, freeze_identities, production_prerequisites
from .pfc_deterministic_replay import _Arithmetic, _control, _held_current, simulate
from .pfc_native_drive import create_drive
from .pfc_source_execution import _case_config, _units, validate_result

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCENARIO = "pfc_working_memory"


def settings():
    campaign = campaign_config(CONFIGS / "neural_population_dynamics.yaml")
    campaign["scenarios"][SCENARIO]["task_protocol"] = task_protocol(SCENARIO)
    identities = list(islice(episode_identities(campaign, SCENARIO), 10))
    base = yaml.safe_load((CONFIGS / "neural_population_pfc_100k_source.yaml").read_text())
    return campaign, identities, base


def test_pfc_protocol_preserves_factorial_and_rejects_schedule_or_clock_edits():
    campaign, identities, _ = settings()
    protocol = campaign["scenarios"][SCENARIO]["task_protocol"]
    assert protocol == task_protocol(SCENARIO)
    assert len({condition["condition_id"] for condition in protocol["conditions"]}) == 10
    for index in range(5):
        cue, omission = (copy.deepcopy(protocol["conditions"][i]) for i in (index, index + 5))
        assert cue.pop("stimulus_strength_nA") == 0.5
        assert omission.pop("stimulus_strength_nA") == 0.0
        assert cue.pop("condition_id").removeprefix("cue_") == omission.pop("condition_id").removeprefix("omission_")
        assert cue == omission
    for name in ("dt_ms", "stimulus_operation_ms", "stimulus_start_ms"):
        invalid = copy.deepcopy(protocol); invalid[name] += 0.01
        with pytest.raises(ValueError, match="PROTOCOL_DIFFERS"):
            validate_protocol(invalid, SCENARIO)
    invalid = copy.deepcopy(protocol); invalid["conditions"][4]["distractor_start_ms"] = 2000.0
    with pytest.raises(ValueError, match="PROTOCOL_DIFFERS"):
        validate_protocol(invalid, SCENARIO)
    for identity in identities:
        resolved = resolve_episode(campaign, SCENARIO, identity)
        assert resolved["source_seeds"] == dict(initial_rng_seed=210000000, external_noise_seed=220000000)
        assert "formation_seed" not in resolved and "retrieval_seeds" not in resolved
        assert resolved["circuit_parameter_domain_bound"] is False
        assert resolved["production_eligible"] is False and resolved["complete_episodes"] == 0


def test_pfc_compiler_matches_delivered_current_at_every_native_operation():
    campaign, identities, base = settings()
    protocol = campaign["scenarios"][SCENARIO]["task_protocol"]
    original = copy.deepcopy(base)
    for identity in identities:
        config = pfc_working_memory_config(base, protocol, identity)
        case = config["cases"][0]
        _case_config(config, case["case_id"])
        n_e = 360
        current = np.full(n_e, 3e-9)
        program = _control(config, case, n_e, _Arithmetic("numpy"))
        cue_ids = np.arange(105, 136)
        center = int(config["distractor_center_deg"])
        distractor_ids = np.arange(center - 20, center + 21) % n_e
        absent = np.zeros(n_e)
        absent_program = copy.deepcopy(program)
        absent_program["distractor"]["end"] = absent_program["distractor"]["start"]
        for operation in range(5000):
            active = _held_current(current, program, operation * 0.001)
            cue_active = 100 <= operation < 301
            start = int(config["distractor_start_ms"])
            distractor_active = start <= operation < start + int(config["distractor_duration_ms"])
            expected = np.zeros(n_e)
            if cue_active: expected[cue_ids] = config["stimulus_strength_nA"] * 1e-9
            if distractor_active: expected[distractor_ids] = config["distractor_strength_nA"] * 1e-9
            assert active == (cue_active, distractor_active)
            np.testing.assert_array_equal(current, expected)
            if identity["condition_index"] in (0, 5):
                _held_current(absent, absent_program, operation * 0.001)
                np.testing.assert_array_equal(current, absent)
        assert config["benchmark_episode_candidate"]["execution_scope"] == "development"
    assert base == original


def test_pfc_compiler_keeps_zero_cue_persistence_and_distractor_overwrite():
    campaign, identities, base = settings()
    # Source semantics remain supported even though prospective conditions do not overlap.
    for identity in (identities[0], identities[5]):
        config = pfc_working_memory_config(base, task_protocol(SCENARIO), identity)
        config.update(stimulus_start_ms=0.0, stimulus_duration_ms=3.0,
                      distractor_start_ms=0.0, distractor_duration_ms=1.0,
                      distractor_center_deg=120.0)
        case = dict(config["cases"][0], distractor_strength_nA=0.25)
        program = _control(config, case, 360, _Arithmetic("numpy"))
        current = np.zeros(360)
        _held_current(current, program, 0.0)
        assert np.all(current[100:141] == 0.25e-9)
        _held_current(current, program, 0.001)
        assert np.all(current[105:136] == case["stimulus_strength_nA"] * 1e-9)
        assert np.all(current[np.r_[100:105, 136:141]] == 0.25e-9)
        _held_current(current, program, 0.003)
        assert not current.any()


def test_pfc_realized_initial_states_and_background_innovations_are_paired():
    campaign, identities, base = settings()
    protocol = campaign["scenarios"][SCENARIO]["task_protocol"]
    initial_reference, draws_reference, remaining_reference, rng_reference = None, None, None, None
    for identity in identities:
        config = pfc_working_memory_config(base, protocol, identity)
        config.update(excitatory_neurons=8, inhibitory_neurons=2)
        initial, drive = create_drive(config, config["cases"][0])
        literal_initial = np.random.RandomState(identity["initial_state_seed"])
        i = literal_initial.uniform(-60.0, -50.0, size=2)
        e = literal_initial.uniform(-60.0, -50.0, size=8)
        np.testing.assert_array_equal(initial, np.r_[e, i])
        literal_noise = np.random.RandomState(identity["background_seed"])
        draws = []
        for _ in range(37):
            expected_i = literal_noise.binomial(1000, 1.4 * 0.00005, size=2)
            expected_e = literal_noise.binomial(1000, 1.4 * 0.00005, size=8)
            block = drive.take(1)
            actual_i, actual_e = block["I"][0], block["E"][0]
            np.testing.assert_array_equal(actual_i, expected_i)
            np.testing.assert_array_equal(actual_e, expected_e)
            draws.append(np.r_[actual_i, actual_e])
        if initial_reference is None: initial_reference, draws_reference = initial, np.asarray(draws)
        np.testing.assert_array_equal(initial, initial_reference)
        np.testing.assert_array_equal(draws, draws_reference)
        remaining = drive.take(drive.remaining)
        if remaining_reference is None:
            remaining_reference, rng_reference = remaining, drive.get_state()
        for population in ("I", "E"):
            np.testing.assert_array_equal(remaining[population], remaining_reference[population])
        for actual, expected in zip(drive.get_state(), rng_reference):
            np.testing.assert_array_equal(actual, expected)
        assert drive.ticks == drive.total_ticks == 100000
    config = pfc_working_memory_config(base, protocol, identities[0])
    config.update(excitatory_neurons=8, inhibitory_neurons=2)
    config["cases"][0]["initial_rng_seed"] += 1
    changed_initial, drive = create_drive(config, config["cases"][0])
    assert not np.array_equal(changed_initial, initial_reference)
    block = drive.take(1)
    actual_i, actual_e = block["I"][0], block["E"][0]
    np.testing.assert_array_equal(np.r_[actual_i, actual_e], draws_reference[0])
    config["cases"][0]["external_noise_seed"] = -1
    with pytest.raises(ValueError, match="SEED_INVALID"):
        create_drive(config, config["cases"][0])


def test_pfc_late_condition_reaches_full_source_current_and_unit_consumer():
    campaign, identities, base = settings()
    config = pfc_working_memory_config(base, task_protocol(SCENARIO), identities[4])
    # Bounded numerical fixture; retains the full 5000ms schedule, not a scale claim.
    config.update(excitatory_neurons=8, inhibitory_neurons=2, weight_scaling_factor=256.0)
    case = config["cases"][0]
    _case_config(config, case["case_id"])
    initial, drive = create_drive(config, case)
    result = simulate(config, case, initial, drive=drive, observe_applied_current=True)
    checks = validate_result(config, result, drive)
    assert all(row["passed"] is True for row in checks.values()), checks
    observed = result["applied_current_observation"]["arrays"]
    np.testing.assert_array_equal(observed["segment_start_ticks"], [0, 2000, 6020, 48000, 49000])
    np.testing.assert_array_equal(observed["segment_stop_ticks"], [2000, 6020, 48000, 49000, 100000])
    expected = np.zeros((5, 8)); expected[1, 3] = 0.5e-9; expected[3, 2] = 0.5e-9
    np.testing.assert_array_equal(observed["current_A"], expected)
    units = _units(8, 2)
    np.testing.assert_array_equal(units["population_index"], [0]*8 + [1]*2)
    np.testing.assert_array_equal(units["excitatory_preferred_angle_deg"], np.arange(8) * 45)
    assert drive.ticks == drive.total_ticks == 100000


def test_pfc_episode_preparation_reuses_source_loader_and_rejects_production(tmp_path):
    campaign, identities, base = settings()
    report = prepare_episode(CONFIGS / "neural_population_dynamics.yaml",
                             CONFIGS / "neural_population_pfc_100k_source.yaml", SCENARIO, 90004, tmp_path / "episode")
    config = yaml.safe_load((tmp_path / "episode/episode.yaml").read_text())
    case, _, dimensions = _case_config(config, config["cases"][0]["case_id"])
    assert report["intended_split"] == "test" and report["execution_scope"] == "development"
    assert report["circuit_parameter_domain_bound"] is False
    assert report["production_eligible"] is False and dimensions["coupled_core_units"] == 100000
    assert case["external_noise_seed"] == 220090000 and case["initial_rng_seed"] == 210090000
    with pytest.raises(FileExistsError, match="PREPARATION_EXISTS"):
        prepare_episode(CONFIGS / "neural_population_dynamics.yaml", CONFIGS / "neural_population_pfc_100k_source.yaml", SCENARIO, 90004, tmp_path / "episode")
    missing = production_prerequisites(campaign, SCENARIO, None)
    assert set(["parameter_domain", "task_protocol_not_frozen", "observation_map_not_frozen", "source_qualification"]) <= set(missing)
    source = tmp_path / "unqualified.json"; source.write_text("{}")
    with pytest.raises(ValueError, match="NOT_QUALIFIED"):
        freeze_identities(campaign, SCENARIO, source, tmp_path / "production.csv")
    assert not (tmp_path / "production.csv").exists()


def test_pfc_compiler_preserves_source_kernel_and_rejects_nonphysical_tuple():
    from .pfc_kernel_parameters import circuit_provenance, source_kernel
    from .pfc_deterministic_replay import SOURCE_ENTRY_SHA256

    campaign, identities, base = settings()
    protocol = campaign["scenarios"][SCENARIO]["task_protocol"]
    original = pfc_working_memory_config(base, protocol, identities[0])
    changed_base = dict(base, sigma_weight_profile=30.0, Jpos_excit2excit=1.7)
    changed = pfc_working_memory_config(changed_base, protocol, identities[0])
    assert changed["sigma_weight_profile"] == 30.0
    assert changed["Jpos_excit2excit"] == 1.7
    assert changed["cases"] == original["cases"]
    assert not np.array_equal(source_kernel(128, changed), source_kernel(128, original))
    before = circuit_provenance(original, SOURCE_ENTRY_SHA256)
    after = circuit_provenance(changed, SOURCE_ENTRY_SHA256)
    assert all(before[key] != after[key] for key in before)
    assert changed["benchmark_episode_candidate"]["circuit_parameter_domain_bound"] is False
    assert changed["benchmark_episode_candidate"]["production_eligible"] is False
    with pytest.raises(ValueError, match="NONPHYSICAL_WEIGHTS"):
        pfc_working_memory_config(dict(base, Jpos_excit2excit=100.0), protocol, identities[0])


def test_uniform_kernel_parameter_labels_do_not_prove_circuit_diversity():
    from .pfc_kernel_parameters import source_kernel

    first = source_kernel(128, dict(sigma_weight_profile=20.0, Jpos_excit2excit=1.0))
    second = source_kernel(128, dict(sigma_weight_profile=30.0, Jpos_excit2excit=1.0))
    np.testing.assert_array_equal(first, np.ones(128))
    np.testing.assert_array_equal(first, second)


def test_pfc_block_preparation_keeps_full_factorial_and_one_noise_identity(tmp_path):
    campaign, _, base = settings()
    source = tmp_path / "source.yaml"
    base.update(sigma_weight_profile=30.0, Jpos_excit2excit=2.3)
    source.write_text(yaml.safe_dump(base))
    output = tmp_path / "block"
    report = prepare_pfc_block(CONFIGS / "neural_population_dynamics.yaml", source, 23, 7, output)
    assert report["actual_execution_scope"] == "development" and report["intended_split"] == "train"
    assert report["production_eligible"] is False and report["complete_episodes"] == 0
    assert report["coupled_core_units"] == 100000 and report["duration_ticks"] == 400000
    assert len(report["conditions"]) == 10
    seeds, ids = set(), set()
    for row in report["conditions"]:
        config = yaml.safe_load((output / row["config_path"]).read_text())
        case, registered, _ = _case_config(config, row["identity"]["episode_id"])
        assert registered == [row["identity"]["episode_id"]]
        assert row["identity"]["circuit_index"] == 23 and row["identity"]["block_index"] == 7
        assert config["benchmark_episode_candidate"]["identity"] == row["identity"]
        assert config["sigma_weight_profile"] == 30.0 and config["Jpos_excit2excit"] == 2.3
        for key, value in row["condition"].items():
            if key != "condition_id": assert config[key] == value
        seeds.add((case["initial_rng_seed"], case["external_noise_seed"]))
        ids.add(row["condition"]["condition_id"])
    assert len(ids) == 10 and len(seeds) == 1
    with pytest.raises(FileExistsError, match="BLOCK_PREPARATION_EXISTS"):
        prepare_pfc_block(CONFIGS / "neural_population_dynamics.yaml", source, 23, 7, output)


@pytest.mark.parametrize("circuit,block", [(True, 0), (0, -1), (100, 0), (0, 100)])
def test_pfc_block_preparation_rejects_wrong_sampling_identity(tmp_path, circuit, block):
    with pytest.raises(ValueError, match="PFC_CIRCUIT_OR_BLOCK_INDEX_INVALID"):
        prepare_pfc_block(CONFIGS / "neural_population_dynamics.yaml",
                          CONFIGS / "neural_population_pfc_100k_source.yaml", circuit, block, tmp_path / "block")
    assert not (tmp_path / "block").exists()


def test_pfc_block_preparation_rejects_small_network_before_writing(tmp_path):
    _, _, base = settings()
    base.update(excitatory_neurons=16, inhibitory_neurons=4, weight_scaling_factor=128.0)
    source = tmp_path / "source.yaml"; source.write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError, match="FULL_COUPLED_SOURCE_REQUIRED"):
        prepare_pfc_block(CONFIGS / "neural_population_dynamics.yaml", source, 0, 0, tmp_path / "block")
    assert not (tmp_path / "block").exists()
