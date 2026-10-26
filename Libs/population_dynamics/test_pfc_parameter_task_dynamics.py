"""Task outcomes retain source-time controls, paired signs and missing directions."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from . import pfc_parameter_task_dynamics as dynamics
from .benchmark_tasks import task_protocol
from .provenance import sha256_file
from .test_pfc_parameter_domain_quality import plan


def summary_report():
    return dict(schema="pfc_parameter_domain_summary_v1",
        status="COMPLETE3000_DESCRIPTIVE_PARAMETER_DOMAIN_COHORT",
        complete_development_episodes=3000, parameter_circuits=100,
        blocks=[6, 7, 8], conditions=list(dynamics.CONDITIONS), circuit_split_counts=dynamics.SPLITS,
        core_units=100000, dt_ms=.0125, duration_ticks=400000,
        observation_bin_ticks=80, observation_bin_ms=1.,
        source_qualification_granted=False, scientific_parameter_domain_qualified=False,
        scientific_claim_qualified=False, uncertainty_coverage_qualified=False, scaling_qualified=False,
        production_episodes=0,
        observation_map=dict(frozen=True, populations=["E", "I"],
            unit_order="all_E_then_all_I_ascending_source_indices",
            temporal_resolution=dict(scientific_bin_ticks=80),
            controls=dict(commands=dynamics.CONTROL_COMMANDS)),
        bindings=[dict(episode_id=r["identity"]["episode_id"]) for r in plan()["rows"]],
        outputs={f"circuit_{c:03d}_joint_activity.npz":dict(sha256="unused") for c in range(100)})


def joint_activity():
    result = {}
    cue = np.deg2rad(120.)
    for k in range(10):
        prefix = f"condition_{k:02d}"
        fields = np.zeros((3, 5000, 17), dtype=np.int64)
        fields[:, :, 0] = 100
        fields[:, :, 16] = 20
        result[prefix + "_population_field_counts1ms"] = fields
        result[prefix + "_vector_x1ms"] = np.full((3, 5000), (k + 1) * np.cos(cue))
        result[prefix + "_vector_y1ms"] = np.full((3, 5000), (k + 1) * np.sin(cue))
        control = [8000, 24080]  # Zero-amplitude cue retains its activation flag.
        if k % 5:
            control += [192000, 196000] if k % 5 == 4 else [32000, 36000]
        result[prefix + "_source_control_transition_ticks"] = np.array(control, dtype=np.int64)
    return result


def test_silent_cancelled_and_weak_directions_keep_distinct_meanings():
    fields = np.zeros((1, 5000, 17), dtype=np.int64)
    fields[:, :, 0] = 100
    x = np.ones((1, 5000)); y = np.zeros_like(x)
    fields[0, 0, 0] = 0; x[0, 0] = 0
    x[0, 1] = 0  # Non-silent population cancellation.
    x[0, 2] = 1e-8  # Numerically defined, with no scientific strength claim.
    value = dynamics.measured_trajectories(fields, x, y, 0.)
    assert np.isnan(value["resultant_length"][0, 0])
    assert value["resultant_length"][0, 1] == 0
    assert not value["numerical_direction_defined"][0, :2].any()
    assert value["numerical_direction_defined"][0, 2]
    assert value["resultant_length"][0, 2] == pytest.approx(1e-10)
    assert np.isnan(value["decoded_angle_deg"][0, :2]).all()
    assert value["target_directed_activity_Hz"][0, 0] == 0


def test_impossible_vector_and_partial_time_domain_reject():
    joint = joint_activity()
    fields = joint["condition_00_population_field_counts1ms"]
    x = np.full((3, 5000), 101.); y = np.zeros_like(x)
    with pytest.raises(ValueError, match="VECTOR_OUTSIDE_EVENT_SUPPORT"):
        dynamics.measured_trajectories(fields, x, y, 120.)
    with pytest.raises(ValueError, match="COMPLETE_FINITE_TRAJECTORIES"):
        dynamics.measured_trajectories(fields[:, :-1], x[:, :-1], y[:, :-1], 120.)


def test_negative_paired_effects_and_observed_recovery_origins():
    trajectories, angular, phases = dynamics.describe_circuit(
        86, joint_activity(), task_protocol("pfc_working_memory", pfc_dt_ms=.0125))
    assert trajectories["cue_minus_omission"].shape == (3, 5000, 4, 5)
    np.testing.assert_allclose(trajectories["cue_minus_omission"][:, :, 0, :], -5. / 80.)
    np.testing.assert_allclose(trajectories["distractor_difference_of_differences"], 0., atol=1e-15)
    assert len(angular) == 10 * 3 * 8 * 3 and len(phases) == 10 * 3 * 8
    assert all(row["split"] == "calibration" for row in phases)
    np.testing.assert_array_equal(trajectories["distractor_removal_ticks"], [36000, 36000, 36000, 196000])
    assert trajectories["time_since_distractor_removal_ms"][0, 450] == 0.
    assert not trajectories["post_distractor_removal"][3, 2449]
    assert trajectories["post_distractor_removal"][3, 2450]
    np.testing.assert_array_equal(trajectories["condition_00_source_control_transition_ticks"], [8000, 24080])


def test_angular_wrap_and_missing_endpoint_preserve_pair_denominator():
    fields = np.zeros((1, 5000, 17), dtype=np.int64); fields[:, :, 0] = 100
    angles = np.tile(np.array([179.] * 5 + [-179.] * 5), 500)[None, :]
    x = 50. * np.cos(np.deg2rad(angles)); y = 50. * np.sin(np.deg2rad(angles))
    value = dynamics.measured_trajectories(fields, x, y, 120.)
    rows = dynamics.angular_rows(fields, x, y, value, 0, [0, 400000])
    assert rows[0]["candidate_pairs"] == 4995
    assert rows[0]["mean_squared_displacement_deg2"] == pytest.approx(4.)
    x[0, 10] = 0.; y[0, 10] = 0.
    value = dynamics.measured_trajectories(fields, x, y, 120.)
    missing = dynamics.angular_rows(fields, x, y, value, 0, [0, 400000])[0]
    assert missing["candidate_pairs"] == rows[0]["candidate_pairs"]
    assert missing["undefined_pairs"] == 2
    assert missing["mean_squared_displacement_deg2"] is None


def test_incomplete_or_duplicate_cohort_rejects():
    report = summary_report(); dynamics.validate_summary(report)
    changed = copy.deepcopy(report); changed["bindings"].pop()
    with pytest.raises(ValueError, match="COMPLETE_EPISODE_IDENTITIES"):
        dynamics.validate_summary(changed)
    changed = copy.deepcopy(report); changed["bindings"][-1] = changed["bindings"][0]
    with pytest.raises(ValueError, match="COMPLETE_EPISODE_IDENTITIES"):
        dynamics.validate_summary(changed)
    changed = copy.deepcopy(report); changed["outputs"].pop("circuit_099_joint_activity.npz")
    with pytest.raises(ValueError, match="ALL100_JOINT_ARRAYS"):
        dynamics.validate_summary(changed)


def test_changed_summary_array_fails_before_loading(tmp_path):
    path = tmp_path / "circuit_000_joint_activity.npz"; path.write_text("changed")
    report = summary_report()
    with pytest.raises(ValueError, match="JOINT_ARRAY_CHANGED"):
        dynamics.load_circuit(tmp_path, report, 0)


def test_full_entry_retains_all_circuits_and_last_failure_withholds_output(tmp_path, monkeypatch):
    report = summary_report()
    preparation = plan(); preparation["task_protocol"] = task_protocol("pfc_working_memory", pfc_dt_ms=.0125)
    preparation["task_protocol"]["frozen"] = True
    prepared = tmp_path / "prepared"; prepared.mkdir()
    (prepared / "execution.json").write_text(json.dumps(preparation))
    report["preparation_sha256"] = sha256_file(prepared / "execution.json")
    source = tmp_path / "summary"; source.mkdir(); (source / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(dynamics, "cpu_execution_provenance", lambda: dict(
        backend="slurm", run_id="123456", trial_id=None, hostname="fixture-host",
        slurm_job_id="123456", slurm_node="cpu05", allocated_cpus=1, allocated_gpus=0))
    seen = []
    monkeypatch.setattr(dynamics, "load_circuit", lambda *args: {})
    def describe(circuit, joint, protocol):
        seen.append(circuit)
        return {"fixture": np.array([circuit])}, [{"circuit_index": circuit}], [{"circuit_index": circuit}]
    monkeypatch.setattr(dynamics, "describe_circuit", describe)
    result = dynamics.analyze(source, prepared, tmp_path / "complete")
    assert seen == list(range(100))
    assert result["complete_development_episodes"] == 3000 and result["source_simulations"] == 0
    assert result["production_episodes"] == 0 and not result["memory_success_qualified"]
    def fail_last(circuit, joint, protocol):
        if circuit == 99:
            raise ValueError("missing final circuit")
        return describe(circuit, joint, protocol)
    monkeypatch.setattr(dynamics, "describe_circuit", fail_last)
    with pytest.raises(ValueError, match="missing final circuit"):
        dynamics.analyze(source, prepared, tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()


def test_source_float_control_schedules_preserve_zero_amplitude_flags():
    protocol = task_protocol("pfc_working_memory", pfc_dt_ms=.0125)
    expected = ([8000, 24080], [8000, 24080, 32000, 36000],
        [8000, 24080, 32000, 36000], [8000, 24080, 32000, 36000],
        [8000, 24080, 192000, 196000], [8000, 24080],
        [8000, 24080, 32000, 36000], [8000, 24080, 32000, 36000],
        [8000, 24080, 32000, 36000], [8000, 24080, 192000, 196000])
    for condition, ticks in zip(protocol["conditions"], expected):
        np.testing.assert_array_equal(dynamics.expected_control_transition_ticks(protocol, condition), ticks)
    zero = dict(protocol["conditions"][6], distractor_strength_nA=0.)
    np.testing.assert_array_equal(dynamics.expected_control_transition_ticks(protocol, zero),
        [8000, 24080, 32000, 36000])
    empty = dict(protocol["conditions"][6], distractor_duration_ms=0.)
    np.testing.assert_array_equal(dynamics.expected_control_transition_ticks(protocol, empty), [8000, 24080])


@pytest.mark.parametrize("condition_index,wrong_distractor", [
    (1, [192000, 196000]), (2, [192000, 196000]),
    (3, [192000, 196000]), (4, [32000, 36000]),
])
def test_internally_paired_early_late_swaps_reject_before_outcomes(
        condition_index, wrong_distractor, monkeypatch):
    joint = joint_activity()
    joint[f"condition_{condition_index:02d}_source_control_transition_ticks"] = np.array(
        [8000, 24080] + wrong_distractor, dtype=np.int64)
    joint[f"condition_{condition_index+5:02d}_source_control_transition_ticks"] = np.array(
        [8000, 24080] + wrong_distractor, dtype=np.int64)
    def no_outcomes(*args):
        raise AssertionError("activity was processed before control admission")
    monkeypatch.setattr(dynamics, "measured_trajectories", no_outcomes)
    with pytest.raises(ValueError, match="CONDITION_CONTROL_SCHEDULE_MISMATCH"):
        dynamics.describe_circuit(0, joint, task_protocol("pfc_working_memory", pfc_dt_ms=.0125))


@pytest.mark.parametrize("wrong_cue", [[8000, 36000], [32000, 36000]])
def test_wrong_cue_using_valid_global_phase_edges_rejects(wrong_cue):
    joint = joint_activity()
    joint["condition_04_source_control_transition_ticks"] = np.array(
        wrong_cue + [192000, 196000], dtype=np.int64)
    with pytest.raises(ValueError, match="CONDITION_CONTROL_SCHEDULE_MISMATCH:cue_late_off_target"):
        dynamics.describe_circuit(0, joint, task_protocol("pfc_working_memory", pfc_dt_ms=.0125))


def test_bound_control_reports_preserve_identity_and_history_checks(tmp_path):
    protocol = task_protocol("pfc_working_memory", pfc_dt_ms=.0125)
    report = dict(bindings=[])
    for k, condition in enumerate(protocol["conditions"]):
        for block in dynamics.BLOCKS:
            episode = f"pfc_working_memory_circuit_000_block_{block:03d}_condition_{k:02d}"
            directory = tmp_path / episode / "observation1ms"; directory.mkdir(parents=True)
            path = directory / "report.json"
            payload = dict(identity=dict(episode_id=episode), condition=condition,
                observation_map=dict(controls=dict(commands=dynamics.CONTROL_COMMANDS)),
                source_control_transitions=[dict(boundary_aligned=True, transition_tick=int(tick))
                    for tick in dynamics.expected_control_transition_ticks(protocol, condition)])
            path.write_text(json.dumps(payload))
            report["bindings"].append(dict(episode_id=episode, path=str(directory.parent),
                observation_report_sha256=sha256_file(path)))
    dynamics.validate_control_transitions(dynamics.load_control_transitions(report, 0), protocol)
    last = report["bindings"][-1]; path = Path(last["path"]) / "observation1ms/report.json"
    changed = json.loads(path.read_text()); changed["source_control_transitions"][-1]["transition_tick"] = 36000
    path.write_text(json.dumps(changed)); last["observation_report_sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="CONTROL_SCHEDULE_DIFFERS_BETWEEN_HISTORIES"):
        dynamics.load_control_transitions(report, 0)


def test_omission_activation_boundaries_cannot_be_removed():
    joint = joint_activity()
    joint["condition_05_source_control_transition_ticks"] = np.array([], dtype=np.int64)
    with pytest.raises(ValueError, match="CONDITION_CONTROL_SCHEDULE_MISMATCH:omission_none"):
        dynamics.describe_circuit(0, joint, task_protocol("pfc_working_memory", pfc_dt_ms=.0125))


def test_actual_current_map_cannot_replace_command_feature_map():
    report = summary_report()
    report["observation_map"]["controls"]["commands"] = "per_neuron_applied_current_A"
    with pytest.raises(ValueError, match="FROZEN_OBSERVATION_MAP_REQUIRED"):
        dynamics.validate_summary(report)
