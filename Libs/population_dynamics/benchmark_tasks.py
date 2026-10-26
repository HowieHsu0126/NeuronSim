"""Executable V1, PFC and CA3 task conditions and circuit/block sampling semantics.

Condition choices are prospective. Numerical qualification remains separate from
the task definition. The original simulator archives retain their identities.
"""
from __future__ import annotations

import copy
import random
import hashlib

TASK_SCHEMA = "neural_circuit_benchmark_tasks_v1"
PFC_DISTRACTORS = (
    ("none", 90.0, 0.0, 400.0, 0.0),
    ("aligned", 120.0, 0.5, 400.0, 50.0),
    ("off_target", 90.0, 0.5, 400.0, 50.0),
    ("antipodal", 300.0, 0.5, 400.0, 50.0),
    ("late_off_target", 90.0, 0.5, 2400.0, 50.0),
)
CA3_CONDITIONS = (
    ("cue_omission", 0, 0),
    ("completion_10", 55, 0),
    ("completion_25", 138, 0),
    ("completion_50", 275, 0),
    ("completion_75", 413, 0),
    ("completion_full", 550, 0),
    ("corruption_10", 247, 28),
    ("corruption_25", 206, 69),
    ("corruption_50", 138, 137),
    ("wrong_memory", 0, 275),
)


def task_protocol(scenario: str, *, retrieval_duration_ms: int = 1001,
                  pfc_dt_ms: float = 0.05) -> dict:
    if type(retrieval_duration_ms) is not int or retrieval_duration_ms not in (1001, 1210):
        raise ValueError("CA3_BENCHMARK_RETRIEVAL_DURATION_UNSUPPORTED")
    if scenario == "v1_visual":
        conditions = [dict(condition_id=f"direction_{direction:03d}",
                           direction_deg=direction, contrast_percent=80)
                      for direction in range(0, 360, 45)]
        conditions += [dict(condition_id="gray", direction_deg=0, contrast_percent=0),
                       dict(condition_id="low_contrast", contrast_percent=20,
                            direction_rule="45 * (block_index % 8)")]
        return dict(schema=TASK_SCHEMA, frozen=False, dt_ms=0.25,
                    condition_duration_ticks=[12000] * 10,
                    gray_screen_s=0.5, duration_s=3.0,
                    spatial_frequency_cycles_per_degree=0.04, temporal_frequency_hz=2.0,
                    conditions=conditions,
                    initial_state_policy="source_reset_then_recorded_full_gray_warmup",
                    noise_pairing="shared_starting_seeds_not_per_unit_event_pairing",
                    inference_unit="circuit_with_nested_noise_blocks",
                    outcomes=["all_unit_spike_counts", "population_time_courses",
                              "direction_selectivity", "contrast_response"],
                    quality_selection="numerical_validity_only_keep_silent_and_weak_responses")
    if scenario == "pfc_working_memory":
        if type(pfc_dt_ms) not in (int, float) or pfc_dt_ms not in (0.05, 0.025, 0.0125):
            raise ValueError("PFC_BENCHMARK_CLOCK_UNSUPPORTED")
        conditions = [dict(condition_id=cue + "_" + name,
                           stimulus_strength_nA=amplitude,
                           distractor_center_deg=center, distractor_width_deg=40.0,
                           distractor_strength_nA=strength, distractor_start_ms=start,
                           distractor_duration_ms=duration)
                      for cue, amplitude in (("cue", 0.5), ("omission", 0.0))
                      for name, center, strength, start, duration in PFC_DISTRACTORS]
        return dict(schema=TASK_SCHEMA, frozen=False, dt_ms=pfc_dt_ms,
                    condition_duration_ticks=[int(5000.0 / pfc_dt_ms)] * 10, duration_ms=5000.0,
                    stimulus_operation_ms=1.0,
                    schedule_arithmetic="source_float_seconds_start_plus_duration",
                    cue_active_operation_indices=[100, 301],
                    stimulus_center_deg=120.0,
                    stimulus_width_deg=30.0, stimulus_start_ms=100.0, stimulus_duration_ms=200.0,
                    conditions=conditions,
                    initial_state_policy="independent_source_uniform_I_then_E_minus60_to_minus50_mV",
                    noise_pairing="same_realized_native_background_draws_across_conditions",
                    random_stream_policy="independent_initial_external",
                    control_semantics="cue_assignment_or_full_reset_then_distractor_overwrite",
                    cue_omission_policy="zero_amplitude_with_same_active_cue_schedule",
                    no_distractor_policy="inactive_zero_duration_distractor",
                    inference_unit="qualified_parameter_circuit_with_nested_initial_noise_blocks",
                    outcomes=["all_unit_spike_counts", "population_time_courses",
                              "target_directed_activity", "angular_displacement", "recovery_trajectories"],
                    paired_contrasts="cue_minus_omission_by_distractor_then_minus_no_distractor_cue_effect",
                    quality_selection="numerical_validity_only_keep_silent_and_failed_memory",
                    qualification_status="prospective_conditions_not_scientifically_qualified")
    if scenario == "ca3_pattern_completion":
        return dict(schema=TASK_SCHEMA, frozen=False, dt_ms=1.0,
                    condition_duration_ticks=[retrieval_duration_ms] * 10, formation_duration_ms=70000,
                    retrieval_duration_ms=retrieval_duration_ms, cue_starts_ms=[520, 750, 980],
                    cue_duration_ms=20, assembly_size=550, cue_current_pA=10000,
                    conditions=[dict(condition_id=name, true_cue_units=correct,
                                     false_cue_units=wrong) for name, correct, wrong in CA3_CONDITIONS],
                    false_cue_policy="next_learned_assembly_cyclic",
                    membership_policy="seeded_permutation_nested_within_block",
                    initial_state_policy="fresh_source_reset_then_520ms_background",
                    noise_pairing="same_realized_background_draws_across_conditions",
                    inference_unit="formed_circuit_with_nested_retrieval_noise_blocks",
                    outcomes=["never_cued_target_activity", "wrong_assembly_activity",
                              "all_unit_spike_counts", "post_cue_persistence"],
                    quality_selection="numerical_validity_only_keep_failed_completion",
                    memory_trials_per_episode=3,
                    memory_trials_are_independent_episodes=False)
    raise ValueError("V1_PFC_OR_CA3_BENCHMARK_SCENARIO_REQUIRED")


def validate_protocol(protocol: dict, scenario: str) -> None:
    if scenario == "pfc_working_memory" and (type(protocol.get("dt_ms")) not in (int, float)
            or protocol["dt_ms"] not in (0.05, 0.025, 0.0125)):
        raise ValueError("BENCHMARK_TASK_PROTOCOL_DIFFERS:" + scenario)
    expected = task_protocol(scenario, retrieval_duration_ms=(
        protocol.get("retrieval_duration_ms", 1001) if scenario == "ca3_pattern_completion" else 1001),
        pfc_dt_ms=protocol.get("dt_ms", 0.05) if scenario == "pfc_working_memory" else 0.05)
    actual = copy.deepcopy(protocol)
    frozen = actual.pop("frozen", None)
    expected.pop("frozen")
    if type(frozen) is not bool or actual != expected:
        raise ValueError("BENCHMARK_TASK_PROTOCOL_DIFFERS:" + scenario)


def condition_for_episode(protocol: dict, scenario: str, identity: dict) -> dict:
    validate_protocol(protocol, scenario)
    index, block = identity["condition_index"], identity["block_index"]
    if type(index) is not int or not 0 <= index < 10 or type(block) is not int or not 0 <= block < 100:
        raise ValueError("BENCHMARK_CONDITION_OR_BLOCK_OUT_OF_RANGE")
    condition = copy.deepcopy(protocol["conditions"][index])
    if scenario == "v1_visual" and condition["condition_id"] == "low_contrast":
        condition["direction_deg"] = 45 * (block % 8)
    return condition


def v1_visual_config(base: dict, protocol: dict, identity: dict) -> dict:
    condition = condition_for_episode(protocol, "v1_visual", identity)
    config = copy.deepcopy(base)
    config["stimulus"].update(
        duration_s=protocol["duration_s"], gray_screen_s=protocol["gray_screen_s"],
        direction_deg=condition["direction_deg"], contrast_percent=condition["contrast_percent"],
        spatial_frequency_cycles_per_degree=protocol["spatial_frequency_cycles_per_degree"],
        temporal_frequency_hz=protocol["temporal_frequency_hz"])
    config["spike_generation"].update(trials=1, starting_noise_seed=identity["stimulus_seed"])
    config["background"]["starting_noise_seed"] = identity["background_seed"]
    return config



def pfc_working_memory_config(base: dict, protocol: dict, identity: dict) -> dict:
    """Bind an episode's controls and two RNG streams to the existing source runner.

    Source kernel parameters remain explicit. A prospective circuit identity
    cannot qualify the parameter domain or release production.
    """
    condition = condition_for_episode(protocol, "pfc_working_memory", identity)
    if base.get("scope") != "source_reproduction":
        raise ValueError("PFC_BENCHMARK_SOURCE_BASE_REQUIRED")
    from .pfc_kernel_parameters import circuit_provenance, resolve_kernel_parameters
    from .pfc_native_drive import SOURCE_ENTRY_SHA256

    config = copy.deepcopy(base)
    resolve_kernel_parameters(config)
    candidate = config.get("benchmark_circuit_candidate")
    production = isinstance(candidate, dict) and candidate.get("schema") == "neural_benchmark_production_circuit_v1"
    if candidate is not None:
        expected_status = ("PREPARED_QUALIFIED_PRODUCTION_CIRCUIT" if production
                           else "PREPARED_PFC_PARAMETER_CIRCUIT_CANDIDATE")
        if (not isinstance(candidate, dict) or candidate.get("status") != expected_status
                or candidate.get("scenario_id") != "pfc_working_memory"
                or any(candidate.get(key) != identity[key] for key in ("circuit_id", "circuit_index"))
                or candidate.get("intended_split") != identity["split"]
                or candidate.get("parameters") != resolve_kernel_parameters(config)):
            raise ValueError("PFC_BENCHMARK_PARAMETER_CIRCUIT_MISMATCH")
        provenance = circuit_provenance(config, SOURCE_ENTRY_SHA256)
        if any(candidate.get(key) != value for key, value in provenance.items()):
            raise ValueError("PFC_BENCHMARK_PARAMETER_CIRCUIT_PROVENANCE_MISMATCH")
        if production:
            from .pfc_kernel_parameters import source_kernel

            if (candidate.get("construction_scope") != "production"
                    or candidate.get("production_eligible") is not True or protocol.get("frozen") is not True
                    or candidate.get("effective_kernel_sha256") != hashlib.sha256(
                        source_kernel(config["excitatory_neurons"], config).tobytes()).hexdigest()):
                raise ValueError("PFC_BENCHMARK_PRODUCTION_RECIPE_MISMATCH")
            config["benchmark_task_protocol"] = copy.deepcopy(protocol)
    for name in ("source_rng_seed", "initial_rng_seed", "external_noise_seed"):
        config.pop(name, None)
    config.update(dt_ms=protocol["dt_ms"], duration_ms=protocol["duration_ms"],
                  random_stream_policy=protocol["random_stream_policy"])
    for name in ("stimulus_center_deg", "stimulus_width_deg", "stimulus_start_ms", "stimulus_duration_ms"):
        config[name] = protocol[name]
    config.update({name: value for name, value in condition.items() if name != "condition_id"})
    seeds = dict(initial_rng_seed=identity["initial_state_seed"], external_noise_seed=identity["background_seed"])
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds.values()):
        raise ValueError("PFC_BENCHMARK_SEED_INVALID")
    config.update(seeds)
    config["cases"] = [dict(case_id=identity["episode_id"], **seeds,
                            random_stream_policy=protocol["random_stream_policy"],
                            stimulus_strength_nA=condition["stimulus_strength_nA"],
                            distractor_strength_nA=condition["distractor_strength_nA"])]
    config["benchmark_episode_candidate"] = dict(
        identity=copy.deepcopy(identity), condition_id=condition["condition_id"],
        execution_scope="production" if production else "development", intended_split=identity["split"],
        circuit_parameter_domain_bound=production, connectivity_seed_used=False, stimulus_seed_used=False,
        production_eligible=production, complete_episodes=0)
    return config

def ca3_cue_targets(condition: dict, cue_seed: int) -> tuple[tuple[int, ...], ...]:
    """Map the source pattern's 550 local slots to three circuit-local targets.

    A -1 entry retains background. Error slots target the next memory. The
    permutation is identical across all conditions in one block. No simulator
    random draws are consumed or skipped by this mapping.
    """
    valid = [dict(condition_id=name, true_cue_units=correct, false_cue_units=wrong)
             for name, correct, wrong in CA3_CONDITIONS]
    if condition not in valid or type(cue_seed) is not int or not 1 <= cue_seed <= 2147483646:
        raise ValueError("CA3_BENCHMARK_CUE_CONDITION_OR_SEED_INVALID")
    rng = random.Random(cue_seed)
    permutation = list(range(550))
    rng.shuffle(permutation)
    ranks = {slot: rank for rank, slot in enumerate(permutation)}
    correct, wrong = condition["true_cue_units"], condition["false_cue_units"]
    return tuple(tuple(memory * 550 + slot if ranks[slot] < correct else
                       ((memory + 1) % 3) * 550 + slot if ranks[slot] < correct + wrong else -1
                       for slot in range(550)) for memory in range(3))


def ca3_commands(pattern, pyramidal_start: int, condition: dict, cue_seed: int) -> dict:
    """Reconstruct ordered delivered cue assignments from the source pattern."""
    import numpy as np

    pattern = np.asarray(pattern)
    if (pattern.shape != (2200,) or pattern.dtype.kind not in "iu"
            or np.any(pattern >= 550) or np.any(pattern < 0)
            or type(pyramidal_start) is not int or pyramidal_start < 0):
        raise ValueError("CA3_BENCHMARK_SOURCE_PATTERN_OR_GROUP_INVALID")
    targets = ca3_cue_targets(condition, cue_seed)
    times, ids, cues = [], [], []
    for memory, start in enumerate((520, 750, 980)):
        for phase in range(20):
            for slot in pattern[phase * 110:(phase + 1) * 110]:
                target = targets[memory][int(slot)]
                if target >= 0:
                    times.append(start + phase)
                    ids.append(pyramidal_start + target)
                    cues.append(memory + 1)
    return dict(time_ms=np.asarray(times, dtype=np.int32), unit_ids=np.asarray(ids, dtype=np.int64),
                cue_id=np.asarray(cues, dtype=np.int8),
                set_current=np.full(len(ids), 10000.0, dtype=np.float32))
