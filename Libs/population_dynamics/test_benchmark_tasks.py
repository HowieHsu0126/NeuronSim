"""Task denominators, nested conditions and native cue-assignment semantics."""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from .benchmark_tasks import (CA3_CONDITIONS, ca3_commands, ca3_cue_targets,
                              task_protocol, validate_protocol, v1_visual_config)
from .ca3_benchmark_cues import cue_program, runtime_program
from .ca3_coupled_extension import extend_source
from .ca3_reference import prepare_retrieval_program
from .circuit_campaign import campaign_config, production_prerequisites
from .test_ca3_coupled_extension import source_fixture
from .ca3_native_observation import Group, Interval
from .ca3_control_trace import BACKGROUND_SIGMA, PYRAMIDAL, check_currents
from .benchmark_sampling import resolve_episode, prepare_circuit
from .ca3_realization import SourceRealization, contract, from_record
from .ca3_benchmark import retrieval_realization
from .v1_circuit_build import load_config as load_v1_circuit_config
from .ca3_event_counters import compare_event_counters
from . import ca3_native_observation as observer
from .ca3_benchmark_readout import describe_events, compare_paired_currents, compare_native_prefix


def test_completion_readout_keeps_uncued_denominator_and_short_terminal_window():
    groups = (Group(0, 0, 1652, 0, 0, 0, 1652, 0, "CA3_Pyramidal"),)
    commands = dict(unit_ids=np.array([0, 550, 1100]), cue_id=np.array([1, 2, 3]))
    rows = describe_events(np.array([520, 521, 1000]), np.array([0, 1, 1101]), groups,
                           commands, "completion_10")
    target = next(row for row in rows if row["memory"] == 0
                  and row["population"] == "target_never_assigned_in_episode" and row["phase"] == "cue")
    assert target["units"] == 549 and target["spikes"] == 1
    post = next(row for row in rows if row["memory"] == 2
                and row["population"] == "target_never_assigned_in_episode" and row["phase"] == "post_cue")
    assert post["duration_ms"] == 1 and post["persistence_duration_limited"] is True
    extended = describe_events(np.array([1000, 1209]), np.array([1101, 1101]), groups,
                              commands, "completion_10", duration_ms=1210)
    post = next(row for row in extended if row["memory"] == 2
                and row["population"] == "target_never_assigned_in_episode" and row["phase"] == "post_cue")
    assert post["duration_ms"] == 210 and post["spikes"] == 2
    assert post["persistence_duration_limited"] is False
    full = dict(unit_ids=np.arange(1650), cue_id=np.repeat([1, 2, 3], 550))
    full_rows = describe_events(np.array([], dtype=np.int64), np.array([], dtype=np.int64), groups, full, "completion_full")
    assert all(row["units"] == 0 and row["mean_rate_hz"] is None and not row["applicable"]
               for row in full_rows if row["population"] == "target_never_assigned_in_episode")


def test_paired_currents_exclude_delivered_assignments_and_reject_background_changes(tmp_path):
    name = b"population"
    prefix = observer.HEADER.pack(observer.MAGIC, 1, 2)
    prefix += observer.GROUP.pack(0, 0, 2, 0, 0, 0, 2, 0, len(name)) + name
    def trace(path, cue=False, changed=False):
        with path.open("wb") as stream:
            stream.write(prefix)
            for tick in range(1001):
                values = [10000 if cue and tick == 520 else 1, 3 if changed and tick == 700 else 2]
                for kind in (b"R", b"A", b"C"):
                    stop = tick if kind == b"R" else tick + 1
                    stream.write(observer.RECORD.pack(kind, tick, stop, 0, 2))
                    stream.write(np.asarray([0, 0] if kind == b"C" else values,
                                             dtype="<i4" if kind == b"C" else "<f4").tobytes())
            stream.write(observer.RECORD.pack(b"E", 1001, 1001, -1, 0))
    reference, current = tmp_path / "reference", tmp_path / "current"
    empty = dict(unit_ids=np.array([], dtype=np.int64), time_ms=np.array([], dtype=np.int64))
    cue = dict(unit_ids=np.array([0]), time_ms=np.array([520]))
    trace(reference); trace(current, cue=True)
    result = compare_paired_currents(reference, current, empty, cue)
    assert result["current_values_compared"] == 2001
    trace(current, cue=True, changed=True)
    with pytest.raises(ValueError, match="PAIRED_BACKGROUND_DIFFERS:ms=700"):
        compare_paired_currents(reference, current, empty, cue)


def test_native_persistence_extension_checks_prefix_counts_currents_and_both_footers(tmp_path):
    name = b"population"
    header = observer.HEADER.pack(observer.MAGIC, 1, 2)
    header += observer.GROUP.pack(0, 0, 2, 0, 0, 0, 2, 0, len(name)) + name
    def trace(path, duration, *, count_error=False, current_error=False, footer=True):
        with path.open("wb") as stream:
            stream.write(header)
            for tick in range(duration):
                current = [1, 3 if current_error and tick == 1000 else 2]
                counts = [tick % 2, 1 if count_error and tick == 1000 else 0]
                for kind in (b"R", b"A", b"C"):
                    stream.write(observer.RECORD.pack(kind, tick, tick if kind == b"R" else tick + 1, 0, 2))
                    stream.write(np.asarray(counts if kind == b"C" else current,
                                            dtype="<i4" if kind == b"C" else "<f4").tobytes())
            if footer:
                stream.write(observer.RECORD.pack(b"E", duration, duration, -1, 0))
    original, extended = tmp_path / "old", tmp_path / "extended"
    trace(original, 1001); trace(extended, 1210)
    result = compare_native_prefix(original, extended)
    assert result["prefix_intervals_compared"] == 1001 and result["extended_intervals_verified"] == 1210
    assert result["unit_millisecond_counts_compared"] == result["current_values_compared"] == 2002
    for mutation in (dict(count_error=True), dict(current_error=True)):
        trace(extended, 1210, **mutation)
        with pytest.raises(ValueError, match="NATIVE_PREFIX_DIFFERS:ms=1000"):
            compare_native_prefix(original, extended)
    trace(extended, 1210, footer=False)
    with pytest.raises(ValueError, match="TRUNCATED"):
        compare_native_prefix(original, extended)
    trace(extended, 1210); trace(original, 1001, footer=False)
    with pytest.raises(ValueError, match="TRUNCATED"):
        compare_native_prefix(original, extended)


def test_counter_reconciliation_rejects_time_unit_swaps_with_equal_marginals(tmp_path):
    trace = tmp_path / "native.bin"
    name = b"population"
    data = observer.HEADER.pack(observer.MAGIC, 1, 2)
    data += observer.GROUP.pack(0, 0, 2, 0, 0, 0, 2, 0, len(name)) + name
    def record(kind, start, stop, values):
        dtype = "<i4" if kind == b"C" else "<f4"
        return observer.RECORD.pack(kind, start, stop, 0, len(values)) + np.asarray(values, dtype=dtype).tobytes()
    data += record(b"R", 0, 0, [0, 0])
    for tick, counts in enumerate(([2, 0], [0, 2])):
        data += record(b"A", tick, tick + 1, [0, 0])
        data += record(b"C", tick, tick + 1, counts)
    data += observer.RECORD.pack(b"E", 2, 2, -1, 0)
    trace.write_bytes(data)
    events = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int64)
    result = {}
    compare_event_counters(trace, events[::-1], 2, result, expected_stop_ms=2)
    assert result == dict(unit_millisecond_counts_compared=4, native_intervals_compared=2)
    swapped = events.copy(); swapped[:, 1] = 1 - swapped[:, 1]
    result = {}
    with pytest.raises(ValueError, match="PER_UNIT_MILLISECOND_COUNTER_MISMATCH"):
        compare_event_counters(trace, swapped, 2, result, expected_stop_ms=2)
    assert result["first_counter_mismatch"]["differing_units"] == 2
    trace.write_bytes(data[:-observer.RECORD.size])
    with pytest.raises(ValueError, match="TERMINAL|TRUNCATED"):
        compare_event_counters(trace, events, 2, {}, expected_stop_ms=2)


def test_candidate_circuit_inputs_reach_existing_loaders_without_production_promotion(tmp_path):
    configs = Path(__file__).resolve().parents[1] / "configs"
    campaign = configs / "neural_population_dynamics.yaml"
    for index, split in ((0, "train"), (70, "validation"), (80, "calibration"), (90, "test")):
        v1_output = tmp_path / ("v1_" + str(index))
        report = prepare_circuit(campaign, configs / "neural_population_v1_circuit_development.yaml",
                                 "v1_visual", index, v1_output)
        settings, _ = load_v1_circuit_config(v1_output / "circuit.yaml")
        assert settings["builder_seed"] == 100000000 + index
        assert settings["glif_assignment_seed"] == 101000000 + index
        assert report["intended_split"] == split and report["construction_scope"] == "development"
        assert report["complete_episodes"] == 0 and report["production_eligible"] is False
        ca3_output = tmp_path / ("ca3_" + str(index))
        report = prepare_circuit(campaign, campaign, "ca3_pattern_completion", index, ca3_output)
        realization = from_record(yaml.safe_load((ca3_output / "circuit.yaml").read_text())["ca3_source_realization"])
        assert realization.circuit_seed == realization.uniform_seed == 300000000 + index
        assert realization.engine_seed == 1 and realization.phase == "formation"
        assert realization.split == "development" and report["intended_split"] == split
    with pytest.raises(FileExistsError, match="PREPARATION_EXISTS"):
        prepare_circuit(campaign, campaign, "ca3_pattern_completion", 0, tmp_path / "ca3_0")
    with pytest.raises(ValueError, match="CIRCUIT_INDEX_INVALID"):
        prepare_circuit(campaign, campaign, "ca3_pattern_completion", 100, tmp_path / "invalid")


def test_retrieval_uses_actual_formation_identity_and_rejects_candidate_alias():
    identity = dict(circuit_id="ca3_pattern_completion_circuit_001", connectivity_seed=300000001,
                    block_id="ca3_pattern_completion_circuit_001_block_004",
                    background_seed=320001004, initial_state_seed=310001004)
    formed = SourceRealization(identity["circuit_id"], "formation", "formation", "development",
                               300000001, 300000001, 1)
    native = dict(source_realization=contract(formed))
    realized = retrieval_realization(native, identity)
    assert realized.circuit_seed == 300000001 and realized.circuit_id == identity["circuit_id"]
    assert realized.uniform_seed == 320001004 and realized.engine_seed == 310001004
    assert realized.phase == "retrieval" and realized.split == "development"
    assert retrieval_realization({}, identity) is None
    for changed in (dict(identity, circuit_id="another_circuit"),
                    dict(identity, connectivity_seed=300000002)):
        with pytest.raises(ValueError, match="OWN_FORMED_CIRCUIT"):
            retrieval_realization(native, changed)
    native["source_realization"]["source_topology_noise_coupling_preserved"] = False
    with pytest.raises(ValueError, match="OWN_FORMED_CIRCUIT"):
        retrieval_realization(native, identity)


def test_canonical_task_conditions_and_unfrozen_production_boundary():
    config = campaign_config(Path(__file__).resolve().parents[1] / "configs/neural_population_dynamics.yaml")
    for scenario in ("v1_visual", "ca3_pattern_completion"):
        validate_protocol(config["scenarios"][scenario]["task_protocol"], scenario)
        missing = production_prerequisites(config, scenario, None)
        assert "task_protocol_not_frozen" in missing
        assert "parameter_domain_not_frozen" in missing
        changed = copy.deepcopy(task_protocol(scenario))
        changed["condition_duration_ticks"][0] -= 1
        with pytest.raises(ValueError, match="PROTOCOL_DIFFERS"):
            validate_protocol(changed, scenario)


@pytest.mark.parametrize("seed", [1, 330000000, 330099099])
def test_cue_membership_counts_correct_error_targets_and_nesting(seed):
    previous = set()
    for name, correct, wrong in CA3_CONDITIONS:
        condition = dict(condition_id=name, true_cue_units=correct, false_cue_units=wrong)
        rows = ca3_cue_targets(condition, seed)
        for memory, row in enumerate(rows):
            targets = [target for target in row if target >= 0]
            assert len(targets) == len(set(targets)) == correct + wrong
            assert sum(memory * 550 <= target < (memory + 1) * 550 for target in targets) == correct
            assert sum(((memory + 1) % 3) * 550 <= target < (((memory + 1) % 3) + 1) * 550
                       for target in targets) == wrong
        if name.startswith("completion"):
            current = set(target for target in rows[0] if 0 <= target < 550)
            assert previous <= current
            previous = current
    assert ca3_cue_targets(dict(condition_id="completion_25", true_cue_units=138, false_cue_units=0), seed) != ca3_cue_targets(
        dict(condition_id="completion_25", true_cue_units=138, false_cue_units=0), seed + 1)


def test_ordered_commands_preserve_repeated_source_assignments():
    pattern = np.tile(np.arange(550, dtype=np.int32), 4)
    condition = dict(condition_id="completion_50", true_cue_units=275, false_cue_units=0)
    commands = ca3_commands(pattern, 4000, condition, 33)
    assert len(commands["unit_ids"]) == 3 * 4 * 275
    assert set(commands["time_ms"] // 10) <= {52, 53, 75, 76, 98, 99}
    assert np.all(commands["set_current"] == 10000)
    omitted = ca3_commands(pattern, 4000, dict(condition_id="cue_omission", true_cue_units=0, false_cue_units=0), 33)
    assert len(omitted["unit_ids"]) == 0
    bad = pattern.copy(); bad[0] = 550
    with pytest.raises(ValueError, match="PATTERN"):
        ca3_commands(bad, 4000, condition, 33)


@pytest.mark.parametrize("duration_ms", [1001, 1210])
def test_corrupted_and_omitted_controls_use_existing_full_current_checker(duration_ms):
    groups, begin = [], 0
    for index, name in enumerate(BACKGROUND_SIGMA):
        size = 1650 if name == PYRAMIDAL else 2
        groups.append(Group(index, begin, begin + size, 0, index, begin, begin + size, 0, name))
        begin += size
    pattern = np.tile(np.arange(550, dtype=np.int32), 4)
    for name, correct, wrong in (CA3_CONDITIONS[0], CA3_CONDITIONS[8]):
        commands = ca3_commands(pattern, 0, dict(condition_id=name, true_cue_units=correct,
                                               false_cue_units=wrong), 42)
        def intervals():
            for tick in range(duration_ms):
                current = np.ones(begin, dtype="<f4")
                current[commands["unit_ids"][commands["time_ms"] == tick]] = 10000
                yield Interval(tick, tick + 1, np.zeros(begin, dtype=np.int64), current.copy(), current)
        _, checks = check_currents(groups, intervals(), commands, expected_stop_ms=duration_ms)
        assert checks["ordered_assignments_verified"] == len(commands["unit_ids"])
        assert checks["actual_current_values_checked"] == duration_ms * begin


def test_sampling_seed_names_do_not_claim_random_microscopic_initial_states():
    config = campaign_config(Path(__file__).resolve().parents[1] / "configs/neural_population_dynamics.yaml")
    identity = dict(condition_index=1, block_index=5, connectivity_seed=300000001,
                    initial_state_seed=310001005, background_seed=320001005, stimulus_seed=330001005)
    resolved = resolve_episode(config, "ca3_pattern_completion", identity)
    assert resolved["initial_state"] == "source_reset_not_initial_state_seed_draw"
    assert resolved["retrieval_seeds"]["engines"] == identity["initial_state_seed"]
    assert resolved["production_eligible"] is False


def test_visual_conditions_resolve_noise_and_low_contrast_direction():
    base = dict(stimulus={}, spike_generation={}, background={})
    identity = dict(condition_index=9, block_index=99, stimulus_seed=130009099, background_seed=120009099)
    actual = v1_visual_config(base, task_protocol("v1_visual"), identity)
    assert actual["stimulus"]["direction_deg"] == 135
    assert actual["stimulus"]["contrast_percent"] == 20
    assert actual["spike_generation"] == dict(trials=1, starting_noise_seed=130009099)
    assert base == dict(stimulus={}, spike_generation={}, background={})


def test_native_cue_and_restore_overlay_retains_clock_and_random_draws(tmp_path):
    header, main, pattern = source_fixture("retrieval")
    header, main, _, _ = extend_source(header, main, pattern, multiplier=2, role="retrieval")
    cpp = tmp_path / "source.cpp"; config = tmp_path / "config.h"
    cpp.write_bytes(main.encode()); config.write_bytes(header.encode())
    prepare_retrieval_program(cpp, config, local_input=True, population_multiplier=2)
    raw = cpp.read_bytes()
    condition = dict(condition_id="corruption_50", true_cue_units=138, false_cue_units=137)
    effective, contract = cue_program(raw, condition, 33)
    assert contract["random_draws_changed"] is False
    assert effective.count(b"benchmark_target >= 0") == 2
    for token in (b"drand48()", b"runNetwork", b"setExternalCurrent", b"lognormal_distribution"):
        assert effective.count(token) == raw.count(token)
    runtime, metadata = runtime_program(raw)
    assert b"argc != 5" in runtime
    assert b"benchmark_controls >> benchmark_extra" in runtime
    assert metadata["integration"] == "RUNGE_KUTTA4_5"
    extended, metadata = runtime_program(raw, duration_ms=1210)
    assert metadata["duration_ms"] == 1210
    assert metadata["terminal_background_extension_ms"] == 209
    assert b"for (int i=0; i<735; i++)" in extended
    assert b"if (i >= 525 && i < 735)" in extended
    assert extended.replace(b"i<735", b"i<526").replace(b"i < 735", b"i < 526") == runtime
    protocol = task_protocol("ca3_pattern_completion", retrieval_duration_ms=1210)
    validate_protocol(protocol, "ca3_pattern_completion")
    assert protocol["condition_duration_ticks"] == [1210] * 10
    with pytest.raises(ValueError, match="DURATION_UNSUPPORTED"):
        runtime_program(raw, duration_ms=1209)
    assert config.read_bytes() == header.encode()
    with pytest.raises(ValueError, match="PINNED"):
        cue_program(raw + b"changed", condition, 33)
