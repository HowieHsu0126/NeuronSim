"""CA3 control contrasts, immutable identity and real-format event checks."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from .benchmark_tasks import CA3_CONDITIONS, ca3_commands
from .ca3_benchmark_readout import bind_episode, condition_contrasts, compare_paired_currents, describe_archived_activity
from . import ca3_native_observation as observer
from .circuit_data import CircuitArchiveWriter, CircuitEpisode, CircuitUnits, SCHEMA
from .provenance import sha256_file


def write_header(path, groups):
    with path.open("wb") as stream:
        stream.write(observer.HEADER.pack(observer.MAGIC, len(groups), sum(group.units for group in groups)))
        for group in groups:
            name = group.name.encode()
            stream.write(observer.GROUP.pack(group.group_id, group.start, group.stop, group.net_id,
                group.local_group, group.local_start, group.local_stop, group.gpu, len(name)) + name)


def fixture(root, *, condition_index=1):
    root.mkdir()
    condition = dict(zip(("condition_id", "true_cue_units", "false_cue_units"), CA3_CONDITIONS[condition_index]))
    groups = (observer.Group(0, 0, 168106, 0, 0, 0, 168106, 0, "CA3_Pyramidal"),)
    pattern = np.arange(2200, dtype=np.int32) % 550
    (root / "native_run").mkdir()
    (root / "native_run/trainPat.csv").write_text("\n".join(map(str, pattern)) + "\n")
    formed = dict(normalized_input_sha256="fixture-formed-input", pattern_sha256=sha256_file(root / "native_run/trainPat.csv"),
                  populations={"CA3_Pyramidal": 168106})
    seeds = dict(circuit_seed=10, uniform_seed=11, engine_seed=12, cue_seed=13)
    commands = ca3_commands(pattern, 0, condition, seeds["cue_seed"])
    np.savez_compressed(root / "ordered_cue_commands.npz", **commands)
    targets = np.unique(commands["unit_ids"])
    controls = np.zeros((1001, len(targets) + len(groups) + 2), dtype=np.float32)
    if len(targets):
        controls[commands["time_ms"], np.searchsorted(targets, commands["unit_ids"])] = 1
    provenance = dict(connectivity_identity=formed["normalized_input_sha256"], parameter_identity="fixture-parameters")
    circuit = CircuitUnits("fixture-ca3", np.arange(168106), np.full(168106, "CA3_Pyramidal"),
        np.zeros(168106, dtype=np.int64), ("CA3_Pyramidal",), "source_cell_types_no_anatomical_geometry",
        "fixture-source", circuit_provenance=provenance)
    episode = CircuitEpisode("fixture-episode", circuit.circuit_id, "fixture-block", "development", 1001, 1.,
        np.array([520, 521, 1000]), np.array([0, 1, 1101]), np.arange(1001), controls,
        tuple(dict(name="feature_" + str(index), unit="dimensionless") for index in range(controls.shape[1])),
        seeds=seeds, source_record=dict(source_revision=circuit.source_revision, circuit_provenance=provenance,
            complete_unit_recording=True, simulation_completed=True, recorded_until_tick=1001,
            expected_duration_ticks=1001, core_unit_count=168106, condition=condition, formation_binding=formed))
    with CircuitArchiveWriter(root / "dataset", dict(schema=SCHEMA, scope="source_reproduction",
        scenario_id="ca3_pattern_completion", source_revision=circuit.source_revision,
        sample_unit="complete_mechanistic_episode")) as writer:
        writer.register_circuit(circuit); writer.add(episode)
    report = dict(status="COMPLETE_CA3_BENCHMARK_DEVELOPMENT_EPISODE", actual_split="development",
        complete_episodes=1, units=168106, events=3, duration_ms=1001, condition=condition,
        candidate_identity=dict(background_seed=11, initial_state_seed=12, stimulus_seed=13), formed_circuit=formed,
        unit_millisecond_counts_compared=168106 * 1001, native_terminal_footer_verified=True,
        all_unit_event_counter_equality=True)
    (root / "report.json").write_text(json.dumps(report))
    write_header(root / "native_observation.bin", groups)
    return episode, report, commands


def test_binding_uses_actual_circuit_block_and_ordered_cues(tmp_path):
    episode, _, commands = fixture(tmp_path / "run")
    data = bind_episode(tmp_path / "run")
    assert data["context"]["circuit_id"] == "fixture-ca3"
    assert data["context"]["block_id"] == "fixture-block"
    np.testing.assert_array_equal(data["episode"].spike_ticks, episode.spike_ticks)
    np.testing.assert_array_equal(data["commands"]["unit_ids"], commands["unit_ids"])


def test_activity_retains_silent_units_and_distinct_field_population_maps(tmp_path):
    fixture(tmp_path / "run")
    data = bind_episode(tmp_path / "run")
    # Field membership partitions a source population; it does not replace that population.
    field_index = np.zeros(168106, dtype=np.int64)
    field_index[1100:1650] = 1
    data["circuit"] = replace(data["circuit"], field_names=("nonmemory", "learned_memory_2"), field_index=field_index)
    counts = describe_archived_activity(data)
    assert counts["unit_counts"].shape == (168106,)
    assert np.count_nonzero(counts["unit_counts"] == 0) == 168103
    assert counts["field_counts"][1000, 1] == 1
    assert counts["population_counts"][1000, 0] == 1
    assert counts["field_counts"].shape == (1001, 2)
    assert int(counts["unit_counts"].sum()) == int(counts["field_counts"].sum()) == 3


@pytest.mark.parametrize("change,expected", [("commands", "ORDERED_COMMANDS_DIFFER"),
    ("condition", "ARCHIVE_IDENTITY_DIFFERS"), ("seeds", "ARCHIVE_IDENTITY_DIFFERS"),
    ("duration", "ARCHIVE_IDENTITY_DIFFERS"), ("pattern", "PATTERN_CHANGED"), ("archive", "ARCHIVE_CONTENT_CHANGED")])
def test_binding_rejects_changed_source_identity_and_controls(tmp_path, change, expected):
    root = tmp_path / "run"
    _, report, commands = fixture(root)
    if change == "commands":
        commands["unit_ids"] = commands["unit_ids"].copy(); commands["unit_ids"][0] += 1
        np.savez_compressed(root / "ordered_cue_commands.npz", **commands)
    elif change == "condition":
        report["condition"] = dict(condition_id="cue_omission", true_cue_units=0, false_cue_units=0)
    elif change == "seeds":
        report["candidate_identity"]["background_seed"] += 1
    elif change == "duration":
        report["duration_ms"] = 1210; report["unit_millisecond_counts_compared"] = 168106 * 1210
    elif change == "pattern":
        (root / "native_run/trainPat.csv").write_text("0\n")
    else:
        with (root / "dataset/episodes_000000.h5").open("ab") as stream: stream.write(b"changed")
    (root / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match=expected): bind_episode(root)


def test_contrasts_compare_same_candidate_units_and_preserve_negative_effects(tmp_path):
    base, _, _ = fixture(tmp_path / "run", condition_index=7)
    groups = (observer.Group(0, 0, 168106, 0, 0, 0, 168106, 0, "CA3_Pyramidal"),)
    empty = dict(unit_ids=np.empty(0, dtype=np.int64), cue_id=np.empty(0, dtype=np.int8))
    candidate = dict(unit_ids=np.array([0, 550, 1100]), cue_id=np.array([1, 2, 3]))
    def data(ticks, ids, commands):
        return dict(episode=replace(base, spike_ticks=np.asarray(ticks, dtype=np.int64),
                    spike_unit_indices=np.asarray(ids, dtype=np.int64)), commands=commands, groups=groups)
    bound = dict(completion_10=data([520], [1], candidate),
        cue_omission=data([520, 520, 520, 520], [0, 0, 1, 1], empty),
        wrong_memory=data([520, 520, 520], [1, 1, 1], candidate))
    rows = condition_contrasts(bound)
    row = next(row for row in rows if row["condition_id"] == "completion_10" and row["reference_condition"] == "cue_omission"
               and row["memory"] == 0 and row["population"] == "target_never_assigned_in_episode" and row["phase"] == "cue")
    assert (row["units"], row["spikes"], row["reference_spikes"], row["spike_difference"]) == (549, 1, 2, -1)
    assert row["rate_difference_hz"] == pytest.approx(-1000 / (549 * 20))
    assert row["subset_defined_by_condition"] == "completion_10"
    assert any(row["reference_condition"] == "wrong_memory" and row["spike_difference"] < 0 for row in rows)


def test_full_cue_empty_target_contrast_is_inapplicable(tmp_path):
    base, _, _ = fixture(tmp_path / "run", condition_index=5)
    groups = (observer.Group(0, 0, 168106, 0, 0, 0, 168106, 0, "CA3_Pyramidal"),)
    full = dict(unit_ids=np.arange(1650), cue_id=np.repeat([1, 2, 3], 550))
    empty = dict(unit_ids=np.empty(0, dtype=np.int64), cue_id=np.empty(0, dtype=np.int8))
    rows = condition_contrasts({condition: dict(episode=base, groups=groups, commands=commands)
                               for condition, commands in (("completion_full", full), ("cue_omission", empty), ("wrong_memory", empty))})
    selected = [row for row in rows if row["condition_id"] == "completion_full" and row["population"] == "target_never_assigned_in_episode"]
    assert selected and all(not row["applicable"] and row["units"] == 0 and row["rate_difference_hz"] is None for row in selected)


def test_paired_trace_checks_exact_event_times_and_counts(tmp_path):
    group = observer.Group(0, 0, 2, 0, 0, 0, 2, 0, "population")
    def trace(path):
        write_header(path, (group,))
        with path.open("ab") as stream:
            for tick in range(1001):
                for kind in (b"R", b"A", b"C"):
                    values = [2, 0] if kind == b"C" and tick == 520 else [0, 0]
                    stream.write(observer.RECORD.pack(kind, tick, tick if kind == b"R" else tick + 1, 0, 2))
                    stream.write(np.asarray(values, dtype="<i4" if kind == b"C" else "<f4").tobytes())
            stream.write(observer.RECORD.pack(b"E", 1001, 1001, -1, 0))
    left, right = tmp_path / "left", tmp_path / "right"
    trace(left); trace(right)
    empty = dict(unit_ids=np.empty(0, dtype=np.int64), time_ms=np.empty(0, dtype=np.int64))
    events = np.array([[520, 0], [520, 0]])
    report = compare_paired_currents(left, right, empty, empty, reference_events=events, current_events=events)
    assert report["archived_events_match_native_counts"] is True
    shifted = events.copy(); shifted[:, 0] = 521
    with pytest.raises(ValueError, match="EVENT_COUNTER_DIFFERS:current:ms=520"):
        compare_paired_currents(left, right, empty, empty, reference_events=events, current_events=shifted)


@pytest.mark.skipif(not os.environ.get("NRNSIM_CA3_READOUT_LAUNCHER"), reason="Compute-node launcher fixture only")
def test_actual_readout_launcher_complete_traces_and_incomplete_rejection(tmp_path):
    """Exercise staged archives, native trace exhaustion and both CLI exit states."""
    launcher = Path(os.environ["NRNSIM_CA3_READOUT_LAUNCHER"])
    repo = Path(os.environ["NRNSIM_CA3_TEST_REPO"])
    sources = tmp_path / "sources"
    sources.mkdir()
    roots = []
    for index in range(10):
        root = sources / ("first/episode" if index == 7 else f"remaining/condition_{index}/episode")
        root.parent.mkdir(parents=True)
        fixture(root, condition_index=index)
        roots.append(root)
    # Sparse zero-current payloads keep this full-unit format fixture small on disk.
    trace = roots[0] / "native_observation.bin"
    with trace.open("r+b") as stream:
        stream.seek(0, 2)
        def record(kind, tick, stop, values=None):
            stream.write(observer.RECORD.pack(kind, tick, stop, 0, 168106))
            begin = stream.tell()
            stream.seek(168106 * 4, 1)
            if values:
                end = stream.tell()
                for unit, count in values.items():
                    stream.seek(begin + unit * 4)
                    stream.write(np.asarray(count, dtype="<i4").tobytes())
                stream.seek(end)
        record(b"R", 0, 0)
        for tick in range(1001):
            record(b"A", tick, tick + 1)
            counts = {0: 1} if tick == 520 else {1: 1} if tick == 521 else {1101: 1} if tick == 1000 else None
            record(b"C", tick, tick + 1, counts)
        stream.write(observer.RECORD.pack(b"E", 1001, 1001, -1, 0))
    for root in roots[1:]:
        (root / "native_observation.bin").unlink()
        os.link(trace, root / "native_observation.bin")
    command = ["bash", str(launcher), str(repo), str(sources / "first"), str(sources / "remaining")]
    complete = tmp_path / "complete"
    subprocess.run(command + [str(complete)], check=True)
    report = json.loads((complete / "analysis/report.json").read_text())
    assert report["status"] == "COMPLETE_CA3_BENCHMARK_TEN_CONDITION_DEVELOPMENT"
    assert report["complete_episodes"] == 10 and report["production_episodes"] == 0
    assert report["formed_block_identity"]["circuit_id"] == "fixture-ca3"
    assert report["formed_circuit_realizations"] == 1 and report["independent_circuit_statistics"] is False
    assert len(report["paired_currents"]) == 9
    assert all(row["archived_events_match_native_counts"] for row in report["paired_currents"])
    assert report["contrast_rows"] > 0 and (complete / "analysis/condition_contrasts.csv").is_file()
    with np.load(complete / "analysis/activity_counts.npz", allow_pickle=False) as activity:
        assert activity["unit_counts"].shape == (10, 168106)
        np.testing.assert_array_equal(activity["unit_ids"], np.arange(168106))
        np.testing.assert_array_equal(activity["unit_counts"].sum(axis=1), np.full(10, 3))
        assert np.all(activity["field_counts_per_ms"][:, 1000, 0] == 1)
        assert np.all(activity["population_counts_per_ms"][:, 1000, 0] == 1)
        assert activity["interval_start_ms"][-1] == 1000
        assert activity["interval_stop_ms"][-1] == 1001
    assert len(report["episode_outcomes"]) == 10
    assert all(row["silent_units"] == 168103 for row in report["episode_outcomes"])
    block = sources / "complete_block"
    block.mkdir()
    for index, root in enumerate(roots):
        (block / f"condition_{index}").symlink_to(root.parent)
    complete_block = tmp_path / "complete_block_readout"
    subprocess.run(["bash", str(launcher), str(repo), str(block), str(complete_block)], check=True)
    block_report = json.loads((complete_block / "analysis/report.json").read_text())
    assert block_report["formed_block_identity"] == report["formed_block_identity"]
    assert block_report["complete_episodes"] == 10 and block_report["contrast_rows"] == report["contrast_rows"]
    with np.load(complete / "analysis/activity_counts.npz", allow_pickle=False) as legacy, \
            np.load(complete_block / "analysis/activity_counts.npz", allow_pickle=False) as whole:
        assert set(legacy.files) == set(whole.files)
        for name in legacy.files:
            np.testing.assert_array_equal(legacy[name], whole[name])
    incomplete = tmp_path / "incomplete"
    roots[-1].joinpath("report.json").unlink()
    failed = subprocess.run(command + [str(incomplete)], capture_output=True, text=True)
    assert failed.returncode != 0 and not (incomplete / "analysis").exists()


@pytest.mark.skipif(not os.environ.get("NRNSIM_CA3_EPISODE_LAUNCHER"), reason="Compute-node launcher routing fixture only")
def test_complete_block_producer_routes_all_conditions_without_changing_identity(tmp_path):
    """Check actual staging and routing; the fixture dispatch does not simulate."""
    launcher = Path(os.environ["NRNSIM_CA3_EPISODE_LAUNCHER"])
    repo = Path(os.environ["NRNSIM_CA3_TEST_REPO"])
    prepared, formation, analysis = [tmp_path / name for name in ("prepared", "formation", "analysis")]
    for path in (prepared / "ca3_runtime", formation / "observations", formation / "native_run", analysis):
        path.mkdir(parents=True)
    for path in (prepared / "host_current_replay", prepared / "host_current_replay_build.json",
                 formation / "native_report.json", formation / "observations/recording.json",
                 formation / "native_run/network65p.dat", analysis / "formation_replay.json",
                 analysis / "normalization.json", analysis / "normalization_replay.json",
                 analysis / "network65p_assem550_divnorm.dat"):
        path.write_text("{}\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    srun = tools / "srun"
    srun.write_text('#!/bin/bash\nset -eu\nshift\nexec "$@"\n')
    srun.chmod(0o755)
    dispatch = tools / "fixture_python"
    dispatch.write_text("#!" + sys.executable + "\n" + '''
import json,pathlib,subprocess,sys
args=sys.argv[1:]
assert args[:4]==['-B','-m','Libs.population_dynamics.ca3_benchmark','run']
subprocess.run([sys.executable,'-B','-m','Libs.population_dynamics.ca3_benchmark','run','--help'],check=True,stdout=subprocess.DEVNULL)
values={args[index]:args[index+1] for index in range(4,len(args),2)}
output=pathlib.Path(values['--output']);output.mkdir()
(output/'native_run').mkdir()
(output/'report.json').write_text(json.dumps(dict(status='FIXTURE_ROUTING_ONLY',arguments=values)))
''')
    dispatch.chmod(0o755)
    environment = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"],
        NRNSIM_BENCHMARK_PYTHON=str(dispatch), NRNSIM_CA3_FORMATION_SOURCE=str(formation),
        NRNSIM_CA3_ANALYSIS_SOURCE=str(analysis), NRNSIM_CA3_CIRCUIT_INDEX="1",
        NRNSIM_CA3_REUSE_CONDITION_ZERO="", NRNSIM_CA3_PREFIX_REFERENCE="",
        NRNSIM_BENCHMARK_CONFIG=str(repo / "Libs/configs/neural_population_dynamics.yaml"))
    output = tmp_path / "block"
    subprocess.run(["bash", str(launcher), str(repo), str(prepared), "2", "all", str(output)],
                   env=environment, check=True)
    assert {path.name for path in output.iterdir()} == {"software"} | {f"condition_{index}" for index in range(10)}
    for index in range(10):
        report = json.loads((output / f"condition_{index}/episode/report.json").read_text())
        arguments = report["arguments"]
        assert report["status"] == "FIXTURE_ROUTING_ONLY"
        assert arguments["--condition-index"] == str(index)
        assert arguments["--block-index"] == "2" and arguments["--circuit-index"] == "1"
        assert "--current-replay-engine" in arguments
    assert (output / "software/population_dynamics/benchmark_dataset.py").is_file()
