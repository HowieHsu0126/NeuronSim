"""Describe CA3 completion controls and compare actual paired background currents.

Readouts retain all ten conditions, silence and completion failures. The output
contains descriptive rates without a fitted success threshold or inference from
neurons. Independent formed circuits are still required for scientific claims.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .benchmark_dataset import read_circuit, read_episode
from .benchmark_tasks import CA3_CONDITIONS, ca3_commands
from .ca3_native_observation import iter_intervals, read_header
from .circuit_data import CircuitDataset
from .provenance import freeze_json, sha256_file


def describe_events(ticks, event_units, groups, commands, condition_id, *, duration_ms=1001) -> list[dict]:
    """Keep cue-specific denominators and unequal post-cue durations explicit."""
    if type(duration_ms) is not int or duration_ms not in (1001, 1210):
        raise ValueError("CA3_BENCHMARK_READOUT_DURATION_UNSUPPORTED")
    pyramidal = next(group for group in groups if group.name == "CA3_Pyramidal")
    ever_cued = np.unique(commands["unit_ids"])
    rows = []
    for memory, start in enumerate((520, 750, 980)):
        target = np.arange(pyramidal.start + memory * 550, pyramidal.start + (memory + 1) * 550)
        wrong = np.arange(pyramidal.start + ((memory + 1) % 3) * 550,
                          pyramidal.start + (((memory + 1) % 3) + 1) * 550)
        assigned = np.unique(commands["unit_ids"][commands["cue_id"] == memory + 1])
        populations = {"target_all": target,
            "target_directly_assigned_this_cue": np.intersect1d(target, assigned),
            "target_never_assigned_in_episode": np.setdiff1d(target, ever_cued),
            "wrong_memory_all": wrong,
            "nonmemory_pyramidal": np.arange(pyramidal.start + 1650, pyramidal.stop)}
        populations.update({group.name: np.arange(group.start, group.stop) for group in groups
                            if group.name != "CA3_Pyramidal"})
        post_stop = (750, 980, duration_ms)[memory]
        for population, ids in populations.items():
            members = np.isin(event_units, ids)
            for phase, begin, stop in (("pre_cue", start - 20, start), ("cue", start, start + 20),
                                       ("post_cue", start + 20, post_stop)):
                count = int(np.count_nonzero(members & (ticks >= begin) & (ticks < stop)))
                rows.append(dict(condition_id=condition_id, memory=memory, population=population,
                    phase=phase, start_ms=begin, stop_ms=stop, duration_ms=stop - begin,
                    units=len(ids), spikes=count, applicable=bool(len(ids)),
                    mean_rate_hz=count * 1000 / (len(ids) * (stop - begin)) if len(ids) else None,
                    persistence_duration_limited=phase == "post_cue" and stop - begin < 210))
    return rows


def _event_offsets(events: np.ndarray, units: int, duration_ms: int) -> tuple[np.ndarray, np.ndarray]:
    events = np.asarray(events)
    if (events.ndim != 2 or events.shape[1] != 2 or events.dtype.kind not in "iu"
            or np.any(events < 0) or np.any(events[:, 0] >= duration_ms) or np.any(events[:, 1] >= units)):
        raise ValueError("CA3_BENCHMARK_READOUT_EVENT_SUPPORT_INVALID")
    events = events[np.argsort(events[:, 0], kind="stable")]
    offsets = np.concatenate(([0], np.cumsum(np.bincount(events[:, 0], minlength=duration_ms))))
    return events, offsets


def compare_paired_currents(reference: Path, current: Path, reference_commands: dict, commands: dict, *, duration_ms=1001,
                            reference_events: np.ndarray | None = None, current_events: np.ndarray | None = None) -> dict:
    """Compare every current except the commands deliberately assigned that ms."""
    units, groups = read_header(reference)
    if read_header(current) != (units, groups):
        raise ValueError("CA3_BENCHMARK_PAIRED_NATIVE_UNIT_MAP_DIFFERS")
    if (reference_events is None) != (current_events is None):
        raise ValueError("CA3_BENCHMARK_BOTH_EVENT_RECORDS_REQUIRED")
    events = None if reference_events is None else [_event_offsets(value, units, duration_ms)
                                                   for value in (reference_events, current_events)]
    compared = 0
    left = iter_intervals(reference, expected_stop_ms=duration_ms)
    right = iter_intervals(current, expected_stop_ms=duration_ms)
    for a, b in zip(left, right, strict=True):
        tick = a.start_ms
        if tick != b.start_ms:
            raise ValueError("CA3_BENCHMARK_PAIRED_CURRENT_CLOCK_DIFFERS")
        if events is not None:
            for side, interval, (rows, offsets) in zip(("reference", "current"), (a, b), events, strict=True):
                counts = np.bincount(rows[offsets[tick]:offsets[tick + 1], 1], minlength=units)
                if not np.array_equal(interval.native_counts, counts):
                    raise ValueError(f"CA3_BENCHMARK_READOUT_EVENT_COUNTER_DIFFERS:{side}:ms={tick}")
        excluded = np.union1d(reference_commands["unit_ids"][reference_commands["time_ms"] == tick],
                              commands["unit_ids"][commands["time_ms"] == tick])
        keep = np.ones(units, dtype=bool); keep[excluded] = False
        mismatches = np.flatnonzero(keep & (a.effective_current.view("<u4") != b.effective_current.view("<u4")))
        if len(mismatches):
            raise ValueError(f"CA3_BENCHMARK_PAIRED_BACKGROUND_DIFFERS:ms={tick},units={mismatches[:8].tolist()}")
        compared += int(keep.sum())
    return dict(exact_float32_bits_equal=True, current_values_compared=compared,
                archived_events_match_native_counts=events is not None,
                intervals_compared=duration_ms, excluded="union_of_actual_assignment_units_in_each_ms", tolerance=0)


def compare_native_prefix(reference: Path, extended: Path) -> dict:
    """Exhaust both traces and compare every count and current in the old prefix."""
    units, groups = read_header(reference)
    if read_header(extended) != (units, groups):
        raise ValueError("CA3_BENCHMARK_EXTENSION_UNIT_MAP_DIFFERS")
    original = iter_intervals(reference, expected_stop_ms=1001)
    prefix_steps = extended_steps = 0
    for right in iter_intervals(extended, expected_stop_ms=1210):
        extended_steps += 1
        if right.start_ms >= 1001:
            continue
        left = next(original)
        if (left.start_ms != right.start_ms
                or not np.array_equal(left.native_counts, right.native_counts)
                or not np.array_equal(left.effective_current.view("<u4"), right.effective_current.view("<u4"))):
            raise ValueError(f"CA3_BENCHMARK_EXTENSION_NATIVE_PREFIX_DIFFERS:ms={right.start_ms}")
        prefix_steps += 1
    # Consume the old footer as well. A truncated original cannot qualify.
    if next(original, None) is not None or prefix_steps != 1001 or extended_steps != 1210:
        raise ValueError("CA3_BENCHMARK_EXTENSION_COMPLETE_TRACES_REQUIRED")
    return dict(prefix_intervals_compared=prefix_steps, extended_intervals_verified=extended_steps,
                unit_millisecond_counts_compared=prefix_steps * units,
                current_values_compared=prefix_steps * units, exact_float32_bits_equal=True,
                counts_exact=True, original_terminal_footer_verified=True,
                extended_terminal_footer_verified=True)


def bind_episode(path: Path) -> dict:
    """Bind the actual circuit, controls and duration to the complete archive."""
    native = json.loads((path / "report.json").read_text())
    duration = native.get("duration_ms")
    if (type(duration) is not int or duration not in (1001, 1210)
            or native.get("status") != "COMPLETE_CA3_BENCHMARK_DEVELOPMENT_EPISODE"
            or native.get("actual_split") != "development" or native.get("complete_episodes") != 1
            or native.get("units") != 168106 or native.get("events", -1) < 0
            or native.get("unit_millisecond_counts_compared") != 168106 * duration
            or native.get("native_terminal_footer_verified") is not True
            or native.get("all_unit_event_counter_equality") is not True):
        raise ValueError("CA3_BENCHMARK_COMPLETE_RECONCILED_EPISODE_REQUIRED:" + str(path))
    dataset = CircuitDataset(path / "dataset", splits=("development",))
    catalog = dataset.catalog
    rows = [row for shard in catalog["shards"] for row in shard["episodes"]]
    if (catalog["complete_episodes"] != 1 or len(rows) != 1 or len(catalog["circuits"]) != 1
            or catalog["binding"].get("scenario_id") != "ca3_pattern_completion"
            or catalog["binding"].get("scope") != "source_reproduction"
            or catalog.get("source_fidelity_qualified") is not False or rows[0]["split"] != "development"):
        raise ValueError("CA3_BENCHMARK_ONE_COMPLETE_ARCHIVED_EPISODE_REQUIRED")
    episode = read_episode(dataset, next(dataset.iter_episodes()))
    circuit = read_circuit(dataset, episode.circuit_id)
    episode.validate(len(circuit.unit_ids))
    condition = native["condition"]
    if condition not in [dict(condition_id=name, true_cue_units=correct, false_cue_units=wrong)
                         for name, correct, wrong in CA3_CONDITIONS]:
        raise ValueError("CA3_BENCHMARK_READOUT_CONDITION_DIFFERS")
    formed = native["formed_circuit"]
    identity = native["candidate_identity"]
    if (episode.dt_ms != 1.0 or episode.duration_ticks != duration
            or len(episode.spike_ticks) != native["events"]
            or (len(episode.spike_ticks) and episode.spike_ticks[-1] >= duration)
            or len(circuit.unit_ids) != 168106 or not np.array_equal(circuit.unit_ids, np.arange(168106))
            or episode.source_record.get("condition") != condition
            or episode.source_record.get("formation_binding") != formed
            or episode.source_record.get("circuit_provenance") != circuit.circuit_provenance
            or circuit.circuit_provenance.get("connectivity_identity") != formed["normalized_input_sha256"]
            or any(episode.seeds.get(name) != identity[key] for name, key in (
                ("uniform_seed", "background_seed"), ("engine_seed", "initial_state_seed"), ("cue_seed", "stimulus_seed")))):
        raise ValueError("CA3_BENCHMARK_READOUT_ARCHIVE_IDENTITY_DIFFERS")
    count, groups = read_header(path / "native_observation.bin")
    if count != len(circuit.unit_ids) or {group.name: group.units for group in groups} != formed["populations"]:
        raise ValueError("CA3_BENCHMARK_READOUT_NATIVE_UNIT_MAP_DIFFERS")
    for group in groups:
        if not np.all(circuit.populations[group.start:group.stop] == group.name):
            raise ValueError("CA3_BENCHMARK_READOUT_NATIVE_UNIT_MAP_DIFFERS")
    pattern_path = path / "native_run/trainPat.csv"
    if sha256_file(pattern_path) != formed["pattern_sha256"]:
        raise ValueError("CA3_BENCHMARK_READOUT_PATTERN_CHANGED")
    pattern = np.asarray(list(map(int, pattern_path.read_text().split())), dtype=np.int32)
    pyramidal = next(group for group in groups if group.name == "CA3_Pyramidal")
    expected = ca3_commands(pattern, pyramidal.start, condition, identity["stimulus_seed"])
    with np.load(path / "ordered_cue_commands.npz", allow_pickle=False) as saved:
        commands = {key: saved[key] for key in saved.files}
    if set(commands) != set(expected) or any(not np.array_equal(commands[name], value) for name, value in expected.items()):
        raise ValueError("CA3_BENCHMARK_READOUT_ORDERED_COMMANDS_DIFFER")
    targets = np.unique(commands["unit_ids"])
    if (episode.control_kind != "piecewise_constant"
            or not np.array_equal(episode.control_ticks, np.arange(duration))
            or len(episode.control_features) != len(targets) + len(groups) + 2):
        raise ValueError("CA3_BENCHMARK_READOUT_ARCHIVED_CONTROL_DIFFERS")
    assigned = np.zeros((duration, len(targets)), dtype=np.float32)
    if len(targets):
        assigned[commands["time_ms"], np.searchsorted(targets, commands["unit_ids"])] = 1
    if not np.array_equal(episode.control_values[:, :len(targets)], assigned):
        raise ValueError("CA3_BENCHMARK_READOUT_ARCHIVED_CONTROL_DIFFERS")
    context = dict(circuit_id=episode.circuit_id, block_id=episode.block_id, duration_ms=duration,
                   seeds=episode.seeds, formed_circuit=formed, circuit_provenance=circuit.circuit_provenance)
    return dict(path=path, native=native, episode=episode, circuit=circuit,
                groups=groups, commands=commands, context=context)


def describe_archived_activity(data: dict) -> dict:
    """Retain all units and both source-population and archived field time courses."""
    episode, circuit, groups = data["episode"], data["circuit"], data["groups"]
    duration, units = episode.duration_ticks, len(circuit.unit_ids)
    ticks, event_units = episode.spike_ticks, episode.spike_unit_indices
    unit_counts = np.bincount(event_units, minlength=units)
    field_counts = np.zeros((duration, len(circuit.field_names)), dtype=np.int64)
    np.add.at(field_counts, (ticks, circuit.field_index[event_units]), 1)
    population_index = np.empty(units, dtype=np.int64)
    for index, group in enumerate(groups):
        population_index[group.start:group.stop] = index
    population_counts = np.zeros((duration, len(groups)), dtype=np.int64)
    np.add.at(population_counts, (ticks, population_index[event_units]), 1)
    if (int(unit_counts.sum()) != len(ticks) or int(field_counts.sum()) != len(ticks)
            or int(population_counts.sum()) != len(ticks)
            or not np.array_equal(field_counts.sum(axis=1), population_counts.sum(axis=1))):
        raise ValueError("CA3_BENCHMARK_ARCHIVED_ACTIVITY_COUNTS_DIFFER")
    return dict(unit_counts=unit_counts, field_counts=field_counts,
                population_counts=population_counts, population_index=population_index)


def condition_contrasts(bound: dict[str, dict]) -> list[dict]:
    """Compare identical candidate-defined unit subsets with both controls.

    Reference counts use the candidate condition's subset. Using each control's
    own never-cued subset would change the denominator and compare different units.
    """
    rows = []
    for condition, data in bound.items():
        episode, commands, groups = data["episode"], data["commands"], data["groups"]
        measured = describe_events(episode.spike_ticks, episode.spike_unit_indices, groups, commands,
                                   condition, duration_ms=episode.duration_ticks)
        for control in ("cue_omission", "wrong_memory"):
            if condition == control:
                continue
            reference = bound[control]["episode"]
            baseline = describe_events(reference.spike_ticks, reference.spike_unit_indices, groups, commands,
                                       control, duration_ms=reference.duration_ticks)
            for current, previous in zip(measured, baseline, strict=True):
                row = dict(current, reference_condition=control, subset_defined_by_condition=condition,
                           reference_spikes=previous["spikes"], reference_mean_rate_hz=previous["mean_rate_hz"],
                           spike_difference=current["spikes"] - previous["spikes"])
                row["rate_difference_hz"] = (current["mean_rate_hz"] - previous["mean_rate_hz"]
                                              if current["applicable"] else None)
                rows.append(row)
    return rows


def readout(episodes: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result = dict(status="RUNNING_CA3_BENCHMARK_TEN_CONDITION_READOUT", complete_episodes=0,
                  formed_circuit_realizations=0, production_episodes=0, scientific_fidelity_qualified=False,
                  success_threshold_frozen=False, inferential_statistics=False, paired_currents=[])
    try:
        expected_conditions = {name for name, _, _ in CA3_CONDITIONS}
        bound, rows = {}, []
        shared = None
        for path in episodes:
            data = bind_episode(path)
            native = data["native"]
            condition = native["condition"]["condition_id"]
            if condition in bound or condition not in expected_conditions:
                raise ValueError("CA3_BENCHMARK_UNIQUE_CONDITION_REQUIRED")
            context = data["context"]
            if shared is not None and context != shared:
                raise ValueError("CA3_BENCHMARK_SAME_FORMED_CIRCUIT_AND_BLOCK_REQUIRED")
            if bound:
                first = next(iter(bound.values()))
                if (first["groups"] != data["groups"]
                        or first["circuit"].field_names != data["circuit"].field_names
                        or not np.array_equal(first["circuit"].field_index, data["circuit"].field_index)):
                    raise ValueError("CA3_BENCHMARK_PAIRED_OBSERVATION_MAP_DIFFERS")
            shared = context
            episode = data["episode"]
            rows.extend(describe_events(episode.spike_ticks, episode.spike_unit_indices, data["groups"],
                                       data["commands"], condition, duration_ms=episode.duration_ticks))
            bound[condition] = data
        if set(bound) != expected_conditions:
            raise ValueError("CA3_BENCHMARK_TEN_CONDITION_ENVELOPE_INCOMPLETE:" + str(sorted(expected_conditions - set(bound))))
        reference = bound["cue_omission"]
        for condition, data in bound.items():
            if condition == "cue_omission":
                continue
            comparison = compare_paired_currents(reference["path"] / "native_observation.bin", data["path"] / "native_observation.bin",
                                                  reference["commands"], data["commands"], duration_ms=shared["duration_ms"],
                reference_events=np.column_stack((reference["episode"].spike_ticks, reference["episode"].spike_unit_indices)),
                current_events=np.column_stack((data["episode"].spike_ticks, data["episode"].spike_unit_indices)))
            result["paired_currents"].append(dict(condition_id=condition, **comparison))
        with (output / "condition_rates.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        contrasts = condition_contrasts(bound)
        with (output / "condition_contrasts.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(contrasts[0])); writer.writeheader(); writer.writerows(contrasts)
        condition_order = tuple(name for name, _, _ in CA3_CONDITIONS)
        activity = [describe_archived_activity(bound[name]) for name in condition_order]
        circuit = reference["circuit"]
        np.savez_compressed(output / "activity_counts.npz", condition_ids=np.asarray(condition_order),
            unit_ids=circuit.unit_ids, unit_counts=np.stack([row["unit_counts"] for row in activity]),
            field_names=np.asarray(circuit.field_names), unit_field_index=circuit.field_index,
            field_counts_per_ms=np.stack([row["field_counts"] for row in activity]),
            population_names=np.asarray([group.name for group in reference["groups"]]),
            unit_population_index=activity[0]["population_index"],
            population_counts_per_ms=np.stack([row["population_counts"] for row in activity]),
            interval_start_ms=np.arange(shared["duration_ms"], dtype=np.int64),
            interval_stop_ms=np.arange(1, shared["duration_ms"] + 1, dtype=np.int64))
        result.update(status="COMPLETE_CA3_BENCHMARK_TEN_CONDITION_DEVELOPMENT", complete_episodes=10,
                      conditions=sorted(bound), paired_background_currents_verified=True,
                      formed_circuit_realizations=1, formed_block_identity=shared, rate_rows=len(rows),
                      contrast_rows=len(contrasts), contrast_controls=["cue_omission", "wrong_memory"],
                      contrast_subset_rule="Identical units defined by each evaluated condition; retain inapplicable empty subsets.",
                      independent_circuit_statistics=False,
                      activity_artifact="activity_counts.npz", activity_count_unit="spikes",
                      activity_interval_unit="ms", activity_intervals="half_open_native_one_ms",
                      activity_condition_order=list(condition_order),
                      silent_units_retained=True, all_activity_counts_reconciled=True,
                      episode_outcomes=[dict(condition_id=name, episode_id=bound[name]["episode"].episode_id,
                          events=len(bound[name]["episode"].spike_ticks),
                          silent_units=int(np.count_nonzero(activity[index]["unit_counts"] == 0)),
                          task_outcome=bound[name]["episode"].task_outcome)
                          for index, name in enumerate(condition_order)],
                      third_cue_post_duration_ms=shared["duration_ms"] - 1000)
    except BaseException as error:
        result.update(status="FAILED_CA3_BENCHMARK_TEN_CONDITION_READOUT", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        freeze_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    readout(args.episode, args.output)


if __name__ == "__main__":
    main()
