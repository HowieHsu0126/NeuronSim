"""Read all-unit responses in complete V1 benchmark circuit/noise blocks.

Source analysis uses (500,3000] ms. Delivered commands instead change at
501 ms, and their response window is half open. Both counts and terminal
events remain explicit. Selectivity and contrast differences are descriptive;
no neuron, block or training seed is treated as an independent circuit.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .benchmark_dataset import read_circuit, read_episode
from .benchmark_tasks import validate_protocol
from .circuit_campaign import campaign_config
from .circuit_data import CircuitDataset, CircuitEpisode, CircuitUnits
from .provenance import freeze_json, sha256_file
from .v1_visual_controls import CONTROL_FEATURES, make_commands
from .v1_feedforward_tuning import harmonic_alignment, source_orientation_for_grating

DIRECTIONS = tuple(range(0, 360, 45))
CONDITIONS = tuple(f"direction_{direction:03d}" for direction in DIRECTIONS) + ("gray", "low_contrast")
SOURCE_BOUNDARY_TICK = 2000
COMMAND_BOUNDARY_TICK = 2004
DURATION_TICKS = 12000
DT_MS = 0.25


def describe_episode(episode: CircuitEpisode, circuit: CircuitUnits) -> dict:
    """Check the actual recipe and command array before assigning a condition."""
    n = len(circuit.unit_ids)
    episode.validate(n)
    record = episode.source_record
    config = record["upstream_bindings"]["spike_inputs"]["config"]
    stimulus = config["stimulus"]
    visual = record.get("visual_commands", {})
    if (episode.duration_ticks != DURATION_TICKS or episode.dt_ms != DT_MS
            or record.get("source_revision") != circuit.source_revision
            or record.get("circuit_provenance") != circuit.circuit_provenance
            or record.get("input_mode") != "constructed_visual_inputs"
            or visual.get("source_command_representation_verified") is not True
            or visual.get("config", {}).get("stimulus") != stimulus
            or episode.control_kind != "piecewise_constant"
            or episode.control_features != CONTROL_FEATURES):
        raise ValueError("V1_BENCHMARK_BOUND_SOURCE_COMMANDS_REQUIRED")
    direction, contrast = stimulus["direction_deg"], stimulus["contrast_percent"]
    if (direction not in DIRECTIONS or contrast not in (0, 20, 80)
            or (contrast == 0 and direction != 0)
            or stimulus["spatial_frequency_cycles_per_degree"] != 0.04
            or stimulus["temporal_frequency_hz"] != 2.0):
        raise ValueError("V1_BENCHMARK_CONDITION_OUTSIDE_TASK_ENVELOPE")
    condition = f"direction_{int(direction):03d}" if contrast == 80 else ("gray" if contrast == 0 else "low_contrast")
    expected = make_commands(stimulus)
    if (not np.array_equal(episode.control_ticks, expected["control_ticks"])
            or not np.array_equal(episode.control_values, expected["control_values"])):
        raise ValueError("V1_BENCHMARK_DELIVERED_COMMAND_PROGRAM_DIFFERS")
    seeds = dict(native_kernel_rng_seed=episode.seeds.get("native_kernel_rng_seed"),
                 lgn_starting_noise_seed=config["spike_generation"]["starting_noise_seed"],
                 background_starting_noise_seed=config["background"]["starting_noise_seed"])
    if seeds != episode.seeds or any(type(value) is not int for value in seeds.values()):
        raise ValueError("V1_BENCHMARK_DECLARED_NOISE_STREAMS_DIFFER")
    ticks, units = episode.spike_ticks, episode.spike_unit_indices
    masks = dict(full=np.ones(len(ticks), dtype=bool), source_baseline=ticks <= SOURCE_BOUNDARY_TICK,
                 source_response=ticks > SOURCE_BOUNDARY_TICK, pre_command=ticks < COMMAND_BOUNDARY_TICK,
                 command_response=(ticks >= COMMAND_BOUNDARY_TICK) & (ticks < DURATION_TICKS),
                 terminal=ticks == DURATION_TICKS)
    counts = {name + "_counts": np.bincount(units[keep], minlength=n) for name, keep in masks.items()}
    if (not np.array_equal(counts["full_counts"], counts["source_baseline_counts"] + counts["source_response_counts"])
            or not np.array_equal(counts["full_counts"], counts["pre_command_counts"] +
                                  counts["command_response_counts"] + counts["terminal_counts"])):
        raise ValueError("V1_BENCHMARK_WINDOW_COUNT_RECONCILIATION_FAILED")
    ordinary = ticks < DURATION_TICKS
    flat = circuit.field_index[units[ordinary]] * DURATION_TICKS + ticks[ordinary]
    population_counts = np.bincount(flat, minlength=len(circuit.field_names) * DURATION_TICKS).reshape(
        len(circuit.field_names), DURATION_TICKS)
    if int(population_counts.sum()) + int(counts["terminal_counts"].sum()) != len(ticks):
        raise ValueError("V1_BENCHMARK_POPULATION_TIME_COUNTS_DIFFER")
    return dict(condition_id=condition, direction_deg=int(direction), contrast_percent=int(contrast),
                episode_id=episode.episode_id, circuit_id=episode.circuit_id, block_id=episode.block_id,
                seeds=seeds, initial_vm_identity=record["initial_vm_identity"], task_outcome=episode.task_outcome,
                population_counts=population_counts, **counts)


def selectivity(rates: np.ndarray) -> dict[str, np.ndarray]:
    """Circular resultant magnitudes of raw rates; all-zero units stay undefined."""
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 2 or rates.shape[0] != 8 or not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("V1_BENCHMARK_EIGHT_NONNEGATIVE_DIRECTION_RATES_REQUIRED")
    denominator = rates.sum(axis=0)
    theta = np.deg2rad(DIRECTIONS)
    result = {}
    for name, harmonic in (("direction_selectivity", 1), ("orientation_selectivity", 2)):
        numerator = np.abs(np.exp(harmonic * 1j * theta) @ rates)
        result[name] = np.divide(numerator, denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
    return result


def _same_circuit(left: CircuitUnits, right: CircuitUnits) -> bool:
    for name in ("unit_ids", "populations", "field_index", "coordinates", "node_type_ids", "source_tuning_angle_deg"):
        a, b = getattr(left, name), getattr(right, name)
        if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
            return False
    return all(getattr(left, name) == getattr(right, name) for name in (
        "field_names", "coordinate_semantics", "coordinate_units", "source_revision", "circuit_provenance"))


def _population_rows(data: dict, circuit: CircuitUnits) -> list[dict]:
    rows = []
    for index, population in [(-1, "all")] + list(enumerate(circuit.field_names)):
        selected = np.ones(len(circuit.unit_ids), dtype=bool) if index == -1 else circuit.field_index == index
        n = int(selected.sum())
        row = {name: data[name] for name in ("episode_id", "circuit_id", "block_id", "condition_id", "direction_deg", "contrast_percent", "task_outcome")}
        row.update(population=population, population_index=index, units=n)
        for name, seconds in (("full", 3.0), ("source_baseline", 0.5), ("source_response", 2.5),
                              ("pre_command", 0.501), ("command_response", 2.499)):
            counts = data[name + "_counts"][selected]
            total = int(counts.sum())
            row.update({name + "_spikes": total, name + "_silent_units": int(np.count_nonzero(counts == 0)),
                        name + "_mean_rate_hz": total / (n * seconds) if n else None})
        row["terminal_spikes"] = int(data["terminal_counts"][selected].sum())
        rows.append(row)
    return rows


def readout(sources: list[Path], output: Path, protocol: dict, *, intake_only: bool = False) -> dict:
    """Require ten distinct conditions per source circuit/block, unless validating intake."""
    validate_protocol(protocol, "v1_visual")
    if not sources or len({source.resolve() for source in sources}) != len(sources):
        raise ValueError("V1_BENCHMARK_UNIQUE_SOURCE_ARCHIVES_REQUIRED")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING_V1_BENCHMARK_READOUT", complete_source_episodes_read=0,
                  production_episodes=0, new_source_simulations=0, scientific_qualification=False,
                  inferential_statistics=False, formal_protocol_frozen=False, intake_only=intake_only,
                  noise_pairing_claim="shared_starting_seeds_not_per_unit_event_pairing", blocks=[], sources=[])
    try:
        circuits, blocks, ids = {}, {}, set()
        for source in sources:
            dataset = CircuitDataset(source, splits=("development",))
            catalog, binding = dataset.catalog, dataset.catalog["binding"]
            rows = [row for shard in catalog["shards"] for row in shard["episodes"]]
            if (binding.get("scenario_id") != "v1_visual" or binding.get("scope") != "source_reproduction"
                    or catalog.get("source_fidelity_qualified") is not False
                    or catalog.get("complete_episodes") != len(rows) or not rows
                    or len(dataset.episodes) != len(rows) or any(row["split"] != "development" for row in rows)):
                raise ValueError("V1_BENCHMARK_COMPLETE_DEVELOPMENT_ARCHIVE_REQUIRED")
            report["sources"].append(dict(path=str(source), catalog_sha256=sha256_file(source / "dataset.json")))
            for circuit_id in catalog["circuits"]:
                circuit = read_circuit(dataset, circuit_id)
                if circuit_id in circuits and not _same_circuit(circuits[circuit_id], circuit):
                    raise ValueError("V1_BENCHMARK_SOURCE_CIRCUIT_REBOUND")
                circuits[circuit_id] = circuit
            for row in dataset.iter_episodes():
                if row["episode_id"] in ids:
                    raise ValueError("V1_BENCHMARK_DUPLICATE_SOURCE_EPISODE")
                ids.add(row["episode_id"])
                blocks.setdefault((row["circuit_id"], row["block_id"]), []).append((dataset, row))
        csv_path = output / "population_summary.csv"
        with csv_path.open("x", newline="") as stream:
            writer = None
            for (circuit_id, block_id), entries in sorted(blocks.items()):
                circuit, accepted, context = circuits[circuit_id], {}, None
                for dataset, row in entries:
                    data = describe_episode(read_episode(dataset, row), circuit)
                    shared = dict(seeds=data["seeds"], initial_vm_identity=data["initial_vm_identity"])
                    if context is not None and context != shared:
                        raise ValueError("V1_BENCHMARK_SAME_INITIAL_STATE_AND_NOISE_BLOCK_REQUIRED")
                    context = shared
                    condition = data["condition_id"]
                    if condition in accepted:
                        raise ValueError("V1_BENCHMARK_DUPLICATE_CONDITION_IN_BLOCK")
                    accepted[condition] = data
                missing = sorted(set(CONDITIONS) - set(accepted))
                if missing and not intake_only:
                    raise ValueError("V1_BENCHMARK_TEN_CONDITION_ENVELOPE_INCOMPLETE:" + ",".join(missing))
                order = [condition for condition in CONDITIONS if condition in accepted]
                arrays = {name: np.stack([accepted[condition][name] for condition in order]) for name in (
                    "full_counts", "source_baseline_counts", "source_response_counts", "pre_command_counts",
                    "command_response_counts", "terminal_counts", "population_counts")}
                arrays["source_response_rate_hz"] = arrays["source_response_counts"] / 2.5
                if not missing:
                    rates = arrays["source_response_rate_hz"]
                    arrays.update(selectivity(rates[:8]))
                    if circuit.source_tuning_angle_deg is not None:
                        angle = source_orientation_for_grating(circuit.source_tuning_angle_deg)
                        aligned = harmonic_alignment(rates[:8], angle)
                        arrays.update(source_tuning_angle_deg=circuit.source_tuning_angle_deg,
                            source_grating_orientation_deg=np.mod(-circuit.source_tuning_angle_deg, 180.),
                            source_orientation_alignment=aligned["harmonic_2_signed_alignment"],
                            source_orientation_quadrature=aligned["harmonic_2_quadrature"])
                    low_direction = accepted["low_contrast"]["direction_deg"]
                    high = DIRECTIONS.index(low_direction)
                    arrays.update(low_minus_high_rate_hz=rates[9] - rates[high],
                                  low_minus_gray_rate_hz=rates[9] - rates[8],
                                  high_minus_gray_rate_hz=rates[high] - rates[8])
                target = output / "blocks" / circuit_id
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / (block_id + ".npz"), unit_ids=circuit.unit_ids,
                    population_index=circuit.field_index, population_names=np.asarray(circuit.field_names),
                    condition_ids=np.asarray(order), episode_ids=np.asarray([accepted[name]["episode_id"] for name in order]),
                    direction_deg=np.asarray([accepted[name]["direction_deg"] for name in order]),
                    contrast_percent=np.asarray([accepted[name]["contrast_percent"] for name in order]), **arrays)
                for condition in order:
                    rows = _population_rows(accepted[condition], circuit)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
                    writer.writerows(rows)
                report["blocks"].append(dict(circuit_id=circuit_id, block_id=block_id, units=len(circuit.unit_ids),
                    conditions=order, missing_conditions=missing, complete_ten_condition_envelope=not missing,
                    shared_source_context=context, source_tuning_angle_available=circuit.source_tuning_angle_deg is not None,
                    artifacts=str((target / (block_id + ".npz")).relative_to(output))))
                report["complete_source_episodes_read"] += len(order)
        report.update(status="V1_BENCHMARK_INTAKE_ONLY" if intake_only else "COMPLETE_V1_BENCHMARK_TEN_CONDITION_DEVELOPMENT",
            source_circuit_realizations=len(circuits), source_noise_blocks=len(blocks),
            complete_ten_condition_blocks=sum(not row["missing_conditions"] for row in report["blocks"]),
            all_unit_counts_and_population_time_courses_reconciled=True,
            native_population_time_step_ms=DT_MS, population_time_courses_stop_exclusive_tick=DURATION_TICKS,
            source_analysis_windows="baseline [0,500] ms; response (500,3000] ms",
            delivered_command_windows="pre [0,501) ms; response [501,3000) ms; terminal 3000 ms stored separately",
            gray_condition="The scheduled post-gray segment also has zero contrast; it is not a grating response.",
            rate_units="spikes per source unit per second",
            selectivity_definition="Magnitude of sum(rate * exp(i*k*theta))/sum(rate), k=1 DSI, k=2 OSI; raw (500,3000] ms rates; zero denominator is NaN.",
            source_orientation_definition="When archived source tuning angles exist, use theta=-phi modulo180 from source geometry; signed orientation harmonic and quadrature use raw rates. No fitted phase. Drift-direction sign unqualified. Missing historical angles are not inferred.",
            contrast_definition="Low minus matched-direction high, low minus gray and matched-direction high minus gray; source response rates; no event-pairing claim.",
            inference_unit="Independent qualified circuits; source blocks nest within circuits. Circuit independence and scientific tolerances remain unqualified.",
            raw_all_unit_time_courses="Retained in bound input archives; no dense neuron-by-time conversion or privileged future records in readout arrays.")
    except BaseException as error:
        report.update(status="FAILED_V1_BENCHMARK_READOUT", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        freeze_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intake-only", action="store_true", help="Validate available episodes without completing a ten-condition block.")
    args = parser.parse_args()
    protocol = campaign_config(args.config)["scenarios"]["v1_visual"]["task_protocol"]
    report = readout(args.source, args.output, protocol, intake_only=args.intake_only)
    print(json.dumps({key: report[key] for key in ("status", "complete_source_episodes_read", "complete_ten_condition_blocks")}))


if __name__ == "__main__":
    main()
