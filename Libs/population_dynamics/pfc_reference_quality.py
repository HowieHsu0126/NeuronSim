"""Prepare and read complete PFC reference-clock source-quality development episodes.

This pilot establishes measurements for later qualification. It grants no source
PASS, confidence interval, production membership or biological claim.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import yaml

from .benchmark_dataset import read_episode
from .benchmark_sampling import prepare_pfc_block, PFC_REFERENCE_COMPARATOR_NAMESPACE_OFFSET
from .benchmark_tasks import task_protocol, condition_for_episode
from .circuit_campaign import campaign_config, episode_identities
from .circuit_data import CircuitDataset
from .pfc_kernel_parameters import circuit_provenance
from .pfc_native_drive import SOURCE_ENTRY_SHA256
from .pfc_parameter_design import parameter_grid
from .pfc_source_dataset import export, _benchmark_registration
from .pfc_temporal_readouts import temporal_rows, COUNT_FIELDS, RATE_FIELDS, MOMENT_FIELDS
from .pfc_scale_analysis import scalar_readout
from .provenance import freeze_json, sha256_file

SCENARIO = "pfc_working_memory"
QUALITY_SCHEMA = "pfc_reference_source_qualification_v1"
PLAN_SCHEMA = "pfc_reference_quality_execution_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def quality_config(path: Path) -> dict:
    settings = yaml.safe_load(path.read_text())
    require(isinstance(settings, dict) and settings.get("schema") == QUALITY_SCHEMA
            and settings.get("scope") == "prospective_source_quality_development"
            and settings.get("reference_source_definition") == "discrete_adapted_EPFL"
            and settings.get("dt_ms") == 0.0125 and settings.get("duration_ms") == 5000.0
            and settings.get("coupled_core_neurons") == 100000
            and settings.get("circuits") == [23, 63, 74, 86, 52]
            and settings.get("initial_noise_block_indices") == [3, 4, 5]
            and settings.get("conditions_per_block") == 10
            and settings.get("complete_development_episodes") == 150
            and settings.get("production_eligible") is False
            and settings.get("formal_confirmation_released") is False,
            "PFC_REFERENCE_QUALITY_FIXED_PILOT_REQUIRED")
    return settings


def prepare(project: Path, settings_path: Path, base_path: Path, qualifications: dict, output: Path) -> dict:
    settings = quality_config(settings_path)
    campaign_path = project / settings["campaign_config"]
    campaign = campaign_config(campaign_path)
    spec = campaign["scenarios"][SCENARIO]
    require(spec["source_admission"]["status"] == "AUTHORIZED"
            and spec["task_protocol"] == task_protocol(SCENARIO, pfc_dt_ms=0.0125),
            "PFC_REFERENCE_QUALITY_UNFROZEN_AUTHORIZED_TASK_REQUIRED")
    require(set(qualifications) == {str(index) for index in settings["circuits"]},
            "PFC_REFERENCE_QUALITY_FIVE_TUPLE_BINDINGS_REQUIRED")
    grid = parameter_grid(yaml.safe_load((project / settings["parameter_design_config"]).read_text()), 100)
    base = yaml.safe_load(base_path.read_text())
    require(base.get("dt_ms") == 0.0125 and base.get("excitatory_neurons") == 80000
            and base.get("inhibitory_neurons") == 20000 and base.get("duration_ms") == 5000.,
            "PFC_REFERENCE_QUALITY_COMPLETE_SOURCE_BASE_REQUIRED")
    require(not output.exists(), "PFC_REFERENCE_QUALITY_PREPARATION_EXISTS")
    output.mkdir(parents=True)
    rows, block_reports = [], []
    circuit_splits = {row["circuit_index"]: row["split"] for row in episode_identities(campaign, SCENARIO)
                      if row["block_index"] == 0 and row["condition_index"] == 0}
    for circuit in settings["circuits"]:
        source = dict(base, **grid[circuit])
        source["benchmark_circuit_candidate"] = dict(
            status="PREPARED_PFC_PARAMETER_CIRCUIT_CANDIDATE", scenario_id=SCENARIO,
            circuit_id=f"pfc_working_memory_circuit_{circuit:03d}", circuit_index=circuit,
            intended_split=circuit_splits[circuit],
            parameters=grid[circuit], construction_scope="development", production_eligible=False,
            **circuit_provenance(source, SOURCE_ENTRY_SHA256))
        circuit_path = output / f"circuit_{circuit:03d}.yaml"
        circuit_path.write_text(yaml.safe_dump(source, sort_keys=False))
        for block in settings["initial_noise_block_indices"]:
            directory = output / "recipes" / f"circuit_{circuit:03d}" / f"block_{block:03d}"
            report = prepare_pfc_block(campaign_path, circuit_path, circuit, block, directory)
            require(report["duration_ticks"] == 400000 and report["coupled_core_units"] == 100000,
                    "PFC_REFERENCE_QUALITY_COMPLETE_BLOCK_REQUIRED")
            block_reports.append(str((directory / "report.json").relative_to(output)))
            for row in report["conditions"]:
                path = directory / row["config_path"]
                rows.append(dict(task_index=len(rows), identity=row["identity"],
                                 condition=row["condition"], config_path=str(path.relative_to(output)),
                                 config_sha256=sha256_file(path), qualification=qualifications[str(circuit)]))
    require(len(rows) == 150 and len({row["identity"]["episode_id"] for row in rows}) == 150,
            "PFC_REFERENCE_QUALITY_COMPLETE_PILOT_DENOMINATOR_REQUIRED")
    plan = dict(schema=PLAN_SCHEMA, scope="development", rows=rows, block_reports=block_reports,
                settings=settings, campaign_sha256=sha256_file(campaign_path),
                settings_sha256=sha256_file(settings_path), parameter_design_sha256=sha256_file(
                    project / settings["parameter_design_config"]),
                complete_episodes=0, production_episodes=0, source_quality_qualified=False)
    freeze_json(output / "execution.json", plan)
    return plan


def load_task(preparation: Path, index: int) -> tuple[dict, dict, Path]:
    plan = json.loads((preparation / "execution.json").read_text())
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("scope") == "development"
            and plan.get("production_episodes") == 0 and len(plan.get("rows", [])) == 150,
            "PFC_REFERENCE_QUALITY_PLAN_SCOPE_OR_DENOMINATOR_CHANGED")
    require(type(index) is int and 0 <= index < 150, "PFC_REFERENCE_QUALITY_TASK_INDEX_INVALID")
    row = plan["rows"][index]
    relative = Path(row["config_path"])
    require(not relative.is_absolute() and ".." not in relative.parts,
            "PFC_REFERENCE_QUALITY_RECIPE_PATH_INVALID")
    path = preparation / relative
    require(row["task_index"] == index and sha256_file(path) == row["config_sha256"],
            "PFC_REFERENCE_QUALITY_RECIPE_CHANGED")
    config = yaml.safe_load(path.read_text())
    require(config["benchmark_episode_candidate"]["identity"] == row["identity"]
            and config["benchmark_episode_candidate"]["production_eligible"] is False
            and config["dt_ms"] == 0.0125 and config["duration_ms"] == 5000.0,
            "PFC_REFERENCE_QUALITY_RECIPE_IDENTITY_CHANGED")
    return plan, row, path


def source_task(project: Path, preparation: Path, index: int, output: Path, *,
                qualification: Path | None = None) -> int:
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURMD_NODENAME") in ("cpu03", "cpu04", "cpu05", "cpu06")
            and not os.environ.get("SLURM_JOB_GPUS"), "PFC_REFERENCE_QUALITY_SOURCE_ALLOCATED_CPU_REQUIRED")
    _, row, config = load_task(preparation, index)
    environment = dict(os.environ, NEURONSIM_PROJECT_ROOT=str(project),
        NEURONSIM_PFC_SOURCE_CONFIG=str(config.resolve()),
        NEURONSIM_PFC_SOURCE_QUALIFICATION=str((qualification or Path(row["qualification"])).resolve()),
        NEURONSIM_PFC_SOURCE_CASE_ID=row["identity"]["episode_id"], NEURONSIM_PFC_SOURCE_BACKEND="numpy",
        NEURONSIM_PFC_OUTPUT_BASE=str(output.resolve()), NEURONSIM_PFC_SCRATCH="/tmp",
        CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    environment.pop("SLURM_ARRAY_TASK_ID", None)
    environment.pop("NEURONSIM_PFC_SOURCE_CASES_JSON", None)
    command = [os.environ.get("NEURONSIM_PFC_PYTHON", os.sys.executable), "-B",
               str(project / "Libs/scripts/run_pfc_source_cases.py")]
    return subprocess.run(command, cwd="/tmp", env=environment).returncode


def write_rows(path: Path, rows: list[dict]) -> None:
    require(bool(rows), "PFC_REFERENCE_QUALITY_EMPTY_TABLE")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def phase_edges(protocol: dict) -> list[int]:
    """Use actual source float-time activity; cue removal is at operation301."""
    duration, operation, dt = protocol["duration_ms"], protocol["stimulus_operation_ms"], protocol["dt_ms"]
    edges = {0, round(duration / dt), round(4000.0 / dt)}
    schedules = [(protocol["stimulus_start_ms"], protocol["stimulus_duration_ms"])]
    schedules += [(c["distractor_start_ms"], c["distractor_duration_ms"]) for c in protocol["conditions"]]
    for onset, length in schedules:
        start, stop = onset * .001, onset * .001 + length * .001
        previous = False
        for tick in range(int(duration / operation)):
            active = start <= tick * operation * .001 < stop
            if active != previous:
                edges.add(round(tick * operation / dt))
            previous = active
    return sorted(edges)


def event_measurements(episode, config: dict) -> tuple[list[dict], dict[str, np.ndarray], list[dict]]:
    n_e, n_i = config["excitatory_neurons"], config["inhibitory_neurons"]
    dt, duration = config["dt_ms"], config["duration_ms"]
    protocol = task_protocol(SCENARIO, pfc_dt_ms=dt)
    mask = episode.spike_ticks < episode.duration_ticks
    ticks, ids = episode.spike_ticks[mask], episode.spike_unit_indices[mask]
    temporal = temporal_rows(ticks, ids, n_e=n_e, n_i=n_i, dt_ms=dt, bin_ms=5., duration_ms=duration)
    boundaries = phase_edges(protocol)
    phases, unit_counts = [], []
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        left, right = np.searchsorted(ticks, [start, stop], side="left")
        values = scalar_readout(ids[left:right], n_e, n_i, (stop - start) * dt)
        values["target_directed_activity_hz"] = (
            values["vector_x_spikes"] * math.cos(math.radians(120.))
            + values["vector_y_spikes"] * math.sin(math.radians(120.))) / n_e / ((stop-start)*dt*.001)
        values["direction_missing_reason"] = "" if values["direction_defined"] else "silent_or_resultant_below_existing_roundoff_rule"
        phases.append(dict(phase_index=index, start_tick=start, stop_tick=stop,
                           start_ms=start*dt, stop_ms=stop*dt, **values))
        unit_counts.append(np.bincount(ids[left:right], minlength=n_e+n_i))
    require(sum(row["excitatory_spikes"]+row["inhibitory_spikes"] for row in phases) == len(ids),
            "PFC_REFERENCE_QUALITY_PHASE_EVENT_CONSERVATION_FAILED")
    bins = round(duration/5.)
    unit_order = np.argsort(ids, kind="stable")
    ordered_ids, ordered_bins = ids[unit_order], ticks[unit_order] // round(5./dt)
    temporal_mean, temporal_variance = np.zeros(n_e+n_i), np.zeros(n_e+n_i)
    fields = np.zeros((bins, 17), dtype=np.int64)
    for first in range(0, n_e+n_i, 4096):
        stop = min(first+4096, n_e+n_i)
        left, right = np.searchsorted(ordered_ids, [first, stop])
        flat = ordered_bins[left:right]*(stop-first)+ordered_ids[left:right]-first
        counts = np.bincount(flat, minlength=bins*(stop-first)).reshape(bins, stop-first)
        temporal_mean[first:stop] = counts.mean(axis=0)
        temporal_variance[first:stop] = counts.var(axis=0, ddof=1)
    field_ids = np.where(ids < n_e, ids*16//n_e, 16)
    fields[:] = np.bincount((ticks//round(5./dt))*17+field_ids, minlength=bins*17).reshape(bins,17)
    arrays = dict(phase_unit_counts=np.stack(unit_counts), temporal_unit_mean_counts=temporal_mean,
                  temporal_unit_variance_counts=temporal_variance, population_field_counts=fields,
                  phase_edges_ticks=np.asarray(boundaries, dtype=np.int64))
    require(int(fields.sum()) == len(ids), "PFC_REFERENCE_QUALITY_FIELD_EVENT_CONSERVATION_FAILED")
    return phases, arrays, temporal


def readout(preparation: Path, index: int, source: Path, output: Path, *,
            input_origins: Path | None = None) -> dict:
    _, row, config_path = load_task(preparation, index)
    return _readout_episode(config_path, row, source, output, input_origins=input_origins,
        block_variance_sample_axis="future_aggregate_of_blocks3_4_5_at_matched_times_within_circuit")


def readout_episode(config_path: Path, source: Path, output: Path, *,
                    input_origins: Path | None = None) -> dict:
    """Read one complete compiled development episode without a fixed pilot plan."""
    config = yaml.safe_load(config_path.read_text())
    require(isinstance(config, dict) and config.get("scope") == "source_reproduction"
            and config.get("dt_ms") == .0125 and config.get("duration_ms") == 5000.
            and config.get("excitatory_neurons") == 80000 and config.get("inhibitory_neurons") == 20000
            and config.get("readout_bin_ms") == 5. and config.get("excitatory_preference_bins") == 16
            and config.get("random_stream_policy") == "independent_initial_external",
            "PFC_REFERENCE_EPISODE_FULL_DEVELOPMENT_SOURCE_REQUIRED")
    candidate = config.get("benchmark_episode_candidate", {})
    circuit = config.get("benchmark_circuit_candidate", {})
    require(isinstance(candidate, dict) and candidate.get("execution_scope") == "development"
            and candidate.get("production_eligible") is False and candidate.get("complete_episodes") == 0
            and candidate.get("circuit_parameter_domain_bound") is False
            and isinstance(circuit, dict) and circuit.get("status") == "PREPARED_PFC_PARAMETER_CIRCUIT_CANDIDATE"
            and circuit.get("construction_scope") == "development" and circuit.get("production_eligible") is False
            and "benchmark_task_protocol" not in config,
            "PFC_REFERENCE_EPISODE_DEVELOPMENT_CANDIDATE_REQUIRED")
    require(all(item.get(name) in (None, False, 0) for item in (config, candidate, circuit)
                for name in ("scientific_fidelity_qualified", "scientific_claim_qualified",
                             "source_quality_qualified", "source_qualification_granted", "production_episodes")),
            "PFC_REFERENCE_EPISODE_PROMOTED_RESULT_REJECTED")
    identity = candidate.get("identity", {})
    require(isinstance(identity, dict) and identity.get("scenario_id") == SCENARIO
            and identity.get("sample_unit") == "complete_mechanistic_episode"
            and all(type(identity.get(name)) is int and 0 <= identity[name] < 100
                    for name in ("circuit_index", "block_index"))
            and type(identity.get("condition_index")) is int and 0 <= identity["condition_index"] < 10,
            "PFC_REFERENCE_EPISODE_IDENTITY_INVALID")
    c, b, k = (identity[name] for name in ("circuit_index", "block_index", "condition_index"))
    circuit_id = f"pfc_working_memory_circuit_{c:03d}"
    block_id = circuit_id + f"_block_{b:03d}"
    expected_split = "train" if c < 70 else "validation" if c < 80 else "calibration" if c < 90 else "test"
    require(identity.get("circuit_id") == circuit_id and identity.get("block_id") == block_id
            and identity.get("episode_id") == block_id + f"_condition_{k:02d}"
            and identity.get("split") == expected_split
            and all(type(identity.get(name)) is int and 0 <= identity[name] < 2**32
                    for name in
                    ("connectivity_seed", "initial_state_seed", "background_seed", "stimulus_seed")),
            "PFC_REFERENCE_EPISODE_IDENTITY_INVALID")
    expected_streams = dict(connectivity_seed=200000000 + c,
        initial_state_seed=210000000 + c * 1000 + b,
        background_seed=220000000 + c * 1000 + b,
        stimulus_seed=230000000 + c * 1000 + b)
    require(all(identity[name] == value for name, value in expected_streams.items()),
            "PFC_REFERENCE_EPISODE_DATASET_STREAM_IDENTITY_INVALID")
    # The existing registration replays the actual compiler and source case
    # checks. The exporter owns archive, code, stream and full-unit validation.
    _benchmark_registration(config)
    condition = condition_for_episode(task_protocol(SCENARIO, pfc_dt_ms=.0125), SCENARIO, identity)
    row = dict(identity=identity, condition=condition, config_sha256=sha256_file(config_path))
    return _readout_episode(config_path, row, source, output, input_origins=input_origins,
        block_variance_sample_axis="joint_initial_external_histories_nested_in_circuit; one_episode_supplies_no_between_block_variance")


def readout_reference_size(config_path: Path, source: Path, output: Path, *,
                           input_origins: Path | None = None) -> dict:
    """Read a declared2560-unit comparator without admitting it as100k data."""
    config = yaml.safe_load(config_path.read_text())
    require(isinstance(config, dict) and config.get("scope") == "source_reproduction"
            and config.get("excitatory_neurons") == 2048 and config.get("inhibitory_neurons") == 512
            and config.get("weight_scaling_factor") == 1. and config.get("dt_ms") == .0125
            and config.get("duration_ms") == 5000. and config.get("readout_bin_ms") == 5.
            and config.get("excitatory_preference_bins") == 16
            and config.get("random_stream_policy") == "independent_initial_external",
            "PFC_REFERENCE_SIZE_COMPLETE_SOURCE_REQUIRED")
    marker = config.get("pfc_source_size_comparison", {})
    candidate = config.get("benchmark_episode_candidate", {})
    circuit = config.get("benchmark_circuit_candidate", {})
    require(isinstance(marker, dict) and marker.get("schema") == "pfc_source_size_comparison_input_v1"
            and marker.get("scope") == "development_size_reference"
            and marker.get("comparator_role") == "original_size2560"
            and marker.get("coupled_core_units") == 2560 and marker.get("target_coupled_core_units") == 100000
            and marker.get("cross_size_noise_pairing") is False
            and marker.get("campaign_episodes") == marker.get("production_episodes") == 0
            and marker.get("production_eligible") is marker.get("source_quality_qualified") is marker.get("scientific_claim_qualified") is False
            and isinstance(candidate, dict) and candidate.get("execution_scope") == "development"
            and candidate.get("production_eligible") is False and candidate.get("circuit_parameter_domain_bound") is False
            and isinstance(circuit, dict) and circuit.get("construction_scope") == "development"
            and circuit.get("production_eligible") is False and "benchmark_task_protocol" not in config,
            "PFC_REFERENCE_SIZE_DECLARATION_REQUIRED")
    canonical, identity = marker.get("canonical_campaign_identity", {}), candidate.get("identity", {})
    require(isinstance(canonical, dict) and isinstance(identity, dict)
            and canonical.get("scenario_id") == SCENARIO
            and canonical.get("sample_unit") == "complete_mechanistic_episode"
            and all(type(canonical.get(name)) is int and 0 <= canonical[name] < 100
                    for name in ("circuit_index", "block_index"))
            and type(canonical.get("condition_index")) is int and 0 <= canonical["condition_index"] < 10,
            "PFC_REFERENCE_SIZE_IDENTITY_INVALID")
    c, b, k = (canonical[name] for name in ("circuit_index", "block_index", "condition_index"))
    circuit_id = f"pfc_working_memory_circuit_{c:03d}"
    block_id = circuit_id + f"_block_{b:03d}"
    expected_split = "train" if c < 70 else "validation" if c < 80 else "calibration" if c < 90 else "test"
    expected_streams = dict(connectivity_seed=200000000+c, initial_state_seed=210000000+c*1000+b,
                           background_seed=220000000+c*1000+b, stimulus_seed=230000000+c*1000+b)
    require(canonical.get("circuit_id") == circuit_id and canonical.get("block_id") == block_id
            and canonical.get("episode_id") == block_id + f"_condition_{k:02d}"
            and canonical.get("split") == expected_split
            and all(canonical.get(name) == value for name, value in expected_streams.items())
            and marker.get("comparator_namespace", {}).get("offset") == PFC_REFERENCE_COMPARATOR_NAMESPACE_OFFSET,
            "PFC_REFERENCE_SIZE_IDENTITY_INVALID")
    expected_identity = dict(canonical)
    expected_identity.update({name: value+PFC_REFERENCE_COMPARATOR_NAMESPACE_OFFSET for name,value in expected_streams.items()})
    require(identity == expected_identity, "PFC_REFERENCE_SIZE_STREAM_IDENTITY_INVALID")
    original = marker.get("original_circuit_candidate", {})
    original_dimensions = marker.get("original_dimensions", {})
    restored = dict(config, excitatory_neurons=80000, inhibitory_neurons=20000, weight_scaling_factor=.0256)
    require(isinstance(original, dict) and original.get("status") == "PREPARED_PFC_PARAMETER_CIRCUIT_CANDIDATE"
            and original.get("construction_scope") == "development" and original.get("production_eligible") is False
            and original.get("circuit_id") == circuit_id and original.get("circuit_index") == c
            and original.get("intended_split") == expected_split
            and original.get("parameters") == circuit.get("parameters")
            and all(original.get(name) == value for name,value in circuit_provenance(restored,SOURCE_ENTRY_SHA256).items())
            and original_dimensions.get("coupled_core_units") == 100000
            and isinstance(marker.get("original_circuit_config_sha256"), str)
            and len(marker["original_circuit_config_sha256"]) == 64,
            "PFC_REFERENCE_SIZE_ORIGINAL_CIRCUIT_BINDING_INVALID")
    _benchmark_registration(config)
    row = dict(identity=identity, config_sha256=sha256_file(config_path),
               condition=condition_for_episode(task_protocol(SCENARIO,pfc_dt_ms=.0125),SCENARIO,identity))
    return _readout_episode(config_path,row,source,output,input_origins=input_origins,
        block_variance_sample_axis="original_size2560_independent_histories; no_cross_size_pairing; one_episode_supplies_no_between_block_variance",
        source_size_comparison=marker)


def _readout_episode(config_path: Path, row: dict, source: Path, output: Path, *,
                     input_origins: Path | None, block_variance_sample_axis: str,
                     source_size_comparison: dict | None = None) -> dict:
    """Share full-unit source export and measurements across both entry points."""
    require(not output.exists(), "PFC_REFERENCE_QUALITY_READOUT_EXISTS")
    result = export([source], config_path, output / "export", input_origins=input_origins)
    require(result["status"] == "EXPORTED_VERIFIED_PFC_DEVELOPMENT_DATASET" and result["complete_episodes"] == 1,
            "PFC_REFERENCE_QUALITY_COMPLETE_EXPORT_REQUIRED")
    output.mkdir(parents=True, exist_ok=True)
    dataset = CircuitDataset(output / "export/dataset", splits=("development",))
    identity = row["identity"]
    episode = read_episode(dataset, dataset.episodes[identity["episode_id"]])
    config = yaml.safe_load(config_path.read_text())
    phases, arrays, temporal = event_measurements(episode, config)
    columns = dict(circuit_index=identity["circuit_index"], block_index=identity["block_index"],
                   episode_id=identity["episode_id"], condition_id=row["condition"]["condition_id"])
    write_rows(output / "phase_responses.csv", [dict(columns, **values) for values in phases])
    write_rows(output / "temporal_responses.csv", [dict(columns, **values) for values in temporal])
    np.savez(output / "unit_population_statistics.npz", **arrays)
    archive = source.parent if source.name == "source_run" else source
    native = json.loads((archive / "source_run/report.json").read_text())
    if input_origins is not None:
        origins = json.loads(input_origins.read_text())
        origin = next(item for item in origins if Path(item["staged"]).resolve() == archive.resolve())
        archive = Path(origin["archive"])
    initial = episode.privileged["initial_voltage_mV"]
    report = dict(status="PASS_REFERENCE_SOURCE_QUALITY_READOUT", identity=identity, condition=row["condition"],
        core_units=len(initial), duration_ticks=episode.duration_ticks, dt_ms=episode.dt_ms,
        events=len(episode.spike_ticks), terminal_tick_events=int(np.count_nonzero(episode.spike_ticks == episode.duration_ticks)),
        source_archive=str(archive.resolve()), source_delivery_sha256=sha256_file(archive / "delivery.json"),
        source_report_sha256=sha256_file((source.parent if source.name == "source_run" else source) / "source_run/report.json"),
        config_sha256=row["config_sha256"], output_dataset_catalog_sha256=result["dataset_catalog_sha256"],
        initial_voltage=dict(units="mV", declared_mean=-55., declared_variance=100./12.,
                             observed_mean=float(initial.mean()), observed_variance=float(initial.var()),
                             file_sha256=native["outputs"]["initial_voltage.npy"]["sha256"]),
        external_input=dict(driver=native["driver"], final_rng_sha256=native["outputs"]["final_rng.npz"]["sha256"]),
        source_cost=native.get("simulator_cost"), source_checks=native["validations"],
        phase_count=len(phases), temporal_bins=len(temporal),
        temporal_variance_sample_axis="5ms_time_bins_within_one_episode; not_independent_blocks",
        block_variance_sample_axis=block_variance_sample_axis,
        source_qualification_granted=False, scientific_claim_qualified=False, production_episodes=0,
        outputs={name:dict(path=name, sha256=sha256_file(output/name)) for name in (
            "phase_responses.csv", "temporal_responses.csv", "unit_population_statistics.npz")})
    if source_size_comparison is not None:
        report.update(source_size_comparison=source_size_comparison, campaign_episodes=0,
                      observation_scope="development_original_size2560_comparator_not100k_episode")
    freeze_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("project", "settings", "base-config", "qualifications", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for mode in ("source", "readout"):
        p = commands.add_parser(mode)
        p.add_argument("--preparation", type=Path, required=True)
        p.add_argument("--task-index", type=int, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--project" if mode == "source" else "--source", type=Path, required=True)
        if mode == "readout":
            p.add_argument("--input-origins", type=Path, help="Bound durable origins for staged source archives")
        else:
            p.add_argument("--qualification", type=Path, help="Recovered exact-tuple qualification; validated by the source engine")
    for mode in ("readout-episode", "readout-reference-size"):
        p = commands.add_parser(mode)
        for name in ("config", "source", "output"):
            p.add_argument("--" + name, type=Path, required=True)
        p.add_argument("--input-origins", type=Path, help="Bound durable origins for staged source archives")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        report = prepare(args.project, args.settings, args.base_config,
                         json.loads(args.qualifications.read_text()), args.output)
        print(json.dumps(dict(schema=report["schema"], prepared_episode_recipes=len(report["rows"]), production_episodes=0)))
    elif args.command == "source":
        raise SystemExit(source_task(args.project, args.preparation, args.task_index, args.output,
                                     qualification=args.qualification))
    elif args.command == "readout-episode":
        print(json.dumps(readout_episode(args.config, args.source, args.output,
                                         input_origins=args.input_origins)))
    elif args.command == "readout-reference-size":
        print(json.dumps(readout_reference_size(args.config, args.source, args.output,
                                                input_origins=args.input_origins)))
    else:
        print(json.dumps(readout(args.preparation, args.task_index, args.source, args.output,
                                 input_origins=args.input_origins)))


if __name__ == "__main__":
    main()
