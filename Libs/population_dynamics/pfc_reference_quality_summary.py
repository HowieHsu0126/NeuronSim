"""Complete-cohort descriptive summaries of PFC reference-source development.

A readout PASS binds acquired data; it is not a source-quality or scientific PASS.
Temporal fluctuations and between-block fluctuations have separate sample axes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import h5py
from typing import Any

import numpy as np
import yaml

from .pfc_fluctuation_analysis import unit_chunks
from .pfc_kernel_parameters import circuit_provenance, resolve_kernel_parameters
from .pfc_temporal_readouts import lagged_angular_displacement

from .provenance import freeze_json, sha256_file

SCHEMA = "pfc_reference_quality_summary_v1"
CONDITIONS = ("cue_none", "cue_aligned", "cue_off_target", "cue_antipodal", "cue_late_off_target",
              "omission_none", "omission_aligned", "omission_off_target", "omission_antipodal", "omission_late_off_target")
RATE_FIELDS = ("excitatory_rate_hz", "inhibitory_rate_hz", "cue_region_rate_hz",
               "noncue_region_rate_hz", "cue_selectivity_hz", "all_unit_rate_hz", "target_directed_activity_hz")
CIRCUITS = (23, 63, 74, 86, 52)
BLOCKS = (3, 4, 5)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def descriptive(values: Any) -> dict:
    """Keep the complete denominator. Undefined values withhold the aggregate."""
    values = list(values)
    defined = [float(x) for x in values if x is not None and x != ""]
    require(all(math.isfinite(x) for x in defined), "PFC_QUALITY_NONFINITE_DESCRIPTIVE_VALUE")
    complete = len(defined) == len(values)
    return dict(total_blocks=len(values), defined_blocks=len(defined), undefined_blocks=len(values)-len(defined),
                mean=float(np.mean(defined)) if complete and defined else None,
                sample_sd=float(np.std(defined, ddof=1)) if complete and len(defined)>1 else None)


def paired_contrasts(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Last axis follows the ten declared cue/omission conditions."""
    x = np.asarray(values, dtype=float)
    require(x.ndim >= 1 and x.shape[-1] == 10 and np.isfinite(x).all(),
            "PFC_QUALITY_COMPLETE_PAIRED_CONDITIONS_REQUIRED")
    cue_minus_omission = x[..., :5] - x[..., 5:]
    return cue_minus_omission, cue_minus_omission[..., 1:] - cue_minus_omission[..., :1]


def covariance_axes(fields: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return mean within-episode temporal and mean matched-time block covariance.

    Input is block x time x field. Blocks are joint initial/external histories.
    Time bins do not become independent simulation units.
    """
    x = np.asarray(fields, dtype=float)
    require(x.ndim == 3 and x.shape[0] >= 2 and x.shape[1] >= 2
            and x.shape[2] > 0 and np.isfinite(x).all(), "PFC_QUALITY_COVARIANCE_AXES_REQUIRED")
    temporal_centered = x - x.mean(axis=1, keepdims=True)
    temporal = np.einsum("btp,btq->pq", temporal_centered, temporal_centered)/(x.shape[0]*(x.shape[1]-1))
    block_centered = x - x.mean(axis=0, keepdims=True)
    between = np.einsum("btp,btq->pq", block_centered, block_centered)/((x.shape[0]-1)*x.shape[1])
    return temporal, between


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"PFC_QUALITY_JSON_OBJECT_REQUIRED:{path}")
    return value


def relative_file(directory: Path, value: str) -> Path:
    p = Path(value)
    require(not p.is_absolute() and ".." not in p.parts, "PFC_QUALITY_BOUND_PATH_INVALID")
    return directory / p


def auxiliary_attempt_failures(executions: Path, readouts: Path, expected: dict,
                               source_jobs: set[str], *,
                               scheduler_timeouts: dict[str, dict] | None = None
                               ) -> tuple[list[dict], list[dict]]:
    """Retain startup and readout failures outside existing source-batch records.

    Source batches already bind source attempts, including recovered scheduler
    timeouts. Their outer reports are not another source attempt. A readout is
    a separate operation and retains every failed retry for the same producer.
    """
    def attempts(root: Path):
        if not root.exists():
            return
        for directory in sorted(root.iterdir()):
            job = directory.name.removeprefix("run_")
            require(not directory.is_symlink() and directory.is_dir()
                    and directory.name == "run_" + job and job.isdecimal(),
                    "PFC_QUALITY_ATTEMPT_DIRECTORY_INVALID:" + str(directory))
            yield job, directory

    def terminal(job: str, directory: Path) -> tuple[dict, dict]:
        path = directory / "report.json"
        require(path.is_file(), "PFC_QUALITY_ATTEMPT_TERMINAL_REPORT_REQUIRED:" + str(path))
        report = read_json(path)
        require(str(report.get("job_id")) == job
                and report.get("task_index") == expected["task_index"]
                and report.get("node") in ("cpu03", "cpu04", "cpu05", "cpu06")
                and report.get("source_qualification_granted") is False
                and report.get("production_episodes") == 0,
                "PFC_QUALITY_ATTEMPT_IDENTITY_OR_SCOPE_CHANGED:" + str(path))
        binding = dict(job_id=job, task_index=expected["task_index"],
                       episode_id=expected["identity"]["episode_id"], node=report["node"],
                       report=str(path), report_sha256=sha256_file(path))
        return report, binding

    startups, failed_readouts = [], []
    for job, directory in attempts(executions):
        if job in source_jobs:
            continue
        report, binding = terminal(job, directory)
        require(report.get("status") == "FAILED"
                and type(report.get("exit_code")) is int and report["exit_code"] != 0
                and isinstance(report.get("error"), str) and bool(report["error"]),
                "PFC_QUALITY_STARTUP_FAILURE_WITHOUT_SOURCE_BATCH_REQUIRED:" + str(directory))
        startups.append(dict(binding, status="STARTUP_FAILURE_BEFORE_SOURCE_BATCH",
                             original_status=report["status"], exit_code=report["exit_code"],
                             error=report["error"]))
    for job, directory in attempts(readouts):
        if not (directory / "report.json").is_file() and job in (scheduler_timeouts or {}):
            binding = scheduler_timeouts[job]
            outcome = binding["terminal"]
            step = outcome.get("native_source_step", {})
            require(job in source_jobs and outcome.get("raw_job_id") == job
                    and outcome.get("state") == "TIMEOUT"
                    and outcome.get("source_execution_complete") is True
                    and step.get("state") == "COMPLETED" and step.get("exit_code") == "0:0"
                    and outcome.get("node") in ("cpu03", "cpu04", "cpu05", "cpu06"),
                    "PFC_QUALITY_INTERRUPTED_READOUT_TIMEOUT_BINDING_REQUIRED:" + str(directory))
            failed_readouts.append(dict(job_id=job, producer_job=job,
                task_index=expected["task_index"], episode_id=expected["identity"]["episode_id"],
                node=outcome["node"], status="READOUT_INTERRUPTED_BY_SCHEDULER_TIMEOUT",
                readout_directory=str(directory), scheduler_terminal=binding))
            continue
        report, binding = terminal(job, directory)
        producer = str(report.get("producer_job"))
        require(producer in source_jobs,
                "PFC_QUALITY_READOUT_PRODUCER_NOT_BOUND:" + str(directory))
        if report.get("status") == "PASS_COMPLETE_REFERENCE_QUALITY_READOUT":
            require(report.get("exit_code") == 0,
                    "PFC_QUALITY_SUCCESSFUL_READOUT_EXIT_CHANGED:" + str(directory))
            continue
        require(report.get("status") == "FAILED"
                and type(report.get("exit_code")) is int and report["exit_code"] != 0
                and isinstance(report.get("error"), str) and bool(report["error"]),
                "PFC_QUALITY_READOUT_TERMINAL_FAILURE_REQUIRED:" + str(directory))
        failed_readouts.append(dict(binding, status="READOUT_FAILURE", producer_job=producer,
                                   original_status=report["status"], exit_code=report["exit_code"],
                                   error=report["error"]))
    return startups, failed_readouts


def bound_source_metadata(archive: Path, delivery: dict, name: str) -> tuple[dict, str]:
    """Read one existing delivery-bound metadata file without rereading arrays."""
    entries = [row for row in delivery["files"] if row["path"] == name]
    require(len(entries) == 1, "PFC_QUALITY_SOURCE_METADATA_BINDING_REQUIRED:" + name)
    path = relative_file(archive, name)
    entry = entries[0]
    require(path.stat().st_size == entry["bytes"] and sha256_file(path) == entry["sha256"],
            "PFC_QUALITY_SOURCE_METADATA_CHANGED:" + name)
    return read_json(path), entry["sha256"]


def validate_source_metadata(report: dict, expected: dict, delivery: dict, *, core_units: int = 100000) -> None:
    """Bind copied input metadata to the source and its qualified stream law."""
    # Existing private math fixtures use small arrays with100k source metadata.
    # The explicit original-size reader alone selects2560 source metadata.
    source_units = 2560 if core_units == 2560 else 100000
    n_e, n_i = source_units * 4 // 5, source_units // 5
    archive = Path(report["source_archive"])
    native, _ = bound_source_metadata(archive, delivery, "source_run/report.json")
    config = native["resolved_config"]
    identity = expected["identity"]
    require(native.get("status") == "PFC_SOURCE_EXECUTION_COMPLETE"
            and native.get("source_execution_complete") is True
            and native.get("case_id") == identity["episode_id"]
            and native.get("config_sha256") == expected["config_sha256"]
            and config["benchmark_episode_candidate"]["identity"] == identity
            and config["dt_ms"] == .0125 and config["duration_ms"] == 5000.
            and config["excitatory_neurons"] == n_e and config["inhibitory_neurons"] == n_i,
            "PFC_QUALITY_NATIVE_SOURCE_IDENTITY_CHANGED")
    checks = native.get("validations", {})
    require(bool(checks) and all(row.get("passed") is True for row in checks.values())
            and report.get("source_checks") == checks
            and report.get("source_cost") == native.get("simulator_cost"),
            "PFC_QUALITY_NATIVE_SOURCE_CHECKS_OR_COST_CHANGED")
    driver = native["driver"]
    require(report.get("external_input", {}).get("driver") == driver
            and driver.get("seeds") == dict(initial_rng_seed=identity["initial_state_seed"],
                                            external_noise_seed=identity["background_seed"])
            and driver.get("random_stream_policy") == "independent_initial_external"
            and driver.get("packed_draw_order") == "tick_major_I_then_E"
            and driver.get("n_e") == n_e and driver.get("n_i") == n_i
            and driver.get("dt_seconds") == .0125 * .001
            and driver.get("ticks_generated") == driver.get("total_ticks") == 400000,
            "PFC_QUALITY_NATIVE_INPUT_METADATA_CHANGED")
    for name, copied in (("initial_voltage.npy", report["initial_voltage"]["file_sha256"]),
                         ("final_rng.npz", report["external_input"]["final_rng_sha256"])):
        output = native["outputs"][name]
        entries = [row for row in delivery["files"] if row["path"] == "source_run/" + name]
        require(len(entries) == 1 and output["path"] == name
                and entries[0]["sha256"] == output["sha256"] == copied
                and entries[0]["bytes"] == output["bytes"],
                "PFC_QUALITY_REALIZED_INPUT_BINDING_CHANGED:" + name)
    qualification, qsha = bound_source_metadata(archive, delivery, "qualification/comparison/qualification.json")
    qualified_delivery, dsha = bound_source_metadata(archive, delivery, "qualification/delivery.json")
    entries = [row for row in qualified_delivery["files"] if row["path"] == "comparison/qualification.json"]
    bound = native["backend_qualification"]
    require(len(entries) == 1 and entries[0]["sha256"] == qsha == bound["receipt_sha256"]
            and dsha == bound["delivery_sha256"] and bound["qualification"] == qualification
            and qualification.get("status") == "PASS_NATIVE_STREAM_SOURCE_EXECUTION"
            and qualification.get("backend") == native.get("backend") == "numpy"
            and qualification.get("validated_dt_ms") == [.0125]
            and qualification.get("validated_kernel_parameters") == [resolve_kernel_parameters(config)]
            and qualification.get("input_replay_status") == "PASS_NATIVE_NUMPY_DRIVE_REPLAY"
            and qualification.get("input_replay_passed_comparisons") == 24
            and qualification.get("source_entry_sha256") == native["source_entry_sha256"] == driver["source_entry_sha256"]
            and qualification.get("numpy_version") == driver["numpy_version"]
            and qualification.get("executor_sha256") == native["code"]["executor_sha256"]
            and qualification.get("drive_sha256") == native["code"]["drive_sha256"]
            and qualification.get("kernel_parameterization_sha256") == native["code"]["kernel_parameterization_sha256"]
            and native["circuit_provenance"] == circuit_provenance(config, native["source_entry_sha256"]),
            "PFC_QUALITY_QUALIFIED_STREAM_BINDING_CHANGED")


def paired_block_binding(rows: list[dict], reports: list[dict]) -> dict:
    """Check realized pairing under the already qualified deterministic stream law.

    Driver totals and final RNG hashes corroborate the bound stream execution.
    Their equality alone is not a proof of equal external event assignments.
    """
    require(len(rows) == len(reports) == 10
            and [row["identity"]["condition_index"] for row in rows] == list(range(10)),
            "PFC_QUALITY_COMPLETE_REALIZED_PAIRING_GROUP_REQUIRED")
    identity = rows[0]["identity"]
    reference = reports[0]
    for row, report in zip(rows, reports):
        require(row["identity"]["circuit_id"] == identity["circuit_id"]
                and row["identity"]["block_id"] == identity["block_id"]
                and report["identity"] == row["identity"]
                and report["initial_voltage"]["file_sha256"] == reference["initial_voltage"]["file_sha256"]
                and report["external_input"] == reference["external_input"],
                "PFC_QUALITY_REALIZED_CONDITION_PAIRING_CHANGED")
    return dict(circuit_index=identity["circuit_index"], block_index=identity["block_index"],
                block_id=identity["block_id"], episode_ids=[row["identity"]["episode_id"] for row in rows],
                initial_voltage_sha256=reference["initial_voltage"]["file_sha256"],
                final_rng_sha256=reference["external_input"]["final_rng_sha256"],
                driver=reference["external_input"]["driver"],
                basis="delivery-bound native metadata and exact-tuple qualified condition-independent stream execution")


def input_statistics(report: dict) -> tuple[dict, list[dict]]:
    """Describe saved input moments under the declared ideal IID sampler.

    Ten paired conditions share one input block. Unit draws define the sampler
    moment calculations; they do not increase the independent neural sample count.
    Standardized deviations are diagnostics without a normal-tail assumption,
    significance threshold, confidence interval or qualification decision.
    """
    identity = report["identity"]
    initial = report["initial_voltage"]
    external = report["external_input"]
    driver = external["driver"]
    n_e, n_i, ticks = (driver[key] for key in ("n_e", "n_i", "total_ticks"))
    require(all(type(value) is int and value > 0 for value in (n_e, n_i, ticks))
            and driver["ticks_generated"] == ticks
            and driver["seeds"] == dict(initial_rng_seed=identity["initial_state_seed"],
                                         external_noise_seed=identity["background_seed"]),
            "PFC_QUALITY_INPUT_STATISTICS_IDENTITY_OR_COUNT_CHANGED")
    units = n_e + n_i
    ideal_mean, ideal_variance = -55., 100. / 12.
    observed_mean, observed_variance = initial["observed_mean"], initial["observed_variance"]
    require(initial["units"] == "mV" and initial["declared_mean"] == ideal_mean
            and initial["declared_variance"] == ideal_variance
            and all(type(value) in (int, float) and math.isfinite(value)
                    for value in (observed_mean, observed_variance))
            and -60. <= observed_mean <= -50. and 0. <= observed_variance <= 25.,
            "PFC_QUALITY_INITIAL_MOMENT_METADATA_INVALID")
    common = dict(circuit_index=identity["circuit_index"], block_index=identity["block_index"],
                  block_id=identity["block_id"], split=identity["split"],
                  paired_conditions=10, independent_input_blocks=1,
                  initial_state_stream=identity["initial_state_seed"],
                  external_noise_stream=identity["background_seed"],
                  source_archive=report["source_archive"], source_delivery_sha256=report["source_delivery_sha256"])
    initial_row = dict(common, units=units, initial_voltage_sha256=initial["file_sha256"],
        ideal_sampler="IID_uniform_minus60_minus50_mV", ideal_mean_mV=ideal_mean,
        ideal_variance_mV2=ideal_variance, saved_variance_ddof=0,
        ideal_expected_saved_variance_mV2=(units - 1) * ideal_variance / units,
        observed_mean_mV=float(observed_mean), observed_variance_mV2=float(observed_variance),
        ideal_mean_standard_error_mV=math.sqrt(ideal_variance / units),
        standardized_mean_deviation=(observed_mean - ideal_mean) / math.sqrt(ideal_variance / units))
    afferents, probability, dt, rate = (driver[key] for key in
        ("binomial_n", "binomial_probability", "dt_seconds", "external_rate_Hz"))
    require(type(afferents) is int and afferents == 1000
            and type(probability) in (int, float) and math.isfinite(probability) and 0. < probability < 1.
            and type(dt) in (int, float) and math.isfinite(dt) and dt > 0.
            and rate == 1.4 and probability == rate * dt,
            "PFC_QUALITY_EXTERNAL_SAMPLER_PARAMETERS_CHANGED")
    duration = ticks * dt
    external_rows = []
    for population, count in (("E", n_e), ("I", n_i)):
        draws = count * ticks
        trials = draws * afferents
        observed = driver["external_event_totals"][population]
        require(type(observed) is int and 0 <= observed <= trials,
                "PFC_QUALITY_EXTERNAL_TOTAL_SUPPORT_INVALID")
        expected = trials * probability
        variance = trials * probability * (1. - probability)
        external_rows.append(dict(common, population=population, units=count, ticks=ticks,
            dt_seconds=dt, duration_seconds=duration, afferents_per_unit=afferents,
            binomial_draws=draws, ideal_bernoulli_trials=trials, probability_per_tick=probability,
            ideal_sampler="IID_binomial_per_unit_and_tick", ideal_expected_total_events=expected,
            ideal_total_event_variance=variance, ideal_total_event_standard_deviation=math.sqrt(variance),
            observed_total_events=observed,
            standardized_total_deviation=(observed - expected) / math.sqrt(variance),
            ideal_rate_hz_per_afferent=rate,
            observed_rate_hz_per_afferent=observed / (count * afferents * duration),
            final_rng_sha256=external["final_rng_sha256"]))
    return initial_row, external_rows


def readout_mapping(path: Path) -> dict[str, Path]:
    """Accept an episode-id mapping or an explicit list of readout directories."""
    value = json.loads(path.read_text())
    rows = list(value.items()) if isinstance(value, dict) else [(None, x) for x in value] if isinstance(value, list) else []
    require(bool(rows), "PFC_QUALITY_EXPLICIT_READOUTS_REQUIRED")
    result, locations = {}, set()
    for key, entry in rows:
        if isinstance(entry, dict):
            key = entry.get("episode_id", key)
            entry = entry.get("path", entry.get("readout"))
        require(isinstance(entry, str) and bool(entry), "PFC_QUALITY_READOUT_PATH_REQUIRED")
        directory = Path(entry)
        if not directory.is_absolute(): directory = path.parent / directory
        directory = directory.resolve()
        require(directory not in locations, "PFC_QUALITY_DUPLICATE_READOUT_DIRECTORY")
        locations.add(directory)
        require((directory/"report.json").is_file(), f"PFC_QUALITY_MISSING_READOUT_REPORT:{directory}")
        report = read_json(directory/"report.json")
        episode = report.get("identity", {}).get("episode_id")
        require(isinstance(episode,str) and (key is None or key == episode), "PFC_QUALITY_READOUT_IDENTITY_MISMATCH")
        require(episode not in result, "PFC_QUALITY_DUPLICATE_READOUT_IDENTITY")
        result[episode] = directory
    return result


def validate_plan(preparation: Path) -> dict:
    plan = read_json(preparation/"execution.json")
    settings = plan.get("settings", {})
    require(plan.get("schema") == "pfc_reference_quality_execution_v1" and plan.get("scope") == "development"
            and plan.get("production_episodes") == 0 and plan.get("source_quality_qualified") is False,
            "PFC_QUALITY_DEVELOPMENT_PLAN_REQUIRED")
    require(settings.get("circuits") == list(CIRCUITS) and settings.get("initial_noise_block_indices") == list(BLOCKS)
            and settings.get("complete_development_episodes") == 150 and settings.get("conditions_per_block") == 10
            and settings.get("coupled_core_neurons") == 100000 and settings.get("excitatory_neurons") == 80000
            and settings.get("inhibitory_neurons") == 20000 and settings.get("dt_ms") == .0125
            and settings.get("duration_ms") == 5000. and settings.get("observation_bin_ms") == 5.
            and settings.get("production_eligible") is False and settings.get("formal_confirmation_released") is False
            and settings.get("training_seeds_are_dataset_streams") is False,
            "PFC_QUALITY_FIXED_COMPLETE_PILOT_REQUIRED")
    rows = plan.get("rows", [])
    require(len(rows) == 150, "PFC_QUALITY_PREPARATION_DENOMINATOR_CHANGED")
    cells, episodes, pair_streams = set(), set(), {}
    for index,row in enumerate(rows):
        identity = row["identity"]
        c,b,k = (identity[key] for key in ("circuit_index","block_index","condition_index"))
        cell=(c,b,k)
        episode=f"pfc_working_memory_circuit_{c:03d}_block_{b:03d}_condition_{k:02d}"
        require(c in CIRCUITS and b in BLOCKS and type(k) is int and 0 <= k < 10
                and row["task_index"] == index and cell not in cells and episode not in episodes
                and identity["episode_id"] == episode and identity["circuit_id"] == f"pfc_working_memory_circuit_{c:03d}"
                and identity["block_id"] == f"pfc_working_memory_circuit_{c:03d}_block_{b:03d}"
                and identity["scenario_id"] == "pfc_working_memory" and identity["sample_unit"] == "complete_mechanistic_episode"
                and row["condition"]["condition_id"] == CONDITIONS[k], "PFC_QUALITY_PREPARATION_CELL_INVALID")
        expected_split = "train" if c < 70 else "validation" if c < 80 else "calibration" if c < 90 else "test"
        require(identity["split"] == expected_split, "PFC_QUALITY_WHOLE_CIRCUIT_SPLIT_CHANGED")
        streams=tuple(identity[key] for key in ("connectivity_seed","initial_state_seed","background_seed","stimulus_seed"))
        require(all(type(v) is int and v not in (41,42,43) for v in streams), "PFC_QUALITY_DATASET_STREAMS_INVALID")
        require(pair_streams.setdefault((c,b),streams) == streams, "PFC_QUALITY_PAIRED_STREAMS_CHANGED")
        cells.add(cell); episodes.add(episode)
        config_path=relative_file(preparation,row["config_path"])
        require(sha256_file(config_path) == row["config_sha256"], "PFC_QUALITY_BOUND_RECIPE_CHANGED")
        config=yaml.safe_load(config_path.read_text())
        candidate=config["benchmark_episode_candidate"]
        require(candidate["identity"] == identity and candidate["condition_id"] == CONDITIONS[k]
                and candidate["production_eligible"] is False and config["dt_ms"] == .0125
                and config["duration_ms"] == 5000. and config["excitatory_neurons"] == 80000
                and config["inhibitory_neurons"] == 20000,
                "PFC_QUALITY_BOUND_RECIPE_CONTRACT_CHANGED")
    require(cells == {(c,b,k) for c in CIRCUITS for b in BLOCKS for k in range(10)},
            "PFC_QUALITY_COMPLETE_COHORT_REQUIRED")
    return plan


def load_cell(directory: Path, expected: dict, *, core_units: int=100000) -> tuple[dict,list[dict],dict]:
    report=read_json(directory/"report.json")
    require(report.get("status") == "PASS_REFERENCE_SOURCE_QUALITY_READOUT"
            and report.get("identity") == expected["identity"] and report.get("condition") == expected["condition"]
            and report.get("config_sha256") == expected["config_sha256"]
            and report.get("source_qualification_granted") is False and report.get("scientific_claim_qualified") is False
            and report.get("production_episodes") == 0, "PFC_QUALITY_FAILED_OR_STALE_READOUT")
    require(report.get("core_units") == core_units and report.get("dt_ms") == .0125
            and report.get("duration_ticks") == 400000 and report.get("temporal_bins") == 1000,
            "PFC_QUALITY_UNIT_OR_CLOCK_MISMATCH")
    archive=Path(report["source_archive"])
    require(sha256_file(archive/"delivery.json") == report["source_delivery_sha256"], "PFC_QUALITY_SOURCE_DELIVERY_CHANGED")
    delivery=read_json(archive/"delivery.json")
    require(delivery.get("status") == "ARCHIVED_PFC_SOURCE_EXECUTION"
            and delivery.get("case_id") == expected["identity"]["episode_id"] and delivery.get("exit_code") == 0
            and delivery.get("source_execution_complete") is True and delivery.get("result_status") == "PFC_SOURCE_EXECUTION_COMPLETE"
            and delivery.get("scientific_fidelity_qualified") is False and delivery.get("production_episodes") == 0,
            "PFC_QUALITY_INCOMPLETE_OR_WRONG_SOURCE_DELIVERY")
    validate_source_metadata(report, expected, delivery, core_units=core_units)
    outputs=report.get("outputs", {})
    for name in ("phase_responses.csv","temporal_responses.csv","unit_population_statistics.npz"):
        require(name in outputs and outputs[name]["path"] == name, "PFC_QUALITY_READOUT_OUTPUT_BINDING_REQUIRED")
        require(sha256_file(relative_file(directory,outputs[name]["path"])) == outputs[name]["sha256"],
                "PFC_QUALITY_BOUND_READOUT_CHANGED")
    with (directory/"phase_responses.csv").open(newline="") as stream: phases=list(csv.DictReader(stream))
    with np.load(directory/"unit_population_statistics.npz",allow_pickle=False) as source:
        arrays={key:source[key] for key in ("phase_unit_counts","temporal_unit_mean_counts",
                "temporal_unit_variance_counts","population_field_counts","phase_edges_ticks")}
    edges=arrays["phase_edges_ticks"]
    require(edges.dtype.kind in "iu" and edges.ndim == 1 and len(edges)>1 and edges[0] == 0
            and edges[-1] == 400000 and np.all(np.diff(edges)>0) and len(phases) == len(edges)-1
            and len(phases) == report["phase_count"], "PFC_QUALITY_PHASE_DOMAIN_MISMATCH")
    counts,fields=arrays["phase_unit_counts"],arrays["population_field_counts"]
    require(counts.shape == (len(phases),core_units) and fields.shape == (1000,17)
            and all(x.dtype.kind in "iu" and np.all(x>=0) for x in (counts,fields)), "PFC_QUALITY_COMPLETE_NONNEGATIVE_COUNTS_REQUIRED")
    for name in ("temporal_unit_mean_counts","temporal_unit_variance_counts"):
        x=arrays[name]
        require(x.shape == (core_units,) and np.isfinite(x).all() and np.all(x>=0), "PFC_QUALITY_UNIT_MOMENT_INVALID")
    retained=report["events"]-report["terminal_tick_events"]
    require(type(retained) is int and retained>=0 and int(counts.sum()) == int(fields.sum()) == retained,
            "PFC_QUALITY_EVENT_CONSERVATION_FAILED")
    np.testing.assert_allclose(arrays["temporal_unit_mean_counts"]*1000,counts.sum(axis=0),rtol=0,atol=1e-9)
    n_e=core_units*4//5
    mapping=np.r_[np.arange(n_e,dtype=np.int64)*16//n_e,np.full(core_units-n_e,16,dtype=np.int64)]
    require(np.all(np.bincount(mapping,minlength=17)>0), "PFC_QUALITY_SOURCE_POPULATIONS_EMPTY")
    np.testing.assert_array_equal(fields.sum(axis=0),np.bincount(mapping,weights=counts.sum(axis=0),minlength=17))
    identity=expected["identity"]
    for p,row in enumerate(phases):
        require(row["episode_id"] == identity["episode_id"] and row["condition_id"] == expected["condition"]["condition_id"]
                and int(row["circuit_index"]) == identity["circuit_index"] and int(row["block_index"]) == identity["block_index"]
                and int(row["phase_index"]) == p and int(row["start_tick"]) == int(edges[p])
                and int(row["stop_tick"]) == int(edges[p+1]) and float(row["start_ms"]) == edges[p]*.0125
                and float(row["stop_ms"]) == edges[p+1]*.0125, "PFC_QUALITY_PHASE_IDENTITY_OR_CLOCK_MISMATCH")
        require(int(row["excitatory_spikes"]) == int(counts[p,:n_e].sum())
                and int(row["inhibitory_spikes"]) == int(counts[p,n_e:].sum()), "PFC_QUALITY_PHASE_COUNT_MISMATCH")
        duration_seconds=(edges[p+1]-edges[p])*.0125*.001
        for field,total,units in (("excitatory_rate_hz",counts[p,:n_e].sum(),n_e),
                                  ("inhibitory_rate_hz",counts[p,n_e:].sum(),core_units-n_e),
                                  ("all_unit_rate_hz",counts[p].sum(),core_units)):
            np.testing.assert_allclose(float(row[field]),total/units/duration_seconds,rtol=1e-12,atol=1e-12)
        for field in RATE_FIELDS: require(math.isfinite(float(row[field])), "PFC_QUALITY_PHASE_RATE_NONFINITE")
        defined=row["direction_defined"] in ("True","true","1")
        require(row["direction_defined"] in ("True","true","1","False","false","0"), "PFC_QUALITY_DIRECTION_FLAG_INVALID")
        require(defined == bool(row["decoded_angle_deg"] and row["circular_cue_error_deg"])
                and (defined or bool(row.get("direction_missing_reason"))), "PFC_QUALITY_UNDEFINED_DIRECTION_NOT_PRESERVED")
        for field in ("decoded_angle_deg","circular_cue_error_deg"):
            require(not defined or math.isfinite(float(row[field])), "PFC_QUALITY_DIRECTION_NONFINITE")
    return report,phases,arrays



def matched_time_unit_statistics(events, phase_counts, field_counts, *, core_units: int, n_e: int) -> dict:
    """Reuse event-to-count chunks to separate unit and shared block fluctuations."""
    require(len(events) == 3 and phase_counts.shape[0] == field_counts.shape[0] == 3,
            "PFC_QUALITY_THREE_MATCHED_BLOCKS_REQUIRED")
    bins = field_counts.shape[1]
    totals = phase_counts.sum(axis=1)
    unit_variance = np.empty(core_units, dtype=np.float64)
    for first, stop, counts in unit_chunks(events, core_units, bins):
        np.testing.assert_array_equal(counts.sum(axis=1), totals[:, first:stop])
        unit_variance[first:stop] = counts.var(axis=0, ddof=1, dtype=np.float64).mean(axis=0)
    population = {}
    for name, groups, first, stop in (("excitatory", range(16), 0, n_e),
                                      ("inhibitory", [16], n_e, core_units),
                                      ("all_units", range(17), 0, core_units)):
        units = stop - first
        values = field_counts[:, :, list(groups)].sum(axis=2) / units
        variance = float(values.var(axis=0, ddof=1).mean())
        diagonal = float(unit_variance[first:stop].sum() / units**2)
        population[name] = dict(
            between_block_matched_time_mean_single_unit_variance_counts2=float(unit_variance[first:stop].mean()),
            between_block_matched_time_diagonal_contribution_counts2=diagonal,
            between_block_matched_time_ordered_offdiagonal_contribution_counts2=variance-diagonal)
    return dict(unit_variance=unit_variance, population=population)


def archived_events(directory: Path, report: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read authenticated exported events. No simulator or measurement rerun occurs."""
    catalog_path = directory / "export/dataset/dataset.json"
    require(sha256_file(catalog_path) == report["output_dataset_catalog_sha256"],
            "PFC_QUALITY_EXPORTED_CATALOG_CHANGED")
    catalog = read_json(catalog_path)
    require(catalog["complete_episodes"] == 1 and len(catalog["shards"]) == 1,
            "PFC_QUALITY_ONE_COMPLETE_EXPORTED_EPISODE_REQUIRED")
    shard = catalog["shards"][0]
    require(len(shard["episodes"]) == 1 and shard["episodes"][0]["episode_id"] == report["identity"]["episode_id"],
            "PFC_QUALITY_EXPORTED_EPISODE_CHANGED")
    path = relative_file(directory / "export/dataset", shard["path"])
    require(sha256_file(path) == shard["sha256"], "PFC_QUALITY_EXPORTED_EVENTS_CHANGED")
    with h5py.File(path, "r") as source:
        group = source[report["identity"]["episode_id"]]
        ticks, ids = group["spike_ticks"][:], group["spike_unit_indices"][:]
    require(ticks.shape == ids.shape == (report["events"],) and ticks.dtype.kind in "iu" and ids.dtype.kind in "iu"
            and np.all((ticks >= 0) & (ticks <= report["duration_ticks"]))
            and np.all((ids >= 0) & (ids < report["core_units"])), "PFC_QUALITY_EXPORTED_EVENT_SUPPORT_CHANGED")
    keep = ticks < report["duration_ticks"]
    require(int((~keep).sum()) == report["terminal_tick_events"], "PFC_QUALITY_TERMINAL_EVENT_COUNT_CHANGED")
    ticks, ids = ticks[keep], ids[keep]
    order = np.argsort(ids, kind="stable")
    return ids[order], ticks[order] // 400

def write_rows(path: Path, rows: list[dict]) -> None:
    require(bool(rows), "PFC_QUALITY_EMPTY_AGGREGATE")
    with path.open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def _aggregate(preparation: Path, readouts: Path, output: Path, *, core_units: int, n_e: int) -> dict:
    require(not output.exists(), "PFC_QUALITY_SUMMARY_OUTPUT_EXISTS")
    plan=validate_plan(preparation)
    mapping=readout_mapping(readouts)
    expected={row["identity"]["episode_id"]:row for row in plan["rows"]}
    require(set(mapping) == set(expected), f"PFC_QUALITY_INCOMPLETE_COHORT:missing={len(set(expected)-set(mapping))};unexpected={len(set(mapping)-set(expected))}")
    # Validate every acquired cell before producing a descriptive headline.
    bindings=[]; phase_edges=None; validated_cells={}
    for episode,row in expected.items():
        validated_cells[episode]=load_cell(mapping[episode],row,core_units=core_units)
        report,_,arrays=validated_cells[episode]
        if phase_edges is None: phase_edges=arrays["phase_edges_ticks"]
        require(np.array_equal(phase_edges,arrays["phase_edges_ticks"]), "PFC_QUALITY_COHORT_PHASE_CLOCK_MISMATCH")
        bindings.append(dict(episode_id=episode,readout=str(mapping[episode]),report_sha256=sha256_file(mapping[episode]/"report.json"),
                config_sha256=report["config_sha256"],source_delivery_sha256=report["source_delivery_sha256"],
                source_archive=report["source_archive"],outputs=report["outputs"]))
    pairings, initial_inputs, external_inputs = [], [], []
    for circuit in CIRCUITS:
        circuit_pairs = []
        for block in BLOCKS:
            rows = [expected[f"pfc_working_memory_circuit_{circuit:03d}_block_{block:03d}_condition_{k:02d}"]
                    for k in range(10)]
            reports = [validated_cells[row["identity"]["episode_id"]][0] for row in rows]
            circuit_pairs.append(paired_block_binding(rows, reports))
            initial_row, external_rows = input_statistics(reports[0])
            initial_inputs.append(initial_row)
            external_inputs.extend(external_rows)
        require(len({row["initial_voltage_sha256"] for row in circuit_pairs}) == 3
                and len({row["final_rng_sha256"] for row in circuit_pairs}) == 3,
                "PFC_QUALITY_REALIZED_BLOCK_INPUT_REUSED")
        pairings.extend(circuit_pairs)
    summaries=[];paired=[];contrast_summary=[];population=[];angular=[];unit_arrays={};field_arrays={}
    for circuit in CIRCUITS:
        phase_values=np.zeros((3,len(phase_edges)-1,len(RATE_FIELDS),10))
        for condition_index,condition in enumerate(CONDITIONS):
            cells=[expected[f"pfc_working_memory_circuit_{circuit:03d}_block_{b:03d}_condition_{condition_index:02d}"] for b in BLOCKS]
            loaded=[validated_cells.pop(row["identity"]["episode_id"]) for row in cells]
            reports,phases,arrays=zip(*loaded)
            for row,report in zip(cells,reports):
                with (mapping[row["identity"]["episode_id"]]/"temporal_responses.csv").open(newline="") as stream:
                    temporal=list(csv.DictReader(stream))
                require(len(temporal)==1000 and all(
                    item["episode_id"]==row["identity"]["episode_id"]
                    and item["condition_id"]==condition
                    and int(item["circuit_index"])==circuit
                    and int(item["block_index"])==row["identity"]["block_index"] for item in temporal),
                    "PFC_QUALITY_TEMPORAL_IDENTITY_CHANGED")
                require(sum(int(item["excitatory_spikes"])+int(item["inhibitory_spikes"]) for item in temporal)
                        == report["events"]-report["terminal_tick_events"],
                        "PFC_QUALITY_TEMPORAL_EVENT_CONSERVATION_FAILED")
                angular.extend(dict(circuit_index=circuit,block_index=row["identity"]["block_index"],
                    episode_id=row["identity"]["episode_id"],condition_id=condition,dt_ms=.0125,
                    lag_ms=item["lag_ticks"]*.0125,**item) for item in lagged_angular_displacement(
                        temporal,phase_edges,bin_ticks=400,lag_ticks=[400,800,8000]))
            for phase in range(len(phase_edges)-1):
                for f,field in enumerate(RATE_FIELDS):
                    values=[float(rows[phase][field]) for rows in phases]
                    phase_values[:,phase,f,condition_index]=values
                    summaries.append(dict(circuit_index=circuit,condition_id=condition,phase_index=phase,metric=field,
                        start_tick=int(phase_edges[phase]),stop_tick=int(phase_edges[phase+1]),**descriptive(values)))
                errors=[float(rows[phase]["circular_cue_error_deg"]) if rows[phase]["circular_cue_error_deg"] else None for rows in phases]
                # Arithmetic means of signed angles are not a circular statistic.
                summaries.append(dict(circuit_index=circuit,condition_id=condition,phase_index=phase,metric="absolute_circular_cue_error_deg",
                    start_tick=int(phase_edges[phase]),stop_tick=int(phase_edges[phase+1]),**descriptive([abs(v) if v is not None else None for v in errors])))
            fields=np.stack([a["population_field_counts"] for a in arrays])
            temporal_cov,between_cov=covariance_axes(fields)
            key=f"circuit_{circuit:03d}_condition_{condition_index:02d}"
            field_arrays[key+"_temporal_covariance_counts2"]=temporal_cov
            field_arrays[key+"_between_block_matched_time_covariance_counts2"]=between_cov
            unit_phase=np.stack([a["phase_unit_counts"] for a in arrays])
            unit_mean=np.stack([a["temporal_unit_mean_counts"] for a in arrays])
            unit_var=np.stack([a["temporal_unit_variance_counts"] for a in arrays])
            unit_arrays[key+"_temporal_mean_count_per_unit"]=unit_mean.mean(axis=0)
            unit_arrays[key+"_mean_temporal_unit_variance_counts2"]=unit_var.mean(axis=0)
            unit_arrays[key+"_between_block_episode_mean_unit_variance_counts2"]=unit_mean.var(axis=0,ddof=1)
            unit_arrays[key+"_between_block_phase_unit_variance_counts2"]=unit_phase.var(axis=0,ddof=1)
            matched = None
            if core_units == 100000:
                events = [archived_events(mapping[row["identity"]["episode_id"]], report)
                          for row, report in zip(cells, reports)]
                matched = matched_time_unit_statistics(events, unit_phase, fields, core_units=core_units, n_e=n_e)
                unit_arrays[key+"_between_block_matched_time_unit_variance_counts2"] = matched["unit_variance"]
            for name,groups,lo,hi in (("excitatory",range(16),0,n_e),("inhibitory",[16],n_e,core_units),("all_units",range(17),0,core_units)):
                groups=list(groups);size=hi-lo
                temporal=float(temporal_cov[np.ix_(groups,groups)].sum()/size**2)
                between=float(between_cov[np.ix_(groups,groups)].sum()/size**2)
                diagonal=float(unit_var[:,lo:hi].mean(axis=0).sum()/size**2)
                mean_population=fields[:,:,groups].sum(axis=2)/size
                np.testing.assert_allclose(temporal,mean_population.var(axis=1,ddof=1).mean(),rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(between,mean_population.var(axis=0,ddof=1).mean(),rtol=1e-12,atol=1e-12)
                population.append(dict(circuit_index=circuit,condition_id=condition,population=name,units=size,blocks=3,time_bins=1000,
                    mean_count_per_unit=float(unit_mean[:,lo:hi].mean()),mean_temporal_single_unit_variance_counts2=float(unit_var[:,lo:hi].mean()),
                    temporal_population_mean_variance_counts2=temporal,temporal_diagonal_contribution_counts2=diagonal,
                    temporal_ordered_offdiagonal_contribution_counts2=temporal-diagonal,
                    between_block_matched_time_population_mean_variance_counts2=between,
                    between_block_episode_mean_population_variance_counts2=float(mean_population.mean(axis=1).var(ddof=1)),
                    **(matched["population"][name] if matched is not None else {
                        "between_block_matched_time_mean_single_unit_variance_counts2": None,
                        "between_block_matched_time_diagonal_contribution_counts2": None,
                        "between_block_matched_time_ordered_offdiagonal_contribution_counts2": None})))
        cue,did=paired_contrasts(phase_values)
        for label,values,conditions in (("cue_minus_omission",cue,CONDITIONS[:5]),("distractor_difference_of_differences",did,CONDITIONS[1:5])):
            for p in range(values.shape[1]):
                for f,field in enumerate(RATE_FIELDS):
                    for k,condition in enumerate(conditions):
                        for b,block in enumerate(BLOCKS):
                            paired.append(dict(circuit_index=circuit,block_index=block,phase_index=p,contrast=label,condition_id=condition,metric=field,value=float(values[b,p,f,k])))
                        contrast_summary.append(dict(circuit_index=circuit,phase_index=p,contrast=label,condition_id=condition,metric=field,**descriptive(values[:,p,f,k])))
    output.mkdir(parents=True)
    write_rows(output/"phase_block_summary.csv",summaries)
    write_rows(output/"paired_block_contrasts.csv",paired)
    write_rows(output/"paired_contrast_summary.csv",contrast_summary)
    write_rows(output/"population_variance_summary.csv",population)
    write_rows(output/"initial_input_statistics.csv",initial_inputs)
    write_rows(output/"external_input_statistics.csv",external_inputs)
    write_rows(output/"lagged_angular_displacement.csv",angular)
    np.savez(output/"unit_variance_statistics.npz",**unit_arrays)
    np.savez(output/"population_covariance_statistics.npz",**field_arrays)
    names=("phase_block_summary.csv","paired_block_contrasts.csv","paired_contrast_summary.csv","population_variance_summary.csv",
            "unit_variance_statistics.npz","population_covariance_statistics.npz",
            "initial_input_statistics.csv","external_input_statistics.csv","lagged_angular_displacement.csv")
    report=dict(schema=SCHEMA,status="COMPLETE_REFERENCE_QUALITY_DESCRIPTIVE_COHORT",complete_episodes=150,
                circuits=list(CIRCUITS),blocks=list(BLOCKS),conditions=list(CONDITIONS),core_units=core_units,duration_ticks=400000,dt_ms=.0125,
                phase_edges_ticks=phase_edges.tolist(),preparation_sha256=sha256_file(preparation/"execution.json"),
                campaign_sha256=plan["campaign_sha256"],parameter_design_sha256=plan["parameter_design_sha256"],settings_sha256=plan["settings_sha256"],
                bindings=bindings,source_qualification_granted=False,scientific_claim_qualified=False,production_episodes=0,
                realized_paired_blocks=pairings,
                input_statistics_scope="15 joint initial/external blocks; paired conditions counted once; ideal IID sampler moment diagnostics only",
                input_statistics_inference="no p-values, confidence intervals, acceptance thresholds or source qualification; external rate is per afferent, not neural firing rate",
                uncertainty="descriptive block mean/sample SD only; no confidence intervals, hypothesis tests or FDR",
                temporal_sample_axis="5ms_time_bins_within_episode",
                between_block_sample_axis="three_joint_initial_external_histories_within_each_circuit; matched_time_population_covariance",
                unit_between_block_scope="phase_count_episode_mean_and_matched5ms_time_bin_variance; full events reused for per-unit diagonal",
                angular_displacement_scope="principal increments of saved5ms population vectors at5/10/100ms lags; full phase-contained bins only; undefined pairs withhold increments; temporal pairs are dependent; no unwrapping or diffusion fit",
                undefined_angle_rule="retain all three blocks; withhold mean/SD when any angle is undefined; no arithmetic signed-angle averaging",
                outputs={name:dict(path=name,sha256=sha256_file(output/name)) for name in names})
    freeze_json(output/"report.json",report)
    return report


def aggregate(preparation: Path, readouts: Path, output: Path) -> dict:
    """Production entry point requires the unchanged 100,000-unit pilot."""
    return _aggregate(preparation, readouts, output, core_units=100000, n_e=80000)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("preparation","readouts","output"): parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        report=aggregate(args.preparation,args.readouts,args.output)
    except (ValueError,KeyError,FileNotFoundError,AssertionError) as error:
        print(json.dumps(dict(status="INCOMPLETE_OR_INVALID_REFERENCE_QUALITY_COHORT",error=str(error),authoritative_aggregate_withheld=True,production_episodes=0)))
        raise SystemExit(1) from error
    print(json.dumps({k:report[k] for k in ("status","complete_episodes","production_episodes")}))


if __name__ == "__main__":
    main()
