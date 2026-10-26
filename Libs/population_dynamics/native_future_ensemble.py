"""Describe distinct native futures conditional on one fixed microscopic prefix.

The observed-history posterior of a learned model is a different conditional
law. These outputs do not qualify that law, matched fidelity or a cost frontier.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

import numpy as np
import torch
import yaml

from .native_renewal_data import NativeRenewalEpisode
from .nest_gpu_network import STATE_FIELDS
from .provenance import sha256_file
from .spatial_activity import Activity, analyze as analyze_activity
from .trajectory_sample_metrics import unit_count_metrics
from .trajectory_scores import energy_score, field_pairs, variogram_score

CORE_UNITS = 100000
FIELD_COUNT = 128
DRAWS = 16
EPISODE_MS = 6000.
DT_MS = .1
BIN_MS = 5.
HORIZONS_MS = [20., 80., 500.]
BOUNDARY_TICK = 10000
RESET_TICK = 10015


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(path.read_text())


def _arrays(path, *, report=None):
    if report is not None:
        _require(report.get("artifacts", {}).get(path.name) == sha256_file(path),
                 "native artifact binding changed: " + path.name)
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _exact_arrays(actual, expected, message):
    _require(set(actual) == set(expected) and all(actual[name].dtype == expected[name].dtype
             and np.array_equal(actual[name], expected[name]) for name in expected), message)


def _peak_rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))


def _clock(data):
    _require(data.dt_ms == DT_MS and data.bin_ms == BIN_MS
             and data.duration_ticks * data.dt_ms == EPISODE_MS
             and len(data.unit_ids) == CORE_UNITS and len(data.field_ids) == FIELD_COUNT,
             "complete native 6s,100000-unit,128-field,5ms clock required")


def _state(path, units, *, report=None):
    arrays = _arrays(path, report=report)
    _require({"unit_ids", *STATE_FIELDS, "P11ex", "P11in"} <= set(arrays)
             and np.array_equal(arrays["unit_ids"], units)
             and all(arrays[name].shape == units.shape and np.isfinite(arrays[name]).all()
                     for name in (*STATE_FIELDS, "P11ex", "P11in")),
             "complete finite identified committed state required")
    return arrays


def _noise_seeds(report, boundary_tick, units):
    config = report.get("config", {})
    noise = config.get("future_background_noise")
    _require(isinstance(noise, dict) and noise.get("boundary_time_ms") == boundary_tick * DT_MS,
             "native future noise boundary differs")
    seeds = noise.get("seeds_per_population")
    _require(isinstance(seeds, list) and len(seeds) == len(set(seeds)) == 8
             and all(type(seed) is int and 0 < seed < 2**31 for seed in seeds),
             "eight distinct positive native future seeds required")
    resets = report.get("future_background_noise")
    _require(isinstance(resets, list) and len(resets) == 8
             and len({row.get("source_node") for row in resets}) == 8
             and [row.get("future_seed") for row in resets] == seeds
             and sum(row.get("outgoing_edges_verified", 0) for row in resets) == units
             and all(row.get("boundary_tick") == boundary_tick and row.get("reset_tick") == RESET_TICK
                     and row.get("native_delay_ticks") == RESET_TICK - boundary_tick + 2
                     and row.get("committed_emissions_through_tick") == boundary_tick
                     and row.get("last_committed_arrival_tick") == RESET_TICK
                     and row.get("first_resampled_arrival_tick") == RESET_TICK + 1
                     and row.get("state_and_queues_reinitialized") is False for row in resets),
             "complete future-only native background reset receipts required")
    return tuple(seeds)


def _protocol(config):
    _require(config.get("joint_samples") == DRAWS and config.get("horizons_ms") == HORIZONS_MS
             and config.get("formal_evaluation") is False and config.get("finite_ensemble_corrections") is True
             and config.get("unit_metric_dtype") == "float32" and config.get("field_metric_dtype") == "float64"
             and config.get("field_scale") == "fitting_block_pooled_standard_deviation_of_per_unit_hz"
             and config.get("matched_output") == "all_identified_unit_counts_and_identical_field_projection"
             and config.get("causal_history") == "all_native_bins_before_origin"
             and config.get("temporal_lags_bins") == [1, 4, 16]
             and config.get("refractory_support_windows_bins") == [1, 2, 4],
             "frozen development sampled-count measurement protocol required")


def analyze_future_ensemble(graph: Path, reference_episode: Path, reference_fields: Path,
                            branches: list[Path], fitting_scale_path: Path, config_path: Path,
                            output: Path, boundary_tick: int = BOUNDARY_TICK, *,
                            committed_state_reference: Path):
    """Convert and score sixteen complete source runs without training or sampling.

    The caller must bind ``committed_state_reference`` to the qualified unchanged
    native replay. Its two snapshots are read here, never added to old reports.
    """
    started = time.perf_counter()
    graph, reference_episode, reference_fields = map(Path, (graph, reference_episode, reference_fields))
    fitting_scale_path, config_path, output = map(Path, (fitting_scale_path, config_path, output))
    committed_state_reference = Path(committed_state_reference)
    branches = list(map(Path, branches))
    _require(type(boundary_tick) is int and boundary_tick == BOUNDARY_TICK,
             "fixed1000ms observed boundary required")
    _require(len(branches) == DRAWS and len({path.resolve() for path in branches}) == DRAWS,
             "sixteen distinct complete native branch paths required")
    config = yaml.safe_load(config_path.read_text())
    _protocol(config)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="RUNNING", scope="fixed_microstate_conditional_development",
                  scientific_claim_qualified=False, matched_fidelity=False, same_hardware=False,
                  end_to_end=False, speedup=False, training_performed=False,
                  mechanistic_simulation_performed=False, hidden_state_posterior_qualified=False,
                  configuration=config, boundary_tick=boundary_tick, reset_tick=RESET_TICK,
                  first_resampled_arrival_tick=RESET_TICK + 1,
                  conditional_scope="one fixed microscopic initial state and committed native event/queue prefix; learned observed-history posterior differs",
                  observation_boundary="history excludes tick10000; native branch fixes the post-update state at10000 and committed arrivals through10015",
                  cost_scope="analysis function only; native simulation/import/staging, process imports and final analysis.json write excluded",
                  peak_process_rss_scope="cumulative process-lifetime resident-memory high-water mark",
                  hardware=dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                                host=platform.node(), cpu=platform.processor(), device="cpu",
                                cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads(),
                                slurm_job_id=os.environ.get("SLURM_JOB_ID")),
                  branches=[], artifacts={})
    def save():
        (output / "analysis.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    save()
    samples = None
    try:
        graph_digest = sha256_file(graph / "manifest.json")
        reference_report = _json(reference_episode / "report.json")
        _require(reference_report.get("status") == "EXECUTED"
                 and reference_report.get("graph_manifest_sha256") == graph_digest,
                 "bound complete original native reference required")
        reference = NativeRenewalEpisode(reference_episode, reference_fields)
        _clock(reference)
        initial = _arrays(reference_episode / "initial_parameters.npz", report=reference_report)
        units, mapping, names = reference.unit_ids, reference.unit_field_index, reference.field_ids
        states = {tick: _state(committed_state_reference / f"state_observation_{tick:09d}.npz", units)
                  for tick in (boundary_tick, RESET_TICK)}
        record["reference_binding"] = dict(reference.binding, graph_manifest_sha256=graph_digest,
            fitting_scale_sha256=sha256_file(fitting_scale_path), config_sha256=sha256_file(config_path),
            committed_state_sha256={str(tick): sha256_file(committed_state_reference / f"state_observation_{tick:09d}.npz")
                                    for tick in states})
        with np.load(fitting_scale_path, allow_pickle=False) as scale:
            mean, sd, denominators = scale["mean_hz"], scale["standard_deviation_hz"], scale["unit_count"]
            _require(np.array_equal(scale["field_ids"], names)
                     and np.array_equal(denominators, np.bincount(mapping, minlength=len(names)))
                     and mean.shape == sd.shape == denominators.shape == (FIELD_COUNT,)
                     and np.isfinite(mean).all() and np.isfinite(sd).all() and np.all(sd > 0),
                     "bound fitting scale and source field denominators required")
        definitions = [json.loads(str(name)) for name in names]
        pairs = field_pairs([row[0] for row in definitions], [row[2] for row in definitions], [row[1] for row in definitions])
        longest = int(max(config["horizons_ms"]) / BIN_MS)
        stop_tick = boundary_tick + longest * reference.bin_ticks
        conversion_seconds = 0.
        conversion_started = time.perf_counter()
        observed, control = reference.window(boundary_tick, stop_tick)
        observed = observed.astype(np.int64)
        conversion_seconds += time.perf_counter() - conversion_started
        sample_path = output / "sampled_unit_counts.npy"
        samples = np.lib.format.open_memmap(sample_path, mode="w+", dtype=np.int64,
                                           shape=(DRAWS, longest, len(units)))
        new_source, seen_seeds, consumed_seeds = None, set(), set()
        activity_seconds = validation_seconds = 0.
        prefix = reference.event_ticks <= RESET_TICK
        for index, branch in enumerate(branches):
            validate_started = time.perf_counter()
            report = _json(branch / "report.json")
            _require(report.get("status") == "EXECUTED" and report.get("duration_ms") == EPISODE_MS
                     and report.get("cortical_neurons") == CORE_UNITS and report.get("graph_manifest_sha256") == graph_digest,
                     "complete source-bound native branch required")
            seeds = _noise_seeds(report, boundary_tick, len(units))
            _require(seeds not in seen_seeds, "duplicate native future seed vector")
            _require(not consumed_seeds.intersection(seeds), "native future streams reuse a seed across draws")
            seen_seeds.add(seeds)
            consumed_seeds.update(seeds)
            _exact_arrays(_arrays(branch / "initial_parameters.npz", report=report), initial,
                          "native branch initial parameters differ from original")
            for tick, expected in states.items():
                _exact_arrays(_state(branch / f"state_observation_{tick:09d}.npz", units, report=report), expected,
                              "native committed full-unit state differs")
            validation_seconds += time.perf_counter() - validate_started
            field_output = output / "branch_fields" / f"draw_{index:02d}"
            activity_started = time.perf_counter()
            field_report = analyze_activity(graph, branch, field_output,
                                            Activity(bin_ms=BIN_MS, cells_per_axis=4, start_ms=0.))
            activity_seconds += time.perf_counter() - activity_started
            _require(field_report.get("status") == "DESCRIBED", "complete native branch field projection required")
            conversion_started = time.perf_counter()
            actual = NativeRenewalEpisode(branch, field_output, expected_unit_ids=units,
                                           expected_unit_field_index=mapping)
            _clock(actual)
            source = actual.source_identity
            _require(all(source[key] == reference.source_identity[key] for key in source if key != "native_library_sha256")
                     and source["native_library_sha256"] != reference.source_identity["native_library_sha256"],
                     "future native library identity must remain explicit and source compatible")
            if new_source is None:
                new_source = source
            _require(source == new_source, "mixed future native libraries or source identities")
            _require(np.array_equal(actual.field_ids, names)
                     and np.array_equal(actual.minimum_interval_ticks, reference.minimum_interval_ticks),
                     "native field identities or refractory law differ")
            committed = actual.event_ticks <= RESET_TICK
            _require(np.array_equal(actual.event_ticks[committed], reference.event_ticks[prefix])
                     and np.array_equal(actual.event_units[committed], reference.event_units[prefix]),
                     "native committed event prefix differs")
            _require(np.array_equal(actual.pulse_ticks, reference.pulse_ticks), "native TC control program changed")
            values, actual_control = actual.window(boundary_tick, stop_tick)
            _require(values.shape == (longest, len(units)) and np.isfinite(values).all()
                     and np.all(values >= 0) and np.array_equal(values, np.floor(values))
                     and np.array_equal(actual_control, control), "complete integer native count/control window required")
            samples[index] = values
            conversion_seconds += time.perf_counter() - conversion_started
            record["branches"].append(dict(draw=index, native_report_sha256=sha256_file(branch / "report.json"),
                source_binding=actual.binding, future_seeds=list(seeds), complete_unit_count_values=int(values.size),
                native_elapsed_seconds=report.get("elapsed_seconds"), native_import_seconds=report.get("import_seconds"),
                native_simulation_seconds=report.get("simulation_seconds")))
            del actual, values, actual_control
        projection_started = time.perf_counter()
        tensor = torch.as_tensor(samples)
        unit_mapping = torch.as_tensor(mapping)
        projected = torch.zeros((DRAWS, longest, len(names)), dtype=torch.int64)
        projected.index_add_(-1, unit_mapping, tensor)
        truth_fields = torch.zeros((longest, len(names)), dtype=torch.int64)
        truth_fields.index_add_(-1, unit_mapping, torch.as_tensor(observed))
        raw_fields, raw_truth = projected.numpy(), truth_fields.numpy()
        projection_seconds = time.perf_counter() - projection_started
        scoring_started = time.perf_counter()
        scaled = (raw_fields / (denominators * BIN_MS / 1000) - mean) / sd
        truth = (raw_truth / (denominators * BIN_MS / 1000) - mean) / sd
        rows = []
        for ms in config["horizons_ms"]:
            horizon = int(ms / BIN_MS)
            row = dict(method="native_fixed_microstate", origin_ms=boundary_tick * DT_MS, horizon_ms=ms,
                       draws=DRAWS, source_unit_count=len(units), field_count=len(names),
                       generated_horizon_ms=max(config["horizons_ms"]),
                       energy_score=energy_score(scaled[:, :horizon], truth[:horizon]),
                       **unit_count_metrics(tensor[:, :horizon], torch.as_tensor(observed[:horizon]),
                                            reference.minimum_interval_ticks, reference.bin_ticks,
                                            config["refractory_support_windows_bins"]))
            for label, pair in pairs.items():
                row["variogram_" + label] = variogram_score(scaled[:, :horizon], truth[:horizon], pair[:, 0], pair[:, 1])
            for lag in config["temporal_lags_bins"]:
                row[f"variogram_temporal_lag_{lag}"] = (variogram_score(scaled[:, :horizon], truth[:horizon],
                    np.arange(len(names)), np.arange(len(names)), lag) if lag < horizon else None)
            rows.append(row)
        scoring_seconds = time.perf_counter() - scoring_started
        serialization_started = time.perf_counter()
        samples.flush()
        np.savez_compressed(output / "ensemble_fields.npz", sampled_field_counts=raw_fields,
            observed_field_counts=raw_truth, observed_unit_counts=observed, unit_ids=units,
            unit_field_index=mapping, field_ids=names, unit_count=denominators, stimulus=control,
            boundary_ready_in_ticks=reference.ready_at(boundary_tick))
        with (output / "scores.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        serialization_seconds = time.perf_counter() - serialization_started
        hash_started = time.perf_counter()
        artifacts = {name: sha256_file(output / name) for name in ("sampled_unit_counts.npy", "ensemble_fields.npz", "scores.csv")}
        hash_seconds = time.perf_counter() - hash_started
        record.update(status="DESCRIBED_FIXED_MICROSTATE_DEVELOPMENT", reference_unit_count=len(units),
            field_count=len(names), draws=DRAWS, distinct_future_seed_vectors=len(seen_seeds),
            native_initial_noise_blocks=1, score_rows=len(rows), generated_bins=longest,
            generated_horizon_ms=max(config["horizons_ms"]), generated_unit_count_values=int(samples.size),
            full_unit_counts_saved=True, count_dtype=str(samples.dtype), tensor_dtype=str(tensor.dtype),
            tensor_device=str(tensor.device), field_metric_dtype=str(scaled.dtype),
            reference_source_identity=reference.source_identity, branch_source_identity=new_source,
            all_committed_events_and_states_exact=True, all_tc_programs_exact=True,
            validation_seconds=validation_seconds, activity_projection_seconds=activity_seconds,
            data_conversion_seconds=conversion_seconds, field_projection_seconds=projection_seconds,
            scoring_seconds=scoring_seconds, serialization_seconds=serialization_seconds, hash_seconds=hash_seconds,
            peak_process_rss_bytes=_peak_rss(), artifacts=artifacts,
            implementation_sha256=sha256_file(Path(__file__)))
        record["function_seconds"] = time.perf_counter() - started
        save()
        return record
    except BaseException as error:
        record.update(status="FAILED", error=str(error), error_type=type(error).__name__,
                      traceback=traceback.format_exc(), completed_branches=len(record["branches"]),
                      partial_unit_counts_authoritative=False, function_seconds=time.perf_counter() - started)
        if samples is not None:
            samples.flush()
        save()
        raise


def replay_delivery(delivery: Path, fitting_scale: Path, output: Path):
    """Verify an archived ensemble from native events without new simulation.

    Integer event binning and field projection are recomputed with NumPy.
    The shared metric definitions are applied to all saved draws and horizons.
    The source delivery is immutable; the result goes to a separate new file.
    """
    started = time.perf_counter()
    delivery, fitting_scale, output = map(Path, (delivery, fitting_scale, output))
    _require(not output.exists() and not output.resolve().is_relative_to(delivery.resolve()),
             "new replay output outside the immutable delivery required")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status="RUNNING", delivery_sha256=sha256_file(delivery / "delivery.json"),
                  simulation_repeated=False, training_performed=False,
                  scientific_claim_qualified=False, matched_fidelity_qualified=False,
                  same_hardware_comparison_qualified=False, speedup_qualified=False)
    try:
        proof = _json(delivery / "delivery.json")
        _require(proof.get("status") == "PASS" and proof.get("exit_code") == 0
                 and proof.get("complete_native_executions") == DRAWS
                 and proof.get("generated_unit_count_values") == DRAWS * 100 * CORE_UNITS
                 and proof.get("full_unit_counts_saved") is True,
                 "complete successful sixteen-future delivery required")
        rows = proof.get("files")
        _require(isinstance(rows, list) and rows and len({row["path"] for row in rows}) == len(rows),
                 "unique archive manifest required")
        archive_bytes = 0
        for row in rows:
            relative = Path(row["path"])
            _require(not relative.is_absolute() and ".." not in relative.parts
                     and type(row["bytes"]) is int and row["bytes"] >= 0,
                     "valid relative archive path and byte length required")
            path = delivery / relative
            _require(path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"],
                     "archived artifact changed: " + str(relative))
            archive_bytes += row["bytes"]
        executions = proof.get("native_executions", [])
        _require(len(executions) == DRAWS and [row.get("draw") for row in executions] == list(range(DRAWS))
                 and all(row.get("exit_code") == 0 for row in executions),
                 "all sixteen ordered native executions required")
        for name in ("process_seconds", "native_function_seconds", "graph_import_seconds",
                     "simulation_seconds", "calibration_seconds"):
            _require(all(type(row.get(name)) in (int, float) and np.isfinite(row[name]) and row[name] >= 0
                         for row in executions), "finite measured native cost required: " + name)
        _require(proof.get("native_process_seconds") == sum(row["process_seconds"] for row in executions)
                 and proof.get("native_function_seconds") == sum(row["native_function_seconds"] for row in executions)
                 and proof.get("simulated_biological_ms") == DRAWS * EPISODE_MS
                 and proof.get("target_output_biological_ms") == DRAWS * 500.,
                 "complete native cost denominator differs")
        for name in ("pipeline_before_archive_seconds", "archive_seconds", "task_through_archive_seconds"):
            _require(type(proof.get(name)) in (int, float) and np.isfinite(proof[name]) and proof[name] >= 0,
                     "finite measured pipeline cost required: " + name)
        _require(proof["task_through_archive_seconds"] >= proof["pipeline_before_archive_seconds"] + proof["archive_seconds"],
                 "archive cost is absent from pipeline total")
        analysis_root = delivery / "analysis"
        analysis = _json(analysis_root / "analysis.json")
        _require(analysis.get("status") == "DESCRIBED_FIXED_MICROSTATE_DEVELOPMENT"
                 and analysis.get("draws") == DRAWS and analysis.get("generated_bins") == 100
                 and analysis.get("reference_unit_count") == CORE_UNITS
                 and analysis.get("field_count") == FIELD_COUNT
                 and analysis.get("all_committed_events_and_states_exact") is True
                 and analysis.get("all_tc_programs_exact") is True,
                 "complete identified native ensemble analysis required")
        _protocol(analysis["configuration"])
        _require(sha256_file(fitting_scale) == analysis["reference_binding"]["fitting_scale_sha256"],
                 "original fitting-only field scale required")
        reference = NativeRenewalEpisode(delivery / "reference", delivery / "reference_fields")
        _clock(reference)
        units, mapping, names = reference.unit_ids, reference.unit_field_index, reference.field_ids
        _require(all(analysis["reference_binding"].get(key) == value for key, value in reference.binding.items()),
                 "original reference binding differs")
        initial = _arrays(delivery / "reference/initial_parameters.npz")
        states = {tick: _state(delivery / "qualified_boundary" / f"state_observation_{tick:09d}.npz", units)
                  for tick in (BOUNDARY_TICK, RESET_TICK)}
        _require(all(sha256_file(delivery / "qualified_boundary" / f"state_observation_{tick:09d}.npz")
                     == analysis["reference_binding"]["committed_state_sha256"][str(tick)] for tick in states),
                 "qualified committed-state binding differs")
        sample_path = analysis_root / "sampled_unit_counts.npy"
        samples = np.load(sample_path, mmap_mode="r", allow_pickle=False)
        _require(samples.shape == (DRAWS, 100, CORE_UNITS) and samples.dtype == np.int64,
                 "complete saved int64 ensemble required")
        arrays = _arrays(analysis_root / "ensemble_fields.npz")
        _require(np.array_equal(arrays["unit_ids"], units)
                 and np.array_equal(arrays["unit_field_index"], mapping)
                 and np.array_equal(arrays["field_ids"], names)
                 and np.array_equal(arrays["boundary_ready_in_ticks"], reference.ready_at(BOUNDARY_TICK)),
                 "saved observation map or causal readiness differs")
        denominators = np.bincount(mapping, minlength=FIELD_COUNT)
        _require(np.array_equal(arrays["unit_count"], denominators), "complete field denominators differ")
        with np.load(fitting_scale, allow_pickle=False) as scale:
            mean, sd = scale["mean_hz"], scale["standard_deviation_hz"]
            _require(np.array_equal(scale["field_ids"], names)
                     and np.array_equal(scale["unit_count"], denominators)
                     and mean.shape == sd.shape == (FIELD_COUNT,)
                     and np.isfinite(mean).all() and np.isfinite(sd).all() and np.all(sd > 0),
                     "finite source-matched fitting scale required")

        def bin_events(native):
            values = np.zeros((100, CORE_UNITS), dtype=np.int64)
            left, right = np.searchsorted(native.event_ticks, [BOUNDARY_TICK, BOUNDARY_TICK + 100 * native.bin_ticks])
            bins = (native.event_ticks[left:right] - BOUNDARY_TICK) // native.bin_ticks
            np.add.at(values, (bins, native.event_units[left:right]), 1)
            return values

        def project(values):
            fields = np.zeros((100, FIELD_COUNT), dtype=np.int64)
            np.add.at(fields.T, mapping, values.T)
            return fields

        observed = bin_events(reference)
        _, control = reference.window(BOUNDARY_TICK, BOUNDARY_TICK + 100 * reference.bin_ticks)
        _require(np.array_equal(observed, arrays["observed_unit_counts"])
                 and np.array_equal(control, arrays["stimulus"]), "saved truth or control differs from native events")
        truth_fields = project(observed)
        _require(np.array_equal(truth_fields, arrays["observed_field_counts"]), "saved truth field projection differs")
        fields = np.zeros((DRAWS, 100, FIELD_COUNT), dtype=np.int64)
        seen_seeds = set()
        prefix = reference.event_ticks <= RESET_TICK
        for index in range(DRAWS):
            branch = delivery / "executions" / f"draw_{index:02d}" / "native"
            native_report = _json(branch / "report.json")
            for measured, source in (("native_function_seconds", "elapsed_seconds"),
                                     ("graph_import_seconds", "import_seconds"),
                                     ("simulation_seconds", "simulation_seconds"),
                                     ("calibration_seconds", "calibration_seconds")):
                _require(executions[index][measured] == native_report.get(source),
                         "native cost differs from source report: " + measured)
            native = NativeRenewalEpisode(branch, analysis_root / "branch_fields" / f"draw_{index:02d}",
                                          expected_unit_ids=units, expected_unit_field_index=mapping,
                                          expected_source=analysis["branch_source_identity"])
            _clock(native)
            seeds = _noise_seeds(native_report, BOUNDARY_TICK, CORE_UNITS)
            _require(not seen_seeds.intersection(seeds) and list(seeds) == executions[index]["seeds_per_population"],
                     "repeated or changed future seeds")
            seen_seeds.update(seeds)
            _require(native.binding == analysis["branches"][index]["source_binding"], "branch binding differs")
            _exact_arrays(_arrays(branch / "initial_parameters.npz"), initial, "branch initial parameters differ")
            for tick, expected in states.items():
                _exact_arrays(_state(branch / f"state_observation_{tick:09d}.npz", units), expected,
                              "branch committed state differs")
            committed = native.event_ticks <= RESET_TICK
            _require(np.array_equal(native.event_ticks[committed], reference.event_ticks[prefix])
                     and np.array_equal(native.event_units[committed], reference.event_units[prefix])
                     and np.array_equal(native.pulse_ticks, reference.pulse_ticks),
                     "branch committed events or TC controls differ")
            counts = bin_events(native)
            _require(np.array_equal(counts, samples[index]), "saved unit counts differ from native events")
            fields[index] = project(counts)
        _require(np.array_equal(fields, arrays["sampled_field_counts"]), "saved sampled field projection differs")
        definitions = [json.loads(str(name)) for name in names]
        pairs = field_pairs([row[0] for row in definitions], [row[2] for row in definitions], [row[1] for row in definitions])
        scaled = (fields / (denominators * BIN_MS / 1000) - mean) / sd
        truth = (truth_fields / (denominators * BIN_MS / 1000) - mean) / sd
        with (analysis_root / "scores.csv").open(newline="") as stream:
            scores = list(csv.DictReader(stream))
        _require(len(scores) == len(HORIZONS_MS), "all three score horizons required")
        checked_scores = 0
        # Metrics are read-only; a copy provides a writable tensor backing store.
        tensor = torch.from_numpy(np.array(samples))
        for row, ms in zip(scores, HORIZONS_MS):
            horizon = int(ms / BIN_MS)
            expected = dict(method="native_fixed_microstate", origin_ms=BOUNDARY_TICK * DT_MS, horizon_ms=ms,
                            draws=DRAWS, source_unit_count=CORE_UNITS, field_count=FIELD_COUNT, generated_horizon_ms=500.,
                            energy_score=energy_score(scaled[:, :horizon], truth[:horizon]),
                            **unit_count_metrics(tensor[:, :horizon], torch.as_tensor(observed[:horizon]),
                                                 reference.minimum_interval_ticks, reference.bin_ticks, [1, 2, 4]))
            for label, pair in pairs.items():
                expected["variogram_" + label] = variogram_score(scaled[:, :horizon], truth[:horizon], pair[:, 0], pair[:, 1])
            for lag in [1, 4, 16]:
                expected[f"variogram_temporal_lag_{lag}"] = (variogram_score(scaled[:, :horizon], truth[:horizon],
                    np.arange(FIELD_COUNT), np.arange(FIELD_COUNT), lag) if lag < horizon else None)
            _require(set(row) == set(expected), "score schema differs")
            for name, value in expected.items():
                actual = row[name]
                _require(actual == "" if value is None else actual == value if isinstance(value, str)
                         else np.isfinite(float(actual)) and float(actual) == value,
                         "native replay score differs: " + name)
                checked_scores += 1
        result.update(status="PASS", archive_files_verified=len(rows), archive_bytes_verified=archive_bytes,
                      complete_native_draws=DRAWS, distinct_future_seeds=len(seen_seeds),
                      exact_unit_count_values=int(samples.size), exact_field_count_values=int(fields.size),
                      exact_committed_states_and_events=True, exact_score_cells=checked_scores,
                      score_horizons_ms=HORIZONS_MS, cost_denominator_verified=True,
                      native_process_seconds=proof["native_process_seconds"],
                      pipeline_before_archive_seconds=proof["pipeline_before_archive_seconds"],
                      archive_seconds=proof["archive_seconds"], task_through_archive_seconds=proof["task_through_archive_seconds"],
                      analysis_sha256=sha256_file(analysis_root / "analysis.json"),
                      conditional_scope=analysis["conditional_scope"])
    except Exception as error:
        result.update(status="FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(replay_seconds=time.perf_counter() - started, peak_process_rss_bytes=_peak_rss(),
                      implementation_sha256=sha256_file(Path(__file__)))
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-delivery", type=Path)
    for name in ("graph", "reference-episode", "reference-fields", "branches", "fitting-scale", "config",
                 "output", "committed-state-reference"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.replay_delivery is not None:
        if args.fitting_scale is None or args.output is None:
            parser.error("replay requires --fitting-scale and --output")
        record = replay_delivery(args.replay_delivery, args.fitting_scale, args.output)
        print(json.dumps(record))
        return
    if any(getattr(args, name) is None for name in ("graph", "reference_episode", "reference_fields", "branches",
                                                  "fitting_scale", "config", "output", "committed_state_reference")):
        parser.error("analysis requires all native inputs, configuration and output")
    branches = _json(args.branches)
    _require(isinstance(branches, list) and all(isinstance(path, str) and path for path in branches),
             "JSON list of native branch paths required")
    record = analyze_future_ensemble(args.graph, args.reference_episode, args.reference_fields,
        [Path(path) for path in branches], args.fitting_scale, args.config, args.output,
        committed_state_reference=args.committed_state_reference)
    print(json.dumps({name: value for name, value in record.items() if name not in ("branches", "artifacts")}))


if __name__ == "__main__":
    main()
