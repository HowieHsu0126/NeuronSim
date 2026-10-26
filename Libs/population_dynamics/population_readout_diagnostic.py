"""Posthoc readout diagnostics from authenticated, already generated trajectories.

The manifest contains parent_campaign and corrections directory paths, plus
qualification, native_comparison, and runtime_receipt entries with path/bytes/
sha256. Paths may be relative to the manifest. Optional cpu_threads and
unit_chunk_size control only this posthoc computation. All runtime is on Merlin.

Cached q is the frozen encoder distribution for one future observed episode,
not predictive latent truth. Signed pairwise A/B/C differences within the same
seed/case/origin cancel that common q reference; squared errors do not. Hazard
moments neither decompose refractory count error nor imply a global count ratio.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import torch

from .population_forecast_analysis import _delivery_file
from .population_residual_correction_analysis import (
    BLOCKS, HORIZONS_MS, METHODS, MODELS, ORIGINS_MS, SEEDS, _admit, _delivery, _require,
)
from .population_readout_moments import trajectory_readout_moments
from .population_state import ControlledPopulationState
from .provenance import freeze_json, sha256_file
from .trajectory_forecast_analysis import _write_rows

RUNTIME_KEYS = ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "RH2_CHECKPOINT_HDFS")
FIELDS, CHANNELS, DRAWS, STEPS, EPISODE_BINS, BIN_MS = 128, 4, 16, 100, 1200, 5


def _path(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else base / candidate


def _manifest_file(base: Path, entry: dict, role: str) -> Path:
    _require(isinstance(entry, dict) and set(entry) == {"path", "bytes", "sha256"}
             and isinstance(entry["path"], str) and type(entry["bytes"]) is int and entry["bytes"] > 0,
             "explicit staged file identity required: " + role)
    return _delivery_file(base, {entry["path"]: entry}, entry["path"], role)


def merlin_runtime(receipt: dict) -> dict:
    environment = {key: os.environ.get(key) for key in RUNTIME_KEYS}
    host = platform.node()
    _require(all(isinstance(value, str) and value for value in environment.values())
             and host.split(".")[0] != "mgmt2", "actual Merlin run, trial and checkpoint environment required")
    _require(receipt.get("execution_backend") == "merlin" and receipt.get("environment") == environment
             and receipt.get("hostname") == host, "Merlin runtime receipt differs from the actual run or host")
    return dict(execution_backend="merlin", environment=environment, hostname=host,
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__)


def posterior_window(cache: dict, checkpoint: dict, case_id: str, origin_ms: float):
    """Slice future q after the origin; authenticate the initialization lineage."""
    binding, parent_binding = cache["binding"], checkpoint["binding"]
    _require(binding.get("case_id") == case_id
             and binding.get("parent_checkpoint_sha256") == checkpoint["initialization"]["parent_checkpoint_sha256"]
             and binding.get("model_implementation_sha256") == parent_binding["model_implementation_sha256"]
             and binding.get("bin_ms") == BIN_MS
             and isinstance(binding.get("cohort"), dict)
             and all(key in parent_binding and parent_binding[key] == value for key, value in binding["cohort"].items()),
             "posterior cache initialization-parent lineage or cohort differs")
    mean, variance = (torch.as_tensor(cache[key]) for key in ("mean", "variance"))
    expected = (1, EPISODE_BINS, FIELDS, CHANNELS)
    _require(mean.shape == variance.shape == expected and not mean.is_complex() and not variance.is_complex()
             and torch.isfinite(mean).all() and torch.isfinite(variance).all() and torch.all(variance > 0),
             "complete finite cached posterior mean and positive variance required")
    _require(origin_ms in ORIGINS_MS and origin_ms / BIN_MS == int(origin_ms / BIN_MS), "registered native origin required")
    boundary = int(origin_ms / BIN_MS)
    _require(boundary + STEPS <= EPISODE_BINS, "complete cached future posterior required")
    return mean[0, boundary:boundary+STEPS], variance[0, boundary:boundary+STEPS]


def load_retained_trajectory(path: Path):
    """Read only retained latent arrays; never materialize full_unit_samples."""
    try:
        with np.load(path, allow_pickle=False) as saved:
            _require({"latent_states", "initial_latent_states"}.issubset(saved.files), "retained latent arrays are missing")
            samples = saved["latent_states"]
            initial = saved["initial_latent_states"]
        _require(samples.shape == (DRAWS, STEPS, FIELDS, CHANNELS)
                 and initial.shape == (DRAWS, 1, FIELDS, CHANNELS)
                 and samples.dtype.kind == initial.dtype.kind == "f"
                 and np.isfinite(samples).all() and np.isfinite(initial).all(),
                 "complete finite retained latent arrays required")
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid retained trajectory {path}: {error}") from error
    return torch.from_numpy(samples), torch.from_numpy(initial[:, 0])


def _moment_columns(moments: dict, previous: tuple[str, ...] | None) -> tuple[str, ...]:
    """Require all 19 original and six arithmetic hazard observables."""
    names = tuple(moments)
    _require(len(names) == 25 and (previous is None or names == previous),
             "complete consistent readout moment columns required")
    return names


def origin_tables(moments: dict, native_by_horizon: dict, native_metrics: list[str]):
    """Keep instantaneous, prefix, endpoint and existing native metrics distinct."""
    _require(set(native_by_horizon) == set(HORIZONS_MS), "all three existing native horizons required")
    first = native_by_horizon[20.]
    identity_keys = ("model", "seed", "block", "case_id", "template_id", "origin_ms")
    identity = {key: first[key] for key in identity_keys}
    _require(all(all(row[key] == value for key, value in identity.items()) for row in native_by_horizon.values()),
             "native score rows must describe one exact origin")
    values = {name: np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value, dtype=np.float64)
              for name, value in moments.items()}
    _require(values and all(value.shape == (STEPS,) and np.isfinite(value).all() for value in values.values()),
             "finite per-time moments for the complete retained trajectory required")
    time_rows = [dict(identity, elapsed_ms=(step+1)*BIN_MS, absolute_bin_end_ms=identity["origin_ms"]+(step+1)*BIN_MS,
                      **{name: float(value[step]) for name, value in values.items()}) for step in range(STEPS)]
    rows, metric_names = [], []
    for horizon in HORIZONS_MS:
        steps = int(horizon / BIN_MS)
        native = native_by_horizon[horizon]
        _require(native["horizon_ms"] == horizon, "native horizon identity differs")
        row = dict(identity, horizon_ms=horizon, prefix_bins=steps,
                   native_method=native["method"], native_hardware_device=native["hardware_device"],
                   native_ensemble_artifact=native["ensemble_artifact"])
        for name, value in values.items():
            prefix = "prefix_max_" if name == "decomposition_closure_max_absolute" else "prefix_mean_"
            row[prefix+name] = float(value[:steps].max() if prefix == "prefix_max_" else value[:steps].mean())
            row["endpoint_"+name] = float(value[steps-1])
        for metric in native_metrics:
            value = native[metric]
            _require(value is None or math.isfinite(value), "nonfinite admitted native metric")
            row["native_"+metric] = value
        if not metric_names:
            metric_names = [name for name in row if name.startswith(("prefix_mean_", "prefix_max_", "endpoint_"))]
            metric_names += ["native_"+metric for metric in native_metrics]
        rows.append(row)
    return time_rows, rows, metric_names


def summarize_origins(rows: list[dict], metrics: list[str], *, by_block: bool = False):
    groups, seen = defaultdict(list), set()
    for row in rows:
        identity = tuple(row[key] for key in ("model", "seed", "case_id", "origin_ms", "horizon_ms"))
        _require(identity not in seen, "duplicate origin/horizon in moment aggregation")
        seen.add(identity)
        key = (row["model"], row["seed"], row["horizon_ms"])
        if by_block:
            key += (row["block"],)
        groups[key].append(row)
    result = []
    for key, selected in sorted(groups.items()):
        identity = dict(model=key[0], seed=key[1], horizon_ms=key[2])
        if by_block:
            identity["block"] = key[3]
        for metric in metrics:
            values = [row[metric] for row in selected if row[metric] is not None]
            _require(all(math.isfinite(value) for value in values), "nonfinite moment aggregate input")
            maximum = metric.endswith("decomposition_closure_max_absolute")
            value = (max(values) if maximum else sum(values)/len(values)) if values else None
            result.append(dict(identity, metric=metric, statistic="maximum" if maximum else "mean", value=value,
                available_origins=len(values), structural_na_origins=len(selected)-len(values), total_origins=len(selected),
                metric_scope=("recorded_native_horizon_metric" if metric.startswith("native_")
                              else "instantaneous_endpoint" if metric.startswith("endpoint_") else "prefix_moment_summary")))
    return result


def _comparison_lookup(comparison: dict):
    _require(comparison.get("status") == "COMPLETE_DESCRIPTIVE_NATIVE_RESIDUAL_COMPARISON"
             and comparison.get("training_seeds") == list(SEEDS) and comparison.get("models") == list(MODELS)
             and comparison.get("paired_origins_per_model_seed") == 48
             and comparison.get("score_rows_per_model_seed") == 144 and comparison.get("total_score_rows") == 1296,
             "complete existing native comparison required")
    rows = comparison["consumed"]
    lookup = {(row["seed"], row["model"]): row for row in rows}
    _require(len(lookup) == len(rows) == 9 and set(lookup) == {(seed, model) for seed in SEEDS for model in MODELS},
             "all nine native comparison identities required")
    return lookup


def _same_consumed(actual: dict, expected: dict):
    _require({key: value for key, value in actual.items() if key != "root"}
             == {key: value for key, value in expected.items() if key != "root"},
             "staged delivery differs from the accepted native comparison")


def _qualification_scales(qualification: dict):
    expected = dict(status="COMPLETE_FITTING_INTERNAL_ROLLOUT_QUALIFICATION", training_seeds=list(SEEDS),
        correction_fitting_blocks=list(range(12)), correction_qualification_blocks=[12, 13],
        original_parent_training_blocks=list(range(14)), original_parent_selection_blocks=list(range(14, 18)),
        original_selection_arrays_accessed=False, source_units=100000, posterior_coordinates=512,
        original_initial_and_process_variances=True, clipping_or_stabilization_applied=False)
    _require(all(qualification.get(key) == value for key, value in expected.items()),
             "completed fitting0..11 qualification coordinate scales required")
    scales = {}
    _require(set(qualification["coordinate_scales"]) == {str(seed) for seed in SEEDS}, "all three coordinate scales required")
    for seed in SEEDS:
        value = np.asarray(qualification["coordinate_scales"][str(seed)], dtype=np.float64)
        _require(value.shape == (512,) and np.isfinite(value).all() and np.all(value > 0), "positive fitting-only coordinate scales required")
        scales[seed] = value
    return scales


def _qualification_binding(cell: dict, fit: dict, qualification: dict, digest: str, model: str):
    identity = cell["evaluation"]["correction_identity"]
    selection = identity.get("regularization_selection")
    if selection is None:
        return False
    chosen = qualification["selected_alphas"][model]
    _require(chosen["status"] == "SELECTED" and selection.get("mode") == model
             and selection.get("qualification_sha256") == digest and fit.get("qualification_sha256") == digest
             and selection.get("selected") == chosen and identity["alpha"] == chosen["alpha"]
             and fit["alpha_by_mode"][model] == chosen["alpha"]
             and selection.get("eligible_training_seeds") == list(SEEDS)
             and selection["protocol"]["correction_fitting_blocks"] == list(range(12))
             and selection["protocol"]["correction_qualification_blocks"] == [12, 13],
             "correction regularization selection differs from supplied qualification")
    return True


@torch.no_grad()
def analyze(manifest_path: Path, output: Path):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    required = {"parent_campaign", "corrections", "qualification", "native_comparison", "runtime_receipt"}
    _require(required <= set(manifest) and set(manifest) <= required | {"cpu_threads", "unit_chunk_size"},
             "minimal staged readout input manifest required")
    files = {role: _manifest_file(base, manifest[role], role) for role in
             ("runtime_receipt", "qualification", "native_comparison")}
    runtime = merlin_runtime(json.loads(files["runtime_receipt"].read_text()))
    threads = manifest.get("cpu_threads", 2)
    chunk = manifest.get("unit_chunk_size", 4096)
    _require(type(threads) is int and threads > 0 and type(chunk) is int and chunk > 0, "positive posthoc CPU and chunk resources required")
    torch.set_num_threads(threads)
    runtime.update(torch_threads=threads, unit_chunk_size=chunk)
    qualification = json.loads(files["qualification"].read_text())
    comparison = json.loads(files["native_comparison"].read_text())
    scales, accepted = _qualification_scales(qualification), _comparison_lookup(comparison)
    qualification_digest = manifest["qualification"]["sha256"]
    parent_campaign, corrections = (_path(base, manifest[key]) for key in ("parent_campaign", "corrections"))
    cells, source_cells, binding_available = {}, {}, {}
    for seed in SEEDS:
        parent_cell, correction_cell = parent_campaign / f"seed_{seed}", corrections / f"seed_{seed}"
        parent_delivery, parent_index, parent_digest = _delivery(parent_cell, seed, corrected=False)
        _require(parent_delivery.get("source_units") == 100000 and parent_delivery.get("frozen_parameters_unchanged") is True,
                 "complete frozen-encoder/readout parent delivery required")
        delivery, index, digest = _delivery(correction_cell, seed, corrected=True)
        original = _admit(parent_cell, parent_delivery, parent_index, parent_digest, seed, "original")
        cells[seed, "original"] = original
        source_cells[seed, "original"] = (parent_cell, parent_index)
        fit_path = _delivery_file(correction_cell, index, "provenance/fit.json", "correction fit")
        fit = json.loads(fit_path.read_text())
        _require(fit.get("status") == "FITTED_ALL_NATIVE_RESIDUAL_CORRECTIONS"
                 and index["provenance/fit.json"]["sha256"] == comparison["correction_fit"]["sha256"]
                 and fit["config"] == comparison["correction_fit"]["configuration"], "accepted correction fit identity differs")
        for model in ("current", "history"):
            cells[seed, model] = _admit(correction_cell, delivery, index, digest, seed, model, original)
            source_cells[seed, model] = (correction_cell, index)
            binding_available[f"{seed}/{model}"] = _qualification_binding(cells[seed, model], fit, qualification, qualification_digest, model)
        for model in MODELS:
            _same_consumed(cells[seed, model]["consumed"], accepted[seed, model])
        source_identity = {name: original["evaluation"][name] for name in comparison["source"]}
        _require(source_identity == comparison["source"], "accepted native source identity differs")
    if output.exists():
        raise FileExistsError("readout diagnostics require a fresh output directory")
    output.mkdir(parents=True)
    started = time.perf_counter()
    state_source = sha256_file(Path(__file__).with_name("population_state.py"))
    _require(qualification["software_sources"]["population_state.py"] == state_source, "frozen population_state.py changed")
    consumed_files, origins, metric_names, moment_names = {}, [], None, None
    time_count = 0
    with (output / "time_moments.csv").open("x", newline="") as time_stream, (output / "origin_horizon_moments.csv").open("x", newline="") as origin_stream:
        time_writer = origin_writer = None
        for seed in SEEDS:
            parent_cell, parent_index = source_cells[seed, "original"]
            checkpoint_path = _delivery_file(parent_cell, parent_index, "training/best.pt", "selected frozen parent")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            binding = checkpoint["binding"]
            _require(binding["seed"] == seed and binding["model_implementation_sha256"] == state_source,
                     "selected parent seed or model source differs")
            parent = ControlledPopulationState(**checkpoint["model_configuration"])
            parent.load_state_dict(checkpoint["model"], strict=True)
            parent.eval()
            _require(len(parent.unit_ids) == 100000 and len(parent.field_names) == FIELDS and parent.latent_channels == CHANNELS
                     and parent.bin_seconds == BIN_MS/1000, "complete native parent dimensions required")
            consumed_files[f"{seed}/original"] = {"training/best.pt": parent_index["training/best.pt"]["sha256"]}
            for model in ("current", "history"):
                consumed_files[f"{seed}/{model}"] = {"provenance/fit.json": source_cells[seed, model][1]["provenance/fit.json"]["sha256"]}
            for block in BLOCKS:
                for template in range(4):
                    case = f"train_block_{block:03d}_familiar_{template}"
                    cache_name = f"training/posterior_cache/{case}.pt"
                    cache_path = _delivery_file(parent_cell, parent_index, cache_name, "frozen posterior cache")
                    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
                    consumed_files[f"{seed}/original"][cache_name] = parent_index[cache_name]["sha256"]
                    for origin in ORIGINS_MS:
                        q_mean, q_variance = posterior_window(cache, checkpoint, case, origin)
                        for model in MODELS:
                            cell = cells[seed, model]
                            native = {h: cell["rows"][seed, case, origin, h] for h in HORIZONS_MS}
                            filename = native[20.]["ensemble_artifact"]
                            prefix = "forecasts/" if model == "original" else f"results/{model}/forecasts/"
                            relative = prefix+filename
                            cell_root, index = source_cells[seed, model]
                            trajectory_path = _delivery_file(cell_root, index, relative, "retained native trajectory")
                            samples, initial = load_retained_trajectory(trajectory_path)
                            consumed_files[f"{seed}/{model}"][relative] = index[relative]["sha256"]
                            moments = trajectory_readout_moments(parent, samples, initial, q_mean, q_variance, scales[seed], unit_chunk_size=chunk)
                            names = _moment_columns(moments, moment_names)
                            moment_names = names
                            time_rows, rows, columns = origin_tables(moments, native, cell["metrics"])
                            _require(metric_names is None or columns == metric_names, "paired summary metric columns differ")
                            metric_names = columns
                            if time_writer is None:
                                time_writer = csv.DictWriter(time_stream, fieldnames=list(time_rows[0]))
                                origin_writer = csv.DictWriter(origin_stream, fieldnames=list(rows[0]))
                                time_writer.writeheader()
                                origin_writer.writeheader()
                            time_writer.writerows(time_rows)
                            origin_writer.writerows(rows)
                            time_count += len(time_rows)
                            origins.extend(rows)
                    print(json.dumps(dict(seed=seed, completed_case=case, models=list(MODELS))), flush=True)
            del parent, checkpoint, cache
    _require(time_count == 3*3*48*100 and len(origins) == 3*3*48*3, "complete retained-trajectory coverage required")
    seed_summary = summarize_origins(origins, metric_names)
    block_summary = summarize_origins(origins, metric_names, by_block=True)
    _require(all(row["total_origins"] == 48 for row in seed_summary)
             and all(row["total_origins"] == 12 for row in block_summary), "complete summary denominators required")
    _write_rows(output / "seed_horizon_summary.csv", seed_summary)
    _write_rows(output / "source_block_summary.csv", block_summary)
    counts = {"time_moments.csv": time_count, "origin_horizon_moments.csv": len(origins),
              "seed_horizon_summary.csv": len(seed_summary), "source_block_summary.csv": len(block_summary)}
    result = dict(status="COMPLETE_POSTHOC_NATIVE_READOUT_DIAGNOSTIC", runtime=runtime,
        training_seeds=list(SEEDS), models=list(MODELS), source_blocks=list(BLOCKS),
        origins_per_model_seed=48, retained_draws=DRAWS, future_bins=STEPS, source_units=100000,
        horizon_ms=list(HORIZONS_MS), coordinate_scale_fitting_blocks=list(range(12)),
        qualification_binding_available=binding_available, native_comparison_reused=True,
        input_manifest_sha256=sha256_file(manifest_path), small_inputs={role: manifest[role] for role in files},
        consumed_deliveries=[cells[seed, model]["consumed"] for seed in SEEDS for model in MODELS], consumed_files=consumed_files,
        cache_lineage="Initialization-parent checkpoint; fixed encoder/readout preserved during native-prior fitting.",
        q_reference="One future observed episode's frozen causal encoder diagnostic distribution, not predictive latent truth or a physical-state oracle.",
        pairwise_interpretation="Signed A/B/C differences across models within the same seed/case/origin/bin cancel the shared q reference. Squared component errors do not cancel it.",
        moment_windows="Per-time moments are instantaneous; prefix_mean averages instantaneous moments; prefix_max retains the largest closure residual; endpoint uses the last bin of the scored prefix.",
        native_metric_windows="native_* columns are unchanged recorded horizon scores or costs. Spike totals and total-count coverage/width concern the cumulative prefix; unit-bin MSE retains its original per-bin definition.",
        hazard_interpretation="Hazard/readout decomposition only. Do not exponentiate a unit-mean log gap into a global count ratio or identify a count-error cause from this decomposition.",
        arithmetic_hazard_interpretation="log_unit_mean_* takes the logarithm after arithmetic averaging of hazards over all units. The draw-to-posterior-mean log ratio equals the mean-path log ratio plus empirical aggregate Jensen gap at each time. Refractory availability is excluded; these are not count ratios. Prefix summaries average per-time log rates or ratios, not pooled prefix hazards.",
        cross_model_initial_sample_equality_assumed=False, full_unit_samples_materialized=False,
        trajectory_sampling_performed=False, parameter_fitting_performed=False, native_generation_performed=False,
        new_encoding_performed=False, qualification_or_native_scores_recomputed=False,
        refit_effect_identified=False, physical_noise_identified=False, significance_tests_computed=False,
        scientific_claim_qualified=False, sota_qualified=False, elapsed_seconds=time.perf_counter()-started,
        software_sources={name: sha256_file(Path(__file__).with_name(name)) for name in (
            "population_readout_diagnostic.py", "population_readout_moments.py", "population_residual_correction_analysis.py",
            "population_forecast_analysis.py", "population_state.py", "provenance.py")},
        artifacts={name: dict(sha256=sha256_file(output/name), rows=count) for name, count in counts.items()})
    freeze_json(output / "analysis.json", result)
    return result



def prefix_hazard_exposures(time_rows: list[dict], native_rows: list[dict]):
    """Integrate arithmetic unit-mean rates in seconds for each exact origin.

    The result excludes refractory availability and the discrete-tick Bernoulli
    transform. It is an unrestricted hazard exposure, not an expected count.
    Native counts remain the recorded finite-draw summaries.
    """
    keys = ("model", "seed", "block", "case_id", "template_id", "origin_ms")
    paths = ("draw", "mean_path", "posterior_mean")
    count_names = ("native_sampled_expected_spikes_per_unit", "native_observed_spikes_per_unit")

    def identity(row):
        return (row["model"], int(row["seed"]), int(row["block"]), row["case_id"],
                row["template_id"], float(row["origin_ms"]))

    grouped = defaultdict(dict)
    for row in time_rows:
        key, elapsed = identity(row), float(row["elapsed_ms"])
        _require(elapsed not in grouped[key], "duplicate origin/time in retained hazards")
        _require(float(row["absolute_bin_end_ms"]) == key[-1] + elapsed,
                 "retained hazard absolute time differs")
        values = tuple(float(row[f"log_unit_mean_{name}_hazard"]) for name in paths)
        _require(all(math.isfinite(value) for value in values), "finite retained log hazards required")
        grouped[key][elapsed] = values
    grid = [(step + 1) * BIN_MS for step in range(STEPS)]
    _require(grouped and all(sorted(values) == grid for values in grouped.values()),
             "complete retained hazard time grid required")
    seen, result = set(), []
    for row in native_rows:
        key, horizon = identity(row), float(row["horizon_ms"])
        _require(key in grouped and horizon in HORIZONS_MS and (key, horizon) not in seen,
                 "unique matched native origin/horizon required")
        seen.add((key, horizon))
        steps = int(horizon / BIN_MS)
        _require(int(row["prefix_bins"]) == steps, "native prefix clock differs")
        value = dict(zip(keys, key), horizon_ms=horizon, prefix_bins=steps)
        for index, name in enumerate(paths):
            logs = [grouped[key][elapsed][index] for elapsed in grid[:steps]]
            maximum = max(logs)
            log_exposure = (maximum + math.log(math.fsum(math.exp(x - maximum) for x in logs))
                            + math.log(BIN_MS / 1000))
            try:
                exposure = math.exp(log_exposure)
            except OverflowError as error:
                raise ValueError("integrated hazard exceeds float64 range") from error
            _require(math.isfinite(exposure) and exposure > 0,
                     "positive representable integrated hazard required")
            value[f"integrated_{name}_hazard_per_unit"] = exposure
        for name in count_names:
            value[name] = float(row[name])
            _require(math.isfinite(value[name]) and value[name] >= 0, "finite native counts required")
        result.append(value)
    _require(seen == {(key, horizon) for key in grouped for horizon in HORIZONS_MS},
             "complete native origin/horizon match required")
    metrics = [f"integrated_{name}_hazard_per_unit" for name in paths] + list(count_names)
    return result, metrics


def analyze_hazard_counts(analysis_path: Path, output: Path) -> dict:
    """Reuse completed time/origin tables; perform no model or source execution."""
    runtime = merlin_runtime(dict(execution_backend="merlin", hostname=platform.node(),
                                 environment={key: os.environ.get(key, "") for key in RUNTIME_KEYS}))
    original = json.loads(analysis_path.read_text())
    _require(original.get("status") == "COMPLETE_POSTHOC_NATIVE_READOUT_DIAGNOSTIC"
             and original.get("training_seeds") == list(SEEDS)
             and set(original.get("models", [])) == set(MODELS)
             and original.get("source_blocks") == list(BLOCKS)
             and original.get("source_units") == 100000
             and original.get("origins_per_model_seed") == 48
             and original.get("future_bins") == STEPS
             and original.get("horizon_ms") == list(HORIZONS_MS),
             "complete registered retained diagnostic required")
    tables = {}
    for name, expected in (("time_moments.csv", 43200), ("origin_horizon_moments.csv", 1296)):
        path = analysis_path.parent / name
        artifact = original["artifacts"][name]
        _require(artifact["sha256"] == sha256_file(path) and artifact["rows"] == expected,
                 "retained diagnostic table identity differs")
        with path.open(newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
        _require(len(tables[name]) == expected, "complete retained diagnostic table required")
    rows, metrics = prefix_hazard_exposures(tables["time_moments.csv"], tables["origin_horizon_moments.csv"])
    cohort = defaultdict(set)
    for row in rows:
        _require(row["origin_ms"] in ORIGINS_MS, "registered retained origin required")
        cohort[row["model"], row["seed"]].add(tuple(row[key] for key in
            ("block", "case_id", "template_id", "origin_ms")))
    _require(set(cohort) == {(model, seed) for model in MODELS for seed in SEEDS},
             "all registered model/seed cells required")
    reference = next(iter(cohort.values()))
    _require(len(reference) == 48 and all(value == reference for value in cohort.values())
             and {key[0] for key in reference} == set(BLOCKS)
             and all(sum(key[0] == block for key in reference) == 12 for block in BLOCKS),
             "same complete source origins required for every model/seed")
    seed_rows = summarize_origins(rows, metrics)
    block_rows = summarize_origins(rows, metrics, by_block=True)
    for row in seed_rows + block_rows:
        if row["metric"].startswith("integrated_"):
            row["metric_scope"] = "integrated_unrestricted_hazard"
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, values in (("origin_horizon_hazard_counts.csv", rows),
                         ("seed_horizon_hazard_counts.csv", seed_rows),
                         ("source_block_hazard_counts.csv", block_rows)):
        _write_rows(output / name, values)
        artifacts[name] = dict(rows=len(values), sha256=sha256_file(output / name))
    seed_values = {}
    for row in seed_rows:
        key = (row["model"], row["seed"], row["horizon_ms"])
        value = seed_values.setdefault(key, dict(model=key[0], seed=key[1], horizon_ms=key[2],
                                                 total_origins=row["total_origins"]))
        value[row["metric"]] = row["value"]
    result = dict(status="COMPLETE_RETAINED_HAZARD_COUNT_COMPARISON", runtime=runtime,
        original_analysis_sha256=sha256_file(analysis_path), original_runtime=original["runtime"],
        source_tables={name: original["artifacts"][name] for name in tables},
        training_seeds=list(SEEDS), models=list(MODELS), source_blocks=list(BLOCKS),
        source_units=100000, origins_per_model_seed=48, horizon_ms=list(HORIZONS_MS), bin_ms=BIN_MS,
        definition="Per origin: BIN_MS/1000 * sum_t exp(log_unit_mean_path_hazard[t]). Summaries average these exposures across the full origin denominator. This does not exponentiate a mean log hazard.",
        interpretation="Unrestricted hazard exposure excludes refractory availability and the Bernoulli tick transform. Recorded native counts are finite-draw summaries. Neither is a causal decomposition of count error.",
        seed_horizon_values=list(seed_values.values()), artifacts=artifacts, trajectory_sampling_performed=False,
        parameter_fitting_performed=False, native_generation_performed=False,
        scientific_claim_qualified=False, sota_qualified=False,
        software_sha256=sha256_file(Path(__file__)))
    freeze_json(output / "analysis.json", result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--retained-analysis", type=Path,
                        help="Completed analysis.json for arithmetic prefix hazard/count comparison")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (analyze(args.manifest, args.output) if args.manifest is not None
              else analyze_hazard_counts(args.retained_analysis, args.output))
    print(json.dumps(dict(status=result["status"], runtime=result["runtime"], artifacts=result["artifacts"])), flush=True)


if __name__ == "__main__":
    main()
