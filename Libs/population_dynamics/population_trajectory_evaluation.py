"""Matched sampled-count evaluation for LDS baselines and population states.

All models use the complete observed prefix, the same future TC program,
unit/field resolution, sampling seeds, draw count and finite-ensemble scores.
Event-time likelihood and unordered-count likelihood are not conflated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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

from .native_renewal_data import NativeRenewalEpisode, load_dataset_index
from .poisson_state_space import ControlledPoissonLDS, LatentRandomStreams
from .poisson_trajectory_evaluation import fitting_scale
from .poisson_trajectory_training import partition
from .provenance import sha256_file
from .renewal_state_space import ControlledRenewalLDS
from .trajectory_sample_metrics import unit_count_metrics
from .trajectory_scores import energy_score, field_pairs, variogram_score

COST_POINT_UNITS = 100000
COST_POINT_FIELDS = 128
COST_POINT_CASE = "train_block_014_familiar_0"


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluation_hardware(model, device):
    """Describe this evaluation process rather than its training process."""
    result = dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                  cuda_runtime=torch.version.cuda, device=str(device), host=platform.node(),
                  cpu=platform.processor(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads(),
                  model_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                  model_parameter_devices=sorted({str(p.device) for p in model.parameters()}),
                  model_buffer_dtypes=sorted({str(b.dtype) for b in model.buffers()}))
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(gpu=props.name, gpu_total_memory_bytes=props.total_memory,
                      cudnn_version=torch.backends.cudnn.version())
    return result


def _peak_process_rss_bytes():
    # ru_maxrss is a process-lifetime high-water mark, not an origin delta.
    scale = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


def _load_model(training, acquisition, fields, model_type, device, native):
    if model_type not in ("poisson", "renewal", "population"):
        raise ValueError("supported model type required")
    run = json.loads((training / "run.json").read_text())
    if run["status"] != "FITTED_DEVELOPMENT":
        raise ValueError("complete development training required")
    checkpoint_path = training / "best.pt"
    if sha256_file(checkpoint_path) != run["best_checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    binding = checkpoint["binding"]
    if (sha256_file(acquisition / "design.json") != binding["acquisition_design_sha256"]
            or sha256_file(fields / "dataset.json") != binding["dataset_sha256"]):
        raise ValueError("evaluation data differ from the fitted cohort")
    implementation = {"poisson": "poisson_state_space.py", "renewal": "renewal_state_space.py",
                      "population": "population_state.py"}[model_type]
    if sha256_file(Path(__file__).with_name(implementation)) != binding["model_implementation_sha256"]:
        raise ValueError("fitted model implementation changed")
    if model_type == "poisson":
        config = binding["config"]
        model = ControlledPoissonLDS(list(map(int, native.unit_ids)), 1, config["latent_size"], config["encoder_size"],
                                    config["bin_ms"] / 1000, initialization_seed=binding["seed"])
        dtype = checkpoint["model"]["loading"].dtype
    elif model_type == "population":
        from .population_state import ControlledPopulationState
        if run["binding"] != binding or binding["config"]["model_family"] != "population":
            raise ValueError("population checkpoint and training record differ")
        for key, filename in [("observation_implementation_sha256", "refractory_observation.py"),
                              ("loader_implementation_sha256", "native_renewal_data.py"),
                              ("latent_model_implementation_sha256", "poisson_state_space.py"),
                              ("native_counts_implementation_sha256", "native_count_windows.py")]:
            if sha256_file(Path(__file__).with_name(filename)) != binding[key]:
                raise ValueError("population fitted numerical or loader implementation changed")
        model = ControlledPopulationState(**checkpoint["model_configuration"])
        if (model.unit_ids != tuple(map(int, native.unit_ids))
                or model.field_names != tuple(map(str, native.field_ids))
                or not np.array_equal(model.unit_field_index.numpy(), native.unit_field_index)
                or not np.array_equal(model.minimum_interval_ticks.numpy(), native.minimum_interval_ticks)
                or model.bin_seconds != native.bin_ms / 1000
                or model.tick_seconds != native.dt_ms / 1000
                or binding["native_source_identity"] != native.source_identity):
            raise ValueError("population checkpoint source, clock or observation map changed")
        # These buffers encode the source map, not fitted parameters. Reject
        # a checkpoint that would overwrite the verified constructor mapping.
        for name, expected in model.named_buffers():
            saved = checkpoint["model"][name]
            if saved.dtype != expected.dtype or not torch.equal(saved.cpu(), expected):
                raise ValueError("population checkpoint state observation map changed: " + name)
        dtype = checkpoint["model"]["unit_loading"].dtype
    else:
        for key, filename in [("observation_implementation_sha256", "refractory_observation.py"),
                              ("loader_implementation_sha256", "native_renewal_data.py")]:
            if sha256_file(Path(__file__).with_name(filename)) != binding[key]:
                raise ValueError("fitted observation or data implementation changed")
        model = ControlledRenewalLDS(**checkpoint["model_configuration"])
        if (tuple(model.latent_model.unit_ids) != tuple(map(int, native.unit_ids))
                or not np.array_equal(model.minimum_interval_ticks.cpu().numpy(), native.minimum_interval_ticks)):
            raise ValueError("renewal checkpoint unit identity or source interval changed")
        dtype = checkpoint["model"]["latent_model.loading"].dtype
    model = model.to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, binding, run["best_checkpoint_sha256"]


def _apply_correction(model, binding, checkpoint_digest, correction_path):
    # Frozen parent-only packages do not stage correction code. Load it only
    # after the native parent checkpoint and source implementation pass admission.
    fit_report = Path(correction_path) / "fit.json"
    if fit_report.is_file():
        schema = json.loads(fit_report.read_text()).get("schema")
        if schema != "native_residual_observation_fit_v1":
            raise ValueError("unsupported residual fit schema")
        from .population_residual_observation_fitting import load_native_observation_correction
        return load_native_observation_correction(correction_path, model, binding, checkpoint_digest).eval()
    from .population_residual_correction import load_correction
    return load_correction(correction_path, model, binding, checkpoint_digest).eval()


def _initial_memory_arrays(state, corrected):
    return {"initial_memory": state.memory.detach().cpu().numpy()} if corrected else {}


def _draw(model, model_type, state, future, seed_base, draws, device):
    streams = LatentRandomStreams.from_seeds(seed_base, seed_base + 1, seed_base + 2, device=device)
    if model_type == "poisson":
        # The public baseline also retains full conditional-mean trajectories.
        # Count-only timing omits that unrequested output; the sampler below
        # uses the same draws and is checked against the public implementation.
        counts, latent, initial = sample_poisson_counts(model, state, future, streams, draws)
        return counts[:, 0], latent[:, 0], initial, None
    elif model_type == "population":
        counts, latent, initial, terminal, _ = sample_population_outputs(
            model, state, future, streams, draws, record_events=False)
        return counts[:, 0], latent[:, 0], initial, terminal
    else:
        sample = model.simulate(state, future, streams, draws, record_events=False)
        if sample.events_recorded or len(sample.events):
            raise ValueError("matched count output unexpectedly materialized events")
        terminal = sample.terminal_ready_in_ticks
    return sample.counts[:, 0], sample.latent_states[:, 0], sample.initial_latent_states, terminal


@torch.no_grad()
def sample_population_outputs(model, state, stimulus, streams, draws, *, record_events=False):
    """Collect the declared forecast output from the streaming population API."""
    counts, latent, events = [], [], []
    initial = terminal = None
    for step in model.iter_simulation(state, stimulus, streams, draws, record_events=record_events):
        if not record_events and len(step.events):
            raise ValueError("count-only population output unexpectedly materialized events")
        counts.append(step.counts)
        latent.append(step.latent)
        if record_events:
            events.append(step.events)
        if initial is None:
            initial = step.initial_latent
        terminal = step.ready_in_ticks
    if not counts:
        raise ValueError("complete nonempty population forecast required")
    event_output = torch.cat(events) if record_events else None
    return torch.stack(counts, 2), torch.stack(latent, 2), initial, terminal, event_output


@torch.no_grad()
def sample_poisson_counts(model, state, stimulus, streams, draws):
    mean, variance, u = model._forecast_inputs(state, stimulus)
    process = model.process_log_variance.exp()
    model._positive_finite(process)
    shape = (draws, mean.shape[0], model.latent_size)
    z = mean + variance.sqrt() * torch.randn(shape, dtype=mean.dtype, device=mean.device, generator=streams.initial)
    initial = z.clone()
    counts, latent = [], []
    for t in range(u.shape[1]):
        z = (z @ model.transition.T + u[:, t] @ model.input_weight.T + model.transition_bias
             + process.sqrt() * torch.randn(shape, dtype=mean.dtype, device=mean.device, generator=streams.dynamics))
        # Keep the arithmetic order of ControlledPoissonLDS.simulate exactly.
        intensity = (z @ model.loading.T + model.log_rate_bias + math.log(model.bin_seconds)).exp()
        if not torch.isfinite(z).all() or not torch.isfinite(intensity).all():
            raise FloatingPointError("nonfinite count-only Poisson generation")
        sampled = torch.poisson(intensity, generator=streams.observation)
        if not torch.isfinite(sampled).all() or torch.any(sampled < 0) or torch.any(sampled >= torch.iinfo(torch.int64).max):
            raise FloatingPointError("invalid count-only Poisson sample")
        counts.append(sampled.to(torch.int64)); latent.append(z)
    return torch.stack(counts, 2), torch.stack(latent, 2), initial


@torch.no_grad()
def evaluate(acquisition, fields, training, config_path, output, model_type, device_name="cuda", *,
             correction_path=None):
    config = yaml.safe_load(config_path.read_text())
    if correction_path is not None and (model_type != "population" or config.get("schema") in
            ("pfc_population_checkpoint_forecast_v1", "v1_population_checkpoint_forecast_v1")):
        raise ValueError("residual correction requires the native population evaluation route")
    if config.get("schema") == "pfc_population_checkpoint_forecast_v1":
        if model_type != "population" or fields is not None:
            raise ValueError("PFC_FORECAST_POPULATION_MODEL_AND_DECLARED_INPUTS_REQUIRED")
        from .pfc_population_evaluation import evaluate as evaluate_pfc
        return evaluate_pfc(acquisition, training, config_path, output, device_name)
    if config.get("schema") == "v1_population_checkpoint_forecast_v1":
        if model_type != "population" or fields is not None:
            raise ValueError("V1_FORECAST_POPULATION_MODEL_AND_DECLARED_INPUTS_REQUIRED")
        from .v1_population_evaluation import evaluate as evaluate_v1
        return evaluate_v1(acquisition, training, config_path, output, device_name)
    function_started = time.perf_counter()
    if model_type not in ("poisson", "renewal", "population"):
        raise ValueError("supported model type required")
    if (config["joint_samples"] < 2 or config["formal_evaluation"] is not False
            or config["finite_ensemble_corrections"] is not True
            or config["unit_metric_dtype"] != "float32" or config["field_metric_dtype"] != "float64"
            or config["field_scale"] != "fitting_block_pooled_standard_deviation_of_per_unit_hz"
            or config["causal_history"] != "all_native_bins_before_origin"
            or config["matched_output"] != "all_identified_unit_counts_and_identical_field_projection"
            or config["retain_full_unit_samples"] != "first_validation_case_and_first_origin_per_seed_and_model"):
        raise ValueError("declared common development forecast protocol required")
    device = torch.device(device_name)
    design = json.loads((acquisition / "design.json").read_text())
    index = load_dataset_index(fields)
    if set(index) != {case["case_id"] for case in design["cases"]}:
        raise ValueError("canonical field cases differ from the acquired cohort")
    for case_id, row in index.items():
        if (sha256_file(acquisition / case_id / "report.json") != row["raw_report_sha256"]
                or sha256_file(fields / case_id / "summary.json") != row["field_summary_sha256"]):
            raise ValueError("fitting or evaluation metadata differ from the canonical field index")
    training_report = json.loads((training / "run.json").read_text())
    fit, validation = partition(design, training_report["binding"]["config"])
    native = NativeRenewalEpisode(acquisition / validation[0]["case_id"], fields / validation[0]["case_id"])
    if (native.source_identity["graph_manifest_sha256"] != design["graph_manifest_sha256"]
            or ("native_library_sha256" in design
                and native.source_identity["native_library_sha256"] != design["native_library_sha256"])):
        raise ValueError("native source differs from the acquisition declaration")
    model, binding, checkpoint_digest = _load_model(training, acquisition, fields, model_type, device, native)
    if correction_path is not None:
        model = _apply_correction(model, binding, checkpoint_digest, correction_path)
    method = model_type if correction_path is None else "population_residual_" + model.mode
    bin_ms = binding["config"]["bin_ms"]
    horizons = [int(round(value / bin_ms)) for value in config["horizons_ms"]]
    if any(h < 1 or h * bin_ms != value for h, value in zip(horizons, config["horizons_ms"])):
        raise ValueError("positive aligned horizons required")
    torch.set_num_threads(binding["config"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    longest, draws = max(horizons), config["joint_samples"]
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="RUNNING", model_type=model_type, configuration=config, training_seed=binding["seed"],
                  checkpoint_sha256=checkpoint_digest, dataset_sha256=binding["dataset_sha256"],
                  acquisition_design_sha256=binding["acquisition_design_sha256"],
                  evaluation_blocks=binding["config"]["validation_blocks"],
                  all_unit_outputs_generated=True, field_outputs_from_unit_counts=True,
                  scientific_claim_qualified=False, formal_test_accessed=False,
                  native_source_identity=native.source_identity,
                  output_contract="identified unit counts; event materialization excluded for all models",
                  unit_metric_estimator="float32 sample means and unbiased finite-draw squared-error correction",
                  full_unit_replay_scope="first complete validation case and origin per model/seed",
                  likelihood_scores_compared=False)
    if correction_path is not None:
        record.update(method=method, correction_identity=model.correction_identity)
    record["hardware"] = _evaluation_hardware(model, device)
    record["cost_scope"] = dict(
        function_total_seconds="function entry through completed score hash and report assembly; final evaluation.json write excluded",
        excluded_costs=["process imports", "launcher input/software staging", "launcher archival", "training"],
        elapsed_seconds="legacy evaluation loop including episode loading, origins, scores and output writes; setup and final score hash excluded",
        setup_seconds="function entry through evaluation setup, source/checkpoint validation and fitting-scale output",
        episode_costs="complete episode loaded, windowed and transferred once; shared by all its origins",
        origin_total_seconds="origin boundary checks through complete generation, projection, scoring, NPZ/hash/CSV writes, cleanup and progress flush",
        horizon_reuse="one longest-horizon ensemble per origin; all horizon score rows share this origin cost",
        peak_process_rss_bytes="cumulative process-lifetime resident-memory high-water mark, including imports and prior work",
        speedup_qualified=False)
    record["poisson_sampling_adapter"] = "same latent/Poisson draws; unrequested conditional-mean traces omitted"
    record["evaluator_implementation_sha256"] = sha256_file(Path(__file__))
    record["unit_metrics_implementation_sha256"] = sha256_file(Path(__file__).with_name("trajectory_sample_metrics.py"))
    (output / "evaluation.json").write_text(json.dumps(record, indent=2) + "\n")
    scale_mean, scale_sd, metadata = fitting_scale(fields, fit)
    if (not np.array_equal(native.field_ids, metadata["field_ids"])
            or not np.array_equal(np.bincount(native.unit_field_index, minlength=len(native.field_ids)), metadata["unit_count"])):
        raise ValueError("evaluation field identities or unit denominators differ from fitting")
    np.savez(output / "fitting_field_scale.npz", mean_hz=scale_mean, standard_deviation_hz=scale_sd, **metadata)
    definitions = [json.loads(str(v)) for v in metadata["field_ids"]]
    pairs = field_pairs([r[0] for r in definitions], [r[2] for r in definitions], [r[1] for r in definitions])
    first_units = native.unit_ids.copy()
    first_mapping = native.unit_field_index.copy()
    source_identity = native.source_identity
    expected_interval = native.minimum_interval_ticks.copy()
    rows, artifacts = [], {}
    episode_costs, origin_costs = [], []
    _synchronize(device)
    started = time.perf_counter()
    record["setup_seconds"] = started - function_started
    for case_index, case in enumerate(validation):
        episode_started = time.perf_counter()
        data = NativeRenewalEpisode(acquisition / case["case_id"], fields / case["case_id"],
                                   expected_source=source_identity, expected_unit_ids=first_units,
                                   expected_unit_field_index=first_mapping)
        row = index[case["case_id"]]
        if (data.binding["episode_report_sha256"] != row["raw_report_sha256"]
                or data.binding["field_summary_sha256"] != row["field_summary_sha256"]
                or not np.array_equal(data.field_ids, metadata["field_ids"])):
            raise ValueError("validation episode differs from its canonical field-table row")
        if not np.array_equal(data.minimum_interval_ticks, expected_interval):
            raise ValueError("source refractory law differs across validation episodes")
        loading_seconds = time.perf_counter() - episode_started
        start = time.perf_counter()
        count, stimulus = data.window(0, data.duration_ticks)
        window_seconds = time.perf_counter() - start
        _synchronize(device); start = time.perf_counter()
        y = torch.as_tensor(count, device=device)
        u = torch.as_tensor(stimulus, device=device)
        mapping = torch.as_tensor(data.unit_field_index, device=device)
        _synchronize(device); transfer_seconds = time.perf_counter() - start
        episode_costs.append(dict(case_id=case["case_id"], block=case["block"],
                                  native_duration_ticks=data.duration_ticks, native_bins=len(y),
                                  unit_count=len(first_units), origins_ms=list(config["origins_ms"]),
                                  loading_seconds=loading_seconds, window_seconds=window_seconds,
                                  transfer_seconds=transfer_seconds,
                                  episode_preparation_seconds=time.perf_counter() - episode_started,
                                  count_dtype=str(count.dtype), stimulus_dtype=str(stimulus.dtype),
                                  count_tensor_dtype=str(y.dtype), stimulus_tensor_dtype=str(u.dtype),
                                  count_tensor_device=str(y.device), stimulus_tensor_device=str(u.device)))
        for origin_index, origin in enumerate(config["origins_ms"]):
            origin_started = time.perf_counter()
            boundary = int(round(origin / bin_ms))
            if boundary < 1 or boundary * bin_ms != origin or boundary + longest > len(y):
                raise ValueError("complete aligned causal prefix and future support required")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _synchronize(device); start = time.perf_counter()
            history = dict(counts=y[None, :boundary], stimulus=u[None, :boundary])
            ready = data.ready_at(boundary * data.bin_ticks)
            if model_type in ("renewal", "population"):
                history["ready_in_ticks"] = ready[None]
            state = model.infer_state(history)
            future = u[None, boundary:boundary + longest]
            _synchronize(device); inference_seconds = time.perf_counter() - start
            seed_base = config["sampling_seed_base"] + binding["seed"] * 100000 + case_index * 100 + origin_index * 3
            start = time.perf_counter()
            generated, latent, initial, terminal = _draw(model, model_type, state, future, seed_base, draws, device)
            _synchronize(device); generation_seconds = time.perf_counter() - start
            observed = y[boundary:boundary + longest]
            start = time.perf_counter()
            projected = torch.zeros((draws, longest, len(metadata["field_ids"])), dtype=torch.int64, device=device)
            projected.index_add_(-1, mapping, generated)
            observed_field = torch.zeros(projected.shape[1:], dtype=torch.int64, device=device)
            observed_field.index_add_(-1, mapping, observed.to(torch.int64))
            raw_fields, raw_truth = projected.cpu().numpy(), observed_field.cpu().numpy()
            _synchronize(device); projection_seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            start = time.perf_counter()
            sampled = (raw_fields / (metadata["unit_count"] * bin_ms / 1000) - scale_mean) / scale_sd
            truth = (raw_truth / (metadata["unit_count"] * bin_ms / 1000) - scale_mean) / scale_sd
            scoring_seconds = time.perf_counter() - start
            name = f"{case['case_id']}_origin_{origin:g}.npz"
            start = time.perf_counter()
            memory_arrays = _initial_memory_arrays(state, correction_path is not None)
            np.savez_compressed(output / name, observed_field_counts=raw_truth, sampled_field_counts=raw_fields,
                                latent_states=latent.cpu().numpy(), initial_latent_states=initial.cpu().numpy(),
                                **memory_arrays)
            serialization_seconds = time.perf_counter() - start
            start = time.perf_counter()
            artifacts[name] = sha256_file(output / name)
            hash_seconds = time.perf_counter() - start
            if case_index == 0 and origin_index == 0:
                start = time.perf_counter()
                payload = dict(sampled_unit_counts=generated.cpu().numpy(), unit_ids=first_units,
                               initial_ready_in_ticks=ready, latent_states=latent.cpu().numpy(),
                               initial_latent_states=initial.cpu().numpy())
                payload.update(memory_arrays)
                if terminal is not None:
                    payload["terminal_ready_in_ticks"] = terminal.cpu().numpy()
                np.savez_compressed(output / "full_unit_samples.npz", **payload)
                serialization_seconds += time.perf_counter() - start
                start = time.perf_counter()
                artifacts["full_unit_samples.npz"] = sha256_file(output / "full_unit_samples.npz")
                hash_seconds += time.perf_counter() - start
                del payload
            first_row = len(rows)
            start = time.perf_counter()
            for ms, horizon in zip(config["horizons_ms"], horizons):
                row = dict(seed=binding["seed"], block=case["block"], case_id=case["case_id"],
                           template_id=case["template_id"], origin_ms=origin, horizon_ms=ms, method=method,
                           energy_score=energy_score(sampled[:, :horizon], truth[:horizon]),
                           **unit_count_metrics(generated[:, :horizon], observed[:horizon], expected_interval,
                                                data.bin_ticks, config["refractory_support_windows_bins"]),
                           inference_seconds=inference_seconds, full_unit_generation_seconds=generation_seconds,
                           field_projection_seconds=projection_seconds, peak_gpu_allocated_bytes=peak,
                           ensemble_artifact=name)
                for label, pair in pairs.items():
                    row["variogram_" + label] = variogram_score(sampled[:, :horizon], truth[:horizon], pair[:, 0], pair[:, 1])
                for lag in config["temporal_lags_bins"]:
                    row[f"variogram_temporal_lag_{lag}"] = (variogram_score(sampled[:, :horizon], truth[:horizon],
                        np.arange(len(metadata["field_ids"])), np.arange(len(metadata["field_ids"])), lag) if lag < horizon else None)
                rows.append(row)
            _synchronize(device); scoring_seconds += time.perf_counter() - start
            start = time.perf_counter()
            with (output / "scores.csv").open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                if first_row == 0:
                    writer.writeheader()
                writer.writerows(rows[first_row:])
            csv_seconds = time.perf_counter() - start
            cost = dict(case_id=case["case_id"], block=case["block"], origin_ms=origin,
                        horizons_ms=list(config["horizons_ms"]), generated_horizon_ms=max(config["horizons_ms"]),
                        generated_bins=longest, draws=draws, unit_count=len(first_units),
                        field_count=len(metadata["field_ids"]), score_rows=len(rows) - first_row,
                        generated_unit_count_values=generated.numel(),
                        generated_unit_count_dtype=str(generated.dtype), generated_unit_count_device=str(generated.device),
                        latent_dtype=str(latent.dtype), latent_device=str(latent.device),
                        field_count_dtype=str(raw_fields.dtype), field_metric_dtype=str(sampled.dtype),
                        stored_full_unit_counts=case_index == 0 and origin_index == 0,
                        inference_seconds=inference_seconds, full_unit_generation_seconds=generation_seconds,
                        field_projection_seconds=projection_seconds, scoring_seconds=scoring_seconds,
                        serialization_seconds=serialization_seconds, hash_seconds=hash_seconds,
                        csv_seconds=csv_seconds, peak_gpu_allocated_bytes=peak)
            del generated, latent, initial, terminal, projected, observed_field, state, memory_arrays
            print(json.dumps(dict(status="EVALUATING_MATCHED_COUNTS", model=model_type, case=case["case_id"],
                                  origin_ms=origin, rows=len(rows))), flush=True)
            cost["origin_total_seconds"] = time.perf_counter() - origin_started
            cost["peak_process_rss_bytes"] = _peak_process_rss_bytes()
            origin_costs.append(cost)
        del y, u, count, stimulus
    record.update(status="EVALUATED_DEVELOPMENT", score_rows=len(rows), episodes=len(validation),
                  independent_blocks=len({c["block"] for c in validation}), elapsed_seconds=time.perf_counter() - started,
                  artifacts=artifacts,
                  inference="descriptive epoch-selection blocks; not independent efficacy or equivalence evidence")
    start = time.perf_counter()
    record["scores_sha256"] = sha256_file(output / "scores.csv")
    record["score_hash_seconds"] = time.perf_counter() - start
    record.update(episode_costs=episode_costs, origin_costs=origin_costs, origins=len(origin_costs),
                  generated_unit_count_values=sum(cost["generated_unit_count_values"] for cost in origin_costs),
                  peak_process_rss_bytes=_peak_process_rss_bytes())
    record["function_total_seconds"] = time.perf_counter() - function_started
    (output / "evaluation.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


@torch.no_grad()
def evaluate_cost_point(acquisition, fields, training, fitting_scale_path, config_path, output,
                        device_name="cuda"):
    """Time one new full-output ensemble from the observed-history posterior.

    The caller binds the completed seed41 checkpoint and its saved fitting scale.
    No fitting episode, completed forecast draw or native hidden state is read.
    """
    function_started = time.perf_counter()
    acquisition, fields, training, fitting_scale_path, config_path, output = map(
        Path, (acquisition, fields, training, fitting_scale_path, config_path, output))
    config = yaml.safe_load(config_path.read_text())
    expected = dict(scope="observed_history_posterior_development_cost_point",
        measurement_config="neural_population_matched_forecast.yaml", case_id=COST_POINT_CASE,
        training_seed=41, boundary_time_ms=1000., horizon_ms=500., joint_samples=16,
        sampling_seed_base=89000, expected_core_units=COST_POINT_UNITS, expected_field_count=COST_POINT_FIELDS,
        bin_ms=5., native_dt_ms=.1, native_episode_ms=6000., formal_evaluation=False)
    if config != expected:
        raise ValueError("registered single-origin population cost-point configuration required")
    measurement_path = config_path.with_name(config["measurement_config"])
    measurement = yaml.safe_load(measurement_path.read_text())
    required_measurement = dict(horizons_ms=[20., 80., 500.], joint_samples=16,
        causal_history="all_native_bins_before_origin", finite_ensemble_corrections=True,
        matched_output="all_identified_unit_counts_and_identical_field_projection", unit_metric_dtype="float32",
        field_metric_dtype="float64", field_scale="fitting_block_pooled_standard_deviation_of_per_unit_hz",
        temporal_lags_bins=[1, 4, 16], refractory_support_windows_bins=[1, 2, 4], formal_evaluation=False)
    if any(measurement.get(name) != value for name, value in required_measurement.items()):
        raise ValueError("frozen sampled-count cost-point measurement protocol required")
    device = torch.device(device_name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("CPU or CUDA cost-point device required")
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="RUNNING", scope=config["scope"], configuration=config,
        measurement_configuration=measurement, scientific_claim_qualified=False, matched_fidelity=False,
        hidden_microstate_truth=False, same_hardware=False, end_to_end=False, speedup=False,
        training_performed=False, fitting_scale_recomputed=False, completed_forecast_draws_reused=False,
        conditional_scope="strict observed history posterior; fixed native microscopic-state futures are a different conditional law",
        cost_scope="function entry through output hash/report assembly; process imports, training, launcher staging/copyout and final cost_point.json write excluded",
        peak_process_rss_scope="cumulative process-lifetime resident-memory high-water mark", artifacts={})
    def save():
        (output / "cost_point.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    save()
    try:
        design = json.loads((acquisition / "design.json").read_text())
        matches = [case for case in design["cases"] if case["case_id"] == config["case_id"]]
        if len(matches) != 1 or matches[0]["split"] != "train" or matches[0]["block"] != 14:
            raise ValueError("one registered source development case required")
        data = NativeRenewalEpisode(acquisition / config["case_id"], fields / config["case_id"])
        if (len(data.unit_ids) != COST_POINT_UNITS or len(data.field_ids) != COST_POINT_FIELDS
                or data.bin_ms != config["bin_ms"] or data.dt_ms != config["native_dt_ms"]
                or data.duration_ticks * data.dt_ms != config["native_episode_ms"]
                or data.source_identity["graph_manifest_sha256"] != design["graph_manifest_sha256"]
                or data.source_identity["native_library_sha256"] != design["native_library_sha256"]):
            raise ValueError("complete source-bound100000-unit128-field6s observation map required")
        model, binding, checkpoint_digest = _load_model(training, acquisition, fields, "population", device, data)
        if binding["seed"] != 41 or binding["config"]["bin_ms"] != data.bin_ms:
            raise ValueError("completed seed41 population checkpoint required")
        torch.set_num_threads(binding["config"]["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        with np.load(fitting_scale_path, allow_pickle=False) as scale:
            mean, sd, field_units = scale["mean_hz"], scale["standard_deviation_hz"], scale["unit_count"]
            if (not np.array_equal(scale["field_ids"], data.field_ids)
                    or not np.array_equal(field_units, np.bincount(data.unit_field_index, minlength=len(data.field_ids)))
                    or mean.shape != sd.shape or mean.shape != (len(data.field_ids),)
                    or not np.isfinite(mean).all() or not np.isfinite(sd).all() or not np.all(sd > 0)):
                raise ValueError("bound saved fitting scale and field denominators required")
        boundary_tick = int(round(config["boundary_time_ms"] / data.dt_ms))
        boundary_bin = int(round(config["boundary_time_ms"] / data.bin_ms))
        longest = int(round(config["horizon_ms"] / data.bin_ms))
        stop_tick = boundary_tick + longest * data.bin_ticks
        if boundary_tick != boundary_bin * data.bin_ticks or stop_tick > data.duration_ticks:
            raise ValueError("complete aligned cost-point history/future window required")
        definitions = [json.loads(str(name)) for name in data.field_ids]
        pairs = field_pairs([row[0] for row in definitions], [row[2] for row in definitions], [row[1] for row in definitions])
        record.update(hardware=_evaluation_hardware(model, device), checkpoint_sha256=checkpoint_digest,
            native_source_identity=data.source_identity, source_binding=data.binding,
            acquisition_design_sha256=binding["acquisition_design_sha256"], dataset_sha256=binding["dataset_sha256"],
            fitting_scale_sha256=sha256_file(fitting_scale_path), config_sha256=sha256_file(config_path),
            measurement_config_sha256=sha256_file(measurement_path), model_type="population", training_seed=41,
            sampling_seeds=dict(initial=89000, dynamics=89001, observation=89002),
            boundary_tick=boundary_tick, observed_history_bins=boundary_bin, generated_bins=longest,
            observed_history_interval="[0,1000ms)", forecast_interval="[1000ms,1500ms)")
        _synchronize(device)
        record["setup_seconds"] = time.perf_counter() - function_started
        started = time.perf_counter()
        counts, controls = data.window(0, stop_tick)
        _synchronize(device)
        record["data_window_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        y, u = torch.as_tensor(counts, device=device), torch.as_tensor(controls, device=device)
        mapping = torch.as_tensor(data.unit_field_index, device=device)
        _synchronize(device)
        record["input_transfer_seconds"] = time.perf_counter() - started
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        ready = data.ready_at(boundary_tick)
        history = dict(counts=y[None, :boundary_bin], stimulus=u[None, :boundary_bin], ready_in_ticks=ready[None])
        state = model.infer_state(history)
        future = u[None, boundary_bin:boundary_bin + longest]
        _synchronize(device)
        record["inference_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        generated, latent, initial, terminal = _draw(model, "population", state, future, 89000, 16, device)
        _synchronize(device)
        record["full_unit_generation_seconds"] = time.perf_counter() - started
        if generated.shape != (16, longest, len(data.unit_ids)) or generated.dtype != torch.int64:
            raise ValueError("complete int64 cost-point unit trajectories required")
        observed = y[boundary_bin:boundary_bin + longest]
        started = time.perf_counter()
        projected = torch.zeros((16, longest, len(data.field_ids)), dtype=torch.int64, device=device)
        projected.index_add_(-1, mapping, generated)
        projected_truth = torch.zeros((longest, len(data.field_ids)), dtype=torch.int64, device=device)
        projected_truth.index_add_(-1, mapping, observed.to(torch.int64))
        raw_fields, raw_truth = projected.cpu().numpy(), projected_truth.cpu().numpy()
        _synchronize(device)
        record["field_projection_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        scaled = (raw_fields / (field_units * data.bin_ms / 1000) - mean) / sd
        truth = (raw_truth / (field_units * data.bin_ms / 1000) - mean) / sd
        rows = []
        for ms in measurement["horizons_ms"]:
            horizon = int(ms / data.bin_ms)
            row = dict(seed=41, block=14, case_id=config["case_id"], origin_ms=1000., horizon_ms=ms,
                method="population", generated_horizon_ms=500., draws=16,
                energy_score=energy_score(scaled[:, :horizon], truth[:horizon]),
                **unit_count_metrics(generated[:, :horizon], observed[:horizon], data.minimum_interval_ticks,
                                     data.bin_ticks, measurement["refractory_support_windows_bins"]))
            for name, pair in pairs.items():
                row["variogram_" + name] = variogram_score(scaled[:, :horizon], truth[:horizon], pair[:, 0], pair[:, 1])
            for lag in measurement["temporal_lags_bins"]:
                row[f"variogram_temporal_lag_{lag}"] = (variogram_score(scaled[:, :horizon], truth[:horizon],
                    np.arange(len(data.field_ids)), np.arange(len(data.field_ids)), lag) if lag < horizon else None)
            rows.append(row)
        _synchronize(device)
        record["scoring_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        np.save(output / "sampled_unit_counts.npy", generated.cpu().numpy(), allow_pickle=False)
        np.savez_compressed(output / "ensemble_fields.npz", sampled_field_counts=raw_fields, observed_field_counts=raw_truth,
            observed_unit_counts=observed.cpu().numpy().astype(np.int64), unit_ids=data.unit_ids,
            unit_field_index=data.unit_field_index, field_ids=data.field_ids, unit_count=field_units,
            stimulus=future.cpu().numpy(), initial_ready_in_ticks=ready,
            latent_states=latent.cpu().numpy(), initial_latent_states=initial.cpu().numpy(),
            terminal_ready_in_ticks=terminal.cpu().numpy())
        _synchronize(device)
        record["serialization_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        with (output / "scores.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        record["csv_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        artifacts = {name: sha256_file(output / name) for name in ("sampled_unit_counts.npy", "ensemble_fields.npz", "scores.csv")}
        record["hash_seconds"] = time.perf_counter() - started
        record.update(status="EVALUATED_OBSERVED_HISTORY_POSTERIOR_COST_POINT", all_unit_outputs_generated=True,
            all_unit_outputs_saved=True, field_outputs_from_unit_counts=True, draws=16, score_rows=len(rows),
            unit_count=len(data.unit_ids), field_count=len(data.field_ids), generated_unit_count_values=generated.numel(),
            generated_field_count_values=projected.numel(), count_dtype=str(generated.dtype), count_device=str(generated.device),
            latent_dtype=str(latent.dtype), latent_device=str(latent.device), field_count_dtype=str(raw_fields.dtype),
            field_metric_dtype=str(scaled.dtype), peak_process_rss_bytes=_peak_process_rss_bytes(),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            evaluator_implementation_sha256=sha256_file(Path(__file__)), artifacts=artifacts)
        record["function_seconds"] = time.perf_counter() - function_started
        save()
        return record
    except BaseException as error:
        record.update(status="FAILED", error=str(error), error_type=type(error).__name__,
                      traceback=traceback.format_exc(), outputs_authoritative=False,
                      function_seconds=time.perf_counter() - function_started)
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("acquisition", "training", "config", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--fields", type=Path)
    parser.add_argument("--model-type", choices=["poisson", "renewal", "population"])
    parser.add_argument("--cost-point", action="store_true")
    parser.add_argument("--correction", type=Path, help="Authenticated native population residual correction directory")
    parser.add_argument("--fitting-scale", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.fields is None and yaml.safe_load(args.config.read_text()).get("schema")
            not in ("pfc_population_checkpoint_forecast_v1", "v1_population_checkpoint_forecast_v1")):
        parser.error("--fields is required for the native Mesocircuit cohort")
    if args.cost_point:
        if args.correction is not None:
            parser.error("--correction is not supported for cost-point evaluation")
        if args.fitting_scale is None or args.model_type not in (None, "population"):
            parser.error("cost-point requires --fitting-scale and population model only")
    elif args.model_type is None or args.fitting_scale is not None:
        parser.error("regular evaluation requires --model-type and no --fitting-scale")
    try:
        result = (evaluate_cost_point(args.acquisition, args.fields, args.training, args.fitting_scale,
                                     args.config, args.output, args.device) if args.cost_point else
                  evaluate(args.acquisition, args.fields, args.training, args.config, args.output, args.model_type, args.device,
                            correction_path=args.correction))
        print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}))
    except Exception as error:
        if args.output.is_dir():
            (args.output / "failure.json").write_text(json.dumps(dict(status="FAILED", error=str(error),
                traceback=traceback.format_exc()), indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
