"""Decode saved Population latent paths without generating any new trajectories."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import traceback

import numpy as np
import torch

SEEDS = (41, 42, 43)
HORIZONS_MS = (20.0, 80.0, 500.0)
CASE_ID = "train_block_014_familiar_0"
ORIGIN_MS = 1000.0
FROZEN_CORE = {
    "population_state.py": "5b2b568a475629ceb0a18f3c806aa4cd98792b877aa1b0a7c85178ea8ae45e18",
    "refractory_observation.py": "0d5c56578c40fba05df5f707a4b3c705599fa90dd5a5a552b2c6d0c82a3dc69f",
    "renewal_trajectory_training.py": "02275a47607c8f734e2824137710216dcafe1862d388b1d69a1d3e7296943dfd",
}


def require(condition, message):
    if not condition:
        raise ValueError("POPULATION_ACTIVITY_DIAGNOSTIC: " + message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


@torch.no_grad()
def decode_activity(model, latent, horizon_bins, *, unit_chunk_size=4096):
    """Return sums over complete units; retain the saved draw axis.

    The finite-tick quantity is a conditional expectation with every native tick
    available. It bounds the conditional expectation with refractory periods.
    It is not an upper bound on each realized finite-ensemble count.
    """
    require(type(unit_chunk_size) is int and unit_chunk_size > 0, "positive unit chunk required")
    require(latent.ndim == 4 and latent.shape[0] > 0 and latent.shape[1] > 0
            and latent.shape[2:] == (len(model.field_names), model.latent_channels),
            "saved draw/time/field/channel latent shape required")
    require(all(type(n) is int and 0 < n <= latent.shape[1] for n in horizon_bins), "supported horizon bins required")
    require(model.bin_ticks > 0 and model.tick_seconds > 0
            and math.isclose(model.bin_seconds, model.bin_ticks * model.tick_seconds, rel_tol=0, abs_tol=1e-12),
            "aligned physical bin and native clock required")
    raw = np.zeros(tuple(latent.shape[:2]), dtype=np.float64)
    available = np.zeros_like(raw)
    seen = 0
    for start in range(0, len(model.unit_ids), unit_chunk_size):
        stop = min(start + unit_chunk_size, len(model.unit_ids))
        log_rate = model.log_hazard(latent, start, stop)
        require(log_rate.shape == (*latent.shape[:2], stop - start) and torch.isfinite(log_rate).all(),
                "complete finite decoded unit hazards required")
        # Match the frozen sampler's decoded intensity and native-tick arithmetic.
        rate = log_rate.exp()
        hazard = rate * model.tick_seconds
        require(torch.isfinite(hazard).all() and torch.all(hazard > 0), "finite positive native hazard required")
        probability = -torch.expm1(-hazard)
        raw += rate.to(torch.float64).sum(-1).cpu().numpy() * model.bin_seconds
        available += probability.to(torch.float64).sum(-1).cpu().numpy() * model.bin_ticks
        seen += stop - start
    require(seen == len(model.unit_ids), "complete unit accounting required")
    raw, available = raw.cumsum(1), available.cumsum(1)
    return {n: dict(raw_integrated_intensity=raw[:, n - 1].copy(),
                    all_ticks_available_expectation=available[:, n - 1].copy()) for n in horizon_bins}


def delivery_index(cell):
    delivery = read(cell / "delivery.json")
    entries = delivery["files"]
    index = {row["path"]: row for row in entries}
    require(len(index) == len(entries), "duplicate delivery artifact path")
    return delivery, index


def stage_file(cell, index, relative, destination, consumed):
    require(relative in index, "missing delivery artifact: " + relative)
    name = Path(relative)
    require(not name.is_absolute() and ".." not in name.parts, "relative delivered artifact required")
    source, target = cell / name, destination / name
    row = index[relative]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    size = target.stat().st_size
    require(("bytes" not in row or size == row["bytes"]) and digest(target) == row["sha256"],
            "delivered artifact differs: " + str(source))
    consumed.append(dict(path=str(source), bytes=size, sha256=row["sha256"]))
    return target


def select_identity(evaluation):
    config = evaluation["configuration"]
    require(evaluation["status"] == "EVALUATED_DEVELOPMENT"
            and evaluation["model_type"] == "population"
            and evaluation["episodes"] == 16 and evaluation["score_rows"] == 144,
            "complete accepted Population forecast required")
    require(config["joint_samples"] == 16 and tuple(config["horizons_ms"]) == HORIZONS_MS
            and config["origins_ms"][0] == ORIGIN_MS
            and config["retain_full_unit_samples"] == "first_validation_case_and_first_origin_per_seed_and_model"
            and evaluation["full_unit_replay_scope"] == "first complete validation case and origin per model/seed",
            "recorded first-origin retention contract changed")
    expected = f"{CASE_ID}_origin_{ORIGIN_MS:g}.npz"
    names = sorted(name for name in evaluation["artifacts"] if name.endswith(f"_origin_{ORIGIN_MS:g}.npz"))
    require(names and names[0] == expected and "full_unit_samples.npz" in evaluation["artifacts"],
            "predeclared first retained identity changed")
    return expected


def analyze_cell(cell, budget, seed, destination, frozen, model_class, common, posterior=None):
    consumed = []
    delivery, index = delivery_index(cell)
    require(delivery["status"] == "PASS" and delivery["exit_code"] == 0
            and delivery["training_seed"] == seed and delivery["completed_epochs"] == budget
            and delivery["verified_forecast_origins"] == 48 and delivery["score_rows"] == 144
            and delivery["core_neurons"] == 100000, "complete bound budget/seed delivery required")
    metadata_paths = {}
    for relative in ("training/run.json", "forecasts/evaluation.json"):
        metadata_paths[relative] = stage_file(cell, index, relative, destination, consumed)
    run, evaluation = (read(metadata_paths[name]) for name in ("training/run.json", "forecasts/evaluation.json"))
    filename = select_identity(evaluation)  # Identity is established before any checkpoint/array is opened.
    checkpoint_path = stage_file(cell, index, "training/best.pt", destination, consumed)
    forecast_path = stage_file(cell, index, "forecasts/" + filename, destination, consumed)
    count_path = stage_file(cell, index, "forecasts/full_unit_samples.npz", destination, consumed)
    checkpoint_hash = index["training/best.pt"]["sha256"]
    require(run["status"] == "FITTED_DEVELOPMENT" and run["completed_epochs"] == budget
            and run["best_checkpoint_sha256"] == evaluation["checkpoint_sha256"] == checkpoint_hash
            and evaluation["training_seed"] == seed
            and index["forecasts/" + filename]["sha256"] == evaluation["artifacts"][filename]
            and index["forecasts/full_unit_samples.npz"]["sha256"] == evaluation["artifacts"]["full_unit_samples.npz"],
            "checkpoint and forecast receipt binding differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    binding = checkpoint["binding"]
    require(binding == run["binding"] and binding["seed"] == seed
            and binding["config"]["model_family"] == "population"
            and checkpoint["model_configuration"] == run["model_configuration"]
            and checkpoint["next_epoch"] == run["best_epoch"] + 1,
            "selected checkpoint metadata differs")
    for key, name in (
        ("model_implementation_sha256", "population_state.py"),
        ("observation_implementation_sha256", "refractory_observation.py"),
        ("loader_implementation_sha256", "native_renewal_data.py"),
        ("latent_model_implementation_sha256", "poisson_state_space.py"),
        ("native_counts_implementation_sha256", "native_count_windows.py"),
    ):
        require(binding[key] == frozen[name], "accepted numerical implementation differs: " + name)
    for name, expected in FROZEN_CORE.items():
        require(frozen[name] == expected, "fixed Population/refractory/trainer runtime differs: " + name)
        relative = "Libs/population_dynamics/" + name
        if relative in index:
            require(index[relative]["sha256"] == expected, "cell runtime differs: " + name)
    require(binding["dataset_sha256"] == evaluation["dataset_sha256"]
            and binding["acquisition_design_sha256"] == evaluation["acquisition_design_sha256"]
            and binding["native_source_identity"] == evaluation["native_source_identity"],
            "forecast data/source identity differs")
    model = model_class(**checkpoint["model_configuration"])
    require(len(model.unit_ids) == 100000 and model.representation == "recurrent"
            and model.initialization_seed == seed and model.bin_seconds == .005 and model.tick_seconds == .0001,
            "full100000-unit recurrent model on native clock required")
    for name, expected in model.named_buffers():
        saved = checkpoint["model"][name]
        require(saved.dtype == expected.dtype and torch.equal(saved.cpu(), expected),
                "checkpoint physical observation map differs: " + name)
    model = model.to(dtype=checkpoint["model"]["unit_loading"].dtype)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    with np.load(forecast_path, allow_pickle=False) as archive:
        require(set(archive.files) == {"observed_field_counts", "sampled_field_counts", "latent_states", "initial_latent_states"},
                "saved field/latent schema differs")
        fields = {name: archive[name] for name in archive.files}
    with np.load(count_path, allow_pickle=False) as archive:
        require(set(archive.files) == {"sampled_unit_counts", "unit_ids", "initial_ready_in_ticks", "latent_states",
                                      "initial_latent_states", "terminal_ready_in_ticks"}, "saved unit schema differs")
        counts = archive["sampled_unit_counts"]
        ids = archive["unit_ids"]
        ready = archive["initial_ready_in_ticks"]
        terminal = archive["terminal_ready_in_ticks"]
        require(np.array_equal(fields["latent_states"], archive["latent_states"])
                and np.array_equal(fields["initial_latent_states"], archive["initial_latent_states"]),
                "field and full-unit latent archives disagree")
    unit_count, field_count = len(model.unit_ids), len(model.field_names)
    require(counts.shape == (16, 100, unit_count) and counts.dtype == np.int64
            and np.array_equal(ids, np.asarray(model.unit_ids, dtype=np.int64)),
            "full unit-count denominator or identity differs")
    intervals = model.minimum_interval_ticks.numpy()
    require(ready.shape == (unit_count,) and terminal.shape == (16, 1, unit_count)
            and ready.dtype.kind in "iu" and terminal.dtype.kind in "iu"
            and np.all(ready >= 0) and np.all(ready < intervals)
            and np.all(terminal >= 0) and np.all(terminal < intervals),
            "native refractory boundary state differs")
    latent = torch.from_numpy(fields["latent_states"])
    initial = torch.from_numpy(fields["initial_latent_states"])
    require(latent.shape == (16, 100, field_count, model.latent_channels)
            and initial.shape == (16, 1, field_count, model.latent_channels)
            and latent.dtype == initial.dtype == model.unit_loading.dtype
            and torch.isfinite(latent).all() and torch.isfinite(initial).all(),
            "complete finite saved latent denominator/dtype differs")
    observed, projected = fields["observed_field_counts"], fields["sampled_field_counts"]
    require(observed.shape == (100, field_count) and projected.shape == (16, 100, field_count)
            and np.isfinite(observed).all() and np.isfinite(projected).all()
            and np.all(observed >= 0) and np.all(projected >= 0)
            and np.equal(observed, np.rint(observed)).all() and np.equal(projected, np.rint(projected)).all(),
            "source field counts must be complete finite integers")
    mapping = model.unit_field_index.numpy()
    for field in range(field_count):
        positions = np.flatnonzero(mapping == field)
        values = counts[..., positions]
        require(np.all(values >= 0) and np.array_equal(values.sum(-1, dtype=np.int64), projected[..., field]),
                "saved full-unit and field count projections disagree")
    physical = dict(unit_ids=ids, unit_field_index=mapping, minimum_interval_ticks=intervals,
                    field_names=model.field_names, observed_field_counts=observed, initial_ready_in_ticks=ready,
                    dataset_sha256=evaluation["dataset_sha256"],
                    acquisition_design_sha256=evaluation["acquisition_design_sha256"],
                    native_source_identity=evaluation["native_source_identity"])
    if common is not None:
        for name, value in physical.items():
            equal = np.array_equal(value, common[name]) if isinstance(value, np.ndarray) else value == common[name]
            require(equal, "budget/seed source observation identity differs: " + name)
    else:
        common = physical
    horizon_bins = tuple(round(ms / (1000 * model.bin_seconds)) for ms in HORIZONS_MS)
    decoded = decode_activity(model, latent, horizon_bins, unit_chunk_size=model.unit_chunk_size)
    boundary = decode_activity(model, initial, (1,), unit_chunk_size=model.unit_chunk_size)[1]
    generated = counts.sum(-1, dtype=np.int64).cumsum(1, dtype=np.int64)
    truth = observed.sum(-1, dtype=np.float64).cumsum()
    summaries, draws = [], []
    identity = dict(training_seed=seed, training_budget_epochs=budget, selected_checkpoint_epoch=run["best_epoch"],
                    case_id=CASE_ID, source_block=14, template_id="familiar_0", origin_ms=ORIGIN_MS,
                    unit_count=unit_count, field_count=field_count, draws=16)
    for ms, bins in zip(HORIZONS_MS, horizon_bins):
        observed_count = float(truth[bins - 1] / unit_count)
        horizon_draws = []
        for draw in range(16):
            raw = float(decoded[bins]["raw_integrated_intensity"][draw] / unit_count)
            bound = float(decoded[bins]["all_ticks_available_expectation"][draw] / unit_count)
            saved = float(generated[draw, bins - 1] / unit_count)
            values = dict(raw_integrated_intensity_spikes_per_unit=raw,
                all_ticks_available_expected_spikes_per_unit=bound, saved_generated_spikes_per_unit=saved,
                observed_spikes_per_unit=observed_count, available_expectation_minus_observed=bound - observed_count,
                saved_generated_minus_available_expectation=saved - bound, raw_minus_finite_tick_expectation=raw - bound)
            horizon_draws.append(values)
            draws.append(dict(identity, horizon_ms=ms, horizon_bins=bins, draw=draw, **values))
        means = {key: math.fsum(row[key] for row in horizon_draws) / 16 for key in horizon_draws[0]}
        summaries.append(dict(identity, horizon_ms=ms, horizon_bins=bins, **means))
    metadata = dict(identity, source_cell=str(cell), source_cell_resolved=str(cell.resolve()), source_job=delivery["job_id"],
        delivery_sha256=digest(cell / "delivery.json"), consumed_artifacts=consumed,
        selected_checkpoint_sha256=checkpoint_hash, bin_seconds=model.bin_seconds, tick_seconds=model.tick_seconds,
        native_ticks_per_bin=model.bin_ticks, latent_dtype=str(latent.dtype), aggregation_dtype="float64",
        saved_count_values_read=int(counts.size), saved_latent_shape=list(latent.shape),
        saved_initial_latent_shape=list(initial.shape),
        initial_unavailable_unit_fraction=float(np.mean(ready > 0)),
        terminal_unavailable_unit_fraction_at500ms=float(np.mean(terminal > 0)),
        initial_latent_raw_intensity_hz_per_unit=float(boundary["raw_integrated_intensity"].mean() / unit_count / model.bin_seconds),
        initial_latent_all_ticks_available_spikes_per_unit_per_bin=float(boundary["all_ticks_available_expectation"].mean() / unit_count),
        complete_field_projection_verified=True, physical_identity_across_cells_verified=True)
    if posterior is not None:
        metadata["posterior_reconstruction"] = posterior_reconstruction(
            model, initial, latent, run, evaluation, physical, metadata, summaries, posterior, delivery)
    return summaries, draws, metadata, common


def gaussian_tick_bounds(log_mean, log_variance, tick_seconds):
    """Rigorous Gaussian log-hazard moment bounds, without latent sampling.

    For x >= 0, x - x*x/2 <= 1-exp(-x) <= x. Clipping the
    expectations to [0,1] preserves these bounds. A wide interval is reported;
    it is not silently replaced by the intensity or a mean-state probability.
    """
    m, v = np.asarray(log_mean, dtype=np.float64), np.asarray(log_variance, dtype=np.float64)
    require(m.shape == v.shape and np.isfinite(m).all() and np.isfinite(v).all()
            and np.all(v >= 0) and math.isfinite(tick_seconds) and tick_seconds > 0,
            "finite Gaussian log-hazard moments and positive clock required")
    rate = np.exp(m + .5 * v)
    first = rate * tick_seconds
    second = np.exp(2 * m + 2 * v) * tick_seconds**2
    require(np.isfinite(rate).all() and np.isfinite(second).all(), "finite analytic hazard moments required")
    return dict(expected_rate_hz=rate, first_tick_hazard_moment=first,
                second_tick_hazard_moment=second,
                probability_lower=np.maximum(0., first - .5 * second),
                probability_upper=np.minimum(1., first))


@torch.no_grad()
def posterior_activity_bounds(model, mean, variance, eligible_ticks, *, unit_chunk_size=4096):
    """Sum exact intensity moments and bounded observed-path expectations."""
    exposure = np.asarray(eligible_ticks)
    require(mean.ndim == 4 and mean.shape[0] == 1 and mean.shape == variance.shape
            and mean.shape[2:] == (len(model.field_names), model.latent_channels)
            and exposure.shape == (mean.shape[1], len(model.unit_ids))
            and np.isfinite(exposure).all() and np.all(exposure >= 0)
            and np.all(exposure <= model.bin_ticks) and np.equal(exposure, np.rint(exposure)).all()
            and type(unit_chunk_size) is int and unit_chunk_size > 0,
            "complete posterior and native eligible exposure required")
    keys = ("mean_state_integrated_intensity", "posterior_integrated_intensity",
            "all_ticks_expectation_lower", "all_ticks_expectation_upper",
            "observed_path_expectation_lower", "observed_path_expectation_upper")
    result = {key: np.zeros(mean.shape[1], dtype=np.float64) for key in keys}
    for start in range(0, len(model.unit_ids), unit_chunk_size):
        stop = min(start + unit_chunk_size, len(model.unit_ids))
        # Reuse the frozen readout for the mean. Diagonal posterior variance
        # propagates analytically through the same source-indexed linear map.
        log_mean = model.log_hazard(mean.to(torch.float64), start, stop)[0].cpu().numpy()
        field = model.unit_field_index[start:stop]
        log_variance = (variance.to(torch.float64).index_select(-2, field)
                        * model.unit_loading[start:stop].to(torch.float64).square()).sum(-1)[0].cpu().numpy()
        moments = gaussian_tick_bounds(log_mean, log_variance, model.tick_seconds)
        lower, upper = moments["probability_lower"], moments["probability_upper"]
        observed = exposure[:, start:stop]
        result["mean_state_integrated_intensity"] += np.exp(log_mean).sum(-1) * model.bin_seconds
        result["posterior_integrated_intensity"] += moments["expected_rate_hz"].sum(-1) * model.bin_seconds
        result["all_ticks_expectation_lower"] += lower.sum(-1) * model.bin_ticks
        result["all_ticks_expectation_upper"] += upper.sum(-1) * model.bin_ticks
        result["observed_path_expectation_lower"] += (lower * observed).sum(-1)
        result["observed_path_expectation_upper"] += (upper * observed).sum(-1)
    require(all(np.isfinite(value).all() for value in result.values()), "finite posterior activity totals required")
    return result


def stage_posterior_native(plan, work, reference, index, frozen):
    """Admit exactly five native payloads from the durable campaign archive."""
    helper_path = Path(plan["helper_path"])
    require(digest(helper_path) == plan["helper_sha256"], "prepared source-staging helper changed")
    spec = importlib.util.spec_from_file_location("activity_source_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    archive = Path(plan["repository"]) / helper.COHORT
    consumed = []
    input_path = stage_file(reference, index, "inputs.json", work / "native_admission", consumed)
    inputs = read(input_path)
    require(digest(archive / "delivery.json") == inputs["delivery_sha256"],
            "durable source archive differs from accepted checkpoint input receipt")
    receipt = read(archive / "delivery.json")
    source_index, accepted = helper.manifest_rows(receipt), helper.manifest_rows(inputs)
    names = [f"train/{CASE_ID}/{name}" for name in ("report.json", "spikes.npz", "initial_parameters.npz")]
    names += [f"train_fields/{CASE_ID}/{name}" for name in ("summary.json", "spatial_fields.npz")]
    destination = work / "native"
    for name in names:
        require(name in source_index and name in accepted
                and source_index[name]["sha256"] == accepted[name]["sha256"]
                and source_index[name]["bytes"] == accepted[name]["bytes"],
                "canonical native case input differs: " + name)
        stage_file(archive, source_index, name, destination, consumed)
    loader = importlib.import_module("population_dynamics.native_renewal_data")
    replay = importlib.import_module("population_dynamics.renewal_check")
    native = loader.NativeRenewalEpisode(destination / "train" / CASE_ID,
                                        destination / "train_fields" / CASE_ID)
    require(native.dt_ms == .1 and native.bin_ms == 5. and len(native.unit_ids) == 100000,
            "native100000-unit case on accepted clock required")
    count, stimulus = native.window(0, 15000)
    window = native.renewal_window(9000, 15000)
    require(np.array_equal(window.counts, count[180:300])
            and np.array_equal(window.stimulus, stimulus[180:300])
            and np.array_equal(window.initial_ready_in_ticks, native.ready_at(9000))
            and np.array_equal(window.terminal_ready_in_ticks, native.ready_at(15000)),
            "native prefix, reconstruction window or boundary exposure differs")
    exposure_replay = replay.replay_exposure(native, window, 9000, 15000, native.minimum_interval_ticks)
    projected = np.stack([count[:, native.unit_field_index == field].sum(-1, dtype=np.float64)
                          for field in range(len(native.field_ids))], -1)
    with np.load(destination / "train_fields" / CASE_ID / "spatial_fields.npz", allow_pickle=False) as fields:
        require(np.array_equal(fields["bin_edges_ticks"][:301], np.arange(301) * native.bin_ticks)
                and np.array_equal(fields["count_sum"][:300], projected),
                "native unit counts and source spatial projection differ")
    native_rows = [dict(case_id=CASE_ID, bin_start_ms=bin_index * 5., bin_stop_ms=(bin_index + 1) * 5.,
                        native_spikes=int(count[bin_index].sum(dtype=np.float64)),
                        native_spikes_per_unit=float(count[bin_index].mean(dtype=np.float64)),
                        actual_tc_pulse_count=int(stimulus[bin_index, 0]),
                        eligible_unit_ticks=int(window.eligible_ticks[bin_index - 180].sum(dtype=np.int64)))
                   for bin_index in range(180, 300)]
    return dict(native=native, count=count, stimulus=stimulus, window=window, projected=projected,
                window_rows=[], bin_rows=[], native_rows=native_rows, frozen=frozen,
                report=dict(durable_archive=str(archive), consumed_artifacts=consumed,
                    native_payload_files=5, native_payload_bytes=sum(accepted[n]["bytes"] for n in names),
                    dataset_sha256=receipt["dataset_sha256"],
                    acquisition_design_sha256=receipt["acquisition_design_sha256"],
                    source_binding=native.binding, full_native_field_projection_verified=True,
                    native_exposure_replay=exposure_replay, observation_interval_ms=[0., 1500.],
                    reconstruction_interval_ms=[900., 1500.], preboundary_interval_ms=[900., 1000.]))


@torch.no_grad()
def posterior_reconstruction(model, initial, latent, run, evaluation, physical, metadata, prior_rows, context, delivery):
    native, count, controls = context["native"], context["count"], context["stimulus"]
    require(delivery["device"] == "cpu" and run["hardware"]["device"] == "cpu"
            and evaluation["configuration"]["causal_history"] == "all_native_bins_before_origin"
            and evaluation["evaluator_implementation_sha256"] == context["frozen"]["population_trajectory_evaluation.py"],
            "accepted CPU forecast and complete causal prefix required for initial RNG replay")
    require(np.array_equal(native.unit_ids, physical["unit_ids"])
            and np.array_equal(native.unit_field_index, physical["unit_field_index"])
            and np.array_equal(native.minimum_interval_ticks, physical["minimum_interval_ticks"])
            and tuple(map(str, native.field_ids)) == tuple(physical["field_names"])
            and native.source_identity == physical["native_source_identity"]
            and context["report"]["dataset_sha256"] == physical["dataset_sha256"]
            and context["report"]["acquisition_design_sha256"] == physical["acquisition_design_sha256"]
            and np.array_equal(context["projected"][200:300], physical["observed_field_counts"])
            and np.array_equal(native.ready_at(10000), physical["initial_ready_in_ticks"]),
            "native case, saved observation projection or boundary readiness differs")
    history = dict(counts=count[None, :200], stimulus=controls[None, :200],
                   ready_in_ticks=physical["initial_ready_in_ticks"][None])
    state = model.infer_state(history, dict(unit_ids=native.unit_ids, field_names=tuple(map(str, native.field_ids))))
    mean, variance = model.encode(count[None], controls[None])
    mean_error = float((state.mean - mean[:, 199]).abs().max())
    variance_error = float((state.variance - variance[:, 199]).abs().max())
    require(torch.equal(state.mean, mean[:, 199]) and torch.equal(state.variance, variance[:, 199]),
            "causal prefix posterior differs when future observations are appended")
    seed_base = evaluation["configuration"]["sampling_seed_base"] + metadata["training_seed"] * 100000
    stream_module = importlib.import_module("population_dynamics.poisson_state_space")
    streams = stream_module.LatentRandomStreams.from_seeds(seed_base, seed_base + 1, seed_base + 2, device="cpu")
    # Replay only the admitted initial Gaussian draw operation. No transition,
    # observation RNG or count sampler is invoked.
    replay_initial = state.mean + state.variance.sqrt() * torch.randn(
        initial.shape, dtype=state.mean.dtype, generator=streams.initial)
    replay_error = float((replay_initial - initial).abs().max())
    require(torch.equal(replay_initial, initial),
            f"admitted initial RNG replay differs from saved initial draws: max_abs={replay_error}")
    result = posterior_activity_bounds(model, mean[:, 180:300], variance[:, 180:300],
                                      context["window"].eligible_ticks, unit_chunk_size=model.unit_chunk_size)
    units = len(model.unit_ids)
    identity = {key: metadata[key] for key in ("training_seed", "training_budget_epochs", "selected_checkpoint_epoch",
                                             "case_id", "origin_ms", "unit_count", "field_count")}
    prior = {row["horizon_ms"]: row for row in prior_rows}
    for start, stop, label in ((180, 200, "preboundary100ms"), (200, 204, "future20ms"),
                               (200, 216, "future80ms"), (200, 300, "future500ms")):
        values = {key + "_spikes_per_unit": float(value[start - 180:stop - 180].sum() / units)
                  for key, value in result.items()}
        values["observed_spikes_per_unit"] = float(count[start:stop].sum(dtype=np.float64) / units)
        values["observed_path_bound_width_spikes_per_unit"] = (
            values["observed_path_expectation_upper_spikes_per_unit"] - values["observed_path_expectation_lower_spikes_per_unit"])
        values["observed_path_lower_minus_observed"] = values["observed_path_expectation_lower_spikes_per_unit"] - values["observed_spikes_per_unit"]
        values["observed_path_upper_minus_observed"] = values["observed_path_expectation_upper_spikes_per_unit"] - values["observed_spikes_per_unit"]
        selected_mean, selected_variance = mean[:, start:stop], variance[:, start:stop]
        saved = latent[:, :stop - 200] if start == 200 else initial
        matched_prior = prior.get(float((stop - start) * 5)) if start == 200 else None
        context["window_rows"].append(dict(identity, window=label, start_ms=start * 5., stop_ms=stop * 5.,
            actual_tc_pulse_count=int(controls[start:stop].sum()),
            posterior_mean_min=float(selected_mean.min()), posterior_mean_max=float(selected_mean.max()),
            posterior_variance_min=float(selected_variance.min()), posterior_variance_mean=float(selected_variance.mean()),
            posterior_variance_max=float(selected_variance.max()),
            saved_latent_scope="future_prior" if start == 200 else "initial_draws_at_origin_only",
            saved_latent_min=float(saved.min()), saved_latent_max=float(saved.max()),
            saved_prior_all_ticks_expectation_spikes_per_unit=(matched_prior["all_ticks_available_expected_spikes_per_unit"] if matched_prior else None),
            saved_generated_spikes_per_unit=(matched_prior["saved_generated_spikes_per_unit"] if matched_prior else None), **values))
    for index in range(180, 300):
        context["bin_rows"].append(dict(identity, bin_start_ms=index * 5., bin_stop_ms=(index + 1) * 5.,
            observed_spikes_per_unit=float(count[index].mean(dtype=np.float64)),
            actual_tc_pulse_count=int(controls[index, 0]),
            posterior_mean_min=float(mean[:, index].min()), posterior_mean_max=float(mean[:, index].max()),
            posterior_variance_mean=float(variance[:, index].mean()),
            **{key + "_spikes_per_unit": float(value[index - 180] / units) for key, value in result.items()}))
    return dict(causal_prefix_bins=200, causal_prefix_interval_ms=[0., 1000.], boundary_posterior_bin=199,
        first_future_bin=200, native_projection_and_readiness_exact=True,
        prefix_vs_full_sequence_mean_max_abs=mean_error, prefix_vs_full_sequence_variance_max_abs=variance_error,
        initial_rng_replay_performed=True, initial_rng_device="cpu", initial_rng_seed=seed_base,
        initial_rng_replay_exact=True, initial_rng_replay_max_abs=replay_error,
        boundary_posterior_mean_min=float(state.mean.min()), boundary_posterior_mean_max=float(state.mean.max()),
        boundary_posterior_variance_min=float(state.variance.min()),
        boundary_posterior_variance_mean=float(state.variance.mean()),
        boundary_posterior_variance_max=float(state.variance.max()),
        saved_initial_draw_min=float(initial.min()), saved_initial_draw_max=float(initial.max()),
        transition_target_range=[-1., 1.], process_noise_has_unbounded_support=True,
        reconstruction_bins=120, reconstruction_unit_count_values=120 * units,
        window_rows=4, observed_path_conditioning=True, free_running_prediction=False,
        probability_integration="analytic_first_and_second_log_normal_moment_bounds", quadrature_performed=False)


def run(plan_path, output, *, mode="saved-prior"):
    require(mode in ("saved-prior", "posterior-reconstruction"), "supported public diagnostic mode required")
    plan_path, output = Path(plan_path), Path(output)
    plan = read(plan_path)
    require(plan["software_sha256"][Path(__file__).name] == digest(__file__), "prepared diagnostic implementation changed")
    require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURMD_NODENAME", socket.gethostname()).split(".")[0] == "cpu03",
            "CPU03 Slurm allocation required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only diagnostic required")
    require(plan["training_seeds"] == list(SEEDS) and plan["training_budgets"] == [20, 80]
            and plan["case_id"] == CASE_ID and plan["origin_ms"] == ORIGIN_MS
            and plan["horizons_ms"] == list(HORIZONS_MS), "predeclared diagnostic identity changed")
    require(not output.exists(), "new diagnostic output required; preserve existing execution")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    scratch = Path(plan["scratch_parent"])
    scratch.mkdir(parents=True, exist_ok=True)
    storage = shutil.disk_usage(scratch)
    require(storage.free >= 20 * 1024**3 and storage.used / storage.total < .95,
            "node-local scratch critical storage threshold")
    summaries, draws, cells, runtime_records = [], [], [], []
    posterior = None
    report = dict(status="RUNNING", started_utc=datetime.now(timezone.utc).isoformat(),
        job_id=os.environ["SLURM_JOB_ID"], node=socket.gethostname(), plan_sha256=digest(plan_path),
        diagnostic_implementation_sha256=digest(__file__), training_seeds=list(SEEDS), training_budgets=[20, 80],
        case_id=CASE_ID, source_block=14, template_id="familiar_0", origin_ms=ORIGIN_MS, horizons_ms=list(HORIZONS_MS),
        source_episode_count=1, connectivity_instances=1, scientific_claim_qualified=False,
        training_performed=False, source_simulation_performed=False, latent_rollout_performed=False,
        count_sampling_performed=False, posterior_mean_state_reconstruction_performed=False,
        source_exposure_conditioning_performed=False,
        meanings=dict(raw_integrated_intensity="Sum of decoded intensity_Hz * bin_seconds; mean over complete units and saved draws.",
            all_ticks_available_expectation="Sum of bin_ticks * [-expm1(-exp(log_hazard)*tick_seconds)] conditional on each saved latent path.",
            expectation_bound="All-ticks availability bounds the conditional expected count with refractory periods; no realized-count inequality is asserted.",
            observed_counts="Saved observed_field_counts summed over all source-defined fields and divided by all100000units.",
            signed_gaps="Finite16-draw realized counts may lie above or below their conditional expectation; signed gaps are retained.",
            boundary_intensity="Readout of saved initial latent draws before the first future transition; not an additional generated bin.",
            posterior="Not computed in this first-stage saved-prior diagnostic."))
    try:
        with tempfile.TemporaryDirectory(prefix="neuronsim-activity-", dir=scratch) as directory:
            work = Path(directory)
            original = Path(plan["original_root"])
            continued = Path(plan["continued_root"])
            reference = original / "seed_41"
            delivery, index = delivery_index(reference)
            runtime = work / "runtime"
            for relative in sorted(index):
                if relative.startswith("Libs/") and relative.endswith(".py"):
                    stage_file(reference, index, relative, runtime, runtime_records)
            frozen = {Path(row["path"]).name: row["sha256"] for row in runtime_records}
            for name, expected in FROZEN_CORE.items():
                require(frozen[name] == expected, "accepted reference runtime changed: " + name)
            sys.path.insert(0, str(runtime / "Libs"))
            module = importlib.import_module("population_dynamics.population_state")
            require(Path(module.__file__).resolve() == (runtime / "Libs/population_dynamics/population_state.py").resolve(),
                    "diagnostic imported an unbound working-tree model")
            if mode == "posterior-reconstruction":
                posterior = stage_posterior_native(plan, work, reference, index, frozen)
            common = None
            for budget, root in ((20, original), (80, continued)):
                for seed in SEEDS:
                    print(f"ADMIT budget={budget} seed={seed}", flush=True)
                    a, b, cell, common = analyze_cell(root / f"seed_{seed}", budget, seed,
                        work / "inputs" / f"budget_{budget}" / f"seed_{seed}", frozen, module.ControlledPopulationState, common, posterior)
                    summaries.extend(a)
                    draws.extend(b)
                    cells.append(cell)
                    print(json.dumps(dict(budget=budget, seed=seed, horizon500=a[-1])), flush=True)
            require(len(cells) == 6 and len(summaries) == 18 and len(draws) == 288,
                    "complete six-cell diagnostic denominator required")
            report.update(status="COMPLETE_SAVED_PRIOR_ACTIVITY_DIAGNOSTIC",
                native_source_identity=common["native_source_identity"], dataset_sha256=common["dataset_sha256"],
                acquisition_design_sha256=common["acquisition_design_sha256"])
            if posterior is not None:
                require(len(posterior["window_rows"]) == 24 and len(posterior["bin_rows"]) == 720,
                        "complete six-cell posterior reconstruction denominator required")
                report.update(status="COMPLETE_POSTERIOR_OBSERVED_PATH_DIAGNOSTIC",
                    posterior_mean_state_reconstruction_performed=True, source_exposure_conditioning_performed=True,
                    initial_rng_replay_performed=True, native_input=posterior["report"],
                    posterior_window_rows=24, posterior_bin_rows=720, quadrature_performed=False)
                report["meanings"]["posterior"] = (
                    "Causal observation-conditioned Gaussian posterior; analytic intensity moments and finite-tick "
                    "expectation bounds. Fixed observed exposure is reconstruction, not free-running prediction. "
                    "No significance or generative-fidelity qualification follows from this one-case diagnostic.")
    except Exception as exc:
        report.update(status="FAILED", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if posterior is not None:
            write_csv(output / "posterior_by_window.csv", posterior["window_rows"])
            write_csv(output / "posterior_by_bin.csv", posterior["bin_rows"])
            write_csv(output / "native_activity_by_bin.csv", posterior["native_rows"])
        write_csv(output / "activity_by_horizon.csv", summaries)
        write_csv(output / "activity_by_draw.csv", draws)
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(), cells=cells,
            completed_cells=len(cells), horizon_rows=len(summaries), draw_rows=len(draws),
            frozen_runtime_artifacts=runtime_records,
            artifacts={name: digest(output / name) for name in ("activity_by_horizon.csv", "activity_by_draw.csv", "posterior_by_window.csv", "posterior_by_bin.csv", "native_activity_by_bin.csv") if (output / name).exists()})
        write_json(output / "analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("saved-prior", "posterior-reconstruction"), default="saved-prior")
    args = parser.parse_args()
    result = run(args.plan, args.output, mode=args.mode)
    print(json.dumps(dict(status=result["status"], completed_cells=result["completed_cells"], output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
