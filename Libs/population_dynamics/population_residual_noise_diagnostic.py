"""Paired, fixed-checkpoint sensitivity to native process variance.

This fitting-block diagnostic does not select a model or qualify a claim.
Native loss/conditional renewal means and sampled field scores remain distinct.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import resource
import time
import traceback

import numpy as np
import torch
import yaml

from .population_native_forecast_decomposition import field_components
from .population_residual_mean_objective import residual_mean_objective
from .population_residual_memory import ResidualMemoryBoundary
from .population_residual_observation import configure_correction_parameters
from .population_residual_observation_fitting import (
    json_digest, load_batch, load_native_observation_correction,
    validate_config as validate_fit_config, write_json,
)
from .population_residual_observation_training import MEAN_OBJECTIVE, _stream_seed
from .population_residual_observation_validation import load_seed
from .population_residual_refit_provenance import build_refit_receipt
from .poisson_state_space import LatentRandomStreams
from .provenance import sha256_file


PROTOCOL = dict(
    schema="native_residual_process_variance_diagnostic_v1",
    training_seeds=[41, 42, 43], modes=["current", "history"], fitting_blocks=[0],
    templates_per_block=4, origins=[200, 600, 1000], horizons=[4, 16, 100],
    prediction_particles=8, panels=[0, 1], variance_multipliers=[0.8, 1.0, 1.25],
    source_units=100000, source_fields=128, latent_channels=4, bin_ms=5,
    native_unit_chunk_size=4096, mean_unit_chunk_size=32768, cpu_threads=2,
    scope="fitting_block_zero_directional_mechanism_diagnostic",
)
SAMPLED_METRICS = (
    "energy_draw_to_observation", "energy_subtracted_draw_to_draw", "energy_reconstruction",
    "energy_score", "field_mean_mse_uncorrected", "field_mean_mse_corrected",
    "standardized_generated_variance",
)
CONDITION_KEYS = ("seed", "mode", "case", "panel", "variance_multiplier")
PAIRING_KEYS = ("initial_seed", "dynamics_seed", "observation_seed", "initial_end_sha256",
                "dynamics_end_sha256", "initial_latent_sha256")


def validate_config(config):
    if not isinstance(config, dict) or config != PROTOCOL:
        raise ValueError("the complete registered process-variance diagnostic protocol is required")
    values = [v for value in config.values() for v in (value if isinstance(value, list) else [value])]
    if any(isinstance(value, bool) for value in values):
        raise ValueError("boolean protocol numbers are not permitted")
    for factor in config["variance_multipliers"]:
        validate_factor(factor)
    return config


def validate_factor(factor):
    if type(factor) not in (int, float) or not math.isfinite(factor) or factor <= 0:
        raise ValueError("a finite positive process variance multiplier is required")


def conditions(config):
    cases = [f"train_block_{block:03d}_familiar_{template}"
             for block in config["fitting_blocks"] for template in range(config["templates_per_block"])]
    return [dict(zip(CONDITION_KEYS, values)) for values in product(
        config["training_seeds"], config["modes"], cases, config["panels"], config["variance_multipliers"])]


def condition_key(row):
    return tuple(row[key] for key in CONDITION_KEYS)


def objective_names(config):
    return ["loss", "native_loss", "field_mean_loss"] + [
        f"{name}_h{h}" for h in config["horizons"]
        for name in ("native_nll", "field_mean_mse", "particle_ess")]


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def unchanged(model, before):
    current = model.state_dict()
    return set(current) == set(before) and all(torch.equal(current[name], value) for name, value in before.items())


@contextmanager
def process_variance_scale(model, factor):
    """Restore the exact original log variance even if forward/sampling fails."""
    validate_factor(factor)
    parent = model.parent
    if parent.metadata is not None or not hasattr(parent, "process_log_variance"):
        raise ValueError("the admitted parent must expose its native process_log_variance parameter")
    parameter = parent.process_log_variance
    original = parameter.detach().clone()
    try:
        with torch.no_grad():
            if factor != 1.0:  # Keep the baseline bit-exact, without an add/copy round trip.
                parameter.copy_(original + math.log(factor))
            variance = parameter.exp()
            if not torch.isfinite(variance).all() or torch.any(variance <= 0):
                raise FloatingPointError("scaled native process variance is outside numerical support")
        yield
    finally:
        with torch.no_grad():
            parameter.copy_(original)
        if not torch.equal(parameter, original):
            raise RuntimeError("native process log variance was not restored bit-exactly")


def named_streams(seed, case, panel, device):
    """Modes/factors share streams; panels use distinct diagnostic-only names."""
    if seed not in (41, 42, 43) or panel not in (0, 1) or not isinstance(case, str) or not case:
        raise ValueError("registered training seed, panel and nonempty case are required")
    ids = {role: _stream_seed(seed, "process_variance_diagnostic_v1", f"panel_{panel}_{role}", case)
           for role in ("initial", "dynamics", "observation")}
    return LatentRandomStreams.from_seeds(**ids, device=device), ids


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def boundary_and_controls(model, batch):
    """Flatten batch/origin in the unchanged native objective's draw order."""
    mean, variance, controls = (batch[name] for name in ("mean", "variance", "controls"))
    indices = torch.tensor(batch["origins"], device=mean.device, dtype=torch.long)
    shape = (-1, len(model.field_names), model.latent_channels)
    boundary = mean.index_select(1, indices - 1).reshape(shape)
    previous = mean.index_select(1, indices - 2).reshape(shape)
    previous_control = controls.index_select(1, indices - 1).reshape(-1, model.input_size)
    memory = boundary - model.parent.transition_mean(previous, previous_control)
    ready = batch["initial_ready_in_ticks"].reshape(len(boundary), len(model.unit_ids))
    state = ResidualMemoryBoundary(boundary, variance.index_select(1, indices - 1).reshape(shape), ready,
        model.unit_ids, model.field_names, model.bin_seconds, model.source_metadata_identity, memory=memory)
    scheduled = torch.stack([controls.index_select(1, indices + step).reshape(-1, model.input_size)
                             for step in range(max(batch["horizons"]))], dim=1)
    return state, scheduled


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_variant(model, batch, seed, case, panel, factor, *, particles=8, native_chunk=4096, mean_chunk=32768):
    """Evaluate unchanged native loss and REAL native-unit samples; never optimize."""
    before = snapshot(model)
    device = batch["mean"].device
    initial_digest = None
    try:
        with process_variance_scale(model, factor):
            loss_rng, ids = named_streams(seed, case, panel, device)
            synchronize(device)
            started = time.perf_counter()
            result = residual_mean_objective(model, **batch, initial_generator=loss_rng.initial,
                dynamics_generator=loss_rng.dynamics, prediction_particles=particles,
                unit_chunk_size=native_chunk, mean_unit_chunk_size=mean_chunk, checkpoint_readout=False)
            synchronize(device)
            objective_seconds = time.perf_counter() - started
            objective = {name: float(value.detach().cpu()) for name, value in result.items()}
            del result
            if not all(math.isfinite(value) for value in objective.values()):
                raise FloatingPointError("nonfinite native objective component")
            rng, replay_ids = named_streams(seed, case, panel, device)
            if ids != replay_ids:
                raise RuntimeError("native objective and simulation stream identities differ")
            state, scheduled = boundary_and_controls(model, batch)
            mapping, fields = model.unit_field_index, len(model.field_names)
            values = []
            started = time.perf_counter()
            for step in model.iter_simulation(state, scheduled, rng, particles, record_events=False):
                if initial_digest is None:
                    initial_digest = tensor_digest(step.initial_latent)
                counts = step.counts.to(torch.int64)
                reduced = counts.new_zeros((*counts.shape[:-1], fields)).index_add(-1, mapping, counts)
                values.append(reduced.cpu())
                del counts, reduced, step
            synchronize(device)
            sampling_seconds = time.perf_counter() - started
            if len(values) != max(batch["horizons"]):
                raise ValueError("native simulator did not complete every requested bin")
            for name in ("initial", "dynamics"):
                if not torch.equal(getattr(loss_rng, name).get_state(), getattr(rng, name).get_state()):
                    raise RuntimeError("native objective/simulation latent RNG advancement differs: " + name)
            sampled = torch.stack(values, dim=2).numpy()
            counts = batch["target_counts"].reshape(len(state.mean), max(batch["horizons"]), len(model.unit_ids)).to(torch.int64)
            observed = counts.new_zeros((*counts.shape[:-1], fields)).index_add(-1, mapping, counts).cpu().numpy()
            if sampled.shape != (particles, *observed.shape) or np.any(sampled < 0) or np.any(observed < 0):
                raise ValueError("complete nonnegative sampled and observed field counts required")
            pairing = {f"{name}_seed": value for name, value in ids.items()}
            pairing.update({f"{name}_end_sha256": tensor_digest(getattr(rng, name).get_state())
                            for name in ("initial", "dynamics", "observation")})
            pairing.update(initial_latent_sha256=initial_digest,
                objective_simulation_latent_rng_equal=True,
                latent_draw_shape=[particles, len(state.mean), fields, model.latent_channels],
                dynamics_steps=max(batch["horizons"]),
                observation_pairing="same stream start only; native sampler consumption may depend on events")
            return dict(objective=objective, sampled_field_counts=sampled, observed_field_counts=observed,
                        pairing=pairing, objective_seconds=objective_seconds, sampling_seconds=sampling_seconds)
    finally:
        if not unchanged(model, before):
            raise RuntimeError("fixed parent/correction state changed during the diagnostic")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("a fixed-checkpoint diagnostic populated parameter gradients")


def score_fields(result, batch, model):
    """Use authenticated inner-fitting SD; zero centering cancels from these scores."""
    sd = batch["field_standard_deviation_hz"].detach().cpu().numpy()
    units = torch.bincount(model.unit_field_index, minlength=len(model.field_names)).cpu().numpy()
    samples, observed = result["sampled_field_counts"], result["observed_field_counts"]
    if batch["mean"].shape[0] != 1 or model.bin_seconds != .005:
        raise ValueError("one complete five-ms diagnostic episode is required")
    rows = []
    for oi, origin in enumerate(batch["origins"]):
        for horizon in batch["horizons"]:
            values = field_components(samples[:, oi, :horizon], observed[oi, :horizon], np.zeros_like(sd), sd, units)
            rows.append(dict(origin=origin, horizon=horizon, horizon_ms=horizon * 5, **values))
    return rows, dict(sampled_field_counts=samples, observed_field_counts=observed,
        field_ids=np.asarray(model.field_names), unit_count=units, standard_deviation_hz=sd,
        origins=np.asarray(batch["origins"]), horizons=np.asarray(batch["horizons"]), bin_ms=np.asarray(5))


def aggregate_evaluations(records, config):
    """Keep the full condition denominator; never average incomplete groups."""
    validate_config(config)
    expected = {condition_key(row) for row in conditions(config)}
    lookup = {condition_key(row): row for row in records}
    if len(lookup) != len(records) or set(lookup) != expected:
        raise ValueError("all144 unique registered evaluations, including failures, are required")
    names = objective_names(config)
    objective_rows, sampled_rows = [], []
    sample_pairs = set(product(config["origins"], config["horizons"]))
    for row in records:
        complete = row["status"] == "COMPLETE"
        if row["status"] not in ("COMPLETE", "FAILED", "INCOMPLETE"):
            raise ValueError("explicit complete, failed or incomplete evaluation status required")
        identity = {key: row[key] for key in CONDITION_KEYS}
        components = row.get("objective", {})
        if complete and (not set(names) <= set(components) or not all(math.isfinite(components[n]) for n in names)):
            raise ValueError("completed evaluation lacks finite objective components")
        objective_rows.append(dict(**identity, status=row["status"],
            **{name: components[name] if complete else None for name in names}))
        metrics = row.get("sampled_metrics", [])
        scored = {(r["origin"], r["horizon"]): r for r in metrics}
        if complete and (len(scored) != len(metrics) or set(scored) != sample_pairs):
            raise ValueError("completed evaluation lacks the complete origin/horizon score denominator")
        for origin, horizon in product(config["origins"], config["horizons"]):
            values = scored[(origin, horizon)] if complete else {}
            if complete and not all(math.isfinite(values[name]) for name in SAMPLED_METRICS):
                raise ValueError("nonfinite sampled field component")
            sampled_rows.append(dict(**identity, origin=origin, horizon=horizon, status=row["status"],
                **{name: values[name] if complete else None for name in SAMPLED_METRICS}))
    paired_groups = 0
    for seed, case, panel in product(config["training_seeds"], sorted({r["case"] for r in records}), config["panels"]):
        group = [r for r in records if (r["seed"], r["case"], r["panel"]) == (seed, case, panel)]
        if all(r["status"] == "COMPLETE" for r in group):
            signatures = {tuple(r["pairing"][key] for key in PAIRING_KEYS) for r in group}
            if len(signatures) != 1 or not all(r["pairing"]["objective_simulation_latent_rng_equal"] for r in group):
                raise ValueError("factor/mode latent pairing differs")
            paired_groups += 1
    summaries = []
    for rows, metrics, with_horizon in ((objective_rows, names, False), (sampled_rows, SAMPLED_METRICS, True)):
        groups = []
        for seed, mode, panel, factor in product(config["training_seeds"], config["modes"], config["panels"], config["variance_multipliers"]):
            for horizon in (config["horizons"] if with_horizon else [None]):
                selected = [r for r in rows if (r["seed"], r["mode"], r["panel"], r["variance_multiplier"]) == (seed, mode, panel, factor)
                            and (horizon is None or r["horizon"] == horizon)]
                required = 4 * (len(config["origins"]) if with_horizon else 1)
                if len(selected) != required:
                    raise ValueError("incomplete declared aggregate denominator")
                count = sum(r["status"] == "COMPLETE" for r in selected)
                summary = dict(seed=seed, mode=mode, panel=panel, variance_multiplier=factor,
                    status="COMPLETE" if count == required else "INCOMPLETE", expected_rows=required, complete_rows=count)
                if with_horizon:
                    summary["horizon"] = horizon
                summary.update({name: float(np.mean([r[name] for r in selected])) if count == required else None for name in metrics})
                groups.append(summary)
        baselines = {(r["seed"], r["mode"], r["panel"], r.get("horizon")): r for r in groups if r["variance_multiplier"] == 1.0}
        for row in groups:
            baseline = baselines[row["seed"], row["mode"], row["panel"], row.get("horizon")]
            complete = row["status"] == baseline["status"] == "COMPLETE"
            row["paired_delta_status"] = "COMPLETE" if complete else "INCOMPLETE"
            row.update({"delta_" + name: row[name] - baseline[name] if complete else None for name in metrics})
        summaries.append(groups)
    return dict(expected_evaluations=len(expected), complete_evaluations=sum(r["status"] == "COMPLETE" for r in records),
        expected_sampled_rows=len(expected) * len(sample_pairs), paired_groups_verified=paired_groups,
        objective_rows=objective_rows, sampled_rows=sampled_rows,
        objective_means=summaries[0], sampled_means=summaries[1])


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluation_path(output, condition):
    factor = str(float(condition["variance_multiplier"])).replace(".", "p")
    return output / "cells" / f"seed_{condition['seed']}" / condition["mode"] / condition["case"] / f"panel_{condition['panel']}" / f"variance_{factor}"


def read_completed(path, condition, identity_sha256):
    report = json.loads(path.read_text())
    if condition_key(report) != condition_key(condition) or report.get("diagnostic_identity_sha256") != identity_sha256:
        raise ValueError("existing diagnostic evaluation identity differs")
    if report.get("status") == "COMPLETE":
        artifact = report["field_artifact"]
        fields = path.parent / "sampled_fields.npz"
        if (artifact["path"] != "sampled_fields.npz" or fields.stat().st_size != artifact["bytes"]
                or sha256_file(fields) != artifact["sha256"] or report.get("state_unchanged") is not True
                or report.get("exit_code") != 0):
            raise ValueError("existing complete diagnostic field artifact or state receipt differs")
    elif report.get("status") not in ("FAILED", "INCOMPLETE"):
        raise ValueError("existing partial evaluation must have an explicit incomplete state")
    return report


def diagnose(inputs, refits, fit_config_path, config_path, output, device="cuda"):
    if platform.node().split(".")[0] == "mgmt2" or not os.environ.get("RH2_JOB_RUN_ID") or not os.environ.get("ARNOLD_TRIAL_ID"):
        raise RuntimeError("process-variance diagnosis requires actual Merlin run and trial identities")
    inputs, refits, fit_config_path, config_path, output = map(Path, (inputs, refits, fit_config_path, config_path, output))
    config = validate_config(yaml.safe_load(config_path.read_text()))
    fit_config = yaml.safe_load(fit_config_path.read_text())
    validate_fit_config(fit_config)
    family = build_refit_receipt(refits)
    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (fit_config.get("objective") != MEAN_OBJECTIVE or family["config"] != fit_config
            or family["shared_binding"]["config_sha256"] != sha256_file(fit_config_path)
            or family["shared_binding"]["input_manifest_sha256"] != sha256_file(manifest_path)
            or manifest.get("scope") != "complete_fitting_and_internal_selection"
            or manifest.get("training_seeds") != [41, 42, 43] or manifest.get("blocks") != list(range(14))):
        raise ValueError("authenticated complete mean-objective refits and original fitting inputs are required")
    for name, digest in family["shared_binding"]["sources"].items():
        if Path(name).name != name or sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError("fitting source identity differs: " + name)
    sources = {name: sha256_file(Path(__file__).with_name(name)) for name in (
        "population_residual_noise_diagnostic.py", "population_native_forecast_decomposition.py", "trajectory_scores.py")}
    identity = dict(config=config, config_sha256=sha256_file(config_path), fit_config_sha256=sha256_file(fit_config_path),
        input_manifest_sha256=sha256_file(manifest_path), refit_receipt=family, diagnostic_sources=sources)
    identity_sha256 = json_digest(identity)
    if output.exists():
        prior = json.loads((output / "progress.json").read_text())
        if prior.get("diagnostic_identity_sha256") != identity_sha256:
            raise ValueError("existing diagnostic output is bound to different inputs or source")
    else:
        output.mkdir(parents=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    execution = dict(backend="merlin", run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ["ARNOLD_TRIAL_ID"],
        hostname=platform.node(), device=str(device), python=platform.python_version(), numpy=np.__version__,
        torch=str(torch.__version__), cuda_runtime=torch.version.cuda, cpu_threads=torch.get_num_threads(),
        cpu_interop_threads=torch.get_num_interop_threads())
    if device.type == "cuda":
        execution.update(gpu=torch.cuda.get_device_name(device), gpu_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    planned, records, reused = conditions(config), [], 0
    started = time.perf_counter()
    def progress():
        write_json(output / "progress.json", dict(status="RUNNING", diagnostic_identity_sha256=identity_sha256,
            expected_evaluations=len(planned), recorded_evaluations=len(records),
            complete_evaluations=sum(r["status"] == "COMPLETE" for r in records), reused_evaluations=reused,
            updated_utc=datetime.now(timezone.utc).isoformat(), execution=execution))
    progress()
    for seed in config["training_seeds"]:
        parent, checkpoint, seed_error = None, None, None
        for mode in config["modes"]:
            cell_records, model, batch, prior_case, cell_error = [], None, None, None, None
            selected = [row for row in planned if row["seed"] == seed and row["mode"] == mode]
            for condition in selected:
                directory = evaluation_path(output, condition)
                report_path = directory / "analysis.json"
                if report_path.is_file():
                    report = read_completed(report_path, condition, identity_sha256)
                    reused += report["status"] == "COMPLETE"
                    records.append(report); cell_records.append(report); progress()
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                report = dict(**condition, schema=config["schema"], diagnostic_identity_sha256=identity_sha256,
                    status="FAILED", exit_code=2, execution=execution, scientific_claim_qualified=False,
                    optimizer_steps=0, started_utc=datetime.now(timezone.utc).isoformat())
                before = None
                evaluation_started = time.perf_counter()
                try:
                    if (directory / "sampled_fields.npz").exists():
                        report["status"] = "INCOMPLETE"
                        raise ValueError("field array exists without a completed evaluation receipt; retained without automatic retry")
                    if seed_error or cell_error:
                        raise RuntimeError(seed_error or cell_error)
                    if parent is None:
                        try:
                            parent, pilot, _ = load_seed(inputs, seed)
                            del pilot
                            checkpoint = torch.load(inputs / f"parent/seed_{seed}/training/best.pt", map_location="cpu", weights_only=True)
                            parent.to(device)
                        except Exception as error:
                            seed_error = f"parent admission failed: {type(error).__name__}: {error}"
                            raise
                    if model is None:
                        try:
                            model = load_native_observation_correction(refits / f"seed_{seed}" / mode, parent,
                                checkpoint["binding"], sha256_file(inputs / f"parent/seed_{seed}/training/best.pt")).to(device).eval()
                            configure_correction_parameters(model)
                        except Exception as error:
                            cell_error = f"correction admission failed: {type(error).__name__}: {error}"
                            raise
                    before = snapshot(model)
                    if prior_case != condition["case"]:
                        batch = load_batch(inputs, seed, condition["case"], parent, checkpoint, fit_config, device,
                            field_mean_normalization=model.correction_identity["binding"]["field_mean_normalization"])
                        prior_case = condition["case"]
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    result = evaluate_variant(model, batch, seed, condition["case"], condition["panel"], condition["variance_multiplier"],
                        particles=config["prediction_particles"], native_chunk=config["native_unit_chunk_size"], mean_chunk=config["mean_unit_chunk_size"])
                    metrics, arrays = score_fields(result, batch, model)
                    fields_path = directory / "sampled_fields.npz"
                    temporary = directory / "sampled_fields.npz.tmp"
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, **arrays)
                    temporary.replace(fields_path)
                    report.update(status="COMPLETE", exit_code=0, objective=result["objective"], sampled_metrics=metrics,
                        pairing=result["pairing"], objective_seconds=result["objective_seconds"], sampling_seconds=result["sampling_seconds"],
                        normalization=model.correction_identity["binding"]["field_mean_normalization"],
                        normalization_scope="authenticated fitting blocks0-11; zero common centering; not comparable to earlier LDNS absolute scores",
                        field_artifact=dict(path=fields_path.name, sha256=sha256_file(fields_path), bytes=fields_path.stat().st_size,
                            sampled_shape=list(arrays["sampled_field_counts"].shape), observed_shape=list(arrays["observed_field_counts"].shape),
                            units="integer native spikes per five-ms bin", unit_outputs_retained=False))
                    del result, arrays
                except Exception as error:
                    report.update(exit_code=2, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                    if report["status"] != "INCOMPLETE":
                        report["status"] = "FAILED"
                finally:
                    report["state_unchanged"] = unchanged(model, before) if model is not None and before is not None else None
                    if before is not None and report["state_unchanged"] is not True:
                        report.update(status="FAILED", exit_code=2, state_error="parent or correction state changed")
                        seed_error = "previous evaluation changed parent/correction state; remaining seed conditions withheld"
                    report["peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    if device.type == "cuda":
                        report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                    report["elapsed_seconds"] = time.perf_counter() - evaluation_started
                    report["timing_scope"] = "evaluation including any first-use admission/input loading, scoring and field artifact write"
                    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
                write_json(report_path, report)
                records.append(report); cell_records.append(report); progress()
                print(json.dumps({key: report.get(key) for key in (*CONDITION_KEYS, "status", "error")}), flush=True)
            cell_path = output / "cells" / f"seed_{seed}" / mode / "analysis.json"
            if not cell_path.exists():
                write_json(cell_path, dict(seed=seed, mode=mode, diagnostic_identity_sha256=identity_sha256,
                    status="COMPLETE" if all(r["status"] == "COMPLETE" for r in cell_records) else "INCOMPLETE",
                    expected_evaluations=24, complete_evaluations=sum(r["status"] == "COMPLETE" for r in cell_records),
                    reports=[dict(path=str((evaluation_path(output, r) / "analysis.json").relative_to(output)),
                                  sha256=sha256_file(evaluation_path(output, r) / "analysis.json")) for r in cell_records]))
            del model, batch
        del parent, checkpoint
    aggregated = aggregate_evaluations(records, config)
    artifacts = {}
    for key in ("objective_rows", "sampled_rows", "objective_means", "sampled_means"):
        path = output / (key + ".csv")
        write_csv(path, aggregated[key])
        artifacts[path.name] = dict(rows=len(aggregated[key]), sha256=sha256_file(path), bytes=path.stat().st_size)
    complete = aggregated["complete_evaluations"] == len(planned)
    report = dict(status="COMPLETE_DIRECTIONAL_PROCESS_VARIANCE_DIAGNOSTIC" if complete else "INCOMPLETE_DIRECTIONAL_PROCESS_VARIANCE_DIAGNOSTIC",
        exit_code=0 if complete else 2, diagnostic_identity_sha256=identity_sha256, identity=identity,
        expected_evaluations=len(planned), complete_evaluations=aggregated["complete_evaluations"],
        expected_sampled_rows=aggregated["expected_sampled_rows"], paired_groups_verified=aggregated["paired_groups_verified"],
        reused_evaluations=reused, artifacts=artifacts, execution=execution, elapsed_seconds=time.perf_counter()-started,
        objective_scope="all origins and horizons: unchanged native mixture NLL plus conditional-renewal field mean objective",
        sampled_scope="real native-unit samples reduced to fields; integer count arrays retained for score replay",
        pairing="exact shared latent draw layout and stream advancement across factors/modes; observation shares only stream start",
        interpretation="directional fitting-block mechanism diagnostic; no validation, calibration or selected-model claim",
        optimizer_steps=0, checkpoints_written=False, scientific_claim_qualified=False, automatic_go_decision=False)
    write_json(output / "analysis.json", report)
    write_json(output / "progress.json", dict(status=report["status"], diagnostic_identity_sha256=identity_sha256,
        expected_evaluations=len(planned), recorded_evaluations=len(records), complete_evaluations=aggregated["complete_evaluations"],
        updated_utc=datetime.now(timezone.utc).isoformat(), execution=execution))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "refits", "fit-config", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    report = diagnose(args.inputs, args.refits, args.fit_config, args.config, args.output, args.device)
    print(json.dumps({key: report[key] for key in ("status", "exit_code", "expected_evaluations", "complete_evaluations")}))
    raise SystemExit(report["exit_code"])


if __name__ == "__main__":
    main()
