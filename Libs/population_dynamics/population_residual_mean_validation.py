"""Run the registered three-origin mean objective and one fresh Adam update.

This entry point authenticates the complete fitting bundle. It initializes the
correction from blocks 0--11 and reports execution evidence for one fitting
episode. It does not load residual refits or generate observation samples.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
import traceback

import numpy as np
import torch
import yaml

from .population_native_mean_benchmark import _peak_memory, _snapshot, _synchronize, _unchanged
from .population_prior_residual_prediction import _coordinates
from .population_residual_memory import ResidualMemoryPopulationState
from .population_residual_mean_objective import residual_mean_objective
from .population_residual_observation import (
    configure_correction_parameters, residual_observation_mixture_loss, standardized_ridge_penalty,
)
from .population_residual_observation_fitting import (
    SOURCES, cases, fit_field_mean_normalization, json_digest, load_batch, save_predictor,
    validate_config, write_json,
)
from .population_residual_observation_training import MEAN_OBJECTIVE, _generator, _metrics, _stream_seed
from .population_residual_observation_validation import CASE, HORIZONS, ORIGINS, SEEDS, load_seed
from .provenance import sha256_file
from .residual_prediction import fit_residual_predictor

MODES = ("current", "history")
SCHEMA = "native_residual_mean_real_validation_v1"
CORRECTION_NAMES = ("correction_weight", "correction_bias")
MEAN_INPUTS = ("initial_ready_in_ticks", "field_standard_deviation_hz")


def _registered_config(config):
    validate_config(config)
    if (config.get("objective") != MEAN_OBJECTIVE or config["learning_rate"] != 1e-4
            or config["unit_chunk_size"] != 4096 or config["regularization_weight"] != 1e-4
            or config["patience"] != 12 or config["max_epochs"] != 80):
        raise ValueError("registered mean objective, Adam1e-4 and4096-unit native chunks required")


def _verify_input_manifest(inputs, manifest_path):
    """Verify staged files once; authenticated loaders establish source semantics."""
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "native_residual_observation_inputs_v1"
            or manifest.get("scope") != "complete_fitting_and_internal_selection"
            or manifest.get("blocks") != list(range(14))
            or manifest.get("training_seeds") != list(SEEDS)):
        raise ValueError("complete fitting input manifest required")
    rows = manifest["files"]
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("nonempty unique input paths required")
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("input path escapes staging root")
        path = inputs / relative
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError("staged input identity differs: " + str(relative))
    return manifest


def _fresh_reference(coordinates, mode, config):
    """Use the admitted parent's coordinates and only inner fitting rows."""
    X, Y, metadata = coordinates
    fitting = config["fitting_blocks"]
    if (mode not in MODES or fitting != list(range(12))
            or X.shape[0] != 14 or Y.shape[0] != 14):
        raise ValueError("fresh current/history initialization on blocks0--11 required")
    width = metadata["current_width"] + (
        metadata["fields"] * metadata["channels"] if mode == "history" else 0)
    return fit_residual_predictor(X[fitting], Y[fitting], width, config["initialization_ridge_alpha"])


def _native_parity(native, combined, final_states, generators, horizons):
    expected_keys = set(native) | {"native_loss", "field_mean_loss"} | {
        f"field_mean_mse_h{horizon}" for horizon in horizons}
    if "loss" not in native or set(combined) != expected_keys:
        raise ValueError("complete unchanged native terms and registered mean terms required")
    for name, value in native.items():
        other = combined["native_loss" if name == "loss" else name]
        if not torch.equal(value, other):
            raise ValueError("native objective term differs: " + name)
    if not all(torch.equal(generator.get_state(), state)
               for generator, state in zip(generators, final_states)):
        raise ValueError("public RNG advancement differs from the pure native objective")


def _components(terms, penalty, regularization_weight):
    result = _metrics(terms)
    result["ridge_penalty"] = float(penalty.detach().cpu())
    result["regularized_loss"] = float(
        (terms["loss"] + regularization_weight * penalty).detach().cpu())
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("nonfinite objective or ridge component")
    return result


def _timed(device, operation):
    _synchronize(device)
    started = time.perf_counter()
    result = operation()
    _synchronize(device)
    return result, time.perf_counter() - started


def exercise_objective(model, reference, batch, config, seed, report):
    """Exercise the unchanged production objective; tiny tests use this same path."""
    device = model.correction_weight.device
    frozen = _snapshot(model.parent)
    parameters = configure_correction_parameters(model)
    model.eval()
    correction_before = {name: getattr(model, name).detach().cpu().clone() for name in CORRECTION_NAMES}
    optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"], weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    stream_ids = {role: _stream_seed(seed, "qualify", role)
                  for role in ("train_initial", "train_dynamics")}
    initial, dynamics = [_generator(device, stream_ids[role])
                         for role in ("train_initial", "train_dynamics")]
    initial_states = (initial.get_state().clone(), dynamics.get_state().clone())
    report["random_streams"] = dict(protocol=config["stream_protocol"], phase="qualify", ids=stream_ids)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        report["baseline_allocated_bytes"] = torch.cuda.memory_allocated(device)
        report["baseline_reserved_bytes"] = torch.cuda.memory_reserved(device)
    started = time.perf_counter()

    def paired_generators():
        generators = (torch.Generator(device=device), torch.Generator(device=device))
        for generator, state in zip(generators, initial_states):
            generator.set_state(state)
        return generators

    def combined(generators):
        terms = residual_mean_objective(model, **batch, initial_generator=generators[0],
            dynamics_generator=generators[1], prediction_particles=config["prediction_particles"],
            unit_chunk_size=config["unit_chunk_size"], mean_unit_chunk_size=config["mean_unit_chunk_size"],
            checkpoint_readout=True)
        penalty = standardized_ridge_penalty(model, reference)
        total = terms["loss"] + config["regularization_weight"] * penalty
        return terms, penalty, total

    try:
        native_batch = {name: value for name, value in batch.items() if name not in MEAN_INPUTS}
        with torch.no_grad():
            native, report["pure_native_forward_seconds"] = _timed(device, lambda:
                residual_observation_mixture_loss(model, **native_batch, initial_generator=initial,
                    dynamics_generator=dynamics, prediction_particles=config["prediction_particles"],
                    unit_chunk_size=config["unit_chunk_size"], checkpoint_readout=True))
        final_states = (initial.get_state().clone(), dynamics.get_state().clone())
        report["pure_native_components"] = _metrics(native)
        generators = paired_generators()
        (before, penalty, total), report["forward_seconds"] = _timed(device, lambda: combined(generators))
        _native_parity(native, before, final_states, generators, batch["horizons"])
        report.update(native_result_keys=sorted(native), native_terms_exact=True, public_rng_exact=True,
                      before=_components(before, penalty, config["regularization_weight"]))
        _, report["backward_seconds"] = _timed(device, total.backward)
        if (any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in parameters)
                or any(parameter.grad is not None for parameter in model.parent.parameters())):
            raise FloatingPointError("missing/nonfinite correction gradient or nonfrozen parent gradient")
        norm = float(torch.stack([parameter.grad.double().square().sum() for parameter in parameters]).sum().sqrt())
        if not math.isfinite(norm) or norm <= 0:
            raise FloatingPointError("finite nonzero correction gradient required")
        report.update(gradient_norm=norm, gradient_finite=True, gradient_nonzero=True)
        _, report["optimizer_seconds"] = _timed(device, optimizer.step)
        report["optimizer_steps"] = 1
        report["correction_finite"] = all(
            bool(torch.isfinite(getattr(model, name)).all()) for name in CORRECTION_NAMES)
        report["correction_changed"] = any(
            not torch.equal(getattr(model, name).detach().cpu(), correction_before[name])
            for name in CORRECTION_NAMES)
        if not report["correction_finite"] or not report["correction_changed"]:
            raise FloatingPointError("finite changed correction required after the Adam step")
        _native_parity(native, before, final_states, generators, batch["horizons"])
        del native, before, penalty, total
        with torch.no_grad():
            (after, penalty, _), report["post_step_forward_seconds"] = _timed(
                device, lambda: combined(paired_generators()))
            report["after"] = _components(after, penalty, config["regularization_weight"])
        report["after_minus_before"] = {
            name: report["after"][name] - value for name, value in report["before"].items()}
        report["post_step_same_random_draws"] = True
        if not _unchanged(model.parent, frozen):
            raise RuntimeError("frozen parent parameters or buffers changed")
    finally:
        report["parent_unchanged"] = _unchanged(model.parent, frozen)
        report.update(_peak_memory(device))
        report["objective_execution_seconds"] = time.perf_counter() - started
    return report


def _finish_cell(directory, report):
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(directory / "analysis.json", report)
    print(json.dumps({key: report.get(key) for key in
        ("seed", "mode", "status", "exit_code", "gradient_norm", "objective_execution_seconds", "error")}), flush=True)
    return report


def measure_cell(parent, batch, coordinates, config, seed, mode, directory, binding, execution,
                 *, preparation_error=None):
    directory.mkdir(parents=True)
    report = dict(schema=SCHEMA, seed=seed, mode=mode, case_id=CASE, binding=binding, execution=execution,
        started_utc=datetime.now(timezone.utc).isoformat(), optimizer="Adam", learning_rate=1e-4,
        optimizer_steps=0, checkpoint_readout=True, prediction_particles=8, unit_chunk_size=4096,
        mean_unit_chunk_size=32768, origins=list(ORIGINS), horizons=list(HORIZONS),
        source_units=100000, fields=128, generated_unit_samples=False, warm_start_used=False,
        future_observed_exposure_used_for_mean=False, improvement_required=False,
        scientific_claim_qualified=False, production_qualified=False, full_fitting_feasibility_qualified=False)
    try:
        if preparation_error is not None:
            raise RuntimeError(preparation_error)
        started = time.perf_counter()
        reference = _fresh_reference(coordinates, mode, config)
        save_predictor(directory / "reference.npz", reference)
        report["initialization"] = dict(method="fresh_fit_residual_predictor", alpha=reference.alpha,
            fitting_blocks=list(range(12)), fitting_cases=cases(range(12)), normalization_blocks=list(range(12)),
            coordinate_source=coordinates[2], reference_sha256=sha256_file(directory / "reference.npz"))
        report["initialization_seconds"] = time.perf_counter() - started
        model = ResidualMemoryPopulationState(parent, reference, mode, {"schema": SCHEMA, "seed": seed})
        report["coverage"] = dict(mean_shape=list(batch["mean"].shape),
            target_counts_shape=list(batch["target_counts"].shape),
            causal_readiness_shape=list(batch["initial_ready_in_ticks"].shape),
            field_scale_shape=list(batch["field_standard_deviation_hz"].shape),
            origins=list(batch["origins"]), horizons=list(batch["horizons"]))
        exercise_objective(model, reference, batch, config, seed, report)
        report.update(status="COMPLETE", exit_code=0)
    except Exception as error:
        report.update(status="FAILED", exit_code=2, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
    return _finish_cell(directory, report)


def _execution(device):
    if (platform.node().split(".")[0] == "mgmt2" or not os.environ.get("RH2_JOB_RUN_ID")
            or not os.environ.get("ARNOLD_TRIAL_ID")):
        raise RuntimeError("real three-origin mean validation requires actual Merlin execution")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    selected = torch.device(device)
    execution = dict(backend="merlin", run_id=os.environ["RH2_JOB_RUN_ID"],
        trial_id=os.environ["ARNOLD_TRIAL_ID"], host=platform.node(), device=str(selected),
        python_version=platform.python_version(), torch_version=str(torch.__version__),
        numpy_version=np.__version__, cuda_version=torch.version.cuda)
    if selected.type == "cuda":
        execution.update(gpu_name=torch.cuda.get_device_name(selected),
                         gpu_total_memory_bytes=torch.cuda.get_device_properties(selected).total_memory)
    return selected, execution


def validate(inputs, manifest_path, config_path, output, device="cuda"):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    device, execution = _execution(device)
    inputs, manifest_path, config_path, output = map(Path, (inputs, manifest_path, config_path, output))
    if output.exists():
        raise FileExistsError("fresh three-origin validation output required")
    torch.set_num_threads(2)
    output.mkdir(parents=True)
    report = dict(schema=SCHEMA, status="RUNNING", execution=execution,
        started_utc=datetime.now(timezone.utc).isoformat(), training_seeds=list(SEEDS), modes=list(MODES),
        case_id=CASE, records=[], scientific_claim_qualified=False, production_qualified=False,
        full_fitting_feasibility_qualified=False, generated_unit_samples=False)
    write_json(output / "analysis.json", report)
    try:
        config = yaml.safe_load(config_path.read_text())
        _registered_config(config)
        admitted = _verify_input_manifest(inputs, manifest_path)
        source_names = tuple(dict.fromkeys(SOURCES + (
            "population_residual_mean_validation.py", "population_residual_mean_objective.py",
            "native_renewal_expectation.py", "poisson_trajectory_evaluation.py",
            "population_native_mean_benchmark.py")))
        shared = dict(config=config, config_sha256=sha256_file(config_path),
            input_manifest_sha256=sha256_file(manifest_path), input_scope=admitted["scope"],
            input_blocks=admitted["blocks"],
            sources={name: sha256_file(Path(__file__).with_name(name)) for name in source_names})
        started = time.perf_counter()
        normalization = fit_field_mean_normalization(inputs, config)
        shared["field_mean_normalization"] = normalization
        report.update(binding=shared, normalization_seconds=time.perf_counter() - started)
        protocol = dict(training_seeds=list(SEEDS), fitting_blocks=list(range(14)),
            excluded_selection_blocks=list(range(14, 18)), templates_per_block=4,
            episode_bins=1200, bin_ms=5, source_units=100000)
        for seed in SEEDS:
            parent, batch, coordinates, preparation_error = None, None, None, None
            binding = dict(shared, seed=seed)
            started = time.perf_counter()
            try:
                parent, legacy_batch, source_binding = load_seed(inputs, seed)
                del legacy_batch
                parent_path = inputs / f"parent/seed_{seed}/training/best.pt"
                checkpoint = torch.load(parent_path, map_location="cpu", weights_only=True)
                binding.update(parent_checkpoint_sha256=sha256_file(parent_path),
                    parent_binding_sha256=json_digest(checkpoint["binding"]), source_binding=source_binding)
                with tempfile.TemporaryDirectory(prefix="mean-validation-coordinates-") as scratch:
                    X, Y, _, metadata = _coordinates(inputs / "parent", seed, Path(scratch), protocol)
                coordinates = (X, Y, metadata)
                parent.to(device)
                batch = load_batch(inputs, seed, CASE, parent, checkpoint, config, device,
                                   field_mean_normalization=normalization)
                del checkpoint, X, Y, metadata
            except Exception:
                preparation_error = traceback.format_exc()
            binding["shared_seed_preparation_seconds"] = time.perf_counter() - started
            for mode in MODES:
                row = measure_cell(parent, batch, coordinates, config, seed, mode,
                    output / "cells" / f"seed_{seed}_{mode}", binding, execution,
                    preparation_error=preparation_error)
                report["records"].append(row)
                write_json(output / "analysis.json", report)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if row.get("parent_unchanged") is False:
                    preparation_error = "parent changed during the preceding cell"
            del parent, batch, coordinates
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    complete = sum(row["status"] == "COMPLETE" for row in report["records"])
    failed = sum(row["status"] == "FAILED" for row in report["records"])
    report.update(status=("COMPLETE_REAL_MEAN_OBJECTIVE_VALIDATION" if complete == 6 else
                          "PARTIAL_REAL_MEAN_OBJECTIVE_VALIDATION" if complete else
                          "FAILED_REAL_MEAN_OBJECTIVE_VALIDATION"),
        exit_code=0 if complete == 6 else 2, complete_cells=complete, failed_cells=failed,
        unstarted_cells=6 - len(report["records"]), finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    report = validate(args.inputs, args.manifest, args.config, args.output, args.device)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
