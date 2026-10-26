"""Measure frozen full-unit renewal means and a diagnostic scalar gradient.

The scalar is total expected spikes per unit averaged over eight latent paths.
It is a gradient probe, not an admitted fitting objective or a fidelity result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch
import yaml

from .native_renewal_data import NativeRenewalEpisode
from .native_renewal_expectation import expected_field_counts
from .population_prior_objective import _validated_prior_inputs
from .population_residual_observation import CORRECTION_PARAMETERS, configure_correction_parameters
from .population_residual_observation_fitting import load_native_observation_correction, validate_config
from .population_residual_observation_validation import CASE, SEEDS, load_seed
from .population_residual_refit_provenance import build_refit_receipt
from .provenance import freeze_json, sha256_file
from .renewal_trajectory_training import variational_seed

MODES = ("current", "history")
ORIGIN, HORIZON, PARTICLES, UNIT_CHUNK_SIZE = 200, 100, 8, 4096
SOURCE_UNITS, SOURCE_FIELDS, CHANNELS = 100000, 128, 4
SCALAR = "total_expected_spikes_per_unit_averaged_over_latent_particles"


def free_residual_path(model, mean, variance, controls, initial, dynamics, *, origin=ORIGIN, horizon=HORIZON,
                       particles=PARTICLES):
    """Use the frozen native residual rollout, without observation labels."""
    if type(origin) is not int or origin < 2 or type(particles) is not int or particles < 1:
        raise ValueError("two-bin causal boundary and positive latent particle count required")
    parent = model.parent
    mean, variance, controls, _, _, process_variance = _validated_prior_inputs(
        parent, mean, variance, controls, (origin,), (horizon,), (initial, dynamics),
        stream_names="initial and dynamics")
    if mean.shape[0] != 1:
        raise ValueError("one fixed fitting episode required")
    boundary, previous = mean[:, origin - 1], mean[:, origin - 2]
    shape = (particles, *boundary.shape)
    memory = (boundary - parent.transition_mean(previous, controls[:, origin - 1])).expand(shape)
    state = boundary + variance[:, origin - 1].sqrt() * torch.randn(
        shape, dtype=mean.dtype, device=mean.device, generator=initial)
    states = []
    for step in range(horizon):
        control = controls[:, origin + step]
        base = parent.transition_mean(state, control)
        state = base + model.correction(state, control, memory) + process_variance.sqrt() * torch.randn(
            shape, dtype=mean.dtype, device=mean.device, generator=dynamics)
        memory = state - base
        states.append(state)
    return torch.stack(states, dim=2)[:, 0]


def _snapshot(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _unchanged(module, before):
    current = module.state_dict()
    return set(current) == set(before) and all(torch.equal(current[name].detach().cpu(), value)
                                               for name, value in before.items())


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory(device):
    if device.type != "cuda":
        return dict(peak_allocated_bytes=None, peak_reserved_bytes=None)
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def _cell_identity(seed, mode, execution):
    return dict(schema="native_renewal_mean_benchmark_cell_v1", seed=seed, mode=mode, case_id=CASE,
        origin_bins=ORIGIN, horizon_bins=HORIZON, horizon_ms=500, particles=PARTICLES,
        requested_source_units=SOURCE_UNITS, requested_fields=SOURCE_FIELDS, channels=CHANNELS,
        unit_chunk_size=UNIT_CHUNK_SIZE, checkpoint_readout=True, scalar=SCALAR,
        admitted_training_objective=False, optimizer_steps=0, generated_unit_samples=False,
        future_observed_exposure_used=False, scientific_claim_qualified=False, execution=execution,
        parent_unchanged=None, correction_unchanged=None, gradient_finite=None,
        started_utc=datetime.now(timezone.utc).isoformat())


def _finish_cell(directory, report):
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    freeze_json(directory / "analysis.json", report)
    print(json.dumps({name: report.get(name) for name in
        ("seed", "mode", "status", "exit_code", "forward_seconds", "backward_seconds", "gradient_norm", "error")}), flush=True)
    return report


def measure_cell(parent, coordinates, ready, run, refits, directory, seed, mode, execution, source_binding):
    directory.mkdir(parents=True)
    report = _cell_identity(seed, mode, execution)
    report.update(parent_checkpoint_sha256=run["best_checkpoint_sha256"], source_binding=source_binding,
        timing_scope="free latent rollout plus checkpointed deterministic readout; prefix loading and artifact I/O excluded")
    device = parent.unit_loading.device
    model, before, parent_before = None, None, None
    phase = "state_snapshot"
    try:
        parent_before = _snapshot(parent)
        phase = "model_load"
        model = load_native_observation_correction(refits / f"seed_{seed}" / mode,
            parent, run["binding"], run["best_checkpoint_sha256"]).to(device).eval()
        parameters = configure_correction_parameters(model)
        before = _snapshot(model)
        report["correction_identity"] = model.correction_identity
        base_seed = run["binding"]["config"]["training_variational_seed_base"]
        initial_seed = variational_seed(base_seed, seed, "native_mean_benchmark_initial", CASE, 0)
        dynamics_seed = variational_seed(base_seed, seed, "native_mean_benchmark_dynamics", CASE, 0)
        initial = torch.Generator(device=device).manual_seed(initial_seed)
        dynamics = torch.Generator(device=device).manual_seed(dynamics_seed)
        report.update(initial_seed=initial_seed, dynamics_seed=dynamics_seed,
                      stream_scope="named initial/process streams shared across correction modes; replicate index zero")
        _synchronize(device)
        if device.type == "cuda":
            report["allocated_before_forward_bytes"] = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        phase = "forward"
        started = time.perf_counter()
        trajectory = free_residual_path(model, **coordinates, initial=initial, dynamics=dynamics)
        if trajectory.shape != (PARTICLES, HORIZON, SOURCE_FIELDS, CHANNELS):
            raise ValueError("complete registered free latent path required")
        fields = expected_field_counts(parent, trajectory, ready, unit_chunk_size=UNIT_CHUNK_SIZE, checkpoint_readout=True)
        scalar = fields.sum() / (PARTICLES * SOURCE_UNITS)
        _synchronize(device)
        report.update(forward_seconds=time.perf_counter() - started, forward_memory=_peak_memory(device),
            scalar_value=float(scalar.detach()), readout_units=len(parent.unit_ids), fields_covered=fields.shape[-1],
            expected_field_shape=list(fields.shape), unit_chunks=(SOURCE_UNITS + UNIT_CHUNK_SIZE - 1) // UNIT_CHUNK_SIZE)
        path = directory / "expected_fields.npz"
        np.savez_compressed(path, expected_field_counts=fields.detach().cpu().numpy(),
            field_ids=np.asarray(parent.field_names),
            unit_count=torch.bincount(parent.unit_field_index, minlength=SOURCE_FIELDS).cpu().numpy(),
            origin_bin=np.asarray(ORIGIN), bin_ms=np.asarray(parent.bin_seconds * 1000),
            particle_indices=np.arange(PARTICLES))
        report["artifacts"] = {path.name: dict(sha256=sha256_file(path), bytes=path.stat().st_size,
            units="expected_spikes_per_model_bin", shape=list(fields.shape))}
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        phase = "backward"
        started = time.perf_counter()
        gradients = torch.autograd.grad(scalar, parameters)
        _synchronize(device)
        report.update(backward_seconds=time.perf_counter() - started, backward_memory=_peak_memory(device))
        if any(not torch.isfinite(value).all() for value in gradients):
            raise FloatingPointError("nonfinite expected-spike gradient probe")
        if any(parameter.grad is not None for parameter in parent.parameters()):
            raise RuntimeError("gradient probe populated a frozen parent gradient")
        norms = {name: float(value.detach().double().norm()) for name, value in zip(CORRECTION_PARAMETERS, gradients)}
        norm = float(torch.stack([value.detach().double().square().sum() for value in gradients]).sum().sqrt())
        report.update(status="COMPLETE", exit_code=0, gradient_finite=True, gradient_norm=norm,
                      gradient_nonzero=norm > 0, parameter_gradient_norms=norms)
    except Exception as error:
        report.update(status="FAILED", exit_code=2, failed_phase=phase, error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        if phase in ("forward", "backward"):
            try:
                report["failure_memory"] = _peak_memory(device)
            except RuntimeError as memory_error:
                report["failure_memory_error"] = str(memory_error)
    try:
        report["parent_unchanged"] = _unchanged(parent, parent_before)
        if model is not None and before is not None:
            report["correction_unchanged"] = _unchanged(model, before)
        if report["parent_unchanged"] is not True or report["correction_unchanged"] is not True:
            raise RuntimeError("parent or correction state changed or could not be verified")
    except Exception as error:
        report.update(status="FAILED", exit_code=2, state_verification_error=str(error))
    return _finish_cell(directory, report)


def benchmark(inputs, refits, config_path, output, device="cuda"):
    if (platform.node().split(".")[0] == "mgmt2" or not os.environ.get("RH2_JOB_RUN_ID")
            or not os.environ.get("ARNOLD_TRIAL_ID")):
        raise ValueError("native mean benchmark requires actual Merlin execution")
    inputs, refits, config_path, output = map(Path, (inputs, refits, config_path, output))
    if output.exists():
        raise FileExistsError("fresh native mean benchmark output required")
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    family = build_refit_receipt(refits)
    manifest = json.loads((inputs / "manifest.json").read_text())
    if (manifest.get("scope") != "complete_fitting_and_internal_selection"
            or manifest.get("training_seeds") != list(SEEDS) or manifest.get("blocks") != list(range(14))
            or family["shared_binding"]["input_manifest_sha256"] != sha256_file(inputs / "manifest.json")
            or family["shared_binding"]["config_sha256"] != sha256_file(config_path)
            or family["config"] != config or config["unit_chunk_size"] != UNIT_CHUNK_SIZE):
        raise ValueError("complete native refits with original fitting inputs/configuration and4096-unit chunks required")
    for name, digest in family["shared_binding"]["sources"].items():
        if Path(name).name != name or sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError("frozen native fitting source differs: " + name)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device(device)
    execution = dict(backend="merlin", run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ["ARNOLD_TRIAL_ID"],
        host=platform.node(), device=str(device), torch_version=torch.__version__, cuda_version=torch.version.cuda)
    if device.type == "cuda":
        execution.update(gpu_name=torch.cuda.get_device_name(device),
                         gpu_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    output.mkdir(parents=True)
    records = []
    for seed in SEEDS:
        parent, coordinates, ready = None, None, None
        try:
            parent, batch, source_binding = load_seed(inputs, seed)
            run = json.loads((inputs / f"parent/seed_{seed}/training/run.json").read_text())
            if list(batch["origins"]) != config["origins"] or list(batch["horizons"]) != config["horizons"]:
                raise ValueError("registered fitting origins and horizons differ")
            source = NativeRenewalEpisode(inputs / f"cohort/train/{CASE}", inputs / f"cohort/train_fields/{CASE}",
                expected_source=run["binding"]["native_source_identity"], expected_unit_ids=np.asarray(parent.unit_ids),
                expected_unit_field_index=parent.unit_field_index.numpy())
            if source.binding != source_binding:
                raise ValueError("causal readiness source differs from the admitted fitting episode")
            ready = torch.from_numpy(source.ready_at(ORIGIN * source.bin_ticks)).to(device)
            # load_seed authenticates its legacy label windows. This benchmark
            # does not pass those future labels or eligible exposures to the path.
            coordinates = {name: batch[name].to(device) for name in ("mean", "variance", "controls")}
            del source, batch
            parent.to(device)
        except Exception as error:
            for mode in MODES:
                directory = output / "cells" / f"seed_{seed}_{mode}"
                directory.mkdir(parents=True)
                record = _cell_identity(seed, mode, execution)
                record.update(status="FAILED", exit_code=2, error_type=type(error).__name__, error=str(error),
                              traceback=traceback.format_exc())
                records.append(_finish_cell(directory, record))
        else:
            for mode in MODES:
                records.append(measure_cell(parent, coordinates, ready, run, refits,
                    output / "cells" / f"seed_{seed}_{mode}", seed, mode, execution, source_binding))
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del parent, coordinates, ready
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    complete = sum(row["status"] == "COMPLETE" and row["exit_code"] == 0 for row in records)
    report = dict(status="COMPLETE_NATIVE_RENEWAL_MEAN_BENCHMARK" if complete == 6 else "PARTIAL_NATIVE_RENEWAL_MEAN_BENCHMARK",
        exit_code=0 if complete == 6 else 2, training_seeds=list(SEEDS), modes=list(MODES), complete_cells=complete,
        failed_cells=6 - complete, case_id=CASE, origin_bins=ORIGIN, horizon_bins=HORIZON, particles=PARTICLES,
        source_units=SOURCE_UNITS, fields=SOURCE_FIELDS, scalar=SCALAR, admitted_training_objective=False,
        optimizer_steps=0, generated_unit_samples=False, future_observed_exposure_used=False,
        input_manifest_sha256=family["shared_binding"]["input_manifest_sha256"], config_sha256=sha256_file(config_path),
        benchmark_sha256=sha256_file(Path(__file__)),
        expectation_sha256=sha256_file(Path(__file__).with_name("native_renewal_expectation.py")),
        refit_receipt=family, execution=execution, scientific_claim_qualified=False,
        full_fitting_feasibility_qualified=False, records=records,
        cell_receipts={f"seed_{row['seed']}_{row['mode']}": sha256_file(
            output / "cells" / f"seed_{row['seed']}_{row['mode']}" / "analysis.json") for row in records})
    freeze_json(output / "analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "refits", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    report = benchmark(args.inputs, args.refits, args.config, args.output, args.device)
    print(json.dumps({name: report[name] for name in ("status", "exit_code", "complete_cells", "failed_cells")}))
    raise SystemExit(report["exit_code"])


if __name__ == "__main__":
    main()
