"""Fitting-only recurrence contrasts for frozen native residual corrections."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch

from .population_prior_residual_prediction import _coordinates
from .population_prior_residual_diagnostic import _csv
from .population_residual_correction import load_correction
from .population_state import ControlledPopulationState
from .provenance import freeze_json, sha256_file
from .residual_prediction import RidgeModel

SEEDS = [41, 42, 43]
CASE = "train_block_000_familiar_0"
ORIGIN_BIN = 200
STEPS = 100
DRAWS = 16


def scalar(value):
    value = float(value)
    return value if np.isfinite(value) else None


def features(z, u, h, mode):
    control = u.expand(*z.shape[:-2], u.shape[-1])
    parts = [z.flatten(-2), control]
    if mode == "history":
        parts.append(h.flatten(-2))
    return torch.cat(parts, -1)


def affine_check(model, predictor, z, u, h):
    raw = features(z, u, h, model.mode).double().numpy()
    expected = predictor.predict(raw)
    actual = model.correction(z, u, h).flatten(-2).double().numpy()
    difference = actual - expected
    return dict(max_absolute_error=scalar(np.abs(difference).max()),
                relative_l2_error=scalar(np.linalg.norm(difference) /
                    max(np.linalg.norm(expected), np.finfo(np.float64).tiny)),
                reference_rms=scalar(np.sqrt(np.square(expected).mean())),
                actual_rms=scalar(np.sqrt(np.square(actual).mean())))


@torch.no_grad()
def diagnose(campaign, corrections, diagnostic_path, output):
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS")
            or (os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION"))
            not in (("cpu03", "gpu"), ("cpu04", "cpu"))):
        raise ValueError("CPU-only admitted Slurm allocation required")
    campaign, corrections, diagnostic_path, output = map(Path, (campaign, corrections, diagnostic_path, output))
    diagnostic = json.loads(diagnostic_path.read_text())
    fit = json.loads((corrections / "fit.json").read_text())
    if fit["status"] != "FITTED_ALL_NATIVE_RESIDUAL_CORRECTIONS" or fit["config"]["training_seeds"] != SEEDS:
        raise ValueError("complete three-seed correction fit required")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    rows, runs, sources, probes = [], [], {}, {}
    for seed in SEEDS:
        with tempfile.TemporaryDirectory(prefix=f"residual-recursion-{seed}-") as name:
            scratch = Path(name)
            X, Y, _, source = _coordinates(campaign, seed, scratch, diagnostic["config"])
            if source != diagnostic["consumed"][str(seed)]:
                raise ValueError("fitting coordinates differ from accepted residual diagnostic")
            sources[str(seed)] = source
            checkpoint = torch.load(scratch / "training/best.pt", weights_only=True, map_location="cpu")
            parent = ControlledPopulationState(**checkpoint["model_configuration"])
            parent.load_state_dict(checkpoint["model"], strict=True)
            parent.eval()
            cache_name = f"training/posterior_cache/{CASE}.pt"
            cached = scratch / "boundary.pt"
            shutil.copyfile(campaign / f"seed_{seed}" / cache_name, cached)
            if sha256_file(cached) != source["files"][cache_name]:
                raise ValueError("fitting boundary cache changed")
            cache = torch.load(cached, weights_only=True, map_location="cpu")
            mean, variance, controls = (cache[key] for key in ("mean", "variance", "controls"))
            boundary_mean, boundary_variance = mean[:, ORIGIN_BIN-1], variance[:, ORIGIN_BIN-1]
            h0 = boundary_mean - parent.transition_mean(mean[:, ORIGIN_BIN-2], controls[:, ORIGIN_BIN-1])
            future = controls[:, ORIGIN_BIN:ORIGIN_BIN+STEPS]
            if len(parent.unit_ids) != 100000 or future.shape[1] != STEPS:
                raise ValueError("full-unit complete fitting episode required")
            process = parent._field_parameters()[3].exp()
            bounds = dict(low=torch.from_numpy(X.min(axis=(0, 1, 2))),
                          high=torch.from_numpy(X.max(axis=(0, 1, 2))))
            models, predictors = {}, {}
            probes[str(seed)] = dict(initial_variance_min=scalar(boundary_variance.min()),
                initial_variance_max=scalar(boundary_variance.max()),
                process_variance_min=scalar(process.min()), process_variance_max=scalar(process.max()),
                source_units=len(parent.unit_ids), source_fields=len(parent.field_names))
            for mode in ("current", "history"):
                directory = corrections / f"seed_{seed}" / mode
                models[mode] = load_correction(directory, parent, checkpoint["binding"], source["files"]["training/best.pt"])
                with np.load(directory / "predictor.npz", allow_pickle=False) as saved:
                    predictors[mode] = RidgeModel(alpha=.0001, **{key: saved[key].copy() for key in saved.files})
                teacher = torch.from_numpy(X[0, 0, :128]).to(boundary_mean)
                width = len(parent.field_names) * parent.latent_channels
                probes[str(seed)][mode] = dict(
                    teacher_affine=affine_check(models[mode], predictors[mode],
                        teacher[:, :width].reshape(-1, len(parent.field_names), parent.latent_channels),
                        teacher[:, width:source["current_width"]],
                        teacher[:, source["current_width"]:].reshape(-1, len(parent.field_names), parent.latent_channels)),
                    raw_weight_absolute_max=scalar(models[mode].correction_weight.abs().max()),
                    input_scale_min=scalar(predictors[mode].x_scale.min()),
                    input_scale_max=scalar(predictors[mode].x_scale.max()))
            del X, Y
            for method in ("parent", "current", "history"):
                for regime, initial_on, process_on in (
                    ("deterministic", False, False), ("initial_only", True, False),
                    ("process_only", False, True), ("both", True, True)):
                    initial_rng = torch.Generator().manual_seed(41)
                    dynamics_rng = torch.Generator().manual_seed(42)
                    # Draw both streams even when disabled, preserving paired stream positions.
                    initial_draw = torch.randn((DRAWS, *boundary_mean.shape), generator=initial_rng)
                    z = boundary_mean + boundary_variance.sqrt() * initial_draw * int(initial_on)
                    h = h0.expand(DRAWS, -1, -1, -1)
                    envelope_mode = "current" if method == "parent" else method
                    predictor = predictors[envelope_mode]
                    width = predictor.x_mean.size
                    scale = torch.from_numpy(predictor.x_scale)
                    run = dict(seed=seed, method=method, regime=regime, first_departure=None,
                               first_invalid_hazard=None, completed_steps=0, dtype=str(z.dtype))
                    for step in range(STEPS):
                        u = future[:, step]
                        vector = features(z, u, h, envelope_mode).double()
                        outside = (vector < bounds["low"][:width]) | (vector > bounds["high"][:width])
                        if run["first_departure"] is None and outside.any():
                            gap = torch.maximum(bounds["low"][:width]-vector, vector-bounds["high"][:width])
                            departure = dict(step=step+1, outside_feature_values=int(outside.sum()),
                                             largest_standardized_excess=scalar((gap / scale).max()))
                            if method != "parent":
                                departure["affine"] = affine_check(models[method], predictors[method], z, u, h)
                            run["first_departure"] = departure
                        base = parent.transition_mean(z, u)
                        correction = torch.zeros_like(base) if method == "parent" else models[method].correction(z, u, h)
                        noise = process.sqrt() * torch.randn(z.shape, generator=dynamics_rng)
                        z = base + correction + noise * int(process_on)
                        h = z - base
                        log_hazard = parent.log_hazard(z)
                        hazard = log_hazard.exp() * parent.tick_seconds
                        valid = bool(torch.isfinite(hazard).all() and torch.all(hazard > 0))
                        row = dict(seed=seed, method=method, regime=regime, step=step+1,
                            latent_abs_max=scalar(z.abs().max()), base_abs_max=scalar(base.abs().max()),
                            correction_abs_max=scalar(correction.abs().max()),
                            log_hazard_min=scalar(log_hazard.min()), log_hazard_max=scalar(log_hazard.max()),
                            latent_nonfinite=int((~torch.isfinite(z)).sum()),
                            log_hazard_nonfinite=int((~torch.isfinite(log_hazard)).sum()),
                            hazard_nonfinite=int((~torch.isfinite(hazard)).sum()),
                            hazard_zero_or_negative=int((hazard <= 0).sum()), hazard_valid=valid)
                        rows.append(row)
                        if not valid:
                            run["first_invalid_hazard"] = row
                            break
                        run["completed_steps"] = step+1
                    run["status"] = "COMPLETE_FINITE_RECURSION" if run["completed_steps"] == STEPS else "INVALID_NATIVE_HAZARD"
                    runs.append(run)
                    print(json.dumps(run, allow_nan=False), flush=True)
            del models, predictors, parent, checkpoint, cache
    _csv(output / "trajectory_by_step.csv", rows)
    result = dict(status="COMPLETE_FITTING_RECURSION_DIAGNOSTIC", case_id=CASE, origin_ms=1000,
        horizon_ms=500, steps=STEPS, draws=DRAWS, training_seeds=SEEDS, noise_stream_seeds=[41, 42],
        paired_draws_across_methods_and_regimes=True, original_process_and_boundary_variances=True,
        source_units=100000, selection_data_accessed=False, observation_sampling_performed=False,
        consumed=sources, probes=probes, runs=runs, fit_sha256=sha256_file(corrections / "fit.json"),
        fitting_diagnostic_sha256=sha256_file(diagnostic_path), job_id=os.environ["SLURM_JOB_ID"],
        elapsed_seconds=time.perf_counter()-started, scientific_claim_qualified=False,
        limits="One fitting case and causal posterior coordinates; finite recursion is not fidelity or physical innovation identification.",
        implementation_sha256=sha256_file(Path(__file__)),
        artifacts={"trajectory_by_step.csv": sha256_file(output / "trajectory_by_step.csv")})
    freeze_json(output / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "corrections", "diagnostic", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.campaign, args.corrections, args.diagnostic, args.output)


if __name__ == "__main__":
    main()
