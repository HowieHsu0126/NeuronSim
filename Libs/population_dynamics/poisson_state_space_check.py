"""Engineering entry point for the controlled shared-latent count baseline."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time
from typing import Any

import h5py
import torch

from .baseline_data import load_engineering_windows
from .poisson_state_space import ControlledPoissonLDS, LatentRandomStreams
from .provenance import freeze_json, runtime_manifest, sha256_file


def check_poisson_state_space(source_dir: Path, output_dir: Path, config: dict[str, Any],
                               command: list[str], seed: int) -> tuple[dict[str, Any], int]:
    """Fit the existing two-unit anchor fixture, save draws and replayable moments.

    No validation selection, formal training, or scientific qualification occurs.
    Input paths use the same source-checked reader as the other two baselines.
    """
    if output_dir.exists():
        raise FileExistsError(f"POISSON_LDS_CHECK_OUTPUT_EXISTS:{output_dir}")
    if seed not in config["training_seeds"]:
        raise ValueError("UNREGISTERED_TRAINING_SEED")
    spec = config["poisson_state_space_check"]
    for name in ("optimizer_steps", "n_samples"):
        if type(spec[name]) is not int or spec[name] < 1:
            raise ValueError(f"POSITIVE_INTEGER_REQUIRED:{name}")
    if type(spec["learning_rate"]) not in (int, float) or not math.isfinite(spec["learning_rate"]) or spec["learning_rate"] <= 0:
        raise ValueError("POSITIVE_LEARNING_RATE_REQUIRED")
    streams = LatentRandomStreams.from_seeds(**spec["sampling_seeds"])
    data = load_engineering_windows(source_dir, config)
    counts, inputs = torch.as_tensor(data.counts, dtype=torch.float64), torch.as_tensor(data.inputs, dtype=torch.float64)
    model = ControlledPoissonLDS(data.unit_ids, inputs.shape[-1], spec["latent_size"], spec["encoder_size"],
                                data.bin_seconds, initialization_seed=seed).double()
    # Training-only rate initialization; this pseudocount never changes the likelihood.
    initialization_pseudocount = 0.5
    with torch.no_grad():
        total_seconds = counts.shape[0] * counts.shape[1] * data.bin_seconds
        model.log_rate_bias.copy_(((counts.sum((0, 1)) + initialization_pseudocount) / total_seconds).log())
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
    output_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        fit_start = time.monotonic()
        with (output_dir / "optimization.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "loss", "expected_observation_nll",
                                                       "initial_kl", "transition_kl", "gradient_norm"])
            writer.writeheader(); stream.flush()
            for step in range(spec["optimizer_steps"]):
                optimizer.zero_grad()
                terms = model.sequence_loss(counts, inputs)
                terms["loss"].backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if any(not torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("NONFINITE_OPTIMIZER_GRADIENT")
                writer.writerow({"step": step, **{name: value.item() for name, value in terms.items()},
                                 "gradient_norm": float(torch.sqrt(sum(g.square().sum() for g in gradients)))})
                stream.flush()
                optimizer.step()
        fit_seconds = time.monotonic() - fit_start
        with torch.no_grad():
            final_terms = {name: value.item() for name, value in model.sequence_loss(counts, inputs).items()}
            inference_start = time.monotonic()
            history = {"counts": counts[:, :data.history_bins], "stimulus": inputs[:, :data.history_bins]}
            state = model.infer_state(history, {"unit_ids": data.unit_ids, "bin_seconds": data.bin_seconds})
            inference_seconds = time.monotonic() - inference_start
            future = inputs[:, data.history_bins:]
            prediction_start = time.monotonic()
            moments = model.forecast_moments(state, future)
            moment_seconds = time.monotonic() - prediction_start
            sampling_start = time.monotonic()
            draws = model.simulate(state, future, streams, spec["n_samples"])
            sampling_seconds = time.monotonic() - sampling_start
        model.save(output_dir / "poisson_state_space.pt")
        with h5py.File(output_dir / "count_samples.h5", "x") as recording:
            for name, value in {"counts": draws.counts, "conditional_expected_counts": draws.conditional_expected_counts,
                                "latent_states": draws.latent_states, "initial_latent_states": draws.initial_latent_states,
                                "boundary_mean": state.mean, "boundary_variance": state.variance,
                                "forecast_mean": moments.mean, "forecast_covariance": moments.covariance,
                                "marginal_expected_counts": moments.expected_counts}.items():
                recording.create_dataset(name, data=value.detach().cpu().numpy())
            recording.create_dataset("unit_ids", data=data.unit_ids)
            recording.create_dataset("trial_ids", data=data.trial_ids, dtype=h5py.string_dtype())
            recording.attrs["bin_seconds"] = data.bin_seconds
            recording.attrs["history_bins"] = data.history_bins
            recording.attrs["scope"] = "engineering_only"
        if data.input_features is not None:
            freeze_json(output_dir / "input_features.json", data.input_features)
        summary = {"status": "PARTIAL", "scope": "engineering_only", "model": "controlled_poisson_lds",
                   "training_seed": seed, "formal_seed_complete": False,
                   "unit_ids": list(data.unit_ids), "trial_count": len(data.trial_ids),
                   "known_input_channels": inputs.shape[-1], "latent_size": model.latent_size,
                   "parameter_count": sum(p.numel() for p in model.parameters()),
                   "generated_shape": list(draws.counts.shape), "final_training_terms": final_terms,
                   "loss_normalization": "sum over units and latent dimensions divided by batch*time",
                   "posterior_family": "causal encoder, Gaussian diagonal per time, joint factorized over time",
                   "sampling_seeds": spec["sampling_seeds"],
                   "elapsed_seconds": time.monotonic() - start,
                   "timings_seconds": {"fit": fit_seconds, "state_inference": inference_seconds,
                                       "analytic_forecast": moment_seconds, "trajectory_sampling": sampling_seconds},
                   "generative_fidelity_qualified": False, "inverse_intervention_qualified": False,
                   "scientific_claims_qualified": False}
        freeze_json(output_dir / "summary.json", summary)
        names = ["poisson_state_space.pt", "count_samples.h5", "optimization.csv", "summary.json"]
        if data.input_features is not None:
            names.append("input_features.json")
        freeze_json(output_dir / "manifest.json", {
            "schema": "neural_population_poisson_lds_check_v1", "scope": "engineering_only",
            "source_manifest": str((source_dir / "manifest.json").resolve()),
            "source_manifest_sha256": sha256_file(source_dir / "manifest.json"),
            "started_at_utc": started_at, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed, "initialization_pseudocount": initialization_pseudocount,
            "torch_version": str(torch.__version__), "dtype": "float64", "resumed": False,
            "artifacts": {name: sha256_file(output_dir / name) for name in names},
            **runtime_manifest(config, command, 2)})
    except Exception as exc:
        freeze_json(output_dir / "failure.json", {
            "status": "FAILED", "scope": "engineering_only", "error_type": type(exc).__name__,
            "error": str(exc), "seed": seed, "started_at_utc": started_at,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(), **runtime_manifest(config, command, 1)})
        raise
    return {**summary, "manifest": str(output_dir / "manifest.json")}, 2
