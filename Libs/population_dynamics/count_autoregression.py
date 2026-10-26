"""Known-input Poisson autoregression: a transparent count-generation baseline.

The conditional log rate is linear in preceding counts and the current known
input. Future samples are fed back into history. Instantaneous counts are
conditionally independent; temporal coupling can induce cross-unit dependence.
This baseline has no hidden-state posterior or instantaneous shared innovation.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor, nn

from .baseline_data import load_engineering_windows
from .provenance import freeze_json, runtime_manifest, sha256_file


@dataclass(frozen=True)
class CountHistoryState:
    counts: Tensor  # batch, history bin, unit
    unit_ids: tuple[int, ...]
    uncertainty: str = "point_mass_conditional_on_complete_observed_history"


@dataclass(frozen=True)
class CountSamples:
    counts: Tensor  # draw, batch, forecast bin, unit; integer spike counts
    conditional_expected_counts: Tensor  # conditional on each sampled past
    unit_ids: tuple[int, ...]
    bin_seconds: float

    @property
    def population_counts(self) -> Tensor:
        return self.counts.sum(dim=-1)


class PoissonAutoregression(nn.Module):
    def __init__(self, unit_ids: Sequence[int], input_size: int, history_bins: int,
                 bin_seconds: float, *, initialization_seed: int):
        super().__init__()
        ids = tuple(unit_ids)
        if not ids or any(type(u) is not int or u < 0 for u in ids) or len(set(ids)) != len(ids):
            raise ValueError("UNIQUE_NONNEGATIVE_INTEGER_UNIT_IDS_REQUIRED")
        if type(input_size) is not int or input_size < 0 or type(history_bins) is not int or history_bins < 1:
            raise ValueError("INVALID_INPUT_OR_HISTORY_SIZE")
        if isinstance(bin_seconds, bool) or not math.isfinite(bin_seconds) or bin_seconds <= 0:
            raise ValueError("POSITIVE_BIN_SECONDS_REQUIRED")
        if type(initialization_seed) is not int or initialization_seed < 0:
            raise ValueError("NONNEGATIVE_INITIALIZATION_SEED_REQUIRED")
        self.unit_ids = ids
        self.input_size = input_size
        self.history_bins = history_bins
        self.bin_seconds = float(bin_seconds)
        self.initialization_seed = initialization_seed
        self.history_weight = nn.Parameter(torch.empty(len(ids), history_bins * len(ids)))
        self.input_weight = nn.Parameter(torch.empty(len(ids), input_size))
        self.log_rate_bias = nn.Parameter(torch.zeros(len(ids)))
        rng = torch.Generator(device="cpu").manual_seed(initialization_seed)
        nn.init.normal_(self.history_weight, std=0.01, generator=rng)
        nn.init.normal_(self.input_weight, std=0.01, generator=rng)

    def _tensor(self, values: Any) -> Tensor:
        return torch.as_tensor(values, dtype=self.log_rate_bias.dtype, device=self.log_rate_bias.device)

    @staticmethod
    def _counts_valid(values: Tensor) -> None:
        if not torch.isfinite(values).all() or torch.any(values < 0) or torch.any(values != values.round()):
            raise ValueError("FINITE_NONNEGATIVE_INTEGER_COUNTS_REQUIRED")

    def infer_state(self, history: Any, metadata: dict[str, Any] | None = None) -> CountHistoryState:
        values = self._tensor(history)
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if values.ndim != 3 or values.shape[1] < self.history_bins or values.shape[2] != len(self.unit_ids):
            raise ValueError("HISTORY_SHAPE_MISMATCH")
        self._counts_valid(values)
        if metadata is not None and tuple(metadata.get("unit_ids", self.unit_ids)) != self.unit_ids:
            raise ValueError("HISTORY_UNIT_ORDER_MISMATCH")
        return CountHistoryState(values[:, -self.history_bins:].detach().clone(), self.unit_ids)

    def conditional_log_expected_counts(self, history: Tensor, stimulus: Tensor) -> Tensor:
        """Exact one-step conditional mean in counts/bin, rather than Hz."""
        if history.ndim != 3 or history.shape[1:] != (self.history_bins, len(self.unit_ids)):
            raise ValueError("HISTORY_SHAPE_MISMATCH")
        if stimulus.shape != (history.shape[0], self.input_size) or not torch.isfinite(stimulus).all():
            raise ValueError("KNOWN_STIMULUS_SHAPE_OR_VALUES_INVALID")
        if not torch.isfinite(history).all():
            raise ValueError("NONFINITE_HISTORY")
        log_rate = (history.flatten(1) @ self.history_weight.T
                    + stimulus @ self.input_weight.T + self.log_rate_bias)
        result = log_rate + math.log(self.bin_seconds)
        if not torch.isfinite(result).all():
            raise FloatingPointError("NONFINITE_CONDITIONAL_LOG_MEAN")
        return result

    def sequence_loss(self, counts: Any, stimulus: Any, *, l2_weight: float = 0.) -> dict[str, Tensor]:
        """Teacher-forced likelihood; each target sees earlier counts only.

        counts/stimulus have matching (batch,time,feature) dimensions, including
        the leading history. The current known stimulus drives the current count.
        The full factorial term is exact, so the likelihood is a proper count score.
        """
        y, u = self._tensor(counts), self._tensor(stimulus)
        if y.ndim != 3 or y.shape[1] <= self.history_bins or y.shape[2] != len(self.unit_ids):
            raise ValueError("COUNT_SEQUENCE_SHAPE_MISMATCH")
        if u.shape != (*y.shape[:2], self.input_size):
            raise ValueError("STIMULUS_SEQUENCE_SHAPE_MISMATCH")
        self._counts_valid(y)
        if not math.isfinite(l2_weight) or l2_weight < 0:
            raise ValueError("NONNEGATIVE_L2_WEIGHT_REQUIRED")
        # unfold appends the window dimension; permute to history-bin then unit.
        windows = y.unfold(1, self.history_bins + 1, 1).permute(0, 1, 3, 2)
        past = windows[:, :, :-1].reshape(-1, self.history_bins, len(self.unit_ids))
        targets = windows[:, :, -1].reshape(-1, len(self.unit_ids))
        target_count = y.shape[0] * (y.shape[1] - self.history_bins)
        log_mu = self.conditional_log_expected_counts(past, u[:, self.history_bins:].reshape(target_count, self.input_size))
        nll = (log_mu.exp() - targets * log_mu + torch.lgamma(targets + 1)).mean()
        penalty = self.history_weight.square().sum() + self.input_weight.square().sum()
        loss = nll + l2_weight * penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("NONFINITE_POISSON_LOSS")
        return {"loss": loss, "poisson_nll": nll, "weight_penalty": penalty}

    @torch.no_grad()
    def simulate(self, state: CountHistoryState, stimulus: Any, rng: torch.Generator,
                 n_samples: int) -> CountSamples:
        if state.unit_ids != self.unit_ids:
            raise ValueError("STATE_UNIT_ORDER_MISMATCH")
        if type(n_samples) is not int or n_samples < 1:
            raise ValueError("POSITIVE_SAMPLE_COUNT_REQUIRED")
        history = self._tensor(state.counts)
        self._counts_valid(history)
        if history.ndim != 3 or history.shape[1:] != (self.history_bins, len(self.unit_ids)):
            raise ValueError("STATE_HISTORY_SHAPE_MISMATCH")
        batch = history.shape[0]
        u = self._tensor(stimulus)
        if u.ndim == 2:
            u = u.unsqueeze(0).expand(batch, -1, -1)
        if u.ndim != 3 or u.shape[0] != batch or u.shape[1] < 1 or u.shape[2] != self.input_size or not torch.isfinite(u).all():
            raise ValueError("STIMULUS_SEQUENCE_SHAPE_OR_VALUES_INVALID")
        if torch.device(rng.device) != history.device:
            raise ValueError("RNG_AND_MODEL_DEVICE_MUST_MATCH")
        history = history.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(-1, self.history_bins, len(self.unit_ids)).clone()
        inputs = u.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(n_samples * batch, u.shape[1], self.input_size)
        draws, means = [], []
        for time in range(u.shape[1]):
            mu = self.conditional_log_expected_counts(history, inputs[:, time]).exp()
            if not torch.isfinite(mu).all():
                raise FloatingPointError(f"NONFINITE_GENERATION_MEAN_AT_BIN:{time}")
            sampled = torch.poisson(mu, generator=rng)
            if not torch.isfinite(sampled).all() or torch.any(sampled >= torch.iinfo(torch.int64).max):
                raise FloatingPointError(f"COUNT_SAMPLE_OUTSIDE_INT64_AT_BIN:{time}")
            draws.append(sampled.to(torch.int64)); means.append(mu)
            history = torch.cat((history[:, 1:], sampled[:, None]), dim=1)
        shape = (n_samples, batch, u.shape[1], len(self.unit_ids))
        return CountSamples(torch.stack(draws, dim=1).reshape(shape),
                            torch.stack(means, dim=1).reshape(shape), self.unit_ids, self.bin_seconds)

    def save(self, path: Path) -> None:
        """Save an explicit baseline identity and tensors; never overwrite evidence."""
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            torch.save({"schema": "neural_population_poisson_autoregression_v1",
                        "configuration": {"unit_ids": list(self.unit_ids), "input_size": self.input_size,
                                          "history_bins": self.history_bins, "bin_seconds": self.bin_seconds,
                                          "initialization_seed": self.initialization_seed},
                        "state_dict": self.state_dict()}, stream)

    @classmethod
    def load(cls, path: Path, *, device: str | torch.device = "cpu") -> PoissonAutoregression:
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload["schema"] != "neural_population_poisson_autoregression_v1":
            raise ValueError("COUNT_BASELINE_SCHEMA_MISMATCH")
        dtype = payload["state_dict"]["log_rate_bias"].dtype
        model = cls(**payload["configuration"]).to(device=device, dtype=dtype)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


def check_count_baseline(source_dir: Path, output_dir: Path, config: dict[str, Any],
                         command: list[str], seed: int) -> tuple[dict[str, Any], int]:
    """Bounded real-count fit and sampling replay, separate from formal training.

    Source-aligned stimulus features are used when present. Earlier count-only
    artifacts retain their explicitly reported no-input special case.
    """
    if output_dir.exists():
        raise FileExistsError(f"COUNT_CHECK_OUTPUT_EXISTS:{output_dir}")
    if seed not in config["training_seeds"]:
        raise ValueError("UNREGISTERED_TRAINING_SEED")
    spec = config["count_baseline_check"]
    if type(spec["optimizer_steps"]) is not int or spec["optimizer_steps"] < 1:
        raise ValueError("POSITIVE_OPTIMIZER_STEPS_REQUIRED")
    if not math.isfinite(spec["learning_rate"]) or spec["learning_rate"] <= 0:
        raise ValueError("POSITIVE_LEARNING_RATE_REQUIRED")
    data = load_engineering_windows(source_dir, config)
    counts = torch.from_numpy(data.counts).to(dtype=torch.float64)
    inputs = torch.from_numpy(data.inputs).to(dtype=torch.float64)
    ids, trials = list(data.unit_ids), list(data.trial_ids)
    history_count, bin_seconds = data.history_bins, data.bin_seconds
    feature_spec, input_size = data.input_features, data.inputs.shape[2]
    model = PoissonAutoregression(ids, input_size, spec["history_bins"], bin_seconds, initialization_seed=seed).double()
    if history_count < model.history_bins or history_count >= counts.shape[1]:
        raise ValueError("SOURCE_HISTORY_OR_RESPONSE_TOO_SHORT")
    model._counts_valid(counts)
    # Half a count initializes silent units finitely; it is not a likelihood
    # modification, fitted posterior, or post-hoc clipping of predictions.
    initialization_pseudocount = 0.5
    with torch.no_grad():
        observed = counts[:, model.history_bins:]
        total_time = observed.shape[0] * observed.shape[1] * bin_seconds
        model.log_rate_bias.copy_(((observed.sum((0, 1)) + initialization_pseudocount) / total_time).log())
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
    output_dir.mkdir(parents=True, exist_ok=False)
    if feature_spec is not None:
        freeze_json(output_dir / "input_features.json", feature_spec)
    start = time.monotonic()
    rows = []
    for step in range(spec["optimizer_steps"]):
        optimizer.zero_grad()
        values = model.sequence_loss(counts, inputs, l2_weight=spec["l2_weight"])
        values["loss"].backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("NONFINITE_OPTIMIZER_GRADIENT")
        rows.append({"step": step, **{name: value.item() for name, value in values.items()},
                     "gradient_norm": float(torch.sqrt(sum(g.square().sum() for g in gradients)))})
        optimizer.step()
    with (output_dir / "optimization.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    final_nll = model.sequence_loss(counts, inputs)["poisson_nll"].item()
    checkpoint = output_dir / "count_autoregression.pt"
    model.save(checkpoint)
    state = model.infer_state(counts[:, :history_count], {"unit_ids": ids})
    future = inputs[:, history_count:]
    draws = model.simulate(state, future, torch.Generator().manual_seed(spec["sampling_seed"]), spec["n_samples"])
    restored = PoissonAutoregression.load(checkpoint)
    replay = restored.simulate(restored.infer_state(counts[:, :history_count]), future,
                               torch.Generator().manual_seed(spec["sampling_seed"]), spec["n_samples"])
    if not torch.equal(draws.counts, replay.counts) or not torch.equal(draws.conditional_expected_counts, replay.conditional_expected_counts):
        raise ValueError("SAVED_BASELINE_GENERATION_REPLAY_MISMATCH")
    with h5py.File(output_dir / "count_samples.h5", "x") as output:
        output.attrs.update(scope="engineering_only", bin_seconds=bin_seconds,
                            conditional_means_are_not_marginal_forecast_means=True)
        output["counts"] = draws.counts.numpy()
        output["conditional_expected_counts"] = draws.conditional_expected_counts.numpy()
        output["population_counts"] = draws.population_counts.numpy()
        output["unit_ids"] = np.asarray(ids, dtype=np.int64)
        output.create_dataset("trial_ids", data=trials, dtype=h5py.string_dtype("utf-8"))
    summary = {"status": "PARTIAL", "software_check": "PASS", "scope": "engineering_only",
               "method": "poisson_count_autoregression", "seed": seed,
               "training_scope": "training_anchor_engineering_fit", "formal_training_completed": False,
               "unit_count": len(ids), "trial_count": counts.shape[0], "sample_shape": list(draws.counts.shape),
               "optimizer_steps": spec["optimizer_steps"], "initial_poisson_nll": rows[0]["poisson_nll"],
               "final_poisson_nll": final_nll, "parameter_count": sum(p.numel() for p in model.parameters()),
               "known_input_channels": input_size,
               "known_input_validation": ("source-aligned recorded visual conditions and movie-frame occupancy" if input_size
                                            else "separate analytical contract tests only"),
               "sampling_seed": spec["sampling_seed"], "checkpoint_sampling_replay": "PASS",
               "elapsed_seconds": time.monotonic() - start, "generative_fidelity_qualified": False,
               "inverse_intervention_qualified": False, "scientific_claims_qualified": False}
    freeze_json(output_dir / "summary.json", summary)
    artifact_names = ["count_autoregression.pt", "optimization.csv", "count_samples.h5", "summary.json"]
    if feature_spec is not None:
        artifact_names.append("input_features.json")
    freeze_json(output_dir / "manifest.json", {
        "schema": "neural_population_count_baseline_check_v1", "source_manifest": str((source_dir / "manifest.json").resolve()),
        "source_manifest_sha256": sha256_file(source_dir / "manifest.json"), "seed": seed,
        "initialization_pseudocount": initialization_pseudocount, "torch_version": str(torch.__version__), "dtype": "float64",
        "ended_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "engineering_only",
        "artifacts": {name: sha256_file(output_dir / name) for name in artifact_names},
        **runtime_manifest(config, command, 2)})
    return {**summary, "manifest": str(output_dir / "manifest.json")}, 2
