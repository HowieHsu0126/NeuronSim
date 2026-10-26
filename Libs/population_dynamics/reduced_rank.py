"""Known-input ridge/reduced-rank autoregression of binned neural counts.

This is an unconstrained mean-response predictor, not a spike generator. Rank
bounds the joint lag/input coefficient map; the observed state is a lag buffer.
Fits use a thin SVD and retain factors instead of a dense unit-by-feature matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor, nn

from .baseline_data import load_engineering_windows
from .provenance import freeze_json, runtime_manifest, sha256_file


@dataclass(frozen=True)
class RegressionHistoryState:
    counts: Tensor
    unit_ids: tuple[int, ...]
    bin_seconds: float
    uncertainty: str = "point_mass_conditional_on_complete_observed_history"


@dataclass(frozen=True)
class MeanForecast:
    values: Tensor  # batch, forecast bin, unit; real-valued predictions in counts/bin
    unit_ids: tuple[int, ...]
    bin_seconds: float
    prediction_kind: str = "unconstrained_mean_response_prediction_not_spike_samples"


class ReducedRankAutoregression(nn.Module):
    def __init__(self, unit_ids: Sequence[int], input_size: int, history_bins: int,
                 bin_seconds: float, rank: int | None = None):
        super().__init__()
        ids = tuple(unit_ids)
        if not ids or any(type(u) is not int or u < 0 for u in ids) or len(set(ids)) != len(ids):
            raise ValueError("UNIQUE_NONNEGATIVE_INTEGER_UNIT_IDS_REQUIRED")
        if type(input_size) is not int or input_size < 0 or type(history_bins) is not int or history_bins < 1:
            raise ValueError("INVALID_INPUT_OR_HISTORY_SIZE")
        if type(bin_seconds) not in (int, float) or not math.isfinite(bin_seconds) or bin_seconds <= 0:
            raise ValueError("POSITIVE_BIN_SECONDS_REQUIRED")
        if rank is not None and (type(rank) is not int or not 0 <= rank <= len(ids)):
            raise ValueError("RANK_MUST_BE_UNDEFINED_OR_BETWEEN_ZERO_AND_UNIT_COUNT")
        self.unit_ids, self.input_size, self.history_bins = ids, input_size, history_bins
        self.bin_seconds, self.rank = float(bin_seconds), rank
        features = history_bins * len(ids) + input_size
        self.register_buffer("input_factor", torch.empty(features, 0, dtype=torch.float64))
        self.register_buffer("output_factor", torch.empty(len(ids), 0, dtype=torch.float64))
        self.register_buffer("bias", torch.zeros(len(ids), dtype=torch.float64))
        self.register_buffer("feature_mean", torch.zeros(features, dtype=torch.float64))
        self.register_buffer("feature_scale", torch.ones(features, dtype=torch.float64))
        self.fit_metadata: dict[str, Any] | None = None

    def _tensor(self, values: Any) -> Tensor:
        return torch.as_tensor(values, dtype=self.bias.dtype, device=self.bias.device)

    @staticmethod
    def _observed_counts(values: Tensor) -> None:
        if not torch.isfinite(values).all() or torch.any(values < 0) or torch.any(values != values.round()):
            raise ValueError("FINITE_NONNEGATIVE_INTEGER_COUNTS_REQUIRED")

    def causal_rows(self, counts: Any, stimulus: Any) -> tuple[Tensor, Tensor]:
        """Oldest-to-newest lag bins and current known input, within each trial."""
        y, u = self._tensor(counts), self._tensor(stimulus)
        if y.ndim != 3 or y.shape[0] < 1 or y.shape[1] <= self.history_bins or y.shape[2] != len(self.unit_ids):
            raise ValueError("COUNT_SEQUENCE_SHAPE_MISMATCH")
        self._observed_counts(y)
        if u.shape != (*y.shape[:2], self.input_size) or not torch.isfinite(u).all():
            raise ValueError("STIMULUS_SEQUENCE_SHAPE_OR_VALUES_INVALID")
        windows = y.unfold(1, self.history_bins + 1, 1).permute(0, 1, 3, 2)
        past = windows[:, :, :-1].reshape(-1, self.history_bins * len(self.unit_ids))
        targets = windows[:, :, -1].reshape(-1, len(self.unit_ids))
        inputs = u[:, self.history_bins:].reshape(targets.shape[0], self.input_size)
        return torch.cat((past, inputs), dim=1), targets

    @torch.no_grad()
    def fit_sequence(self, counts: Any, stimulus: Any, *, ridge_weight: float,
                     standardize: bool = True, rcond: float | None = None) -> dict[str, Any]:
        """Minimize summed SSE plus ridge on standardized coefficients.

        The intercept is unpenalized. For positive ridge every SVD mode is used;
        for zero ridge rcond defines the numerical pseudoinverse subspace. Constant
        training features have scale one. No responses outside supplied fit rows
        enter centering, scaling or the least-squares solution.
        """
        if self.bias.dtype != torch.float64:
            raise ValueError("FLOAT64_FIT_REQUIRED")
        if type(ridge_weight) not in (int, float) or not math.isfinite(ridge_weight) or ridge_weight < 0:
            raise ValueError("NONNEGATIVE_FINITE_RIDGE_WEIGHT_REQUIRED")
        if type(standardize) is not bool:
            raise ValueError("STANDARDIZE_BOOLEAN_REQUIRED")
        if rcond is not None and (type(rcond) not in (int, float) or not math.isfinite(rcond) or not 0 <= rcond < 1):
            raise ValueError("RELATIVE_SVD_CUTOFF_INVALID")
        design, target = self.causal_rows(counts, stimulus)
        x, y = design.detach().cpu().numpy(), target.detach().cpu().numpy()
        x_mean, y_mean = x.mean(0), y.mean(0)
        centered, y_centered = x - x_mean, y - y_mean
        training_scale = centered.std(0)
        constant = training_scale == 0
        scale = training_scale.copy() if standardize else np.ones(x.shape[1])
        scale[constant] = 1.0
        if not np.isfinite(scale).all() or not np.isfinite(y_centered).all():
            raise FloatingPointError("NONFINITE_REGRESSION_CENTER_OR_SCALE")
        x_scaled = centered / scale
        left, singular, right = np.linalg.svd(x_scaled, full_matrices=False)
        cutoff_relative = float(rcond if rcond is not None else np.finfo(np.float64).eps * max(x_scaled.shape))
        cutoff = cutoff_relative * (float(singular[0]) if singular.size else 0.0)
        retained = singular > cutoff
        if ridge_weight > 0:
            denominator = singular ** 2 + ridge_weight
            inverse = singular / denominator
            projection_weight = singular / np.sqrt(denominator)
        else:
            inverse = np.zeros_like(singular)
            inverse[retained] = 1.0 / singular[retained]
            projection_weight = retained.astype(np.float64)
        projected_target = left.T @ y_centered
        weighted = projection_weight[:, None] * projected_target
        _, response_singular, response_right = np.linalg.svd(weighted, full_matrices=False)
        requested = len(self.unit_ids) if self.rank is None else self.rank
        width = min(requested, response_right.shape[0])
        output_factor = response_right[:width].T.copy()
        standardized_factor = right.T @ (inverse[:, None] * (projected_target @ output_factor))
        input_factor = standardized_factor / scale[:, None]
        bias = y_mean - (x_mean @ input_factor) @ output_factor.T
        prediction = (x @ input_factor) @ output_factor.T + bias
        residual_sse = float(np.square(y - prediction).sum())
        penalty = float(np.square(standardized_factor).sum())
        objective = residual_sse + float(ridge_weight) * penalty
        certificate = float(np.square(y_centered).sum() - np.square(response_singular[:width]).sum())
        if not all(np.isfinite(v).all() for v in (input_factor, output_factor, bias)) or not math.isfinite(objective + certificate):
            raise FloatingPointError("NONFINITE_REGRESSION_FIT")
        self.input_factor = self._tensor(input_factor).clone()
        self.output_factor = self._tensor(output_factor).clone()
        self.bias.copy_(self._tensor(bias))
        self.feature_mean.copy_(self._tensor(x_mean))
        self.feature_scale.copy_(self._tensor(scale))
        self.fit_metadata = {
            "fit_rows": int(x.shape[0]), "feature_count": int(x.shape[1]), "rank_requested": self.rank,
            "factor_width": width, "ridge_weight": float(ridge_weight), "standardize": standardize,
            "ridge_penalty_coordinates": "standardized_features" if standardize else "raw_features",
            "intercept_penalized": False, "objective_normalization": "sum_over_rows_and_units",
            "design_rank_at_cutoff": int(retained.sum()), "rcond": cutoff_relative,
            "svd_cutoff_applied_to_fit": ridge_weight == 0,
            "constant_feature_count": int(constant.sum()), "residual_sse": residual_sse,
            "coefficient_penalty": penalty, "objective": objective,
            "spectral_optimum_objective": certificate, "objective_certificate_error": objective - certificate,
            "largest_design_singular_value": float(singular[0]),
            "smallest_design_singular_value": float(singular[-1]),
            "deterministic_fit": True, "stochastic_training_seed_used": False,
        }
        return dict(self.fit_metadata)

    def infer_state(self, history: Any, metadata: dict[str, Any] | None = None) -> RegressionHistoryState:
        values = self._tensor(history)
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < self.history_bins or values.shape[2] != len(self.unit_ids):
            raise ValueError("HISTORY_SHAPE_MISMATCH")
        self._observed_counts(values)
        if metadata is not None:
            if tuple(metadata.get("unit_ids", self.unit_ids)) != self.unit_ids:
                raise ValueError("HISTORY_UNIT_ORDER_MISMATCH")
            if metadata.get("bin_seconds", self.bin_seconds) != self.bin_seconds:
                raise ValueError("HISTORY_BIN_WIDTH_MISMATCH")
        return RegressionHistoryState(values[:, -self.history_bins:].detach().clone(), self.unit_ids, self.bin_seconds)

    def forecast(self, state: RegressionHistoryState, stimulus: Any) -> MeanForecast:
        """Recursive prediction without future responses; input gradients are kept."""
        if self.fit_metadata is None:
            raise ValueError("REGRESSION_MODEL_NOT_FITTED")
        if state.unit_ids != self.unit_ids or state.bin_seconds != self.bin_seconds:
            raise ValueError("STATE_UNIT_ORDER_OR_BIN_WIDTH_MISMATCH")
        history = self._tensor(state.counts)
        if history.ndim != 3 or history.shape[0] < 1 or history.shape[1:] != (self.history_bins, len(self.unit_ids)):
            raise ValueError("STATE_HISTORY_SHAPE_MISMATCH")
        self._observed_counts(history)
        batch = history.shape[0]
        u = self._tensor(stimulus)
        if u.ndim == 2:
            u = u.unsqueeze(0).expand(batch, -1, -1)
        if u.ndim != 3 or u.shape[0] != batch or u.shape[1] < 1 or u.shape[2] != self.input_size or not torch.isfinite(u).all():
            raise ValueError("STIMULUS_SEQUENCE_SHAPE_OR_VALUES_INVALID")
        predictions = []
        for step in range(u.shape[1]):
            features = torch.cat((history.flatten(1), u[:, step]), dim=1)
            value = (features @ self.input_factor) @ self.output_factor.T + self.bias
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"NONFINITE_REGRESSION_FORECAST_AT_BIN:{step}")
            predictions.append(value)
            history = torch.cat((history[:, 1:], value[:, None]), dim=1)
        return MeanForecast(torch.stack(predictions, dim=1), self.unit_ids, self.bin_seconds)

    def save(self, path: Path) -> None:
        if self.fit_metadata is None:
            raise ValueError("REGRESSION_MODEL_NOT_FITTED")
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            torch.save({"schema": "neural_population_reduced_rank_autoregression_v1",
                        "configuration": {"unit_ids": list(self.unit_ids), "input_size": self.input_size,
                                          "history_bins": self.history_bins, "bin_seconds": self.bin_seconds,
                                          "rank": self.rank},
                        "fit_metadata": self.fit_metadata, "state_dict": self.state_dict()}, stream)

    @classmethod
    def load(cls, path: Path, *, device: str | torch.device = "cpu") -> ReducedRankAutoregression:
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload["schema"] != "neural_population_reduced_rank_autoregression_v1":
            raise ValueError("REGRESSION_CHECKPOINT_SCHEMA_MISMATCH")
        state = payload["state_dict"]
        model = cls(**payload["configuration"]).to(device=device, dtype=state["bias"].dtype)
        width = state["input_factor"].shape[1]
        if (state["input_factor"].shape != (model.feature_mean.numel(), width)
                or state["output_factor"].shape != (len(model.unit_ids), width)
                or width > (len(model.unit_ids) if model.rank is None else model.rank)):
            raise ValueError("REGRESSION_CHECKPOINT_FACTOR_SHAPE_MISMATCH")
        model.input_factor = torch.empty_like(state["input_factor"])
        model.output_factor = torch.empty_like(state["output_factor"])
        model.load_state_dict(state, strict=True)
        if any(not torch.isfinite(tensor).all() for tensor in model.buffers()):
            raise ValueError("NONFINITE_REGRESSION_CHECKPOINT")
        model.fit_metadata = payload["fit_metadata"]
        if not isinstance(model.fit_metadata, dict):
            raise ValueError("REGRESSION_CHECKPOINT_FIT_METADATA_REQUIRED")
        return model



def check_reduced_rank(source_dir: Path, output_dir: Path, config: dict[str, Any],
                       command: list[str], seed: int) -> tuple[dict[str, Any], int]:
    """Deterministic, in-sample engineering fit; no formal comparison or samples."""
    if output_dir.exists():
        raise FileExistsError(f"REGRESSION_CHECK_OUTPUT_EXISTS:{output_dir}")
    if seed not in config["training_seeds"]:
        raise ValueError("UNREGISTERED_TRAINING_SEED")
    spec = config["ridge_baseline_check"]
    data = load_engineering_windows(source_dir, config)
    model = ReducedRankAutoregression(data.unit_ids, data.inputs.shape[2], spec["history_bins"],
                                      data.bin_seconds, spec["rank"])
    if data.history_bins < model.history_bins:
        raise ValueError("SOURCE_HISTORY_OR_RESPONSE_TOO_SHORT")
    started = time.monotonic()
    fit = model.fit_sequence(data.counts, data.inputs, ridge_weight=spec["ridge_weight"],
                             standardize=spec["standardize"], rcond=spec["rcond"])
    fit_seconds = time.monotonic() - started
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "reduced_rank.pt"
    model.save(checkpoint)
    prefix = data.counts[:, :data.history_bins]
    future = data.inputs[:, data.history_bins:]
    prediction = model.forecast(model.infer_state(prefix, {"unit_ids": data.unit_ids,
                                                           "bin_seconds": data.bin_seconds}), future).values
    restored = ReducedRankAutoregression.load(checkpoint)
    replay = restored.forecast(restored.infer_state(prefix), future).values
    if not torch.equal(prediction, replay):
        raise ValueError("SAVED_REGRESSION_FORECAST_REPLAY_MISMATCH")
    targets = torch.as_tensor(data.counts[:, data.history_bins:], dtype=prediction.dtype)
    with h5py.File(output_dir / "mean_forecasts.h5", "x") as output:
        output.attrs.update(scope="engineering_only", bin_seconds=data.bin_seconds,
                            prediction_kind="unconstrained_mean_response_prediction_not_spike_samples",
                            evaluation_scope="in_sample_engineering_replay_no_holdout")
        output["predictions"] = prediction.detach().cpu().numpy()
        output["observed_counts"] = targets.numpy()
        output["unit_ids"] = np.asarray(data.unit_ids, dtype=np.int64)
        output.create_dataset("trial_ids", data=data.trial_ids, dtype=h5py.string_dtype("utf-8"))
    if data.input_features is not None:
        freeze_json(output_dir / "input_features.json", data.input_features)
    freeze_json(output_dir / "fit.json", fit)
    summary = {
        "status": "PARTIAL", "software_check": "PASS", "scope": "engineering_only",
        "method": "ridge_reduced_rank_autoregression", "seed": seed,
        "deterministic_fit": True, "stochastic_training_seed_used": False,
        "formal_training_completed": False, "evaluation_scope": "in_sample_engineering_replay_no_holdout",
        "unit_count": len(data.unit_ids), "trial_count": int(data.counts.shape[0]),
        "forecast_shape": list(prediction.shape), "known_input_channels": model.input_size,
        "history_bins": model.history_bins, "rank_requested": model.rank,
        "factor_width": model.input_factor.shape[1],
        "parameter_count": model.input_factor.numel() + model.output_factor.numel() + model.bias.numel(),
        "state_scalar_count_per_trial": model.history_bins * len(model.unit_ids),
        "fit_seconds": fit_seconds, "forecast_rmse": float((prediction - targets).square().mean().sqrt()),
        "negative_prediction_count": int((prediction < 0).sum()),
        "minimum_prediction": float(prediction.min()), "maximum_prediction": float(prediction.max()),
        "checkpoint_forecast_replay": "PASS", "elapsed_seconds": time.monotonic() - started,
        "provides_joint_spike_samples": False, "generative_fidelity_qualified": False,
        "inverse_intervention_qualified": False, "scientific_claims_qualified": False,
    }
    freeze_json(output_dir / "summary.json", summary)
    names = ["reduced_rank.pt", "fit.json", "mean_forecasts.h5", "summary.json"]
    if data.input_features is not None:
        names.append("input_features.json")
    freeze_json(output_dir / "manifest.json", {
        "schema": "neural_population_reduced_rank_check_v1",
        "source_manifest": str((source_dir / "manifest.json").resolve()),
        "source_manifest_sha256": sha256_file(source_dir / "manifest.json"),
        "seed": seed, "torch_version": str(torch.__version__), "dtype": "float64",
        "ended_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "engineering_only",
        "artifacts": {name: sha256_file(output_dir / name) for name in names},
        **runtime_manifest(config, command, 2),
    })
    return {**summary, "manifest": str(output_dir / "manifest.json")}, 2
