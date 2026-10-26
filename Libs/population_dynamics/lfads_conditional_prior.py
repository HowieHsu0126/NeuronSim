"""Fitting-derived controller mean for one causal LFADS development contrast.

The original adapter, recurrent weights and eight fitted AR scalars stay fixed.
The added curve is a variational training target, not identified process noise.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import yaml

from .lfads import LFADSCountSamples
from .provenance import freeze_json, sha256_file

TEMPLATES = tuple(f"familiar_{i}" for i in range(4))
COMPLETE_BINS = 1200
CHANNELS = 4
SCHEMA = "lfads_fitting_template_mean_v1"


@dataclass(frozen=True)
class ConditionalPriorSamples(LFADSCountSamples):
    base_future_controller_inputs: np.ndarray


@dataclass(frozen=True)
class ConditionalMeanPrior:
    means: np.ndarray

    def curve(self, template_id: str, boundary: int, bins: int) -> np.ndarray:
        if (template_id not in TEMPLATES or type(boundary) is not int or type(bins) is not int
                or boundary < 1 or bins < 1 or boundary + bins > COMPLETE_BINS):
            raise ValueError("KNOWN_TEMPLATE_AND_FITTED_NATIVE_TIME_SUPPORT_REQUIRED")
        return self.means[TEMPLATES.index(template_id), boundary - 1:boundary + bins]


def anchored_offsets(curve: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Float32 recurrence delta_t=phi*delta_(t-1)+(mu_t-phi*mu_(t-1))."""
    curve, phi = np.asarray(curve), np.asarray(phi)
    if (curve.dtype != np.float32 or phi.dtype != np.float32 or curve.ndim != 2
            or curve.shape[0] < 2 or curve.shape[1] != CHANNELS or phi.shape != (CHANNELS,)
            or not np.isfinite(curve).all() or not np.isfinite(phi).all()
            or (phi <= 0).any() or (phi >= 1).any()):
        raise ValueError("FINITE_NATIVE_FLOAT32_MEAN_CURVE_AND_INTERIOR_AR_REQUIRED")
    offset = np.zeros(CHANNELS, dtype=np.float32)
    result = []
    for previous, current in zip(curve[:-1], curve[1:]):
        bias = (current - phi * previous).astype(np.float32)
        offset = (phi * offset + bias).astype(np.float32)
        result.append(offset)
    result = np.stack(result)
    if not np.isfinite(result).all():
        raise FloatingPointError("NONFINITE_CONDITIONAL_PRIOR_OFFSET")
    return result


def apply_mean_to_feed(model, feed, latent, curve: np.ndarray) -> None:
    """Change future feed values only; preserve causal past and all random draws."""
    curve = np.asarray(curve)
    future = latent["future_controller_inputs"]
    if (future.dtype != np.float32 or future.ndim != 4 or future.shape[-1] != CHANNELS
            or curve.dtype != np.float32 or curve.shape != (future.shape[2] + 1, CHANNELS)
            or not np.isfinite(curve).all()):
        raise ValueError("ALIGNED_NATIVE_MEAN_CURVE_REQUIRED")
    # Preserve signed zeros and exact source arithmetic for the zero-curve control.
    if not np.any(curve):
        return
    m = model._full.model
    native_phi = np.asarray(model._full.session.run(m.prior_zs_ar_con.phis_bxu, feed_dict=feed))
    phi_rows = native_phi.reshape(-1, CHANNELS)
    if not np.array_equal(phi_rows, np.broadcast_to(phi_rows[0], phi_rows.shape)):
        raise ValueError("SHARED_FITTED_AR_PARAMETERS_REQUIRED")
    offsets = anchored_offsets(curve, phi_rows[0])
    corrected = (future + offsets[None, None]).astype(np.float32)
    if not np.isfinite(corrected).all():
        raise FloatingPointError("NONFINITE_CONDITIONAL_CONTROLLER_INPUT")
    for t in range(future.shape[2]):
        feed[m.controller_outputs[model.history_bins + t]] = corrected[:, :, t].reshape(-1, CHANNELS)
    latent["future_controller_inputs"] = corrected


def simulate_with_mean(model, state, stimulus, rng, n_samples: int, curve: np.ndarray):
    feed, latent, bins = model._draw_feed(state, stimulus, rng, n_samples)
    base_future = latent["future_controller_inputs"]
    apply_mean_to_feed(model, feed, latent, curve)
    future = model._read_future(feed, n_samples, bins)
    counts = rng.observation.poisson(future["conditional_expected_counts"])
    return ConditionalPriorSamples(counts=counts, **future, **latent,
        unit_ids=model.unit_ids, bin_seconds=model.bin_seconds, base_future_controller_inputs=base_future)


def _protocol_cell(protocol, config, training_seed, network_budget_epochs):
    legacy = protocol.get("scope") == "one_development_causal_prior_contrast"
    if legacy:
        seed, budget = 41, 20
        if (protocol.get("training_seed") != seed or protocol.get("network_budget_epochs") != budget
                or training_seed not in (None, seed) or network_budget_epochs not in (None, budget)):
            raise ValueError("DECLARED_SINGLE_CELL_MEAN_COMPARISON_REQUIRED")
    else:
        base_path = config.with_name("neural_population_lfads_conditional_prior.yaml")
        if (protocol.get("scope") != "registered_development_causal_prior_matrix"
                or protocol.get("training_seeds") != [41, 42, 43]
                or protocol.get("network_budget_epoch_ceilings") != [20, 80]
                or type(training_seed) is not int or training_seed not in [41, 42, 43]
                or type(network_budget_epochs) is not int or network_budget_epochs not in [20, 80]
                or sha256_file(base_path) != protocol.get("single_cell_protocol_sha256")):
            raise ValueError("EXPLICIT_REGISTERED_CONDITIONAL_MEAN_MATRIX_CELL_REQUIRED")
        base = yaml.safe_load(base_path.read_text())
        registration = {"scope", "training_seeds", "network_budget_epoch_ceilings",
                        "single_cell_protocol_sha256", "routing_reference"}
        if ({key: value for key, value in protocol.items() if key not in registration}
                != {key: value for key, value in base.items()
                    if key not in {"scope", "training_seed", "network_budget_epochs"}}):
            raise ValueError("SINGLE_CELL_SCIENTIFIC_PROTOCOL_CHANGED")
        seed, budget = training_seed, network_budget_epochs
    if (protocol["fitting_blocks"] != list(range(14))
            or protocol["selection_blocks"] != list(range(14, 18))
            or protocol["mean_estimator"] != "equal_block_and_draw_mean_at_each_template_and_native_bin"
            or protocol["formal_test_access"] is not False):
        raise ValueError("DECLARED_FITTING_ONLY_MEAN_COMPARISON_REQUIRED")
    return seed, budget, not legacy


def _training_cell(training_run, binding, comparison, prior, seed, budget):
    """Authenticate the existing run; an epoch ceiling is not a selected epoch."""
    from .lfads_trajectory_training import _checkpoint_budget

    if training_run is None:
        raise ValueError("BOUND_TRAINING_RUN_REQUIRED_FOR_MATRIX_CELL")
    training_run = Path(training_run)
    if sha256_file(training_run) != binding["training_run_sha256"]:
        raise ValueError("CONDITIONAL_MEAN_TRAINING_RUN_BINDING_CHANGED")
    run = json.loads(training_run.read_text())
    effective, _ = _checkpoint_budget(run, binding["training_binding"]["config"])
    selected = binding["selected_epoch"]
    if (run.get("status") != "FITTED_DEVELOPMENT" or run.get("phase") != "stopped"
            or run.get("binding") != binding["training_binding"]
            or run.get("training_seed") != seed or binding["training_seed"] != seed
            or run["binding"].get("seed") != seed or comparison.get("training_seed") != seed
            or prior.get("training_seed") != seed or effective != budget
            or type(selected) is not int or prior.get("selected_epoch") != selected
            or type(run.get("best_epoch")) is not int or selected != run["best_epoch"] + 1
            or type(run.get("completed_epochs")) is not int or not 1 <= selected <= run["completed_epochs"] <= budget
            or run.get("test_outcomes_accessed") is not False
            or run.get("calibration_outcomes_accessed") is not False
            or binding["training_binding"]["config"].get("fit_blocks") != list(range(14))
            or binding["training_binding"]["config"].get("validation_blocks") != list(range(14, 18))):
        raise ValueError("REGISTERED_TRAINING_CELL_OR_SELECTED_CHECKPOINT_CHANGED")
    return dict(training_seed=seed, network_budget_epochs=budget, selected_epoch=selected,
                completed_training_epochs=run["completed_epochs"],
                training_run_sha256=binding["training_run_sha256"],
                selected_model_artifacts=binding["selected_model_artifacts"])


def fit_cache_mean(cache: Path, paired_analysis: Path, config: Path, output: Path, *,
                   training_seed: int | None = None, network_budget_epochs: int | None = None,
                   training_run: Path | None = None) -> dict:
    # The cache verifier imports its existing evaluator; use it only in fitting.
    from .lfads_prior_refit import _cache_identity, _cached_paths

    cache, paired_analysis, config, output = map(Path, (cache, paired_analysis, config, output))
    protocol = yaml.safe_load(config.read_text())
    seed, budget, matrix = _protocol_cell(protocol, config, training_seed, network_budget_epochs)
    if output.exists():
        raise FileExistsError(output)
    path = paired_analysis / "analysis/analysis.json"
    delivery = json.loads((paired_analysis / "delivery.json").read_text())
    rows = [row for row in delivery["files"] if row["path"] == "analysis/analysis.json"]
    if (delivery["status"] != "DESCRIBED_REPLAYED_PRIOR_COMPARISON" or len(rows) != 1
            or sha256_file(path) != rows[0]["sha256"]):
        raise ValueError("VERIFIED_PAIRED_PRIOR_ANALYSIS_REQUIRED")
    comparison = json.loads(path.read_text())
    prior = comparison["arm_bindings"]["fitted_prior"]["prior_refit"]
    record_path = cache / "posterior_cache.json"
    if (sha256_file(record_path) != prior["posterior_cache_sha256"]
            or sha256_file(cache / "prior_parameters.npz") != prior["prior_parameters_sha256"]):
        raise ValueError("PAIRED_FITTING_CACHE_CHANGED")
    record = json.loads(record_path.read_text())
    binding = record["binding"]
    ids = [f"train_block_{block:03d}_{template}" for block in range(14) for template in TEMPLATES]
    if (record["status"] != "FITTED_FROZEN_NETWORK_AR_PRIOR" or not record["fitting_cohort_extracted"]
            or record["formal_test_accessed"] or record["complete_episodes"] != 56
            or record["posterior_paths"] != 896 or binding["training_seed"] != seed
            or binding["fitting_blocks"] != list(range(14))
            or binding["excluded_selection_blocks"] != list(range(14, 18))
            or binding["selected_model_artifacts"] != comparison["arm_bindings"]["fitted_prior"]["selected_model_artifacts"]
            or binding["training_binding"] != comparison["training_binding"]
            or [row["case_id"] for row in record["episodes"]] != ids):
        raise ValueError("COMPLETE_SOURCE_BOUND_FITTING_ONLY_CACHE_REQUIRED")
    cell = _training_cell(training_run, binding, comparison, prior, seed, budget) if matrix else None
    # Each block and each draw has equal weight; no response in blocks14-17 enters.
    total = np.zeros((len(TEMPLATES), COMPLETE_BINS, CHANNELS), dtype=np.float64)
    for row in record["episodes"]:
        case = {key: row[key] for key in ("case_id", "block", "template_id")}
        paths = _cached_paths(cache, case, row, _cache_identity(binding, case))
        total[TEMPLATES.index(case["template_id"])] += paths.astype(np.float64).mean(axis=0)
    means = (total / 14).astype(np.float32)
    if not np.isfinite(means).all():
        raise FloatingPointError("NONFINITE_FITTED_CONDITIONAL_MEAN")
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "controller_means.npz", means=means)
    report = dict(status="FITTED_TEMPLATE_TIME_CONTROLLER_MEAN", schema=SCHEMA,
        training_seed=seed, prior_refit=prior, comparison_protocol=protocol,
        comparison_protocol_sha256=sha256_file(config), fitting_blocks=list(range(14)),
        excluded_selection_blocks=list(range(14, 18)), episodes=56, posterior_paths=896,
        templates=list(TEMPLATES), complete_bins=COMPLETE_BINS, bin_seconds=0.005,
        additional_fitted_scalars=int(means.size), fit_accumulation_dtype="float64",
        saved_mean_dtype="float32", estimator="Equal initial/background-block and posterior-draw mean at each known template/native bin",
        paired_analysis_sha256=sha256_file(path),
        means_sha256=sha256_file(output / "controller_means.npz"),
        implementation_sha256=sha256_file(Path(__file__)),
        selected_model_artifacts=binding["selected_model_artifacts"],
        conditioning="Preplanned familiar stimulus template and elapsed native episode time",
        forecast_formula="Original float32 sampled AR path plus anchored float32 deterministic mean offset",
        boundary_condition="Zero offset at the last observed controller bin",
        network_weights_changed=False, ar_parameters_changed=False,
        process_noise_identified=False, formal_test_accessed=False, scientific_claim_qualified=False)
    if matrix:
        report.update(network_budget_epochs=budget, training_cell=cell)
    freeze_json(output / "mean_prior.json", report)
    return report


def load_mean_prior(directory: Path, prior_binding: dict, *, training_seed: int | None = None,
                    network_budget_epochs: int | None = None,
                    selected_model_artifacts: dict | None = None) -> tuple[ConditionalMeanPrior, dict]:
    directory = Path(directory)
    path = directory / "mean_prior.json"
    report = json.loads(path.read_text())
    matrix = report["comparison_protocol"].get("scope") == "registered_development_causal_prior_matrix"
    if matrix:
        cell = report.get("training_cell", {})
        if (type(training_seed) is not int or training_seed not in [41, 42, 43]
                or type(network_budget_epochs) is not int or network_budget_epochs not in [20, 80]
                or report.get("training_seed") != training_seed or prior_binding.get("training_seed") != training_seed
                or report.get("network_budget_epochs") != network_budget_epochs
                or cell.get("training_seed") != training_seed or cell.get("network_budget_epochs") != network_budget_epochs
                or cell.get("selected_epoch") != prior_binding.get("selected_epoch")
                or type(cell.get("selected_epoch")) is not int or type(cell.get("completed_training_epochs")) is not int
                or not 1 <= cell["selected_epoch"] <= cell["completed_training_epochs"] <= network_budget_epochs
                or not isinstance(selected_model_artifacts, dict) or not selected_model_artifacts
                or report.get("selected_model_artifacts") != selected_model_artifacts
                or cell.get("selected_model_artifacts") != selected_model_artifacts):
            raise ValueError("CONDITIONAL_PRIOR_MATRIX_CELL_BINDING_CHANGED")
    elif (report["training_seed"] != 41 or training_seed not in (None, 41)
          or network_budget_epochs not in (None, 20)
          or (selected_model_artifacts is not None and selected_model_artifacts != report["selected_model_artifacts"])):
        raise ValueError("CONDITIONAL_PRIOR_FITTING_BINDING_CHANGED")
    if (report["status"] != "FITTED_TEMPLATE_TIME_CONTROLLER_MEAN" or report["schema"] != SCHEMA
            or report["prior_refit"] != prior_binding
            or report["fitting_blocks"] != list(range(14)) or report["excluded_selection_blocks"] != list(range(14, 18))
            or report["templates"] != list(TEMPLATES) or report["complete_bins"] != COMPLETE_BINS
            or report["bin_seconds"] != 0.005 or report["formal_test_accessed"] is not False
            or report["network_weights_changed"] is not False or report["ar_parameters_changed"] is not False
            or sha256_file(directory / "controller_means.npz") != report["means_sha256"]):
        raise ValueError("CONDITIONAL_PRIOR_FITTING_BINDING_CHANGED")
    with np.load(directory / "controller_means.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"means"}:
            raise ValueError("EXACT_CONDITIONAL_MEAN_ARRAY_REQUIRED")
        means = archive["means"]
    if means.dtype != np.float32 or means.shape != (len(TEMPLATES), COMPLETE_BINS, CHANNELS) or not np.isfinite(means).all():
        raise ValueError("FINITE_TEMPLATE_TIME_FOUR_CHANNEL_MEAN_REQUIRED")
    means.flags.writeable = False
    binding = dict(mean_prior_sha256=sha256_file(path), means_sha256=report["means_sha256"],
        implementation_sha256=sha256_file(Path(__file__)), additional_fitted_scalars=int(means.size),
        conditioning=report["conditioning"], forecast_formula=report["forecast_formula"],
        comparison_protocol=report["comparison_protocol"],
        comparison_protocol_sha256=report["comparison_protocol_sha256"],
        formal_test_accessed=False, process_noise_identified=False)
    if matrix:
        binding.update(training_seed=training_seed, network_budget_epochs=network_budget_epochs,
                       training_cell=report["training_cell"])
    return ConditionalMeanPrior(means), binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "paired-analysis", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--network-budget-epochs", type=int)
    parser.add_argument("--training-run", type=Path)
    args = parser.parse_args()
    print(json.dumps(fit_cache_mean(args.cache, args.paired_analysis, args.config, args.output,
        training_seed=args.training_seed, network_budget_epochs=args.network_budget_epochs,
        training_run=args.training_run)))


if __name__ == "__main__":
    main()
