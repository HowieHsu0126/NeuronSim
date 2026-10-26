"""Replay and describe count forecasts on epoch-selection blocks.

Four initial/background realization blocks are the source replication units,
conditional on one fixed graph. Training seeds, units, phases and ensemble draws
are not additional independent realizations. No inferential CI, p or q is made.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
import traceback

import numpy as np
import torch
import yaml

from .native_renewal_data import NativeRenewalEpisode, load_dataset_index
from .poisson_trajectory_evaluation import fitting_scale
from .poisson_trajectory_training import partition, validate_prior_budget_binding
from .provenance import sha256_file
from .population_trajectory_evaluation import _load_model
from .trajectory_forecast_analysis import (_write_rows, covariance_components,
                                           pair_sets, replay_scores)

SEEDS = (41, 42, 43)
MODELS = ("poisson", "renewal")
BLOCKS = (14, 15, 16, 17)
IDENTITY_COLUMNS = ("seed", "block", "case_id", "template_id", "origin_ms", "horizon_ms", "method", "ensemble_artifact")
UNIT_METRICS = ("unit_bin_mean_mse_corrected", "sampled_expected_spikes_per_unit", "observed_spikes_per_unit",
                "unit_total_coverage_90", "unit_total_interval_width_90")
COST_METRICS = ("inference_seconds", "full_unit_generation_seconds", "field_projection_seconds", "peak_gpu_allocated_bytes")
COST_COMPONENTS = COST_METRICS + (
    "input_loading_seconds", "sample_serialization_seconds", "graph_setup_seconds",
    "scoring_seconds", "origin_total_seconds", "process_peak_rss_bytes",
    "peak_tensorflow_gpu_allocated_bytes",
)
EVALUATOR_ORIGIN_COSTS = (
    "scoring_seconds", "serialization_seconds", "hash_seconds", "csv_seconds",
    "origin_total_seconds", "peak_process_rss_bytes",
)
EVALUATOR_EPISODE_COSTS = (
    "loading_seconds", "window_seconds", "transfer_seconds", "episode_preparation_seconds",
)
EVALUATOR_COST_COLUMNS = EVALUATOR_ORIGIN_COSTS + (
    "episode_loading_seconds", "episode_window_seconds", "episode_transfer_seconds",
    "episode_preparation_seconds", "episode_cost_origin_ms", "episode_cost_scope",
    "evaluation_hardware", "evaluation_cost_scope", "generated_bins", "draws", "unit_count",
    "field_count", "score_rows", "generated_unit_count_values", "generated_unit_count_dtype",
    "generated_unit_count_device", "latent_dtype", "latent_device", "field_count_dtype",
    "field_metric_dtype", "stored_full_unit_counts",
)
CANDIDATE_COST_SCOPE = (
    "inference, full 500ms unit generation and field projection; input loading, "
    "serialization, scoring and setup are unmeasured; event materialization excluded"
)
CONFIG_PATH = Path(__file__).parents[1] / "configs/neural_population_matched_forecast.yaml"
PRIOR_REFIT_SCHEMA = "neural_population_prior_refit_v1"
PRIOR_REFIT_OBJECTIVE = "fixed_posterior_multistep_prior_kl"
PRIOR_MIXTURE_SCHEMA = "neural_population_prior_predictive_mixture_v1"
PRIOR_MIXTURE_OBJECTIVE = "fixed_posterior_joint_predictive_mixture_kl"
PRIOR_OBSERVATION_SCHEMA = "neural_population_native_observation_prior_v1"
PRIOR_OBSERVATION_OBJECTIVE = "fixed_readout_native_event_path_mixture_nll"
PRIOR_OBSERVATION_MODULE = "population_observation_prior.py"
PRIOR_OBSERVATION_DECLARATIONS = {
    "mixture_scope": "joint_all_source_units_and_time_prefix",
    "observation_score_normalization": "nats_per_identified_unit_and_bin",
    "posterior_cache_role": "causal_boundary_only",
    "posterior_targets_used": False,
    "future_exposure_role": "observed_event_path_scoring_label_only",
    "future_exposure_used_in_prior_dynamics": False,
    "marginal_likelihood_exact": False,
}
PRIOR_PROFILES = {
    PRIOR_REFIT_SCHEMA: (PRIOR_REFIT_OBJECTIVE, "validation_multistep_prior_kl",
                         "fixed_draw_validation_multistep_kl"),
    PRIOR_MIXTURE_SCHEMA: (PRIOR_MIXTURE_OBJECTIVE, "validation_joint_predictive_mixture_kl",
                           "fixed_draw_validation_joint_predictive_mixture_kl"),
    PRIOR_OBSERVATION_SCHEMA: (PRIOR_OBSERVATION_OBJECTIVE, "validation_native_event_path_mixture_nll",
                              "fixed_draw_validation_native_event_path_mixture_nll"),
}
PRIOR_METHODS = {
    PRIOR_REFIT_OBJECTIVE: "population_recurrent_prior_refit",
    PRIOR_MIXTURE_OBJECTIVE: "population_recurrent_predictive_mixture",
    PRIOR_OBSERVATION_OBJECTIVE: "population_recurrent_native_observation",
}
PRIOR_COMPARISON_LABELS = {
    PRIOR_REFIT_OBJECTIVE: "analytic_prior_kl",
    PRIOR_MIXTURE_OBJECTIVE: "predictive_mixture",
    PRIOR_OBSERVATION_OBJECTIVE: "native_observation",
}
PRIOR_OBJECTIVE_MODULES = {
    PRIOR_REFIT_OBJECTIVE: {"population_prior_objective.py"},
    # Preserve the historical aggregate key as well as the source filename.
    PRIOR_MIXTURE_OBJECTIVE: {"population_prior_objective.py", "population_predictive_mixture.py",
                              "population_prior_predictive_mixture.py"},
    PRIOR_OBSERVATION_OBJECTIVE: {"population_prior_objective.py", PRIOR_OBSERVATION_MODULE},
}
PRIOR_PROTOCOL_KEYS = ("schema", "training_sequence", "observation_objective",
                       "prior_origins_ms", "prior_horizons_ms", "prior_selection", "prior_target_gradient")
PRIOR_MIXTURE_CONFIG_KEYS = ("prediction_particles", "posterior_target_draws")
PRIOR_MIXTURE_SCOPE = "joint_all_source_fields_and_latent_channels"
# The reviewed refit revision replaces one type annotation/import with a Protocol.
# All observation function bodies are identical, including event-time likelihood
# and native-clock refractory sampling. Other changed source pairs are rejected.
PRIOR_OBSERVATION_COMPATIBILITY = {
    ("0d5c56578c40fba05df5f707a4b3c705599fa90dd5a5a552b2c6d0c82a3dc69f",
     "5fd537d4c20b550478e9f342a3519062bf0b361a87664035489472d3ff4e4030"):
        "typing-only RenewalEpisode Protocol replaces NativeEpisodeCounts annotation/import; all runtime function bodies unchanged"
}
PRIOR_SOURCE_BINDINGS = ("acquisition_design_sha256", "acquisition_ledger_sha256", "dataset_sha256",
                         "episode_table_sha256", "native_source_identity", "episode_bindings")



def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    return json.loads(path.read_text())


def _finite(value, name):
    result = float(value)
    _require(np.isfinite(result), "finite numeric value required: " + name)
    return result


def _integer(value, name):
    result = _finite(value, name)
    _require(result == int(result), "integer identity required: " + name)
    return int(result)


def _counts(values, shape, name):
    _require(values.shape == shape and values.dtype.kind in "iu" and np.all(values >= 0),
             "identified nonnegative integer count schema required: " + name)


class CanonicalFields:
    """Bind all source views; reuse the fitting-scale estimator on fit blocks only."""

    def __init__(self, acquisition: Path, fields: Path):
        design, dataset = _read_json(acquisition / "design.json"), _read_json(fields / "dataset.json")
        split = dict(training_seeds=list(SEEDS), fit_blocks=list(range(14)), validation_blocks=list(BLOCKS))
        fit, self.validation = partition(design, split)
        _require(len(fit) == 56 and len(self.validation) == 16 and len({c["case_id"] for c in design["cases"]}) == 72,
                 "complete72-episode training cohort required")
        _require(all({c["template_id"] for c in design["cases"] if c["block"] == block}
                     == {f"familiar_{i}" for i in range(4)} for block in range(18)), "four familiar templates per block required")
        _require(dataset["status"] == "BUILT" and dataset["split"] == "train" and dataset["completed_episodes"] == 72
                 and dataset["cortical_units"] == [100000] and dataset["settings"]["bin_ms"] == 5.
                 and dataset["settings"]["cells_per_axis"] == 4 and dataset["settings"]["start_ms"] == 0.,
                 "complete source-defined100000-unit 5ms training fields required")
        _require(dataset["design_sha256"] == sha256_file(acquisition / "design.json"), "canonical dataset/design binding changed")
        index = load_dataset_index(fields)
        _require(set(index) == {c["case_id"] for c in design["cases"]}, "canonical episode identities changed")
        self.bin_ms = 5.
        self.truth = {}
        self.source_identity = None
        self.unit_ids = self.unit_field_index = self.field_ids = self.unit_count = None
        field_bindings = {}
        for case in sorted(design["cases"], key=lambda c: c["case_id"]):
            identity = case["case_id"]
            folder = fields / identity
            report_path = acquisition / identity / "report.json"
            native, summary = _read_json(report_path), _read_json(folder / "summary.json")
            _require(sha256_file(report_path) == index[identity]["raw_report_sha256"]
                     and sha256_file(folder / "summary.json") == index[identity]["field_summary_sha256"],
                     "canonical native/field metadata changed: " + identity)
            _require(native["status"] == "EXECUTED" and summary["status"] == "DESCRIBED"
                     and native["config"] == case["settings"] and native["duration_ms"] == 6000.
                     and summary["window_start_tick"] == 0 and summary["window_stop_tick"] == 60000,
                     "complete canonical episode and matched field window required")
            source = {key: native[key] for key in ("gpu_source_revision", "native_library_sha256",
                                                   "native_poisson_rng_family", "graph_manifest_sha256", "dt_ms")}
            _require(source["graph_manifest_sha256"] == design["graph_manifest_sha256"]
                     and source["native_library_sha256"] == design["native_library_sha256"]
                     and source["native_poisson_rng_family"] == "per_edge_philox4_32_10" and source["dt_ms"] == .1,
                     "canonical Philox source/clock differs")
            if self.source_identity is None:
                self.source_identity = source
            _require(source == self.source_identity, "mixed native sources in canonical cohort")
            field_path = folder / "spatial_fields.npz"
            field_digest = sha256_file(field_path)
            _require(field_digest == summary["artifacts"][field_path.name], "canonical field array changed")
            with np.load(field_path, allow_pickle=False) as array:
                ids, mapping, names, units = (array[k] for k in ("source_unit_ids", "unit_field_index", "field_ids", "total_unit_count"))
                _require(ids.shape == mapping.shape == (100000,) and ids.dtype.kind in "iu" and np.all(np.diff(ids) > 0)
                         and mapping.dtype.kind in "iu" and np.all(mapping >= 0) and np.all(mapping < 128)
                         and names.shape == units.shape == (128,) and units.dtype.kind in "iu" and np.all(units > 0)
                         and np.array_equal(np.bincount(mapping, minlength=128), units), "source unit/field observation map invalid")
                _counts(array["count_sum"], (1200, 128), identity)
                if self.unit_ids is None:
                    self.unit_ids, self.unit_field_index, self.field_ids, self.unit_count = ids.copy(), mapping.copy(), names.copy(), units.copy()
                _require(all(np.array_equal(a, b) for a, b in zip((ids, mapping, names, units),
                         (self.unit_ids, self.unit_field_index, self.field_ids, self.unit_count))), "source unit/field identities changed")
                if case["block"] in BLOCKS:
                    self.truth[identity] = array["count_sum"].copy()
            field_bindings[identity] = dict(raw_report_sha256=index[identity]["raw_report_sha256"],
                                            field_summary_sha256=index[identity]["field_summary_sha256"],
                                            spatial_fields_sha256=field_digest)
        self.mean, self.sd, metadata = fitting_scale(fields, fit)
        _require(np.isfinite(self.mean).all() and np.isfinite(self.sd).all() and np.all(self.sd > 0)
                 and np.array_equal(metadata["field_ids"], self.field_ids)
                 and np.array_equal(metadata["unit_count"], self.unit_count), "canonical fitting scale invalid")
        self.binding = dict(acquisition_design_sha256=sha256_file(acquisition / "design.json"),
                            dataset_sha256=sha256_file(fields / "dataset.json"),
                            episode_table_sha256=dataset["episode_table_sha256"], native_source_identity=self.source_identity,
                            field_bindings=field_bindings)


def _metric_names(config):
    field = ["energy_score", "variogram_spatial", "variogram_cross_population"]
    field += [f"variogram_temporal_lag_{lag}" for lag in config["temporal_lags_bins"]]
    support = [f"refractory_count_violation_{width}bin" for width in config["refractory_support_windows_bins"]]
    return field, list(UNIT_METRICS) + support + list(COST_METRICS)


def _expected_rows(source, config, seed, model):
    return {(seed, c["block"], c["case_id"], c["template_id"], float(origin), float(horizon), model)
            for c, origin, horizon in itertools.product(source.validation, config["origins_ms"], config["horizons_ms"])}


def _is_prior_refit(run):
    return run.get("binding", {}).get("config", {}).get("schema") in PRIOR_PROFILES


def _prior_config_protocol(config):
    particle_keys = (PRIOR_MIXTURE_CONFIG_KEYS if config["schema"] == PRIOR_MIXTURE_SCHEMA else
                     ("prediction_particles",) if config["schema"] == PRIOR_OBSERVATION_SCHEMA else ())
    keys = PRIOR_PROTOCOL_KEYS + particle_keys
    return {key: config[key] for key in keys}


def _prior_sampling_budget(run):
    """Expose the implicit single path in the original analytic-KL objective."""
    mixture = run["training_objective"] == PRIOR_MIXTURE_OBJECTIVE
    observation = run["training_objective"] == PRIOR_OBSERVATION_OBJECTIVE
    return dict(prediction_particles=run["prediction_particles"] if mixture or observation else 1,
                posterior_target_draws=run["posterior_target_draws"] if mixture else None,
                posterior_target_integration="monte_carlo" if mixture else "not_used" if observation else "analytic_gaussian_kl",
                mixture_scope=run["mixture_scope"] if mixture or observation else None)


def _prior_protocol(run, checkpoint=None):
    """Keep the latent objective distinct from whole-episode observation ELBO."""
    config = run["binding"]["config"]
    profile = PRIOR_PROFILES.get(config.get("schema"))
    _require(profile is not None, "registered fixed-posterior prior profile required")
    objective, selection, prior_selection = profile
    expected = dict(training_objective=objective,
        selection_metric=selection, complete_episode_updates=False,
        complete_source_episode_encoding=True, posterior_targets_fixed=True,
        observation_parameters_refitted=False,
        observation_likelihood_evaluated_during_refit=config["schema"] == PRIOR_OBSERVATION_SCHEMA,
        fitting_episodes=56, validation_episodes=16, fitting_origin_windows=168, validation_origin_windows=48)
    _require(_is_prior_refit(run) and all(run.get(key) == value for key, value in expected.items())
             and config.get("observation_objective") == objective
             and config.get("training_sequence") == "complete_episode_causal_posterior_cache"
             and config.get("prior_origins_ms") == [1000, 3000, 5000]
             and config.get("prior_horizons_ms") == [5, 20, 80, 500]
             and config.get("prior_selection") == prior_selection
             and config.get("prior_target_gradient") == "stopped"
             and run.get("frozen_parameters_unchanged_in_attempt") is True,
             "registered fixed-posterior prior-refit objective and complete source encoding required")
    if config["schema"] == PRIOR_MIXTURE_SCHEMA:
        _require(all(type(config.get(key)) is int and type(run.get(key)) is int
                     and config[key] == run[key] == value
                     for key, value in (("prediction_particles", 8), ("posterior_target_draws", 4)))
                 and run.get("mixture_scope") == PRIOR_MIXTURE_SCOPE
                 and config.get("variational_rng") == "explicit_separate_checkpointed_generator",
                 "registered joint predictive-mixture particle, target-draw and random-stream profile required")
        if checkpoint is not None:
            _require(all(torch.is_tensor(checkpoint.get(key)) and checkpoint[key].dtype == torch.uint8
                         and checkpoint[key].ndim == 1 and checkpoint[key].numel() > 0
                         for key in ("variational_rng", "prior_dynamics_rng", "prior_target_rng")),
                     "predictive-mixture checkpoint requires three recorded random streams")
    elif config["schema"] == PRIOR_OBSERVATION_SCHEMA:
        _require(type(config.get("prediction_particles")) is int
                 and type(run.get("prediction_particles")) is int
                 and config["prediction_particles"] == run["prediction_particles"] == 8
                 and "posterior_target_draws" not in config and "posterior_target_draws" not in run
                 and config.get("variational_rng") == "explicit_separate_checkpointed_generator"
                 and config.get("maximum_epochs") == run.get("effective_maximum_epochs") == 80
                 and config.get("early_stopping_patience") == 12 and run.get("continuation") is None,
                 "registered native observation prior particle and fresh-refit budget required")
        _require(all((run.get(key) is value) if isinstance(value, bool) else run.get(key) == value
                     for key, value in PRIOR_OBSERVATION_DECLARATIONS.items()),
                 "native event-label, unused future-posterior or causal-boundary declaration differs")
        sources = run.get("trainer_sources", {})
        digest = sources.get(PRIOR_OBSERVATION_MODULE) if isinstance(sources, dict) else None
        _require(isinstance(digest, str) and len(digest) == 64
                 and all(character in "0123456789abcdef" for character in digest),
                 "registered native observation objective source required")
        if checkpoint is not None:
            _require(checkpoint.get("trainer_sources") == sources and "prior_target_rng" not in checkpoint
                     and all(torch.is_tensor(checkpoint.get(key)) and checkpoint[key].dtype == torch.uint8
                             and checkpoint[key].ndim == 1 and checkpoint[key].numel() > 0
                             for key in ("variational_rng", "prior_dynamics_rng")),
                     "native observation checkpoint requires its source and two training random streams")
    else:
        _require(not any(key in config for key in PRIOR_MIXTURE_CONFIG_KEYS),
                 "analytic prior profile cannot declare predictive-mixture particle settings")
    continuation = run.get("continuation")
    if continuation is not None:
        _require(isinstance(continuation, dict)
                 and continuation.get("schema") == "neural_population_trajectory_budget_continuation_v1"
                 and continuation.get("base_maximum_epochs") == config["maximum_epochs"]
                 and continuation.get("total_epochs") == run.get("effective_maximum_epochs")
                 and continuation.get("total_epochs", 0) > config["maximum_epochs"]
                 and continuation.get("current_source_hashes") == run.get("trainer_sources")
                 and continuation.get("inherited_early_stopping_patience") == config["early_stopping_patience"],
                 "prior continuation budget or source declaration changed")
        migration = validate_prior_budget_binding(continuation.get("parent_binding"), run["binding"],
            continuation.get("parent_source_hashes"), run.get("trainer_sources"))
        _require(migration == continuation.get("source_migration"),
                 "prior continuation source migration changed")
    initialization = run.get("refit_initialization")
    _require(isinstance(initialization, dict) and initialization.get("mode") == "selected_parent_prior_refit"
             and initialization.get("optimizer_and_random_streams_reinitialized") is True,
             "selected-parent prior-refit initialization required")
    expected_parameters = {"local_weight", "field_mix_logits", "control_weight", "transition_bias",
                           "decay_logit", "process_log_variance"}
    names = initialization.get("trainable_parameter_names", [])
    _require(isinstance(names, list) and len(names) == len(expected_parameters) and set(names) == expected_parameters,
             "prior-refit trainable parameter contract changed")
    _require(config.get("parent_checkpoint_sha256", {}).get(str(run["training_seed"]))
             == initialization.get("parent_checkpoint_sha256"), "prospective prior parent checkpoint changed")
    if checkpoint is not None:
        _require(checkpoint.get("initialization") == initialization, "prior-refit checkpoint and report initialization differ")
    return initialization


def _delivery_file(root, manifest, name, scope):
    _require(name in manifest, scope + " artifact is unavailable: " + name)
    path, row = root / name, manifest[name]
    _require(path.is_file() and path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"],
             scope + " binding changed: " + name)
    return path


def _prior_observation_compatibility(cell, rows, parent, parent_rows, old, current):
    """Accept identical code or the explicitly reviewed, delivered typing change."""
    _require(isinstance(old, str) and old and isinstance(current, str) and current,
             "prior parent and refit observation source hashes required")
    result = dict(parent_sha256=old, prior_refit_sha256=current)
    if old == current:
        return dict(result, mode="identical_source")
    reason = PRIOR_OBSERVATION_COMPATIBILITY.get((old, current))
    _require(reason is not None, "unreviewed prior observation implementation change")
    name = "Libs/population_dynamics/refractory_observation.py"
    for root, manifest, expected in ((parent, parent_rows, old), (cell, rows, current)):
        _delivery_file(root, manifest, name, "observation compatibility source")
        _require(manifest[name]["sha256"] == expected, "observation compatibility source hash differs from training")
    return dict(result, mode="verified_typing_only_change", reason=reason)


def _prior_parent_evidence(cell, run, checkpoint=None):
    """Bind the selected parent to staged files and its durable original delivery."""
    initialization = _prior_protocol(run, checkpoint)
    delivery = _read_json(cell / "delivery.json")
    rows = {row["path"]: row for row in delivery["files"]}
    _require(len(rows) == len(delivery["files"]), "duplicate prior-refit delivery paths")
    provenance = _read_json(_delivery_file(cell, rows, "prerequisites/prior_parent.json", "prior parent"))
    parent = Path(provenance["parent_cell"])
    _require(parent.resolve() != cell.resolve() and parent.resolve() == Path(delivery["parent_cell"]).resolve(),
             "prior parent cell changed")
    _require(sha256_file(parent / "delivery.json") == provenance["delivery_sha256"] == delivery["parent_delivery_sha256"],
             "original prior parent delivery changed")
    parent_delivery = _read_json(parent / "delivery.json")
    parent_rows = {row["path"]: row for row in parent_delivery["files"]}
    declared = {row["path"]: row for row in provenance["files"]}
    _require(len(parent_rows) == len(parent_delivery["files"]) and len(declared) == len(provenance["files"])
             and set(declared) == {"training/run.json", "training/best.pt"},
             "complete unique selected-parent file provenance required")
    for original, staged in (("training/run.json", "parent_training/run.json"), ("training/best.pt", "parent_training/best.pt")):
        _require(declared[original] == parent_rows.get(original), "prior parent file provenance changed")
        _delivery_file(parent, parent_rows, original, "original prior parent")
        _delivery_file(cell, rows, staged, "staged prior parent")
        _require(rows[staged]["sha256"] == parent_rows[original]["sha256"]
                 and rows[staged]["bytes"] == parent_rows[original]["bytes"], "staged prior parent differs")
    parent_run = _read_json(parent / "training/run.json")
    staged_run = _read_json(cell / "parent_training/run.json")
    seed, parent_binding, binding = run["training_seed"], parent_run["binding"], run["binding"]
    _require(parent_delivery.get("status") == "PASS" and parent_delivery.get("exit_code") == 0
             and parent_delivery.get("training_seed") == seed and parent_delivery.get("scientific_claim_qualified") is False
             and parent_delivery.get("calibration_or_test_accessed") is False
             and parent_delivery.get("core_neurons") == 100000 and parent_delivery.get("complete_episodes") == 72
             and parent_delivery.get("episode_ms") == 6000
             and parent_run == staged_run and parent_run.get("status") == "FITTED_DEVELOPMENT"
             and parent_run.get("training_seed") == parent_binding.get("seed") == binding.get("seed") == seed
             and parent_binding == initialization.get("parent_binding") and not _is_prior_refit(parent_run),
             "complete same-seed original prior parent training binding required")
    _require(all(key in binding and binding[key] == parent_binding.get(key) for key in PRIOR_SOURCE_BINDINGS),
             "prior parent native source or complete cohort differs")
    _require(parent_binding["config"].get("model_family") == binding["config"].get("model_family") == "population"
             and parent_binding["config"].get("representation") == binding["config"].get("representation") == "recurrent"
             and parent_run["model_configuration"] == run["model_configuration"],
             "prior parent native constructor or representation differs")
    observation_compatibility = _prior_observation_compatibility(cell, rows, parent, parent_rows,
        parent_binding.get("observation_implementation_sha256"), binding.get("observation_implementation_sha256"))
    _require(initialization.get("parent_report_sha256") == parent_rows["training/run.json"]["sha256"]
             and initialization.get("parent_checkpoint_sha256") == parent_rows["training/best.pt"]["sha256"] == parent_run["best_checkpoint_sha256"]
             and initialization.get("parent_selected_epoch") == parent_run["best_epoch"]
             and initialization.get("parent_completed_epochs") == parent_run["completed_epochs"],
             "prior selected checkpoint, epoch or report lineage changed")
    if checkpoint is not None:
        _require(checkpoint.get("binding") == binding and checkpoint.get("next_epoch") == run["best_epoch"] + 1,
                 "prior selected child checkpoint differs from its training report")
        selected = torch.load(cell / "parent_training/best.pt", map_location="cpu", weights_only=True)
        _require(selected.get("binding") == parent_binding and selected.get("next_epoch") == parent_run["best_epoch"] + 1
                 and selected.get("model_configuration") == checkpoint.get("model_configuration") == run["model_configuration"]
                 and set(selected["model"]) == set(checkpoint["model"]), "prior parent selected state differs")
        trainable = set(initialization["trainable_parameter_names"])
        _require(all(value.dtype == selected["model"][name].dtype and torch.equal(value, selected["model"][name])
                     for name, value in checkpoint["model"].items() if name not in trainable),
                 "prior refit changed a frozen parameter or source buffer")
    lineage = {key: initialization[key] for key in ("mode", "parent_checkpoint_sha256", "parent_report_sha256",
        "parent_selected_epoch", "parent_completed_epochs", "trainable_parameter_names", "optimizer_and_random_streams_reinitialized")}
    lineage.update(parent_cell=str(parent.resolve()), parent_delivery_sha256=provenance["delivery_sha256"],
                   observation_source_compatibility=observation_compatibility,
                   parent_training_config=parent_binding["config"],
                   parent_effective_maximum_epochs=parent_run.get("effective_maximum_epochs", parent_binding["config"]["maximum_epochs"]))
    return parent, parent_run, lineage


def _training_budget(run, checkpoint, checkpoint_digest):
    """Separate immutable base config from the executed continuation ceiling."""
    config = run['binding']['config']
    base = config['maximum_epochs']
    effective = run.get('effective_maximum_epochs', base)
    completed, best = run['completed_epochs'], run['best_epoch']
    _require(all(type(v) is int for v in (base, effective, completed, best))
             and 0 <= best < completed <= effective and base > 0,
             'valid executed epoch and selection boundaries required')
    continuation = run.get('continuation')
    if continuation is None:
        _require(effective == base and checkpoint.get('continuation') is None,
                 'extended training budget lacks registered continuation')
        lineage = None
    else:
        _require(continuation['schema'] == 'neural_population_trajectory_budget_continuation_v1'
                 and continuation['base_maximum_epochs'] == base
                 and continuation['total_epochs'] == effective and effective > base
                 and continuation['inherited_early_stopping_patience'] == config['early_stopping_patience']
                 and continuation['parent_completed_epochs'] <= completed
                 and continuation['current_source_hashes'] == run['trainer_sources'],
                 'continuation ceiling, patience or implementation lineage changed')
        if checkpoint.get('continuation') != continuation:
            _require(best < continuation['parent_completed_epochs']
                     and checkpoint_digest == continuation['parent_artifact_sha256']['best.pt'],
                     'selected checkpoint differs from continued or inherited best lineage')
        lineage = {key: continuation[key] for key in ('schema', 'total_epochs', 'base_maximum_epochs',
            'parent_completed_epochs', 'parent_checkpoint_sha256', 'parent_artifact_sha256',
            'source_migration', 'inherited_early_stopping_patience')}
    stale = completed - best - 1
    patience = config['early_stopping_patience']
    expected_stop = 'validation_patience' if stale >= patience else 'epoch_budget'
    _require(run['stop_reason'] == expected_stop and (stale >= patience or completed == effective),
             'fit did not stop at registered patience or effective ceiling')
    result = dict(base_maximum_epochs=base, effective_maximum_epochs=effective,
                  completed_epochs=completed, best_epoch=best, stop_reason=run['stop_reason'], continuation_lineage=lineage)
    if _is_prior_refit(run):
        _prior_protocol(run, checkpoint)
        result.update(training_objective=run["training_objective"], selection_metric=run["selection_metric"], inherited_training_is_additional=True)
    return result


def _campaign_cell(root, seed):
    """Require the completed fit/evaluation/replay cell before consuming results."""
    cell = root / f"seed_{seed}"
    delivery = _read_json(cell / "delivery.json")
    _require(delivery["status"] == "PASS" and delivery["exit_code"] == 0
             and delivery["training_seed"] == seed and delivery["scientific_claim_qualified"] is False,
             "complete registered population campaign cell required")
    rows = {row["path"]: row for row in delivery["files"]}
    _require(len(rows) == len(delivery["files"]), "duplicate campaign delivery paths")
    for name in ("training/run.json", "forecasts/evaluation.json", "replay.json"):
        row = rows[name]
        _require((cell / name).stat().st_size == row["bytes"]
                 and sha256_file(cell / name) == row["sha256"], "campaign delivery binding changed: " + name)
    replay = _read_json(cell / "replay.json")
    evaluation = _read_json(cell / "forecasts/evaluation.json")
    fit = _read_json(cell / "training/run.json")
    if _is_prior_refit(fit):
        _prior_parent_evidence(cell, fit)
        _require(delivery.get("source_units") == fit.get("units") == 100000
                 and delivery.get("fit_episodes") == 56 and delivery.get("validation_episodes") == 16
                 and delivery.get("complete_origins") == delivery.get("verified_origins") == 48
                 and delivery.get("score_rows") == 144 and delivery.get("frozen_parameters_unchanged") is True
                 and fit.get("status") == "FITTED_DEVELOPMENT"
                 and fit.get("calibration_outcomes_accessed") is False and fit.get("test_outcomes_accessed") is False,
                 "complete native prior-refit delivery and full-source encoding required")
    else:
        _require(delivery.get("core_neurons") == 100000 and delivery.get("complete_episodes") == 72
                 and delivery.get("episode_ms") == 6000 and delivery.get("fit_episodes") == 56
                 and delivery.get("validation_episodes") == 16 and delivery.get("verified_forecast_origins") == 48
                 and delivery.get("score_rows") == 144 and delivery.get("calibration_or_test_accessed") is False,
                 "complete registered population campaign cell required")
    _require(replay["status"] == "PASS" and replay["model_type"] == "population"
             and replay["training_seed"] == seed
             and replay["verified_origins"] == replay["expected_origins"] == 48
             and replay["verified_score_rows"] == 144
             and replay["all_forecast_unit_metrics_replayed"] is True
             and replay["all_origin_fields_and_latents_exact"] is True
             and replay["evaluation_sha256"] == sha256_file(cell / "forecasts/evaluation.json")
             and replay["scores_sha256"] == evaluation["scores_sha256"]
             and replay["source_checkpoint_sha256"] == fit["best_checkpoint_sha256"],
             "complete source-bound population all-origin replay required")
    return cell


def _load_run(root, model, seed, source, config, code, *, campaign_layout=False):
    if campaign_layout:
        _require(model == "population", "population model required for campaign layout")
        cell = _campaign_cell(root, seed)
        folder, training = cell / "forecasts", cell / "training"
    else:
        folder = root / model / f"seed_{seed}_forecast"
        training = root / model / f"seed_{seed}"
    report = _read_json(folder / "evaluation.json")
    run = _read_json(training / "run.json")
    _require(report["status"] == "EVALUATED_DEVELOPMENT" and run["status"] == "FITTED_DEVELOPMENT",
             "all six completed evaluations and training runs required")
    _require(report["configuration"] == config and report["model_type"] == model and report["training_seed"] == seed
             and report["evaluation_blocks"] == list(BLOCKS) and report["independent_blocks"] == 4
             and report["episodes"] == 16 and report["score_rows"] == 144,
             "matched frozen model/seed/4block/16episode forecast protocol required")
    _require(report["scientific_claim_qualified"] is False and report["formal_test_accessed"] is False
             and report["likelihood_scores_compared"] is False and report["all_unit_outputs_generated"] is True
             and report["field_outputs_from_unit_counts"] is True, "development sampled-count evidence scope required")
    _require(report["native_source_identity"] == source.source_identity, "evaluation native source changed")
    for name in ("acquisition_design_sha256", "dataset_sha256"):
        _require(report[name] == source.binding[name], "evaluation canonical data binding changed")
    _require(report["evaluator_implementation_sha256"] == code["population_trajectory_evaluation.py"]
             and report["unit_metrics_implementation_sha256"] == code["trajectory_sample_metrics.py"], "evaluated implementation changed")
    checkpoint_path = training / "best.pt"
    checkpoint_digest = sha256_file(checkpoint_path)
    _require(checkpoint_digest == report["checkpoint_sha256"] == run["best_checkpoint_sha256"], "selected checkpoint changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    binding = checkpoint["binding"]
    _require(binding == run["binding"] and binding["seed"] == seed, "training/checkpoint binding changed")
    for name in ("acquisition_design_sha256", "dataset_sha256"):
        _require(binding[name] == source.binding[name], "checkpoint source data changed")
    expected_model = {"poisson": "poisson_state_space.py", "renewal": "renewal_state_space.py",
                      "population": "population_state.py"}[model]
    _require(binding["model_implementation_sha256"] == code[expected_model], "checkpoint model implementation changed")
    if model in ("renewal", "population"):
        for key, filename in (("observation_implementation_sha256", "refractory_observation.py"),
                              ("loader_implementation_sha256", "native_renewal_data.py")):
            _require(binding[key] == code[filename], "checkpoint observation/loader implementation changed")
    if model == "population":
        _require(run["model_family"] == binding["config"]["model_family"] == "population",
                 "population training family changed")
        # Reuse the evaluator's constructor, native-clock, immutable-buffer and
        # implementation checks. Do not create a second observation-map validator.
        _load_model(training, source.acquisition, source.fields, model,
                    torch.device("cpu"), source.population_native)
    fit_config = binding["config"]
    _require(fit_config["training_seeds"] == list(SEEDS) and fit_config["fit_blocks"] == list(range(14))
             and fit_config["validation_blocks"] == list(BLOCKS) and fit_config["bin_ms"] == source.bin_ms,
             "registered training seeds and unchanged development split required")
    executed_budget = _training_budget(run, checkpoint, checkpoint_digest)
    prior_lineage = _prior_parent_evidence(training.parent, run, checkpoint)[2] if _is_prior_refit(run) else None
    if run.get("training_objective") == PRIOR_OBSERVATION_OBJECTIVE:
        if PRIOR_OBSERVATION_MODULE not in code:
            code[PRIOR_OBSERVATION_MODULE] = sha256_file(Path(__file__).with_name(PRIOR_OBSERVATION_MODULE))
        _require(code[PRIOR_OBSERVATION_MODULE] == run["trainer_sources"][PRIOR_OBSERVATION_MODULE],
                 "native observation objective differs from the analyzed source")
    del checkpoint
    field_metrics, other_metrics = _metric_names(config)
    metric_names = field_metrics + other_metrics
    with (folder / "scores.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames))
                 and set(reader.fieldnames) == set(IDENTITY_COLUMNS) | set(metric_names), "matched sampled-count score schema required")
        raw_rows = list(reader)
    _require(sha256_file(folder / "scores.csv") == report["scores_sha256"], "forecast score table changed")
    rows, keys = [], set()
    for raw in raw_rows:
        row = {k: raw[k] for k in IDENTITY_COLUMNS}
        for name in ("seed", "block"):
            row[name] = _integer(raw[name], name)
        for name in ("origin_ms", "horizon_ms"):
            row[name] = _finite(raw[name], name)
        key = tuple(row[name] for name in IDENTITY_COLUMNS[:-1])
        _require(key not in keys, "duplicate forecast case/origin/horizon")
        keys.add(key)
        bins = row["horizon_ms"] / source.bin_ms
        _require(bins == int(bins) and bins > 0, "aligned forecast horizon required")
        for name in metric_names:
            unavailable = ((name.startswith("variogram_temporal_lag_") and int(name.rsplit("_", 1)[1]) >= bins)
                           or (name.startswith("refractory_count_violation_") and int(name.rsplit("_", 1)[1][:-3]) > bins))
            if unavailable:
                _require(raw[name] == "", "unavailable temporal/support metric must be blank")
                row[name] = None
            elif name == "peak_gpu_allocated_bytes" and raw[name] == "":
                row[name] = None
            else:
                _require(raw[name] not in (None, ""), "missing eligible numeric metric: " + name)
                row[name] = _finite(raw[name], name)
                if name in COST_METRICS or name in UNIT_METRICS[1:]:
                    _require(row[name] >= 0, "nonnegative cost/rate/interval metric required")
                if name == "unit_total_coverage_90" or name.startswith("refractory_count_violation_"):
                    _require(0 <= row[name] <= 1, "coverage/support fraction outside [0,1]")
                if name == "peak_gpu_allocated_bytes":
                    _require(row[name] == int(row[name]), "integer GPU allocation bytes required")
        expected_name = f"{row['case_id']}_origin_{row['origin_ms']:g}.npz"
        _require(row["ensemble_artifact"] == expected_name, "ensemble artifact identity changed")
        rows.append(row)
    _require(len(rows) == 144 and keys == _expected_rows(source, config, seed, model), "missing, extra or unmatched forecast coverage")
    expected_artifacts = {row["ensemble_artifact"] for row in rows} | {"full_unit_samples.npz"}
    artifact_names = set(report["artifacts"])
    _require(artifact_names in (expected_artifacts, expected_artifacts | {"fitting_field_scale.npz"}), "missing or undeclared ensemble artifacts")
    for name, expected in report["artifacts"].items():
        _require(Path(name).name == name and sha256_file(folder / name) == expected, "forecast artifact changed: " + name)
    scale_path = folder / "fitting_field_scale.npz"
    with np.load(scale_path, allow_pickle=False) as scale:
        for name, expected in (("mean_hz", source.mean), ("standard_deviation_hz", source.sd),
                               ("unit_count", source.unit_count), ("field_ids", source.field_ids)):
            _require(np.array_equal(scale[name], expected), "saved scale differs from canonical fitting data: " + name)
    # Only retained unit identity is consumed here. Score math is separately
    # unit-tested; the complete raw unit sample arrays exist at one origin only.
    with np.load(folder / "full_unit_samples.npz", allow_pickle=False) as samples:
        _require(np.array_equal(samples["unit_ids"], source.unit_ids), "retained unit sample identity changed")
    receipt = dict(evaluation_sha256=sha256_file(folder / "evaluation.json"), training_report_sha256=sha256_file(training / "run.json"),
                   checkpoint_sha256=checkpoint_digest, scores_sha256=report["scores_sha256"],
                   fitting_field_scale_sha256=sha256_file(scale_path), artifacts=report["artifacts"], training_config=fit_config, executed_training_budget=executed_budget)
    if prior_lineage is not None:
        receipt.update(prior_refit_lineage=prior_lineage, training_objective=run["training_objective"], selection_metric=run["selection_metric"])
    measured_costs = _evaluation_costs(report, rows, source, config)
    if measured_costs is not None:
        receipt["measured_evaluation_costs"] = measured_costs
    return folder, rows, receipt


def _mean(values):
    finite = [value for value in values if value is not None]
    _require(not finite or len(finite) == len(values), "mixed missing/available values in matched aggregate")
    if not finite:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.mean(finite))
    _require(np.isfinite(result), "nonfinite aggregate mean")
    return result


def _cost_value(value, name):
    if value in (None, ""):
        return None
    result = _finite(value, name)
    _require(result >= 0, "nonnegative measured cost required: " + name)
    if name.endswith("_bytes"):
        _require(result == int(result), "integer measured allocation bytes required: " + name)
    return result


def _origin_key(row):
    return (_integer(row["seed"], "seed"), str(row["case_id"]),
            _finite(row["origin_ms"], "origin_ms"), str(row["method"]))


def _evaluation_costs(report, rows, source, config):
    """Validate optional costs inside the already source-bound evaluation report."""
    return validate_evaluation_costs(report, rows, config, validation=source.validation,
        unit_count=len(source.unit_ids), field_count=len(source.field_ids), bin_ms=source.bin_ms,
        source_dt_ms=source.source_identity["dt_ms"],
        episode_bin_counts={identity: len(values) for identity, values in source.truth.items()})


def validate_evaluation_costs(report, rows, config, *, validation, unit_count,
                              field_count, bin_ms, source_dt_ms, episode_bin_counts, case_cost_contracts=None):
    """Validate evaluator costs against already-bound source metadata.

    Callers must establish source, split, unit/field and complete-episode
    identities before calling. Optional per-case contracts describe source-bound
    maps whose complete episode tensors are not materialized. This validator does
    not load source outcomes or change the default fixed-map cost contract.
    """
    present = [name in report for name in ("origin_costs", "episode_costs")]
    if not any(present):
        return None
    _require(all(present) and all(isinstance(report[name], list) for name in ("origin_costs", "episode_costs")),
             "complete measured origin and episode cost records required")
    hardware = report.get("hardware")
    _require(isinstance(hardware, dict), "measured evaluation hardware required")
    for name in ("python", "numpy", "torch", "host", "device"):
        _require(isinstance(hardware.get(name), str) and hardware[name], "measured evaluation hardware field required: " + name)
    for name in ("cpu_threads", "cpu_interop_threads"):
        _require(type(hardware.get(name)) is int and hardware[name] > 0, "positive measured evaluation threads required")
    device = hardware["device"]
    device_type = device.split(":", 1)[0]
    _require(device_type in ("cpu", "cuda"), "measured CPU or CUDA evaluation device required")
    for name in ("model_parameter_dtypes", "model_parameter_devices"):
        values = hardware.get(name)
        _require(isinstance(values, list) and values and all(isinstance(value, str) and value for value in values)
                 and len(values) == len(set(values)), "unique measured model runtime values required: " + name)
    _require(all(value.split(":", 1)[0] == device_type for value in hardware["model_parameter_devices"]),
             "measured parameter and evaluation devices differ")
    scope = report.get("cost_scope")
    _require(isinstance(scope, dict) and scope.get("speedup_qualified") is False,
             "unqualified measured evaluator cost scope required")
    for name in ("function_total_seconds", "elapsed_seconds", "setup_seconds", "episode_costs",
                 "origin_total_seconds", "horizon_reuse", "peak_process_rss_bytes"):
        _require(isinstance(scope.get(name), str) and scope[name], "explicit measured evaluator cost boundary required: " + name)
    _require(isinstance(scope.get("excluded_costs"), list) and scope["excluded_costs"],
             "excluded evaluator costs must remain explicit")

    def cost(value, name):
        _require(not isinstance(value, bool), "numeric measured evaluator cost required: " + name)
        result = _cost_value(value, name)
        _require(result is not None, "missing measured evaluator cost: " + name)
        return result

    def number(value, name, *, integer=False):
        _require(value is not None and not isinstance(value, bool), "missing measured cost identity: " + name)
        return _integer(value, name) if integer else _finite(value, name)

    function_costs = {name: cost(report.get(name), name) for name in (
        "setup_seconds", "elapsed_seconds", "score_hash_seconds", "function_total_seconds", "peak_process_rss_bytes")}
    _require(function_costs["function_total_seconds"] >= sum(function_costs[name] for name in (
        "setup_seconds", "elapsed_seconds", "score_hash_seconds")), "measured evaluator function cost boundary differs")
    cases = {case["case_id"]: case for case in validation}
    _require(len(cases) == len(validation), "unique source validation cases required for costs")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["case_id"], row["origin_ms"])].append(row)
    expected = {(identity, float(origin)) for identity, origin in itertools.product(cases, config["origins_ms"])}
    _require(set(grouped) == expected, "complete source forecast origin identities required for costs")
    horizons = list(config["horizons_ms"])
    draws = config["joint_samples"]
    contracts = None
    if case_cost_contracts is not None:
        _require(isinstance(case_cost_contracts, dict) and set(case_cost_contracts) == set(cases),
                 "measured per-case cost contracts must exactly cover validation cases")
        contracts = {}
        required = {"block", "unit_count", "field_count", "bin_ms", "source_dt_ms", "episode_bins",
                    "full_unit_retention", "episode_inputs_materialized"}
        for identity, raw in case_cost_contracts.items():
            _require(isinstance(raw, dict) and set(raw) == required, "complete measured per-case cost contract required")
            _require(isinstance(raw["block"], str) and raw["block"]
                     and raw["block"] == cases[identity]["block"], "measured per-case source block differs")
            contract = dict(raw)
            for name in ("unit_count", "field_count", "episode_bins"):
                contract[name] = number(raw[name], name, integer=True)
                _require(contract[name] > 0, "positive measured per-case output denominator required: " + name)
            for name in ("bin_ms", "source_dt_ms"):
                contract[name] = number(raw[name], name)
                _require(contract[name] > 0, "positive measured per-case clock required: " + name)
            _require(raw["full_unit_retention"] == "all_origins" and raw["episode_inputs_materialized"] is False,
                     "explicit all-origin retention and unmaterialized episode cost contract required")
            for horizon in horizons:
                _require(number(horizon / contract["bin_ms"], "generated_bins", integer=True) > 0,
                         "positive measured per-case forecast horizon required")
            contracts[identity] = contract
    else:
        longest = number(max(horizons) / bin_ms, "generated_bins", integer=True)
        units, fields = unit_count, field_count
    first_identity = (validation[0]["case_id"], float(config["origins_ms"][0]))
    origins, seen = [], set()
    for raw in report["origin_costs"]:
        _require(isinstance(raw, dict), "measured origin cost object required")
        key = (raw.get("case_id"), number(raw.get("origin_ms"), "origin_ms"))
        _require(key in expected and key not in seen, "duplicate or unmatched measured origin cost")
        seen.add(key)
        if contracts is None:
            _require(number(raw.get("block"), "block", integer=True) == cases[key[0]]["block"], "measured origin source block differs")
        else:
            contract = contracts[key[0]]
            _require(raw.get("block") == contract["block"], "measured origin source block differs")
            units, fields = contract["unit_count"], contract["field_count"]
            longest = number(max(horizons) / contract["bin_ms"], "generated_bins", integer=True)
        _require(raw.get("horizons_ms") == horizons, "measured origin horizon siblings differ")
        for name, wanted in (("generated_horizon_ms", max(horizons)), ("generated_bins", longest),
                             ("draws", draws), ("unit_count", units), ("field_count", fields),
                             ("score_rows", len(horizons)), ("generated_unit_count_values", draws * longest * units)):
            _require(number(raw.get(name), name, integer=name != "generated_horizon_ms") == wanted,
                     "measured origin output denominator differs: " + name)
        retained = key == first_identity if contracts is None else True
        _require(type(raw.get("stored_full_unit_counts")) is bool
                 and raw["stored_full_unit_counts"] == retained, "measured full-unit retention origin differs")
        _require(raw.get("generated_unit_count_dtype") == "torch.int64" and raw.get("field_count_dtype") == "int64"
                 and raw.get("field_metric_dtype") == config["field_metric_dtype"], "measured output dtypes differ")
        _require(raw.get("latent_dtype") in hardware["model_parameter_dtypes"], "measured latent and parameter dtypes differ")
        for name in ("generated_unit_count_device", "latent_device"):
            _require(raw.get(name) in hardware["model_parameter_devices"], "measured output and parameter devices differ")
        siblings = grouped[key]
        _require(len(siblings) == len(horizons) and {row["horizon_ms"] for row in siblings} == set(horizons),
                 "complete measured origin horizon score rows required")
        measured = dict(raw)
        _require(all(name in raw for name in COST_METRICS), "complete measured origin legacy timers required")
        for name in EVALUATOR_ORIGIN_COSTS + COST_METRICS:
            measured[name] = (None if name == "peak_gpu_allocated_bytes" and raw.get(name) is None
                              else cost(raw.get(name), name))
        _require(device_type != "cpu" or measured["peak_gpu_allocated_bytes"] is None,
                 "CPU evaluation cannot report CUDA allocation")
        for name in COST_METRICS:
            _require(all(row[name] == measured[name] for row in siblings), "measured origin legacy timer differs: " + name)
        component_total = sum(measured[name] for name in (*COST_METRICS[:3], "scoring_seconds", "serialization_seconds", "hash_seconds", "csv_seconds"))
        _require(measured["origin_total_seconds"] >= component_total, "measured origin total excludes a reported component")
        _require(measured["peak_process_rss_bytes"] <= function_costs["peak_process_rss_bytes"],
                 "measured origin RSS exceeds process high-water mark")
        origins.append(measured)
    _require(seen == expected, "missing measured origin cost")
    episodes, seen = [], set()
    for raw in report["episode_costs"]:
        _require(isinstance(raw, dict), "measured episode cost object required")
        identity = raw.get("case_id")
        _require(identity in cases and identity not in seen, "duplicate or unmatched measured episode cost")
        seen.add(identity)
        if contracts is None:
            matching_block = number(raw.get("block"), "block", integer=True) == cases[identity]["block"]
            expected_bins, case_bin_ms, case_dt_ms = episode_bin_counts[identity], bin_ms, source_dt_ms
            units = unit_count
        else:
            contract = contracts[identity]
            matching_block = raw.get("block") == contract["block"]
            expected_bins, case_bin_ms, case_dt_ms = contract["episode_bins"], contract["bin_ms"], contract["source_dt_ms"]
            units = contract["unit_count"]
        _require(matching_block and raw.get("origins_ms") == list(config["origins_ms"]),
                 "measured episode source or origin identities differ")
        ticks, bins = number(raw.get("native_duration_ticks"), "native_duration_ticks", integer=True), number(raw.get("native_bins"), "native_bins", integer=True)
        _require(number(raw.get("unit_count"), "unit_count", integer=True) == units and bins == expected_bins
                 and ticks * case_dt_ms == bins * case_bin_ms,
                 "measured complete episode output denominator differs")
        if contracts is None:
            _require(all(raw.get(name) == "float32" for name in ("count_dtype", "stimulus_dtype"))
                     and all(raw.get(name) == "torch.float32" for name in ("count_tensor_dtype", "stimulus_tensor_dtype")),
                     "measured episode input dtypes differ")
            _require(all(isinstance(raw.get(name), str) and raw[name].split(":", 1)[0] == device_type
                         for name in ("count_tensor_device", "stimulus_tensor_device")), "measured episode input device differs")
        else:
            absent = ("count_dtype", "stimulus_dtype", "count_tensor_dtype", "stimulus_tensor_dtype",
                      "count_tensor_device", "stimulus_tensor_device")
            _require(raw.get("episode_inputs_materialized") is False
                     and all(name in raw and raw[name] is None for name in absent),
                     "measured unmaterialized episode must not report input dtype or device")
        measured = dict(raw)
        measured.update({name: cost(raw.get(name), name) for name in EVALUATOR_EPISODE_COSTS})
        if contracts is not None:
            _require(measured["window_seconds"] == measured["transfer_seconds"] == 0,
                     "measured unmaterialized episode cannot repeat origin window or transfer costs")
        _require(measured["episode_preparation_seconds"] >= sum(measured[name] for name in EVALUATOR_EPISODE_COSTS[:3]),
                 "measured episode preparation excludes a reported component")
        episodes.append(measured)
    _require(seen == set(cases), "missing measured episode cost")
    _require(number(report.get("origins"), "origins", integer=True) == len(origins)
             and number(report.get("generated_unit_count_values"), "generated_unit_count_values", integer=True)
             == sum(origin["generated_unit_count_values"] for origin in origins), "measured evaluation output denominator differs")
    _require(report["episodes"] == len(episodes) and report["score_rows"] == len(rows), "measured evaluation row counts differ")
    _require(function_costs["elapsed_seconds"] >= sum(origin["origin_total_seconds"] for origin in origins)
             + sum(episode["episode_preparation_seconds"] for episode in episodes),
             "measured evaluation loop excludes a reported origin or episode cost")
    return dict(hardware=hardware, cost_scope=scope, function_costs=function_costs,
                origin_costs=origins, episode_costs=episodes)


def _measured_origin_columns(receipt, case_id, origin_ms):
    """Charge complete episode preparation once at its first configured origin."""
    result = {name: None for name in EVALUATOR_COST_COLUMNS}
    costs = receipt.get("measured_evaluation_costs")
    if costs is None:
        return result
    origin = next(row for row in costs["origin_costs"] if (row["case_id"], row["origin_ms"]) == (case_id, origin_ms))
    episode = next(row for row in costs["episode_costs"] if row["case_id"] == case_id)
    result.update({name: origin[name] for name in EVALUATOR_COST_COLUMNS if name in origin})
    first_origin = episode["origins_ms"][0]
    result.update(episode_cost_origin_ms=first_origin,
                  episode_cost_scope="one complete episode preparation shared by all origins; recorded only at episode_cost_origin_ms",
                  evaluation_hardware=json.dumps(costs["hardware"], sort_keys=True),
                  evaluation_cost_scope=json.dumps(costs["cost_scope"], sort_keys=True))
    if origin_ms == first_origin:
        result.update(episode_loading_seconds=episode["loading_seconds"], episode_window_seconds=episode["window_seconds"],
                      episode_transfer_seconds=episode["transfer_seconds"], episode_preparation_seconds=episode["episode_preparation_seconds"])
    return result


def _origin_cost_fidelity(fidelity_rows, cost_rows, config, cost_scope):
    """Join measured components to complete longest-horizon fidelity conditions."""
    horizon = max(config["horizons_ms"])
    _require(horizon == 500. and config["joint_samples"] == 16,
             "the measured 500ms, sixteen-draw output contract is required")
    metric_names = _metric_names(config)[0] + list(UNIT_METRICS)
    metric_names += [f"refractory_count_violation_{w}bin" for w in config["refractory_support_windows_bins"]]
    grouped = defaultdict(list)
    for row in fidelity_rows:
        grouped[_origin_key(row)].append(row)
    longest = {}
    for key, siblings in grouped.items():
        _require(len(siblings) == len(config["horizons_ms"])
                 and {_finite(row["horizon_ms"], "horizon_ms") for row in siblings}
                 == set(config["horizons_ms"]), "complete horizon siblings required for measured cost")
        longest[key] = next(row for row in siblings if _finite(row["horizon_ms"], "horizon_ms") == horizon)
    methods = {key[-1] for key in longest}
    _require(len(longest) == 144 * len(methods), "complete three-seed cost/fidelity origin coverage required")
    cost_index, scope_index, numeric_columns = {}, {}, set(COST_COMPONENTS)
    for raw in cost_rows:
        row = dict(raw)
        if row.get("method") in (None, ""):
            _require(len(methods) == 1, "missing cost method requires one identified baseline model")
            row["method"] = next(iter(methods))
        key = _origin_key(row)
        _require(key not in cost_index, "duplicate measured cost origin")
        _require(key in longest, "measured cost has an unmatched fidelity condition")
        wanted = longest[key]
        for name in ("block", "template_id"):
            if row.get(name) not in (None, ""):
                actual = _integer(row[name], name) if name == "block" else str(row[name])
                expected = _integer(wanted[name], name) if name == "block" else str(wanted[name])
                _require(actual == expected, "measured cost source condition identity differs: " + name)
        if row.get("generated_horizon_ms") not in (None, ""):
            _require(_finite(row["generated_horizon_ms"], "generated_horizon_ms") == horizon,
                     "measured cost belongs to an ineligible generation horizon")
        measured = {name: _cost_value(value, name) for name, value in row.items()
                    if name.endswith(("_seconds", "_bytes"))}
        numeric_columns.update(measured)
        _require(all(measured.get(name) is not None for name in COST_METRICS[:3]),
                 "measured inference, full-unit generation and projection components are required")
        for name in measured:
            values = [_cost_value(sibling.get(name), name) for sibling in grouped[key]
                      if name in sibling]
            if values:
                _require(len(values) == len(grouped[key]) and all(value == measured[name] for value in values),
                         "full-horizon cost differs across reused horizon rows: " + name)
        cost_index[key] = measured
        if row.get('graph_setup_id') not in (None, ''):
            _require(measured.get('graph_setup_seconds') is not None
                     and measured.get('graph_setup_charged_seconds') is not None,
                     'identified graph setup requires raw and charged measurements')
            scope_index[key] = dict(graph_setup_id=str(row['graph_setup_id']),
                graph_setup_cases=_integer(row.get('graph_setup_cases'), 'graph_setup_cases'),
                graph_setup_source=str(row.get('graph_setup_source', '')))
    _require(set(cost_index) == set(longest), "missing measured cost condition")
    # LFADS setup is recorded on each case row, but each graph lifetime serves
    # the sixteen cases at a fixed seed/origin. Preserve those raw measurements;
    # never sum the repeated values or manufacture an amortized total here.
    lifetimes = defaultdict(list)
    for key, costs in cost_index.items():
        if costs.get("graph_setup_seconds") is not None:
            scope = scope_index.get(key)
            lifetime = (key[0], key[3], scope['graph_setup_id']) if scope else (key[0], key[3], key[2])
            lifetimes[lifetime].append(key)
    for keys in lifetimes.values():
        scopes = [scope_index.get(key) for key in keys]
        values = [cost_index[key]['graph_setup_seconds'] for key in keys]
        _require(all(value == values[0] for value in values), 'graph setup differs within its measured lifetime')
        if any(scopes):
            _require(all(scopes) and len({key[2] for key in keys}) == 1
                     and all(scope['graph_setup_cases'] == len(keys) and scope['graph_setup_source'] in ('parent', 'current') for scope in scopes)
                     and len({scope['graph_setup_source'] for scope in scopes}) == 1,
                     'complete identified graph setup lifetime required')
            charges = [cost_index[key]['graph_setup_charged_seconds'] for key in keys]
            _require(charges.count(values[0]) == 1 and all(value in (0., values[0]) for value in charges),
                     'each measured graph setup must be charged exactly once')
        else:
            _require(len(values) == 16, 'complete sixteen-case seed/origin graph setup lifetime required')
    columns = list(COST_COMPONENTS) + sorted(numeric_columns - set(COST_COMPONENTS))
    result = []
    for key in sorted(longest):
        row, costs = longest[key], cost_index[key]
        output = dict(seed=key[0], block=_integer(row["block"], "block"), case_id=key[1],
                      template_id=str(row["template_id"]), origin_ms=key[2], method=key[3],
                      horizon_ms=horizon, generated_horizon_ms=horizon, joint_samples=16,
                      source_unit_count=100000, field_count=128, all_unit_outputs_generated=True,
                      ensemble_artifact=row["ensemble_artifact"], cost_scope=cost_scope,
                      scientific_claim_qualified=False)
        output.update({name: costs.get(name) for name in columns})
        if scope_index:
            output.update({name: scope_index.get(key, {}).get(name) for name in (
                'graph_setup_id', 'graph_setup_cases', 'graph_setup_source')})
        output.update({name: None if row.get(name) in (None, "") else _finite(row[name], name)
                       for name in metric_names})
        result.append(output)
    return result


def _baseline_comparisons(roots, report, all_rows, config, *, cost_fidelity_rows=None):
    """Consume already verified baseline tables without repeating generation."""
    comparisons, receipts = [], []
    objective = report.get("common_training_budget", {}).get("training_objective")
    prior = objective in PRIOR_METHODS
    candidate_method = PRIOR_METHODS[objective] if prior else "population"
    metric_names = _metric_names(config)[0] + list(UNIT_METRICS)
    metric_names += [f"refractory_count_violation_{w}bin" for w in config["refractory_support_windows_bins"]]
    identity = ("seed", "block", "case_id", "template_id", "origin_ms", "horizon_ms")
    candidates = {tuple(str(row[k]) for k in identity): row for row in all_rows}
    _require(len(candidates) == 432, "complete three-seed candidate comparison required")
    seen = set()
    for root in roots:
        delivery = _read_json(root / "delivery.json")
        _require(delivery["status"] == "ARCHIVED_VERIFIED_DEVELOPMENT"
                 and delivery["training_seeds"] == list(SEEDS)
                 and delivery["calibration_outcomes_accessed"] is False
                 and delivery["test_outcomes_accessed"] is False,
                 "verified development-only baseline delivery required")
        manifest = {row["path"]: row for row in delivery["files"]}
        for name in ("analysis/analysis.json", "analysis/forecast_all_seeds.csv", "analysis/costs_by_origin.csv"):
            _require(name in manifest, "baseline aggregate manifest lacks required measured costs: " + name)
            row = manifest[name]
            _require((root / name).stat().st_size == row["bytes"]
                     and sha256_file(root / name) == row["sha256"], "baseline aggregate delivery changed: " + name)
        baseline = _read_json(root / "analysis/analysis.json")
        _require(baseline["status"] in ("REPLAYED_DEVELOPMENT", "DESCRIBED_REPLAYED_DEVELOPMENT")
                 and baseline["scientific_claim_qualified"] is False
                 and baseline["formal_test_accessed"] is False,
                 "completed descriptive baseline analysis required")
        cost_path = root / "analysis/costs_by_origin.csv"
        cost_digest = sha256_file(cost_path)
        _require(baseline.get("artifacts", {}).get("costs_by_origin.csv") == cost_digest,
                 "baseline analysis measured-cost artifact binding changed")
        _require(isinstance(baseline.get("costs"), str) and baseline["costs"],
                 "baseline measured cost scope is required")
        source = baseline["source_binding"]
        for key in ("acquisition_design_sha256", "dataset_sha256"):
            _require(source[key] == report["source_binding"][key], "baseline source cohort differs")
        for key, value in source["native_source_identity"].items():
            _require(value == report["source_binding"]["native_source_identity"][key], "baseline native source differs")
        # Scope labels may have changed; every measurement/input setting must match.
        _require({k: v for k, v in baseline["configuration"].items() if k != "scope"}
                 == {k: v for k, v in config.items() if k != "scope"}, "baseline measurement protocol differs")
        verification = delivery["verification"]
        if isinstance(verification, dict):
            _require(verification["all_unit_metrics_independently_replayed"] is True
                     and verification["seeds"] == list(SEEDS)
                     and verification["score_rows"] == baseline["score_rows"], "baseline unit replay incomplete")
        else:
            _require(len(verification) == 3 and {r["seed"] for r in verification} == set(SEEDS)
                     and all(r["verified_origins"] == 48 and r["verified_score_rows"] == 144
                             and r["field_scores_recomputed"] == 720 for r in verification)
                     and baseline["all_unit_metrics_independently_replayed"] is True,
                     "baseline full field/unit replay incomplete")
        with (root / "analysis/forecast_all_seeds.csv").open(newline="") as stream:
            raw = list(csv.DictReader(stream))
        _require(len(raw) == baseline["score_rows"], "baseline table coverage changed")
        with cost_path.open(newline="") as stream:
            measured_costs = list(csv.DictReader(stream))
        methods = sorted({row["method"] for row in raw})
        _require(not seen.intersection(methods), "duplicate baseline model comparison")
        seen.update(methods)
        for method in methods:
            rows = [r for r in raw if r["method"] == method]
            indexed = {tuple(r[k] for k in identity): r for r in rows}
            _require(len(rows) == len(indexed) == 432 and set(indexed) == set(candidates),
                     "baseline/candidate complete condition identities differ")
            for block, horizon, metric in itertools.product(BLOCKS, config["horizons_ms"], metric_names):
                keys = [key for key, row in candidates.items() if row["block"] == block and row["horizon_ms"] == horizon]
                _require(len(keys) == 36, "complete three-seed template/origin block required")
                values = [candidates[key][metric] for key in keys]
                reference = [None if indexed[key][metric] == "" else _finite(indexed[key][metric], metric) for key in keys]
                _require(all((a is None) == (b is None) for a, b in zip(values, reference)),
                         "baseline/candidate eligible metric availability differs")
                if values[0] is None:
                    continue
                candidate_mean, baseline_mean = _mean(values), _mean(reference)
                comparisons.append(dict(block=block, horizon_ms=horizon, metric=metric,
                    candidate=candidate_method, baseline=method, candidate_mean=candidate_mean,
                    **({"candidate_training_objective": objective} if prior else {}),
                    baseline_mean=baseline_mean, candidate_minus_baseline=candidate_mean-baseline_mean,
                    training_seeds=3, template_origins_per_seed=12, interpretation="development_descriptive"))
        # Normalize sparse legacy cost identities only after the complete
        # source/protocol/fidelity join has established their condition meaning.
        joined_costs = _origin_cost_fidelity(raw, measured_costs, config, baseline["costs"])
        if cost_fidelity_rows is not None:
            cost_fidelity_rows.extend(joined_costs)
        receipts.append(dict(root=str(root), delivery_sha256=sha256_file(root / "delivery.json"),
                             analysis_sha256=sha256_file(root / "analysis/analysis.json"),
                             models=methods, training_bindings=baseline["model_seed_bindings"],
                             cost_artifact_sha256=cost_digest, cost_scope=baseline["costs"]))
    return comparisons, receipts



def _continuation_training_cost(cell, run):
    """Join delivered parent costs to incremental continuation attempt costs."""
    continuation = run["continuation"]
    delivery = _read_json(cell / "delivery.json")
    rows = {row["path"]: row for row in delivery["files"]}
    _require(len(rows) == len(delivery["files"]), "duplicate continuation delivery paths")

    def bound_record(root, manifest, name):
        _require(name in manifest, "continuation training cost artifact is unavailable: " + name)
        path, row = root / name, manifest[name]
        _require(path.is_file() and path.stat().st_size == row["bytes"]
                 and sha256_file(path) == row["sha256"], "continuation training cost binding changed: " + name)
        return _read_json(path)

    staged_delivery_name = "prerequisites/budget_parent/delivery.json"
    parent_delivery = bound_record(cell, rows, staged_delivery_name)
    lineage = delivery.get("optimization_budget_continuation", {})
    parent = Path(lineage.get("parent", ""))
    expected_digest = lineage.get("parent_delivery_sha256")
    _require(expected_digest == rows[staged_delivery_name]["sha256"]
             and (parent / "delivery.json").is_file()
             and sha256_file(parent / "delivery.json") == expected_digest,
             "continuation original parent delivery binding changed")
    _require(parent.resolve() != cell.resolve(), "continuation cost cannot reuse its own parent")
    parent_rows = {row["path"]: row for row in parent_delivery["files"]}
    _require(len(parent_rows) == len(parent_delivery["files"]), "duplicate parent training cost paths")
    parent_run = bound_record(parent, parent_rows, "training/run.json")
    _require(parent_rows.get("training/latest.pt", {}).get("sha256") == continuation.get("parent_checkpoint_sha256"),
             "continuation parent latest checkpoint cost receipt changed")
    staged_run = bound_record(cell, rows, "prerequisites/budget_parent/training/run.json")
    seed, binding = run["training_seed"], run["binding"]
    prior = _is_prior_refit(run)
    _require(parent_delivery.get("status") == "PASS" and parent_delivery.get("exit_code") == 0
             and parent_delivery.get("training_seed") == seed
             and parent_delivery.get("scientific_claim_qualified") is False
             and ((parent_run.get("calibration_outcomes_accessed") is False
                   and parent_run.get("test_outcomes_accessed") is False) if prior else
                  parent_delivery.get("calibration_or_test_accessed") is False),
             "complete same-seed development parent cost delivery required")
    _require(parent_run == staged_run and parent_run.get("status") == "FITTED_DEVELOPMENT"
             and parent_run.get("training_seed") == binding.get("seed") == seed
             and parent_run.get("binding") == continuation.get("parent_binding")
             and parent_run.get("continuation") is None,
             "continuation parent training model/data/seed binding changed")
    sources = run.get("trainer_sources")
    _require(isinstance(sources, dict) and bool(sources)
             and parent_run.get("trainer_sources") == continuation.get("parent_source_hashes")
             and sources == continuation.get("current_source_hashes"),
             "continuation parent training source binding changed")
    if prior:
        _require(_is_prior_refit(parent_run)
                 and parent_run.get("refit_initialization") == run.get("refit_initialization"),
                 "prior continuation must preserve the original refit initialization")
        migration = validate_prior_budget_binding(parent_run["binding"], binding,
                                                  parent_run["trainer_sources"], sources)
        _require(continuation.get("source_migration") == migration,
                 "prior continuation source migration changed")
        _prior_parent_evidence(parent, parent_run)
    else:
        _require(parent_run["binding"] == binding and sources == parent_run["trainer_sources"]
                 and continuation.get("source_migration") == {},
                 "continuation parent training source binding changed")
    _require(continuation.get("schema") == "neural_population_trajectory_budget_continuation_v1"
             and continuation.get("parent_artifact_sha256", {}).get("run.json")
                 == parent_rows["training/run.json"]["sha256"]
             and continuation.get("parent_checkpoint_sha256") == parent_run.get("latest_checkpoint_sha256")
             == lineage.get("parent_checkpoint_sha256")
             and continuation.get("parent_completed_epochs") == parent_run.get("completed_epochs")
             == lineage.get("parent_completed_epochs")
             and continuation.get("base_maximum_epochs") == binding["config"]["maximum_epochs"]
             and continuation.get("total_epochs") == run.get("effective_maximum_epochs")
             == lineage.get("total_epochs"),
             "continuation parent checkpoint/report or effective budget cost lineage changed")
    parent_cost = _training_attempt_cost(parent, parent_run)
    parent_attempt = parent_run.get("attempt", 0)
    attempt = run.get("attempt")
    _require(type(attempt) is int and attempt > parent_attempt,
             "continuation attempt must follow the parent committed attempt")
    incremental = _cost_value(run["elapsed_seconds"], "training_elapsed_seconds")
    for previous in range(parent_attempt + 1, attempt):
        name = f"training/attempts/attempt_{previous:03d}.json"
        record = bound_record(cell, rows, name)
        _require(record.get("attempt") == previous and record.get("training_seed") == seed
                 and record.get("binding") == binding and record.get("continuation") == continuation
                 and record.get("trainer_sources") == sources,
                 "prior continuation training attempt identity changed")
        elapsed = _cost_value(record["elapsed_seconds"], "prior_training_elapsed_seconds")
        _require(elapsed is not None, "prior training elapsed time is unavailable")
        incremental = None if incremental is None else incremental + elapsed
    parent_elapsed = parent_cost["training_elapsed_seconds"]
    total = None if parent_elapsed is None or incremental is None else parent_elapsed + incremental
    result = dict(training_elapsed_seconds=total,
                training_attempts=parent_cost["training_attempts"] + attempt - parent_attempt,
                parent_training_elapsed_seconds=parent_elapsed, incremental_training_elapsed_seconds=incremental,
                parent_training_attempts=parent_cost["training_attempts"],
                incremental_training_attempts=attempt - parent_attempt,
                training_cost_scope="sum of bound parent and incremental continuation attempt wall times, each once; time after the last interrupted report is unmeasured")
    if prior:
        result.update(launcher_staging_seconds=None, launcher_archive_seconds=None,
                      full_pipeline_elapsed_seconds=None, full_pipeline_cost_measured=False)
    return result


def _training_attempt_cost(cell, run):
    """Charge full selected-parent training and each refit attempt exactly once."""
    if _is_prior_refit(run):
        parent, parent_run, _ = _prior_parent_evidence(cell, run)
        if run.get("continuation") is not None:
            return _continuation_training_cost(cell, run)
        inherited = _training_attempt_cost(parent, parent_run)
        incremental = _recorded_training_attempt_cost(cell, run)
        parent_elapsed, own_elapsed = inherited["training_elapsed_seconds"], incremental["training_elapsed_seconds"]
        return dict(training_elapsed_seconds=None if parent_elapsed is None or own_elapsed is None else parent_elapsed + own_elapsed,
            training_attempts=inherited["training_attempts"] + incremental["training_attempts"],
            parent_training_elapsed_seconds=parent_elapsed, incremental_training_elapsed_seconds=own_elapsed,
            parent_training_attempts=inherited["training_attempts"], incremental_training_attempts=incremental["training_attempts"],
            parent_training_hardware=json.dumps(parent_run.get("hardware"), sort_keys=True),
            launcher_staging_seconds=None, launcher_archive_seconds=None, full_pipeline_elapsed_seconds=None,
            full_pipeline_cost_measured=False,
            training_cost_scope="complete selected-parent training, including its budget ancestry, plus each refit attempt once; launcher staging/archive and time after the last interrupted report are unmeasured")
    if run.get("continuation") is not None:
        return _continuation_training_cost(cell, run)
    return _recorded_training_attempt_cost(cell, run)


def _recorded_training_attempt_cost(cell, run):
    """Sum attempts belonging to this objective, without adding inherited training."""
    attempt = run.get("attempt", 0)
    _require(type(attempt) is int and attempt >= 0, "nonnegative training attempt identity required")
    total = _cost_value(run["elapsed_seconds"], "training_elapsed_seconds")
    if attempt:
        delivery = _read_json(cell / "delivery.json")
        rows = {row["path"]: row for row in delivery["files"]}
        for previous in range(attempt):
            name = f"training/attempts/attempt_{previous:03d}.json"
            _require(name in rows, "prior training attempt cost is unavailable: " + name)
            path, row = cell / name, rows[name]
            _require(path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"],
                     "prior training attempt cost binding changed: " + name)
            record = _read_json(path)
            _require(record["attempt"] == previous and record["training_seed"] == run["training_seed"]
                     and record["binding"] == run["binding"], "prior training attempt identity changed")
            if _is_prior_refit(run):
                _require(record.get("refit_initialization") == run.get("refit_initialization")
                         and record.get("training_objective") == run.get("training_objective")
                         and record.get("trainer_sources") == run.get("trainer_sources"),
                         "prior-refit attempt initialization or objective changed")
            elapsed = _cost_value(record["elapsed_seconds"], "prior_training_elapsed_seconds")
            _require(elapsed is not None, "prior training elapsed time is unavailable")
            total = None if total is None else total + elapsed
    return dict(training_elapsed_seconds=total, training_attempts=attempt + 1,
                training_cost_scope="sum of recorded attempt wall times; time after the last interrupted report is unmeasured")


def _delivered_population_analysis(root):
    """Read accepted aggregate tables and bound training metadata, not samples."""
    delivery = _read_json(root / "delivery.json")
    _require(delivery["status"] == "PASS" and delivery["exit_code"] == 0
             and delivery["mode"] == "complete" and delivery["training_seeds"] == list(SEEDS)
             and delivery["scientific_claim_qualified"] is False
             and delivery["calibration_or_test_accessed"] is False,
             "complete development-only population analysis delivery required")
    manifest = {row["path"]: row for row in delivery["files"]}
    _require(len(manifest) == len(delivery["files"]), "duplicate analysis delivery paths")
    names = ("analysis/analysis.json", "analysis/forecast_all_seeds.csv", "analysis/costs_by_origin.csv")
    for name in names:
        bound = manifest[name]
        _require((root / name).stat().st_size == bound["bytes"]
                 and sha256_file(root / name) == bound["sha256"], "population aggregate delivery changed: " + name)
    report = _read_json(root / names[0])
    _require(report["status"] == "REPLAYED_DEVELOPMENT" and report["models"] == ["population"]
             and report["seeds"] == list(SEEDS) and report["score_rows"] == 432
             and report["cost_origin_rows"] == 144 and report["independent_blocks"] == 4
             and report["scientific_claim_qualified"] is False
             and report["source_fidelity_qualified"] is False and report["formal_test_accessed"] is False
             and report["all_unit_metrics_independently_replayed"] is True
             and report["all_observed_field_counts_checked_against_source"] is True,
             "complete independently replayed population development aggregate required")
    for name in names[1:]:
        _require(report["artifacts"][Path(name).name] == manifest[name]["sha256"],
                 "population analysis artifact binding changed")
    config = report["configuration"]
    _require(config["formal_evaluation"] is False and config["joint_samples"] == 16
             and config["origins_ms"] == [1000., 3000., 5000.]
             and config["horizons_ms"] == [20., 80., 500.], "registered development forecast protocol required")
    representation = report["common_training_budget"]["representation"]
    _require(representation in ("fixed", "recurrent"), "identified population representation required")
    campaign = Path(report["inputs"]["forecasts"])
    training_rows, prior_flags, prior_protocols = [], [], []
    bindings = report["model_seed_bindings"]
    _require(set(bindings) == {f"population_{seed}" for seed in SEEDS}
             and set(report["campaign_deliveries"]) == {str(seed) for seed in SEEDS},
             "three complete population training bindings required")
    for seed in SEEDS:
        cell = campaign / f"seed_{seed}"
        _require(sha256_file(cell / "delivery.json") == report["campaign_deliveries"][str(seed)],
                 "analyzed population campaign delivery changed")
        bound = bindings[f"population_{seed}"]
        _require(sha256_file(cell / "training/run.json") == bound["training_report_sha256"],
                 "analyzed population training metadata changed")
        run = _read_json(cell / "training/run.json")
        _require(run["status"] == "FITTED_DEVELOPMENT" and run["model_family"] == "population"
                 and run["representation"] == representation
                 and run["binding"]["config"] == bound["training_config"]
                 and run["binding"]["seed"] == seed
                 and run["best_checkpoint_sha256"] == bound["checkpoint_sha256"]
                 and bound["training_config"]["representation"] == representation,
                 "analyzed representation/training binding differs")
        budget = bound["executed_training_budget"]
        _require(run["completed_epochs"] == budget["completed_epochs"]
                 and run["best_epoch"] == budget["best_epoch"] and run["stop_reason"] == budget["stop_reason"],
                 "analyzed executed training budget differs")
        prior = _is_prior_refit(run)
        prior_flags.append(prior)
        extra = dict(training_objective=run.get("training_objective", "native_event_time_mc_negative_elbo"),
                     selection_metric=run.get("selection_metric", "validation_negative_elbo"))
        if prior:
            path = cell / "training/best.pt"
            _require(sha256_file(path) == bound["checkpoint_sha256"], "analyzed prior checkpoint changed")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            lineage = _prior_parent_evidence(cell, run, checkpoint)[2]
            _require(bound.get("prior_refit_lineage") == lineage
                     and bound.get("training_objective") == run["training_objective"]
                     and bound.get("selection_metric") == run["selection_metric"], "analyzed prior objective lineage differs")
            extra.update({key: lineage[key] for key in ("parent_selected_epoch", "parent_completed_epochs",
                "parent_effective_maximum_epochs", "parent_checkpoint_sha256", "parent_report_sha256")})
            extra.update(observation_parameters_refitted=False, posterior_targets_fixed=True,
                         **_prior_sampling_budget(run))
            if run["training_objective"] == PRIOR_OBSERVATION_OBJECTIVE:
                _require(report["code_sha256"].get(PRIOR_OBSERVATION_MODULE)
                         == run["trainer_sources"][PRIOR_OBSERVATION_MODULE],
                         "analyzed native observation objective source differs")
                extra.update(PRIOR_OBSERVATION_DECLARATIONS)
            prior_protocols.append(_prior_config_protocol(run["binding"]["config"]))
            del checkpoint
        parameters = run["parameter_count"]
        _require(type(parameters) is int and parameters > 0, "measured model parameter count required")
        method = PRIOR_METHODS[run["training_objective"]] if prior else "population_" + representation
        training_rows.append(dict(seed=seed, representation=representation, method=method, **extra,
            parameter_count=parameters, completed_epochs=run["completed_epochs"], best_epoch=run["best_epoch"],
            stop_reason=run["stop_reason"], effective_maximum_epochs=budget["effective_maximum_epochs"],
            **_training_attempt_cost(cell, run),
            training_peak_process_rss_bytes=_cost_value(run["peak_process_rss_bytes"], "training_peak_process_rss_bytes"),
            training_hardware=json.dumps(run["hardware"], sort_keys=True)))
    _require(len(set(prior_flags)) == 1
             and prior_flags[0] == (report["common_training_budget"].get("training_objective") in PRIOR_METHODS)
             and (not prior_flags[0] or all(row["training_objective"]
                    == report["common_training_budget"].get("training_objective") for row in training_rows)),
             "population aggregate objective differs across seeds or its common protocol")
    if prior_protocols:
        _require(all(value == prior_protocols[0] for value in prior_protocols),
                 "population aggregate mixes prior training protocols")
        if prior_protocols[0]["schema"] == PRIOR_MIXTURE_SCHEMA:
            _require(all(report["common_training_budget"].get(key) == value
                         for key, value in prior_protocols[0].items())
                     and report["common_training_budget"].get("selection_metric") == training_rows[0]["selection_metric"]
                     and report["common_training_budget"].get("mixture_scope") == PRIOR_MIXTURE_SCOPE,
                     "population aggregate joint predictive-mixture protocol differs")
        elif prior_protocols[0]["schema"] == PRIOR_OBSERVATION_SCHEMA:
            budget = report["common_training_budget"]
            _require(all(budget.get(key) == value for key, value in prior_protocols[0].items())
                     and budget.get("selection_metric") == training_rows[0]["selection_metric"]
                     and "posterior_target_draws" not in budget
                     and all((budget.get(key) is value) if isinstance(value, bool) else budget.get(key) == value
                             for key, value in PRIOR_OBSERVATION_DECLARATIONS.items()),
                     "population aggregate native observation protocol differs")
    with (root / names[1]).open(newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
                 "unique population aggregate score columns required")
        raw = list(reader)
    field_metrics, other_metrics = _metric_names(config)
    metric_names = field_metrics + other_metrics
    _require(set(reader.fieldnames) == set(IDENTITY_COLUMNS) | set(metric_names),
             "population aggregate score schema differs")
    expected = {(seed, block, f"train_block_{block:03d}_familiar_{template}", f"familiar_{template}", origin, horizon)
                for seed, block, template, origin, horizon in itertools.product(
                    SEEDS, BLOCKS, range(4), config["origins_ms"], config["horizons_ms"])}
    rows, seen = [], set()
    for row in raw:
        _require(row["method"] == "population", "one population model per aggregate required")
        value = dict(row)
        for name in ("seed", "block"):
            value[name] = _integer(row[name], name)
        for name in ("origin_ms", "horizon_ms"):
            value[name] = _finite(row[name], name)
        key = tuple(value[k] for k in ("seed", "block", "case_id", "template_id", "origin_ms", "horizon_ms"))
        _require(key in expected and key not in seen, "duplicate or foreign population comparison condition")
        seen.add(key)
        bins = value["horizon_ms"] / 5.
        for name in metric_names:
            unavailable = ((name.startswith("variogram_temporal_lag_") and int(name.rsplit("_", 1)[1]) >= bins)
                or (name.startswith("refractory_count_violation_") and int(name.rsplit("_", 1)[1][:-3]) > bins))
            _require((row[name] == "") == (unavailable or (name == "peak_gpu_allocated_bytes" and row[name] == "")),
                     "population eligible metric availability differs: " + name)
            value[name] = None if row[name] == "" else _finite(row[name], name)
            if value[name] is not None:
                if name in COST_METRICS or name in UNIT_METRICS[1:]:
                    _require(value[name] >= 0, "nonnegative population cost/rate/interval required")
                if name == "unit_total_coverage_90" or name.startswith("refractory_count_violation_"):
                    _require(0 <= value[name] <= 1, "population coverage/support outside [0,1]")
                if name == "peak_gpu_allocated_bytes":
                    _require(value[name] == int(value[name]), "integer measured GPU allocation required")
        rows.append(value)
    _require(seen == expected and len(rows) == 432, "complete population comparison conditions required")
    with (root / names[2]).open(newline="") as stream:
        costs = list(csv.DictReader(stream))
    joined = _origin_cost_fidelity(rows, costs, config, CANDIDATE_COST_SCOPE)
    for row in [*rows, *joined]:
        row["method"] = method
    receipt = dict(root=str(root.resolve()), method=method,
        training_objective=training_rows[0]["training_objective"], selection_metric=training_rows[0]["selection_metric"],
        delivery_sha256=sha256_file(root / "delivery.json"),
        analysis_sha256=manifest[names[0]]["sha256"], representation=representation,
        training_bindings=bindings, campaign_deliveries=report["campaign_deliveries"],
        score_artifact_sha256=manifest[names[1]]["sha256"], cost_artifact_sha256=manifest[names[2]]["sha256"])
    return report, rows, joined, training_rows, receipt


def compare_saved_population_baselines(root, baselines, output):
    """Compare complete saved aggregates without repeating generation."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    report, rows, costs, training, receipt = _delivered_population_analysis(Path(root))
    for row in rows:
        row['method'] = 'population'
    differences, bindings = _baseline_comparisons(
        [Path(path) for path in baselines], report, rows, report['configuration'], cost_fidelity_rows=costs)
    _require(bool(bindings), 'at least one saved baseline aggregate required')
    if receipt["training_objective"] in PRIOR_METHODS:
        for row in differences:
            row.update(candidate=receipt["method"], candidate_training_objective=receipt["training_objective"])
    output.mkdir(parents=True)
    tables = {'forecast_paired_block_differences.csv': differences,
              'cost_fidelity_by_origin.csv': costs, 'population_training_resources.csv': training}
    for name, values in tables.items():
        columns = list(dict.fromkeys(key for row in values for key in row))
        _write_rows(output/name, [{key:row.get(key) for key in columns} for row in values])
    result = dict(status='DESCRIBED_SAVED_BASELINE_COMPARISON', population_binding=receipt,
        baseline_bindings=bindings, training_seeds=list(SEEDS), independent_blocks=4, connectivity_instances=1,
        configuration=report['configuration'], source_binding=report['source_binding'],
        paired_rows=len(differences), cost_fidelity_origins=len(costs), new_forecasts=0, new_training_updates=0,
        scientific_claim_qualified=False, formal_test_accessed=False, end_to_end_cost_measured=False,
        matched_fidelity_cost_comparison=False, speedup_qualified=False,
        interpretation='descriptive development comparison at matched source/output protocol; model conditioning, capacities and training budgets may differ',
        artifacts={name:sha256_file(output/name) for name in tables})
    (output/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def _population_budget_pair(items):
    """Require one original recurrent run and its unchanged numerical continuation."""
    _require(all(item[0]["common_training_budget"]["representation"] == "recurrent" for item in items),
             "budget comparison requires two recurrent population analyses")
    ceilings = []
    for item in items:
        budgets = [item[0]["model_seed_bindings"][f"population_{seed}"]["executed_training_budget"]["effective_maximum_epochs"]
                   for seed in SEEDS]
        _require(all(type(value) is int and value > 0 for value in budgets) and len(set(budgets)) == 1,
                 "budget comparison requires one executed ceiling per three-seed analysis")
        _require(item[0]["common_training_budget"]["maximum_epochs"] == budgets[0],
                 "budget comparison common and executed ceilings differ")
        ceilings.append(budgets[0])
    _require(ceilings[0] != ceilings[1], "budget comparison requires distinct executed ceilings")
    parent, child = [item for _, item in sorted(zip(ceilings, items), key=lambda pair: pair[0])]
    for name in ("source_binding", "configuration"):
        _require(parent[0][name] == child[0][name], "population budget source/code/measurement differs: " + name)
    parent_code, child_code = parent[0]["code_sha256"], child[0]["code_sha256"]
    code_changes = {}
    if parent_code != child_code:
        _require(isinstance(parent_code, dict) and isinstance(child_code, dict)
                 and set(parent_code) == set(child_code), "population budget aggregate code key set differs")
        code_changes = {name: dict(parent=digest, current=child_code[name])
                        for name, digest in parent_code.items() if digest != child_code[name]}
    _require({k: v for k, v in parent[0]["common_training_budget"].items() if k != "maximum_epochs"}
             == {k: v for k, v in child[0]["common_training_budget"].items() if k != "maximum_epochs"},
             "population budget base training settings differ")
    parent_campaign, child_campaign = [Path(item[0]["inputs"]["forecasts"]) for item in (parent, child)]
    for seed in SEEDS:
        bound_parent, bound_child = [item[0]["model_seed_bindings"][f"population_{seed}"] for item in (parent, child)]
        _require(bound_parent["training_config"] == bound_child["training_config"],
                 "population budget full base training configuration differs")
        parent_cell, child_cell = [campaign / f"seed_{seed}" for campaign in (parent_campaign, child_campaign)]
        parent_run = _read_json(parent_cell / "training/run.json")
        child_run = _read_json(child_cell / "training/run.json")
        lineage = child_run.get("continuation")
        _require(parent_run.get("continuation") is None and isinstance(lineage, dict)
                 and lineage.get("parent_artifact_sha256", {}).get("run.json") == bound_parent["training_report_sha256"]
                 and lineage.get("parent_checkpoint_sha256") == parent_run.get("latest_checkpoint_sha256")
                 and lineage.get("parent_completed_epochs") == parent_run["completed_epochs"]
                 and lineage.get("parent_effective_maximum_epochs") == bound_parent["executed_training_budget"]["effective_maximum_epochs"]
                 and lineage.get("total_epochs") == bound_child["executed_training_budget"]["effective_maximum_epochs"]
                 and lineage.get("parent_binding") == parent_run["binding"]
                 and lineage.get("parent_source_hashes") == parent_run.get("trainer_sources")
                 and lineage.get("current_source_hashes") == child_run.get("trainer_sources"),
                 "population budget continuation parent checkpoint/report/source lineage differs")
        if _is_prior_refit(parent_run) or _is_prior_refit(child_run):
            _require(_is_prior_refit(parent_run) and _is_prior_refit(child_run)
                     and parent_run.get("refit_initialization") == child_run.get("refit_initialization"),
                     "prior budget comparison must preserve its original refit objective and initialization")
            migration = validate_prior_budget_binding(parent_run["binding"], child_run["binding"],
                parent_run["trainer_sources"], child_run["trainer_sources"])
            _require(lineage.get("source_migration") == migration, "prior budget source migration differs")
            # Every aggregate difference must match the validated trainer migration
            # for this seed. Numerical and evaluation changes have no such entry.
            _require(all(migration.get(name) == change for name, change in code_changes.items()),
                     "population budget aggregate code differs from registered prior source migration")
        else:
            _require(not code_changes, "population budget non-prior aggregate code differs")
            _require(parent_run["binding"] == child_run["binding"]
                     and parent_run.get("trainer_sources") == child_run.get("trainer_sources")
                     and lineage.get("source_migration") == {},
                     "population budget continuation source lineage differs")
        proof = _read_json(parent_cell / "delivery.json")
        rows = {row["path"]: row for row in proof["files"]}
        _require(rows.get("training/latest.pt", {}).get("sha256") == lineage["parent_checkpoint_sha256"],
                 "population budget parent latest checkpoint is not delivery-bound")
        child_proof = _read_json(child_cell / "delivery.json")
        _require(Path(child_proof["optimization_budget_continuation"]["parent"]).resolve() == parent_cell.resolve(),
                 "population budget continuation identifies another parent cell")
    return parent, child


def compare_population_budget_analyses(roots, output):
    """Pair an original recurrent budget and its registered continuation."""
    return compare_population_analyses(roots, output, budget_comparison=True)


def _population_prior_objective_pair(items):
    """Contrast registered prior objectives from the same original selected state."""
    indexed = {item[0]["common_training_budget"].get("training_objective"): item for item in items}
    _require(len(items) == len(indexed) == 2 and set(indexed) <= set(PRIOR_METHODS),
             "two different registered prior objectives required")
    left, right = [indexed[objective] for objective in PRIOR_METHODS if objective in indexed]
    _require(all(item[0]["common_training_budget"]["representation"] == "recurrent" for item in items),
             "prior-objective comparison requires recurrent population models")
    for name in ("source_binding", "configuration"):
        _require(left[0][name] == right[0][name], "prior-objective source or measurement differs: " + name)
    codes = [item[0]["code_sha256"] for item in (left, right)]
    modules = [PRIOR_OBJECTIVE_MODULES[item[0]["common_training_budget"]["training_objective"]]
               for item in (left, right)]
    objective_modules = set().union(*PRIOR_OBJECTIVE_MODULES.values())
    for name in set(codes[0]) | set(codes[1]):
        present = [name in code for code in codes]
        if name in objective_modules:
            _require(all(not included or name in allowed for included, allowed in zip(present, modules)),
                     "unregistered objective source for its profile: " + name)
            if all(present):
                _require(codes[0][name] == codes[1][name], "shared prior objective source differs: " + name)
        else:
            _require(all(present), "prior-objective code key set differs: " + name)
            # These are comparison/trainer control sources, not model or score
            # equations. Their exact differences remain in the comparison report.
            if name not in ("population_forecast_analysis.py", "poisson_trajectory_training.py"):
                _require(codes[0][name] == codes[1][name],
                         "prior-objective numerical or measurement source differs: " + name)
    for item in (left, right):
        if item[0]["common_training_budget"]["training_objective"] != PRIOR_OBSERVATION_OBJECTIVE:
            continue
        digest = item[0]["code_sha256"].get(PRIOR_OBSERVATION_MODULE)
        for seed in SEEDS:
            run = _read_json(Path(item[0]["inputs"]["forecasts"]) / f"seed_{seed}" / "training/run.json")
            _require(isinstance(digest, str) and run.get("training_objective") == PRIOR_OBSERVATION_OBJECTIVE
                     and run["binding"]["config"] == item[0]["model_seed_bindings"][f"population_{seed}"]["training_config"]
                     and run.get("trainer_sources", {}).get(PRIOR_OBSERVATION_MODULE) == digest,
                     "native observation aggregate source differs from a training seed")
    differing_config = {"schema", "observation_objective", "prior_selection", *PRIOR_MIXTURE_CONFIG_KEYS}
    differing_budget = differing_config | {"training_objective", "selection_metric", "mixture_scope", "interpretation",
                                           *PRIOR_OBSERVATION_DECLARATIONS}
    _require({key: value for key, value in left[0]["common_training_budget"].items() if key not in differing_budget}
             == {key: value for key, value in right[0]["common_training_budget"].items() if key not in differing_budget},
             "prior-objective registered training budget differs")
    parent_keys = ("parent_cell", "parent_delivery_sha256", "parent_checkpoint_sha256", "parent_report_sha256",
                   "parent_training_config", "parent_completed_epochs", "parent_selected_epoch",
                   "parent_effective_maximum_epochs", "trainable_parameter_names")
    for seed in SEEDS:
        a, b = [item[0]["model_seed_bindings"][f"population_{seed}"] for item in (left, right)]
        _require({key: value for key, value in a["training_config"].items() if key not in differing_config}
                 == {key: value for key, value in b["training_config"].items() if key not in differing_config},
                 "prior-objective non-objective training configuration differs")
        _require(all(a["executed_training_budget"].get(key) == b["executed_training_budget"].get(key)
                     for key in ("base_maximum_epochs", "effective_maximum_epochs")),
                 "prior-objective registered epoch ceilings differ")
        first, second = a.get("prior_refit_lineage", {}), b.get("prior_refit_lineage", {})
        _require(all(key in first and key in second and first[key] == second[key] for key in parent_keys),
                 "prior-objective original parent checkpoint/report or frozen-state lineage differs")
    # Each input's _delivered_population_analysis has already compared every
    # non-prior tensor to this authenticated common parent, with exact dtypes.
    return left, right


def _population_prior_pair(items):
    """Match a new prior objective to the exact selected parent that initialized it."""
    children = [item for item in items if item[0]["common_training_budget"].get("training_objective") in PRIOR_METHODS]
    if len(children) == 2:
        return _population_prior_objective_pair(children)
    _require(len(children) == 1, "prior-objective comparison requires one prior refit and one original parent")
    child = children[0]
    parent = next(item for item in items if item is not child)
    _require(all(item[0]["common_training_budget"]["representation"] == "recurrent" for item in items),
             "prior-objective comparison requires recurrent population models")
    for name in ("source_binding", "configuration"):
        _require(parent[0][name] == child[0][name], "prior-objective source or measurement differs: " + name)
    for seed in SEEDS:
        left, right = [item[0]["model_seed_bindings"][f"population_{seed}"] for item in (parent, child)]
        lineage = right.get("prior_refit_lineage", {})
        _require(lineage.get("parent_checkpoint_sha256") == left["checkpoint_sha256"]
                 and lineage.get("parent_report_sha256") == left["training_report_sha256"]
                 and lineage.get("parent_training_config") == left["training_config"]
                 and lineage.get("parent_completed_epochs") == left["executed_training_budget"]["completed_epochs"]
                 and lineage.get("parent_selected_epoch") == left["executed_training_budget"]["best_epoch"]
                 and lineage.get("parent_effective_maximum_epochs") == left["executed_training_budget"]["effective_maximum_epochs"]
                 and Path(lineage.get("parent_cell", "")).resolve()
                     == (Path(parent[0]["inputs"]["forecasts"]) / f"seed_{seed}").resolve()
                 and lineage.get("parent_delivery_sha256") == parent[0]["campaign_deliveries"][str(seed)],
                 "prior-objective comparison selected-parent lineage differs")
    return parent, child


def compare_population_analyses(roots, output, *, budget_comparison=False, prior_objective_comparison=False):
    """Pair delivered results with explicit representation, budget or prior-objective semantics."""
    _require(len(roots) == 2, "exactly two delivered population analyses required")
    _require(not (budget_comparison and prior_objective_comparison), "choose one population comparison mode")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", seeds=list(SEEDS), models=["population_fixed", "population_recurrent"],
        comparison_mode="prior_objective" if prior_objective_comparison else "training_budget" if budget_comparison else "representation",
        independent_blocks=4, scientific_claim_qualified=False, source_fidelity_qualified=False,
        formal_test_accessed=False, state_selection_enabled=False, formal_tolerances_qualified=False,
        uncertainty="descriptive checkpoint-selection blocks; no inferential CI, p-value or q-value",
        replication_unit="initial/background realization block conditional on one fixed connectivity graph",
        inputs=[str(root.resolve()) for root in roots],
        comparison_implementation_sha256=sha256_file(Path(__file__)),
        generation_repeated=False, training_repeated=False, end_to_end_cost_measured=False,
        same_hardware_cost_comparison=False, matched_fidelity_cost_comparison=False, speedup_qualified=False)
    def save():
        (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        items = list(map(_delivered_population_analysis, roots))
        if prior_objective_comparison:
            fixed, recurrent = _population_prior_pair(items)
            two_prior_objectives = fixed[0]["common_training_budget"].get("training_objective") in PRIOR_METHODS
            if two_prior_objectives:
                left_label, right_label = [PRIOR_COMPARISON_LABELS[item[0]["common_training_budget"]["training_objective"]]
                                           for item in (fixed, recurrent)]
                difference_label = right_label + "_minus_" + left_label
                report["models"] = [PRIOR_METHODS[item[0]["common_training_budget"]["training_objective"]]
                                    for item in (fixed, recurrent)]
            else:
                left_label, right_label, difference_label = "parent", "prior_refit", "prior_refit_minus_parent"
                report["models"] = ["population_recurrent_parent",
                                    PRIOR_METHODS[recurrent[0]["common_training_budget"]["training_objective"]]]
            prefix = "prior_objective"
            report["prior_objective_comparison_kind"] = "same_parent_prior_objectives" if two_prior_objectives else "selected_parent_and_refit"
            for item, method in zip((fixed, recurrent), report["models"]):
                for row in [*item[1], *item[2], *item[3]]:
                    row["method"] = method
            report["implementation_differences"] = {
                name: {left_label: fixed[0]["code_sha256"].get(name), right_label: recurrent[0]["code_sha256"].get(name)}
                for name in sorted(set(fixed[0]["code_sha256"]) | set(recurrent[0]["code_sha256"]))
                if fixed[0]["code_sha256"].get(name) != recurrent[0]["code_sha256"].get(name)}
        elif budget_comparison:
            fixed, recurrent = _population_budget_pair(items)
            left_label, right_label, difference_label = "parent", "continuation", "continuation_minus_parent"
            prefix = "budget"
            objective = fixed[0]["common_training_budget"].get("training_objective")
            prior_budget = objective in PRIOR_METHODS
            model_prefix = PRIOR_METHODS[objective] + "_budget" if prior_budget else "population_recurrent_budget"
            report["models"] = [model_prefix + str(item[0]["common_training_budget"]["maximum_epochs"])
                                for item in (fixed, recurrent)]
            report["prior_refit_budget_comparison"] = prior_budget
            for item, method in zip((fixed, recurrent), report["models"]):
                for row in [*item[1], *item[2], *item[3]]:
                    row["method"] = method
        else:
            loaded = {item[0]["common_training_budget"]["representation"]: item for item in items}
            _require(set(loaded) == {"fixed", "recurrent"}, "one fixed and one recurrent analysis required")
            fixed, recurrent = loaded["fixed"], loaded["recurrent"]
            left_label, right_label, difference_label = "fixed", "recurrent", "recurrent_minus_fixed"
            prefix = "representation"
            for name in ("source_binding", "code_sha256"):
                _require(fixed[0][name] == recurrent[0][name], "population representation " + name + " differs")
            for name in ("configuration", "common_training_budget"):
                exclude = "scope" if name == "configuration" else "representation"
                _require({k: v for k, v in fixed[0][name].items() if k != exclude}
                         == {k: v for k, v in recurrent[0][name].items() if k != exclude},
                         "population representation measurement or training budget differs")
            for seed in SEEDS:
                configs = [item[0]["model_seed_bindings"][f"population_{seed}"]["training_config"] for item in (fixed, recurrent)]
                _require({k: v for k, v in configs[0].items() if k != "representation"}
                         == {k: v for k, v in configs[1].items() if k != "representation"},
                         "population representation full training configuration differs")
        config = fixed[0]["configuration"]
        identity = ("seed", "block", "case_id", "template_id", "origin_ms", "horizon_ms")
        indexes = [{tuple(row[k] for k in identity): row for row in item[1]} for item in (fixed, recurrent)]
        _require(set(indexes[0]) == set(indexes[1]), "population paired conditions differ")
        metrics = _metric_names(config)[0] + list(UNIT_METRICS)
        metrics += [f"refractory_count_violation_{w}bin" for w in config["refractory_support_windows_bins"]]
        differences, blocks = [], []
        for key in sorted(indexes[0]):
            left, right = indexes[0][key], indexes[1][key]
            for name in metrics:
                _require((left[name] is None) == (right[name] is None), "paired population metric availability differs")
                if left[name] is not None:
                    differences.append(dict(zip(identity, key), metric=name,
                        **{left_label: left[name], right_label: right[name],
                           difference_label: _finite(right[name]-left[name], "paired population difference")},
                        interpretation="development_descriptive"))
        for block, horizon, name in itertools.product(BLOCKS, config["horizons_ms"], metrics):
            members = [r for r in differences if r["block"] == block and r["horizon_ms"] == horizon and r["metric"] == name]
            if not members:
                continue
            _require(len(members) == 36, "complete three-seed block comparison required")
            blocks.append(dict(block=block, horizon_ms=horizon, metric=name,
                **{left_label + "_mean": _mean([r[left_label] for r in members]),
                   right_label + "_mean": _mean([r[right_label] for r in members]),
                   difference_label: _mean([r[difference_label] for r in members])}, training_seeds=3,
                template_origins_per_seed=12, interpretation="development_descriptive"))
        costs = [row for item in (fixed, recurrent) for row in item[2]]
        cost_indexes = [{_origin_key(r)[:-1]: r for r in item[2]} for item in (fixed, recurrent)]
        _require(set(cost_indexes[0]) == set(cost_indexes[1]), "population cost conditions differ")
        cost_differences = []
        for key in sorted(cost_indexes[0]):
            left, right = [index[key] for index in cost_indexes]
            for name in COST_COMPONENTS:
                cost_differences.append(dict(seed=key[0], case_id=key[1], origin_ms=key[2], horizon_ms=500.,
                    metric=name, **{left_label: left[name], right_label: right[name],
                    difference_label: None if left[name] is None or right[name] is None else _finite(right[name]-left[name], "paired cost difference")},
                    interpretation="descriptive measured component; no end-to-end or same-hardware cost claim"))
        outputs = {"forecast_all_seeds.csv": fixed[1] + recurrent[1],
            prefix + "_paired_conditions.csv": differences, prefix + "_paired_blocks.csv": blocks,
            "cost_fidelity_by_origin.csv": costs, prefix + "_cost_differences.csv": cost_differences,
            prefix + "_training.csv": fixed[3] + recurrent[3]}
        for name, rows in outputs.items():
            columns = list(dict.fromkeys(key for row in rows for key in row))
            _write_rows(output / name, [{key: row.get(key) for key in columns} for row in rows])
        report.update(status="DESCRIBED_REPLAYED_DEVELOPMENT", configuration=config,
            common_training_budget={k: v for k, v in fixed[0]["common_training_budget"].items() if k != "representation"},
            source_binding=fixed[0]["source_binding"], population_analysis_bindings=[fixed[4], recurrent[4]],
            score_rows=864, paired_forecast_conditions=432, cost_fidelity_origin_rows=len(costs),
            comparison_scope=("same recurrent model and base configuration; registered training ceiling differs; descriptive selection-block results"
                if budget_comparison else "only encoder representation differs; capacities and actual training costs are reported; selection tolerances remain unqualified"),
            artifacts={name: sha256_file(output / name) for name in outputs})
        if prior_objective_comparison:
            report.update(comparison_scope=(
                "two registered prior objectives initialized from the same original selected parent; identical frozen encoder/readout/source buffers, cohort, forecast protocol and registered epoch budget; particle budgets and measured training costs differ; descriptive selection-block results"
                if two_prior_objectives else
                "matched native source and forecast protocol; new prior objective initialized from the selected parent; fixed encoder/readout and source buffers verified; implementation differences reported; descriptive selection-block results"),
                training_protocols={left_label: fixed[0]["common_training_budget"], right_label: recurrent[0]["common_training_budget"]},
                inherited_training_cost_included=True, launcher_staging_cost_measured=False,
                training_compute_matched=False)
            if any(item[0]["common_training_budget"].get("training_objective") == PRIOR_OBSERVATION_OBJECTIVE
                   for item in (fixed, recurrent)):
                report.update(training_label_information_match_qualified=False,
                    native_event_label_disclosure="native event-path counts and eligible ticks are training labels; equal training-label information with count-only methods is not qualified")
            if two_prior_objectives:
                report["training_particle_budgets"] = {
                    label: {key: item[3][0][key] for key in
                            ("prediction_particles", "posterior_target_draws", "posterior_target_integration", "mixture_scope")}
                    for label, item in ((left_label, fixed), (right_label, recurrent))}
            report.pop("common_training_budget")
        elif budget_comparison:
            report["executed_training_ceilings"] = [item[0]["common_training_budget"]["maximum_epochs"] for item in (fixed, recurrent)]
            report["common_training_budget"] = {k: v for k, v in fixed[0]["common_training_budget"].items() if k != "maximum_epochs"}
        save()
        return report
    except BaseException as error:
        report.update(status="FAILED", error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
        save()
        raise


def analyze(root: Path, acquisition: Path, fields: Path, output: Path, *, population_campaign=False,
            baseline_analyses=()):
    _require(population_campaign or not baseline_analyses, "baseline aggregate comparison requires population campaign")
    models = ("population",) if population_campaign else MODELS
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", seeds=list(SEEDS), models=list(models), independent_blocks=4,
                  scientific_claim_qualified=False, source_fidelity_qualified=False, formal_test_accessed=False,
                  uncertainty="descriptive epoch-selection blocks; no inferential CI, p-value or q-value",
                  replication_unit="initial/background realization block conditional on one fixed connectivity graph",
                  inputs=dict(forecasts=str(root.resolve()), acquisition=str(acquisition.resolve()), fields=str(fields.resolve())),
                  all_unit_metrics_independently_replayed=False,
                  unit_metric_scope="reported finite sampled-count metrics; only retained unit IDs checked here; full raw unit samples retained at one case/origin per model/seed",
                  costs="inference, full-unit generation and field projection reported separately; each cost covers the full500ms forecast and is repeated on horizon rows; no matched mechanistic-source speedup or separately attested evaluation-hardware comparison")
    def save():
        (output / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text())
        _require(len(config["origins_ms"]) == len(set(config["origins_ms"])) == 3
                 and len(config["horizons_ms"]) == len(set(config["horizons_ms"])) == 3
                 and config["joint_samples"] == 16 and config["formal_evaluation"] is False
                 and config["likelihood_comparison"] == "omitted_observation_spaces_differ", "frozen common development forecast config required")
        for model, seed in itertools.product(models, SEEDS):
            folder = (_campaign_cell(root, seed) / "forecasts" if population_campaign
                      else root / model / f"seed_{seed}_forecast")
            evaluation = _read_json(folder / "evaluation.json")
            _require(evaluation["status"] == "EVALUATED_DEVELOPMENT", "all six completed evaluations required")
        source = CanonicalFields(acquisition, fields)
        if population_campaign:
            source.acquisition, source.fields = acquisition, fields
            case = source.validation[0]["case_id"]
            source.population_native = NativeRenewalEpisode(
                acquisition / case, fields / case, expected_source=source.source_identity,
                expected_unit_ids=source.unit_ids, expected_unit_field_index=source.unit_field_index)
        files = (Path(__file__).name, "trajectory_forecast_analysis.py", "population_trajectory_evaluation.py", "trajectory_sample_metrics.py",
                 "poisson_state_space.py", "renewal_state_space.py", "refractory_observation.py", "native_renewal_data.py",
                 "poisson_trajectory_evaluation.py", "poisson_trajectory_training.py")
        if population_campaign:
            files += ("population_state.py", "native_count_windows.py")
        code = {name: sha256_file(Path(__file__).with_name(name)) for name in files}
        runs = [(model, seed, *_load_run(root, model, seed, source, config, code,
                                       campaign_layout=population_campaign))
                for model, seed in itertools.product(models, SEEDS)]
        budget_keys = ("latent_size", "encoder_size", "bin_ms", "learning_rate", "maximum_epochs", "early_stopping_patience",
                       "initialization_pseudocount", "gradient_norm_limit", "initial_transition_diagonal")
        if population_campaign:
            budget_keys = tuple(key for key in budget_keys
                                if key not in ("latent_size", "initial_transition_diagonal"))
            budget_keys += ("latent_channels", "representation", "unit_chunk_size")
        budget = {key: runs[0][-1]["training_config"][key] for key in budget_keys}
        _require(all({key: run[-1]["training_config"][key] for key in budget_keys} == budget for run in runs), "matched base training settings differ")
        ceilings = {run[-1]['executed_training_budget']['effective_maximum_epochs'] for run in runs}
        _require(len(ceilings) == 1, 'matched effective training ceilings differ')
        budget['base_maximum_epochs'] = budget['maximum_epochs']
        budget['maximum_epochs'] = ceilings.pop()
        budget['interpretation'] = 'matched registered epoch ceilings and patience; actual epochs and wall time may differ'
        prior_runs = [run for run in runs if run[-1]["training_config"].get("schema") in PRIOR_PROFILES]
        if prior_runs:
            _require(len(prior_runs) == len(runs), "mixed population training objectives in one aggregate")
            protocol = _prior_config_protocol(runs[0][-1]["training_config"])
            _require(all(_prior_config_protocol(run[-1]["training_config"]) == protocol for run in runs),
                     "matched prior-refit training protocols differ")
            objective, selection, _ = PRIOR_PROFILES[protocol["schema"]]
            budget.update(protocol, training_objective=objective, selection_metric=selection,
                interpretation="fixed-encoder/readout prior objective; epoch counts describe refit updates; complete parent training cost is additional")
            if protocol["schema"] == PRIOR_MIXTURE_SCHEMA:
                budget["mixture_scope"] = PRIOR_MIXTURE_SCOPE
            elif protocol["schema"] == PRIOR_OBSERVATION_SCHEMA:
                budget.update(PRIOR_OBSERVATION_DECLARATIONS)
                report.update(training_label_information_match_qualified=False,
                    native_event_label_disclosure="native event-path counts and eligible ticks are training labels; equal training-label information with count-only methods is not qualified")
        report.update(configuration=config, configuration_sha256=sha256_file(CONFIG_PATH), source_binding=source.binding,
                      code_sha256=code, model_seed_bindings={f"{model}_{seed}": receipt for model, seed, _, _, receipt in runs},
                      common_training_budget=budget, field_scale_recomputed_from_fitting_episodes=56)
        measured_runs = sum("measured_evaluation_costs" in run[-1] for run in runs)
        if measured_runs:
            report.update(measured_evaluation_runs=measured_runs,
                          costs="source-bound evaluator origin components and evaluation hardware when recorded; complete episode preparation recorded once at its first origin; legacy missing costs remain null; imports, training, launcher staging/archive and final evaluator-report write excluded; no end-to-end, same-hardware or speedup qualification")
        pairs = pair_sets(source.field_ids)
        field_metrics, other_metrics = _metric_names(config)
        metrics = field_metrics + other_metrics
        all_rows, cov_rows, cost_rows = [], [], []
        cov_arrays = {}
        comparisons, maximum_delta = 0, 0.
        longest = int(max(config["horizons_ms"]) / source.bin_ms)
        for model, seed, folder, rows, receipt in runs:
            ensembles = {}
            grouped = defaultdict(list)
            for row in rows:
                name = row["ensemble_artifact"]
                if name not in ensembles:
                    with np.load(folder / name, allow_pickle=False) as data:
                        sampled, observed = data["sampled_field_counts"], data["observed_field_counts"]
                        _counts(sampled, (config["joint_samples"], longest, len(source.field_ids)), name)
                        _counts(observed, (longest, len(source.field_ids)), name + " truth")
                        boundary = int(row["origin_ms"] / source.bin_ms)
                        _require(np.array_equal(observed, source.truth[row["case_id"]][boundary:boundary + longest]), "forecast observed field truth differs from canonical source")
                        for key in ("latent_states", "initial_latent_states"):
                            _require(np.isfinite(data[key]).all(), "nonfinite retained latent state")
                        denominator = source.unit_count * source.bin_ms / 1000
                        ensembles[name] = ((sampled / denominator - source.mean) / source.sd,
                                           (observed / denominator - source.mean) / source.sd)
                    cost_rows.append(dict(seed=seed, block=row["block"], case_id=row["case_id"], template_id=row["template_id"],
                                          origin_ms=row["origin_ms"], method=model, generated_horizon_ms=max(config["horizons_ms"]),
                                          **{key: row[key] for key in COST_METRICS},
                                          **_measured_origin_columns(receipt, row["case_id"], row["origin_ms"])))
                sample, truth = ensembles[name]
                horizon = int(row["horizon_ms"] / source.bin_ms)
                replayed = replay_scores(sample[:, :horizon], truth[:horizon], pairs, config["temporal_lags_bins"])
                for key, value in replayed.items():
                    delta = abs(value - row[key])
                    _require(np.isfinite(value) and np.isclose(value, row[key], rtol=1e-10, atol=1e-12), "independent field score replay mismatch: " + key)
                    maximum_delta = max(maximum_delta, delta)
                    comparisons += 1
                siblings = [r for r in rows if r["ensemble_artifact"] == name]
                _require(all(all(r[key] == row[key] for key in COST_METRICS) for r in siblings), "full-horizon cost differs across reused horizon rows")
                grouped[(row["template_id"], row["origin_ms"], row["horizon_ms"])].append(row)
            for key, members in sorted(grouped.items()):
                members.sort(key=lambda row: row["block"])
                _require([row["block"] for row in members] == list(BLOCKS), "each covariance group requires exactly four source blocks")
                horizon = int(key[2] / source.bin_ms)
                sample = np.stack([ensembles[r["ensemble_artifact"]][0][:, :horizon] for r in members])
                truth = np.stack([ensembles[r["ensemble_artifact"]][1][:horizon] for r in members])
                requests = [(name, pair[:, 0], pair[:, 1], 0) for name, pair in pairs.items()]
                requests += [(f"temporal_lag_{lag}", np.arange(len(source.field_ids)), np.arange(len(source.field_ids)), lag)
                             for lag in config["temporal_lags_bins"] if lag < horizon]
                for name, left, right, lag in requests:
                    observed, predicted, within, between = covariance_components(sample, truth, left, right, lag)
                    payload = np.stack((observed, predicted, within, between))
                    _require(np.isfinite(payload).all(), "nonfinite covariance diagnostic")
                    array_key = f"g{len(cov_rows)}"
                    cov_arrays[array_key] = payload
                    cov_rows.append(dict(seed=seed, method=model, template_id=key[0], origin_ms=key[1], horizon_ms=key[2],
                                         statistic=name, independent_blocks=4, pairs=len(left), phases=horizon-lag,
                                         observed_covariance_mean=float(observed.mean()), predicted_covariance_mean=float(predicted.mean()),
                                         covariance_mse=_finite(((predicted-observed)**2).mean(), "covariance_mse"), covariance_mae=_finite(np.abs(predicted-observed).mean(), "covariance_mae"),
                                         array_key=array_key))
            all_rows.extend(rows)
        block_groups = defaultdict(list)
        for row in all_rows:
            block_groups[(row["seed"], row["block"], row["method"], row["horizon_ms"])].append(row)
        block_rows = []
        for key, members in sorted(block_groups.items()):
            _require(len(members) == 12, "all four templates and three origins per block required")
            block_rows.append(dict(zip(("seed", "block", "method", "horizon_ms"), key), episodes_origins=12,
                                   **{metric: _mean([row[metric] for row in members]) for metric in metrics}))
        summary = []
        for model, horizon, metric in itertools.product(models, config["horizons_ms"], metrics):
            values = [_mean([r[metric] for r in block_rows if r["method"] == model and r["horizon_ms"] == horizon and r["seed"] == seed]) for seed in SEEDS]
            if values[0] is None:
                _require(all(value is None for value in values), "unmatched missing summary metric")
                continue
            summary.append(dict(method=model, horizon_ms=horizon, metric=metric, mean=_mean(values),
                                minimum_seed_mean=min(values), maximum_seed_mean=max(values), independent_blocks=4,
                                training_seeds=3, interpretation="development_descriptive"))
        paired = []
        for block, horizon, metric in itertools.product(BLOCKS if len(models) > 1 else (), config["horizons_ms"], metrics):
            values = {model: _mean([r[metric] for r in block_rows if r["method"] == model and r["horizon_ms"] == horizon and r["block"] == block]) for model in MODELS}
            if any(value is None for value in values.values()):
                _require(all(value is None for value in values.values()), "unmatched missing paired metric")
                continue
            paired.append(dict(block=block, horizon_ms=horizon, metric=metric, poisson_mean=values["poisson"],
                               renewal_mean=values["renewal"], renewal_minus_poisson=values["renewal"]-values["poisson"], training_seeds=3))
        cov_groups = defaultdict(list)
        for row in cov_rows:
            cov_groups[(row["method"], row["horizon_ms"], row["statistic"])].append(row)
        cov_summary = []
        for key, members in sorted(cov_groups.items()):
            cov_summary.append(dict(zip(("method", "horizon_ms", "statistic"), key), training_seeds=3, independent_blocks=4,
                                    matched_template_origins=12,
                                    **{metric: _mean([r[metric] for r in members]) for metric in
                                       ("observed_covariance_mean", "predicted_covariance_mean", "covariance_mse", "covariance_mae")}))
        outputs = {"forecast_all_seeds.csv": all_rows, "forecast_by_block.csv": block_rows,
                   "forecast_summary.csv": summary,
                   "covariance_diagnostic.csv": cov_rows, "covariance_summary.csv": cov_summary,
                   "costs_by_origin.csv": cost_rows}
        if paired:
            outputs["forecast_paired_block_differences.csv"] = paired
        cost_fidelity_rows = _origin_cost_fidelity(all_rows, cost_rows, config,
                                                  report["costs"] if measured_runs else CANDIDATE_COST_SCOPE)
        if baseline_analyses:
            paired, baseline_receipts = _baseline_comparisons(
                baseline_analyses, report, all_rows, config, cost_fidelity_rows=cost_fidelity_rows)
            outputs["forecast_paired_block_differences.csv"] = paired
            report.update(baseline_aggregate_bindings=baseline_receipts,
                          comparison_scope="matched source, complete conditions and outputs; model capacities and training objectives differ; no cost speedup or convergence claim")
        # A baseline may retain additional measured component columns. Expand
        # every row to the common schema without replacing unknowns by zero.
        columns = list(dict.fromkeys(name for row in cost_fidelity_rows for name in row))
        outputs["cost_fidelity_by_origin.csv"] = [
            {name: row.get(name) for name in columns} for row in cost_fidelity_rows]
        report.update(cost_fidelity_origin_rows=len(cost_fidelity_rows),
                      cost_fidelity_scope="descriptive measured components at the complete500ms output contract; model output and setup scopes differ; no end-to-end, same-hardware, matched-fidelity or speedup claim",
                      graph_setup_scope="raw setup repeated across sixteen cases per seed/origin graph lifetime; no setup charge or total inferred",
                      end_to_end_cost_measured=False, same_hardware_cost_comparison=False,
                      matched_fidelity_cost_comparison=False, speedup_qualified=False)
        for name, rows in outputs.items():
            _write_rows(output / name, rows)
        np.savez_compressed(output / "covariance_arrays.npz", **cov_arrays)
        report.update(status="REPLAYED_DEVELOPMENT", score_rows=len(all_rows), field_scores_recomputed=comparisons,
                      maximum_absolute_replay_difference=maximum_delta, covariance_rows=len(cov_rows),
                      cost_origin_rows=len(cost_rows), all_observed_field_counts_checked_against_source=True,
                      covariance_scope="across four realization blocks at fixed template, origin and phase; total predictive covariance diagnostic",
                      covariance_array_axis=["observed", "predicted_total", "mean_within", "between_sample_means"],
                      covariance_interpretation="descriptive noisy estimates; not conditional covariance identified from one continuation",
                      artifacts={name: sha256_file(output / name) for name in [*outputs, "covariance_arrays.npz"]})
        if population_campaign:
            report.update(all_unit_metrics_independently_replayed=True,
                          unit_metric_scope="complete 48-origin independent replay per training seed, bound to each campaign delivery",
                          campaign_deliveries={str(seed): sha256_file(root / f"seed_{seed}/delivery.json") for seed in SEEDS},
                          paired_model_comparison_available=bool(baseline_analyses))
        save()
        return report
    except BaseException as error:
        report.update(status="FAILED", error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
        save()
        raise


def _covariance_decomposition(payload, draws, rate_ratio):
    """Describe saved block covariance; mean-rate scaling is sensitivity only."""
    values = np.asarray(payload, dtype=np.float64)
    _require(values.ndim == 3 and values.shape[0] == 4 and all(values.shape[1:])
             and np.isfinite(values).all(), "four finite covariance component arrays required")
    _require(type(draws) is int and draws >= 2, "at least two covariance draws required")
    _require(np.isfinite(rate_ratio) and rate_ratio >= 0, "finite nonnegative mean-rate ratio required")
    observed, total, within, between = values
    corrected_between = between - within / draws
    reconstructed = corrected_between + within
    residual = np.abs(total - reconstructed)
    scale = max(float(np.abs(total).max()), float(np.abs(within).max()),
                float(np.abs(between).max()), np.finfo(np.float64).tiny)
    _require(np.allclose(total, reconstructed, rtol=1e-10, atol=1e-12),
             "saved total covariance differs from its declared components")
    multiplier = rate_ratio ** 2
    _require(np.isfinite(multiplier), "finite mean-rate covariance multiplier required")
    return dict(observed_covariance_mean=float(observed.mean()),
        predicted_covariance_mean=float(total.mean()), within_covariance_mean=float(within.mean()),
        raw_between_covariance_mean=float(between.mean()),
        corrected_between_covariance_mean=float(corrected_between.mean()),
        scaled_predicted_covariance_mean=float((total * multiplier).mean()),
        scaled_within_covariance_mean=float((within * multiplier).mean()),
        scaled_corrected_between_covariance_mean=float((corrected_between * multiplier).mean()),
        covariance_mae=float(np.abs(total - observed).mean()),
        scaled_covariance_mae=float(np.abs(total * multiplier - observed).mean()),
        maximum_reconstruction_residual=float(residual.max()),
        scale_normalized_reconstruction_residual=float(residual.max() / scale))


def describe_saved_covariance(root, output):
    """Decompose delivered covariance without model or source execution."""
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    report, scores, _, _, receipt = _delivered_population_analysis(root)
    manifest = {row["path"]: row for row in _read_json(root / "delivery.json")["files"]}
    names = ("covariance_diagnostic.csv", "covariance_arrays.npz")
    for name in names:
        path, bound = root / "analysis" / name, manifest["analysis/" + name]
        _require(path.stat().st_size == bound["bytes"] and sha256_file(path) == bound["sha256"]
                 and report["artifacts"][name] == bound["sha256"], "saved covariance binding changed: " + name)
    _require(report["covariance_array_axis"] ==
             ["observed", "predicted_total", "mean_within", "between_sample_means"],
             "saved covariance axis semantics differ")
    with (root / "analysis" / names[0]).open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    config = report["configuration"]
    expected = {(seed, f"familiar_{template}", origin, horizon, statistic)
        for seed, template, origin, horizon in itertools.product(SEEDS, range(4), config["origins_ms"], config["horizons_ms"])
        for statistic in ["spatial", "cross_population"] +
        [f"temporal_lag_{lag}" for lag in config["temporal_lags_bins"] if lag < horizon / 5.]}
    identities, array_keys = set(), set()
    for row in raw:
        key = (_integer(row["seed"], "seed"), row["template_id"], _finite(row["origin_ms"], "origin"),
               _finite(row["horizon_ms"], "horizon"), row["statistic"])
        _require(key in expected and key not in identities and row["method"] == "population"
                 and _integer(row["independent_blocks"], "blocks") == len(BLOCKS)
                 and row["array_key"] not in array_keys, "duplicate or foreign covariance condition")
        identities.add(key); array_keys.add(row["array_key"])
    _require(identities == expected and report["covariance_rows"] == len(raw),
             "complete saved covariance conditions required")
    ratios = {}
    for seed, horizon in itertools.product(SEEDS, config["horizons_ms"]):
        cells = [row for row in scores if row["seed"] == seed and row["horizon_ms"] == horizon]
        _require(len(cells) == 48, "complete matched rate conditions required")
        predicted = _mean([row["sampled_expected_spikes_per_unit"] for row in cells])
        observed = _mean([row["observed_spikes_per_unit"] for row in cells])
        _require(predicted > 0, "positive predicted mean required for rate sensitivity")
        ratios[(seed, horizon)] = observed / predicted
    rows = []
    with np.load(root / "analysis" / names[1], allow_pickle=False) as arrays:
        _require(set(arrays.files) == array_keys, "saved covariance array identities differ")
        for row in raw:
            seed, horizon = _integer(row["seed"], "seed"), _finite(row["horizon_ms"], "horizon")
            payload = arrays[row["array_key"]]
            _require(payload.shape == (4, _integer(row["phases"], "phases"), _integer(row["pairs"], "pairs")),
                     "saved covariance shape differs")
            result = _covariance_decomposition(payload, config["joint_samples"], ratios[(seed, horizon)])
            for name in ("observed_covariance_mean", "predicted_covariance_mean", "covariance_mae"):
                _require(np.isclose(result[name], _finite(row[name], name), rtol=1e-10, atol=1e-12),
                         "saved covariance table disagrees with arrays: " + name)
            rows.append(dict(seed=seed, template_id=row["template_id"], origin_ms=float(row["origin_ms"]),
                horizon_ms=horizon, statistic=row["statistic"], rate_ratio=ratios[(seed, horizon)], **result))
    summary = []
    numeric = [name for name in rows[0] if name not in ("seed", "template_id", "origin_ms", "horizon_ms", "statistic")]
    for horizon, statistic in sorted({(row["horizon_ms"], row["statistic"]) for row in rows}):
        cells = [row for row in rows if (row["horizon_ms"], row["statistic"]) == (horizon, statistic)]
        _require(len(cells) == 36, "all three seeds and twelve matched template/origins required")
        summary.append(dict(horizon_ms=horizon, statistic=statistic, training_seeds=3,
            independent_blocks=4, matched_template_origins=12,
            **{name: max(row[name] for row in cells) if "residual" in name else _mean([row[name] for row in cells])
               for name in numeric}))
    output.mkdir(parents=True)
    _write_rows(output / "covariance_components.csv", rows)
    _write_rows(output / "covariance_component_summary.csv", summary)
    result = dict(status="DESCRIBED_SAVED_COVARIANCE_DECOMPOSITION", input=receipt, score_rows=len(scores),
        covariance_rows=len(rows), input_artifacts={name: manifest["analysis/" + name]["sha256"] for name in names},
        code_sha256=sha256_file(Path(__file__)), representation=receipt["representation"],
        units="covariance of fitting-standardized per-unit Hz fields",
        decomposition="corrected_between=between_sample_means-mean_within/draws; total=corrected_between+mean_within",
        within_interpretation="all variation among generated draws, including latent and observation variation",
        covariance_scope=report["covariance_scope"],
        rate_scaling="per-seed/horizon observed mean count divided by predicted mean; covariance multiplied by ratio squared",
        rate_scaling_interpretation="outcome-informed uniform-rate sensitivity; not a corrected count generator or formal score",
        scientific_claim_qualified=False, confidence_interval_or_significance=False,
        training_or_generation_repeated=False, calibration_or_test_accessed=False,
        artifacts={name: sha256_file(output / name) for name in
                   ("covariance_components.csv", "covariance_component_summary.csv")})
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _latent_projection(latents, loading, unit_field_index, process_variance):
    """Project saved latent draws into log hazards without new simulation."""
    z, weights = np.asarray(latents, dtype=np.float64), np.asarray(loading, dtype=np.float64)
    index, process = np.asarray(unit_field_index), np.asarray(process_variance, dtype=np.float64)
    _require(z.ndim == 5 and z.shape[1] >= 2 and z.shape[2] >= 1
             and weights.ndim == 2 and weights.shape[1] == z.shape[-1]
             and index.shape == (len(weights),) and index.dtype.kind in "iu"
             and np.all(index >= 0) and np.all(index < z.shape[-2])
             and process.shape == z.shape[-2:] and np.all(process > 0)
             and all(np.isfinite(x).all() for x in (z, weights, process)),
             "finite aligned latent, unit-loading and process-variance arrays required")
    counts = np.bincount(index, minlength=z.shape[-2])
    means = np.zeros(z.shape[-2:])
    grams = np.zeros((z.shape[-2], z.shape[-1], z.shape[-1]))
    for field in range(z.shape[-2]):
        selected = weights[index == field]
        if len(selected):
            means[field] = selected.mean(0)
            grams[field] = selected.T @ selected / len(selected)
    centered = z - z.mean(1, keepdims=True)
    cov = np.einsum("bdtfk,bdtfl->btfkl", centered, centered) / (z.shape[1] - 1)
    projected = np.einsum("btfkl,fkl->btf", cov, grams)
    field_hazards = np.einsum("bdtfk,fk->bdtf", z, means)
    field_variance = field_hazards.var(axis=1, ddof=1)
    latent_variance = np.diagonal(cov, axis1=-2, axis2=-1).mean(-1)
    return dict(field_mean_log_hazards=field_hazards,
        latent_variance=latent_variance,
        unit_log_hazard_variance=projected,
        field_mean_log_hazard_variance=field_variance,
        unit_counts=counts, mean_loading=means,
        projected_process_unit_variance=np.einsum("fk,fkk->f", process, grams),
        projected_process_field_mean_variance=(process * means ** 2).sum(-1))


def describe_saved_latents(root, output):
    """Measure latent-to-log-hazard variation from all delivered trajectories."""
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    report, scores, _, _, receipt = _delivered_population_analysis(root)
    campaign = Path(report["inputs"]["forecasts"])
    pairs, source_map, rows, parameters, optimization, bindings = None, None, [], [], [], {}
    for seed in SEEDS:
        cell = campaign / f"seed_{seed}"
        bound = report["model_seed_bindings"][f"population_{seed}"]
        checkpoint_path = cell / "training/best.pt"
        _require(sha256_file(checkpoint_path) == bound["checkpoint_sha256"], "latent projection checkpoint changed")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _require(checkpoint["binding"]["seed"] == seed
                 and checkpoint["binding"]["config"] == bound["training_config"], "latent checkpoint training identity differs")
        state, config = checkpoint["model"], checkpoint["model_configuration"]
        run = _read_json(cell / "training/run.json")
        _require(config == run["model_configuration"], "checkpoint latent observation map differs from bound training metadata")
        weights = state["unit_loading"].detach().numpy().astype(np.float64)
        index = state["unit_field_index"].detach().numpy()
        process = state["process_log_variance"].exp().detach().numpy().astype(np.float64)
        fields = np.asarray(config["field_names"])
        _require(np.array_equal(index, np.asarray(config["unit_field_index"]))
                 and len(weights) == len(config["unit_ids"]) == 100000 and len(fields) == 128
                 and config["bin_seconds"] == .005,
                 "source-defined100000-unit128-field5ms latent map required")
        current_pairs = pair_sets(fields)
        current_map = {name: config[name] for name in ("unit_ids", "unit_field_index", "field_names")}
        if source_map is not None:
            _require(current_map == source_map, "latent observation maps differ across seeds")
        source_map = current_map
        if pairs is not None:
            _require(all(np.array_equal(pairs[key], current_pairs[key]) for key in pairs),
                     "latent field pair maps differ across seeds")
        pairs = current_pairs
        decay = state["decay_logit"].sigmoid().detach().numpy().astype(np.float64)
        mix = state["field_mix_logits"].softmax(-1).detach().numpy().astype(np.float64)
        _require(np.isfinite(decay).all() and np.isfinite(mix).all(), "finite transition parameters required")
        parameters.append(dict(seed=seed, unit_loading_rms=float(np.sqrt((weights ** 2).mean())),
            process_variance_mean=float(process.mean()), decay_mean=float(decay.mean()),
            field_mix_self_weight_mean=float(np.diag(mix).mean()),
            field_mix_maximum_weight_mean=float(mix.max(-1).mean())))
        history = checkpoint["history"]
        _require([row["epoch"] for row in history] == list(range(checkpoint["next_epoch"])),
                 "complete committed optimization history required")
        for entry in history:
            for partition_name in ("training", "validation"):
                terms = entry[partition_name]
                _require(all(np.isfinite(terms[name]) for name in ("loss", "sampled_observation_nll", "initial_kl", "transition_kl")),
                         "finite committed loss terms required")
                optimization.append(dict(seed=seed, epoch=entry["epoch"], partition=partition_name, **terms))
        bindings[str(seed)] = dict(checkpoint_sha256=bound["checkpoint_sha256"], forecast_artifacts={})
        selected = [row for row in scores if row["seed"] == seed and row["horizon_ms"] == 500.]
        _require(len(selected) == 48, "complete latent forecast origins required")
        groups = defaultdict(list)
        for row in selected:
            groups[(row["template_id"], row["origin_ms"])].append(row)
        for (template, origin), members in sorted(groups.items()):
            members.sort(key=lambda row: row["block"])
            _require([row["block"] for row in members] == list(BLOCKS), "four matched latent source blocks required")
            saved = []
            for row in members:
                name = row["ensemble_artifact"]
                _require(Path(name).name == name, "plain latent artifact filename required")
                artifact = cell / "forecasts" / name
                _require(sha256_file(artifact) == bound["artifacts"][name], "saved latent trajectory changed")
                bindings[str(seed)]["forecast_artifacts"][name] = bound["artifacts"][name]
                with np.load(artifact, allow_pickle=False) as data:
                    z, initial = data["latent_states"], data["initial_latent_states"]
                    _require(z.shape == (16, 100, 128, config["latent_channels"])
                             and initial.shape == (16, 1, 128, config["latent_channels"]),
                             "aligned complete16-draw500ms latent trajectory required")
                    saved.append(np.concatenate((initial, z), axis=1))
            projection = _latent_projection(np.stack(saved), weights, index, process)
            hazards = projection["field_mean_log_hazards"]
            population_weights = projection["unit_counts"] / len(weights)
            for phase in range(101):
                for statistic, pair in pairs.items():
                    left, right = pair[:, 0], pair[:, 1]
                    centered = hazards[:, :, phase] - hazards[:, :, phase].mean(1, keepdims=True)
                    within = (centered[:, :, left] * centered[:, :, right]).sum(1).mean(0) / 15
                    means = hazards[:, :, phase].mean(1)
                    between = ((means[:, left] - means[:, left].mean(0)) *
                               (means[:, right] - means[:, right].mean(0))).sum(0) / 3
                    corrected_between = between - within / 16
                    denominator = np.sqrt(projection["field_mean_log_hazard_variance"][:, phase, left]
                                          * projection["field_mean_log_hazard_variance"][:, phase, right])
                    valid = denominator > 0
                    per_block_covariance = (centered[:, :, left] * centered[:, :, right]).sum(1) / 15
                    correlations = np.zeros_like(denominator)
                    np.divide(per_block_covariance, denominator, out=correlations, where=valid)
                    rows.append(dict(seed=seed, template_id=template, origin_ms=origin, forecast_ms=phase * 5,
                        statistic=statistic, latent_variance_mean=float(projection["latent_variance"][:, phase].mean()),
                        unit_log_hazard_variance_mean=float((projection["unit_log_hazard_variance"][:, phase] * population_weights).sum(-1).mean()),
                        field_mean_log_hazard_variance_mean=float(projection["field_mean_log_hazard_variance"][:, phase].mean()),
                        within_field_mean_log_hazard_covariance=float(within.mean()),
                        corrected_between_field_mean_log_hazard_covariance=float(corrected_between.mean()),
                        total_field_mean_log_hazard_covariance=float((corrected_between + within).mean()),
                        within_field_mean_log_hazard_correlation=None if not valid.any() else float(correlations[valid].mean()),
                        correlation_eligible_block_pairs=int(valid.sum()),
                        process_unit_log_hazard_innovation_variance=float((projection["projected_process_unit_variance"] * population_weights).sum()),
                        process_field_mean_log_hazard_innovation_variance=float(projection["projected_process_field_mean_variance"].mean())))
    summary = []
    values = [name for name in rows[0] if name not in ("seed", "template_id", "origin_ms", "forecast_ms", "statistic")]
    for phase, statistic in sorted({(row["forecast_ms"], row["statistic"]) for row in rows}):
        group = [row for row in rows if (row["forecast_ms"], row["statistic"]) == (phase, statistic)]
        _require(len(group) == 36, "all three seeds and twelve latent template/origins required")
        summary.append(dict(forecast_ms=phase, statistic=statistic, training_seeds=3,
            matched_template_origins=12, independent_blocks=4,
            **{name: None if all(row[name] is None for row in group) else
               _mean([row[name] for row in group if row[name] is not None]) for name in values}))
    output.mkdir(parents=True)
    tables = {"latent_projection_by_origin.csv": rows, "latent_projection_summary.csv": summary,
              "checkpoint_parameters.csv": parameters, "committed_loss_history.csv": optimization}
    for name, values in tables.items():
        _write_rows(output / name, values)
    result = dict(status="DESCRIBED_SAVED_LATENT_HAZARD_VARIANCE", input=receipt, trajectory_bindings=bindings,
        code_sha256=sha256_file(Path(__file__)), forecast_origins=144, phases_per_origin=101,
        rows=len(rows), units=100000, fields=128, draws=16, training_seeds=list(SEEDS),
        output_units="latent variance; natural-log hazard variance/covariance; dimensionless correlation",
        projection="each unit loading projects its own field latent covariance; field log hazard is the mean of unit log hazards",
        conditioning="four source initial/background blocks at matched template, origin and forecast phase; one fixed graph",
        process_noise="source model uses independent Gaussian field/channel innovations; projected innovation has no interfield covariance",
        loss_normalization="NLL sums all units; initial/transition KL sum all field channels; each divided by batch*time; all loss weights1",
        interpretation="descriptive latent-to-log-hazard diagnostic; nonlinear hazards and refractory sampling preclude direct count-covariance equivalence",
        variance_decomposition="between is corrected by subtracting mean within/16; signed estimates retained",
        scientific_claim_qualified=False, confidence_interval_or_significance=False,
        training_or_generation_repeated=False, calibration_or_test_accessed=False,
        artifacts={name: sha256_file(output / name) for name in tables})
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("forecasts", "acquisition", "fields"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population-analysis", type=Path, action="append", default=[],
                        help="Pair exactly two completed fixed/recurrent population analyses without generation")
    parser.add_argument("--budget-comparison", action="store_true",
                        help="Compare exactly two delivered recurrent budgets with authenticated parent lineage")
    parser.add_argument("--prior-objective-comparison", action="store_true",
                        help="Compare a prior refit with its selected parent, or two registered prior objectives from one parent; include inherited training costs")
    parser.add_argument("--population-campaign", action="store_true",
                        help="Read completed seed_<seed>/{training,forecasts,replay.json} campaign cells")
    parser.add_argument("--baseline-analysis", type=Path, action="append", default=[],
                        help="Reuse a delivered baseline analysis; repeat for distinct model families")
    parser.add_argument("--covariance-analysis", type=Path,
                        help="Describe saved covariance components and uniform mean-rate sensitivity")
    parser.add_argument("--latent-analysis", type=Path,
                        help="Project every saved latent trajectory through its bound checkpoint unit loadings")
    args = parser.parse_args()
    if args.budget_comparison and args.prior_objective_comparison:
        parser.error("choose budget-comparison or prior-objective-comparison")
    if (args.budget_comparison or args.prior_objective_comparison) and (len(args.population_analysis) != 2 or args.baseline_analysis
            or args.latent_analysis or args.covariance_analysis or args.population_campaign
            or any(getattr(args, name) is not None for name in ("forecasts", "acquisition", "fields"))):
        parser.error("budget or prior-objective comparison requires exactly two population analyses and output")
    if args.latent_analysis:
        if (args.covariance_analysis or args.population_analysis or args.baseline_analysis or args.population_campaign
                or any(getattr(args, name) is not None for name in ("forecasts", "acquisition", "fields"))):
            parser.error("saved latent projection requires only latent-analysis and output")
        report = describe_saved_latents(args.latent_analysis, args.output)
    elif args.covariance_analysis:
        if (args.population_analysis or args.baseline_analysis or args.population_campaign
                or any(getattr(args, name) is not None for name in ("forecasts", "acquisition", "fields"))):
            parser.error("saved covariance decomposition requires only covariance-analysis and output")
        report = describe_saved_covariance(args.covariance_analysis, args.output)
    elif args.population_analysis:
        if (args.population_campaign or any(getattr(args, name) is not None for name in ("forecasts", "acquisition", "fields"))
                or not (len(args.population_analysis) == 2 and not args.baseline_analysis
                        or len(args.population_analysis) == 1 and args.baseline_analysis)):
            parser.error("saved comparison requires two population analyses or one population analysis with baselines")
        report = (compare_saved_population_baselines(args.population_analysis[0], args.baseline_analysis, args.output)
                  if args.baseline_analysis else compare_population_budget_analyses(args.population_analysis, args.output)
                  if args.budget_comparison else compare_population_analyses(args.population_analysis, args.output,
                      prior_objective_comparison=args.prior_objective_comparison))
    else:
        if any(getattr(args, name) is None for name in ("forecasts", "acquisition", "fields")):
            parser.error("forecasts, acquisition and fields required for source replay")
        report = analyze(args.forecasts, args.acquisition, args.fields, args.output,
                         population_campaign=args.population_campaign, baseline_analyses=args.baseline_analysis)
    print(json.dumps({key: value for key, value in report.items() if key not in ("model_seed_bindings", "source_binding", "artifacts")}))


if __name__ == "__main__":
    main()
