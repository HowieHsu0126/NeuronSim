"""Fixed-posterior data and initialization for native population prior refits.

The complete accepted native cohort supplies causal posterior targets. Only
transition/process parameters change; the shared trainer owns optimization,
checkpointing and resource recovery. Validation measures the declared latent
or native event-path objective. Full-unit generation remains a separate required evaluation.
"""
from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
import time

import numpy as np
import torch

from .population_prior_objective import PRIOR_PARAMETER_NAMES, prior_rollout_loss
from .population_predictive_mixture import prior_predictive_mixture_loss
from .population_observation_prior import prior_observation_mixture_loss
from .provenance import sha256_file

SCHEMA = "neural_population_prior_refit_v1"
MIXTURE_SCHEMA = "neural_population_prior_predictive_mixture_v1"
OBSERVATION_SCHEMA = "neural_population_native_observation_prior_v1"
SCHEMAS = (SCHEMA, MIXTURE_SCHEMA, OBSERVATION_SCHEMA)
OVERRIDES = dict(schema=SCHEMA, maximum_epochs=80, early_stopping_patience=12,
    training_sequence="complete_episode_causal_posterior_cache",
    observation_objective="fixed_posterior_multistep_prior_kl",
    prior_origins_ms=[1000, 3000, 5000], prior_horizons_ms=[5, 20, 80, 500],
    prior_target_gradient="stopped", prior_selection="fixed_draw_validation_multistep_kl")
MIXTURE_OVERRIDES = dict(OVERRIDES, schema=MIXTURE_SCHEMA,
    observation_objective="fixed_posterior_joint_predictive_mixture_kl",
    prior_selection="fixed_draw_validation_joint_predictive_mixture_kl",
    prediction_particles=8, posterior_target_draws=4)

OBSERVATION_OVERRIDES = dict(OVERRIDES, schema=OBSERVATION_SCHEMA,
    observation_objective="fixed_readout_native_event_path_mixture_nll",
    prior_selection="fixed_draw_validation_native_event_path_mixture_nll",
    prediction_particles=8)


def uses_observation(config):
    return config.get("schema") == OBSERVATION_SCHEMA


def uses_mixture(config):
    return config.get("schema") == MIXTURE_SCHEMA


def validate_config(config, seed, frozen):
    overrides = OBSERVATION_OVERRIDES if uses_observation(config) else MIXTURE_OVERRIDES if uses_mixture(config) else OVERRIDES
    expected = dict(frozen, **overrides)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("configuration differs from the prospective native prior refit")
    if type(seed) is not int or seed not in (41, 42, 43):
        raise ValueError("registered prior refit seed41/42/43 required")
    if type(config.get("cpu_threads")) is not int or not 1 <= config["cpu_threads"] <= 2:
        raise ValueError("one or two CPU threads required")
    if torch.device(config["device"]).type not in ("cpu", "cuda"):
        raise ValueError("CPU or CUDA prior placement required")
    for key in ("training_variational_seed_base", "validation_seed_base"):
        if type(config.get(key)) is not int or not 0 <= config[key] < 2**63:
            raise ValueError("explicit variational seed base required")
    parents = config.get("parent_checkpoint_sha256", {})
    if set(parents) != {"41", "42", "43"} or any(
            not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in parents.values()):
        raise ValueError("three prospectively bound parent checkpoints required")


def loss_terms(config):
    if uses_observation(config):
        return ("loss", *[f"{prefix}{int(h / config['bin_ms'])}"
            for h in config["prior_horizons_ms"]
            for prefix in ("prior_observation_nll_h", "prior_observation_particle_ess_h")])
    prefix = "prior_mixture_kl_h" if uses_mixture(config) else "prior_kl_h"
    return ("loss", *[f"{prefix}{int(h / config['bin_ms'])}" for h in config["prior_horizons_ms"]])


def initialize_from_parent(model, cohort, config, seed, parent, constructor):
    """Start a new objective from a selected parent, without its optimizer/RNG."""
    parent = Path(parent)
    report = json.loads((parent / "run.json").read_text())
    expected = config["parent_checkpoint_sha256"][str(seed)]
    if (report.get("status") != "FITTED_DEVELOPMENT"
            or report.get("best_checkpoint_sha256") != expected
            or sha256_file(parent / "best.pt") != expected):
        raise ValueError("accepted selected parent checkpoint required for prior refit")
    checkpoint = torch.load(parent / "best.pt", map_location="cpu", weights_only=True)
    binding = checkpoint["binding"]
    if (report["binding"] != binding or binding["seed"] != seed
            or binding["config"].get("model_family") != "population"
            or binding["config"].get("representation") != "recurrent"
            or any(binding.get(key) != value for key, value in cohort.binding.items())):
        raise ValueError("prior parent seed, representation or native cohort differs")
    if (checkpoint["model_configuration"] != report["model_configuration"]
            or checkpoint["model_configuration"] != constructor):
        raise ValueError("parent model constructor and report differ")
    current = model.state_dict()
    if set(current) != set(checkpoint["model"]):
        raise ValueError("parent/current native model state keys differ")
    for name, expected_buffer in model.named_buffers():
        if not torch.equal(checkpoint["model"][name], expected_buffer.detach().cpu()):
            raise ValueError("parent native identity or physical buffer differs: " + name)
    model.load_state_dict(checkpoint["model"], strict=True)
    return dict(mode="selected_parent_prior_refit", parent_training=str(parent.resolve()),
        parent_checkpoint_sha256=expected, parent_report_sha256=sha256_file(parent / "run.json"),
        parent_selected_epoch=checkpoint["next_epoch"] - 1,
        parent_completed_epochs=report["completed_epochs"], parent_binding=binding,
        parent_initialization=checkpoint.get("initialization"),
        trainable_parameter_names=list(PRIOR_PARAMETER_NAMES),
        optimizer_and_random_streams_reinitialized=True)


def reuse_posterior_cache(source, output, initialization, seed, case_ids):
    """Copy only authenticated fixed-posterior caches from a completed KL control."""
    source, output = Path(source), Path(output)
    proof_bytes = (source / "delivery.json").read_bytes()
    proof = json.loads(proof_bytes)
    if (proof.get("status") != "PASS" or proof.get("exit_code") != 0
            or proof.get("training_seed") != seed):
        raise ValueError("accepted same-seed posterior cache delivery required")
    files = {row["path"]: row for row in proof["files"]}
    if len(files) != len(proof["files"]):
        raise ValueError("duplicate posterior cache delivery paths")
    data = (source / "training/run.json").read_bytes()
    row = files["training/run.json"]
    if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("posterior cache source report changed")
    report = json.loads(data)
    if (report.get("status") != "FITTED_DEVELOPMENT"
            or report["binding"]["seed"] != seed
            or report["binding"]["config"].get("schema") != SCHEMA
            or report.get("posterior_targets_fixed") is not True
            or report.get("frozen_parameters_unchanged_in_attempt") is not True
            or not isinstance(report.get("refit_initialization"), dict)
            or {k: v for k, v in report["refit_initialization"].items() if k != "parent_training"}
            != {k: v for k, v in initialization.items() if k != "parent_training"}):
        raise ValueError("posterior cache source or frozen parent differs")
    names = {f"training/posterior_cache/{case}.pt" for case in case_ids}
    if len(names) != len(case_ids) or {name for name in files if name.startswith("training/posterior_cache/")} != names:
        raise ValueError("complete same-case posterior cache required")
    destination = output / "posterior_cache"
    destination.mkdir(exist_ok=False)
    consumed = {}
    for name in sorted(names):
        target = destination / Path(name).name
        shutil.copyfile(source / name, target)
        row = files[name]
        if target.stat().st_size != row["bytes"] or sha256_file(target) != row["sha256"]:
            raise ValueError("posterior cache payload changed: " + name)
        consumed[name] = row["sha256"]
    return dict(source=str(source.resolve()), delivery_sha256=hashlib.sha256(proof_bytes).hexdigest(),
                source_report_sha256=files["training/run.json"]["sha256"],
                parent_checkpoint_sha256=initialization["parent_checkpoint_sha256"],
                reused_episodes=len(consumed), files=consumed,
                optimizer_or_prior_weights_reused=False, posterior_encoding_performed=False)


class PosteriorCache:
    """Cache complete episode encodings under an immutable encoder/readout parent."""
    def __init__(self, model, cohort, config, output, initialization, save_checkpoint):
        self.model, self.cohort, self.config = model, cohort, config
        self.directory = Path(output) / "posterior_cache"
        self.directory.mkdir(exist_ok=True)
        self.save_checkpoint = save_checkpoint
        self.binding = dict(parent_checkpoint_sha256=initialization["parent_checkpoint_sha256"],
            cohort=cohort.binding, model_implementation_sha256=sha256_file(
                Path(__file__).with_name("population_state.py")), bin_ms=config["bin_ms"])
        self.origins = tuple(int(value / config["bin_ms"]) for value in config["prior_origins_ms"])
        self.horizons = tuple(int(value / config["bin_ms"]) for value in config["prior_horizons_ms"])
        self.device = model.unit_loading.device
        self.dtype = model.unit_loading.dtype

    @torch.no_grad()
    def inputs(self, case):
        case_id = case["case_id"]
        path = self.directory / (case_id + ".pt")
        expected = dict(self.binding, case_id=case_id)
        window = None
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["binding"] != expected:
                raise ValueError("cached posterior belongs to a different parent/source")
        else:
            window = self.cohort.window(case)
            counts = torch.as_tensor(window.counts, device=self.device).unsqueeze(0)
            controls = torch.as_tensor(window.stimulus, device=self.device, dtype=self.dtype).unsqueeze(0)
            self.model.eval()
            mean, variance = self.model.encode(counts, controls)
            saved = dict(binding=expected, mean=mean.cpu(), variance=variance.cpu(), controls=controls.cpu())
            self.save_checkpoint(saved, path)
        mean, variance, controls = (saved[key] for key in ("mean", "variance", "controls"))
        expected_shape = (1, self.cohort.duration_ticks // round(self.model.bin_seconds / self.model.tick_seconds),
                          len(self.model.field_names), self.model.latent_channels)
        if (mean.shape != expected_shape or variance.shape != expected_shape
                or controls.shape != (*expected_shape[:2], self.model.input_size)
                or any(value.dtype != self.dtype or not torch.isfinite(value).all()
                       for value in (mean, variance, controls)) or torch.any(variance <= 0)):
            raise ValueError("finite complete-episode posterior cache required")
        latent_inputs = tuple(value.to(self.device) for value in (mean, variance, controls))
        if not uses_observation(self.config):
            return latent_inputs
        if window is None:
            window = self.cohort.window(case)
        if (window.counts.shape != (expected_shape[1], len(self.model.unit_ids))
                or window.eligible_ticks.shape != window.counts.shape
                or not torch.equal(torch.as_tensor(window.stimulus, dtype=self.dtype), controls[0])):
            raise ValueError("complete same-source native event labels and controls required")
        longest = max(self.horizons)
        labels = [torch.as_tensor(np.stack([value[origin:origin + longest] for origin in self.origins]),
                                  device=self.device).unsqueeze(0)
                  for value in (window.counts, window.eligible_ticks)]
        return (*latent_inputs, *labels)

    def loss(self, inputs, initial_generator, dynamics_generator, target_generator=None):
        if uses_observation(self.config):
            if target_generator is not None:
                raise ValueError("native observation likelihood does not sample posterior targets")
            return prior_observation_mixture_loss(self.model, *inputs, self.origins, self.horizons,
                initial_generator, dynamics_generator, prediction_particles=self.config["prediction_particles"])
        if uses_mixture(self.config):
            return prior_predictive_mixture_loss(self.model, *inputs, self.origins, self.horizons,
                initial_generator, dynamics_generator, target_generator,
                prediction_particles=self.config["prediction_particles"],
                posterior_target_draws=self.config["posterior_target_draws"])
        if target_generator is not None:
            raise ValueError("target stream belongs only to the predictive-mixture objective")
        return prior_rollout_loss(self.model, *inputs, self.origins, self.horizons,
                                  initial_generator, dynamics_generator)

    @torch.no_grad()
    def validation(self, seed, seed_function, on_draw=None):
        self.model.eval()
        rows, terms = [], loss_terms(self.config)
        for case in self.cohort.validation:
            inputs = self.inputs(case)
            for draw in range(self.config["validation_draws"]):
                initial_seed = seed_function(self.config["validation_seed_base"], seed,
                    "prior_validation_initial", case["case_id"], draw)
                dynamics_seed = seed_function(self.config["validation_seed_base"], seed,
                    "prior_validation_dynamics", case["case_id"], draw)
                initial = torch.Generator(device=self.device).manual_seed(initial_seed)
                dynamics = torch.Generator(device=self.device).manual_seed(dynamics_seed)
                target, target_metadata = None, {}
                if uses_mixture(self.config):
                    target_seed = seed_function(self.config["validation_seed_base"], seed,
                        "prior_validation_target", case["case_id"], draw)
                    target = torch.Generator(device=self.device).manual_seed(target_seed)
                    target_metadata["prior_target_seed"] = target_seed
                started = time.perf_counter()
                values = {key: float(value.detach()) for key, value in self.loss(inputs, initial, dynamics, target).items()}
                rows.append(values)
                if on_draw is not None:
                    on_draw(dict(case_id=case["case_id"], draw=draw,
                        variational_seed=initial_seed, prior_dynamics_seed=dynamics_seed,
                        **target_metadata, **values, seconds=time.perf_counter() - started))
        return {key: float(np.mean([row[key] for row in rows])) for key in terms}
