"""Epoch-boundary optimization and exact resume for native residual fitting."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from .population_residual_observation import (
    residual_observation_mixture_loss, standardized_ridge_penalty,
)

from .population_residual_mean_objective import residual_mean_objective

NATIVE_OBJECTIVE = "native_event_mixture_nll"
MEAN_OBJECTIVE = "native_event_nll_plus_unbiased_field_mean_mse"

_NAMES = ("correction_weight", "correction_bias")
_REF_ARRAYS = ("x_mean", "x_scale", "y_mean", "y_scale", "coefficients")


def _capture(model):
    return {name: getattr(model, name).detach().cpu().clone() for name in _NAMES}


def _restore(model, values):
    if set(values) != set(_NAMES):
        raise ValueError("incomplete correction checkpoint")
    with torch.no_grad():
        for name in _NAMES:
            parameter, value = getattr(model, name), values[name]
            if parameter.shape != value.shape or parameter.dtype != value.dtype:
                raise ValueError("correction shape or dtype differs")
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite correction checkpoint")
            parameter.copy_(value.to(parameter.device))


def _reference(reference):
    return {"alpha": float(reference.alpha),
        **{name: torch.as_tensor(getattr(reference, name)).detach().cpu().clone()
           for name in _REF_ARRAYS}}


def _atomic_save(path, state):
    descriptor, temporary = tempfile.mkstemp(prefix=".epoch-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _stream_seed(seed, phase, role, case=""):
    # These are random-stream identities, not additional experiment replicates.
    encoded = json.dumps(["native_residual_observation_named_streams_v1",
                          seed, phase, role, case], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63)


def _generator(device, value):
    return torch.Generator(device=device).manual_seed(value)


def _metrics(result):
    values = {name: float(value.detach().cpu()) if torch.is_tensor(value)
              else float(value) for name, value in result.items()}
    if "loss" not in values or not all(math.isfinite(value) for value in values.values()):
        raise ValueError("incomplete or nonfinite native metrics")
    if any(not 1 <= value <= 8 for name, value in values.items() if "ess" in name):
        raise ValueError("particle effective sample size outside one to8")
    return values


def run_epochs(model, reference, batch_loader, fit_cases, validation_cases,
               config, phase, seed, output, binding, *,
               selected_epochs=None, epoch_limit=None, checkpoint_callback=None):
    """epoch_limit bounds additional complete epochs; it does not alter selection."""
    keys = {"learning_rate", "regularization_weight", "max_epochs", "patience",
            "prediction_particles", "unit_chunk_size"}
    if not keys <= set(config) or seed not in (41, 42, 43):
        raise ValueError("registered config keys and training seed required")
    for key, ceiling in (("max_epochs", 80), ("patience", 12),
                         ("prediction_particles", None), ("unit_chunk_size", None)):
        value = config[key]
        if type(value) is not int or value < 1 or (ceiling and value > ceiling):
            raise ValueError("invalid integer setting: " + key)
    for key in ("learning_rate", "regularization_weight"):
        value = config[key]
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid optimizer setting: " + key)
    if config["learning_rate"] == 0:
        raise ValueError("positive learning rate required")
    if phase not in ("qualify", "refit") or not fit_cases:
        raise ValueError("valid phase and fitting cases required")
    if len(set(fit_cases)) != len(fit_cases) or len(set(validation_cases)) != len(validation_cases):
        raise ValueError("duplicate cases")
    if set(fit_cases) & set(validation_cases):
        raise ValueError("fitting and validation cases overlap")
    if phase == "qualify" and (not validation_cases or selected_epochs is not None):
        raise ValueError("qualification requires validation and no selected epoch")
    if phase == "refit" and (validation_cases or type(selected_epochs) is not int
                            or not 0 <= selected_epochs <= config["max_epochs"]):
        raise ValueError("refit requires a selected epoch and no validation")
    if epoch_limit is not None and (type(epoch_limit) is not int or epoch_limit < 0):
        raise ValueError("epoch_limit must be a nonnegative integer")
    identity = json.loads(json.dumps(
        dict(schema="native_affine_epochs_v1", binding=binding, config=config,
             fit_cases=fit_cases, validation_cases=validation_cases,
             phase=phase, seed=seed, selected_epochs=selected_epochs),
        sort_keys=True, allow_nan=False))
    objective_name = config.get("objective", NATIVE_OBJECTIVE)
    if objective_name not in (NATIVE_OBJECTIVE, MEAN_OBJECTIVE):
        raise ValueError("unsupported native residual fitting objective")
    if objective_name == MEAN_OBJECTIVE:
        if (type(config.get("field_mean_weight")) not in (int, float) or config["field_mean_weight"] != 1.0
                or type(config.get("mean_unit_chunk_size")) is not int or config["mean_unit_chunk_size"] < 1):
            raise ValueError("weight-one field mean objective and positive mean chunk required")
    parameters = {name: getattr(model, name) for name in _NAMES}
    enabled = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if enabled != {id(parameter) for parameter in parameters.values()}:
        raise ValueError("only correction weight and bias may be trainable")
    optimizer = torch.optim.Adam(parameters.values(), lr=config["learning_rate"], weight_decay=0)
    device = model.correction_weight.device
    model.eval()
    reference_state = _reference(reference)
    stream_ids = {role: _stream_seed(seed, phase, role)
                  for role in ("order", "train_initial", "train_dynamics")}
    generators = {role: _generator("cpu" if role == "order" else device, value)
                  for role, value in stream_ids.items()}
    target = config["max_epochs"] if phase == "qualify" else selected_epochs
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "epoch_checkpoint.pt"

    def evaluate_case(case, training):
        if training:
            initial, dynamics = generators["train_initial"], generators["train_dynamics"]
        else:
            initial = _generator(device, _stream_seed(seed, phase, "validation_initial", case))
            dynamics = _generator(device, _stream_seed(seed, phase, "validation_dynamics", case))
        objective = residual_mean_objective if objective_name == MEAN_OBJECTIVE else residual_observation_mixture_loss
        extra = {"mean_unit_chunk_size": config["mean_unit_chunk_size"]} if objective_name == MEAN_OBJECTIVE else {}
        return objective(model, **batch_loader(case), initial_generator=initial,
            dynamics_generator=dynamics, prediction_particles=config["prediction_particles"],
            unit_chunk_size=config["unit_chunk_size"], checkpoint_readout=True, **extra)

    def validate():
        with torch.no_grad():
            rows = [dict(case=case, **_metrics(evaluate_case(case, False)))
                    for case in validation_cases]
        return rows, sum(row["loss"] for row in rows) / len(rows)

    def persist():
        state["correction"] = _capture(model)
        state["optimizer"] = optimizer.state_dict()
        state["rng"] = {name: generator.get_state() for name, generator in generators.items()}
        finished = state["epoch"] >= target or (
            phase == "qualify" and state["patience"] >= config["patience"])
        state["status"] = "COMPLETE" if finished else "PARTIAL"
        _atomic_save(path, state)
        if checkpoint_callback is not None:
            checkpoint_callback(path, state["epoch"])
        print(json.dumps(dict(phase=phase, seed=seed, mode=binding.get("mode"),
            completed_epochs=state["epoch"], selected_epoch=state["selected_epoch"],
            best_validation_loss=state["best_loss"], status=state["status"])), flush=True)

    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["identity"] != identity or state["reference"]["alpha"] != reference_state["alpha"]:
            raise ValueError("checkpoint binding, protocol or reference differs")
        if any(not torch.equal(state["reference"][key], reference_state[key]) for key in _REF_ARRAYS):
            raise ValueError("checkpoint reference arrays differ")
        _restore(model, state["correction"])
        optimizer.load_state_dict(state["optimizer"])
        for name, generator in generators.items():
            generator.set_state(state["rng"][name])
    else:
        rows, score = validate() if phase == "qualify" else (None, None)
        state = dict(identity=identity, reference=reference_state, stream_ids=stream_ids,
                     epoch=0, patience=0, selected_epoch=0 if phase == "qualify" else target,
                     best_loss=score, best_correction=_capture(model) if phase == "qualify" else None,
                     records=[dict(epoch=0, training=[], validation=rows, validation_loss=score)])
        persist()

    stop_at = target if epoch_limit is None else min(target, state["epoch"] + epoch_limit)
    while state["status"] != "COMPLETE" and state["epoch"] < stop_at:
        training_rows = []
        order = torch.randperm(len(fit_cases), generator=generators["order"]).tolist()
        for index in order:
            case = fit_cases[index]
            optimizer.zero_grad(set_to_none=True)
            result = evaluate_case(case, True)
            metrics = _metrics(result)
            penalty = standardized_ridge_penalty(model, reference)
            objective = result["loss"] + config["regularization_weight"] * penalty
            if not torch.isfinite(objective).all() or not torch.isfinite(penalty).all():
                raise ValueError("nonfinite training objective")
            objective.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                   for parameter in parameters.values()):
                raise ValueError("missing or nonfinite correction gradients")
            grad_norm = math.sqrt(sum(float(parameter.grad.detach().double().square().sum())
                                      for parameter in parameters.values()))
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in parameters.values()):
                raise ValueError("nonfinite optimizer result")
            if any(torch.is_tensor(value) and not torch.isfinite(value).all()
                   for slot in optimizer.state.values() for value in slot.values()):
                raise ValueError("nonfinite optimizer state")
            training_rows.append(dict(case=case, **metrics, penalty=float(penalty.detach()),
                                      objective=float(objective.detach()), gradient_norm=grad_norm))
        rows, score = validate() if phase == "qualify" else (None, None)
        epoch = state["epoch"] + 1
        if phase == "qualify":
            if score < state["best_loss"]:
                state.update(best_loss=score, best_correction=_capture(model),
                             selected_epoch=epoch, patience=0)
            else:
                state["patience"] += 1
        state["epoch"] = epoch
        state["records"].append(dict(epoch=epoch, training=training_rows,
                                     validation=rows, validation_loss=score))
        persist()

    if phase == "qualify":
        _restore(model, state["best_correction"])
    return dict(status=state["status"], completed_epochs=state["epoch"],
                selected_epoch=state["selected_epoch"], best_validation_loss=state["best_loss"],
                patience=state["patience"], records=state["records"], checkpoint=str(path))
