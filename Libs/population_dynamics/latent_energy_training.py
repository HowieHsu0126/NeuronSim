"""Train the separately conditioned EAG autoencoder and latent energy model.

One energy pass visits all fitting origin windows. Logical batches span shuffled
passes, so 4000 passes of 168 windows produce 2625 complete batches of 256.
A terminal partial logical batch preserves every presentation for source cohorts
whose complete-pass budget is not divisible by 256. Only completed autoencoder
epochs and energy optimizer updates are committed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch
import yaml

from .latent_energy import ControlledLatentEnergy
from .latent_spike_diffusion import SOURCE_REVISION
from .latent_spike_diffusion_training import _ae_epoch, _ae_sequences, _seed, _write_json
from .latent_spike_diffusion_training import _scheduler as _ae_scheduler
from .native_trajectory_cohort import TrajectoryCohort
from .poisson_trajectory_training import _save_checkpoint
from .provenance import sha256_file

FIXED_PROTOCOL = dict(
    schema="neural_population_latent_energy_training_v1",
    training_seeds=[41, 42, 43], fit_blocks=list(range(14)), validation_blocks=list(range(14, 18)),
    bin_ms=5.0, complete_episode_bins=1200, expected_poisson_rng_family="per_edge_philox4_32_10",
    history_bins=200, horizon_bins=1000, forecast_origins_bins=[200, 600, 1000], forecast_bins=100,
    model=dict(input_size=1, bin_seconds=.005, complete_episode_bins=1200, maximum_prefix_bins=1000,
        forecast_bins=100, latent_channels=16, hidden_channels=256, encoder_blocks=4,
        decoder_blocks=4, linear_layers_per_block=4,
        generator_options=dict(encoder_width=256, encoder_depth=4, encoder_heads=4,
            decoder_width=256, decoder_depth=4, decoder_heads=4, score_mlp_width=768,
            score_mlp_depth=6, noise_channels=64, alpha=1.0, training_temperature=1.0,
            mask_ratio_min=.7, mask_ratio_max=1.0, label_dropout=.1,
            attention_dropout=.1, projection_dropout=.1, guidance_schedule="linear")),
    autoencoder=dict(maximum_epochs=260, learning_rate=.001, weight_decay=.01,
        mask_probability=.5, latent_beta=.001, temporal_beta=.2, gradient_limit=2.0,
        warmup_epochs=10, cosine_budget_factor=1.1, validation_every=10, validation_draws=4),
    energy=dict(maximum_epochs=4000, learning_rate=.0001, weight_decay=.02,
        optimizer_betas=[.9, .95], gradient_limit=3.0, warmup_epochs=100,
        learning_rate_schedule="constant_after_warmup", source_batch_size=256,
        validation_every=100, validation_draws=4),
    sampling=dict(joint_samples=16, steps=50, guidance=4.0, temperature=.7),
    profile="paper_MC_Maze_dimensions_and_budget_with_recorded_source_numerical_defaults",
    formal_training_completed=False,
)


V1_SCHEMA = "neural_population_v1_latent_energy_training_v1"


def validate_config(config, seed):
    if config.get("schema") == V1_SCHEMA:
        # Historical frozen Mesocircuit packages do not stage source adapters.
        from .v1_latent_energy_training import validate_config as validate_v1
        return validate_v1(config, seed)
    for name, expected in FIXED_PROTOCOL.items():
        actual = config.get(name)
        if name == "energy" and isinstance(actual, dict):
            actual = {key: value for key, value in actual.items() if key != "microbatch_size"}
        if actual != expected:
            raise ValueError("frozen complete-cohort EAG protocol differs: " + name)
    if type(seed) is not int or seed not in (41, 42, 43):
        raise ValueError("registered training seed41/42/43 required")
    microbatch = config["energy"].get("microbatch_size")
    if type(microbatch) is not int or not 1 <= microbatch <= config["energy"]["source_batch_size"]:
        raise ValueError("positive measured microbatch_size within source batch required")
    if config.get("device") not in ("cpu", "cuda") or config.get("cpu_threads") not in (1, 2):
        raise ValueError("explicit supported device and bounded CPU threads required")
    if config["device"] == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU training requires CUDA_VISIBLE_DEVICES empty")


def _sources(config=None):
    names = ("latent_energy_training.py", "latent_energy.py", "latent_energy_generator.py",
             "latent_spike_diffusion_training.py", "latent_spike_diffusion.py",
             "native_trajectory_cohort.py", "native_count_windows.py", "provenance.py",
             "poisson_trajectory_training.py", "poisson_state_space.py")
    if config is not None and config.get("schema") == V1_SCHEMA:
        names += ("v1_latent_energy_training.py", "v1_latent_energy_data.py",
                  "v1_population_training_data.py", "v1_renewal_data.py", "circuit_data.py",
                  "v1_nest_time_semantics.py", "v1_visual_controls.py")
    return {name: sha256_file(Path(__file__).with_name(name)) for name in names}


def _model(cohort, config):
    # ControlledLatentEnergy prepares both bidirectional S4 stacks before an optimizer exists.
    return ControlledLatentEnergy(tuple(map(int, cohort.unit_ids)), **config["model"]).to(config["device"])


def _autoencoder_config(config):
    ae = config["autoencoder"]
    return dict(config, maximum_autoencoder_epochs=ae["maximum_epochs"],
        autoencoder_warmup_epochs=ae["warmup_epochs"], autoencoder_cosine_budget_factor=ae["cosine_budget_factor"],
        mask_probability=ae["mask_probability"], latent_beta=ae["latent_beta"], temporal_beta=ae["temporal_beta"],
        autoencoder_gradient_limit=ae["gradient_limit"], validation_draws=ae["validation_draws"])


def _optimizer(model, config, stage, fit_episodes):
    if stage == "autoencoder":
        ae = config["autoencoder"]
        model.autoencoder.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.autoencoder.parameters(), lr=ae["learning_rate"], weight_decay=ae["weight_decay"])
        scheduler = _ae_scheduler(optimizer, fit_episodes * 3, _autoencoder_config(config), "autoencoder")
    else:
        energy = config["energy"]
        model.autoencoder.requires_grad_(False)
        model.autoencoder.eval()
        optimizer = torch.optim.AdamW(model.energy.parameters(), lr=energy["learning_rate"],
                                      weight_decay=energy["weight_decay"], betas=tuple(energy["optimizer_betas"]))
        warmup = energy["warmup_epochs"] * fit_episodes * len(config["forecast_origins_bins"])
        batch = energy["source_batch_size"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda step: min(1.0, step * batch / warmup) if warmup else 1.0)
    return optimizer, scheduler


@torch.no_grad()
def build_latent_cache(model, cohort, config):
    """Fit target moments using fitting episodes; encode every prefix separately."""
    model.eval()
    fit_ids = [case["case_id"] for case in cohort.fit]
    validation_ids = [case["case_id"] for case in cohort.validation]
    if (len(set(fit_ids)) != len(fit_ids) or len(set(validation_ids)) != len(validation_ids)
            or set(fit_ids) & set(validation_ids)):
        raise ValueError("disjoint identified fitting and validation episodes required")
    rows = {"fit": [], "validation": []}
    total = squared = None
    number = 0
    for split, cases in (("fit", cohort.fit), ("validation", cohort.validation)):
        for case in cases:
            counts, controls = cohort.window(case)
            counts = torch.as_tensor(counts, device=config["device"], dtype=torch.float32)[None]
            controls = torch.as_tensor(controls, device=config["device"], dtype=torch.float32)[None]
            for origin in config["forecast_origins_bins"]:
                stop = origin + config["forecast_bins"]
                if origin < 2 or stop > counts.shape[1] or counts.shape[:2] != controls.shape[:2]:
                    raise ValueError("complete aligned observed prefix and future target required")
                prefix = model.autoencoder.encode(model._counts(counts[:, :origin]))
                target = model.autoencoder.encode(model._counts(counts[:, origin:stop]))
                if not torch.isfinite(prefix).all() or not torch.isfinite(target).all():
                    raise FloatingPointError("nonfinite separately encoded EAG latent")
                if split == "fit":
                    value = target.double()
                    sums, squares = value.sum((0, 2), keepdim=True), value.square().sum((0, 2), keepdim=True)
                    total = sums if total is None else total + sums
                    squared = squares if squared is None else squared + squares
                    number += value.shape[0] * value.shape[2]
                rows[split].append(dict(case_id=case["case_id"], origin=origin,
                    prefix=prefix.cpu(), target=target.cpu(),
                    observed_controls=controls[:, :origin].cpu().clone(), stimulus=controls[:, origin:stop].cpu().clone()))
    if not number or not rows["validation"]:
        raise ValueError("nonempty fitting targets and validation windows required")
    mean = total / number
    variance = squared / number - mean.square()
    if not torch.isfinite(variance).all() or torch.any(variance <= 0):
        raise ValueError("positive finite fitting-only EAG target variation required")
    model.set_latent_scale(mean, variance.sqrt())
    mean, sd = model.latent_mean.cpu(), model.latent_sd.cpu()
    for values in rows.values():
        for row in values:
            row["prefix"] = ((row["prefix"] - mean) / sd).transpose(1, 2).contiguous()
            row["target"] = ((row["target"] - mean) / sd).transpose(1, 2).contiguous()
    return dict(rows=rows, moments=dict(fit_cases=fit_ids, fit_windows=len(rows["fit"]),
        fit_time_values_per_channel=number, mean=mean, population_sd=sd,
        transform="all_fit_origins_separately_encoded_future_windows_population_sd"),
        source_denominators=dict(fit_complete_episodes=len(fit_ids), validation_complete_episodes=len(validation_ids),
            fit_origin_windows=len(rows["fit"]), validation_origin_windows=len(rows["validation"]),
            future_bins=config["forecast_bins"], units=len(cohort.unit_ids)))


def new_sampler():
    return dict(completed_passes=0, order=[], cursor=0, presentations=0)


def take_logical_batch(sampler, number, maximum_passes, batch_size, shuffle_rng):
    """Consume complete shuffled passes, retaining the cursor across batches."""
    total = number * maximum_passes
    if any(type(value) is not int or value < 1 for value in (number, maximum_passes, batch_size)):
        raise ValueError("positive complete-pass and logical batch sizes required")
    if (sampler["completed_passes"] * number + sampler["cursor"] != sampler["presentations"]
            or not 0 <= sampler["presentations"] <= total
            or not 0 <= sampler["cursor"] < number
            or (sampler["order"] and sorted(sampler["order"]) != list(range(number)))
            or (not sampler["order"] and sampler["cursor"] != 0)):
        raise ValueError("energy shuffled-pass cursor is inconsistent")
    if sampler["presentations"] == total:
        return []
    result = []
    wanted = min(batch_size, total - sampler["presentations"])
    while len(result) < wanted:
        if not sampler["order"]:
            sampler["order"] = torch.randperm(number, generator=shuffle_rng).tolist()
        take = min(wanted - len(result), number - sampler["cursor"])
        indices = sampler["order"][sampler["cursor"]:sampler["cursor"] + take]
        result.extend(dict(index=index, pass_index=sampler["completed_passes"]) for index in indices)
        sampler["cursor"] += take
        sampler["presentations"] += take
        if sampler["cursor"] == number:
            sampler["completed_passes"] += 1
            sampler["order"] = []
            sampler["cursor"] = 0
    return result


def _loss_values(terms):
    values = {key: float(value.detach()) for key, value in terms.items()
              if isinstance(value, torch.Tensor) and value.numel() == 1}
    if "loss" not in values or not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("finite named EAG loss terms required")
    return values


def _energy_update(model, cache, config, batch, rng, optimizer, scheduler):
    energy = config["energy"]
    if not 1 <= len(batch) <= energy["source_batch_size"]:
        raise ValueError("nonempty source logical batch within declared size required")
    model.autoencoder.eval()
    model.energy.train()
    options = config["model"]["generator_options"]
    ratio = options["mask_ratio_min"] + (options["mask_ratio_max"] - options["mask_ratio_min"]) * float(
        torch.rand((), generator=rng, device=config["device"]))
    # A pass is kept separate within each prefix-length group so pass losses
    # have exact example denominators even when a logical batch crosses passes.
    groups = {}
    for item in batch:
        row = cache["rows"]["fit"][item["index"]]
        groups.setdefault((item["pass_index"], row["origin"]), []).append(row)
    optimizer.zero_grad(set_to_none=True)
    aggregate, by_pass, microbatches = {}, {}, 0
    for (pass_index, _), rows in groups.items():
        for start in range(0, len(rows), energy["microbatch_size"]):
            chunk = rows[start:start + energy["microbatch_size"]]
            target = torch.cat([row["target"] for row in chunk]).to(config["device"])
            prefix = torch.cat([row["prefix"] for row in chunk]).to(config["device"])
            controls = torch.cat([torch.cat((row["observed_controls"], row["stimulus"]), dim=1) for row in chunk]).to(config["device"])
            terms = model.energy.loss(target, prefix, controls, generator=rng, mask_ratio=ratio)
            values = _loss_values(terms)
            (terms["loss"] * (len(chunk) / len(batch))).backward()
            contribution = by_pass.setdefault(str(pass_index), dict(examples=0, sums={}))
            contribution["examples"] += len(chunk)
            for key, value in values.items():
                aggregate[key] = aggregate.get(key, 0.0) + value * len(chunk) / len(batch)
                contribution["sums"][key] = contribution["sums"].get(key, 0.0) + value * len(chunk)
            microbatches += 1
    torch.nn.utils.clip_grad_norm_(model.energy.parameters(), energy["gradient_limit"], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return dict(mean=aggregate, pass_contributions=by_pass, mask_ratio=ratio, microbatches=microbatches)


@torch.no_grad()
def _energy_validation(model, cache, config, seed):
    model.energy.eval()
    rows = []
    for row in cache["rows"]["validation"]:
        target, prefix = row["target"].to(config["device"]), row["prefix"].to(config["device"])
        controls = torch.cat((row["observed_controls"], row["stimulus"]), dim=1).to(config["device"])
        draws = []
        for draw in range(config["energy"]["validation_draws"]):
            stream = torch.Generator(device=config["device"]).manual_seed(_seed(seed, "energy_validation", row["case_id"], row["origin"], draw))
            draws.append(_loss_values(model.energy.loss(target, prefix, controls, generator=stream)))
        rows.append(dict(case_id=row["case_id"], origin=row["origin"], draws=draws))
    keys = rows[0]["draws"][0]
    return dict(mean={key: math.fsum(draw[key] for row in rows for draw in row["draws"])
                      / sum(len(row["draws"]) for row in rows) for key in keys}, windows=rows)


def _record_pass_losses(state, contributions, number):
    for name, contribution in contributions.items():
        current = state["energy_pass_accumulators"].setdefault(name, dict(examples=0, sums={}))
        current["examples"] += contribution["examples"]
        for key, value in contribution["sums"].items():
            current["sums"][key] = current["sums"].get(key, 0.0) + value
        if current["examples"] > number:
            raise ValueError("energy pass loss denominator exceeded fitting window count")
    for name in sorted(tuple(state["energy_pass_accumulators"]), key=int):
        row = state["energy_pass_accumulators"][name]
        if row["examples"] == number:
            state["energy_epochs"].append(dict(epoch=int(name), presentations=number,
                mean={key: value / number for key, value in row["sums"].items()}))
            del state["energy_pass_accumulators"][name]


def _reports(output, model, state, digest):
    _write_json(output / "epochs.json", state["history"])
    _write_json(output / "energy_epochs.json", state["energy_epochs"])
    sampler = state["energy_sampler"]
    report = dict(status="FITTED_DEVELOPMENT" if state["stage"] == "complete" else "RUNNING",
        training_seed=state["binding"]["seed"], stage=state["stage"], completed_stage_epochs=state["next_epoch"],
        autoencoder_completed_epochs=state["autoencoder_completed_epochs"], energy_completed_passes=sampler["completed_passes"],
        energy_presentations=sampler["presentations"], energy_optimizer_steps=state["steps"] if state["stage"] != "autoencoder" else 0,
        training_commits=state["commits"], history_entries=len(state["history"]), energy_epoch_rows=len(state["energy_epochs"]),
        best_epoch=state["best_epoch"], best_loss=state["best_loss"], best_presentations=state["best_presentations"],
        latest_checkpoint_sha256=digest, selected_checkpoint_sha256=state["selected_digest"],
        selected_autoencoder_checkpoint_sha256=state["autoencoder_selected_digest"], cache_sha256=state["cache_digest"],
        source_denominators=state["source_denominators"], parameter_count=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=state["elapsed_seconds"], resumed=state["resumed"],
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        energy_selection_timing="first_complete_optimizer_update_at_or_after_validation_pass_boundary",
        source_revision=SOURCE_REVISION, scientific_claim_qualified=False, recorded_utc=datetime.now(timezone.utc).isoformat())
    _write_json(output / "run.json", report)
    return report


def _commit(output, model, optimizer, scheduler, rng, shuffle_rng, state):
    previous = json.loads((output / "committed.json").read_text()) if (output / "committed.json").exists() else None
    payload = dict(state=state, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                   rng=rng.get_state(), shuffle_rng=shuffle_rng.get_state(), global_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if model.latent_mean.is_cuda else None)
    directory = output / "checkpoints"
    directory.mkdir(exist_ok=True)
    checkpoint = directory / f"{state['stage']}_{state['commits']:06d}.pt"
    _save_checkpoint(payload, checkpoint)
    digest = sha256_file(checkpoint)
    _write_json(output / "committed.json", dict(checkpoint=str(checkpoint.relative_to(output)), sha256=digest,
        stage=state["stage"], next_epoch=state["next_epoch"], commits=state["commits"]))
    report = _reports(output, model, state, digest)
    if previous is not None and previous["checkpoint"] != str(checkpoint.relative_to(output)):
        (output / previous["checkpoint"]).unlink()
    return report


def _select(output, model, state, score):
    if state["best_loss"] is not None and score >= state["best_loss"]:
        return
    epoch = state["next_epoch"] - 1
    state.update(best_epoch=epoch, best_loss=score, best_presentations=state["energy_sampler"]["presentations"])
    path = output / "checkpoints" / f"selected_{state['stage']}_{state['commits'] + 1:06d}.pt"
    _save_checkpoint(dict(model=model.state_dict(), binding=state["binding"], stage=state["stage"], epoch=epoch,
        score=score, cache_digest=state["cache_digest"], energy_presentations=state["best_presentations"],
        autoencoder_selected_digest=state["autoencoder_selected_digest"]), path)
    state["selected_path"], state["selected_digest"] = str(path.relative_to(output)), sha256_file(path)


def _load_selected(output, model, state):
    path = output / state["selected_path"]
    if sha256_file(path) != state["selected_digest"]:
        raise ValueError("EAG selected stage checkpoint changed")
    payload = torch.load(path, map_location=model.latent_mean.device, weights_only=False)
    if (payload["binding"] != state["binding"] or payload["stage"] != state["stage"]
            or payload["epoch"] != state["best_epoch"] or payload["score"] != state["best_loss"]
            or payload["cache_digest"] != state["cache_digest"]):
        raise ValueError("EAG selected stage identity differs")
    model.load_state_dict(payload["model"])


def train(acquisition, fields, config_path, output, seed, resume=False, *, stop_after_commits=None):
    acquisition, config_path, output = map(Path, (acquisition, config_path, output))
    fields = None if fields is None else Path(fields)
    config = yaml.safe_load(config_path.read_text())
    validate_config(config, seed)
    if stop_after_commits is not None and (type(stop_after_commits) is not int or stop_after_commits < 1):
        raise ValueError("positive per-invocation complete-commit boundary required")
    if output.exists() and not resume:
        raise FileExistsError("existing EAG output requires explicit resume")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    if config.get("schema") == V1_SCHEMA:
        from .v1_latent_energy_data import V1LatentEnergyCohort
        from .v1_latent_energy_training import validate_config as validate_v1
        cohort = V1LatentEnergyCohort(acquisition, config, config_validator=validate_v1)
    else:
        if fields is None:
            raise ValueError("Mesocircuit EAG training requires its field dataset")
        cohort = TrajectoryCohort(acquisition, fields, config)
    fit_windows = len(cohort.fit) * len(config["forecast_origins_bins"])
    binding = dict(config=config, seed=seed, cohort=cohort.binding, sources=_sources(config),
                   source_revision=SOURCE_REVISION, torch_version=torch.__version__)
    model = _model(cohort, config)
    rng = torch.Generator(device=config["device"]).manual_seed(_seed(seed, "training"))
    shuffle_rng = torch.Generator().manual_seed(_seed(seed, "energy_shuffle"))
    cache = None
    if resume:
        pointer = json.loads((output / "committed.json").read_text())
        if sha256_file(output / pointer["checkpoint"]) != pointer["sha256"]:
            raise ValueError("EAG committed checkpoint changed")
        payload = torch.load(output / pointer["checkpoint"], map_location=config["device"], weights_only=False)
        state = payload["state"]
        if (state["binding"] != binding or pointer["stage"] != state["stage"]
                or pointer["next_epoch"] != state["next_epoch"] or pointer["commits"] != state["commits"]):
            raise ValueError("EAG resume source/data/protocol or cursor identity differs")
        model.load_state_dict(payload["model"])
        state["resumed"] = True
        optimizer, scheduler = _optimizer(model, config, "energy" if state["stage"] == "complete" else state["stage"], len(cohort.fit))
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        rng.set_state(payload["rng"].cpu())
        shuffle_rng.set_state(payload["shuffle_rng"].cpu())
        torch.set_rng_state(payload["global_rng"].cpu())
        if payload["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in payload["cuda_rng"]])
        if state["stage"] in ("energy", "complete"):
            if sha256_file(output / "latent_cache.pt") != state["cache_digest"]:
                raise ValueError("EAG fitting-only latent cache changed")
            if state["stage"] == "energy":
                cache = torch.load(output / "latent_cache.pt", map_location="cpu", weights_only=False)
        if state["stage"] == "complete":
            # The commit pointer is authoritative if interruption preceded derived JSON reports.
            return _reports(output, model, state, pointer["sha256"])
    else:
        output.mkdir(parents=True)
        state = dict(binding=binding, stage="autoencoder", next_epoch=0, commits=0, steps=0,
            best_epoch=None, best_loss=None, best_presentations=None, selected_path=None, selected_digest=None,
            autoencoder_selected_path=None, autoencoder_selected_digest=None, autoencoder_completed_epochs=0,
            cache_digest=None, energy_sampler=new_sampler(), energy_pass_accumulators={}, energy_epochs=[],
            next_validation_pass=config["energy"]["validation_every"], history=[], elapsed_seconds=0.0, resumed=False,
            source_denominators=dict(fit_complete_episodes=len(cohort.fit), validation_complete_episodes=len(cohort.validation),
                fit_origin_windows=fit_windows, validation_origin_windows=len(cohort.validation) * len(config["forecast_origins_bins"]),
                future_bins=config["forecast_bins"], units=len(cohort.unit_ids)))
        optimizer, scheduler = _optimizer(model, config, "autoencoder", len(cohort.fit))
        _commit(output, model, optimizer, scheduler, rng, shuffle_rng, state)
    start, previous, invocation_commits = time.monotonic(), state["elapsed_seconds"], 0
    try:
        while state["stage"] != "complete":
            if state["stage"] == "autoencoder":
                epoch = state["next_epoch"]
                fitting = _ae_epoch(model, cohort, _autoencoder_config(config), seed, epoch, rng, optimizer, scheduler)
                state["next_epoch"] = epoch + 1
                state["autoencoder_completed_epochs"] = epoch + 1
                state["steps"] += len(cohort.fit) * 3
                validation = None
                if (epoch + 1) % config["autoencoder"]["validation_every"] == 0 or epoch + 1 == config["autoencoder"]["maximum_epochs"]:
                    validation = _ae_epoch(model, cohort, _autoencoder_config(config), seed, epoch, rng)
                    _select(output, model, state, validation["mean"]["loss"])
                state["history"].append(dict(stage="autoencoder", epoch=epoch, training=fitting, validation=validation))
                if state["next_epoch"] == config["autoencoder"]["maximum_epochs"]:
                    _load_selected(output, model, state)
                    state["autoencoder_selected_path"], state["autoencoder_selected_digest"] = state["selected_path"], state["selected_digest"]
                    cache = build_latent_cache(model, cohort, config)
                    _save_checkpoint(cache, output / "latent_cache.pt")
                    state.update(stage="energy", next_epoch=0, steps=0, best_epoch=None, best_loss=None,
                        best_presentations=None, selected_path=None, selected_digest=None,
                        cache_digest=sha256_file(output / "latent_cache.pt"))
                    optimizer, scheduler = _optimizer(model, config, "energy", len(cohort.fit))
            else:
                sampler = state["energy_sampler"]
                before = sampler["presentations"]
                batch = take_logical_batch(sampler, fit_windows, config["energy"]["maximum_epochs"],
                                           config["energy"]["source_batch_size"], shuffle_rng)
                if not batch:
                    raise ValueError("terminal energy cursor lacks its completed checkpoint")
                fitting = _energy_update(model, cache, config, batch, rng, optimizer, scheduler)
                state["steps"] += 1
                state["next_epoch"] = sampler["completed_passes"]
                _record_pass_losses(state, fitting["pass_contributions"], fit_windows)
                terminal = sampler["presentations"] == fit_windows * config["energy"]["maximum_epochs"]
                validation = None
                if sampler["completed_passes"] >= state["next_validation_pass"] or terminal:
                    validation = _energy_validation(model, cache, config, seed)
                    validation["first_scheduled_pass"] = state["next_validation_pass"]
                    validation["actual_presentations"] = sampler["presentations"]
                    _select(output, model, state, validation["mean"]["loss"])
                    while state["next_validation_pass"] <= sampler["completed_passes"]:
                        state["next_validation_pass"] += config["energy"]["validation_every"]
                state["history"].append(dict(stage="energy", update=state["steps"] - 1,
                    start_presentations=before, end_presentations=sampler["presentations"], completed_passes=sampler["completed_passes"],
                    mask_ratio=fitting["mask_ratio"], microbatches=fitting["microbatches"], training=fitting["mean"], validation=validation))
                if terminal:
                    if state["energy_pass_accumulators"] or len(state["energy_epochs"]) != config["energy"]["maximum_epochs"]:
                        raise ValueError("complete energy pass losses required")
                    _load_selected(output, model, state)
                    state["stage"] = "complete"
            state["commits"] += 1
            state["elapsed_seconds"] = previous + time.monotonic() - start
            report = _commit(output, model, optimizer, scheduler, rng, shuffle_rng, state)
            invocation_commits += 1
            if stop_after_commits is not None and invocation_commits >= stop_after_commits:
                break
        return report
    except Exception as exc:
        name = "failure_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"
        _write_json(output / name, dict(status="FAILED", error_type=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(), stage=state["stage"], incomplete_epoch=state["next_epoch"],
            incomplete_presentations=state["energy_sampler"]["presentations"], scientific_claim_qualified=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition", "fields", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-commits", type=int)
    args = parser.parse_args()
    print(json.dumps(train(args.acquisition, args.fields, args.config, args.output, args.seed,
                           args.resume, stop_after_commits=args.stop_after_commits)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
