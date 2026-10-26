"""Decompose the fixed-posterior prior objective without fitting or unit sampling.

The finite-draw identity concerns expected conditional KL. It does not measure
KL to the marginalized predictive law or identify joint posterior covariance.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from .population_prior_objective import PRIOR_PARAMETER_NAMES, prior_rollout_loss
from .population_state import ControlledPopulationState
from .poisson_trajectory_training import validate_prior_budget_binding
from .provenance import freeze_json, sha256_file
from .renewal_trajectory_training import variational_seed


@torch.no_grad()
def recorded_rollout(model, mean, variance, controls, origins, horizons, initial, dynamics):
    """Observe conditional means while leaving the objective and RNG use intact."""
    original = model.transition_mean
    recorded = {}
    step = 0

    def transition(state, control):
        nonlocal step
        result = original(state, control)
        step += 1
        if step in horizons:
            recorded[step] = result.detach().cpu().clone()
        return result

    with patch.object(model, "transition_mean", side_effect=transition):
        terms = prior_rollout_loss(model, mean, variance, controls, origins, horizons, initial, dynamics)
    if set(recorded) != set(horizons) or step != max(horizons):
        raise ValueError("complete registered horizon observations required")
    return terms, recorded


def conditional_kl_components(means, target_mean, target_variance, process_variance):
    """Keep coordinates; means has one leading fixed-draw axis.

    Use the empirical population variance (correction=0). That makes the
    decomposition exact for the actual finite draws, without a variance estimate.
    """
    means, target_mean, target_variance, process_variance = (
        value.detach().to(device="cpu", dtype=torch.float64) for value in (means, target_mean, target_variance, process_variance))
    if (means.ndim < 2 or means.shape[0] < 2 or means.shape[1:] != target_mean.shape
            or target_variance.shape != target_mean.shape
            or process_variance.shape != target_mean.shape[-2:]
            or not all(torch.isfinite(x).all() for x in (means, target_mean, target_variance, process_variance))
            or torch.any(target_variance <= 0) or torch.any(process_variance <= 0)):
        raise ValueError("aligned finite draws and strictly positive variances required")
    center = means.mean(0)
    bias = (target_mean - center).square()
    spread = (means - center).square().mean(0)
    variance_kl = .5 * (torch.log(process_variance / target_variance) + target_variance / process_variance - 1)
    bias_kl = .5 * bias / process_variance
    spread_kl = .5 * spread / process_variance
    direct = .5 * (torch.log(process_variance / target_variance) - 1
                   + (target_variance + (target_mean - means).square()) / process_variance).mean(0)
    torch.testing.assert_close(variance_kl + bias_kl + spread_kl, direct, rtol=1e-12, atol=1e-12)
    return dict(variance_kl=variance_kl, mean_bias_kl=bias_kl, path_spread_kl=spread_kl,
                direct_kl=direct, mean_squared_bias=bias, between_path_variance=spread,
                process_variance=process_variance.expand_as(target_mean),
                target_variance=target_variance, conditional_mean=center, target_mean=target_mean)


def _csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stage(cell, files, name, destination, consumed):
    row = files[name]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cell / name, destination)
    if destination.stat().st_size != row["bytes"] or sha256_file(destination) != row["sha256"]:
        raise ValueError("delivery input changed: " + name)
    consumed[name] = row["sha256"]
    return destination


def _checkpoint(cell, seed, scratch, source_hash, device):
    delivery = json.loads((cell / "delivery.json").read_text())
    if (delivery.get("status") != "PASS" or delivery.get("exit_code") != 0
            or delivery.get("training_seed") != seed or delivery.get("source_units") != 100000
            or not delivery.get("frozen_parameters_unchanged")):
        raise ValueError("accepted complete same-seed prior delivery required")
    files = {row["path"]: row for row in delivery["files"]}
    if len(files) != len(delivery["files"]):
        raise ValueError("duplicate delivery input paths")
    consumed = {}
    def stage(name):
        return _stage(cell, files, name, scratch / name, consumed)
    report = json.loads(stage("training/run.json").read_text())
    checkpoint = torch.load(stage("training/best.pt"), map_location="cpu", weights_only=True)
    binding = checkpoint["binding"]
    if (binding != report["binding"] or binding["seed"] != seed
            or binding["model_implementation_sha256"] != source_hash
            or binding["config"]["observation_objective"] != "fixed_posterior_multistep_prior_kl"
            or checkpoint["model_configuration"] != report["model_configuration"]
            or report["best_checkpoint_sha256"] != consumed["training/best.pt"]
            or checkpoint["next_epoch"] != report["best_epoch"] + 1):
        raise ValueError("prior checkpoint, report, source or objective mismatch")
    constructor = checkpoint["model_configuration"]
    if len(constructor["unit_ids"]) != 100000 or constructor["representation"] != "recurrent":
        raise ValueError("complete native recurrent model required")
    model = ControlledPopulationState(**constructor)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return dict(cell=cell, files=files, consumed=consumed, report=report,
                checkpoint=checkpoint, model=model, stage=stage,
                delivery_sha256=sha256_file(cell / "delivery.json"))


def validate_pair(parent, child):
    """Keep the admitted source migration and immutable target/readout contract."""
    a, b = parent["checkpoint"], child["checkpoint"]
    ar, br = parent["report"], child["report"]
    validate_prior_budget_binding(a["binding"], b["binding"], ar["trainer_sources"], br["trainer_sources"])
    if (a["model_configuration"] != b["model_configuration"] or a["initialization"] != b["initialization"]
            or ar["completed_epochs"] != 80 or br.get("effective_maximum_epochs") != 320
            or br["continuation"]["parent_artifact_sha256"]["best.pt"] != parent["consumed"]["training/best.pt"]
            or set(a["model"]) != set(b["model"])):
        raise ValueError("registered paired prior-budget lineage required")
    for name, tensor in a["model"].items():
        if name not in PRIOR_PARAMETER_NAMES and (tensor.dtype != b["model"][name].dtype or not torch.equal(tensor, b["model"][name])):
            raise ValueError("frozen parameter or buffer changed: " + name)


def diagnose(parent_campaign, continuation_campaign, output, device="cpu"):
    device = torch.device(device)
    node, partition = os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION")
    gpu_job = bool(os.environ.get("SLURM_JOB_GPUS"))
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Slurm allocation required")
    if device.type == "cpu":
        if gpu_job or (node, partition) not in (("cpu03", "gpu"), ("cpu04", "cpu")):
            raise ValueError("CPU-only allocation on an admitted partition/node pair required")
    elif device.type == "cuda":
        if not gpu_job or node not in {"cpu03", "cpu04", "cpu05", "cpu06", "gpu03"} or not torch.cuda.is_available():
            raise ValueError("allocated available GPU on an authorized node required")
    else:
        raise ValueError("CPU or CUDA diagnostic device required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_hash = sha256_file(Path(__file__).with_name("population_state.py"))
    rows, replay_rows, historical_rows, bindings = [], [], [], {}
    for seed in (41, 42, 43):
        with tempfile.TemporaryDirectory(prefix=f"neuronsim-prior-kl-{seed}-", dir="/tmp") as scratch:
            scratch = Path(scratch)
            pair = {label: _checkpoint(Path(campaign) / f"seed_{seed}", seed, scratch / label, source_hash, device)
                    for label, campaign in (("prior80", parent_campaign), ("continued", continuation_campaign))}
            parent, child = pair["prior80"], pair["continued"]
            validate_pair(parent, child)
            config = parent["checkpoint"]["binding"]["config"]
            constructor = parent["checkpoint"]["model_configuration"]
            origins = tuple(int(value / config["bin_ms"]) for value in config["prior_origins_ms"])
            horizons = tuple(int(value / config["bin_ms"]) for value in config["prior_horizons_ms"])
            cases = [(split, f"train_block_{block:03d}_familiar_{template}")
                     for split, key in (("fit", "fit_blocks"), ("selection", "validation_blocks"))
                     for block in config[key] for template in range(4)]
            expected = {f"training/posterior_cache/{case}.pt" for _, case in cases}
            if len(cases) != 72 or sum(split == "fit" for split, _ in cases) != 56:
                raise ValueError("complete56/16 development partition required")
            for item in pair.values():
                if {name for name in item["files"] if name.startswith("training/posterior_cache/")} != expected:
                    raise ValueError("complete posterior cache set required")
            historical = {}
            for label, item in pair.items():
                with item["stage"]("training/validation.csv").open() as stream:
                    selected = [row for row in csv.DictReader(stream) if int(row["epoch"]) == item["report"]["best_epoch"]]
                keys = [(row["case_id"], int(row["draw"])) for row in selected]
                expected_keys = {(case, draw) for split, case in cases if split == "selection" for draw in range(config["validation_draws"])}
                if len(keys) != len(set(keys)) or set(keys) != expected_keys:
                    raise ValueError("complete selected-epoch validation case/draw history required")
                historical[label] = dict(zip(keys, selected))
            arrays = {label: {} for label in pair}
            for split, case in cases:
                name = f"training/posterior_cache/{case}.pt"
                if parent["files"][name] != child["files"][name]:
                    raise ValueError("paired posterior cache binding changed")
                saved = torch.load(parent["stage"](name), map_location="cpu", weights_only=True)
                # One staged read authenticates the shared cache bytes; child manifest binds the same bytes.
                child["consumed"][name] = child["files"][name]["sha256"]
                cb = saved["binding"]
                if (cb["case_id"] != case or cb["parent_checkpoint_sha256"] != parent["checkpoint"]["initialization"]["parent_checkpoint_sha256"]
                        or cb["model_implementation_sha256"] != source_hash or cb["bin_ms"] != config["bin_ms"]
                        or any(parent["checkpoint"]["binding"][key] != value for key, value in cb["cohort"].items())):
                    raise ValueError("posterior identity, original parent, source or clock changed")
                mean, variance, controls = (saved[key] for key in ("mean", "variance", "controls"))
                if mean.shape != (1, 1200, len(constructor["field_names"]), constructor["latent_channels"]):
                    raise ValueError("complete six-second posterior axes required")
                for label, item in pair.items():
                    model_device = item["model"].unit_loading.device
                    inputs = tuple(value.to(model_device) for value in (mean, variance, controls))
                    recorded = {horizon: [] for horizon in horizons}
                    losses = []
                    for draw in range(config["validation_draws"]):
                        streams = [torch.Generator(device=model_device).manual_seed(variational_seed(config["validation_seed_base"], seed, domain, case, draw))
                                   for domain in ("prior_validation_initial", "prior_validation_dynamics")]
                        terms, means = recorded_rollout(item["model"], *inputs, origins, horizons, *streams)
                        losses.append({key: float(value) for key, value in terms.items()})
                        if split == "selection":
                            saved_terms = historical[label][(case, draw)]
                            for key, value in losses[-1].items():
                                reference = float(saved_terms[key])
                                if device.type == "cuda":
                                    np.testing.assert_allclose(value, reference, rtol=1e-5, atol=1e-5)
                                historical_rows.append(dict(seed=seed, checkpoint=label, case_id=case, draw=draw,
                                    term=key, historical_native_kl=reference, diagnostic_native_kl=value,
                                    absolute_difference=abs(value-reference), same_device_type=device.type=="cuda"))
                        for horizon in horizons:
                            recorded[horizon].append(means[horizon])
                    by_case = {}
                    for horizon in horizons:
                        indices = [origin + horizon - 1 for origin in origins]
                        target_mean = mean[:, indices].reshape(len(origins), *mean.shape[-2:])
                        target_var = variance[:, indices].reshape_as(target_mean)
                        # Native exp precedes float64 promotion, matching the actual objective.
                        process_var = item["model"].process_log_variance.detach().exp()
                        parts = conditional_kl_components(torch.stack(recorded[horizon]), target_mean, target_var, process_var)
                        native = np.mean([loss[f"prior_kl_h{horizon}"] for loss in losses])
                        direct = float(parts["direct_kl"].sum((-1, -2)).mean())
                        np.testing.assert_allclose(direct, native, rtol=1e-5, atol=1e-5)
                        replay_rows.append(dict(seed=seed, checkpoint=label, split=split, case_id=case,
                            horizon_ms=horizon * config["bin_ms"], native_float32_kl=native,
                            direct_float64_kl=direct, absolute_roundoff_difference=abs(direct-native)))
                        for key, value in parts.items():
                            by_case.setdefault(key, []).append(value.numpy())
                        for i, origin in enumerate(origins):
                            row = dict(seed=seed, checkpoint=label, selected_epoch=item["report"]["best_epoch"],
                                split=split, case_id=case, origin_ms=origin * config["bin_ms"],
                                horizon_ms=horizon * config["bin_ms"], draws=config["validation_draws"])
                            for key, value in parts.items():
                                row[key + ("_sum" if key.endswith("kl") else "_coordinate_mean")] = float(value[i].sum() if key.endswith("kl") else value[i].mean())
                            rows.append(row)
                    for key, values in by_case.items():
                        arrays[label].setdefault(key, []).append(np.stack(values, axis=1))
                (scratch / "prior80" / name).unlink()
            for label, item in pair.items():
                np.savez_compressed(output / f"seed_{seed}_{label}_coordinates.npz",
                    **{key: np.asarray(value) for key, value in arrays[label].items()},
                    case_ids=np.asarray([case for _, case in cases]), splits=np.asarray([split for split, _ in cases]),
                    field_names=np.asarray(constructor["field_names"]), origins_ms=np.asarray(config["prior_origins_ms"]),
                    horizons_ms=np.asarray(config["prior_horizons_ms"]))
                bindings[f"{seed}/{label}"] = dict(cell=str(item["cell"].resolve()), delivery_sha256=item["delivery_sha256"],
                    files=item["consumed"], completed_epochs=item["report"]["completed_epochs"], best_epoch=item["report"]["best_epoch"])
    _csv(output / "kl_by_case.csv", rows)
    _csv(output / "objective_replay.csv", replay_rows)
    _csv(output / "selected_validation_replay.csv", historical_rows)
    metrics = [key for key in rows[0] if key.endswith(("_sum", "_coordinate_mean"))]
    summaries = []
    for seed in (41, 42, 43):
        for label in ("prior80", "continued"):
            for split in ("fit", "selection"):
                for horizon in config["prior_horizons_ms"]:
                    selected = [row for row in rows if (row["seed"], row["checkpoint"], row["split"], row["horizon_ms"]) == (seed, label, split, horizon)]
                    summaries.append(dict(seed=seed, checkpoint=label, split=split, horizon_ms=horizon,
                        case_origins=len(selected), **{key: float(np.mean([row[key] for row in selected])) for key in metrics}))
    _csv(output / "kl_summary.csv", summaries)
    names = ["kl_by_case.csv", "objective_replay.csv", "selected_validation_replay.csv", "kl_summary.csv",
             *[f"seed_{seed}_{label}_coordinates.npz" for seed in (41,42,43) for label in ("prior80","continued")]]
    result = dict(status="COMPLETE_PAIRED_CONDITIONAL_KL_DIAGNOSTIC", training_seeds=[41,42,43],
        cases_per_seed=72, fit_cases_per_seed=56, selection_cases_per_seed=16,
        case_origin_horizon_rows=len(rows), replay_rows=len(replay_rows), summary_rows=len(summaries),
        maximum_absolute_native_roundoff_difference=max(row["absolute_roundoff_difference"] for row in replay_rows),
        coordinate_axes=["case","origin","horizon","source_field","latent_channel"],
        draws=config["validation_draws"], random_streams="registered validation seed identities and draw layout reused on the same device for both checkpoints",
        diagnostic_device=str(model_device), diagnostic_node=node,
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(model_device) if device.type=="cuda" else None,
        historical_gpu_draws_replayed=device.type=="cuda", selected_validation_replay_rows=len(historical_rows),
        maximum_absolute_historical_validation_difference=max(row["absolute_difference"] for row in historical_rows),
        fitting_split_draws="offline diagnostic draws; not historical optimizer draws",
        interpretation="finite-draw expected conditional KL allocation; not marginalized predictive KL or a covariance capacity test",
        training_performed=False, source_simulation_performed=False, encoding_performed=False,
        latent_sampling_performed=True, unit_sampling_performed=False, scientific_claim_qualified=False,
        consumed=bindings, job_id=os.environ["SLURM_JOB_ID"], diagnostic_source_sha256=sha256_file(Path(__file__)),
        objective_source_sha256=sha256_file(Path(__file__).with_name("population_prior_objective.py")),
        artifacts={name:sha256_file(output/name) for name in names})
    freeze_json(output / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-campaign", type=Path, required=True)
    parser.add_argument("--continuation-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = diagnose(args.parent_campaign, args.continuation_campaign, args.output, args.device)
    print(json.dumps({key:result[key] for key in ("status","case_origin_horizon_rows","replay_rows","summary_rows")}))


if __name__ == "__main__":
    main()
