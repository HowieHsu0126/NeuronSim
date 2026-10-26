"""Inspect cached posterior targets against the native prior's mean envelopes.

This is a development diagnostic of fixed posterior targets. Future posterior
means condition on future observations. An incompatible target is not an
impossible future sample, a count bound, or evidence of miscalibration.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from .provenance import freeze_json, sha256_file


def mean_envelopes(boundary, decay, steps):
    """Outer bounds for d*z + (1-d)*tanh(...) with zero-mean innovations."""
    boundary, decay = np.asarray(boundary, dtype=np.float64), np.asarray(decay, dtype=np.float64)
    if (boundary.shape != decay.shape or boundary.size == 0
            or not np.isfinite(boundary).all() or not np.isfinite(decay).all()
            or np.any(decay < 0) or np.any(decay > 1)
            or type(steps) is not int or steps < 1):
        raise ValueError("aligned finite boundary/decay in [0,1] and positive steps required")
    retention = decay ** steps
    return dict(
        current_lower=retention * boundary - (1 - retention),
        current_upper=retention * boundary + (1 - retention),
        structural_lower=np.minimum(boundary, -1),
        structural_upper=np.maximum(boundary, 1),
    )


def signed_violation(target, lower, upper):
    """Keep signed distance outside an interval; zero means inside."""
    target, lower, upper = (np.asarray(x, dtype=np.float64) for x in (target, lower, upper))
    if (target.shape != lower.shape or target.shape != upper.shape
            or not all(np.isfinite(x).all() for x in (target, lower, upper))
            or np.any(lower > upper)):
        raise ValueError("aligned finite targets and ordered bounds required")
    return target - np.clip(target, lower, upper)


def _write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def epoch_means(rows, split, seed, config):
    """Summarize complete committed epochs; reject duplicate or missing rows."""
    terms = ("loss", "prior_kl_h1", "prior_kl_h4", "prior_kl_h16", "prior_kl_h100")
    blocks = config["fit_blocks" if split == "fit" else "validation_blocks"]
    cases = {f"train_block_{block:03d}_familiar_{template}" for block in blocks for template in range(4)}
    expected = ({(case,) for case in cases} if split == "fit" else
                {(case, draw) for case in cases for draw in range(config["validation_draws"])})
    groups = {}
    for row in rows:
        groups.setdefault(int(row["epoch"]), []).append(row)
    if set(groups) != set(range(config["maximum_epochs"])):
        raise ValueError("complete registered epoch history required")
    result = []
    for epoch, group in sorted(groups.items()):
        keys = [(row["case_id"],) if split == "fit" else
                (row["case_id"], int(row["draw"])) for row in group]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("duplicate or missing committed epoch case/draw")
        values = np.array([[float(row[term]) for term in terms] for row in group])
        if not np.isfinite(values).all():
            raise ValueError("finite committed loss terms required")
        result.append(dict(seed=seed, split=split, epoch=epoch, rows=len(group),
                           **dict(zip(terms, values.mean(axis=0).tolist()))))
    return result


def diagnose(campaign, output):
    """Read accepted deliveries and all 72 cached episodes per registered seed.

    Stage only declared inputs to node-local scratch, verify their delivery
    hashes, and preserve case/origin/horizon/field/channel axes in NPZ outputs.
    No source simulation, posterior encoding, fitting, or sampling is performed.
    """
    if (not os.environ.get("SLURM_JOB_ID")
            or os.environ.get("SLURMD_NODENAME") not in {"cpu03", "cpu04"}
            or os.environ.get("SLURM_JOB_GPUS")):
        raise ValueError("CPU-only Slurm allocation on cpu03/cpu04 required")
    campaign, output = Path(campaign), Path(output)
    if output.exists():
        raise FileExistsError("retain the existing diagnostic output")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    summaries, histories, consumed = [], [], {}
    source_hash = sha256_file(Path(__file__).with_name("population_state.py"))
    for seed in (41, 42, 43):
        cell = campaign / f"seed_{seed}"
        delivery = json.loads((cell / "delivery.json").read_text())
        if (delivery.get("status") != "PASS" or delivery.get("training_seed") != seed
                or delivery.get("source_units") != 100000
                or not delivery.get("frozen_parameters_unchanged")):
            raise ValueError("accepted same-seed complete prior refit required")
        files = {row["path"]: row for row in delivery["files"]}
        bound = {}
        with tempfile.TemporaryDirectory(prefix=f"neuronsim-prior-mean-{seed}-", dir="/tmp") as scratch:
            scratch = Path(scratch)
            def stage(name):
                row = files[name]
                source = cell / name
                destination = scratch / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if destination.stat().st_size != row["bytes"] or sha256_file(destination) != row["sha256"]:
                    raise ValueError("delivery input changed: " + name)
                bound[name] = row["sha256"]
                return destination
            report = json.loads(stage("training/run.json").read_text())
            checkpoint = torch.load(stage("training/best.pt"), map_location="cpu", weights_only=True)
            binding, config = checkpoint["binding"], checkpoint["binding"]["config"]
            if (binding != report["binding"] or binding["seed"] != seed
                    or binding["model_implementation_sha256"] != source_hash
                    or config["observation_objective"] != "fixed_posterior_multistep_prior_kl"
                    or config["maximum_epochs"] != 80
                    or checkpoint["model_configuration"] != report["model_configuration"]
                    or report["best_checkpoint_sha256"] != bound["training/best.pt"]):
                raise ValueError("prior checkpoint, source, objective, or report mismatch")
            constructor = checkpoint["model_configuration"]
            if len(constructor["unit_ids"]) != 100000 or constructor["representation"] != "recurrent":
                raise ValueError("complete native recurrent model required")
            decay = checkpoint["model"]["decay_logit"].double().sigmoid().numpy()
            fields = constructor["field_names"]
            origins = [int(x / config["bin_ms"]) for x in config["prior_origins_ms"]]
            horizons = [int(x / config["bin_ms"]) for x in config["prior_horizons_ms"]]
            cases = [(split, f"train_block_{block:03d}_familiar_{template}")
                     for split, key in (("fit", "fit_blocks"), ("selection", "validation_blocks"))
                     for block in config[key] for template in range(4)]
            expected_names = {f"training/posterior_cache/{case}.pt" for _, case in cases}
            if {name for name in files if name.startswith("training/posterior_cache/")} != expected_names:
                raise ValueError("complete 56 fit and 16 selection caches required")
            arrays = {name: [] for name in ("boundary_mean", "target_mean", "current_lower", "current_upper",
                                            "structural_lower", "structural_upper",
                                            "current_signed_violation", "structural_signed_violation")}
            for split, case in cases:
                name = f"training/posterior_cache/{case}.pt"
                saved = torch.load(stage(name), map_location="cpu", weights_only=True)
                cache_binding = saved["binding"]
                if (cache_binding["case_id"] != case
                        or cache_binding["parent_checkpoint_sha256"] != checkpoint["initialization"]["parent_checkpoint_sha256"]
                        or cache_binding["model_implementation_sha256"] != source_hash
                        or cache_binding["bin_ms"] != config["bin_ms"]
                        or any(binding[key] != value for key, value in cache_binding["cohort"].items())):
                    raise ValueError("posterior cache parent, source, or clock mismatch")
                mean = saved["mean"].double().numpy()
                if (mean.shape != (1, 1200, len(fields), constructor["latent_channels"])
                        or not np.isfinite(mean).all()):
                    raise ValueError("complete finite six-second posterior mean required")
                per_case = {name: [] for name in arrays}
                for origin in origins:
                    by_origin = {name: [] for name in arrays}
                    for horizon in horizons:
                        boundary, target = mean[0, origin - 1], mean[0, origin + horizon - 1]
                        limits = mean_envelopes(boundary, decay, horizon)
                        values = dict(boundary_mean=boundary, target_mean=target, **limits)
                        row = dict(seed=seed, split=split, case_id=case,
                                   origin_ms=origin * config["bin_ms"], horizon_ms=horizon * config["bin_ms"],
                                   coordinates=target.size)
                        for mode in ("current", "structural"):
                            distance = signed_violation(target, limits[mode + "_lower"], limits[mode + "_upper"])
                            values[mode + "_signed_violation"] = distance
                            row.update({mode + "_outside_fraction": float(np.mean(distance != 0)),
                                        mode + "_mean_absolute_distance": float(np.abs(distance).mean()),
                                        mode + "_rms_distance": float(np.sqrt(np.square(distance).mean())),
                                        mode + "_maximum_absolute_distance": float(np.abs(distance).max())})
                        summaries.append(row)
                        for name, value in values.items():
                            by_origin[name].append(value)
                    for name in arrays:
                        per_case[name].append(by_origin[name])
                for name in arrays:
                    arrays[name].append(per_case[name])
                (scratch / f"training/posterior_cache/{case}.pt").unlink()
            np.savez_compressed(output / f"seed_{seed}_coordinates.npz",
                                **{name: np.asarray(value) for name, value in arrays.items()},
                                case_ids=np.asarray([case for _, case in cases]),
                                splits=np.asarray([split for split, _ in cases]),
                                field_names=np.asarray(fields), origins_ms=np.asarray(config["prior_origins_ms"]),
                                horizons_ms=np.asarray(config["prior_horizons_ms"]), decay=decay)
            for split, name in (("fit", "optimization.csv"), ("selection", "validation.csv")):
                with stage("training/" + name).open() as stream:
                    histories.extend(epoch_means(list(csv.DictReader(stream)), split, seed, config))
            consumed[str(seed)] = dict(cell=str(cell.resolve()),
                                      delivery_sha256=sha256_file(cell / "delivery.json"), files=bound)
    _write_csv(output / "mean_envelope_by_case.csv", summaries)
    _write_csv(output / "loss_by_epoch.csv", histories)
    names = ["mean_envelope_by_case.csv", "loss_by_epoch.csv",
             *[f"seed_{seed}_coordinates.npz" for seed in (41, 42, 43)]]
    result = dict(status="COMPLETE_CACHED_POSTERIOR_MEAN_DIAGNOSTIC",
                  training_seeds=[41, 42, 43], source_units=100000, cases_per_seed=72,
                  fit_cases_per_seed=56, selection_cases_per_seed=16,
                  case_origin_horizon_rows=len(summaries), epoch_rows=len(histories),
                  coordinate_axes=["case", "origin", "horizon", "source_field", "latent_channel"],
                  current_mean_bound="d^h*m0 +/- (1-d^h)",
                  structural_mean_bound="[min(m0,-1),max(m0,1)]",
                  conditioning="cached_future_observation_assisted_posterior_target",
                  interpretation="Necessary predictive-mean bounds, not reachable sets, sample/count bounds or a calibration test.",
                  covariance_qualification=False, convergence_qualification=False,
                  scientific_claim_qualified=False, source_simulation_performed=False,
                  encoding_performed=False, training_performed=False, sampling_performed=False,
                  consumed=consumed, model_source_sha256=source_hash,
                  diagnostic_source_sha256=sha256_file(Path(__file__)),
                  job_id=os.environ["SLURM_JOB_ID"],
                  artifacts={name: sha256_file(output / name) for name in names})
    freeze_json(output / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.campaign, args.output)
    print(json.dumps({key: result[key] for key in ("status", "case_origin_horizon_rows", "epoch_rows")}))


if __name__ == "__main__":
    main()
