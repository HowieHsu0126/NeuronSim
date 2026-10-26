"""Run the three-plan posterior diagnostic from authenticated saved evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import torch


NAMES = ("omit", "first_future_pulse", "original_future_pulses")
SEEDS = (41, 42, 43)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def stage(source, target, row):
    expected_bytes = row.get("bytes", row.get("destination_bytes"))
    if source.stat().st_size != expected_bytes:
        raise ValueError("declared source size differs: " + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if digest(target) != row["sha256"]:
        raise ValueError("declared source hash differs: " + str(source))


def save_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(repo, output, scratch_root):
    if os.environ.get("SLURMD_NODENAME") not in ("cpu03", "cpu04"):
        raise RuntimeError("run this diagnostic on an allowed CPU compute node")
    if shutil.disk_usage(scratch_root).free < 20 * 1024 ** 3:
        raise RuntimeError("compute scratch below unchanged 20GiB admission boundary")
    if output.exists():
        raise FileExistsError("preserve existing diagnostic output: " + str(output))
    output.mkdir(parents=True)
    execution = dict(status="RUNNING", training_performed=False,
        mechanistic_simulation_performed=False, forecast_generation_performed=False)
    started = time.monotonic()
    root = repo / "Output/neural_population_dynamics"
    source = root / "model/native_pulse_execution/run_607327"
    search = root / "model/intervention_development/run_606504"
    campaign = root / "model/trajectory_development/run_606139"
    boundaries = root / "model/boundary_closure/run_606387"
    cohort = root / "mechanistic_reference/spatial_connectivity/stimulus_trajectory_cohort/philox_development"
    delivery = json.loads((source / "delivery.json").read_text())
    search_delivery = json.loads((search / "delivery.json").read_text())
    if (delivery["status"] != "EXECUTED_DEVELOPMENT" or delivery["exit_code"] != 0
            or delivery["training_seeds"] != list(SEEDS)
            or delivery["native_initial_noise_blocks"] != 1
            or tuple(row["candidate"] for row in delivery["candidates"]) != NAMES
            or digest(search / "delivery.json") != delivery["source_search_delivery_sha256"]):
        raise ValueError("accepted original three-plan evidence required")
    source_index = {row["path"]: row for row in delivery["files"]}
    code_index = {row["path"]: row for row in search_delivery["files"]}
    input_rows, checkpoint_rows, summary, per_bin, paths = [], [], [], [], {}
    try:
        with tempfile.TemporaryDirectory(prefix="neuronsim-posterior-diagnostic-", dir=scratch_root) as scratch:
            work = Path(scratch)
            for name, row in code_index.items():
                if name.startswith("Libs/population_dynamics/") and name.endswith(".py"):
                    stage(search / name, work / name, row)
                    input_rows.append(dict(source=str(search / name), sha256=row["sha256"]))
            helper = repo / "Libs/population_dynamics/population_posterior_diagnostic.py"
            shutil.copy2(helper, work / "Libs/population_dynamics/population_posterior_diagnostic.py")
            shutil.copy2(helper, output / "population_posterior_diagnostic.py")
            shutil.copy2(Path(__file__), output / "diagnose_population_posterior.py")
            sys.path.insert(0, str(work / "Libs"))
            from population_dynamics.native_renewal_data import NativeRenewalEpisode
            from population_dynamics.population_state import ControlledPopulationState
            from population_dynamics.population_posterior_diagnostic import diagnose_native_plan, diagnostic_boundary_vector
            torch.set_num_threads(1)
            rng_before = torch.get_rng_state().clone()
            native_episodes = {}
            for name in NAMES:
                for suffix in ("native/report.json", "native/spikes.npz", "native/initial_parameters.npz",
                               "fields/summary.json", "fields/spatial_fields.npz"):
                    relative = "candidates/" + name + "/" + suffix
                    stage(source / relative, work / relative, source_index[relative])
                    input_rows.append(dict(source=str(source / relative), sha256=source_index[relative]["sha256"]))
                native_episodes[name] = NativeRenewalEpisode(work / "candidates" / name / "native",
                                                            work / "candidates" / name / "fields")
            cohort_proof = json.loads((cohort / "delivery.json").read_text())
            cohort_index = {row["path"]: row for row in cohort_proof["files"]}
            case = "train_block_014_familiar_0"
            for prefix, files in (("train", ("report.json", "spikes.npz", "initial_parameters.npz")),
                                  ("train_fields", ("summary.json", "spatial_fields.npz"))):
                for name in files:
                    relative = prefix + "/" + case + "/" + name
                    stage(cohort / relative, work / "original" / prefix / name, cohort_index[relative])
                    input_rows.append(dict(source=str(cohort / relative), sha256=cohort_index[relative]["sha256"]))
            original = NativeRenewalEpisode(work / "original/train", work / "original/train_fields")
            count, controls = original.window(0, 10000)
            for native in native_episodes.values():
                actual_count, actual_control = native.window(0, 10000)
                if (not np.array_equal(count, actual_count) or not np.array_equal(controls, actual_control)
                        or not np.array_equal(original.ready_at(10000), native.ready_at(10000))):
                    raise ValueError("native plan differs from the original observed causal prefix")
            for seed in SEEDS:
                report = json.loads((search / f"results/seed_{seed}/analysis.json").read_text())
                checkpoint = campaign / f"seed_{seed}/training/best.pt"
                saved = work / f"seed_{seed}.pt"
                shutil.copy2(checkpoint, saved)
                if digest(saved) != report["checkpoint_sha256"]:
                    raise ValueError("accepted checkpoint changed")
                payload = torch.load(saved, map_location="cpu", weights_only=True)
                if (payload["binding"] != report["training_binding"]
                        or payload["binding"]["model_implementation_sha256"] !=
                        digest(work / "Libs/population_dynamics/population_state.py")):
                    raise ValueError("checkpoint and archived model implementation differ")
                model = ControlledPopulationState(**payload["model_configuration"]).eval()
                model.load_state_dict(payload["model"], strict=True)
                model.requires_grad_(False)
                witness = json.loads((boundaries / f"analysis/seed_{seed}/witness.json").read_text())
                boundary_source = boundaries / f"analysis/seed_{seed}/boundaries.npz"
                boundary_file = work / f"boundary_{seed}.npz"
                shutil.copy2(boundary_source, boundary_file)
                if (digest(boundary_file) != witness["boundary_artifact_sha256"]
                        or witness["checkpoint_sha256"] != report["checkpoint_sha256"]):
                    raise ValueError("saved boundary and checkpoint binding differ")
                with np.load(boundary_file, allow_pickle=False) as archive:
                    selected = ((archive["blocks"] == 14) & (archive["cells"] == 0)
                                & (archive["origins_ms"] == 1000))
                    if selected.sum() != 1:
                        raise ValueError("one saved source-block boundary required")
                    saved_boundary = archive["boundaries"][selected].copy()
                boundary_file.unlink()
                checkpoint_rows.append(dict(training_seed=seed, checkpoint_sha256=report["checkpoint_sha256"],
                    boundary_sha256=witness["boundary_artifact_sha256"], model_source_sha256=payload["binding"]["model_implementation_sha256"]))
                for name in NAMES:
                    values = diagnose_native_plan(model, native_episodes[name], 10000, 15000)
                    boundary = diagnostic_boundary_vector(values, model)
                    if not np.array_equal(boundary, saved_boundary):
                        raise ValueError("reconstructed posterior and complete readiness differ from saved boundary")
                    arrays = {k: v for k, v in values.items() if not k.startswith("boundary_")}
                    np.savez_compressed(output / f"seed_{seed}_{name}.npz", **arrays)
                    paths[seed, name] = values
                    scalar_names = [k for k, v in values.items() if v.ndim == 1]
                    for bin_index in range(len(values["bin_start_tick"])):
                        per_bin.append(dict(training_seed=seed, candidate=name,
                            **{k: v[bin_index].item() for k, v in values.items() if k in scalar_names},
                            pulse_count=float(values["controls"][bin_index].sum())))
                    summary.append(dict(training_seed=seed, candidate=name,
                        observed_total_spikes=int(values["observed_total_spikes"].sum()),
                        posterior_event_nll=float(values["posterior_event_nll"].sum()),
                        one_step_event_nll=float(values["one_step_event_nll"].sum()),
                        one_step_minus_posterior_event_nll=float((values["one_step_event_nll"] - values["posterior_event_nll"]).sum()),
                        posterior_observed_exposure_probability_sum=float(values["posterior_observed_exposure_probability_sum"].sum()),
                        one_step_observed_exposure_probability_sum=float(values["one_step_observed_exposure_probability_sum"].sum()),
                        latent_transition_residual_rms=float(np.sqrt(np.mean(values["latent_transition_residual_rms"] ** 2))),
                        delta_standardized_residual_rms=float(np.sqrt(np.mean(values["delta_standardized_residual_rms"] ** 2)))))
                    # Bound memory: retain only arrays needed for schedule contrasts.
                    paths[seed, name] = {k: v for k, v in values.items() if k in
                        ("observed_total_spikes", "posterior_observed_exposure_probability_sum",
                         "one_step_observed_exposure_probability_sum", "posterior_field_point_rate_hz",
                         "one_step_field_point_rate_hz")}
                del model, payload
            if not torch.equal(rng_before, torch.get_rng_state()):
                raise ValueError("checkpoint diagnostic consumed the global random stream")
        contrasts = []
        for seed in SEEDS:
            for left, right in combinations(NAMES, 2):
                a, b = paths[seed, left], paths[seed, right]
                contrasts.append(dict(training_seed=seed, left=left, right=right,
                    observed_spike_difference=float((b["observed_total_spikes"] - a["observed_total_spikes"]).sum()),
                    posterior_exposure_probability_sum_difference=float((b["posterior_observed_exposure_probability_sum"] - a["posterior_observed_exposure_probability_sum"]).sum()),
                    one_step_exposure_probability_sum_difference=float((b["one_step_observed_exposure_probability_sum"] - a["one_step_observed_exposure_probability_sum"]).sum()),
                    posterior_field_point_rate_contrast_rms_hz=float(np.sqrt(np.mean((b["posterior_field_point_rate_hz"] - a["posterior_field_point_rate_hz"]) ** 2))),
                    one_step_field_point_rate_contrast_rms_hz=float(np.sqrt(np.mean((b["one_step_field_point_rate_hz"] - a["one_step_field_point_rate_hz"]) ** 2)))))
        save_csv(output / "summary.csv", summary)
        save_csv(output / "per_bin.csv", per_bin)
        save_csv(output / "schedule_contrasts.csv", contrasts)
        result = dict(status="COMPLETED_CHECKPOINT_POSTERIOR_DIAGNOSTIC", training_seeds=list(SEEDS),
            native_plans=list(NAMES), native_initial_noise_blocks=1, source_units=100000, source_fields=128,
            origin_tick=10000, stop_tick=15000, bin_ticks=50, tick_seconds=.0001,
            posterior_includes_current_observation=True, one_step_is_conditional_mean_from_previous_posterior=True,
            prior_variance_method="first-order diagonal J diag(previous variance) J^T plus process variance",
            residual_standardization_is_exact_marginal=False,
            exposure_probability_sum_is_generative_expected_count=False,
            posterior_and_one_step_are_point_latent_likelihoods=True,
            nll_normalization="native sum over all units; per-unit-bin NLL divides each bin by source_units",
            scientific_claim_qualified=False, causal_root_cause_established=False,
            preserved_full_boundary_exact=True, global_rng_preserved=True,
            checkpoints=checkpoint_rows, inputs=input_rows,
            source_delivery_sha256=digest(source / "delivery.json"),
            helper_sha256=digest(output / "population_posterior_diagnostic.py"),
            summary_rows=len(summary), per_bin_rows=len(per_bin), contrast_rows=len(contrasts),
            allocation=os.environ["SLURM_JOB_ID"], node=os.environ["SLURMD_NODENAME"],
            seconds=time.monotonic() - started,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            training_performed=False, mechanistic_simulation_performed=False, forecast_generation_performed=False)
        (output / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        execution.update(status="COMPLETED", recorded_utc=datetime.now(timezone.utc).isoformat())
        print(json.dumps({k: v for k, v in result.items() if k not in ("inputs", "checkpoints")}))
    except Exception as error:
        execution.update(status="FAILED", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        (output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.repo.resolve(), arguments.output.resolve(), arguments.scratch_root.resolve())
