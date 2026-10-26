"""Measure finite-particle likelihood/gradient variation on one fixed fitting case.

The unchanged joint event-path objective is evaluated at K=8 and K=32 with
nested normal draws. No optimizer step, parameter selection, or generation of
unit observations is performed. The single case diagnoses this estimator;
it does not measure biological generalization or generative fidelity.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch

from .native_renewal_data import NativeRenewalEpisode
from .population_observation_prior import prior_observation_mixture_loss
from .population_prior_objective import PRIOR_PARAMETER_NAMES, configure_prior_parameters
from .population_state import ControlledPopulationState
from .provenance import freeze_json, sha256_file
from .renewal_trajectory_training import variational_seed

CASE = "train_block_000_familiar_0"
BUDGETS = (8, 32)
REPLICATES = (0, 1, 2)


def nested_objective(model, inputs, origins, horizons, initial, dynamics, *, particles, maximum_particles,
                     objective=None, reconstructs_paths=False):
    """Preserve iid marginals; both budgets use the same first K normal draws.

    Each stream consumes [maximum_particles, case, field, channel] at each
    invocation. This is a diagnostic draw layout, not replay of training RNG.
    """
    if type(particles) is not int or type(maximum_particles) is not int or not 1 <= particles <= maximum_particles:
        raise ValueError("positive ordered diagnostic particle budgets required")
    if type(reconstructs_paths) is not bool:
        raise ValueError("boolean path-reconstruction contract required")
    original = torch.randn
    streams = {id(initial): initial, id(dynamics): dynamics}
    starts = {key: value.get_state().clone() for key, value in streams.items()}
    replays = {}
    counts = {id(initial): 0, id(dynamics): 0}
    expected_tail = (inputs[0].shape[0] * len(origins), len(model.field_names), model.latent_channels)
    def nested(shape, *, generator, **kwargs):
        key = id(generator)
        if key not in counts and reconstructs_paths:
            matches = [owner for owner, state in starts.items()
                       if torch.equal(generator.get_state(), state)]
            if len(matches) != 1 or matches[0] in replays:
                raise ValueError("unexpected reconstructed random stream")
            replays[matches[0]] = key
            streams[key] = generator
            counts[key] = 0
        if tuple(shape) != (particles, *expected_tail) or key not in counts:
            raise ValueError("unexpected objective normal draw or random stream")
        counts[key] += 1
        return original((maximum_particles, *expected_tail), generator=generator, **kwargs)[:particles]
    loss_function = prior_observation_mixture_loss if objective is None else objective
    with patch.object(torch, "randn", new=nested):
        terms = loss_function(model, *inputs, origins, horizons, initial, dynamics,
                              prediction_particles=particles)
    expected = {id(initial): 1, id(dynamics): max(horizons)}
    if reconstructs_paths:
        if set(replays) != set(expected):
            raise ValueError("both native random streams must be reconstructed exactly once")
        for owner, replay in replays.items():
            expected[replay] = expected[owner]
            if not torch.equal(streams[owner].get_state(), streams[replay].get_state()):
                raise ValueError("reconstructed native RNG advancement differs")
    if counts != expected:
        raise ValueError("complete uninterrupted initial/process draw layout required")
    return terms


def gradient_probe(model, inputs, origins, horizons, initial_seed, dynamics_seed, particles, maximum_particles):
    parameters = configure_prior_parameters(model)
    initial = torch.Generator(device="cpu").manual_seed(initial_seed)
    dynamics = torch.Generator(device="cpu").manual_seed(dynamics_seed)
    model.zero_grad(set_to_none=True)
    started = time.perf_counter()
    terms = nested_objective(model, inputs, origins, horizons, initial, dynamics,
                             particles=particles, maximum_particles=maximum_particles)
    gradients = torch.autograd.grad(terms["loss"], parameters)
    if any(not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("nonfinite finite-particle likelihood gradient")
    vector = torch.cat([g.detach().double().reshape(-1) for g in gradients]).numpy()
    row = dict(particles=particles, **{key:float(value.detach()) for key,value in terms.items()},
               gradient_norm=float(np.linalg.norm(vector)), seconds=time.perf_counter()-started,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    row.update({name + "_gradient_norm":float(g.detach().double().norm()) for name,g in zip(PRIOR_PARAMETER_NAMES, gradients)})
    return row, vector


def _cosine(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm else None


def _stage(root, name, row, scratch, consumed):
    target = scratch / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / name, target)
    size = row.get("bytes", row.get("destination_bytes"))
    if target.stat().st_size != size or sha256_file(target) != row["sha256"]:
        raise ValueError("bound diagnostic input changed: " + name)
    consumed[name] = row["sha256"]
    return target


def diagnose(campaign, cohort, output):
    if (not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS")
            or (os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION"))
            not in (("cpu03", "gpu"), ("cpu04", "cpu"))):
        raise ValueError("CPU-only admitted Slurm allocation required")
    campaign, cohort, output = map(Path, (campaign, cohort, output))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    source_hash = sha256_file(Path(__file__).with_name("population_state.py"))
    objective_hash = sha256_file(Path(__file__).with_name("population_observation_prior.py"))
    receipt = json.loads((cohort / "delivery.json").read_text())
    if receipt.get("status") != "VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE":
        raise ValueError("accepted native source cohort required")
    rows = {r["path"]: r for r in receipt["files"]}
    if len(rows) != len(receipt["files"]):
        raise ValueError("duplicate source delivery paths")
    observations, vectors, comparisons, consumed = [], [], [], {}
    with tempfile.TemporaryDirectory(prefix="neuronsim-particle-diagnostic-") as scratch_name:
        scratch = Path(scratch_name)
        source_names = [f"train/{CASE}/{name}" for name in ("report.json", "spikes.npz", "initial_parameters.npz")]
        source_names += [f"train_fields/{CASE}/{name}" for name in ("summary.json", "spatial_fields.npz")]
        source_consumed = {}
        for name in source_names:
            _stage(cohort, name, rows[name], scratch / "source", source_consumed)
        for seed in (41, 42, 43):
            cell = campaign / f"seed_{seed}"
            proof = json.loads((cell / "delivery.json").read_text())
            if (proof.get("status") != "PASS" or proof.get("exit_code") != 0
                    or proof.get("training_seed") != seed or proof.get("frozen_parameters_unchanged") is not True):
                raise ValueError("accepted same-seed fixed-encoder/readout native model required")
            index = {r["path"]:r for r in proof["files"]}
            if len(index) != len(proof["files"]):
                raise ValueError("duplicate training delivery paths")
            bound = {}
            def stage(name):
                return _stage(cell, name, index[name], scratch / f"seed_{seed}", bound)
            report = json.loads(stage("training/run.json").read_text())
            checkpoint = torch.load(stage("training/best.pt"), map_location="cpu", weights_only=True)
            binding = checkpoint["binding"]
            config = binding["config"]
            if (binding != report["binding"] or binding["seed"] != seed
                    or binding["model_implementation_sha256"] != source_hash
                    or report["trainer_sources"]["population_observation_prior.py"] != objective_hash
                    or config["observation_objective"] != "fixed_readout_native_event_path_mixture_nll"
                    or config["prediction_particles"] != 8 or config["training_seeds"] != [41,42,43]
                    or 0 not in config["fit_blocks"] or 0 in config["validation_blocks"]
                    or report["completed_epochs"] != 80
                    or report["best_checkpoint_sha256"] != bound["training/best.pt"]
                    or checkpoint["model_configuration"] != report["model_configuration"]
                    or checkpoint["next_epoch"] != report["best_epoch"] + 1):
                raise ValueError("selected native checkpoint, source or fitting identity differs")
            constructor = checkpoint["model_configuration"]
            if len(constructor["unit_ids"]) != 100000:
                raise ValueError("complete 100000-unit likelihood required")
            model = ControlledPopulationState(**constructor)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            saved = torch.load(stage(f"training/posterior_cache/{CASE}.pt"), map_location="cpu", weights_only=True)
            cb = saved["binding"]
            if (cb["case_id"] != CASE or cb["parent_checkpoint_sha256"] != checkpoint["initialization"]["parent_checkpoint_sha256"]
                    or cb["model_implementation_sha256"] != source_hash or cb["bin_ms"] != config["bin_ms"]
                    or any(binding[key] != value for key,value in cb["cohort"].items())):
                raise ValueError("posterior cache lineage differs")
            expected = binding["episode_bindings"][CASE]
            if (source_consumed[f"train/{CASE}/report.json"] != expected["raw_report_sha256"]
                    or source_consumed[f"train_fields/{CASE}/summary.json"] != expected["field_summary_sha256"]):
                raise ValueError("likelihood labels and fitting posterior have different sources")
            episode = NativeRenewalEpisode(scratch / "source/train" / CASE, scratch / "source/train_fields" / CASE,
                expected_source=binding["native_source_identity"], expected_unit_ids=np.asarray(constructor["unit_ids"]),
                expected_unit_field_index=np.asarray(constructor["unit_field_index"]))
            if (episode.duration_ticks != 60000 or episode.bin_ticks != 50
                    or list(map(str,episode.field_ids)) != constructor["field_names"]
                    or not np.array_equal(episode.minimum_interval_ticks, constructor["minimum_interval_ticks"])):
                raise ValueError("native clock, field order or renewal support differs")
            origins = [int(x/config["bin_ms"]) for x in config["prior_origins_ms"]]
            horizons = [int(x/config["bin_ms"]) for x in config["prior_horizons_ms"]]
            labels, exposure = [], []
            for origin in origins:
                window = episode.renewal_window(origin * episode.bin_ticks, (origin + max(horizons)) * episode.bin_ticks)
                if not torch.equal(torch.from_numpy(window.stimulus), saved["controls"][0,origin:origin+max(horizons)]):
                    raise ValueError("native source controls differ from posterior cache")
                labels.append(window.counts)
                exposure.append(window.eligible_ticks)
            inputs = (saved["mean"], saved["variance"], saved["controls"],
                      torch.from_numpy(np.stack(labels)[None]), torch.from_numpy(np.stack(exposure)[None]))
            del labels, exposure, window, episode
            seed_rows, seed_vectors = [], {}
            with (output / f"seed_{seed}_draws.csv").open("x", newline="") as stream:
                writer = None
                for replicate in REPLICATES:
                    initial_seed = variational_seed(config["training_variational_seed_base"],seed,"particle_diagnostic_initial",CASE,replicate)
                    dynamics_seed = variational_seed(config["training_variational_seed_base"],seed,"particle_diagnostic_dynamics",CASE,replicate)
                    for budget in BUDGETS:
                        row, vector = gradient_probe(model, inputs, origins, horizons, initial_seed, dynamics_seed, budget, max(BUDGETS))
                        row = dict(seed=seed, case_id=CASE, replicate=replicate, initial_seed=initial_seed, dynamics_seed=dynamics_seed, **row)
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row); stream.flush()
                        observations.append(row); seed_rows.append(row)
                        seed_vectors[budget,replicate] = vector
                        vectors.append(vector)
                        print(json.dumps(row),flush=True)
            for budget in BUDGETS:
                selected = [r for r in seed_rows if r["particles"] == budget]
                g = np.stack([seed_vectors[budget,r] for r in REPLICATES])
                comparisons.append(dict(seed=seed, particles=budget,
                    loss_mean=float(np.mean([r["loss"] for r in selected])), loss_sd=float(np.std([r["loss"] for r in selected],ddof=1)),
                    pairwise_gradient_cosines=[_cosine(g[i],g[j]) for i in REPLICATES for j in REPLICATES if i<j],
                    centered_gradient_rms=float(np.sqrt(np.mean(np.sum((g-g.mean(0))**2,axis=1)))),
                    mean_gradient_norm=float(np.linalg.norm(g.mean(0))),
                    paired_8_vs_32_gradient_cosines=[_cosine(seed_vectors[8,r],seed_vectors[32,r]) for r in REPLICATES]))
            consumed[str(seed)] = dict(delivery_sha256=sha256_file(cell / "delivery.json"),files=bound)
            del model, inputs, saved, checkpoint, seed_vectors
        np.savez_compressed(output / "gradients.npz", gradients=np.stack(vectors),
                            seed=np.asarray([r["seed"] for r in observations]),
                            particles=np.asarray([r["particles"] for r in observations]),
                            replicate=np.asarray([r["replicate"] for r in observations]),
                            parameter_names=np.asarray(PRIOR_PARAMETER_NAMES))
    result = dict(status="COMPLETE_NATIVE_PARTICLE_GRADIENT_DIAGNOSTIC",case_id=CASE,training_seeds=[41,42,43],
        particle_budgets=list(BUDGETS),replicate_indices=list(REPLICATES),source_units=100000,
        origins_ms=config["prior_origins_ms"],horizons_ms=config["prior_horizons_ms"],objective_evaluations=len(observations),
        fixed_selected_checkpoints=True,nested_draw_layout="per-step maximum K=32; first K paths share initial and process normals",
        independent_streams="SHA256 namespace through existing variational_seed; replicate indices are not training seeds",
        parameters_updated=False,validation_outcomes_accessed=False,posterior_targets_used=False,
        generated_unit_samples=False,scientific_claim_qualified=False,
        limitation="One fixed fitting episode; three stream replicates per seed. Describes estimator variation, not biological significance or joint fidelity.",
        comparisons=comparisons,consumed=consumed,source_consumed=source_consumed,
        source_delivery_sha256=sha256_file(cohort/"delivery.json"),model_source_sha256=source_hash,
        objective_source_sha256=objective_hash,diagnostic_source_sha256=sha256_file(Path(__file__)),
        job_id=os.environ["SLURM_JOB_ID"],artifacts={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
    freeze_json(output/"analysis.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,required=True)
    parser.add_argument("--cohort",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=diagnose(args.campaign,args.cohort,args.output)
    print(json.dumps({key:result[key] for key in ("status","objective_evaluations")}))


if __name__ == "__main__":
    main()
