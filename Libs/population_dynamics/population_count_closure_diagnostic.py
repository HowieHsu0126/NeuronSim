"""Diagnose frozen readout closure with future-posterior means and native clocks.

Future observations define the latent path. Only the factual boundary initializes
refractory clocks. Saved source-exposure reconstruction is reused without rerunning
it. These are development diagnostics, not free forecasts or benchmark results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .native_renewal_data import NativeRenewalEpisode
from .population_prior_kl_diagnostic import _checkpoint, _csv, _stage
from .population_posterior_field_diagnostic import _native_archive_files
from .posterior_count_closure import sample_prescribed_latents
from .provenance import freeze_json, sha256_file
from .trajectory_sample_metrics import unit_count_metrics
from .trajectory_scores import energy_score, variogram_score
from .trajectory_forecast_analysis import pair_sets

SEEDS = (41, 42, 43)


def diagnose(protocol_path, output, device_name="cpu"):
    protocol_path, output = Path(protocol_path), Path(output)
    protocol = json.loads(protocol_path.read_text())
    if (protocol.get("schema") != "posterior_native_count_closure_v1"
            or protocol.get("training_seeds") != list(SEEDS)
            or protocol.get("joint_samples") != 16
            or protocol.get("future_observations_accessed") is not True
            or protocol.get("future_source_exposure_used_for_sampling") is not False
            or protocol.get("scientific_claim_qualified") is not False):
        raise ValueError("registered future-posterior diagnostic protocol required")
    node, partition = os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Slurm compute allocation required")
    if device_name == "cpu":
        if os.environ.get("SLURM_JOB_GPUS") or (node, partition) not in (("cpu03", "gpu"), ("cpu04", "cpu")):
            raise ValueError("authorized CPU-only allocation required")
    elif device_name != "cuda" or not os.environ.get("SLURM_JOB_GPUS") or node not in {"cpu03","cpu04","cpu05","cpu06","gpu03"}:
        raise ValueError("authorized GPU allocation required")
    device = torch.device(device_name)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.use_deterministic_algorithms(True)
    campaign, cohort, reference = (Path(protocol[key]) for key in ("campaign", "cohort", "source_exposure_reconstruction"))
    proof = json.loads((reference / "analysis.json").read_text())
    if (json.loads((reference / "delivery.json").read_text())["exit_code"] != 0
            or proof["status"] != "COMPLETE_FROZEN_POSTERIOR_FIELD_RECONSTRUCTION"
            or proof["training_seeds"] != list(SEEDS)):
        raise ValueError("accepted saved source-exposure reconstruction required")
    source_receipt = json.loads((cohort / "delivery.json").read_text())
    if (source_receipt["status"] != "VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE"
            or source_receipt["completed_episodes"] != 72 or source_receipt["cortical_units"] != 100000
            or sha256_file(cohort / "delivery.json") != proof["source_delivery_sha256"]):
        raise ValueError("same complete native source archive required")
    source_files = _native_archive_files(source_receipt)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows, consumed_source, replays, artifacts = [], {}, [], {}
    code_hash = sha256_file(Path(__file__).with_name("population_state.py"))
    with tempfile.TemporaryDirectory(prefix="neuronsim-count-closure-", dir="/tmp") as temp:
        work = Path(temp)
        cells = {seed: _checkpoint(campaign/f"seed_{seed}", seed, work/f"seed_{seed}", code_hash, device) for seed in SEEDS}
        reference_arrays = {}
        for name in ["observed_fields.npz", *[f"seed_{seed}_reconstructed_fields.npz" for seed in SEEDS]]:
            local = work/name
            _stage(reference, {name: dict(bytes=(reference/name).stat().st_size, sha256=proof["artifacts"][name])}, name, local, {})
            with np.load(local, allow_pickle=False) as saved:
                reference_arrays[name] = {key:saved[key].copy() for key in saved.files}
        observed_reference = reference_arrays["observed_fields.npz"]
        scale_mean, sd, units, field_ids = (observed_reference[key] for key in ("mean_hz", "standard_deviation_hz", "unit_count", "field_ids"))
        if int(units.sum()) != 100000 or np.any(sd <= 0) or not np.isfinite(sd).all():
            raise ValueError("complete source fields and frozen fitting normalization required")
        pairs = pair_sets(field_ids)
        common = cells[41]["checkpoint"]["binding"]
        settings = None
        for seed, item in cells.items():
            binding = item["checkpoint"]["binding"]
            if proof["consumed"][str(seed)]["delivery_sha256"] != item["delivery_sha256"]:
                raise ValueError("saved reconstruction uses a different frozen readout")
            for key in ("dataset_sha256", "acquisition_design_sha256"):
                if binding[key] != source_receipt[key]:
                    raise ValueError("checkpoint and native source differ")
            if (binding["episode_bindings"] != common["episode_bindings"]
                    or binding["native_source_identity"] != common["native_source_identity"]
                    or binding["config"]["validation_blocks"] != [14,15,16,17]):
                raise ValueError("same registered selection cohort required")
            evaluation = json.loads(item["stage"]("forecasts/evaluation.json").read_text())["configuration"]
            if (evaluation["formal_evaluation"] or evaluation["horizons_ms"] != [20.,80.,500.]
                    or evaluation["origins_ms"] != [1000.,3000.,5000.] or evaluation["joint_samples"] != 16):
                raise ValueError("unchanged complete development measurement required")
            if settings is None: settings = evaluation
            elif settings != evaluation: raise ValueError("evaluation settings differ across seeds")
        for case_index in range(16):
            block, template = 14+case_index//4, case_index%4
            case = f"train_block_{block:03d}_familiar_{template}"
            native_work = work/"native"/case
            for name in [*[f"train/{case}/{part}" for part in ("report.json","spikes.npz","initial_parameters.npz")],
                         *[f"train_fields/{case}/{part}" for part in ("summary.json","spatial_fields.npz")]]:
                _stage(cohort, source_files, name, native_work/name, consumed_source)
            native = NativeRenewalEpisode(native_work/"train"/case, native_work/"train_fields"/case,
                expected_source=common["native_source_identity"], expected_unit_ids=np.asarray(cells[41]["model"].unit_ids),
                expected_unit_field_index=cells[41]["model"].unit_field_index.cpu().numpy())
            if (native.bin_ms != 5. or native.dt_ms != .1 or native.duration_ticks != 60000
                    or not np.array_equal(native.field_ids, field_ids)
                    or not np.array_equal(np.bincount(native.unit_field_index,minlength=len(field_ids)),units)):
                raise ValueError("native clock, field map or denominator changed")
            declared = common["episode_bindings"][case]
            if (native.binding["episode_report_sha256"] != declared["raw_report_sha256"]
                    or native.binding["field_summary_sha256"] != declared["field_summary_sha256"]):
                raise ValueError("source case identity differs from checkpoint")
            caches = {}
            for seed,item in cells.items():
                saved = torch.load(item["stage"](f"training/posterior_cache/{case}.pt"),map_location="cpu",weights_only=True)
                cb = saved["binding"]
                if (cb["case_id"] != case or cb["model_implementation_sha256"] != code_hash or cb["bin_ms"] != 5.
                        or cb["parent_checkpoint_sha256"] != item["checkpoint"]["initialization"]["parent_checkpoint_sha256"]
                        or any(item["checkpoint"]["binding"][key] != value for key,value in cb["cohort"].items())
                        or saved["mean"].shape != (1,1200,len(field_ids),item["model"].latent_channels)):
                    raise ValueError("future posterior cache identity or time axis differs")
                if not np.array_equal(native.minimum_interval_ticks,item["checkpoint"]["model_configuration"]["minimum_interval_ticks"]):
                    raise ValueError("source refractory intervals differ")
                caches[seed] = saved["mean"]
            for origin_index, origin in enumerate(settings["origins_ms"]):
                identity = case_index*3+origin_index
                if observed_reference["case_ids"][identity] != case or observed_reference["origins_ms"][identity] != origin:
                    raise ValueError("saved reconstruction origin ordering differs")
                start_bin, start_tick = int(origin/5), int(origin/.1)
                count, _ = native.window(start_tick,start_tick+5000)
                truth_raw = np.zeros((100,len(field_ids)),dtype=np.int64)
                np.add.at(truth_raw.T,native.unit_field_index,count.T)
                if not np.array_equal(truth_raw,observed_reference["observed_field_counts"][identity]):
                    raise ValueError("native all-unit projection differs from saved source truth")
                truth = (truth_raw/(units*.005)-scale_mean)/sd
                ready = torch.as_tensor(native.ready_at(start_tick)[None],device=device,dtype=torch.long)
                observed = torch.as_tensor(count,device=device)
                for seed,item in cells.items():
                    model = item["model"]
                    path = caches[seed][:,start_bin:start_bin+100].to(device).unsqueeze(0).expand(16,-1,-1,-1,-1)
                    observation_seed = settings["sampling_seed_base"]+seed*100000+case_index*100+origin_index*3+2
                    generator = torch.Generator(device=device).manual_seed(observation_seed)
                    tick = time.monotonic()
                    generated, terminal = sample_prescribed_latents(model,path,ready,generator)
                    if device.type == "cuda": torch.cuda.synchronize()
                    generation_seconds = time.monotonic()-tick
                    if identity == 0:
                        replay,end = sample_prescribed_latents(model,path,ready,torch.Generator(device=device).manual_seed(observation_seed))
                        if not torch.equal(generated,replay) or not torch.equal(terminal,end):
                            raise ValueError("native all-unit diagnostic replay differs")
                        replays.append(dict(seed=seed,case_id=case,origin_ms=origin,count_values=generated.numel(),exact=True))
                        del replay,end
                    samples = generated[:,0]
                    projected = torch.zeros((16,100,len(field_ids)),device=device,dtype=torch.int64)
                    projected.index_add_(-1,model.unit_field_index,samples)
                    raw = projected.cpu().numpy()
                    fields = (raw/(units*.005)-scale_mean)/sd
                    reference_counts = reference_arrays[f"seed_{seed}_reconstructed_fields.npz"]["mean_state_count"][identity]
                    source_mean = (reference_counts/(units*.005)-scale_mean)/sd
                    name = f"seed_{seed}_{case}_origin_{origin:g}.npz"
                    np.savez_compressed(output/name,sampled_field_counts=raw,observed_field_counts=truth_raw,
                        source_exposure_mean_counts=reference_counts,terminal_ready_min=int(terminal.min()),terminal_ready_max=int(terminal.max()))
                    artifacts[name] = sha256_file(output/name)
                    for ms in settings["horizons_ms"]:
                        bins = int(ms/5)
                        row = dict(seed=seed,case_id=case,block=block,template_id=f"familiar_{template}",origin_ms=origin,horizon_ms=ms,
                            method="future_posterior_mean_endogenous_clock",source_units=100000,
                            energy_score=energy_score(fields[:,:bins],truth[:bins]),
                            field_mean_mse_corrected=float((np.square(fields[:,:bins].mean(0)-truth[:bins])-fields[:,:bins].var(0,ddof=1)/16).mean()),
                            source_exposure_field_mean_mse=float(np.square(source_mean[:bins]-truth[:bins]).mean()),
                            source_exposure_spikes_per_unit=float(reference_counts[:bins].sum()/100000),
                            full_horizon_generation_seconds=generation_seconds,
                            **unit_count_metrics(samples[:,:bins],observed[:bins],native.minimum_interval_ticks,native.bin_ticks,settings["refractory_support_windows_bins"]))
                        for label,pair in pairs.items():
                            row["variogram_"+label] = variogram_score(fields[:,:bins],truth[:bins],pair[:,0],pair[:,1])
                        for lag in settings["temporal_lags_bins"]:
                            row[f"variogram_temporal_lag_{lag}"] = (variogram_score(fields[:,:bins],truth[:bins],np.arange(len(field_ids)),np.arange(len(field_ids)),lag) if lag < bins else None)
                        rows.append(row)
                    del generated,samples,projected,path,terminal
                print(json.dumps(dict(case_id=case,origin_ms=origin,completed_source_origins=identity+1,condition_rows=len(rows),elapsed_seconds=time.monotonic()-started)),flush=True)
                (output/"progress.json").write_text(json.dumps(dict(completed_source_origins=identity+1,condition_rows=len(rows),elapsed_seconds=time.monotonic()-started))+"\n")
        bindings = {str(seed):dict(delivery_sha256=item["delivery_sha256"],files=item["consumed"]) for seed,item in cells.items()}
    if len(rows) != 432 or len(replays) != 3:
        raise ValueError("all48 origins,three seeds,three horizons and full-unit replays required")
    _csv(output/"scores.csv",rows)
    metrics = [key for key in rows[0] if key not in ("seed","case_id","block","template_id","origin_ms","horizon_ms","method","source_units")]
    summary = []
    for seed in SEEDS:
        for ms in settings["horizons_ms"]:
            group = [row for row in rows if row["seed"]==seed and row["horizon_ms"]==ms]
            summary.append(dict(seed=seed,horizon_ms=ms,source_cases=16,origins=48,**{key:float(np.mean([row[key] for row in group])) if group[0][key] is not None else None for key in metrics}))
    _csv(output/"summary.csv",summary)
    artifacts.update({name:sha256_file(output/name) for name in ("scores.csv","summary.csv")})
    result = dict(status="COMPLETE_POSTERIOR_NATIVE_COUNT_CLOSURE_DIAGNOSTIC",training_seeds=list(SEEDS),source_units=100000,
        source_episodes=16,source_origins=48,model_origins=144,condition_rows=len(rows),joint_samples=16,
        future_observations_accessed=True,future_source_exposure_used_for_sampling=False,
        source_exposure_reconstruction_reused=True,posterior_encoding_performed=False,training_performed=False,
        free_predictive_capacity_qualified=False,scientific_claim_qualified=False,full_unit_replays=replays,
        interpretation="Future-observation posterior means with native endogenous clocks; no prior forecast or posterior predictive covariance qualification.",
        protocol=str(protocol_path),protocol_sha256=sha256_file(protocol_path),source_reconstruction=str(reference),
        source_reconstruction_analysis_sha256=sha256_file(reference/"analysis.json"),source_delivery_sha256=sha256_file(cohort/"delivery.json"),
        consumed=bindings,consumed_source=consumed_source,artifacts=artifacts,
        code_sha256={name:sha256_file(Path(__file__).with_name(name)) for name in ("population_state.py","refractory_observation.py","posterior_count_closure.py","trajectory_sample_metrics.py","trajectory_scores.py",Path(__file__).name)},
        job_id=os.environ["SLURM_JOB_ID"],node=node,device=str(device),elapsed_seconds=time.monotonic()-started)
    freeze_json(output/"analysis.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    result=diagnose(args.protocol,args.output,args.device)
    print(json.dumps({key:result[key] for key in ("status","condition_rows","elapsed_seconds")}))


if __name__ == "__main__":
    main()
