"""Reconstruct selection fields from frozen posterior caches and native exposure.

Future observations supply both posterior states and eligible native ticks.
These are deterministic response diagnostics, not free-running forecasts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .native_renewal_data import NativeRenewalEpisode
from .population_prior_kl_diagnostic import _checkpoint, _csv, _stage
from .posterior_field_reconstruction import posterior_field_reconstruction, response_covariance
from .provenance import freeze_json, sha256_file
from .trajectory_forecast_analysis import pair_sets


KEYS = ("mean_state_count", "posterior_expected_count_lower", "posterior_expected_count_upper", "posterior_integrated_intensity")
SEEDS = (41, 42, 43)


def _native_archive_files(receipt):
    """Bind native archive destinations to the shared staging size contract."""
    files = {}
    for row in receipt["files"]:
        source_size, archive_size = row["source_bytes"], row["destination_bytes"]
        if (type(source_size) is not int or type(archive_size) is not int
                or archive_size < 0 or source_size != archive_size):
            raise ValueError("native archive source and destination sizes differ or are invalid")
        name = row["path"]
        if name in files:
            raise ValueError("duplicate native source file bindings")
        files[name] = {**row, "bytes": archive_size}
    return files


def diagnose(campaign, cohort, output, prior_diagnostic, device="cuda"):
    node, partition = os.environ.get("SLURMD_NODENAME"), os.environ.get("SLURM_JOB_PARTITION")
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Slurm allocation required")
    if device == "cpu":
        if os.environ.get("SLURM_JOB_GPUS") or (node, partition) not in (("cpu03", "gpu"), ("cpu04", "cpu")):
            raise ValueError("CPU-only allocation on an admitted partition/node pair required")
    elif device == "cuda":
        if not os.environ.get("SLURM_JOB_GPUS") or node not in {"cpu03", "cpu04", "cpu05", "cpu06", "gpu03"} or not torch.cuda.is_available():
            raise ValueError("allocated GPU on an authorized compute node required")
    else:
        raise ValueError("CPU or CUDA diagnostic device required")
    campaign, cohort, output = map(Path, (campaign, cohort, output))
    output.mkdir(parents=True, exist_ok=False)
    prior_diagnostic = Path(prior_diagnostic)
    prior_evidence = json.loads(prior_diagnostic.read_text())
    if prior_evidence.get("status") != "COMPLETE_PAIRED_CONDITIONAL_KL_DIAGNOSTIC" or prior_evidence.get("training_seeds") != list(SEEDS):
        raise ValueError("accepted paired frozen-parameter/cache diagnostic required")
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_hash = sha256_file(Path(__file__).with_name("population_state.py"))
    receipt = json.loads((cohort / "delivery.json").read_text())
    if (receipt.get("status") != "VERIFIED_PHILOX_DEVELOPMENT_COHORT_ARCHIVE"
            or receipt.get("completed_episodes") != 72 or receipt.get("cortical_units") != 100000):
        raise ValueError("accepted complete native development cohort required")
    source_files = _native_archive_files(receipt)
    rows, cov_rows, origin_ids, observed, consumed_source = [], [], [], [], {}
    reconstructions = {seed: {key: [] for key in KEYS} for seed in SEEDS}
    with tempfile.TemporaryDirectory(prefix="neuronsim-posterior-fields-", dir="/tmp") as temporary:
        work = Path(temporary)
        cells = {seed: _checkpoint(campaign / f"seed_{seed}", seed, work / f"seed_{seed}", source_hash, device) for seed in SEEDS}
        reference = cells[41]
        common_binding = reference["checkpoint"]["binding"]
        settings = None
        scale = None
        for seed, item in cells.items():
            binding, cfg = item["checkpoint"]["binding"], item["checkpoint"]["binding"]["config"]
            if (prior_evidence["consumed"][f"{seed}/prior80"]["delivery_sha256"] != item["delivery_sha256"]
                    or item["model"].unit_ids != reference["model"].unit_ids
                    or item["model"].field_names != reference["model"].field_names
                    or not torch.equal(item["model"].unit_field_index, reference["model"].unit_field_index)
                    or item["checkpoint"]["model_configuration"]["minimum_interval_ticks"] != reference["checkpoint"]["model_configuration"]["minimum_interval_ticks"]):
                raise ValueError("paired frozen readout evidence or source-unit mapping differs")
            if cfg["validation_blocks"] != [14, 15, 16, 17]:
                raise ValueError("complete registered four-block selection partition required")
            for key in ("dataset_sha256", "acquisition_design_sha256"):
                if binding[key] != receipt[key]:
                    raise ValueError("native cohort differs from the frozen checkpoint")
            for key in ("episode_bindings", "native_source_identity", "acquisition_ledger_sha256", "episode_table_sha256"):
                if binding[key] != common_binding[key]:
                    raise ValueError("source identity differs across registered seeds")
            evaluation = json.loads(item["stage"]("forecasts/evaluation.json").read_text())
            current = evaluation["configuration"]
            if current["formal_evaluation"] or current["horizons_ms"] != [20., 80., 500.] or current["origins_ms"] != [1000., 3000., 5000.]:
                raise ValueError("registered development origin and horizon envelope required")
            if settings is None:
                settings = current
            elif settings != current:
                raise ValueError("forecast observation settings differ across seeds")
            with np.load(item["stage"]("forecasts/fitting_field_scale.npz"), allow_pickle=False) as saved:
                current_scale = {key: saved[key].copy() for key in ("mean_hz", "standard_deviation_hz", "unit_count", "field_ids")}
            if scale is None:
                scale = current_scale
            elif any(not np.array_equal(scale[key], value) for key, value in current_scale.items()):
                raise ValueError("frozen fitting-field normalization differs across seeds")
        mean_hz, sd, units, field_ids = (scale[key] for key in ("mean_hz", "standard_deviation_hz", "unit_count", "field_ids"))
        if not np.isfinite(sd).all() or np.any(sd <= 0) or np.any(units <= 0) or int(units.sum()) != 100000:
            raise ValueError("positive frozen field scale and complete field denominators required")
        pairs = pair_sets(field_ids)
        for block in (14, 15, 16, 17):
            for template in range(4):
                case = f"train_block_{block:03d}_familiar_{template}"
                native_work = work / "native" / case
                for name in [*[f"train/{case}/{part}" for part in ("report.json", "spikes.npz", "initial_parameters.npz")],
                             *[f"train_fields/{case}/{part}" for part in ("summary.json", "spatial_fields.npz")]]:
                    _stage(cohort, source_files, name, native_work / name, consumed_source)
                native = NativeRenewalEpisode(native_work / "train" / case, native_work / "train_fields" / case,
                    expected_source=common_binding["native_source_identity"],
                    expected_unit_ids=np.asarray(reference["model"].unit_ids),
                    expected_unit_field_index=reference["model"].unit_field_index.cpu().numpy())
                if (native.bin_ms != 5. or native.dt_ms != .1 or native.duration_ticks != 60000
                        or not np.array_equal(native.field_ids, field_ids)
                        or not np.array_equal(np.bincount(native.unit_field_index, minlength=len(field_ids)), units)
                        or not np.array_equal(native.minimum_interval_ticks, reference["checkpoint"]["model_configuration"]["minimum_interval_ticks"])):
                    raise ValueError("native clocks, source field map or refractory intervals changed")
                declared = common_binding["episode_bindings"][case]
                if (native.binding["episode_report_sha256"] != declared["raw_report_sha256"]
                        or native.binding["field_summary_sha256"] != declared["field_summary_sha256"]):
                    raise ValueError("native case report/field binding differs from training")
                with np.load(native_work / f"train_fields/{case}/spatial_fields.npz", allow_pickle=False) as fields:
                    count_sum = fields["count_sum"].copy()
                cached = {}
                for seed, item in cells.items():
                    name = f"training/posterior_cache/{case}.pt"
                    saved = torch.load(item["stage"](name), map_location="cpu", weights_only=True)
                    cb = saved["binding"]
                    if (cb["case_id"] != case or cb["parent_checkpoint_sha256"] != item["checkpoint"]["initialization"]["parent_checkpoint_sha256"]
                            or cb["model_implementation_sha256"] != source_hash or cb["bin_ms"] != 5.
                            or any(item["checkpoint"]["binding"][key] != value for key, value in cb["cohort"].items())):
                        raise ValueError("cached posterior source, parent or time binding changed")
                    m, v = saved["mean"], saved["variance"]
                    if (m.shape != (1, 1200, len(field_ids), item["model"].latent_channels)
                            or v.shape != m.shape or not torch.isfinite(m).all() or not torch.isfinite(v).all() or torch.any(v <= 0)):
                        raise ValueError("complete finite positive-variance posterior cache required")
                    cached[seed] = (m, v)
                for origin in settings["origins_ms"]:
                    start_bin = int(origin / native.bin_ms)
                    start_tick = int(origin / native.dt_ms)
                    window = native.renewal_window(start_tick, start_tick + 5000)
                    truth = np.zeros((100, len(field_ids)), dtype=np.float64)
                    np.add.at(truth.T, native.unit_field_index, window.counts.T)
                    if not np.array_equal(truth, count_sum[start_bin:start_bin + 100]):
                        raise ValueError("all-unit native counts differ from recorded field projection")
                    origin_ids.append(dict(case_id=case, block=block, template_id=f"familiar_{template}", origin_ms=origin))
                    observed.append(truth)
                    for seed, item in cells.items():
                        m, v = cached[seed]
                        result = posterior_field_reconstruction(item["model"], m[:, start_bin:start_bin+100], v[:, start_bin:start_bin+100],
                            window.eligible_ticks, unit_chunk_size=item["model"].unit_chunk_size)
                        for key in KEYS:
                            reconstructions[seed][key].append(result[key])
                        for horizon in settings["horizons_ms"]:
                            bins = int(horizon / native.bin_ms)
                            a = (result["mean_state_count"][:bins] / (units * .005) - mean_hz) / sd
                            b = (truth[:bins] / (units * .005) - mean_hz) / sd
                            rows.append(dict(seed=seed, case_id=case, block=block, template_id=f"familiar_{template}", origin_ms=origin,
                                horizon_ms=horizon, source_units=100000,
                                observed_spikes_per_unit=float(truth[:bins].sum()/100000),
                                mean_state_field_mse_standardized=float(np.square(a-b).mean()),
                                **{key+"_spikes_per_unit":float(result[key][:bins].sum()/100000) for key in KEYS}))
                print(json.dumps(dict(case_id=case, completed_source_cases=len(origin_ids)//3, reconstructed_model_origins=len(origin_ids)*3)), flush=True)
        if len(origin_ids) != 48 or len(rows) != 432:
            raise ValueError("complete16-case48-origin three-seed reconstruction required")
        observed = np.asarray(observed)
        for seed in SEEDS:
            values = reconstructions[seed]
            for key in KEYS:
                values[key] = np.asarray(values[key])
            for template in range(4):
                for origin in settings["origins_ms"]:
                    indices = [i for i, identity in enumerate(origin_ids) if identity["template_id"]==f"familiar_{template}" and identity["origin_ms"]==origin]
                    if [origin_ids[i]["block"] for i in indices] != [14,15,16,17]:
                        raise ValueError("matched covariance requires the four source blocks")
                    for horizon in settings["horizons_ms"]:
                        bins = int(horizon / 5.)
                        truth = (observed[indices,:bins]/(units*.005)-mean_hz)/sd
                        point = (values["mean_state_count"][indices,:bins]/(units*.005)-mean_hz)/sd
                        requests = [(key, pair[:,0], pair[:,1], 0) for key,pair in pairs.items()]
                        requests += [(f"temporal_lag_{lag}",np.arange(len(field_ids)),np.arange(len(field_ids)),lag)
                                     for lag in settings["temporal_lags_bins"] if lag<bins]
                        for statistic,left,right,lag in requests:
                            real_cov = response_covariance(truth,left,right,lag)
                            reconstruction_cov = response_covariance(point,left,right,lag)
                            cov_rows.append(dict(seed=seed,template_id=f"familiar_{template}",origin_ms=origin,horizon_ms=horizon,
                                statistic=statistic,source_blocks=4,pairs=len(left),phases=bins-lag,
                                observed_covariance_mean=float(real_cov.mean()),
                                mean_state_response_covariance_mean=float(reconstruction_cov.mean()),
                                covariance_mse=float(np.square(reconstruction_cov-real_cov).mean()),
                                covariance_mae=float(np.abs(reconstruction_cov-real_cov).mean())))
            np.savez_compressed(output/f"seed_{seed}_reconstructed_fields.npz",**values)
        np.savez_compressed(output/"observed_fields.npz",observed_field_counts=observed,
            case_ids=np.asarray([row["case_id"] for row in origin_ids]),origins_ms=np.asarray([row["origin_ms"] for row in origin_ids]),
            field_ids=field_ids,unit_count=units,mean_hz=mean_hz,standard_deviation_hz=sd)
        bindings={str(seed):dict(cell=str(item["cell"].resolve()),delivery_sha256=item["delivery_sha256"],files=item["consumed"])
                  for seed,item in cells.items()}
    _csv(output/"reconstruction_by_condition.csv",rows)
    _csv(output/"response_covariance.csv",cov_rows)
    summaries=[]
    for seed in SEEDS:
        for horizon in settings["horizons_ms"]:
            group=[row for row in rows if row["seed"]==seed and row["horizon_ms"]==horizon]
            metrics=[key for key in rows[0] if key.endswith("_spikes_per_unit") or key=="mean_state_field_mse_standardized"]
            summaries.append(dict(seed=seed,horizon_ms=horizon,source_cases=16,origins_per_case=3,
                **{key:float(np.mean([row[key] for row in group])) for key in metrics}))
    _csv(output/"reconstruction_summary.csv",summaries)
    cov_summary=[]
    for seed in SEEDS:
        for horizon in settings["horizons_ms"]:
            for statistic in sorted({row["statistic"] for row in cov_rows if row["horizon_ms"]==horizon}):
                group=[row for row in cov_rows if(row["seed"],row["horizon_ms"],row["statistic"])==(seed,horizon,statistic)]
                cov_summary.append(dict(seed=seed,horizon_ms=horizon,statistic=statistic,matched_template_origins=len(group),
                    **{key:float(np.mean([row[key] for row in group])) for key in ("observed_covariance_mean","mean_state_response_covariance_mean","covariance_mse","covariance_mae")}))
    _csv(output/"response_covariance_summary.csv",cov_summary)
    artifacts=["reconstruction_by_condition.csv","reconstruction_summary.csv","response_covariance.csv","response_covariance_summary.csv",
               "observed_fields.npz",*[f"seed_{seed}_reconstructed_fields.npz" for seed in SEEDS]]
    result=dict(status="COMPLETE_FROZEN_POSTERIOR_FIELD_RECONSTRUCTION",training_seeds=list(SEEDS),source_units=100000,
        selection_episodes=16,source_origins=48,reconstructed_model_origins=144,condition_rows=len(rows),covariance_rows=len(cov_rows),
        readout_reused_for_both_prior_budgets=True,source_exposure_conditioned=True,future_observations_accessed=True,
        prior_budget_frozen_parameter_evidence=str(prior_diagnostic.resolve()), prior_budget_evidence_sha256=sha256_file(prior_diagnostic),
        posterior_encoding_performed=False,training_performed=False,source_simulation_performed=False,unit_sampling_performed=False,
        scientific_claim_qualified=False,free_predictive_capacity_qualified=False,
        interpretation="observed-exposure and cached-posterior deterministic reconstruction; response covariance is not predictive covariance and probability bounds are not confidence or covariance bounds",
        job_id=os.environ["SLURM_JOB_ID"],node=node,device=device,gpu=torch.cuda.get_device_name() if device=="cuda" else None,torch_version=torch.__version__,
        elapsed_seconds=time.monotonic()-started,normalization="unchanged fitting-block field normalization",
        consumed=bindings,source_archive=str(cohort.resolve()),source_delivery_sha256=sha256_file(cohort/"delivery.json"),consumed_source=consumed_source,
        helper_sha256=sha256_file(Path(__file__).with_name("posterior_field_reconstruction.py")),
        diagnostic_sha256=sha256_file(Path(__file__)),artifacts={name:sha256_file(output/name) for name in artifacts})
    freeze_json(output/"analysis.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,required=True)
    parser.add_argument("--cohort",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prior-diagnostic",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=parser.parse_args()
    result=diagnose(args.campaign,args.cohort,args.output,args.prior_diagnostic,args.device)
    print(json.dumps({key:result[key] for key in ("status","condition_rows","covariance_rows")}))


if __name__=="__main__":
    main()
