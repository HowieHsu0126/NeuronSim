"""All-unit V1 EAG forecasts under the existing source-field Hz protocol.

Model channels are positions in each source episode. Artifacts retain the
native neuron IDs and field map for that episode. EAG emits Poisson counts;
no renewal initial state or terminal readiness is asserted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import latent_energy_training as training
from . import source_population_evaluation as source
from .provenance import sha256_file
from .v1_latent_energy_data import V1LatentEnergyCohort
from .v1_latent_energy_training import validate_config
from .v1_population_evaluation import _config

EVALUATION_STATUS = "EVALUATED_V1_EAG_SOURCE_COUNTS"


@dataclass(frozen=True)
class SourceCountMap:
    unit_ids: tuple[int, ...]
    field_names: tuple[str, ...]
    unit_field_index: torch.Tensor
    field_unit_counts: torch.Tensor
    minimum_interval_ticks: torch.Tensor
    field_units: tuple[torch.Tensor, ...]
    bin_ticks: int
    bin_seconds: float
    source_metadata_identity: None = None


def _load(inputs, directory, device):
    directory=Path(directory)
    run=json.loads((directory/"run.json").read_text())
    if run.get("status")!="FITTED_DEVELOPMENT":
        raise ValueError("COMPLETE_V1_EAG_TRAINING_REQUIRED")
    pointer=json.loads((directory/"committed.json").read_text())
    path=directory/pointer["checkpoint"]
    if sha256_file(path)!=pointer["sha256"]:
        raise ValueError("V1_EAG_TERMINAL_CHECKPOINT_CHANGED")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    state=payload["state"]
    # The committed checkpoint owns the binding; run.json is a derived summary.
    config=state["binding"]["config"]
    seed=state["binding"]["seed"]
    if config.get("schema")!=training.V1_SCHEMA:
        raise ValueError("COMPLETE_V1_EAG_TRAINING_REQUIRED")
    validate_config(config,seed)
    # Check original circuit membership before source arrays or model construction.
    cohort=V1LatentEnergyCohort(inputs,config,config_validator=validate_config)
    binding=dict(config=config,seed=seed,cohort=cohort.binding,sources=training._sources(config),
                 source_revision=training.SOURCE_REVISION,torch_version=torch.__version__)
    windows=len(cohort.fit)*len(config["forecast_origins_bins"])
    presentations=windows*config["energy"]["maximum_epochs"]
    batch=config["energy"]["source_batch_size"]
    if (run["training_seed"]!=seed or state["binding"]!=binding or state["stage"]!="complete"
            or pointer["stage"]!="complete" or pointer["commits"]!=state["commits"]
            or state["autoencoder_completed_epochs"]!=config["autoencoder"]["maximum_epochs"]
            or state["energy_sampler"]["presentations"]!=presentations
            or state["energy_sampler"]["completed_passes"]!=config["energy"]["maximum_epochs"]
            or state["energy_sampler"]["cursor"]!=0 or state["energy_sampler"]["order"]
            or state["steps"]!=(presentations+batch-1)//batch):
        raise ValueError("V1_EAG_COMPLETE_TRAINING_SOURCE_OR_BUDGET_CHANGED")
    for name,digest in ((state["selected_path"],state["selected_digest"]),
                        (state["autoencoder_selected_path"],state["autoencoder_selected_digest"]),
                        ("latent_cache.pt",state["cache_digest"])):
        if sha256_file(directory/name)!=digest:
            raise ValueError("V1_EAG_SELECTED_WEIGHTS_OR_FITTING_SCALE_CHANGED")
    model=training._model(cohort,dict(config,device=str(device)))
    model.load_state_dict(payload["model"],strict=True)
    model.eval()
    metadata=dict(status=run["status"],binding=cohort.binding,training_seed=seed,
        best_checkpoint_sha256=pointer["sha256"],
        evaluation_training_metadata=dict(method="eag",observation_law="softplus_poisson_counts",
            native_refractory_support_qualified=False,
            autoencoder_completed_epochs=state["autoencoder_completed_epochs"],
            energy_completed_passes=state["energy_sampler"]["completed_passes"],
            energy_presentations=presentations,energy_optimizer_steps=state["steps"],
            selected_energy_checkpoint_sha256=state["selected_digest"],
            selected_autoencoder_checkpoint_sha256=state["autoencoder_selected_digest"],
            latent_axes=["draw","latent_channel","future_bin"],
            count_channel_identity="per_episode_native_source_order"))
    return model,cohort,metadata


def _prepare_case(model,cohort,case):
    data=cohort.native_episode(case)
    if len(data.unit_ids)!=len(model.unit_ids) or tuple(model.unit_ids)!=tuple(range(len(data.unit_ids))):
        raise ValueError("V1_EAG_COMPLETE_CANONICAL_CHANNEL_WIDTH_REQUIRED")
    indices=np.asarray(data.unit_field_index)
    names=tuple(map(str,data.field_ids))
    view=SourceCountMap(tuple(map(int,data.unit_ids)),names,torch.as_tensor(indices.copy(),dtype=torch.int64),
        torch.as_tensor(np.bincount(indices,minlength=len(names)),dtype=torch.int64),
        torch.as_tensor(data.minimum_interval_ticks.copy(),dtype=torch.int64),
        tuple(torch.as_tensor(np.flatnonzero(indices==f)) for f in range(len(names))),
        data.bin_ticks,data.dt_ms*data.bin_ticks/1000)
    if view.bin_seconds!=model.bin_seconds:
        raise ValueError("V1_EAG_SOURCE_MODEL_CLOCK_CHANGED")
    return view,data


@torch.no_grad()
def _forecast(model,cohort,case,origin,config,seed_base,device,*,prepared=None,costs=None):
    view,data=_prepare_case(model,cohort,case) if prepared is None else prepared
    horizon=max(config["horizons_bins"])
    if (origin%data.bin_ticks or origin//data.bin_ticks not in cohort.config["forecast_origins_bins"]
            or horizon!=model.forecast_bins or origin+horizon*data.bin_ticks>data.duration_ticks
            or config.get("random_seed_policy")!="exact_registered_stream_seeds" or seed_base!=41):
        raise ValueError("V1_EAG_DECLARED_SOURCE_FORECAST_WINDOW_REQUIRED")
    if costs is not None and device.type=="cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source._synchronize(device);started=time.perf_counter()
    counts,controls=data.window(0,origin)
    scheduled=source.source_controls(data,0,origin)
    future=source.source_controls(data,origin,origin+horizon*data.bin_ticks)
    features=tuple(cohort.binding["count_adapter"]["control_features"])
    if (tuple(scheduled["control_features"])!=features or tuple(future["control_features"])!=features
            or not np.array_equal(controls,scheduled["known_controls"])
            or counts.shape!=(origin//data.bin_ticks,len(view.unit_ids))
            or controls.shape!=(origin//data.bin_ticks,model.input_size)
            or future["known_controls"].shape!=(horizon,model.input_size)):
        raise ValueError("V1_EAG_SOURCE_PREFIX_AND_CONTROL_CONTRACT_CHANGED")
    dtype=model.latent_mean.dtype
    y=torch.as_tensor(counts[None],device=device,dtype=dtype)
    u=torch.as_tensor(controls[None],device=device,dtype=dtype)
    state=model.infer_state(y,u)
    future_controls=torch.as_tensor(future["known_controls"][None],device=device,dtype=dtype)
    source._synchronize(device);inference_seconds=time.perf_counter()-started
    started=time.perf_counter()
    global_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else []
    # Explicit latent and observation streams advance through every ensemble draw.
    latent_rng=torch.Generator(device=device).manual_seed(42)
    observation_rng=torch.Generator(device=device).manual_seed(43)
    generated,latents=[],[]
    sampling=cohort.config["sampling"]
    for _ in range(config["joint_samples"]):
        result=model.simulate(state,future_controls,latent_rng=latent_rng,observation_rng=observation_rng,
                              steps=sampling["steps"],guidance=sampling["guidance"],temperature=sampling["temperature"])
        if (tuple(result["unit_ids"])!=tuple(model.unit_ids) or result["counts"].shape!=(1,horizon,len(view.unit_ids))
                or result["bin_seconds"]!=view.bin_seconds):
            raise ValueError("V1_EAG_GENERATED_SOURCE_CHANNEL_OR_CLOCK_CHANGED")
        generated.append(result["counts"][0].cpu());latents.append(result["latent"][0].cpu())
    if (not torch.equal(global_rng,torch.get_rng_state())
            or any(not torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all() if cuda_rng else []))):
        raise RuntimeError("V1_EAG_FORECAST_CHANGED_GLOBAL_RANDOM_STREAM")
    generated,latent=torch.stack(generated),torch.stack(latents)
    source._synchronize(device);generation_seconds=time.perf_counter()-started
    started=time.perf_counter()
    # Observed future counts are read only after the complete generated ensemble.
    truth,observed_controls=data.window(origin,origin+horizon*data.bin_ticks)
    if not np.array_equal(observed_controls,future["known_controls"]):
        raise ValueError("V1_EAG_FACTUAL_FUTURE_CONTROLS_DIFFER")
    fields=torch.stack([generated.index_select(-1,index).sum(-1) for index in view.field_units],-1).numpy()
    observed=np.column_stack([truth[:,data.unit_field_index==f].sum(-1) for f in range(len(view.field_names))])
    source._synchronize(device)
    if costs is not None:
        costs.update(inference_seconds=inference_seconds,full_unit_generation_seconds=generation_seconds,
            field_projection_seconds=time.perf_counter()-started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None)
    return view,data,truth,generated,fields,observed,latent,None,None


def evaluate(inputs,training_dir,config_path,output,device_name="cpu"):
    return source.evaluate(Path(inputs),Path(training_dir),Path(config_path),Path(output),device_name,
        load=_load,read_config=_config,evaluator_path=Path(__file__),evaluation_status=EVALUATION_STATUS,
        forecast=_forecast,prepare_case=_prepare_case)


def replay(inputs,training_dir,forecasts,output,device_name="cpu"):
    return source.replay(Path(inputs),Path(training_dir),Path(forecasts),Path(output),device_name,
        load=_load,evaluator_path=Path(__file__),evaluation_status=EVALUATION_STATUS,forecast=_forecast)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("inputs","training","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--replay",action="store_true")
    args=parser.parse_args()
    if args.replay:
        report=replay(args.inputs,args.training,args.output,args.output/"replay.json",args.device)
    else:
        if args.config is None:
            parser.error("--config is required for evaluation")
        report=evaluate(args.inputs,args.training,args.config,args.output,args.device)
    print(json.dumps(report))


if __name__=="__main__":
    main()
