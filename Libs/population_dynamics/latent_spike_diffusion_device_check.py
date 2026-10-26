"""Validate the prepared full-unit LDNS path on an allocated GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import yaml

from .latent_spike_diffusion_training import _model, _seed, _write_json, _ae_epoch, _optimizer, build_latent_cache
from .latent_spike_diffusion import PrefixCondition
from .native_trajectory_cohort import TrajectoryCohort
from .provenance import sha256_file


def diffusion_updates(model, target, state, stimulus, config, generator):
    """Exercise the source warmup schedule and prove a denoiser weight update."""
    optimizer, scheduler = _optimizer(model, config, "diffusion")
    before = {name: parameter.detach().clone() for name, parameter in model.denoiser.named_parameters()}
    updates = []
    for index in range(2):
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        if index == 0 and any(rate != 0 for rate in learning_rates):
            raise ValueError("LDNS_DEVICE_CHECK_EXPECTS_INITIAL_ZERO_WARMUP_RATE")
        if index == 1 and not all(rate > 0 for rate in learning_rates):
            raise ValueError("LDNS_DEVICE_CHECK_WARMUP_DID_NOT_ADVANCE")
        loss = model.diffusion_loss(target, state, stimulus,
            noise=torch.randn(target.shape, device=target.device, generator=generator),
            timestep=torch.tensor([500], device=target.device))
        if not torch.isfinite(loss):
            raise FloatingPointError("LDNS_DEVICE_CHECK_NONFINITE_DIFFUSION_LOSS")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(),
            config["diffusion_gradient_limit"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if index == 0 and any(not torch.equal(parameter, before[name])
                              for name, parameter in model.denoiser.named_parameters()):
            raise ValueError("LDNS_DEVICE_CHECK_ZERO_RATE_CHANGED_PARAMETERS")
        updates.append(dict(loss=float(loss.detach()), learning_rates=learning_rates,
                            gradient_norm=float(gradient.detach())))
    deltas = [(parameter.detach() - before[name]).abs().max().item()
              for name, parameter in model.denoiser.named_parameters()]
    if not deltas or not np.isfinite(deltas).all() or max(deltas) <= 0:
        raise ValueError("LDNS_DEVICE_CHECK_NO_DENOISER_PARAMETER_UPDATE")
    return dict(optimizer_steps=len(updates), scheduler_steps=len(updates),
                updates=updates, changed_parameter_tensors=sum(delta > 0 for delta in deltas),
                maximum_parameter_change=max(deltas), initial_zero_rate_parameters_unchanged=True)


def check(acquisition, fields, config_path, output, *, replay=False):
    acquisition,fields,config_path,output=map(Path,(acquisition,fields,config_path,output))
    config=yaml.safe_load(config_path.read_text())
    if config["device"]!="cuda" or not torch.cuda.is_available():raise ValueError("allocated CUDA device required")
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(41)
    cohort=TrajectoryCohort(acquisition,fields,config)
    model=_model(cohort,config)
    if replay:
        report=json.loads((output/"check.json").read_text())
        if report["cohort"]!=cohort.binding or sha256_file(output/"checkpoint.pt")!=report["checkpoint_sha256"]:
            raise ValueError("GPU check input or checkpoint changed")
        model.load_state_dict(torch.load(output/"checkpoint.pt",map_location="cuda",weights_only=False))
    else:
        output.mkdir(parents=True,exist_ok=False)
        # This checks one representative full episode. It is not campaign fit.
        cohort.fit=cohort.fit[:1];cohort.validation=[]
        optimizer,scheduler=_optimizer(model,config,"autoencoder")
        start=time.perf_counter()
        fitting=_ae_epoch(model,cohort,config,41,0,torch.Generator(device="cuda").manual_seed(42000),optimizer,scheduler)
        torch.cuda.synchronize();update_seconds=time.perf_counter()-start
        cache=build_latent_cache(model,cohort,config)
        row=cache["rows"]["fit"][0];target=row["target"].cuda()
        state=PrefixCondition(row["prefix"].cuda(),model.unit_ids,row["origin"])
        generator=torch.Generator(device="cuda").manual_seed(43000)
        updates=diffusion_updates(model,target,state,row["stimulus"].cuda(),config,generator)
        torch.save(model.state_dict(),output/"checkpoint.pt")
        report=dict(status="PASS_REAL_FULL_UNIT_CUDA_UPDATES_AND_GENERATION",cohort=cohort.binding,
            checkpoint_sha256=sha256_file(output/"checkpoint.pt"),ae_losses=fitting["mean"],
            ae_update_seconds=update_seconds,diffusion_loss=updates["updates"][-1]["loss"],
            diffusion_updates=updates,
            diagnostic_only=True,registered_training_completed=False,latent_scale="one_fit_episode_three_target_windows",
            source_units=100000,complete_episode_bins=1200,scientific_claim_qualified=False)
    counts,controls=cohort.window(cohort.fit[0])
    model.eval();state=model.infer_state(torch.as_tensor(counts[None,:200],device="cuda",dtype=torch.float32))
    stimulus=torch.as_tensor(controls[None,200:300],device="cuda",dtype=torch.float32)
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    torch.cuda.synchronize();started=time.perf_counter()
    sample=model.simulate(state,stimulus,latent_rng=torch.Generator(device="cuda").manual_seed(44000),
                          observation_rng=torch.Generator(device="cuda").manual_seed(45000))
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    values=sample["counts"].cpu().numpy();digest=hashlib.sha256(values.tobytes()).hexdigest()
    if not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state()):
        raise ValueError("GPU LDNS sampler changed global RNG")
    if replay:
        if digest!=report["counts_sha256"]:raise ValueError("GPU LDNS full count replay differs")
        with np.load(output/"outputs.npz",allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["counts"],values)
            np.testing.assert_array_equal(saved["latent"],sample["latent"].cpu().numpy())
            np.testing.assert_array_equal(saved["unit_ids"],cohort.unit_ids)
        result=dict(status="PASS_FRESH_PROCESS_FULL_UNIT_CUDA_REPLAY",count_values=values.size,
            seconds=seconds,global_cpu_cuda_rng_preserved=True,scientific_claim_qualified=False)
        _write_json(output/"replay.json",result)
        archive=Path("/workspace/NeuronSim/Output/neural_population_dynamics/baselines/latent_spike_diffusion/cuda_validation_20261012")
        archive.mkdir(parents=True,exist_ok=False)
        files=[]
        for name in ("check.json","checkpoint.pt","outputs.npz","replay.json"):
            source=output/name;target=archive/name;shutil.copy2(source,target)
            digest=sha256_file(source)
            if sha256_file(target)!=digest:raise ValueError("CUDA validation archive changed")
            files.append(dict(path=name,sha256=digest,bytes=source.stat().st_size))
        _write_json(archive/"delivery.json",dict(status="PASS",files=files,
            source_cohort=cohort.binding,device_check_sha256=sha256_file(Path(__file__)),
            trainer_sha256=sha256_file(Path(__file__).with_name("latent_spike_diffusion_training.py")),
            adapter_sha256=sha256_file(Path(__file__).with_name("latent_spike_diffusion.py")),
            configuration_sha256=sha256_file(config_path),diagnostic_only=True,scientific_claim_qualified=False))
    else:
        np.savez_compressed(output/"outputs.npz",counts=values,latent=sample["latent"].cpu().numpy(),unit_ids=cohort.unit_ids)
        report.update(counts_sha256=digest,generation_seconds=seconds,count_values=values.size,
            gpu=torch.cuda.get_device_name(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            global_cpu_cuda_rng_preserved=True)
        _write_json(output/"check.json",report);result=report
    print(json.dumps({k:v for k,v in result.items() if k!="cohort"}));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition","fields","config","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--replay",action="store_true");args=parser.parse_args()
    check(args.acquisition,args.fields,args.config,args.output,replay=args.replay)


if __name__=="__main__":main()
