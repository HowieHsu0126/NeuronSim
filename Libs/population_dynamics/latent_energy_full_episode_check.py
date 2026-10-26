"""Exercise the EAG architecture on one complete source episode, then replay.

This is an engineering check. The two updates and single-fit-case latent scale
are not a trained baseline or a held-out scientific comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch
import yaml

from .latent_energy import ControlledLatentEnergy
from .native_trajectory_cohort import TrajectoryCohort
from .provenance import sha256_file


def array_record(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(memoryview(array)).hexdigest())


def run(acquisition, fields, config_path, output, seed, replay=False):
    config = yaml.safe_load(config_path.read_text())
    if config['training_seeds'] != [41, 42, 43] or seed not in config['training_seeds']:
        raise ValueError('registered three-seed configuration required')
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    if config['device'] != 'cuda' or not torch.cuda.is_available():
        raise ValueError('allocated CUDA device required for the complete-unit check')
    started = time.monotonic()
    cohort = TrajectoryCohort(acquisition, fields, config)
    case = cohort.fit[0]
    values, control_values = cohort.window(case)
    assert values.shape == (1200, 100000)
    model = ControlledLatentEnergy(tuple(map(int, cohort.unit_ids)), **config['model']).cuda()
    counts = torch.as_tensor(values, device='cuda')[None]
    controls = torch.as_tensor(control_values, device='cuda')[None]
    binding = dict(config=config, seed=seed, cohort=cohort.binding, case_id=case['case_id'])
    checkpoint = output / 'checkpoint.pt'
    if replay:
        original = json.loads((output / 'generation.json').read_text())
        if sha256_file(checkpoint) != original['checkpoint_sha256']:
            raise ValueError('probe checkpoint changed')
        payload = torch.load(checkpoint, map_location='cuda', weights_only=True)
        if payload['binding'] != binding:
            raise ValueError('probe source or configuration binding changed')
        model.load_state_dict(payload['model'])
    else:
        output.mkdir(parents=True, exist_ok=False)
        model.train()
        ae = config['autoencoder']
        optimizer = torch.optim.AdamW(model.autoencoder.parameters(), lr=ae['learning_rate'],
                                     weight_decay=ae['weight_decay'])
        mask = torch.rand((1, 100000, 1200), device='cuda',
                          generator=torch.Generator(device='cuda').manual_seed(41)) > ae['mask_probability']
        ae_terms = model.autoencoder_loss(counts, mask=mask, mask_probability=ae['mask_probability'],
                                         latent_beta=ae['latent_beta'], temporal_beta=ae['temporal_beta'])
        ae_terms['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.autoencoder.parameters(), ae['gradient_limit'], error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del optimizer, mask
        with torch.no_grad():
            target_latent = model.autoencoder.encode(model._counts(counts[:, 1000:1100]))
            model.set_latent_scale(target_latent.double().mean((0, 2), keepdim=True),
                                   target_latent.double().std((0, 2), correction=0, keepdim=True))
        state = model.infer_state(counts[:, :1000], controls[:, :1000])
        target = model.encode_target(counts[:, 1000:1100])
        energy = config['energy']
        optimizer = torch.optim.AdamW(model.energy.parameters(), lr=energy['learning_rate'],
            weight_decay=energy['weight_decay'], betas=tuple(energy['optimizer_betas']))
        energy_terms = model.energy.loss(target, state.latent, controls[:, :1100],
                                         generator=torch.Generator(device='cuda').manual_seed(42))
        energy_terms['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.energy.parameters(), energy['gradient_limit'], error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_record = dict(autoencoder={k: float(v.detach()) for k, v in ae_terms.items()},
                             energy={k: float(energy_terms[k].detach()) for k in ('loss', 'attraction', 'repulsion')})
        del optimizer, ae_terms, energy_terms, state, target, target_latent
        torch.save(dict(binding=binding, model=model.state_dict()), checkpoint)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    count_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone()
    count = config['sampling']['joint_samples']
    state = model.infer_state(counts[:, :1000], controls[:, :1000])
    state = type(state)(state.latent.expand(count, -1, -1).contiguous(),
                        state.controls.expand(count, -1, -1).contiguous(), state.unit_ids)
    result = model.simulate(state, controls[:, 1000:1100].expand(count, -1, -1).contiguous(),
        latent_rng=torch.Generator(device='cuda').manual_seed(42),
        observation_rng=torch.Generator(device='cuda').manual_seed(43),
        **{k: config['sampling'][k] for k in ('steps', 'guidance', 'temperature')})
    torch.cuda.synchronize()
    assert torch.equal(count_before, torch.random.get_rng_state())
    assert torch.equal(cuda_before, torch.cuda.get_rng_state())
    if result['counts'].shape != (16, 100, 100000) or torch.any(result['counts'] < 0):
        raise ValueError('complete nonnegative all-unit output required')
    arrays = {key: array_record(result[key]) for key in ('counts', 'expected_counts', 'latent')}
    record = dict(status='PASS', phase='replay' if replay else 'fit_generate', binding=binding,
        parameters=sum(p.numel() for p in model.parameters()), checkpoint_sha256=sha256_file(checkpoint),
        generated_unit_count_values=result['counts'].numel(), arrays=arrays,
        peak_inference_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        elapsed_seconds=time.monotonic()-started, full_episode_bins=1200, source_units=100000,
        observed_prefix_bins=1000, forecast_bins=100, joint_draws=16,
        global_rng_unchanged=True, normalization_scope='single-fitting-case engineering diagnostic',
        autoencoder_training_updates=1, energy_training_updates=1,
        formal_training_completed=False, scientific_claim_qualified=False)
    if replay:
        if arrays != original['arrays']:
            raise ValueError('fresh-process all-unit replay mismatch')
    else:
        record['update_losses'] = update_record
    name = 'replay.json' if replay else 'generation.json'
    (output / name).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: value for key, value in record.items() if key != 'binding'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('acquisition', 'fields', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(41, 42, 43), required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    run(args.acquisition, args.fields, args.config, args.output, args.seed, args.replay)


if __name__ == '__main__':
    main()
