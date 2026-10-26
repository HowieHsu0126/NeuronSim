"""Validate native residual fitting on one authentic complete fitting episode.

This is a full-unit implementation check, not a fitted model or selection result.
All three seeds and both correction modes use zero correction initialization and
one fixed SGD step. Every result, including a worsening loss, is retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

import numpy as np
import torch

from .native_renewal_data import NativeRenewalEpisode
from .population_residual_memory import ResidualMemoryPopulationState
from .population_residual_observation import (
    configure_correction_parameters, export_affine_predictor,
    residual_observation_mixture_loss,
)
from .population_state import ControlledPopulationState
from .provenance import sha256_file
from .residual_prediction import RidgeModel


CASE = "train_block_000_familiar_0"
SEEDS = (41, 42, 43)
ORIGINS = (200, 600, 1000)
HORIZONS = (4, 16, 100)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_seed(inputs, seed):
    cell = inputs / f"parent/seed_{seed}"
    delivery = json.loads((cell / "delivery.json").read_text())
    report = json.loads((cell / "training/run.json").read_text())
    path = cell / "training/best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    binding = checkpoint["binding"]
    if (delivery.get("status") != "PASS" or delivery.get("exit_code") != 0
            or delivery.get("training_seed") != seed or delivery.get("source_units") != 100000
            or report.get("status") != "FITTED_DEVELOPMENT" or report["binding"] != binding
            or binding["seed"] != seed or report["completed_epochs"] != 80
            or report["model_configuration"] != checkpoint["model_configuration"]
            or report["best_checkpoint_sha256"] != sha256_file(path)):
        raise ValueError("complete accepted same-seed parent required")
    for key, name in (("model_implementation_sha256", "population_state.py"),
                      ("observation_implementation_sha256", "refractory_observation.py"),
                      ("loader_implementation_sha256", "native_renewal_data.py"),
                      ("latent_model_implementation_sha256", "poisson_state_space.py"),
                      ("native_counts_implementation_sha256", "native_count_windows.py")):
        if binding[key] != sha256_file(Path(__file__).with_name(name)):
            raise ValueError("parent implementation differs: " + name)
    model = ControlledPopulationState(**checkpoint["model_configuration"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    if (len(model.unit_ids) != 100000 or len(model.field_names) != 128 or model.latent_channels != 4
            or model.bin_seconds != .005 or model.tick_seconds != .0001 or model.metadata is not None):
        raise ValueError("registered100000-unit native parent and clock required")
    cache = torch.load(cell / f"training/posterior_cache/{CASE}.pt", map_location="cpu", weights_only=True)
    cb = cache["binding"]
    if (cb["case_id"] != CASE or cb["bin_ms"] != 5.
            or cb["parent_checkpoint_sha256"] != checkpoint["initialization"]["parent_checkpoint_sha256"]
            or cb["model_implementation_sha256"] != binding["model_implementation_sha256"]
            or any(binding[key] != value for key, value in cb["cohort"].items())):
        raise ValueError("fitting cache lineage differs")
    source = NativeRenewalEpisode(inputs / f"cohort/train/{CASE}", inputs / f"cohort/train_fields/{CASE}",
        expected_source=binding["native_source_identity"], expected_unit_ids=np.asarray(model.unit_ids),
        expected_unit_field_index=model.unit_field_index.numpy())
    record = binding["episode_bindings"][CASE]
    if (source.binding["episode_report_sha256"] != record["raw_report_sha256"]
            or source.binding["field_summary_sha256"] != record["field_summary_sha256"]
            or tuple(map(str, source.field_ids)) != tuple(model.field_names)
            or source.duration_ticks != 60000 or source.bin_ticks != 50
            or not np.array_equal(source.minimum_interval_ticks, model.minimum_interval_ticks.numpy())):
        raise ValueError("native fitting episode identity, duration or refractory rule differs")
    mean, variance, controls = (cache[key] for key in ("mean", "variance", "controls"))
    if (mean.shape != (1, 1200, 128, 4) or variance.shape != mean.shape or controls.shape != (1, 1200, 1)
            or any(v.dtype != model.unit_loading.dtype or not torch.isfinite(v).all()
                   for v in (mean, variance, controls)) or torch.any(variance <= 0)):
        raise ValueError("complete finite causal posterior cache required")
    pulses = source.pulse_ticks[source.pulse_ticks < source.duration_ticks]
    native_controls = np.bincount(pulses // source.bin_ticks, minlength=1200).astype(np.float32)[:, None]
    if not np.array_equal(controls[0].numpy(), native_controls):
        raise ValueError("cached controls differ from native TC events")
    windows = [source.renewal_window(o * source.bin_ticks, (o + 100) * source.bin_ticks) for o in ORIGINS]
    for origin, window in zip(ORIGINS, windows):
        if window.counts.shape != (100, 100000) or not np.array_equal(window.stimulus, native_controls[origin:origin + 100]):
            raise ValueError("full-unit native future window mismatch")
    counts = torch.from_numpy(np.stack([w.counts for w in windows])).unsqueeze(0)
    eligible = torch.from_numpy(np.stack([w.eligible_ticks for w in windows])).unsqueeze(0)
    return model, dict(mean=mean, variance=variance, controls=controls, target_counts=counts,
                      target_eligible_ticks=eligible, origins=ORIGINS, horizons=HORIZONS), source.binding


def native_loss(model, inputs, seed, *, capture=False):
    initial = torch.Generator().manual_seed(seed)
    dynamics = torch.Generator().manual_seed(SEEDS[(SEEDS.index(seed) + 1) % 3])
    paths = []
    original = model.parent.log_hazard
    if capture:
        def readout(z, start=0, stop=None):
            if not paths:
                paths.append(z.detach().clone())
            return original(z, start, stop)
        model.parent.log_hazard = readout
    try:
        values = residual_observation_mixture_loss(model, **inputs,
            initial_generator=initial, dynamics_generator=dynamics, prediction_particles=8,
            unit_chunk_size=4096, checkpoint_readout=True)
    finally:
        if capture:
            model.parent.log_hazard = original
    for name, value in values.items():
        if not torch.isfinite(value) or ('ess' in name and not 1 <= float(value) <= 8):
            raise FloatingPointError("invalid native objective metric: " + name)
    return values, paths


def validate(inputs, manifest_path, output):
    if (platform.node().split('.')[0] == 'mgmt2' or not os.environ.get('RH2_JOB_RUN_ID')
            or not os.environ.get('ARNOLD_TRIAL_ID')):
        raise RuntimeError("real objective validation requires Merlin execution")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('scope') != 'one_fitting_episode_full_unit_objective_validation'
            or manifest.get('case_id') != CASE or manifest.get('training_seeds') != list(SEEDS)):
        raise ValueError("registered bounded fitting input manifest required")
    for row in manifest['files']:
        path = inputs / row['path']
        if path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']:
            raise ValueError("staged input identity differs: " + row['path'])
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', run_id=os.environ['RH2_JOB_RUN_ID'], trial_id=os.environ['ARNOLD_TRIAL_ID'],
        host=platform.node(), started_at=datetime.now(timezone.utc).isoformat(), records=[],
        runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cpu_threads=torch.get_num_threads()),
        input_manifest_sha256=sha256_file(manifest_path), training_seeds=list(SEEDS), case_id=CASE,
        source_units=100000, source_episode_ms=6000, origins_ms=[1000, 3000, 5000], horizons_ms=[20, 80, 500],
        prediction_particles=8, optimizer='SGD', learning_rate=.0001, updates_per_seed_mode=1,
        correction_initialization='zero_affine_diagnostic', fitting_complete=False,
        source_simulation_performed=False, posterior_encoding_performed=False, scientific_claim_qualified=False)
    try:
        for seed in SEEDS:
            report["active_seed"] = seed
            report["active_mode"] = None
            parent, batch, source_binding = load_seed(inputs, seed)
            frozen = {name: value.clone() for name, value in parent.state_dict().items()}
            for mode in ('current', 'history'):
                report['active_mode'] = mode
                started = time.perf_counter()
                d = len(parent.field_names) * parent.latent_channels
                p = d + parent.input_size + (d if mode == 'history' else 0)
                reference = RidgeModel(1., np.zeros(p), np.ones(p), np.zeros(d), np.ones(d), np.zeros((p, d)))
                model = ResidualMemoryPopulationState(parent, reference, mode, {'scope': 'real_objective_validation'})
                parameters = configure_correction_parameters(model)
                optimizer = torch.optim.SGD(parameters, lr=.0001)
                optimizer.zero_grad(set_to_none=True)
                before, _ = native_loss(model, batch, seed)
                before['loss'].backward()
                if (any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters)
                        or any(p.grad is not None for p in parent.parameters())):
                    raise FloatingPointError("missing/nonfinite correction gradient or changed parent gradient")
                norm = float(torch.stack([p.grad.double().square().sum() for p in parameters]).sum().sqrt())
                if norm == 0:
                    raise FloatingPointError("real native objective has zero correction gradient")
                optimizer.step()
                with torch.no_grad():
                    after, paths = native_loss(model, batch, seed, capture=True)
                    exported = export_affine_predictor(model, reference)
                    restored = ResidualMemoryPopulationState(parent, exported, mode, {'scope': 'diagnostic_export'})
                    configure_correction_parameters(restored)
                    replay, replay_paths = native_loss(restored, batch, seed, capture=True)
                    if not torch.equal(paths[0], replay_paths[0]):
                        raise ValueError("exported affine native trajectory differs")
                    if any(not torch.equal(after[key], replay[key]) for key in after):
                        raise ValueError("exported affine native likelihood differs")
                if any(not torch.equal(value, parent.state_dict()[name]) for name, value in frozen.items()):
                    raise ValueError("frozen parent parameter or buffer changed")
                row = dict(seed=seed, mode=mode, before={k:float(v.detach()) for k,v in before.items()},
                    after={k:float(v.detach()) for k,v in after.items()}, gradient_norm=norm,
                    source_binding=source_binding, parent_unchanged=True, exported_latent_replay_exact=True,
                    seconds=time.perf_counter()-started)
                report['records'].append(row)
                write_json(output/'validation.json', report)
                print(json.dumps({k:row[k] for k in ('seed','mode','gradient_norm','seconds')}), flush=True)
                del optimizer, model, restored, before, after, replay, paths, replay_paths
            del parent, batch, frozen
        report.update(status='PASS_REAL_OBJECTIVE_VALIDATION', exit_code=0)
    except Exception:
        report.update(status='FAILED_REAL_OBJECTIVE_VALIDATION', exit_code=1, error=traceback.format_exc())
    report.update(finished_at=datetime.now(timezone.utc).isoformat(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(output/'validation.json', report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    if not 1 <= args.threads <= 8:raise ValueError('one to eight CPU threads required')
    torch.set_num_threads(args.threads)
    result=validate(args.inputs,args.manifest,args.output)
    raise SystemExit(result['exit_code'])


if __name__ == '__main__':
    main()
