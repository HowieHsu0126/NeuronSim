"""Replay the complete-unit LDNS development probe without repeating updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .latent_spike_diffusion import ControlledLatentSpikeDiffusion
from .native_trajectory_cohort import TrajectoryCohort
from .provenance import sha256_file


def replay(acquisition, fields, cohort_config, probe_directory):
    acquisition, fields, cohort_config, probe_directory = map(Path,
        (acquisition, fields, cohort_config, probe_directory))
    record = json.loads((probe_directory / "full_unit_probe.json").read_text())
    if (record["status"] != "PASS_REAL_FULL_UNIT_IMPLEMENTATION_PROBE"
            or record["source_units"] != 100000 or record["complete_episode_bins"] != 1200
            or record["denoising_steps"] != 1000 or record["training_seed"] != 41):
        raise ValueError("complete declared development probe required")
    torch.set_num_threads(1)
    config = yaml.safe_load(cohort_config.read_text())
    cohort = TrajectoryCohort(acquisition, fields, config)
    if record["case_id"] != cohort.fit[0]["case_id"]:
        raise ValueError("probe complete fit episode differs")
    counts, controls = cohort.window(cohort.fit[0])
    model = ControlledLatentSpikeDiffusion(tuple(map(int,cohort.unit_ids)), input_size=1,
                                         bin_seconds=.005,diffusion_steps=1000)
    checkpoint = probe_directory / "full_unit_probe_checkpoint.pt"
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False))
    state = model.infer_state(torch.as_tensor(counts[:200],dtype=torch.float32)[None])
    future = torch.as_tensor(controls[200:300],dtype=torch.float32)[None]
    start = time.perf_counter()
    global_rng = torch.random.get_rng_state().clone()
    generated = model.simulate(state,future,latent_rng=torch.Generator().manual_seed(42000),
                               observation_rng=torch.Generator().manual_seed(43000))
    digest = hashlib.sha256(generated["counts"].numpy().tobytes()).hexdigest()
    if digest != record["generated_counts_sha256"] or not torch.equal(global_rng,torch.random.get_rng_state()):
        raise ValueError("LDNS full-unit replay or global RNG differs")
    field_counts = np.stack([np.bincount(cohort.unit_field_index,weights=row,minlength=len(cohort.field_ids))
                             for row in generated["counts"].numpy() [0]])
    np.savez_compressed(probe_directory / "replayed_outputs.npz",counts=generated["counts"].numpy(),
                        expected_counts=generated["expected_counts"].numpy(),field_counts=field_counts,
                        unit_ids=cohort.unit_ids,field_ids=cohort.field_ids)
    result = dict(status="PASS_FRESH_PROCESS_COMPLETE_UNIT_COUNT_DIGEST_REPLAY",
        generated_count_values=generated["counts"].numel(),unit_ids=100000,
        field_values=field_counts.size,source_unit_projection_checked=True,
        original_count_sha256=digest,checkpoint_sha256=sha256_file(checkpoint),
        source_cohort=cohort.binding,seconds=time.perf_counter()-start,
        saved_original_full_arrays_available=False,replayed_output_artifact="replayed_outputs.npz",
        training_repeated=False,scientific_claim_qualified=False)
    (probe_directory / "replay.json").write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("acquisition","fields","cohort-config","probe-directory"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    result = replay(args.acquisition,args.fields,args.cohort_config,args.probe_directory)
    print(json.dumps({k:v for k,v in result.items() if k!='source_cohort'}))


if __name__=='__main__':
    main()
