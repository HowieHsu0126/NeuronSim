"""Train the existing EAG baseline on complete source-ordered V1 episodes."""
from __future__ import annotations

import argparse
import copy
import json
import os

import yaml
from pathlib import Path

from . import latent_energy_training as shared
from .v1_population_training_data import inspect_inputs


def fixed_protocol():
    protocol = copy.deepcopy(shared.FIXED_PROTOCOL)
    for name in ("fit_blocks", "validation_blocks", "expected_poisson_rng_family", "history_bins", "horizon_bins"):
        del protocol[name]
    protocol.update(schema=shared.V1_SCHEMA, input_family="v1_qualified_source_episodes",
        training_sequence="complete_source_episode", partition_scope="original_whole_circuit_splits",
        bin_ticks=20, bin_ms=5.0, complete_episode_bins=600,
        forecast_origins_bins=[100,300,500], forecast_bins=100,
        profile="EAG_source_architecture_and_budget_with_V1_native_counts_and_controls")
    protocol["model"].update(input_size=6,complete_episode_bins=600,maximum_prefix_bins=500,
                             forecast_bins=100,bin_seconds=.005)
    return protocol


def validate_config(config, seed):
    if type(seed) is not int or seed not in (41,42,43):
        raise ValueError("registered EAG training seed41/42/43 required")
    for name,expected in fixed_protocol().items():
        actual=config.get(name)
        if name=="energy" and isinstance(actual,dict):
            actual={key:value for key,value in actual.items() if key!="microbatch_size"}
        if actual!=expected:
            raise ValueError("V1_EAG_COMPLETE_SOURCE_PROTOCOL_DIFFERS:"+name)
    declared=[]
    for name in ("fit_circuit_ids","validation_circuit_ids"):
        values=config.get(name)
        if (not isinstance(values,list) or not values or any(not isinstance(v,str) or not v for v in values)
                or len(set(values))!=len(values)):
            raise ValueError("V1_EAG_EXPLICIT_UNIQUE_CIRCUIT_PARTITION_REQUIRED:"+name)
        declared.append(set(values))
    if declared[0] & declared[1]:
        raise ValueError("V1_EAG_WHOLE_CIRCUIT_SPLIT_LEAKAGE")
    microbatch=config["energy"].get("microbatch_size")
    if type(microbatch) is not int or not 1<=microbatch<=config["energy"]["source_batch_size"]:
        raise ValueError("V1_EAG_MEASURED_MICROBATCH_REQUIRED")
    if config.get("device") not in ("cpu","cuda") or type(config.get("cpu_threads")) is not int or config["cpu_threads"] not in (1,2):
        raise ValueError("V1_EAG_SUPPORTED_DEVICE_AND_CPU_THREADS_REQUIRED")
    if config["device"]=="cpu" and os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("CPU training requires CUDA_VISIBLE_DEVICES empty")


def train(inputs, config, output, seed, *, resume=False, stop_after_commits=None):
    return shared.train(inputs,None,config,output,seed,resume,
                        stop_after_commits=stop_after_commits)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("inputs","config"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--check-inputs",action="store_true")
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stop-after-commits",type=int)
    args=parser.parse_args()
    if args.check_inputs:
        config=yaml.safe_load(args.config.read_text())
        validate_config(config,args.seed)
        inspect_inputs(args.inputs,config,config_validator=validate_config)
        print(json.dumps(dict(status="V1_EAG_INPUT_METADATA_ADMITTED",training_seed=args.seed,
                              source_arrays_read=False,model_created=False)))
        return
    if args.output is None:
        parser.error("--output is required for training")
    print(json.dumps(train(args.inputs,args.config,args.output,args.seed,
                           resume=args.resume,stop_after_commits=args.stop_after_commits)))


if __name__=="__main__":
    main()
