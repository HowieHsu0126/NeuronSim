"""Complete native episode update, full-unit sampling and fresh-process replay.

This is an implementation and cost check. One in-sample episode does not
establish convergence, held-out fidelity or scientific source qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .native_renewal_data import NativeRenewalEpisode
from .population_state import ControlledPopulationState
from .population_trajectory_evaluation import sample_population_outputs
from .poisson_state_space import LatentRandomStreams
from .poisson_trajectory_training import _trainer_sources
from .provenance import sha256_file
from .renewal_check import replay_events
from .renewal_trajectory_training import (initialize_observation, model_configuration,
                                         validate_config, variational_seed)


def run(episode, fields, config_path, output, *, device_name="cpu", seed=41, replay=False):
    config = yaml.safe_load(config_path.read_text())
    config["device"] = device_name
    validate_config(config, seed, backend="population")
    device = torch.device(device_name)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    data = NativeRenewalEpisode(episode, fields)
    if (len(data.unit_ids) != 100000 or data.duration_ticks * data.dt_ms != 6000
            or data.bin_ms != 5 or data.dt_ms != .1
            or data.source_identity["native_poisson_rng_family"] != config["expected_poisson_rng_family"]):
        raise ValueError("complete100000unit6s Philox development episode required")
    binding = dict(configuration=config, training_seed=seed, native=data.binding,
                   implementation_sources=_trainer_sources("population"),
                   check_implementation_sha256=sha256_file(Path(__file__)),
                   evaluator_implementation_sha256=sha256_file(Path(__file__).with_name("population_trajectory_evaluation.py")))
    if replay:
        report = json.loads((output / "run.json").read_text())
        if report["status"] != "ENGINEERING_EXECUTED" or report["binding"] != binding:
            raise ValueError("same completed update, native inputs and implementation required")
        for name in ("model.pt", "samples.npz", "optimization_state.pt"):
            if sha256_file(output / name) != report["artifacts"][name]:
                raise ValueError("implementation output changed:" + name)
        model = ControlledPopulationState.load(output / "model.pt", device=device)
    else:
        output.mkdir(parents=True, exist_ok=False)
        cohort = SimpleNamespace(unit_ids=data.unit_ids, unit_field_index=data.unit_field_index,
            field_ids=data.field_ids, minimum_interval_ticks=data.minimum_interval_ticks, dt_ms=data.dt_ms,
            fit=[dict(case_id=episode.name)])
        started = time.perf_counter()
        training = data.renewal_window(0, data.duration_ticks)
        cohort.window = lambda case: training
        model = ControlledPopulationState(**model_configuration(cohort, config, seed, "population")).to(device)
        initialization = initialize_observation(model, cohort, config["initialization_pseudocount"], backend="population")
        preparation_seconds = time.perf_counter() - started
        y = torch.as_tensor(training.counts[None], device=device)
        u = torch.as_tensor(training.stimulus[None], device=device)
        exposure = torch.as_tensor(training.eligible_ticks[None], device=device)
        if y.shape != (1, 1200, 100000):
            raise ValueError("full complete-episode input required")
        generator = torch.Generator(device=model.unit_loading.device).manual_seed(
            variational_seed(config["training_variational_seed_base"], seed, "training"))
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        before = model.unit_loading.detach().clone()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        terms = model.sequence_loss(y, u, exposure, generator)
        terms["loss"].backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_norm_limit"], error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":torch.cuda.synchronize(device)
        update_seconds = time.perf_counter() - started
        if torch.equal(before, model.unit_loading):
            raise ValueError("complete-episode update did not change observation loading")
        optimization = dict(**{name:float(value.detach()) for name,value in terms.items()},
            gradient_norm=float(gradient),update_seconds=update_seconds,preparation_seconds=preparation_seconds,
            training_input_shape=list(y.shape),complete_episode_updates=1,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
            initialization=initialization)
        model.save(output / "model.pt")
        torch.save(dict(optimizer=optimizer.state_dict(),variational_rng=generator.get_state()),output / "optimization_state.pt")
        (output / "optimization.json").write_text(json.dumps(optimization,indent=2)+'\n')
        del y,u,exposure,training,optimizer,terms,before
    model.eval()
    count,stimulus=data.window(0,data.duration_ticks)
    boundary = 200  # one full second of observed causal history
    horizon = 100  # matched500ms output at5ms bins
    ready=data.ready_at(boundary*data.bin_ticks)
    started=time.perf_counter()
    with torch.no_grad():
        state=model.infer_state(dict(counts=count[None,:boundary],stimulus=stimulus[None,:boundary],ready_in_ticks=ready[None]))
        generated=sample_population_outputs(model,state,stimulus[None,boundary:boundary+horizon],
            LatentRandomStreams.from_seeds(4186000,4186001,4186002,device=model.unit_loading.device),2,record_events=True)
    if device.type=="cuda":torch.cuda.synchronize(device)
    generation_seconds=time.perf_counter()-started
    names=("counts","latent_states","initial_latent_states","terminal_ready_in_ticks","events")
    arrays={name:value.cpu().numpy() for name,value in zip(names,generated)}
    arrays.update(unit_ids=data.unit_ids,unit_field_index=data.unit_field_index,field_ids=data.field_ids,
        initial_ready_in_ticks=ready[None],minimum_interval_ticks=data.minimum_interval_ticks)
    support,terminal=replay_events(arrays["counts"],arrays["events"],ready[None],data.minimum_interval_ticks,data.bin_ticks)
    if not np.array_equal(terminal,arrays["terminal_ready_in_ticks"]):
        raise ValueError("event-derived terminal availability differs")
    projected=np.zeros((*arrays["counts"].shape[:3],len(data.field_ids)),dtype=np.int64)
    for field in range(len(data.field_ids)):
        projected[...,field]=arrays["counts"][...,data.unit_field_index==field].sum(-1,dtype=np.int64)
    arrays["field_counts"]=projected
    if replay:
        with np.load(output / "samples.npz",allow_pickle=False) as saved:
            if set(saved.files)!=set(arrays):raise ValueError("sample output schema differs")
            for name,value in arrays.items():
                if saved[name].dtype!=value.dtype or not np.array_equal(saved[name],value):
                    raise ValueError("fresh-process elementwise replay differs:"+name)
        result=dict(status="PASS_EXACT_FULL_UNIT_EVENT_FIELD_AND_LATENT_REPLAY",binding=binding,
            support=support,compared_values=sum(a.size for a in arrays.values()),generation_seconds=generation_seconds,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        with (output / "replay.json").open("x") as stream:json.dump(result,stream,indent=2)
    else:
        np.savez_compressed(output / "samples.npz",**arrays)
        result=dict(status="ENGINEERING_EXECUTED",binding=binding,core_neurons=100000,episode_ms=6000,
            optimization=optimization,generation_seconds=generation_seconds,support=support,samples=2,forecast_ms=500,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            artifacts={name:sha256_file(output / name) for name in ("model.pt","samples.npz","optimization_state.pt")},
            source_qualified=False,formal_training_completed=False,scientific_claim_qualified=False,
            inference_scope="one in-sample complete episode; implementation and cost only")
        (output / "run.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("episode","fields","config","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--seed",type=int,choices=[41,42,43],default=41)
    parser.add_argument("--replay",action="store_true")
    args=parser.parse_args()
    run(args.episode,args.fields,args.config,args.output,device_name=args.device,seed=args.seed,replay=args.replay)


if __name__=="__main__":main()
