"""Bounded all-unit V1 native-observation integration check; no model selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .circuit_data import CircuitDataset
from .population_state import ControlledPopulationState
from .provenance import sha256_file
from .v1_renewal_data import V1RenewalEpisode


def tick_oracle(episode: V1RenewalEpisode, start: int, stop: int):
    """Count eligibility one source tick at a time over a bounded all-unit window."""
    n = len(episode.unit_ids)
    counts = np.zeros(((stop-start)//episode.bin_ticks, n), dtype=np.int64)
    eligible = np.zeros_like(counts)
    ready = np.ones(n, dtype=np.int64)
    before = np.searchsorted(episode.event_ticks, start, side="left")
    np.maximum.at(ready, episode.event_units[:before],
                  episode.event_ticks[:before] + episode.minimum_interval_ticks[episode.event_units[:before]])
    initial = np.maximum(ready-start, 0)
    for tick in range(start, stop):
        index = (tick-start)//episode.bin_ticks
        eligible[index] += tick >= ready
        left, right = np.searchsorted(episode.event_ticks, [tick, tick+1], side="left")
        units = episode.event_units[left:right]
        if len(units):
            if np.any(tick < ready[units]):
                raise ValueError("V1_ORACLE_EVENT_DURING_REFRACTORY_INTERVAL")
            np.add.at(counts[index], units, 1)
            ready[units] = tick + episode.minimum_interval_ticks[units]
    return counts, eligible, initial, np.maximum(ready-stop, 0)


def check(dataset: Path, episode_id: str, native_parameters: Path, output: Path, *, bin_ticks: int = 20):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("V1_NATIVE_OBSERVATION_CHECK_REQUIRES_COMPUTE_ALLOCATION")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status="FAILED", scope="bounded_native_observation_engineering_check",
        observed_utc=datetime.now(timezone.utc).isoformat(), slurm_job_id=os.environ["SLURM_JOB_ID"],
        source_dataset=str(dataset), episode_id=episode_id, native_parameters=str(native_parameters),
        bin_ticks=bin_ticks, training_seeds=[41,42,43], formal_training_completed=False,
        validation_outcomes_accessed=False, calibration_outcomes_accessed=False, test_outcomes_accessed=False,
        source_simulations_executed=0, scientific_claim_qualified=False)
    try:
        torch.set_num_threads(1)
        data = CircuitDataset(dataset, splits=("train",))
        episode = V1RenewalEpisode(data, episode_id, bin_ticks=bin_ticks, native_parameters=native_parameters)
        width = 2 * bin_ticks
        if width > episode.duration_ticks:
            raise ValueError("TWO_COMPLETE_MODEL_BINS_REQUIRED")
        starts = sorted({0, min((episode.duration_ticks//2//bin_ticks)*bin_ticks, episode.duration_ticks-width),
                         episode.duration_ticks-width})
        windows = []
        for start in starts:
            stop = start+width
            value = episode.renewal_window(start,stop)
            expected = tick_oracle(episode,start,stop)
            for name, actual, wanted in zip(("counts","exposure","initial_ready","terminal_ready"),
                    (value.counts,value.eligible_ticks,value.initial_ready_in_ticks,value.terminal_ready_in_ticks),expected):
                if not np.array_equal(actual,wanted):
                    raise ValueError("V1_NATIVE_TICK_ORACLE_DIFFERS:"+name)
            for tick, wanted in ((start, expected[2]), (stop, expected[3])):
                if not np.array_equal(episode.ready_at(tick), wanted):
                    raise ValueError("V1_NATIVE_CAUSAL_BOUNDARY_DIFFERS")
            windows.append(dict(start_tick=start, stop_tick_exclusive=stop, units=len(episode.unit_ids),
                                observed_events=int(value.counts.sum()), eligible_ticks=int(value.eligible_ticks.sum()),
                                all_counts_exposures_and_ready_states_exact=True))
        metadata = episode.population_metadata()
        value = episode.renewal_window(0,width)
        inputs = [torch.as_tensor(x,dtype=torch.float32).unsqueeze(0)
                  for x in (value.counts,value.stimulus,value.eligible_ticks)]
        updates=[]
        for seed in (41,42,43):
            model=ControlledPopulationState(unit_ids=list(map(int,episode.unit_ids)),
                unit_field_index=episode.unit_field_index.tolist(),field_names=list(episode.field_names),
                minimum_interval_ticks=episode.minimum_interval_ticks.tolist(),tick_seconds=episode.dt_ms/1000,
                bin_seconds=episode.dt_ms/1000*bin_ticks,input_size=6,initialization_seed=seed,
                metadata=metadata)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            before=model.shared_unit_readout.weight.detach().clone()
            terms=model.sequence_loss(*inputs,torch.Generator().manual_seed(seed))
            terms["loss"].backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("V1_POPULATION_NATIVE_GRADIENT_INVALID")
            optimizer.step()
            if torch.equal(before,model.shared_unit_readout.weight):
                raise ValueError("V1_POPULATION_NATIVE_UPDATE_MISSING")
            updates.append(dict(seed=seed, loss=float(terms["loss"].detach()),
                parameter_count=sum(p.numel() for p in model.parameters()), all_parameter_gradients_finite=True,
                shared_observation_weights_updated=True, units=len(episode.unit_ids), bins=2))
            del model,optimizer,terms
        report.update(status="PASS", binding=episode.binding, source_units=len(episode.unit_ids),
            source_fields=len(episode.field_names), source_node_types=len(episode.parameters),
            minimum_interval_ticks=[int(episode.minimum_interval_ticks.min()),int(episode.minimum_interval_ticks.max())],
            source_duration_ticks=episode.duration_ticks, raw_events=len(episode.event_ticks),
            endpoint_events_retained=int(np.count_nonzero(episode.event_ticks==episode.duration_ticks)),
            windows=windows, updates=updates, unit_feature_count=len(metadata["unit_feature_schema"]),
            field_feature_count=len(metadata["field_feature_schema"]), legal_visual_controls=len(metadata["control_features"]),
            implementation_sha256={name:sha256_file(Path(__file__).with_name(name))
                                   for name in ("v1_renewal_data.py","v1_renewal_check.py")})
        return report
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--episode-id",required=True)
    parser.add_argument("--native-parameters",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--bin-ticks",type=int,default=20)
    args=parser.parse_args(argv)
    result=check(args.dataset,args.episode_id,args.native_parameters,args.output,bin_ticks=args.bin_ticks)
    print(json.dumps(dict(status=result["status"],source_units=result["source_units"],windows=len(result["windows"]),
                          training_seeds=result["training_seeds"],output=str(args.output))))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
