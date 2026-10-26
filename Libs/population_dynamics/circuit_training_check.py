"""One existing model update through the complete-episode data interface.

This diagnostic establishes consumer compatibility, not trained-model quality.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from math import gcd
from pathlib import Path

import numpy as np
import torch

from .circuit_data import CircuitDataset
from .poisson_state_space import ControlledPoissonLDS, LatentRandomStreams
from .pfc_renewal_data import PfcRenewalEpisode
from .population_state import ControlledPopulationState
from .provenance import freeze_json, sha256_file


def check(dataset_path: Path, output: Path, *, source_run: Path | None = None,
          source_entry: Path | None = None, episode_id: str | None = None, seed: int = 41):
    if (source_run is None) != (source_entry is None):
        raise ValueError("PFC_SOURCE_RUN_AND_PINNED_ENTRY_REQUIRED_TOGETHER")
    if type(seed) is not int or seed not in (41, 42, 43):
        raise ValueError("REGISTERED_TRAINING_SEED_REQUIRED")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = CircuitDataset(dataset_path, splits=("train", "development"))
    row = data.episodes[episode_id] if episode_id is not None else next(data.iter_episodes(), None)
    if row is None:
        raise ValueError("NO_TRAIN_OR_DEVELOPMENT_EPISODES")
    duration = row["duration_ticks"]
    if duration < 2:
        raise ValueError("CIRCUIT_CONSUMER_REQUIRES_TWO_SOURCE_TICKS")
    # Retain 5 ms bins when exact on the source clock. A common divisor covers
    # the complete episode; never round away a final partial bin or source tick.
    preferred_ticks = Fraction("5") / Fraction(str(row["dt_ms"]))
    bin_ticks = gcd(duration, int(preferred_ticks)) if preferred_ticks.denominator == 1 else 1
    if duration // bin_ticks < 2:
        bin_ticks = 1
    history_ticks = (duration // bin_ticks // 2) * bin_ticks
    horizon_ticks = duration - history_ticks
    sequence = data.complete_episode(row["episode_id"], bin_ticks=bin_ticks)
    counts = torch.as_tensor(sequence["counts"], dtype=torch.float32)[None]
    control = torch.as_tensor(sequence["known_controls"], dtype=torch.float32)[None]
    adapter = None
    if source_run is None:
        model = ControlledPoissonLDS(tuple(map(int, sequence["unit_ids"])), control.shape[-1], 4, 8,
                                    sequence["bin_seconds"], initialization_seed=seed)
        inputs = (counts, control)
    else:
        adapter = PfcRenewalEpisode(data, row["episode_id"], bin_ticks=bin_ticks,
                                   source_run=source_run, source_entry=source_entry)
        observation = adapter.renewal_window(0, duration)
        if not np.array_equal(sequence["counts"], observation.counts) or not np.array_equal(
                sequence["known_controls"], observation.stimulus):
            raise ValueError("PFC_POPULATION_SOURCE_OBSERVATIONS_DISAGREE")
        model = ControlledPopulationState(unit_ids=list(map(int, adapter.unit_ids)),
            unit_field_index=adapter.unit_field_index, field_names=adapter.field_names,
            minimum_interval_ticks=adapter.minimum_interval_ticks, tick_seconds=adapter.dt_ms / 1000,
            bin_seconds=sequence["bin_seconds"], input_size=control.shape[-1],
            initialization_seed=seed, metadata=adapter.population_metadata())
        inputs = (counts, control, torch.as_tensor(observation.eligible_ticks)[None],
                  torch.Generator().manual_seed(61001 + seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    terms = model.sequence_loss(*inputs)
    terms["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or any(not torch.isfinite(value).all() for value in gradients):
        raise ValueError("CIRCUIT_CONSUMER_GRADIENT_INVALID")
    optimizer.step()
    changed = any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
    numbers = {name: float(value.detach()) for name, value in terms.items()}
    if not changed or not all(np.isfinite(v) for v in numbers.values()):
        raise ValueError("CIRCUIT_CONSUMER_UPDATE_FAILED")
    report = dict(status="ENGINEERING_UPDATE_PASS", source_episode=row["episode_id"], split=row["split"],
                  units=len(sequence["unit_ids"]), complete_sequence_bins=counts.shape[1],
                  source_duration_ticks=duration, source_dt_ms=row["dt_ms"],
                  consumed_start_tick=0, consumed_stop_tick_exclusive=duration,
                  history_ticks=history_ticks, horizon_ticks=horizon_ticks,
                  bin_ticks=bin_ticks, bin_seconds=sequence["bin_seconds"],
                  training_seed=seed, optimizer_updates=1, terms=numbers,
                  gradient_finite=True, parameters_changed=True,
                  formal_training_completed=False, generative_fidelity_qualified=False,
                  dataset_sha256=sha256_file(dataset_path / "dataset.json"),
                  torch_version=torch.__version__)
    if adapter is not None:
        boundary = model.infer_state(dict(counts=counts[:, :history_ticks // bin_ticks],
            stimulus=control[:, :history_ticks // bin_ticks],
            ready_in_ticks=adapter.ready_at(history_ticks)[None]))
        report.update(model_family="metadata_conditioned_population", source_binding=adapter.binding,
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      parameter_shapes={n: list(p.shape) for n, p in model.named_parameters()},
                      boundary_latent_shape=list(boundary.mean.shape),
                      source_event_count=len(adapter.event_ticks), scientific_claim_qualified=False)
        # One model bin checks all-unit generation. It is not a fidelity trial.
        future = control[:, history_ticks // bin_ticks:history_ticks // bin_ticks + 1]
        step = next(model.iter_simulation(boundary, future,
            LatentRandomStreams.from_seeds(77001, 77002, 77003), 1, record_events=True))
        replay = next(model.iter_simulation(boundary, future,
            LatentRandomStreams.from_seeds(77001, 77002, 77003), 1, record_events=True))
        if any(not torch.equal(getattr(step, name), getattr(replay, name))
               for name in ("counts", "events", "latent", "ready_in_ticks", "initial_latent")):
            raise ValueError("PFC_POPULATION_SAMPLING_REPLAY_DIFFERS")
        report.update(generated_unit_count_values=step.counts.numel(), generated_events=len(step.events),
                      replay_exact=True, generated_bins=1,
                      unit_feature_schema=model.metadata["unit_feature_schema"],
                      field_feature_schema=model.metadata["field_feature_schema"])
    freeze_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--source-entry", type=Path)
    parser.add_argument("--episode-id")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args(argv)
    print(json.dumps(check(args.dataset, args.output, source_run=args.source_run,
                           source_entry=args.source_entry, episode_id=args.episode_id, seed=args.seed)))


if __name__ == "__main__":
    main()
