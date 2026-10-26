"""Real-input full-unit renewal path: exposure, update, event sampling and replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from .native_count_windows import NativeEpisodeCounts
from .nest_gpu_network import ticks
from .poisson_state_space import LatentRandomStreams
from .provenance import sha256_file
from .refractory_observation import renewal_window
from .renewal_state_space import ControlledRenewalLDS


def replay_exposure(data, window, start_tick, stop_tick, intervals):
    """Direct tick recursion for a fixed evenly spaced set of observed units."""
    selected = np.unique(np.linspace(0, len(data.unit_ids) - 1, min(64, len(data.unit_ids)), dtype=int))
    for unit in selected:
        events = data.event_ticks[(data.event_units == unit) & (data.event_ticks < stop_tick)]
        before = events[events < start_tick]
        last = int(before[-1]) if len(before) else -int(intervals[unit])
        initial = max(last + int(intervals[unit]) - start_tick, 0)
        if initial != int(window.initial_ready_in_ticks[unit]):
            raise ValueError("independent observed boundary availability mismatch")
        emitted = set(map(int, events[events >= start_tick]))
        counts = np.zeros(len(window.counts), dtype=np.int64)
        exposure = counts.copy()
        for tick in range(start_tick, stop_tick):
            index = (tick - start_tick) // data.bin_ticks
            exposure[index] += tick >= last + intervals[unit]
            if tick in emitted:
                counts[index] += 1
                last = tick
        if (not np.array_equal(counts, window.counts[:, unit])
                or not np.array_equal(exposure, window.eligible_ticks[:, unit])
                or max(last + int(intervals[unit]) - stop_tick, 0) != window.terminal_ready_in_ticks[unit]):
            raise ValueError("independent native-tick exposure recursion mismatch")
    return dict(selected_unit_ids=data.unit_ids[selected].tolist(), units=len(selected),
                native_unit_ticks=(stop_tick-start_tick)*len(selected),
                count_and_exposure_values=2*len(selected)*len(window.counts), exact=True)


def replay_events(counts, events, ready, intervals, bin_ticks):
    """Independent sparse-event histogram and per-unit chronological support."""
    if (events.ndim != 2 or events.shape[1] != 4 or counts.ndim != 4
            or events.dtype.kind not in "iu" or counts.dtype.kind not in "iu"):
        raise ValueError("integer identified event and count arrays required")
    draws, batch, bins, units = counts.shape
    if batch != 1:
        raise ValueError("this real-input replay expects one observed episode")
    if (np.any(events < 0) or np.any(events[:, 0] >= draws)
            or np.any(events[:, 1] >= batch) or np.any(events[:, 2] >= units)
            or np.any(events[:, 3] >= bins * bin_ticks)):
        raise ValueError("sample event identity or emission clock outside support")
    histogram = np.zeros_like(counts)
    np.add.at(histogram, (events[:, 0], events[:, 1], events[:, 3] // bin_ticks, events[:, 2]), 1)
    if not np.array_equal(histogram, counts):
        raise ValueError("generated events do not reconstruct generated counts")
    key = events[:, 0] * units + events[:, 2]
    order = np.lexsort((events[:, 3], key))
    same = np.diff(key[order]) == 0
    gaps = np.diff(events[order, 3])[same]
    if np.any(gaps < intervals[events[order[1:][same], 2]]):
        raise ValueError("generated event train violates the source minimum interval")
    if np.any(events[:, 3] < ready[0, events[:, 2]]):
        raise ValueError("generated events violate observed boundary refractory history")
    last = np.tile(ready[0] - intervals, (draws, 1)).reshape(-1)
    np.maximum.at(last, key, events[:, 3])
    terminal = np.maximum(last.reshape(draws, 1, units) + intervals - bins * bin_ticks, 0)
    return dict(count_values=int(counts.size), sampled_events=len(events),
                minimum_inter_spike_ticks=int(gaps.min()) if len(gaps) else None,
                refractory_violations=0, boundary_violations=0), terminal


def run(episode: Path, fields: Path, config_path: Path, output: Path, replay=False):
    config = yaml.safe_load(config_path.read_text())
    if (config["schema"] != "neural_population_renewal_implementation_check_v1"
            or config["formal_training_seeds"] != [41, 42, 43]
            or config["minimum_interval_policy"] != "nest_gpu_iaf_psc_exp_grid"):
        raise ValueError("declared engineering configuration required")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    data = NativeEpisodeCounts(episode, fields)
    source = json.loads((episode / "report.json").read_text())
    if source["gpu_source_revision"] != "74fc021cad2b4bd5c011c390b3f95a302b663483":
        raise ValueError("minimum-interval policy requires the inspected native source revision")
    if sha256_file(episode / "initial_parameters.npz") != source["artifacts"]["initial_parameters.npz"]:
        raise ValueError("native neuron parameters changed")
    with np.load(episode / "initial_parameters.npz", allow_pickle=False) as p:
        if (not np.array_equal(p["unit_ids"], data.unit_ids)
                or np.any(p["V_reset_rel"] >= p["Theta_rel"])):
            raise ValueError("source unit order/reset semantics incompatible with dead time")
        ref = p["t_ref"].astype(np.float64) / data.dt_ms
        if not np.allclose(ref, np.rint(ref), atol=1e-6, rtol=0):
            raise ValueError("off-grid refractory periods require a separate native policy")
        # Native iaf_psc_exp.cu: a spike sets round(t_ref/dt); each following
        # positive-counter update only decrements. Integration resumes one tick
        # after the final decrement, so earliest next emission is R+1 ticks.
        intervals = np.rint(ref).astype(np.int64) + 1
    def clock(name):
        return int(ticks([config[name]], data.dt_ms)[0])
    training_start, training_stop = clock("training_start_ms"), clock("training_stop_ms")
    history_start, forecast_start, forecast_stop = (clock(k) for k in
        ("history_start_ms", "forecast_start_ms", "forecast_stop_ms"))
    binding = dict(config=config, native_report_sha256=sha256_file(episode / "report.json"),
                   field_summary_sha256=sha256_file(fields / "summary.json"),
                   model_sha256=sha256_file(Path(__file__).with_name("renewal_state_space.py")),
                   observation_sha256=sha256_file(Path(__file__).with_name("refractory_observation.py")))
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if replay:
        report = json.loads((output / "run.json").read_text())
        if report["binding"] != binding or report["status"] != "ENGINEERING_EXECUTED":
            raise ValueError("replay requires the same completed implementation and inputs")
        if sha256_file(output / "model.pt") != report["model_sha256"]:
            raise ValueError("replay model checkpoint changed")
        model = ControlledRenewalLDS.load(output / "model.pt", device=device)
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        model = ControlledRenewalLDS(unit_ids=list(map(int, data.unit_ids)), input_size=1,
                    latent_size=config["latent_size"], encoder_size=config["encoder_size"],
                    bin_seconds=data.bin_ms / 1000, initialization_seed=config["training_seed"],
                    tick_seconds=data.dt_ms / 1000, minimum_interval_ticks=intervals).to(device)
        training = renewal_window(data, training_start, training_stop, intervals)
        exposure_replay = replay_exposure(data, training, training_start, training_stop, intervals)
        (output / "exposure_replay.json").write_text(json.dumps(exposure_replay, indent=2) + "\n")
        successes, exposure = training.counts.sum(0, dtype=np.float64), training.eligible_ticks.sum(0, dtype=np.float64)
        probability = ((successes + config["initialization_pseudocount"])
                       / (exposure + 2 * config["initialization_pseudocount"]))
        with torch.no_grad():
            model.latent_model.log_rate_bias.copy_(torch.as_tensor(
                np.log(-np.log1p(-probability) / model.tick_seconds), device=device))
            model.latent_model.transition.copy_(config["initial_transition_diagonal"]
                                                * torch.eye(config["latent_size"], device=device))
        y = torch.as_tensor(training.counts, device=device).unsqueeze(0)
        e = torch.as_tensor(training.eligible_ticks, device=device).unsqueeze(0)
        u = torch.as_tensor(training.stimulus, device=device).unsqueeze(0)
        actual_device = model.latent_model.loading.device
        generator = torch.Generator(device=actual_device).manual_seed(config["variational_seed"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        optimization = []
        for update in range(config["training_updates"]):
            optimizer.zero_grad(set_to_none=True)
            terms = model.sequence_loss(y, u, e, generator)
            terms["loss"].backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_norm_limit"], error_if_nonfinite=True)
            optimizer.step()
            optimization.append(dict(update=update, gradient_norm=float(gradient),
                                     **{k: float(v.detach()) for k, v in terms.items()}))
        model.save(output / "model.pt")
        torch.save(dict(optimizer=optimizer.state_dict(), variational_rng=generator.get_state()),
                   output / "optimization_state.pt")
        (output / "optimization.json").write_text(json.dumps(optimization, indent=2) + "\n")
        del y, e, u, training, optimizer, terms
    history = renewal_window(data, history_start, forecast_start, intervals)
    observed = dict(counts=history.counts[None], stimulus=history.stimulus[None],
                    ready_in_ticks=history.terminal_ready_in_ticks[None])
    state = model.infer_state(observed)
    _, stimulus = data.window(forecast_start, forecast_stop)
    generated = model.simulate(state, stimulus[None], LatentRandomStreams.from_seeds(
                              *config["sampling_seeds"], device=device), config["samples"])
    arrays = {name: getattr(generated, name).cpu().numpy() for name in
              ("counts", "events", "latent_states", "initial_latent_states", "terminal_ready_in_ticks")}
    arrays.update(unit_ids=data.unit_ids, minimum_interval_ticks=intervals,
                  initial_ready_in_ticks=history.terminal_ready_in_ticks[None])
    support, terminal = replay_events(arrays["counts"], arrays["events"], arrays["initial_ready_in_ticks"],
                                     intervals, model.bin_ticks)
    if not np.array_equal(terminal, arrays["terminal_ready_in_ticks"]):
        raise ValueError("generated terminal refractory history differs from independent event replay")
    if replay:
        if sha256_file(output / "samples.npz") != report["samples_sha256"]:
            raise ValueError("archived generated samples changed")
        with np.load(output / "samples.npz", allow_pickle=False) as original:
            if set(original.files) != set(arrays) or any(not np.array_equal(original[k], v) for k, v in arrays.items()):
                raise ValueError("fresh-process generated arrays do not match")
        result = dict(status="EXACT_FRESH_PROCESS_REPLAY", support=support,
                      elapsed_seconds=time.perf_counter() - started, scientific_fidelity_qualified=False)
        (output / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        np.savez_compressed(output / "samples.npz", **arrays)
        result = dict(status="ENGINEERING_EXECUTED", binding=binding, units=len(data.unit_ids),
                      training_updates=config["training_updates"], training_bins=(training_stop-training_start)//data.bin_ticks,
                      forecast_bins=(forecast_stop-forecast_start)//data.bin_ticks, support=support,
                      minimum_interval_ticks=sorted(set(map(int, intervals))),
                      elapsed_seconds=time.perf_counter() - started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                      native_rng_family=source.get("native_poisson_rng_family", "not_reported"),
                      model_sha256=sha256_file(output / "model.pt"), samples_sha256=sha256_file(output / "samples.npz"),
                      formal_training_completed=False, scientific_fidelity_qualified=False,
                      inference_scope="in-sample engineering on one source episode")
        (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("episode", "fields", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    a = parser.parse_args()
    print(json.dumps(run(a.episode, a.fields, a.config, a.output, a.replay)))


if __name__ == "__main__":
    main()
