"""Uniform unit-window pilot of conditional PFC second-spike probabilities.

Samples include no-spike windows. Their mean targets the archived histories'
transformed finite-population mean, not their realized multispike frequency.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import yaml

from . import pfc_deterministic_replay as source
from . import pfc_native_drive as native
from . import pfc_coupled_drive as coupled
from . import pfc_first_passage_reference as reference
from . import pfc_first_passage_importance as importance
from .pfc_fluctuation_analysis import unit_mapping
from .pfc_delay_analysis import PHASES


def require(condition, message):
    if not condition:
        raise ValueError("PFC_FIRST_PASSAGE_HISTORY_" + message)


def create_input_drive(config, case, source_report, kind):
    """Use the bound source's actual native or coupled input construction."""
    if kind == "native_numpy_stream":
        require("coarse_dt_ms" not in source_report, "NATIVE_MODE_WITH_COUPLED_SOURCE")
        return native.create_drive(config, case)
    require(kind == "coupled_binomial_clock_refinement_stream", "INPUT_STREAM_KIND")
    require(source_report["resolved_config"] == config and source_report["selected_case"] == case,
            "COUPLED_SOURCE_IDENTITY")
    return coupled.create_coupled_drive(config, case, source_report["coarse_dt_ms"],
                                        source_report["refinement_seeds"])


def prefix_state_arrays(drive, kind):
    """Observe the pinned drive without advancing it or adding a resume API.

    CoupledFineDrive has no public prefix-state exporter. Its version is bound
    to the original source receipt by the launcher; these explicit internals
    preserve the three RNGs and buffered child rows at an incomplete prefix.
    """
    common = dict(prefix_kind=np.asarray(kind), fine_ticks=np.asarray(drive.ticks),
                  fine_total_ticks=np.asarray(drive.total_ticks),
                  events_E=np.asarray(drive.event_totals["E"]), events_I=np.asarray(drive.event_totals["I"]))
    if kind == "native_numpy_stream":
        require(isinstance(drive, native.NativeDrive), "NATIVE_PREFIX_TYPE")
        return {**{k:np.array(v, copy=True) for k,v in native._state_arrays(drive.get_state()).items()}, **common}
    require(kind == "coupled_binomial_clock_refinement_stream" and isinstance(drive, coupled.CoupledFineDrive),
            "COUPLED_PREFIX_TYPE")
    factor, offset = drive.factor, drive.ticks % drive.factor
    rows = drive._rows
    require(rows.coarse.ticks == (drive.ticks+factor-1)//factor
            and drive._buffer_offset == offset, "COUPLED_PREFIX_CURSOR")
    if offset:
        require(drive._buffer is not None and drive._buffer.dtype == np.uint16
                and drive._buffer.shape == (factor, drive.n_e+drive.n_i), "COUPLED_PREFIX_BUFFER")
        buffer = drive._buffer.copy()
    else:
        require(drive._buffer is None, "COUPLED_PREFIX_UNEXPECTED_BUFFER")
        buffer = np.empty((0,drive.n_e+drive.n_i), dtype=np.uint16)
    states = {k:np.array(v,copy=True) for k,v in rows.state_arrays().items()}
    return {**states, **common, "coarse_ticks":np.asarray(rows.coarse.ticks),
            "coarse_events_E":np.asarray(rows.coarse.event_totals["E"]),
            "coarse_events_I":np.asarray(rows.coarse.event_totals["I"]),
            "refinement_factor":np.asarray(factor), "buffer_offset":np.asarray(offset),
            "buffered_parent_children":buffer}


def sample_frame(config, *, field, phase, window_ms, sample_size, seed_components):
    mapping, _ = unit_mapping(config["excitatory_neurons"], config["inhibitory_neurons"],
                              config["excitatory_preference_bins"])
    units = np.flatnonzero(mapping[:config["excitatory_neurons"]] == field)
    start_ms, stop_ms = PHASES[phase]
    windows = round((stop_ms - start_ms) / window_ms)
    require(len(units) > 0 and windows > 0 and windows * window_ms == stop_ms - start_ms,
            "COMPLETE_SAMPLING_FRAME")
    size = len(units) * windows
    require(type(sample_size) is int and 0 < sample_size <= size, "SAMPLE_SIZE")
    rng = np.random.RandomState(np.asarray(seed_components, dtype=np.uint32))
    ids = rng.choice(size, size=sample_size, replace=False)
    rows = []
    for index, item in enumerate(ids):
        start = start_ms + (int(item) // len(units)) * window_ms
        lo, hi = round(start / config["dt_ms"]), round((start + window_ms) / config["dt_ms"])
        require(lo * config["dt_ms"] == start and hi * config["dt_ms"] == start + window_ms,
                "PHYSICAL_WINDOW_CLOCK")
        rows.append(dict(sample_index=index, frame_index=int(item), unit_index=int(units[int(item) % len(units)]),
                         window_start_ms=start, window_stop_ms=start + window_ms,
                         window_start_tick=lo, window_stop_tick=hi))
    metadata = dict(field_index=field, population="E", phase=phase, phase_start_ms=start_ms,
                    phase_stop_ms=stop_ms, window_ms=window_ms, units=len(units), windows_per_unit=windows,
                    frame_size=size, sample_size=sample_size, seed_components=seed_components,
                    sampling="uniform_without_replacement_before_event_inspection",
                    unit_order="source_indices_ascending", frame_order="window_major_then_unit")
    return rows, metadata


def classify(config, tape, rows):
    dt = config["dt_ms"] * 1e-3
    refractory = source._constants(config, "E")["refractory"]
    refractory_ticks = int(np.asarray((refractory + 1e-3 * dt) / dt, dtype=np.int64))
    result = []
    for original in rows:
        row = dict(original)
        lo = np.searchsorted(tape.events[:, 0], row["window_start_tick"])
        hi = np.searchsorted(tape.events[:, 0], row["window_stop_tick"])
        events = tape.events[lo:hi]
        ticks = events[events[:, 1] == row["unit_index"], 0]
        row.update(observed_count=len(ticks), observed_window_ticks=ticks.tolist(),
                   first_spike_tick=int(ticks[0]) if len(ticks) else None,
                   refractory_ticks=refractory_ticks)
        if len(ticks) == 0:
            row["category"] = "no_first_spike"
        elif len(ticks) > 1:
            row["category"] = "suppressed_network_reference_required"
        elif int(ticks[0]) + refractory_ticks >= row["window_stop_tick"]:
            row["category"] = "refractory_covers_remaining_window"
        else:
            row["category"] = "conditional_integration_required"
        result.append(row)
    return result


def _extract_observed_single_spikes(config, case, tape, rows, *, drive=None, counts_E=None, counts_I=None):
    eligible = [r for r in rows if r["category"] == "conditional_integration_required"]
    if not eligible:
        return {}, dict(ticks_reconstructed=0, source_events_consumed=0)
    captures = {}; initials = {}; forcing = {}; conditional = {}; filled = {}
    for row in eligible:
        index, tau, stop = row["sample_index"], row["first_spike_tick"], row["window_stop_tick"]
        initials.setdefault(tau, []).append(row)
        captures.setdefault(tau, [])
        length = stop - tau - 1
        require(length > 0, "EXTRACTED_WINDOW_INCOMPLETE")
        forcing[index] = {"tick": np.empty(length, dtype=np.int64),
                          "time_seconds": np.empty(length, dtype=np.float64),
                          **{key: np.empty(length, dtype=np.float64)
                             for key in ("s_GABA", "s_NMDA_total", "I_stim", "s_NMDA", "x")}}
        filled[index] = 0
        for tick in range(tau + 1, stop): captures.setdefault(tick, []).append(row)
    constants = source._constants(config, "E")
    def snapshot(stage, tick, state):
        if stage == "pre_input":
            for row in initials.get(tick, []):
                u = row["unit_index"]
                values = {key: state["E__"+key][u].item() for key in reference.FIELDS + reference.E_FIELDS}
                values.update(v=constants["reset"], x=float(state["E__x"][u] + 1.0))
                conditional[row["sample_index"]] = dict(unit_index=u, population="E", first_spike_tick=tick,
                    window_stop_tick=row["window_stop_tick"], state=values,
                    timing="first_spike_reset_with_preinput_ampa", fresh_input_included=False,
                    voltage_units="V", time_units="s")
        elif stage == "pre_rk2":
            for row in captures.get(tick, []):
                index = row["sample_index"]
                offset = tick - row["first_spike_tick"] - 1
                require(offset == filled[index], "EXTRACTED_WINDOW_TICK_ORDER")
                arrays = forcing[index]
                arrays["tick"][offset] = tick
                arrays["time_seconds"][offset] = tick * tape.dt_seconds
                for key in ("s_GABA", "s_NMDA_total", "I_stim", "s_NMDA", "x"):
                    arrays[key][offset] = state["E__" + key][row["unit_index"]]
                filled[index] += 1
    result = reference.reconstruct(config, case, tape, stop_tick=max(r["window_stop_tick"] for r in eligible),
        drive=drive, counts_E=counts_E, counts_I=counts_I, on_state_view=snapshot,
        capture_ticks=list(map(int, captures)))
    refs = {}
    for row in eligible:
        index = row["sample_index"]
        require(index in conditional and filled[index] == row["window_stop_tick"]-row["first_spike_tick"]-1,
                "EXTRACTED_WINDOW_INCOMPLETE")
        refs[index] = (conditional[index], forcing[index])
    return refs, {k: result[k] for k in ("ticks_reconstructed", "source_events_consumed")}


def extract(config, case, tape, rows, *, drive=None, counts_E=None, counts_I=None, suppressed_replay=None):
    """Resolve every sampled history without using outcome-based replacement.

    The optional replayer recomputes one complete network per multispike
    window. An absent or failed replayer leaves that history unresolved.
    """
    multiple = [row for row in rows if row["category"] == "suppressed_network_reference_required"]
    require(not multiple or callable(suppressed_replay), "SUPPRESSED_NETWORK_REPLAY_REQUIRED_NO_ROWS_DROPPED")
    refs, diagnostic = _extract_observed_single_spikes(config,case,tape,rows,
        drive=drive,counts_E=counts_E,counts_I=counts_I)
    for row in multiple:
        (conditional,forcing), details = suppressed_replay(dict(row))
        require(conditional["unit_index"] == row["unit_index"]
                and conditional["first_spike_tick"] == row["first_spike_tick"]
                and conditional["window_stop_tick"] == row["window_stop_tick"]
                and conditional["fresh_input_included"] is False
                and np.array_equal(forcing["tick"],np.arange(row["first_spike_tick"]+1,row["window_stop_tick"]))
                and details["network_feedback_recomputed"] is True
                and details["source_law_trajectory"] is False
                and details["observed_focal_window_ticks"] == row["observed_window_ticks"],
                "SUPPRESSED_REFERENCE_IDENTITY_OR_FEEDBACK")
        refs[row["sample_index"]] = (conditional,forcing)
        row["suppressed_reference_resolved"] = True
        if "report_path" in details: row["suppressed_reference_report"] = details["report_path"]
    if multiple:
        diagnostic = dict(diagnostic,suppressed_network_replays=len(multiple),
                          source_window_denominator=len(rows),all_sampled_histories_retained=True)
    return refs, diagnostic


def integrate_rows(config, settings, rows, refs, output):
    """Integrate references in row order and preserve structural-zero rows.

    The caller creates ``output``. Rows are updated in place and saved after
    each completed row. An empty frame is saved as an empty list. Optional
    ``integration_seed_context`` is a list or tuple of nonnegative uint32
    integers appended to the unchanged legacy seed vector.
    """
    context = settings.get("integration_seed_context", [])
    require(isinstance(context, (list, tuple))
            and all(type(value) is int and 0 <= value < 2**32 for value in context),
            "INTEGRATION_SEED_CONTEXT")
    values = []; inner_variances = []
    for row in rows:
        index = row["sample_index"]
        row.update(conditional_estimate=0., inner_monte_carlo_variance=0., proposal_hits=0)
        if index in refs:
            conditional, arrays = refs[index]
            relative = Path("sample_"+str(index).zfill(3)); base = output/relative
            (base/"reference").mkdir(parents=True)
            (base/"reference/conditional_initial.json").write_text(json.dumps(conditional, indent=2)+"\n")
            np.savez_compressed(base/"reference/reference_window.npz", **arrays)
            inner = dict(settings["importance_sampling"])
            inner["seed_components"] = [20261009, settings["source_rng_seed"], settings["field_index"],
                                         row["frame_index"], round(config["dt_ms"]*1000),
                                         settings.get("integration_seed_namespace", 2)] + list(context)
            inner["reference"] = settings["source"]
            (base/"integration").mkdir()
            result = importance.run(config, conditional, arrays, inner, base/"integration")
            result.update(reference_archive_role="accepted_source_provenance_for_extracted_reference",
                          extracted_reference="../reference", sample_frame_row=row)
            (base/"integration/report.json").write_text(json.dumps(result, indent=2)+"\n")
            score = result["stopped_prefix"]
            row.update(conditional_estimate=score["estimate"],
                       inner_monte_carlo_variance=score["empirical_score_variance"]/inner["samples"],
                       proposal_hits=result["proposal_hits"], integration_report=str(relative/"integration/report.json"))
        values.append(row["conditional_estimate"]); inner_variances.append(row["inner_monte_carlo_variance"])
        (output/"integrated_frame.json").write_text(json.dumps(rows, indent=2)+"\n")
    if not rows:
        (output/"integrated_frame.json").write_text(json.dumps(rows, indent=2)+"\n")
    return values, inner_variances


def run(settings, source_dir, output, paired_frame=None, *, suppressed_replay=None,
        frame_sampler=None, frame_summary=None):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_PFC_FIRST_PASSAGE_HISTORY", new_independent_source_blocks=0,
        new_full_network_episodes=0, production_episodes=0, confirmation_execution_released=False,
        population_history_uncertainty_established=False, scientific_fidelity_qualified=False)
    started = time.perf_counter()
    try:
        config = yaml.safe_load((source_dir/"config.yaml").read_text())
        sr = json.loads((source_dir/"report.json").read_text()); case = sr["selected_case"]
        require(config == sr["resolved_config"] and case["source_rng_seed"] == settings["source_rng_seed"]
                and case["case_id"] == settings["case_id"] and config["dt_ms"] == settings["dt_ms"], "SOURCE_IDENTITY")
        require(config["excitatory_neurons"]+config["inhibitory_neurons"] == settings["expected_units"], "SOURCE_SCALE")
        require((frame_sampler is None and frame_summary is None)
                or (callable(frame_sampler) and callable(frame_summary)), "SAMPLING_AND_SUMMARY_REQUIRED_TOGETHER")
        sampler = sample_frame if frame_sampler is None else frame_sampler
        rows, frame = sampler(config, field=settings["field_index"], phase=settings["phase"],
            window_ms=settings["window_ms"], sample_size=settings["sample_size"], seed_components=settings["sampling_seed_components"])
        # Census-based selection may use archived timing labels. Persist its exact
        # sampling design before opening the event tape for reference extraction.
        (output/"selected_frame.json").write_text(json.dumps(dict(frame=frame, rows=rows), indent=2)+"\n")
        if "paired_coarse" in settings:
            require(paired_frame is not None, "PAIRED_PHYSICAL_FRAME_REQUIRED")
            paired = json.loads(paired_frame.read_text())
            keys = ("sample_index", "frame_index", "unit_index", "window_start_ms", "window_stop_ms")
            require(frame == paired["frame"]
                    and [[r[k] for k in keys] for r in rows] == [[r[k] for k in keys] for r in paired["rows"]],
                    "PAIRED_PHYSICAL_FRAME_CHANGED")
            report.update(paired_physical_frame_equal=True, paired_coarse=settings["paired_coarse"])
        with np.load(source_dir/"raw.npz", allow_pickle=False) as data:
            raw = {key: data[key] for key in ("spike_ticks", "spike_unit_indices", "native_spike_counts", "spike_seconds")}
        tape = reference.validate_tape(config, raw); del raw
        rows = classify(config, tape, rows)
        (output/"classified_frame.json").write_text(json.dumps(rows, indent=2)+"\n")
        report.update(frame=frame, categories=dict(Counter(r["category"] for r in rows)), source_rng_seed=case["source_rng_seed"])
        kind = settings.get("input_stream", "native_numpy_stream")
        initial, drive = create_input_drive(config, case, sr, kind)
        require(np.array_equal(initial, np.load(source_dir/"initial_voltage.npy", allow_pickle=False)), "SOURCE_INITIAL_STATE")
        begin = time.perf_counter()
        def replay_row(row):
            replay_initial, replay_drive = create_input_drive(config,case,sr,kind)
            require(np.array_equal(replay_initial,initial), "SUPPRESSED_REPLAY_INITIAL_CHANGED")
            relative = Path("suppressed_sample_"+str(row["sample_index"]).zfill(3))
            destination = output/relative
            destination.mkdir()
            details = dict(status="FAILED_PFC_HISTORY_SUPPRESSED_REFERENCE", sample_index=row["sample_index"],
                           source_rng_seed=case["source_rng_seed"], input_stream=kind)
            begin_replay = time.perf_counter()
            try:
                actual, checks = suppressed_replay(config,case,replay_initial,tape,row,drive=replay_drive)
                np.savez_compressed(destination/"source_raw.npz",**actual["raw"])
                np.savez_compressed(destination/"source_terminal.npz",**actual["terminal_state"])
                np.savez_compressed(destination/"prefix_rng.npz",**prefix_state_arrays(replay_drive,kind))
                result = actual["suppressed_reference"]
                np.savez_compressed(destination/"threshold_candidates.npz",ticks=result["threshold_candidate_ticks"])
                details.update(checks,status="COMPLETE_PFC_HISTORY_SUPPRESSED_REFERENCE",
                    source_cost=actual["cost"],source_units=len(replay_initial),source_reference=settings["source"],
                    unit_index=row["unit_index"],first_spike_tick=row["first_spike_tick"],
                    window_stop_tick=row["window_stop_tick"],dt_ms=config["dt_ms"],
                    source_law_trajectory=False,new_independent_source_blocks=0,production_episodes=0)
            except Exception as exc:
                details["error"] = dict(type=type(exc).__name__,message=str(exc));raise
            finally:
                details["wall_seconds"] = time.perf_counter()-begin_replay
                (destination/"report.json").write_text(json.dumps(details,indent=2)+"\n")
            return (result["conditional_initial"],result["forcing"]),dict(details,report_path=str(relative/"report.json"))
        refs, reconstruction = extract(config, case, tape, rows, drive=drive,
            suppressed_replay=replay_row if suppressed_replay is not None else None)
        report.update(reconstruction=reconstruction, reconstruction_seconds=time.perf_counter()-begin)
        np.savez_compressed(output/"reconstructed_prefix_rng.npz", **prefix_state_arrays(drive, kind))
        report.update(input_stream=kind, reconstruction_drive=drive.metadata())
        values, inner_variances = integrate_rows(config, settings, rows, refs, output)
        n = len(rows)
        report.update(status="COMPLETE_PFC_FIRST_PASSAGE_HISTORY_PILOT", sample_denominator=n,
            integrated_histories=len(refs), original_source_blocks=1, clock_pair_completed=False,
            third_spike_probability_established=False,
            actual_multispike_frequency=(float(np.mean([r["observed_count"] >= 2 for r in rows])) if n else None))
        if frame_summary is None:
            report.update(sample_mean=float(np.mean(values)),
                empirical_inner_mc_variance_of_sample_mean=float(sum(inner_variances)/n**2),
                empirical_variance_across_sampled_estimates=float(np.var(values, ddof=1)),
                expected_inner_variance_contribution_to_sample_variance=float(np.mean(inner_variances)),
                latent_sample_variance_diagnostic=float(np.var(values, ddof=1)-np.mean(inner_variances)),
                finite_population_sampling_fraction=n/frame["frame_size"],
                uncertainty_scope="Inner Monte Carlo and unit-window variation diagnostics only; one existing source block, no source-history confidence interval.",
                estimand="uniform unit-window mean of first-spike-indicator times integrated second-spike probability on archived source histories")
        else:
            report.update(frame_summary(rows, frame))
    except Exception as exc:
        report["status"] = "FAILED_PFC_FIRST_PASSAGE_HISTORY"
        report["error"] = dict(type=type(exc).__name__, message=str(exc)); raise
    finally:
        report["wall_seconds"] = time.perf_counter()-started
        (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source", "output"): parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--paired-frame", type=Path)
    args = parser.parse_args(); settings = yaml.safe_load(args.config.read_text())
    require(settings["schema"] == "pfc_first_passage_history_v1", "CONFIG_SCHEMA")
    result = run(settings, args.source, args.output, args.paired_frame)
    print(json.dumps({k:result[k] for k in ("status", "categories", "sample_mean", "wall_seconds")}))


if __name__ == "__main__": main()
