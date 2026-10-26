"""Validate a saved focal first-hit input in the complete coupled PFC source.

The native stream advances unchanged before a single focal column is replaced.
The result is a counterfactual prefix, never a new independent source episode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from . import pfc_deterministic_replay as source
from . import pfc_native_drive as native
from . import pfc_first_passage_reference as reference
from .pfc_first_passage_probability import first_passage_ticks


def require(condition, message):
    if not condition:
        raise ValueError("PFC_FIRST_PASSAGE_WITNESS_" + message)


def create_bound_drive(config, case, source_report, settings):
    """Use the declared source input law and its archived seed identities."""
    kind = settings.get("input_stream", "native_numpy_stream")
    if kind == "native_numpy_stream":
        require("coarse_dt_ms" not in source_report, "NATIVE_MODE_WITH_COUPLED_SOURCE")
        initial, drive = native.create_drive(config, case)
        return initial, drive, kind
    # Frozen native witness bundles predate the history adapter dependency.
    from . import pfc_first_passage_history as history
    initial, drive = history.create_input_drive(config, case, source_report, kind)
    return initial, drive, kind


def bound_prefix_state(drive, kind):
    """Retain the legacy native arrays; include coupled RNGs and buffered rows."""
    if kind == "native_numpy_stream":
        return native._state_arrays(drive.get_state())
    from . import pfc_first_passage_history as history
    return history.prefix_state_arrays(drive, kind)


class FocalOverrideDrive:
    """Replace a declared E-unit interval after all native draws and accounting."""

    def __init__(self, drive, *, unit_index, start_tick, stop_tick, counts):
        values = np.asarray(counts)
        require(drive.ticks == 0, "FRESH_NATIVE_DRIVE_REQUIRED")
        require(type(unit_index) is int and 0 <= unit_index < drive.n_e, "FOCAL_E_INDEX")
        require(type(start_tick) is int and type(stop_tick) is int
                and 0 <= start_tick < stop_tick <= drive.total_ticks, "INPUT_INTERVAL")
        require(values.ndim == 1 and values.shape == (stop_tick - start_tick,)
                and values.dtype.kind in "iu" and np.all((values >= 0) & (values <= 1000)),
                "COMPLETE_INTEGER_INPUT_COUNTS")
        self.native = drive
        self.n_e, self.n_i, self.dt_seconds = drive.n_e, drive.n_i, drive.dt_seconds
        self.total_ticks = drive.total_ticks
        self.unit_index, self.start_tick, self.stop_tick = unit_index, start_tick, stop_tick
        self.counts = values.astype(np.uint16, copy=True)
        self.original_counts = np.zeros_like(self.counts)
        self.replaced = np.zeros(len(values), dtype=bool)

    @property
    def ticks(self):
        return self.native.ticks

    def take(self, rows):
        start = self.ticks
        blocks = self.native.take(rows)
        lo, hi = max(start, self.start_tick), min(start + rows, self.stop_tick)
        if lo < hi:
            target = slice(lo - start, hi - start)
            witness = slice(lo - self.start_tick, hi - self.start_tick)
            require(not self.replaced[witness].any(), "REPEATED_INPUT_REPLACEMENT")
            self.original_counts[witness] = blocks["E"][target, self.unit_index]
            blocks["E"][target, self.unit_index] = self.counts[witness]
            self.replaced[witness] = True
        return blocks

    def get_state(self):
        return self.native.get_state()

    def evidence(self):
        require(self.replaced.all(), "INCOMPLETE_INPUT_REPLACEMENT")
        delta = int(self.counts.sum(dtype=np.int64) - self.original_counts.sum(dtype=np.int64))
        return dict(unit_index=self.unit_index, population="E", input_start_tick=self.start_tick,
                    input_stop_tick_exclusive=self.stop_tick, input_slots=len(self.counts),
                    original_focal_events=int(self.original_counts.sum(dtype=np.int64)),
                    delivered_focal_events=int(self.counts.sum(dtype=np.int64)),
                    event_delta_E=delta, event_delta_I=0,
                    original_native_metadata=self.native.metadata())


def check_event_boundary(original, changed, *, unit_index, window_start_tick,
                         first_spike_tick, expected_hit_tick, window_stop_tick):
    """Compare the causal prefix and first two focal events in the frozen window."""
    require(window_start_tick <= first_spike_tick < expected_hit_tick < window_stop_tick,
            "FIRST_HIT_INTERVAL")
    original = np.asarray(original); changed = np.asarray(changed)
    baseline_focal = original[(original[:, 1] == unit_index)
                              & (original[:, 0] >= window_start_tick)
                              & (original[:, 0] < window_stop_tick), 0]
    focal = changed[(changed[:, 1] == unit_index)
                    & (changed[:, 0] >= window_start_tick)
                    & (changed[:, 0] < window_stop_tick), 0]
    require(np.array_equal(baseline_focal, [first_spike_tick]), "BASELINE_NOT_SINGLE_WINDOW_SPIKE")
    require(len(focal) >= 2 and focal[0] == first_spike_tick and focal[1] == expected_hit_tick,
            "FIRST_ADDITIONAL_SPIKE_MISMATCH")
    old_prefix = original[original[:, 0] < expected_hit_tick]
    new_prefix = changed[changed[:, 0] < expected_hit_tick]
    require(np.array_equal(old_prefix, new_prefix), "PREHIT_NETWORK_EVENTS_CHANGED")
    old_tick = original[(original[:, 0] == expected_hit_tick) & (original[:, 1] != unit_index)]
    new_tick = changed[(changed[:, 0] == expected_hit_tick) & (changed[:, 1] != unit_index)]
    require(np.array_equal(old_tick, new_tick), "HIT_TICK_NONFOCAL_EVENTS_CHANGED")
    return dict(all_prehit_events_equal=True, prehit_events=len(old_prefix),
                hit_tick_nonfocal_events_equal=True, hit_tick_nonfocal_events=len(old_tick),
                original_focal_window_ticks=baseline_focal.tolist(),
                modified_focal_window_ticks=focal.tolist(), actual_first_hit_tick=int(focal[1]),
                original_already_hit_at_expected_tick=False,
                posthit_network_equality_required=False)


def validate(settings, source_dir, reference_dir, witnesses_path, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_PFC_FIRST_PASSAGE_WITNESS", new_full_network_episodes=0,
                  new_independent_source_blocks=0, production_episodes=0,
                  confirmation_execution_released=False, scientific_fidelity_qualified=False,
                  population_history_uncertainty_established=False)
    started = time.perf_counter()
    try:
        config = yaml.safe_load((source_dir / "config.yaml").read_text())
        origin = json.loads((source_dir / "report.json").read_text())
        conditional = json.loads((reference_dir / "conditional_initial.json").read_text())
        case = origin["selected_case"]
        require(config == origin["resolved_config"], "SOURCE_CONFIG")
        require(case["case_id"] == settings["case_id"]
                and case["source_rng_seed"] == settings["source_rng_seed"], "SOURCE_CASE")
        require(config["excitatory_neurons"] + config["inhibitory_neurons"] == settings["expected_units"]
                and config["duration_ms"] == settings["source_duration_ms"]
                and config["dt_ms"] == settings["dt_ms"], "SOURCE_SCALE_AND_CLOCK")
        row = settings["witness_row"]
        require(row == 0, "FIRST_SAVED_WITNESS_REQUIRED")
        with np.load(witnesses_path, allow_pickle=False) as data:
            counts = data["counts"][row].copy()
            witness = {k: data[k][row].item() for k in data.files if k != "counts"}
        tau, stop, unit = (conditional[k] for k in ("first_spike_tick", "window_stop_tick", "unit_index"))
        require(unit == settings["unit_index"] and tau == settings["first_spike_tick"]
                and stop == settings["window_stop_tick"], "REFERENCE_IDENTITY")
        require(witness["sample_index"] == settings["expected_sample_index"]
                and witness["first_hit_tick"] == settings["expected_hit_tick"], "WITNESS_IDENTITY")
        expected_hit = int(witness["first_hit_tick"])
        with np.load(reference_dir / "reference_window.npz", allow_pickle=False) as data:
            forcing = {key: data[key] for key in data.files}
        prediction = first_passage_ticks(config, conditional, forcing, counts[None, :])
        require(int(prediction[0]) == expected_hit, "SAVED_KERNEL_PREDICTION_CHANGED")
        require(int(counts[:expected_hit - tau].sum(dtype=np.int64)) == witness["prefix_input_count"]
                and int(counts.sum(dtype=np.int64)) == witness["total_input_count"], "WITNESS_COUNT_TOTALS")
        with np.load(source_dir / "raw.npz", allow_pickle=False) as data:
            raw = {key: data[key] for key in ("spike_ticks", "spike_unit_indices", "native_spike_counts", "spike_seconds")}
        tape = reference.validate_tape(config, raw)
        del raw
        original_events = tape.events[tape.events[:, 0] < stop]
        initial, native_drive, kind = create_bound_drive(config, case, origin, settings)
        require(np.array_equal(initial, np.load(source_dir / "initial_voltage.npy", allow_pickle=False)),
                "ORIGINAL_INITIAL_STATE")
        drive = FocalOverrideDrive(native_drive, unit_index=unit, start_tick=tau, stop_tick=stop - 1, counts=counts)
        prefix_config = dict(config, duration_ms=stop * config["dt_ms"])
        result = source.simulate(prefix_config, case, initial, backend="numpy", drive=drive)
        result["inputs_used"]["input_mode"] = kind+"_with_saved_focal_counterfactual_inputs"
        result["cost"]["arithmetic_path"]["external_inputs"] = (
            kind+" draws retained; one E-unit interval replaced after drawing")
        # Save the complete modified outputs before any acceptance assertion.
        np.savez_compressed(output / "source_raw.npz", **result["raw"])
        np.savez_compressed(output / "source_terminal.npz", **result["terminal_state"])
        np.savez_compressed(output / "replacement_inputs.npz", tick=np.arange(tau, stop - 1),
                            counts=drive.counts, original_counts=drive.original_counts,
                            replaced=drive.replaced, **{k: np.asarray(v) for k, v in witness.items()})
        actual_rng = bound_prefix_state(native_drive, kind)
        np.savez_compressed(output / "native_prefix_rng.npz", **actual_rng)
        report.update(witness=witness, source_units=tape.n_e + tape.n_i, unit_index=unit,
                      window_start_tick=settings["window_start_tick"], first_spike_tick=tau,
                      prefix_ticks=stop, prefix_duration_ms=prefix_config["duration_ms"],
                      source_cost=result["cost"], inputs_used=result["inputs_used"], replacement=drive.evidence())
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        # This regenerates inputs only, not a second membrane/network simulation.
        begin = time.perf_counter()
        second_initial, check_drive, check_kind = create_bound_drive(config, case, origin, settings)
        require(check_kind == kind, "REGENERATED_INPUT_KIND")
        originals = []
        while check_drive.ticks < stop:
            start = check_drive.ticks
            size = min(256, check_drive.max_rows, stop - start)
            blocks = check_drive.take(size)
            lo, hi = max(start, tau), min(start + size, stop - 1)
            if lo < hi:
                originals.append(blocks["E"][lo - start:hi - start, unit].copy())
        np.savez_compressed(output / "regenerated_prefix_rng.npz", **bound_prefix_state(check_drive, kind))
        require(np.array_equal(initial, second_initial), "REGENERATED_INITIAL_STATE")
        require(all(np.array_equal(v, bound_prefix_state(check_drive, kind)[k]) for k, v in actual_rng.items()),
                "PREFIX_NATIVE_RNG_CHANGED")
        require(np.array_equal(np.concatenate(originals), drive.original_counts), "ORIGINAL_REPLACED_INPUTS_CHANGED")
        require(check_drive.event_totals == native_drive.event_totals, "ORIGINAL_NATIVE_TOTALS_CHANGED")
        require(result["inputs_used"]["E"]["events"] == check_drive.event_totals["E"] + report["replacement"]["event_delta_E"]
                and result["inputs_used"]["I"]["events"] == check_drive.event_totals["I"], "DELIVERED_INPUT_TOTALS")
        changed = reference.canonical_events(result["raw"]["spike_ticks"], result["raw"]["spike_unit_indices"],
                                             tape.n_e + tape.n_i, stop)
        boundary = check_event_boundary(original_events, changed, unit_index=unit,
            window_start_tick=settings["window_start_tick"], first_spike_tick=tau,
            expected_hit_tick=expected_hit, window_stop_tick=stop)
        require(np.array_equal(result["raw"]["native_spike_counts"],
                               np.bincount(changed[:, 1], minlength=tape.n_e + tape.n_i)), "EVENT_COUNTS")
        report.update(status="COMPLETE_PFC_FIRST_PASSAGE_WITNESS", event_boundary=boundary,
                      modified_prefix_events=len(changed), original_prefix_events=len(original_events),
                      prefix_rng_equal=True, input_stream=kind, input_regeneration_seconds=time.perf_counter() - begin,
                      original_replaced_inputs_equal=True, delivered_input_totals_equal=True,
                      actual_source_prefix_replays=1,
                      interpretation="One saved counterfactual first crossing validated in the full coupled source; no population or third-spike probability qualification.")
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source", "reference", "witnesses", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    require(settings["schema"] == "pfc_first_passage_witness_v1", "CONFIG_SCHEMA")
    result = validate(settings, args.source, args.reference, args.witnesses, args.output)
    print(json.dumps({k: result[k] for k in ("status", "event_boundary", "wall_seconds")}))


if __name__ == "__main__":
    main()
