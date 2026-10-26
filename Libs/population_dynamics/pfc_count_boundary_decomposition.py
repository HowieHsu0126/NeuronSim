"""Exact window-boundary accounting for fixed PFC spike paths; no inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from . import pfc_count_histograms as histograms
from .pfc_fluctuation_analysis import unit_mapping
from . import pfc_threshold_branch_probe as branch
from .provenance import freeze_json, sha256_file


def require(value, message):
    if not value:
        raise ValueError("PFC_COUNT_BOUNDARY_" + message)


def counts(events, units, stop, width):
    require(stop > 0 and width > 0 and stop % width == 0, "WINDOW_CLOCK")
    code = (events[:, 0] // width) * units + events[:, 1]
    return np.bincount(code, minlength=(stop // width) * units).reshape(-1, units)


def intervals(original, other, units, stop):
    """D(t)=N_other(<t)-N_original(<t); constant segments occupy (a,b]."""
    a = original[:, 0] * units + original[:, 1]
    b = other[:, 0] * units + other[:, 1]
    jumps = {}
    for codes, sign in ((np.setdiff1d(b, a), 1), (np.setdiff1d(a, b), -1)):
        for code in codes:
            tick, unit = divmod(int(code), units)
            row = jumps.setdefault(unit, {})
            row[tick] = row.get(tick, 0) + sign
    return segments_from_jumps(jumps, stop)


def segments_from_jumps(jumps, stop):
    """Integrate integer signed event jumps, preserving multiplicity and cancellation.

    Counts are strictly before the query time, so each constant segment is (a,b].
    Callers supply complete event streams with 0 <= event tick < stop.
    """
    segments, excursions = [], []
    for unit, changes in sorted(jumps.items()):
        value, previous, onset, maximum = 0, None, None, 0
        for tick, jump in sorted(changes.items()):
            if not jump:
                continue
            if value:
                segments.append(dict(unit=unit, start_exclusive_tick=previous,
                    stop_inclusive_tick=tick, delta=value, unresolved_at_horizon=False))
            old = value
            value += jump
            if old == 0 and value:
                onset, maximum = tick, abs(value)
            maximum = max(maximum, abs(value))
            if old and value == 0:
                excursions.append(dict(unit=unit, onset_event_tick=onset,
                    compensation_event_tick=tick, unresolved_at_horizon=False,
                    maximum_absolute_imbalance=maximum))
                onset = None
            previous = tick
        if value:
            segments.append(dict(unit=unit, start_exclusive_tick=previous,
                stop_inclusive_tick=stop, delta=value, unresolved_at_horizon=True))
            excursions.append(dict(unit=unit, onset_event_tick=onset,
                compensation_event_tick=None, unresolved_at_horizon=True,
                maximum_absolute_imbalance=maximum))
    return segments, excursions


def decompose(original, other, units, stop, widths, fields, sizes):
    for events in (original, other):
        require(events.ndim == 2 and events.shape[1] == 2 and events.dtype.kind in "iu"
                and np.all(events >= 0) and np.all(events[:, 0] < stop)
                and np.all(events[:, 1] < units), "EVENT_SCHEMA")
        require(len(events) == len(np.unique(events, axis=0)), "DUPLICATE_EVENT")
    segments, excursions = intervals(original, other, units, stop)
    by_unit = {}
    for row in segments:
        by_unit.setdefault(row["unit"], []).append(row)
    sparse, changed, field_rows, cdf_rows = {}, [], [], []
    numerical_cdf_errors, saved = [], {}
    base_width = widths[0]
    base = {name: counts(events, units, stop, base_width)
            for name, events in (("original", original), ("other", other))}
    for wi, width in enumerate(widths):
        require(width % base_width == 0, "NESTED_WINDOWS_REQUIRED")
        grouped = {name: values.reshape(-1, width // base_width, units).sum(axis=1)
                   for name, values in base.items()}
        delta = grouped["other"] - grouped["original"]
        cumulative = np.vstack((np.zeros((1, units), dtype=np.int64), np.cumsum(delta, axis=0)))
        boundaries = np.arange(0, stop + 1, width, dtype=np.int64)
        for unit, rows in by_unit.items():
            expected = np.zeros(len(boundaries), dtype=np.int64)
            for row in rows:
                mask = (boundaries > row["start_exclusive_tick"]) & (boundaries <= row["stop_inclusive_tick"])
                expected[mask] = row["delta"]
            require(np.array_equal(expected, cumulative[:, unit]), "INTERVAL_COUNT_BOUNDARY_IDENTITY")
        require(set(np.flatnonzero(np.any(delta != 0, axis=0))) <= set(by_unit), "MISSING_CHANGED_UNIT")
        for condition, name in enumerate(("original", "other")):
            histograms.accumulate(sparse, base[name][None], fields, width // base_width,
                                  condition=condition, phase=0, window=wi)
        rows, columns = np.nonzero(delta)
        for window, unit in zip(rows, columns):
            changed.append(dict(width_ticks=width, start_tick=int(window * width), stop_tick=int((window + 1) * width),
                unit=int(unit), field=int(fields[unit]), original_count=int(grouped["original"][window, unit]),
                other_count=int(grouped["other"][window, unit]), delta=int(delta[window, unit]),
                D_start=int(cumulative[window, unit]), D_stop=int(cumulative[window + 1, unit])))
        saved[width] = grouped
    coordinates = np.asarray(sorted(sparse), dtype=np.int64)
    mass = np.asarray([sparse[tuple(key)] for key in coordinates], dtype=np.int64)
    support = np.arange(int(coordinates[:, 5].max()) + 1, dtype=np.int64)
    cdfs = histograms.cdf(coordinates, mass, support, (2, 1, 1, len(widths), len(sizes)))
    require(np.all(cdfs[..., -1] == 1), "INCOMPLETE_EMPIRICAL_SUPPORT")
    for wi, width in enumerate(widths):
        for field, size in enumerate(sizes):
            a = saved[width]["original"][:, fields == field].ravel()
            b = saved[width]["other"][:, fields == field].ravel()
            denominator, touched = len(a), int(np.count_nonzero(a != b))
            signed = []
            for k in support:
                positive = int(np.count_nonzero((b <= k) & (a > k)))
                negative = int(np.count_nonzero((a <= k) & (b > k)))
                numerator = positive - negative
                require(abs(numerator) <= touched, "CDF_TOUCHED_FRACTION_BOUND")
                observed = float(cdfs[1, 0, 0, wi, field, k] - cdfs[0, 0, 0, wi, field, k])
                error = abs(observed - numerator / denominator)
                numerical_cdf_errors.append(error)
                require(error <= 2 * np.finfo(np.float64).eps, "CDF_INTEGER_CONTRIBUTION_IDENTITY")
                signed.append(numerator)
                cdf_rows.append(dict(width_ticks=width, field=field, threshold=int(k), denominator=denominator,
                    positive_crossings=positive, negative_crossings=negative, signed_numerator=numerator,
                    empirical_cdf_difference=observed, touched_observations=touched))
            field_rows.append(dict(width_ticks=width, field=field, field_units=int(size), denominator=denominator,
                touched_observations=touched, touched_fraction_bound=touched / denominator,
                maximum_absolute_cdf_numerator=max(abs(x) for x in signed),
                maximum_absolute_empirical_cdf_difference=max(abs(x) for x in signed) / denominator))
    focus = sorted({row["unit"] for row in changed if row["width_ticks"] == base_width and row["start_tick"] == 0})
    first_delta = saved[base_width]["other"][0] - saved[base_width]["original"][0]
    summary = dict(original_event_count=len(original), other_event_count=len(other),
        event_disagreement_units=len(by_unit), constant_disagreement_segments=len(segments),
        excursions=len(excursions), unresolved_excursions_at_horizon=sum(x["unresolved_at_horizon"] for x in excursions),
        first_window_changed_units=focus, first_window_absolute_count_delta=int(np.abs(first_delta).sum()),
        first_window_signed_count_delta=int(first_delta.sum()), total_changed_unit_windows=len(changed),
        count_support_max=int(support[-1]), field_count=len(sizes),
        maximum_absolute_empirical_cdf_difference=max(x["maximum_absolute_empirical_cdf_difference"] for x in field_rows),
        maximum_cdf_arithmetic_error=max(numerical_cdf_errors), all_boundary_identities_exact=True,
        all_integer_cdf_bounds_passed=True, compensation_does_not_identify_same_spike=True,
        interval_semantics="D(t) uses events strictly before t; nonzero segments are (start,stop]; compensation occurs immediately after its event tick",
        focus_excursions=[x for x in excursions if x["unit"] in focus])
    return summary, dict(segments=segments, excursions=excursions, changed=changed,
                        fields=field_rows, cdf_contributions=cdf_rows)


def write_csv(path, rows, empty_fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty_fields)
        writer.writeheader()
        writer.writerows(rows)


def bound_archive(root, digest, names):
    require(sha256_file(root / "delivery.json") == digest, "ARCHIVE_BINDING")
    delivery = json.loads((root / "delivery.json").read_text())
    require(delivery["verified"] is True, "ARCHIVE_UNVERIFIED")
    manifest = {row["path"]: row for row in delivery["files"]}
    return {name: branch._bound_file(root, name, manifest[name]) for name in names}


def read_events(path, config, stop):
    with np.load(path, allow_pickle=False) as raw:
        events = branch.canonical_events(raw["spike_ticks"], raw["spike_unit_indices"],
            config["excitatory_neurons"] + config["inhibitory_neurons"],
            round(config["duration_ms"] / config["dt_ms"]))
    return events[events[:, 0] < stop]


def analyze(project_root, probe_root, probe_digest, continuation_root, continuation_digest, output):
    output.mkdir(parents=True, exist_ok=False)
    started, names = time.perf_counter(), []
    report = dict(status="FAILED_PFC_COUNT_BOUNDARY_DECOMPOSITION",
        scientific_fidelity_qualified=False, cdf_qualified=False,
        source_probability_bound_established=False, confirmation_execution_released=False,
        new_full_episodes=0, new_independent_blocks=0, source_simulations=0, production_episodes=0)
    try:
        probe = bound_archive(probe_root, probe_digest,
            ("execution/report.json", "prepared/plan.json", "prepared/reversed_events.npz"))
        continuation = bound_archive(continuation_root, continuation_digest,
            ("execution/report.json", "prepared/plan.json", "prepared/original_events.npz", "execution/counterfactual_raw.npz"))
        prior = json.loads(probe["execution/report.json"].read_text())
        current = json.loads(continuation["execution/report.json"].read_text())
        probe_plan = json.loads(probe["prepared/plan.json"].read_text())
        plan = json.loads(continuation["prepared/plan.json"].read_text())
        require(prior["status"] == "PASS_PFC_THRESHOLD_BRANCH_TWO_PATH_WITNESS"
                and current["status"] == "COMPLETE_PFC_SPIKE_BRANCH_CONTINUATION", "ACCEPTED_SOURCE_REPORTS")
        config, settings = plan["full_config"], plan["settings"]
        require(config == probe_plan["full_config"] and settings["stop_ms"] == 100.
                and settings["windows_ms"] == [5., 50., 100.] and settings["expected_units"] == 100000,
                "FROZEN_OBSERVATION_SCOPE")
        stop = round(settings["stop_ms"] / config["dt_ms"])
        widths = [round(x / config["dt_ms"]) for x in settings["windows_ms"]]
        ne, ni = config["excitatory_neurons"], config["inhibitory_neurons"]
        fields, sizes = unit_mapping(ne, ni, config["excitatory_preference_bins"])
        binding = probe_plan["source_bindings"]["reversed"]
        reversed_root = project_root / binding["archive"]
        case = binding["case_relative_path"]
        reversed_files = bound_archive(reversed_root, binding["delivery_sha256"], (case + "/raw.npz", case + "/report.json"))
        source = json.loads(reversed_files[case + "/report.json"].read_text())
        require(source["resolved_config"] == config and source["selected_case"] == plan["selected_case"],
                "REVERSED_SOURCE_CONFIG")
        events = dict(original=read_events(continuation["prepared/original_events.npz"], config, stop),
            reversal=read_events(reversed_files[case + "/raw.npz"], config, stop),
            isolated_spike=read_events(continuation["execution/counterfactual_raw.npz"], config, stop))
        short_reversal = read_events(probe["prepared/reversed_events.npz"], config, widths[0])
        require(np.array_equal(short_reversal, events["reversal"][events["reversal"][:, 0] < widths[0]]),
                "EXACT_ACCEPTED_REVERSED_PREFIX")
        require(len(events["original"]) == current["original_event_count"]
                and len(events["isolated_spike"]) == current["counterfactual_event_count"], "ACCEPTED_PREFIX_COUNTS")
        np.savez(output / "prefix_events.npz", **{name + "_" + column: values[:, index]
            for name, values in events.items() for index, column in enumerate(("ticks", "units"))})
        names.append("prefix_events.npz")
        summaries = {}
        empty_headers = dict(segments=["unit", "start_exclusive_tick", "stop_inclusive_tick", "delta", "unresolved_at_horizon"],
            excursions=["unit", "onset_event_tick", "compensation_event_tick", "unresolved_at_horizon", "maximum_absolute_imbalance"],
            changed=["width_ticks", "start_tick", "stop_tick", "unit", "field", "original_count", "other_count", "delta", "D_start", "D_stop"],
            fields=["width_ticks", "field", "denominator"], cdf_contributions=["width_ticks", "field", "threshold", "denominator"])
        for name in ("reversal", "isolated_spike"):
            summary, rows = decompose(events["original"], events[name], ne + ni, stop, widths, fields, sizes)
            for kind, values in rows.items():
                filename = name + "_" + kind + ".csv"
                write_csv(output / filename, values, empty_headers[kind])
                names.append(filename)
            summaries[name] = summary
        expected = prior["prefix_count_differences"]
        actual = summaries["reversal"]
        require(len(actual["first_window_changed_units"]) == expected["units_with_different_counts"]
                and actual["first_window_absolute_count_delta"] == expected["absolute_count_difference_sum"]
                and actual["first_window_signed_count_delta"] == expected["reversed_minus_original_total"],
                "ACCEPTED_TWO_UNIT_COUNT_DIFFERENCE")
        control = summaries["isolated_spike"]
        require(control["event_disagreement_units"] == 1 and control["total_changed_unit_windows"] == 0
                and control["maximum_absolute_empirical_cdf_difference"] == 0
                and control["unresolved_excursions_at_horizon"] == 0, "ISOLATED_SPIKE_BOUNDARY_CONTROL")
        report.update(status="COMPLETE_PFC_COUNT_BOUNDARY_DECOMPOSITION", comparisons=summaries,
            config=config, selected_case=plan["selected_case"], stop_tick=stop, stop_ms=settings["stop_ms"],
            windows_ms=settings["windows_ms"], field_units=sizes.tolist(), dt_ms=config["dt_ms"],
            input_bindings=dict(probe_archive=str(probe_root), probe_delivery_sha256=probe_digest,
                continuation_archive=str(continuation_root), continuation_delivery_sha256=continuation_digest,
                reversal_source=binding),
            cdf_bound_scope="Exact empirical touched-unit-window bound only, not a source-law probability bound",
            complete_unit_denominator=ne + ni, compensation_is_event_identity_claim=False,
            inference="Retain boundary crossings and finite-horizon residual count imbalance; equal cumulative counts do not identify the same biological event")
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
    report.update(wall_seconds=time.perf_counter() - started,
                  outputs={name: branch._file_record(output / name) for name in names})
    freeze_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--probe-archive", type=Path, required=True)
    parser.add_argument("--probe-delivery-sha256", required=True)
    parser.add_argument("--continuation-archive", type=Path, required=True)
    parser.add_argument("--continuation-delivery-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.project_root, args.probe_archive, args.probe_delivery_sha256,
                     args.continuation_archive, args.continuation_delivery_sha256, args.output)
    print(json.dumps({key: report[key] for key in ("status", "error", "comparisons") if key in report}, indent=2))
    return 0 if report["status"] == "COMPLETE_PFC_COUNT_BOUNDARY_DECOMPOSITION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
