"""Attribute frozen PFC response scores to complete signed event counts.

This is descriptive accounting of a selected existing RNG block, not inference.
The four streams retain their source identity, multiplicity, and control sign.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import yaml

from .pfc_count_boundary_decomposition import segments_from_jumps, write_csv
from .pfc_delay_analysis import PHASES
from .pfc_fluctuation_analysis import unit_mapping
from .pfc_threshold_branch_probe import canonical_events
from .provenance import freeze_json, sha256_file, versions

SIGNS = np.array([1, -1, -1, 1], dtype=np.int64)
COMPONENTS = ("left_cue", "left_omission", "right_cue", "right_omission")
SEGMENT_FIELDS = ("unit", "start_exclusive_tick", "stop_inclusive_tick", "delta", "unresolved_at_horizon")
EXCURSION_FIELDS = ("unit", "onset_event_tick", "compensation_event_tick", "unresolved_at_horizon", "maximum_absolute_imbalance")


def require(value, message):
    if not value:
        raise ValueError("PFC_RESPONSE_EVENT_" + message)


def physical_ticks(milliseconds, base_tick_ms):
    ratio = milliseconds / base_tick_ms
    tick = round(ratio)
    require(milliseconds >= 0 and base_tick_ms > 0 and abs(ratio - tick) <= 1e-9,
            "NONINTEGER_PHYSICAL_CLOCK")
    return tick


def read_events(path, clock_ms, base_tick_ms, units, duration_ms):
    factor = physical_ticks(clock_ms, base_tick_ms)
    require(factor > 0, "CLOCK_FACTOR")
    with np.load(path, allow_pickle=False) as raw:
        events = canonical_events(raw["spike_ticks"], raw["spike_unit_indices"], units,
                                  physical_ticks(duration_ms, clock_ms))
    events[:, 0] *= factor
    return events


def signed_jumps(streams, units, horizon):
    """Sum coincident events as integers; never apply set union across streams."""
    require(len(streams) == 4, "FOUR_STREAMS_REQUIRED")
    codes, weights = [], []
    for events, sign in zip(streams, SIGNS):
        checked = canonical_events(events[:, 0], events[:, 1], units, horizon)
        codes.append(checked[:, 1] * horizon + checked[:, 0])
        weights.append(np.full(len(checked), sign, dtype=np.int64))
    code, weight = np.concatenate(codes), np.concatenate(weights)
    if len(code) == 0:
        return {}, dict(source_events=0, coincident_coordinates=0, cancelled_coordinates=0,
                        nonzero_coordinates=0, absolute_signed_jump_sum=0, maximum_jump_magnitude=0)
    order = np.argsort(code, kind="stable")
    code, weight = code[order], weight[order]
    starts = np.r_[0, np.flatnonzero(code[1:] != code[:-1]) + 1]
    unique, counts = code[starts], np.diff(np.r_[starts, len(code)])
    net = np.add.reduceat(weight, starts)
    jumps = {}
    for encoded, value in zip(unique[net != 0], net[net != 0]):
        unit, tick = divmod(int(encoded), horizon)
        jumps.setdefault(unit, {})[tick] = int(value)
    return jumps, dict(source_events=len(code), coincident_coordinates=int(np.count_nonzero(counts > 1)),
        cancelled_coordinates=int(np.count_nonzero(net == 0)), nonzero_coordinates=int(np.count_nonzero(net)),
        absolute_signed_jump_sum=int(np.abs(net).sum()), maximum_jump_magnitude=int(np.abs(net).max()))


def count_window(events, units, start, stop):
    return np.bincount(events[(events[:, 0] >= start) & (events[:, 0] < stop), 1], minlength=units)


def signed_summary(values):
    return dict(signed_sum=int(values.sum()), positive_sum=int(values[values > 0].sum()),
        negative_sum=int(values[values < 0].sum()), absolute_sum=int(np.abs(values).sum()),
        positive_units=int(np.count_nonzero(values > 0)), negative_units=int(np.count_nonzero(values < 0)),
        nonzero_units=int(np.count_nonzero(values)), max_abs=int(np.abs(values).max(initial=0)))


def score_identity(integer_score, denominator, saved_score, tolerance):
    actual = integer_score / denominator
    error = abs(actual - saved_score)
    require(np.isfinite(saved_score) and error <= tolerance, "SAVED_SCORE_RECONSTRUCTION")
    return error


def attribute(streams, units, start, stop, horizon):
    require(0 <= start < stop <= horizon, "WINDOW_RANGE")
    jumps, jump_summary = signed_jumps(streams, units, horizon)
    segments, excursions = segments_from_jumps(jumps, horizon)
    # Direct window counting and integration at boundaries are separate computations.
    components = np.stack([count_window(events, units, start, stop) for events in streams])
    boundaries = np.array([start, stop, horizon], dtype=np.int64)
    cumulative = np.stack([np.stack([count_window(e, units, 0, t) for e in streams]) for t in boundaries])
    direct = SIGNS @ components
    boundary_delta = np.einsum("c,bcu->bu", SIGNS, cumulative, dtype=np.int64)
    integrated = np.zeros((3, units), dtype=np.int64)
    selected_segments = []
    for row in segments:
        for bi, tick in enumerate(boundaries):
            if row["start_exclusive_tick"] < tick <= row["stop_inclusive_tick"]:
                integrated[bi, row["unit"]] = row["delta"]
                if bi < 2:
                    selected_segments.append(dict(boundary_tick=int(tick), **row))
    require(np.array_equal(integrated, boundary_delta), "SEGMENT_BOUNDARY_IDENTITY")
    require(np.array_equal(direct, boundary_delta[1] - boundary_delta[0]), "WINDOW_INTEGER_IDENTITY")
    arrays = dict(component_window_counts=components, cumulative_component_counts=cumulative,
        boundary_ticks=boundaries, cumulative_signed_counts=boundary_delta, unit_signed_counts=direct,
        cue_clock_counts=components[0] - components[2], omission_clock_counts=components[1] - components[3])
    cue, omission = arrays["cue_clock_counts"], arrays["omission_clock_counts"]
    # Save units where the two control-specific changes cancel as well as nonzero scores.
    touched = np.flatnonzero((cue != 0) | (omission != 0) | (boundary_delta[0] != 0) | (boundary_delta[1] != 0))
    contributors = []
    for unit in touched:
        contributors.append(dict(unit=int(unit), **{name: int(components[j, unit]) for j, name in enumerate(COMPONENTS)},
            cue_clock_count=int(cue[unit]), omission_clock_count=int(omission[unit]),
            signed_count=int(direct[unit]), D_start=int(boundary_delta[0, unit]),
            D_stop=int(boundary_delta[1, unit]), D_horizon=int(boundary_delta[2, unit])))
    summary = dict(window=signed_summary(direct), cue_clock_difference=signed_summary(cue),
        omission_clock_difference=signed_summary(omission),
        component_counts={name: int(components[j].sum()) for j, name in enumerate(COMPONENTS)},
        boundary_counts={name: signed_summary(boundary_delta[j]) for j, name in enumerate(("start", "stop", "horizon"))},
        control_change_units=int(np.count_nonzero((cue != 0) | (omission != 0))),
        cancelled_control_change_units=int(np.count_nonzero((direct == 0) & ((cue != 0) | (omission != 0)))),
        jump_accounting=jump_summary, segments=len(segments), excursions=len(excursions),
        unresolved_excursions_at_horizon=sum(r["unresolved_at_horizon"] for r in excursions),
        all_unit_integer_identities_exact=True)
    return summary, arrays, contributors, selected_segments, segments, excursions


def archive_metadata(root, settings):
    result = {}
    for key, spec in settings["archives"].items():
        path = root / spec["path"] / "delivery.json"
        require(sha256_file(path) == spec["delivery_sha256"], "DELIVERY_BINDING:" + key)
        delivery = json.loads(path.read_text())
        require(delivery["status"] == spec["status"], "DELIVERY_STATUS:" + key)
        if "verified" in delivery:
            require(delivery["verified"] is True, "UNVERIFIED_DELIVERY:" + key)
        elif spec["status"] == "ARCHIVED_PFC_SOURCE_EXECUTION":
            require(delivery["source_execution_complete"] is True and delivery["exit_code"] == 0
                    and delivery["result_status"] == "PFC_SOURCE_EXECUTION_COMPLETE", "COARSE_DELIVERY_COMPLETION")
        else:
            require(key == "original_scores" and delivery["status"] == "COMPLETE", "UNRECOGNIZED_LEGACY_DELIVERY")
        result[key] = {entry["path"]: entry for entry in delivery["files"]}
    return result


def bound_file(root, settings, manifests, archive, relative):
    entry = manifests[archive][relative]
    path = root / settings["archives"][archive]["path"] / relative
    require(path.stat().st_size == entry["bytes"] and sha256_file(path) == entry["sha256"],
            "INPUT_BINDING:" + archive + "/" + relative)
    return path


def load_sources(root, settings, manifests):
    events, reports, receipts = {}, {}, []
    n = settings["excitatory_neurons"] + settings["inhibitory_neurons"]
    for case in settings["cases"]:
        key, archive, relative = case["id"], case["archive"], case["path"]
        require(key not in events, "DUPLICATE_SOURCE_CASE")
        files = {name: bound_file(root, settings, manifests, archive, relative + "/" + name)
                 for name in ("raw.npz", "report.json", "config.yaml", "unit_metadata.json")}
        report = json.loads(files["report.json"].read_text())
        cfg = yaml.safe_load(files["config.yaml"].read_text())
        require(cfg == report["resolved_config"], "SOURCE_CONFIG_AGREEMENT")
        require(report["source_execution_complete"] is True and report["status"] in
                ("PFC_SOURCE_EXECUTION_COMPLETE", "COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE"), "SOURCE_COMPLETION")
        selected = report["selected_case"]
        require(selected["source_rng_seed"] == settings["source_rng_seed"]
                and selected["case_id"] == "pfc_" + str(settings["source_rng_seed"]) + "_" + case["control"], "SELECTED_CASE_IDENTITY")
        require(selected["stimulus_strength_nA"] == {"cue": .06, "omission": 0.}[case["control"]]
                and selected["distractor_strength_nA"] == 0., "CONTROL_SEMANTICS")
        require(cfg["dt_ms"] == case["clock_ms"] and cfg["duration_ms"] == settings["duration_ms"]
                and cfg["excitatory_neurons"] == settings["excitatory_neurons"]
                and cfg["inhibitory_neurons"] == settings["inhibitory_neurons"]
                and cfg["excitatory_preference_bins"] == settings["excitatory_preference_bins"], "SOURCE_DIMENSIONS")
        require(report["completed_duration_ticks"] == physical_ticks(settings["duration_ms"], case["clock_ms"]), "FULL_DURATION")
        metadata = json.loads(files["unit_metadata.json"].read_text())
        require(sha256_file(files["unit_metadata.json"]) == settings["unit_metadata_sha256"]
                and metadata["unit_count"] == n
                and metadata["archive_order"] == "all ascending E source indices, then all ascending I source indices", "UNIT_MAPPING")
        for name in ("raw.npz", "config.yaml", "unit_metadata.json"):
            require(report["outputs"][name]["sha256"] == manifests[archive][relative + "/" + name]["sha256"], "SOURCE_OUTPUT_BINDING")
        events[key] = read_events(files["raw.npz"], case["clock_ms"], settings["base_tick_ms"], n, settings["duration_ms"])
        reports[key] = report
        receipts.append(dict(**case, event_count=len(events[key]), report_sha256=sha256_file(files["report.json"]),
            raw_sha256=report["outputs"]["raw.npz"]["sha256"], refinement_seeds=report.get("refinement_seeds")))
    require(len(events) == 10, "TEN_UNIQUE_SOURCE_CASES_REQUIRED")
    for case in settings["cases"]:
        if case["clock_ms"] == .05:
            continue
        coarse = settings["archives"]["coarse_" + case["control"]]
        reference = reports[case["id"]]["coarse_reference"]
        require(reference["archive"] == coarse["path"] and reference["delivery_sha256"] == coarse["delivery_sha256"]
                and reference["report_sha256"] == next(r["report_sha256"] for r in receipts
                    if r["id"] == "shared_coarse_" + case["control"]), "SHARED_COARSE_REFERENCE")
    return events, receipts


def selected_score_rows(root, settings, manifests):
    tables = {}
    for key, spec in settings["score_tables"].items():
        path = bound_file(root, settings, manifests, spec["archive"], spec["path"])
        require(sha256_file(path) == spec["sha256"], "SCORE_TABLE_BINDING")
        with path.open() as stream:
            rows = [r for r in csv.DictReader(stream) if int(r["source_rng_seed"]) == settings["source_rng_seed"]]
        tables[key] = rows
    selected = {}
    for endpoint in settings["endpoints"]:
        index = endpoint["domain_row_index"]
        selected[index] = {}
        for key, rows in tables.items():
            matches = [r for r in rows if int(r["domain_row_index"]) == index]
            require(len(matches) == 1, "UNIQUE_SCORE_ROW")
            require(json.loads(matches[0]["endpoint_identity"]) == endpoint["identity"], "FROZEN_ENDPOINT_IDENTITY")
            selected[index][key] = matches[0]
        paired = selected[index]["paired"]
        require(paired["units"] == "Hz" and float(paired["margin"]) == endpoint["margin_hz"], "FROZEN_MARGIN")
    return selected


def integer_table(rows, fields):
    return np.asarray([[(-1 if r[k] is None else int(r[k])) for k in fields] for r in rows], dtype=np.int64).reshape(-1, len(fields))


def analyze(root, config, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status="FAILED_PFC_RESPONSE_EVENT_ATTRIBUTION", source_simulations=0, new_full_episodes=0,
        new_independent_blocks=0, scientific_fidelity_qualified=False, confirmation_execution_released=False,
        production_episodes=0, full_84330_endpoint_qualified=False)
    try:
        settings = yaml.safe_load(config.read_text())
        require(settings["schema"] == "pfc_response_event_attribution_v1", "CONFIG_SCHEMA")
        require(list(PHASES[settings["phase"]]) == settings["window_ms"], "FROZEN_PHASE_WINDOW")
        manifests = archive_metadata(root, settings)
        events, receipts = load_sources(root, settings, manifests)
        selected = selected_score_rows(root, settings, manifests)
        base = settings["base_tick_ms"]
        start, stop = [physical_ticks(t, base) for t in settings["window_ms"]]
        horizon = physical_ticks(settings["duration_ms"], base)
        ne, ni = settings["excitatory_neurons"], settings["inhibitory_neurons"]
        units, denominator = ne + ni, (ne + ni) * (stop - start) * base / 1000
        fields, _ = unit_mapping(ne, ni, settings["excitatory_preference_bins"])
        population = np.r_[np.zeros(ne, dtype=np.int8), np.ones(ni, dtype=np.int8)]
        panels, rows, outputs = {}, [], []
        for endpoint in settings["endpoints"]:
            index = endpoint["domain_row_index"]
            saved = selected[index]
            for realization in ("A", "B"):
                label = realization + "_" + str(index)
                case_ids = endpoint[realization]
                summary, arrays, contributions, bounds, segments, excursions = attribute(
                    [events[k] for k in case_ids], units, start, stop, horizon)
                total = summary["window"]["signed_sum"]
                paired_score = float(saved["paired"]["original_score" if realization == "A" else "repeated_score"])
                error = score_identity(total, denominator, paired_score, settings["score_serialization_tolerance_hz"])
                if realization == "A":
                    error = max(error, score_identity(total, denominator, float(saved["original"]["signed_block_score"]),
                                                     settings["score_serialization_tolerance_hz"]))
                    for name, column in zip(COMPONENTS, ("left_positive", "left_negative", "right_positive", "right_negative")):
                        score_identity(summary["component_counts"][name], denominator, float(saved["original"][column]),
                                       settings["score_serialization_tolerance_hz"])
                summary.update(case_ids=case_ids, domain_row_index=index, realization=realization,
                    signed_score_hz=total / denominator, saved_score_hz=paired_score,
                    reconstruction_error_hz=error, margin_hz=endpoint["margin_hz"], denominator_neuron_seconds=denominator,
                    population_contributions={name: signed_summary(arrays["unit_signed_counts"][population == pi])
                                              for pi, name in enumerate(("E", "I"))})
                for r in contributions:
                    r.update(population="E" if r["unit"] < ne else "I", field=int(fields[r["unit"]]))
                for r in bounds:
                    r.update(start_exclusive_ms=r["start_exclusive_tick"] * base,
                             stop_inclusive_ms=r["stop_inclusive_tick"] * base)
                array_path = label + "_all_units.npz"
                np.savez_compressed(output / array_path, **arrays, unit=np.arange(units), population=population,
                                    field=fields, component_names=np.array(COMPONENTS))
                segment_path = label + "_cumulative_segments.npz"
                np.savez_compressed(output / segment_path,
                    segments=integer_table(segments, SEGMENT_FIELDS), segment_columns=np.array(SEGMENT_FIELDS),
                    excursions=integer_table(excursions, EXCURSION_FIELDS), excursion_columns=np.array(EXCURSION_FIELDS))
                contribution_path, boundary_path = label + "_contributors.csv", label + "_boundary_segments.csv"
                write_csv(output / contribution_path, contributions, ["unit", *COMPONENTS, "signed_count"])
                write_csv(output / boundary_path, bounds, ["boundary_tick", *SEGMENT_FIELDS])
                outputs.extend([array_path, segment_path, contribution_path, boundary_path])
                rows.append(dict(panel=label, source_rng_seed=settings["source_rng_seed"], domain_row_index=index,
                    realization=realization, net_events=total, positive_events=summary["window"]["positive_sum"],
                    negative_events=summary["window"]["negative_sum"], changed_units=summary["window"]["nonzero_units"],
                    signed_score_hz=total / denominator, saved_score_hz=paired_score, reconstruction_error_hz=error,
                    cancelled_control_change_units=summary["cancelled_control_change_units"]))
                panels[label] = summary
        write_csv(output / "endpoint_summary.csv", rows, ["panel"])
        outputs.append("endpoint_summary.csv")
        report.update(status="COMPLETE_PFC_RESPONSE_EVENT_ATTRIBUTION", panels=panels, source_bindings=receipts,
            settings=settings, config_sha256=sha256_file(config), versions=versions(),
            event_count_semantics="Half-open physical window [1000,2000) ms; N(<stop)-N(<start).",
            cumulative_semantics="D=N_Lcue-N_Lomission-N_Rcue+N_Romission; segments (start,stop]; -1 denotes absent compensation tick.",
            interpretation="Selected development block only. Count cancellation does not identify retimed spikes, causal paths, numerical bias, convergence, coverage, or equivalence.",
            all_units_retained=units, unique_source_cases=len(events), original_independent_blocks=32,
            output_files={name: dict(bytes=(output/name).stat().st_size, sha256=sha256_file(output/name)) for name in outputs})
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        freeze_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.project_root, args.config, args.output)
    print(json.dumps(dict(status=report["status"], panels={k: v["window"]["signed_sum"] for k,v in report["panels"].items()})))


if __name__ == "__main__":
    main()
