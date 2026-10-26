"""Compare complete original-size calibration cells with the fixed100k summary."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import pfc_reference_quality_summary as quality
from .provenance import freeze_json, sha256_file

CIRCUITS = (23, 63, 52, 86)
BLOCKS = (3, 4, 5)
REFERENCE_SUMMARY_SHA256 = "77b2ef54b5acb3be207def5655447abee3b883dc1f9f127b7889889309a17c58"


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def calibration_plan(path: Path) -> dict:
    plan = quality.read_json(path)
    rows = plan.get("rows", [])
    quality.require(plan.get("status") == "PREPARED120_ORIGINAL_SIZE_CALIBRATION_EPISODES"
        and plan.get("scope") == "descriptive_training_calibration_size_comparison"
        and plan.get("circuits") == list(CIRCUITS) and plan.get("blocks") == list(BLOCKS)
        and plan.get("conditions_per_block") == 10 and len(rows) == 120
        and plan.get("source_qualification_granted") is False
        and plan.get("scientific_parameter_domain_qualified") is False
        and plan.get("production_episodes") == 0,
        "PFC_SIZE_CALIBRATION_COMPLETE_PLAN_REQUIRED")
    expected = {(c, b, k) for c in CIRCUITS for b in BLOCKS for k in range(10)}
    actual = set()
    for index, row in enumerate(rows):
        identity, canonical = row["identity"], row["canonical_campaign_identity"]
        cell = tuple(identity[key] for key in ("circuit_index", "block_index", "condition_index"))
        quality.require(row["task_index"] == index and cell not in actual
            and cell in expected and row["condition"]["condition_id"] == quality.CONDITIONS[cell[2]]
            and identity["split"] == ("calibration" if cell[0] == 86 else "train")
            and row["split"] == identity["split"], "PFC_SIZE_CALIBRATION_IDENTITY_CHANGED")
        restored = dict(identity)
        for name in ("connectivity_seed", "initial_state_seed", "background_seed", "stimulus_seed"):
            restored[name] -= 100000000
        quality.require(restored == canonical, "PFC_SIZE_CALIBRATION_STREAM_NAMESPACE_CHANGED")
        name = Path(row["config_path"])
        quality.require(not name.is_absolute() and ".." not in name.parts,
            "PFC_SIZE_CALIBRATION_CONFIG_PATH_INVALID")
        quality.require(sha256_file(path.parent / name) == row["config_sha256"],
            "PFC_SIZE_CALIBRATION_CONFIG_CHANGED")
        actual.add(cell)
    quality.require(actual == expected, "PFC_SIZE_CALIBRATION_COMPLETE_DESIGN_REQUIRED")
    sampling = plan["comparison_sampling"]
    quality.require(sampling.get("within_size_conditions_paired") is True
        and sampling.get("cross_size_noise_pairing") is False
        and sampling.get("comparator_namespace_offset") == 100000000,
        "PFC_SIZE_CALIBRATION_PAIRING_SCOPE_CHANGED")
    return plan


def reference_tables(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    quality.require(sha256_file(path) == REFERENCE_SUMMARY_SHA256,
        "PFC_SIZE_CALIBRATION_FIXED_REFERENCE_CHANGED")
    report = quality.read_json(path)
    quality.require(report.get("status") == "COMPLETE_REFERENCE_QUALITY_DESCRIPTIVE_COHORT"
        and report.get("complete_episodes") == 150 and report.get("core_units") == 100000
        and report.get("blocks") == list(BLOCKS) and report.get("dt_ms") == .0125
        and report.get("source_qualification_granted") is False,
        "PFC_SIZE_CALIBRATION_COMPLETE_REFERENCE_REQUIRED")
    tables = {}
    for name in ("phase_block_summary.csv", "paired_contrast_summary.csv", "population_variance_summary.csv"):
        entry = report["outputs"][name]
        quality.require(entry["path"] == name and sha256_file(path.parent / name) == entry["sha256"],
            "PFC_SIZE_CALIBRATION_REFERENCE_TABLE_CHANGED")
        tables[name] = [row for row in csv_rows(path.parent / name) if int(row["circuit_index"]) in CIRCUITS]
    return report, tables


def descriptive_difference(original: dict, reference: dict, metric: str) -> dict:
    first, second = original.get(metric), reference.get(metric)
    available = first not in (None, "") and second not in (None, "")
    return dict(original_size_value=float(first) if first not in (None, "") else None,
        expanded_size_value=float(second) if second not in (None, "") else None,
        expanded_minus_original=float(second)-float(first) if available else None,
        difference_defined=available)


def compare_rows(original: list[dict], reference: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    def identity(row):
        return tuple(str(row[key]) for key in keys)
    lookup = {identity(row): row for row in reference}
    quality.require(len(lookup) == len(reference) == len(original) and set(map(identity, original)) == set(lookup),
        "PFC_SIZE_CALIBRATION_MATCHED_ESTIMANDS_REQUIRED")
    result = []
    for row in original:
        other = lookup[identity(row)]
        for metric in metrics:
            result.append(dict({key: row[key] for key in keys}, statistic=metric,
                **descriptive_difference(row, other, metric),
                comparison="unpaired_size_descriptive_difference"))
    return result


def aggregate(plan_path: Path, reference_path: Path, output: Path) -> dict:
    quality.require(not output.exists(), "PFC_SIZE_CALIBRATION_OUTPUT_EXISTS")
    plan = calibration_plan(plan_path)
    reference, reference_data = reference_tables(reference_path)
    reference_bindings = {row["episode_id"]: row for row in reference["bindings"]}
    # Resolve the complete120 denominator before loading arrays or creating output.
    paths = {}
    for row in plan["rows"]:
        directory = plan_path.parent / "readout" / f"task_{row['task_index']:03d}"
        quality.require((directory / "report.json").is_file(),
            f"PFC_SIZE_CALIBRATION_INCOMPLETE_CELL:{row['task_index']}")
        paths[row["identity"]["episode_id"]] = directory
    loaded, bindings, costs = {}, [], []
    for row in plan["rows"]:
        episode = row["identity"]["episode_id"]
        directory = paths[episode]
        report, phases, arrays = quality.load_cell(directory, row, core_units=2560)
        marker = report.get("source_size_comparison", {})
        quality.require(marker.get("comparator_role") == "original_size2560"
            and marker.get("coupled_core_units") == 2560 and marker.get("target_coupled_core_units") == 100000
            and marker.get("cross_size_noise_pairing") is False
            and marker.get("canonical_campaign_identity") == row["canonical_campaign_identity"]
            and marker.get("campaign_episodes") == marker.get("production_episodes") == 0,
            "PFC_SIZE_CALIBRATION_SOURCE_SIZE_DECLARATION_CHANGED")
        target = reference_bindings[episode]
        target_report_path = Path(target["readout"]) / "report.json"
        quality.require(sha256_file(target_report_path) == target["report_sha256"],
            "PFC_SIZE_CALIBRATION_REFERENCE_CELL_CHANGED")
        target_report = quality.read_json(target_report_path)
        quality.require(target_report["identity"] == row["canonical_campaign_identity"],
            "PFC_SIZE_CALIBRATION_REFERENCE_IDENTITY_CHANGED")
        loaded[episode] = report, phases, arrays
        bindings.append(dict(episode_id=episode, original_readout=str(directory.resolve()),
            original_report_sha256=sha256_file(directory / "report.json"), expanded_readout=target["readout"],
            expanded_report_sha256=target["report_sha256"], source_delivery_sha256=report["source_delivery_sha256"]))
        for scale, item in ((2560, report), (100000, target_report)):
            costs.append(dict(episode_id=episode, core_units=scale, source_simulator_cost=item.get("source_cost")))
    phase_summary, contrasts, contrast_summary, population, angles, pairings = [], [], [], [], [], []
    for circuit in CIRCUITS:
        for block in BLOCKS:
            rows = [row for row in plan["rows"] if row["identity"]["circuit_index"] == circuit
                    and row["identity"]["block_index"] == block]
            pairings.append(quality.paired_block_binding(rows, [loaded[row["identity"]["episode_id"]][0] for row in rows]))
        pairs = pairings[-3:]
        quality.require(len({row["initial_voltage_sha256"] for row in pairs}) == 3
            and len({row["final_rng_sha256"] for row in pairs}) == 3,
            "PFC_SIZE_CALIBRATION_DUPLICATED_HISTORIES")
        values = np.zeros((3, 8, len(quality.RATE_FIELDS), 10))
        for index, condition in enumerate(quality.CONDITIONS):
            rows = [next(row for row in plan["rows"] if (row["identity"]["circuit_index"], row["identity"]["block_index"], row["identity"]["condition_index"]) == (circuit, block, index)) for block in BLOCKS]
            reports, phases, arrays = zip(*(loaded.pop(row["identity"]["episode_id"]) for row in rows))
            for phase in range(8):
                for f, metric in enumerate(quality.RATE_FIELDS):
                    vector = [float(item[phase][metric]) for item in phases]
                    values[:, phase, f, index] = vector
                    phase_summary.append(dict(circuit_index=circuit, condition_id=condition, phase_index=phase,
                        metric=metric, **quality.descriptive(vector)))
                errors = [abs(float(item[phase]["circular_cue_error_deg"])) if item[phase]["circular_cue_error_deg"] else None for item in phases]
                phase_summary.append(dict(circuit_index=circuit, condition_id=condition, phase_index=phase,
                    metric="absolute_circular_cue_error_deg", **quality.descriptive(errors)))
            fields = np.stack([item["population_field_counts"] for item in arrays])
            unit_phase = np.stack([item["phase_unit_counts"] for item in arrays])
            unit_var = np.stack([item["temporal_unit_variance_counts"] for item in arrays])
            events = [quality.archived_events(paths[row["identity"]["episode_id"]], report) for row, report in zip(rows, reports)]
            matched = quality.matched_time_unit_statistics(events, unit_phase, fields, core_units=2560, n_e=2048)
            temporal, between = quality.covariance_axes(fields)
            for name, groups, lo, hi in (("excitatory", list(range(16)), 0, 2048), ("inhibitory", [16], 2048, 2560), ("all_units", list(range(17)), 0, 2560)):
                units = hi-lo
                temporal_variance = float(temporal[np.ix_(groups, groups)].sum()/units**2)
                diagonal = float(unit_var[:, lo:hi].mean(axis=0).sum()/units**2)
                population.append(dict(circuit_index=circuit, condition_id=condition, population=name,
                    units=units, blocks=3, time_bins=1000,
                    temporal_population_mean_variance_counts2=temporal_variance,
                    temporal_diagonal_contribution_counts2=diagonal,
                    temporal_ordered_offdiagonal_contribution_counts2=temporal_variance-diagonal,
                    between_block_matched_time_population_mean_variance_counts2=float(between[np.ix_(groups, groups)].sum()/units**2),
                    **matched["population"][name]))
            for row, report, item in zip(rows, reports, arrays):
                temporal_rows = csv_rows(paths[row["identity"]["episode_id"]]/"temporal_responses.csv")
                angles.extend(dict(circuit_index=circuit, block_index=row["identity"]["block_index"], condition_id=condition,
                    **value) for value in quality.lagged_angular_displacement(temporal_rows,
                        item["phase_edges_ticks"], bin_ticks=400, lag_ticks=[400, 800, 8000]))
        cue, did = quality.paired_contrasts(values)
        for label, data, conditions in (("cue_minus_omission", cue, quality.CONDITIONS[:5]), ("distractor_difference_of_differences", did, quality.CONDITIONS[1:5])):
            for phase in range(8):
                for f, metric in enumerate(quality.RATE_FIELDS):
                    for k, condition in enumerate(conditions):
                        key = dict(circuit_index=circuit, phase_index=phase, contrast=label, condition_id=condition, metric=metric)
                        contrast_summary.append(dict(key, **quality.descriptive(data[:, phase, f, k])))
                        contrasts.extend(dict(key, block_index=block, value=float(data[i, phase, f, k])) for i, block in enumerate(BLOCKS))
    phase_difference = compare_rows(phase_summary, reference_data["phase_block_summary.csv"],
        ("circuit_index", "condition_id", "phase_index", "metric"), ("mean", "sample_sd"))
    contrast_difference = compare_rows(contrast_summary, reference_data["paired_contrast_summary.csv"],
        ("circuit_index", "condition_id", "phase_index", "metric", "contrast"), ("mean", "sample_sd"))
    variance_metrics = tuple(key for key in population[0] if key.endswith("counts2"))
    variance_difference = compare_rows(population, reference_data["population_variance_summary.csv"],
        ("circuit_index", "condition_id", "population"), variance_metrics)
    output.mkdir(parents=True)
    tables = {"phase_summary.csv": phase_summary, "paired_block_contrasts.csv": contrasts,
        "paired_contrast_summary.csv": contrast_summary, "population_variance_summary.csv": population,
        "lagged_angular_displacement.csv": angles, "phase_size_differences.csv": phase_difference,
        "contrast_size_differences.csv": contrast_difference, "variance_size_differences.csv": variance_difference}
    for name, rows in tables.items():
        quality.write_rows(output/name, rows)
    freeze_json(output/"source_costs.json", costs)
    report = dict(status="COMPLETE120_DESCRIPTIVE_SIZE_CALIBRATION", original_size_episodes=120,
        expanded_size_episodes_reused=120, circuits=list(CIRCUITS), blocks=list(BLOCKS), conditions=list(quality.CONDITIONS),
        source_sizes=[2560, 100000], dt_ms=.0125, duration_ticks=400000, observation_bin_ticks=400,
        plan_sha256=sha256_file(plan_path), reference_summary_sha256=sha256_file(reference_path),
        bindings=bindings, realized_paired_blocks=pairings, cross_size_noise_pairing=False,
        uncertainty="three histories per circuit and size; descriptive mean/sample SD only; no confidence intervals, significance or equivalence",
        temporal_sample_axis="dependent5ms bins within each episode", between_block_sample_axis="matched-time joint initial/external histories",
        undefined_angle_rule="all three histories retained; undefined angles withhold mean and SD",
        scientific_parameter_domain_qualified=False, source_qualification_granted=False,
        scientific_claim_qualified=False, production_episodes=0, source_simulations=0,
        outputs={name: dict(path=name, sha256=sha256_file(output/name)) for name in (*tables, "source_costs.json")})
    freeze_json(output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-plan", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.comparison_plan, args.reference_summary, args.output)))


if __name__ == "__main__":
    main()
