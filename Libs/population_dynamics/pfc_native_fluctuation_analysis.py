"""Reuse a completed native PFC reference for matched-time fluctuations."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import yaml

from . import pfc_scale_analysis as source
from .pfc_delay_analysis import PHASES, read, require, write_json, write_rows
from .pfc_fluctuation_analysis import unit_moments, unit_mapping, field_covariance, population_summary
from .provenance import sha256_file


def load_reference(archive, temporal):
    archive, temporal = Path(archive), Path(temporal)
    receipt = read(temporal/"delivery.json")
    require(receipt.get("status") == "ARCHIVED_PFC_SOURCE_READOUTS"
            and receipt.get("exit_code") == 0 and receipt.get("readout_mode") == "reference-temporal",
            "completed native temporal readout delivery required")
    inventory = source._source_manifest(receipt)
    report_path = source._source_bound_file(temporal, inventory, "analysis/analysis.json")
    previous = read(report_path)
    require(previous.get("status") == "COMPLETE_SOURCE_TEMPORAL_READOUTS"
            and previous.get("coupled_core_units") == 2560 and previous.get("bins") == 9000
            and previous.get("source_rng_seeds") == list(source.SOURCE_SEEDS)
            and previous.get("source_backend") == "native_brian2"
            and previous.get("scientific_fidelity_qualified") is False
            and previous.get("new_source_episodes") == 0,
            "complete nine-case native temporal analysis required")
    original = read(archive/"delivery.json")
    require(sha256_file(archive/"delivery.json") == previous["reference_delivery_sha256"]
            and original.get("status") == "ARCHIVED_REPLAYED_REFERENCE_SIZE_DIAGNOSTIC"
            and original.get("complete_source_episodes") == 9 and original.get("units") == 2560,
            "canonical native reference differs from verified temporal source")
    manifest = source._source_manifest(original)
    consumed = previous["consumed_manifest_files"]
    for name, digest in consumed.items():
        path = source._source_bound_file(archive, manifest, name)
        require(sha256_file(path) == digest, "native input changed since temporal verification: "+name)
    native_name = previous["native_report_path"]
    require(native_name in consumed and consumed[native_name] == previous["native_report_sha256"]
            == original["native_report_sha256"], "complete native report binding differs")
    native_root = (archive/native_name).parent
    native = read(archive/native_name)
    config_name = "Libs/configs/neural_population_pfc_reference_size.yaml"
    require(config_name in consumed and consumed[config_name] == previous["config_sha256"],
            "native reference configuration identity changed")
    config = yaml.safe_load((archive/config_name).read_text())
    require((config["excitatory_neurons"], config["inhibitory_neurons"], config["dt_ms"],
             config["readout_bin_ms"], config["duration_ms"]) == (2048, 512, .05, 5., 5000.),
            "original native units and clocks required")
    expected = {f"pfc_{seed}_{condition}": (seed, condition)
                for seed in source.SOURCE_SEEDS for condition in source.CONDITIONS}
    cases = {row["case_id"]: row for row in previous["completed_cases"]}
    native_cases = {row["case_id"]: row for row in native["cases"]}
    require(len(cases) == len(previous["completed_cases"]) == len(native["cases"]) == 9
            and set(cases) == set(native_cases) == set(expected), "all native reference cases required")
    for case_id, (seed, condition) in expected.items():
        row, original_case = cases[case_id], native_cases[case_id]
        binding = row["source_binding"]
        raw_path = (Path(native_name).parent/original_case["raw_file"]).as_posix()
        require(row["source_rng_seed"] == seed and row["condition"] == condition and row["bins"] == 1000
                and binding["raw_path"] == raw_path and raw_path in consumed
                and binding["raw_sha256"] == consumed[raw_path] == original_case["raw_sha256"],
                "native case raw identity differs from completed temporal readout")
    dataset, identity = source.dataset_binding(native_root, native, config, previous["config_sha256"], expected)
    return config, previous, native_root, native_cases, cases, dataset, identity


def run(archive, temporal, output):
    output = Path(output);output.mkdir(parents=True, exist_ok=False)
    result = dict(status="RUNNING", new_source_simulations=0, production_episodes=0,
                  scientific_fidelity_qualified=False, finite_size_scaling_qualified=False,
                  confidence_intervals=False, p_values=False, q_values=False)
    started = time.perf_counter()
    try:
        config, previous, root, native_cases, prior_cases, dataset, identity = load_reference(archive, temporal)
        n_e, n_i = config["excitatory_neurons"], config["inhibitory_neurons"]
        total, bin_ms, fields = n_e+n_i, config["readout_bin_ms"], config["excitatory_preference_bins"]
        require(fields == 16, "original native preference fields required")
        mapping, sizes = unit_mapping(n_e, n_i, fields)
        populations = {"E": list(range(fields)), "I": [fields], "all": list(range(fields+1))}
        phases = [(name, round(start/bin_ms), round(stop/bin_ms))
                  for name, (start, stop) in PHASES.items()]
        moments, shared, rows, bindings = {}, {}, [], []
        exclusions = values_consumed = 0
        for condition in ("omission", "cue", "strong_cue"):
            blocks = []
            for seed in source.SOURCE_SEEDS:
                case_id = f"pfc_{seed}_{condition}";case = native_cases[case_id]
                raw = source.raw_values(root, case)
                dense = source.archive_window(dataset, case, raw, seed, condition, identity)
                require(dense.dtype.kind in "iu", "native archive reader must return exact integer counts")
                excluded = int(np.count_nonzero(raw["spike_ticks"] >= source.DURATION_TICKS))
                require(excluded == prior_cases[case_id]["terminal_tick_events_excluded"],
                        "native terminal-boundary accounting changed")
                blocks.append(dense)
                bindings.append(dict(case_id=case_id, source_rng_seed=seed, condition=condition,
                    **prior_cases[case_id]["source_binding"], terminal_tick_events_excluded=excluded))
                exclusions += excluded;values_consumed += int(dense.size)
            counts = np.stack(blocks)
            group_counts = np.stack([counts[:, :, mapping == group].sum(axis=2)
                                     for group in range(len(sizes))], axis=2)
            require(np.array_equal(group_counts.sum(axis=2), counts.sum(axis=2)),
                    "complete native unit-to-field counts changed")
            means, variances, covariance = [], [], []
            for phase, left, right in phases:
                mean, variance = unit_moments(counts[:, left:right])
                current = group_counts[:, left:right];cov = field_covariance(current)
                means.append(mean);variances.append(variance);covariance.append(cov)
                for population, groups in populations.items():
                    rows.append(dict(condition=condition, phase=phase, population=population,
                        joint_rng_blocks=len(source.SOURCE_SEEDS), time_bins=right-left, bin_ms=bin_ms,
                        **population_summary(mean, variance, current, cov, mapping, groups)))
            moments[condition+"_mean_counts"] = np.stack(means)
            moments[condition+"_variance_counts"] = np.stack(variances)
            shared[condition+"_field_counts"] = group_counts
            shared[condition+"_field_covariance_counts"] = np.stack(covariance)
        np.savez_compressed(output/"unit_moments.npz", unit_ids=np.arange(total, dtype=np.int64),
                            unit_field_indices=mapping, **moments)
        np.savez_compressed(output/"field_fluctuations.npz", field_unit_counts=sizes,
                            source_rng_seeds=np.asarray(source.SOURCE_SEEDS), **shared)
        write_rows(output/"population_variance.csv", rows)
        result.update(status="COMPLETE_SOURCE_FLUCTUATION_CALIBRATION", source_format="legacy_native_brian2",
            source_backend=previous["source_backend"], source_versions=previous["source_versions"],
            registered_source_cases=9, completed_source_cases=9, coupled_core_units=total,
            joint_rng_blocks=list(source.SOURCE_SEEDS), dt_ms=config["dt_ms"], bin_ms=bin_ms,
            phases_ms=PHASES, excitatory_preference_bins=fields, field_unit_counts=sizes.tolist(),
            source_archive=str(archive), reference_archive=str(temporal),
            reference_delivery_sha256=sha256_file(Path(temporal)/"delivery.json"),
            source_delivery_sha256=previous["reference_delivery_sha256"], config_sha256=previous["config_sha256"],
            source_bindings=bindings, native_report_path=previous["native_report_path"],
            unit_bin_count_values_consumed=values_consumed, terminal_tick_events_excluded=exclusions,
            prior_full_unit_bin_verification_reused=True, population_summaries=rows,
            current_observation_status=previous["current_observation_status"],
            estimator="sample variance/covariance across joint RNG blocks at each matched 5ms bin, then average within registered phases",
            interpretation="Native reference development point estimates; size comparisons also differ in backend and source RNG cohort, and possibly clock. No size-only inference or finite-size law follows. Time bins and units are not independent replicates.",
            moments_units="mean: spikes/unit/bin; variance and covariance: counts squared",
            shared_calculation_sha256=sha256_file(Path(__file__).with_name("pfc_fluctuation_analysis.py")),
            reused_statistics_implementation_sha256=sha256_file(Path(__file__).with_name("source_variant_analysis.py")),
            implementation_sha256=sha256_file(Path(__file__)),
            elapsed_analysis_seconds=time.perf_counter()-started,
            outputs={name: dict(sha256=sha256_file(output/name), bytes=(output/name).stat().st_size)
                     for name in ("unit_moments.npz", "field_fluctuations.npz", "population_variance.csv")})
    except Exception as error:
        result.update(status="FAILED", error=str(error), error_type=type(error).__name__)
        write_json(output/"report.json", result)
        raise
    write_json(output/"report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-reference-archive", type=Path, required=True)
    parser.add_argument("--reference-readouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.native_reference_archive, args.reference_readouts, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
