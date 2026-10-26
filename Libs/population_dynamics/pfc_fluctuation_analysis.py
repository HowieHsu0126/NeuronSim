"""Describe matched-time all-unit and shared PFC count fluctuations.

Every condition retains its three complete joint initial/background blocks.
Time bins and units are observations, not additional independent replicates.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import yaml

from .pfc_delay_analysis import PHASES, read, require, write_json, write_rows
from .pfc_paired_source_difference import load_reference
from . import pfc_count_histograms
from .pfc_scale_analysis import _source_registration, _source_header, _source_values
from .provenance import sha256_file
from .source_variant_analysis import time_inner_product_gram, covariance_from_gram

UNIT_CHUNK = 4096


def unit_moments(counts):
    x = np.asarray(counts)
    require(x.ndim == 3 and x.shape[0] >= 2 and min(x.shape[1:]) > 0
            and x.dtype.kind in "iu" and np.all(x >= 0),
            "complete nonnegative block/time/unit integer counts required")
    # Remove the matched-time block mean, not a pooled temporal mean.
    return x.mean(axis=(0, 1)), x.var(axis=0, ddof=1, dtype=np.float64).mean(axis=0)


def field_covariance(counts):
    x = np.asarray(counts)
    require(x.ndim == 3 and x.shape[0] >= 2 and min(x.shape[1:]) > 0
            and x.dtype.kind in "iu" and np.all(x >= 0),
            "complete nonnegative block/time/field counts required")
    weights = np.ones((1, len(x)), dtype=np.int64)
    return covariance_from_gram(time_inner_product_gram(x), weights)[0]


def population_summary(mean, variance, group_counts, covariance, unit_groups, selected_groups):
    mask = np.isin(unit_groups, selected_groups)
    units = int(mask.sum())
    require(units > 0, "population must retain its complete units")
    population_counts = group_counts[:, :, selected_groups].sum(axis=2)
    observed = float((population_counts / units).var(axis=0, ddof=1).mean())
    diagonal = float(variance[mask].sum() / units**2)
    reconstructed = float(covariance[np.ix_(selected_groups, selected_groups)].sum() / units**2)
    tolerance = 128 * np.finfo(float).eps * max(1., abs(observed), abs(reconstructed))
    require(abs(observed-reconstructed) <= tolerance,
            "field covariance fails to reconstruct all-unit population variance")
    total_mean = float(mean[mask].sum())
    return dict(units=units, mean_count_per_unit=float(mean[mask].mean()),
        mean_single_unit_count_variance=float(variance[mask].mean()),
        population_mean_count_variance=observed,
        diagonal_contribution_to_population_variance=diagonal,
        ordered_offdiagonal_contribution_to_population_variance=observed-diagonal,
        field_reconstructed_population_variance=reconstructed,
        reconstruction_absolute_error=abs(observed-reconstructed),
        pooled_single_unit_ensemble_fano=None if total_mean == 0 else float(variance[mask].sum()/total_mean),
        silent_units=int(np.count_nonzero(mean[mask] == 0)))


def unit_mapping(n_e, n_i, preference_bins):
    require(all(type(x) is int and x > 0 for x in (n_e, n_i, preference_bins))
            and preference_bins <= n_e, "nonempty source-defined fields required")
    # Source coordinates are preferred angles, not anatomical positions.
    mapping = np.r_[np.arange(n_e, dtype=np.int64)*preference_bins//n_e,
                    np.full(n_i, preference_bins, dtype=np.int64)]
    sizes = np.bincount(mapping, minlength=preference_bins+1)
    require(np.all(sizes > 0), "each registered field must contain source units")
    return mapping, sizes


def prepared_events(raw, total_units, bin_ticks, total_bins):
    ticks, ids = raw["spike_ticks"], raw["spike_unit_indices"]
    require(ticks.ndim == ids.ndim == 1 and ticks.shape == ids.shape
            and ticks.dtype.kind in "iu" and ids.dtype.kind in "iu"
            and np.all((ids >= 0) & (ids < total_units))
            and np.all((ticks >= 0) & (ticks < bin_ticks*total_bins)),
            "full integer source event identities and time support required")
    order = np.argsort(ids, kind="stable")
    return ids[order], ticks[order]//bin_ticks


def unit_chunks(events, total_units, total_bins, chunk_units=UNIT_CHUNK):
    require(type(chunk_units) is int and chunk_units > 0, "positive unit chunk size required")
    for start in range(0, total_units, chunk_units):
        stop = min(total_units, start+chunk_units)
        blocks = []
        for ids, bins in events:
            left, right = np.searchsorted(ids, [start, stop])
            flat = bins[left:right]*(stop-start) + ids[left:right]-start
            blocks.append(np.bincount(flat, minlength=total_bins*(stop-start)).reshape(total_bins, stop-start))
        yield start, stop, np.stack(blocks)


def run(source_runs, config_path, reference, output, *, measurement_protocol=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    start_time = time.perf_counter()
    result = dict(status="RUNNING", new_source_simulations=0, production_episodes=0,
                  scientific_fidelity_qualified=False, finite_size_scaling_qualified=False,
                  confidence_intervals=False, p_values=False, q_values=False)
    try:
        config, registered, seeds, dimensions = _source_registration(config_path)
        config_sha = sha256_file(config_path)
        reference_report, bindings, _, count_path = load_reference(Path(reference), config_sha, registered)
        n_e, n_i, total = dimensions["n_e"], dimensions["n_i"], dimensions["coupled_core_units"]
        bin_ms, preference_bins = config["readout_bin_ms"], config["excitatory_preference_bins"]
        bin_ticks = round(bin_ms/config["dt_ms"])
        require(bin_ticks > 0 and np.isclose(bin_ticks*config["dt_ms"], bin_ms, rtol=0, atol=1e-9)
                and dimensions["duration_ticks"] % bin_ticks == 0, "complete source-aligned bins required")
        total_bins = dimensions["duration_ticks"]//bin_ticks
        mapping, sizes = unit_mapping(n_e, n_i, preference_bins)
        phases = []
        for name, (start, stop) in PHASES.items():
            left, right = round(start/bin_ms), round(stop/bin_ms)
            require(left < right <= total_bins and np.isclose(left*bin_ms, start, rtol=0, atol=1e-9)
                    and np.isclose(right*bin_ms, stop, rtol=0, atol=1e-9), "whole phase bins required")
            phases.append((name, left, right))
        headers = {}
        for source in source_runs:
            status, header = _source_header(source, config, registered, config_sha)
            require(header is not None and status["status"] == "COMPLETE"
                    and status["case_id"] not in headers, "one complete archive per source case required")
            case = status["case_id"]
            require(all(header["binding"][key] == bindings[case][key]
                        for key in ("report_sha256", "delivery_sha256", "source_runner_sha256")),
                    "source differs from complete phase-readout binding")
            headers[case] = header
        require(set(headers) == set(registered), "all nine registered cases required" if len(registered) == 9 else "all registered calibration cases required")
        populations = {"E": list(range(preference_bins)), "I": [preference_bins],
                       "all": list(range(preference_bins+1))}
        windows_ms = []
        if measurement_protocol is not None:
            protocol = yaml.safe_load(Path(measurement_protocol).read_text())
            measurements = protocol['measurements']
            windows_ms = measurements['all_unit_count_windows_ms']
            require(protocol['scope'] == 'development_calibration'
                    and measurements['observation_bin_ms'] == bin_ms
                    and measurements['phases'] == list(PHASES)
                    and isinstance(windows_ms, list) and windows_ms
                    and all(type(value) in (int, float) and np.isfinite(value) and value > 0
                            and value/bin_ms == round(value/bin_ms) for value in windows_ms)
                    and windows_ms == sorted(set(windows_ms))
                    and all((stop-start)/width == round((stop-start)/width)
                            for start, stop in PHASES.values() for width in windows_ms),
                    'complete registered count-observation windows required')
        histograms = {}
        conditions = ('omission', 'cue', 'strong_cue')
        moments, shared, summaries = {}, {}, []
        phase_values_checked = group_values_checked = 0
        with np.load(count_path, allow_pickle=False) as original_counts:
            for condition_index, condition in enumerate(conditions):
                events, direct_fields = [], []
                for seed in seeds:
                    case = f"pfc_{seed}_{condition}";header = headers[case]
                    raw, common, _ = _source_values(header, config, registered, dimensions, config_sha)
                    require(common == reference_report["common_execution_binding"],
                            "source execution differs from reference readout")
                    events.append(prepared_events(raw, total, bin_ticks, total_bins))
                    ids, bins = events[-1]
                    direct_fields.append(np.bincount(bins*len(sizes)+mapping[ids],
                        minlength=total_bins*len(sizes)).reshape(total_bins, len(sizes)))
                group_counts = np.stack(direct_fields)
                counted_fields = np.zeros_like(group_counts)
                means = np.empty((len(phases), total), dtype=np.float64)
                variances = np.empty_like(means)
                for left, right, counts in unit_chunks(events, total, total_bins):
                    for group in np.unique(mapping[left:right]):
                        counted_fields[:, :, group] += counts[:, :, mapping[left:right] == group].sum(axis=2)
                    for index, (phase, begin, end) in enumerate(phases):
                        selected = counts[:, begin:end]
                        for window_index, width in enumerate(windows_ms):
                            pfc_count_histograms.accumulate(histograms, selected, mapping[left:right],
                                round(width/bin_ms), condition=condition_index, phase=index, window=window_index)
                        means[index, left:right], variances[index, left:right] = unit_moments(selected)
                        for block, seed in enumerate(seeds):
                            expected = original_counts[f"pfc_{seed}_{condition}_{phase}"][left:right]
                            require(np.array_equal(selected[block].sum(axis=0), expected),
                                    "unit/bin counts disagree with independently replayed phase counts")
                            phase_values_checked += right-left
                require(np.array_equal(counted_fields, group_counts),
                        "direct field histogram differs from all-unit bin aggregation")
                group_values_checked += int(group_counts.size)
                moments[condition+"_mean_counts"] = means
                moments[condition+"_variance_counts"] = variances
                shared[condition+"_field_counts"] = group_counts
                covariance = []
                for index, (phase, begin, end) in enumerate(phases):
                    current = group_counts[:, begin:end]
                    cov = field_covariance(current);covariance.append(cov)
                    for population, selected_groups in populations.items():
                        row = population_summary(means[index], variances[index], current, cov, mapping, selected_groups)
                        summaries.append(dict(condition=condition, phase=phase, population=population,
                            joint_rng_blocks=len(seeds), time_bins=end-begin, bin_ms=bin_ms, **row))
                shared[condition+"_field_covariance_counts"] = np.stack(covariance)
        np.savez_compressed(output/"unit_moments.npz", unit_ids=np.arange(total, dtype=np.int64),
                            unit_field_indices=mapping, **moments)
        np.savez_compressed(output/"field_fluctuations.npz", field_unit_counts=sizes,
                            source_rng_seeds=np.asarray(seeds), **shared)
        write_rows(output/"population_variance.csv", summaries)
        output_names = ['unit_moments.npz', 'field_fluctuations.npz', 'population_variance.csv']
        if measurement_protocol is not None:
            histogram_file = 'unit_count_histograms.npz'
            result['unit_count_histograms'] = pfc_count_histograms.write(output/histogram_file, histograms,
                conditions=conditions, seeds=seeds, phases=PHASES, windows_ms=windows_ms, field_sizes=sizes)
            result['measurement_protocol_sha256'] = sha256_file(measurement_protocol)
            result['measurement_protocol_scope'] = 'observation definitions only; source run configuration fixes the actual complete cohort'
            result['count_histogram_implementation_sha256'] = sha256_file(Path(pfc_count_histograms.__file__))
            output_names.append(histogram_file)
        result.update(status="COMPLETE_SOURCE_FLUCTUATION_CALIBRATION",
            registered_source_cases=len(registered), completed_source_cases=len(headers),
            joint_rng_blocks=seeds, coupled_core_units=total, dt_ms=config["dt_ms"], bin_ms=bin_ms,
            phases_ms=PHASES, excitatory_preference_bins=preference_bins,
            field_mapping="floor(E_source_index * preference_bins / N_E); one complete I field",
            field_unit_counts=sizes.tolist(), unit_chunk=UNIT_CHUNK,
            source_bindings=[headers[case]["binding"] for case in registered],
            common_execution_binding=reference_report["common_execution_binding"],
            reference_archive=str(reference), reference_delivery_sha256=sha256_file(Path(reference)/"delivery.json"),
            config_sha256=config_sha, unit_phase_count_values_verified=phase_values_checked,
            field_bin_count_values_verified=group_values_checked, population_summaries=summaries,
            moments_units="mean: spikes/unit/bin; variance and covariance: counts squared",
            estimator="center across joint RNG blocks separately at every matched bin; sample covariance denominator blocks-1; average within each registered phase",
            covariance_decomposition="Var(sum_i X_i/N) = sum_i Var(X_i)/N^2 + sum_{i!=j} Cov(X_i,X_j)/N^2",
            interpretation=f"Development point estimates from {len(seeds)} joint initial/background blocks at one circuit size. Conditions and time bins do not add independent replicates. Offdiagonal covariance is signed. These measurements do not establish finite-size scaling, stationarity, conditional-on-history covariance, numerical equivalence or biological validity.",
            reused_statistics_implementation_sha256=sha256_file(Path(__file__).with_name("source_variant_analysis.py")),
            implementation_sha256=sha256_file(Path(__file__)),
            elapsed_analysis_seconds=time.perf_counter()-start_time,
            outputs={name: dict(sha256=sha256_file(output/name), bytes=(output/name).stat().st_size)
                     for name in output_names})
    except Exception as error:
        result.update(status="FAILED", error=str(error), error_type=type(error).__name__)
        write_json(output/"report.json", result)
        raise
    write_json(output/"report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-readouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-protocol", type=Path, help="Development observation definitions; cohort identity remains in --config")
    args = parser.parse_args(argv)
    run(args.source_run, args.config, args.reference_readouts, args.output, measurement_protocol=args.measurement_protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
