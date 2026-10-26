"""Fresh stratified sampling of paired archived PFC first-passage histories.

The full census supplies observed timing labels and finite-frame weights.
Conditional-probability estimates require separate integration; no old estimates
enter the fresh selection or weighted mean. No population confidence coverage
or scientific qualification follows from the conditional variance diagnostics.
SRSWOR means simple random sampling without replacement.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from . import pfc_first_passage_history as history
from .pfc_first_passage_clock_comparison import IDENTITY
from .pfc_first_passage_history_summary import variance_components as stratum_variance


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("PFC_STRATIFIED_HISTORY_" + message)


def validate_settings(settings: Mapping[str, Any]) -> None:
    """Accept only the two registered window, trial and seed-namespace profiles."""
    profile = (settings["sample_size"], settings["importance_sampling"]["samples"],
               settings["integration_seed_namespace"])
    require(all(type(value) is int for value in profile)
            and profile in ((64, 100000, 3), (640, 10000, 4)), "REGISTERED_WINDOW_INTEGRATION_BUDGET")
    seed, namespace = settings["source_rng_seed"], profile[2]
    require(type(seed) is int and settings["case_id"] == f"pfc_{seed}_strong_cue"
            and settings["expected_units"] == 100000 and settings["field_index"] == 5
            and settings["phase"] == "cue" and settings["window_ms"] == 5., "FROZEN_FRESH_STRATIFIED_DESIGN")
    components = settings["sampling_seed_components"]
    require(isinstance(components, list) and components == [20261009, seed, 5, namespace]
            and all(type(value) is int for value in components), "SAMPLING_SEED_NAMESPACE")


def allocation(counts: Sequence[int], budget: int = 64) -> list[dict[str, Any]]:
    """Allocate at least two per noncensus positive stratum, then deficit shares."""
    populations = np.asarray(counts)
    require(populations.shape == (16,) and populations.dtype.kind in "iu"
            and np.all(populations >= 0) and populations.sum() > 0, "SIXTEEN_FULL_FRAME_COUNTS")
    require(type(budget) is int and budget > 0, "POSITIVE_NOMINAL_BUDGET")
    populations = populations.astype(np.int64)
    positive = int(populations[1:].sum())
    target = min(budget, positive)
    selected = np.minimum(populations, 2)
    selected[0] = 0
    require(int(selected.sum()) <= target, "BUDGET_BELOW_STRATUM_MINIMUM")
    while int(selected.sum()) < target:
        available = [h for h in range(1, 16) if selected[h] < populations[h]]
        # Integer deficits implement proportional shares without floating tie drift.
        h = min(available, key=lambda h: (-(target * int(populations[h]) - int(selected[h]) * positive), h))
        selected[h] += 1
    total = int(populations.sum())
    return [dict(stratum=h, population_size=int(n), sample_size=int(selected[h]), weight=float(n / total),
                 inclusion_probability=None if n == 0 else float(selected[h] / n),
                 sampling=("empty" if n == 0 else "known_both_clock_zero" if h == 0
                           else "census" if selected[h] == n else "SRSWOR"))
            for h, n in enumerate(populations)]


def stratified_components(
    estimates: Sequence[float], inner_variances: Sequence[float], strata: Sequence[int],
    populations: Sequence[int], *, budget: int = 64, allow_signed: bool = False,
) -> dict[str, Any]:
    """Weighted finite-frame mean and unbiased conditional variance diagnostics."""
    design = allocation(populations, budget)
    values, variances, labels = np.asarray(estimates, dtype=float), np.asarray(inner_variances, dtype=float), np.asarray(strata)
    require(values.ndim == variances.ndim == labels.ndim == 1
            and values.shape == variances.shape == labels.shape
            and (labels.dtype.kind in "iu" or labels.size == 0)
            and np.isfinite(values).all() and np.isfinite(variances).all()
            and np.all(variances >= 0) and (allow_signed or np.all(values >= 0)), "SCORE_OR_VARIANCE_SUPPORT")
    require(values.size == sum(row["sample_size"] for row in design)
            and np.all((labels > 0) & (labels < 16)), "COMPLETE_SELECTED_DENOMINATOR")
    diagnostics = []
    mean = within = inner = sampling = 0.
    for spec in design:
        h, n, population = spec["stratum"], spec["sample_size"], spec["population_size"]
        mask = labels == h
        require(int(mask.sum()) == n, "MISSING_OR_EXTRA_STRATUM:" + str(h))
        if n > 1:
            components = stratum_variance(values[mask], variances[mask], population, allow_signed=allow_signed)
        elif n == 1:
            require(population == 1, "NONCENSUS_SINGLETON")
            variance = float(variances[mask][0])
            components = dict(sample_mean=float(values[mask][0]), sampled_estimate_variance=None,
                              mean_inner_variance=variance, latent_unit_window_variance_diagnostic=None,
                              within_archive_mean_variance_diagnostic=variance,
                              inner_mc_mean_variance_diagnostic=variance,
                              unit_window_sampling_mean_variance_diagnostic=0.)
        else:
            require(h == 0 or population == 0, "UNSAMPLED_POSITIVE_STRATUM")
            components = dict(sample_mean=0., sampled_estimate_variance=0., mean_inner_variance=0.,
                              latent_unit_window_variance_diagnostic=0., within_archive_mean_variance_diagnostic=0.,
                              inner_mc_mean_variance_diagnostic=0., unit_window_sampling_mean_variance_diagnostic=0.)
        w = spec["weight"]
        row = dict(spec, **components)
        row.update(weighted_mean_contribution=w * components["sample_mean"],
                   weighted_within_variance=w * w * components["within_archive_mean_variance_diagnostic"],
                   weighted_inner_variance=w * w * components["inner_mc_mean_variance_diagnostic"],
                   weighted_sampling_variance=w * w * components["unit_window_sampling_mean_variance_diagnostic"])
        diagnostics.append(row)
        mean += row["weighted_mean_contribution"]
        within += row["weighted_within_variance"]
        inner += row["weighted_inner_variance"]
        sampling += row["weighted_sampling_variance"]
    return dict(sample_mean=mean, sample_denominator=int(values.size), nominal_sample_budget=budget,
                frame_size=int(np.asarray(populations).sum()), stratum_diagnostics=diagnostics,
                within_archive_mean_variance_diagnostic=within, inner_mc_mean_variance_diagnostic=inner,
                unit_window_sampling_mean_variance_diagnostic=sampling,
                empirical_inner_mc_variance_of_sample_mean=inner,
                negative_variance_diagnostics_preserved=True, importance_estimates_clipped=False,
                population_confidence_interval_computed=False,
                uncertainty_scope="Conditional archived-frame sampling and inner Monte Carlo diagnostics; no source-history confidence interval.")


def _census_populations(report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    labels = np.asarray(arrays["joint_time_bin"])
    require(labels.shape == (report["frame_size"],) and labels.dtype.kind in "iu"
            and np.all((labels >= 0) & (labels < 16)), "FULL_CENSUS_ASSIGNMENTS")
    counts = np.bincount(labels, minlength=16)
    require(report["joint_time_bin_counts"] == {str(h): int(n) for h, n in enumerate(counts)}, "CENSUS_COUNTS_CHANGED")
    return counts


def sampling_callbacks(report: Mapping[str, Any], arrays: Mapping[str, np.ndarray], clock: str):
    """Return existing history.run callbacks bound to one clock's full census."""
    require(clock in ("coarse", "fine"), "CLOCK_NAME")
    populations = _census_populations(report, arrays)
    meta = report["clocks"][clock]
    unit_indices = np.asarray(arrays["unit_indices"])

    def frame_sampler(config, *, field, phase, window_ms, sample_size, seed_components):
        require(all(config[key] == meta[key] for key in
                    ("excitatory_neurons", "inhibitory_neurons", "excitatory_preference_bins", "duration_ms", "dt_ms"))
                and (field, phase, window_ms) == (meta["field_index"], meta["phase"], meta["window_ms"]), "CLOCK_FRAME_CONFIG")
        mapping, _ = history.unit_mapping(config["excitatory_neurons"], config["inhibitory_neurons"], config["excitatory_preference_bins"])
        require(np.array_equal(unit_indices, np.flatnonzero(mapping == field)), "SOURCE_UNIT_MAPPING")
        require(list(history.PHASES[phase]) == [meta["phase_start_ms"], meta["phase_stop_ms"]]
                and report["frame_size"] == len(unit_indices) * meta["windows"], "PHYSICAL_FRAME_BOUNDS")
        require(isinstance(seed_components, (list, tuple)) and len(seed_components) > 0
                and all(type(value) is int and 0 <= value < 2**32 for value in seed_components), "FRESH_SAMPLING_SEED_VECTOR")
        design = allocation(populations, sample_size)
        rows = []
        for spec in design:
            h, n = spec["stratum"], spec["sample_size"]
            if n == 0:
                continue
            ids = np.flatnonzero(arrays["joint_time_bin"] == h)
            rng = np.random.RandomState(np.asarray([*seed_components, h], dtype=np.uint32))
            for item in rng.choice(ids, size=n, replace=False):
                frame_index = int(item)
                window = frame_index // len(unit_indices)
                start = meta["phase_start_ms"] + window * window_ms
                stop = start + window_ms
                lo, hi = round(start / config["dt_ms"]), round(stop / config["dt_ms"])
                require(np.isclose(lo * config["dt_ms"], start, rtol=0, atol=1e-12)
                        and np.isclose(hi * config["dt_ms"], stop, rtol=0, atol=1e-12)
                        and lo == arrays[clock + "_window_start_tick"][window]
                        and hi == arrays[clock + "_window_stop_tick"][window], "PHYSICAL_WINDOW_CLOCK")
                rows.append(dict(sample_index=len(rows), frame_index=frame_index,
                                 unit_index=int(unit_indices[frame_index % len(unit_indices)]),
                                 window_start_ms=start, window_stop_ms=stop, window_start_tick=lo, window_stop_tick=hi,
                                 joint_time_bin=h, inclusion_probability=spec["inclusion_probability"]))
        frame = dict(field_index=field, population="E", phase=phase, phase_start_ms=meta["phase_start_ms"],
                     phase_stop_ms=meta["phase_stop_ms"], window_ms=window_ms, units=len(unit_indices),
                     windows_per_unit=meta["windows"], frame_size=report["frame_size"], sample_size=len(rows),
                     nominal_sample_budget=sample_size, seed_components=list(seed_components), allocation=design,
                     selected_frame_indices=[row["frame_index"] for row in rows],
                     sampling="SRSWOR_within_census_joint_timing_strata; independent_seed_vector_plus_stratum",
                     unit_order="source_indices_ascending", frame_order="window_major_then_unit")
        return rows, frame

    def frame_summary(rows, frame):
        require(frame["allocation"] == allocation(populations, frame["nominal_sample_budget"])
                and len(rows) == frame["sample_size"]
                and [row["frame_index"] for row in rows] == frame["selected_frame_indices"]
                and len({row["frame_index"] for row in rows}) == len(rows), "SELECTED_FRAME_CHANGED")
        for sample_index, row in enumerate(rows):
            item = row["frame_index"]
            require(type(item) is int and 0 <= item < report["frame_size"] and row["sample_index"] == sample_index,
                    "SAMPLE_IDENTITY")
            window, unit = divmod(item, len(unit_indices))
            count = int(arrays[clock + "_observed_spike_count"][item])
            first = int(arrays[clock + "_first_spike_tick"][item])
            refractory = int(arrays[clock + "_refractory_ticks"][item])
            lo, hi = (int(arrays[clock + suffix][window]) for suffix in ("_window_start_tick", "_window_stop_tick"))
            require(row["unit_index"] == unit_indices[unit]
                    and (row["window_start_tick"], row["window_stop_tick"]) == (lo, hi)
                    and (row["window_start_ms"], row["window_stop_ms"]) ==
                        (meta["phase_start_ms"] + window * meta["window_ms"], meta["phase_start_ms"] + (window + 1) * meta["window_ms"])
                    and row["observed_count"] == count and row["first_spike_tick"] == (None if first < 0 else first)
                    and row["refractory_ticks"] == refractory, "CLASSIFICATION_DIFFERS_FROM_CENSUS")
            remaining = 0 if first < 0 else hi - first - refractory
            edges = np.rint(np.asarray(meta["time_edges_ms"]) / meta["dt_ms"]).astype(np.int64)
            time_bin = -1 if first < 0 or remaining <= 0 else int(np.searchsorted(edges[1:], remaining, side="left"))
            h = int(arrays["joint_time_bin"][item])
            require(time_bin == arrays[clock + "_time_bin"][item] and row["joint_time_bin"] == h
                    and row["inclusion_probability"] == frame["allocation"][h]["inclusion_probability"], "TIMING_STRATUM_CHANGED")
            category = ("no_first_spike" if count == 0 else "suppressed_network_reference_required" if count > 1
                        else "refractory_covers_remaining_window" if remaining <= 0 else "conditional_integration_required")
            require(row["category"] == category, "CLASSIFICATION_CATEGORY_CHANGED")
            if time_bin < 0:
                require(row["conditional_estimate"] == row["inner_monte_carlo_variance"] == 0., "STRUCTURAL_ZERO_SCORE")
            if count > 1:
                require(row.get("suppressed_reference_resolved") is True, "UNRESOLVED_MULTISPIKE_RETAINED")
        result = stratified_components([r["conditional_estimate"] for r in rows],
                                       [r["inner_monte_carlo_variance"] for r in rows],
                                       [r["joint_time_bin"] for r in rows], populations,
                                       budget=frame["nominal_sample_budget"])
        result.update(actual_multispike_frequency=meta["multispike_windows"] / report["frame_size"],
                      actual_multispike_frequency_scope="Exact complete archived-frame census",
                      selected_physical_windows=len(rows), census_classification_matches=True,
                      zero_hit_eligible_histories=sum(row["conditional_estimate"] == 0 and row["category"] in
                                                     ("conditional_integration_required", "suppressed_network_reference_required") for row in rows),
                      estimand="Full archived unit-window mean of first-spike-indicator times conditional second-spike probability",
                      weight_scope="Full census weights retained, including unsampled both-clock structural zeros")
        return result

    return frame_sampler, frame_summary


def paired_components(coarse_rows, fine_rows, frame):
    require(len(coarse_rows) == len(fine_rows) == frame["sample_size"]
            and [r["frame_index"] for r in coarse_rows] == frame["selected_frame_indices"]
            and len({r["frame_index"] for r in coarse_rows}) == len(coarse_rows), "PAIRED_DENOMINATOR")
    pairs = []
    for coarse, fine in zip(coarse_rows, fine_rows):
        require(all(coarse[key] == fine[key] for key in (*IDENTITY, "joint_time_bin", "inclusion_probability")), "PAIRED_PHYSICAL_IDS")
        delta = fine["conditional_estimate"] - coarse["conditional_estimate"]
        pairs.append({**{key: coarse[key] for key in (*IDENTITY, "joint_time_bin")},
                      "coarse_count": coarse["observed_count"], "fine_count": fine["observed_count"],
                      "coarse_first_spike_tick": coarse["first_spike_tick"], "fine_first_spike_tick": fine["first_spike_tick"],
                      "coarse_category": coarse["category"], "fine_category": fine["category"],
                      "coarse_estimate": coarse["conditional_estimate"], "fine_estimate": fine["conditional_estimate"],
                      "fine_minus_coarse_count_ge2": delta, "fine_minus_coarse_cdf_le1": -delta,
                      "inner_variance_of_difference": fine["inner_monte_carlo_variance"] + coarse["inner_monte_carlo_variance"]})
    populations = [row["population_size"] for row in frame["allocation"]]
    labels = [row["joint_time_bin"] for row in pairs]
    stats = stratified_components([r["fine_minus_coarse_count_ge2"] for r in pairs],
                                  [r["inner_variance_of_difference"] for r in pairs], labels, populations,
                                  budget=frame["nominal_sample_budget"], allow_signed=True)
    stats.update(fine_minus_coarse_count_ge2=stats["sample_mean"], fine_minus_coarse_cdf_le1=-stats["sample_mean"],
                 paired_physical_frame_equal=True, third_spike_probability_established=False,
                 inner_randomization_scope="Independent auxiliary seed vectors distinguished by clock tick tag; finite-PRNG independence is not certified",
                 scientific_fidelity_qualified=False, confirmation_execution_released=False, production_episodes=0)
    return stats, pairs


def run_pair(settings, census_dir, coarse_dir, fine_dir, output):
    """Integrate a fresh paired sample through the existing source-reference path."""
    validate_settings(settings)
    seed = settings["source_rng_seed"]
    census_dir, output = Path(census_dir), Path(output)
    census = json.loads((census_dir / "report.json").read_text())
    require(census["status"] == "COMPLETE_PFC_HISTORY_TIMING_CENSUS" and census["source_rng_seed"] == seed
            and census["case_id"] == settings["case_id"] and census["frame_size"] == 200000, "COMPLETE_BOUND_CENSUS")
    with np.load(census_dir / "frame_assignments.npz", allow_pickle=False) as archive:
        assignments = {key: archive[key] for key in archive.files}
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = dict(status="FAILED_PFC_STRATIFIED_HISTORY_PAIR", source_rng_seed=seed, case_id=settings["case_id"],
                  nominal_sample_budget=settings["sample_size"],
                  nominal_inner_trial_budget_per_clock=settings["sample_size"] * settings["importance_sampling"]["samples"],
                  integration_seed_namespace=settings["integration_seed_namespace"],
                  sampling_seed_components=list(settings["sampling_seed_components"]),
                  new_independent_source_blocks=0, new_full_network_episodes=0,
                  production_episodes=0, scientific_fidelity_qualified=False, confirmation_execution_released=False,
                  population_confidence_interval_computed=False, third_spike_probability_established=False,
                  future_stratified_precision_established=False, population_history_uncertainty_established=False)
    try:
        clock_reports, clock_rows = {}, {}
        for clock, directory, dt, stream in (("coarse", coarse_dir, .05, "native_numpy_stream"),
                                             ("fine", fine_dir, .025, "coupled_binomial_clock_refinement_stream")):
            spec = settings["clocks"][clock]
            require(spec["dt_ms"] == dt and spec["input_stream"] == stream, "REGISTERED_CLOCK_INPUT_STREAM")
            arguments = dict(settings, **spec)
            paired_frame = output / "coarse" / "selected_frame.json" if clock == "fine" else None
            if paired_frame is not None:
                arguments["paired_coarse"] = "../coarse/selected_frame.json"
            sampler, summary = sampling_callbacks(census, assignments, clock)
            clock_reports[clock] = history.run(arguments, Path(directory), output / clock, paired_frame,
                                               frame_sampler=sampler, frame_summary=summary)
            clock_rows[clock] = (json.loads((output / clock / "integrated_frame.json").read_text())
                                 if clock_reports[clock]["sample_denominator"] else [])
        frame = clock_reports["coarse"]["frame"]
        require(frame == clock_reports["fine"]["frame"], "PAIRED_FRAME_CHANGED")
        stats, rows = paired_components(clock_rows["coarse"], clock_rows["fine"], frame)
        (output / "paired_frame.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        diagnostics = [dict(clock=clock, **row) for clock, report in
                       [*clock_reports.items(), ("fine_minus_coarse", stats)] for row in report["stratum_diagnostics"]]
        with (output / "stratum_diagnostics.csv").open("w", newline="") as stream:
            fields = list(dict.fromkeys(key for row in diagnostics for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(diagnostics)
        result.update(status="COMPLETE_PFC_STRATIFIED_HISTORY_PAIR", statistics=stats, frame=frame,
                      selected_physical_windows=frame["sample_size"],
                      per_clock={clock: dict(report="./" + clock + "/report.json", sample_mean=report["sample_mean"],
                                                integrated_histories=report["integrated_histories"],
                                                inner_trials=report["integrated_histories"] * settings["importance_sampling"]["samples"],
                                                reconstruction=report["reconstruction"], reconstruction_seconds=report["reconstruction_seconds"],
                                                wall_seconds=report["wall_seconds"], zero_hit_eligible_histories=report["zero_hit_eligible_histories"])
                                 for clock, report in clock_reports.items()},
                      paired_frame_file="paired_frame.json", stratum_diagnostics_file="stratum_diagnostics.csv",
                      scope="Fresh conditional archived-frame diagnostic using full timing-census strata; no population inference or release")
    except Exception as exc:
        result["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "census", "coarse", "fine", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run_pair(yaml.safe_load(args.config.read_text()), args.census, args.coarse, args.fine, args.output)
    print(json.dumps({key: result[key] for key in ("status", "source_rng_seed", "selected_physical_windows")}))


if __name__ == "__main__":
    main()
