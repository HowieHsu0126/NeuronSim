"""Verify all 32 fresh stratified history pairs before descriptive aggregation.

The census and importance reports are read from authenticated staged artifacts.
Missing receipts do not imply terminal jobs. Conditional variance diagnostics
and comparisons with uniform sampling do not establish population confidence
coverage, clock equivalence, source qualification or matched execution cost.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import yaml

from .pfc_first_passage_history_summary import bound_json, between_components
from .pfc_stratified_history import sampling_callbacks, paired_components, validate_settings


PRODUCER_CONFIG = "Libs/configs/neural_population_pfc_stratified_history.yaml"
TESTED_FILES = tuple("Libs/population_dynamics/" + name + ".py" for name in (
    "pfc_first_passage_history", "pfc_stratified_history", "pfc_first_passage_history_summary",
    "pfc_first_passage_clock_comparison", "pfc_history_timing_census", "pfc_first_passage_importance",
    "pfc_first_passage_reference")) + (PRODUCER_CONFIG,)
STAT_FIELDS = ("sample_mean", "within_archive_mean_variance_diagnostic", "inner_mc_mean_variance_diagnostic",
               "unit_window_sampling_mean_variance_diagnostic")
COST_FIELDS = ("integrated_histories", "inner_trials", "eligible_zero_estimates", "zero_hit_histories",
               "nonzero_conditional_estimates", "inner_integration_seconds", "reconstruction_seconds",
               "analysis_seconds", "reconstruction_ticks", "source_events_consumed")


def require(condition, message):
    if not condition:
        raise ValueError("PFC_STRATIFIED_SUMMARY_" + message)


def bound_payload(root, entries, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and relative in entries, "BOUND_RELATIVE_PATH")
    payload = (root / path).read_bytes()
    require(len(payload) == entries[relative]["bytes"]
            and hashlib.sha256(payload).hexdigest() == entries[relative]["sha256"], "ARTIFACT_CHANGED:" + relative)
    return payload


def receipt(root, status, expected_sha256=None):
    payload = (root / "delivery.json").read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(expected_sha256 is None or digest == expected_sha256, "DELIVERY_CHANGED")
    delivery = json.loads(payload)
    require(delivery["verified"] is True and delivery["status"] == status, "COMPLETE_DELIVERY_REQUIRED")
    entries = {entry["path"]: entry for entry in delivery["files"]}
    require(len(entries) == len(delivery["files"]), "DUPLICATE_DELIVERY_ENTRY")
    return entries, digest


def unreleased(report):
    require(all(report[key] is False for key in ("scientific_fidelity_qualified", "confirmation_execution_released"))
            and report["production_episodes"] == report["new_independent_source_blocks"] == 0,
            "UNRELEASED_SCIENTIFIC_SCOPE")


def _read_clock(root, entries, seed, clock, block_config, census, arrays):
    prefix = "validation/" + clock + "/"
    actual = bound_json(root, entries, prefix + "report.json")
    selected = bound_json(root, entries, prefix + "selected_frame.json")
    require(actual["status"] == "COMPLETE_PFC_FIRST_PASSAGE_HISTORY_PILOT" and actual.get("error") is None
            and actual["source_rng_seed"] == seed, "COMPLETE_CLOCK_RESULT")
    unreleased(actual)
    require(actual["population_confidence_interval_computed"] is False
            and actual["population_history_uncertainty_established"] is False
            and actual["third_spike_probability_established"] is False, "CLOCK_INFERENCE_SCOPE")
    meta = census["clocks"][clock]
    config = {key: meta[key] for key in ("excitatory_neurons", "inhibitory_neurons", "excitatory_preference_bins", "duration_ms", "dt_ms")}
    spec = block_config["clocks"][clock]
    require(config["dt_ms"] == spec["dt_ms"] and config["duration_ms"] == 5000.
            and config["excitatory_neurons"] + config["inhibitory_neurons"] == 100000
            and actual["input_stream"] == spec["input_stream"], "SOURCE_CLOCK_SCALE")
    sampler, summary = sampling_callbacks(census, arrays, clock)
    expected_rows, frame = sampler(config, field=5, phase="cue", window_ms=5., sample_size=block_config["sample_size"],
                                    seed_components=block_config["sampling_seed_components"])
    require(selected == dict(frame=frame, rows=expected_rows) and actual["frame"] == frame,
            "FRESH_RANDOM_SELECTION_CHANGED")
    relative = prefix + "integrated_frame.json"
    require(relative in entries or len(expected_rows) == 0, "MISSING_INTEGRATED_ROWS")
    rows = bound_json(root, entries, relative) if relative in entries else []
    statistics = summary(rows, frame)
    require(all(actual[key] == value for key, value in statistics.items()), "WEIGHTED_CLOCK_ARITHMETIC")
    counts = dict(integrated_histories=0, inner_trials=0, eligible_zero_estimates=0, zero_hit_histories=0,
                  nonzero_conditional_estimates=0, inner_integration_seconds=0.)
    importance = block_config["importance_sampling"]
    for original, row in zip(expected_rows, rows):
        require(all(row[key] == value for key, value in original.items()), "SELECTED_ROW_IDENTITY")
        category = row["category"]
        if category in ("conditional_integration_required", "suppressed_network_reference_required"):
            if category == "suppressed_network_reference_required":
                replay = bound_json(root, entries, prefix + row["suppressed_reference_report"])
                require(row.get("suppressed_reference_resolved") is True
                        and replay["status"] == "COMPLETE_PFC_HISTORY_SUPPRESSED_REFERENCE"
                        and replay["source_rng_seed"] == seed and replay["source_reference"] == spec["source"]
                        and replay["dt_ms"] == config["dt_ms"] and replay["network_feedback_recomputed"] is True
                        and replay["source_law_trajectory"] is False
                        and all(replay[key] == row[key] for key in ("unit_index", "first_spike_tick", "window_stop_tick"))
                        and replay["observed_focal_window_ticks"] == row["observed_window_ticks"], "UNRESOLVED_MULTISPIKE_REFERENCE")
            inner = bound_json(root, entries, prefix + row["integration_report"])
            require(inner["status"] == "COMPLETE_CONDITIONAL_FIRST_PASSAGE_IMPORTANCE" and inner.get("error") is None
                    and inner["samples"] == importance["samples"]
                    and inner["reference_archive"] == spec["source"]["archive"]
                    and inner["reference_delivery_sha256"] == spec["source"]["delivery_sha256"]
                    and inner["seed_components"] == [20261009, seed, 5, row["frame_index"], round(config["dt_ms"] * 1000), block_config["integration_seed_namespace"]],
                    "INNER_SOURCE_SEED_OR_BUDGET")
            require(inner["batch_size"] == importance["batch_size"]
                    and inner["mixture_probabilities"] == importance["mixture_probabilities"]
                    and np.array_equal(inner["proposal_probabilities"],
                                       np.asarray(importance["proposal_multipliers"]) * inner["nominal_probability"]),
                    "INNER_PROPOSAL_CHANGED")
            score = inner["stopped_prefix"]
            require(row["conditional_estimate"] == score["estimate"]
                    and row["inner_monte_carlo_variance"] == score["empirical_score_variance"] / inner["samples"]
                    and row["proposal_hits"] == inner["proposal_hits"]
                    and 0 <= inner["proposal_hits"] <= inner["samples"], "INNER_SCORE_ARITHMETIC")
            identity = tuple(original) + ("observed_count", "observed_window_ticks", "first_spike_tick", "refractory_ticks", "category")
            require(all(inner["sample_frame_row"][key] == row[key] for key in identity), "INNER_PREINTEGRATION_IDENTITY")
            # sample_frame_row contains pre-integration score placeholders. Only
            # stopped_prefix and the bound integrated row supply numerical scores.
            unreleased(inner)
            require(inner["third_spike_probability_established"] is False
                    and np.isfinite(inner["wall_seconds"]) and inner["wall_seconds"] >= 0, "INNER_SCOPE_OR_COST")
            counts["integrated_histories"] += 1
            counts["inner_trials"] += inner["samples"]
            counts["eligible_zero_estimates"] += int(row["conditional_estimate"] == 0)
            counts["zero_hit_histories"] += int(inner["proposal_hits"] == 0)
            counts["nonzero_conditional_estimates"] += int(row["conditional_estimate"] > 0)
            counts["inner_integration_seconds"] += inner["wall_seconds"]
        else:
            require(category in ("no_first_spike", "refractory_covers_remaining_window")
                    and row["conditional_estimate"] == row["inner_monte_carlo_variance"] == row["proposal_hits"] == 0,
                    "STRUCTURAL_ZERO_OR_MISSING_REFERENCE")
    require(actual["categories"] == dict(Counter(row["category"] for row in rows))
            and actual["integrated_histories"] == counts["integrated_histories"]
            and actual["zero_hit_eligible_histories"] == counts["eligible_zero_estimates"], "CLOCK_COUNTS_CHANGED")
    require(all(np.isfinite(actual[key]) and actual[key] >= 0 for key in ("wall_seconds", "reconstruction_seconds")), "CLOCK_COST_INVALID")
    counts.update(reconstruction_seconds=actual["reconstruction_seconds"], analysis_seconds=actual["wall_seconds"],
                  reconstruction_ticks=actual["reconstruction"]["ticks_reconstructed"],
                  source_events_consumed=actual["reconstruction"]["source_events_consumed"])
    if clock == "fine":
        require(actual["paired_physical_frame_equal"] is True
                and actual["paired_coarse"] == "../coarse/selected_frame.json", "PAIRED_COARSE_REFERENCE")
    return dict(statistics, **counts), rows, frame, actual


def read_block(root, seed, settings, preparation_entries):
    root = Path(root)
    entries, digest = receipt(root, "COMPLETE_PFC_STRATIFIED_HISTORY_COMPLETE",
                              settings["pilot_delivery_sha256"].get(str(seed)))
    for path in TESTED_FILES:
        require(path in entries and path in preparation_entries
                and all(entries[path][key] == preparation_entries[path][key] for key in ("sha256", "bytes")),
                "TESTED_IMPLEMENTATION_OR_CONFIG_CHANGED:" + path)
    wrapper = bound_json(root, entries, "report.json")
    require(wrapper["status"] == "COMPLETE_PFC_STRATIFIED_HISTORY_COMPLETE"
            and wrapper["exit_code"] == 0 and wrapper.get("error") is None, "FAILED_WRAPPER")
    unreleased(wrapper)
    canonical = yaml.safe_load(bound_payload(root, entries, PRODUCER_CONFIG))
    block = yaml.safe_load(bound_payload(root, entries, "stratified_block_config.yaml"))
    validate_settings(block)
    require(all(block[key] == value for key, value in canonical.items() if key != "clocks")
            and all(all(block["clocks"][clock][key] == value for key, value in spec.items())
                    for clock, spec in canonical["clocks"].items())
            and block["source_rng_seed"] == seed and block["case_id"] == f"pfc_{seed}_strong_cue"
            and block["sampling_seed_components"] == [20261009, seed, 5, block["integration_seed_namespace"]],
            "FROZEN_BLOCK_CONFIG")
    for clock in ("coarse", "fine"):
        registered = settings["source_registry"][str(seed)][clock]
        expected = {key: registered[key] for key in ("archive", "case_path", "delivery_sha256")}
        require(block["clocks"][clock]["source"] == expected, "FROZEN_CENSUS_SOURCE_CHANGED")
    census = bound_json(root, entries, "census/report.json")
    require(census["status"] == "COMPLETE_PFC_HISTORY_TIMING_CENSUS" and census["source_rng_seed"] == seed
            and census["case_id"] == block["case_id"] and census["frame_size"] == 200000, "COMPLETE_PHYSICAL_CENSUS")
    with np.load(BytesIO(bound_payload(root, entries, "census/frame_assignments.npz")), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    actual = bound_json(root, entries, "validation/report.json")
    require(actual["status"] == "COMPLETE_PFC_STRATIFIED_HISTORY_PAIR" and actual.get("error") is None
            and actual["source_rng_seed"] == seed and actual["case_id"] == block["case_id"], "COMPLETE_PAIR_REQUIRED")
    unreleased(actual)
    require(actual["population_confidence_interval_computed"] is False
            and actual["third_spike_probability_established"] is False, "PAIR_INFERENCE_SCOPE")
    clocks, rows, frames = {}, {}, {}
    for clock in ("coarse", "fine"):
        clocks[clock], rows[clock], frames[clock], clock_report = _read_clock(root, entries, seed, clock, block, census, arrays)
        cost = actual["per_clock"][clock]
        require(cost["sample_mean"] == clocks[clock]["sample_mean"]
                and cost["integrated_histories"] == clocks[clock]["integrated_histories"]
                and cost["inner_trials"] == clocks[clock]["inner_trials"]
                and cost["zero_hit_eligible_histories"] == clocks[clock]["eligible_zero_estimates"]
                and cost["reconstruction"] == clock_report["reconstruction"]
                and cost["reconstruction_seconds"] == clocks[clock]["reconstruction_seconds"]
                and cost["wall_seconds"] == clocks[clock]["analysis_seconds"], "PAIR_CLOCK_COST_ARITHMETIC")
    require(frames["coarse"] == frames["fine"] == actual["frame"], "PAIRED_FRAME_CHANGED")
    paired, _ = paired_components(rows["coarse"], rows["fine"], frames["coarse"])
    require(paired == actual["statistics"] and actual["selected_physical_windows"] == paired["sample_denominator"], "PAIRED_WEIGHTED_ARITHMETIC")
    require(all(np.isfinite(value) and value >= 0 for value in (actual["wall_seconds"], wrapper["wall_seconds"])), "PAIR_COST_INVALID")
    zero = frames["coarse"]["allocation"][0]
    return dict(source_rng_seed=seed, archive=str(root), delivery_sha256=digest, clocks=clocks, paired=paired,
                selected_physical_windows=paired["sample_denominator"], both_zero_population=zero["population_size"],
                both_zero_weight=zero["weight"], pair_analysis_seconds=actual["wall_seconds"], wrapper_seconds=wrapper["wall_seconds"])


def _upstream(project, settings):
    root = project / settings["preparation"]["archive"]
    entries, _ = receipt(root, "COMPLETE_PFC_STRATIFIED_HISTORY_PREPARE", settings["preparation"]["delivery_sha256"])
    prepared = bound_json(root, entries, "report.json")
    require(prepared["status"] == "COMPLETE_PFC_STRATIFIED_HISTORY_PREPARE" and prepared["exit_code"] == 0
            and prepared.get("error") is None, "ACCEPTED_PREPARATION_REQUIRED")
    canonical = yaml.safe_load(bound_payload(root, entries, PRODUCER_CONFIG))
    require(canonical["required_source_seeds"] == settings["required_source_seeds"]
            and canonical["accepted_uniform_comparison"] == settings["accepted_uniform_comparison"], "FROZEN_UPSTREAM_REFERENCES")
    namespace = canonical["integration_seed_namespace"]
    seed = canonical["required_source_seeds"][0]
    validate_settings(dict(canonical, source_rng_seed=seed, case_id=f"pfc_{seed}_strong_cue",
                           sampling_seed_components=[20261009, seed, 5, namespace]))
    profile = dict(sample_size=canonical["sample_size"], inner_samples=canonical["importance_sampling"]["samples"],
                   integration_seed_namespace=namespace,
                   nominal_inner_trial_budget_per_clock=canonical["sample_size"] * canonical["importance_sampling"]["samples"])
    spec = canonical["accepted_census"]
    origin = project / spec["archive"]
    census_entries, _ = receipt(origin, "COMPLETE_PFC_HISTORY_TIMING_CENSUS_COMPLETE", spec["delivery_sha256"])
    census_report = bound_json(origin, census_entries, "analysis/report.json")
    require(census_report["status"] == "COMPLETE_FULL_FRAME_CENSUS"
            and census_report["completed_original_blocks"] == census_report["required_original_blocks"] == 32
            and census_report["completed_physical_windows"] == census_report["required_physical_windows"] == 6400000
            and census_report["failures"] == [], "COMPLETE32_SOURCE_CENSUS_REQUIRED")
    registry = bound_json(origin, census_entries, "source_index.json")["source_blocks"]
    require(set(registry) == {str(seed) for seed in settings["required_source_seeds"]}, "COMPLETE_SOURCE_REGISTRY")
    return entries, dict(settings, source_registry=registry, sampling_profile=profile), spec


def _uniform_comparison(project, settings, paired, selected_windows):
    spec = settings["accepted_uniform_comparison"]
    root = project / spec["archive"]
    entries, _ = receipt(root, "COMPLETE_PFC_CLOCK_HISTORY_COMPARISON_COMPLETE", spec["delivery_sha256"])
    wrapper = bound_json(root, entries, "report.json")
    uniform = bound_json(root, entries, "analysis/report.json")
    require(wrapper["status"] == "COMPLETE_PFC_CLOCK_HISTORY_COMPARISON_COMPLETE" and wrapper["exit_code"] == 0
            and wrapper.get("error") is None and uniform["status"] == "COMPLETE_PFC_PAIRED_HISTORY_DIAGNOSTICS"
            and uniform["available_original_block_pairs"] == uniform["required_original_block_pairs"] == 32
            and uniform["missing_receipts"] == [] and uniform["paired_unit_windows"] == 2048
            and (uniform["field_index"], uniform["phase"], uniform["window_ms"]) == (5, "cue", 5.)
            and uniform["coarse_dt_ms"] == .05 and uniform["fine_dt_ms"] == .025, "COMPLETE_MATCHING_UNIFORM_COMPARISON")
    unreleased(uniform)
    reference = uniform["statistics"]
    ratios = {}
    for key in ("mean_within_archive_variance_diagnostic", "observed_between_block_variance", "empirical_mean_variance_diagnostic"):
        require(np.isfinite(reference[key]) and reference[key] >= 0, "UNIFORM_VARIANCE_INVALID")
        ratios[key] = None if reference[key] == 0 else paired[key] / reference[key]
    return dict(reference=spec, uniform_paired_statistics=reference,
                stratified_minus_uniform_event_mean=paired["mean"] - reference["mean"],
                stratified_over_uniform_variance_ratios=ratios,
                nominal_window_budget_per_block=64 if settings["sampling_profile"]["sample_size"] == 64 else None,
                uniform_nominal_window_budget_per_block=64,
                stratified_nominal_window_budget_per_block=settings["sampling_profile"]["sample_size"],
                uniform_inner_samples_per_eligible_clock_window=100000,
                stratified_inner_samples_per_eligible_clock_window=settings["sampling_profile"]["inner_samples"],
                uniform_nominal_inner_trial_budget_per_clock=6400000,
                stratified_nominal_inner_trial_budget_per_clock=settings["sampling_profile"]["nominal_inner_trial_budget_per_clock"],
                uniform_selected_physical_windows=2048, stratified_selected_physical_windows=selected_windows,
                matched_runtime_comparison=False, significance_test_computed=False,
                scope="Descriptive estimates on the same archived blocks with fresh sampling; nominal window budgets do not imply matched integration counts or runtime.")


def _write_blocks(path, blocks):
    with path.open("w", newline="") as stream:
        rows = []
        for block in blocks:
            row = {key: block[key] for key in ("source_rng_seed", "archive", "delivery_sha256", "selected_physical_windows",
                                               "both_zero_population", "both_zero_weight", "pair_analysis_seconds", "wrapper_seconds")}
            for clock in ("coarse", "fine"):
                row.update({clock + "_" + key: block["clocks"][clock][key] for key in (*STAT_FIELDS, *COST_FIELDS)})
            row.update({"paired_" + key: block["paired"][key] for key in STAT_FIELDS})
            row["fine_minus_coarse_cdf_le1"] = block["paired"]["fine_minus_coarse_cdf_le1"]
            rows.append(row)
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(project, settings, output, allow_partial=False):
    project, output = Path(project), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    blocks, missing, inspected = [], [], []
    report = dict(status="FAILED_PFC_STRATIFIED_HISTORY_SUMMARY", required_original_blocks=32,
                  statistics=None, uniform_comparison=None, missing_receipt_is_terminal_job_evidence=False,
                  population_confidence_interval_computed=False, significance_test_computed=False,
                  clock_equivalence_qualified=False, scientific_fidelity_qualified=False,
                  confirmation_execution_released=False, production_episodes=0, new_independent_source_blocks=0,
                  third_spike_probability_established=False, cdf_at_count2_contrast_estimated=False,
                  future_stratified_precision_established=False,
                  variance_scope="Within-frame sampling and inner Monte Carlo diagnostics; corrected between-block values describe latent archived means, without population confidence coverage.",
                  cost_scope="Clock analysis time includes reconstruction and inner integration. Cost levels are reported separately; equal nominal inner-trial budgets do not imply equal integration counts or runtime.")
    current_seed = None
    try:
        seeds = settings["required_source_seeds"]
        require(settings["schema"] == "pfc_stratified_history_summary_v1" and seeds == list(range(95001, 95033)), "ALL32_REGISTERED_BLOCKS_REQUIRED")
        preparation_entries, local, census_spec = _upstream(project, settings)
        report.update(preparation=settings["preparation"], source_census=census_spec, sampling_profile=local["sampling_profile"])
        for seed in seeds:
            current_seed = seed
            relative = settings["pilot_archives"].get(str(seed), settings["array_archive_prefix"] + str(seed - 95001))
            root = project / relative
            if not (root / "delivery.json").exists():
                missing.append(seed)
            else:
                block = read_block(root, seed, local, preparation_entries)
                block["archive"] = relative
                blocks.append(block)
            inspected.append(seed)
        current_seed = None
        if not missing:
            statistics = {clock: between_components([block["clocks"][clock] for block in blocks]) for clock in ("coarse", "fine")}
            statistics["paired"] = between_components([block["paired"] for block in blocks])
            statistics["paired"]["fine_minus_coarse_cdf_le1_mean"] = -statistics["paired"]["mean"]
            comparison = _uniform_comparison(project, local, statistics["paired"], sum(block["selected_physical_windows"] for block in blocks))
            report.update(status="COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY", statistics=statistics, uniform_comparison=comparison)
        else:
            report["status"] = "PARTIAL_PFC_STRATIFIED_HISTORY_SUMMARY"
    except Exception as exc:
        report.update(error=dict(type=type(exc).__name__, message=str(exc)), failed_source_seed=current_seed)
        raise
    finally:
        report.update(available_original_blocks=len(blocks), validated_source_seeds=[block["source_rng_seed"] for block in blocks],
                      missing_receipt_source_seeds=missing,
                      unexamined_source_seeds=[seed for seed in settings.get("required_source_seeds", []) if seed not in inspected and seed != current_seed],
                      selected_physical_windows=sum(block["selected_physical_windows"] for block in blocks),
                      totals={clock: {key: sum(block["clocks"][clock][key] for block in blocks) for key in COST_FIELDS} for clock in ("coarse", "fine")},
                      pair_analysis_seconds=sum(block["pair_analysis_seconds"] for block in blocks),
                      wrapper_seconds=sum(block["wrapper_seconds"] for block in blocks), block_diagnostics_file="block_diagnostics.csv")
        _write_blocks(output / "block_diagnostics.csv", blocks)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    require(report["status"] == "COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY" or allow_partial, "INCOMPLETE32_BLOCK_RECEIPTS")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "project-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = run(args.project_root, yaml.safe_load(args.config.read_text()), args.output, args.allow_partial)
    print(json.dumps({key: result[key] for key in ("status", "available_original_blocks", "selected_physical_windows")}))


if __name__ == "__main__":
    main()
