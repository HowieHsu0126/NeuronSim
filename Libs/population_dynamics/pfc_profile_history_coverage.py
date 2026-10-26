"""Receipt-bound coverage and descriptive aggregation of PFC profile records.

Reuse accepted scalar readbacks without reopening history samples. Aggregate
original blocks only when all 32 registered blocks have all 72 profile keys.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import yaml

from .pfc_first_passage_history_summary import between_components
from .pfc_history_profile_census import BOUNDARY, write_json
from .pfc_profile_history import profile_key
from .pfc_profile_history_readback import readback_requests


STAT_FIELDS = (
    "sample_mean",
    "within_archive_mean_variance_diagnostic",
    "inner_mc_mean_variance_diagnostic",
    "unit_window_sampling_mean_variance_diagnostic",
)
PAIRED_STAT_FIELDS = STAT_FIELDS + ("fine_minus_coarse_cdf_le1",)
_COMPONENTS = ("coarse", "fine", "paired")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("PFC_PROFILE_HISTORY_COVERAGE_" + message)


def _validate_statistics(statistics: Mapping[str, Any], component: str) -> None:
    fields = PAIRED_STAT_FIELDS if component == "paired" else STAT_FIELDS
    require(isinstance(statistics, Mapping) and all(field in statistics for field in fields),
            "STATISTIC_FIELDS:" + component)
    for field, value in statistics.items():
        require(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value),
                "FINITE_NUMERIC_STATISTIC:" + component + ":" + str(field))
    if component == "paired":
        require(statistics["fine_minus_coarse_cdf_le1"] == -statistics["sample_mean"], "PAIRED_CDF_LE1_SIGN")


def _scope(history_settings, requested_source_seeds):
    original_seeds = history_settings["required_source_seeds"]
    require(original_seeds == list(range(95001, 95033)), "ORIGINAL_SOURCE_DOMAIN")
    profiles, repeats = history_settings["profiles"], history_settings["auxiliary_repeats"]
    require(len(profiles) == 36 and repeats == ["A", "B"], "REGISTERED_PROFILES_AND_REPEATS")
    keys = [profile_key(profile, repeat) for repeat in repeats for profile in profiles]
    require(len(keys) == len(set(keys)) == 72, "UNIQUE_REGISTERED_PROFILE_KEYS")
    require(isinstance(requested_source_seeds, (list, tuple)) and len(requested_source_seeds) > 0
            and all(type(seed) is int and seed in original_seeds for seed in requested_source_seeds)
            and len(set(requested_source_seeds)) == len(requested_source_seeds), "REQUESTED_SOURCE_DOMAIN")
    requested = list(requested_source_seeds)
    return original_seeds, keys, requested


def aggregate(history_settings: Mapping[str, Any], requested_source_seeds: Sequence[int],
              records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Require all 72 keys per requested source; summarize only all 32 sources.

    Statistics contain finite numeric scalars. Signed variance diagnostics are
    retained. Record order, additional record fields, and provenance are kept.
    A/B keys remain separate groups, each with 32 original blocks at full scope.
    """
    original_seeds, keys, requested = _scope(history_settings, requested_source_seeds)
    requested_set, expected_keys = set(requested), set(keys)
    retained = list(records)
    indexed, seen = {}, {seed: set() for seed in requested}
    for record in retained:
        require(isinstance(record, Mapping)
                and all(field in record for field in ("source_rng_seed", "key", "provenance", *_COMPONENTS)),
                "RECORD_FIELDS")
        seed, key = record["source_rng_seed"], record["key"]
        require(type(seed) is int and seed in requested_set, "RECORD_SOURCE_DOMAIN")
        require(isinstance(key, str) and key in expected_keys, "UNKNOWN_PROFILE_KEY")
        require(bool(record["provenance"]), "PROVENANCE_REQUIRED")
        identity = (seed, key)
        require(identity not in indexed, "DUPLICATE_PAIR:" + str(seed) + ":" + key)
        for component in _COMPONENTS:
            _validate_statistics(record[component], component)
        indexed[identity] = record
        seen[seed].add(key)
    for seed in requested:
        require(seen[seed] == expected_keys,
                "MISSING_PAIR_KEYS:" + str(seed) + ":" + str(len(expected_keys) - len(seen[seed])))
    available = [seed for seed in original_seeds if seed in requested_set and seen[seed]]
    full_scope = requested_set == set(original_seeds)
    statistics = None
    if full_scope:
        statistics = {
            key: {component: between_components([indexed[(seed, key)][component] for seed in original_seeds])
                  for component in _COMPONENTS}
            for key in keys
        }
    return dict(
        status="COMPLETE_REQUESTED_PFC_PROFILE_HISTORY_COVERAGE",
        required_original_blocks=32, requested_original_blocks=len(requested),
        available_original_blocks=len(available), requested_source_seeds=requested,
        available_source_seeds=available, expected_keys_per_original_block=72,
        expected_pairs=len(requested) * 72, completed_pairs=len(retained), required_registered_pairs=2304,
        full_registered_scope_covered=full_scope, profile_keys=keys, statistics=statistics, records=retained,
        negative_variance_diagnostics_preserved=True, new_full_network_episodes=0,
        source_history_population_uncertainty_established=False,
        third_spike_probability_established=False, third_spike_probability_or_bound_qualified=False,
        scope="Requested record coverage and descriptive original-block aggregation; no source-history uncertainty or scientific qualification",
        **BOUNDARY,
    )


_READBACK_CONFIG = "Libs/configs/neural_population_pfc_profile_history_readback.yaml"
_LEGACY_CONFIG = "Libs/configs/neural_population_pfc_stratified_history_summary.yaml"
_READBACK_STATUS = "COMPLETE_PFC_PROFILE_HISTORY_READBACK_COMPLETE"
_PREPARATION_STATUS = "COMPLETE_PFC_PROFILE_HISTORY_READBACK_PREPARE"


def _entry(delivery, name):
    entries = [row for row in delivery["files"] if row["path"] == name]
    require(len(entries) == 1, "UNIQUE_MANIFEST_ENTRY:" + name)
    return entries[0]


def _bind_file(bundle, name, work, copy):
    entry = _entry(bundle["delivery"], name)
    target = bundle["target"] / name
    copy(bundle["origin"] / name, target, entry["sha256"])
    local = work / target
    require(local.stat().st_size == entry["bytes"], "BOUND_FILE_SIZE:" + name)
    return local


def _bind_archive(reference, repo, work, copy, label, status):
    origin, target = repo / reference["archive"], Path("coverage_inputs") / label
    copy(origin / "delivery.json", target / "delivery.json", reference.get("delivery_sha256"))
    payload = (work / target / "delivery.json").read_bytes()
    delivery = json.loads(payload)
    require(delivery["verified"] is True and delivery["status"] == status,
            "COMPLETE_DELIVERY:" + reference["archive"])
    return dict(origin=origin, target=target, delivery=delivery, archive=reference["archive"],
                delivery_sha256=hashlib.sha256(payload).hexdigest())


def _scope_gates(report, false_keys, zero_keys):
    require(all(report.get(key) is False for key in false_keys), "UNRELEASED_SCOPE")
    require(all(type(report.get(key)) is int and report[key] == 0 for key in zero_keys),
            "NO_NEW_SOURCE_EPISODES")


def _reference(bundle):
    return dict(archive=bundle["archive"], delivery_sha256=bundle["delivery_sha256"],
                staged_inputs=str(bundle["target"]))


def _legacy_records(history, requested, repo, work, copy):
    bundle = _bind_archive(history["reused_integration_summary"], repo, work, copy, "legacy",
                           "COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY_COMPLETE")
    summary = json.loads(_bind_file(bundle, "analysis/report.json", work, copy).read_text())
    _bind_file(bundle, _LEGACY_CONFIG, work, copy)
    table = _bind_file(bundle, "analysis/block_diagnostics.csv", work, copy)
    require(summary["status"] == "COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY"
            and summary.get("error") is None
            and summary["required_original_blocks"] == summary["available_original_blocks"] == 32
            and summary["validated_source_seeds"] == history["required_source_seeds"]
            and summary["missing_receipt_source_seeds"] == summary["unexamined_source_seeds"] == []
            and summary["selected_physical_windows"] == 32 * 640
            and summary["block_diagnostics_file"] == "block_diagnostics.csv", "LEGACY_ALL32_SUMMARY")
    design = summary["sampling_profile"]
    require(design["sample_size"] == 640 and design["inner_samples"] == 10000
            and design["integration_seed_namespace"] == 4, "LEGACY_SAMPLING_DESIGN")
    _scope_gates(summary, ("population_confidence_interval_computed", "significance_test_computed",
                          "clock_equivalence_qualified", "scientific_fidelity_qualified",
                          "confirmation_execution_released", "third_spike_probability_established",
                          "cdf_at_count2_contrast_estimated", "future_stratified_precision_established"),
                 ("production_episodes", "new_independent_source_blocks"))
    with table.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    seeds = [int(row["source_rng_seed"]) for row in rows]
    require(len(seeds) == len(set(seeds)) == 32
            and set(seeds) == set(history["required_source_seeds"]), "LEGACY_UNIQUE32_SOURCE_BLOCKS")
    records = []
    statistical_columns = {f"{component}_{field}" for component in _COMPONENTS for field in STAT_FIELDS}
    statistical_columns.add("fine_minus_coarse_cdf_le1")
    for seed, row in zip(seeds, rows):
        require(int(row["selected_physical_windows"]) == 640, "LEGACY_PHYSICAL_WINDOWS")
        for clock in ("coarse", "fine"):
            require(int(row[clock + "_inner_trials"]) == int(row[clock + "_integrated_histories"]) * 10000,
                    "LEGACY_INNER_TRIAL_BUDGET")
        components = {component: {field: float(row[f"{component}_{field}"]) for field in STAT_FIELDS}
                      for component in _COMPONENTS}
        components["paired"]["fine_minus_coarse_cdf_le1"] = float(row["fine_minus_coarse_cdf_le1"])
        for component, statistics in components.items():
            _validate_statistics(statistics, component)
        require(bool(row["archive"]) and bool(row["delivery_sha256"]), "LEGACY_SOURCE_PROVENANCE")
        if seed in requested:
            records.append(dict(source_rng_seed=seed, key=history["reused_profile_keys"][0], **components,
                                provenance=dict(kind="legacy_summary", **_reference(bundle),
                                                integration_seed_namespace=4,
                                                row_metadata={key: value for key, value in row.items()
                                                              if key not in statistical_columns})))
    costs = dict(kind="legacy_summary", **_reference(bundle),
                 block_diagnostics=str(table.relative_to(work)),
                 summary=str(bundle["target"] / "analysis/report.json"),
                 cost_scope=summary.get("cost_scope"),
                 pair_analysis_seconds=summary.get("pair_analysis_seconds"),
                 wrapper_seconds=summary.get("wrapper_seconds"))
    return records, costs


def _completed_readback(reference, repo, work, copy, label):
    bundle = _bind_archive(reference, repo, work, copy, label, _READBACK_STATUS)
    wrapper = json.loads(_bind_file(bundle, "report.json", work, copy).read_text())
    analysis = json.loads(_bind_file(bundle, "analysis/report.json", work, copy).read_text())
    source_config = yaml.safe_load(_bind_file(bundle, "source_config.yaml", work, copy).read_text())
    require(wrapper["status"] == _READBACK_STATUS and wrapper["exit_code"] == 0
            and wrapper.get("error") is None and wrapper["task"] == "profile_readback"
            and wrapper["mode"] == "complete"
            and wrapper["job_id"] == bundle["delivery"]["job_id"], "READBACK_WRAPPER")
    _scope_gates(wrapper, ("scientific_fidelity_qualified", "confirmation_execution_released"),
                 ("production_episodes", "new_independent_source_blocks"))
    require(analysis["status"] == "COMPLETE_PFC_SHARED_PROFILE_HISTORY_READBACK"
            and analysis.get("error") is None and analysis["original_source_blocks"] == 1,
            "READBACK_ANALYSIS")
    _scope_gates(analysis, ("third_spike_probability_or_bound_qualified",
                           "source_history_population_uncertainty_established",
                           "population_confidence_interval_computed", "scientific_fidelity_qualified",
                           "confirmation_execution_released"),
                 ("new_neural_trajectories", "new_independent_source_blocks", "production_episodes"))
    return bundle, analysis, source_config


def _normalize_readback(bundle, analysis, seed, keys, labels, kind):
    require(analysis["source_rng_seed"] == seed and analysis["paired_profiles"] == len(keys)
            and [pair["key"] for pair in analysis["pairs"]] == keys, "READBACK_PAIR_IDENTITIES")
    records = []
    for pair in analysis["pairs"]:
        require(pair["paired_selected_physical_windows"] == 640, "PAIRED_PHYSICAL_WINDOWS")
        inputs = dict(paired=pair["statistics"], **pair["per_clock"])
        components, metadata = {}, {}
        for component in _COMPONENTS:
            values = inputs[component]
            fields = PAIRED_STAT_FIELDS if component == "paired" else STAT_FIELDS
            components[component] = {field: values[field] for field in fields}
            _validate_statistics(components[component], component)
            require(values["sample_denominator"] == values["nominal_sample_budget"] == 640
                    and values["population_confidence_interval_computed"] is False
                    and values["importance_estimates_clipped"] is False, "READBACK_SAMPLING_SCOPE")
            metadata[component] = {key: value for key, value in values.items()
                                   if key not in fields and (not isinstance(value, (dict, list))
                                                             or key == "source_binding")}
            if component != "paired":
                require(values["source_label"] == labels[component]
                        and values["source_binding"] == analysis["source_costs"][component]["source_binding"]
                        and values["inner_trials"] == values["integrated_histories"] * 10000,
                        "READBACK_CLOCK_SOURCE_OR_BUDGET")
        _scope_gates(inputs["paired"], ("third_spike_probability_established", "scientific_fidelity_qualified",
                                         "confirmation_execution_released"), ("production_episodes",))
        records.append(dict(source_rng_seed=seed, key=pair["key"], **components,
                            provenance=dict(kind=kind, **_reference(bundle), integration_seed_namespace=5,
                                            paired_selected_physical_windows=640, component_metadata=metadata)))
    return records


def _pilot_records(history, requested, repo, work, copy):
    accepted, pilot = history["accepted_pilot"], history["pilot"]
    require(pilot["source_rng_seed"] == 95001 and pilot["source_labels"] == ["half_A", "quarter_A"]
            and len(pilot["profile_keys"]) == len(set(pilot["profile_keys"])) == 2, "PILOT_SCOPE")
    bundle, analysis, old_history = _completed_readback(accepted["readback"], repo, work, copy, "pilot")
    require(old_history == {key: value for key, value in history.items() if key != "accepted_pilot"},
            "PILOT_SOURCE_CONFIG")
    records = _normalize_readback(bundle, analysis, 95001, pilot["profile_keys"],
                                  dict(coarse="half_A", fine="quarter_A"), "accepted_pilot")
    costs = dict(kind="accepted_pilot", **_reference(bundle), source_costs=analysis["source_costs"],
                 source_execution_references=accepted["source_runs"])
    return (records if 95001 in requested else []), costs


def _group_records(descriptor, index, history, source_sha, repo, work, copy, label):
    reference = descriptor.get("task_archives", {}).get(index)
    archive = (reference["archive"] if reference else descriptor["archive_prefix"] + "_" + str(index))
    bundle, analysis, source_config = _completed_readback(
        reference or dict(archive=archive), repo, work, copy, label)
    if reference:
        wrapper = json.loads((work / bundle["target"] / "report.json").read_text())
        require(bundle["delivery"]["job_id"] == reference["job_id"]
                and wrapper.get("slurm_step_id") == reference["slurm_step_id"]
                and isinstance(reference["slurm_step_id"], str)
                and bool(reference["slurm_step_id"]), "READBACK_STEP_JOB_IDENTITY")
    else:
        array_name = Path(descriptor["archive_prefix"]).name
        require(array_name.startswith("run_")
                and bundle["delivery"]["job_id"] == array_name[4:] + "_" + str(index), "READBACK_ARRAY_JOB_IDENTITY")
    require(_entry(bundle["delivery"], "source_config.yaml")["sha256"] == source_sha
            and source_config == history, "READBACK_SOURCE_CONFIG")
    preparation = json.loads(_bind_file(bundle, "preparation_delivery.json", work, copy).read_text())
    require(_entry(bundle["delivery"], "preparation_delivery.json")["sha256"]
            == descriptor["preparation_delivery_sha256"]
            and preparation["verified"] is True and preparation["status"] == _PREPARATION_STATUS,
            "READBACK_PREPARATION_BINDING")
    config = yaml.safe_load(_bind_file(bundle, _READBACK_CONFIG, work, copy).read_text())
    require(_entry(preparation, _READBACK_CONFIG)["sha256"]
            == _entry(bundle["delivery"], _READBACK_CONFIG)["sha256"]
            and _entry(preparation, "source_config.yaml")["sha256"] == source_sha
            and config["schema"] == "pfc_profile_history_readback_v1"
            and config["source_config_sha256"] == source_sha, "READBACK_CONFIG_PREPARATION")
    requests = readback_requests(config, history)
    require(0 <= index < len(requests), "READBACK_TASK_INDEX")
    request = requests[index]
    require(request["full_source_requests"] is True and analysis["full_source_requests"] is True
            and analysis["readback_group"] == request["group_name"]
            and analysis["readback_group_index"] == request["group_index"] == index, "READBACK_GROUP_IDENTITY")
    labels = {clock: spec["source_label"] for clock, spec in request["source_runs"].items()}
    records = _normalize_readback(bundle, analysis, request["source_rng_seed"], request["profile_keys"],
                                  labels, "profile_readback")
    costs = analysis["source_costs"]
    require(set(costs) == {"coarse", "fine"}, "READBACK_SOURCE_COST_CLOCKS")
    for clock, spec in request["source_runs"].items():
        require(costs[clock]["history_execution_archive"] == spec["archive"], "SOURCE_COST_EXECUTION_BINDING")
    return records, costs


def _write_records(records, output):
    fields = ["source_rng_seed", "key", "provenance_kind", "archive", "delivery_sha256", "integration_seed_namespace"]
    fields += [f"{component}_{field}" for component in _COMPONENTS
               for field in (PAIRED_STAT_FIELDS if component == "paired" else STAT_FIELDS)]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            provenance = record["provenance"]
            row = dict(source_rng_seed=record["source_rng_seed"], key=record["key"], provenance_kind=provenance["kind"],
                       **{field: provenance[field] for field in ("archive", "delivery_sha256", "integration_seed_namespace")})
            row.update({f"{component}_{field}": value for component in _COMPONENTS
                        for field, value in record[component].items()})
            writer.writerow(row)


def run(settings, repo, work, copy, mode, foundation_delivery):
    """Bind small accepted outputs; never replay sampling or open history tars."""
    repo, work = Path(repo), Path(work)
    output = work / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="FAILED_PFC_PROFILE_HISTORY_COVERAGE", statistics=None,
                  source_history_population_uncertainty_established=False,
                  third_spike_probability_established=False, third_spike_probability_or_bound_qualified=False,
                  new_full_network_episodes=0, **BOUNDARY)
    try:
        require(settings["schema"] == "pfc_profile_history_coverage_v1" and mode in ("prepare", "complete"),
                "CONFIG_SCHEMA_OR_MODE")
        require(foundation_delivery["verified"] is True and foundation_delivery["status"] == _PREPARATION_STATUS,
                "FOUNDATION_DELIVERY")
        foundation = dict(origin=repo / settings["source_preparation"]["archive"],
                          target=Path("coverage_inputs/foundation"), delivery=foundation_delivery)
        require(_entry(foundation_delivery, "source_config.yaml")["sha256"] == settings["source_config_sha256"],
                "FOUNDATION_SOURCE_CONFIG")
        history = yaml.safe_load(_bind_file(foundation, "source_config.yaml", work, copy).read_text())
        original, keys, requested = _scope(history, settings["requested_source_seeds"])
        require(history["schema"] == "pfc_profile_history_v1" and history["sample_size"] == 640
                and history["importance_sampling"]["samples"] == 10000 and history["integration_seed_namespace"] == 5
                and history["reused_profile_keys"] == ["A/native_half/field_5_cue"], "FROZEN_HISTORY_DESIGN")
        arrays = settings["readback_arrays"]
        require(isinstance(arrays, list) and len(arrays) > 0, "READBACK_ARRAYS")
        groups = []
        for descriptor in arrays:
            indices = descriptor["task_indices"]
            require(isinstance(descriptor["archive_prefix"], str) and bool(descriptor["archive_prefix"])
                    and isinstance(descriptor["preparation_delivery_sha256"], str)
                    and isinstance(indices, list) and len(indices) > 0
                    and all(type(index) is int and index >= 0 for index in indices), "READBACK_ARRAY_DESCRIPTOR")
            overrides = descriptor.get("task_archives", {})
            require(isinstance(overrides, Mapping) and set(overrides) <= set(indices)
                    and all(isinstance(value, Mapping)
                            and set(value) == {"archive", "delivery_sha256", "job_id", "slurm_step_id"}
                            and all(isinstance(item, str) and bool(item) for item in value.values())
                            for value in overrides.values()), "READBACK_TASK_ARCHIVES")
            groups.extend((descriptor["archive_prefix"], index) for index in indices)
        require(len(groups) == len(set(groups)), "DUPLICATE_READBACK_GROUP")
        legacy, legacy_costs = _legacy_records(history, requested, repo, work, copy)
        pilot, pilot_costs = _pilot_records(history, requested, repo, work, copy)
        records = legacy + pilot
        reused = {(record["source_rng_seed"], record["key"]) for record in records}
        require(len(reused) == len(records)
                and all(seed in requested and key in keys for seed, key in reused), "REUSED_RECORD_IDENTITIES")
        awaited = [dict(source_rng_seed=seed, key=key) for seed in requested for key in keys if (seed, key) not in reused]
        report.update(required_original_blocks=len(original), requested_original_blocks=len(requested),
                      requested_source_seeds=requested, expected_keys_per_original_block=72,
                      expected_pairs=72 * len(requested), required_registered_pairs=2304,
                      reused_legacy_pairs=len(legacy), reused_pilot_pairs=len(pilot), reused_pairs=len(records),
                      awaited_pairs=len(awaited), awaited_pair_keys=awaited, required_readback_groups=len(groups),
                      reused_cost_references=[legacy_costs, pilot_costs],
                      new_source_costs=[], new_source_execution_count=0)
        if mode == "prepare":
            report.update(status="PREPARED_PFC_PROFILE_HISTORY_COVERAGE", completed_pairs=len(records),
                          full_registered_scope_covered=False, records=records, source_execution_verified=False)
        else:
            costs = {}
            for array_index, descriptor in enumerate(arrays):
                for index in descriptor["task_indices"]:
                    new_records, clock_costs = _group_records(descriptor, index, history, settings["source_config_sha256"],
                                                             repo, work, copy, f"groups/array_{array_index}/task_{index}")
                    records.extend(new_records)
                    for cost in clock_costs.values():
                        archive = cost["history_execution_archive"]
                        require(archive not in costs or costs[archive] == cost, "SOURCE_COST_MISMATCH:" + archive)
                        costs[archive] = cost
            report.update(aggregate(history, requested, records), awaited_pairs=0, awaited_pair_keys=[],
                          new_source_costs=list(costs.values()), new_source_execution_count=len(costs))
            _write_records(records, output / "profile_records.csv")
            report["record_file"] = "profile_records.csv"
    except Exception as exc:
        report["status"] = "FAILED_PFC_PROFILE_HISTORY_COVERAGE"
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        write_json(output / "report.json", report)
    return report
