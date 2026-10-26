"""Numerical readback of caller-authenticated paired profile history artifacts.

Importance members are read from their tar archive without extraction. This
checks archived-frame calculations; it does not qualify source-law inference.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack
import json
from pathlib import Path
import tarfile
import time
from typing import Any

import numpy as np

from . import pfc_first_passage_importance as importance
from .pfc_history_profile_census import BOUNDARY, CLOCKS, SOURCE_CLOCKS, unpack_pair, write_json
from .pfc_profile_history import pair_name, seed_identity, source_requests
from .pfc_stratified_history import paired_components, sampling_callbacks


_TARGET = "second_focal_spike_before_window_end_given_first_spike_past_and_other_unit_future_innovations"
_PHYSICAL_KEYS = ("sample_index", "frame_index", "unit_index", "window_start_ms",
                  "window_stop_ms", "joint_time_bin", "inclusion_probability")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("PFC_PROFILE_HISTORY_READBACK_" + message)


def readback_requests(settings: Mapping[str, Any], history_settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map readback groups to complete source-case archives without changing scope.

    Legacy settings describe one request. Array settings describe inclusive
    global source-index ranges; group_index follows target-seed order, then
    A/native_half, B/native_half, A/half_quarter, B/half_quarter.
    """
    if "source_arrays" not in settings:
        return [dict(settings, full_source_requests=False)]
    original_seeds = history_settings["required_source_seeds"]
    require(original_seeds == list(range(95001, 95033)), "ORIGINAL_SOURCE_SEEDS")
    seeds = settings["source_rng_seeds"]
    require(isinstance(seeds, list) and len(seeds) > 0
            and all(type(seed) is int and seed in original_seeds for seed in seeds)
            and len(set(seeds)) == len(seeds), "TARGET_SOURCE_SEEDS")
    arrays = settings["source_arrays"]
    require(isinstance(arrays, list) and len(arrays) > 0
            and all(isinstance(array, Mapping) for array in arrays), "SOURCE_ARRAYS")
    labels = list(SOURCE_CLOCKS)
    source_count = len(original_seeds) * len(labels)
    ranges = []
    for array in arrays:
        start, stop = array.get("source_index_start"), array.get("source_index_stop")
        prefix = array.get("archive_prefix")
        require(type(start) is int and type(stop) is int and 0 <= start <= stop < source_count,
                "SOURCE_INDEX_RANGE")
        require(isinstance(prefix, str) and len(prefix) > 0, "SOURCE_ARCHIVE_PREFIX")
        ranges.append((start, stop, prefix))
    ordered = sorted(ranges)
    require(all(left[1] < right[0] for left, right in zip(ordered, ordered[1:])), "SOURCE_INDEX_RANGE_OVERLAP")
    groups = (("A/native_half", "native", "half_A"), ("B/native_half", "native", "half_B"),
              ("A/half_quarter", "half_A", "quarter_A"), ("B/half_quarter", "half_B", "quarter_B"))
    requests = []
    for seed in seeds:
        resolved = {}
        for label_index, label in enumerate(labels):
            source_index = original_seeds.index(seed) * len(labels) + label_index
            matches = [entry for entry in ranges if entry[0] <= source_index <= entry[1]]
            require(len(matches) == 1, "SOURCE_INDEX_COVERAGE:" + str(source_index))
            keys = [request["key"] for request in source_requests(history_settings, label, pilot=False, seed=seed)]
            resolved[label] = dict(source_label=label, archive=matches[0][2]+"_"+str(source_index),
                                   expected_profile_keys=keys)
        for name, coarse_label, fine_label in groups:
            coarse, fine = resolved[coarse_label], resolved[fine_label]
            fine_keys = set(fine["expected_profile_keys"])
            keys = [key for key in coarse["expected_profile_keys"] if key in fine_keys]
            requests.append(dict(settings, source_rng_seed=seed, profile_keys=keys,
                                 source_runs=dict(coarse=dict(coarse), fine=dict(fine)),
                                 full_source_requests=True, group_index=len(requests), group_name=name))
    return requests


def require_false(report: Mapping[str, Any], *keys: str) -> None:
    require(all(report[key] is False for key in keys), "UNRELEASED_SCOPE")


def tar_json(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    member = archive.getmember(name)
    require(member.isfile(), "REGULAR_TAR_MEMBER_REQUIRED:" + name)
    with archive.extractfile(member) as stream:
        return json.load(stream)


def reconstruct_scores(inner: Mapping[str, Any], settings: Mapping[str, Any], dt_ms: float):
    """Reconstruct both stored score summaries with the production function."""
    options = settings["importance_sampling"]
    samples = inner["samples"]
    require(type(samples) is int and samples == options["samples"] == 10000
            and inner["batch_size"] == options["batch_size"], "INNER_BUDGET")
    nominal = importance.native._RATE_HZ * (dt_ms * 1e-3)
    proposals = nominal * np.asarray(options["proposal_multipliers"], dtype=np.float64)
    mixture = np.asarray(options["mixture_probabilities"], dtype=np.float64)
    importance.log_likelihood_ratio(np.array([0]), np.array([0]), nominal, proposals, mixture)
    bound = 1 / float(mixture[proposals == nominal].sum())
    require(inner["nominal_probability"] == nominal
            and np.array_equal(inner["proposal_probabilities"], proposals)
            and np.array_equal(inner["mixture_probabilities"], mixture)
            and inner["weight_bound"] == bound, "INNER_PROPOSAL_OR_BOUND")
    hits = inner["proposal_hits"]
    require(type(hits) is int and 0 <= hits <= samples
            and inner["proposal_no_hits"] == samples - hits, "INNER_HIT_COUNTS")
    reconstructed = {}
    for name in ("stopped_prefix", "full_path"):
        stored = inner[name]
        total, squares = stored["score_sum"], stored["score_square_sum"]
        require(np.isfinite(total) and np.isfinite(squares) and total >= 0 and squares >= 0,
                "FINITE_SCORE_MOMENTS:" + name)
        if hits == 0:
            require(total == squares == 0, "ZERO_HIT_MOMENTS:" + name)
        calculated = importance.summarize_score(total, squares, samples, bound, options["confidence_level"])
        require(stored == calculated, "SCORE_MOMENT_RECONSTRUCTION:" + name)
        reconstructed[name] = calculated
    return reconstructed


def _read_clock(settings, census_report, census_arrays, directory, binding, *, clock, label,
                profile, repeat, seed, sampling_seeds, context):
    actual = json.loads((directory / "report.json").read_text())
    selected = json.loads((directory / "selected_frame.json").read_text())
    classified = json.loads((directory / "classified_frame.json").read_text())
    rows = json.loads((directory / "integrated_frame.json").read_text())
    meta, _ = unpack_pair(census_report, census_arrays, clock)
    dt = CLOCKS[profile["left_cohort" if clock == "coarse" else "right_cohort"]]
    native = label == "native"
    require(actual["status"] == "COMPLETE_PFC_PROFILE_CONDITIONAL_HISTORY" and actual.get("error") is None
            and actual["profile"] == profile and actual["auxiliary_repeat"] == repeat
            and actual["source_rng_seed"] == seed and actual["clock"] == clock
            and actual["source_label"] == label, "CLOCK_PROFILE_SOURCE_IDENTITY")
    require(actual["dt_ms"] == meta["dt_ms"] == dt
            and actual["source_parent_dt_ms"] == (None if native else .05)
            and actual["input_stream"] == ("native_numpy_stream" if native else "coupled_binomial_clock_refinement_stream")
            and meta["excitatory_neurons"] + meta["inhibitory_neurons"] == settings["expected_units"]
            and meta["duration_ms"] == 5000., "SOURCE_CLOCK_OR_ORIGINAL_PARENT")
    require(actual["integration_seed_namespace"] == settings["integration_seed_namespace"] == 5
            and actual["integration_seed_context"] == context, "CLOCK_SEED_CONTEXT")
    require_false(actual, "population_confidence_interval_computed", "scientific_fidelity_qualified",
                  "confirmation_execution_released", "third_spike_probability_established",
                  "source_history_population_uncertainty_established")
    require(all(actual[key] == 0 for key in ("new_neural_trajectories", "new_independent_source_blocks", "production_episodes")),
            "NO_NEW_SOURCE_EPISODES")
    sampler, summarize = sampling_callbacks(census_report, census_arrays, clock)
    expected_rows, frame = sampler(meta, field=profile["field_index"], phase=profile["phase"],
                                  window_ms=profile["window_ms"], sample_size=settings["sample_size"],
                                  seed_components=sampling_seeds)
    require(selected == dict(frame=frame, rows=expected_rows) and actual["frame"] == frame,
            "SELECTED_FRAME_CHANGED")
    require(len(rows) == len(classified) == len(expected_rows), "COMPLETE_SELECTED_DENOMINATOR")
    eligible = sum(row["category"] == "conditional_integration_required" for row in classified)
    counts = dict(integrated_histories=0, inner_trials=0, structural_zero_histories=0,
                  eligible_zero_estimates=0, moment_reports_recomputed=0)
    with ExitStack() as stack:
        archive = stack.enter_context(tarfile.open(directory / "integration_details.tar", "r:*")) if eligible else None
        for original, before, row in zip(expected_rows, classified, rows):
            require(all(before[key] == value for key, value in original.items())
                    and all(row[key] == value for key, value in before.items()), "ROW_CLASSIFICATION_IDENTITY")
            category = row["category"]
            require(category != "suppressed_network_reference_required", "UNQUALIFIED_SUPPRESSED_REFERENCE")
            if category == "conditional_integration_required":
                require(row["observed_count"] == 1 and row["observed_window_ticks"] == [row["first_spike_tick"]],
                        "OBSERVED_SINGLE_SPIKE_IDENTITY")
                relative = f"sample_{row['sample_index']:03d}"
                name = relative + "/integration/report.json"
                require(row["integration_report"] == name, "INTEGRATION_REPORT_PATH")
                inner = tar_json(archive, name)
                require(inner["status"] == "COMPLETE_CONDITIONAL_FIRST_PASSAGE_IMPORTANCE" and inner.get("error") is None
                        and inner["reference_archive"] == binding["archive"]
                        and inner["reference_delivery_sha256"] == binding["delivery_sha256"], "INNER_SOURCE_IDENTITY")
                expected_seed = [20261009, seed, profile["field_index"], row["frame_index"], round(dt * 1000), 5, *context]
                require(inner["seed_components"] == expected_seed, "INNER_SEED_IDENTITY")
                require(inner["reference_archive_role"] == "accepted_source_provenance_for_extracted_reference"
                        and inner["extracted_reference"] == "../reference"
                        and inner["target"] == _TARGET, "INNER_REFERENCE_ROLE")
                # Numeric fields in this snapshot were written before integration.
                require(all(inner["sample_frame_row"][key] == value for key, value in before.items()),
                        "INNER_PREINTEGRATION_IDENTITY")
                conditional = tar_json(archive, relative + "/reference/conditional_initial.json")
                require(all(conditional[key] == row[key] for key in ("unit_index", "first_spike_tick", "window_stop_tick"))
                        and conditional["population"] == "E" and conditional["fresh_input_included"] is False,
                        "CONDITIONAL_REFERENCE_IDENTITY")
                require(archive.getmember(relative + "/reference/reference_window.npz").isfile(), "REFERENCE_WINDOW_MEMBER")
                require_false(inner, "scientific_fidelity_qualified", "confirmation_execution_released",
                              "population_history_uncertainty_established", "population_first_passage_probability_established",
                              "third_spike_probability_established", "finite_PRNG_law_certified")
                require(all(inner[key] == 0 for key in ("new_full_network_episodes", "new_independent_source_blocks", "production_episodes")),
                        "INNER_SOURCE_SCOPE")
                score = reconstruct_scores(inner, settings, dt)["stopped_prefix"]
                require(row["conditional_estimate"] == score["estimate"]
                        and row["inner_monte_carlo_variance"] == score["empirical_score_variance"] / inner["samples"]
                        and row["proposal_hits"] == inner["proposal_hits"], "INTEGRATED_SCORE_ARITHMETIC")
                counts["integrated_histories"] += 1
                counts["inner_trials"] += inner["samples"]
                counts["eligible_zero_estimates"] += int(score["estimate"] == 0)
                counts["moment_reports_recomputed"] += 2
            else:
                require(category in ("no_first_spike", "refractory_covers_remaining_window")
                        and row["conditional_estimate"] == row["inner_monte_carlo_variance"] == row["proposal_hits"] == 0,
                        "STRUCTURAL_ZERO_SCORE")
                require((category == "no_first_spike" and row["observed_count"] == 0
                         and row["first_spike_tick"] is None and row["observed_window_ticks"] == [])
                        or (category == "refractory_covers_remaining_window" and row["observed_count"] == 1
                            and row["observed_window_ticks"] == [row["first_spike_tick"]]
                            and row["first_spike_tick"] + row["refractory_ticks"] >= row["window_stop_tick"]),
                        "STRUCTURAL_ZERO_REASON")
                counts["structural_zero_histories"] += 1
    statistics = summarize(rows, frame)
    require(all(actual[key] == value for key, value in statistics.items()), "WEIGHTED_CLOCK_ARITHMETIC")
    require(actual["integrated_histories"] == counts["integrated_histories"]
            and actual["inner_trials"] == counts["inner_trials"]
            and actual["categories"] == dict(Counter(row["category"] for row in rows)), "CLOCK_COUNTS_CHANGED")
    return dict(statistics, **counts, source_label=label, source_binding=dict(binding)), rows, frame, selected["rows"]


def read_pair(settings: Mapping[str, Any], census_report: Mapping[str, Any],
              census_arrays: Mapping[str, np.ndarray], coarse_dir: Path, fine_dir: Path,
              source_bindings: Mapping[str, dict], output: Path) -> dict[str, Any]:
    """Read an authenticated pair and write numerical results to a new directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_PFC_PROFILE_HISTORY_READBACK", new_full_network_episodes=0,
                  third_spike_probability_established=False,
                  source_history_population_uncertainty_established=False, **BOUNDARY)
    started = time.perf_counter()
    try:
        profile, repeat, seed = census_report["profile"], census_report["auxiliary_repeat"], census_report["source_rng_seed"]
        require(census_report["status"] == "COMPLETE_PFC_HISTORY_PROFILE_FRAME"
                and profile in settings["profiles"] and repeat in settings["auxiliary_repeats"]
                and seed in settings["required_source_seeds"], "CENSUS_PROFILE_IDENTITY")
        require(settings["importance_sampling"]["samples"] == 10000
                and settings["integration_seed_namespace"] == 5, "REGISTERED_INTEGRATION_DESIGN")
        require_false(census_report, "population_confidence_interval_computed", "scientific_fidelity_qualified",
                      "confirmation_execution_released")
        pair = pair_name(profile)
        request = dict(profile=profile, auxiliary_repeat=repeat, pair_name=pair)
        sampling_seeds, context = seed_identity(settings, seed, request)
        labels = ("native", "half_" + repeat) if pair == "native_half" else ("half_" + repeat, "quarter_" + repeat)
        for clock, label in zip(("coarse", "fine"), labels):
            seeds = source_bindings[clock]["refinement_seeds"]
            if label == "native":
                require(seeds is None, "NATIVE_REFINEMENT_IDENTITY")
            else:
                require(isinstance(seeds, (list, tuple)) and len(seeds) == 2
                        and all(type(value) is int and 0 <= value < 2**32 for value in seeds)
                        and seeds[0] != seeds[1], "REFINEMENT_SEED_IDENTITY")
        if pair == "half_quarter":
            require(source_bindings["coarse"]["refinement_seeds"] == source_bindings["fine"]["refinement_seeds"],
                    "PAIRED_REFINEMENT_IDENTITY")
        clocks = {}
        for clock, directory, label in zip(("coarse", "fine"), (Path(coarse_dir), Path(fine_dir)), labels):
            clocks[clock] = _read_clock(settings, census_report, census_arrays, directory, source_bindings[clock],
                                       clock=clock, label=label, profile=profile, repeat=repeat, seed=seed,
                                       sampling_seeds=sampling_seeds, context=context)
        coarse_stats, coarse_rows, frame, coarse_selected = clocks["coarse"]
        fine_stats, fine_rows, fine_frame, fine_selected = clocks["fine"]
        require(frame == fine_frame
                and [[row[key] for key in _PHYSICAL_KEYS] for row in coarse_selected]
                == [[row[key] for key in _PHYSICAL_KEYS] for row in fine_selected], "PAIRED_PHYSICAL_FRAME_CHANGED")
        statistics, paired_rows = paired_components(coarse_rows, fine_rows, frame)
        write_json(output / "paired_frame.json", dict(frame=frame, rows=paired_rows))
        report.update(status="COMPLETE_PFC_PROFILE_HISTORY_READBACK", source_rng_seed=seed,
                      profile=profile, auxiliary_repeat=repeat, pair_name=pair, frame=frame,
                      paired_selected_physical_windows=frame["sample_size"],
                      per_clock=dict(coarse=coarse_stats, fine=fine_stats), statistics=statistics,
                      paired_frame_file="paired_frame.json",
                      scope="Numerical checks of authenticated archived-frame diagnostics; no population confidence or source-law qualification")
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        write_json(output / "report.json", report)
    return report
