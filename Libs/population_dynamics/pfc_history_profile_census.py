"""Shared source loading for the unresolved PFC count-CDF physical frames.

The outputs enumerate archived windows. They do not estimate source-law coverage.
Source archives and existing census assignments are authenticated before reuse.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
import json
from pathlib import Path
import resource
import shutil
import time
from typing import Any

import numpy as np
import yaml

from .pfc_history_timing_census import (
    _validate_census, census_many, pair, require,
)


Copy = Callable[[Path, Path, str | None], None]
CLOCKS = {"numpy_dt_0p05": .05, "numpy_dt_0p025": .025, "numpy_dt_0p0125": .0125}
SOURCE_CLOCKS = {"native": .05, "half_A": .025, "quarter_A": .0125,
                 "half_B": .025, "quarter_B": .0125}
BOUNDARY = dict(population_confidence_interval_computed=False,
                scientific_fidelity_qualified=False, confirmation_execution_released=False,
                new_neural_trajectories=0, new_independent_source_blocks=0, production_episodes=0)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def frame_key(profile: Mapping[str, Any]) -> str:
    return f"field_{profile['field_index']}_{profile['phase']}"


def frame_requests(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Deduplicate physical frames while retaining all endpoint specifications."""
    frames = {}
    rows = set()
    for profile in settings["profiles"]:
        require(profile["condition"] == "strong_cue" and profile["window_ms"] == 5.,
                "PROFILE_CONDITION_OR_WINDOW_CHANGED")
        require((profile["left_cohort"], profile["right_cohort"]) in (
            ("numpy_dt_0p05", "numpy_dt_0p025"), ("numpy_dt_0p025", "numpy_dt_0p0125")),
            "PROFILE_CLOCK_PAIR_CHANGED")
        for endpoint in profile["endpoints"]:
            require(endpoint["count_threshold"] in (1, 2) and endpoint["row_index"] not in rows,
                    "PROFILE_ENDPOINT_IDENTITY")
            rows.add(endpoint["row_index"])
        request = {key: profile[key] for key in ("field_index", "phase", "window_ms")}
        request["time_edges_ms"] = settings["time_edges_ms"]
        key = frame_key(profile)
        require(key not in frames or frames[key] == request, "PROFILE_FRAME_COLLISION")
        frames[key] = request
    return list(frames.values())


def validate_source(config: Mapping[str, Any], report: Mapping[str, Any], *,
                    seed: int, label: str, expected_units: int,
                    refinement_seeds: Sequence[int] | None,
                    coarse_reference: Mapping[str, Any]) -> None:
    require(report["resolved_config"] == config, "PROFILE_SOURCE_CONFIG_REPORT_MISMATCH")
    require(report["source_execution_complete"] is True, "PROFILE_SOURCE_INCOMPLETE")
    selected = report["selected_case"]
    require(selected["source_rng_seed"] == seed and selected["case_id"] == f"pfc_{seed}_strong_cue",
            "PROFILE_SOURCE_IDENTITY")
    require(config["excitatory_neurons"] + config["inhibitory_neurons"] == expected_units
            and config["excitatory_preference_bins"] == 16
            and config["dt_ms"] == SOURCE_CLOCKS[label] and config["duration_ms"] == 5000.,
            "PROFILE_SOURCE_DIMENSIONS")
    if label == "native":
        require(report["status"] == "PFC_SOURCE_EXECUTION_COMPLETE", "PROFILE_NATIVE_STATUS")
    else:
        require(report["status"] == "COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE"
                and report["coarse_dt_ms"] == .05
                and report["refinement_seeds"] == list(refinement_seeds)
                and report["drive"]["refinement_factor"] == round(.05 / SOURCE_CLOCKS[label]),
                "PROFILE_ORIGINAL_PARENT_OR_AUXILIARY_CHANGED")
        require(all(report["coarse_reference"][key] == coarse_reference[key]
                    for key in ("archive", "delivery_sha256")), "PROFILE_NATIVE_PARENT_CHANGED")
        require(report["completed_duration_ticks"] == round(5000. / SOURCE_CLOCKS[label]),
                "PROFILE_SOURCE_TRUNCATED")


def prepare(settings: Mapping[str, Any], repo: Path, work: Path, copy: Copy) -> dict[str, Any]:
    requests = frame_requests(settings)
    require(len(requests) == 18 and len(settings["profiles"]) == 36
            and sum(len(p["endpoints"]) for p in settings["profiles"]) == 64,
            "ALL_UNRESOLVED_PROFILE_IDENTITIES_REQUIRED")
    require(settings["required_source_seeds"] == list(range(95001, 95033))
            and settings["expected_units"] == 100000 and settings["source_parent_dt_ms"] == .05
            and settings["time_edges_ms"] == [0., 1., 2., 3.], "PROFILE_PROTOCOL_CHANGED")
    accepted = settings["accepted_census"]
    origin = repo / accepted["archive"]
    copy(origin / "delivery.json", Path("accepted_census_delivery.json"), accepted["delivery_sha256"])
    delivery = json.loads((work / "accepted_census_delivery.json").read_text())
    require(delivery["verified"] and delivery["status"] == "COMPLETE_PFC_HISTORY_TIMING_CENSUS_COMPLETE",
            "ACCEPTED_CENSUS_INCOMPLETE")
    entries = {row["path"]: row for row in delivery["files"]}
    copy(origin / "source_index.json", Path("accepted_source_index.json"), entries["source_index.json"]["sha256"])
    original = json.loads((work / "accepted_source_index.json").read_text())["source_blocks"]
    registry = settings["fine_registry"]
    copy(repo / registry["path"], Path(registry["path"]), registry["sha256"])
    fine = {row["source_rng_seed"]: row for row in yaml.safe_load((work / registry["path"]).read_text())["source_blocks"]}
    repeat = {row["source_rng_seed"]: row for row in settings["repeat_blocks"]}
    require(set(fine) == set(repeat) == set(settings["required_source_seeds"]), "PROFILE_ALL32_SOURCE_BINDINGS")
    blocks = {}
    for seed in settings["required_source_seeds"]:
        base = original[str(seed)]["coarse"]
        sources = {}
        selected_case = None
        for label, dt in SOURCE_CLOCKS.items():
            ref = base if label == "native" else fine[seed]["source"] if label.endswith("A") else repeat[seed]
            metadata = Path("source_metadata") / str(seed) / label
            source_origin = repo / ref["archive"]
            copy(source_origin / "delivery.json", metadata / "delivery.json", ref["delivery_sha256"])
            source_delivery = json.loads((work / metadata / "delivery.json").read_text())
            if label == "native":
                require(source_delivery["status"] == "ARCHIVED_PFC_SOURCE_EXECUTION"
                        and source_delivery["source_execution_complete"] and source_delivery["exit_code"] == 0,
                        "PROFILE_NATIVE_DELIVERY")
            else:
                allowed = ("COMPLETE_COUPLED_CLOCK_DEVELOPMENT_PILOT", "COMPLETE_COUPLED_CLOCK_CALIBRATION_BLOCK") if label.endswith("A") else ("COMPLETE_COUPLED_REFINEMENT_REPEAT_BLOCK",)
                require(source_delivery["verified"] and source_delivery["status"] in allowed,
                        "PROFILE_FINE_DELIVERY")
            source_entries = {row["path"]: row for row in source_delivery["files"]}
            case = Path("source_run") if label == "native" else Path("source_cases") / ("dt_" + str(dt).replace(".", "p")) / f"pfc_{seed}_strong_cue"
            for filename in ("config.yaml", "report.json"):
                copy(source_origin / case / filename, metadata / filename, source_entries[str(case / filename)]["sha256"])
            cfg = yaml.safe_load((work / metadata / "config.yaml").read_text())
            report = json.loads((work / metadata / "report.json").read_text())
            seeds = None if label == "native" else fine[seed]["refinement_seeds"] if label.endswith("A") else [99411 + 2 * (seed - 95001), 99412 + 2 * (seed - 95001)]
            validate_source(cfg, report, seed=seed, label=label, expected_units=settings["expected_units"],
                            refinement_seeds=seeds, coarse_reference=base)
            if selected_case is None:
                selected_case = report["selected_case"]
            require(report["selected_case"] == selected_case, "PROFILE_PAIRED_SOURCE_CASE_CHANGED")
            raw = source_entries[str(case / "raw.npz")]
            if label in ("native", "half_A"):
                prior = original[str(seed)]["coarse" if label == "native" else "fine"]
                require(raw == prior["raw"] and ref["archive"] == prior["archive"]
                        and ref["delivery_sha256"] == prior["delivery_sha256"], "PROFILE_REUSED_CENSUS_SOURCE_CHANGED")
            sources[label] = dict(archive=ref["archive"], delivery_sha256=ref["delivery_sha256"],
                                  metadata=str(metadata), raw=raw, dt_ms=dt, refinement_seeds=seeds,
                                  source_parent_dt_ms=None if label == "native" else .05)
        reused = {name: entries[f"analysis/{seed}/{name}"] for name in ("report.json", "frame_assignments.npz")}
        blocks[str(seed)] = dict(sources=sources, selected_case=selected_case, reused_census=reused)
    index = dict(source_blocks=blocks, required_original_blocks=32, profiles=settings["profiles"],
                 distinct_frames_per_source=len(requests), distinct_sources_per_block=5,
                 raw_bytes=sum(s["raw"]["bytes"] for b in blocks.values() for s in b["sources"].values()),
                 max_source_bytes=max(s["raw"]["bytes"] for b in blocks.values() for s in b["sources"].values()))
    write_json(work / "source_index.json", index)
    report = dict(status="PREPARED_SHARED_PROFILE_CENSUS", required_original_blocks=32,
                  source_cases=len(blocks) * 5, profiles=36, endpoints=64,
                  raw_bytes_declared=index["raw_bytes"], source_events_read=False, **BOUNDARY)
    write_json(work / "analysis/report.json", report)
    return report


def unpack_pair(summary: Mapping[str, Any], arrays: Mapping[str, np.ndarray], clock: str):
    values = {"unit_indices": arrays["unit_indices"]}
    values.update({key[len(clock) + 1:]: value for key, value in arrays.items() if key.startswith(clock + "_")})
    metadata = summary["clocks"][clock]
    _validate_census(metadata, values)
    return metadata, values


def endpoint_counts(profile: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Retain exact integer paired observations for both original CDF thresholds."""
    coarse, fine = arrays["coarse_observed_spike_count"], arrays["fine_observed_spike_count"]
    require(coarse.shape == fine.shape and coarse.size > 0, "PROFILE_CDF_FRAME_CHANGED")
    rows = []
    for endpoint in profile["endpoints"]:
        threshold = endpoint["count_threshold"]
        left, right = coarse <= threshold, fine <= threshold
        delta = int(np.count_nonzero(right)) - int(np.count_nonzero(left))
        rows.append(dict(**endpoint, physical_window_denominator=int(coarse.size),
                         coarse_cdf_count=int(left.sum()), fine_cdf_count=int(right.sum()),
                         coarse_tail_count=int((~left).sum()), fine_tail_count=int((~right).sum()),
                         paired_discordant_windows=int(np.count_nonzero(left != right)),
                         fine_minus_coarse_cdf_count=delta,
                         fine_minus_coarse_empirical_cdf=delta / coarse.size))
    return rows


def execute(settings: Mapping[str, Any], repo: Path, work: Path, prepared: Path,
            prepared_entries: Mapping[str, Any], copy: Copy, seed: int) -> dict[str, Any]:
    require(seed in settings["required_source_seeds"], "PROFILE_ORIGINAL_SOURCE_BLOCK_REQUIRED")
    copy(prepared / "source_index.json", Path("source_index.json"), prepared_entries["source_index.json"]["sha256"])
    index = json.loads((work / "source_index.json").read_text())
    require(index["profiles"] == settings["profiles"], "PROFILE_ENDPOINT_REGISTRY_CHANGED")
    block = index["source_blocks"][str(seed)]
    frames = frame_requests(settings)
    scratch = work / "scratch" / "census"
    scratch.mkdir(parents=True)
    analysis = work / "analysis"
    analysis.mkdir()
    origin = repo / settings["accepted_census"]["archive"]
    for name, entry in block["reused_census"].items():
        copy(origin / entry["path"], Path("scratch/reused") / name, entry["sha256"])
    old_report = json.loads((work / "scratch/reused/report.json").read_text())
    require(old_report["source_rng_seed"] == seed and old_report["selected_case"] == block["selected_case"],
            "PROFILE_REUSED_CENSUS_IDENTITY")
    with np.load(work / "scratch/reused/frame_assignments.npz", allow_pickle=False) as archive:
        old_arrays = {key: archive[key] for key in archive.files}
    for label, clock in (("native", "coarse"), ("half_A", "fine")):
        metadata, arrays = unpack_pair(old_report, old_arrays, clock)
        require(metadata["field_index"] == 5 and metadata["phase"] == "cue"
                and metadata["dt_ms"] == SOURCE_CLOCKS[label], "PROFILE_REUSED_FRAME_CHANGED")
        target = scratch / label / "field_5_cue"
        target.mkdir(parents=True)
        write_json(target / "report.json", metadata)
        np.savez_compressed(target / "arrays.npz", **arrays)
    del arrays, old_arrays
    source_cost = []
    for label, source in block["sources"].items():
        started = time.perf_counter()
        directory = Path("source_inputs") / label
        metadata_path = Path(source["metadata"])
        for filename in ("config.yaml", "report.json"):
            relative = metadata_path / filename
            copy(prepared / relative, directory / filename, prepared_entries[str(relative)]["sha256"])
        raw = source["raw"]
        require(shutil.disk_usage(work).free >= raw["bytes"] + 1024**3, "PROFILE_SCRATCH_CAPACITY")
        copy(repo / source["archive"] / raw["path"], directory / "raw.npz", raw["sha256"])
        cfg = yaml.safe_load((work / directory / "config.yaml").read_text())
        staged_seconds = time.perf_counter() - started
        with np.load(work / directory / "raw.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in ("spike_ticks", "spike_unit_indices", "native_spike_counts", "spike_seconds") if key in archive.files}
        requests = [request for request in frames if not (label in ("native", "half_A") and frame_key(request) == "field_5_cue")]
        sizes = []
        for metadata, arrays in census_many(cfg, data, requests):
            target = scratch / label / frame_key(metadata)
            target.mkdir(parents=True)
            write_json(target / "report.json", metadata)
            np.savez_compressed(target / "arrays.npz", **arrays)
            sizes.append(metadata["frame_size"])
        del data, arrays
        shutil.rmtree(work / directory)
        source_cost.append(dict(source=label, raw_bytes=raw["bytes"], source_loads=1,
                                tape_validations=1, computed_frames=len(sizes),
                                computed_physical_windows=sum(sizes), staging_seconds=staged_seconds,
                                wall_seconds=time.perf_counter() - started))
        print(json.dumps(source_cost[-1]), flush=True)
    results = []
    for repeat in settings["auxiliary_repeats"]:
        for profile in settings["profiles"]:
            key = frame_key(profile)
            pair_name = "native_half" if profile["left_cohort"] == "numpy_dt_0p05" else "half_quarter"
            relative = Path(repeat) / pair_name / key
            target = analysis / relative
            target.mkdir(parents=True)
            reused = repeat == "A" and pair_name == "native_half" and key == "field_5_cue"
            if reused:
                # Keep the accepted paired file byte-for-byte; do not repeat its census.
                shutil.copy2(work / "scratch/reused/frame_assignments.npz", target / "frame_assignments.npz")
                summary = dict(old_report)
                with np.load(target / "frame_assignments.npz", allow_pickle=False) as archive:
                    paired_arrays = {key: archive[key] for key in archive.files}
            else:
                labels = ("native", "half_" + repeat) if pair_name == "native_half" else ("half_" + repeat, "quarter_" + repeat)
                values = []
                for label in labels:
                    frame_dir = scratch / label / key
                    with np.load(frame_dir / "arrays.npz", allow_pickle=False) as archive:
                        arrays = {key: archive[key] for key in archive.files}
                    values.extend((json.loads((frame_dir / "report.json").read_text()), arrays))
                summary, paired_arrays = pair(*values)
                del values, arrays
                np.savez_compressed(target / "frame_assignments.npz", **paired_arrays)
            rows = endpoint_counts(profile, paired_arrays)
            del paired_arrays
            summary.update(status="COMPLETE_PFC_HISTORY_PROFILE_FRAME", source_rng_seed=seed,
                           auxiliary_repeat=repeat, profile=profile, endpoints=rows,
                           reused_accepted_pair=reused, assignments_file="frame_assignments.npz", **BOUNDARY)
            write_json(target / "report.json", summary)
            results.append(dict(profile=profile, auxiliary_repeat=repeat, report=str(relative / "report.json"),
                                frame_size=summary["frame_size"], endpoints=rows,
                                multispike_windows={clock: summary["clocks"][clock]["multispike_windows"] for clock in ("coarse", "fine")}))
    report = dict(status="COMPLETE_PFC_HISTORY_PROFILE_BLOCK", source_rng_seed=seed,
                  distinct_source_cases=5, computed_single_clock_frames=88, reused_single_clock_frames=2,
                  completed_profile_pairs=len(results), expected_profile_pairs=72,
                  reused_paired_frames=1, computed_paired_frames=71, source_cost=source_cost,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  paired_frame_visits=sum(row["frame_size"] for row in results),
                  frame_visit_scope="Repeated clock and auxiliary comparisons share physical IDs; not independent observations",
                  original_blocks=1, source_parent_dt_ms=.05, results=results, **BOUNDARY)
    write_json(analysis / "report.json", report)
    fields = ["source_rng_seed", "auxiliary_repeat", "left_cohort", "right_cohort", "field_index", "phase",
              "row_index", "count_threshold", "physical_window_denominator", "coarse_cdf_count", "fine_cdf_count",
              "coarse_tail_count", "fine_tail_count", "paired_discordant_windows", "fine_minus_coarse_cdf_count",
              "fine_minus_coarse_empirical_cdf"]
    with (analysis / "endpoint_counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            profile = result["profile"]
            for endpoint in result["endpoints"]:
                writer.writerow(dict(source_rng_seed=seed, auxiliary_repeat=result["auxiliary_repeat"],
                                     **{key: profile[key] for key in ("left_cohort", "right_cohort", "field_index", "phase")}, **endpoint))
    require(len(results) == 72, "PROFILE_BLOCK_INCOMPLETE")
    return report
