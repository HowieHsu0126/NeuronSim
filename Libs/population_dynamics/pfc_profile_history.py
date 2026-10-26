"""Shared archived-source replay for multiple PFC conditional-history frames.

Selection and integration use declared profile identities. The source tape and
input stream are reconstructed once per source case. These conditional estimates
are not population confidence intervals or source qualification.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import resource
import shutil
import tarfile
import time
from typing import Any

import numpy as np
import yaml

from . import pfc_first_passage_history as history
from .pfc_history_profile_census import BOUNDARY, SOURCE_CLOCKS, frame_key, unpack_pair, write_json
from .pfc_stratified_history import sampling_callbacks


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("PFC_PROFILE_HISTORY_" + message)


def pair_name(profile: Mapping[str, Any]) -> str:
    left, right = profile["left_cohort"], profile["right_cohort"]
    require((left, right) in (("numpy_dt_0p05", "numpy_dt_0p025"),
                              ("numpy_dt_0p025", "numpy_dt_0p0125")), "REGISTERED_CLOCK_PAIR")
    return "native_half" if left == "numpy_dt_0p05" else "half_quarter"


def profile_key(profile: Mapping[str, Any], repeat: str) -> str:
    return str(Path(repeat) / pair_name(profile) / frame_key(profile))


def source_requests(settings: Mapping[str, Any], label: str, *, pilot: bool, seed: int | None = None) -> list[dict[str, Any]]:
    require(label in SOURCE_CLOCKS, "SOURCE_LABEL")
    result = []
    for repeat in settings["auxiliary_repeats"]:
        for profile in settings["profiles"]:
            pair = pair_name(profile)
            labels = ("native", "half_" + repeat) if pair == "native_half" else ("half_" + repeat, "quarter_" + repeat)
            if label not in labels:
                continue
            key = profile_key(profile, repeat)
            if key in settings["reused_profile_keys"]:
                continue
            if (not pilot and seed is not None and settings.get("accepted_pilot")
                    and seed == settings["pilot"]["source_rng_seed"] == 95001
                    and label in settings["pilot"]["source_labels"]
                    and key in settings["pilot"]["profile_keys"]):
                continue
            if pilot and key not in settings["pilot"]["profile_keys"]:
                continue
            result.append(dict(key=key, profile=profile, auxiliary_repeat=repeat,
                               pair_name=pair, clock="coarse" if label == labels[0] else "fine"))
    return result


def seed_identity(settings: Mapping[str, Any], seed: int, request: Mapping[str, Any]):
    profile = request["profile"]
    context = [settings["phase_codes"][profile["phase"]],
               settings["clock_pair_codes"][request["pair_name"]],
               settings["auxiliary_repeat_codes"][request["auxiliary_repeat"]]]
    sampling = [settings["sampling_seed_epoch"], seed, profile["field_index"],
                *context, settings["integration_seed_namespace"]]
    require(all(type(value) is int and 0 <= value < 2**32 for value in sampling), "SEED_IDENTITY")
    return sampling, context


def merge_selected_frames(frames: Sequence[Sequence[Mapping[str, Any]]]):
    """Deduplicate physical windows; keep a local-to-shared index for every row."""
    shared, lookup, mappings = [], {}, []
    for rows in frames:
        require([row["sample_index"] for row in rows] == list(range(len(rows))), "LOCAL_SAMPLE_ORDER")
        mapping = []
        for row in rows:
            identity = (row["unit_index"], row["window_start_tick"], row["window_stop_tick"])
            if identity not in lookup:
                lookup[identity] = len(shared)
                shared.append(dict(row, sample_index=len(shared)))
            mapping.append(lookup[identity])
        mappings.append(mapping)
    return shared, mappings


def local_classified_rows(selected, mapping, classified):
    require(len(selected) == len(mapping), "LOCAL_MAPPING_SIZE")
    rows = []
    for row, shared_index in zip(selected, mapping):
        require(type(shared_index) is int and 0 <= shared_index < len(classified)
                and classified[shared_index]["sample_index"] == shared_index, "SHARED_INDEX_IDENTITY")
        original = classified[shared_index]
        require(all(row[key] == original[key] for key in
                    ("unit_index", "window_start_tick", "window_stop_tick", "window_start_ms", "window_stop_ms")),
                "SHARED_PHYSICAL_WINDOW_CHANGED")
        rows.append(dict(original, **row, shared_history_index=shared_index))
    return rows


def pack_integration_details(output: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Verify an archive before removing this task's node-local detail folders."""
    folders = [output / Path(row["integration_report"]).parts[0]
               for row in rows if "integration_report" in row]
    require(len(folders) == len(set(folders)), "INTEGRATION_DETAIL_DUPLICATE")
    if not folders:
        return dict(integration_details_archive=None, integration_detail_files=0)
    files = [path for folder in folders for path in sorted(folder.rglob("*")) if path.is_file()]
    archive = output / "integration_details.tar"
    with tarfile.open(archive, "w") as stream:
        for folder in folders:
            stream.add(folder, arcname=str(folder.relative_to(output)))
    with tarfile.open(archive, "r:") as stream:
        require(sum(member.isfile() for member in stream.getmembers()) == len(files), "DETAIL_ARCHIVE_FILE_COUNT")
        for path in files:
            member = stream.extractfile(str(path.relative_to(output)))
            require(member is not None and member.read() == path.read_bytes(), "DETAIL_ARCHIVE_CONTENT_CHANGED")
    for folder in folders:
        shutil.rmtree(folder)
    return dict(integration_details_archive=archive.name, integration_detail_files=len(files),
                integration_details_scope="Relative sample paths are members of the verified per-profile tar archive")


def run_source(settings: Mapping[str, Any], source_dir: Path, census_root: Path,
               output: Path, *, seed: int, label: str, pilot: bool) -> dict[str, Any]:
    require(settings["schema"] == "pfc_profile_history_v1"
            and settings["required_source_seeds"] == list(range(95001, 95033))
            and seed in settings["required_source_seeds"]
            and settings["sample_size"] == 640 and settings["importance_sampling"]["samples"] == 10000
            and settings["integration_seed_namespace"] == 5, "REGISTERED_PROFILE_DESIGN")
    requests = source_requests(settings, label, pilot=pilot, seed=seed)
    require(requests and (not pilot or seed == settings["pilot"]["source_rng_seed"]), "REGISTERED_EXECUTION_SCOPE")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="FAILED_PFC_PROFILE_SOURCE_HISTORY", source_rng_seed=seed, source_label=label,
                  pilot=pilot, requested_profile_keys=[row["key"] for row in requests],
                  completed_profile_keys=[], new_full_network_episodes=0,
                  source_history_population_uncertainty_established=False, **BOUNDARY)
    started = time.perf_counter()
    try:
        cfg = yaml.safe_load((source_dir / "config.yaml").read_text())
        sr = json.loads((source_dir / "report.json").read_text())
        case = sr["selected_case"]
        require(cfg == sr["resolved_config"] and cfg["dt_ms"] == SOURCE_CLOCKS[label]
                and cfg["excitatory_neurons"] + cfg["inhibitory_neurons"] == settings["expected_units"] == 100000
                and cfg["duration_ms"] == 5000. and case["source_rng_seed"] == seed
                and case["case_id"] == f"pfc_{seed}_strong_cue" and sr["source_execution_complete"],
                "SOURCE_IDENTITY_OR_DIMENSIONS")
        kind = "native_numpy_stream" if label == "native" else "coupled_binomial_clock_refinement_stream"
        if label != "native":
            require(sr["coarse_dt_ms"] == .05 and sr["drive"]["refinement_factor"] == round(.05 / cfg["dt_ms"])
                    and sr["refinement_seeds"] == settings["source_binding"]["refinement_seeds"],
                    "ORIGINAL_NATIVE_PARENT_REQUIRED")
        selected_frames = []
        frame_metadata = []
        for request in requests:
            directory = census_root / request["key"]
            census_report = json.loads((directory / "report.json").read_text())
            require(census_report["status"] == "COMPLETE_PFC_HISTORY_PROFILE_FRAME"
                    and census_report["source_rng_seed"] == seed
                    and census_report["profile"] == request["profile"]
                    and census_report["auxiliary_repeat"] == request["auxiliary_repeat"], "CENSUS_PROFILE_IDENTITY")
            with np.load(directory / "frame_assignments.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            unpack_pair(census_report, arrays, request["clock"])
            sampler, summary = sampling_callbacks(census_report, arrays, request["clock"])
            sampling_seeds, context = seed_identity(settings, seed, request)
            p = request["profile"]
            rows, frame = sampler(cfg, field=p["field_index"], phase=p["phase"], window_ms=p["window_ms"],
                                  sample_size=settings["sample_size"], seed_components=sampling_seeds)
            target = output / "frames" / request["key"]
            write_json(target / "selected_frame.json", dict(frame=frame, rows=rows))
            selected_frames.append(rows)
            frame_metadata.append(frame)
            del arrays, sampler, summary
        shared, mappings = merge_selected_frames(selected_frames)
        with np.load(source_dir / "raw.npz", allow_pickle=False) as archive:
            raw = {key: archive[key] for key in ("spike_ticks", "spike_unit_indices", "native_spike_counts", "spike_seconds") if key in archive.files}
        tape = history.reference.validate_tape(cfg, raw)
        del raw
        classified = history.classify(cfg, tape, shared)
        write_json(output / "shared_classified_frame.json", classified)
        write_json(output / "profile_reference_mapping.json", {r["key"]: m for r, m in zip(requests, mappings)})
        initial, drive = history.create_input_drive(cfg, case, sr, kind)
        require(np.array_equal(initial, np.load(source_dir / "initial_voltage.npy", allow_pickle=False)),
                "SOURCE_INITIAL_STATE_CHANGED")
        begin = time.perf_counter()
        refs, reconstruction = history.extract(cfg, case, tape, classified, drive=drive)
        reconstruction_seconds = time.perf_counter() - begin
        np.savez_compressed(output / "reconstructed_prefix_rng.npz", **history.prefix_state_arrays(drive, kind))
        report.update(distinct_source_loads=1, tape_validations=1, history_extraction_calls=1, source_reconstruction_calls=int(reconstruction["ticks_reconstructed"] > 0),
                      selected_clock_profile_windows=sum(map(len, selected_frames)),
                      distinct_selected_physical_windows=len(shared),
                      duplicate_reference_extractions_avoided=sum(map(len, selected_frames))-len(shared),
                      distinct_integrated_references=len(refs), shared_categories=dict(Counter(r["category"] for r in classified)),
                      reconstruction=reconstruction, reconstruction_seconds=reconstruction_seconds,
                      reconstruction_drive=drive.metadata(), input_stream=kind,
                      replay_scope="Existing archived spike tape and original input stream; no new source-law episode")
        frame_reports = []
        for request, selected, frame, mapping in zip(requests, selected_frames, frame_metadata, mappings):
            target = output / "frames" / request["key"]
            rows = local_classified_rows(selected, mapping, classified)
            local_refs = {row["sample_index"]: refs[index] for row, index in zip(rows, mapping) if index in refs}
            write_json(target / "classified_frame.json", rows)
            _, context = seed_identity(settings, seed, request)
            per_frame = dict(settings, source_rng_seed=seed, source=settings["source_binding"],
                             field_index=request["profile"]["field_index"], integration_seed_context=context)
            begin = time.perf_counter()
            history.integrate_rows(cfg, per_frame, rows, local_refs, target)
            elapsed = time.perf_counter() - begin
            directory = census_root / request["key"]
            census_report = json.loads((directory / "report.json").read_text())
            with np.load(directory / "frame_assignments.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            sampler, summarize = sampling_callbacks(census_report, arrays, request["clock"])
            summary = summarize(rows, frame)
            del arrays, sampler, summarize
            summary.update(status="COMPLETE_PFC_PROFILE_CONDITIONAL_HISTORY", source_rng_seed=seed,
                           source_label=label, input_stream=kind, dt_ms=cfg["dt_ms"],
                           source_parent_dt_ms=None if label == "native" else sr["coarse_dt_ms"],
                           profile=request["profile"], auxiliary_repeat=request["auxiliary_repeat"], clock=request["clock"],
                           frame=frame, integrated_histories=len(local_refs),
                           inner_trials=len(local_refs)*settings["importance_sampling"]["samples"],
                           categories=dict(Counter(r["category"] for r in rows)),
                           integration_seed_namespace=settings["integration_seed_namespace"], integration_seed_context=context,
                           integration_seconds=elapsed, third_spike_probability_established=False,
                           source_history_population_uncertainty_established=False, **BOUNDARY)
            summary.update(pack_integration_details(target, rows))
            write_json(target / "report.json", summary)
            frame_reports.append(dict(key=request["key"], report=str(target.relative_to(output) / "report.json"),
                                      sample_mean=summary["sample_mean"], integrated_histories=len(local_refs), inner_trials=summary["inner_trials"]))
            report["completed_profile_keys"].append(request["key"])
        report.update(status="COMPLETE_PFC_PROFILE_SOURCE_HISTORY", frame_reports=frame_reports,
                      total_inner_trials=sum(row["inner_trials"] for row in frame_reports))
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report.update(wall_seconds=time.perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(output / "report.json", report)
    return report


def bind_accepted_pilot(settings, repo, work, copy, source_index, source_preparation_entries):
    """Bind accepted pilot receipts before excluding their source/profile requests."""
    if "accepted_pilot" not in settings:
        return 0
    accepted = settings["accepted_pilot"]
    pilot = settings["pilot"]
    seed, labels, keys = pilot["source_rng_seed"], pilot["source_labels"], pilot["profile_keys"]
    require(isinstance(accepted, Mapping) and seed == 95001 and labels == ["half_A", "quarter_A"]
            and len(keys) == len(set(keys)) == 2
            and set(accepted["source_runs"]) == set(labels), "ACCEPTED_PILOT_SCOPE")
    expected_settings = {key: value for key, value in settings.items() if key != "accepted_pilot"}

    def completed(reference, label, status):
        origin = repo / reference["archive"]
        target = Path("accepted_pilot") / label
        copy(origin / "delivery.json", target / "delivery.json", reference["delivery_sha256"])
        receipt = json.loads((work / target / "delivery.json").read_text())
        require(receipt["verified"] is True and receipt["status"] == status, "ACCEPTED_PILOT_DELIVERY:" + label)
        entries = {row["path"]: row for row in receipt["files"]}
        for name in ("report.json", "analysis/report.json"):
            copy(origin / name, target / name, entries[name]["sha256"])
        wrapper = json.loads((work / target / "report.json").read_text())
        require(wrapper["status"] == status and wrapper["exit_code"] == 0 and wrapper.get("error") is None,
                "ACCEPTED_PILOT_WRAPPER:" + label)
        report = json.loads((work / target / "analysis/report.json").read_text())
        return origin, target, entries, report

    _, _, _, readback = completed(accepted["readback"], "readback", "COMPLETE_PFC_PROFILE_HISTORY_READBACK_COMPLETE")
    require(readback["status"] == "COMPLETE_PFC_SHARED_PROFILE_HISTORY_READBACK"
            and readback.get("error") is None and readback["source_rng_seed"] == seed
            and readback["original_source_blocks"] == 1 and readback["paired_profiles"] == len(keys)
            and [pair["key"] for pair in readback["pairs"]] == keys, "ACCEPTED_PILOT_READBACK_IDENTITY")
    for clock, label in zip(("coarse", "fine"), labels):
        origin, target, entries, report = completed(accepted["source_runs"][label], label, "COMPLETE_PFC_PROFILE_HISTORY_COMPLETE")
        require(report["status"] == "COMPLETE_PFC_PROFILE_SOURCE_HISTORY" and report.get("error") is None
                and report["source_rng_seed"] == seed and report["source_label"] == label and report["pilot"] is True
                and report["requested_profile_keys"] == report["completed_profile_keys"] == keys
                and [frame["key"] for frame in report["frame_reports"]] == keys, "ACCEPTED_PILOT_SOURCE_IDENTITY:" + label)
        config_path = "Libs/configs/neural_population_pfc_profile_history.yaml"
        for name in (config_path, "runtime_config.json"):
            copy(origin / name, target / name, entries[name]["sha256"])
        old_settings = yaml.safe_load((work / target / config_path).read_text())
        runtime = json.loads((work / target / "runtime_config.json").read_text())
        require(old_settings == expected_settings
                and {key: value for key, value in runtime.items() if key != "source_binding"} == old_settings,
                "ACCEPTED_PILOT_SETTINGS_CHANGED:" + label)
        source = source_index["source_blocks"][str(seed)]["sources"][label]
        report_path = str(Path(source["metadata"]) / "report.json")
        expected_binding = dict(archive=source["archive"], delivery_sha256=source["delivery_sha256"],
                                case_path=str(Path(source["raw"]["path"]).parent),
                                report_sha256=source_preparation_entries[report_path]["sha256"],
                                refinement_seeds=source["refinement_seeds"])
        require(runtime["source_binding"] == expected_binding, "ACCEPTED_PILOT_SOURCE_BINDING:" + label)
        require(readback["source_costs"][clock]["source_binding"] == expected_binding,
                "ACCEPTED_PILOT_READBACK_SOURCE_BINDING:" + label)
        for frame, pair in zip(report["frame_reports"], readback["pairs"]):
            observed = pair["per_clock"][clock]
            require(observed["source_label"] == label and observed["source_binding"] == expected_binding
                    and all(frame[key] == observed[key] for key in ("sample_mean", "integrated_histories", "inner_trials")),
                    "ACCEPTED_PILOT_PAIRED_SOURCE_IDENTITY:" + label)
    return len(labels) * len(keys)


def prepare_inputs(settings, repo, work, copy):
    """Reuse accepted all32 source metadata and census readback bindings."""
    references = (("source_preparation", "source_preparation_delivery.json", "COMPLETE_PFC_HISTORY_PROFILE_CENSUS_PREPARE"),
                  ("census_readback", "census_readback_delivery.json", "COMPLETE_PFC_HISTORY_PROFILE_SUMMARY_PREPARE"))
    receipts = {}
    for key, destination, status in references:
        reference = settings[key]
        origin = repo / reference["archive"]
        copy(origin / "delivery.json", Path(destination), reference["delivery_sha256"])
        receipt = json.loads((work / destination).read_text())
        require(receipt["verified"] and receipt["status"] == status, "ACCEPTED_INPUT_INCOMPLETE")
        receipts[key] = {row["path"]: row for row in receipt["files"]}
    origin = repo / settings["source_preparation"]["archive"]
    copy(origin / "source_index.json", Path("source_index.json"), receipts["source_preparation"]["source_index.json"]["sha256"])
    source_index = json.loads((work / "source_index.json").read_text())
    require(source_index["profiles"] == settings["profiles"]
            and sorted(map(int, source_index["source_blocks"])) == settings["required_source_seeds"], "SOURCE_REGISTRY_CHANGED")
    origin = repo / settings["census_readback"]["archive"]
    copy(origin / "run_index.json", Path("census_run_index.json"), receipts["census_readback"]["run_index.json"]["sha256"])
    copy(origin / "analysis/report.json", Path("census_readback_report.json"), receipts["census_readback"]["analysis/report.json"]["sha256"])
    readback = json.loads((work / "census_readback_report.json").read_text())
    require(readback["status"] == "COMPLETE_SHARED_PROFILE_CENSUS_READBACK"
            and readback["completed_original_blocks"] == 32 and not readback["failures"], "FULL32_CENSUS_REQUIRED")
    origin = repo / settings["reused_integration_summary"]["archive"]
    copy(origin / "delivery.json", Path("reused_integration_delivery.json"), settings["reused_integration_summary"]["delivery_sha256"])
    receipt = json.loads((work / "reused_integration_delivery.json").read_text())
    require(receipt["verified"] and receipt["status"] == "COMPLETE_PFC_STRATIFIED_HISTORY_SUMMARY_COMPLETE", "REUSED_INTEGRATION_INCOMPLETE")
    counts = {label: len(source_requests(settings, label, pilot=False)) for label in SOURCE_CLOCKS}
    require(sum(counts.values()) == 142 and len(settings["profiles"]) == 36
            and settings["reused_profile_keys"] == ["A/native_half/field_5_cue"], "ALL_REMAINING_PROFILES_REQUIRED")
    pilot = {label: source_requests(settings, label, pilot=True) for label in settings["pilot"]["source_labels"]}
    require(settings["pilot"]["source_labels"] == ["half_A", "quarter_A"]
            and all([r["key"] for r in requests] == settings["pilot"]["profile_keys"] for requests in pilot.values()),
            "MATCHED_PILOT_FRAMES_REQUIRED")
    reused_pilot = bind_accepted_pilot(settings, repo, work, copy, source_index, receipts["source_preparation"])
    remaining = {str(seed): {label: len(source_requests(settings, label, pilot=False, seed=seed))
                            for label in SOURCE_CLOCKS} for seed in settings["required_source_seeds"]}
    remaining_total = sum(sum(counts.values()) for counts in remaining.values())
    require(remaining_total == sum(counts.values()) * len(settings["required_source_seeds"]) - reused_pilot,
            "REMAINING_PROFILE_ACCOUNTING")
    report = dict(status="PREPARED_SHARED_PROFILE_HISTORY", full_source_case_requests=counts,
                  remaining_source_case_requests=remaining, remaining_clock_frames=remaining_total,
                  reused_pilot_clock_frames=reused_pilot, remaining_paired_profiles=remaining_total // 2,
                  full_paired_profiles_per_block=72, reused_paired_profiles_per_block=1,
                  planned_new_paired_profiles_per_block=71, required_original_blocks=32,
                  pilot=settings["pilot"], sampling_namespace=settings["integration_seed_namespace"], **BOUNDARY)
    write_json(work / "analysis/report.json", report)
    return report


def execute_staged(settings, repo, work, prepared, prepared_entries, copy, *, task_index, pilot):
    """Stage only this source and requested census frames, then use local files."""
    for name in ("source_index.json", "source_preparation_delivery.json", "census_run_index.json"):
        copy(prepared / name, Path(name), prepared_entries[name]["sha256"])
    source_index = json.loads((work / "source_index.json").read_text())
    census_index = json.loads((work / "census_run_index.json").read_text())
    if pilot:
        require(0 <= task_index < len(settings["pilot"]["source_labels"]), "PILOT_SOURCE_INDEX")
        seed, label = settings["pilot"]["source_rng_seed"], settings["pilot"]["source_labels"][task_index]
    else:
        labels = list(SOURCE_CLOCKS)
        require(0 <= task_index < len(settings["required_source_seeds"]) * len(labels), "SOURCE_INDEX")
        seed, label = settings["required_source_seeds"][task_index // len(labels)], labels[task_index % len(labels)]
    source = source_index["source_blocks"][str(seed)]["sources"][label]
    source_prepared = repo / settings["source_preparation"]["archive"]
    original_entries = {row["path"]: row for row in json.loads((work / "source_preparation_delivery.json").read_text())["files"]}
    directory = Path("source_inputs/source")
    metadata = Path(source["metadata"])
    for name in ("config.yaml", "report.json", "delivery.json"):
        relative = metadata / name
        copy(source_prepared / relative, directory / name, original_entries[str(relative)]["sha256"])
    source_entries = {row["path"]: row for row in json.loads((work / directory / "delivery.json").read_text())["files"]}
    case = Path(source["raw"]["path"]).parent
    for name in ("raw.npz", "initial_voltage.npy"):
        entry = source_entries[str(case / name)]
        require(shutil.disk_usage(work).free >= entry["bytes"] + 1024**3, "LOCAL_SCRATCH_CAPACITY")
        copy(repo / source["archive"] / entry["path"], directory / name, entry["sha256"])
    census = next(row for row in census_index["runs"] if row["source_rng_seed"] == seed)
    origin = repo / census["archive"]
    copy(origin / "delivery.json", Path("census_delivery.json"), census["delivery_sha256"])
    receipt = json.loads((work / "census_delivery.json").read_text())
    require(receipt["verified"] and receipt["status"] == "COMPLETE_PFC_HISTORY_PROFILE_CENSUS_COMPLETE", "CENSUS_DELIVERY")
    entries = {row["path"]: row for row in receipt["files"]}
    for request in source_requests(settings, label, pilot=pilot, seed=seed):
        for name in ("report.json", "frame_assignments.npz"):
            relative = Path("analysis") / request["key"] / name
            copy(origin / relative, Path("source_inputs/census") / request["key"] / name, entries[str(relative)]["sha256"])
    runtime = dict(settings, source_binding=dict(archive=source["archive"], delivery_sha256=source["delivery_sha256"],
                                                case_path=str(case), report_sha256=source_entries[str(case / "report.json")]["sha256"],
                                                refinement_seeds=source["refinement_seeds"]))
    write_json(work / "runtime_config.json", runtime)
    return run_source(runtime, work / directory, work / "source_inputs/census", work / "analysis",
                      seed=seed, label=label, pilot=pilot)
