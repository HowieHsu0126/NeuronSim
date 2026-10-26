"""Native source spikes to a fixed periodic population-field observation.

This bounded check retains all source identities and original events. It does
not qualify the source's dynamics, a learned simulator, or biological fidelity.
"""
from __future__ import annotations

import json
from os.path import relpath
from pathlib import Path

import numpy as np

from .provenance import freeze_json, runtime_manifest, sha256_file
from .spatial_field import PeriodicPopulationGrid
from .spike_windows import count_spikes_at_edges, window_edges


def validate_field_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {
        "onset_seconds", "history_seconds", "horizon_seconds",
        "bin_seconds", "cells_per_axis", "unit_chunk_size"
    }:
        raise ValueError("MESOCIRCUIT_FIELD_CONFIG_FIELDS_INVALID")
    onset = spec["onset_seconds"]
    if type(onset) not in (int, float) or not np.isfinite(onset) or onset < 0:
        raise ValueError("MESOCIRCUIT_FIELD_FINITE_NONNEGATIVE_ONSET_REQUIRED")
    window_edges(spec["history_seconds"], spec["horizon_seconds"], spec["bin_seconds"])
    for key in ("cells_per_axis", "unit_chunk_size"):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError("MESOCIRCUIT_FIELD_POSITIVE_INTEGER_REQUIRED:" + key)


def check_mesocircuit_field(source_dir: Path, output: Path, config: dict,
                           command: list[str]) -> tuple[dict, int]:
    if output.exists():
        raise FileExistsError("MESOCIRCUIT_FIELD_OUTPUT_EXISTS:" + str(output))
    spec = config["mesocircuit_field_check"]
    validate_field_spec(spec)
    manifest = json.loads((source_dir / "manifest.json").read_text())
    if manifest.get("schema") != "mesocircuit_spatial_reference_v1":
        raise ValueError("MESOCIRCUIT_FIELD_NATIVE_SOURCE_REQUIRED")
    for name in ("summary.json", "spikes.npz"):
        if sha256_file(source_dir / name) != manifest["artifacts"][name]:
            raise ValueError("MESOCIRCUIT_FIELD_SOURCE_DIGEST_MISMATCH:" + name)
    source = json.loads((source_dir / "summary.json").read_text())
    if source["scope"] != "engineering_diagnostic" or source["boundary"] != "periodic square":
        raise ValueError("MESOCIRCUIT_FIELD_ENGINEERING_PERIODIC_SOURCE_REQUIRED")
    with np.load(source_dir / "spikes.npz", allow_pickle=False) as arrays:
        ids = arrays["unit_ids"]
        populations = arrays["populations"]
        xy = arrays["positions_mm"]
        senders = arrays["spike_senders"]
        times = arrays["spike_times_s"]
    observer = PeriodicPopulationGrid.from_positions(
        ids, populations, xy, extent_mm=source["extent_mm"],
        cells_per_axis=spec["cells_per_axis"],
        definition="Native Mesocircuit soma positions in mm; disjoint periodic square top-hat cells, separately by source population.")
    if (len(ids) != source["unit_count"] or times.ndim != 1
            or senders.shape != times.shape or len(times) != source["spike_count"]
            or senders.dtype.kind not in "iu" or times.dtype.kind not in "fiu"
            or not np.isfinite(times).all() or np.any(np.diff(times) < 0)
            or not np.isin(senders, ids).all() or np.any(times <= 0)
            or np.any(times > source["simulation_stop_s"])):
        raise ValueError("MESOCIRCUIT_FIELD_NATIVE_EVENT_IDENTITY_OR_TIME_MISMATCH")
    relative, history_bins = window_edges(
        spec["history_seconds"], spec["horizon_seconds"], spec["bin_seconds"])
    absolute = spec["onset_seconds"] + relative
    # Native events are reported in ms; build aligned edges through the same
    # units. This avoids .05 + .1 extending an intended .15-s half-open window.
    dt_ms = manifest["resolved_config"]["simulation"]["sim_resolution"]
    dt_s = dt_ms / 1000
    ticks = absolute / dt_s
    if not np.allclose(ticks, np.rint(ticks), rtol=0, atol=1e-9):
        raise ValueError("MESOCIRCUIT_FIELD_EDGES_MUST_ALIGN_TO_NATIVE_CLOCK")
    absolute = (np.rint(ticks) * dt_ms) / 1000
    if absolute[0] < 0 or absolute[-1] > source["simulation_stop_s"]:
        raise ValueError("MESOCIRCUIT_FIELD_WINDOW_OUTSIDE_SIMULATED_SUPPORT")
    chunk_size = spec["unit_chunk_size"]
    def chunks():
        for start in range(0, len(ids), chunk_size):
            selected = ids[start:start + chunk_size]
            spikes = {int(uid): times[senders == uid] for uid in selected}
            counts = count_spikes_at_edges(spikes, selected, absolute[None, :])
            yield selected, counts, None
    readout = observer.project(chunks(), bin_seconds=spec["bin_seconds"])
    field = readout.field
    inside = (times >= absolute[0]) & (times < absolute[-1])
    if int(field.count_sum.sum()) != int(inside.sum()) or not field.complete.all():
        raise ValueError("MESOCIRCUIT_FIELD_COUNT_OR_COVERAGE_MISMATCH")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "spatial_fields.npz",
        source_unit_ids=ids, source_populations=populations, source_positions_mm=xy,
        unit_field_index=observer.mapping.group_index,
        field_ids=field.field_ids, field_populations=observer.population_labels,
        ix=observer.ix, iy=observer.iy, centers_mm=observer.centers_mm,
        cell_area_mm2=observer.cell_area_mm2, count_sum=field.count_sum,
        observed_unit_count=field.observed_unit_count, total_unit_count=field.total_unit_count,
        observed_rate_hz=field.observed_rate_hz, complete=field.complete,
        density_hz_per_mm2=readout.density_hz_per_mm2,
        absolute_edges_seconds=absolute, relative_edges_seconds=relative,
        history_bin_count=history_bins)
    summary = {
        "status": "PARTIAL", "scope": "native_spatial_observation_engineering",
        "source_unit_count": len(ids), "population_count": len(set(populations)),
        "source_spike_count": len(times), "counted_spikes": int(inside.sum()),
        "outside_window_spikes": int((~inside).sum()),
        "final_right_boundary_spikes": int((times == absolute[-1]).sum()),
        "window_start_s": float(absolute[0]), "window_stop_s": float(absolute[-1]),
        "bin_convention": "[left,right)", "history_bin_count": history_bins,
        "field_shape": list(field.count_sum.shape), "field_count": len(field.field_ids),
        "empty_field_count": int((field.total_unit_count == 0).sum()),
        "count_unit": "spikes_per_bin", "density_unit": "spikes_per_second_per_mm2",
        "observed_rate_unit": "spikes_per_second_per_observed_unit",
        "cell_area_mm2": observer.cell_area_mm2, "extent_mm": observer.extent_mm,
        "mapping_definition": observer.mapping.definition,
        "spatial_kernel": "disjoint square top-hat cells; no smoothing",
        "all_numerical_inputs_complete": bool(field.complete.all()),
        "scientific_reference_qualified": False, "large_scale_fidelity_qualified": False,
        "formal_training_completed": False, "dataset_ready": False,
        "original_unit_trajectories_retained_at": relpath(source_dir.resolve(), output.resolve())
    }
    freeze_json(output / "summary.json", summary)
    freeze_json(output / "manifest.json", {
        "schema": "mesocircuit_spatial_field_check_v1",
        "source_directory": relpath(source_dir.resolve(), output.resolve()),
        "source_manifest_sha256": sha256_file(source_dir / "manifest.json"),
        "artifacts": {name: sha256_file(output / name)
                      for name in ("spatial_fields.npz", "summary.json")},
        **runtime_manifest(config, command, 2)})
    return summary, 2
