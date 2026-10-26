"""Source-indexed population readout on a bounded Opto engineering fixture."""
from __future__ import annotations

import json
from os.path import relpath
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline_data import load_engineering_windows
from .population_field import PopulationFieldMapping
from .provenance import freeze_json, runtime_manifest, sha256_file


def check_population_field(source_dir: Path, window_dir: Path, output: Path,
                           config: dict[str, Any], command: list[str]) -> tuple[dict[str, Any], int]:
    """Check the observation map, preserving the complete source-unit denominator."""
    if output.exists():
        raise FileExistsError(f"POPULATION_FIELD_OUTPUT_EXISTS:{output}")
    source = json.loads((source_dir / "manifest.json").read_text())
    window_manifest = json.loads((window_dir / "manifest.json").read_text())
    if (window_manifest.get("schema") != "opto_complete_control_windows_v1"
            or window_manifest.get("source_manifest_sha256") != sha256_file(source_dir / "manifest.json")
            or source.get("subject_id") != config["opto_source"]["engineering_anchor_subject_id"]
            or source.get("future_split_requirement") != "development_training_only"):
        raise ValueError("FIELD_SOURCE_AND_WINDOW_IDENTITY_MISMATCH")
    if sha256_file(source_dir / "source_tables.npz") != source["artifacts"]["source_tables.npz"]:
        raise ValueError("FIELD_SOURCE_TABLE_CHANGED")
    spec = config["population_field_check"]
    with np.load(source_dir / "source_tables.npz", allow_pickle=False) as data:
        source_ids = data["units/id"]
        groups = data["units/" + spec["group_by"]]
    mapping = PopulationFieldMapping.from_groups(
        source_ids, groups,
        definition="Original Opto units/" + spec["group_by"] + "; source labels only, no spatial kernel or soma coordinates.")
    windows = load_engineering_windows(window_dir, config)
    chunk_size = spec["unit_chunk_size"]
    readout = mapping.project(((windows.unit_ids[i:i + chunk_size],
                               windows.counts[..., i:i + chunk_size], None)
                              for i in range(0, len(windows.unit_ids), chunk_size)),
                             bin_seconds=windows.bin_seconds)
    # Direct grouped counts provide a second formulation for numerical replay.
    source_labels = dict(zip(map(int, source_ids), groups.tolist()))
    reference = np.stack([windows.counts[..., [j for j, uid in enumerate(windows.unit_ids)
                         if source_labels[uid] == group]].sum(axis=-1)
                         for group in readout.field_ids], axis=-1)
    if not np.array_equal(reference, readout.count_sum):
        raise ValueError("FIELD_DIRECT_SUM_REPLAY_MISMATCH")
    if not np.array_equal(readout.count_sum.sum(axis=-1), windows.counts.sum(axis=-1)):
        raise ValueError("FIELD_UNIT_COUNT_CONSERVATION_MISMATCH")
    output.mkdir(parents=True)
    np.savez_compressed(output / "population_fields.npz", count_sum=readout.count_sum,
                        observed_unit_count=readout.observed_unit_count,
                        total_unit_count=readout.total_unit_count,
                        observed_rate_hz=readout.observed_rate_hz, complete=readout.complete,
                        field_ids=np.asarray(readout.field_ids, dtype=str),
                        supplied_unit_ids=readout.supplied_unit_ids,
                        history_bin_count=windows.history_bins)
    pd.DataFrame({"unit_id": source_ids, "field_label": groups}).to_csv(output / "source_unit_map.csv", index=False)
    summary = {"status": "PARTIAL", "scope": "source_indexed_observation_map_engineering",
               "subject_id": source["subject_id"], "source_unit_count": len(source_ids),
               "supplied_unit_count": len(windows.unit_ids), "supplied_unit_ids": list(windows.unit_ids),
               "field_ids": list(readout.field_ids), "field_count": len(readout.field_ids),
               "field_shape": list(readout.count_sum.shape), "window_count": len(windows.trial_ids),
               "total_counted_spikes_in_overlapping_windows": int(readout.count_sum.sum()),
               "source_group_field": spec["group_by"], "mapping_definition": mapping.definition,
               "count_unit": "spikes_per_bin", "rate_unit": "spikes_per_second_per_observed_unit",
               "bin_seconds": windows.bin_seconds, "unit_chunk_size": chunk_size,
               "input_complete_elements": int(readout.complete.sum()),
               "input_total_elements": readout.complete.size,
               "missing_unit_policy": "Absent units retain source denominator; zero-observation rates are NaN, not silence.",
               "coverage_note": "complete describes supplied numerical inputs only; recording support remains unqualified.",
               "field_observation_map_qualified": False, "spatial_field_qualified": False,
               "recording_coverage_qualified": False, "dataset_ready": False,
               "formal_training_completed": False, "original_unit_trajectories_retained_at": relpath(window_dir.resolve(), output.resolve())}
    freeze_json(output / "summary.json", summary)
    names = ("population_fields.npz", "source_unit_map.csv", "summary.json")
    freeze_json(output / "manifest.json", {"schema": "source_indexed_population_field_check_v1",
                "source_manifest_sha256": sha256_file(source_dir / "manifest.json"),
                "window_manifest_sha256": sha256_file(window_dir / "manifest.json"),
                "mapping_implementation_sha256": sha256_file(Path(__file__).with_name("population_field.py")),
                "artifacts": {name: sha256_file(output / name) for name in names},
                **runtime_manifest(config, command, 2)})
    return summary, 2
