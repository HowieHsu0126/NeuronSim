"""Source identity, denominator and shared-window-consumer integration checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from .__main__ import run, parser
from .baseline_data import load_engineering_windows
from .config import load_config
from .population_field_check import check_population_field
from .provenance import sha256_file

CONFIG = Path(__file__).resolve().parents[1] / "configs/neural_population_dynamics.yaml"


def fixture(tmp_path):
    config = load_config(CONFIG)
    source, windows = tmp_path / "source", tmp_path / "windows"
    source.mkdir(); windows.mkdir()
    np.savez(source / "source_tables.npz", **{
        "units/id": np.array([4, 7, 9, 11]),
        "units/brain_region": np.array(["A", "B", "A", "C"])})
    sm = {"subject_id": "655565", "future_split_requirement": "development_training_only",
          "artifacts": {"source_tables.npz": sha256_file(source / "source_tables.npz")}}
    (source / "manifest.json").write_text(json.dumps(sm))
    counts = np.array([[[2, 0], [0, 4], [1, 2], [5, 0]]])
    np.savez(windows / "windows.npz", counts=counts, unit_ids=[7, 4], history_bin_count=2,
             relative_edges_seconds=np.array([-.002, -.001, 0, .001, .002]),
             stimulus_features=np.ones((1, 4, 1)))
    (windows / "summary.json").write_text(json.dumps({"subject_id": "655565", "dataset_ready": False,
                                                     "selected_trial_ids": [3]}))
    (windows / "stimulus_features.json").write_text(json.dumps([{"feature_index": 0,
                                                               "feature_kind": "unannotated_fraction"}]))
    wm = {"schema": "opto_complete_control_windows_v1", "subject_id": "655565",
          "source_manifest_sha256": sha256_file(source / "manifest.json"),
          "resolved_config": config,
          "artifacts": {name: sha256_file(windows / name) for name in
                        ("windows.npz", "summary.json", "stimulus_features.json")}}
    (windows / "manifest.json").write_text(json.dumps(wm))
    return source, windows, config, counts


def test_real_cli_contract_keeps_missing_source_units_and_units_of_rate(tmp_path):
    source, windows, config, counts = fixture(tmp_path)
    args = parser().parse_args(["--config", str(CONFIG), "--project-root", str(tmp_path),
                               "population-field-check", "--source-dir", str(source),
                               "--window-dir", str(windows), "--output-dir", str(tmp_path / "output")])
    summary, exit_code = run(args, ["test-cli"])
    assert exit_code == 2 and summary["dataset_ready"] is False
    pointer = Path(summary["original_unit_trajectories_retained_at"])
    assert not pointer.is_absolute()
    assert (tmp_path / "output" / pointer).resolve() == windows.resolve()
    assert summary["source_unit_count"] == 4 and summary["supplied_unit_count"] == 2
    with np.load(tmp_path / "output/population_fields.npz", allow_pickle=False) as out:
        np.testing.assert_array_equal(out["count_sum"][..., 0], counts[..., 1])
        np.testing.assert_array_equal(out["count_sum"][..., 1], counts[..., 0])
        np.testing.assert_array_equal(out["total_unit_count"], [2, 1, 1])
        np.testing.assert_array_equal(out["observed_rate_hz"][..., 0], counts[..., 1] * 1000)
        assert np.isnan(out["observed_rate_hz"][..., 2]).all()
        assert not out["complete"][..., 0].any()
        assert out["complete"][..., 1].all()
        assert not out["complete"][..., 2].any()
    assert json.loads((tmp_path / "output/manifest.json").read_text())["exit_code"] == 2


def test_shared_loader_retains_control_vocabulary_and_causal_boundary(tmp_path):
    _, windows, config, counts = fixture(tmp_path)
    data = load_engineering_windows(windows, config)
    np.testing.assert_array_equal(data.counts, counts)
    assert data.unit_ids == (7, 4) and data.history_bins == 2 and data.trial_ids == ("3",)
    assert data.input_features["vocabulary_scope"] == "fixed_training_anchor_control_catalog"
    assert data.inputs.shape == (1, 4, 1)


def test_source_mismatch_and_changed_control_metadata_reject_before_output(tmp_path):
    source, windows, config, _ = fixture(tmp_path)
    source_manifest = source / "manifest.json"
    data = json.loads(source_manifest.read_text()); data["subject_id"] = "different"
    source_manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="IDENTITY_MISMATCH"):
        check_population_field(source, windows, tmp_path / "rejected", config, [])
    assert not (tmp_path / "rejected").exists()
    (windows / "stimulus_features.json").write_text("[]")
    with pytest.raises(ValueError, match="DIGEST_MISMATCH"):
        load_engineering_windows(windows, config)


@pytest.mark.parametrize("field", [{"group_by": "soma_position", "unit_chunk_size": 2},
                                   {"group_by": "brain_region", "unit_chunk_size": True},
                                   {"group_by": "brain_region", "unit_chunk_size": 0}])
def test_unimplemented_space_and_invalid_chunk_sizes_reject(tmp_path, field):
    config = yaml.safe_load(CONFIG.read_text()); config["population_field_check"] = field
    path = tmp_path / "config.yaml"; path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="POPULATION_FIELD"):
        load_config(path)
