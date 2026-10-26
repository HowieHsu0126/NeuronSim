"""Source provenance, native clock boundaries and actual field CLI contract."""
import json
from pathlib import Path

import numpy as np
import pytest

from .mesocircuit_field import check_mesocircuit_field
from .provenance import freeze_json, sha256_file
from .spike_windows import count_spikes_at_edges


def source_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    np.savez_compressed(source / "spikes.npz",
        unit_ids=np.array([2, 11, 4]), populations=np.array(["E", "I", "E"]),
        positions_mm=np.array([[-.4, -.4], [.4, .4], [.1, -.1]]),
        spike_senders=np.array([2, 11, 2, 11]), spike_times_s=np.array([.01, .05, .149, .15]))
    freeze_json(source / "summary.json", {
        "scope": "engineering_diagnostic", "boundary": "periodic square",
        "extent_mm": 1., "unit_count": 3, "spike_count": 4, "simulation_stop_s": .15})
    freeze_json(source / "manifest.json", {
        "schema": "mesocircuit_spatial_reference_v1",
        "resolved_config": {"simulation": {"sim_resolution": .1}},
        "artifacts": {name: sha256_file(source / name) for name in ("spikes.npz", "summary.json")}})
    config = {"mesocircuit_field_check": dict(onset_seconds=.05, history_seconds=.05,
        horizon_seconds=.1, bin_seconds=.001, cells_per_axis=2, unit_chunk_size=1)}
    return source, config


def test_native_clock_right_edge_excluded_and_history_boundary_is_response(tmp_path):
    source, config = source_fixture(tmp_path)
    output = tmp_path / "readout"
    summary, code = check_mesocircuit_field(source, output, config, ["test"])
    assert code == 2 and summary["status"] == "PARTIAL"
    assert summary["source_unit_count"] == 3 and summary["counted_spikes"] == 3
    assert summary["final_right_boundary_spikes"] == 1
    assert summary["window_stop_s"] == .15 and not summary["dataset_ready"]
    with np.load(output / "spatial_fields.npz", allow_pickle=False) as field:
        assert field["count_sum"][:, :50].sum() == 1
        assert field["count_sum"][:, 50:].sum() == 2
        assert field["count_sum"].shape == (1, 150, 8)
        assert field["total_unit_count"].sum() == 3
        assert field["complete"].all()
        np.testing.assert_allclose(field["density_hz_per_mm2"] * .001 * .25, field["count_sum"])
        empty = field["total_unit_count"] == 0
        assert (field["density_hz_per_mm2"][..., empty] == 0).all()
        assert np.isnan(field["observed_rate_hz"][..., empty]).all()
    manifest = json.loads((output / "manifest.json").read_text())
    assert (output / manifest["source_directory"]).resolve() == source.resolve()


def test_source_digest_and_existing_output_are_rejected(tmp_path):
    source, config = source_fixture(tmp_path)
    output = tmp_path / "readout"
    output.mkdir()
    with pytest.raises(FileExistsError, match="OUTPUT_EXISTS"):
        check_mesocircuit_field(source, output, config, [])
    with (source / "spikes.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="DIGEST_MISMATCH"):
        check_mesocircuit_field(source, tmp_path / "fresh", config, [])
    assert not (tmp_path / "fresh").exists()


@pytest.mark.parametrize("override", [
    {"onset_seconds": .2}, {"bin_seconds": .00015}, {"unit_chunk_size": 0},
    {"cells_per_axis": True}, {"history_seconds": 0}, {"extra": 1}
])
def test_invalid_window_or_grid_rejected_before_writing(tmp_path, override):
    source, config = source_fixture(tmp_path)
    config["mesocircuit_field_check"].update(override)
    with pytest.raises(ValueError):
        check_mesocircuit_field(source, tmp_path / "bad", config, [])
    assert not (tmp_path / "bad").exists()


def test_supplied_edges_count_boundaries_and_silent_units():
    counts = count_spikes_at_edges({4: np.array([0., .05, .1, .15]), 2: np.array([])},
                                  [2, 4], np.array([[0., .05, .1, .15]]))
    np.testing.assert_array_equal(counts, [[[0, 1], [0, 1], [0, 1]]])


@pytest.mark.parametrize("edges", [np.array([0., 1.]), np.array([[1.]]),
                                  np.array([[0., 0.]]), np.array([[0., np.nan]])])
def test_invalid_supplied_edges_rejected(edges):
    with pytest.raises(ValueError):
        count_spikes_at_edges({1: np.array([])}, [1], edges)
