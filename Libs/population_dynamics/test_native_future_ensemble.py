"""Small explicit source-format fixtures, not neuroscientific evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

from . import native_future_ensemble as ensemble
from .native_renewal_data import NativeRenewalEpisode, NATIVE_SOURCE_REVISION
from .provenance import sha256_file
from .spatial_activity import Activity, analyze as activity
from .trajectory_sample_metrics import unit_count_metrics
from .trajectory_scores import energy_score


def _json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Only the scale constants change. Real native loading, projection,
    # serialization and scoring paths are exercised with32 identified units.
    monkeypatch.setattr(ensemble, "CORE_UNITS", 32)
    monkeypatch.setattr(ensemble, "FIELD_COUNT", 32)
    graph = tmp_path / "graph"
    graph.mkdir()
    manifest = dict(source_config=dict(simulation=dict(t_presim=0.)),
                    resolved_source_network=dict(extent=1., populations=["P0", "P1", "TC"],
                                                 neuron_params=dict(t_ref=2.)))
    _json(graph / "manifest.json", manifest)
    ids = np.arange(1, 34, dtype=np.int64)
    populations = np.array(["P0"] * 16 + ["P1"] * 16 + ["TC"])
    cell = np.array([(x, y) for y in (-.375, -.125, .125, .375) for x in (-.375, -.125, .125, .375)])
    coordinates = np.vstack((cell, cell, [[0., 0.]]))
    base_ticks = np.array([100, 9000, 10000, 10010, 12000], dtype=np.int64)
    base_units = np.array([1, 33, 2, 3, 33], dtype=np.int64)
    states = dict(unit_ids=ids[:32], **{name: np.zeros(32, dtype=np.float32) for name in
                                     (*ensemble.STATE_FIELDS, "P11ex", "P11in")})
    states["P11ex"][:] = .9
    states["P11in"][:] = .8
    qualified = tmp_path / "qualified_boundary"
    qualified.mkdir()
    for tick in (10000, 10015):
        np.savez(qualified / f"state_observation_{tick:09d}.npz", **states)

    def episode(path, draw=None):
        path.mkdir()
        ticks, units = base_ticks.copy(), base_units.copy()
        if draw is not None:
            ticks = np.append(ticks, 10100 + draw * 50)
            units = np.append(units, 4 + draw % 28)
        order = np.lexsort((units, ticks))
        ticks, units = ticks[order], units[order]
        np.savez(path / "initial_parameters.npz", unit_ids=ids[:32],
                 V_reset_rel=np.full(32, -65., dtype=np.float32),
                 Theta_rel=np.full(32, -50., dtype=np.float32), t_ref=np.full(32, 2., dtype=np.float32))
        np.savez_compressed(path / "spikes.npz", unit_ids=ids, populations=populations,
            positions_mm=coordinates, spike_unit_ids=units, spike_ticks=ticks,
            raw_spike_times_ms=ticks * .1, dt_ms=.1)
        counts = np.bincount(units - 1, minlength=len(ids)).astype(np.int64)
        np.savez(path / "spike_counts.npz", unit_ids=ids, native_counts=counts, recorded_counts=counts)
        report = dict(status="EXECUTED", duration_ms=6000., cortical_neurons=32, dt_ms=.1,
            gpu_source_revision=NATIVE_SOURCE_REVISION, graph_manifest_sha256=sha256_file(graph / "manifest.json"),
            native_library_sha256=("a" if draw is None else "b") * 64,
            native_poisson_rng_family="per_edge_philox4_32_10", checks=dict(spike_count_recording_complete=True),
            config={}, elapsed_seconds=1., import_seconds=.1, simulation_seconds=.2, calibration_seconds=.1)
        if draw is not None:
            seeds = list(range(87000 + draw * 8, 87008 + draw * 8))
            report["config"]["future_background_noise"] = dict(boundary_time_ms=1000., seeds_per_population=seeds)
            report["future_background_noise"] = [dict(source_node=index + 100, future_seed=seed,
                outgoing_edges_verified=4, boundary_tick=10000, reset_tick=10015,
                native_delay_ticks=17, committed_emissions_through_tick=10000,
                last_committed_arrival_tick=10015, first_resampled_arrival_tick=10016,
                state_and_queues_reinitialized=False) for index, seed in enumerate(seeds)]
            for tick in (10000, 10015):
                np.savez(path / f"state_observation_{tick:09d}.npz", **states)
        report["artifacts"] = {file.name: sha256_file(file) for file in path.iterdir() if file.suffix == ".npz"}
        _json(path / "report.json", report)
        return path

    reference = episode(tmp_path / "reference")
    fields = tmp_path / "reference_fields"
    assert activity(graph, reference, fields, Activity(bin_ms=5., cells_per_axis=4, start_ms=0.))["status"] == "DESCRIBED"
    native = NativeRenewalEpisode(reference, fields)
    scale = tmp_path / "scale.npz"
    np.savez(scale, mean_hz=np.zeros(32), standard_deviation_hz=np.ones(32),
             unit_count=np.bincount(native.unit_field_index, minlength=32), field_ids=native.field_ids)
    config = tmp_path / "config.yaml"
    actual_config = yaml.safe_load((Path(__file__).parents[1] / "configs/neural_population_matched_forecast.yaml").read_text())
    config.write_text(yaml.safe_dump(actual_config))
    branches = [episode(tmp_path / f"draw_{draw:02d}", draw) for draw in range(16)]
    return dict(graph=graph, reference_episode=reference, reference_fields=fields, branches=branches,
                fitting_scale_path=scale, config_path=config, committed_state_reference=qualified,
                output=tmp_path / "analysis")


def _resign_native(path):
    report = json.loads((path / "report.json").read_text())
    report["artifacts"] = {file.name: sha256_file(file) for file in path.iterdir() if file.suffix == ".npz"}
    _json(path / "report.json", report)


def test_complete_ensemble_real_loading_projection_metrics_and_full_persistence(fixture):
    result = ensemble.analyze_future_ensemble(**fixture)
    assert result["status"] == "DESCRIBED_FIXED_MICROSTATE_DEVELOPMENT"
    assert result["scope"] == "fixed_microstate_conditional_development"
    assert (result["draws"], result["generated_bins"], result["field_count"], result["score_rows"]) == (16, 100, 32, 3)
    assert result["generated_unit_count_values"] == 16 * 100 * 32
    assert result["distinct_future_seed_vectors"] == 16
    assert all(result[name] is False for name in ("scientific_claim_qualified", "matched_fidelity", "same_hardware",
                                                 "end_to_end", "speedup", "training_performed"))
    assert result["reference_source_identity"]["native_library_sha256"] == "a" * 64
    assert result["branch_source_identity"]["native_library_sha256"] == "b" * 64
    output = fixture["output"]
    samples = np.load(output / "sampled_unit_counts.npy", mmap_mode="r", allow_pickle=False)
    assert samples.shape == (16, 100, 32) and samples.dtype == np.int64
    assert samples.sum() == 16 * 2 + 16
    native = NativeRenewalEpisode(fixture["branches"][0], output / "branch_fields/draw_00")
    values, _ = native.window(10000, 15000)
    np.testing.assert_array_equal(samples[0], values)
    with np.load(output / "ensemble_fields.npz", allow_pickle=False) as arrays:
        assert arrays["sampled_field_counts"].shape == (16, 100, 32)
        np.testing.assert_array_equal(arrays["sampled_field_counts"].sum(-1), samples.sum(-1))
        expected = unit_count_metrics(torch.as_tensor(np.array(samples)), torch.as_tensor(arrays["observed_unit_counts"]),
                                      native.minimum_interval_ticks, 50, [1, 2, 4])
        standardized = arrays["sampled_field_counts"] / .005
        truth = arrays["observed_field_counts"] / .005
        expected_energy = energy_score(standardized, truth)
    with (output / "scores.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["horizon_ms"]) for row in rows] == [20., 80., 500.]
    assert float(rows[-1]["unit_bin_mean_mse_corrected"]) == expected["unit_bin_mean_mse_corrected"]
    assert float(rows[-1]["energy_score"]) == expected_energy
    assert rows[0]["variogram_temporal_lag_4"] == ""
    assert all(np.isfinite(value) and value >= 0 for name, value in result.items() if name.endswith("_seconds"))
    assert result["tensor_device"] == "cpu" and result["tensor_dtype"] == "torch.int64"
    assert result["peak_process_rss_bytes"] > 0
    assert all(sha256_file(output / name) == digest for name, digest in result["artifacts"].items())


def test_cli_uses_explicit_branch_list_and_committed_state_reference(fixture, monkeypatch, capsys):
    branch_list = fixture["output"].parent / "branches.json"
    _json(branch_list, [str(path) for path in fixture["branches"]])
    arguments = ["native_future_ensemble"]
    for flag, name in (("graph", "graph"), ("reference-episode", "reference_episode"),
                       ("reference-fields", "reference_fields"), ("fitting-scale", "fitting_scale_path"),
                       ("config", "config_path"), ("output", "output"),
                       ("committed-state-reference", "committed_state_reference")):
        arguments.extend(["--" + flag, str(fixture[name])])
    arguments.extend(["--branches", str(branch_list)])
    monkeypatch.setattr(sys, "argv", arguments)
    ensemble.main()
    report = json.loads((fixture["output"] / "analysis.json").read_text())
    assert report["draws"] == 16 and report["full_unit_counts_saved"] is True
    assert '"status": "DESCRIBED_FIXED_MICROSTATE_DEVELOPMENT"' in capsys.readouterr().out


def _archived_ensemble(fixture):
    root = fixture["output"].parent / "delivered"
    root.mkdir()
    for name, relative in (("reference_episode", "reference"), ("reference_fields", "reference_fields"),
                           ("committed_state_reference", "qualified_boundary")):
        destination = root / relative
        fixture[name].rename(destination)
        fixture[name] = destination
    for index, branch in enumerate(fixture["branches"]):
        destination = root / "executions" / f"draw_{index:02d}" / "native"
        destination.parent.mkdir(parents=True)
        branch.rename(destination)
        fixture["branches"][index] = destination
    fixture["output"] = root / "analysis"
    ensemble.analyze_future_ensemble(**fixture)
    executions = [dict(draw=index, exit_code=0, seeds_per_population=list(range(87000 + index * 8, 87008 + index * 8)),
                       process_seconds=2., native_function_seconds=1., graph_import_seconds=.1,
                       simulation_seconds=.2, calibration_seconds=.1) for index in range(16)]
    proof = dict(status="PASS", exit_code=0, complete_native_executions=16,
                 generated_unit_count_values=16 * 100 * 32, full_unit_counts_saved=True,
                 native_executions=executions, native_process_seconds=32., native_function_seconds=16.,
                 simulated_biological_ms=96000., target_output_biological_ms=8000.,
                 pipeline_before_archive_seconds=50., archive_seconds=5., task_through_archive_seconds=55.)
    _archive_manifest(root, proof)
    return root, proof


def _archive_manifest(root, proof):
    proof["files"] = [dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha256_file(path))
                      for path in sorted(root.rglob("*")) if path.is_file() and path.name != "delivery.json"]
    _json(root / "delivery.json", proof)


def test_archived_replay_checks_all_native_counts_fields_scores_and_cost(fixture):
    root, proof = _archived_ensemble(fixture)
    output = root.parent / "replay.json"
    result = ensemble.replay_delivery(root, fixture["fitting_scale_path"], output)
    assert result["status"] == "PASS"
    assert result["exact_unit_count_values"] == 16 * 100 * 32
    assert result["exact_field_count_values"] == 16 * 100 * 32
    assert result["distinct_future_seeds"] == 128
    assert result["cost_denominator_verified"] is True
    assert result["archive_files_verified"] == len(proof["files"])
    assert result["simulation_repeated"] is False
    assert json.loads(output.read_text()) == result
    with pytest.raises(ValueError, match="new replay output"):
        ensemble.replay_delivery(root, fixture["fitting_scale_path"], root / "replay.json")


def test_replay_cli_uses_archived_inputs_and_requires_scale(fixture, monkeypatch, capsys):
    root, _ = _archived_ensemble(fixture)
    output = root.parent / "cli_replay.json"
    monkeypatch.setattr(sys, "argv", ["native_future_ensemble", "--replay-delivery", str(root),
                                     "--fitting-scale", str(fixture["fitting_scale_path"]), "--output", str(output)])
    ensemble.main()
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    monkeypatch.setattr(sys, "argv", ["native_future_ensemble", "--replay-delivery", str(root), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        ensemble.main()
    assert error.value.code == 2


@pytest.mark.parametrize("change", ["counts", "score", "cost", "source_cost", "missing_draw", "archive"])
def test_archived_replay_rejects_corrupt_or_numerically_changed_delivery(fixture, change):
    root, proof = _archived_ensemble(fixture)
    if change == "counts":
        path = root / "analysis/sampled_unit_counts.npy"
        data = np.load(path, mmap_mode="r+")
        data[0, 0, 0] += 1
        data.flush()
        del data
    elif change == "score":
        path = root / "analysis/scores.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["energy_score"] = str(float(rows[0]["energy_score"]) + 1.)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    elif change == "cost":
        proof["native_process_seconds"] += 1.
    elif change == "source_cost":
        proof["native_executions"][0]["graph_import_seconds"] += .01
    elif change == "missing_draw":
        proof["native_executions"].pop()
    else:
        (root / "analysis/scores.csv").write_text("changed")
    if change != "archive":
        _archive_manifest(root, proof)
    output = root.parent / "failed_replay.json"
    with pytest.raises(ValueError):
        ensemble.replay_delivery(root, fixture["fitting_scale_path"], output)
    assert json.loads(output.read_text())["status"] == "FAILED"


@pytest.mark.parametrize("failure", ["missing_branch", "same_branch", "failed", "duplicate_seed", "reused_seed",
    "missing_reset", "reset_tick", "reset_edges", "source_graph", "same_library", "mixed_library",
    "initial_parameters", "committed_state", "missing_state", "committed_events", "tc_control", "scale", "protocol"])
def test_incomplete_or_changed_native_evidence_rejected(fixture, failure):
    branch = fixture["branches"][0]
    report_path = branch / "report.json"
    report = json.loads(report_path.read_text())
    if failure == "missing_branch": fixture["branches"].pop()
    elif failure == "same_branch": fixture["branches"][-1] = branch
    elif failure == "failed": report["status"] = "FAILED"
    elif failure in ("duplicate_seed", "reused_seed"):
        later = fixture["branches"][1]
        other = json.loads((later / "report.json").read_text())
        seeds = report["config"]["future_background_noise"]["seeds_per_population"]
        if failure == "reused_seed": seeds = [seeds[0]] + list(range(89001, 89008))
        other["config"]["future_background_noise"]["seeds_per_population"] = seeds
        for row, seed in zip(other["future_background_noise"], seeds): row["future_seed"] = seed
        _json(later / "report.json", other)
    elif failure == "missing_reset": report.pop("future_background_noise")
    elif failure == "reset_tick": report["future_background_noise"][0]["reset_tick"] += 1
    elif failure == "reset_edges": report["future_background_noise"][0]["outgoing_edges_verified"] -= 1
    elif failure == "source_graph": report["graph_manifest_sha256"] = "f" * 64
    elif failure == "same_library": report["native_library_sha256"] = "a" * 64
    elif failure == "mixed_library":
        later = fixture["branches"][1]
        other = json.loads((later / "report.json").read_text())
        other["native_library_sha256"] = "c" * 64
        _json(later / "report.json", other)
    elif failure in ("initial_parameters", "committed_state", "committed_events", "tc_control"):
        name = ("initial_parameters.npz" if failure == "initial_parameters" else
                "state_observation_000010000.npz" if failure == "committed_state" else "spikes.npz")
        with np.load(branch / name, allow_pickle=False) as arrays: values = {key: arrays[key] for key in arrays.files}
        if failure == "initial_parameters": values["t_ref"][0] = 3.
        elif failure == "committed_state": values["V_m_rel"][0] = 1.
        elif failure == "committed_events": values["spike_ticks"][0] += 1
        else: values["spike_ticks"][values["spike_unit_ids"] == 33] += 1
        np.savez_compressed(branch / name, **values)
        _resign_native(branch)
        report = json.loads(report_path.read_text())
    elif failure == "missing_state": (branch / "state_observation_000010015.npz").unlink()
    elif failure == "scale":
        with np.load(fixture["fitting_scale_path"]) as arrays: values = {key: arrays[key] for key in arrays.files}
        values["unit_count"][0] = 2
        np.savez(fixture["fitting_scale_path"], **values)
    elif failure == "protocol":
        config = yaml.safe_load(fixture["config_path"].read_text())
        config["joint_samples"] = 8
        fixture["config_path"].write_text(yaml.safe_dump(config))
    _json(report_path, report)
    with pytest.raises((ValueError, FileNotFoundError)):
        ensemble.analyze_future_ensemble(**fixture)
    if (fixture["output"] / "analysis.json").exists():
        result = json.loads((fixture["output"] / "analysis.json").read_text())
        assert result["status"] == "FAILED"
        assert result["scientific_claim_qualified"] is False
        assert result["partial_unit_counts_authoritative"] is False
