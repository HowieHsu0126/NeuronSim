"""Fitting-only PFC reference arithmetic and source identity contracts."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from . import pfc_intervention_references as factory
from .benchmark_tasks import pfc_working_memory_config, task_protocol
from .circuit_data import CircuitArchiveWriter, CircuitDataset, SCHEMA
from .pfc_kernel_parameters import circuit_provenance
from .provenance import canonical_json, sha256_file
from .test_circuit_data import binding, episode, units


def field_counts(identity):
    circuit, block, condition = identity
    result = np.zeros((980, 17), dtype=np.int64)
    if identity != (23, 0, 0):
        result[:, 0] = condition + block + 1
        result[:, 1] = (np.arange(980) % 3) * (circuit % 10 + 1)
    return result


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    monkeypatch.delenv("NEURONSIM_EXECUTION_PATH_MAP", raising=False)
    config_path = Path(__file__).resolve().parents[1] / "configs/neural_population_pfc_intervention_reference.yaml"
    config = yaml.safe_load(config_path.read_text())
    source_entry = Path(__file__).resolve().parents[1] / (
        "external/pfc_working_memory/neurodynex3/working_memory_network/wm_model.py")
    spec = dict(schema="pfc_population_development_inputs_v1", source_entry=str(source_entry), episodes=[])
    base = dict(scope="source_reproduction", expected_source_revision=factory.SOURCE_REVISION,
                excitatory_neurons=80000, inhibitory_neurons=20000, weight_scaling_factor=.0256,
                excitatory_preference_bins=16, readout_bin_ms=5., state_monitored_per_population=8,
                synapse_execution="factorized_uniform_projections", sigma_weight_profile=30.)
    index_by_root, metadata_by_circuit = {}, {}
    for circuit, blocks in config["fitting_circuit_blocks"].items():
        for block in blocks:
            for condition in config["condition_indices"]:
                alias = f"pfc_working_memory_circuit_{circuit:03d}"
                block_id = f"{alias}_block_{block:03d}"
                name = f"{block_id}_condition_{condition:02d}"
                identity = dict(scenario_id="pfc_working_memory", sample_unit="complete_mechanistic_episode",
                    circuit_id=alias, circuit_index=circuit, block_id=block_id, block_index=block,
                    episode_id=name, condition_index=condition, split="train", connectivity_seed=200000000 + circuit,
                    initial_state_seed=210000000 + 1000 * circuit + block,
                    background_seed=220000000 + 1000 * circuit + block,
                    stimulus_seed=230000000 + 1000 * circuit + block)
                source_config = pfc_working_memory_config(
                    dict(base, Jpos_excit2excit={23: 2.3, 52: 2., 63: 2.5}[circuit]),
                    task_protocol("pfc_working_memory"), identity)
                provenance = circuit_provenance(source_config, factory.SOURCE_ENTRY_SHA256)
                native_circuit = "pfc_" + provenance["parameter_identity"][:16]
                root = tmp_path / name / "dataset"
                source = tmp_path / name / "archive/source_run"
                root.mkdir(parents=True)
                source.mkdir(parents=True)
                (source / "config.yaml").write_text(yaml.safe_dump(source_config))
                config_sha = sha256_file(source / "config.yaml")
                report = dict(case_id=name, resolved_config=source_config, config_sha256=config_sha)
                (source / "report.json").write_text(canonical_json(report))
                report_sha = sha256_file(source / "report.json")
                delivery = dict(case_id=name, status="ARCHIVED_PFC_SOURCE_EXECUTION",
                    result_status="PFC_SOURCE_EXECUTION_COMPLETE", source_execution_complete=True,
                    files=[dict(path="source_run/report.json", sha256=report_sha,
                                bytes=(source / "report.json").stat().st_size)])
                (source.parent / "delivery.json").write_text(canonical_json(delivery))
                episode_row = dict(episode_id=name, circuit_id=native_circuit,
                    block_id=f"pfc_initial_{identity['initial_state_seed']}_external_{identity['background_seed']}",
                    split="development", dt_ms=.05, duration_ticks=100000, event_count=int(field_counts((circuit, block, condition)).sum()))
                catalog = dict(binding=dict(schema=SCHEMA, scope="source_reproduction",
                    scenario_id="pfc_working_memory", sample_unit="complete_mechanistic_episode",
                    source_revision=factory.SOURCE_REVISION, source_entry_sha256=factory.SOURCE_ENTRY_SHA256,
                    resolved_source_config=source_config, config_sha256=config_sha,
                    source_bindings=[dict(case_id=name, path=str(source),
                        delivery_sha256=sha256_file(source.parent / "delivery.json"), report_sha256=report_sha)]),
                    circuits={native_circuit: dict(units=100000, path="units.h5", sha256="a" * 64,
                                                   provenance=provenance)},
                    shards=[dict(path="events.h5", sha256="b" * 64, episodes=[episode_row])], complete_episodes=1)
                (root / "dataset.json").write_text(canonical_json(catalog))
                spec["episodes"].append(dict(dataset=str(root), source_run=str(source), episode_id=name,
                                            partition="development_fit"))
                index_by_root[root] = (circuit, block, condition)
                if native_circuit not in metadata_by_circuit:
                    fields = np.r_[np.arange(80000) * 16 // 80000, np.full(20000, 16)]
                    metadata_by_circuit[native_circuit] = dict(
                        unit_ids=np.arange(100000), unit_field_index=fields, field_names=factory._field_names(),
                        populations=np.where(fields == 16, "I", "E"), source_revision=factory.SOURCE_REVISION,
                        circuit_provenance=provenance,
                        coordinate_semantics="E angle=360*source_index/N_E degrees; I has no preference coordinate")
    # These paths deliberately do not exist. Their payloads must never be resolved.
    for block in range(3):
        for condition in range(10):
            spec["episodes"].append(dict(dataset=str(tmp_path / "validation_not_opened"),
                source_run=str(tmp_path / "validation_source_not_opened"),
                episode_id=f"pfc_working_memory_circuit_074_block_{block:03d}_condition_{condition:02d}",
                partition="development_validation"))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(canonical_json(spec))
    opened, reads = [], []
    class Dataset(CircuitDataset):
        def __init__(self, root, *, splits):
            assert Path(root) in index_by_root
            opened.append(Path(root))
            super().__init__(root, splits=splits)
        def circuit_metadata(self, circuit_id):
            return metadata_by_circuit[circuit_id]
        def _read(self, name, start, stop, bin_ticks, unit_start, unit_stop):
            assert (start, stop, bin_ticks, unit_start, unit_stop) == (2000, 100000, 100, 0, 0)
            assert len(opened) == 80  # No fitting outcome before complete metadata admission.
            reads.append(name)
            row = self.episodes[name]
            return dict(episode_id=name, circuit_id=row["circuit_id"], counts=np.empty((980, 0), dtype=np.int64),
                        field_counts=field_counts(index_by_root[self.root]), unit_ids=np.empty(0, dtype=np.int64),
                        field_names=factory._field_names(), bin_seconds=.005)
    monkeypatch.setattr(factory, "CircuitDataset", Dataset)
    return dict(inputs=inputs, config=config_path, output=tmp_path / "prepared", spec=spec,
                opened=opened, reads=reads, metadata=metadata_by_circuit, root=tmp_path)


def prepare(cohort):
    return factory.prepare_references(cohort["inputs"], cohort["config"], cohort["output"])


def test_complete_fitting_denominator_rms_quantum_silence_and_roundtrip(cohort):
    report = prepare(cohort)
    assert len(cohort["reads"]) == 80 and len(set(cohort["reads"])) == 80
    assert report["scale_episode_count"] == 80 and report["reference_block_count"] == 8
    assert report["fitting_circuit_count"] == 3
    assert report["validation_rows_skipped"] == 30
    assert report["validation_catalog_metadata_reads"] == report["validation_outcome_reads"] == 0
    expected_units = np.array([5000] * 16 + [20000])
    rates = [field_counts(factory._identity(row["episode_id"])) / (expected_units * .005)
             for row in cohort["spec"]["episodes"] if row["partition"] == "development_fit"]
    rms = np.sqrt(np.mean(np.stack(rates) ** 2, axis=(0, 1)))
    np.testing.assert_allclose(report["rms_rate_hz"], rms, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(report["single_spike_quantum_hz"], 1 / (expected_units * .005))
    np.testing.assert_allclose(report["reference_target"]["scale_hz"], np.maximum(rms, 1 / (expected_units * .005)))
    target, loaded = factory.load_reference(cohort["output"] / "reference.json")
    assert target.field_rate_hz.shape == (8, 980, 17)
    assert np.all(target.field_rate_hz[0] == 0)
    assert len(target.independent_block_ids) == 8
    assert len(loaded["source_bindings"]) == 80
    assert loaded["inference_scope"] == "eight_circuit_initial_noise_blocks_nested_in_three_fitting_circuits"
    assert loaded["scientific_claim_qualified"] is loaded["memory_recovery_qualified"] is False
    assert (cohort["output"] / "reference.json").read_text() == canonical_json(report)
    assert loaded["source_bindings"][0]["candidate_identity"]["circuit_id"] != (
        loaded["source_bindings"][0]["episode"]["circuit_id"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_fit_condition", "calibration", "test"])
def test_manifest_failures_reject_before_any_path_resolution_or_outcomes(cohort, change):
    spec = deepcopy(cohort["spec"])
    if change == "missing":
        spec["episodes"].pop(79)
    elif change == "duplicate":
        spec["episodes"].append(deepcopy(spec["episodes"][0]))
    elif change == "wrong_fit_condition":
        spec["episodes"][79]["episode_id"] = spec["episodes"][79]["episode_id"][:-2] + "10"
    else:
        spec["episodes"][-1]["partition"] = change
    cohort["inputs"].write_text(canonical_json(spec))
    with patch.object(factory, "_declared_path", side_effect=AssertionError("EARLY_PATH_RESOLUTION")):
        with pytest.raises(ValueError, match="PFC_REFERENCE_"):
            prepare(cohort)
    assert not cohort["reads"] and not cohort["opened"]


@pytest.mark.parametrize("change", ["intended_split", "native_block", "condition_label", "source_metadata"])
def test_last_fitting_source_is_checked_before_any_outcome_read(cohort, change):
    row = cohort["spec"]["episodes"][79]
    catalog_path = Path(row["dataset"]) / "dataset.json"
    catalog = json.loads(catalog_path.read_text())
    if change == "intended_split":
        catalog["binding"]["resolved_source_config"]["benchmark_episode_candidate"]["intended_split"] = "validation"
    elif change == "native_block":
        catalog["shards"][0]["episodes"][0]["block_id"] = "foreign_native_block"
    elif change == "condition_label":
        catalog["binding"]["resolved_source_config"]["benchmark_episode_candidate"]["condition_id"] = "cue_none"
    else:
        (Path(row["source_run"]) / "report.json").write_text("{}")
    catalog_path.write_text(canonical_json(catalog))
    with pytest.raises(ValueError, match="PFC_"):
        prepare(cohort)
    assert not cohort["reads"]


def test_field_order_or_population_denominator_mismatch_rejects_before_outcomes(cohort):
    metadata = next(iter(cohort["metadata"].values()))
    metadata["unit_field_index"] = metadata["unit_field_index"].copy()
    metadata["unit_field_index"][0] = 1
    with pytest.raises(ValueError, match="COMMON_SOURCE_UNIT_ORDER_FIELDS"):
        prepare(cohort)
    assert not cohort["reads"]


@pytest.mark.parametrize("change", ["scale", "missing_reference", "missing_source", "scientific_claim"])
def test_loader_rejects_inconsistent_reference_binding(cohort, change):
    report = prepare(cohort)
    if change == "scale":
        report["reference_target"]["scale_hz"][-1] *= 2
    elif change == "missing_reference":
        report["reference_target"]["field_rate_hz"].pop()
    elif change == "missing_source":
        report["source_bindings"].pop()
    else:
        report["scientific_claim_qualified"] = True
    path = cohort["root"] / "inconsistent_reference.json"
    path.write_text(canonical_json(report))
    with pytest.raises(ValueError, match="PFC_REFERENCE_|identified finite"):
        factory.load_reference(path)


@pytest.mark.parametrize("key,value", [("origin_tick", 6000), ("stop_tick_exclusive", 50000),
                                      ("expected_units", 20000), ("training_seeds", [41, 42, 44]),
                                      ("scientific_claim_qualified", True)])
def test_config_rejects_changed_scope_before_source_reads(cohort, key, value):
    config = yaml.safe_load(cohort["config"].read_text())
    config[key] = value
    changed = cohort["root"] / "changed.yaml"
    changed.write_text(yaml.safe_dump(config))
    with patch.object(factory, "_declared_path", side_effect=AssertionError("CHANGED_SCOPE_READ")):
        with pytest.raises(ValueError, match="REGISTERED_CONFIG_REQUIRED"):
            factory.prepare_references(cohort["inputs"], changed, cohort["output"])


def test_streaming_reader_uses_all_units_without_dense_unit_time_arrays(tmp_path):
    with CircuitArchiveWriter(tmp_path, binding()) as writer:
        writer.register_circuit(units())
        writer.add(replace(episode(), split="development"))
    data = CircuitDataset(tmp_path, splits=("development",))
    streamed = data._read("episode_a", 0, 10, 2, unit_start=0, unit_stop=0)
    complete = data.complete_episode("episode_a", bin_ticks=2)
    assert streamed["counts"].shape == (5, 0) and streamed["unit_ids"].shape == (0,)
    np.testing.assert_array_equal(streamed["field_counts"], complete["field_counts"])
    assert streamed["field_counts"].sum() == 5


def test_cli_writes_one_reference_artifact(cohort, capsys):
    assert factory.main(["--inputs", str(cohort["inputs"]), "--config", str(cohort["config"]),
                         "--output", str(cohort["output"])]) == 0
    assert json.loads(capsys.readouterr().out)["reference_blocks"] == 8
    assert [path.name for path in cohort["output"].iterdir()] == ["reference.json"]
