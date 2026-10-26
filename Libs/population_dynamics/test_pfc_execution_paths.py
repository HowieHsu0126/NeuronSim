"""Local-copy admission tests without rewriting PFC input identities."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from . import pfc_population_training_data as consumer
from . import pfc_source_dataset as exporter
from . import test_pfc_source_dataset as source_fixtures
from .test_pfc_population_training_data import configuration


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def relocated(tmp_path, monkeypatch):
    """Copy two tiny authenticated archives, then remove their original roots."""
    monkeypatch.delenv("NEURONSIM_EXECUTION_PATH_MAP", raising=False)
    fixtures, records, circuits, copies = [], [], [], []
    try:
        for index, case in enumerate(("pfc_94001_strong_cue", "pfc_94002_strong_cue")):
            fixture = source_fixtures.PfcSourceDatasetTests()
            fixture.setUp()
            fixtures.append(fixture)
            if index:
                fixture.fixture.config.update(sigma_weight_profile=30., Jpos_excit2excit=1.7)
                fixture.config.write_text(yaml.safe_dump(fixture.fixture.config))
            source = fixture.source(case)
            if index:
                fixture.fixture.parameterize_archive(source)
            exporter.export([source], fixture.config, fixture.root / "export")
            dataset = fixture.root / "export/dataset"
            catalog = json.loads((dataset / "dataset.json").read_text())
            circuits.append(next(iter(catalog["circuits"])))
            records.append(dict(dataset=str(dataset), source_run=str(source), episode_id=case,
                                partition="development_fit" if not index else "development_validation"))
            copies.append((dataset, source))
        config = configuration()
        config.update(fit_circuit_ids=circuits[:1], validation_circuit_ids=circuits[1:], bin_ticks=100)
        pinned = Path(__file__).resolve().parents[1] / "external/pfc_working_memory/neurodynex3/working_memory_network/wm_model.py"
        inputs = tmp_path / "inputs.json"
        _write(inputs, dict(schema="pfc_population_development_inputs_v1",
                            source_entry=str(pinned), episodes=records))
        direct = consumer.PfcPopulationTrainingCohort(inputs, config)
        expected_window = direct.window(direct.validation[0])
        expected = dict(binding=deepcopy(direct.binding),
                        counts=expected_window.counts.copy(),
                        stimulus=expected_window.stimulus.copy(),
                        metadata=direct.active_episode.population_metadata())
        local = tmp_path / "local"
        local.mkdir()
        local_source = local / "wm_model.py"
        shutil.copy2(pinned, local_source)
        paths = {str(pinned): dict(path=str(local_source), sha256=consumer.sha256_file(pinned))}
        for index, (dataset, source) in enumerate(copies):
            local_dataset = local / str(index) / "dataset"
            local_archive = local / str(index) / "source_archive"
            shutil.copytree(dataset, local_dataset)
            shutil.copytree(source.parent, local_archive)
            paths[str(dataset.resolve())] = dict(path=str(local_dataset),
                                                sha256=consumer.sha256_file(dataset / "dataset.json"))
            paths[str(source.resolve())] = dict(path=str(local_archive / "source_run"),
                                               sha256=consumer.sha256_file(source.parent / "delivery.json"))
        map_file = tmp_path / "execution_path_map.json"
        mapping = dict(schema="pfc_population_execution_path_map_v1",
                       inputs_sha256=consumer.sha256_file(inputs), paths=paths)
        _write(map_file, mapping)
        before = inputs.read_bytes()
        for dataset, source in copies:
            shutil.rmtree(dataset)
            shutil.rmtree(source.parent)
        monkeypatch.setenv("NEURONSIM_EXECUTION_PATH_MAP", str(map_file))
        yield dict(inputs=inputs, config=config, mapping=mapping, map_file=map_file,
                   original_bytes=before, expected=expected, copies=copies, local=local)
    finally:
        for fixture in fixtures:
            fixture.doCleanups()


def test_relocated_authenticated_loading_retains_input_and_episode_checkpoint_bindings(relocated):
    cohort = consumer.PfcPopulationTrainingCohort(relocated["inputs"], relocated["config"])
    assert cohort.binding == relocated["expected"]["binding"]
    assert cohort.binding["circuit_inputs_sha256"] == relocated["mapping"]["inputs_sha256"]
    assert relocated["inputs"].read_bytes() == relocated["original_bytes"]
    assert all(case["dataset"].is_relative_to(relocated["local"])
               and case["source_run"].is_relative_to(relocated["local"])
               for case in cohort.fit + cohort.validation)
    window = cohort.window(cohort.validation[0])
    np.testing.assert_array_equal(window.counts, relocated["expected"]["counts"])
    np.testing.assert_array_equal(window.stimulus, relocated["expected"]["stimulus"])
    assert cohort.active_episode.population_metadata() == relocated["expected"]["metadata"]


def _before_arrays(monkeypatch):
    monkeypatch.setattr(consumer, "PfcRenewalEpisode",
                        lambda *args, **kwargs: pytest.fail("mapped input rejected after source array admission"))


@pytest.mark.parametrize("kind", ["dataset", "source_run", "source_entry"])
def test_missing_declared_mapping_rejects_before_array_admission(relocated, monkeypatch, kind):
    value = deepcopy(relocated["mapping"])
    if kind == "source_entry":
        key = next(key for key in value["paths"] if key.endswith("wm_model.py"))
    elif kind == "source_run":
        key = next(key for key in value["paths"] if key.endswith("source_run"))
    else:
        key = next(key for key in value["paths"] if key.endswith("dataset"))
    del value["paths"][key]
    _write(relocated["map_file"], value)
    _before_arrays(monkeypatch)
    with pytest.raises(ValueError, match="DECLARED_IDENTITIES_REQUIRED"):
        consumer.PfcPopulationTrainingCohort(relocated["inputs"], relocated["config"])


@pytest.mark.parametrize("change,error", [
    ("foreign_key", "DECLARED_IDENTITIES_REQUIRED"),
    ("input_digest", "INPUT_CHANGED"),
    ("unknown_schema", "SCHEMA_REQUIRED"),
    ("bad_digest", "SCHEMA_REQUIRED"),
    ("relative_target", "LOCAL_TARGET_REQUIRED"),
    ("shared_target", "LOCAL_TARGET_REQUIRED"),
    ("target_absent", "LOCAL_TARGET_REQUIRED"),
    ("target_escape", "LOCAL_TARGET_REQUIRED"),
    ("dataset_digest", "METADATA_CHANGED"),
    ("source_digest", "METADATA_CHANGED"),
    ("swapped_catalog", "METADATA_CHANGED"),
])
def test_malformed_or_foreign_map_rejects_before_source_arrays(relocated, monkeypatch, change, error):
    value = deepcopy(relocated["mapping"])
    dataset = next(key for key in value["paths"] if key.endswith("dataset"))
    source = next(key for key in value["paths"] if key.endswith("source_run"))
    pinned = next(key for key in value["paths"] if key.endswith("wm_model.py"))
    if change == "foreign_key":
        value["paths"]["/foreign/catalog"] = value["paths"][dataset]
    elif change == "input_digest":
        value["inputs_sha256"] = "0" * 64
    elif change == "unknown_schema":
        value["schema"] = "unregistered"
    elif change == "bad_digest":
        value["paths"][dataset]["sha256"] = "not-a-digest"
    elif change == "relative_target":
        value["paths"][dataset]["path"] = "local/0/dataset"
    elif change == "shared_target":
        value["paths"][pinned]["path"] = pinned
    elif change == "target_absent":
        value["paths"][dataset]["path"] += "_absent"
    elif change == "target_escape":
        link = relocated["local"] / "escape"
        link.symlink_to(Path(pinned).parent, target_is_directory=True)
        value["paths"][dataset]["path"] = str(link)
    elif change == "dataset_digest":
        value["paths"][dataset]["sha256"] = "0" * 64
    elif change == "source_digest":
        value["paths"][source]["sha256"] = "0" * 64
    elif change == "swapped_catalog":
        other = next(key for key in value["paths"] if key.endswith("dataset") and key != dataset)
        value["paths"][dataset]["path"] = value["paths"][other]["path"]
    _write(relocated["map_file"], value)
    _before_arrays(monkeypatch)
    with pytest.raises(ValueError, match=error):
        consumer.PfcPopulationTrainingCohort(relocated["inputs"], relocated["config"])


@pytest.mark.parametrize("change,error", [
    ("protected_role", "PROSPECTIVE_IDENTITY_CHANGED"),
    ("foreign_source_path", "SOURCE_IDENTITY_CHANGED"),
    ("foreign_delivery_case", "SOURCE_IDENTITY_CHANGED"),
    ("foreign_report", "METADATA_CHANGED"),
])
def test_nested_identity_rejects_even_when_outer_map_digest_is_resealed(relocated, monkeypatch, change, error):
    value = deepcopy(relocated["mapping"])
    dataset = next(key for key in value["paths"] if key.endswith("dataset"))
    source = next(key for key in value["paths"] if key.endswith("source_run"))
    catalog_path = Path(value["paths"][dataset]["path"]) / "dataset.json"
    catalog = json.loads(catalog_path.read_text())
    if change == "protected_role":
        catalog["binding"]["resolved_source_config"]["benchmark_episode_candidate"] = {"intended_split": "calibration"}
    elif change == "foreign_source_path":
        catalog["binding"]["source_bindings"][0]["path"] += "_foreign"
    elif change == "foreign_delivery_case":
        delivery_path = Path(value["paths"][source]["path"]).parent / "delivery.json"
        delivery = json.loads(delivery_path.read_text())
        delivery["case_id"] = "foreign_episode"
        _write(delivery_path, delivery)
        value["paths"][source]["sha256"] = consumer.sha256_file(delivery_path)
        catalog["binding"]["source_bindings"][0]["delivery_sha256"] = value["paths"][source]["sha256"]
    elif change == "foreign_report":
        report_path = Path(value["paths"][source]["path"]) / "report.json"
        report = json.loads(report_path.read_text())
        report["case_id"] = "foreign_episode"
        _write(report_path, report)
    _write(catalog_path, catalog)
    value["paths"][dataset]["sha256"] = consumer.sha256_file(catalog_path)
    _write(relocated["map_file"], value)
    _before_arrays(monkeypatch)
    with pytest.raises(ValueError, match=error):
        consumer.PfcPopulationTrainingCohort(relocated["inputs"], relocated["config"])


def test_last_declared_identity_is_checked_before_first_source_array_read(relocated, monkeypatch):
    value = deepcopy(relocated["mapping"])
    datasets = [key for key in value["paths"] if key.endswith("dataset")]
    path = Path(value["paths"][datasets[-1]]["path"]) / "dataset.json"
    catalog = json.loads(path.read_text())
    catalog["shards"][0]["episodes"][0]["episode_id"] = "foreign_validation_episode"
    _write(path, catalog)
    value["paths"][datasets[-1]]["sha256"] = consumer.sha256_file(path)
    _write(relocated["map_file"], value)
    _before_arrays(monkeypatch)
    with pytest.raises(ValueError, match="PROSPECTIVE_IDENTITY_CHANGED"):
        consumer.PfcPopulationTrainingCohort(relocated["inputs"], relocated["config"])


def test_duplicate_json_keys_are_rejected_before_array_admission(relocated, monkeypatch):
    raw = relocated["map_file"].read_text()
    relocated["map_file"].write_text(raw.replace('"schema":', '"schema": "foreign", "schema":', 1))
    _before_arrays(monkeypatch)
    with pytest.raises(ValueError, match="DUPLICATE_KEY"):
        consumer.PfcPopulationTrainingCohort(relocated["inputs"], relocated["config"])
