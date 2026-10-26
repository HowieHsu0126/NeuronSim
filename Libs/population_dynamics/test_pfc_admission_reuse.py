"""One-time local source admission with bounded active-event reloads."""
from __future__ import annotations

from copy import deepcopy
import gc
import json
from pathlib import Path
import shutil
import weakref

import h5py
import numpy as np
import pytest

from . import pfc_renewal_data as adapter
from . import pfc_source_dataset as exporter
from . import test_pfc_source_dataset as fixtures
from .circuit_data import CircuitDataset


@pytest.fixture
def local_episode(tmp_path):
    fixture = fixtures.PfcSourceDatasetTests()
    fixture.setUp()
    try:
        original_source = fixture.source()
        exporter.export([original_source], fixture.config, fixture.root / "export")
        root = tmp_path / "local"
        source_archive = root / "source_archive"
        dataset_root = root / "dataset"
        shutil.copytree(original_source.parent, source_archive)
        shutil.copytree(fixture.root / "export/dataset", dataset_root)
        pinned = Path(__file__).resolve().parents[1] / (
            "external/pfc_working_memory/neurodynex3/working_memory_network/wm_model.py")
        source_entry = root / "source_entry/wm_model.py"
        source_entry.parent.mkdir(parents=True)
        shutil.copy2(pinned, source_entry)
        data = CircuitDataset(dataset_root, splits=("development",))
        yield dict(data=data, source=source_archive / "source_run", source_entry=source_entry,
                   original_entry=pinned, episode="pfc_94001_strong_cue", root=root)
    finally:
        fixture.doCleanups()


def _load(fixture, **changes):
    options = dict(bin_ticks=100, source_run=fixture["source"], source_entry=fixture["source_entry"])
    options.update(changes)
    return adapter.PfcRenewalEpisode(fixture["data"], fixture["episode"], **options)


def _spy_source_values(monkeypatch):
    original = adapter._source_values
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "_source_values", counted)
    return calls


def _same_window(first, second):
    assert first.__dict__.keys() == second.__dict__.keys()
    for name in first.__dict__:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def _small_record(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    assert not isinstance(value, (np.ndarray, adapter.PfcRenewalEpisode, CircuitDataset))
    assert not isinstance(value, h5py.File)
    if isinstance(value, dict):
        for key, item in value.items():
            _small_record(key, seen)
            _small_record(item, seen)
    elif isinstance(value, (tuple, list, set)):
        for item in value:
            _small_record(item, seen)
    else:
        assert isinstance(value, (str, int, float, bool, type(None)))


def test_local_repeat_reuses_authentication_and_reloads_exact_active_arrays(local_episode, monkeypatch):
    calls = _spy_source_values(monkeypatch)
    first = _load(local_episode)
    assert len(calls) == 1
    events = (first.event_ticks.copy(), first.event_units.copy())
    metadata, binding = first.population_metadata(), deepcopy(first.binding)
    windows = [first.renewal_window(start, stop) for start, stop in
               ((0, 100000), (2000, 2500), (80000, 80500))]
    second = _load(local_episode)
    assert len(calls) == 1
    np.testing.assert_array_equal(second.event_ticks, events[0])
    np.testing.assert_array_equal(second.event_units, events[1])
    assert second.binding == binding and second.population_metadata() == metadata
    assert not np.shares_memory(first.event_ticks, second.event_ticks)
    assert not np.shares_memory(first.event_units, second.event_units)
    for expected, (start, stop) in zip(windows, ((0, 100000), (2000, 2500), (80000, 80500))):
        _same_window(expected, second.renewal_window(start, stop))
        np.testing.assert_array_equal(first.ready_at(stop), second.ready_at(stop))


@pytest.mark.parametrize("file_kind", [
    "raw", "source_config", "report", "source_entry", "catalog", "circuit", "shard",
    "delivery", "current", "producer", "qualification",
])
def test_changed_consumed_local_file_rejects_before_any_array_reload(
        local_episode, monkeypatch, file_kind):
    _load(local_episode)
    data, source = local_episode["data"], local_episode["source"]
    row = data.episodes[local_episode["episode"]]
    paths = dict(
        raw=source / "raw.npz", source_config=source / "config.yaml", report=source / "report.json",
        source_entry=local_episode["source_entry"], catalog=data.root / "dataset.json",
        circuit=data.root / data.catalog["circuits"][row["circuit_id"]]["path"],
        shard=data.root / data.shards[row["shard"]]["path"],
        delivery=source.parent / "delivery.json", current=source / "applied_current_observation.npz",
        producer=source.parent / "Libs/population_dynamics/pfc_source_execution.py",
        qualification=source.parent / "qualification/comparison/qualification.json",
    )
    path = paths[file_kind]
    with path.open("ab") as stream:
        stream.write(b"\nchanged after admission\n")

    def forbidden(*args, **kwargs):
        pytest.fail("changed evidence reached source/HDF5 array reload")

    monkeypatch.setattr(adapter, "_source_values", forbidden)
    monkeypatch.setattr(adapter.np, "load", forbidden)
    monkeypatch.setattr(adapter.h5py, "File", forbidden)
    with pytest.raises(ValueError, match="ADMITTED_FILE_CHANGED"):
        _load(local_episode)


@pytest.mark.parametrize("identity", ["catalog", "row", "circuit"])
def test_changed_in_memory_identity_rejects_before_array_reload(
        local_episode, monkeypatch, identity):
    _load(local_episode)
    data = local_episode["data"]
    if identity == "catalog":
        data.catalog["binding"]["resolved_source_config"]["stimulus_center_deg"] += 1
    elif identity == "row":
        data.episodes[local_episode["episode"]]["block_id"] = "foreign_block"
    else:
        data.episodes[local_episode["episode"]]["circuit_id"] = "foreign_circuit"

    def forbidden(*args, **kwargs):
        pytest.fail("changed catalog/row reached array reload")

    monkeypatch.setattr(adapter, "_source_values", forbidden)
    monkeypatch.setattr(adapter.h5py, "File", forbidden)
    with pytest.raises(ValueError, match="ADMITTED_IDENTITY_CHANGED"):
        _load(local_episode)


def test_distinct_bin_width_gets_its_own_full_admission(local_episode, monkeypatch):
    calls = _spy_source_values(monkeypatch)
    first = _load(local_episode)
    second = _load(local_episode, bin_ticks=200)
    third = _load(local_episode, bin_ticks=200)
    assert len(calls) == 2
    assert first.binding["bin_ticks"] == 100
    assert second.binding["bin_ticks"] == third.binding["bin_ticks"] == 200
    _same_window(second.renewal_window(0, 100000), third.renewal_window(0, 100000))
    assert len(local_episode["data"].pfc_source_admissions) == 2


def test_cache_records_do_not_retain_event_arrays_or_active_adapters(local_episode):
    loaded = _load(local_episode)
    ref = weakref.ref(loaded)
    ticks = weakref.ref(loaded.event_ticks)
    units = weakref.ref(loaded.event_units)
    records = local_episode["data"].pfc_source_admissions
    _small_record(records)
    del loaded
    gc.collect()
    assert ref() is None and ticks() is None and units() is None
    _small_record(records)


def test_unmapped_original_source_entry_keeps_full_authentication(local_episode, monkeypatch):
    calls = _spy_source_values(monkeypatch)
    first = _load(local_episode, source_entry=local_episode["original_entry"])
    second = _load(local_episode, source_entry=local_episode["original_entry"])
    assert len(calls) == 2
    assert not getattr(local_episode["data"], "pfc_source_admissions", {})
    assert first.binding == second.binding
    _same_window(first.renewal_window(2000, 2500), second.renewal_window(2000, 2500))


def test_unselected_episode_rejects_before_admission_or_file_reads(local_episode, monkeypatch):
    _load(local_episode)
    local_episode["data"].episodes = {}

    def forbidden(*args, **kwargs):
        pytest.fail("unselected episode reached file admission")

    monkeypatch.setattr(adapter, "sha256_file", forbidden)
    monkeypatch.setattr(adapter, "_source_values", forbidden)
    monkeypatch.setattr(adapter.h5py, "File", forbidden)
    with pytest.raises(KeyError):
        _load(local_episode)
