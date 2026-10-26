"""Metadata-only coverage and binding tests for PFC training-input assembly."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from . import pfc_training_inputs as intake


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def _file_binding(path, relative):
    return dict(path=relative, bytes=path.stat().st_size, sha256=intake.digest(path))


def _reader(base, job, circuit, block, tasks=()):
    """Create authentic small metadata files; no source events or arrays."""
    archive = base / ("reader_" + job)
    archive.mkdir()
    sources, bindings, catalogs = [], [], []
    for task in tasks:
        condition, episode = task["condition_id"], task["episode_id"]
        origin = base / "sources" / episode
        delivery = _write(origin / "delivery.json", dict(status="PASS", episode_id=episode))
        source = dict(episode_id=episode, condition_id=condition,
                      source_archive=str(origin), delivery_sha256=intake.digest(delivery))
        sources.append(source)
        relative = (f"circuit_{circuit:03d}/block_{block:03d}/"
                    f"{condition}_dataset/dataset/dataset.json")
        catalog = _write(archive / relative, dict(
            binding=dict(scope="source_reproduction", config_sha256=task["config_sha256"],
                         source_entry_sha256=intake.SOURCE_ENTRY_SHA256,
                         resolved_source_config=dict(benchmark_episode_candidate=dict(
                             identity=intake.identity(task), intended_split=task["split"],
                             condition_id=condition)),
                         source_bindings=[dict(case_id=episode, path=str(origin / "source_run"),
                                               delivery_sha256=source["delivery_sha256"])]),
            complete_episodes=1, circuits={"source_circuit": {"units": 100000}},
            shards=[dict(episodes=[dict(episode_id=episode, duration_ticks=100000,
                                       dt_ms=.05, split="development")])]))
        bindings.append(_file_binding(catalog, relative))
        catalogs.append(catalog)
    report = _write(archive / "report.json", dict(
        status="PASS_PFC_TEN_CONDITION_SOURCE_READBACK", job_id=job, exit_code=0,
        circuits=[dict(source_bindings=deepcopy(sources))]))
    delivery = _write(archive / "delivery.json", dict(
        verified=True, status="PASS_PFC_TEN_CONDITION_SOURCE_READBACK", job_id=job,
        files=[_file_binding(report, "report.json"), *bindings]))
    receipt = _write(base / f"parameter_block_readback_acceptance_{job}.json", dict(
        status="ACCEPTED_PFC_TEN_CONDITION_TASK_READBACK", complete_development_episodes=10,
        job_id=job, circuit_index=circuit, block_index=block, archive=str(archive),
        delivery_sha256=intake.digest(delivery), report_sha256=intake.digest(report), sources=sources))
    return dict(receipt=receipt, archive=archive, catalogs=catalogs)


def _reseal(cell):
    """Keep outer receipts valid when testing a deeper semantic mismatch."""
    delivery_path = cell["archive"] / "delivery.json"
    delivery = intake.read_json(delivery_path)
    for row in delivery["files"]:
        row.update(_file_binding(cell["archive"] / row["path"], row["path"]))
    _write(delivery_path, delivery)
    receipt = intake.read_json(cell["receipt"])
    receipt.update(delivery_sha256=intake.digest(delivery_path),
                   report_sha256=intake.digest(cell["archive"] / "report.json"))
    _write(cell["receipt"], receipt)


class MetadataFixture:
    def __init__(self, root):
        self.base = root / "acquisition"
        self.base.mkdir()
        self.baseline = root / "original_inputs.json"
        self.validation = root / "original_validation.json"
        self.plan = root / "plan"
        self.plan.mkdir()
        self.jobs = [str(700000 + group) for group in range(10)]
        self.tasks = []
        for circuit, role in ((23, "train"), (63, "train"), (74, "validation"),
                              (86, "calibration"), (52, "train")):
            for block in (1, 2):
                for offset, condition in enumerate(intake.CONDITIONS):
                    self.tasks.append(dict(
                        scenario_id="pfc_working_memory",
                        circuit_id=f"pfc_working_memory_circuit_{circuit:03d}",
                        circuit_index=circuit,
                        episode_id=f"pfc_working_memory_circuit_{circuit:03d}_block_{block:03d}_condition_{offset:02d}",
                        block_id=f"pfc_working_memory_circuit_{circuit:03d}_block_{block:03d}",
                        block_index=block, condition_index=offset, condition_id=condition, split=role,
                        connectivity_seed=200000000 + circuit,
                        initial_state_seed=210000000 + 1000 * circuit + block,
                        background_seed=220000000 + 1000 * circuit + block,
                        stimulus_seed=230000000 + 1000 * circuit + block,
                        sample_unit="complete_mechanistic_episode", config_sha256="a" * 64))
        source = root / "pinned_source.py"
        source.write_text("# Metadata-only pinned-source fixture.\n")
        self.source = source
        episodes = []
        for circuit, role in ((23, "development_fit"), (63, "development_fit"),
                              (74, "development_validation")):
            for offset in range(10):
                episode = f"pfc_working_memory_circuit_{circuit:03d}_block_000_condition_{offset:02d}"
                episodes.append(dict(dataset=str(root / "block0" / episode / "dataset"),
                                     episode_id=episode, source_run=str(root / "block0" / episode / "source_run"),
                                     partition=role))
        self.original = dict(schema="pfc_population_development_inputs_v1",
                             source_entry=str(source), episodes=episodes)
        _write(self.baseline, self.original)
        self.output, self.coverage = root / "inputs.json", root / "coverage.json"

    def bind_baseline(self):
        receipts = []
        for circuit in (23, 63, 74, 86):
            cell = _reader(self.base, str(600000 + circuit), circuit, 0)
            receipt = intake.read_json(cell["receipt"])
            receipts.append(dict(path=str(cell["receipt"]), sha256=intake.digest(cell["receipt"]),
                                 authenticated_delivery_sha256=receipt["delivery_sha256"],
                                 authenticated_report_sha256=receipt["report_sha256"]))
        bindings = _write(self.baseline.parent / "input_receipt_bindings.json", receipts)
        _write(self.validation, dict(files=[_file_binding(self.baseline, self.baseline.name),
                                           _file_binding(bindings, bindings.name)]))

    def accept(self, group):
        tasks = self.tasks[group * 10:(group + 1) * 10]
        return _reader(self.base, self.jobs[group], tasks[0]["circuit_index"],
                       tasks[0]["block_index"], tasks)

    def assemble(self):
        return intake.assemble(self.baseline, self.validation, self.base, self.plan, self.jobs)

    def cli(self, monkeypatch):
        values = ("baseline", self.baseline, "baseline-validation", self.validation,
                  "base", self.base, "plan", self.plan, "output", self.output, "coverage", self.coverage)
        argv = ["pfc_training_inputs"]
        for index in range(0, len(values), 2):
            argv.extend(("--" + values[index], str(values[index + 1])))
        argv.extend(("--reader-jobs", *self.jobs))
        monkeypatch.setattr(sys, "argv", argv)
        intake.main()


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    fixture = MetadataFixture(tmp_path)
    monkeypatch.setattr(intake, "SOURCE_ENTRY_SHA256", intake.digest(fixture.source))
    fixture.bind_baseline()

    def load_plan(path, *, require_release=False):
        assert path == fixture.plan and require_release is True
        return deepcopy(fixture.tasks)

    monkeypatch.setattr(intake, "load_plan", load_plan)
    return fixture


def test_original_block0_rows_remain_exact_and_missing_readers_are_pending(metadata):
    spec, coverage = metadata.assemble()
    assert spec == metadata.original
    assert coverage["status"] == "PARTIAL_FIT_VALIDATION_INPUT_COVERAGE"
    assert (coverage["fitting_episodes"], coverage["validation_episodes"]) == (20, 10)
    assert coverage["accepted_export_episodes"] == 40
    assert coverage["reserved_calibration_episodes"] == 10
    assert len(coverage["reader_groups"]) == 10
    assert all(cell["status"] == "PENDING_READER_ACCEPTANCE" for cell in coverage["reader_groups"])
    assert coverage["fit_validation_coverage_ready"] is False
    assert all(coverage[name] is False for name in
               ("bulk_authentication_repeated", "raw_source_read_repeated",
                "training_submitted", "scientific_fidelity_qualified"))
    assert coverage["source_simulations"] == coverage["optimizer_updates"] == 0


@pytest.mark.parametrize("group,counts", [(0, (30, 10)), (4, (20, 20)), (8, (30, 10))])
def test_complete_ten_condition_group_adds_only_its_declared_role(metadata, group, counts):
    metadata.accept(group)
    spec, coverage = metadata.assemble()
    assert spec["episodes"][:30] == metadata.original["episodes"]
    assert len(spec["episodes"]) == 40
    assert (coverage["fitting_episodes"], coverage["validation_episodes"]) == counts
    task_rows = metadata.tasks[group * 10:(group + 1) * 10]
    assert [row["episode_id"] for row in spec["episodes"][30:]] == [row["episode_id"] for row in task_rows]
    expected_role = "development_fit" if task_rows[0]["split"] == "train" else "development_validation"
    assert all(row["partition"] == expected_role for row in spec["episodes"][30:])
    assert coverage["reader_groups"][group]["status"] == "ACCEPTED_TRAINING_INPUT"
    assert coverage["accepted_export_episodes"] == 50
    assert coverage["fit_validation_coverage_ready"] is False


def test_authenticated_calibration_group_counts_toward_acquisition_but_stays_excluded(metadata):
    metadata.accept(6)
    spec, coverage = metadata.assemble()
    assert spec == metadata.original
    assert coverage["reader_groups"][6]["status"] == "ACCEPTED_EXCLUDED_CALIBRATION"
    assert coverage["reader_groups"][6]["complete_episodes"] == 10
    assert coverage["accepted_export_episodes"] == 50
    assert coverage["reserved_calibration_episodes"] == 20
    assert len(coverage["receipt_bindings"]) == 5


def test_full_authorized_envelope_preserves_block0_and_excludes_all_calibration(metadata):
    for group in range(10):
        metadata.accept(group)
    spec, coverage = metadata.assemble()
    assert spec["episodes"][:30] == metadata.original["episodes"]
    assert len(spec["episodes"]) == len({row["episode_id"] for row in spec["episodes"]}) == 110
    assert (coverage["fitting_episodes"], coverage["validation_episodes"]) == (80, 30)
    assert coverage["status"] == "COMPLETE_FIT_VALIDATION_INPUT_COVERAGE"
    assert coverage["accepted_export_episodes"] == coverage["declared_complete_episode_envelope"] == 140
    assert coverage["reserved_calibration_episodes"] == 30
    assert coverage["fit_validation_coverage_ready"] is True
    assert not any("circuit_086" in row["episode_id"] for row in spec["episodes"])
    assert {row["circuit_index"] for row in coverage["reader_groups"]
            if row["status"] == "ACCEPTED_EXCLUDED_CALIBRATION"} == {86}


def test_fit_validation_readiness_does_not_open_or_require_calibration(metadata):
    for group in (0, 1, 2, 3, 4, 5, 8, 9):
        metadata.accept(group)
    spec, coverage = metadata.assemble()
    assert len(spec["episodes"]) == 110
    assert coverage["fit_validation_coverage_ready"] is True
    assert coverage["accepted_export_episodes"] == 120
    assert coverage["reserved_calibration_episodes"] == 10
    assert all(coverage["reader_groups"][group]["status"] == "PENDING_READER_ACCEPTANCE"
               for group in (6, 7))


def test_incremental_cli_rebuild_keeps_prior_inputs_and_adds_new_group(metadata, monkeypatch):
    metadata.accept(0)
    metadata.cli(monkeypatch)
    first = intake.read_json(metadata.output)
    metadata.accept(4)
    metadata.cli(monkeypatch)
    second, coverage = intake.read_json(metadata.output), intake.read_json(metadata.coverage)
    assert second["episodes"][:len(first["episodes"])] == first["episodes"]
    assert len(second["episodes"]) == 50
    assert (coverage["fitting_episodes"], coverage["validation_episodes"]) == (30, 20)
    assert coverage["inputs_sha256"] == intake.digest(metadata.output)


def test_identical_cli_does_not_rewrite_existing_inputs(metadata, monkeypatch):
    metadata.accept(0)
    metadata.cli(monkeypatch)
    before, stamp = metadata.output.read_bytes(), metadata.output.stat().st_mtime_ns
    original_write = Path.write_text
    writes = []

    def tracked_write(path, *args, **kwargs):
        writes.append(path)
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", tracked_write)
    metadata.cli(monkeypatch)
    assert metadata.output not in writes
    assert metadata.coverage in writes
    assert metadata.output.read_bytes() == before
    assert metadata.output.stat().st_mtime_ns == stamp


def _alter(cell, change):
    receipt = intake.read_json(cell["receipt"])
    if change == "source_identity":
        receipt["sources"][0]["source_archive"] += "_foreign"
        _write(cell["receipt"], receipt)
    elif change == "condition_identity":
        receipt["sources"][0]["condition_id"] = intake.CONDITIONS[1]
        _write(cell["receipt"], receipt)
    elif change == "reader_block":
        receipt["block_index"] = 2
        _write(cell["receipt"], receipt)
    elif change == "source_delivery":
        source = Path(receipt["sources"][0]["source_archive"]) / "delivery.json"
        _write(source, dict(status="PASS", episode_id="foreign"))
    elif change == "catalog_bytes":
        catalog = intake.read_json(cell["catalogs"][0])
        catalog["binding"]["resolved_source_config"]["benchmark_episode_candidate"]["intended_split"] = "calibration"
        _write(cell["catalogs"][0], catalog)
    else:
        catalog = intake.read_json(cell["catalogs"][0])
        candidate = catalog["binding"]["resolved_source_config"]["benchmark_episode_candidate"]
        if change == "nested_protected_split":
            candidate["intended_split"] = "calibration"
        elif change == "nested_block":
            candidate["identity"]["block_index"] = 2
        elif change == "nested_condition":
            candidate["identity"]["condition_index"] = 1
        elif change == "complete_clock":
            catalog["shards"][0]["episodes"][0]["duration_ticks"] = 99999
        elif change == "full_unit_count":
            catalog["circuits"]["source_circuit"]["units"] = 99999
        else:
            raise AssertionError(change)
        _write(cell["catalogs"][0], catalog)
        _reseal(cell)


@pytest.mark.parametrize("change,error", [
    ("source_identity", "SOURCE_IDENTITY_MISMATCH"),
    ("condition_identity", "ALL_TEN_SOURCE_CONDITIONS_REQUIRED"),
    ("reader_block", "READER_CIRCUIT_BLOCK_MISMATCH"),
    ("source_delivery", "SOURCE_DELIVERY_CHANGED"),
    ("catalog_bytes", "EXPORTED_CATALOG_CHANGED"),
    ("nested_protected_split", "EXPORTED_PROSPECTIVE_IDENTITY_MISMATCH"),
    ("nested_block", "EXPORTED_PROSPECTIVE_IDENTITY_MISMATCH"),
    ("nested_condition", "EXPORTED_PROSPECTIVE_IDENTITY_MISMATCH"),
    ("complete_clock", "EXPORTED_EPISODE_CLOCK_MISMATCH"),
    ("full_unit_count", "COMPLETE_FULL_UNIT_EPISODE_REQUIRED"),
])
def test_changed_binding_rejects_before_cli_writes(metadata, monkeypatch, change, error):
    cell = metadata.accept(0)
    _alter(cell, change)
    metadata.output.write_text("preserved inputs\n")
    metadata.coverage.write_text("preserved coverage\n")
    before = (metadata.output.read_bytes(), metadata.coverage.read_bytes())
    with pytest.raises(ValueError, match=error):
        metadata.cli(monkeypatch)
    assert (metadata.output.read_bytes(), metadata.coverage.read_bytes()) == before


def test_changed_baseline_rejects_before_new_output_creation(metadata, monkeypatch):
    changed = deepcopy(metadata.original)
    changed["episodes"][0]["partition"] = "development_validation"
    _write(metadata.baseline, changed)
    with pytest.raises(ValueError, match="VALIDATED_BASELINE_CHANGED"):
        metadata.cli(monkeypatch)
    assert not metadata.output.exists() and not metadata.coverage.exists()


def test_baseline_cannot_be_used_as_output(metadata, monkeypatch):
    before = metadata.baseline.read_bytes()
    metadata.output = metadata.baseline
    with pytest.raises(ValueError, match="PRESERVE_VALIDATED_BASELINE"):
        metadata.cli(monkeypatch)
    assert metadata.baseline.read_bytes() == before
    assert not metadata.coverage.exists()
