"""Manifest-bound recovery of initial population checkpoints without model loads."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


_spec = importlib.util.spec_from_file_location(
    "population_initial_checkpoint_campaign",
    Path(__file__).with_name("run_population_trajectory_campaign.py"),
)
campaign = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(campaign)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _binding(path, relative):
    return dict(path=relative, bytes=path.stat().st_size, sha256=campaign.digest(path))


class RecoveryFixture:
    def __init__(self, tmp_path, epochs=0):
        self.parent, self.work = tmp_path / "parent", tmp_path / "work"
        self.parent.mkdir()
        self.work.mkdir()
        self.config = dict(maximum_epochs=20, representation="fixed",
                           training_seeds=[41, 42, 43], device="cpu")
        self.report = dict(training_seed=43, completed_epochs=epochs,
                           best_epoch=None if epochs == 0 else epochs - 1,
                           effective_maximum_epochs=20,
                           calibration_outcomes_accessed=False, test_outcomes_accessed=False,
                           status="RUNNING", representation="fixed")
        _write(self.parent / "training_configuration.json", self.config)
        _write(self.work / "training_configuration.json", self.config)
        _write(self.parent / "training/run.json", self.report)
        latest = self.parent / "training/latest.pt"
        latest.write_bytes(b"Opaque committed optimizer, model, and RNG fixture.\n")
        self.source_name = "Libs/population_dynamics/population_state.py"
        source = self.parent / self.source_name
        source.parent.mkdir(parents=True)
        source.write_text("# Exact frozen numerical source fixture.\n")
        target = self.work / self.source_name
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        self.names = ["training_configuration.json", self.source_name,
                      "training/latest.pt", "training/run.json"]
        if epochs:
            (self.parent / "training/best.pt").write_bytes(b"Opaque selected checkpoint.\n")
            _write(self.parent / "training/epochs.json", [dict(epoch=index) for index in range(epochs)])
            self.names.extend(["training/best.pt", "training/epochs.json"])
        self.proof = dict(status="RESOURCE_BLOCKED", exit_code=3, training_seed=43,
                          array_job_id="607379", job_id="610541", device="cpu",
                          representation="fixed", calibration_or_test_accessed=False,
                          scientific_claim_qualified=False)
        self.reseal()

    def reseal(self):
        self.proof["files"] = [_binding(self.parent / name, name) for name in self.names]
        _write(self.parent / "delivery.json", self.proof)

    def restore(self):
        return campaign.stage_recovery(self.parent, self.work, 43, "607379", "cpu", "fixed")


@pytest.fixture
def initial(tmp_path):
    return RecoveryFixture(tmp_path)


def test_initial_latest_only_checkpoint_restores_exact_manifest_bound_state(initial):
    original = {name: (initial.parent / name).read_bytes() for name in initial.names}
    receipt = initial.restore()
    assert receipt["committed_epochs"] == 0
    assert receipt["maximum_epochs"] == 20
    assert receipt["parent_job_id"] == "610541"
    assert receipt["parent_delivery_sha256"] == campaign.digest(initial.parent / "delivery.json")
    assert receipt["latest_checkpoint_sha256"] == campaign.digest(initial.parent / "training/latest.pt")
    assert (initial.work / "training/latest.pt").read_bytes() == original["training/latest.pt"]
    assert (initial.work / "training/run.json").read_bytes() == original["training/run.json"]
    assert (initial.work / initial.source_name).read_bytes() == original[initial.source_name]
    assert (initial.work / "prerequisites/recovery_configuration.json").read_bytes() == original["training_configuration.json"]
    assert (initial.work / "prerequisites/recovery_delivery.json").read_bytes() == (initial.parent / "delivery.json").read_bytes()
    assert not (initial.work / "training/best.pt").exists()
    assert not (initial.work / "training/epochs.json").exists()
    assert original == {name: (initial.parent / name).read_bytes() for name in initial.names}


@pytest.mark.parametrize("change,error", [
    ("config", "training configuration changed"),
    ("source", "numerical source changed"),
    ("checkpoint", "source manifest identity changed"),
])
def test_initial_recovery_rejects_changed_configuration_source_or_checkpoint(initial, change, error):
    if change == "config":
        value = deepcopy(initial.config)
        value["maximum_epochs"] = 21
        _write(initial.work / "training_configuration.json", value)
    elif change == "source":
        (initial.work / initial.source_name).write_text("# Changed numerical source.\n")
    else:
        (initial.parent / "training/latest.pt").write_bytes(b"Foreign opaque checkpoint.\n")
    with pytest.raises(ValueError, match=error):
        initial.restore()
    assert not (initial.work / "prerequisites/recovery_delivery.json").exists()


@pytest.mark.parametrize("artifact", ["training/best.pt", "training/epochs.json"])
def test_positive_epoch_requires_both_selected_checkpoint_and_history(tmp_path, artifact):
    fixture = RecoveryFixture(tmp_path, epochs=1)
    fixture.names.remove(artifact)
    fixture.reseal()
    with pytest.raises(ValueError, match="committed recovery artifact missing: " + artifact):
        fixture.restore()
    assert not (fixture.work / "prerequisites/recovery_delivery.json").exists()


def test_positive_epoch_restores_latest_best_and_history_without_payload_loading(tmp_path):
    fixture = RecoveryFixture(tmp_path, epochs=1)
    receipt = fixture.restore()
    assert receipt["committed_epochs"] == 1
    for name in ("training/latest.pt", "training/best.pt", "training/epochs.json"):
        assert (fixture.work / name).read_bytes() == (fixture.parent / name).read_bytes()


def test_initial_checkpoint_cannot_declare_selected_best_epoch(initial):
    initial.report["best_epoch"] = 0
    _write(initial.parent / "training/run.json", initial.report)
    initial.reseal()
    with pytest.raises(ValueError, match="zero-epoch recovery cannot declare a selected epoch"):
        initial.restore()
    assert not (initial.work / "prerequisites/recovery_delivery.json").exists()


@pytest.mark.parametrize("epochs", [-1, True, 20])
def test_recovery_rejects_invalid_or_complete_epoch_counts(initial, epochs):
    initial.report["completed_epochs"] = epochs
    _write(initial.parent / "training/run.json", initial.report)
    initial.reseal()
    with pytest.raises(ValueError, match="incomplete committed training"):
        initial.restore()


@pytest.mark.parametrize("field,value", [
    ("effective_maximum_epochs", 21),
    ("calibration_outcomes_accessed", True),
    ("test_outcomes_accessed", True),
])
def test_initial_recovery_preserves_original_budget_and_protected_outcome_boundary(initial, field, value):
    initial.report[field] = value
    _write(initial.parent / "training/run.json", initial.report)
    initial.reseal()
    with pytest.raises(ValueError, match="incomplete committed training"):
        initial.restore()


@pytest.mark.parametrize("artifact", ["training/latest.pt", "training/run.json"])
def test_initial_recovery_requires_committed_latest_and_report(initial, artifact):
    initial.names.remove(artifact)
    initial.reseal()
    with pytest.raises(ValueError, match="committed recovery artifact missing: " + artifact):
        initial.restore()
