"""Manifest-bound recovery of a stopped BRAID fit through its existing caller."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "braid_campaign", Path(__file__).parents[1] / "scripts/run_braid_trajectory_campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


@pytest.fixture
def stopped(tmp_path):
    parent = tmp_path / "parent"
    (parent / "training").mkdir(parents=True)
    campaign.write(parent / "training/run.json", dict(training_seed=41, status="RUNNING",
        completed_epochs=4, stop_reason=None))
    campaign.write(parent / "training/committed.json", dict(checkpoint="checkpoints/epoch_003"))
    (parent / "training/checkpoints").mkdir()
    (parent / "training/checkpoints/model").write_bytes(b"committed model and optimizer fixture")
    campaign.write(parent / "fit_receipt.json", dict(status="RESOURCE_BLOCKED", exit_code=-15))
    rows = [dict(path=str(p.relative_to(parent)), bytes=p.stat().st_size, sha256=campaign.digest(p))
            for p in parent.rglob("*") if p.is_file()]
    campaign.write(parent / "delivery.json", dict(status="RESOURCE_BLOCKED", training_seed=41,
        preparation_sha256="bound preparation", scientific_claim_qualified=False, files=rows))
    cell = tmp_path / "cell"
    cell.mkdir()
    return parent, cell


def test_recovery_copies_committed_fit_and_preserves_parent(stopped):
    parent, cell = stopped
    before = {str(p.relative_to(parent)): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    report = campaign.stage_recovery(parent, cell, 41, "bound preparation")
    assert report["committed_epochs"] == 4
    for name in before:
        if name.startswith("training/"):
            assert (cell / name).read_bytes() == before[name]
        assert (parent / name).read_bytes() == before[name]
    assert (cell / "parent_fit_receipt.json").is_file()
    assert (cell / "recovery_delivery.json").is_file()


def test_recovery_rejects_foreign_seed_source_and_changed_checkpoint(stopped):
    parent, cell = stopped
    for seed, preparation in ((42, "bound preparation"), (41, "different preparation")):
        with pytest.raises(ValueError, match="same-seed"):
            campaign.stage_recovery(parent, cell, seed, preparation)
    (parent / "training/checkpoints/model").write_bytes(b"changed model")
    with pytest.raises(ValueError, match="artifact changed"):
        campaign.stage_recovery(parent, cell, 41, "bound preparation")
