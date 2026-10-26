"""PFC resource recovery preserves parent evidence and completed phases."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pytest
import yaml


spec = importlib.util.spec_from_file_location("population_pfc_recovery", Path(__file__).with_name("run_population_trajectory_campaign.py"))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def parent_fixture(root, *, epoch=2, completed=(), committed=True):
    parent, work, preparation = root / "parent", root / "work", root / "prepared"
    parent.mkdir(parents=True)
    work.mkdir()
    preparation.mkdir()
    campaign.write_json(preparation / "preparation.json", {"frozen_fixture": True})
    config = dict(maximum_epochs=20, device="cpu", cpu_threads=2, representation="recurrent")
    campaign.write_json(work / "training_configuration.json", config)
    campaign.write_json(parent / "training_configuration.json", config)
    module = parent / "Libs/population_dynamics/fixture.py"
    module.parent.mkdir(parents=True)
    module.write_text("# Frozen numerical implementation fixture.\n")
    target = work / "Libs/population_dynamics/fixture.py"
    target.parent.mkdir(parents=True)
    shutil.copy2(module, target)
    if committed:
        training = parent / "training"
        training.mkdir()
        (training / "latest.pt").write_bytes(b"opaque committed checkpoint; launcher never deserializes it")
        if epoch:
            (training / "best.pt").write_bytes(b"opaque best checkpoint")
            campaign.write_json(training / "epochs.json", [{"epoch": i} for i in range(epoch)])
        campaign.write_json(training / "run.json", dict(training_seed=41, completed_epochs=epoch,
            best_epoch=epoch - 1 if epoch else None, effective_maximum_epochs=20,
            status="FITTED_DEVELOPMENT" if "fit" in completed else "RUNNING",
            fitting_episodes=80, validation_episodes=30, stop_reason="epoch_budget" if "fit" in completed else None,
            calibration_outcomes_accessed=False, test_outcomes_accessed=False))
    else:
        campaign.write_json(parent / "training/run.json", dict(status="RUNNING", completed_epochs=0))
        (parent / "training/optimization.csv").write_text("partial diagnostic\n")
    for name in completed:
        campaign.write_json(parent / "phases" / name / "receipt.json", dict(
            status="PASS", exit_code=0, phase=name, command=["original", name]))
    if "evaluate" in completed:
        forecasts = parent / "forecasts"
        forecasts.mkdir()
        (forecasts / "counts.npz").write_bytes(b"opaque saved all-unit outputs")
        campaign.write_json(forecasts / "evaluation.json", dict(
            status="EVALUATED_PFC_DEVELOPMENT_CHECKPOINT", score_rows=360,
            origins=[{"fixture_origin": i} for i in range(120)],
            checkpoint_sha256=campaign.digest(parent / "training/best.pt")))
    if "replay" in completed:
        campaign.write_json(parent / "replay.json", dict(
            status="PASS_ALL_UNIT_AND_INDEPENDENT_METRIC_REPLAY", origins=list(range(120))))
    rows = [dict(path=str(path.relative_to(parent)), bytes=path.stat().st_size, sha256=campaign.digest(path))
            for path in sorted(parent.rglob("*")) if path.is_file()]
    campaign.write_json(parent / "delivery.json", dict(status="RESOURCE_BLOCKED", exit_code=3,
        input_family="pfc_source_reproduction", training_seed=41, array_job_id="123", job_id="124",
        device="cpu", representation="recurrent", preparation_sha256=campaign.digest(preparation / "preparation.json"),
        calibration_or_test_accessed=False, scientific_claim_qualified=False, files=rows))
    return parent, work, preparation


@pytest.mark.parametrize("epoch,completed,resume", [
    (0, (), True), (3, (), True), (20, ("fit",), False),
    (20, ("fit", "evaluate"), False), (20, ("fit", "evaluate", "replay"), False),
])
def test_committed_boundaries_restore_and_skip_completed_phases(tmp_path, epoch, completed, resume):
    parent, work, prepared = parent_fixture(tmp_path, epoch=epoch, completed=completed)
    before = (parent / "delivery.json").read_bytes()
    result = campaign.stage_pfc_recovery(parent, work, 41, "123", "cpu", "recurrent",
                                        campaign.digest(prepared / "preparation.json"))
    assert result["resume_training"] is resume
    assert result["completed_phases"] == list(completed)
    assert (work / "training/latest.pt").read_bytes() == (parent / "training/latest.pt").read_bytes()
    if epoch == 0:
        assert not (work / "training/best.pt").exists()
    for name in completed:
        receipt = json.loads((work / "phases" / name / "receipt.json").read_text())
        assert receipt["status"] == "PASS" and receipt["original_phase_command_not_repeated"]
    if "evaluate" in completed:
        assert (work / "forecasts/counts.npz").read_bytes() == (parent / "forecasts/counts.npz").read_bytes()
    assert (parent / "delivery.json").read_bytes() == before


def test_stop_before_initial_commit_keeps_diagnostics_outside_fresh_training(tmp_path):
    parent, work, prepared = parent_fixture(tmp_path, committed=False)
    result = campaign.stage_pfc_recovery(parent, work, 41, "123", "cpu", "recurrent",
                                        campaign.digest(prepared / "preparation.json"))
    assert not result["resume_training"] and result["completed_phases"] == []
    assert not (work / "training").exists()
    assert (work / "prerequisites/uncommitted_attempt/training/optimization.csv").read_text() == "partial diagnostic\n"


@pytest.mark.parametrize("change,match", [
    ("seed", "SAME_IDENTITY"), ("preparation", "SAME_IDENTITY"),
    ("software", "SOFTWARE_OR_CONFIGURATION"), ("checkpoint", "source manifest identity changed"),
    ("config", "TRAINING_CONFIGURATION"),
])
def test_recovery_rejects_changed_scientific_bindings(tmp_path, change, match):
    parent, work, prepared = parent_fixture(tmp_path)
    seed, digest = 41, campaign.digest(prepared / "preparation.json")
    if change == "seed":
        seed = 42
    elif change == "preparation":
        digest = "b" * 64
    elif change == "software":
        (work / "Libs/population_dynamics/fixture.py").write_text("changed")
    elif change == "checkpoint":
        (parent / "training/latest.pt").write_bytes(b"corrupt")
    else:
        campaign.write_json(work / "training_configuration.json", dict(maximum_epochs=1))
    with pytest.raises(ValueError, match=match):
        campaign.stage_pfc_recovery(parent, work, seed, "123", "cpu", "recurrent", digest)


@pytest.mark.parametrize("completed", [(), ("fit",), ("fit", "evaluate")])
def test_run_recovers_to_separate_archive_and_executes_only_remaining_phases(tmp_path, monkeypatch, completed):
    parent, fixture_work, prepared = parent_fixture(tmp_path, epoch=20 if completed else 2, completed=completed)
    repo = tmp_path / "repo"
    original = repo / campaign.PFC_CAMPAIGN / "run_123/seed_41"
    original.parent.mkdir(parents=True)
    shutil.copytree(parent, original)
    original_proof = (original / "delivery.json").read_bytes()
    work = Path("/tmp/neuronsim") / ("pfc-recover-contract-" + tmp_path.name)
    calls, archived = [], {}
    monkeypatch.setattr(campaign, "execution_identity", lambda *a: dict(seed=41, task=0, array_id="123",
        node="cpu04", partition="cpu", replaces_job="123_0"))
    monkeypatch.setenv("SLURM_JOB_ID", "125")
    def stage(preparation, local):
        local.mkdir(parents=True)
        shutil.copytree(fixture_work / "Libs", local / "Libs")
        config = local / "Libs/configs"
        config.mkdir()
        (config / campaign.PFC_CONFIGS[0]).write_text(yaml.safe_dump(dict(maximum_epochs=20)))
        (config / campaign.PFC_CONFIGS[1]).write_text(yaml.safe_dump(dict(joint_samples=16,
            origins_ticks=[2000, 8000, 48000, 80000], horizons_bins=[4, 16, 100])))
        return lambda *a, **k: dict(status="PASS", reasons=[]), {}
    def phase(local, name, command, probe):
        calls.append((name, command))
        if name == "fit":
            assert "--resume" in command
            campaign.write_json(local / "training/run.json", dict(status="FITTED_DEVELOPMENT", training_seed=41,
                fitting_episodes=80, validation_episodes=30, completed_epochs=20, stop_reason="epoch_budget"))
        elif name == "evaluate":
            campaign.write_json(local / "forecasts/evaluation.json", dict(
                status="EVALUATED_PFC_DEVELOPMENT_CHECKPOINT", score_rows=360))
        else:
            campaign.write_json(local / "replay.json", dict(
                status="PASS_ALL_UNIT_AND_INDEPENDENT_METRIC_REPLAY", origins=list(range(120))))
        return 0
    monkeypatch.setattr(campaign, "stage_pfc", stage)
    monkeypatch.setattr(campaign, "phase", phase)
    monkeypatch.setattr(campaign, "archive", lambda local, target, record: archived.update(target=str(target), **record))
    try:
        assert campaign.run(repo, work, "cpu", 41, "123", "123_0", recovery_from=original, pfc_preparation=prepared) == 0
        assert [name for name, _ in calls] == [name for name in ("fit", "evaluate", "replay") if name not in completed]
        assert archived["target"] == str(repo / campaign.PFC_CAMPAIGN / "run_123/resource_attempts/seed_41_job_125")
        assert archived["array_job_id"] == "123" and archived["training_seed"] == 41
        assert (original / "delivery.json").read_bytes() == original_proof
    finally:
        if work.exists():
            shutil.rmtree(work)
