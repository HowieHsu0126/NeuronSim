"""Retain failed source/readout attempts and reuse an explicit complete source."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import pfc_parameter_domain_execution as execution


@pytest.fixture
def task(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture_job")
    monkeypatch.setenv("SLURMD_NODENAME", "cpu05")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.setattr(execution, "qualified_mapping", lambda p: ({}, {}))
    monkeypatch.setattr(execution, "load_task", lambda *a: ({},
        dict(identity={"episode_id": "declared_episode"}, config_sha256="bound_recipe"), Path("recipe")))
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("frozen campaign")
    return dict(source_project=tmp_path / "source_software", observation_project=tmp_path / "observation_software",
                prepared=tmp_path / "prepared", qualifications=tmp_path / "qualifications.json", index=7,
                source_output=tmp_path / "sources", observation_output=tmp_path / "observations",
                campaign=campaign, output=tmp_path / "attempt")


def test_source_failure_stops_readout_and_retains_nonzero_exit(task, monkeypatch):
    calls = []
    monkeypatch.setattr(execution.subprocess, "run", lambda c, **kw:
                        (calls.append((c, kw)) or SimpleNamespace(returncode=17)))
    with pytest.raises(RuntimeError, match="SOURCE_EXIT:17"):
        execution.run_task(**task)
    report = json.loads((task["output"] / "report.json").read_text())
    assert len(calls) == 1 and calls[0][0][4] == "source"
    assert calls[0][1]["env"]["PYTHONPATH"] == str(task["source_project"] / "Libs")
    assert report["source_exit_code"] == 17 and report["observation_exit_code"] is None
    assert report["completed_episodes"] == report["production_episodes"] == 0


def test_readout_failure_retains_completed_source_path_for_explicit_reuse(task, monkeypatch):
    codes = iter([0, 23])
    monkeypatch.setattr(execution.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=next(codes)))
    with pytest.raises(RuntimeError, match="OBSERVATION_EXIT:23"):
        execution.run_task(**task)
    report = json.loads((task["output"] / "report.json").read_text())
    assert report["source_exit_code"] == 0 and report["observation_exit_code"] == 23
    assert report["source_archive"] == str(task["source_output"] / "run_fixture_job")
    assert report["completed_episodes"] == 0
    with pytest.raises(FileExistsError, match="ATTEMPT_EXISTS"):
        execution.run_task(**task)


def test_explicit_source_reuse_runs_only_frozen_observation(task, monkeypatch):
    archive = task["source_output"] / "run_original_job"
    archive.mkdir(parents=True)
    (archive / "delivery.json").write_text("{}"); calls = []
    monkeypatch.setattr(execution.subprocess, "run", lambda c, **kw:
                        (calls.append((c, kw)) or SimpleNamespace(returncode=0)))
    result = execution.run_task(**task, source_archive=archive)
    assert len(calls) == 1 and calls[0][0][4] == "readout"
    assert calls[0][1]["env"]["PYTHONPATH"] == str(task["observation_project"] / "Libs")
    assert calls[0][0][calls[0][0].index("--source") + 1] == str(archive)
    assert result["new_source_calls"] == 0 and result["source_exit_code"] is None
    assert result["completed_episodes"] == 1 and not result["source_qualification_granted"]
    assert result["production_episodes"] == 0


def test_missing_reused_source_fails_before_attempt_or_invocation(task, monkeypatch):
    monkeypatch.setattr(execution.subprocess, "run", lambda *a, **kw: pytest.fail("started an invalid source"))
    with pytest.raises(ValueError, match="COMPLETE_SOURCE_ARCHIVE_REQUIRED"):
        execution.run_task(**task, source_archive=task["source_output"] / "missing")
    assert not task["output"].exists()


def test_existing_task_requires_explicit_source_reuse(task, monkeypatch):
    task["source_output"].mkdir()
    monkeypatch.setattr(execution.subprocess, "run", lambda *a, **kw: pytest.fail("repeated existing task"))
    with pytest.raises(FileExistsError, match="EXISTING_SOURCE_OR_OBSERVATION"):
        execution.run_task(**task)
    assert not task["output"].exists()
