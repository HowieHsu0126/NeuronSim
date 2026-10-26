"""Frozen PFC staging, execution dispatch and partial-cohort rejection."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import os
import subprocess
from types import SimpleNamespace

import pytest
import yaml


spec = importlib.util.spec_from_file_location("population_campaign", Path(__file__).with_name("run_population_trajectory_campaign.py"))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def frozen_preparation(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    source = tmp_path / "original.json"
    source.write_text('{"original":true}\n')
    probe = root / "Libs/population_dynamics/resource_probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("def probe(path, required_memory_gib):\n    return {'status':'PASS','reasons':[]}\n")
    row = dict(path=str(probe.relative_to(root)), bytes=probe.stat().st_size, sha256=campaign.digest(probe))
    caller = root / "Libs/scripts/run_population_trajectory_campaign.py"
    caller.parent.mkdir(parents=True)
    shutil.copy2(Path(campaign.__file__), caller)
    caller_row = dict(path=str(caller.relative_to(root)), bytes=caller.stat().st_size, sha256=campaign.digest(caller))
    copy = campaign.input_copy(source, "inputs/cohort.json", campaign.digest(source))
    proof = dict(schema="pfc_population_campaign_preparation_v1", status="PREPARED_COMPLETE_DEVELOPMENT_COHORT",
        training_seeds=[41, 42, 43], fitting_episodes=80, validation_episodes=30,
        files=[row, caller_row], input_files=[copy], paths={str(source): dict(path="inputs/cohort.json", sha256=copy["sha256"])},
        input_sha256=copy["sha256"], input_bytes=copy["bytes"], output_storage_bytes=100)
    campaign.write_json(root / "preparation.json", proof)
    return root, source, proof


def test_staging_preserves_input_and_explicit_map(tmp_path):
    prepared, source, proof = frozen_preparation(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    probe, actual = campaign.stage_pfc(prepared, work)
    assert actual == proof
    assert campaign.digest(work / "inputs/cohort.json") == campaign.digest(source)
    mapping = json.loads((work / "execution_path_map.json").read_text())
    assert mapping["inputs_sha256"] == proof["input_sha256"]
    assert mapping["paths"][str(source)]["path"] == str(work / "inputs/cohort.json")
    assert source.read_text() == '{"original":true}\n'


@pytest.mark.parametrize("mutation,match", [
    ("software", "source manifest identity changed"),
    ("input", "source manifest identity changed"),
    ("space", "ALL_UNIT_OUTPUT_SPACE"),
    ("escape", "RELATIVE_INPUT_PATH"),
    ("incomplete", "FROZEN_COMPLETE_PREPARATION"),
])
def test_staging_rejects_invalid_preparation(tmp_path, monkeypatch, mutation, match):
    prepared, source, proof = frozen_preparation(tmp_path)
    if mutation == "software":
        (prepared / proof["files"][0]["path"]).write_text("changed")
    elif mutation == "input":
        source.write_text("changed")
    elif mutation == "space":
        monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    elif mutation == "escape":
        proof["input_files"][0]["path"] = "../escape"
    else:
        proof["fitting_episodes"] = 30
    campaign.write_json(prepared / "preparation.json", proof)
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(ValueError, match=match):
        campaign.stage_pfc(prepared, work)


def test_prepare_rejects_partial_coverage_before_writes(tmp_path, monkeypatch):
    from population_dynamics import pfc_training_inputs
    manifest = dict(schema="pfc_population_development_inputs_v1", source_entry="unused", episodes=[])
    coverage = dict(baseline_inputs="unused", baseline_validation="unused", plan="unused/plan",
        reader_jobs=[], fitting_episodes=30, validation_episodes=10, fit_validation_coverage_ready=False)
    monkeypatch.setattr(pfc_training_inputs, "assemble", lambda *args: (manifest, coverage))
    inputs, receipt, output = tmp_path / "inputs.json", tmp_path / "coverage.json", tmp_path / "output"
    campaign.write_json(inputs, manifest)
    campaign.write_json(receipt, dict(coverage, inputs=str(inputs), inputs_sha256=campaign.digest(inputs)))
    with pytest.raises(ValueError, match="80_FIT_30_VALIDATION"):
        campaign.prepare_pfc(Path(__file__).resolve().parents[2], inputs, receipt, output)
    assert not output.exists()


def test_output_space_includes_all_units_and_origins():
    forecast = dict(origins_ticks=[2000, 8000, 48000, 80000], horizons_bins=[4, 16, 100], joint_samples=16)
    bound = campaign.pfc_output_storage_bytes(30, forecast, 4, 16)
    assert bound > 30 * 4 * 16 * 100 * 100000 * 8
    assert campaign.pfc_output_storage_bytes(60, forecast, 4, 16) > bound


def test_software_closure_includes_pfc_and_shared_entry_points():
    repo = Path(__file__).resolve().parents[2]
    files = campaign.software_closure(repo, ["population_trajectory_training", "population_trajectory_evaluation",
        "population_forecast_replay", "pfc_population_evaluation", "pfc_training_inputs"])
    for name in ("source_population_evaluation", "v1_population_evaluation",
                 "source_population_training", "v1_population_training_data", "v1_renewal_data",
                 "pfc_population_training_data", "pfc_renewal_data", "circuit_data", "pfc_scale_analysis",
                 "pfc_source_dataset", "pfc_kernel_parameters", "pfc_recurrent_gain_comparison", "benchmark_tasks"):
        assert "Libs/population_dynamics/" + name + ".py" in files
    assert len(files) == len(set(files))


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("flag", ["--pfc-preparation", "--v1-preparation"])
def test_wrapper_uses_frozen_source_caller(tmp_path, inline, flag):
    binary = tmp_path / "bin"
    binary.mkdir()
    output = tmp_path / "arguments.txt"
    srun = binary / "srun"
    srun.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$PFC_TEST_ARGUMENTS"\n'
        'for ((i=1;i<=$#;i++)); do if [ "${!i}" = --work ]; then j=$((i+1)); touch "${!j}/archive_complete"; fi; done\n')
    srun.chmod(0o755)
    preparation = tmp_path / "prepared"
    arguments = [flag + "=" + str(preparation)] if inline else [flag, str(preparation)]
    environment = dict(os.environ, PATH=str(binary) + ":" + os.environ["PATH"], SLURM_JOB_ID="pfc-wrapper-fixture",
        SLURMD_NODENAME="cpu04", SLURM_JOB_PARTITION="cpu", SLURM_JOB_GPUS="", PFC_TEST_ARGUMENTS=str(output))
    result = subprocess.run(["bash", str(Path(campaign.__file__).with_suffix(".sbatch")), "cpu", *arguments],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    passed = output.read_text().splitlines()
    assert passed[1] == str(preparation / "Libs/scripts/run_population_trajectory_campaign.py")
    assert not Path(passed[passed.index("--work") + 1]).exists()


def test_phase_passes_local_map_to_fresh_child(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "scratch").mkdir()
    mapping = work / "execution_path_map.json"
    mapping.write_text("{}")
    command = [sys.executable, "-c", "import os; from pathlib import Path; "
        "Path('child_map.txt').write_text(os.environ['NEURONSIM_EXECUTION_PATH_MAP'])"]
    assert campaign.phase(work, "map", command, lambda *a, **k: dict(status="PASS", reasons=[])) == 0
    assert (work / "child_map.txt").read_text() == str(mapping)


def test_staging_resource_rejection_preserves_attempt_before_children(tmp_path, monkeypatch):
    repo, prepared, work = tmp_path / "repo", tmp_path / "prepared", Path("/tmp/neuronsim") / ("pfc-block-" + tmp_path.name)
    prepared.mkdir()
    campaign.write_json(prepared / "preparation.json", {"frozen_fixture": True})
    monkeypatch.setattr(campaign, "execution_identity", lambda *a: dict(seed=41, task=0, array_id="123",
        node="cpu04", partition="cpu", replaces_job=None))
    monkeypatch.setenv("SLURM_JOB_ID", "124")
    monkeypatch.setattr(campaign, "stage_pfc", lambda *a: (_ for _ in ()).throw(campaign.ResourceBlocked(["PERSISTENT_SWAPPING"])))
    observed = {}
    monkeypatch.setattr(campaign, "archive", lambda local, target, record: observed.update(record))
    try:
        work.mkdir()
        assert campaign.run(repo, work, "cpu", pfc_preparation=prepared) == 3
        assert observed["status"] == "RESOURCE_BLOCKED"
        assert observed["reasons"] == ["PERSISTENT_SWAPPING"]
        assert observed["complete_episodes"] == 110 and observed["fit_episodes"] == 80
        assert (work / "archive_complete").exists()
        assert not (work / "training").exists()
    finally:
        if work.exists():
            shutil.rmtree(work)


def test_pfc_dispatch_retains_training_evaluation_replay_denominators(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    work = Path("/tmp/neuronsim") / ("pfc-dispatch-" + tmp_path.name)
    commands, archive = [], {}
    monkeypatch.setattr(campaign, "execution_identity", lambda *a: dict(seed=41, task=0, array_id="123",
        node="cpu04", partition="cpu", replaces_job=None))
    monkeypatch.setenv("SLURM_JOB_ID", "124")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    campaign.write_json(prepared / "preparation.json", {})
    def stage(preparation, local):
        config = local / "Libs/configs"
        config.mkdir(parents=True)
        (config / campaign.PFC_CONFIGS[0]).write_text(yaml.safe_dump(dict(maximum_epochs=20)))
        (config / campaign.PFC_CONFIGS[1]).write_text(yaml.safe_dump(dict(joint_samples=16,
            origins_ticks=[2000, 8000, 48000, 80000], horizons_bins=[4, 16, 100])))
        return lambda *a, **k: dict(status="PASS", reasons=[]), {}
    def phase(local, name, command, probe):
        commands.append(command)
        if name == "fit":
            campaign.write_json(local / "training/run.json", dict(status="FITTED_DEVELOPMENT", training_seed=41,
                fitting_episodes=80, validation_episodes=30, completed_epochs=20, stop_reason="epoch_budget"))
        elif name == "evaluate":
            campaign.write_json(local / "forecasts/evaluation.json", dict(status="EVALUATED_PFC_DEVELOPMENT_CHECKPOINT", score_rows=360))
        else:
            campaign.write_json(local / "replay.json", dict(status="PASS_ALL_UNIT_AND_INDEPENDENT_METRIC_REPLAY", origins=list(range(120))))
        return 0
    monkeypatch.setattr(campaign, "stage_pfc", stage)
    monkeypatch.setattr(campaign, "phase", phase)
    monkeypatch.setattr(campaign, "archive", lambda local, target, record: archive.update(record))
    try:
        assert campaign.run(repo, work, "cpu", pfc_preparation=prepared) == 0
        assert [command[2] for command in commands] == ["population_dynamics.population_trajectory_training",
            "population_dynamics.population_trajectory_evaluation", "population_dynamics.population_forecast_replay"]
        assert all("--fields" not in command and "inputs/cohort.json" in command for command in commands)
        assert archive["verified_forecast_origins"] == 120 and archive["score_rows"] == 360
        assert archive["fit_episodes"] == 80 and archive["validation_episodes"] == 30
        assert archive["scientific_claim_qualified"] is False
    finally:
        if work.exists():
            shutil.rmtree(work)
