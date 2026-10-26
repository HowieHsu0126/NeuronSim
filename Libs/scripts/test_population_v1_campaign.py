"""V1 frozen-source staging and actual fresh-process execution/recovery."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from population_dynamics import v1_population_training_data as consumer
from population_dynamics.test_v1_population_training_data import inputs, configuration

spec = importlib.util.spec_from_file_location("population_v1_campaign", Path(__file__).with_name("run_population_trajectory_campaign.py"))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
REPO = Path(__file__).resolve().parents[2]


def healthy(path, required_memory_gib):
    return dict(status="PASS", reasons=[])


@pytest.fixture
def source(tmp_path):
    original = tmp_path / "original"; original.mkdir()
    manifest = inputs(original)
    # An unavailable protected shard stays in the immutable catalog metadata.
    catalog_path = original / "dataset.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["circuits"]["protected"] = dict(catalog["circuits"]["v1_fit"], path="protected/units.npz")
    catalog["shards"].append(dict(path="protected/never_read.h5", sha256="f"*64,
        episodes=[dict(episode_id="protected_test", circuit_id="protected", block_id="protected",
                       split="test", duration_ticks=40, dt_ms=.25, event_count=0)]))
    catalog["complete_episodes"] += 1
    catalog_path.write_text(json.dumps(catalog))
    repo = tmp_path / "repo"; configs = repo / "Libs/configs"; configs.mkdir(parents=True)
    config = configuration(); config.update(maximum_epochs=1, early_stopping_patience=1)
    (configs/campaign.V1_CONFIGS[0]).write_text(yaml.safe_dump(config))
    forecast = yaml.safe_load((REPO/"Libs/configs"/campaign.V1_CONFIGS[1]).read_text())
    forecast.update(origins_ticks=[10,20], horizons_bins=[1,2], support_windows=[1,2], joint_samples=3)
    (configs/campaign.V1_CONFIGS[1]).write_text(yaml.safe_dump(forecast))
    # Reuse the actual software files; only the declared fixture configs differ.
    (repo/"Libs/population_dynamics").symlink_to(REPO/"Libs/population_dynamics", target_is_directory=True)
    (repo/"Libs/scripts").symlink_to(REPO/"Libs/scripts", target_is_directory=True)
    shutil.copyfile(REPO/"Libs/__init__.py", repo/"Libs/__init__.py")
    shutil.copyfile(REPO/"Libs/configs/neural_population_pfc_recurrent_gain.yaml",
                    configs/"neural_population_pfc_recurrent_gain.yaml")
    return SimpleNamespace(repo=repo, inputs=manifest, original=original, config=config, forecast=forecast)


@pytest.fixture
def prepared(source, tmp_path, monkeypatch):
    monkeypatch.delenv("NEURONSIM_EXECUTION_PATH_MAP", raising=False)
    output = tmp_path / "prepared"
    campaign.prepare_v1(source.repo, source.inputs, output)
    monkeypatch.setattr(campaign, "staging_probe", lambda work: healthy)
    return source, output


def test_staged_v1_preserves_original_catalog_and_selected_cohort(prepared, tmp_path, monkeypatch):
    source, prepared_path = prepared
    expected = consumer.V1PopulationTrainingCohort(source.inputs, source.config).binding
    work = tmp_path / "work"; work.mkdir()
    _, proof = campaign.stage_v1(prepared_path, work)
    assert (work/"inputs/cohort.json").read_bytes() == source.inputs.read_bytes()
    assert (work/"inputs/dataset/dataset.json").read_bytes() == (source.original/"dataset.json").read_bytes()
    assert not (work/"inputs/dataset/protected").exists()
    assert all("protected" not in row["path"] for row in proof["input_files"])
    source.original.rename(tmp_path/"unavailable_original")
    monkeypatch.setenv("NEURONSIM_EXECUTION_PATH_MAP", str(work/"execution_path_map.json"))
    actual = consumer.V1PopulationTrainingCohort(work/"inputs/cohort.json", source.config)
    assert actual.binding == expected == proof["cohort_binding"]
    assert len(actual.fit) == 2 and len(actual.validation) == 1
    for case in [*actual.fit, *actual.validation]:
        assert actual.window(case).counts.shape == (8,3)


@pytest.mark.parametrize("split", ["train", "calibration", "test"])
def test_prepare_requires_original_validation_before_payloads_or_output(tmp_path, split):
    original = tmp_path/"input"; original.mkdir()
    manifest = inputs(original, validation_split=split)
    with (
        patch.object(consumer, "V1RenewalEpisode", side_effect=AssertionError("source arrays read")),
        patch.object(consumer, "sha256_file", side_effect=AssertionError("parameter hash")),
    ):
        with pytest.raises(ValueError, match="ORIGINAL_VALIDATION"):
            campaign.prepare_v1(REPO, manifest, tmp_path/"must_not_exist")
    assert not (tmp_path/"must_not_exist").exists()


@pytest.mark.parametrize("change", ["catalog", "parameters", "software", "runtime_map"])
def test_stage_rejects_changed_frozen_inputs(prepared, tmp_path, change):
    source, root = prepared
    proof = json.loads((root/"preparation.json").read_text())
    if change == "catalog":
        path = source.original/"dataset.json"; path.write_text(path.read_text()+" ")
    elif change == "parameters":
        path = Path(next(row["source"] for row in proof["input_files"] if row["path"].startswith("inputs/parameters/")))
        path.write_text(path.read_text()+" ")
    elif change == "software":
        path = root/"Libs/population_dynamics/v1_population_training_data.py"
        path.write_text(path.read_text()+"# changed\n")
    else:
        proof["runtime_paths"]["dataset"]["sha256"] = "f"*64
        campaign.write_json(root/"preparation.json", proof)
    work=tmp_path/"work";work.mkdir()
    with pytest.raises(ValueError, match="identity changed|RUNTIME_MAP_IDENTITY"):
        campaign.stage_v1(root,work)


def test_staging_space_rejection_precedes_input_copy(prepared, tmp_path, monkeypatch):
    _, root = prepared
    monkeypatch.setattr(campaign.shutil,"disk_usage",lambda path:SimpleNamespace(free=1))
    work=tmp_path/"work";work.mkdir()
    with pytest.raises(campaign.ResourceBlocked, match="required_bytes=.*free_bytes=1"):
        campaign.stage_v1(root,work)
    assert not (work/"inputs").exists()


def test_storage_uses_every_circuit_unit_and_origin():
    forecast=dict(origins_ticks=[10,20],horizons_bins=[1,4],joint_samples=3)
    dimensions={"small":dict(units=3,fields=2),"large":dict(units=230924,fields=21)}
    cases=[dict(circuit_id="small"),dict(circuit_id="large"),dict(circuit_id="large")]
    size=campaign.v1_output_storage_bytes(cases,dimensions,forecast,4)
    assert size > (3+230924*2)*3*4*2*8 + 21*1024**3
    assert size > campaign.v1_output_storage_bytes(cases[:2],dimensions,forecast,4)


def identity(monkeypatch, *, job="901"):
    monkeypatch.setenv("SLURM_JOB_ID",job)
    monkeypatch.setattr(campaign,"execution_identity",lambda *args:dict(
        seed=41,task=0,array_id="900",node="cpu04",partition="cpu",replaces_job=None))


def report_for(source, *, recovered=False, job="901"):
    base=source.repo/campaign.V1_CAMPAIGN/"run_900"
    return base/"resource_attempts"/("seed_41_job_"+job) if recovered else base/"seed_41"


def assert_success(path):
    report=json.loads((path/"delivery.json").read_text())
    assert report["status"]=="PASS",report
    assert report["core_neurons"]==3 and report["complete_episodes"]==3
    assert report["fit_episodes"]==2 and report["validation_episodes"]==1
    assert report["verified_forecast_origins"]==2 and report["score_rows"]==4
    assert json.loads((path/"replay.json").read_text())["unit_values_consumed"]==36


def test_actual_v1_campaign_runs_fresh_training_evaluation_and_replay(prepared,tmp_path,monkeypatch):
    source, root=prepared
    identity(monkeypatch)
    work=tmp_path/"work";work.mkdir()
    code=campaign.run(source.repo,work,"cpu",v1_preparation=root)
    target=report_for(source)
    assert code==0,json.loads((target/"delivery.json").read_text())
    assert_success(target)
    for name in ("fit","evaluate","replay"):
        receipt=json.loads((target/"phases"/name/"receipt.json").read_text())
        assert receipt["status"]=="PASS" and receipt["command_started"] is True
    assert not (target/"inputs").exists()


@pytest.mark.parametrize("stopped_phase",["evaluate","replay"])
def test_v1_recovery_reuses_completed_real_phase_outputs(prepared,tmp_path,monkeypatch,stopped_phase):
    source,root=prepared
    identity(monkeypatch)
    phase=campaign.phase
    def stop(work,name,command,probe):
        if name==stopped_phase:
            campaign.write_json(work/"phases"/name/"receipt.json",
                                dict(status="RESOURCE_BLOCKED",exit_code=None,command_started=False))
            return 3
        return phase(work,name,command,probe)
    monkeypatch.setattr(campaign,"phase",stop)
    work=tmp_path/"work";work.mkdir()
    assert campaign.run(source.repo,work,"cpu",v1_preparation=root)==3
    parent=report_for(source);before=(parent/"delivery.json").read_bytes()
    checkpoint=campaign.digest(parent/"training/best.pt")
    calls=[]
    def remaining(work,name,command,probe):
        calls.append(name);return phase(work,name,command,probe)
    monkeypatch.setattr(campaign,"phase",remaining)
    identity(monkeypatch,job="902")
    work=tmp_path/"resumed";work.mkdir()
    code=campaign.run(source.repo,work,"cpu",v1_preparation=root,recovery_from=parent)
    target=report_for(source,recovered=True,job="902")
    assert code==0,json.loads((target/"delivery.json").read_text())
    assert calls==(["evaluate","replay"] if stopped_phase=="evaluate" else ["replay"])
    assert campaign.digest(target/"training/best.pt")==checkpoint
    assert (parent/"delivery.json").read_bytes()==before
    assert_success(target)
    reused=json.loads((target/"phases/fit/receipt.json").read_text())
    assert reused["original_phase_command_not_repeated"] is True


@pytest.mark.parametrize("node,partition",[("cpu03","gpu"),("cpu04","cpu"),("cpu05","gpu"),("cpu06","gpu"),("gpu03","gpu1")])
def test_authorized_gpu_nodes_keep_single_seed_allocation(monkeypatch,node,partition):
    for key,value in dict(SLURM_JOB_ID="100",SLURMD_NODENAME=node,SLURM_JOB_PARTITION=partition,
                          SLURM_JOB_GPUS="0",SLURM_ARRAY_TASK_ID="0",SLURM_ARRAY_JOB_ID="99").items():
        monkeypatch.setenv(key,value)
    result=campaign.execution_identity("cuda")
    assert result["seed"]==41 and result["node"]==node


def test_prepare_v1_cli_freezes_declared_source_and_never_submits(source,tmp_path):
    output=tmp_path/"cli_prepared"
    result=subprocess.run([sys.executable,"-B",str(REPO/"Libs/scripts/run_population_trajectory_campaign.py"),
        "--repo",str(source.repo),"--prepare-v1","--v1-inputs",str(source.inputs),"--output",str(output)],
        capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    proof=json.loads((output/"preparation.json").read_text())
    assert not proof["training_submitted"] and proof["fitting_episodes"]==2 and proof["validation_episodes"]==1
