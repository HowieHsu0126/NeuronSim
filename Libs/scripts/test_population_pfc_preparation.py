"""Complete receipt-driven PFC preparation with metadata-only source fixtures."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from population_dynamics import pfc_scale_analysis, pfc_training_inputs
from population_dynamics.test_pfc_training_inputs import (
    MetadataFixture, _file_binding, _reader, _reseal, _write,
)


spec = importlib.util.spec_from_file_location(
    "population_complete_preparation", Path(__file__).with_name("run_population_trajectory_campaign.py"))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)

SOURCE_OUTPUTS = (
    "config.yaml", "request.json", "raw.npz", "terminal_state.npz", "units.npz",
    "unit_metadata.json", "initial_voltage.npy", "final_rng.npz",
    "applied_current_observation.npz", "applied_current_metadata.json",
)
PRODUCERS = (
    "pfc_source_execution.py", "pfc_deterministic_replay.py",
    "pfc_native_drive.py", "pfc_kernel_parameters.py",
)
CIRCUIT_IDS = {
    23: "pfc_f878b5b987319436", 63: "pfc_be8cefad129550b6",
    74: "pfc_e7072637ba0e1076", 86: "calibration_reserved",
    52: "pfc_c4cb0a8dcfbeb9c3",
}


def _placeholder(path, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("Metadata fixture only; no scientific arrays: " + label + "\n").encode())
    return path


def _enrich_episode(task, dataset, archive, source_hash):
    """Supply the exact declared-file inventory without executable simulation data."""
    source = archive / "source_run"
    candidate = dict(identity=pfc_training_inputs.identity(task),
                     intended_split=task["split"], condition_id=task["condition_id"])
    config = dict(benchmark_episode_candidate=candidate, source_metadata_fixture=True)
    config_path = source / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config))
    task["config_sha256"] = campaign.digest(config_path)
    outputs = {}
    for name in SOURCE_OUTPUTS:
        path = source / name
        if name != "config.yaml":
            _placeholder(path, task["episode_id"] + "/" + name)
        outputs[name] = _file_binding(path, name)
    report = _write(source / "report.json", dict(
        case_id=task["episode_id"], status="PFC_SOURCE_EXECUTION_COMPLETE",
        source_execution_complete=True, config_sha256=task["config_sha256"],
        resolved_config=config, outputs=outputs,
        code={"kernel_parameterization_sha256": "fixture"},
    ))
    producer_paths = [
        _placeholder(archive / "Libs/population_dynamics" / name, name) for name in PRODUCERS
    ]
    qualification_paths = [
        _write(archive / "qualification/delivery.json", {"metadata_fixture": True}),
        _write(archive / "qualification/comparison/qualification.json", {"metadata_fixture": True}),
    ]
    declared = [report, *[source / name for name in SOURCE_OUTPUTS],
                *producer_paths, *qualification_paths]
    delivery = _write(archive / "delivery.json", dict(
        status="ARCHIVED_PFC_SOURCE_EXECUTION", result_status="PFC_SOURCE_EXECUTION_COMPLETE",
        source_execution_complete=True, exit_code=0, case_id=task["episode_id"],
        files=[_file_binding(path, str(path.relative_to(archive))) for path in declared],
    ))
    circuit_id = CIRCUIT_IDS[int(task["circuit_index"])]
    circuit = _placeholder(dataset / "circuits" / (circuit_id + ".h5"), task["episode_id"] + "/circuit")
    shard = _placeholder(dataset / "episodes_000000.h5", task["episode_id"] + "/episode")
    catalog = _write(dataset / "dataset.json", dict(
        binding=dict(scope="source_reproduction", source_entry_sha256=source_hash,
            config_sha256=task["config_sha256"], resolved_source_config=config,
            source_bindings=[dict(case_id=task["episode_id"], path=str(source),
                delivery_sha256=campaign.digest(delivery), report_sha256=campaign.digest(report))]),
        complete_episodes=1,
        circuits={circuit_id: dict(path=str(circuit.relative_to(dataset)),
            units=100000, sha256=campaign.digest(circuit))},
        shards=[dict(path=str(shard.relative_to(dataset)), sha256=campaign.digest(shard),
            episodes=[dict(episode_id=task["episode_id"], circuit_id=circuit_id,
                duration_ticks=100000, dt_ms=.05, split="development")])],
    ))
    binding = dict(episode_id=task["episode_id"], condition_id=task["condition_id"],
        source_archive=str(archive), delivery_sha256=campaign.digest(delivery))
    return binding, (catalog, circuit, shard)


def _enrich_group(fixture, group, cell, source_hash):
    tasks = fixture.tasks[group * 10:(group + 1) * 10]
    bindings, payloads = [], []
    for task, catalog in zip(tasks, cell["catalogs"]):
        archive = fixture.base / "sources" / task["episode_id"]
        binding, paths = _enrich_episode(task, catalog.parent, archive, source_hash)
        bindings.append(binding)
        payloads.extend(paths[1:])
    report = json.loads((cell["archive"] / "report.json").read_text())
    report["circuits"][0]["source_bindings"] = bindings
    _write(cell["archive"] / "report.json", report)
    receipt = json.loads(cell["receipt"].read_text())
    receipt["sources"] = bindings
    _write(cell["receipt"], receipt)
    delivery = json.loads((cell["archive"] / "delivery.json").read_text())
    delivery["files"].extend(_file_binding(path, str(path.relative_to(cell["archive"])))
                              for path in payloads)
    _write(cell["archive"] / "delivery.json", delivery)
    _reseal(cell)


@pytest.fixture
def complete(tmp_path, monkeypatch):
    fixture = MetadataFixture(tmp_path)
    # The maintained launcher derives the acquisition base from plan.parent.
    fixture.plan = fixture.base / "plan"
    fixture.plan.mkdir()
    source_hash = campaign.digest(fixture.source)
    monkeypatch.setattr(pfc_training_inputs, "SOURCE_ENTRY_SHA256", source_hash)
    monkeypatch.setattr(pfc_scale_analysis, "SOURCE_ENTRY_SHA256", source_hash)
    template = {(int(row["circuit_index"]), row["condition_id"]): row
                for row in fixture.tasks if int(row["block_index"]) == 1}
    baseline_receipts = []
    baseline_by_circuit = {}
    for record in fixture.original["episodes"]:
        circuit = int(record["episode_id"].split("_circuit_")[1][:3])
        offset = int(record["episode_id"].rsplit("_", 1)[1])
        task = deepcopy(template[(circuit, pfc_training_inputs.CONDITIONS[offset])])
        task.update(block_index=0, block_id=f"pfc_working_memory_circuit_{circuit:03d}_block_000",
                    episode_id=record["episode_id"],
                    initial_state_seed=210000000 + 1000 * circuit,
                    background_seed=220000000 + 1000 * circuit,
                    stimulus_seed=230000000 + 1000 * circuit)
        binding, _ = _enrich_episode(task, Path(record["dataset"]),
                                     Path(record["source_run"]).parent, source_hash)
        baseline_by_circuit.setdefault(circuit, []).append(binding)
    for circuit in (23, 63, 74, 86):
        cell = _reader(fixture.base, str(600000 + circuit), circuit, 0)
        report = json.loads((cell["archive"] / "report.json").read_text())
        report["circuits"][0]["source_bindings"] = baseline_by_circuit.get(circuit, [])
        _write(cell["archive"] / "report.json", report)
        receipt = json.loads(cell["receipt"].read_text())
        receipt["sources"] = baseline_by_circuit.get(circuit, [])
        _write(cell["receipt"], receipt)
        _reseal(cell)
        receipt = json.loads(cell["receipt"].read_text())
        baseline_receipts.append(dict(path=str(cell["receipt"]),
            sha256=campaign.digest(cell["receipt"]),
            authenticated_delivery_sha256=receipt["delivery_sha256"],
            authenticated_report_sha256=receipt["report_sha256"]))
    receipt_bindings = _write(fixture.baseline.parent / "input_receipt_bindings.json", baseline_receipts)
    _write(fixture.validation, dict(files=[
        _file_binding(fixture.baseline, fixture.baseline.name),
        _file_binding(receipt_bindings, receipt_bindings.name),
    ]))
    for group in range(10):
        cell = fixture.accept(group)
        _enrich_group(fixture, group, cell, source_hash)

    def load_plan(path, *, require_release=False):
        assert path == fixture.plan and require_release is True
        return deepcopy(fixture.tasks)

    monkeypatch.setattr(pfc_training_inputs, "load_plan", load_plan)
    fixture.cli(monkeypatch)
    fixture.prepared = tmp_path / "prepared"
    fixture.repo = Path(__file__).resolve().parents[2]
    fixture.initial_inputs = fixture.output.read_bytes()
    return fixture


def _prepare_cli(fixture, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_population_trajectory_campaign", "--prepare-pfc",
        "--repo", str(fixture.repo), "--pfc-inputs", str(fixture.output),
        "--pfc-coverage", str(fixture.coverage), "--output", str(fixture.prepared)])
    campaign.main()


def test_complete_actual_intake_cli_prepares_and_stages_all_declared_files(complete, monkeypatch):
    saved = json.loads(complete.coverage.read_text())
    assert saved["inputs"] == str(complete.output)
    assert saved["inputs_sha256"] == campaign.digest(complete.output)
    assert (saved["fitting_episodes"], saved["validation_episodes"]) == (80, 30)
    assert saved["accepted_export_episodes"] == 140
    _prepare_cli(complete, monkeypatch)
    proof = json.loads((complete.prepared / "preparation.json").read_text())
    assert proof["status"] == "PREPARED_COMPLETE_DEVELOPMENT_COHORT"
    assert proof["training_seeds"] == [41, 42, 43]
    assert (proof["fitting_episodes"], proof["validation_episodes"]) == (80, 30)
    assert proof["input_sha256"] == saved["inputs_sha256"]
    assert json.loads((complete.prepared / "input_coverage.json").read_text()) == saved
    assert not proof["training_submitted"] and not proof["scientific_claim_qualified"]
    manifest = json.loads(complete.output.read_text())
    assert len(manifest["episodes"]) == 110
    assert not any("circuit_086" in row["episode_id"] for row in manifest["episodes"])
    assert len(proof["paths"]) == 221  # one source entry plus two roots per episode
    index = {row["path"]: row for row in proof["input_files"]}
    assert len(index) == len(proof["input_files"])
    for offset, episode in enumerate(manifest["episodes"]):
        source_prefix = f"inputs/sources/{offset:03d}/"
        dataset_prefix = f"inputs/datasets/{offset:03d}/"
        names = ("delivery.json", "source_run/report.json",
                 *["source_run/" + name for name in SOURCE_OUTPUTS],
                 *["Libs/population_dynamics/" + name for name in PRODUCERS],
                 "qualification/delivery.json", "qualification/comparison/qualification.json")
        assert all(source_prefix + name in index for name in names)
        assert dataset_prefix + "dataset.json" in index
        assert any(path.startswith(dataset_prefix + "circuits/") for path in index)
        assert dataset_prefix + "episodes_000000.h5" in index
    software = {row["path"] for row in proof["files"]}
    for name in ("source_population_training", "pfc_population_training_data", "pfc_renewal_data", "pfc_source_dataset",
                 "pfc_scale_analysis", "pfc_kernel_parameters", "pfc_recurrent_gain_comparison",
                 "population_trajectory_training", "renewal_trajectory_training",
                 "population_trajectory_evaluation", "population_forecast_replay", "benchmark_tasks"):
        assert "Libs/population_dynamics/" + name + ".py" in software
    assert all("Libs/configs/" + name in software for name in
               (*campaign.PFC_CONFIGS, "neural_population_pfc_recurrent_gain.yaml"))
    # Use explicit fixture admission without changing the frozen real source.
    original_spec = campaign.importlib.util.spec_from_file_location

    def fixture_probe(name, location, *args, **kwargs):
        if name == "population_resource_probe":
            return SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: setattr(
                module, "probe", lambda *a, **k: {"status": "PASS", "reasons": []})))
        return original_spec(name, location, *args, **kwargs)

    original_module = campaign.importlib.util.module_from_spec
    monkeypatch.setattr(campaign.importlib.util, "spec_from_file_location", fixture_probe)
    monkeypatch.setattr(campaign.importlib.util, "module_from_spec",
        lambda spec: SimpleNamespace() if isinstance(spec, SimpleNamespace) else original_module(spec))
    monkeypatch.setattr(campaign.shutil, "disk_usage", lambda *a: SimpleNamespace(free=10**15))
    work = complete.prepared.parent / "local_execution"
    work.mkdir()
    _, staged = campaign.stage_pfc(complete.prepared, work)
    assert staged == proof
    assert (work / "inputs/cohort.json").read_bytes() == complete.initial_inputs
    assert complete.output.read_bytes() == complete.initial_inputs
    for row in proof["input_files"]:
        assert (work / row["path"]).stat().st_size == row["bytes"]
        assert campaign.digest(work / row["path"]) == row["sha256"]
    for row in proof["files"]:
        assert campaign.digest(work / row["path"]) == row["sha256"]
    mapping = json.loads((work / "execution_path_map.json").read_text())
    assert mapping["inputs_sha256"] == proof["input_sha256"]
    assert all(Path(value["path"]).is_relative_to(work) for value in mapping["paths"].values())
    assert not (work / "training").exists()


@pytest.mark.parametrize("change,error", [
    ("digest", "COVERAGE_OUTPUT_INPUT_BINDING_CHANGED"),
    ("input_path", "COVERAGE_OUTPUT_INPUT_BINDING_CHANGED"),
    ("extra_coverage", "80_FIT_30_VALIDATION_REQUIRED"),
])
def test_saved_coverage_binding_failures_reject_before_preparation_directory(complete, monkeypatch, change, error):
    value = json.loads(complete.coverage.read_text())
    if change == "digest":
        value["inputs_sha256"] = "0" * 64
    elif change == "input_path":
        value["inputs"] = str(complete.output.with_name("foreign_inputs.json"))
    else:
        value["unregistered_metadata"] = True
    _write(complete.coverage, value)
    with pytest.raises(ValueError, match=error):
        _prepare_cli(complete, monkeypatch)
    assert not complete.prepared.exists()
    assert complete.output.read_bytes() == complete.initial_inputs



def _wrapper_snapshot(fixture):
    """Freeze actual preparation software; payload fixtures remain separate."""
    snapshot = fixture.prepared.parent / "submission_snapshot"
    modules = [
        "population_trajectory_training", "population_trajectory_evaluation",
        "population_forecast_replay", "pfc_population_evaluation", "pfc_training_inputs",
        "resource_probe", "poisson_trajectory_training", "poisson_state_space",
        "native_count_windows", "native_trajectory_cohort", "renewal_trajectory_training",
        "population_state", "renewal_state_space", "refractory_observation", "native_renewal_data",
        "pfc_population_training_data", "pfc_renewal_data", "circuit_data", "pfc_scale_analysis",
        "pfc_source_dataset", "pfc_kernel_parameters", "pfc_recurrent_gain_comparison",
        "benchmark_tasks",
    ]
    names = campaign.software_closure(fixture.repo, modules)
    names += ["Libs/configs/" + name for name in
              (*campaign.PFC_CONFIGS, "neural_population_pfc_recurrent_gain.yaml")]
    names += ["Libs/scripts/run_population_trajectory_campaign.py",
              "Libs/scripts/run_population_trajectory_campaign.sbatch",
              "Libs/scripts/prepare_pfc_population_campaign.sbatch"]
    rows = []
    for name in names:
        source = fixture.repo / name
        row = _file_binding(source, name)
        campaign.copy_bound(source, snapshot / name, row)
        rows.append(row)
    campaign.write_json(snapshot / "software_snapshot.json", dict(
        schema="pfc_population_campaign_software_snapshot_v1", files=rows,
        training_seeds=[41, 42, 43], training_submitted=False,
    ))
    return snapshot, rows


def _execute_prepare_wrapper(fixture, snapshot, monkeypatch):
    wrapper = fixture.repo / "Libs/scripts/prepare_pfc_population_campaign.sbatch"
    text = wrapper.read_text()
    assert text.count("<<'PY'\n") == 1
    body = text.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    monkeypatch.setattr(sys, "argv", ["-", str(snapshot), str(fixture.output),
                                     str(fixture.coverage), str(fixture.prepared)])
    for name, value in dict(SLURMD_NODENAME="cpu04", SLURM_JOB_PARTITION="cpu",
                            SLURM_JOB_ID="metadata-preparation-fixture", SLURM_JOB_GPUS="").items():
        monkeypatch.setenv(name, value)
    exec(compile(body, str(wrapper), "exec"), {"__name__": "__main__"})


def test_prepare_wrapper_refreshes_stale_cohort_and_freezes_complete_execution(complete, monkeypatch):
    snapshot, rows = _wrapper_snapshot(complete)
    stale = json.loads(complete.coverage.read_text())
    stale.update(status="PARTIAL_FIT_VALIDATION_INPUT_COVERAGE", fitting_episodes=20,
                 validation_episodes=10, fit_validation_coverage_ready=False)
    _write(complete.output, complete.original)
    stale.update(inputs_sha256=campaign.digest(complete.output))
    _write(complete.coverage, stale)
    before_baseline = complete.baseline.read_bytes()
    _execute_prepare_wrapper(complete, snapshot, monkeypatch)
    refreshed = json.loads(complete.coverage.read_text())
    assert (refreshed["fitting_episodes"], refreshed["validation_episodes"]) == (80, 30)
    assert refreshed["fit_validation_coverage_ready"] is True
    assert refreshed["inputs_sha256"] == campaign.digest(complete.output)
    assert complete.output.read_bytes() == complete.initial_inputs
    assert complete.baseline.read_bytes() == before_baseline
    execution = json.loads((complete.prepared / "preparation_execution.json").read_text())
    assert execution["status"] == "PREPARED_COMPLETE_AUTHENTICATED_PFC_TRAINING"
    assert execution["job_id"] == "metadata-preparation-fixture"
    assert execution["training_seeds"] == [41, 42, 43]
    assert (execution["fitting_episodes"], execution["validation_episodes"]) == (80, 30)
    assert not execution["training_started"] and not execution["scientific_claim_qualified"]
    assert execution["source_snapshot"] == str(snapshot)
    assert execution["software_snapshot_sha256"] == campaign.digest(snapshot / "software_snapshot.json")
    assert execution["preparation_sha256"] == campaign.digest(complete.prepared / "preparation.json")
    proof = json.loads((complete.prepared / "preparation.json").read_text())
    frozen = {row["path"]: row for row in proof["files"]}
    for row in rows:
        if row["path"] == "Libs/scripts/prepare_pfc_population_campaign.sbatch":
            continue
        assert frozen[row["path"]] == row
        assert campaign.digest(complete.prepared / row["path"]) == row["sha256"]
    assert proof["input_sha256"] == refreshed["inputs_sha256"]
    assert not (complete.prepared / "training").exists()


def test_prepare_wrapper_corrupted_frozen_source_rejects_before_refresh(complete, monkeypatch):
    snapshot, _ = _wrapper_snapshot(complete)
    source = snapshot / "Libs/population_dynamics/pfc_population_training_data.py"
    with source.open("a") as stream:
        stream.write("\n# Changed after snapshot binding.\n")
    preserved = (complete.output.read_bytes(), complete.coverage.read_bytes())
    with pytest.raises(ValueError, match="FROZEN_PREPARATION_SOFTWARE_CHANGED"):
        _execute_prepare_wrapper(complete, snapshot, monkeypatch)
    assert (complete.output.read_bytes(), complete.coverage.read_bytes()) == preserved
    assert not complete.prepared.exists()


def test_prepare_wrapper_partial_readers_reject_before_inputs_or_coverage_write(complete, monkeypatch):
    snapshot, _ = _wrapper_snapshot(complete)
    missing = complete.base / f"parameter_block_readback_acceptance_{complete.jobs[8]}.json"
    missing.unlink()
    preserved = (complete.output.read_bytes(), complete.coverage.read_bytes())
    with pytest.raises(ValueError, match="COMPLETE_80_FIT_30_VALIDATION_BEFORE_PREPARATION"):
        _execute_prepare_wrapper(complete, snapshot, monkeypatch)
    assert (complete.output.read_bytes(), complete.coverage.read_bytes()) == preserved
    assert not complete.prepared.exists()


def test_prepare_wrapper_has_valid_shell_syntax():
    import subprocess
    path = Path(__file__).with_name("prepare_pfc_population_campaign.sbatch")
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_actuation_configs_are_frozen_by_real_prepare_entry(complete,monkeypatch):
    training=complete.repo/"Libs/configs/neural_population_pfc_actuation_training.yaml"
    forecast=complete.repo/"Libs/configs/neural_population_pfc_actuation_forecast.yaml"
    monkeypatch.setattr(sys,"argv",["run_population_trajectory_campaign","--prepare-pfc",
        "--repo",str(complete.repo),"--pfc-inputs",str(complete.output),
        "--pfc-coverage",str(complete.coverage),"--output",str(complete.prepared),
        "--pfc-training-config",str(training),"--pfc-forecast-config",str(forecast)])
    campaign.main()
    proof=json.loads((complete.prepared/"preparation.json").read_text())
    assert proof["control_geometry"]=="actuation_only"
    assert proof["training_config"]==training.name and proof["forecast_config"]==forecast.name
    assert proof["input_sha256"]==campaign.digest(complete.output)
    assert (complete.prepared/"Libs/configs"/training.name).read_bytes()==training.read_bytes()
    assert (complete.prepared/"Libs/configs"/forecast.name).read_bytes()==forecast.read_bytes()
    assert proof["training_seeds"]==[41,42,43]
    assert (proof["fitting_episodes"],proof["validation_episodes"])==(80,30)
