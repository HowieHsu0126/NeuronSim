"""Qualified PFC admission and native archive lifecycle on disposable fixtures.

Synthetic receipts and archive arrays test engineering contracts. They do not
qualify source fidelity or contribute production evidence.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from . import benchmark_production as production
from . import pfc_source_execution as executor
from . import pfc_kernel_parameters as kernels
from . import pfc_scale_analysis as readout
from .benchmark_dataset import read_episode, read_circuit
from .benchmark_tasks import task_protocol
from .circuit_campaign import campaign_config, freeze_identities
from .circuit_data import CircuitDataset
from .pfc_native_drive import create_drive
from .provenance import sha256_file

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCENARIO = "pfc_working_memory"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def native_qualification(root, config):
    path = root / "comparison/qualification.json"
    receipt = dict(status="PASS_NATIVE_STREAM_SOURCE_EXECUTION",
        source_entry_sha256=executor.native_drive.SOURCE_ENTRY_SHA256, backend="numpy",
        numpy_version=np.__version__, executor_sha256=sha256_file(Path(executor.executor.__file__)),
        drive_sha256=sha256_file(Path(executor.native_drive.__file__)),
        input_replay_status="PASS_NATIVE_NUMPY_DRIVE_REPLAY", input_replay_passed_comparisons=24,
        global_cases_passed=2, global_arrays_passed=10, validated_dt_ms=[config["dt_ms"]],
        validated_kernel_parameters=[kernels.resolve_kernel_parameters(config)],
        kernel_parameterization_sha256=sha256_file(Path(kernels.__file__)),
        scientific_fidelity_qualified=False, full_scale_qualified=False, production_episodes=0)
    write_json(path, receipt)
    write_json(root / "delivery.json", dict(status="ARCHIVED_PFC_STREAM_VALIDATION",
        result_status=receipt["status"], files=[dict(path="comparison/qualification.json",
            bytes=path.stat().st_size, sha256=sha256_file(path))]))
    return executor._qualification(root, "numpy", config["dt_ms"], config)


@pytest.fixture(scope="module")
def pfc_cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("pfc-production-fixture")
    config = yaml.safe_load((CONFIGS / "neural_population_dynamics.yaml").read_text())
    specification = config["simulation_campaign"]["scenarios"][SCENARIO]
    specification.pop("source_admission", None)
    specification["task_protocol"] = task_protocol(SCENARIO)
    specification["parameter_domain"] = dict(frozen=True, source="disposable numerical-kernel fixture")
    specification["task_protocol"]["frozen"] = True
    specification["observation_map"]["frozen"] = True
    path = root / "campaign.yaml"; path.write_text(yaml.safe_dump(config, sort_keys=False))
    base = yaml.safe_load((CONFIGS / "neural_population_pfc_100k_source.yaml").read_text())
    configs, native = {}, {}
    for index in range(100):
        name = f"pfc_working_memory_circuit_{index:03d}"
        recipe = dict(base, sigma_weight_profile=20.0 + index * .01, Jpos_excit2excit=1.6)
        configs[name] = recipe
        bound = native_qualification(root / name, recipe)
        native[name] = {key: bound[key] for key in ("receipt_sha256", "delivery_sha256")}
        native[name]["backend"] = "numpy"
    proof = dict(status="PASS", scenario_id=SCENARIO, coupled_core_neurons=100000,
        source_revision=executor._SOURCE_REVISION, evidence=["disposable engineering fixture only"],
        checks={name: True for name in ("source_reproduction", "numerical_integrity", "driver_statistics",
                                       "control_delivery", "full_unit_recording", "coupling", "scaling_fidelity")},
        source_recipes=dict(circuit_configs=configs, native_qualifications=native))
    qualification = root / "qualification.json"; write_json(qualification, proof)
    table = root / "identities.csv"
    freeze_identities(campaign_config(path), SCENARIO, qualification, table)
    return production.ProductionCohort.load(path, SCENARIO, qualification, table)


def archive(root, config_path, cohort, identity, *, failed=False):
    """Build the existing native archive format with complete observed arrays."""
    config = yaml.safe_load(config_path.read_text()); case = config["cases"][0]
    source = root / "source_run"; source.mkdir(parents=True)
    if failed:
        report = dict(status="PFC_SOURCE_EXECUTION_FAILED", source_execution_complete=False,
            case_id=identity["episode_id"], resolved_config=config, selected_case=case,
            scientific_fidelity_qualified=False, production_episodes=0)
        write_json(source / "report.json", report)
        names = ["source_run/report.json"]
    else:
        n_e, n_i, total = 80000, 20000, 100000
        _, _, dimensions = executor._case_config(config, identity["episode_id"])
        code = dict(runner_sha256=sha256_file(Path(executor.__file__)),
            executor_sha256=sha256_file(Path(executor.executor.__file__)),
            drive_sha256=sha256_file(Path(executor.native_drive.__file__)),
            kernel_parameterization_sha256=sha256_file(Path(kernels.__file__)))
        for key, name in readout._SOURCE_KERNEL_CODE_FILES.items():
            target = root / "Libs/population_dynamics" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(executor.__file__).with_name(name).read_bytes())
        qroot = root / "qualification"
        bound = native_qualification(qroot, config)
        initial, drive = create_drive(config, case)
        np.save(source / "initial_voltage.npy", initial)
        ticks = np.array([0, 2000, 99999], dtype=np.int64)
        ids = np.array([0, 1, 99999], dtype=np.int64)
        counts = np.bincount(ids, minlength=total)
        terminal = {}
        for population, size, offset in (("E", n_e, 0), ("I", n_i, n_e)):
            for name in executor._STATE_FIELDS[population]:
                terminal[population + "__" + name] = np.zeros(size)
            terminal[population + "__v"][:] = -.06
            terminal[population + "__not_refractory"] = np.ones(size, dtype=np.bool_)
            terminal[population + "__source_indices"] = np.arange(size, dtype=np.int64)
            terminal[population + "__archive_indices"] = np.arange(size, dtype=np.int64) + offset
            terminal[population + "__spike_counts"] = counts[offset:offset+size]
        np.savez(source / "raw.npz", spike_ticks=ticks, spike_unit_indices=ids, spike_seconds=ticks * .00005,
            native_spike_counts=counts, terminal_voltage_mV=np.r_[terminal["E__v"], terminal["I__v"]] / .001)
        np.savez(source / "terminal_state.npz", **terminal)
        np.savez(source / "units.npz", **executor._units(n_e, n_i))
        np.savez(source / "final_rng.npz", keys=np.array([41, 42, 43], dtype=np.uint32))
        controls = np.zeros((3, n_e)); controls[1, 23334:30001] = case["stimulus_strength_nA"] * 1e-9
        np.savez(source / "applied_current_observation.npz", source_unit_indices=np.arange(n_e, dtype=np.int64),
            archive_unit_indices=np.arange(n_e, dtype=np.int64),
            segment_start_ticks=np.array([0, 2000, 6020], dtype=np.int64),
            segment_stop_ticks=np.array([2000, 6020, 100000], dtype=np.int64),
            cue_active=np.array([False, True, False]), distractor_active=np.zeros(3, dtype=np.bool_), current_A=controls)
        metadata = executor.executor._applied_current_metadata(n_e, .00005, 100000, 20)
        write_json(source / "applied_current_metadata.json", metadata)
        provenance = kernels.circuit_provenance(config, executor.native_drive.SOURCE_ENTRY_SHA256)
        write_json(source / "unit_metadata.json", dict(unit_file="units.npz", unit_count=total,
            population_index_labels={"0": "E", "1": "I"}, inhibitory_preference_coordinate=None,
            anatomical_soma_coordinates_provided=False, initial_voltage_file="initial_voltage.npy", initial_voltage_units="mV",
            terminal_state_units=dict(v="V", I_stim="A", lastspike="s", synaptic_state_variables="dimensionless", not_refractory="boolean"),
            applied_current_observation_file="applied_current_observation.npz",
            applied_current_metadata_file="applied_current_metadata.json",
            coupling=dict(normalization=config["weight_scaling_factor"], representation="implicit source rules; no independent network tiling",
                resolved_kernel_parameters=kernels.resolve_kernel_parameters(config), circuit_provenance=provenance)))
        (source / "config.yaml").write_bytes(config_path.read_bytes())
        seeds = dict(initial_rng_seed=identity["initial_state_seed"], external_noise_seed=identity["background_seed"])
        request = dict(status="PFC_SOURCE_EXECUTION_REQUESTED", case_id=identity["episode_id"], selected_case=case,
            registered_case_ids=[identity["episode_id"]], registered_case_count=1, config_sha256=sha256_file(config_path),
            backend="numpy", dimensions=dimensions, code=code, backend_qualification=bound,
            resolved_kernel_parameters=kernels.resolve_kernel_parameters(config), circuit_provenance=provenance)
        write_json(source / "request.json", request)
        totals = dict(E=0, I=0)
        driver = drive.metadata()
        driver.update(ticks_generated=100000, external_event_totals=totals)
        report = dict(request, status="PFC_SOURCE_EXECUTION_COMPLETE", source_execution_complete=True,
            resolved_config=config, selected_case_count=1, completed_case_count=1,
            source_entry_sha256=executor.native_drive.SOURCE_ENTRY_SHA256,
            scientific_fidelity_qualified=False, working_memory_qualified=False,
            fidelity_at_requested_scale_qualified=False, learned_simulator=False, production_episodes=0,
            validations={key: dict(passed=True) for key in readout._SOURCE_EXECUTION_CHECKS},
            random_stream_policy="independent_initial_external", seeds=seeds, driver=driver,
            completed_duration_ticks=100000, neural_event_count=len(ids), source_external_event_totals=totals,
            inputs_used=dict(input_mode="native_numpy_stream", ticks_consumed=100000,
                             E=dict(values=n_e*100000, events=0), I=dict(values=n_i*100000, events=0)),
            costs=dict(applied_current_capture_seconds=.001),
            simulator_cost=dict(wall_seconds=123.0, native_input_generation_seconds=3.0,
                scope="simulate: validation, setup, input generation if streamed, transfer, all ticks, full output extraction and optional current observation; excludes file hashing and reference comparison"))
        report["outputs"] = {name: dict(path=name, bytes=(source/name).stat().st_size, sha256=sha256_file(source/name))
                             for name in readout._SOURCE_OUTPUTS | readout._SOURCE_CURRENT_OUTPUTS}
        write_json(source / "report.json", report)
        names = ["source_run/" + name for name in report["outputs"]] + ["source_run/report.json",
            "qualification/delivery.json", "qualification/comparison/qualification.json"]
        names += ["Libs/population_dynamics/" + name for name in readout._SOURCE_KERNEL_CODE_FILES.values()]
    write_json(root / "delivery.json", dict(status="ARCHIVED_PFC_SOURCE_EXECUTION", result_status=report["status"],
        source_execution_complete=report["source_execution_complete"], case_id=identity["episode_id"],
        scientific_fidelity_qualified=False, production_episodes=0,
        files=[dict(path=name, bytes=(root/name).stat().st_size, sha256=sha256_file(root/name)) for name in names]))
    return root


@pytest.mark.parametrize("domain_state", ["null", "missing"])
def test_unqualified_current_campaign_rejects_before_output(tmp_path, domain_state):
    proof = tmp_path / "qualification.json"; write_json(proof, {})
    campaign_path = CONFIGS / "neural_population_dynamics.yaml"
    if domain_state == "missing":
        campaign = yaml.safe_load(campaign_path.read_text())
        del campaign["simulation_campaign"]["scenarios"][SCENARIO]["parameter_domain"]
        campaign_path = tmp_path / "campaign.yaml"
        campaign_path.write_text(yaml.safe_dump(campaign, sort_keys=False))
    output = tmp_path / "production"
    with pytest.raises(ValueError, match="BENCHMARK_PRODUCTION_NOT_QUALIFIED:.*parameter_domain"):
        production.main(["--config", str(campaign_path), "--scenario", SCENARIO,
            "--qualification", str(proof), "--identity-table", str(tmp_path / "missing.csv"), "--circuit-index", "0",
            "prepare-pfc-episode", "--native-qualification", str(tmp_path / "native"), "--backend", "numpy",
            "--block-index", "0", "--condition-index", "0", "--output", str(output)])
    assert not output.exists()


def test_frozen_mapping_rejects_seed_labels_and_duplicate_actual_kernels(pfc_cohort):
    proof = deepcopy(pfc_cohort.qualification)
    configs = proof["source_recipes"]["circuit_configs"]
    configs["pfc_working_memory_circuit_001"] = deepcopy(configs["pfc_working_memory_circuit_000"])
    configs["pfc_working_memory_circuit_001"]["source_rng_seed"] = 42
    with pytest.raises(ValueError, match="DUPLICATE_NUMERICAL_KERNEL"):
        production.validate_pfc_recipes(pfc_cohort.campaign, proof)
    proof = deepcopy(pfc_cohort.qualification)
    for recipe in proof["source_recipes"]["circuit_configs"].values():
        recipe["Jpos_excit2excit"] = 1.0
    with pytest.raises(ValueError, match="DUPLICATE_NUMERICAL_KERNEL"):
        production.validate_pfc_recipes(pfc_cohort.campaign, proof)


def test_preparation_keeps_canonical_ten_conditions_two_rngs_and_frozen_protocol(pfc_cohort, tmp_path):
    original = deepcopy(pfc_cohort.qualification)
    for condition in range(10):
        identity = pfc_cohort.identity(0, 2, condition)
        config = pfc_cohort.pfc_episode_config(identity)
        case = config["cases"][0]
        assert case["initial_rng_seed"] == identity["initial_state_seed"]
        assert case["external_noise_seed"] == identity["background_seed"]
        assert config["benchmark_episode_candidate"]["execution_scope"] == "production"
        assert config["benchmark_task_protocol"] == task_protocol(SCENARIO) | {"frozen": True}
        assert executor._case_config(config, identity["episode_id"])[2]["coupled_core_units"] == 100000
        assert production.pfc_source_dataset._benchmark_registration(config)[1] == {
            identity["episode_id"]: (identity["background_seed"], config["benchmark_episode_candidate"]["condition_id"])}
    identity = pfc_cohort.identity(0, 0, 0)
    result = production.prepare_pfc_episode(pfc_cohort, identity,
        pfc_cohort.qualification_path.parent / identity["circuit_id"], "numpy", tmp_path / "prepared")
    assert result["source_simulations"] == 0 and result["complete_episodes"] == 0
    assert result["source_environment"]["NEURONSIM_PFC_SOURCE_CASE_ID"] == identity["episode_id"]
    assert pfc_cohort.qualification == original
    with pytest.raises(ValueError, match="CLOCK_NOT_QUALIFIED"):
        executor._qualification(pfc_cohort.qualification_path.parent / identity["circuit_id"], "numpy", .025,
                                pfc_cohort.pfc_episode_config(identity))
    with pytest.raises(ValueError, match="KERNEL_NOT_QUALIFIED"):
        production.prepare_pfc_episode(pfc_cohort, pfc_cohort.identity(1, 0, 0),
            pfc_cohort.qualification_path.parent / identity["circuit_id"], "numpy", tmp_path / "wrong_tuple")
    assert not (tmp_path / "wrong_tuple").exists()


@pytest.mark.parametrize("node", ["cpu04", "cpu05", "cpu06"])
def test_actual_native_archive_export_preserves_all_units_currents_states_and_resume(pfc_cohort, tmp_path, monkeypatch, node):
    identity = pfc_cohort.identity(0, 0, 0)
    prepared = tmp_path / "prepared"
    production.prepare_pfc_episode(pfc_cohort, identity,
        pfc_cohort.qualification_path.parent / identity["circuit_id"], "numpy", prepared)
    source = archive(tmp_path / "native", prepared / "episode.yaml", pfc_cohort, identity)
    monkeypatch.setenv("SLURMD_NODENAME", node); monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    output = tmp_path / "production"
    result = production.export_pfc(pfc_cohort, identity, prepared / "episode.yaml", source, output)
    assert result["status"] == "EXPORTED_QUALIFIED_PRODUCTION_EPISODE"
    dataset = CircuitDataset(output, splits=("train",))
    episode = read_episode(dataset, dataset.episodes[identity["episode_id"]])
    units = read_circuit(dataset, identity["circuit_id"])
    assert len(units.unit_ids) == 100000 and np.array_equal(units.populations[80000:], np.full(20000, "I"))
    assert episode.spike_ticks[-1] == 99999 and episode.spike_unit_indices[-1] == 99999
    assert episode.privileged["delivered_current__current_A"].shape == (3, 80000)
    assert episode.privileged["terminal__I__v"].shape == (20000,)
    assert len(episode.privileged["initial_voltage_mV"]) == 100000
    assert episode.source_record["native_seed_streams"] == dict(initial_rng_seed=identity["initial_state_seed"],
                                                               external_noise_seed=identity["background_seed"])
    view = dataset.window(identity["episode_id"], boundary_tick=2000, history_ticks=1000, horizon_ticks=1000, bin_ticks=100)
    assert "privileged" not in view and view["known_controls"][10, 2] == 1
    original = sha256_file(output / "dataset.json")
    monkeypatch.setattr(production, "qualify_pfc_backend", Mock(side_effect=AssertionError("native must not rerun")))
    reused = production.export_pfc(pfc_cohort, identity, tmp_path / "missing-config", tmp_path / "missing-source", output, resume=True)
    assert reused["new_source_simulations"] == 0 and sha256_file(output / "dataset.json") == original
    skipped = production.prepare_pfc_episode(pfc_cohort, identity, tmp_path / "missing-native", "numpy",
        tmp_path / "unused-preparation", production_output=output, resume=True)
    assert skipped["status"] == "REUSED_COMMITTED_PRODUCTION_EPISODE"
    assert not (tmp_path / "unused-preparation").exists()


def test_failed_or_rebound_native_archive_rejects_before_production_output(pfc_cohort, tmp_path):
    identity = pfc_cohort.identity(0, 0, 0)
    prepared = tmp_path / "prepared"
    production.prepare_pfc_episode(pfc_cohort, identity,
        pfc_cohort.qualification_path.parent / identity["circuit_id"], "numpy", prepared)
    failed = archive(tmp_path / "failed", prepared / "episode.yaml", pfc_cohort, identity, failed=True)
    with pytest.raises(ValueError, match="COMPLETE_NATIVE_ARCHIVE"):
        production.export_pfc(pfc_cohort, identity, prepared / "episode.yaml", failed, tmp_path / "production")
    assert (failed / "delivery.json").exists() and not (tmp_path / "production").exists()
    source = archive(tmp_path / "changed", prepared / "episode.yaml", pfc_cohort, identity)
    payload = source / "source_run/config.yaml"; payload.write_text(payload.read_text() + "\n")
    with pytest.raises(ValueError, match="SOURCE_DOCUMENT_DIFFERS"):
        production.export_pfc(pfc_cohort, identity, prepared / "episode.yaml", source, tmp_path / "changed-production")
    assert not (tmp_path / "changed-production").exists()


def test_bounded_real_source_execution_and_terminal_archive(tmp_path):
    # This 8E+2I,4.05ms execution checks the real runner only; no scale claim.
    config = dict(scope="source_reproduction", expected_source_revision=executor._SOURCE_REVISION,
        synapse_execution="factorized_uniform_projections", excitatory_neurons=8, inhibitory_neurons=2,
        weight_scaling_factor=256.0, dt_ms=.05, duration_ms=4.05,
        stimulus_center_deg=0.0, stimulus_width_deg=90.0, stimulus_start_ms=1.0,
        stimulus_duration_ms=2.0, random_stream_policy="independent_initial_external",
        cases=[dict(case_id="pfc_archive_lifecycle", initial_rng_seed=41, external_noise_seed=42,
                    stimulus_strength_nA=.5, distractor_strength_nA=0.0)])
    path = tmp_path / "config.yaml"; path.write_text(yaml.safe_dump(config))
    native_qualification(tmp_path / "qualification", config)
    report = executor.execute(path, config["cases"][0]["case_id"], tmp_path / "source_run", tmp_path / "qualification", "numpy")
    assert report["source_execution_complete"] is True, report.get("failure")
    assert all(row["passed"] is True for row in report["validations"].values())
    assert (tmp_path / "source_run/config.yaml").read_bytes() == path.read_bytes()
    assert report["seeds"] == dict(initial_rng_seed=41, external_noise_seed=42)
    assert report["backend_qualification"]["qualification"]["validated_kernel_parameters"] == [kernels.resolve_kernel_parameters(config)]


_PFC_STALE_SOURCE_ENV = (
    "SLURM_ARRAY_TASK_ID",
    "NEURONSIM_PFC_SOURCE_CASES_JSON",
    "NEURONSIM_PFC_SOURCE_CONFIG_SHA256",
    "NEURONSIM_PFC_SOURCE_CASES_SHA256",
    "NEURONSIM_PFC_SOURCE_CODE_SHA256",
)


def pfc_runtime(monkeypatch, job):
    """Select a disposable CPU invocation without submitting a Slurm job."""
    monkeypatch.setenv("SLURMD_NODENAME", "cpu04")
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setenv("SLURM_JOBID", job)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)


def run_pfc_fixture(cohort, identity, output, *, resume=False):
    return production.run_pfc(
        cohort, identity, cohort.qualification_path.parent / identity["circuit_id"],
        Path(sys.executable), Path(__file__).resolve().parents[2], output, resume=resume,
    )


def pfc_source_launcher(cohort, *, real_archive=False, launcher_exit=0):
    """Replace only source execution; optionally build the full native fixture."""
    calls = []

    def launch(command, **kwargs):
        env = dict(kwargs["env"])
        source_root = Path(env["NEURONSIM_PFC_OUTPUT_BASE"]).parent
        intent = json.loads((source_root / "execution.json").read_text())
        identity, job = intent["identity"], intent["producer_job"]
        native = Path(intent["source_archive"])
        if real_archive:
            archive(native, source_root / "prepared/episode.yaml", cohort, identity)
        else:
            # Lightweight orchestration cases mock export separately. This
            # receipt alone must never reach the real archive validator.
            write_json(native / "delivery.json", dict(
                status="ARCHIVED_PFC_SOURCE_EXECUTION", source_execution_complete=True,
                case_id=identity["episode_id"], fixture_scope="orchestration_only",
            ))
        write_json(Path(intent["batch_delivery"]), dict(
            status="COMPLETE", job_id=job, exit_code=0, srun_exit_code=0,
            case_ids=[identity["episode_id"]], cases=[dict(
                case_id=identity["episode_id"], job_id=job, exit_code=0,
                source_execution_complete=True, archive_status="COMPLETE",
                archive=str(native),
            )],
        ))
        calls.append(dict(command=list(command), env=env, source_root=source_root,
                          source_archive=native, intent=deepcopy(intent)))
        return subprocess.CompletedProcess(command, returncode=launcher_exit)

    return launch, calls


def test_run_pfc_same_job_isolates_source_archives_and_committed_reuse(
    pfc_cohort, tmp_path, monkeypatch,
):
    pfc_runtime(monkeypatch, "777003")
    for name in _PFC_STALE_SOURCE_ENV:
        monkeypatch.setenv(name, "stale-parent-value")
    output = tmp_path / "production"
    identities = [pfc_cohort.identity(0, block, 0) for block in (0, 1)]
    launch, calls = pfc_source_launcher(pfc_cohort, real_archive=True)
    monkeypatch.setattr(production.subprocess, "run", launch)

    results = [run_pfc_fixture(pfc_cohort, identity, output, resume=index > 0)
               for index, identity in enumerate(identities)]

    assert [row["new_source_simulations"] for row in results] == [1, 1]
    assert len(calls) == 2
    assert calls[0]["source_root"] != calls[1]["source_root"]
    assert calls[0]["source_archive"] != calls[1]["source_archive"]
    for call, identity in zip(calls, identities):
        env = call["env"]
        assert env["SLURM_JOB_ID"] == env["SLURM_JOBID"] == "777003"
        assert env["SLURM_NTASKS"] == env["SLURM_CPUS_PER_TASK"] == "1"
        assert env["NEURONSIM_PFC_SOURCE_CASE_ID"] == identity["episode_id"]
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert not set(_PFC_STALE_SOURCE_ENV).intersection(env)
        assert call["source_archive"].name == "run_777003"
        assert call["intent"]["identity"] == identity
        launcher_cost = json.loads((call["source_root"] / "launcher_exit.json").read_text())
        assert launcher_cost["producer_job"] == "777003" and launcher_cost["exit_code"] == 0
        assert launcher_cost["wall_seconds"] >= 0
        assert "archive/hash" in launcher_cost["cost_scope"]
    for result, call in zip(results, calls):
        costs = result["production_costs"]
        native_report = json.loads((call["source_archive"] / "source_run/report.json").read_text())
        assert costs["source_simulator_cost"] == native_report["simulator_cost"]
        assert costs["source_simulator_cost"]["wall_seconds"] == 123.0
        assert "excludes file hashing" in costs["source_simulator_cost"]["scope"]
        assert costs["source_launcher_wall_seconds"] == json.loads(
            (call["source_root"] / "launcher_exit.json").read_text())["wall_seconds"]
        assert costs["source_exporter_elapsed_seconds"] >= 0
        assert costs["export_wall_seconds"] >= (
            costs["source_exporter_wall_seconds"] + costs["cohort_writer_wall_seconds"])
        assert "all-unit" in costs["scopes"]["cohort_writer"]
        completion = json.loads((call["source_root"] / "report.json").read_text())
        assert completion["production_costs"] == costs
        assert completion["identity"] == call["intent"]["identity"]
    dataset = CircuitDataset(output, splits=("train",))
    assert set(dataset.episodes) == {identity["episode_id"] for identity in identities}
    before = sha256_file(output / "dataset.json")
    forbidden = Mock(side_effect=AssertionError("committed episode must not prepare or execute source"))
    monkeypatch.setattr(production, "prepare_pfc_episode", forbidden)
    monkeypatch.setattr(production.subprocess, "run", forbidden)

    reused = production.run_pfc(
        pfc_cohort, identities[0], tmp_path / "missing-qualification",
        tmp_path / "missing-python", tmp_path / "missing-project", output, resume=True,
    )

    assert reused["new_source_simulations"] == 0
    assert reused["production_costs"] == results[0]["production_costs"]
    assert reused["source_producer_job"] == "777003"
    assert sha256_file(output / "dataset.json") == before
    forbidden.assert_not_called()

    # Receipt validation must reject changed source/dataset bindings, while
    # preserving the already committed payload and original receipt bytes.
    completion_path = calls[0]["source_root"] / "report.json"
    original = completion_path.read_bytes()
    for name, value, expected in (
        ("source_delivery_sha256", "0" * 64, "COMPLETION_RECEIPT_DIFFERS"),
        ("source_producer_job", "777999", "COMPLETION_PRODUCER_DIFFERS"),
        ("dataset_episode_binding", dict(shard="wrong.h5", sha256="0" * 64),
         "COMPLETION_RECEIPT_DIFFERS"),
    ):
        changed = json.loads(original)
        changed[name] = value
        write_json(completion_path, changed)
        with pytest.raises(ValueError, match=expected):
            run_pfc_fixture(pfc_cohort, identities[0], output, resume=True)
        completion_path.write_bytes(original)
        assert sha256_file(output / "dataset.json") == before
    # A killed completion-receipt write cannot retrospectively measure export.
    completion_path.unlink()
    unavailable = run_pfc_fixture(pfc_cohort, identities[0], output, resume=True)
    assert unavailable["production_cost_status"] == "UNAVAILABLE_ORIGINAL_COMPLETION_RECEIPT"
    assert unavailable["production_costs"] is None and unavailable["new_source_simulations"] == 0
    assert not completion_path.exists()
    assert sha256_file(output / "dataset.json") == before


def test_run_pfc_export_interruption_reuses_original_job_archive(
    pfc_cohort, tmp_path, monkeypatch,
):
    identity = pfc_cohort.identity(0, 0, 0)
    output = tmp_path / "production"
    pfc_runtime(monkeypatch, "777001")
    launch, calls = pfc_source_launcher(pfc_cohort, real_archive=True)
    monkeypatch.setattr(production.subprocess, "run", launch)
    real_export = production.export_pfc
    interrupted_export = Mock(side_effect=RuntimeError("injected export interruption"))
    monkeypatch.setattr(production, "export_pfc", interrupted_export)

    with pytest.raises(RuntimeError, match="injected export interruption"):
        run_pfc_fixture(pfc_cohort, identity, output)

    assert len(calls) == 1
    interrupted_export.assert_called_once()
    source_root = calls[0]["source_root"]
    intent_bytes = (source_root / "execution.json").read_bytes()
    source_delivery_sha256 = sha256_file(calls[0]["source_archive"] / "delivery.json")
    assert not (output / "dataset.json").exists()
    pfc_runtime(monkeypatch, "777002")
    monkeypatch.setattr(production, "export_pfc", real_export)
    forbidden = Mock(side_effect=AssertionError("complete retained source must not execute again"))
    monkeypatch.setattr(production.subprocess, "run", forbidden)

    result = run_pfc_fixture(pfc_cohort, identity, output, resume=True)

    assert result["new_source_simulations"] == 0
    assert result["source_producer_job"] == "777001"
    assert Path(result["source_archive"]) == calls[0]["source_archive"]
    assert (source_root / "execution.json").read_bytes() == intent_bytes
    assert sha256_file(calls[0]["source_archive"] / "delivery.json") == source_delivery_sha256
    assert not (source_root / "native/run_777002").exists()
    costs = result["production_costs"]
    assert costs["source_launcher_wall_seconds"] == json.loads(
        (source_root / "launcher_exit.json").read_text())["wall_seconds"]
    assert result["production_cost_status"] == "RECORDED_ORIGINAL_COMPLETION"
    completion_bytes = (source_root / "report.json").read_bytes()
    committed = run_pfc_fixture(pfc_cohort, identity, output, resume=True)
    assert committed["production_costs"] == costs and committed["new_source_simulations"] == 0
    assert (source_root / "report.json").read_bytes() == completion_bytes
    assert production.completed_episode(pfc_cohort, identity, output, resume=True) is not None
    forbidden.assert_not_called()


def test_run_pfc_preparation_only_resume_starts_one_source_attempt(
    pfc_cohort, tmp_path, monkeypatch,
):
    identity = pfc_cohort.identity(0, 0, 0)
    output = tmp_path / "production"
    source_root = output / "source_execution" / identity["episode_id"]
    prepared = source_root / "prepared"
    qualification = pfc_cohort.qualification_path.parent / identity["circuit_id"]
    production.prepare_pfc_episode(pfc_cohort, identity, qualification, "numpy", prepared)
    config_before = (prepared / "episode.yaml").read_bytes()
    assert not (source_root / "execution.json").exists()
    pfc_runtime(monkeypatch, "777004")
    launch, calls = pfc_source_launcher(pfc_cohort)
    monkeypatch.setattr(production.subprocess, "run", launch)
    exporter = Mock(return_value=dict(status="EXPORTED_QUALIFIED_PRODUCTION_EPISODE"))
    monkeypatch.setattr(production, "export_pfc", exporter)

    result = run_pfc_fixture(pfc_cohort, identity, output, resume=True)

    assert result["new_source_simulations"] == 1
    assert len(calls) == 1
    assert (prepared / "episode.yaml").read_bytes() == config_before
    assert calls[0]["intent"]["identity"] == identity
    exporter.assert_called_once()


@pytest.mark.parametrize("evidence", ["native/delivery.json", "launcher_exit.json"])
def test_run_pfc_preparation_without_intent_rejects_source_evidence(
    pfc_cohort, tmp_path, monkeypatch, evidence,
):
    identity = pfc_cohort.identity(0, 0, 0)
    output = tmp_path / "production"
    source_root = output / "source_execution" / identity["episode_id"]
    (source_root / "prepared").mkdir(parents=True)
    write_json(source_root / evidence, dict(original_failure="retained contradictory source evidence"))
    before = (source_root / evidence).read_bytes()
    forbidden = Mock(side_effect=AssertionError("unresolved source evidence must not be replaced"))
    monkeypatch.setattr(production, "prepare_pfc_episode", forbidden)
    monkeypatch.setattr(production.subprocess, "run", forbidden)

    with pytest.raises(ValueError, match="PFC_UNRESOLVED_SOURCE_EXECUTION"):
        run_pfc_fixture(pfc_cohort, identity, output, resume=True)

    assert (source_root / evidence).read_bytes() == before
    assert not (source_root / "execution.json").exists()
    forbidden.assert_not_called()


@pytest.mark.parametrize("damage,expected", [
    ("missing_intent", "PFC_UNRESOLVED_SOURCE_EXECUTION"),
    ("missing_preparation_report", "PFC_UNRESOLVED_SOURCE_EXECUTION"),
    ("missing_batch", "PFC_UNRESOLVED_SOURCE_EXECUTION"),
    ("changed_prepared_hash", "PFC_RETAINED_PREPARATION_CHANGED"),
    ("failed_source", "PFC_COMPLETE_RETAINED_SOURCE_REQUIRED"),
])
def test_run_pfc_rejects_unresolved_or_failed_attempt_without_repeat(
    pfc_cohort, tmp_path, monkeypatch, damage, expected,
):
    identity = pfc_cohort.identity(0, 0, 0)
    output = tmp_path / "production"
    pfc_runtime(monkeypatch, "777005")
    launch, calls = pfc_source_launcher(pfc_cohort)
    monkeypatch.setattr(production.subprocess, "run", launch)
    exporter = Mock(side_effect=RuntimeError("retain attempted source for recovery test"))
    monkeypatch.setattr(production, "export_pfc", exporter)
    with pytest.raises(RuntimeError, match="retain attempted source"):
        run_pfc_fixture(pfc_cohort, identity, output)
    source_root = calls[0]["source_root"]
    batch_path = Path(calls[0]["intent"]["batch_delivery"])
    if damage == "missing_intent":
        (source_root / "execution.json").unlink()
    elif damage == "missing_preparation_report":
        (source_root / "prepared/report.json").unlink()
    elif damage == "missing_batch":
        batch_path.unlink()
    elif damage == "changed_prepared_hash":
        config_path = source_root / "prepared/episode.yaml"
        config_path.write_bytes(config_path.read_bytes() + b"\n")
    else:
        batch = json.loads(batch_path.read_text())
        batch.update(status="FAILED", exit_code=1, srun_exit_code=1)
        batch["cases"][0].update(exit_code=1, source_execution_complete=False,
                                  archive_status="FAILED", error="original nonresource source failure")
        write_json(batch_path, batch)
        write_json(calls[0]["source_archive"] / "delivery.json", dict(
            status="ARCHIVED_PFC_SOURCE_EXECUTION", source_execution_complete=False,
            case_id=identity["episode_id"], error="original nonresource source failure",
        ))
    retained_batch = batch_path.read_bytes() if batch_path.exists() else None
    pfc_runtime(monkeypatch, "777006")
    forbidden = Mock(side_effect=AssertionError("failed or unknown source must not repeat"))
    monkeypatch.setattr(production.subprocess, "run", forbidden)
    exporter.reset_mock()

    with pytest.raises(ValueError, match=expected):
        run_pfc_fixture(pfc_cohort, identity, output, resume=True)

    assert len(calls) == 1
    assert (source_root / "launcher_exit.json").exists()
    if retained_batch is not None:
        assert batch_path.read_bytes() == retained_batch
    forbidden.assert_not_called()
    exporter.assert_not_called()


def test_run_pfc_complete_batch_preserves_nonzero_outer_launcher_exit(
    pfc_cohort, tmp_path, monkeypatch,
):
    identity = pfc_cohort.identity(0, 0, 0)
    output = tmp_path / "production"
    pfc_runtime(monkeypatch, "777007")
    launch, calls = pfc_source_launcher(pfc_cohort, launcher_exit=7)
    monkeypatch.setattr(production.subprocess, "run", launch)
    exporter = Mock(return_value=dict(status="EXPORTED_QUALIFIED_PRODUCTION_EPISODE"))
    monkeypatch.setattr(production, "export_pfc", exporter)
    monkeypatch.setattr(production, "time", Mock(perf_counter=Mock(side_effect=[10.0, 13.0])))

    result = run_pfc_fixture(pfc_cohort, identity, output)

    assert result["new_source_simulations"] == 1
    assert len(calls) == 1
    launcher = json.loads((calls[0]["source_root"] / "launcher_exit.json").read_text())
    assert launcher["producer_job"] == "777007" and launcher["exit_code"] == 7
    assert launcher["wall_seconds"] == 3.0
    assert "excludes production preparation and export" in launcher["cost_scope"]
    assert json.loads(Path(result["source_batch_delivery"]).read_text())["status"] == "COMPLETE"
    exporter.assert_called_once()


def test_pfc_export_cost_receipt_separates_exporter_writer_and_total(tmp_path, monkeypatch):
    identity = dict(episode_id="disposable_cost_episode", split="train")
    config_path = tmp_path / "episode.yaml"
    config_path.write_text("{}\n")
    archive_path = tmp_path / "native"
    output = tmp_path / "production"
    cohort = Mock(scenario=SCENARIO, qualification_sha256="qualification-hash",
                  identity_table_sha256="identity-hash")
    cohort.pfc_episode_config.return_value = {}
    simulator_cost = dict(wall_seconds=100.0, scope="native simulator; excludes archive and export")
    header = dict(archive=archive_path, report=dict(backend="numpy", simulator_cost=simulator_cost,
        outputs={"request.json": dict(sha256="request-hash")}))
    source_exporter = Mock()
    source_exporter._benchmark_registration.return_value = ({}, {}, {}, {})
    source_exporter._source_header.return_value = (
        dict(status="COMPLETE", report_sha256="source-report-hash", delivery_sha256="delivery-hash"), header)
    source_exporter.export.return_value = dict(
        status="EXPORTED_VERIFIED_PFC_DEVELOPMENT_DATASET", elapsed_seconds=2.5)
    source_dataset = Mock()
    committed_dataset = Mock(episodes={identity["episode_id"]: dict(shard="episodes_000000.h5")},
                             shards={"episodes_000000.h5": dict(sha256="committed-shard-hash")})
    monkeypatch.setattr(production, "pfc_source_dataset", source_exporter)
    monkeypatch.setattr(production, "completed_episode", Mock(return_value=None))
    monkeypatch.setattr(production, "qualify_pfc_backend", Mock())
    monkeypatch.setattr(production, "verify_source_admission", Mock())
    monkeypatch.setattr(production, "source_document", Mock(return_value={}))
    monkeypatch.setattr(production, "CircuitDataset", Mock(side_effect=[source_dataset, committed_dataset]))
    writer = Mock(return_value=dict(status="EXPORTED_QUALIFIED_PRODUCTION_EPISODE", identity=identity,
                                   complete_episodes=1, dataset_catalog_sha256="catalog-at-commit"))
    monkeypatch.setattr(production, "write_episode", writer)
    monkeypatch.setattr(production, "time", Mock(perf_counter=Mock(side_effect=[0., 1., 4., 5., 9., 11.])))
    pfc_runtime(monkeypatch, "777008")

    result = production.export_pfc(cohort, identity, config_path, archive_path, output)

    costs = result["production_costs"]
    assert costs["source_exporter_elapsed_seconds"] == 2.5
    assert costs["source_exporter_wall_seconds"] == 3.0
    assert costs["cohort_writer_wall_seconds"] == 4.0
    assert costs["export_wall_seconds"] == 11.0
    assert costs["source_simulator_cost"] == simulator_cost
    assert costs["source_launcher_wall_seconds"] is None
    assert "excludes cohort loading, source launch and completion receipt write" in costs["scopes"]["export"]
    assert result["source_producer_job"] is None
    completion_path = output / "source_execution" / identity["episode_id"] / "report.json"
    saved = json.loads(completion_path.read_text())
    assert saved["production_costs"] == costs
    assert saved["dataset_episode_binding"] == dict(shard="episodes_000000.h5", sha256="committed-shard-hash")
    assert saved["source_delivery_sha256"] == "delivery-hash"
    assert saved["source_archive"] == str(archive_path.resolve())
    writer.assert_called_once()


def test_run_pfc_blocks_runs_all_ten_conditions_before_next_block(
    pfc_cohort, tmp_path, monkeypatch,
):
    output = tmp_path / "production"
    calls = []
    qualification = pfc_cohort.qualification_path.parent / pfc_cohort.identity(0, 2, 0)["circuit_id"]

    def episode(cohort, identity, native_qualification, source_python, project, destination, *, resume):
        assert cohort is pfc_cohort and native_qualification == qualification
        assert destination == output and resume is True
        calls.append((identity["block_index"], identity["condition_index"]))
        return dict(new_source_simulations=int(identity["condition_index"] != 0), identity=identity)

    monkeypatch.setattr(production, "run_pfc", episode)
    result = production.run_pfc_blocks(
        pfc_cohort, 0, 2, 2, qualification, Path(sys.executable),
        Path(__file__).resolve().parents[2], output,
    )

    assert calls == [(block, condition) for block in (2, 3) for condition in range(10)]
    assert result["complete_episodes"] == 20
    assert result["new_source_simulations"] == 18
    assert len(result["episodes"]) == 20


def test_run_pfc_blocks_stops_first_failure_and_preserves_prior_outputs(
    pfc_cohort, tmp_path, monkeypatch,
):
    output = tmp_path / "production"
    calls = []
    qualification = pfc_cohort.qualification_path.parent / pfc_cohort.identity(0, 2, 0)["circuit_id"]

    def episode(cohort, identity, native_qualification, source_python, project, destination, *, resume):
        pair = (identity["block_index"], identity["condition_index"])
        calls.append(pair)
        if pair == (2, 3):
            raise RuntimeError("injected condition failure")
        # These markers represent existing successful caller outputs. The loop
        # must preserve them; real dataset commits are covered above.
        write_json(destination / (identity["episode_id"] + ".json"), dict(identity=identity))
        return dict(new_source_simulations=1)

    monkeypatch.setattr(production, "run_pfc", episode)
    with pytest.raises(RuntimeError, match="injected condition failure"):
        production.run_pfc_blocks(
            pfc_cohort, 0, 2, 2, qualification, Path(sys.executable),
            Path(__file__).resolve().parents[2], output,
        )

    assert calls == [(2, condition) for condition in range(4)]
    assert {path.stem for path in output.glob("*.json")} == {
        pfc_cohort.identity(0, 2, condition)["episode_id"] for condition in range(3)
    }


@pytest.mark.parametrize("block_start,block_count", [
    (-1, 1), (100, 1), (0, 0), (99, 2), (False, 1), (0, True), (1.5, 1),
])
def test_run_pfc_blocks_invalid_range_rejects_before_source(
    pfc_cohort, tmp_path, monkeypatch, block_start, block_count,
):
    output = tmp_path / "production"
    forbidden = Mock(side_effect=AssertionError("invalid range must not start an episode"))
    monkeypatch.setattr(production, "run_pfc", forbidden)

    with pytest.raises(ValueError, match="PFC_BLOCK_RANGE_INVALID"):
        production.run_pfc_blocks(
            pfc_cohort, 0, block_start, block_count, pfc_cohort.qualification_path.parent,
            Path(sys.executable), Path(__file__).resolve().parents[2], output,
        )

    assert not output.exists()
    forbidden.assert_not_called()


def test_run_pfc_blocks_existing_output_requires_explicit_resume(
    pfc_cohort, tmp_path, monkeypatch,
):
    output = tmp_path / "production"
    output.mkdir()
    forbidden = Mock(side_effect=AssertionError("missing resume must not start an episode"))
    monkeypatch.setattr(production, "run_pfc", forbidden)

    with pytest.raises(ValueError, match="EXPLICIT_RESUME_REQUIRED"):
        production.run_pfc_blocks(
            pfc_cohort, 0, 0, 1, pfc_cohort.qualification_path.parent,
            Path(sys.executable), Path(__file__).resolve().parents[2], output,
        )

    forbidden.assert_not_called()


def pfc_block_launcher_fixture(tmp_path):
    """Provide disposable staged code and an argument-recording Python runtime."""
    launcher = Path(__file__).resolve().parents[1] / "scripts/run_pfc_production_blocks.sbatch"
    text = launcher.read_text()
    modules = re.search(r"^modules=\((.*?)^\)", text, flags=re.MULTILINE | re.DOTALL)
    assert modules is not None
    project = tmp_path / "project with spaces"
    for module in modules.group(1).split():
        target = project / "Libs/population_dynamics" / (module + ".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Disposable launcher fixture.\n")
    for name in ("run_pfc_source_cases.sbatch", "run_pfc_source_cases.py"):
        target = project / "Libs/scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Source execution is forbidden in this fixture.\n")
    runtime = tmp_path / "recording python"
    runtime.write_text(
        "#!" + sys.executable + "\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['PFC_LAUNCHER_TEST_CAPTURE']).write_text(json.dumps({"
        "'argv': sys.argv[1:], 'cwd': os.getcwd(), 'pythonpath': os.environ['PYTHONPATH'], "
        "'threads': {k: os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', "
        "'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}, 'cuda': os.environ['CUDA_VISIBLE_DEVICES']}))\n"
        "sys.exit(int(os.environ.get('PFC_LAUNCHER_TEST_EXIT', '0')))\n"
    )
    runtime.chmod(0o755)
    capture = tmp_path / "runtime.json"
    env = dict(os.environ, SLURMD_NODENAME="cpu05", SLURM_JOB_ID="777100",
               SLURM_NTASKS="1", SLURM_CPUS_PER_TASK="1",
               NRNSIM_BENCHMARK_PYTHON=str(runtime),
               PFC_LAUNCHER_TEST_CAPTURE=str(capture))
    env.pop("SLURM_JOB_GPUS", None)
    env.pop("SLURM_STEP_GPUS", None)
    arguments = [str(tmp_path / name) for name in
                 ("campaign with spaces.yaml", "qualification.json", "identities.csv")]
    arguments += ["23", "4", "2", str(tmp_path / "native qualification"),
                  str(runtime), str(project), str(tmp_path / "production output")]
    return launcher, arguments, env, capture


@pytest.mark.parametrize("resume,child_exit", [(False, 0), (True, 37)])
def test_pfc_block_launcher_forwards_arguments_and_preserves_exit(
    tmp_path, resume, child_exit,
):
    launcher, arguments, env, capture = pfc_block_launcher_fixture(tmp_path)
    env["PFC_LAUNCHER_TEST_EXIT"] = str(child_exit)
    command = ["bash", str(launcher), *arguments, *(["resume"] if resume else [])]

    child = subprocess.run(command, env=env, capture_output=True, text=True)

    assert child.returncode == child_exit, child.stderr
    recorded = json.loads(capture.read_text())
    assert recorded["argv"] == [
        "-B", "-m", "population_dynamics.benchmark_production",
        "--config", arguments[0], "--scenario", SCENARIO,
        "--qualification", arguments[1], "--identity-table", arguments[2],
        "--circuit-index", "23", "run-pfc-blocks", "--block-start", "4",
        "--block-count", "2", "--native-qualification", arguments[6],
        "--source-python", arguments[7], "--project", arguments[8],
        "--output", arguments[9], *(["--resume"] if resume else []),
    ]
    scratch = Path(recorded["cwd"])
    assert scratch.name.startswith("neuronsim-pfc-production-blocks.")
    assert recorded["pythonpath"] == str(scratch / "Libs")
    assert not scratch.exists()
    assert recorded["threads"] == dict.fromkeys(
        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
    assert recorded["cuda"] == ""
    assert not Path(arguments[9]).exists()
    text = launcher.read_text()
    assert not re.search(r"^\s*(?:exec\s+)?srun\b", text, flags=re.MULTILINE)
    assert "#SBATCH --partition=cpu" in text
    assert "#SBATCH --nodelist=cpu05,cpu06" in text
    assert "#SBATCH --gres=none" in text


@pytest.mark.parametrize("invalid", [
    "missing_argument", "unknown_mode", "wrong_node", "gpu", "two_cpus",
    "two_tasks", "relative_config",
])
def test_pfc_block_launcher_rejects_invalid_invocation_before_runtime(tmp_path, invalid):
    launcher, arguments, env, capture = pfc_block_launcher_fixture(tmp_path)
    if invalid == "missing_argument":
        arguments.pop()
    elif invalid == "unknown_mode":
        arguments.append("restart")
    elif invalid == "wrong_node":
        env["SLURMD_NODENAME"] = "cpu04"
    elif invalid == "gpu":
        env["SLURM_JOB_GPUS"] = "0"
    elif invalid == "two_cpus":
        env["SLURM_CPUS_PER_TASK"] = "2"
    elif invalid == "two_tasks":
        env["SLURM_NTASKS"] = "2"
    else:
        arguments[0] = "relative-campaign.yaml"

    child = subprocess.run(["bash", str(launcher), *arguments], env=env,
                           capture_output=True, text=True)

    assert child.returncode == 2
    assert not capture.exists()
    assert not (tmp_path / "production output").exists()


@pytest.mark.parametrize("resume", [False, True])
def test_pfc_blocks_cli_dispatch_preserves_explicit_arguments(tmp_path, monkeypatch, resume):
    paths = {name: tmp_path / name for name in
             ("config", "qualification", "identities", "native", "python", "project", "output")}
    cohort = object()
    load = Mock(return_value=cohort)
    run = Mock(return_value=dict(status="TEST_ONLY_CLI_FORWARDING", complete_episodes=0))
    monkeypatch.setattr(production.ProductionCohort, "load", load)
    monkeypatch.setattr(production, "run_pfc_blocks", run)

    exit_code = production.main([
        "--config", str(paths["config"]), "--scenario", SCENARIO,
        "--qualification", str(paths["qualification"]),
        "--identity-table", str(paths["identities"]), "--circuit-index", "23",
        "run-pfc-blocks", "--block-start", "4", "--block-count", "2",
        "--native-qualification", str(paths["native"]), "--source-python", str(paths["python"]),
        "--project", str(paths["project"]), "--output", str(paths["output"]),
        *(["--resume"] if resume else []),
    ])

    assert exit_code == 0
    load.assert_called_once_with(paths["config"], SCENARIO, paths["qualification"], paths["identities"])
    run.assert_called_once_with(cohort, 23, 4, 2, paths["native"], paths["python"],
                                paths["project"], paths["output"], resume=resume)
    assert not paths["output"].exists()


def test_pfc_blocks_real_cli_rejects_unqualified_campaign_before_output(tmp_path):
    qualification = tmp_path / "qualification.json"
    write_json(qualification, {})
    output = tmp_path / "production"
    project = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(project / "Libs"), PYTHONDONTWRITEBYTECODE="1")

    child = subprocess.run([
        sys.executable, "-B", "-m", "population_dynamics.benchmark_production",
        "--config", str(CONFIGS / "neural_population_dynamics.yaml"), "--scenario", SCENARIO,
        "--qualification", str(qualification), "--identity-table", str(tmp_path / "missing-identities.csv"),
        "--circuit-index", "23", "run-pfc-blocks", "--block-start", "4", "--block-count", "2",
        "--native-qualification", str(tmp_path / "missing-native"), "--source-python", sys.executable,
        "--project", str(project), "--output", str(output),
    ], cwd=tmp_path, env=env, capture_output=True, text=True)

    assert child.returncode != 0
    assert "BENCHMARK_PRODUCTION_NOT_QUALIFIED:" in child.stderr
    assert "parameter_domain" in child.stderr
    assert not output.exists()

