"""Source-chain routing uses disposable metadata and a fake Slurm client."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from . import v1_production_execution as execution
from .benchmark_production import ProductionCohort
from .benchmark_production_block import load_block
from .provenance import sha256_file
from .test_benchmark_production import CONFIG, fixture_configuration

REPO = CONFIG.parents[2]


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    config, qualification, identities = fixture_configuration(tmp_path_factory.mktemp("v1_execution"), "v1_visual")
    return ProductionCohort.load(config, "v1_visual", qualification, identities)


def archive(root, status, documents):
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for relative, value in documents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False) if relative.endswith(".yaml") else json.dumps(value))
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha256_file(path)))
    (root / "delivery.json").write_text(json.dumps(dict(status=status, exit_code=0, files=files)))
    return root


def resources(root, cohort, *, reuse=False):
    root.mkdir()
    identity = cohort.identity(0, 1, 0)
    construction = cohort.v1_construction_config(identity)
    circuit = root / "circuit.yaml"
    circuit.write_text(yaml.safe_dump(construction, sort_keys=False))
    shared = dict(mode="full", original_config=construction, original_config_sha256=sha256_file(circuit))
    recurrent = archive(root / "recurrent", "synthetic", {"primary/report.json": dict(shared,
        rng=dict(builder_seed=identity["connectivity_seed"], glif_assignment_seed=identity["connectivity_seed"] + 1000000))})
    inputs = archive(root / "inputs", "synthetic", {"primary/report.json": dict(shared,
        rng=dict(lgn_selection_python_seed=identity["connectivity_seed"]))})
    runtime = archive(root / "runtime", "synthetic", {"prepared/prepared.json": dict(status="SOURCE_RUNTIME_PREPARED")})
    native = archive(root / "native_canary", "PASS_POINTNET_NATIVE_VISUAL_INPUTS_DEVELOPMENT",
                     {"primary/report.json": {}, "original_config.yaml": construction})
    for name in ("parameters", "point_config", "authority"):
        (root / name).mkdir()
    conditions = []
    for index in range(10):
        recipe = cohort.v1_visual_config(cohort.identity(0, 1, index))
        representative = dict(recipe, unit_coverage="model_representatives")
        rate_code = archive(root / f"rate_code_{index}", "SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT",
            {"probe/probe.json": dict(status="SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT", config=representative)})
        spike = archive(root / f"spike_canary_{index}", "PASS_CONSTRUCTED_VISUAL_INPUT_CANARY",
                        {"visual_inputs/report.json": dict(config=representative)})
        rate_source = None
        if reuse:
            other_noise = deepcopy(recipe)
            other_noise["spike_generation"]["starting_noise_seed"] += 8
            other_noise["background"]["starting_noise_seed"] += 8
            rate_source = archive(root / f"rates_{index}", "SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT", {
                "probe/probe.json": dict(status="SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT", source_units=17400,
                    config=other_noise, input_construction_binding=dict(delivery_sha256=sha256_file(inputs / "delivery.json")))})
        conditions.append(dict(condition_index=index, rate_code_reference=str(rate_code) if not reuse else None,
                               spike_canary=str(spike), rate_source=str(rate_source) if rate_source else None))
    document = dict(schema=execution.SCHEMA, circuit_index=0, block_index=1,
        circuit_config=str(circuit), recurrent_source=str(recurrent), input_source=str(inputs),
        prepared_runtime=None if reuse else str(runtime), native_canary=str(native), parameters=str(root / "parameters"),
        point_config=str(root / "point_config"), authority_source=str(root / "authority"),
        native_python="/node-local/validated-nest/python", export_python=sys.executable,
        memory_gib=dict(rates=4, spikes=4, native=40, export=4), conditions=conditions)
    path = root / "resources.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path, document


class Scheduler:
    def __init__(self):
        self.calls, self.states = [], {}

    def __call__(self, command, **kwargs):
        if command[0] == "sacct":
            jobs = [int(job) for job in command[command.index("-j") + 1].split(",")]
            output = "".join(f"{job}|{self.states[job]}|0:0|\n" for job in jobs if job in self.states)
            return subprocess.CompletedProcess(command, 0, output, "")
        if command[0] == "scontrol":
            return subprocess.CompletedProcess(command, 1, "", "Invalid job id specified")
        assert command[0] == "sbatch"
        self.calls.append((command, kwargs))
        job = 700000 + len(self.calls)
        self.states[job] = "PENDING"
        return subprocess.CompletedProcess(command, 0, str(job) + "\n", "")


def prepare(cohort, path, output, *, resume=False):
    return execution.prepare(cohort, REPO, cohort.qualification_path.parent / "campaign.yaml",
                             path, output, resume=resume)


def test_native_execution_uses_declared_walltime(cohort, tmp_path, monkeypatch):
    path, document = resources(tmp_path / "resources", cohort, reuse=True)
    budgets = dict(rates=240, spikes=120, native=240, export=120)
    document["walltime_minutes"] = budgets
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    output = tmp_path / "execution"
    plan = prepare(cohort, path, output)
    scheduler = Scheduler()
    monkeypatch.setattr(execution.subprocess, "run", scheduler)
    result = execution.submit(cohort, plan, output)
    for (command, _), row in zip(scheduler.calls, result["jobs"]):
        assert "--time=" + str(budgets[row["phase"]]) in command


@pytest.mark.parametrize("reuse,expected_jobs", ((False, 31), (True, 21)))
def test_complete_source_chain_and_existing_export_manifest(cohort, tmp_path, monkeypatch, reuse, expected_jobs):
    path, document = resources(tmp_path / "resources", cohort, reuse=reuse)
    output = tmp_path / "execution"
    plan = prepare(cohort, path, output)
    scheduler = Scheduler()
    monkeypatch.setattr(execution.subprocess, "run", scheduler)
    for key in ("SLURM_JOB_ID", "SLURMD_NODENAME", "SRUN_CPUS_PER_TASK", "SBATCH_GRES", "NRNSIM_V1_CLOCK_PROTOCOL"):
        monkeypatch.setenv(key, "must_not_reach_source_job")
    result = execution.submit(cohort, plan, output)
    assert len(scheduler.calls) == expected_jobs
    assert "walltime_minutes" not in plan
    assert result["status"] == "SUBMITTED_V1_PRODUCTION_BLOCK" and result["production_episodes"] == 0
    previous = None
    for (command, options), row in zip(scheduler.calls, result["jobs"]):
        assert {"--partition=gpu", "--nodelist=cpu03", "--gres=none", "--no-requeue", "--cpus-per-task=1"} <= set(command)
        assert "--time=0" in command
        assert options["env"]["TMPDIR"] == "/tmp" and options["env"]["PROJECT_PREFIX"] == "NrnSim"
        assert not any(key.startswith(("SLURM_", "SLURMD_", "SRUN_", "SBATCH_")) for key in options["env"])
        assert "NRNSIM_V1_CLOCK_PROTOCOL" not in options["env"]
        dependencies = [arg for arg in command if arg.startswith("--dependency=")]
        assert dependencies == ([] if previous is None else ["--dependency=afterok:" + str(previous)])
        previous = row["job_id"]
        if row["phase"] == "native":
            assert options["env"]["NRNSIM_V1_NEST_PYTHON"] == document["native_python"]
            assert options["env"]["NRNSIM_V1_NATIVE_CODE_REFERENCE"] == document["native_canary"]
            assert command[-7:-1] == [document["circuit_config"], "full", document["recurrent_source"],
                                       document["input_source"], document["parameters"], document["point_config"]]
        if row["phase"] == "rates":
            assert options["env"]["NRNSIM_V1_LGN_INPUT_CONSTRUCTION"] == document["input_source"]
            assert "NRNSIM_V1_LGN_REFERENCE_ARCHIVE" not in options["env"]
    entries = load_block(cohort, output / "sources.json", 0, 1)
    assert [entry.identity for entry in entries] == [cohort.identity(0, 1, index) for index in range(10)]
    assert len({entry.sources["native_source"] for entry in entries}) == 10
    assert sum(row["phase"] == "rates" for row in result["jobs"]) == (0 if reuse else 10)
    # The queued export reads the prepared campaign snapshot.
    assert scheduler.calls[-1][0][-7] == str(output.resolve() / "campaign.yaml")
    count = len(scheduler.calls)
    for job in scheduler.states:
        scheduler.states[job] = "COMPLETED"
    resumed = execution.submit(cohort, prepare(cohort, path, output, resume=True), output, resume=True)
    assert resumed == result and len(scheduler.calls) == count


def test_last_condition_mismatch_rejects_before_preparation(cohort, tmp_path):
    path, document = resources(tmp_path / "resources", cohort)
    archive_path = Path(document["conditions"][9]["rate_code_reference"])
    report = json.loads((archive_path / "probe/probe.json").read_text())
    report["config"]["stimulus"]["direction_deg"] = 90
    archive(archive_path, "SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT", {"probe/probe.json": report})
    with pytest.raises(ValueError, match="VISUAL_RECIPE_DIFFERS"):
        prepare(cohort, path, tmp_path / "execution")
    assert not (tmp_path / "execution").exists()


def test_reused_rate_from_another_circuit_rejects(cohort, tmp_path):
    path, document = resources(tmp_path / "resources", cohort, reuse=True)
    archive_path = Path(document["conditions"][9]["rate_source"])
    report = json.loads((archive_path / "probe/probe.json").read_text())
    report["input_construction_binding"]["delivery_sha256"] = "another circuit"
    archive(archive_path, "SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT", {"probe/probe.json": report})
    with pytest.raises(ValueError, match="REUSED_RATE_CIRCUIT_OR_CONDITION_DIFFERS"):
        prepare(cohort, path, tmp_path / "execution")
    assert not (tmp_path / "execution").exists()


def test_changed_recipe_rejects_before_submission(cohort, tmp_path, monkeypatch):
    path, _ = resources(tmp_path / "resources", cohort)
    output = tmp_path / "execution"
    plan = prepare(cohort, path, output)
    recipe_path = output / "recipes/condition_09.yaml"
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe["background"]["starting_noise_seed"] += 1
    recipe_path.write_text(yaml.safe_dump(recipe))
    scheduler = Scheduler()
    monkeypatch.setattr(execution.subprocess, "run", scheduler)
    with pytest.raises(ValueError, match="PREPARED_RECIPE_CHANGED"):
        execution.submit(cohort, plan, output)
    assert not scheduler.calls


def test_unknown_submission_is_not_retried(cohort, tmp_path, monkeypatch):
    path, _ = resources(tmp_path / "resources", cohort)
    output = tmp_path / "execution"
    plan = prepare(cohort, path, output)

    def disconnect(*args, **kwargs):
        raise OSError("synthetic transport outcome unknown")

    monkeypatch.setattr(execution.subprocess, "run", disconnect)
    with pytest.raises(OSError, match="outcome unknown"):
        execution.submit(cohort, plan, output)
    scheduler = Scheduler()
    monkeypatch.setattr(execution.subprocess, "run", scheduler)
    with pytest.raises(ValueError, match="SUBMISSION_OUTCOME_UNKNOWN"):
        execution.submit(cohort, plan, output, resume=True)
    assert not scheduler.calls


def test_failed_retained_job_is_not_automatically_replaced(cohort, tmp_path, monkeypatch):
    path, _ = resources(tmp_path / "resources", cohort, reuse=True)
    output = tmp_path / "execution"
    plan = prepare(cohort, path, output)
    scheduler = Scheduler()
    monkeypatch.setattr(execution.subprocess, "run", scheduler)
    execution.submit(cohort, plan, output)
    scheduler.states[700001] = "CANCELLED"
    with pytest.raises(ValueError, match="RETAINED_JOB_NOT_SUCCESSFUL"):
        execution.submit(cohort, plan, output, resume=True)
    assert len(scheduler.calls) == 21


@pytest.mark.parametrize("field", ("parameter_domain", "task_protocol", "observation_map"))
def test_unfrozen_campaign_rejects_through_real_cli(cohort, tmp_path, field):
    document = yaml.safe_load(CONFIG.read_text())
    document["simulation_campaign"]["scenarios"]["v1_visual"][field]["frozen"] = False
    config = tmp_path / "unfrozen_campaign.yaml"
    config.write_text(yaml.safe_dump(document, sort_keys=False))
    output = tmp_path / "rejected"
    result = subprocess.run([sys.executable, "-B", "-m", "population_dynamics.v1_production_execution",
        "--repo", str(REPO), "--config", str(config), "--qualification", str(cohort.qualification_path),
        "--identity-table", str(cohort.identity_table), "--resources", str(tmp_path / "absent.yaml"),
        "--output", str(output), "--submit"], text=True, capture_output=True)
    assert result.returncode != 0 and "BENCHMARK_PRODUCTION_NOT_QUALIFIED" in result.stderr
    assert not output.exists()


def test_export_launcher_routes_existing_cli_and_removes_scratch(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    record = tmp_path / "srun.json"
    srun = binary / "srun"
    srun.write_text(f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
                    "Path(os.environ['ROUTING_RECORD']).write_text(json.dumps(dict(args=sys.argv[1:],pythonpath=os.environ['PYTHONPATH'])))\n")
    srun.chmod(0o755)
    environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
        SLURMD_NODENAME="cpu03", SLURM_JOB_ID="synthetic-routing", SLURM_JOB_GPUS="",
        NRNSIM_BENCHMARK_PYTHON=sys.executable, ROUTING_RECORD=str(record))
    arguments = [str(REPO), "campaign.yaml", "qualification.json", "identities.csv", "0", "1", "sources.json", "block"]
    command = ["bash", str(REPO / "Libs/scripts/run_v1_production_block_export.sbatch"), *arguments]
    result = subprocess.run(command, env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    called = json.loads(record.read_text())
    assert "population_dynamics.benchmark_production_block" in called["args"]
    assert called["args"][-2:] == ["--output", "block"]
    assert called["args"][called["args"].index("--config") + 1] == "campaign.yaml"
    assert not Path(called["pythonpath"]).exists()
    record.unlink()
    result = subprocess.run(command, env=dict(environment, SLURMD_NODENAME="cpu04"), text=True, capture_output=True)
    assert result.returncode != 0 and not record.exists()


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED", "TIMEOUT"])
def test_resource_barrier_omits_terminal_failure_and_excludes_gpu(monkeypatch, state):
    scheduler = Scheduler()
    scheduler.states.update({10: state, 11: "COMPLETED", 12: "RUNNING"})
    monkeypatch.setattr(execution.subprocess, "run", scheduler)
    submission = dict(jobs=[dict(phase="spikes", job_id=10),
                            dict(phase="source_plan", job_id=11),
                            dict(phase="gpu_primary", job_id=12)])
    assert execution.source_barrier_jobs(submission) == ()


def test_cancelled_plan_does_not_release_running_spike_or_inherited_lane(monkeypatch):
    scheduler = Scheduler()
    scheduler.states.update({10: "RUNNING", 11: "CANCELLED", 9: "RUNNING"})
    monkeypatch.setattr(execution.subprocess, "run", scheduler)
    submission = dict(jobs=[dict(phase="spikes", job_id=10), dict(phase="source_plan", job_id=11)],
                      start_after_any_job_ids=[9])
    assert execution.source_barrier_jobs(submission) == (9, 10)


@pytest.mark.parametrize("state", [None, "UNKNOWN"])
def test_unknown_barrier_state_stops_dispatch(monkeypatch, state):
    scheduler = Scheduler()
    if state is not None:
        scheduler.states[10] = state
    monkeypatch.setattr(execution.subprocess, "run", scheduler)
    with pytest.raises(ValueError, match="JOB_STATE_UNKNOWN"):
        execution.source_barrier_jobs(dict(jobs=[dict(phase="spikes", job_id=10)]))


def test_fresh_submission_uses_controller_when_accounting_lags(monkeypatch):
    calls = []
    def scheduler(command, **kwargs):
        calls.append(command)
        if command[0] == "sacct":
            return subprocess.CompletedProcess(command, 0, "11|CANCELLED|0:0|\n", "")
        assert command == ["scontrol", "show", "job", "-o", "10"]
        return subprocess.CompletedProcess(command, 0, "JobId=10 JobState=RUNNING ExitCode=0:0", "")
    monkeypatch.setattr(execution.subprocess, "run", scheduler)
    assert execution.source_barrier_jobs(dict(jobs=[
        dict(phase="spikes", job_id=10), dict(phase="source_plan", job_id=11)])) == (10,)
    assert len(calls) == 2


@pytest.mark.parametrize("partial_output", ["", "10|COMPLETED|0:0|\n"])
def test_failed_accounting_query_uses_exact_controller_states(monkeypatch, partial_output):
    calls = []

    def scheduler(command, **kwargs):
        calls.append(command)
        if command[0] == "sacct":
            return subprocess.CompletedProcess(command, 1, partial_output, "Accounting connection refused")
        assert command[:4] == ["scontrol", "show", "job", "-o"]
        job = command[4]
        state = "RUNNING" if job == "10" else "COMPLETED"
        return subprocess.CompletedProcess(command, 0, f"JobId={job} JobState={state} ExitCode=0:0", "")

    monkeypatch.setattr(execution.subprocess, "run", scheduler)
    with pytest.warns(RuntimeWarning, match="ACCOUNTING_UNAVAILABLE"):
        assert execution.source_barrier_jobs(dict(jobs=[
            dict(phase="spikes", job_id=10), dict(phase="source_plan", job_id=11)])) == (10,)
    assert len(calls) == 3


def test_unavailable_accounting_and_controller_do_not_release_source_lane(monkeypatch):
    def scheduler(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "Accounting unavailable" if command[0] == "sacct"
                                           else "Invalid job id specified")

    monkeypatch.setattr(execution.subprocess, "run", scheduler)
    with pytest.warns(RuntimeWarning, match="ACCOUNTING_UNAVAILABLE"):
        with pytest.raises(ValueError, match="JOB_STATE_UNKNOWN"):
            execution.source_barrier_jobs(dict(jobs=[dict(phase="spikes", job_id=10)]))
