"""Node qualification and routing tests use disposable source metadata."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest
import yaml

from . import v1_source_node_qualification as node
from . import v1_production_execution as execution
from . import v1_noise_block_expansion as expansion
from . import v1_glif_production as glif
from .provenance import sha256_file
from .test_v1_production_execution import cohort, resources, prepare, Scheduler, archive
from .test_v1_glif_production import gpu_cohort, gpu_resources, GpuScheduler


def _deliver(root, report, values, clock):
    root.mkdir()
    probe = root / "probe"
    probe.mkdir()
    (probe / "probe.json").write_text(json.dumps(report))
    with h5py.File(probe / "firing_rates_f_tot.h5", "w") as output:
        output["time"] = clock
        output["firing_rates_Hz"] = values
    entries = [dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                    sha256=sha256_file(path)) for path in probe.iterdir()]
    (root / "delivery.json").write_text(json.dumps(dict(exit_code=0, files=entries)))


@pytest.mark.parametrize("change", [None, "value", "clock", "config", "identity"])
def test_rates_compare_reference_units_clock_and_config(tmp_path, change):
    settings = dict(unit_coverage="all_source_units", stimulus="fixed",
                    spike_generation=dict(starting_noise_seed=101),
                    background=dict(starting_noise_seed=202))
    report = dict(config=settings, input_construction_binding={"source": "fixture"},
                  source_units=17400, source_unit_ids=list(range(17400)), rate_shape=[17400, 3])
    values = np.arange(17400 * 3, dtype=np.float64).reshape(17400, 3)
    clock = np.arange(3, dtype=np.float64)
    _deliver(tmp_path / "before", report, values, clock)
    selected = list(range(0, 28, 2))
    current = deepcopy(report)
    current.update(source_units=14, source_unit_ids=selected, rate_shape=[14, 3])
    current["config"]["unit_coverage"] = "model_representatives"
    current["config"]["spike_generation"]["starting_noise_seed"] = 909
    observed = values[selected].copy()
    if change == "value":
        observed[0, 0] += 1
    elif change == "clock":
        clock[1] += .1
    elif change == "config":
        current["config"]["stimulus"] = "different"
    elif change == "identity":
        current["input_construction_binding"] = {"source": "different"}
    _deliver(tmp_path / "after", current, observed, clock)
    work = tmp_path / "work"
    work.mkdir()
    if change is None:
        result = node.compare_rates(tmp_path / "before", tmp_path / "after", work, complete=False)
        assert result["compared_rate_values"] == 42 and result["mismatched_rate_values"] == 0
    else:
        with pytest.raises(ValueError, match="NODE_QUALIFICATION_"):
            node.compare_rates(tmp_path / "before", tmp_path / "after", work, complete=False)


def _qualification(root, cohort, document):
    root.mkdir()
    launchers = {}
    bindings = {}
    for role in ("rates", "spikes"):
        path = root / (role + ".sbatch")
        path.write_text("# synthetic routing fixture\n")
        digest = sha256_file(path)
        launchers[role] = dict(path=str(path), sha256=digest)
        bindings[str(path)] = digest
    cases = []
    for index, row in enumerate(document["conditions"]):
        cases.append(dict(config=cohort.v1_visual_config(cohort.identity(0, 1, index)),
                          rate_code_reference=row["rate_code_reference"], spike_canary=row["spike_canary"]))
    for block in (0, 2, 3, 4, 5, 6, 7):
        cases.append(dict(cases[9], config=cohort.v1_visual_config(cohort.identity(0, block, 9))))
    result = dict(status="PASS_V1_SOURCE_NODE_QUALIFICATION", node="cpu05", partition="cpu",
                  source_qualification_sha256=cohort.qualification_sha256,
                  completed_cases=dict(rates_canary=17, spikes_canary=17, rates_full=3, spikes_full=10),
                  launchers=launchers, cases=cases, source_bindings=bindings)
    path = root / "qualification.json"
    path.write_text(json.dumps(result))
    return path, result


@pytest.mark.parametrize("change", ["missing_case", "qualification", "file"])
def test_node_route_rejects_incomplete_or_changed_evidence(cohort, tmp_path, change):
    _, document = resources(tmp_path / "resources", cohort)
    path, proof = _qualification(tmp_path / "proof", cohort, document)
    if change == "missing_case":
        proof["completed_cases"]["spikes_full"] = 9
        path.write_text(json.dumps(proof))
    elif change == "qualification":
        proof["source_qualification_sha256"] = "different"
        path.write_text(json.dumps(proof))
    else:
        Path(proof["launchers"]["rates"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="NODE_QUALIFICATION_"):
        node.load_source_node_qualification(path, cohort.qualification_sha256, {})


def test_qualified_source_node_routes_sources_and_export(cohort, tmp_path, monkeypatch):
    path, document = resources(tmp_path / "resources", cohort)
    qualification, proof = _qualification(tmp_path / "proof", cohort, document)
    document["source_node_qualification"] = str(qualification)
    path.write_text(yaml.safe_dump(document))
    output = tmp_path / "execution"
    plan = prepare(cohort, path, output)
    scheduler = Scheduler()

    def run(command, **kwargs):
        if command[0] == "sinfo":
            return subprocess.CompletedProcess(command, 0, "cpu05|cpu|mix\n", "")
        return scheduler(command, **kwargs)

    monkeypatch.setattr(execution.subprocess, "run", run)
    report = execution.submit(cohort, plan, output)
    for (command, _), job in zip(scheduler.calls, report["jobs"]):
        if job["phase"] in ("rates", "spikes"):
            assert "--nodelist=cpu05" in command and "--partition=cpu" in command
            assert proof["launchers"][job["phase"]]["path"] in command
        elif job["phase"] == "export":
            assert "--nodelist=cpu05" in command and "--partition=cpu" in command
            assert plan["launchers"]["export"] in command
            launcher = Path(plan["launchers"]["export"]).read_text()
            assert '[[ "${SLURMD_NODENAME:-}" =~ ^(cpu03|cpu05)$ &&' in launcher
        else:
            assert "--nodelist=cpu03" in command
        assert "--gres=none" in command
    assert plan["source_cpu"]["qualification"] == str(qualification)


def test_collector_rejects_pending_full_comparisons(tmp_path):
    manifest = dict(node="cpu05", numerical_source_modules_changed=False, canonical_launchers_changed=False,
                    full_rate_cases=[0, 8, 9], full_spike_cases=list(range(10)),
                    cases=[{} for _ in range(17)], software_bindings={}, launchers={})
    source = tmp_path / "source.json"
    source.write_text("{}")
    manifest.update(source_qualification=str(source), source_qualification_sha256=sha256_file(source))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "qualification.json"
    with pytest.raises(FileNotFoundError):
        node.collect_qualification(path, output)
    assert not output.exists()


@pytest.mark.parametrize("qualified,concurrency", [(False, 2), (True, 7), (True, 3)])
def test_independent_direction_dispatch_and_success_gates(cohort, tmp_path, monkeypatch, qualified, concurrency):
    _, document = resources(tmp_path / "resources", cohort)
    root = tmp_path / "cohort/circuit_006/noise_block_expansion"
    root.mkdir(parents=True)
    template = dict(document, circuit_index=6)
    if qualified:
        proof_path, _ = _qualification(tmp_path / "proof", cohort, document)
        template["source_node_qualification"] = str(proof_path)
    (root / "execution_template.yaml").write_text(yaml.safe_dump(template))
    records = [dict(block_index=i, direction_deg=45 * i,
                    recipe=str(tmp_path / f"recipe_{i}.yaml"),
                    rate_code_reference=document["conditions"][9]["rate_code_reference"],
                    spike_canary=document["conditions"][9]["spike_canary"]) for i in range(1, 8)]
    prepared = dict(status="PREPARED_SEVEN_LOW_CONTRAST_DIRECTIONS", circuit_index=6,
                    repo=str(tmp_path), input_source=document["input_source"], records=records)
    if qualified:
        prepared["rate_concurrency"] = concurrency
    (root / "preparation.json").write_text(json.dumps(prepared))
    monkeypatch.setattr(expansion, "_cohort", lambda _: cohort)
    scheduler = Scheduler()

    def run(command, **kwargs):
        if command[0] == "sinfo":
            return subprocess.CompletedProcess(command, 0, "mix\n", "")
        return scheduler(command, **kwargs)

    monkeypatch.setattr(expansion.subprocess, "run", run)
    result = expansion.submit_directions(tmp_path / "cohort", 6)
    assert len(scheduler.calls) == 15 and result["rate_concurrency"] == concurrency
    for i, row in enumerate(result["jobs"]):
        rate_command = scheduler.calls[2 * i][0]
        prep_command = scheduler.calls[2 * i + 1][0]
        assert "--gres=none" in rate_command
        assert "--nodelist=" + ("cpu05" if qualified else "cpu03") in rate_command
        assert "--dependency=afterok:" + str(row["rate_job"]) in prep_command
        if i >= concurrency:
            predecessor = result["jobs"][i - concurrency]["rate_job"]
            assert "--dependency=afterany:" + str(predecessor) in rate_command
            assert "--dependency=afterok:" + str(predecessor) not in rate_command
        else:
            assert not any(x.startswith("--dependency=") for x in rate_command)
    completion = scheduler.calls[-1][0]
    assert "--dependency=afterok:" + ":".join(str(row["preparation_job"]) for row in result["jobs"]) in completion


def test_gpu_block_keeps_all_source_plan_and_export_work_on_bulk_cpu(gpu_cohort, tmp_path, monkeypatch):
    path, document = gpu_resources(tmp_path / "resources", gpu_cohort, reuse=False)
    qualification, _ = _qualification(tmp_path / "proof", gpu_cohort, document)
    document["source_node_qualification"] = str(qualification)
    path.write_text(yaml.safe_dump(document))
    output = tmp_path / "execution"
    plan = prepare(gpu_cohort, path, output)
    scheduler = GpuScheduler()

    def run(command, **kwargs):
        if command[0] == "sinfo" and "--nodes=cpu05" in command:
            return subprocess.CompletedProcess(command, 0, "cpu05|cpu|mix\n", "")
        return scheduler(command, **kwargs)

    monkeypatch.setattr(execution.subprocess, "run", run)
    report = execution.submit(gpu_cohort, plan, output)
    assert plan["launchers"]["source_plan"] == plan["launchers"]["export"]
    for (command, _), row in zip(scheduler.calls, report["jobs"]):
        gpu = row["phase"] in ("gpu_primary", "gpu_replay")
        assert "--nodelist=" + ("cpu06" if gpu else "cpu05") in command
        assert ("--gres=gpu:1" if gpu else "--gres=none") in command



@pytest.mark.parametrize("node_name", ["cpu03", "cpu05", "cpu06"])
def test_bulk_export_nodes_reach_source_admission(cohort, tmp_path, monkeypatch, node_name):
    monkeypatch.setenv("SLURMD_NODENAME", node_name)
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-export-placement")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.setattr(glif.production, "glif_backend_qualification", lambda _: None)

    def admission(*args):
        raise RuntimeError("REACHED_SOURCE_ADMISSION")

    monkeypatch.setattr(glif, "bind_sources", admission)
    with pytest.raises(RuntimeError, match="REACHED_SOURCE_ADMISSION"):
        with glif.prepared_source_block(cohort, [object()], tmp_path):
            pytest.fail("Synthetic admission must stop before data processing")


@pytest.mark.parametrize("failure", [None, "qualification", "budget"])
def test_qualified_recovery_keeps_retained_spikes_and_recipes(gpu_cohort, tmp_path, monkeypatch, failure):
    path, document = gpu_resources(tmp_path / "resources", gpu_cohort, reuse=True)
    original = tmp_path / "original"
    plan = prepare(gpu_cohort, path, original)
    actual_repo = Path(plan["repo"])
    plan["repo"] = str(tmp_path)
    export_source = tmp_path / "Libs/scripts/run_v1_production_block_export.sbatch"
    export_source.parent.mkdir(parents=True)
    export_source.write_text((actual_repo / "Libs/scripts/run_v1_production_block_export.sbatch").read_text())
    (original / "plan.json").write_text(json.dumps(plan))
    scheduler = GpuScheduler()
    def run(command, **kwargs):
        if command[0] == "sinfo":
            node_name = next(value.split("=", 1)[1] for value in command if value.startswith("--nodes="))
            partition = next(value.split("=", 1)[1] for value in command if value.startswith("--partition="))
            return subprocess.CompletedProcess(command, 0, f"{node_name}|{partition}|mix\n", "")
        return scheduler(command, **kwargs)
    monkeypatch.setattr(execution.subprocess, "run", run)
    first = execution.submit(gpu_cohort, plan, original)
    before = {name: (original / name).read_bytes() for name in ["plan.json", "submission.json", "campaign.yaml"]}
    for index, row in enumerate(first["jobs"]):
        scheduler.states[row["job_id"]] = "COMPLETED" if index < 2 else ("FAILED" if index == 2 else "CANCELLED")
    for index in range(2):
        root = tmp_path / execution.FAMILY / "visual_inputs" / ("spikes_" + str(first["jobs"][index]["job_id"]))
        recipe = gpu_cohort.v1_visual_config(gpu_cohort.identity(0, 1, index))
        archive(root, "PASS_CONSTRUCTED_VISUAL_INPUT_FULL", {"visual_inputs/report.json": dict(
            config=recipe, exit_code=0, complete_lgn_population=True,
            input_construction_binding=dict(delivery_sha256=sha256_file(Path(document["input_source"]) / "delivery.json")))})
        delivery = json.loads((root / "delivery.json").read_text())
        delivery.update(rate_archive=document["conditions"][index]["rate_source"], input_construction_archive=document["input_source"])
        (root / "delivery.json").write_text(json.dumps(delivery))
    qualification, proof = _qualification(tmp_path / "proof", gpu_cohort, document)
    for index, case in enumerate(proof["cases"]):
        candidate = archive(tmp_path / f"cpu05_canary_{index}", "PASS_CONSTRUCTED_VISUAL_INPUT_CANARY",
                            {"visual_inputs/report.json": dict(config=case["config"])})
        case["spike_canary"] = str(candidate)
    if failure == "qualification":
        proof["completed_cases"]["spikes_full"] = 9
    qualification.write_text(json.dumps(proof))
    budgets = dict(rates=240, spikes=15, gpu=120, export=30) if failure != "budget" else {"gpu": 120}
    output = tmp_path / "recovery"
    calls = len(scheduler.calls)
    if failure:
        with pytest.raises(ValueError, match="NODE_QUALIFICATION_|WALLTIME_MINUTES_INVALID"):
            execution.recover_source_block(gpu_cohort, original, output,
                source_node_qualification=qualification, walltime_minutes=budgets, gpu_node="cpu05")
        assert not output.exists() and len(scheduler.calls) == calls
        return
    result = execution.recover_source_block(gpu_cohort, original, output,
        source_node_qualification=qualification, walltime_minutes=budgets, gpu_node="cpu05")
    recovered = json.loads((output / "plan.json").read_text())
    assert recovered["source_cpu"] == dict(node="cpu05", partition="cpu", qualification=str(qualification))
    assert recovered["walltime_minutes"] == budgets
    assert len(scheduler.calls) - calls == 12
    assert [row["job_id"] for row in result["jobs"][:2]] == [row["job_id"] for row in first["jobs"][:2]]
    assert all(row["reused"] for row in result["jobs"][:2])
    for index, condition in enumerate(recovered["conditions"]):
        recipe = gpu_cohort.v1_visual_config(gpu_cohort.identity(0, 1, index))
        assert condition["identity"] == plan["conditions"][index]["identity"]
        assert condition["rate_source"] == plan["conditions"][index]["rate_source"]
        assert condition["spike_canary"] == node.qualified_source_case(proof, recipe)["spike_canary"]
        assert (output / f"recipes/condition_{index:02d}.yaml").read_bytes() == (original / f"recipes/condition_{index:02d}.yaml").read_bytes()
    for (command, _), row in zip(scheduler.calls[calls:], result["jobs"][2:]):
        assert "--nodelist=cpu05" in command
        if row["phase"] in ("spikes", "source_plan", "export"):
            assert "--gres=none" in command
            budget = 15 if row["phase"] == "spikes" else 30
            assert "--time=" + str(budget) in command
    assert all((original / name).read_bytes() == contents for name, contents in before.items())
    assert (output / "campaign.yaml").read_bytes() == before["campaign.yaml"]
