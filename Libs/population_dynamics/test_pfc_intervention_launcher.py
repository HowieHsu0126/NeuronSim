"""Thin PFC launcher admission, composition and archival control flow."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_pfc_intervention_experiment.py"
SPEC = importlib.util.spec_from_file_location("pfc_intervention_launcher_under_test", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
HELPER_SPEC = importlib.util.spec_from_file_location(
    "pfc_campaign_helper_under_test", SCRIPT.with_name("run_population_trajectory_campaign.py"))
helper = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(helper)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return dict(path="", bytes=path.stat().st_size, sha256=helper.digest(path))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    preparation, software = tmp_path / "preparation", tmp_path / "software"
    preparation.mkdir(); software.mkdir()
    names = [launcher.FROZEN_HELPER, launcher.RESOURCE_PROBE, "Libs/population_dynamics/core_model.py"]
    frozen = []
    for name in names:
        row = write(preparation / name, "frozen " + name)
        row["path"] = name
        frozen.append(row)
        write(software / name, (preparation / name).read_text())
    added = "Libs/population_dynamics/new_entry.py"
    added_row = write(software / added, "tested caller")
    added_row["path"] = added
    proof = dict(files=frozen, training_seeds=[41, 42, 43], input_sha256="a" * 64)
    write(preparation / "preparation.json", json.dumps(proof))
    software_rows = [*deepcopy(frozen), added_row]
    write(software / "source_snapshot.json", json.dumps(dict(files=software_rows)))
    reference = tmp_path / "reference.json"
    write(reference, '{"reference": "bound"}')
    config = tmp_path / "intervention.yaml"
    write(config, yaml.safe_dump(dict(reference_sha256=helper.digest(reference))))
    seed_files = ("training/run.json", "training/best.pt", "forecasts/evaluation.json", "replay.json",
                  "phases/fit/receipt.json", "phases/evaluate/receipt.json", "phases/replay/receipt.json")
    cells, admitted, events, commands = {}, {}, [], []
    for seed in launcher.SEEDS:
        root = tmp_path / f"seed_{seed}"
        root.mkdir()
        index = {}
        for name in seed_files:
            row = write(root / name, f"bound seed {seed} {name}")
            row["path"] = name
            index[name] = row
        cell_proof = dict(preparation_sha256=helper.digest(preparation / "preparation.json"))
        write(root / "delivery.json", json.dumps(cell_proof))
        cell = SimpleNamespace(root=root, files=index, proof=cell_proof, path=lambda name, root=root: root / name)
        binding = dict(delivery_sha256=helper.digest(root / "delivery.json"),
                       consumed_files={name: row["sha256"] for name, row in index.items()})
        admitted[seed] = dict(cell=cell, binding=binding,
            run=dict(trainer_sources={"core_model.py": frozen[-1]["sha256"]}))
        cells[seed] = root
    def admission(given):
        assert given == cells
        events.append("admission")
        return admitted
    caller = SimpleNamespace(validate_config=lambda value: value, admit_seed_cells=admission, SEED_CELL_FILES=seed_files)
    def stage(prepared, work):
        assert events == ["admission"]
        events.append("stage")
        for row in frozen:
            helper.copy_bound(prepared / row["path"], work / row["path"], row)
        write(work / "inputs/cohort.json", "{}")
        write(work / "execution_path_map.json", "{}")
        return object(), proof
    return_codes, outcome_overrides = {}, {}
    def phase(work, name, command, probe):
        events.append(name); commands.append(command)
        assert command[command.index("--preparation") + 1] == str(work)
        assert command[command.index("--inputs") + 1] == str(work / "inputs/cohort.json")
        assert command.count("--seed-cell") == (0 if name == "source_outcomes" else 3)
        if name == "source_outcomes":
            assert command[command.index("--search") + 1] == str(work / "results")
            assert command[command.index("--replay") + 1] == str(work / "replay")
        assert (work / "preparation.json").is_file()
        assert (work / names[-1]).read_text() == (preparation / names[-1]).read_text()
        assert (work / added).read_text() == "tested caller"
        directory = work / "phases" / name
        code = return_codes.get(name, 0)
        helper.write_json(directory / "receipt.json", dict(status="PASS" if code == 0 else "FAILED", exit_code=code))
        if code == 0:
            output = Path(command[command.index("--output") + 1])
            replay = name == "intervention_replay"
            report = dict(status="PASS_PFC_INTERVENTION_EXPERIMENT_REPLAY" if replay
                          else "COMPLETE_PFC_INTERVENTION_EXPERIMENT",
                candidate_rows=90, seed_block_cases=9, generated_unit_count_values_across_completed_cases=141120000000,
                per_bin_full_unit_counts_saved=False, per_bin_full_unit_counts_exact_replay_claimed=False,
                training_performed=False, native_execution_performed=False, success_tolerance_qualified=False,
                memory_recovery_qualified=False, scientific_claim_qualified=False)
            if replay:
                report["saved_candidate_array_values_compared"] = 344982330
            if name == "source_outcomes":
                report = dict(status="COMPLETE_PFC_INTERVENTION_SOURCE_OUTCOMES",
                    candidate_rows=90, native_source_episodes=30, native_source_blocks=3,
                    native_circuits=1, policy_rows=3, program_rows=10,
                    observed_unit_bin_values_aggregated=2940000000, newly_generated_samples=0,
                    all_candidate_outcomes_retained=True, observed_prefixes_equal=True,
                    native_future_noise_pairing_verified=False, independent_intervention_qualified=False,
                    success_tolerance_qualified=False, memory_recovery_qualified=False,
                    scientific_claim_qualified=False, native_execution_performed=False)
                report.update(outcome_overrides)
            helper.write_json(output / "analysis.json", report)
        return code
    monkeypatch.setattr(launcher, "load_frozen_helper", lambda path: (helper, proof))
    monkeypatch.setattr(launcher, "validated_caller", lambda path: caller)
    monkeypatch.setattr(helper, "stage_pfc", stage)
    monkeypatch.setattr(helper, "software_closure", lambda root, modules: [added])
    monkeypatch.setattr(helper, "phase", phase)
    monkeypatch.setattr(launcher, "SCRATCH_ROOT", tmp_path)
    for key, value in dict(SLURMD_NODENAME="cpu03", SLURM_JOB_PARTITION="gpu",
                           SLURM_JOB_ID="test", SLURM_CPUS_PER_TASK="4").items():
        monkeypatch.setenv(key, value)
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        monkeypatch.delenv(key, raising=False)
    arguments = dict(preparation=preparation, software=software, reference=reference, config=config,
                     seed_cells=cells, work=tmp_path / "work", output=tmp_path / "delivery")
    return SimpleNamespace(arguments=arguments, events=events, commands=commands, admitted=admitted,
                           return_codes=return_codes, outcome_overrides=outcome_overrides, caller=caller, proof=proof, software_rows=software_rows)


def test_terminal_admission_precedes_bulk_stage_and_bound_composition_runs_then_replays(setup):
    report = launcher.launch(**setup.arguments)
    assert report["status"] == "PASS" and report["exit_code"] == 0
    assert setup.events == ["admission", "stage", "intervention", "intervention_replay", "source_outcomes"]
    work, output = setup.arguments["work"], setup.arguments["output"]
    assert (work / "archive_complete").is_file()
    assert json.loads((work / "preparation.json").read_text()) == setup.proof
    snapshot = json.loads((output / "source_snapshot.json").read_text())
    assert snapshot["protected_sources"]["Libs/population_dynamics/core_model.py"] == setup.proof["files"][-1]["sha256"]
    assert not (output / "inputs").exists()
    assert (output / "admission.json").is_file()
    assert json.loads((output / "staged_seed_cells.json").read_text())["completed_forecast_payloads_copied"] is False
    assert "--replay-from" not in setup.commands[0] and "--replay-from" in setup.commands[1]
    assert report["generation_and_replay_complete"] and report["source_outcome_readback_performed"]
    assert report["native_source_episodes_reused"] == 30


def test_pending_seed_never_stages_bulk_inputs_but_archives_failure(setup):
    setup.caller.admit_seed_cells = lambda cells: (_ for _ in ()).throw(ValueError("pending seed43"))
    report = launcher.launch(**setup.arguments)
    assert report["status"] == "FAILED" and report["exit_code"] != 0
    assert setup.events == []
    assert (setup.arguments["output"] / "failure.json").is_file()
    assert not setup.commands


def test_protected_source_difference_is_fatal_before_staging(setup):
    name = "Libs/population_dynamics/core_model.py"
    software = setup.arguments["software"]
    changed = write(software / name, "different protected implementation")
    rows = deepcopy(setup.software_rows)
    for row in rows:
        if row["path"] == name:
            row.update(bytes=changed["bytes"], sha256=changed["sha256"])
    write(software / "source_snapshot.json", json.dumps(dict(files=rows)))
    report = launcher.launch(**setup.arguments)
    assert report["status"] == "FAILED" and "PROTECTED_SOURCE_DIFFERENCE" in report["error"]
    assert setup.events == ["admission"]


@pytest.mark.parametrize("phase,code", [("intervention", 2), ("intervention", 3), ("intervention_replay", 2), ("source_outcomes", 2), ("source_outcomes", 3)])
def test_phase_failure_propagates_and_archives_complete_receipts(setup, phase, code):
    setup.return_codes[phase] = code
    report = launcher.launch(**setup.arguments)
    assert report["exit_code"] == code and report["failed_phase"] == phase
    assert report["status"] == ("RESOURCE_BLOCKED" if code == 3 else "FAILED")
    assert (setup.arguments["output"] / f"phases/{phase}/receipt.json").is_file()
    assert (setup.arguments["work"] / "archive_complete").is_file()
    if phase == "intervention":
        assert "intervention_replay" not in setup.events
    if phase == "source_outcomes":
        assert report["generation_and_replay_complete"]
        assert not report["source_outcome_readback_performed"]
        assert (setup.arguments["output"] / "results/analysis.json").is_file()
        assert (setup.arguments["output"] / "replay/analysis.json").is_file()


def test_failed_archive_never_enables_scratch_cleanup(setup, monkeypatch):
    monkeypatch.setattr(helper, "archive", lambda *args: (_ for _ in ()).throw(OSError("archive unavailable")))
    with pytest.raises(OSError, match="archive unavailable"):
        launcher.launch(**setup.arguments)
    assert not (setup.arguments["work"] / "archive_complete").exists()
    assert (setup.arguments["work"] / "execution.json").is_file()


def test_reference_binding_rejects_before_admission(setup):
    setup.arguments["reference"].write_text("changed")
    report = launcher.launch(**setup.arguments)
    assert report["status"] == "FAILED" and "REFERENCE_CONFIG_BINDING_CHANGED" in report["error"]
    assert setup.events == []


def test_cli_keeps_explicit_cells_and_optional_reuse(tmp_path):
    arguments = []
    for name in ("preparation", "software", "reference", "config", "work", "output"):
        arguments.extend(["--" + name, str(tmp_path / name)])
    for seed in launcher.SEEDS:
        arguments.extend(["--seed-cell", str(seed), str(tmp_path / str(seed))])
    arguments.extend(["--reuse-completed-from", str(tmp_path / "previous/results")])
    with patch.object(launcher, "launch", return_value=dict(exit_code=0)) as launch:
        assert launcher.main(arguments) == 0
    assert set(launch.call_args.args[4]) == {41, 42, 43}
    assert launch.call_args.kwargs["reuse_completed_from"] == tmp_path / "previous/results"


def test_batch_placement_and_cleanup_contract():
    text = SCRIPT.with_suffix(".sbatch").read_text()
    for required in ("--job-name=NrnSim_PfcIntervention", "--partition=gpu", "--nodelist=cpu03",
                     "--cpus-per-task=4", "--mem=40G", "--gres=none", "--no-requeue",
                     "PROJECT_PREFIX", 'task_user=$(id -un)', 'if [ -f "$work/archive_complete" ]'):
        assert required in text
    assert "--array=" not in text and "--exclusive" not in text and "--gres=gpu" not in text
    assert "/tmp/neuronsim" not in text


def test_source_readback_never_counts_model_seeds_as_additional_native_runs(setup):
    setup.outcome_overrides["native_source_episodes"] = 90
    report = launcher.launch(**setup.arguments)
    assert report["status"] == "FAILED" and report["exit_code"] != 0
    assert "COMPLETE_BOUND_NATIVE_SOURCE_READBACK_REQUIRED" in report["error"]
    assert report["generation_and_replay_complete"]
    assert not report["source_outcome_readback_performed"]
