"""Exercise the actual array launcher with disposable source-constructor fixtures."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import yaml

from .v1_circuit_construction_array import FAMILY, STAGES, ROUTING_CONFIG, ROUTING_SCHEMA, node_launcher_text


@pytest.fixture
def construction(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "Libs/scripts"
    scripts.mkdir(parents=True)
    (repo / "Libs/population_dynamics").symlink_to(Path(__file__).parent)
    circuits = []
    for index in (2, 3):
        config = tmp_path / f"circuit_{index}.yaml"
        config.write_text(f"builder_seed: {400000000 + index}\n")
        circuits.append(dict(circuit_index=index, config=str(config),
                             config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()))
    preparation = tmp_path / "preparation.json"
    preparation.write_text(json.dumps(dict(circuits=circuits)))
    fixture = tmp_path / "constructor.py"
    fixture.write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "stage,repo=sys.argv[1:3]\n"
        "status={'recurrent':'PASS_SOURCE_RECURRENT_BUILD_REPEATABILITY',"
        "'inputs':'PASS_SOURCE_INPUT_BUILD_REPEATABILITY'}[stage]\n"
        f"family=Path({str(FAMILY)!r})\n"
        "family=family/('recurrent_construction' if stage=='recurrent' else 'input_construction')\n"
        "target=Path(repo)/family/('run_'+os.environ['SLURM_JOB_ID'])\n"
        "target.mkdir(parents=True)\n"
        "(target/'arguments.json').write_text(json.dumps(sys.argv[2:]))\n"
        "(target/'environment.json').write_text(json.dumps({'input_code_reference':os.environ.get('NRNSIM_V1_INPUT_CODE_REFERENCE')}))\n"
        "code=int(os.environ.get('FIXTURE_EXIT_CODE','0'))\n"
        "if not os.environ.get('FIXTURE_MISSING_DELIVERY'):\n"
        " (target/'delivery.json').write_text(json.dumps(dict(status=status,"
        "exit_code=code,job_id=os.environ['SLURM_JOB_ID'],mode='full',node=os.environ['SLURMD_NODENAME'])))\n"
        "raise SystemExit(code)\n"
    )
    for stage, (launcher, _, _) in STAGES.items():
        (scripts / launcher).write_text(
            "#!/bin/bash\n# node not in ('cpu03', 'cpu04')\n# mode == 'full' and node != 'cpu03'\nexec "
            + shlex.join([sys.executable, str(fixture), stage]) + ' "$@"\n')
    return repo, preparation, tmp_path / "executions", tmp_path / "qualification"


def launch(construction, stage, index, raw_job_id, **extra):
    repo, preparation, output, qualification = construction
    env = dict(os.environ, SLURM_ARRAY_TASK_ID=str(index),
               SLURM_ARRAY_JOB_ID="1000" if stage == "recurrent" else "2000",
               SLURM_JOB_ID=str(raw_job_id), NRNSIM_V1_PYTHON=sys.executable,
               SLURMD_NODENAME="cpu03")
    env.update(extra)
    launcher = Path(__file__).parents[1] / "scripts/run_v1_circuit_construction_array.sbatch"
    return subprocess.run(["bash", str(launcher), str(repo), stage, str(preparation),
                           str(output), str(qualification)], env=env, text=True, capture_output=True)


def test_out_of_order_raw_job_ids_select_matching_parent(construction):
    repo, _, output, _ = construction
    for index, raw_job in ((3, 1101), (2, 1199)):
        result = launch(construction, "recurrent", index, raw_job)
        assert result.returncode == 0, result.stderr
    result = launch(construction, "inputs", 2, 2003)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "circuit_002/inputs/report.json").read_text())
    assert report["parent_job_id"] == "1199"
    assert report["array_task_id"] == report["circuit"]["circuit_index"] == 2
    arguments = json.loads((Path(report["archive"]) / "arguments.json").read_text())
    assert arguments[3] == str(repo / FAMILY / "recurrent_construction/run_1199")
    assert report["status"] == "COMPLETE_SOURCE_CONSTRUCTION"
    assert report["production_episodes"] == 0


def test_changed_config_rejected_before_constructor(construction):
    repo, preparation, output, _ = construction
    config = Path(json.loads(preparation.read_text())["circuits"][0]["config"])
    config.write_text("builder_seed: 41\n")
    result = launch(construction, "recurrent", 2, 1002)
    assert result.returncode != 0
    assert "V1_CONSTRUCTION_PREPARED_CONFIG_CHANGED" in result.stderr
    assert not output.exists() and not (repo / FAMILY).exists()


@pytest.mark.parametrize("extra,expected", [({"FIXTURE_EXIT_CODE": "7"}, 7),
                                          ({"FIXTURE_MISSING_DELIVERY": "1"}, 1)])
def test_failed_source_exit_and_missing_delivery_remain_failed(construction, extra, expected):
    _, _, output, _ = construction
    result = launch(construction, "recurrent", 2, 1002, **extra)
    assert result.returncode == expected
    report_path = output / "circuit_002/recurrent/report.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "FAILED_SOURCE_CONSTRUCTION"
    assert report["exit_code"] == expected
    original = report_path.read_bytes()
    child = launch(construction, "inputs", 2, 2002)
    assert child.returncode != 0
    assert "V1_CONSTRUCTION_MATCHING_RECURRENT_COMPLETION_REQUIRED" in child.stderr
    retry = launch(construction, "recurrent", 2, 1003)
    assert retry.returncode != 0
    assert report_path.read_bytes() == original


def test_missing_parent_prevents_input_constructor(construction):
    repo, _, _, _ = construction
    result = launch(construction, "inputs", 2, 2002)
    assert result.returncode != 0
    assert not (repo / FAMILY).exists()


def configure_bulk_node(construction):
    repo, preparation, _, _ = construction
    destination = repo / "node_launchers"
    destination.mkdir()
    launchers = {}
    for stage, (name, _, _) in STAGES.items():
        path = destination / name
        path.write_text(node_launcher_text(repo, stage, "cpu05"))
        launchers[stage] = str(path.relative_to(repo))
    config = repo / ROUTING_CONFIG
    config.parent.mkdir()
    config.write_text(yaml.safe_dump(dict(
        schema=ROUTING_SCHEMA,
        preparation_sha256=hashlib.sha256(preparation.read_bytes()).hexdigest(),
        nodes={"cpu05": dict(pilot_circuit_index=2, launchers=launchers)})))
    return config


def test_bulk_node_requires_full_pilot_for_each_constructor_stage(construction):
    _, _, output, _ = construction
    configure_bulk_node(construction)
    blocked = launch(construction, "recurrent", 3, 1103, SLURMD_NODENAME="cpu05")
    assert blocked.returncode != 0 and "COMPLETE_NODE_PILOT_REQUIRED" in blocked.stderr
    assert not output.exists()
    pilot = launch(construction, "recurrent", 2, 1102, SLURMD_NODENAME="cpu05")
    assert pilot.returncode == 0, pilot.stderr
    following = launch(construction, "recurrent", 3, 1103, SLURMD_NODENAME="cpu05")
    assert following.returncode == 0, following.stderr
    report = json.loads((output / "circuit_003/recurrent/report.json").read_text())
    assert report["node"] == "cpu05" and report["node_route"]["is_pilot"] is False
    assert report["node_route"]["qualification_report"].endswith("circuit_002/recurrent/report.json")
    blocked = launch(construction, "inputs", 3, 2103, SLURMD_NODENAME="cpu05")
    assert blocked.returncode != 0 and "COMPLETE_NODE_PILOT_REQUIRED" in blocked.stderr
    for index in (2, 3):
        result = launch(construction, "inputs", index, 2100 + index, SLURMD_NODENAME="cpu05")
        assert result.returncode == 0, result.stderr
    report = json.loads((output / "circuit_003/inputs/report.json").read_text())
    assert report["parent_job_id"] == "1103" and report["production_episodes"] == 0


def test_changed_pilot_delivery_blocks_further_source_work(construction):
    _, _, output, _ = construction
    configure_bulk_node(construction)
    result = launch(construction, "recurrent", 2, 1102, SLURMD_NODENAME="cpu05")
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "circuit_002/recurrent/report.json").read_text())
    delivery = Path(report["archive"]) / "delivery.json"
    document = json.loads(delivery.read_text())
    document["exit_code"] = 1
    delivery.write_text(json.dumps(document))
    blocked = launch(construction, "recurrent", 3, 1103, SLURMD_NODENAME="cpu05")
    assert blocked.returncode != 0 and "NODE_PILOT_DELIVERY_CHANGED" in blocked.stderr
    assert not (output / "circuit_003").exists()


def test_bulk_route_is_bound_to_the_frozen_circuit_preparation(construction):
    _, preparation, output, _ = construction
    configure_bulk_node(construction)
    preparation.write_text(preparation.read_text() + "\n")
    blocked = launch(construction, "recurrent", 2, 1102, SLURMD_NODENAME="cpu05")
    assert blocked.returncode != 0 and "NODE_ROUTE_NOT_CONFIGURED" in blocked.stderr
    assert not output.exists()


def test_modified_node_wrapper_is_rejected_before_construction(construction):
    repo, _, output, _ = construction
    config = configure_bulk_node(construction)
    selected = yaml.safe_load(config.read_text())["nodes"]["cpu05"]
    launcher = repo / selected["launchers"]["recurrent"]
    launcher.write_text(launcher.read_text() + "# changed\n")
    blocked = launch(construction, "recurrent", 2, 1102, SLURMD_NODENAME="cpu05")
    assert blocked.returncode != 0 and "NODE_LAUNCHER_CHANGED" in blocked.stderr
    assert not output.exists()


def test_unsupported_bulk_node_is_rejected(construction):
    _, _, output, _ = construction
    blocked = launch(construction, "recurrent", 2, 1102, SLURMD_NODENAME="cpu07")
    assert blocked.returncode != 0 and "CPU_PLACEMENT_INVALID" in blocked.stderr
    assert not output.exists()

@pytest.mark.parametrize("changed_node", [False, True])
def test_recovered_stage_receipt_preserves_original_failure_and_other_stage(construction, changed_node):
    repo, _, output, _ = construction
    config = configure_bulk_node(construction)
    for stage, job in [("recurrent", 1102), ("inputs", 2102)]:
        result = launch(construction, stage, 2, job, SLURMD_NODENAME="cpu05")
        assert result.returncode == 0, result.stderr
    original = output / "circuit_002/inputs/report.json"
    passed = json.loads(original.read_text())
    recovered = repo / "recovered_inputs/report.json"
    recovered.parent.mkdir()
    if changed_node:
        passed["node"] = "cpu03"
    recovered.write_text(json.dumps(passed))
    failed = dict(json.loads(original.read_text()), status="FAILED_SOURCE_CONSTRUCTION", exit_code=1)
    original.write_text(json.dumps(failed))
    failure_bytes = original.read_bytes()
    settings = yaml.safe_load(config.read_text())
    settings["qualification_reports"] = {"cpu05": {"inputs": str(recovered.relative_to(repo))}}
    config.write_text(yaml.safe_dump(settings))
    result = launch(construction, "recurrent", 3, 1103, SLURMD_NODENAME="cpu05")
    assert result.returncode == 0, result.stderr
    result = launch(construction, "inputs", 3, 2103, SLURMD_NODENAME="cpu05")
    if changed_node:
        assert result.returncode != 0 and "COMPLETE_NODE_PILOT_REQUIRED" in result.stderr
        assert not (output / "circuit_003/inputs").exists()
    else:
        assert result.returncode == 0, result.stderr
        report = json.loads((output / "circuit_003/inputs/report.json").read_text())
        assert report["node_route"]["qualification_report"] == str(recovered)
    assert original.read_bytes() == failure_bytes

def test_bulk_input_qualification_routes_arguments_and_snapshot_together(construction):
    repo, _, output, _ = construction
    config = configure_bulk_node(construction)
    settings = yaml.safe_load(config.read_text())
    profile = settings["nodes"]["cpu05"]
    settings["input_qualifications"] = {"cpu05": "qualified_inputs"}
    config.write_text(yaml.safe_dump(settings))
    for index in (2, 3):
        result = launch(construction, "recurrent", index, 1100 + index, SLURMD_NODENAME="cpu05")
        assert result.returncode == 0, result.stderr
    expected = str(repo / "qualified_inputs")
    for index in (2, 3):
        result = launch(construction, "inputs", index, 2100 + index, SLURMD_NODENAME="cpu05",
                        NRNSIM_V1_INPUT_CODE_REFERENCE="inherited_old_snapshot")
        assert result.returncode == 0, result.stderr
        report = json.loads((output / f"circuit_{index:03d}/inputs/report.json").read_text())
        archive = Path(report["archive"])
        assert json.loads((archive / "arguments.json").read_text())[-1] == expected
        assert json.loads((archive / "environment.json").read_text())["input_code_reference"] == expected
        assert report["environment_overrides"]["NRNSIM_V1_INPUT_CODE_REFERENCE"] == expected
        assert report["node_route"]["input_qualification"] == expected
        assert report["node_route"]["profile"] == profile


def test_changed_input_qualification_requires_a_matching_full_pilot(construction):
    repo, _, output, _ = construction
    config = configure_bulk_node(construction)
    settings = yaml.safe_load(config.read_text())
    settings["input_qualifications"] = {"cpu05": "qualified_inputs"}
    config.write_text(yaml.safe_dump(settings))
    for index in (2, 3):
        result = launch(construction, "recurrent", index, 1100 + index, SLURMD_NODENAME="cpu05")
        assert result.returncode == 0, result.stderr
    result = launch(construction, "inputs", 2, 2102, SLURMD_NODENAME="cpu05")
    assert result.returncode == 0, result.stderr
    settings["input_qualifications"]["cpu05"] = "different_qualification"
    config.write_text(yaml.safe_dump(settings))
    result = launch(construction, "inputs", 3, 2103, SLURMD_NODENAME="cpu05")
    assert result.returncode != 0
    assert "INPUT_QUALIFICATION_DIFFERS_FROM_PILOT" in result.stderr
    assert not (output / "circuit_003/inputs").exists()
    assert not (repo / FAMILY / "input_construction/run_2103").exists()


def configure_cpu03_inputs(construction, qualification="qualified_inputs"):
    repo, preparation, _, _ = construction
    config = repo / ROUTING_CONFIG
    config.parent.mkdir()
    config.write_text(yaml.safe_dump(dict(
        schema=ROUTING_SCHEMA,
        preparation_sha256=hashlib.sha256(preparation.read_bytes()).hexdigest(),
        input_qualifications={"cpu03": qualification})))
    return config


def test_cpu03_input_qualification_routes_canonical_launcher_and_matching_parent(construction):
    repo, _, output, old_qualification = construction
    configure_cpu03_inputs(construction)
    for index, raw_job in ((3, 1101), (2, 1199)):
        result = launch(construction, "recurrent", index, raw_job)
        assert result.returncode == 0, result.stderr
    recurrent = json.loads((output / "circuit_002/recurrent/report.json").read_text())
    assert recurrent["command"][-2] == str(old_qualification)
    assert "node_route" not in recurrent
    result = launch(construction, "inputs", 2, 2003,
                    NRNSIM_V1_INPUT_CODE_REFERENCE="inherited_old_snapshot")
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "circuit_002/inputs/report.json").read_text())
    archive = Path(report["archive"])
    arguments = json.loads((archive / "arguments.json").read_text())
    environment = json.loads((archive / "environment.json").read_text())
    expected = str(repo / "qualified_inputs")
    assert report["command"][1] == str(repo / "Libs/scripts" / STAGES["inputs"][0])
    assert report["node"] == report["node_route"]["node"] == "cpu03"
    assert report["parent_job_id"] == "1199"
    assert arguments[3] == str(repo / FAMILY / "recurrent_construction/run_1199")
    assert arguments[-1] == expected != str(old_qualification)
    assert environment["input_code_reference"] == expected
    assert report["environment_overrides"]["NRNSIM_V1_INPUT_CODE_REFERENCE"] == expected
    assert report["node_route"]["input_qualification"] == expected
    assert report["status"] == "COMPLETE_SOURCE_CONSTRUCTION"
    assert report["production_episodes"] == 0


@pytest.mark.parametrize("with_route_config", [False, True])
def test_cpu03_input_without_explicit_qualification_retains_legacy_binding(construction,
                                                                         with_route_config):
    repo, _, output, qualification = construction
    if with_route_config:
        configure_bulk_node(construction)
    result = launch(construction, "recurrent", 2, 1102)
    assert result.returncode == 0, result.stderr
    result = launch(construction, "inputs", 2, 2102,
                    NRNSIM_V1_INPUT_CODE_REFERENCE="inherited_old_snapshot")
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "circuit_002/inputs/report.json").read_text())
    archive = Path(report["archive"])
    assert report["command"][1] == str(repo / "Libs/scripts" / STAGES["inputs"][0])
    assert json.loads((archive / "arguments.json").read_text())[-1] == str(qualification)
    assert json.loads((archive / "environment.json").read_text())["input_code_reference"] == "inherited_old_snapshot"
    assert "node_route" not in report
    assert "environment_overrides" not in report


@pytest.mark.parametrize("changed_binding", ["preparation", "schema"])
def test_cpu03_input_route_rejects_stale_binding_before_constructor(construction, changed_binding):
    repo, preparation, output, _ = construction
    config = configure_cpu03_inputs(construction)
    result = launch(construction, "recurrent", 2, 1102)
    assert result.returncode == 0, result.stderr
    if changed_binding == "preparation":
        preparation.write_text(preparation.read_text() + "\n")
    else:
        settings = yaml.safe_load(config.read_text())
        settings["schema"] = "invalid_schema"
        config.write_text(yaml.safe_dump(settings))
    result = launch(construction, "inputs", 2, 2102)
    assert result.returncode != 0
    assert "V1_CONSTRUCTION_NODE_ROUTE_NOT_CONFIGURED" in result.stderr
    assert not (output / "circuit_002/inputs").exists()
    assert not (repo / FAMILY / "input_construction/run_2102").exists()


@pytest.mark.parametrize("qualification", [None, "", "   ", 7, ["qualified_inputs"]])
def test_cpu03_input_route_rejects_invalid_explicit_qualification_before_constructor(construction,
                                                                                   qualification):
    repo, _, output, _ = construction
    configure_cpu03_inputs(construction, qualification)
    result = launch(construction, "recurrent", 2, 1102)
    assert result.returncode == 0, result.stderr
    result = launch(construction, "inputs", 2, 2102)
    assert result.returncode != 0
    assert "V1_CONSTRUCTION_INPUT_QUALIFICATION_INVALID" in result.stderr
    assert not (output / "circuit_002/inputs").exists()
    assert not (repo / FAMILY / "input_construction/run_2102").exists()
