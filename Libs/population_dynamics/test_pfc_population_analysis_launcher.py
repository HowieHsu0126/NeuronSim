"""Exercise the PFC analysis shell launcher through a real Python child."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


LAUNCHER = Path(__file__).parents[1] / "scripts/run_pfc_population_analysis.sbatch"
CONSUMER = """from pathlib import Path
import argparse,json,sys
parser=argparse.ArgumentParser()
for name in ("campaign","preparation","output"):
    parser.add_argument("--"+name,type=Path,required=True)
for seed in (41,42,43):
    parser.add_argument(f"--seed{seed}-cell",type=Path)
args=parser.parse_args()
if not args.campaign.is_dir() or not (args.preparation/"preparation.json").is_file():
    raise ValueError("FIXTURE_PFC_INPUTS_REQUIRED")
for seed in (41,42,43):
    cell=getattr(args,f"seed{seed}_cell") or args.campaign/f"seed_{seed}"
    proof=json.loads((cell/"delivery.json").read_text())
    if proof.get("status")!="PASS" or proof.get("training_seed")!=seed:
        raise ValueError("FIXTURE_COMPLETE_SAME_SEED_CELL_REQUIRED")
args.output.mkdir(parents=True,exist_ok=False)
(args.output/"command.json").write_text(json.dumps(dict(argv=sys.argv[1:],
    cwd=str(Path.cwd()),staged_files=70)))
(args.output/"report.json").write_text(json.dumps(dict(
    status="COMPLETE_DESCRIPTIVE_PFC_POPULATION_FORECAST_ANALYSIS",
    score_rows=1080,origin_cost_rows=360,artifacts=["command.json"])))
"""


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def fixture(tmp_path):
    snapshot = tmp_path / "snapshot"
    sources = {"Libs/__init__.py": "", "Libs/population_dynamics/__init__.py": "",
               "Libs/population_dynamics/pfc_population_forecast_analysis.py": CONSUMER}
    sources.update({f"Libs/configs/frozen_{i:02d}.yaml": "scope: frozen_fixture\n" for i in range(67)})
    rows = []
    for name, content in sources.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        rows.append(dict(path=name, bytes=target.stat().st_size,
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    assert len(rows) == 70
    write_json(snapshot / "delivery.json", dict(status="FROZEN_TESTED_PFC_THREE_SEED_ANALYSIS",
                                                tests_passed=15, files=rows))
    campaign, preparation = tmp_path / "campaign", tmp_path / "preparation"
    write_json(preparation / "preparation.json", dict(status="FROZEN_FIXTURE_PREPARATION"))
    for seed in (41, 42, 43):
        write_json(campaign / f"seed_{seed}/delivery.json", dict(status="PASS", training_seed=seed))
    overrides = {}
    for seed in (41, 42, 43):
        cell = tmp_path / f"seed_{seed}_recovery"
        write_json(cell / "delivery.json", dict(status="PASS", training_seed=seed))
        overrides[seed] = cell
    binary = tmp_path / "bin"
    binary.mkdir()
    fake_srun = binary / "srun"
    fake_srun.write_text("#!/bin/bash\nset -euo pipefail\nprintf '%s\\n' \"$@\" > \"$PFC_FAKE_SRUN_CAPTURE\"\nexec \"$PFC_TEST_PYTHON\" \"${@:2}\"\n")
    fake_srun.chmod(0o755)
    environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
        PFC_TEST_PYTHON=sys.executable, PFC_FAKE_SRUN_CAPTURE=str(tmp_path / "srun_argv.txt"),
        SLURM_JOB_ID="pfc-analysis-fixture", SLURMD_NODENAME="cpu04",
        SLURM_JOB_PARTITION="cpu", SLURM_JOB_GPUS="", SLURM_STEP_GPUS="")
    return dict(root=tmp_path, snapshot=snapshot, campaign=campaign, preparation=preparation,
                overrides=overrides, environment=environment)


def run_launcher(fixture, output, optional=()):
    arguments = [str(fixture["snapshot"]), str(fixture["campaign"]), str(fixture["preparation"]),
                 str(fixture["overrides"][41]), str(output), *map(str, optional)]
    return subprocess.run(["bash", str(LAUNCHER), *arguments], env=fixture["environment"],
                          text=True, capture_output=True, check=False)


@pytest.mark.parametrize("node,partition", [("cpu03", "gpu"), ("cpu04", "cpu")])
@pytest.mark.parametrize("overridden", [(), (42,), (42, 43), (43,)])
def test_shell_srun_and_python_child_preserve_original_and_override_arguments(fixture, overridden, node, partition):
    fixture["environment"].update(SLURMD_NODENAME=node, SLURM_JOB_PARTITION=partition)
    output = fixture["root"] / "output"
    optional = []
    if 42 in overridden or 43 in overridden:
        optional.append(fixture["overrides"][42] if 42 in overridden else "")
    if 43 in overridden:
        optional.append(fixture["overrides"][43])
    result = run_launcher(fixture, output, optional)
    assert result.returncode == 0, result.stdout + result.stderr
    captured = (fixture["root"] / "srun_argv.txt").read_text().splitlines()
    assert captured[:3] == ["/opt/conda/bin/python", "-B", "-"]
    child = json.loads((output / "command.json").read_text())
    assert not Path(child["cwd"]).exists()
    command = child["argv"]
    assert command[:8] == ["--campaign", str(fixture["campaign"]),
        "--preparation", str(fixture["preparation"]), "--seed41-cell", str(fixture["overrides"][41]),
        "--output", str(output)]
    expected = [value for seed in overridden for value in (f"--seed{seed}-cell", str(fixture["overrides"][seed]))]
    assert command[8:] == expected
    proof = json.loads((output / "delivery.json").read_text())
    assert proof["status"] == "PASS" and proof["job_id"] == "pfc-analysis-fixture"
    assert proof["scientific_claim_qualified"] is False
    assert {row["path"] for row in proof["files"]} == {"report.json", "command.json"}
    for row in proof["files"]:
        assert row["sha256"] == hashlib.sha256((output / row["path"]).read_bytes()).hexdigest()


def test_existing_output_is_rejected_before_child_or_output_changes(fixture):
    output = fixture["root"] / "existing"
    output.mkdir()
    original = b"preserved existing analysis\n"
    (output / "sentinel.txt").write_bytes(original)
    result = run_launcher(fixture, output, (fixture["overrides"][42],))
    assert result.returncode != 0
    assert "Existing PFC analysis must be consumed before repeating" in result.stderr
    assert (output / "sentinel.txt").read_bytes() == original
    assert not (output / "command.json").exists()
    assert not (output / "delivery.json").exists()


def test_invalid_same_seed_override_does_not_create_output(fixture):
    write_json(fixture["overrides"][42] / "delivery.json", dict(status="PASS", training_seed=41))
    output = fixture["root"] / "invalid_cell_output"
    result = run_launcher(fixture, output, (fixture["overrides"][42],))
    assert result.returncode != 0
    assert "FIXTURE_COMPLETE_SAME_SEED_CELL_REQUIRED" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("node,partition", [
    ("cpu03", "cpu"), ("cpu04", "gpu"), ("gpu03", "gpu"), ("cpu02", "cpu")])
def test_forbidden_node_or_partition_is_rejected_before_output_creation(fixture, node, partition):
    fixture["environment"].update(SLURMD_NODENAME=node, SLURM_JOB_PARTITION=partition)
    output = fixture["root"] / "invalid_placement"
    result = run_launcher(fixture, output)
    assert result.returncode != 0
    assert "PFC_ANALYSIS_CPU_ONLY_PLACEMENT_REQUIRED" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("node,partition", [("cpu03", "gpu"), ("cpu04", "cpu")])
@pytest.mark.parametrize("allocated_gpu_variable", ["SLURM_JOB_GPUS", "SLURM_STEP_GPUS"])
def test_allocated_job_or_step_gpu_is_rejected_on_each_allowed_node(fixture, node, partition, allocated_gpu_variable):
    fixture["environment"].update(SLURMD_NODENAME=node, SLURM_JOB_PARTITION=partition)
    fixture["environment"][allocated_gpu_variable] = "0"
    output = fixture["root"] / "allocated_gpu"
    result = run_launcher(fixture, output)
    assert result.returncode != 0
    assert "PFC_ANALYSIS_CPU_ONLY_PLACEMENT_REQUIRED" in result.stderr
    assert not output.exists()
