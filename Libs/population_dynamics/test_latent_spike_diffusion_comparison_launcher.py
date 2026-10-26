"""Exercise the LDNS Bash launcher without executing scientific modules."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/run_latent_spike_diffusion_comparison.sbatch"
REPO = Path("/workspace/NeuronSim")
PREFLIGHT = "Libs.population_dynamics.resource_probe"
COMPARISON = "Libs.population_dynamics.latent_spike_diffusion_comparison"


@pytest.fixture
def launch(tmp_path):
    preparation = tmp_path / "frozen software with spaces"
    (preparation / "Libs").mkdir(parents=True)
    (preparation / "Libs/launcher_fixture.txt").write_text("duplicate software fixture\n")
    stub_bin = tmp_path / "stub bin"
    stub_bin.mkdir()
    stub = stub_bin / "srun"
    stub.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""\
        import json
        import os
        from pathlib import Path
        import sys

        arguments = sys.argv[1:]
        assert arguments[:2] == ["python3", "-m"], arguments
        module = arguments[2]
        assert module in {
            "Libs.population_dynamics.resource_probe",
            "Libs.population_dynamics.latent_spike_diffusion_comparison",
        }, module
        work = Path(os.environ["PYTHONPATH"])
        record = dict(
            arguments=arguments,
            work=str(work),
            copied_software=(work / "Libs/launcher_fixture.txt").read_text(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            mkl_num_threads=os.environ.get("MKL_NUM_THREADS"),
            openblas_num_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
            dont_write_bytecode=os.environ.get("PYTHONDONTWRITEBYTECODE"),
        )
        with Path(os.environ["STUB_SRUN_RECORD"]).open("a") as stream:
            stream.write(json.dumps(record) + "\\n")
        if module == "Libs.population_dynamics.resource_probe":
            code = int(os.environ["STUB_PREFLIGHT_EXIT"])
            report = dict(status="BLOCKED" if code else "PASS", reasons=["FIXTURE_PREFLIGHT_FAILURE"] if code else [])
            output = Path(arguments[arguments.index("--output") + 1])
            output.write_text(json.dumps(report) + "\\n")
            print(json.dumps(report))
            raise SystemExit(code)
        """))
    stub.chmod(0o755)

    def run(*flags, preflight_exit=0):
        record_path = tmp_path / "srun_calls.jsonl"
        environment = dict(
            os.environ,
            PATH=str(stub_bin) + os.pathsep + os.environ["PATH"],
            SLURM_JOB_ID="ldns_launcher_fixture",
            STUB_SRUN_RECORD=str(record_path),
            STUB_PREFLIGHT_EXIT=str(preflight_exit),
        )
        result = subprocess.run(
            ["bash", str(LAUNCHER), str(preparation), *map(str, flags)],
            env=environment, capture_output=True, text=True,
        )
        calls = [json.loads(line) for line in record_path.read_text().splitlines()] if record_path.exists() else []
        for call in calls:
            assert call["copied_software"] == "duplicate software fixture\n"
            assert call["cuda_visible_devices"] == ""
            assert call["omp_num_threads"] == call["mkl_num_threads"] == call["openblas_num_threads"] == "1"
            assert call["dont_write_bytecode"] == "1"
            assert not Path(call["work"]).exists(), "Task-owned scratch was not removed."
        assert (preparation / "Libs/launcher_fixture.txt").read_text() == "duplicate software fixture\n"
        return result, calls

    return run


def _comparison_options(call):
    arguments = call["arguments"][3:]
    assert len(arguments) == 6
    assert set(arguments[::2]) == {"--ldns-root", "--population-root", "--output"}
    return dict(zip(arguments[::2], arguments[1::2]))


def test_single_software_argument_preserves_default_roots(launch):
    result, calls = launch()
    assert result.returncode == 0, result.stderr
    assert [call["arguments"][2] for call in calls] == [PREFLIGHT, COMPARISON]
    assert calls[0]["arguments"][3:] == [
        "--output", str(Path(calls[0]["work"]) / "preflight.json"),
        "--required-memory-gib", "4",
    ]
    assert _comparison_options(calls[1]) == {
        "--ldns-root": str(REPO / "Output/neural_population_dynamics/baselines/latent_spike_diffusion/complete_cohort_development/campaign_20261012"),
        "--population-root": str(REPO / "Output/neural_population_dynamics/model/trajectory_development/run_606139"),
        "--output": str(REPO / "Output/neural_population_dynamics/baselines/latent_spike_diffusion/joint_comparison_20261012"),
    }


def test_explicit_continued_budget_roots_preserve_literal_boundaries(launch, tmp_path):
    ldns = tmp_path / "LDNS complete cohort with spaces"
    population = tmp_path / "population continued80 with spaces"
    output = tmp_path / "fresh continued80 joint comparison with spaces"
    ldns.mkdir()
    population.mkdir()
    flags = ["--ldns-root", str(ldns), "--population-root", str(population), "--output", str(output)]
    result, calls = launch(*flags)
    assert result.returncode == 0, result.stderr
    assert [call["arguments"][2] for call in calls] == [PREFLIGHT, COMPARISON]
    assert calls[1]["arguments"][3:] == flags
    assert _comparison_options(calls[1]) == {
        "--ldns-root": str(ldns), "--population-root": str(population), "--output": str(output),
    }
    assert ldns.is_dir() and population.is_dir()
    assert not output.exists(), "The fixture must not execute the scientific comparison."


def test_preflight_failure_stops_comparison_and_keeps_failure_log(launch):
    result, calls = launch(preflight_exit=2)
    assert result.returncode == 2, result.stderr
    assert [call["arguments"][2] for call in calls] == [PREFLIGHT]
    assert json.loads(result.stdout) == {"status": "BLOCKED", "reasons": ["FIXTURE_PREFLIGHT_FAILURE"]}


def test_cpu04_resource_contract():
    directives = {line for line in LAUNCHER.read_text().splitlines() if line.startswith("#SBATCH")}
    assert {"#SBATCH --partition=cpu", "#SBATCH --nodelist=cpu04", "#SBATCH --cpus-per-task=1", "#SBATCH --mem=4G"} <= directives
    assert "#SBATCH --no-requeue" in directives
    assert "#SBATCH --output=/tmp/neuronsim/neuronsim-ldns-comparison-%j.log" in directives
    assert all("--gres" not in line and "--gpus" not in line and "--exclusive" not in line for line in directives)
