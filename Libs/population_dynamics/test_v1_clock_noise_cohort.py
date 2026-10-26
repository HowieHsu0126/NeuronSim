"""Exercise cohort orchestration with disposable comparison fixtures."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _write_comparison(output, label, circuit, first, second):
    output.mkdir(parents=True)
    row_count = 310 if label == "noise" else 240
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "clock_to_noise_ratio", "undefined_units"])
        for index in range(row_count):
            writer.writerow([f"metric_{index}", "NaN" if index == 0 else "0.0012300", "34434"])
    data = (output / "comparison.csv").read_bytes()
    report = dict(exit_code=0, circuit_id=circuit, block_indices=[first, second],
                  units=230924, noise_block_pairs=1, comparison_rows=row_count,
                  artifacts={"comparison.csv": dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())})
    if label == "binning":
        report.update(bin_ms=[1, 5, 10], lag_ms=[0, 10, 100],
                      window=dict(start_ms=501, stop_ms_exclusive=2991))
    (output / "report.json").write_text(json.dumps(report))


@pytest.fixture
def cohort(tmp_path):
    extension = tmp_path / "extension"
    combined = tmp_path / "combined"
    validation = tmp_path / "validation"
    software = validation / "software"
    destination = tmp_path / "result"
    extension.mkdir()
    combined.mkdir()
    software.mkdir(parents=True)
    (combined / "delivery.json").write_text(json.dumps(dict(exit_code=0, verified_pairs=60)))
    (validation / "delivery.json").write_text(json.dumps(dict(exit_code=0, legacy_csv_exact=True)))
    (extension / "submission.json").write_text(json.dumps(
        dict(combined_panel_job=dict(destination=str(combined), expected_pairs=60))))
    (extension / "preparation.json").write_text(json.dumps(dict(
        circuit_block_pairs=[[0, 0, 8], [1, 1, 9], [2, 1, 9]],
        prior_pair_reused=[1, 1, 9])))
    (software / "__init__.py").write_text("")
    imports = "from pathlib import Path\nimport csv,hashlib,json\n"
    (software / "_fixture.py").write_text(imports + inspect.getsource(_write_comparison))
    for label in ("noise", "binning"):
        _write_comparison(validation / ("legacy_" + label), label, "v1_visual_circuit_001", 1, 9)
        module = "v1_clock_" + label + "_comparison"
        (software / (module + ".py")).write_text(
            "import argparse\nfrom pathlib import Path\nfrom ._fixture import _write_comparison\n"
            "parser=argparse.ArgumentParser()\n"
            "for name in ('panel','circuit-id','first-block','second-block','output'):\n"
            "    parser.add_argument('--'+name,required=True)\n"
            "args=parser.parse_args()\n"
            f"_write_comparison(Path(args.output),{label!r},args.circuit_id,"
            "int(args.first_block),int(args.second_block))\n")
    return extension, software, destination, combined


def _run(cohort):
    extension, software, destination, _ = cohort
    launcher = Path(__file__).resolve().parents[1] / "scripts/run_v1_clock_noise_cohort.sbatch"
    env = dict(os.environ, NRNSIM_BENCHMARK_PYTHON=sys.executable,
               SLURMD_NODENAME="cpu03", SLURM_JOB_ID="123", SLURM_JOB_GPUS="",
               TMPDIR=str(extension.parent))
    result = subprocess.run(["bash", str(launcher), str(extension), str(software), str(destination)],
                            capture_output=True, text=True, env=env)
    return result, json.loads((destination / "report.json").read_text())


def test_cohort_preserves_rows_strings_and_persistent_sources(cohort):
    result, report = _run(cohort)
    assert result.returncode == 0, result.stderr
    assert report["reused_pairs"] == 1
    assert len(report["commands"]) == 4
    assert report["scientific_qualification"] is False
    assert report["new_source_simulations"] == 0
    assert [pair["low_contrast_direction_deg"] for pair in report["pairs"]] == [0, 45, 45]
    destination = cohort[2]
    for label, count in (("noise", 930), ("binning", 720)):
        with (destination / (label + "_by_circuit.csv")).open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count
        assert {row["circuit_id"] for row in rows} == {f"v1_visual_circuit_{i:03d}" for i in range(3)}
        assert rows[0]["clock_to_noise_ratio"] == "NaN"
        assert rows[1]["clock_to_noise_ratio"] == "0.0012300"
        assert all(row["undefined_units"] == "34434" for row in rows)
        for pair in report["pairs"]:
            source = pair["sources"][label]
            path = Path(source["report"])
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["report_sha256"]
            assert path.is_relative_to(destination) != source["reused"]


def test_cohort_rejects_changed_comparison_and_preserves_failure(cohort):
    path = cohort[1].parent / "legacy_noise/comparison.csv"
    path.write_text(path.read_text().replace("0.0012300", "0.1", 1))
    result, report = _run(cohort)
    assert result.returncode != 0
    assert report["exit_code"] == 1
    assert report["error"] == "CLOCK_NOISE_COHORT_COMPARISON_CHANGED"
    assert (cohort[2] / "error.log").is_file()
    assert not (cohort[2] / "noise_by_circuit.csv").exists()


def test_cohort_requires_complete_combined_delivery(cohort):
    (cohort[3] / "delivery.json").write_text(json.dumps(dict(exit_code=0, verified_pairs=40)))
    result, report = _run(cohort)
    assert result.returncode != 0
    assert report["error"] == "CLOCK_NOISE_COHORT_COMPLETE_COMBINED_PANEL_REQUIRED"
    assert report["commands"] == []
    assert (cohort[2] / "error.log").is_file()
