"""Source bindings and exact CPU deployment inputs for the saved-field comparison."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_memory_field_comparison import consumed_names, cpu_placement, seed_inputs
from run_population_trajectory_campaign import copy_bound


def test_consumed_names_preserve_all_origins_without_unread_products():
    memory = consumed_names("memory")
    population = consumed_names("population")
    assert len(memory) == len(set(memory)) == 51
    assert len(population) == len(set(population)) == 50
    assert "evaluation/train_block_014_familiar_0_origin_0200.npz" in memory
    assert "evaluation/train_block_017_familiar_3_origin_1000.npz" in memory
    assert "forecasts/train_block_014_familiar_0_origin_1000.npz" in population
    assert "forecasts/train_block_017_familiar_3_origin_5000.npz" in population
    assert all("checkpoint" not in name and "scores.csv" not in name for name in memory + population)
    with pytest.raises(ValueError, match="unknown comparison source"):
        consumed_names("other")


def test_cpu_placement_requires_slurm_and_an_allowed_node_without_gpu(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    for node, partition in (("cpu04", "cpu"), ("cpu03", "gpu")):
        monkeypatch.setenv("SLURMD_NODENAME", node)
        monkeypatch.setenv("SLURM_JOB_PARTITION", partition)
        assert cpu_placement()["node"] == node
    monkeypatch.setenv("SLURMD_NODENAME", "gpu03")
    with pytest.raises(ValueError, match="requires CPU04"):
        cpu_placement()
    monkeypatch.setenv("SLURMD_NODENAME", "cpu03")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    with pytest.raises(ValueError, match="without GPU"):
        cpu_placement()
    monkeypatch.delenv("SLURM_JOB_GPUS")
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(ValueError, match="requires CPU04"):
        cpu_placement()


def write_seed(root, **changes):
    root.mkdir()
    proof = dict(status="PASS", exit_code=0, training_seed=43, job_id="actual-delivered-job",
                 scientific_claim_qualified=False, complete_training=True, replay_performed=True,
                 preparation_sha256="frozen-preparation", files=[dict(path=name, bytes=1, sha256="a" * 64)
                    for name in consumed_names("memory")])
    proof.update(changes)
    (root / "delivery.json").write_text(json.dumps(proof))


def test_memno_accepts_actual_delivered_job_bound_to_frozen_preparation(tmp_path):
    root = tmp_path / "seed_43"
    write_seed(root)
    rows, binding = seed_inputs(root, 43, "memory", "frozen-preparation")
    assert len(rows) == 52
    assert rows[0]["path"] == "memory/seed_43/delivery.json"
    assert binding["job_id"] == "actual-delivered-job"
    assert binding["training_seed"] == 43
    assert len(binding["delivery_sha256"]) == 64


@pytest.mark.parametrize("changes", [dict(training_seed=42), dict(status="FAILED"),
    dict(complete_training=False), dict(replay_performed=False), dict(preparation_sha256="changed"),
    dict(scientific_claim_qualified=True), dict(files=[])])
def test_memno_rejects_partial_foreign_or_unbound_delivery(tmp_path, changes):
    root = tmp_path / "seed_43"
    write_seed(root, **changes)
    with pytest.raises(ValueError):
        seed_inputs(root, 43, "memory", "frozen-preparation")


def test_input_copy_rejects_changed_declared_source(tmp_path):
    source = tmp_path / "input.npz"
    source.write_bytes(b"original")
    row = dict(bytes=source.stat().st_size, sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    source.write_bytes(b"modified")
    with pytest.raises(ValueError, match="source manifest identity changed"):
        copy_bound(source, tmp_path / "staged.npz", row)
    assert not (tmp_path / "staged.npz").exists()


@pytest.mark.parametrize("node,partition", [("cpu04", "cpu"), ("cpu03", "gpu")])
def test_source_frozen_launcher_end_to_end_and_changed_input_failure(tmp_path, node, partition):
    """Exercise the real child consumer and archive on disposable CPU scratch."""
    import csv
    import itertools
    import os
    import shutil
    import subprocess

    import numpy as np

    from run_memory_field_comparison import ENTRY, BATCH, HELPER, RESULTS
    from run_population_trajectory_campaign import digest, software_closure, write_json

    repo = Path(__file__).parents[2]
    fixture_repo = tmp_path / "source_repo"
    names = software_closure(repo, ["memory_field_comparison", "resource_probe"]) + [ENTRY, BATCH, HELPER]
    for name in dict.fromkeys(names):
        target = fixture_repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)

    memory = tmp_path / "memory"
    population = tmp_path / "population"
    braid = tmp_path / "braid"
    memory_preparation = tmp_path / "memory_preparation"
    write_json(memory_preparation / "preparation.json", dict(status="PREPARED_SOURCE_BOUND_MEMNO",
                                                           training_seeds=[41, 42, 43]))
    memory_preparation_sha256 = digest(memory_preparation / "preparation.json")
    unit_ids = list(range(100000))
    field_ids = ["fixture_field_0", "fixture_field_1"]
    field_index = [0] * 50000 + [1] * 50000
    cohort = dict(acquisition_design_sha256="fixture_design", acquisition_ledger_sha256="fixture_ledger",
                  dataset_sha256="fixture_dataset", episode_table_sha256="fixture_episodes",
                  episode_bindings={"fixture": "binding"}, native_source_identity=dict(dt_ms=.1))
    source = dict(field_units="spikes_per_second_per_neuron", bin_seconds=.005, cohort=cohort,
                  source_unit_ids=unit_ids, source_field_ids=field_ids)
    binding = dict(dataset=source, config=dict(fit_blocks=list(range(14)), validation_blocks=list(range(14, 18))))
    truth = np.tile(np.array([2., 4.], dtype=np.float32), (100, 1))
    prediction = truth + np.array([1., 2.], dtype=np.float32)
    population_deliveries = {}

    def deliver(root, seed, kind):
        rows = [dict(path=name, bytes=(root / name).stat().st_size, sha256=digest(root / name))
                for name in consumed_names(kind)]
        proof = dict(status="PASS", exit_code=0, training_seed=seed, job_id=f"fixture_actual_{seed}",
                     scientific_claim_qualified=False, files=rows)
        if kind == "memory":
            proof.update(complete_training=True, replay_performed=True,
                         preparation_sha256=memory_preparation_sha256)
        write_json(root / "delivery.json", proof)
        return digest(root / "delivery.json")

    for seed in (41, 42, 43):
        mroot = memory / f"seed_{seed}"
        proot = population / f"seed_{seed}"
        write_json(mroot / "training/run.json", dict(status="FITTED_DEVELOPMENT", completed_epochs=300,
                                                     binding=binding, field_mean_only=True))
        write_json(proot / "training/run.json", dict(binding=cohort,
                   model_configuration=dict(unit_ids=unit_ids, field_names=field_ids,
                                            unit_field_index=field_index, bin_seconds=.005)))
        (proot / "forecasts").mkdir()
        np.savez(proot / "forecasts/fitting_field_scale.npz", standard_deviation_hz=np.array([1., 2.]),
                 unit_count=np.array([50000, 50000]), field_ids=np.asarray(field_ids))
        artifacts = {}
        for block, template, origin in itertools.product(range(14, 18), range(4), (1000., 3000., 5000.)):
            case = f"train_block_{block:03d}_familiar_{template}"
            mname = f"evaluation/{case}_origin_{int(origin / 5):04d}.npz"
            (mroot / "evaluation").mkdir(exist_ok=True)
            np.savez(mroot / mname, predicted_rate_hz=prediction, truth_rate_hz=truth,
                     source_field_ids=np.asarray(field_ids), source_unit_ids=np.asarray(unit_ids),
                     origin_bin=int(origin / 5), bin_seconds=.005)
            artifacts[Path(mname).name] = digest(mroot / mname)
            np.savez(proot / f"forecasts/{case}_origin_{origin:g}.npz",
                     observed_field_counts=truth.astype(np.float64) * 250.)
        write_json(mroot / "evaluation/evaluation.json", dict(status="EVALUATED_DEVELOPMENT", score_rows=144,
                   origins=48, partial_training_diagnostic=False, binding=binding, artifacts=artifacts))
        write_json(mroot / "replay/evaluation.json", dict(status="REPLAYED_DEVELOPMENT", score_rows=144,
                                                         binding=binding))
        deliver(mroot, seed, "memory")
        population_deliveries[str(seed)] = deliver(proot, seed, "population")
        (mroot / "training/unread_checkpoint.pt").write_bytes(b"not consumed")

    write_json(braid / "comparison/comparison.json", dict(consumed_deliveries={seed: dict(
        population=dict(delivery_sha256=value)) for seed, value in population_deliveries.items()}))
    prior_rows = [dict(seed=seed, block=block, case_id=f"train_block_{block:03d}_familiar_{template}",
                      origin_ms=origin, horizon_ms=horizon, braid_field_mean_mse=2.,
                      population_field_mean_mse_corrected=3.)
                  for seed, block, template, origin, horizon in itertools.product(
                      (41, 42, 43), range(14, 18), range(4), (1000., 3000., 5000.), (20., 80., 500.))]
    with (braid / "comparison/mean_comparison_all_seeds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prior_rows[0]))
        writer.writeheader()
        writer.writerows(prior_rows)
    prior_files = [dict(path=name, bytes=(braid / name).stat().st_size, sha256=digest(braid / name))
                   for name in ("comparison/comparison.json", "comparison/mean_comparison_all_seeds.csv")]
    write_json(braid / "delivery.json", dict(status="PASS", exit_code=0, files=prior_files))
    preparation = tmp_path / "comparison_preparation"
    command = [sys.executable, "-B", str(fixture_repo / ENTRY), "--prepare", "--repo", str(fixture_repo),
               "--preparation", str(preparation), "--memory-campaign", str(memory),
               "--braid-comparison", str(braid), "--population-campaign", str(population),
               "--memory-preparation", str(memory_preparation)]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        SLURM_JOB_ID="memory-fixture-" + node, SLURMD_NODENAME=node, SLURM_JOB_PARTITION=partition,
        SLURM_JOB_GPUS="", SLURM_STEP_GPUS="")
    child = subprocess.run(command, env=environment, text=True, capture_output=True)
    assert child.returncode == 0, child.stdout + child.stderr
    # Frozen software must supply the child even if its original source changes.
    (fixture_repo / "Libs/population_dynamics/memory_field_comparison.py").write_text("raise RuntimeError('mutable source executed')\n")
    # Execute the frozen sbatch shell and the frozen Python caller. The fake
    # srun removes only the scheduler transport; both real consumers still run.
    binary = tmp_path / "bin"
    binary.mkdir()
    fake_srun = binary / "srun"
    fake_srun.write_text("#!/bin/bash\nset -euo pipefail\nexec \"$MEMORY_TEST_PYTHON\" \"${@:2}\"\n")
    fake_srun.chmod(0o755)
    work_root = tmp_path / "node_scratch"
    work_root.mkdir()
    environment.update(PATH=str(binary) + os.pathsep + os.environ["PATH"], MEMORY_TEST_PYTHON=sys.executable,
                       MEMORY_FIELD_COMPARISON_WORK_ROOT=str(work_root))
    work = work_root / ("neuronsim-memory-comparison-" + environment["SLURM_JOB_ID"])
    destination_root = tmp_path / "comparison_delivery"
    destination = destination_root / ("run_" + environment["SLURM_JOB_ID"])
    command = ["bash", str(preparation / BATCH), str(preparation), str(destination_root)]
    child = subprocess.run(command, env=environment, text=True, capture_output=True)
    assert child.returncode == 0, child.stdout + child.stderr + (destination / "delivery.json").read_text()
    result = json.loads((destination / "comparison/comparison.json").read_text())
    assert result["score_rows"] == 432 and result["block_rows"] == 36
    assert result["scientific_claim_qualified"] is result["inference_performed"] is False
    receipt = json.loads((destination / "delivery.json").read_text())
    assert receipt["status"] == "PASS" and receipt["exit_code"] == 0
    assert receipt["retraining_performed"] is False
    assert receipt["placement"]["node"] == node and receipt["placement"]["partition"] == partition
    assert set(RESULTS).issubset({row["path"] for row in receipt["files"]})
    for row in receipt["files"]:
        assert digest(destination / row["path"]) == row["sha256"]
    for seed in (41, 42, 43):
        assert receipt["consumed_deliveries"][str(seed)]["memory"]["job_id"] == f"fixture_actual_{seed}"
        for kind in ("memory", "population"):
            staged = json.loads((destination / "input_delivery.json").read_text())["files"]
            prefix = f"{kind}/seed_{seed}/"
            actual = {row["path"].removeprefix(prefix) for row in staged if row["path"].startswith(prefix)}
            assert actual == set(consumed_names(kind)) | {"delivery.json"}
    executed = json.loads((destination / "command.json").read_text())["command"]
    assert "population_dynamics.memory_field_comparison" in executed
    assert not work.exists()  # The frozen shell cleans only its archived scratch.

    damaged = memory / "seed_41" / consumed_names("memory")[3]
    damaged.write_bytes(b"changed declared source")
    failed_work = tmp_path / "failed_work"
    failed_delivery = tmp_path / "failed_delivery"
    command = [sys.executable, "-B", str(preparation / ENTRY), "--run", "--preparation", str(preparation),
               "--work", str(failed_work), "--destination", str(failed_delivery)]
    child = subprocess.run(command, env=environment, text=True, capture_output=True)
    assert child.returncode != 0
    failure = json.loads((failed_delivery / "delivery.json").read_text())
    assert failure["status"] == "FAILED" and failure["exit_code"] != 0
    assert "source manifest identity changed" in failure["error"]
    assert not (failed_work / "command.json").exists()
    assert not (failed_delivery / "comparison/comparison.json").exists()


@pytest.mark.parametrize("node,partition,job_gpu,step_gpu", [
    ("mgmt2", "cpu", "", ""), ("cpu03", "cpu", "", ""), ("cpu04", "gpu", "", ""),
    ("cpu03", "gpu", "0", ""), ("cpu04", "cpu", "", "0")])
def test_frozen_caller_rejects_unsupported_nodes_partitions_and_gpu_before_outputs(tmp_path, node, partition, job_gpu, step_gpu):
    import os
    import shutil
    import subprocess

    from run_memory_field_comparison import ENTRY, BATCH, HELPER

    repo = Path(__file__).parents[2]
    preparation = tmp_path / "preparation"
    for name in (ENTRY, BATCH, HELPER):
        target = preparation / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
    binary = tmp_path / "bin"
    binary.mkdir()
    fake_srun = binary / "srun"
    fake_srun.write_text("#!/bin/bash\nset -euo pipefail\nexec \"$MEMORY_TEST_PYTHON\" \"${@:2}\"\n")
    fake_srun.chmod(0o755)
    environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
        MEMORY_TEST_PYTHON=sys.executable, SLURM_JOB_ID="rejected", SLURMD_NODENAME=node,
        SLURM_JOB_PARTITION=partition, SLURM_JOB_GPUS=job_gpu, SLURM_STEP_GPUS=step_gpu,
        MEMORY_FIELD_COMPARISON_WORK_ROOT=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    destination = tmp_path / "delivery"
    child = subprocess.run(["bash", str(preparation / BATCH), str(preparation), str(destination)],
                           env=environment, capture_output=True, text=True, check=False)
    assert child.returncode != 0
    assert "without GPU" in child.stderr
    assert not destination.exists()
    assert not (tmp_path / "neuronsim-memory-comparison-rejected").exists()
