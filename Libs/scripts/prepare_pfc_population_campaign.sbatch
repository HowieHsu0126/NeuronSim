#!/bin/bash
#SBATCH --job-name=NrnSim_PfcTrainingPrepare
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-pfc-training-prepare-%j.log
set -euo pipefail
snapshot="$1"
inputs="$2"
coverage="$3"
prepared="$4"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
srun /opt/conda/bin/python -B - "$snapshot" "$inputs" "$coverage" "$prepared" <<'PY'
from pathlib import Path
import importlib.util
import json
import os
import sys

snapshot, inputs, coverage, prepared = map(Path, sys.argv[1:])
if (os.environ.get("SLURMD_NODENAME") != "cpu04"
        or os.environ.get("SLURM_JOB_PARTITION") != "cpu" or os.environ.get("SLURM_JOB_GPUS")):
    raise ValueError("PFC_PREPARATION_CPU04_CPU_ALLOCATION_REQUIRED")
spec = importlib.util.spec_from_file_location("population_campaign", snapshot / "Libs/scripts/run_population_trajectory_campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
frozen = json.loads((snapshot / "software_snapshot.json").read_text())
for row in frozen["files"]:
    path = snapshot / row["path"]
    if path.stat().st_size != row["bytes"] or campaign.digest(path) != row["sha256"]:
        raise ValueError("PFC_FROZEN_PREPARATION_SOFTWARE_CHANGED:" + row["path"])
sys.path.insert(0, str(snapshot / "Libs"))
from population_dynamics.pfc_training_inputs import assemble

previous = json.loads(coverage.read_text())
cohort, refreshed = assemble(Path(previous["baseline_inputs"]), Path(previous["baseline_validation"]),
    Path(previous["plan"]).parent, Path(previous["plan"]), previous["reader_jobs"])
if (refreshed["fit_validation_coverage_ready"] is not True
        or (refreshed["fitting_episodes"], refreshed["validation_episodes"]) != (80, 30)):
    raise ValueError("PFC_COMPLETE_80_FIT_30_VALIDATION_BEFORE_PREPARATION")
payload = json.dumps(cohort, indent=2) + "\n"
if not inputs.exists() or inputs.read_text() != payload:
    campaign.write_json(inputs, cohort)
refreshed.update(inputs=str(inputs), inputs_sha256=campaign.digest(inputs))
campaign.write_json(coverage, refreshed)
campaign.prepare_pfc(snapshot, inputs, coverage, prepared)
campaign.write_json(prepared / "preparation_execution.json", dict(
    status="PREPARED_COMPLETE_AUTHENTICATED_PFC_TRAINING", job_id=os.environ["SLURM_JOB_ID"],
    source_snapshot=str(snapshot), software_snapshot_sha256=campaign.digest(snapshot / "software_snapshot.json"),
    preparation_sha256=campaign.digest(prepared / "preparation.json"),
    fitting_episodes=80, validation_episodes=30, training_seeds=[41, 42, 43],
    training_started=False, scientific_claim_qualified=False))
print(json.dumps(dict(status="PREPARED_COMPLETE_AUTHENTICATED_PFC_TRAINING", output=str(prepared))))
PY
