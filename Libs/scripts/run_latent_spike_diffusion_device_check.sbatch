#!/bin/bash
#SBATCH --job-name=NrnSim_ldns_cuda_check
#SBATCH --partition=gpu1
#SBATCH --nodelist=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-ldns-cuda-%j.log
set -euo pipefail
preparation="$1"
work="/tmp/neuronsim/neuronsim-ldns-cuda-${SLURM_JOB_ID}"
mkdir -p "$work/prerequisites"
cp -r "$preparation/Libs" "$work/"
export PYTHONPATH="$work:$work/Libs/external/LDNS"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
srun python3 -m Libs.population_dynamics.resource_probe --output "$work/preflight.json" --required-memory-gib 32
srun python3 - "$work" <<'PY'
import importlib.util
from pathlib import Path
import sys
work = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("inputs", work / "Libs/scripts/run_population_trajectory_campaign.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.stage_inputs(Path("/workspace/NeuronSim"), work)
PY
common=(--acquisition "$work/inputs/train" --fields "$work/inputs/train_fields"
        --config "$work/Libs/configs/neural_population_latent_spike_diffusion.yaml" --output "$work/results")
srun python3 -m Libs.population_dynamics.latent_spike_diffusion_device_check "${common[@]}"
srun python3 -m Libs.population_dynamics.latent_spike_diffusion_device_check "${common[@]}" --replay
