#!/bin/bash
#SBATCH --job-name=NrnSim_live_control_canary
#SBATCH --partition=gpu1
#SBATCH --nodelist=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-live-control-canary-%j.log
set -euo pipefail
repo=/workspace/NeuronSim
srun python3 "$repo/Libs/scripts/run_nest_gpu_live_control.py" \
    --repo "$repo" --work "/tmp/neuronsim/neuronsim-live-control-canary-${SLURM_JOB_ID}" \
    --output "$repo/Output/neural_population_dynamics/model/native_live_control/run_${SLURM_JOB_ID}" "$@"
