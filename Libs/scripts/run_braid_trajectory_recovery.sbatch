#!/bin/bash
#SBATCH --job-name=NrnSim_braid_fit_resume
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-braid-recovery-%j.log
set -euo pipefail
preparation="$1"
parent="$2"
seed="$3"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
srun python3 "$preparation/Libs/scripts/run_braid_trajectory_campaign.py" \
    --seed "$seed" --recovery-from "$parent"
