#!/bin/bash
#SBATCH --job-name=NrnSim_braid_population_fit
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2%1
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-braid-population-%A-%a.log
set -euo pipefail
if [ "$SLURMD_NODENAME" != cpu04 ] || [ "$SLURM_JOB_PARTITION" != cpu ] || [ -n "${SLURM_JOB_GPUS:-}" ]; then exit 2; fi
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
seed=$((41 + SLURM_ARRAY_TASK_ID))
srun python3 /tmp/neuronsim/neuronsim-braid-matched-development-20261011/deployment.py --seed "$seed"
