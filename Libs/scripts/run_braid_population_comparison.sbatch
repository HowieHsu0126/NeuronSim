#!/bin/bash
#SBATCH --job-name=NrnSim_braid_mean_comparison
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-braid-mean-comparison-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
repo="$1"
shift
srun python3 "$repo/Libs/scripts/run_braid_population_comparison.py" --repo "$repo" "$@"
