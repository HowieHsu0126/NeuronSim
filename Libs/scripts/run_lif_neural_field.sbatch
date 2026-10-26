#!/bin/bash
#SBATCH --job-name=NrnSim_lif_field
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-lif-field-%j.log
set -euo pipefail
if [ -n "${SLURM_JOB_GPUS:-}" ]; then exit 2; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
repo=/workspace/NeuronSim
work=$(mktemp -d "/tmp/neuronsim/neuronsim-lif-field-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf "$work"' EXIT
srun python3 "$repo/Libs/scripts/run_lif_neural_field.py" --repo "$repo" --work "$work" "$@"
