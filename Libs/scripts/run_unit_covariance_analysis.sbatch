#!/bin/bash
#SBATCH --job-name=NrnSim_unit_covariance
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-unit-covariance-%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
repo=/workspace/NeuronSim
work=$(mktemp -d "/tmp/neuronsim/neuronsim-unit-covariance-${SLURM_JOB_ID}.XXXXXX")
cleanup() {
    if [ -f "$work/archive_complete" ]; then rm -rf "$work";
    else printf 'Covariance scratch retained: %s\n' "$work" >&2; fi
}
trap cleanup EXIT
srun python3 "$repo/Libs/scripts/run_unit_covariance_analysis.py" \
    --repo "$repo" --work "$work" \
    --output "$repo/Output/neural_population_dynamics/model/unit_covariance/run_${SLURM_JOB_ID}" "$@"
