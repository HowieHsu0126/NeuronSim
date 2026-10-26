#!/bin/bash
#SBATCH --job-name=NrnSim_population_analysis
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-population-analysis-%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
repo=/workspace/NeuronSim
work=$(mktemp -d "/tmp/neuronsim/neuronsim-population-analysis-${SLURM_JOB_ID}.XXXXXX")
cleanup() {
    if [ -f "$work/archive_complete" ]; then rm -rf "$work";
    else printf 'Analysis scratch retained: %s\n' "$work" >&2; fi
}
trap cleanup EXIT
srun python3 "$repo/Libs/scripts/run_population_forecast_analysis.py" \
    --repo "$repo" --work "$work" "$@"
