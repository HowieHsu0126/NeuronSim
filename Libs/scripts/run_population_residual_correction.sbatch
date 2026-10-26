#!/bin/bash
#SBATCH --job-name=NrnSim_ResidualCorrection
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --chdir=/tmp
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --array=41-43%1

# Usage: sbatch [scheduler overrides] this-file REPO PREPARATION FIT_JOB OUTPUT_ROOT
# Set --output to a campaign log path when submitting. No parent training runs.
# CUDA alternative: override placement/GRES and export NEURONSIM_DEVICE=cuda.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "usage: $0 REPO PREPARATION FIT_JOB OUTPUT_ROOT" >&2
    exit 2
fi
residual_repo=$(realpath -e -- "$1")
residual_preparation=$(realpath -e -- "$2")
residual_fit_job=$3
mkdir -p -- "$4" /tmp/neuronsim
residual_output_root=$(realpath -e -- "$4")
residual_seed=${SLURM_ARRAY_TASK_ID:?Slurm seed array is required}
residual_device=${NEURONSIM_DEVICE:-cpu}
residual_python=${NEURONSIM_PYTHON:-python3}
residual_work=$(mktemp -d "/tmp/neuronsim/neuronsim-residual-eval-${SLURM_JOB_ID}-${residual_seed}-XXXXXX")

residual_cleanup() {
    residual_exit=$?
    trap - EXIT
    if [[ -f "$residual_work/archive_complete" ]]; then
        rm -rf -- "$residual_work"
    else
        echo "Archive incomplete; task scratch retained for recovery: $residual_work" >&2
    fi
    exit "$residual_exit"
}
trap residual_cleanup EXIT
mkdir -p -- "$residual_work/scratch"
export TMPDIR="$residual_work/scratch"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$residual_device" == cpu ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi
"$residual_python" "$residual_repo/Libs/scripts/run_population_residual_correction.py" \
    --repo "$residual_repo" --preparation "$residual_preparation" --fit-job "$residual_fit_job" \
    --seed "$residual_seed" --work "$residual_work" \
    --output "$residual_output_root/seed_$residual_seed" --device "$residual_device"
