#!/bin/bash
#SBATCH --job-name=NrnSim_PfcSourceCases
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim-PfcSource-%A_%a.log

# Packed NumPy callers override task count and memory at submission.
# PFC CPU production callers may select cpu05/cpu06 in the cpu partition.
# Allocate CPU-only work explicitly without GPUs and retain resource checks.
# CUDA callers may request GPUs on cpu03, cpu04, cpu05, cpu06 or gpu03.
# Request Slurm GPU resources explicitly and select an available node/partition combination.
set -euo pipefail

: "${NEURONSIM_PFC_PYTHON:?Set NEURONSIM_PFC_PYTHON to an absolute executable Python path}"
case "$NEURONSIM_PFC_PYTHON" in
    /*) ;;
    *) echo "NEURONSIM_PFC_PYTHON must be an absolute executable path" >&2; exit 1 ;;
esac
if [[ ! -f "$NEURONSIM_PFC_PYTHON" || ! -x "$NEURONSIM_PFC_PYTHON" ]]; then
    echo "NEURONSIM_PFC_PYTHON must be an executable file" >&2
    exit 1
fi

task_project_root="${NEURONSIM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
case "$task_project_root" in
    /*) ;;
    *) echo "Set an absolute NEURONSIM_PROJECT_ROOT or repository SLURM_SUBMIT_DIR" >&2; exit 1 ;;
esac
if [[ ! -d "$task_project_root" || ! -f "$task_project_root/Libs/scripts/run_pfc_source_cases.py" ]]; then
    echo "Project root must contain Libs/scripts/run_pfc_source_cases.py" >&2
    exit 1
fi
export NEURONSIM_PROJECT_ROOT="$task_project_root"
export PYTHONPATH="$task_project_root/Libs"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NEURONSIM_PFC_SOURCE_BACKEND="${NEURONSIM_PFC_SOURCE_BACKEND:-numpy}"
case "$NEURONSIM_PFC_SOURCE_BACKEND" in
    numpy) export CUDA_VISIBLE_DEVICES='' ;;
    cuda) ;; # Preserve the allocated CUDA visibility.
    *) echo "NEURONSIM_PFC_SOURCE_BACKEND must be numpy or cuda" >&2; exit 1 ;;
esac
exec "$NEURONSIM_PFC_PYTHON" -B "$task_project_root/Libs/scripts/run_pfc_source_cases.py" "$@"
