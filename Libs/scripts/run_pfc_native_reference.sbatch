#!/bin/bash
#SBATCH --job-name=NrnSim_PfcNativeReference
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=none
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim-PfcNativeReference-%j.log
set -euo pipefail
: "${NEURONSIM_PFC_PYTHON:?Set an explicit absolute controller Python path}"
: "${NEURONSIM_PROJECT_ROOT:?Set an explicit absolute project root}"
case "$NEURONSIM_PFC_PYTHON" in /*) ;; *) echo "Absolute controller Python required" >&2; exit 1 ;; esac
case "$NEURONSIM_PROJECT_ROOT" in /*) ;; *) echo "Absolute project root required" >&2; exit 1 ;; esac
if [[ ! -f "$NEURONSIM_PFC_PYTHON" || ! -x "$NEURONSIM_PFC_PYTHON" ]]; then
    echo "Executable controller Python required" >&2
    exit 1
fi
if [[ ! -f "$NEURONSIM_PROJECT_ROOT/Libs/population_dynamics/pfc_native_reference_execution.py" ]]; then
    echo "Project root lacks native reference adapter" >&2
    exit 1
fi
export PYTHONPATH="$NEURONSIM_PROJECT_ROOT/Libs"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$NEURONSIM_PFC_PYTHON" -B -m population_dynamics.pfc_native_reference_execution "$@"
