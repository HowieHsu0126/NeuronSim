#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_ReferencePerf
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-reference-performance-%j.out
#SBATCH --error=/tmp/neuronsim-reference-performance-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export NRNSIM_PERFORMANCE_BATCH="$0"
"$NEURONSIM_PYTHON" -B "$NEURONSIM_PROJECT_ROOT/Libs/scripts/profile_pfc_reference_extraction.py" "$@"
