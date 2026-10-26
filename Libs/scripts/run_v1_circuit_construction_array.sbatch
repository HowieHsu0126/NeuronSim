#!/bin/bash
#SBATCH --job-name=NrnSim_V1CircuitConstruction
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_V1Construction-%A_%a.log
set -euo pipefail
[[ $# == 5 ]] || exit 2
repo=$1
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$repo/Libs"
exec "${NRNSIM_V1_PYTHON:?Set validated scientific Python}" -B \
    -m population_dynamics.v1_circuit_construction_array \
    --repo "$repo" --stage "$2" --preparation "$3" --output "$4" --qualification "$5"
