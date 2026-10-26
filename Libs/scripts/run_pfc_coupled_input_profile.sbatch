#!/bin/bash
#SBATCH --job-name=NrnSim_PfcInputProfile
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-pfc-input-profile-%j.out
#SBATCH --error=/tmp/neuronsim-pfc-input-profile-%j.err
set -euo pipefail
: "${PROJECT_PREFIX:?PROJECT_PREFIX_REQUIRED}"
[[ "$PROJECT_PREFIX" == "NrnSim" ]]
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
unset PYTHONPATH
export NRNSIM_PROFILE_BATCH="$0"
/opt/conda/bin/python3 -B /workspace/NeuronSim/Libs/scripts/profile_pfc_coupled_input.py "$@"
