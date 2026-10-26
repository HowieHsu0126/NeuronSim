#!/bin/bash
#SBATCH --job-name=NrnSim_PfcTaskDynamics
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_PfcTaskDynamics-%j.log

# Arguments: analysis-project dependencies-project prepared summary output
set -euo pipefail
[[ $# == 5 ]] || { echo 'Expected analysis-project dependencies-project prepared summary output' >&2; exit 2; }
: "${NRNSIM_BENCHMARK_PYTHON:?Set the configured scientific Python executable}"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
scratch=$(mktemp -d /tmp/neuronsim-pfc-task-dynamics.XXXXXX)
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/Libs/population_dynamics"
cp "$1/Libs/population_dynamics/pfc_parameter_task_dynamics.py" "$scratch/Libs/population_dynamics/"
export NRNSIM_DYNAMICS_DEPENDENCIES="$2/Libs/population_dynamics"
printf '%s\n' 'import os' '__path__.append(os.environ["NRNSIM_DYNAMICS_DEPENDENCIES"])' > "$scratch/Libs/population_dynamics/__init__.py"
export PYTHONPATH="$scratch/Libs" TMPDIR="$scratch"
"$NRNSIM_BENCHMARK_PYTHON" -B -m population_dynamics.resource_probe \
    --required-memory-gib 4 --output "${5}.resource_${SLURM_JOB_ID}.json"
"$NRNSIM_BENCHMARK_PYTHON" -B -m population_dynamics.pfc_parameter_task_dynamics \
    --prepared "$3" --summary "$4" --output "$5"
