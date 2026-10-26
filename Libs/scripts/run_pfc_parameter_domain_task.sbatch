#!/bin/bash
#SBATCH --job-name=NrnSim_PfcDomainTask
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu05,cpu06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_PfcDomainTask-%A_%a.log

# Arguments: runner-project source-project observation-project prepared
# qualifications task-index source-output observation-output campaign execution-output
# [completed-source-archive]. Each array caller supplies disjoint output paths.
set -euo pipefail
[[ $# == 10 || $# == 11 ]] || { echo 'Expected ten arguments and an optional completed source archive' >&2; exit 2; }
: "${NRNSIM_BENCHMARK_PYTHON:?Set the configured scientific Python executable}"
runner_project=$1
options=()
if [[ $# == 11 ]]; then options+=(--source-archive "${11}"); fi
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
scratch=$(mktemp -d /tmp/neuronsim-pfc-domain-task.XXXXXX)
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/Libs/population_dynamics"
cp "$runner_project/Libs/population_dynamics/pfc_parameter_domain_execution.py" "$scratch/Libs/population_dynamics/"
touch "$scratch/Libs/population_dynamics/__init__.py"
# Extend the package with the existing observation snapshot's concrete imports.
export NRNSIM_PFC_OBSERVATION_PROJECT="$3"
printf '%s\n' 'import os' '__path__.append(os.path.join(os.environ["NRNSIM_PFC_OBSERVATION_PROJECT"], "Libs", "population_dynamics"))' >> "$scratch/Libs/population_dynamics/__init__.py"
export PYTHONPATH="$scratch/Libs" TMPDIR="$scratch"
"$NRNSIM_BENCHMARK_PYTHON" -B -m population_dynamics.pfc_parameter_domain_execution \
    --source-project "$2" --observation-project "$3" --prepared "$4" \
    --qualifications "$5" --index "$6" --source-output "$7" \
    --observation-output "$8" --campaign "$9" --output "${10}" "${options[@]}"
