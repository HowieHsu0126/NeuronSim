#!/bin/bash
#SBATCH --job-name=NrnSim_V1ProductionExport
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_V1ProductionExport-%j.log
set -euo pipefail
[[ $# == 8 || $# == 9 ]] || exit 2
repo=$1
config=$2
qualification=$3
identities=$4
circuit_index=$5
block_index=$6
inputs=$7
destination=$8
mode=${9:-export}
options=()
case "$mode" in
  prepare_glif) options+=(--prepare-glif-inputs) ;;
  resume) options+=(--resume) ;;
  export) ;;
  *) exit 2 ;;
esac
[[ "${SLURMD_NODENAME:-}" == cpu03 && -n "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_GPUS:-}" ]] || exit 2
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
scratch=$(mktemp -d /tmp/neuronsim-v1-production-export.XXXXXX)
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/Libs/population_dynamics"
# Source modules stay on node-local scratch during conversion. This also keeps
# optional CA3/PFC imports used by the established exporter available.
cp "$repo"/Libs/population_dynamics/*.py "$scratch/Libs/population_dynamics/"
cd "$scratch"
export PYTHONPATH="$scratch/Libs"
srun --ntasks=1 --cpu-bind=none "${NRNSIM_BENCHMARK_PYTHON:?Set existing scientific Python}" -B \
  -m population_dynamics.benchmark_production_block --config "$config" --scenario v1_visual \
  --qualification "$qualification" --identity-table "$identities" --circuit-index "$circuit_index" \
  --block-index "$block_index" --inputs "$inputs" "${options[@]}" --output "$destination"
