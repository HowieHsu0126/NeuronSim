#!/bin/bash
#SBATCH --job-name=NrnSim_V1BenchmarkExport
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_V1BenchmarkExport-%j.log
set -euo pipefail
[[ $# == 6 || $# == 8 ]] || exit 2
repo=$1
rates=$2
spikes=$3
native=$4
condition=$5
destination=$6
circuit_index=${7:-0}
block_index=${8:-0}
[[ "$circuit_index" =~ ^[0-9]+$ && "$block_index" =~ ^[0-9]+$ ]] || exit 2
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
scratch=$(mktemp -d /tmp/neuronsim-v1-export-"$SLURM_JOB_ID".XXXXXX)
trap 'if [[ -f "$scratch/run/report.json" ]]; then mkdir -p "$destination"; cp "$scratch/run/report.json" "$destination/terminal_report.json"; fi; rm -rf "$scratch"' EXIT
mkdir -p "$scratch/Libs/population_dynamics" "$scratch/Libs/configs"
touch "$scratch/Libs/__init__.py" "$scratch/Libs/population_dynamics/__init__.py"
modules=(benchmark_tasks circuit_campaign circuit_data config population_field provenance
  v1_benchmark v1_lgn_runtime v1_lgn_source v1_lgn_spikes v1_lgn_control_causality
  v1_visual_controls v1_circuit_stimulus v1_configured_inputs v1_nest_input_replay
  v1_nest_time_semantics v1_nest_repeatability v1_source_dataset)
for module in "${modules[@]}"; do cp "$repo/Libs/population_dynamics/$module.py" "$scratch/Libs/population_dynamics/"; done
cp "$repo/Libs/configs/neural_population_dynamics.yaml" "$scratch/Libs/configs/"
authority="$repo/Output/neural_population_dynamics/multicircuit/v1_asset_intake/original_graph_route/circuit_construction_source/coordinate_authority/run_587701"
cd "$scratch"
srun --cpu-bind=none "${NRNSIM_BENCHMARK_PYTHON:?Set existing scientific Python}" -B -m Libs.population_dynamics.v1_benchmark \
  --config Libs/configs/neural_population_dynamics.yaml --rate-source "$rates" --spike-source "$spikes" \
  --native-source "$native" --authority-source "$authority" --output "$scratch/run" \
  --block-index "$block_index" --condition-index "$condition" --circuit-index "$circuit_index"
[[ ! -e "$destination" ]]
mkdir -p "$destination"
cp -r "$scratch/run" "$destination/episode"
cp -r "$scratch/Libs" "$destination/software"
rm -rf "$scratch"
trap - EXIT
