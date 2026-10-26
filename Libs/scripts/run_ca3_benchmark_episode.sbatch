#!/bin/bash
#SBATCH --job-name=NrnSim_CA3BenchmarkEpisode
#SBATCH --partition=gpu1
#SBATCH --nodelist=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_CA3BenchmarkEpisode-%j.log
set -euo pipefail
[[ $# == 5 ]] || exit 2
repo=$1
prepared=$2
block=$3
condition=$4
destination=$5
[[ "$condition" == remaining || "$condition" == all || "$condition" =~ ^[0-9]$ ]] || exit 2
[[ ! -e "$destination" ]] || exit 2
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
scratch=$(mktemp -d /data/neuronsim/neuronsim-ca3-benchmark-"$SLURM_JOB_ID".XXXXXX)
preserve_failure() {
  result=$?
  trap - EXIT
  if [[ -d "$scratch" ]]; then
    mkdir -p "$destination/failure"
    if [[ -f "$scratch/run/report.json" ]]; then
      cp "$scratch/run/report.json" "$destination/terminal_report.json"
    fi
    for file in native.log resource_preflight.json resource_latest.json cue_targets.txt; do
      if [[ -f "$scratch/run/$file" ]]; then
        cp "$scratch/run/$file" "$destination/failure/"
      fi
    done
    printf '%s\n' "$scratch" > "$destination/failure/retained_scratch.txt"
    printf '%s\n' "$result" > "$destination/failure/exit_code.txt"
  fi
  exit "$result"
}
trap preserve_failure EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
mkdir -p "$scratch/Libs/population_dynamics" "$scratch/Libs/configs" "$scratch/formation/observations" "$scratch/analysis"
touch "$scratch/Libs/__init__.py" "$scratch/Libs/population_dynamics/__init__.py"
modules=(benchmark_tasks benchmark_sampling benchmark_dataset ca3_benchmark_cues ca3_benchmark ca3_benchmark_readout ca3_cue_omission
  ca3_realization ca3_coupled_extension ca3_parameter_comparison ca3_reference config provenance
  circuit_campaign circuit_data population_field circuit_sources ca3_native_observation ca3_event_counters ca3_host_current_replay
  ca3_retrieval_execution ca3_formation_execution ca3_control_trace ca3_retrieval_readouts carlsim_records resource_probe)
for module in "${modules[@]}"; do
  cp "$repo/Libs/population_dynamics/$module.py" "$scratch/Libs/population_dynamics/"
done
campaign_config=${NRNSIM_BENCHMARK_CONFIG:-$repo/Libs/configs/neural_population_dynamics.yaml}
cp "$campaign_config" "$scratch/Libs/configs/neural_population_dynamics.yaml"
cp -r "$prepared/ca3_runtime" "$scratch/packaged"
replay_args=()
if [[ -f "$prepared/host_current_replay" && -f "$prepared/host_current_replay_build.json" ]]; then
  cp "$prepared/host_current_replay" "$scratch/host_current_replay"
  cp "$prepared/host_current_replay_build.json" "$scratch/host_current_replay_build.json"
  replay_args=(--current-replay-engine "$scratch/host_current_replay")
elif [[ "$condition" == remaining || "$condition" == all ]]; then
  exit 2
fi
formation="$repo/Output/neural_population_dynamics/multicircuit/ca3_source_reproduction/coupled_extension/formation_589360"
analysis="$repo/Output/neural_population_dynamics/multicircuit/ca3_source_reproduction/coupled_extension/formation_analysis_589369"
formation=${NRNSIM_CA3_FORMATION_SOURCE:-$formation}
analysis=${NRNSIM_CA3_ANALYSIS_SOURCE:-$analysis}
circuit_index=${NRNSIM_CA3_CIRCUIT_INDEX:-0}
[[ "$circuit_index" =~ ^[0-9]+$ ]] || exit 2
cp "$formation/native_report.json" "$scratch/formation/"
cp "$formation/observations/recording.json" "$scratch/formation/observations/"
mkdir -p "$scratch/formation/native_run"
cp "$formation/native_run/network65p.dat" "$scratch/formation/native_run/"
for file in formation_replay.json normalization.json normalization_replay.json network65p_assem550_divnorm.dat; do
  cp "$analysis/$file" "$scratch/analysis/"
done
cd "$scratch"
prefix_args=()
if [[ -n "${NRNSIM_CA3_PREFIX_REFERENCE:-}" ]]; then
  prefix_args=(--prefix-reference "$NRNSIM_CA3_PREFIX_REFERENCE")
fi
conditions=("$condition")
if [[ "$condition" == remaining ]]; then conditions=(0 1 2 3 4 5 6 8 9); fi
if [[ "$condition" == all ]]; then conditions=(0 1 2 3 4 5 6 7 8 9); fi
reuse_zero=${NRNSIM_CA3_REUSE_CONDITION_ZERO:-}
if [[ -n "$reuse_zero" ]]; then
  [[ "$condition" == remaining && -f "$reuse_zero/episode/report.json" ]] || exit 2
  srun --cpu-bind=none "${NRNSIM_BENCHMARK_PYTHON:?}" -B - "$reuse_zero/episode/report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report['status'] == 'COMPLETE_CA3_BENCHMARK_DEVELOPMENT_EPISODE'
assert report['complete_episodes'] == 1 and report['actual_split'] == 'development'
assert report['candidate_identity']['condition_index'] == 0
assert report['unit_millisecond_counts_compared'] == report['units'] * report['duration_ms']
assert report['native_terminal_footer_verified'] is True
assert report['independent_current_replay']['exact_float32_bits_equal'] is True
PY
  mkdir -p "$destination"
  ln -s "$reuse_zero" "$destination/condition_0"
  conditions=(1 2 3 4 5 6 8 9)
fi
for selected in "${conditions[@]}"; do
  srun --cpu-bind=none "${NRNSIM_BENCHMARK_PYTHON:?Set existing scientific Python}" -B -m Libs.population_dynamics.ca3_benchmark run \
    --packaged "$scratch/packaged" --formation "$scratch/formation" --analysis "$scratch/analysis" \
    --config Libs/configs/neural_population_dynamics.yaml --output "$scratch/run" \
    --circuit-index "$circuit_index" --block-index "$block" --condition-index "$selected" "${replay_args[@]}" "${prefix_args[@]}"
  target="$destination"
  if [[ "$condition" == remaining || "$condition" == all ]]; then target="$destination/condition_$selected"; fi
  mkdir -p "$target"
  # Keep the normalized input reference after owned scratch is removed.
  ln -sfn "$analysis/network65p_assem550_divnorm.dat" "$scratch/run/native_run/network65p_assem550_divnorm.dat"
  cp -r "$scratch/run" "$target/episode"
  rm -rf "$scratch/run"
done
cp -r "$scratch/Libs" "$destination/software"
rm -rf "$scratch"
trap - EXIT
