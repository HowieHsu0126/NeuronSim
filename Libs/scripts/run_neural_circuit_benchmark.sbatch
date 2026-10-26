#!/bin/bash
#SBATCH --job-name=NrnSim_BenchmarkPrepare
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_BenchmarkPrepare-%j.log
set -euo pipefail
scenario=${NRNSIM_BENCHMARK_SCENARIO:-}
validation_scope=${NRNSIM_BENCHMARK_VALIDATION_SCOPE:-tasks}
[[ "$validation_scope" == tasks || "$validation_scope" == dataset ]] || exit 2
[[ -z "$scenario" || "$scenario" == pfc_working_memory ]] || exit 2
[[ $# == 2 ]] || exit 2
repo=$1
output=$2
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
scratch=$(mktemp -d /tmp/neuronsim-benchmark-"$SLURM_JOB_ID".XXXXXX)
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/Libs/population_dynamics" "$scratch/Libs/configs"
touch "$scratch/Libs/__init__.py" "$scratch/Libs/population_dynamics/__init__.py"
modules=(benchmark_tasks benchmark_sampling ca3_benchmark_cues ca3_benchmark ca3_benchmark_readout ca3_cue_omission
  ca3_realization ca3_coupled_extension ca3_parameter_comparison ca3_reference config provenance
  circuit_campaign circuit_data population_field circuit_sources ca3_native_observation ca3_event_counters ca3_host_current_replay
  ca3_retrieval_execution ca3_formation_execution ca3_control_trace ca3_retrieval_readouts carlsim_records resource_probe
  v1_circuit_build v1_circuit_source v1_graph_identity v1_source_conversion
  test_benchmark_tasks test_circuit_campaign test_ca3_coupled_extension
  test_ca3_host_current_replay test_ca3_control_trace)
if [[ "$scenario" == pfc_working_memory ]]; then
  modules+=(pfc_kernel_parameters pfc_input_pairing pfc_native_drive pfc_deterministic_replay pfc_source_execution
    test_pfc_benchmark_tasks)
  if [[ "$validation_scope" == dataset ]]; then
    modules+=(pfc_source_dataset pfc_scale_analysis pfc_reference_replay pfc_source_comparison
      pfc_delay_analysis pfc_response_analysis pfc_temporal_readouts pfc_distractor_response
      test_pfc_source_dataset test_pfc_source_scale_analysis test_pfc_distractor_response)
  fi
fi
for module in "${modules[@]}"; do
  cp "$repo/Libs/population_dynamics/$module.py" "$scratch/Libs/population_dynamics/"
done
campaign_config=${NRNSIM_BENCHMARK_CONFIG:-$repo/Libs/configs/neural_population_dynamics.yaml}
cp "$campaign_config" "$scratch/Libs/configs/neural_population_dynamics.yaml"
if [[ "$scenario" == pfc_working_memory ]]; then
  cp "$repo/Libs/configs/neural_population_pfc_100k_source.yaml" "$scratch/Libs/configs/"
fi
cp "$repo/Libs/population_dynamics/ca3_host_current_replay.cpp" "$scratch/Libs/population_dynamics/"
cp "$repo/Libs/configs/neural_population_v1_circuit_development.yaml" "$scratch/Libs/configs/"
mkdir -p "$scratch/Libs/external/ca3_formation_retrieval/retrieval/project/ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275/src"
mkdir -p "$scratch/Libs/external/ca3_formation_retrieval/retrieval/build/ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275"
project=ca3_full_net_test_patcomp_lognormalI_09_05_23_assem_275
cp "$repo/Libs/external/ca3_formation_retrieval/retrieval/project/$project/generateCONFIGStateSTP.h" "$scratch/Libs/external/ca3_formation_retrieval/retrieval/project/$project/"
cp "$repo/Libs/external/ca3_formation_retrieval/retrieval/project/$project/generateSETUPStateSTP.h" "$scratch/Libs/external/ca3_formation_retrieval/retrieval/project/$project/"
cp "$repo/Libs/external/ca3_formation_retrieval/retrieval/project/$project/src/main_$project.cpp" "$scratch/Libs/external/ca3_formation_retrieval/retrieval/project/$project/src/"
cp "$repo/Libs/external/ca3_formation_retrieval/retrieval/build/$project/trainPat.csv" "$scratch/Libs/external/ca3_formation_retrieval/retrieval/build/$project/"
cd "$scratch"
python=${NRNSIM_BENCHMARK_PYTHON:?Set the existing scientific Python}
mkdir -p "$output"
if [[ "$scenario" == pfc_working_memory ]]; then
  cp "$0" "$output/launcher.sbatch"
  mkdir -p "$output/software"
  cp -r "$scratch/Libs" "$output/software/"
  srun "$python" -B -m Libs.population_dynamics.resource_probe --output "$output/resource_preflight.json" --required-memory-gib 1
  if [[ "$validation_scope" == dataset ]]; then
    srun "$python" -B -m pytest -q Libs/population_dynamics/test_pfc_source_dataset.py \
      Libs/population_dynamics/test_pfc_source_scale_analysis.py \
      --basetemp="$scratch/test_tmp" -p no:cacheprovider --junitxml="$output/tests.xml"
    exit 0
  fi
  srun "$python" -B -m pytest -q Libs/population_dynamics/test_pfc_benchmark_tasks.py \
    Libs/population_dynamics/test_benchmark_tasks.py Libs/population_dynamics/test_circuit_campaign.py \
    --basetemp="$scratch/test_tmp" -p no:cacheprovider --junitxml="$output/tests.xml"
  srun "$python" -B -m Libs.population_dynamics.benchmark_sampling --config Libs/configs/neural_population_dynamics.yaml \
    --scenario "$scenario" --output "$output/sampling"
  srun "$python" -B -m Libs.population_dynamics.benchmark_sampling --config Libs/configs/neural_population_dynamics.yaml \
    --episode-scenario "$scenario" --episode-index 4 --base-config Libs/configs/neural_population_pfc_100k_source.yaml \
    --output "$output/episode"
  exit 0
fi
srun "$python" -B -m pytest -q Libs/population_dynamics/test_benchmark_tasks.py Libs/population_dynamics/test_circuit_campaign.py \
  Libs/population_dynamics/test_ca3_host_current_replay.py Libs/population_dynamics/test_ca3_control_trace.py \
  --basetemp="$scratch/test_tmp" -p no:cacheprovider --junitxml="$output/tests.xml"
srun "$python" -B -m Libs.population_dynamics.benchmark_sampling --config Libs/configs/neural_population_dynamics.yaml --output "$output/sampling"
srun "$python" -B -m Libs.population_dynamics.ca3_benchmark build \
  --author-root "$scratch/Libs/external/ca3_formation_retrieval" \
  --framework /tmp/neuronsim/neuronsim-ca3-reference-589123/execution/carlsim \
  --backend "$repo/Output/neural_population_dynamics/multicircuit/ca3_source_reproduction/coupled_extension/runtime_589146/retrieval" \
  --cuda /tmp/neuronsim/neuronsim-ca3-reference-603327/cuda-11.8 \
  --retrieval-duration-ms "${NRNSIM_CA3_RETRIEVAL_DURATION_MS:-1001}" \
  --output "$scratch/ca3_runtime"
srun g++ -std=c++11 -O3 "$scratch/Libs/population_dynamics/ca3_host_current_replay.cpp" -o "$scratch/host_current_replay"
srun "$python" -B - "$scratch" <<'PY'
from pathlib import Path
import json, os
from Libs.population_dynamics.provenance import sha256_file
scratch = Path(__import__('sys').argv[1])
record = dict(status='BUILT_AND_VALIDATED', job_id=os.environ['SLURM_JOB_ID'],
              binary_sha256=sha256_file(scratch/'host_current_replay'),
              source_sha256=sha256_file(scratch/'Libs/population_dynamics/ca3_host_current_replay.cpp'),
              duration_ms_supported=[1001, 1210])
(scratch/'host_current_replay_build.json').write_text(json.dumps(record, indent=2)+'\n')
PY
cp -r "$scratch/ca3_runtime" "$output/ca3_runtime"
cp "$scratch/host_current_replay" "$scratch/host_current_replay_build.json" "$output/"
cp -r "$scratch/Libs/population_dynamics" "$output/software"
cp "$scratch/Libs/configs/neural_population_dynamics.yaml" "$output/"
