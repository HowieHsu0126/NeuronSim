#!/usr/bin/env bash
# Native GPU source qualification; keep the checkpoint off the node root disk.
set -euo pipefail
project=${SLURM_SUBMIT_DIR:?Submit from the NeuronSim repository}
python=${CA3_PYTHON:-python3}
scratch="/tmp/${USER}/neuronsim-ca3-reference-${SLURM_JOB_ID}"
native="/dev/shm/neuronsim-ca3-reference-${SLURM_JOB_ID}"
archive="$project/Output/neural_population_dynamics/multicircuit/ca3_source_reproduction/attempt_${SLURM_JOB_ID}"
if [[ -n "${CA3_COUPLED_EXTENSION_CONFIG:-}" ]]; then
  archive="$project/Output/neural_population_dynamics/multicircuit/ca3_source_reproduction/coupled_extension/build_${SLURM_JOB_ID}"
fi
case "${CA3_NATIVE_REFERENCE:-0}" in
  0) ;;
  1)
    if [[ -n "${CA3_COUPLED_EXTENSION_CONFIG:-}" || "${CA3_CONDUCTANCE_MEMORY:-0}" != 1 ]]; then
      echo "Native reference requires conductance correction without coupled configuration" >&2; exit 2
    fi
    archive="$project/Output/neural_population_dynamics/multicircuit/ca3_source_reproduction/source_scale_reference/build_${SLURM_JOB_ID}"
    ;;
  *) echo "CA3_NATIVE_REFERENCE must be 0 or 1" >&2; exit 2 ;;
esac
mkdir -p "$scratch/Libs/population_dynamics" "$scratch/tmp" "$archive"
export PYTHONPATH="$scratch" TMPDIR="$scratch/tmp" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
touch "$scratch/Libs/__init__.py"
for name in __init__.py provenance.py config.py circuit_sources.py carlsim_records.py ca3_native_observation.py ca3_parameter_comparison.py ca3_coupled_extension.py ca3_conductance_memory.py ca3_reference.py ca3_toolchain.py; do
  cp "$project/Libs/population_dynamics/$name" "$scratch/Libs/population_dynamics/$name"
done
if [[ "${CA3_CUE_OMISSION:-0}" == 1 ]]; then
  cp "$project/Libs/population_dynamics/ca3_cue_omission.py" "$scratch/Libs/population_dynamics/ca3_cue_omission.py"
fi
if [[ -n "${CA3_NEURON_SUBSTEPS:-}" ]]; then
  cp "$project/Libs/population_dynamics/ca3_neuron_substeps.py" "$scratch/Libs/population_dynamics/ca3_neuron_substeps.py"
fi
if [[ -n "${CA3_SOURCE_REALIZATION_CONFIG:-}" ]]; then
  cp "$project/Libs/population_dynamics/ca3_realization.py" "$scratch/Libs/population_dynamics/ca3_realization.py"
fi
cd "$scratch"
"$python" - <<'PY'
import pathlib
states=[]
for p in pathlib.Path('/proc').iterdir():
    if p.name.isdigit():
        try: states.append((p/'stat').read_text().rsplit(')',1)[1].split()[0])
        except (FileNotFoundError,ProcessLookupError,PermissionError): pass
print({'D_state':states.count('D')},flush=True)
if states.count('D') >= 15: raise SystemExit('Node health prevents native source execution')
PY
build_args=(--scope "${CA3_SCOPE:-prefix}")
case "${CA3_NEURON_SUBSTEPS:-}" in
  '') ;;
  5|10|20|40) build_args+=(--neuron-substeps "$CA3_NEURON_SUBSTEPS") ;;
  *) echo "CA3_NEURON_SUBSTEPS must be 5, 10, 20 or 40" >&2; exit 2 ;;
esac
case "${CA3_CUE_OMISSION:-0}" in
  0) ;;
  1) build_args+=(--cue-omission --local-retrieval-input) ;;
  *) echo "CA3_CUE_OMISSION must be 0 or 1" >&2; exit 2 ;;
esac
case "${CA3_CONDUCTANCE_MEMORY:-0}" in
  0) ;;
  1) build_args+=(--conductance-memory) ;;
  *) echo "CA3_CONDUCTANCE_MEMORY must be 0 or 1" >&2; exit 2 ;;
esac
case "${CA3_NATIVE_OBSERVATION:-0}" in
  0) ;;
  1) build_args+=(--native-observation) ;;
  *) echo "CA3_NATIVE_OBSERVATION must be 0 or 1" >&2; exit 2 ;;
esac
if [[ -n "${CA3_COUPLED_EXTENSION_CONFIG:-}" ]]; then
  cp "$CA3_COUPLED_EXTENSION_CONFIG" "$scratch/coupled_extension.yaml"
  build_args+=(--coupled-extension-config "$scratch/coupled_extension.yaml")
  if [[ "${CA3_SCOPE:-prefix}" == retrieval ]]; then
    build_args+=(--local-retrieval-input)
  fi
fi
if [[ "${CA3_NATIVE_REFERENCE:-0}" == 1 && "${CA3_SCOPE:-prefix}" == retrieval ]]; then
  build_args+=(--local-retrieval-input)
fi
if [[ -n "${CA3_SOURCE_REALIZATION_CONFIG:-}" ]]; then
  cp "$CA3_SOURCE_REALIZATION_CONFIG" "$scratch/source_realization.yaml"
  build_args+=(--source-realization-config "$scratch/source_realization.yaml")
fi
install_args=()
if [[ "${2:-}" == --build-only ]]; then
  arch=${3:?Specify the target CUDA architecture for a CPU build}
  build_args+=(--build-only)
  install_args=(--compile-only)
else
  arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" | tr -d '.')
fi
cuda_root=${1:-}
if [[ "$cuda_root" == install-cuda118 ]]; then
  set +e
  "$python" -m Libs.population_dynamics.ca3_toolchain --output "$scratch/cuda-11.8" --cuda-arch "$arch" "${install_args[@]}"
  result=$?
  set -e
  for file in toolchain.json redistribution.json probe_build.log; do
    if [[ -f "$scratch/cuda-11.8/$file" ]]; then cp "$scratch/cuda-11.8/$file" "$archive/"; fi
  done
  if [[ "$result" != 0 ]]; then exit "$result"; fi
  cuda_root="$scratch/cuda-11.8"
fi
cuda_args=()
boost_include=${CA3_BOOST_INCLUDE:-$("$python" -c 'import sys; print(sys.prefix + "/include")')}
if [[ -f "$boost_include/boost/iterator/counting_iterator.hpp" ]]; then
  build_args+=(--boost-include "$boost_include")
fi
if [[ -n "$cuda_root" ]]; then
  cuda_args=(--cuda-root "$cuda_root")
  export PATH="$cuda_root/bin:$PATH" LD_LIBRARY_PATH="$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
fi
set +e
"$python" -m Libs.population_dynamics.ca3_reference --source-root "$project/Libs/external" \
  --workdir "$scratch/execution" --cuda-arch "$arch" --native-run-dir "$native" "${cuda_args[@]}" "${build_args[@]}"
result=$?
set -e
if [[ -d execution ]]; then
  for file in report.json materialized_build_inputs.json cuda_samples_source.json build_0.log build_1.log native.log; do
    if [[ -f execution/$file ]]; then cp "execution/$file" "$archive/"; fi
  done
fi
printf '%s\n' "$scratch" > "$archive/runtime.txt"
printf '%s\n' "$native" > "$archive/native_run_directory.txt"
exit "$result"
