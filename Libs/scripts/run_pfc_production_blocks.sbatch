#!/bin/bash
#SBATCH --job-name=NrnSim_PfcProductionBlocks
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu05,cpu06
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_PfcProductionBlocks-%j.log

# Arguments: config qualification identities circuit block-start block-count
# native-qualification source-python project output [resume].
# The CLI retains scientific admission and completed-source resume checks.
set -euo pipefail
if [[ $# != 10 && $# != 11 ]]; then
    echo "Expected config qualification identities circuit block-start block-count native-qualification source-python project output [resume]" >&2
    exit 2
fi
config=$1
qualification=$2
identities=$3
circuit_index=$4
block_start=$5
block_count=$6
native_qualification=$7
source_python=$8
task_project=$9
destination=${10}
options=()
if [[ $# == 11 ]]; then
    [[ ${11} == resume ]] || { echo "Optional mode must be resume" >&2; exit 2; }
    options+=(--resume)
fi
case "${SLURMD_NODENAME:-}" in
    cpu05|cpu06) ;;
    *) echo "PFC production blocks require an allocated cpu05 or cpu06 CPU task" >&2; exit 2 ;;
esac
[[ -n ${SLURM_JOB_ID:-} && -z ${SLURM_JOB_GPUS:-} && -z ${SLURM_STEP_GPUS:-}
   && ${SLURM_NTASKS:-1} == 1 && ${SLURM_CPUS_PER_TASK:-1} == 1 ]] || exit 2
: "${NRNSIM_BENCHMARK_PYTHON:?Set the configured scientific Python executable}"
for value in "$config" "$qualification" "$identities" "$native_qualification" \
             "$source_python" "$task_project" "$destination" "$NRNSIM_BENCHMARK_PYTHON"; do
    case "$value" in
        /*) ;;
        *) echo "Use absolute input, project, Python and output paths" >&2; exit 2 ;;
    esac
done
[[ -f "$NRNSIM_BENCHMARK_PYTHON" && -x "$NRNSIM_BENCHMARK_PYTHON"
   && -f "$source_python" && -x "$source_python"
   && -f "$task_project/Libs/scripts/run_pfc_source_cases.sbatch"
   && -f "$task_project/Libs/scripts/run_pfc_source_cases.py" ]] || exit 2

export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
scratch=$(mktemp -d /tmp/neuronsim-pfc-production-blocks.XXXXXX)
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/Libs/population_dynamics"
# Stage the CLI's concrete import closure. Source execution uses its existing
# wrapper and the original --project path; native archives remain authoritative.
modules=(
    __init__ benchmark_dataset benchmark_production benchmark_tasks
    ca3_benchmark ca3_benchmark_cues ca3_benchmark_readout ca3_conductance_memory
    ca3_control_trace ca3_coupled_extension ca3_cue_omission ca3_event_counters
    ca3_formation_execution ca3_host_current_replay ca3_native_observation
    ca3_neuron_substeps ca3_parameter_comparison ca3_realization ca3_reference
    ca3_retrieval_execution ca3_retrieval_readouts carlsim_records circuit_campaign
    circuit_data circuit_sources config mechanistic mesocircuit_continuation
    nest_gpu_connectivity nest_gpu_live_control nest_gpu_network nest_gpu_poisson
    nest_gpu_spatial nest_reference network_snapshot pfc_clock_coupling
    pfc_count_histograms pfc_coupled_drive pfc_delay_analysis pfc_deterministic_replay
    pfc_distractor_response pfc_fluctuation_analysis pfc_input_pairing
    pfc_kernel_parameters pfc_native_drive pfc_paired_source_difference
    pfc_recurrent_gain_comparison pfc_reference_replay pfc_response_analysis
    pfc_scale_analysis pfc_source_comparison pfc_source_dataset pfc_source_execution
    pfc_temporal_readouts population_field provenance pulse_history
    pulse_history_analysis resource_probe source_variant_analysis source_variant_cohort
    spatial_activity spatial_connectivity spatial_field stimulus_trajectory trajectory_data
    v1_benchmark v1_benchmark_readout v1_circuit_stimulus v1_configured_inputs
    v1_feedforward_tuning v1_lgn_control_causality v1_lgn_runtime v1_lgn_source
    v1_lgn_spikes v1_nest_input_replay v1_nest_repeatability v1_nest_time_semantics
    v1_source_dataset v1_visual_controls
)
for module in "${modules[@]}"; do
    cp "$task_project/Libs/population_dynamics/$module.py" "$scratch/Libs/population_dynamics/"
done
cd "$scratch"
export PYTHONPATH="$scratch/Libs"
# Do not wrap this process in srun: the source wrapper starts its own one-CPU
# step. The batch process must leave that CPU available to the source step.
"$NRNSIM_BENCHMARK_PYTHON" -B -m population_dynamics.benchmark_production \
    --config "$config" --scenario pfc_working_memory \
    --qualification "$qualification" --identity-table "$identities" \
    --circuit-index "$circuit_index" run-pfc-blocks \
    --block-start "$block_start" --block-count "$block_count" \
    --native-qualification "$native_qualification" --source-python "$source_python" \
    --project "$task_project" --output "$destination" "${options[@]}"
