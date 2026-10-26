#!/bin/bash
#SBATCH --job-name=NrnSim_population_trajectory
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --array=0-2%1
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-population-trajectory-%A-%a.log
set -euo pipefail
repo=/workspace/NeuronSim
device=${1:-cpu}
if [ "$#" -gt 0 ]; then shift; fi
case "$device" in
 cpu) export CUDA_VISIBLE_DEVICES=""; if { [ "$SLURMD_NODENAME" != cpu03 ] || [ "$SLURM_JOB_PARTITION" != gpu ]; } && { [ "$SLURMD_NODENAME" != cpu04 ] || [ "$SLURM_JOB_PARTITION" != cpu ]; }; then exit 2; fi; if [ -n "${SLURM_JOB_GPUS:-}" ]; then exit 2; fi ;;
 cuda) case "$SLURMD_NODENAME" in cpu03|cpu04|cpu05|cpu06|gpu03) ;; *) exit 2 ;; esac
       if [ -z "${SLURM_JOB_GPUS:-}" ]; then exit 2; fi ;;
 *) exit 2 ;;
esac
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
work=$(mktemp -d "/tmp/neuronsim/neuronsim-population-trajectory-${SLURM_JOB_ID}.XXXXXX")
cleanup() {
 if [ -f "$work/archive_complete" ]; then rm -rf "$work";
 else printf 'Campaign scratch retained for recovery: %s\n' "$work" >&2; fi
}
trap cleanup EXIT
runner="$repo/Libs/scripts/run_population_trajectory_campaign.py"
# A source preparation pins the caller as well as the numerical software.
arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
 if [ "${arguments[$index]}" = --pfc-preparation ] || [ "${arguments[$index]}" = --v1-preparation ]; then
  if ((index+1>=${#arguments[@]})); then exit 2; fi
  runner="${arguments[$((index+1))]}/Libs/scripts/run_population_trajectory_campaign.py"
 elif [[ "${arguments[$index]}" == --pfc-preparation=* ]] || [[ "${arguments[$index]}" == --v1-preparation=* ]]; then
  runner="${arguments[$index]#*=}/Libs/scripts/run_population_trajectory_campaign.py"
 fi
done
srun python3 "$runner" --repo "$repo" --work "$work" --device "$device" "$@"
