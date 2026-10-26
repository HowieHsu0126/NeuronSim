#!/bin/bash
#SBATCH --job-name=NrnSim_PfcIntervention
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-pfc-intervention-%j.log
set -euo pipefail
if [ "${SLURMD_NODENAME:-}" != cpu03 ] || [ "${SLURM_JOB_PARTITION:-}" != gpu ] \
   || [ -n "${SLURM_JOB_GPUS:-}" ] || [ -n "${SLURM_STEP_GPUS:-}" ]; then
    exit 2
fi
prefix=$(cat "${SLURM_SUBMIT_DIR:?}/PROJECT_PREFIX")
if [ "$prefix" != NrnSim ]; then
    printf 'Expected NeuronSim PROJECT_PREFIX NrnSim.\n' >&2
    exit 2
fi
software=""
arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
    if [ "${arguments[$index]}" = --software ]; then
        if ((index+1>=${#arguments[@]})); then exit 2; fi
        software="${arguments[$((index+1))]}"
    elif [[ "${arguments[$index]}" == --software=* ]]; then
        software="${arguments[$index]#--software=}"
    fi
done
if [ -z "$software" ]; then exit 2; fi
task_user=$(id -un)
mkdir -p "/tmp/$task_user"
work=$(mktemp -d "/tmp/$task_user/neuronsim-pfc-intervention-${SLURM_JOB_ID}.XXXXXX")
cleanup() {
    if [ -f "$work/archive_complete" ]; then
        rm -rf "$work"
    else
        printf 'PFC intervention scratch retained: %s\n' "$work" >&2
    fi
}
trap cleanup EXIT
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
srun python3 "$software/Libs/scripts/run_pfc_intervention_experiment.py" "$@" --work "$work"
