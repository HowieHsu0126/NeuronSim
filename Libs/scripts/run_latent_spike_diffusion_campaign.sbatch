#!/bin/bash
#SBATCH --job-name=NrnSim_latent_spike_diffusion
#SBATCH --partition=gpu1
#SBATCH --nodelist=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-ldns-%A-%a.log
set -euo pipefail
repo=/workspace/NeuronSim
preparation="$1"
launcher="${2:-$preparation/Libs/scripts/run_latent_spike_diffusion_campaign.py}"
reuse_prefix="${3:-}"
work_root="${4:-/tmp/neuronsim}"
if [[ "$work_root" != /* ]] || [[ ! -d "$work_root" ]]; then
    printf 'LDNS work root must be an existing absolute node-local directory: %s\n' "$work_root" >&2
    exit 2
fi
case "$(findmnt -n -T "$work_root" -o FSTYPE)" in
    xfs|ext4) ;;
    *) printf 'LDNS work root must use a node-local disk filesystem: %s\n' "$work_root" >&2; exit 2 ;;
esac
seed=$((41 + SLURM_ARRAY_TASK_ID))
work="${work_root%/}/neuronsim-ldns-${SLURM_ARRAY_JOB_ID}-${seed}"
cleanup() {
    if [ -f "$work/archive_complete" ]; then rm -rf "$work";
    else printf 'LDNS scratch retained: %s\n' "$work" >&2; fi
}
trap cleanup EXIT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
arguments=(--repo "$repo" --preparation "$preparation" --work "$work" --seed "$seed")
if [ -n "$reuse_prefix" ]; then
    arguments+=(--reuse-inputs-from "${reuse_prefix}-${seed}")
fi
srun python3 -B "$launcher" "${arguments[@]}"
