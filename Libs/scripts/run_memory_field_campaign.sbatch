#!/bin/bash
#SBATCH --job-name=NrnSim_MemoryFieldTraining
#SBATCH --partition=gpu1
#SBATCH --nodelist=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=UNLIMITED
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-memno-%A-%a.log
set -euo pipefail
preparation="$1"
destination="$2"
seed=$((41 + ${SLURM_ARRAY_TASK_ID:-0}))
work_root="${MEMORY_FIELD_WORK_ROOT:-/data/neuronsim}"
work="${work_root%/}/neuronsim-memno-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${seed}"
case "$(findmnt -n -T "$work_root" -o FSTYPE)" in
    xfs|ext4) ;;
    *) printf 'MemNO requires a node-local work disk\n' >&2; exit 2 ;;
esac
cleanup() {
    if [[ -f "$work/archive_complete" ]]; then rm -rf "$work";
    else printf 'MemNO scratch retained: %s\n' "$work" >&2; fi
}
trap cleanup EXIT
shift 2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0
srun python3 -B "$preparation/Libs/scripts/run_memory_field_campaign.py" \
    --preparation "$preparation" --work "$work" --destination "$destination/seed_$seed" --seed "$seed" "$@"
