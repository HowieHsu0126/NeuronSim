#!/bin/bash
#SBATCH --job-name=NrnSim_MemoryFieldComparison
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=UNLIMITED
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-memory-comparison-%j.log
set -euo pipefail
preparation="$1"
destination="$2/run_$SLURM_JOB_ID"
work_root="${MEMORY_FIELD_COMPARISON_WORK_ROOT:-/tmp/neuronsim}"
case "$(findmnt -n -T "$work_root" -o FSTYPE)" in
    xfs|ext4) ;;
    *) printf 'Memory field comparison requires node-local scratch\n' >&2; exit 2 ;;
esac
work="${work_root%/}/neuronsim-memory-comparison-$SLURM_JOB_ID"
cleanup() {
    if [[ -f "$work/archive_complete" ]]; then rm -rf "$work";
    else printf 'Memory comparison scratch retained: %s\n' "$work" >&2; fi
}
trap cleanup EXIT
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
srun python3 -B "$preparation/Libs/scripts/run_memory_field_comparison.py" \
    --run --preparation "$preparation" --work "$work" --destination "$destination"
