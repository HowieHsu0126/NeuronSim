#!/bin/bash
#SBATCH --job-name=NrnSim_ldns_joint_comparison
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-ldns-comparison-%j.log
set -euo pipefail
preparation="$1"
shift
repo=/workspace/NeuronSim
mkdir -p /tmp/neuronsim
work=$(mktemp -d "/tmp/neuronsim/neuronsim-ldns-comparison-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cp -r "$preparation/Libs" "$work/"
export PYTHONPATH="$work" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
srun python3 -m Libs.population_dynamics.resource_probe --output "$work/preflight.json" --required-memory-gib 4
comparison_args=(
    --ldns-root "$repo/Output/neural_population_dynamics/baselines/latent_spike_diffusion/complete_cohort_development/campaign_20261012"
    --population-root "$repo/Output/neural_population_dynamics/model/trajectory_development/run_606139"
    --output "$repo/Output/neural_population_dynamics/baselines/latent_spike_diffusion/joint_comparison_20261012"
)
# Explicit calls retain the Python CLI required-input contract.
if (( $# > 0 )); then
    comparison_args=("$@")
fi
srun python3 -m Libs.population_dynamics.latent_spike_diffusion_comparison "${comparison_args[@]}"
