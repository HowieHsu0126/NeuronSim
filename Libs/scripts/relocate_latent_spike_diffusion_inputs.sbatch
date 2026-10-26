#!/bin/bash
#SBATCH --job-name=NrnSim_ldns_input_relocation
#SBATCH --partition=gpu1
#SBATCH --nodelist=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-ldns-input-relocation-%j.log
set -euo pipefail
snapshot="$1"
repo=/workspace/NeuronSim
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun python3 -B "$snapshot/Libs/scripts/relocate_latent_spike_diffusion_inputs.py" \
    --repo "$repo" \
    --preparation "$repo/Output/neural_population_dynamics/baselines/latent_spike_diffusion/preparation_20261012" \
    --failed-prefix /tmp/neuronsim/neuronsim-ldns-611529 \
    --cuda-cell /tmp/neuronsim/neuronsim-ldns-cuda-611535 \
    --destination-root /data/neuronsim/neuronsim-ldns-retained-inputs-611529 \
    --output "$snapshot/run_${SLURM_JOB_ID}"
