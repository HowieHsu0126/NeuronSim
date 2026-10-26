#!/bin/bash
#SBATCH --job-name=NrnSim_BenchmarkDataset
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_BenchmarkDataset-%j.log
set -euo pipefail
[[ $# -ge 4 ]] || exit 2
repo=$1
destination=$2
scenario=$3
shift 3
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
scratch=$(mktemp -d /tmp/neuronsim-benchmark-dataset-"$SLURM_JOB_ID".XXXXXX)
trap 'if [[ -f "$scratch/run/report.json" ]]; then mkdir -p "$destination"; cp "$scratch/run/report.json" "$destination/terminal_report.json"; fi; rm -rf "$scratch"' EXIT
mkdir -p "$scratch/Libs/population_dynamics" "$scratch/Libs/configs" "$destination"
touch "$scratch/Libs/__init__.py" "$scratch/Libs/population_dynamics/__init__.py"
for module in benchmark_dataset test_benchmark_dataset test_circuit_data circuit_data circuit_campaign benchmark_tasks config provenance population_field; do
  cp "$repo/Libs/population_dynamics/$module.py" "$scratch/Libs/population_dynamics/"
done
cp "$repo/Libs/configs/neural_population_dynamics.yaml" "$scratch/Libs/configs/"
cd "$scratch"
python=${NRNSIM_BENCHMARK_PYTHON:?Set existing scientific Python}
srun --cpu-bind=none "$python" -B -m pytest -q Libs/population_dynamics/test_benchmark_dataset.py \
  --basetemp="$scratch/test_tmp" -p no:cacheprovider --junitxml="$destination/tests.xml"
args=()
for source in "$@"; do args+=(--source "$source"); done
srun --cpu-bind=none "$python" -B -m Libs.population_dynamics.benchmark_dataset \
  --scenario "$scenario" "${args[@]}" --output "$scratch/run"
cp -r "$scratch/run" "$destination/assembly"
cp -r "$scratch/Libs" "$destination/software"
rm -rf "$scratch"
trap - EXIT
