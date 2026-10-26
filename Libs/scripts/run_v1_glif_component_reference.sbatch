#!/bin/bash
#SBATCH --job-name=NrnSim_GlifReference
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_GlifReference-%j.log
set -euo pipefail
repo=$1
package=$2
python=$3
parameters=$4
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
scratch=$(mktemp -d /tmp/neuronsim-glif-reference-"$SLURM_JOB_ID".XXXXXX)
trap 'rm -rf "$scratch"' EXIT
output="$package/bank_reference/run_$SLURM_JOB_ID"
[[ ! -e "$output" ]] || exit 2
mkdir -p "$scratch/Libs/population_dynamics" "$output"
touch "$scratch/Libs/__init__.py" "$scratch/Libs/population_dynamics/__init__.py"
for module in v1_glif_component_reference test_v1_glif_component_reference v1_cell_parameters v1_nest_input_replay v1_nest_time_semantics provenance; do
  cp "$repo/Libs/population_dynamics/$module.py" "$scratch/Libs/population_dynamics/"
done
cp "$package/detect_resources.py" "$scratch/detect_resources.py"
cp -r "$scratch/Libs" "$output/software"
cd "$scratch"
srun --cpu-bind=none "$python" -B "$scratch/detect_resources.py" -o "$output/resources.json"
srun --cpu-bind=none "$python" -B -m pytest -q Libs/population_dynamics/test_v1_glif_component_reference.py --basetemp="$scratch/test_tmp" -p no:cacheprovider --junitxml="$output/tests.xml"
set +e
srun --cpu-bind=none "$python" -B -m Libs.population_dynamics.v1_glif_component_reference --parameters "$parameters" --output "$scratch/reference"
code=$?
set -e
if [[ -d "$scratch/reference" ]]; then cp -r "$scratch/reference" "$output/reference"; fi
exit "$code"
