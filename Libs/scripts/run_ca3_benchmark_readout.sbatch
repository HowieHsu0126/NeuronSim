#!/bin/bash
#SBATCH --job-name=NrnSim_CA3BenchmarkReadout
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_CA3BenchmarkReadout-%j.log
set -euo pipefail
[[ $# == 3 || $# == 4 ]] || exit 2
repo=$1
if [[ $# == 3 ]]; then
  block=$2
  destination=$3
  sources=()
  for condition in 0 1 2 3 4 5 6 7 8 9; do sources+=("$block/condition_$condition/episode"); done
else
  first=$2
  remaining=$3
  destination=$4
  sources=("$first/episode")
  for condition in 0 1 2 3 4 5 6 8 9; do sources+=("$remaining/condition_$condition/episode"); done
fi
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
scratch=$(mktemp -d /tmp/neuronsim-ca3-readout-"$SLURM_JOB_ID".XXXXXX)
trap 'if [[ -f "$scratch/readout/report.json" ]]; then mkdir -p "$destination"; cp "$scratch/readout/report.json" "$destination/terminal_report.json"; fi; rm -rf "$scratch"' EXIT
mkdir -p "$scratch/Libs/population_dynamics"
touch "$scratch/Libs/__init__.py" "$scratch/Libs/population_dynamics/__init__.py"
for module in benchmark_tasks benchmark_dataset circuit_data circuit_campaign config population_field ca3_benchmark_readout ca3_native_observation provenance; do
  cp "$repo/Libs/population_dynamics/$module.py" "$scratch/Libs/population_dynamics/"
done
args=()
index=0
for source in "${sources[@]}"; do
  target="$scratch/episode_$index"
  mkdir -p "$target/dataset" "$target/native_run"
  for file in report.json ordered_cue_commands.npz native_observation.bin; do cp "$source/$file" "$target/"; done
  cp "$source/native_run/trainPat.csv" "$target/native_run/"
  "${NRNSIM_BENCHMARK_PYTHON:?Set existing scientific Python}" -B - "$source/dataset" "$target/dataset" <<'PY'
import json, pathlib, shutil, sys
source, target = map(pathlib.Path, sys.argv[1:])
catalog = json.loads((source / 'dataset.json').read_text())
names = ['dataset.json'] + [row['path'] for row in catalog['circuits'].values()] + [row['path'] for row in catalog['shards']]
for name in names:
    relative = pathlib.Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('CA3_BENCHMARK_RELATIVE_ARCHIVE_PATH_REQUIRED')
    path = target / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / relative, path)
PY
  args+=(--episode "$target")
  index=$((index + 1))
done
cd "$scratch"
srun --cpu-bind=none "${NRNSIM_BENCHMARK_PYTHON:?Set existing scientific Python}" -B -m Libs.population_dynamics.ca3_benchmark_readout \
  "${args[@]}" --output "$scratch/readout"
[[ ! -e "$destination" ]]
mkdir -p "$destination"
cp -r "$scratch/readout" "$destination/analysis"
cp -r "$scratch/Libs" "$destination/software"
rm -rf "$scratch"
trap - EXIT
