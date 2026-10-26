#!/bin/bash
#SBATCH --job-name=NrnSim_V1BenchmarkReadout
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_V1BenchmarkReadout-%j.log
set -euo pipefail
[[ $# -ge 3 ]] || exit 2
repo=$1
destination=$2
shift 2
[[ ! -e "$destination" ]] || exit 2
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
scratch=$(mktemp -d /tmp/neuronsim-v1-readout-"$SLURM_JOB_ID".XXXXXX)
trap 'if [[ -f "$scratch/run/report.json" ]]; then mkdir -p "$destination"; cp "$scratch/run/report.json" "$destination/terminal_report.json"; fi; rm -rf "$scratch"' EXIT
mkdir -p "$scratch/Libs/population_dynamics" "$scratch/Libs/configs"
touch "$scratch/Libs/__init__.py" "$scratch/Libs/population_dynamics/__init__.py"
modules=(benchmark_dataset benchmark_tasks circuit_campaign circuit_data config population_field provenance
  v1_benchmark_readout v1_feedforward_tuning v1_visual_controls v1_circuit_stimulus v1_configured_inputs
  v1_lgn_control_causality v1_lgn_runtime v1_lgn_source v1_lgn_spikes v1_nest_input_replay v1_nest_time_semantics)
for module in "${modules[@]}"; do cp "$repo/Libs/population_dynamics/$module.py" "$scratch/Libs/population_dynamics/"; done
cp "$repo/Libs/configs/neural_population_dynamics.yaml" "$scratch/Libs/configs/"
cd "$scratch"
args=()
for source in "$@"; do
  # Copy only catalog-declared circuits and episode shards to node-local scratch.
  staged="$scratch/input_${#args[@]}"
  "${NRNSIM_BENCHMARK_PYTHON:?Set existing scientific Python}" -B - "$source" "$staged" "$scratch/source_locations.json" <<'PY'
import json, pathlib, shutil, sys
from Libs.population_dynamics.provenance import sha256_file
source, target, locations = map(pathlib.Path, sys.argv[1:])
catalog = json.loads((source / 'dataset.json').read_text())
names = ['dataset.json'] + [row['path'] for row in catalog['circuits'].values()] + [row['path'] for row in catalog['shards']]
for name in names:
    relative = pathlib.Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('V1_BENCHMARK_RELATIVE_ARCHIVE_PATH_REQUIRED')
    path = target / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / relative, path)
records = json.loads(locations.read_text()) if locations.exists() else []
records.append(dict(source=str(source), staged_path=str(target), catalog_sha256=sha256_file(target / 'dataset.json')))
locations.write_text(json.dumps(records, indent=2) + '\n')
PY
  args+=(--source "$staged")
done
srun --cpu-bind=none "$NRNSIM_BENCHMARK_PYTHON" -B -m Libs.population_dynamics.v1_benchmark_readout \
  --config Libs/configs/neural_population_dynamics.yaml "${args[@]}" --output "$scratch/run"
cp "$scratch/source_locations.json" "$scratch/run/source_locations.json"
mkdir -p "$destination"
cp -r "$scratch/run" "$destination/readout"
cp -r "$scratch/Libs" "$destination/software"
rm -rf "$scratch"
trap - EXIT
