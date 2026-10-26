#!/usr/bin/env bash
# Resolve small public asset metadata on a compute node before bulk intake.
set -euo pipefail
project=${SLURM_SUBMIT_DIR:?Submit from the NeuronSim repository}
scratch="/tmp/${USER}/neuronsim-v1-intake-${SLURM_JOB_ID}"
archive="$project/Output/neural_population_dynamics/multicircuit/v1_asset_intake/attempt_${SLURM_JOB_ID}"
mkdir -p "$scratch/Libs/population_dynamics" "$scratch/tmp" "$archive"
export TMPDIR="$scratch/tmp" PYTHONPATH="$scratch" PYTHONDONTWRITEBYTECODE=1
touch "$scratch/Libs/__init__.py"
for name in __init__.py provenance.py v1_source_assets.py; do
  cp "$project/Libs/population_dynamics/$name" "$scratch/Libs/population_dynamics/$name"
done
cp "$project/Output/neural_population_dynamics/multicircuit/v1_asset_intake/point_index/asset_index.html" "$scratch/point_index.html"
cd "$scratch"
if [[ -n "${1:-}" ]]; then
  cp "$project/$1" download.json
  set +e
  python3 -m Libs.population_dynamics.v1_source_assets --source-receipt download.json --output asset
  result=$?
  set -e
  if [[ -f asset/source.json ]]; then cp asset/source.json "$archive/source.json"; fi
  printf "%s\n" "$scratch/asset" > "$archive/runtime.txt"
  exit "$result"
fi
set +e
python3 -m Libs.population_dynamics.v1_source_assets --parent-index point_index.html \
  --child-name config.json --text-asset --output config
result=$?
if [[ "$result" == 0 ]]; then
  python3 -m Libs.population_dynamics.v1_source_assets --parent-index point_index.html \
    --child-name network --output network
  result=$?
fi
set -e
for item in config network; do
  if [[ -d "$item" ]]; then cp -r "$item" "$archive/"; fi
done
printf '%s\n' "$scratch" > "$archive/runtime.txt"
exit "$result"
