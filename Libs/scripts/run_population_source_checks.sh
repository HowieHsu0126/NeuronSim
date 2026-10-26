#!/usr/bin/env bash
set -euo pipefail
project=${SLURM_SUBMIT_DIR:?Submit from the NeuronSim repository}
scratch="/tmp/${USER}/neuronsim-source-checks-${SLURM_JOB_ID}"
archive="$project/Output/neural_population_dynamics/multicircuit/validation"
mkdir -p "$scratch/Libs/population_dynamics" "$scratch/tmp"
export PYTHONPATH="$scratch" TMPDIR="$scratch/tmp" PYTHONDONTWRITEBYTECODE=1
touch "$scratch/Libs/__init__.py"
for name in __init__.py provenance.py v1_source_assets.py test_v1_source_assets.py; do
  cp "$project/Libs/population_dynamics/$name" "$scratch/Libs/population_dynamics/$name"
done
cd "$scratch"
set +e
python3 -m unittest Libs.population_dynamics.test_v1_source_assets -v > checks.log 2>&1
result=$?
set -e
cp checks.log "$archive/source_checks_${SLURM_JOB_ID}.log"
printf '%s\n' "$result" > "$archive/source_checks_${SLURM_JOB_ID}.exit"
exit "$result"
