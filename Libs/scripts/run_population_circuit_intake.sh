#!/usr/bin/env bash
# Run inside a CPU allocation; stage sources and environments on local storage.
set -euo pipefail
project=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
test -f "$project/Libs/population_dynamics/circuit_sources.py"
scratch=${1:?Pass a task-owned compute-node-local scratch directory}
source_name=${2:-all}
archive="$project/Output/neural_population_dynamics/multicircuit/source_intake"
mkdir -p "$scratch/Libs/population_dynamics" "$archive" "$scratch/tmp"
export TMPDIR="$scratch/tmp" MPLCONFIGDIR="$scratch/matplotlib"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONDONTWRITEBYTECODE=1
cd "$scratch"
for name in __init__.py provenance.py circuit_sources.py requirements-circuits.txt; do
  cp "$project/Libs/population_dynamics/$name" "Libs/population_dynamics/$name"
done
cp "$project/Output/neural_population_dynamics/multicircuit/operations/detect_resources.py" detect_resources.py
if [[ ! -x "$scratch/venv/bin/python" ]]; then
  /usr/bin/python3 -m venv "$scratch/venv"
fi
if ! env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    "$scratch/venv/bin/python" -m pip --isolated install --disable-pip-version-check \
    --index-url https://pypi.org/simple --cache-dir "$scratch/pip-cache" --retries 0 \
    -r Libs/population_dynamics/requirements-circuits.txt \
    > "install_${SLURM_JOB_ID}.log" 2>&1; then
  cp "install_${SLURM_JOB_ID}.log" "$archive/"
  exit 1
fi
"$scratch/venv/bin/python" -m pip freeze > "$archive/packages_${SLURM_JOB_ID}.txt"
"$scratch/venv/bin/python" detect_resources.py -o "$archive/resources_${SLURM_JOB_ID}.json"
printf '%s\n' "$scratch" > "$archive/runtime_${SLURM_JOB_ID}.txt"
set +e
"$scratch/venv/bin/python" -m Libs.population_dynamics.circuit_sources --source "$source_name" --output "$scratch/sources"
result=$?
set -e
"$scratch/venv/bin/python" - "$scratch/sources" "$project/Libs/external" "$archive" <<'PY'
import json, pathlib, shutil, sys
source, destination, archive = map(pathlib.Path, sys.argv[1:])
for folder in source.iterdir():
    if folder.is_dir() and (folder/'source.json').is_file():
        manifest = json.loads((folder/'source.json').read_text())
        target = destination/folder.name
        target.mkdir(exist_ok=True)
        for row in [*manifest['files'], {'path':'source.json'}]:
            src, dst = folder/row['path'], target/row['path']
            if dst.exists() and src.read_bytes() != dst.read_bytes():
                raise RuntimeError('Existing external source differs: '+str(dst))
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists(): shutil.copy2(src, dst)
        (archive/(folder.name+'.json')).write_text(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2)+'\n')
    elif folder.is_file():
        shutil.copy2(folder,archive/folder.name)
PY
exit "$result"
