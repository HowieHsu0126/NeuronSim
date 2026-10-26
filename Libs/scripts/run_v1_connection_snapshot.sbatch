#!/bin/bash
#SBATCH --job-name=NrnSim_ConnectionSnapshot
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_ConnectionSnapshot-%j.log
set -euo pipefail
repo=$1
reference=$2
package=$3
python=$4
detector=$5
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
scratch=$(mktemp -d /tmp/neuronsim-connection-snapshot-"$SLURM_JOB_ID".XXXXXX)
trap 'rm -rf "$scratch"' EXIT
output="$package/run_$SLURM_JOB_ID"
[[ ! -e "$output" ]] || exit 2
mkdir -p "$output" "$scratch/Libs/population_dynamics"
"$python" -B - "$repo" "$scratch" <<'PY'
import ast,pathlib,shutil,sys
root,local=map(pathlib.Path,sys.argv[1:])
destination=local/"Libs/population_dynamics"
(local/"Libs/__init__.py").touch();(destination/"__init__.py").touch()
queue=["v1_connection_snapshot","test_v1_connection_snapshot"];seen=set()
while queue:
    name=queue.pop()
    if name in seen:continue
    seen.add(name)
    if len(seen)>150:raise ValueError("SNAPSHOT_STAGING_SCOPE_EXCEEDED")
    source=root/"Libs/population_dynamics"/(name+".py")
    body=source.read_text();shutil.copy2(source,destination/source.name)
    for item in ast.walk(ast.parse(body)):
        if isinstance(item,ast.ImportFrom) and item.level==1:
            names=[item.module.split(".")[0]] if item.module else [x.name for x in item.names]
            for dep in names:
                if (root/"Libs/population_dynamics"/(dep+".py")).is_file():queue.append(dep)
PY
cp -r "$scratch/Libs" "$output/software"
cp "$detector" "$scratch/detect_resources.py"
export PYTHONPATH="$scratch/Libs"
cd "$scratch"
srun --cpu-bind=none "$python" -B "$scratch/detect_resources.py" -o "$output/resources.json"
srun --cpu-bind=none "$python" -B -m pytest -q Libs/population_dynamics/test_v1_connection_snapshot.py -p no:cacheprovider --basetemp="$scratch/test_tmp" --junitxml="$output/tests.xml" >"$output/tests.log" 2>&1
set +e
srun --cpu-bind=none "$python" -B -m population_dynamics.v1_connection_snapshot --reference "$reference" --output "$scratch/snapshot" >"$output/execution.log" 2>&1
code=$?
set -e
if [[ -d "$scratch/snapshot" ]]; then cp -r "$scratch/snapshot" "$output/snapshot"; fi
exit "$code"
