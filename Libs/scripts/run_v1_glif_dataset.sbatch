#!/bin/bash
#SBATCH --job-name=NrnSim_GlifDataset
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_GlifDataset-%j.log
set -euo pipefail
repo=$1
execution=$2
controls=$3
authority=$4
package=$5
python=$6
detector=$7
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
scratch=$(mktemp -d /tmp/neuronsim-glif-dataset-"$SLURM_JOB_ID".XXXXXX)
trap 'rm -rf "$scratch"' EXIT
output="$package/run_$SLURM_JOB_ID"
[[ ! -e "$output" ]] || exit 2
mkdir -p "$output" "$scratch/Libs/population_dynamics"
"$python" -B - "$repo" "$scratch" <<'PY'
import ast,pathlib,shutil,sys
root,local=map(pathlib.Path,sys.argv[1:])
dest=local/"Libs/population_dynamics"
(local/"Libs/__init__.py").touch();(dest/"__init__.py").touch()
queue=["v1_glif_dataset","test_v1_glif_dataset"];seen=set()
while queue:
    name=queue.pop()
    if name in seen:continue
    seen.add(name)
    if len(seen)>150:raise ValueError("GLIF_DATASET_STAGING_SCOPE_EXCEEDED")
    source=root/"Libs/population_dynamics"/(name+".py")
    body=source.read_text();shutil.copy2(source,dest/source.name)
    for node in ast.walk(ast.parse(body)):
        if isinstance(node,ast.ImportFrom) and node.level==1:
            names=[node.module.split(".")[0]] if node.module else [x.name for x in node.names]
            for dep in names:
                if (root/"Libs/population_dynamics"/(dep+".py")).is_file():queue.append(dep)
PY
cp -r "$scratch/Libs" "$output/software"
cp "$detector" "$scratch/detect_resources.py"
export PYTHONPATH="$scratch/Libs"
cd "$scratch"
srun --cpu-bind=none "$python" -B "$scratch/detect_resources.py" -o "$output/resources.json"
srun --cpu-bind=none "$python" -B -m pytest -q Libs/population_dynamics/test_v1_glif_dataset.py -p no:cacheprovider --basetemp="$scratch/test_tmp" --junitxml="$output/tests.xml" >"$output/tests.log" 2>&1
if [[ ! -e "$execution/delivery.json" ]]; then
  srun --cpu-bind=none "$python" -B -m population_dynamics.v1_glif_dataset deliver --execution "$execution" >"$output/delivery.log" 2>&1
fi
set +e
srun --cpu-bind=none "$python" -B -m population_dynamics.v1_glif_dataset export --execution "$execution" --controls "$controls" --authority "$authority" --output "$scratch/export" >"$output/export.log" 2>&1
code=$?
set -e
if [[ -d "$scratch/export" ]]; then cp -r "$scratch/export" "$output/export"; fi
exit "$code"
