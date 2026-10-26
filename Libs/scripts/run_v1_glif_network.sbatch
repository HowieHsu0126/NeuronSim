#!/bin/bash
#SBATCH --job-name=NrnSim_GlifNetwork
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_GlifNetwork-%j.log
set -euo pipefail
repo=$1
reference=$2
snapshot=$3
package=$4
python=$5
library=$6
detector=$7
check=""
if (( $# >= 8 )); then check=$8; fi
[[ -z "$check" || "$check" = compare || "$check" = check || "$check" = batch || "$check" = batch_check || "$check" = source || "$check" = source_check || "$check" = source_batch || "$check" = source_batch_check ]] || exit 2
if [[ "$check" = source || "$check" = source_check || "$check" = source_batch || "$check" = source_batch_check ]]; then
  [[ $# = 9 || $# = 10 ]] || exit 2
  source_inputs=$9
  code_root=${10:-"$repo/Libs/population_dynamics"}
else
  [[ $# = 7 || $# = 8 || $# = 9 ]] || exit 2
  # A circuit comparison can use the same immutable numerical build as production.
  code_root=${9:-"$repo/Libs/population_dynamics"}
fi
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
output="$package/run_$SLURM_JOB_ID"
[[ ! -e "$output" ]] || exit 2
mkdir -p "$output"
# The resource inventory below records hardware; this probe enforces admission
# before source staging, simulation, or writes to compute-node scratch.
cp "$repo/Libs/population_dynamics/resource_probe.py" "$output/resource_probe.py"
srun --cpu-bind=none "$python" -B "$output/resource_probe.py" --output "$output/resource_preflight.json"
scratch=$(mktemp -d /tmp/neuronsim-glif-network-"$SLURM_JOB_ID".XXXXXX)
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/Libs/population_dynamics"
"$python" -B - "$code_root" "$scratch" <<'PY'
import ast,pathlib,shutil,sys
root,local=map(pathlib.Path,sys.argv[1:])
dest=local/"Libs/population_dynamics"
(local/"Libs/__init__.py").touch();(dest/"__init__.py").touch()
queue=["v1_glif_network","test_v1_glif_network","test_v1_connection_snapshot","v1_glif_dataset"];seen=set()
while queue:
    name=queue.pop()
    if name in seen:continue
    seen.add(name)
    if len(seen)>150:raise ValueError("GLIF_NETWORK_STAGING_SCOPE_EXCEEDED")
    source=root/(name+".py")
    body=source.read_text();shutil.copy2(source,dest/source.name)
    for node in ast.walk(ast.parse(body)):
        if isinstance(node,ast.ImportFrom) and node.level==1:
            names=[node.module.split(".")[0]] if node.module else [x.name for x in node.names]
            for dep in names:
                if (root/(dep+".py")).is_file():queue.append(dep)
PY
cp -r "$scratch/Libs" "$output/software"
cp "$detector" "$scratch/detect_resources.py"
export PYTHONPATH="$scratch/Libs"
cd "$scratch"
srun --cpu-bind=none "$python" -B "$scratch/detect_resources.py" -o "$output/resources.json"
if [[ "$check" = check || "$check" = batch_check || "$check" = source_check || "$check" = source_batch_check ]]; then
  srun --cpu-bind=none "$python" -B -m pytest -q Libs/population_dynamics/test_v1_glif_network.py Libs/population_dynamics/test_v1_connection_snapshot.py -p no:cacheprovider --basetemp="$scratch/test_tmp" --junitxml="$output/tests.xml" >"$output/tests.log" 2>&1
fi
if [[ "$check" = check ]]; then
  set +e
  srun --cpu-bind=none "$python" -B -m population_dynamics.v1_glif_network --archive "$reference" --cuda-library "$library" --output "$scratch/legacy" >"$output/legacy.log" 2>&1
  code=$?
  set -e
  if [[ -d "$scratch/legacy" ]]; then cp -r "$scratch/legacy" "$output/legacy"; fi
  [[ "$code" = 0 ]] || exit "$code"
fi
set +e
if [[ "$check" = source_batch || "$check" = source_batch_check ]]; then
  srun --cpu-bind=none "$python" -B -m population_dynamics.v1_glif_network --archive "$reference" --source-cases "$source_inputs" --snapshot "$snapshot" --cuda-library "$library" --output "$scratch/coupled" >"$output/execution.log" 2>&1
elif [[ "$check" = source || "$check" = source_check ]]; then
  srun --cpu-bind=none "$python" -B -m population_dynamics.v1_glif_network --archive "$reference" --source-inputs "$source_inputs" --snapshot "$snapshot" --cuda-library "$library" --output "$scratch/coupled" >"$output/execution.log" 2>&1
elif [[ "$check" = batch || "$check" = batch_check ]]; then
  srun --cpu-bind=none "$python" -B -m population_dynamics.v1_glif_network --cases "$reference" --snapshot "$snapshot" --cuda-library "$library" --output "$scratch/coupled" >"$output/execution.log" 2>&1
else
  srun --cpu-bind=none "$python" -B -m population_dynamics.v1_glif_network --archive "$reference" --snapshot "$snapshot" --cuda-library "$library" --output "$scratch/coupled" >"$output/execution.log" 2>&1
fi
code=$?
set -e
if [[ -d "$scratch/coupled" ]]; then cp -r "$scratch/coupled" "$output/coupled"; fi
[[ "$code" = 0 ]] || exit "$code"
if [[ "$check" = check ]]; then
 "$python" -B - "$output" <<'PY'
import json,numpy as np,pathlib,sys
root=pathlib.Path(sys.argv[1]);comparisons={}
for name in ["cortical_state.npz","cortical_spikes.npz"]:
    with np.load(root/"legacy"/name,allow_pickle=False) as a,np.load(root/"coupled"/name,allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        comparisons[name]={key:bool(np.array_equal(a[key],b[key])) for key in a.files}
report=dict(status="PASS_LEGACY_AND_SNAPSHOT_CONSUMERS" if all(all(v.values()) for v in comparisons.values()) else "FAILED_CONSUMER_COMPARISON",arrays=comparisons,benchmark_episodes=0)
(root/"consumer_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
assert report["status"].startswith("PASS_")
PY
fi

if [[ "$check" = source || "$check" = source_check || "$check" = source_batch || "$check" = source_batch_check ]]; then
  srun --cpu-bind=none "$python" -B -m population_dynamics.v1_glif_dataset deliver --execution "$output" >"$output/delivery.log" 2>&1
fi
