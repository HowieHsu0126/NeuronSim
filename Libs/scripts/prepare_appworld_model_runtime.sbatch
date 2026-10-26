#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_AppWorldModelSetup
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-appworld-model-setup-%j.out
#SBATCH --error=/tmp/neuronsim-appworld-model-setup-%j.err
set -euo pipefail
repo=${1:?repository root required}
scratch_root=${2:?node-local scratch root required}
base_python=${3:?base Python required}
runtime="$scratch_root/neuronsim_appworld_benchmark/vllm_0p11p2_${SLURM_JOB_ID}"
archive="$repo/Output/neural_population_dynamics/agentic_extension/benchmark/model_runtime/run_${SLURM_JOB_ID}"
test "$(hostname -s)" = cpu04
fstype=$(findmnt -T "$scratch_root" -n -o FSTYPE)
case "$fstype" in nfs|nfs4|cifs) printf '%s\n' NODE_LOCAL_SCRATCH_REQUIRED >&2; exit 1;; esac
mkdir -p "$(dirname "$runtime")" "$(dirname "$archive")"
mkdir "$runtime" "$archive"
mkdir -p "$runtime/project/population_dynamics" "$runtime/tmp" "$runtime/cache"
cp "$repo/Libs/population_dynamics/__init__.py" "$repo/Libs/population_dynamics/resource_probe.py" "$runtime/project/population_dynamics/"
cp "${BASH_SOURCE[0]}" "$archive/executed_launcher.sbatch"
export PYTHONPATH="$runtime/project" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="" TMPDIR="$runtime/tmp" PIP_CACHE_DIR="$runtime/cache/pip"
export XDG_CACHE_HOME="$runtime/cache" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# CPU04 direct access to the official index passed the scoped route check643772.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PIP_INDEX_URL=https://pypi.org/simple
finish() {
    result=$?
    trap - EXIT
    for name in resource_admission.json setup.log package_freeze.txt environment.json; do
        if [ -f "$runtime/$name" ]; then cp "$runtime/$name" "$archive/"; fi
    done
    printf '%s\n' "$runtime" > "$archive/runtime_path.txt"
    printf '%s\n' "$result" > "$archive/exit_code.txt"
    exit "$result"
}
trap finish EXIT
exec > "$runtime/setup.log" 2>&1
"$base_python" -m population_dynamics.resource_probe --output "$runtime/resource_admission.json" --required-memory-gib 8
"$base_python" -m venv "$runtime/venv"
python="$runtime/venv/bin/python"
"$python" -m pip install --index-url https://pypi.org/simple vllm==0.11.2 transformers==4.57.0 accelerate==1.13.0 huggingface-hub==0.35.3
"$python" -m pip check
"$python" -m pip freeze > "$runtime/package_freeze.txt"
"$python" - "$runtime" <<'PY'
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
runtime=Path(sys.argv[1])
packages={name:importlib.metadata.version(name) for name in
          ("vllm","torch","transformers","accelerate","huggingface-hub")}
expected={"vllm":"0.11.2","torch":"2.9.0","transformers":"4.57.0",
          "accelerate":"1.13.0","huggingface-hub":"0.35.3"}
if packages != expected:
    raise ValueError("PINNED_MODEL_PACKAGE_VERSIONS_MISMATCH")
report={"status":"MODEL_PACKAGES_PREPARED_GPU_VALIDATION_REQUIRED",
        "observed_utc":datetime.now(timezone.utc).isoformat(),
        "job_id":os.environ["SLURM_JOB_ID"],"hostname":socket.gethostname(),
        "python":str(runtime/"venv/bin/python"),"packages":packages,
        "package_freeze_sha256":hashlib.sha256((runtime/"package_freeze.txt").read_bytes()).hexdigest(),
        "model_loaded":False,"model_generations":0,"tool_call_transport_validated":False}
(runtime/"environment.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report))
PY
