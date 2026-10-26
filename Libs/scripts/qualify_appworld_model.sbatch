#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_AppWorldModelProbe
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:L40:2
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-appworld-model-probe-%j.out
#SBATCH --error=/tmp/neuronsim-appworld-model-probe-%j.err
set -euo pipefail
repo=${1:?repository root required}
scratch_root=${2:?node-local scratch root required}
model_python=${3:?prepared model-environment Python required}
model_directory=${4:?existing node-local model directory required}
package_receipt=${5:?completed model-package environment.json required}
software_validation=${6:?completed model-probe software validation.json required}
work="$scratch_root/neuronsim_appworld_benchmark/model_probe_${SLURM_JOB_ID}"
archive="$repo/Output/neural_population_dynamics/agentic_extension/benchmark/model_transport/run_${SLURM_JOB_ID}"
test "$(hostname -s)" = cpu04
fstype=$(findmnt -T "$scratch_root" -n -o FSTYPE)
case "$fstype" in nfs|nfs4|cifs) printf '%s\n' NODE_LOCAL_SCRATCH_REQUIRED >&2; exit 1;; esac
mkdir -p "$(dirname "$work")" "$(dirname "$archive")"
mkdir "$work" "$archive"
mkdir -p "$work/Libs/population_dynamics" "$work/Libs/configs" "$work/tmp" "$work/cache"
cp "$0" "$archive/executed_launcher.sbatch"
for name in __init__.py provenance.py resource_probe.py appworld_reference.py appworld_collaboration.py appworld_action_decoder.py appworld_local_actor.py appworld_benchmark_protocol.py appworld_benchmark_execution.py appworld_benchmark_runner.py appworld_benchmark_transport.py appworld_benchmark_development.py appworld_benchmark_model_probe.py appworld_benchmark_model_runtime.py; do
    cp "$repo/Libs/population_dynamics/$name" "$work/Libs/population_dynamics/"
done
cp "$repo/Libs/configs/neural_population_appworld_benchmark.yaml" "$repo/Libs/configs/neural_population_appworld_local_actor.yaml" "$work/Libs/configs/"
cp "$repo/Output/neural_population_dynamics/agentic_extension/local_model_intake/qwen3_coder_30b_a3b_b2cff646/source_metadata.json" "$work/source_metadata.json"
cp "$package_receipt" "$work/package_environment.json"
cp "$software_validation" "$work/software_validation.json"
export PYTHONPATH="$work/Libs" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$work/tmp" XDG_CACHE_HOME="$work/cache" HF_HOME="$work/cache/huggingface"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
# Keep Slurm's CUDA_VISIBLE_DEVICES unchanged. No environment installation,
# model download, benchmark task, or historical actor execution occurs here.
finish() {
    result=$?
    trap - EXIT
    mkdir -p "$archive/source"
    cp -a "$work/Libs" "$archive/source/"
    cp "$work/source_metadata.json" "$work/package_environment.json" "$work/software_validation.json" "$archive/"
    if [ -d "$work/result" ]; then cp -a "$work/result" "$archive/result"; fi
    for name in qualification.log source_binding.log; do
        if [ -f "$work/$name" ]; then cp "$work/$name" "$archive/"; fi
    done
    printf '%s\n' "$work" > "$archive/runtime_path.txt"
    printf '%s\n' "$result" > "$archive/exit_code.txt"
    exit "$result"
}
trap finish EXIT
cd "$work"
"$model_python" - "$work" "$archive/executed_launcher.sbatch" > "$work/source_binding.log" 2>&1 <<'PYBIND'
import hashlib
import json
from pathlib import Path
import sys
work, executed_launcher = map(Path, sys.argv[1:])
validation = json.loads((work / "software_validation.json").read_text())
for key in ("resource_exit_code", "shell_syntax_exit_code", "pytest_exit_code"):
    if type(validation.get(key)) is not int or validation[key] != 0:
        raise ValueError("MODEL_PROBE_SOFTWARE_VALIDATION_NOT_PASSED:" + key)
expected = validation["source_sha256"]
paths = list((work / "Libs/population_dynamics").iterdir())
paths.append(work / "Libs/configs/neural_population_appworld_benchmark.yaml")
for path in paths:
    relative = path.relative_to(work).as_posix()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected.get(relative):
        raise ValueError("MODEL_PROBE_TESTED_SOURCE_MISMATCH:" + relative)
launcher_key = "Libs/scripts/qualify_appworld_model.sbatch"
if hashlib.sha256(executed_launcher.read_bytes()).hexdigest() != expected.get(launcher_key):
    raise ValueError("MODEL_PROBE_TESTED_LAUNCHER_MISMATCH")
print("PASS: executable source and canonical configuration match software test job", validation["job_id"])
PYBIND
"$model_python" -m population_dynamics.appworld_benchmark_model_runtime \
    --config "$work/Libs/configs/neural_population_appworld_benchmark.yaml" \
    --model-directory "$model_directory" --source-metadata "$work/source_metadata.json" \
    --snapshot-config "$work/Libs/configs/neural_population_appworld_local_actor.yaml" \
    --runtime-receipt "$work/package_environment.json" --output "$work/result" \
    > "$work/qualification.log" 2>&1
