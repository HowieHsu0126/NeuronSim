#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_AppWorldDevelopment
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
#SBATCH --output=/tmp/neuronsim-appworld-development-%j.out
#SBATCH --error=/tmp/neuronsim-appworld-development-%j.err
set -euo pipefail
repo=${1:?repository root required}
scratch_root=${2:?node-local scratch root required}
model_python=${3:?prepared model-environment Python required}
model_directory=${4:?existing node-local model directory required}
native_environment=${5:?native environment.json required}
split_manifest=${6:?native split_manifest.json required}
package_receipt=${7:?model package environment.json required}
model_qualification=${8:?passed model qualification.json required}
software_validation=${9:?pipeline software validation.json required}
agent=${10:?development agent required}
split=${11:?train or dev required}
membership_index=${12:?zero-based index among non-demo split members required}
seed=${13:?seed 41, 42 or 43 required}
shift 13
# Remaining arguments are explicit collaboration budgets, when applicable.
work="$scratch_root/neuronsim_appworld_benchmark/development_${SLURM_JOB_ID}"
archive="$repo/Output/neural_population_dynamics/agentic_extension/benchmark/native_development/run_${SLURM_JOB_ID}"
test "$(hostname -s)" = cpu04
fstype=$(findmnt -T "$scratch_root" -n -o FSTYPE)
case "$fstype" in nfs|nfs4|cifs) printf '%s\n' NODE_LOCAL_SCRATCH_REQUIRED >&2; exit 1;; esac
mkdir -p "$(dirname "$work")" "$(dirname "$archive")"
mkdir "$work" "$archive"
mkdir -p "$work/Libs/population_dynamics" "$work/Libs/configs" "$work/tmp" "$work/cache"
cp "$0" "$archive/executed_launcher.sbatch"
finish() {
    result=$?
    trap - EXIT
    mkdir -p "$archive/source"
    cp -a "$work/Libs" "$archive/source/"
    if [ -d "$work/result" ]; then cp -a "$work/result" "$archive/result"; fi
    for name in source_metadata.json package_environment.json native_environment.json split_manifest.json model_qualification.json software_validation.json selection.json source_binding.log pipeline.log; do
        if [ -f "$work/$name" ]; then cp "$work/$name" "$archive/"; fi
    done
    printf '%s\n' "$work" > "$archive/runtime_path.txt"
    printf '%s\n' "$result" > "$archive/exit_code.txt"
    exit "$result"
}
trap finish EXIT
for name in __init__.py provenance.py resource_probe.py appworld_reference.py appworld_collaboration.py appworld_action_decoder.py appworld_local_actor.py appworld_benchmark_protocol.py appworld_benchmark_execution.py appworld_benchmark_runner.py appworld_benchmark_transport.py appworld_benchmark_development.py appworld_benchmark_scoring.py appworld_benchmark_model_probe.py appworld_benchmark_model_runtime.py appworld_benchmark_collaboration.py appworld_benchmark_host.py appworld_benchmark_collaboration_development.py appworld_benchmark_pipeline.py; do
    cp "$repo/Libs/population_dynamics/$name" "$work/Libs/population_dynamics/"
done
cp "$repo/Libs/configs/neural_population_appworld_benchmark.yaml" "$repo/Libs/configs/neural_population_appworld_local_actor.yaml" "$work/Libs/configs/"
cp "$repo/Output/neural_population_dynamics/agentic_extension/local_model_intake/qwen3_coder_30b_a3b_b2cff646/source_metadata.json" "$work/source_metadata.json"
cp "$package_receipt" "$work/package_environment.json"
cp "$native_environment" "$work/native_environment.json"
cp "$split_manifest" "$work/split_manifest.json"
cp "$model_qualification" "$work/model_qualification.json"
cp "$software_validation" "$work/software_validation.json"
export PYTHONPATH="$work/Libs" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$work/tmp" XDG_CACHE_HOME="$work/cache" HF_HOME="$work/cache/huggingface"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
# The owner uses Slurm-visible GPUs. Its actor/scorer children receive no GPUs.
cd "$work"
"$model_python" - "$work" "$archive/executed_launcher.sbatch" "$split" "$membership_index" > "$work/source_binding.log" 2>&1 <<'PYBIND'
import hashlib
import json
from pathlib import Path
import sys
from population_dynamics.appworld_benchmark_development import load_benchmark_config
from population_dynamics.appworld_benchmark_pipeline import select_development_task

work, launcher = map(Path, sys.argv[1:3])
split, index_text = sys.argv[3:]
validation = json.loads((work / 'software_validation.json').read_text())
if validation.get('status') != 'PASS' or any(type(validation.get(key)) is not int or validation[key] != 0
        for key in ('resource_exit_code', 'shell_syntax_exit_code', 'pytest_exit_code')):
    raise ValueError('PIPELINE_SOFTWARE_VALIDATION_NOT_PASSED')
expected = validation['source_sha256']
paths = [*(work / 'Libs/population_dynamics').iterdir(), *(work / 'Libs/configs').iterdir()]
for path in paths:
    relative = path.relative_to(work).as_posix()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected.get(relative):
        raise ValueError('PIPELINE_TESTED_SOURCE_MISMATCH:' + relative)
if hashlib.sha256(launcher.read_bytes()).hexdigest() != expected.get('Libs/scripts/run_appworld_development.sbatch'):
    raise ValueError('PIPELINE_TESTED_LAUNCHER_MISMATCH')
config = load_benchmark_config(work / 'Libs/configs/neural_population_appworld_benchmark.yaml')
manifest = json.loads((work / 'split_manifest.json').read_text())
selection = select_development_task(config, manifest, split=split, index=int(index_text))
(work / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
(work / 'selected_task_id.txt').write_text(selection['task_id'] + '\n')
print('PASS: source matches pipeline validation', validation['job_id'])
print(json.dumps(selection))
PYBIND
read -r task_id < "$work/selected_task_id.txt"
"$model_python" -m population_dynamics.appworld_benchmark_pipeline \
    --config "$work/Libs/configs/neural_population_appworld_benchmark.yaml" \
    --native-environment "$work/native_environment.json" --split-manifest "$work/split_manifest.json" \
    --model-qualification "$work/model_qualification.json" --model-directory "$model_directory" \
    --source-metadata "$work/source_metadata.json" --snapshot-config "$work/Libs/configs/neural_population_appworld_local_actor.yaml" \
    --runtime-receipt "$work/package_environment.json" --output "$work/result" \
    --agent "$agent" --split "$split" --task-id "$task_id" --seed "$seed" "$@" \
    > "$work/pipeline.log" 2>&1
