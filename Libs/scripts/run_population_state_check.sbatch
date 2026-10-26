#!/bin/bash
#SBATCH --job-name=NrnSim_population_episode
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-population-episode-%j.log
set -euo pipefail
repo=/workspace/NeuronSim
export NRNSIM_CHECK_DEVICE=${1:-cpu}
case "$NRNSIM_CHECK_DEVICE" in
 cpu) export CUDA_VISIBLE_DEVICES=""; if [ "$SLURMD_NODENAME" != cpu03 ]; then exit 2; fi ;;
 cuda) if [ "$SLURMD_NODENAME" != gpu03 ] || [ -z "${SLURM_JOB_GPUS:-}" ]; then exit 2; fi ;;
 *) exit 2 ;;
esac
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 NVIDIA_TF32_OVERRIDE=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
work=$(mktemp -d "${TMPDIR:-/tmp/neuronsim}/neuronsim-population-episode-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf "$work"' EXIT
srun python3 - "$repo" "$work" <<'PREPARE'
import ast,hashlib,json,shutil,sys
from pathlib import Path
repo,work=map(Path,sys.argv[1:]);root=repo/'Libs/population_dynamics'
queue=['population_state_check','population_trajectory_training','resource_probe'];seen=set()
while queue:
    name=queue.pop(0)
    if name in seen:continue
    source=root/(name+'.py');tree=ast.parse(source.read_text());seen.add(name)
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.level==1:
            queue.extend([node.module] if node.module else [a.name for a in node.names])
for name in ['__init__',*sorted(seen)]:
    target=work/'Libs/population_dynamics'/(name+'.py');target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/(name+'.py'),target)
shutil.copy2(repo/'Libs/__init__.py',work/'Libs/__init__.py')
(work/'Libs/configs').mkdir()
shutil.copy2(repo/'Libs/configs/neural_population_state_trajectory.yaml',work/'Libs/configs/neural_population_state_trajectory.yaml')
base=repo/'Output/neural_population_dynamics/mechanistic_reference/spatial_connectivity/stimulus_trajectory_cohort/philox_development'
case='train_block_000_familiar_0';episode=base/'train'/case;fields=base/'train_fields'/case
report=json.loads((episode/'report.json').read_text());view=json.loads((fields/'summary.json').read_text())
rows=[]
for source,relative,expected in [(episode/'report.json','inputs/episode/report.json',None),
    (episode/'spikes.npz','inputs/episode/spikes.npz',report['artifacts']['spikes.npz']),
    (episode/'initial_parameters.npz','inputs/episode/initial_parameters.npz',report['artifacts']['initial_parameters.npz']),
    (fields/'summary.json','inputs/fields/summary.json',None),
    (fields/'spatial_fields.npz','inputs/fields/spatial_fields.npz',view['artifacts']['spatial_fields.npz'])]:
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if expected is not None and digest!=expected:raise ValueError('NATIVE_INPUT_CHANGED:'+str(source))
    target=work/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('STAGED_INPUT_CHANGED:'+relative)
    rows.append(dict(source=str(source),path=relative,sha256=digest,bytes=source.stat().st_size))
(work/'inputs.json').write_text(json.dumps(rows,indent=2)+'\n')
guard=repo/'Output/neural_population_dynamics/multicircuit/ca3_source_reproduction/native_formation_retrieval/analysis/observed_reconciliation/run_579393/run_guarded.py'
text=guard.read_text().replace('required_memory_gib=1','required_memory_gib=32')
(work/'run_guarded.py').write_text(text)
shutil.copy2(repo/'Libs/scripts/run_population_state_check.sbatch',work/'launcher.sbatch')
(work/'phases').mkdir();(work/'scratch').mkdir()
PREPARE
export PYTHONPATH="$work/Libs" TMPDIR="$work/scratch"
args=(--episode inputs/episode --fields inputs/fields --config Libs/configs/neural_population_state_trajectory.yaml --output results --device "$NRNSIM_CHECK_DEVICE" --seed 41)
set +e
srun python3 "$work/run_guarded.py" update python3 -m population_dynamics.population_state_check "${args[@]}"
result=$?
if [ "$result" = 0 ]; then
 srun python3 "$work/run_guarded.py" replay python3 -m population_dynamics.population_state_check "${args[@]}" --replay
 result=$?
fi
set -e
srun python3 - "$repo" "$work" "$result" <<'ARCHIVE'
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,shutil,sys
repo,work=map(Path,sys.argv[1:3]);code=int(sys.argv[3])
target=repo/'Output/neural_population_dynamics/model/full_episode_check'/('run_'+os.environ['SLURM_JOB_ID'])
target.mkdir(parents=True,exist_ok=False);rows=[]
for source in sorted(work.rglob('*')):
    relative=source.relative_to(work)
    if not source.is_file() or set(relative.parts)&{'inputs','scratch','__pycache__','.pytest_cache'}:continue
    destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(source.read_bytes()).hexdigest();shutil.copy2(source,destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:raise ValueError('ARCHIVE_DIFFERS:'+str(relative))
    rows.append(dict(path=str(relative),sha256=digest,bytes=source.stat().st_size))
record=dict(status='PASS' if code==0 else 'FAILED',exit_code=code,job_id=os.environ['SLURM_JOB_ID'],
    node=os.environ['SLURMD_NODENAME'],device=os.environ['NRNSIM_CHECK_DEVICE'],files=rows,
    core_neurons=100000,episode_ms=6000,scientific_claim_qualified=False,formal_training_completed=False,
    finished_utc=datetime.now(timezone.utc).isoformat())
(target/'delivery.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
ARCHIVE
exit "$result"
