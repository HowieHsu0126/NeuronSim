#!/bin/bash
#SBATCH --job-name=NrnSim_mesoscopic_author
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-mesoscopic-author-%j.log
set -euo pipefail
repo=/workspace/NeuronSim
# The two-second author example evolves only eight population states.
# CPU04 is permitted for this bounded source implementation check.
if [ -n "${SLURM_JOB_GPUS:-}" ]; then exit 2; fi
case "$SLURMD_NODENAME:$SLURM_JOB_PARTITION" in
 cpu03:gpu|cpu04:cpu) ;;
 *) exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
work=$(mktemp -d "/tmp/neuronsim/neuronsim-mesoscopic-author-${SLURM_JOB_ID}.XXXXXX")
trap 'if [ -f "$work/archive_complete" ]; then rm -rf "$work"; else printf "Recovery scratch: %s\n" "$work" >&2; fi' EXIT
srun python3 - "$repo" "$work" <<'PREPARE'
from pathlib import Path
import json,shutil,sys
repo,work=map(Path,sys.argv[1:]);files=['Libs/__init__.py','Libs/population_dynamics/__init__.py',
 'Libs/population_dynamics/mesoscopic_author_reference.py','Libs/population_dynamics/provenance.py',
 'Libs/population_dynamics/resource_probe.py','Libs/configs/neural_population_mesoscopic_reference.yaml',
 'Libs/scripts/run_mesoscopic_author_reference.sbatch']
files+=['Libs/external/mesopopdyn_microcircuit/'+name for name in
 ['README.md','single_run/sim4.py','single_run/potjans.py','single_run/PDmodel_mesopython.py']]
for name in files:
 target=work/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
(work/'scratch').mkdir();(work/'phases').mkdir()
source=repo/'Output/neural_population_dynamics/multicircuit/ca3_source_reproduction/native_formation_retrieval/analysis/observed_reconciliation/run_579393/run_guarded.py'
(work/'run_guarded.py').write_text(source.read_text().replace('required_memory_gib=1','required_memory_gib=4'))
PREPARE
export PYTHONPATH="$work/Libs" TMPDIR="$work/scratch" NUMBA_CACHE_DIR="$work/scratch/numba"
common=(--source Libs/external/mesopopdyn_microcircuit --config Libs/configs/neural_population_mesoscopic_reference.yaml)
result=0
for phase in coupled replay noise uncoupled; do
 args=("${common[@]}" --output "results/$phase")
 case "$phase" in
 coupled) args+=(--condition coupled --noise-seed 12345) ;;
 replay) args+=(--condition coupled --noise-seed 12345 --reference results/coupled) ;;
 noise) args+=(--condition coupled --noise-seed 12346 --reference results/coupled) ;;
 uncoupled) args+=(--condition uncoupled --noise-seed 12345 --reference results/coupled) ;;
 esac
 set +e
 srun python3 "$work/run_guarded.py" "$phase" python3 -m population_dynamics.mesoscopic_author_reference "${args[@]}"
 result=$?
 set -e
 if [ "$result" != 0 ]; then break; fi
done
srun python3 - "$repo" "$work" "$result" <<'ARCHIVE'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,sys
repo,work=map(Path,sys.argv[1:3]);code=int(sys.argv[3]);target=repo/'Output/neural_population_dynamics/baselines/mesoscopic_author_reference'/('run_'+os.environ['SLURM_JOB_ID'])
target.mkdir(parents=True,exist_ok=False);rows=[]
for source in sorted(work.rglob('*')):
 relative=source.relative_to(work)
 if not source.is_file() or set(relative.parts)&{'scratch','__pycache__','.pytest_cache'}:continue
 value=hashlib.sha256(source.read_bytes()).hexdigest();destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
 if hashlib.sha256(destination.read_bytes()).hexdigest()!=value:raise ValueError('ARCHIVE_DIFFERS:'+str(relative))
 rows.append(dict(path=str(relative),bytes=source.stat().st_size,sha256=value))
reports={name:json.loads((target/'results'/name/'run.json').read_text()) for name in ['coupled','replay','noise','uncoupled'] if (target/'results'/name/'run.json').is_file()}
record=dict(status='PASS' if code==0 else 'FAILED',exit_code=code,job_id=os.environ['SLURM_JOB_ID'],node=os.environ['SLURMD_NODENAME'],files=rows,
 outcomes={name:dict(status=value['status'],reference_check=value.get('reference_check')) for name,value in reports.items()},
 represented_units=100000,population_outputs_only=True,production_episodes_counted=0,scientific_claim_qualified=False,source_fidelity_qualified=False,
 license='unresolved; internal evaluation only',finished_utc=datetime.now(timezone.utc).isoformat())
(target/'delivery.json').write_text(json.dumps(record,indent=2)+'\n');(work/'archive_complete').write_text(str(target));print(json.dumps(record))
ARCHIVE
exit "$result"
