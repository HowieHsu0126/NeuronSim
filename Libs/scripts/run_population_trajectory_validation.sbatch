#!/bin/bash
#SBATCH --job-name=NrnSim_population_tests
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim/neuronsim-population-tests-%j.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
repo=/workspace/NeuronSim
work=$(mktemp -d "${TMPDIR:-/tmp/neuronsim}/neuronsim-population-tests-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf "$work"' EXIT
srun python3 - "$repo" "$work" <<'PREPARE'
import ast,json,shutil,sys
from pathlib import Path
repo,work=map(Path,sys.argv[1:]); root=repo/'Libs/population_dynamics'
starts=['test_population_trajectory_training','test_renewal_trajectory_training',
        'test_poisson_trajectory_training','test_population_state',
        'test_population_state_evaluation','test_population_trajectory_evaluation',
        'test_population_forecast_unit_metrics_replay']
queue=starts[:];seen=set()
while queue:
    name=queue.pop(0)
    if name in seen:continue
    source=root/(name+'.py')
    tree=ast.parse(source.read_text());seen.add(name)
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.level==1:
            queue.extend([node.module] if node.module else [a.name for a in node.names])
for name in ['__init__',*sorted(seen)]:
    target=work/'Libs/population_dynamics'/(name+'.py');target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/(name+'.py'),target)
for name in ['neural_population_state_trajectory','neural_population_renewal_trajectory','neural_population_dynamics',
             'neural_population_matched_forecast']:
    target=work/'Libs/configs'/(name+'.yaml');target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(repo/'Libs/configs'/(name+'.yaml'),target)
shutil.copy2(repo/'Libs/__init__.py',work/'Libs/__init__.py')
guard=repo/'Output/neural_population_dynamics/multicircuit/ca3_source_reproduction/native_formation_retrieval/analysis/observed_reconciliation/run_579393/run_guarded.py'
shutil.copy2(guard,work/'run_guarded.py')
# The guard imports the existing resource preflight utility.
shutil.copy2(root/'resource_probe.py',work/'Libs/population_dynamics/resource_probe.py')
shutil.copy2(repo/'Libs/scripts/run_population_trajectory_validation.sbatch',work/'launcher.sbatch')
(work/'phases').mkdir();(work/'scratch').mkdir()
(work/'test_command.json').write_text(json.dumps(['python3','-m','pytest','-q',
    *['Libs/population_dynamics/'+n+'.py' for n in starts],'--junitxml=tests.xml']))
PREPARE
export PYTHONPATH="$work/Libs" TMPDIR="$work/scratch"
set +e
srun python3 "$work/run_guarded.py" training python3 -m pytest -q   Libs/population_dynamics/test_population_trajectory_training.py   Libs/population_dynamics/test_renewal_trajectory_training.py   Libs/population_dynamics/test_poisson_trajectory_training.py   Libs/population_dynamics/test_population_state.py \
  Libs/population_dynamics/test_population_state_evaluation.py \
  Libs/population_dynamics/test_population_trajectory_evaluation.py \
  Libs/population_dynamics/test_population_forecast_unit_metrics_replay.py --junitxml=tests.xml
result=$?
set -e
srun python3 - "$repo" "$work" "$result" <<'ARCHIVE'
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,shutil,sys,xml.etree.ElementTree as ET
repo,work=map(Path,sys.argv[1:3]);code=int(sys.argv[3])
target=repo/'Output/neural_population_dynamics/model/trajectory_software'/('run_'+os.environ['SLURM_JOB_ID'])
target.mkdir(parents=True,exist_ok=False)
rows=[]
for source in sorted(work.rglob('*')):
    relative=source.relative_to(work)
    if not source.is_file() or set(relative.parts)&{'scratch','__pycache__','.pytest_cache'}:continue
    destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(source.read_bytes()).hexdigest();shutil.copy2(source,destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:raise ValueError('ARCHIVE_DIFFERS:'+str(relative))
    rows.append(dict(path=str(relative),sha256=digest,bytes=source.stat().st_size))
suites=ET.parse(work/'tests.xml').getroot().findall('testsuite') if (work/'tests.xml').is_file() else []
tests={name:sum(int(s.attrib.get(name,0)) for s in suites) for name in ('tests','failures','errors','skipped')}
record=dict(status='PASS' if code==0 else 'FAILED',exit_code=code,job_id=os.environ['SLURM_JOB_ID'],
    node=os.environ['SLURMD_NODENAME'],tests=tests,files=rows,scientific_claim_qualified=False,
    finished_utc=datetime.now(timezone.utc).isoformat())
(target/'delivery.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
ARCHIVE
exit "$result"
