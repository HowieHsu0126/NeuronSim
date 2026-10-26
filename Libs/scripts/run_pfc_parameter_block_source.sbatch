#!/bin/bash
#SBATCH --job-name=NrnSim_PfcBlockSource
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim-PfcBlockSource-%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/opt/conda/bin/python3 -B - <<'PY'
from pathlib import Path
import hashlib,json,os,shutil,socket,subprocess,sys,tempfile
repo=Path('/workspace/NeuronSim')
assert socket.gethostname().split('.')[0]=='cpu03' and not os.environ.get('SLURM_JOB_GPUS')
base=repo/'Output/neural_population_dynamics/benchmark/pfc_parameter_design'
prepared=base/'parameter_block_prepare_613107'
plan_root=Path(os.environ['NEURONSIM_PFC_BLOCK_PLAN'])
index=int(os.environ['SLURM_ARRAY_TASK_ID']);assert 0<=index<100
source=base/'task_source_prepare_610582'
delivery=json.loads((source/'delivery.json').read_text())
assert delivery['status']=='STAGED_ACCEPTED_PFC_SOURCE_EXECUTOR'
scratch=Path(tempfile.mkdtemp(prefix='neuronsim-pfc-task-launch-'+os.environ['SLURM_JOB_ID']+'.',dir='/tmp'))
try:
 for row in delivery['files']:
  name=Path(row['path']);assert not name.is_absolute() and '..' not in name.parts
  p=source/name;target=scratch/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  assert target.stat().st_size==row['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
 manifest=json.loads((plan_root/'delivery.json').read_text())
 for row in manifest['files']:
  if not row['path'].startswith('Libs/'):continue
  p=plan_root/row['path'];target=scratch/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
  else:shutil.copy2(p,target)
  assert target.stat().st_size==row['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
 sys.path.insert(0,str(scratch/'Libs'))
 from population_dynamics.pfc_parameter_block_execution import load_plan,recipe_path,source_path,qualification_path
 rows=load_plan(plan_root,require_release=True);task=rows[index]
 assert int(task['task_index'])==index
 config=recipe_path(task,prepared);qualification=qualification_path(base,int(task['circuit_index']))
 env=dict(os.environ,NEURONSIM_PROJECT_ROOT=str(scratch),NEURONSIM_PFC_SOURCE_BACKEND='numpy',
          NEURONSIM_PFC_SOURCE_CONFIG=str(config),NEURONSIM_PFC_SOURCE_QUALIFICATION=str(qualification),
          NEURONSIM_PFC_SOURCE_CASE_ID=task['episode_id'],
          NEURONSIM_PFC_OUTPUT_BASE=str(source_path(base,task)))
 env.pop('SLURM_ARRAY_TASK_ID',None)
 env.pop('NEURONSIM_PFC_SOURCE_CASES_JSON',None)
 result=subprocess.run([sys.executable,'-B',str(scratch/'Libs/scripts/run_pfc_source_cases.py')],env=env,cwd=scratch)
 code=result.returncode
finally:shutil.rmtree(scratch)
raise SystemExit(code)
PY
