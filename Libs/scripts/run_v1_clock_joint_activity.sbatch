#!/bin/bash
#SBATCH --job-name=NrnSim_V1ClockJointActivity
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_V1ClockJointActivity-%j.log
set -euo pipefail
[[ $# == 4 || $# == 5 ]] || exit 2
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${NRNSIM_BENCHMARK_PYTHON:?Set benchmark Python}" -B - "$@" <<'PYTHON'
from pathlib import Path
import hashlib,json,os,resource,shutil,subprocess,sys,tempfile,time,traceback
software,cases,submissions,destination=map(Path,sys.argv[1:5]);selected=sys.argv[5] if len(sys.argv)==6 else None
if os.environ['SLURMD_NODENAME'] not in ('cpu03','cpu04') or os.environ.get('SLURM_JOB_GPUS'):
 raise ValueError('JOINT_CLOCK_CPU_PLACEMENT_REQUIRED')
keys=[row['identity']['episode_id'] for row in json.loads(cases.read_text())['cases']]
if selected is not None:
 if selected not in keys:raise ValueError('JOINT_CLOCK_UNKNOWN_CASE')
 keys=[selected]
started=time.monotonic();code=1
receipt=dict(job_id=os.environ['SLURM_JOB_ID'],node=os.environ['SLURMD_NODENAME'],software=str(software),cases=str(cases),submissions=str(submissions),selected_episode=selected,new_source_simulations=0,production_episodes=0,scientific_qualification=False)
def digest(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
with tempfile.TemporaryDirectory(prefix='neuronsim-clock-joint-') as temp:
 work=Path(temp);output=work/'analysis'
 try:
  sys.path.insert(0,str(software/'Libs'))
  from population_dynamics.resource_probe import probe
  if probe(work/'resources.json',required_memory_gib=4)['status']!='PASS':raise RuntimeError('JOINT_CLOCK_RESOURCE_STOP')
  for key in keys:
   original=destination/'cases'/key
   if original.exists():
    report=json.loads((original/'report.json').read_text())
    target=output/'cases'/key;target.mkdir(parents=True)
    for name in ('report.json',*report['artifacts']):
     if Path(name).name!=name:raise ValueError('JOINT_CLOCK_ARTIFACT_PATH_INVALID')
     shutil.copy2(original/name,target/name)
  command=[sys.executable,'-B','-m','population_dynamics.v1_clock_joint_activity','--cases',str(cases),'--submissions',str(submissions),'--output',str(output)]
  if selected is not None:command+=['--episode-id',selected]
  receipt['command']=command
  with (work/'analysis.log').open('w') as log:
   code=subprocess.run(command,cwd=work,env=dict(os.environ,PYTHONPATH=str(software/'Libs')),stdout=log,stderr=subprocess.STDOUT).returncode
 except BaseException as error:
  receipt.update(error_type=type(error).__name__,error=str(error));(work/'failure.txt').write_text(traceback.format_exc());code=1
 finally:
  destination.mkdir(parents=True,exist_ok=True)
  for key in keys:
   for name in ('report.json','full.npz','pre_command.npz','command_response.npz'):
    path=output/'cases'/key/name
    if not path.exists():continue
    target=destination/'cases'/key/name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
     if digest(target)!=digest(path):raise ValueError('JOINT_CLOCK_EXISTING_OUTPUT_DIFFERS')
    else:shutil.copy2(path,target)
  if (output/'report.json').exists():
   target=destination/'report.json'
   if target.exists():raise ValueError('JOINT_CLOCK_PANEL_ALREADY_COMMITTED')
   shutil.copy2(output/'report.json',target)
  run=destination/'runs'/os.environ['SLURM_JOB_ID'];run.mkdir(parents=True,exist_ok=False)
  for name in ('analysis.log','failure.txt','resources.json'):
   if (work/name).exists():shutil.copy2(work/name,run/name)
  receipt.update(exit_code=code,elapsed_seconds=time.monotonic()-started,child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
  (run/'report.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
sys.exit(code)
PYTHON
