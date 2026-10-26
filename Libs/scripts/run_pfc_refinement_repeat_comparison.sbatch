#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_RepeatReadout
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=none
#SBATCH --mem=4G
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-repeat-readout-%j.out
#SBATCH --error=/tmp/neuronsim-repeat-readout-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export NRNSIM_SUBMITTED_BATCH="$0"
/opt/conda/bin/python3 -B - <<'PY_REPEAT'
from pathlib import Path
import datetime,hashlib,json,os,shutil,socket,subprocess,sys,tempfile,time
import yaml
repo=Path(os.environ.get('NEURONSIM_PROJECT_ROOT','/workspace/NeuronSim'))
job=os.environ['SLURM_JOB_ID']
if 'SLURM_ARRAY_TASK_ID' in os.environ:job=os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID']
mode=os.environ['NEURONSIM_PFC_REPEAT_MODE'];assert mode in ('validate','block','compare')
family=Path('Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/refinement_repeat')
archive=repo/family/('comparison_validation' if mode=='validate' else 'block_readouts' if mode=='block' else 'comparison')/('run_'+job)
config_name='Libs/configs/neural_population_pfc_refinement_repeat_comparison.yaml'
protocol_name='Libs/configs/neural_population_pfc_refinement_repeat.yaml'
prior=repo/'Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/coupled_precision/run_592283'
assert socket.gethostname().split('.')[0]=='cpu04' and not archive.exists() and not os.environ.get('SLURM_JOB_GPUS')
work=Path(tempfile.mkdtemp(prefix='neuronsim-repeat-readout-'+job+'.',dir='/tmp'))
inputs=[];commands=[];code=1;status='FAILED';delivered=False;started=time.perf_counter()
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(source,relative,expected=None):
 target=work/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);sha=digest(target)
 if expected is not None and sha!=expected:raise RuntimeError('STAGED_INPUT_MISMATCH:'+str(relative))
 inputs.append(dict(source=str(source),path=str(relative),sha256=sha))
def stage_report(reference,names):
 root=Path(reference['path']);copy(repo/root/'report.json',root/'report.json',reference['report_sha256'])
 r=json.loads((work/root/'report.json').read_text())
 for name in names:copy(repo/root/name,root/name,r['outputs'][name]['sha256'])
 return r
def run(label,command):
 begin=time.perf_counter()
 with (work/(label+'.stdout')).open('w') as out,(work/(label+'.stderr')).open('w') as err:
  result=subprocess.run(command,cwd=work,env=env,stdout=out,stderr=err)
 commands.append(dict(phase=label,command=command,exit_code=result.returncode,wall_seconds=time.perf_counter()-begin))
 print(json.dumps(commands[-1]),flush=True)
 if result.returncode:raise RuntimeError('REPEAT_'+label.upper()+'_FAILED:'+str(result.returncode))
try:
 snapshot=os.environ.get('NEURONSIM_PFC_REPEAT_VALIDATION')
 if mode=='validate':
  for row in json.loads((prior/'delivery.json').read_text())['files']:
   if row['path'].startswith('Libs/') and row['path'].endswith('.py'):copy(prior/row['path'],row['path'],row['sha256'])
  for name in ['Libs/population_dynamics/pfc_coupled_clock_execution.py','Libs/population_dynamics/pfc_refinement_repeat_comparison.py','Libs/population_dynamics/test_pfc_refinement_repeat_comparison.py','Libs/population_dynamics/resource_probe.py',config_name,protocol_name]:
   copy(repo/name,name,digest(repo/name))
 else:
  assert snapshot,'ACCEPTED_VALIDATION_SNAPSHOT_REQUIRED'
  snapshot=repo/snapshot
  tested=json.loads((snapshot/'report.json').read_text())
  assert tested['exit_code']==0 and all(c['exit_code']==0 for c in tested['commands'])
  assert {c['phase'] for c in tested['commands']} >= {'tests','analysis'}
  for row in json.loads((snapshot/'delivery.json').read_text())['files']:
   if row['path'].startswith('Libs/') and row['path'].endswith(('.py','.yaml')):copy(snapshot/row['path'],row['path'],row['sha256'])
  commands.append(dict(phase='tests_reused',exit_code=0,archive=str(snapshot),report_sha256=digest(snapshot/'report.json')))
 copy(Path(os.environ['NRNSIM_SUBMITTED_BATCH']),'launcher.sbatch')
 assert digest(work/config_name)==os.environ['NEURONSIM_PFC_REPEAT_CONFIG_SHA256'],'FROZEN_COMPARISON_CONFIG_CHANGED'
 sys.path.insert(0,str(work/'Libs'))
 from population_dynamics.resource_probe import probe
 health=probe(work/'resource_preflight.json',4);assert health['status']=='PASS',health['reasons']
 settings=yaml.safe_load((work/config_name).read_text())
 if mode in ('validate','compare'):
  original=stage_report(settings['original'],['endpoint_precision_comparison.csv','coupled_field_rates.npz','coupled_responses.npz','coupled_count_cdfs.npz','coupled_block_scores.npz'])
  stage_report(original['settings']['budget'],['frozen_numerical_margins.csv'])
 selection=[]
 if mode=='block':
  index=int(os.environ.get('SLURM_ARRAY_TASK_ID',os.environ.get('NEURONSIM_PFC_REPEAT_BLOCK_INDEX','-1')))
  assert 0<=index<32
  entry=settings['source_plan'][index];source=Path(entry['archive'])
  copy(repo/source/'delivery.json',source/'delivery.json')
  delivery=json.loads((work/source/'delivery.json').read_text())
  assert delivery['status']=='COMPLETE_COUPLED_REFINEMENT_REPEAT_BLOCK' and str(delivery['job_id'])==entry['job']
  for row in delivery['files']:
   if row['path']=='report.json' or row['path'].startswith('source_cases/'):
    copy(repo/source/row['path'],source/row['path'],row['sha256'])
  selection=['--source-rng-seed',str(entry['source_rng_seed'])]
 if mode=='compare':
  source_plan=Path(os.environ['NEURONSIM_PFC_REPEAT_READOUT_PLAN'])
  copy(repo/source_plan,Path('readout_plan.json'))
  from population_dynamics.pfc_refinement_repeat_comparison import index_from_plan
  index=index_from_plan(repo,json.loads((work/'readout_plan.json').read_text()),digest(work/config_name))
  (work/'readout_index.json').write_text(json.dumps(index,indent=2)+'\n')
  assert [r['source_rng_seed'] for r in index['blocks']]==list(range(95001,95033))
  for entry in index['blocks']:stage_report(entry,['block_readout.npz'])
  selection=['--readout-index',str(work/'readout_index.json')]
 env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(work),str(work/'Libs')]),TMPDIR=str(work))
 if mode=='validate':
  run('tests',[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',
      'Libs/population_dynamics/test_pfc_refinement_repeat_comparison.py',
      '--basetemp='+str(work/'test_tmp'),'--junitxml='+str(work/'tests.xml')])
 run('analysis',[sys.executable,'-B','-m','Libs.population_dynamics.pfc_refinement_repeat_comparison',
     '--config',str(work/config_name),'--project-root',str(work),'--output',str(work/'analysis'),'--mode',mode,*selection])
 status='COMPLETE_PFC_REFINEMENT_REPEAT_'+mode.upper();code=0
except Exception as exc:
 (work/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,error=str(exc)),indent=2)+'\n')
finally:
 (work/'report.json').write_text(json.dumps(dict(status=status,exit_code=code,job_id=job,mode=mode,
     inputs=inputs,commands=commands,source_simulations=0,production_episodes=0,confirmation_execution_released=False,
     wall_seconds=time.perf_counter()-started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
 try:
  archive.mkdir(parents=True,exist_ok=False);files=[]
  for source in sorted(work.rglob('*')):
   if not source.is_file():continue
   relative=source.relative_to(work)
   if relative.parts[0] in ('Output','test_tmp'):continue
   target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);sha=digest(source)
   if digest(target)!=sha:raise RuntimeError('DELIVERY_MISMATCH:'+str(relative))
   files.append(dict(path=str(relative),bytes=source.stat().st_size,sha256=sha))
  (archive/'delivery.json').write_text(json.dumps(dict(status=status,job_id=job,verified=True,files=files),indent=2)+'\n')
  delivered=True;print(json.dumps(dict(status=status,exit_code=code,archive=str(archive))),flush=True)
 finally:
  if delivered:shutil.rmtree(work)
  else:print(json.dumps(dict(status='COPYBACK_FAILED_SCRATCH_PRESERVED',work=str(work))),file=sys.stderr,flush=True)
raise SystemExit(code)
PY_REPEAT
