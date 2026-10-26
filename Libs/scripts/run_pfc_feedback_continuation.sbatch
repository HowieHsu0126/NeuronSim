#!/bin/bash
#SBATCH --job-name=NrnSim_FeedbackContinuation
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-feedback-%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
"$NEURONSIM_PYTHON" -B - <<'PY_CONTINUE'
from pathlib import Path
import hashlib,json,os,shutil,socket,subprocess,sys,tempfile,time,traceback
repo=Path(os.environ['NEURONSIM_PROJECT_ROOT'])
assert (repo/'PROJECT_PREFIX').read_text().strip()=='NrnSim'
assert socket.gethostname().split('.')[0]=='cpu03' and not os.environ.get('SLURM_JOB_GPUS')
job=os.environ['SLURM_JOB_ID']
family=Path('Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/first_passage_fine_suppression')
preparation=Path(os.environ.get('NEURONSIM_FEEDBACK_PREPARATION',str(repo/family/'feedback_continuation_software/validation_609322')))
case_name=os.environ.get('NEURONSIM_FEEDBACK_CASE','half_A')
assert case_name in ('half_A','quarter_A')
original_job=608505 if case_name=='half_A' else 609431
work=Path(tempfile.mkdtemp(prefix='neuronsim-feedback-'+job+'.',dir='/tmp'))
output=repo/family/'feedback_continuation_execute'/('run_'+job)
assert not output.exists();output.mkdir(parents=True)
record=dict(status='FAILED_PFC_FEEDBACK_CONTINUATION_EXECUTION',job_id=job,node=socket.gethostname(),
    original_failed_job=original_job,case_name=case_name,production_episodes=0,scientific_fidelity_qualified=False)
started=time.perf_counter();code=1

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def copy(origin,target,expected=None):
 destination=work/target;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,destination)
 if expected is not None:assert destination.stat().st_size==expected['bytes'] and digest(destination)==expected['sha256'],target
 return destination

try:
 manifest=json.loads((preparation/'caller_software_manifest.json').read_text())
 assert manifest['caller_tests_passed']>=1 and manifest['replay_tests_passed']==22
 if case_name=='quarter_A':assert manifest['caller_tests_passed']==2 and 609431 in manifest['original_failed_jobs']
 for row in manifest['files']:copy(preparation/'software'/row['path'],row['path'],row)
 sys.path.insert(0,str(work))
 from Libs.population_dynamics.resource_probe import probe
 health=probe(work/'resource_preflight.json',4)
 assert health['status']=='PASS',health['reasons']
 jobs=subprocess.check_output(['squeue','-u',os.environ['USER'],'-t','RUNNING','-h','-o','%i %j %N'],text=True)
 source_jobs=[line for line in jobs.splitlines() if 'cpu03' in line and any(name in line for name in ('NrnSim_ProfileKernelSources','NrnSim_QuarterSuppression','NrnSim_FeedbackContinuation','NrnSim_PassageSources','NrnSim_PassageRecovery'))]
 record['source_concurrency_at_admission']=source_jobs
 assert len(source_jobs)<=2,'PFC_SOURCE_CONCURRENCY_ADMISSION_BLOCKED'
 original=repo/family/'execute'/('run_'+str(original_job))
 archived=json.loads((original/'delivery.json').read_text());entries={row['path']:row for row in archived['files']}
 copy(original/'delivery.json','saved/delivery.json')
 names=['report.json' if case_name=='half_A' else 'execution.json']+[f'validation/{case_name}/{branch}/{name}' for branch in ('witness','suppression')
     for name in ('report.json','source_terminal.npz','source_raw.npz','native_prefix_rng.npz')]
 for name in names:copy(original/name,'saved/'+name,entries[name])
 config=Path('Libs/configs/neural_population_pfc_first_passage_fine_suppression.yaml')
 if case_name=='quarter_A':
  import yaml
  settings=yaml.safe_load((work/config).read_text())
  rows=[row for row in settings['cases'] if row['name']==case_name];assert len(rows)==1
  source=rows[0]['source'];origin=repo/source['archive']
  copy(origin/'delivery.json',str(Path(source['archive'])/'delivery.json'))
  assert digest(work/source['archive']/'delivery.json')==source['delivery_sha256']
  receipt=json.loads((work/source['archive']/'delivery.json').read_text())
  bound={row['path']:row for row in receipt['files']}
  for name in ('config.yaml','report.json','initial_voltage.npy'):
   relative=str(Path(source['case_path'])/name)
   copy(origin/relative,str(Path(source['archive'])/relative),bound[relative])
 command=[sys.executable,'-B','-m','Libs.population_dynamics.pfc_first_passage_suppression',
     '--config',str(config),'--output','continuation','--continue-feedback-from','saved','--case-name',case_name]
 with (work/'continuation.stdout').open('w') as out,(work/'continuation.stderr').open('w') as err:
  result=subprocess.run(command,cwd=work,stdout=out,stderr=err,env=dict(os.environ,PYTHONPATH=str(work)))
 record.update(command=command,child_exit_code=result.returncode)
 assert result.returncode==0,'FEEDBACK_CONTINUATION_CHILD_FAILED:'+str(result.returncode)
 numerical=json.loads((work/'continuation/report.json').read_text())
 assert numerical['status']=='COMPLETE_PFC_DELAYED_FEEDBACK_DIAGNOSTIC'
 assert numerical['original_probability_gate_passed'] is False and numerical['original_probability_window_changed'] is False
 record.update(status='COMPLETE_PFC_FEEDBACK_CONTINUATION_EXECUTION',numerical_status=numerical['status']);code=0
except Exception as error:
 record.update(error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
finally:
 record.update(exit_code=code,wall_seconds=time.perf_counter()-started)
 (work/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
 files=[]
 for path in work.rglob('*'):
  relative=path.relative_to(work)
  if not path.is_file() or relative.parts[0] in ('Libs','saved'):continue
  target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
  sha=digest(path);assert digest(target)==sha
  files.append(dict(path=str(relative),bytes=path.stat().st_size,sha256=sha))
 shutil.copy2(preparation/'caller_software_manifest.json',output/'software_manifest.json')
 (output/'delivery.json').write_text(json.dumps(dict(record,files=files),indent=2)+'\n')
 shutil.rmtree(work)
print(json.dumps(record));raise SystemExit(code)
PY_CONTINUE
