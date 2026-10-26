#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_HistoryJointPrecision
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-history-joint-precision-%j.out
#SBATCH --error=/tmp/neuronsim-history-joint-precision-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export NEURONSIM_PRECISION_LAUNCHER="$0"
"$NEURONSIM_PYTHON" -B - <<'PY_PRECISION'
from pathlib import Path
import ast,datetime,hashlib,json,os,shutil,socket,subprocess,sys,tempfile,time
import xml.etree.ElementTree as ET
import yaml
repo=Path(os.environ['NEURONSIM_PROJECT_ROOT']);job=os.environ['SLURM_JOB_ID']
mode=os.environ.get('NEURONSIM_PRECISION_MODE','validate')
assert mode in ('validate','complete') and socket.gethostname().split('.')[0]=='cpu04'
assert (repo/'PROJECT_PREFIX').read_text().strip()=='NrnSim' and not os.environ.get('SLURM_JOB_GPUS')
config_name='Libs/configs/neural_population_pfc_conditional_history_precision.yaml'
base=Path('Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/conditional_history_joint_precision')
archive=repo/base/mode/('run_'+job);assert not archive.exists()
work=Path(tempfile.mkdtemp(prefix='neuronsim-history-joint-'+job+'.',dir='/tmp'))
status='FAILED_CONDITIONAL_HISTORY_JOINT_PRECISION';code=1;inputs=[];commands=[];delivered=False
start=time.perf_counter()
def digest(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def copy(source,name,expected=None,size=None):
 target=work/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 sha=digest(target)
 if (expected is not None and sha!=expected) or (size is not None and target.stat().st_size!=size):
  raise RuntimeError('PRECISION_STAGED_INPUT_CHANGED:'+str(name))
 inputs.append(dict(source=str(source),path=str(name),bytes=target.stat().st_size,sha256=sha))
def bound_archive(origin,name,delivery):
 rows=[entry for entry in delivery['files'] if entry['path']==name];assert len(rows)==1,name
 copy(origin/name,str(origin.relative_to(repo)/name),rows[0]['sha256'],rows[0]['bytes'])
def phase(label,command):
 begin=time.perf_counter()
 with (work/(label+'.stdout')).open('w') as out,(work/(label+'.stderr')).open('w') as err:
  result=subprocess.run(command,cwd=work,env=env,stdout=out,stderr=err)
 commands.append(dict(phase=label,command=command,exit_code=result.returncode,wall_seconds=time.perf_counter()-begin))
 print(json.dumps(commands[-1]),flush=True)
 if result.returncode:raise RuntimeError('PRECISION_'+label.upper()+'_FAILED:'+str(result.returncode))
try:
 # Reuse the declared module closure from two verified existing preparations.
 foundations=[repo/'Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/profile_history_coverage/prepare/run_605115',
              repo/'Output/neural_population_dynamics/multicircuit/pfc_source_reproduction/clock_refinement/coupled_precision/uncertainty_diagnostic/run_596499']
 names=set()
 for foundation in foundations:
  delivery=json.loads((foundation/'delivery.json').read_text());assert delivery['verified'] is True
  names.update(entry['path'] for entry in delivery['files'] if entry['path'].startswith('Libs/')
               and entry['path'].endswith('.py') and not Path(entry['path']).name.startswith('test_'))
 names.update(('Libs/population_dynamics/test_pfc_coupled_uncertainty.py','Libs/population_dynamics/test_pfc_confirmation_uncertainty.py'))
 software=os.environ.get('NEURONSIM_PRECISION_SOFTWARE')
 if software:
  origin=Path(software);pd=json.loads((origin/'delivery.json').read_text());assert pd['verified'] and pd['status'] in ('PASS_CONDITIONAL_HISTORY_JOINT_PRECISION_SOFTWARE','FAILED_CONDITIONAL_HISTORY_JOINT_PRECISION')
  prior=json.loads((origin/'report.json').read_text())
  assert prior['test_suites'] and any(row['phase'] in ('tests','tests_reused') and row['exit_code']==0 for row in prior['commands'])
  assert all(int(row['failures'])==int(row['errors'])==int(row['skipped'])==0 for row in prior['test_suites'])
  pe={entry['path']:entry for entry in pd['files']}
  for name in sorted(names|{config_name}):copy(origin/name,name,pe[name]['sha256'],pe[name]['bytes'])
  copy(origin/'tests.xml','tests.xml',pe['tests.xml']['sha256'],pe['tests.xml']['bytes'])
  commands.append(dict(phase='tests_reused',archive=str(origin),exit_code=0))
 else:
  assert mode=='validate','COMPLETE_REQUIRES_TESTED_SOFTWARE'
  for name in sorted(names|{config_name}):copy(repo/name,name)
 copy(Path(os.environ['NEURONSIM_PRECISION_LAUNCHER']),'launcher.sbatch')
 copy(repo/'Libs/population_dynamics/resource_probe.py','Libs/population_dynamics/resource_probe.py')
 (work/'tmp').mkdir();env=dict(os.environ,PYTHONPATH=str(work),TMPDIR=str(work/'tmp'))
 sys.path.insert(0,str(work))
 from Libs.population_dynamics.resource_probe import probe
 health=probe(work/'resource_preflight.json',4);assert health['status']=='PASS',health['reasons']
 settings=yaml.safe_load((work/config_name).read_text())
 coverage_archive=os.environ.get('NEURONSIM_PRECISION_COVERAGE_ARCHIVE')
 if coverage_archive:
  assert mode=='complete'
  prior_coverage=settings['conditional_history']['coverage_archive']
  settings['conditional_history']['coverage_archive']=coverage_archive
  (work/config_name).write_text(yaml.safe_dump(settings,sort_keys=False))
  commands.append(dict(phase='coverage_binding_override',original_archive=prior_coverage,archive=coverage_archive,scientific_settings_changed=False,exit_code=0))
 if mode=='validate':
  if not software:phase('tests',[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','--import-mode=importlib',
       'Libs/population_dynamics/test_pfc_coupled_uncertainty.py','Libs/population_dynamics/test_pfc_confirmation_uncertainty.py',
       '--basetemp='+str(work/'test_tmp'),'--junitxml='+str(work/'tests.xml')])
  from Libs.population_dynamics.pfc_coupled_uncertainty import validate_settings,identity_key,IDENTITY,UNDEFINED
  validate_settings(settings)
  domain_ref=settings['inputs']['domain'];origin=repo/domain_ref['path']
  copy(origin/'report.json',Path(domain_ref['path'])/'report.json',domain_ref['report_sha256'])
  report=json.loads((work/domain_ref['path']/'report.json').read_text());entry=report['outputs']['endpoint_domains.csv']
  copy(origin/'endpoint_domains.csv',Path(domain_ref['path'])/'endpoint_domains.csv',entry['sha256'],entry['bytes'])
  history_path='Libs/configs/neural_population_pfc_profile_history.yaml'
  copy(repo/history_path,history_path,settings['conditional_history']['history_config_sha256'])
  import csv
  with (work/domain_ref['path']/'endpoint_domains.csv').open(newline='') as stream:domains=list(csv.DictReader(stream))
  history=yaml.safe_load((work/history_path).read_text())
  assert len(domains)==84330 and sum(row['definition_status']==UNDEFINED for row in domains)==1734
  for profile in history['profiles']:
   assert sum(endpoint['count_threshold']==1 for endpoint in profile['endpoints'])==1
   for endpoint in profile['endpoints']:
    row=domains[endpoint['row_index']];expected={name:profile[name] for name in IDENTITY['count_cdf'] if name!='count_threshold'}
    expected['count_threshold']=endpoint['count_threshold']
    assert row['family']=='count_cdf' and identity_key('count_cdf',json.loads(row['endpoint_identity']))==identity_key('count_cdf',expected)
  (work/'domain_mapping.json').write_text(json.dumps(dict(status='PASS_REGISTERED_PROFILE_DOMAIN_MAPPING',
      domain_rows=84330,undefined_domain_rows=1734,profiles=len(history['profiles']),source_config_sha256=digest(work/history_path),
      real_source_inference_run=False),indent=2)+'\n')
  status='PASS_CONDITIONAL_HISTORY_JOINT_PRECISION_SOFTWARE'
 else:
  donor_ref=settings['conditional_history']['joint_precision'];origin=repo/donor_ref['path']
  copy(origin/'report.json',Path(donor_ref['path'])/'report.json',donor_ref['report_sha256'])
  report=json.loads((work/donor_ref['path']/'report.json').read_text());entry=report['outputs']['joint_bootstrap_maximum.npz']
  copy(origin/'joint_bootstrap_maximum.npz',Path(donor_ref['path'])/'joint_bootstrap_maximum.npz',entry['sha256'],entry['bytes'])
  for key,filename in [('domain','endpoint_domains.csv'),('budget','frozen_numerical_margins.csv')]:
   ref=settings['inputs'][key];origin=repo/ref['path']
   copy(origin/'report.json',Path(ref['path'])/'report.json',ref['report_sha256'])
   report=json.loads((work/ref['path']/'report.json').read_text());entry=report['outputs'][filename]
   copy(origin/filename,Path(ref['path'])/filename,entry['sha256'],entry['bytes'])
  origin=repo/settings['conditional_history']['coverage_archive']
  copy(origin/'delivery.json',Path(origin.relative_to(repo))/'delivery.json')
  delivery=json.loads((work/origin.relative_to(repo)/'delivery.json').read_text())
  assert delivery['verified'] and delivery['status']=='COMPLETE_PFC_PROFILE_HISTORY_COVERAGE_COMPLETE'
  for name in ('analysis/report.json','coverage_inputs/foundation/source_config.yaml'):bound_archive(origin,name,delivery)
  phase('analysis',[sys.executable,'-B','-m','Libs.population_dynamics.pfc_coupled_uncertainty',
        '--config',str(work/config_name),'--project-root',str(work),'--output',str(work/'analysis')])
  status='COMPLETE_CONDITIONAL_HISTORY_JOINT_PRECISION'
 code=0
except Exception as error:
 (work/'failure.json').write_text(json.dumps(dict(type=type(error).__name__,error=str(error)),indent=2)+'\n')
finally:
 suites=[]
 if (work/'tests.xml').exists():
  tree=ET.parse(work/'tests.xml');suites=[dict(suite.attrib) for suite in tree.iter('testsuite')]
 (work/'report.json').write_text(json.dumps(dict(status=status,exit_code=code,job_id=job,mode=mode,node=socket.gethostname(),
      inputs=inputs,commands=commands,test_suites=suites,source_simulations=0,new_auxiliary_trials=0,production_episodes=0,
      coverage_qualified=False,scientific_fidelity_qualified=False,confirmation_execution_released=False,
      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start),indent=2)+'\n')
 try:
  archive.mkdir(parents=True,exist_ok=False);files=[]
  declared=sorted(names|{config_name,'launcher.sbatch','report.json','failure.json','domain_mapping.json',
             'resource_preflight.json','tests.xml','tests.stdout','tests.stderr','analysis.stdout','analysis.stderr'})
  declared += ['analysis/'+name for name in ('report.json','conditional_history_sampling_precision.csv','unresolved_threshold2.csv','conditional_history_joint_precision.npz')]
  for name in declared:
   source=work/name
   if not source.is_file():continue
   target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
   sha=digest(source);assert digest(target)==sha
   files.append(dict(path=name,bytes=source.stat().st_size,sha256=sha))
  (archive/'delivery.json').write_text(json.dumps(dict(status=status,job_id=job,verified=True,files=files),indent=2)+'\n');delivered=True
  print(json.dumps(dict(status=status,archive=str(archive),exit_code=code,files=len(files))),flush=True)
 finally:
  if delivered:shutil.rmtree(work)
  else:print(json.dumps(dict(status='COPYBACK_FAILED_SCRATCH_PRESERVED',work=str(work))),file=sys.stderr,flush=True)
raise SystemExit(code)
PY_PRECISION
