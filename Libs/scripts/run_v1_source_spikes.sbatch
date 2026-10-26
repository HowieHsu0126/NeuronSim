#!/bin/bash
#SBATCH --job-name=NrnSim_v1_visual_inputs
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_v1_visual_inputs-%j.log
set -euo pipefail
[[ $# == 4 || $# == 5 || $# == 6 ]] || exit 2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --ntasks=1 /opt/conda/bin/python3 -B - "$@" "$0" <<'PYTHON'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys,tempfile,traceback,xml.etree.ElementTree as ET
import yaml
repo=Path(sys.argv[1]);mode=sys.argv[2];rate_source=Path(sys.argv[3]);construction_source=Path(sys.argv[4])
launcher=Path(sys.argv[-1]);prior=Path(sys.argv[5]) if len(sys.argv)>=7 else None
noise_config=Path(sys.argv[6]) if len(sys.argv)==8 else None
override=os.environ.get('NRNSIM_V1_VISUAL_CONFIG','')
validation_path=os.environ.get('NRNSIM_V1_VISUAL_SOURCE_VALIDATION','')
if bool(override)!=bool(validation_path):raise ValueError('VISUAL_NOISE_CONFIG_REQUIRES_SOURCE_VALIDATION')
if noise_config is not None and (override or validation_path):
 raise ValueError('VISUAL_NOISE_CONFIG_REQUIRES_ONE_CONFIG_INTERFACE')
job=os.environ['SLURM_JOB_ID'];node=os.environ['SLURMD_NODENAME']
assert mode in ('canary','full') and node in ('cpu03','cpu04') and not os.environ.get('SLURM_JOB_GPUS')
assert mode!='full' or node=='cpu03'
modules=['v1_circuit_stimulus','test_v1_circuit_stimulus','v1_lgn_source','v1_lgn_runtime','v1_lgn_spikes',
         'v1_nest_input_replay','v1_configured_inputs','v1_nest_time_semantics','provenance','resource_probe']
receipt=dict(job_id=job,mode=mode,node=node,status='PREPARING',scientific_qualification=False,production_episodes=0,
             rate_archive=str(rate_source),input_construction_archive=str(construction_source))
code=1
with tempfile.TemporaryDirectory(prefix='neuronsim-visual-inputs-'+job+'.',dir='/tmp') as temporary:
 work=Path(temporary)
 def digest(path):
  with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
 def copy(source,target):
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 def stage(source,target,names):
  origin=json.loads((source/'delivery.json').read_text())
  if origin['exit_code']!=0:raise ValueError('VISUAL_INPUT_SOURCE_NOT_COMPLETE:'+str(source))
  indexed={r['path']:r for r in origin['files']}
  copy(source/'delivery.json',target/'delivery.json')
  for relative in names:
   row=indexed[relative];dest=target/relative;copy(source/relative,dest)
   if dest.stat().st_size!=row['bytes'] or digest(dest)!=row['sha256']:raise ValueError('VISUAL_INPUT_SOURCE_CHANGED:'+relative)
 def manifest(root):
  return [dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest(p))
          for p in sorted((root/'visual_inputs').rglob('*')) if p.is_file()]
 try:
  package=work/'Libs/population_dynamics';package.mkdir(parents=True);(package/'__init__.py').write_text('')
  validation_rows=None
  if validation_path:
   validation=Path(validation_path).resolve();proof=json.loads((validation/'delivery.json').read_text())
   validation_rows={r['path']:r for r in proof['files']}
   xml=validation/'junit.xml';row=validation_rows['junit.xml']
   if xml.stat().st_size!=row['bytes'] or digest(xml)!=row['sha256']:raise ValueError('VISUAL_NOISE_JUNIT_CHANGED')
   counts={k:sum(int(x.get(k,'0')) for x in ET.parse(xml).getroot().iter('testsuite')) for k in ('tests','failures','errors','skipped')}
   if (proof['status']!='PASS_SOURCE_AND_CONSTRUCTED_CONSUMER_TESTS' or proof['exit_code']!=0
       or counts!=proof['tests'] or counts['tests']<1 or any(counts[k] for k in ('failures','errors','skipped'))):
    raise ValueError('VISUAL_NOISE_PASSED_SOURCE_TESTS_REQUIRED')
   for relative in ['Libs/population_dynamics/test_v1_circuit_stimulus.py','Libs/population_dynamics/test_v1_visual_controls.py']:
    if digest(validation/relative)!=validation_rows[relative]['sha256']:raise ValueError('VISUAL_NOISE_TEST_SOURCE_CHANGED')
   copy(validation/'delivery.json',work/'source_validation_delivery.json');copy(xml,work/'tests.xml')
   receipt['source_validation']=dict(archive=str(validation),delivery_sha256=digest(validation/'delivery.json'),junit=counts,tests_rerun=False)
  for name in modules:
   relative='Libs/population_dynamics/'+name+'.py'
   # Keep the existing operational health helper; scientific modules come
   # from the completed isolated tests, without a mutable registry merge.
   source=validation/relative if validation_rows is not None and name!='resource_probe' else repo/relative
   copy(source,package/(name+'.py'))
   if validation_rows is not None and name!='resource_probe' and digest(package/(name+'.py'))!=validation_rows[relative]['sha256']:
    raise ValueError('VISUAL_NOISE_VALIDATED_SOURCE_CHANGED:'+relative)
  for name in ('neural_population_v1_constructed_visual_development.yaml','neural_population_v1_noise_development.yaml'):
   copy(repo/'Libs/configs'/name,work/'Libs/configs'/name)
  copy(launcher,work/'launcher.sbatch')
  copy(repo/'Libs/population_dynamics/v1_execution_recipe.py',package/'v1_execution_recipe.py')
  sys.path.insert(0,str(work/'Libs'))
  from population_dynamics.v1_execution_recipe import require_same_recipe
  source_delivery=json.loads((rate_source/'delivery.json').read_text())
  prepared=json.loads((rate_source/'prepared/prepared.json').read_text())
  names=['probe/probe.json','probe/firing_rates_f_tot.h5','config.yaml','prepared/prepared.json']
  names+=['prepared/'+row['path'] for row in prepared['files']]
  stage(rate_source,work/'rate_reference',names)
  names=['primary/report.json']+['primary/glif/'+p+'_'+s for p in ('lgn','bkg') for s in ('nodes.h5','node_types.csv')]
  stage(construction_source,work/'construction',names)
  config_source=(noise_config.resolve() if noise_config is not None else
                 Path(override).resolve() if override else work/'rate_reference/config.yaml')
  copy(config_source,work/'config.yaml')
  if override:
   receipt['explicit_visual_config']=dict(path=str(Path(override).resolve()),sha256=digest(work/'config.yaml'))
  if noise_config is not None:
   receipt['explicit_visual_config']=dict(path=str(noise_config.resolve()),sha256=digest(work/'config.yaml'))
   receipt['rate_recomputed']=False
  config=yaml.safe_load((work/'config.yaml').read_text())
  if config['unit_coverage']!=('model_representatives' if mode=='canary' else 'all_source_units'):
   raise ValueError('VISUAL_INPUT_MODE_AND_RATE_COVERAGE_DIFFER')
  if mode=='full' or prior is not None:
   if prior is None:raise ValueError('FULL_VISUAL_INPUT_REQUIRES_PASSED_CANARY')
   previous=json.loads((prior/'delivery.json').read_text())
   if previous['status']!='PASS_CONSTRUCTED_VISUAL_INPUT_CANARY' or previous['exit_code']!=0:raise ValueError('FULL_VISUAL_CANARY_NOT_PASSED')
   indexed={r['path']:r for r in previous['files']}
   for relative in ['Libs/population_dynamics/'+name+'.py' for name in modules]:
    if digest(work/relative)!=indexed[relative]['sha256']:raise ValueError('FULL_VISUAL_CODE_CHANGED_SINCE_CANARY:'+relative)
   if mode=='full':
    for relative in ('launcher.sbatch','Libs/population_dynamics/v1_execution_recipe.py'):
     if relative not in indexed or digest(work/relative)!=indexed[relative]['sha256']:
      raise ValueError('FULL_VISUAL_EXECUTION_CODE_CHANGED_SINCE_CANARY:'+relative)
   old_config=yaml.safe_load((prior/'config.yaml').read_text())
   require_same_recipe(config,old_config,family='visual')
   receipt['canary_reference']=str(prior)
  env=dict(os.environ,PYTHONPATH=str(work/'Libs'),TMPDIR=str(work),MPLCONFIGDIR=str(work/'mpl'),XDG_CACHE_HOME=str(work/'cache'))
  commands=[('resource',[sys.executable,'-B','-m','population_dynamics.resource_probe','--output',str(work/'resources.json')]),
   ('tests',[sys.executable,'-B','-m','pytest',str(package/'test_v1_circuit_stimulus.py'),'-q','-p','no:cacheprovider','--junitxml='+str(work/'tests.xml')]),
   ('generate',[sys.executable,'-B','-m','population_dynamics.v1_circuit_stimulus','--rate-archive',str(work/'rate_reference'),
     '--input-construction',str(work/'construction'),'--config',str(work/'config.yaml'),'--output',str(work/'visual_inputs')])]
  if validation_rows is not None:
   commands=[row for row in commands if row[0]!='tests']
  reuse_generation=os.environ.get('NRNSIM_V1_VISUAL_REUSE_GENERATION')
  if reuse_generation:
   previous_path=Path(reuse_generation);previous=json.loads((previous_path/'delivery.json').read_text())
   generated=json.loads((previous_path/'visual_inputs/report.json').read_text())
   if not any(row['phase']=='generate' and row['exit_code']==0 for row in previous['commands']):
    raise ValueError('VISUAL_REUSE_REQUIRES_COMPLETED_GENERATOR_PHASE')
   if generated.get('exit_code')!=0 or generated.get('config')!=config or generated.get('rate_report_sha256')!=digest(work/'rate_reference/probe/probe.json'):
    raise ValueError('VISUAL_REUSE_RATE_OR_CONFIG_DIFFERS')
   entries=[row for row in previous['files'] if row['path'].startswith('visual_inputs/')]
   for row in entries:
    dest=work/row['path'];copy(previous_path/row['path'],dest)
    if dest.stat().st_size!=row['bytes'] or digest(dest)!=row['sha256']:raise ValueError('VISUAL_REUSED_GENERATION_CHANGED')
   commands=[row for row in commands if row[0]!='generate']
   receipt['generation_reused_from']=dict(archive=str(previous_path),delivery_sha256=digest(previous_path/'delivery.json'),
       producer_completed=True,previous_overall_exit_code=previous['exit_code'])
  receipt['commands']=[]
  for phase,args in commands:
   with (work/(phase+'.log')).open('w') as log:code=subprocess.run(args,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
   receipt['commands'].append(dict(phase=phase,command=args,exit_code=code))
   if code:raise RuntimeError('VISUAL_INPUT_'+phase.upper()+'_FAILED')
  # The existing loader consumes this local delivery before publication.
  (work/'delivery.json').write_text(json.dumps(dict(exit_code=0,files=manifest(work))))
  sys.path.insert(0,str(work/'Libs'))
  from population_dynamics.v1_circuit_stimulus import load_inputs
  generated=json.loads((work/'visual_inputs/report.json').read_text())
  from population_dynamics.v1_lgn_source import read_input_construction_cells
  _,_,expected_binding=read_input_construction_cells(work/'construction')
  plan,loaded=load_inputs(work,expected_binding,require_complete=mode=='full')
  receipt['consumer_readback']=dict(generators=len(plan['source_id']),events=len(plan['source_event_time_ms']),
    input_identity_sha256=loaded['input_identity_sha256'],complete_lgn_population=loaded['complete_lgn_population'])
  receipt['status']='PASS_CONSTRUCTED_VISUAL_INPUT_'+mode.upper();code=0
 except BaseException as error:
  code=code or 1;receipt.update(status='FAILED_CONSTRUCTED_VISUAL_INPUT_'+mode.upper(),error_type=type(error).__name__,error=str(error));(work/'failure.txt').write_text(traceback.format_exc())
 finally:
  if (work/'tests.xml').exists():
   suites=list(ET.parse(work/'tests.xml').getroot().iter('testsuite'));receipt['junit']={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
  out=repo/'Output/neural_population_dynamics/multicircuit/v1_asset_intake/original_graph_route/circuit_construction_source/visual_inputs'/('spikes_'+job)
  out.mkdir(parents=True,exist_ok=False);files=[]
  for path in sorted(work.rglob('*')):
   relative=path.relative_to(work)
   if not path.is_file() or relative.parts[0] in ('rate_reference','construction','mpl','cache') or '__pycache__' in relative.parts or relative==Path('delivery.json'):continue
   dest=out/relative;copy(path,dest);sha=digest(path)
   if digest(dest)!=sha:raise ValueError('VISUAL_INPUT_DELIVERY_CHANGED:'+str(relative))
   files.append(dict(path=relative.as_posix(),bytes=dest.stat().st_size,sha256=sha))
  receipt.update(exit_code=code,files=files,recorded_utc=datetime.now(timezone.utc).isoformat())
  (out/'delivery.json').write_text(json.dumps(receipt,indent=2)+'\n')
  print(json.dumps({k:v for k,v in receipt.items() if k not in ('files','commands')}))
sys.exit(code)
PYTHON
