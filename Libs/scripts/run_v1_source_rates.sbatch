#!/bin/bash
#SBATCH --job-name=NrnSim_lgn_source_runtime
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_lgn_source_runtime-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
srun --ntasks=1 env CUDA_VISIBLE_DEVICES= /opt/conda/bin/python3 -B - "$1" "$0" <<'NATIVE'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys,tempfile,traceback,yaml
repo,launcher=map(Path,sys.argv[1:]);job=os.environ['SLURM_JOB_ID'];family=repo/'Output/neural_population_dynamics/multicircuit/v1_asset_intake/original_graph_route';code=1
assert os.environ['SLURMD_NODENAME'] in {'cpu03','cpu04'} and not os.environ.get('SLURM_JOB_GPUS')
construction_source=None
r=dict(job_id=job,status='PREPARING',source_runtime_qualified=False,source_fidelity_qualified=False,production_episodes=0)

def configured_path(name,default):
 value=os.environ.get(name)
 path=Path(value) if value else default
 return path if path.is_absolute() else repo/path

with tempfile.TemporaryDirectory(prefix='neuronsim-lgn-runtime-'+job+'.',dir='/tmp') as tmp:
 work=Path(tmp)
 try:
  pkg=work/'Libs/population_dynamics';pkg.mkdir(parents=True);(pkg/'__init__.py').write_text('')
  for name in ['resource_probe','provenance','v1_lgn_runtime','v1_lgn_transfer','v1_lgn_source','v1_configured_inputs','v1_nest_input_replay','v1_nest_time_semantics']:shutil.copy2(repo/'Libs/population_dynamics'/(name+'.py'),pkg/(name+'.py'))
  shutil.copy2(repo/'Libs/population_dynamics/v1_execution_recipe.py',pkg/'v1_execution_recipe.py')
  sys.path.insert(0,str(work/'Libs'))
  from population_dynamics.v1_execution_recipe import require_same_recipe
  from population_dynamics.resource_probe import probe
  from population_dynamics.v1_nest_input_replay import bound_json
  if probe(work/'preflight.json',required_memory_gib=4)['status']!='PASS':raise RuntimeError('SOURCE_RUNTIME_RESOURCE_BLOCKED')
  config_source=configured_path('NRNSIM_V1_LGN_CONFIG',repo/'Libs/configs/neural_population_v1_source_input_development.yaml')
  construction_source=(configured_path('NRNSIM_V1_LGN_INPUT_CONSTRUCTION',repo) if os.environ.get('NRNSIM_V1_LGN_INPUT_CONSTRUCTION') else None)
  reference_source=(configured_path('NRNSIM_V1_LGN_REFERENCE_ARCHIVE',family/'lgn_source_tables/source_runtime_585990')
                    if construction_source is None or os.environ.get('NRNSIM_V1_LGN_REFERENCE_ARCHIVE') else None)
  reference_prefix=Path(os.environ.get('NRNSIM_V1_LGN_REFERENCE_PREFIX') or 'probe')
  if reference_prefix.is_absolute() or '..' in reference_prefix.parts or reference_prefix==Path('.'):
   raise ValueError('SOURCE_RATE_REFERENCE_PREFIX_MUST_BE_RELATIVE')
  reference_prefix=reference_prefix.as_posix()
  reference_members={reference_prefix+'/'+name for name in ('probe.json','firing_rates_f_tot.h5')}
  prepared_source=(configured_path('NRNSIM_V1_LGN_PREPARED_ARCHIVE',family/'lgn_source_tables/source_runtime_586021')
                   if os.environ.get('NRNSIM_V1_LGN_PREPARED_ARCHIVE') else None)
  specifications=[] if reference_source is None else [(reference_source,'rate_reference',lambda n:n in reference_members)]
  if construction_source is None:
   specifications.append((family/'lgn_source_tables/run_585699','tables',lambda n:n in ('tables/intake.json','tables/lgn_full_col_cells_3.csv','tables/lgn_full_col_cell_models_3.csv')))
  else:
   members={'primary/report.json'}|{'primary/glif/'+pop+'_'+suffix for pop in ('lgn','bkg') for suffix in ('nodes.h5','node_types.csv')}
   specifications.append((construction_source,'construction',lambda n:n in members))
  if prepared_source is None:
   specifications[:0]=[(family/'lgn_source_tables/members_585899','members',lambda n:n.startswith('runtime/')),
    (family/'configured_inputs/legacy_metadata_20261001T101100Z','legacy',lambda n:n=='lgn_functions.py/source_asset.txt'),
    (family/'lgn_source_tables/compatibility_references_20261001T112622Z','references',lambda n:n in ('v1.5.3.py','v2.2.5.py'))]
  else:
   specifications.insert(0,(prepared_source,'prepared_source',lambda n:n.startswith('prepared/')))
  r.update(source_archives=[str(source) for source,_,_ in specifications],
           prepared_runtime_reused=prepared_source is not None,config_source=str(config_source))
  staged={}
  for source,label,select in specifications:
   dest=work/label;dest.mkdir();delivery=json.loads((source/'delivery.json').read_text());shutil.copy2(source/'delivery.json',dest/'delivery.json')
   selected=[]
   for row in delivery['files']:
    if not select(row['path']):continue
    target=dest/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/row['path'],target)
    if target.stat().st_size!=row['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest()!=row['sha256']:
     raise ValueError('SOURCE_RUNTIME_STAGED_CONTENT_DIFFERS:'+row['path'])
    selected.append(row)
   staged[label]=dict(delivery=delivery,rows=selected,
                      delivery_sha256=hashlib.sha256((dest/'delivery.json').read_bytes()).hexdigest())
  if reference_source is not None:
   # Rebase only the two verified rate artifacts; original archives remain untouched.
   reference=staged['rate_reference']
   if len(reference['rows'])!=2 or {row['path'] for row in reference['rows']}!=reference_members:
    raise ValueError('SOURCE_RATE_REFERENCE_MEMBERS_INCOMPLETE')
   rebased=[];r['reference_rebase']=[]
   for row in reference['rows']:
    original=work/'rate_reference'/row['path'];relative='probe/'+Path(row['path']).name
    target=work/'rate_reference'/relative;target.parent.mkdir(parents=True,exist_ok=True)
    if original!=target:original.rename(target)
    rebased.append(dict(row,path=relative))
    r['reference_rebase'].append(dict(original_path=row['path'],staged_path=relative,bytes=row['bytes'],sha256=row['sha256']))
   (work/'rate_reference/delivery.json').write_text(json.dumps(dict(reference['delivery'],files=rebased),indent=2)+'\n')
   r['reference_origin']=dict(archive=str(reference_source),prefix=reference_prefix,
                              delivery_sha256=reference['delivery_sha256'])
  if prepared_source is not None:
   (work/'prepared_source/prepared').rename(work/'prepared')
   prepared=json.loads((work/'prepared/prepared.json').read_text())
   if prepared.get('status')!='SOURCE_RUNTIME_PREPARED':raise ValueError('SOURCE_PREPARED_RUNTIME_REQUIRED')
   r['prepared_runtime_origin']=dict(archive=str(prepared_source),
    delivery_sha256=staged['prepared_source']['delivery_sha256'],
    prepared_runtime_sha256=hashlib.sha256((work/'prepared/prepared.json').read_bytes()).hexdigest())
  config=work/'config.yaml';shutil.copy2(config_source,work/'original_config.yaml');settings=yaml.safe_load(config_source.read_text())
  coverage=os.environ.get('NRNSIM_V1_LGN_UNIT_COVERAGE')
  if coverage:
   if coverage not in ('all_source_units','model_representatives'):raise ValueError('SOURCE_RATE_COVERAGE_OVERRIDE_INVALID')
   settings['unit_coverage']=coverage
  config.write_text(yaml.safe_dump(settings,sort_keys=False))
  r['overrides']={} if coverage is None else {'unit_coverage':coverage}
  r['original_config_sha256']=hashlib.sha256((work/'original_config.yaml').read_bytes()).hexdigest()
  code_reference=os.environ.get('NRNSIM_V1_LGN_CODE_REFERENCE')
  if construction_source is not None and settings['unit_coverage']=='all_source_units':
   if os.environ['SLURMD_NODENAME']!='cpu03':raise ValueError('CONSTRUCTED_FULL_LGN_REQUIRES_CPU03')
   if not code_reference:raise ValueError('CONSTRUCTED_FULL_LGN_REQUIRES_PASSED_CODE_REFERENCE')
  if code_reference:
   prior_path=Path(code_reference);prior_path=prior_path if prior_path.is_absolute() else repo/prior_path
   previous=json.loads((prior_path/'delivery.json').read_text())
   if previous['exit_code']!=0 or previous['status']!='SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT':raise ValueError('SOURCE_RATE_CODE_REFERENCE_NOT_PASSED')
   indexed={row['path']:row for row in previous['files']}
   paths=[(source,source.relative_to(work).as_posix()) for source in pkg.glob('*.py') if source.name!='__init__.py']
   paths=[(source,relative) for source,relative in paths if relative!='Libs/population_dynamics/v1_execution_recipe.py']
   if settings['unit_coverage']=='all_source_units':
    paths.extend([(launcher,'launcher.sbatch'),(pkg/'v1_execution_recipe.py','Libs/population_dynamics/v1_execution_recipe.py')])
   require_same_recipe(settings,yaml.safe_load((prior_path/'original_config.yaml').read_text()),family='visual')
   for source,relative in paths:
    if hashlib.sha256(source.read_bytes()).hexdigest()!=indexed[relative]['sha256']:raise ValueError('SOURCE_RATE_CODE_CHANGED_SINCE_CANARY:'+relative)
   r['code_reference']=str(prior_path)
  r.update(config=settings,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
  if reference_source is not None:
   prior,_=bound_json(work/'rate_reference','probe/probe.json')
   prior_config=prior.get('config',{});ids=prior.get('source_unit_ids',[]);models=prior.get('source_models',[])
   target_rate_config={key:value for key,value in settings.items() if key not in ('unit_coverage','spike_generation')}
   prior_rate_config={key:value for key,value in prior_config.items() if key not in ('unit_coverage','spike_generation')}
   if (prior.get('status')!='SOURCE_RATE_FUNCTION_EXECUTED_DEVELOPMENT'
       or prior.get('source_units')!=14 or len(ids)!=14 or len(set(ids))!=14
       or any(type(value) is not int or not 0<=value<17400 for value in ids)
       or len(models)!=14 or len(set(models))!=14
       or prior.get('complete_source_table_units')!=17400
       or prior.get('rate_shape')!=[14,3001]
       or prior.get('time_units')!='s' or prior.get('firing_rate_units')!='Hz'
       or prior_config.get('unit_coverage')!='model_representatives'
       or settings.get('scope')!='development_source_runtime_qualification'
       or settings.get('unit_coverage') not in ('all_source_units','model_representatives')
       or prior_rate_config!=target_rate_config):
    raise ValueError('SOURCE_RATE_REFERENCE_COVERAGE_OR_CONFIG_DIFFERS')
   r['reference_preflight']=dict(status='PASS',source_units=14,source_unit_ids=ids,source_models=models,
                                 stimulus=prior_config['stimulus'],requested_coverage=settings['unit_coverage'])
  env=dict(os.environ,PYTHONPATH=str(work/'Libs'),TMPDIR=str(work),MPLCONFIGDIR=str(work/'matplotlib'),XDG_CACHE_HOME=str(work/'cache'),NUMBA_CACHE_DIR=str(work/'numba'))
  commands=[]
  if prepared_source is None:
   commands.append(('prepare',[sys.executable,'-B','-m','population_dynamics.v1_lgn_runtime','prepare','--members-archive',str(work/'members'),'--legacy-archive',str(work/'legacy'),'--reference-archive',str(work/'references'),'--converter-python',settings['converter_python'],'--output',str(work/'prepared')]))
  source_args=['--tables-archive',str(work/'tables')] if construction_source is None else ['--input-construction',str(work/'construction')]
  reference_args=[] if reference_source is None else ['--reference-rate-archive',str(work/'rate_reference')]
  r['transfer_execution']='numeric_heaviside_dispatch_with_source_sympy_arithmetic'
  commands.append(('probe',[sys.executable,'-B','-m','population_dynamics.v1_lgn_transfer','--runtime',str(work/'prepared'),*source_args,'--config',str(config),*reference_args,'--output',str(work/'probe')]))
  r['commands']=[]
  for stage,cmd in commands:
   with (work/(stage+'.log')).open('w') as log:code=subprocess.run(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
   r['commands'].append(dict(stage=stage,command=cmd,exit_code=code))
   if code:raise RuntimeError('SOURCE_RUNTIME_'+stage.upper()+'_FAILED')
  r['status']=json.loads((work/'probe/probe.json').read_text())['status']
 except BaseException as error:r.update(status='FAILED',error=str(error));(work/'failure.txt').write_text(traceback.format_exc());code=code or 1
 finally:
  r['completed_utc']=datetime.now(timezone.utc).isoformat();(work/'run.json').write_text(json.dumps(r,indent=2)+'\n');shutil.copy2(launcher,work/'launcher.sbatch');out=(family/'lgn_source_tables'/('source_runtime_'+job) if construction_source is None else family/'circuit_construction_source/visual_inputs'/('rates_'+job));out.mkdir(parents=True,exist_ok=False);files=[]
  for path in sorted(work.rglob('*')):
   if not path.is_file():continue
   rel=path.relative_to(work)
   if rel.parts[0] in ('members','legacy','tables','references','rate_reference','prepared_source','matplotlib','cache','numba') or '__pycache__' in rel.parts:continue
   target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);digest=hashlib.sha256(path.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==digest;files.append(dict(path=str(rel),bytes=target.stat().st_size,sha256=digest))
  (out/'delivery.json').write_text(json.dumps(dict(status=r['status'],exit_code=code,files=files),indent=2)+'\n');print(json.dumps(r),flush=True)
sys.exit(code)
NATIVE
