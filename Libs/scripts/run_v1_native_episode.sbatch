#!/bin/bash
#SBATCH --job-name=NrnSim_v1_pointnet
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim_v1_pointnet-%j.log
set -euo pipefail
[[ $# == 7 || $# == 8 ]] || exit 2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${NRNSIM_V1_NEST_PYTHON:?Set validated NEST interpreter}" -B - "$@" "$0" <<'PYTHON'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys,tempfile,traceback,xml.etree.ElementTree as ET
import yaml
repo=Path(sys.argv[1]);config=Path(sys.argv[2]);mode=sys.argv[3]
recurrent=Path(sys.argv[4]);inputs=Path(sys.argv[5]);parameters=Path(sys.argv[6]);source_config=Path(sys.argv[7])
launcher=Path(sys.argv[-1]);prior=Path(sys.argv[8]) if len(sys.argv)==10 else None
native_code_reference=Path(os.environ['NRNSIM_V1_NATIVE_CODE_REFERENCE']) if os.environ.get('NRNSIM_V1_NATIVE_CODE_REFERENCE') else None
if native_code_reference is not None and (prior is None or native_code_reference.resolve()!=prior.resolve()):
 raise ValueError('NATIVE_CODE_REFERENCE_REQUIRES_THE_COMPLETED_CANARY')
spike_inputs=Path(os.environ['NRNSIM_V1_SPIKE_INPUTS']) if os.environ.get('NRNSIM_V1_SPIKE_INPUTS') else None
if spike_inputs is not None and not spike_inputs.is_absolute():spike_inputs=repo/spike_inputs
clock_protocol=Path(os.environ['NRNSIM_V1_CLOCK_PROTOCOL']) if os.environ.get('NRNSIM_V1_CLOCK_PROTOCOL') else None
if clock_protocol is not None and not clock_protocol.is_absolute():clock_protocol=repo/clock_protocol
input_mode='constructed_visual_inputs' if spike_inputs is not None else 'diagnostic_pulses'
passed_status='PASS_POINTNET_NATIVE_VISUAL_INPUTS_DEVELOPMENT' if spike_inputs is not None else 'PASS_POINTNET_NATIVE_DIAGNOSTIC'
failed_status='FAILED_POINTNET_NATIVE_VISUAL_INPUTS' if spike_inputs is not None else 'FAILED_POINTNET_NATIVE_DIAGNOSTIC'
if clock_protocol is not None:
 passed_status='PASS_POINTNET_NATIVE_TEMPORAL_SENSITIVITY_DEVELOPMENT'
 failed_status='FAILED_POINTNET_NATIVE_TEMPORAL_SENSITIVITY'
job=os.environ['SLURM_JOB_ID'];node=os.environ['SLURMD_NODENAME']
assert mode in {'canary','full'} and node in {'cpu03','cpu04'} and not os.environ.get('SLURM_JOB_GPUS')
assert mode!='full' or node=='cpu03'
assert int(os.environ.get('SLURM_NTASKS','1'))==1
family=repo/'Output/neural_population_dynamics/multicircuit/v1_asset_intake/original_graph_route/circuit_construction_source/native_pointnet'
receipt=dict(status='PREPARING',job_id=job,node=node,mode=mode,recurrent_reference=str(recurrent),
 input_reference=str(inputs),parameter_reference=str(parameters),source_config=str(source_config),
 input_mode=input_mode,spike_inputs_reference=str(spike_inputs) if spike_inputs is not None else None,
 stimulus_kind='constructed_visual_and_background_spike_replay' if spike_inputs is not None else 'diagnostic_synchronous_precise_pulses',
 visual_episode=False,scientific_qualification=False,production_episodes=0,
 clock_protocol=str(clock_protocol) if clock_protocol is not None else None)
modules=['v1_pointnet_native','test_v1_pointnet_native','v1_circuit_inputs','v1_circuit_build',
 'v1_circuit_source','v1_source_conversion','v1_graph_identity','v1_synaptic_weights','v1_cell_parameters',
 'v1_nest_network','v1_nest_input_replay','v1_nest_time_semantics','test_v1_nest_time_semantics','v1_lgn_spikes','v1_lgn_runtime',
 'v1_lgn_source','v1_circuit_stimulus','v1_configured_inputs','provenance','resource_probe']
sources=['original_build_network','original_node_functions','original_edge_functions','original_node_parameters',
 'original_edge_parameters','original_connection_parameters','original_lgn_connection_parameters',
 'original_glif_node_parameters','original_bio_to_glif_map','original_glif_recurrent_parameters',
 'weight_semantics/legacy_glif_converter','weight_semantics/legacy_recurrent_table',
 'original_lgn_node_parameters','original_lgn_edge_parameters','original_bkg_edge_parameters',
 'original_glif_lgn_parameters','original_glif_bkg_parameters','weight_semantics/legacy_pointnet_runner',
 'weight_semantics/legacy_e2e','weight_semantics/legacy_e2i','weight_semantics/legacy_i2e','weight_semantics/legacy_i2i']
v1_files=['v1_nodes.h5','v1_node_types.csv','v1_v1_edges.h5','v1_v1_edge_types.csv']
input_files=[pop+'_'+suffix for pop in ['lgn','bkg'] for suffix in ['nodes.h5','node_types.csv','v1_edges.h5','v1_edge_types.csv']]
code=1
with tempfile.TemporaryDirectory(prefix='neuronsim-v1-pointnet-'+job+'.',dir='/tmp') as temporary:
 work=Path(temporary)
 def digest(path):
  with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
 def copy(source,dest):
  dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
 static_references = {}
 try:
  package=work/'Libs/population_dynamics';package.mkdir(parents=True);(package/'__init__.py').write_text('')
  code_rows=None
  if native_code_reference is not None:
   code_delivery=json.loads((native_code_reference/'delivery.json').read_text())
   if code_delivery.get('exit_code')!=0:raise ValueError('NATIVE_CODE_REFERENCE_NOT_COMPLETE')
   code_rows={row['path']:row for row in code_delivery['files']}
   receipt['native_code_reference']=dict(archive=str(native_code_reference),delivery_sha256=digest(native_code_reference/'delivery.json'))
  for name in modules:
   relative='Libs/population_dynamics/'+name+'.py'
   copy((native_code_reference if native_code_reference is not None else repo)/relative,package/(name+'.py'))
   if code_rows is not None and digest(package/(name+'.py'))!=code_rows[relative]['sha256']:
    raise ValueError('NATIVE_CODE_REFERENCE_MODULE_CHANGED:'+relative)
  copy(launcher,work/'launcher.sbatch')
  copy(repo/'Libs/population_dynamics/v1_execution_recipe.py',package/'v1_execution_recipe.py')
  sys.path.insert(0,str(work/'Libs'))
  from population_dynamics.v1_execution_recipe import require_same_recipe
  canonical=yaml.safe_load(config.read_text());source_root=Path(canonical['source_root'])
  if not source_root.is_absolute():source_root=repo/source_root
  for name in sources:
   for leaf in ['source.json','source_asset.txt']:copy(source_root/name/leaf,work/'sources'/name/leaf)
  for leaf in ['source.json','source_asset.txt']:copy(source_config/leaf,work/'source_config'/leaf)
  rec_files=['delivery.json','primary/report.json']+['primary/glif/'+name for name in v1_files]+['primary/biophysical/'+name for name in v1_files[:2]]
  inp_files=['delivery.json','primary/report.json']+['primary/glif/'+name for name in v1_files[:2]+input_files]
  for original,destination,files in [(recurrent,work/'recurrent',rec_files),(inputs,work/'inputs_reference',inp_files)]:
   for relative in files:copy(original/relative,destination/relative)
  if json.loads((work/'recurrent/primary/report.json').read_text())['mode']!=mode:raise ValueError('POINTNET_MODE_DIFFERS_FROM_RECURRENT')
  if spike_inputs is not None:
   spike_files=['delivery.json','visual_inputs/report.json']+['visual_inputs/'+pop+'/trial_0000.'+suffix for pop in ['lgn','bkg'] for suffix in ['npz','h5']]
   for relative in spike_files:copy(spike_inputs/relative,work/'spike_inputs'/relative)
  if clock_protocol is not None:
   original_clock=yaml.safe_load(clock_protocol.read_text())
   reference_key='reference_'+mode+'_archive'
   clock_reference=Path(original_clock[reference_key])
   if not clock_reference.is_absolute():clock_reference=repo/clock_reference
   for relative in ['delivery.json','primary/report.json','replay/report.json','comparison/report.json']:
    copy(clock_reference/relative,work/'clock_reference'/relative)
   copy(clock_protocol,work/'clock_protocol.original.yaml')
   resolved_clock=dict(original_clock,**{reference_key:str(work/'clock_reference')})
   (work/'clock_protocol.resolved.yaml').write_text(yaml.safe_dump(resolved_clock,sort_keys=False))
   receipt['clock_reference']=str(clock_reference)
   receipt['clock_protocol_original_sha256']=digest(clock_protocol)
  ref=json.loads((parameters/'reference/intake.json').read_text())
  ref_files=['delivery.json','reference/intake.json']+['reference/'+row['source']['path'] for row in ref['references']]
  for relative in ref_files:
   path=Path(relative)
   if path.is_absolute() or '..' in path.parts:raise ValueError('POINTNET_REFERENCE_PATH_INVALID')
   copy(parameters/path,work/'parameters'/path)
  wheel=Path(canonical['bmtk_wheel']['path'])
  if not wheel.is_absolute():wheel=repo/wheel
  assert digest(wheel)==canonical['bmtk_wheel']['sha256']
  copy(wheel,work/'runtime'/wheel.name)
  resolved=dict(canonical,source_root=str(work/'sources'),bmtk_wheel=dict(canonical['bmtk_wheel'],path=str(work/'runtime'/wheel.name)),original_config=canonical,original_config_sha256=digest(config))
  copy(config,work/'original_config.yaml');(work/'resolved_config.yaml').write_text(yaml.safe_dump(resolved,sort_keys=False))
  (work/'scratch').mkdir();env=dict(os.environ,PYTHONPATH=str(work/'Libs'),TMPDIR=str(work/'scratch'))
  if mode=='full' or prior is not None:
   if prior is None:raise ValueError('FULL_POINTNET_REQUIRES_COMPLETED_CANARY')
   previous=json.loads((prior/'delivery.json').read_text())
   allowed_canary_statuses={'PASS_POINTNET_NATIVE_TEMPORAL_SENSITIVITY_DEVELOPMENT'} if clock_protocol is not None else {'PASS_POINTNET_NATIVE_DIAGNOSTIC','PASS_POINTNET_NATIVE_VISUAL_INPUTS_DEVELOPMENT'}
   if previous['status'] not in allowed_canary_statuses or previous['exit_code']!=0:raise ValueError('FULL_POINTNET_CANARY_NOT_PASSED')
   indexed={row['path']:row for row in previous['files']}
   fixed=['Libs/population_dynamics/'+name+'.py' for name in modules]
   fixed+=['sources/'+name+'/'+leaf for name in sources for leaf in ['source.json','source_asset.txt']]
   fixed+=['source_config/'+leaf for leaf in ['source.json','source_asset.txt']]+['parameters/'+name for name in ref_files]
   if clock_protocol is not None:
    from population_dynamics.v1_pointnet_native import BENCHMARK_CLOCK_SCOPE, require_clock_canary_protocol
    prior_clock_path = prior/'clock_protocol.original.yaml'
    if digest(prior_clock_path) != indexed['clock_protocol.original.yaml']['sha256']:
     raise ValueError('FULL_POINTNET_CHANGED_SINCE_CANARY:clock_protocol.original.yaml')
    prior_clock = yaml.safe_load(prior_clock_path.read_text())
    require_clock_canary_protocol(original_clock, prior_clock)
    if original_clock['scope'] != BENCHMARK_CLOCK_SCOPE:fixed+=['clock_protocol.original.yaml']
   for relative in fixed:
    if digest(work/relative)!=indexed[relative]['sha256']:raise ValueError('FULL_POINTNET_CHANGED_SINCE_CANARY:'+relative)
   if mode=='full':
    for relative in ('launcher.sbatch','Libs/population_dynamics/v1_execution_recipe.py'):
     if relative not in indexed or digest(work/relative)!=indexed[relative]['sha256']:
      raise ValueError('FULL_POINTNET_EXECUTION_CODE_CHANGED_SINCE_CANARY:'+relative)
   require_same_recipe(canonical,yaml.safe_load((prior/'original_config.yaml').read_text()),family='circuit')
   receipt['canary_reference']=str(prior)
  command=[sys.executable,'-B','-m','population_dynamics.v1_pointnet_native']
  probe=command+['probe','--config',str(work/'resolved_config.yaml'),'--recurrent',str(work/'recurrent'),
   '--inputs',str(work/'inputs_reference'),'--parameters',str(work/'parameters'),'--source-config',str(work/'source_config')]
  if spike_inputs is not None:probe+=['--spike-inputs',str(work/'spike_inputs')]
  if clock_protocol is not None:probe+=['--clock-protocol',str(work/'clock_protocol.resolved.yaml')]
  phases=[('resources',[sys.executable,'-B','-m','population_dynamics.resource_probe','--output',str(work/'resources.json')]),
   ('tests',[sys.executable,'-B','-m','pytest',str(package/'test_v1_pointnet_native.py'),str(package/'test_v1_nest_time_semantics.py'),'-q','-p','no:cacheprovider','--junitxml='+str(work/'tests.xml')]),
   ('primary',probe+['--output',str(work/'primary')]),('replay',probe+['--output',str(work/'replay')]),
   ('compare',command+['compare','--left',str(work/'primary'),'--right',str(work/'replay'),'--output',str(work/'comparison')])]
  receipt['commands']=[args for _,args in phases]
  for phase,args in phases:
   print(json.dumps(dict(job_id=job,phase=phase,state='STARTED')),flush=True)
   with (work/(phase+'.log')).open('w') as log:code=subprocess.run(args,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
   print(json.dumps(dict(job_id=job,phase=phase,exit_code=code)),flush=True)
   if code:break
  if (work/'tests.xml').exists():
   suites=list(ET.parse(work/'tests.xml').getroot().iter('testsuite'));receipt['junit']={key:sum(int(s.get(key,'0')) for s in suites) for key in ['tests','failures','errors','skipped']}
  if code == 0:
   primary_report = json.loads((work/'primary/report.json').read_text())
   for population in ('v1','lgn','bkg'):
    role = 'recurrent' if population == 'v1' else 'inputs'
    archive = recurrent if population == 'v1' else inputs
    source_relative = 'primary/glif/'+population+'_v1_edges.h5'
    origin = json.loads((archive/'delivery.json').read_text())
    rows = [row for row in origin['files'] if row['path'] == source_relative]
    if origin.get('exit_code') != 0 or len(rows) != 1:
     raise ValueError('POINTNET_STATIC_SOURCE_NOT_COMPLETE:'+population)
    expected = {key: rows[0][key] for key in ('bytes','sha256')}
    if (primary_report['bindings'][role]['delivery_sha256'] != digest(archive/'delivery.json')
        or primary_report['bindings'][role]['files'][source_relative] != expected):
     raise ValueError('POINTNET_STATIC_SOURCE_BINDING_DIFFERS:'+population)
    names = ['primary/network/'+population+'_v1_edges.h5',
             'replay/network/'+population+'_v1_edges.h5',
             ('recurrent' if population == 'v1' else 'inputs_reference')+'/'+source_relative]
    for name in names:
     path = work/name
     if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
      raise ValueError('POINTNET_STATIC_COPY_DIFFERS:'+name)
     static_references[name] = dict(archive=str(archive),path=source_relative,
       delivery_sha256=primary_report['bindings'][role]['delivery_sha256'],**expected)
   receipt['static_edge_references'] = static_references
   receipt['static_edge_copies_not_delivered'] = len(static_references)
   receipt['static_edge_copy_bytes_not_delivered'] = sum(row['bytes'] for row in static_references.values())
  receipt['status']=passed_status if code==0 else failed_status
 except BaseException as error:
  static_references = {}
  code=code or 1;receipt.update(status=failed_status,error_type=type(error).__name__,error=str(error));(work/'failure.txt').write_text(traceback.format_exc())
 finally:
  target=family/('run_'+job);target.mkdir(parents=True,exist_ok=False);files=[]
  for path in sorted(work.rglob('*')):
   relative=path.relative_to(work)
   if relative.as_posix() in static_references:continue
   if not path.is_file() or relative.parts[0] in {'runtime','scratch'} or '__pycache__' in relative.parts:continue
   if any(part in {'_bmtk_runtime','runtime'} for part in relative.parts):continue
   dest=target/relative;copy(path,dest);value=digest(path)
   if digest(dest)!=value:raise ValueError('POINTNET_DELIVERY_BYTES_DIFFER:'+str(relative))
   files.append(dict(path=str(relative),bytes=dest.stat().st_size,sha256=value))
  receipt.update(files=files,exit_code=code,recorded_utc=datetime.now(timezone.utc).isoformat())
  (target/'delivery.json').write_text(json.dumps(receipt,indent=2)+'\n')
  print(json.dumps({k:v for k,v in receipt.items() if k not in {'files','commands'}}),flush=True)
sys.exit(code)
PYTHON
