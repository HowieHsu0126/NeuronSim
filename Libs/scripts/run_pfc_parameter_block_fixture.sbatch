#!/bin/bash
#SBATCH --job-name=NrnSim_PfcBlockFixture
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
#SBATCH --output=/tmp/NrnSim-PfcBlockFixture-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/opt/conda/bin/python3 -B - <<'PY'
from pathlib import Path
import csv,hashlib,json,os,shutil,socket,subprocess,sys,tempfile,traceback
import yaml
assert socket.gethostname().split('.')[0]=='cpu03' and not os.environ.get('SLURM_JOB_GPUS')
repo=Path('/workspace/NeuronSim');base=repo/'Output/neural_population_dynamics/benchmark/pfc_parameter_design';job=os.environ['SLURM_JOB_ID']
plan=Path(os.environ['NEURONSIM_PFC_BLOCK_PLAN']);archive=base/('parameter_block_fixture_source_'+job);assert not archive.exists()
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
record=dict(status='FAILED_PFC_PARAMETER_BLOCK_SOURCE_FIXTURE',job_id=job,fixture_units=20,duration_ms=5000,circuit_index=52,block_index=2,production_episodes=0,qualifying_source_episodes=0,scientific_fidelity_qualified=False,native_qualification_stubbed=False,conditions=[])
code=1
archive.mkdir()
with tempfile.TemporaryDirectory(prefix='neuronsim-pfc-block-fixture-'+job+'.',dir='/tmp') as tmp:
 work=Path(tmp);fake=archive/'fixture_repo';fake_base=fake/'Output/neural_population_dynamics/benchmark/pfc_parameter_design'
 try:
  for row in json.loads((plan/'delivery.json').read_text())['files']:
   if not row['path'].startswith('Libs/'):continue
   dst=work/row['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(plan/row['path'],dst);assert digest(dst)==row['sha256']
  sys.path.insert(0,str(work/'Libs'))
  from population_dynamics.pfc_parameter_block_execution import load_plan,group_rows
  from population_dynamics.pfc_kernel_parameters import circuit_provenance,source_kernel
  from population_dynamics.pfc_native_drive import SOURCE_ENTRY_SHA256
  from population_dynamics.benchmark_tasks import pfc_working_memory_config,task_protocol
  rows=group_rows(load_plan(plan),9);assert [int(r['task_index']) for r in rows]==list(range(90,100))
  source=base/'task_source_prepare_610582';delivery=json.loads((source/'delivery.json').read_text())
  target=fake_base/'task_source_prepare_610582';target.mkdir(parents=True)
  for row in delivery['files']:
   p=target/row['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/row['path'],p);assert digest(p)==row['sha256']
  shutil.copy2(source/'delivery.json',target/'delivery.json')
  preparation=base/'parameter_block_prepare_613107'
  design=json.loads((preparation/'recipes/circuit_052/block_002/report.json').read_text())
  for task in rows:
   config=yaml.safe_load((preparation/task['config_path']).read_text())
   config.update(excitatory_neurons=16,inhibitory_neurons=4,weight_scaling_factor=128.0,state_monitored_per_population=4,excitatory_preference_bins=4)
   config['benchmark_circuit_candidate'].update(circuit_provenance(config,SOURCE_ENTRY_SHA256))
   config['benchmark_circuit_candidate']['effective_kernel_sha256']=hashlib.sha256(source_kernel(16,config).tobytes()).hexdigest()
   config=pfc_working_memory_config(config,task_protocol('pfc_working_memory'),config['benchmark_episode_candidate']['identity'])
   dst=fake_base/'parameter_block_prepare_613107'/task['config_path'];dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(yaml.safe_dump(config,sort_keys=False))
  d=fake_base/'parameter_block_prepare_613107/recipes/circuit_052/block_002/report.json';d.write_text(json.dumps(design,indent=2)+'\n')
  launcher=(plan/'Libs/scripts/run_pfc_parameter_block_source.sbatch').read_text()
  launcher=launcher.replace("repo=Path('/workspace/NeuronSim')","repo=Path("+repr(str(fake))+")")
  launcher=launcher.replace('load_plan(plan_root,require_release=True)','load_plan(plan_root)')
  launcher=launcher.replace("config=recipe_path(task,prepared);qualification=qualification_path(base,int(task['circuit_index']))","config=prepared/task['config_path'];qualification=Path("+repr(str(base/'interior_native_parity/stream_numpy_613135'))+")")
  script=archive/'fixture_source.sbatch';script.write_text(launcher)
  for task in rows:
   env=dict(os.environ,SLURM_ARRAY_TASK_ID=task['task_index'],NEURONSIM_PFC_BLOCK_PLAN=str(plan))
   log=archive/(task['condition_id']+'.log')
   with log.open('x') as stream:result=subprocess.run(['bash',str(script)],env=env,stdout=stream,stderr=subprocess.STDOUT,cwd=work)
   assert result.returncode==0,task['condition_id']
   origin=fake_base/'parameter_block_source/circuit_052/block_002'/task['condition_id']/f"index_{int(task['task_index']):03d}"/('run_'+job)
   delivered=json.loads((origin/'delivery.json').read_text());assert delivered['source_execution_complete'] is True
   report=json.loads((origin/'source_run/report.json').read_text());assert report['dimensions']['coupled_core_units']==20 and report['completed_duration_ticks']==100000
   record['conditions'].append(dict(task_index=int(task['task_index']),episode_id=task['episode_id'],condition_id=task['condition_id'],source_archive=str(origin),delivery_sha256=digest(origin/'delivery.json')))
  record.update(status='PASS_PFC_PARAMETER_BLOCK_SOURCE_FIXTURE',complete_engineering_episodes=10,source_launcher_sha256=digest(plan/'Libs/scripts/run_pfc_parameter_block_source.sbatch'),fixture_launcher_sha256=digest(script),fixture_repo=str(fake),plan=str(plan))
  code=0
 except Exception as error:record.update(error=str(error),traceback=traceback.format_exc())
 record['exit_code']=code;(archive/'report.json').write_text(json.dumps(record,indent=2)+'\n')
 files=[]
 for p in sorted(archive.rglob('*')):
  if not p.is_file():continue
  files.append(dict(path=str(p.relative_to(archive)),bytes=p.stat().st_size,sha256=digest(p)))
 (archive/'delivery.json').write_text(json.dumps(dict(status=record['status'],job_id=job,files=files),indent=2)+'\n')
 print(json.dumps(record))
raise SystemExit(code)
PY
