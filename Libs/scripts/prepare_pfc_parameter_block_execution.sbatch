#!/bin/bash
#SBATCH --job-name=NrnSim_PfcBlockPrepare
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=none
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim-PfcBlockPrepare-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/opt/conda/bin/python3 -B - <<'PY'
from pathlib import Path
import csv,hashlib,json,os,shutil,socket,sys,tempfile,traceback,subprocess
assert socket.gethostname().split('.')[0]=='cpu03' and not os.environ.get('SLURM_JOB_GPUS')
repo=Path('/workspace/NeuronSim');base=repo/'Output/neural_population_dynamics/benchmark/pfc_parameter_design'
job=os.environ['SLURM_JOB_ID'];archive=base/('parameter_block_execution_prepare_'+job);assert not archive.exists()
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
record=dict(status='FAILED_PFC_PARAMETER_BLOCK_EXECUTION_PREPARATION',job_id=job,new_source_simulations=0,production_episodes=0,scientific_fidelity_qualified=False,source_execution_released=False)
code=1
with tempfile.TemporaryDirectory(prefix='neuronsim-pfc-block-plan-'+job+'.',dir='/tmp') as tmp:
 work=Path(tmp);files=[]
 try:
  frozen=base/'corner_task_reader_prepare_610692'
  software=json.loads((frozen/'delivery.json').read_text())
  for row in software['files']:
   if not row['path'].startswith('Libs/'):continue
   src=frozen/row['path'];dst=work/row['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
   assert dst.stat().st_size==row['bytes'] and digest(dst)==row['sha256']
  names=['Libs/population_dynamics/pfc_parameter_block_execution.py','Libs/scripts/run_pfc_parameter_block_source.sbatch','Libs/scripts/read_pfc_parameter_block.sbatch','Libs/scripts/intake_pfc_parameter_block.sbatch','Libs/scripts/run_pfc_parameter_block_fixture.sbatch','Libs/scripts/validate_pfc_parameter_block_fixture.sbatch']
  for name in names:
   src=repo/name;dst=work/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  for name in names[1:]:
   assert subprocess.run(['bash','-n',str(work/name)]).returncode==0
   script=(work/name).read_text();marker=script.split("<<'",1)[1].split("'",1)[0]
   body=script.split("<<'"+marker+"'\n",1)[1].rsplit("\n"+marker,1)[0];compile(body,name,'exec')
  compile((work/names[0]).read_text(),names[0],'exec')
  preparation=base/'parameter_block_prepare_613107';delivery=json.loads((preparation/'delivery.json').read_text())
  binding=next(row for row in delivery['files'] if row['path']=='episodes.csv')
  assert digest(preparation/'episodes.csv')==binding['sha256']
  with (preparation/'episodes.csv').open(newline='') as f:original=list(csv.DictReader(f))
  rows=[dict(row,task_index=index) for index,row in enumerate(original)];assert len(rows)==100
  with (work/'execution.csv').open('x',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
  # Authenticate every prepared recipe once; later consumers validate only their selected cells.
  sys.path.insert(0,str(work/'Libs'))
  from population_dynamics.pfc_parameter_block_execution import identity,recipe_path,source_path,group_rows,qualification_path
  for row in rows:recipe_path(row,preparation)
  for group in range(10):
   selected=group_rows(rows,group);assert len({row['episode_id'] for row in selected})==10
  for circuit in [23,63,74,86,52]:
   receipt=qualification_path(base,circuit)
   d=json.loads((receipt/'delivery.json').read_text())
   assert d['status']=='ARCHIVED_PFC_STREAM_VALIDATION' and d['result_status']=='PASS_NATIVE_STREAM_SOURCE_EXECUTION'
  assert str(source_path(base,rows[90])).endswith('circuit_052/block_002/cue_none/index_090')
  assert str(source_path(base,rows[99])).endswith('circuit_052/block_002/omission_late_off_target/index_099')
  wrong=dict(rows[90],block_index=1)
  try:recipe_path(wrong,preparation)
  except ValueError:pass
  else:raise AssertionError('WRONG_BLOCK_RECIPE_ACCEPTED')
  record.update(status='PREPARED_PFC_PARAMETER_BLOCK_EXECUTION',complete_recipe_cells=100,complete_condition_groups=10,recipe_authentication_passed=100,wrong_block_recipe_rejected=True,three_digit_source_paths_checked=[90,99],frozen_source_executor='task_source_prepare_610582',frozen_reader_software='corner_task_reader_prepare_610692',execution_sha256=digest(work/'execution.csv'),execution_entry_checks=6)
  code=0
 except Exception as error:record.update(error=str(error),traceback=traceback.format_exc())
 record['exit_code']=code;(work/'report.json').write_text(json.dumps(record,indent=2)+'\n')
 archive.mkdir()
 for src in sorted(work.rglob('*')):
  if not src.is_file() or '__pycache__' in src.parts:continue
  name=src.relative_to(work);dst=archive/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  files.append(dict(path=str(name),bytes=dst.stat().st_size,sha256=digest(dst)))
 (archive/'delivery.json').write_text(json.dumps(dict(status=record['status'],job_id=job,verified=True,files=files),indent=2)+'\n')
 print(json.dumps(record))
raise SystemExit(code)
PY
