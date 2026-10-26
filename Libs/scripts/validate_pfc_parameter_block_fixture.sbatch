#!/bin/bash
#SBATCH --job-name=NrnSim_PfcBlockFixtureRead
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
#SBATCH --output=/tmp/NrnSim-PfcBlockFixtureRead-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/opt/conda/bin/python3 -B - <<'PY'
from pathlib import Path
import csv,hashlib,json,os,shutil,socket,subprocess,tempfile,traceback
assert socket.gethostname().split('.')[0]=='cpu03' and not os.environ.get('SLURM_JOB_GPUS')
repo=Path('/workspace/NeuronSim');base=repo/'Output/neural_population_dynamics/benchmark/pfc_parameter_design';job=os.environ['SLURM_JOB_ID']
source_job=os.environ['NEURONSIM_PFC_FIXTURE_SOURCE_JOB'];source=base/('parameter_block_fixture_source_'+source_job)
plan=Path(os.environ['NEURONSIM_PFC_BLOCK_PLAN']);archive=base/('parameter_block_fixture_validation_'+job);assert not archive.exists()
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
record=dict(status='FAILED_PFC_PARAMETER_BLOCK_READER_FIXTURE',job_id=job,source_fixture_job=source_job,new_source_simulations=0,production_episodes=0,qualifying_source_episodes=0,scientific_fidelity_qualified=False,fixture_units=20)
code=1
archive.mkdir()
with tempfile.TemporaryDirectory(prefix='neuronsim-pfc-block-fixture-read-'+job+'.',dir='/tmp') as tmp:
 work=Path(tmp)
 try:
  original=json.loads((source/'report.json').read_text());assert original['status']=='PASS_PFC_PARAMETER_BLOCK_SOURCE_FIXTURE' and original['complete_engineering_episodes']==10
  delivered=json.loads((source/'delivery.json').read_text())
  for row in delivered['files']:
   if row['path']=='report.json':
    p=source/row['path'];assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256']
  fake=Path(original['fixture_repo']);fake_base=fake/'Output/neural_population_dynamics/benchmark/pfc_parameter_design'
  reader=(plan/'Libs/scripts/read_pfc_parameter_block.sbatch').read_text()
  reader=reader.replace("repo=Path('/workspace/NeuronSim')","repo=Path("+repr(str(fake))+")")
  reader=reader.replace("socket.gethostname().split('.')[0]=='cpu04'","socket.gethostname().split('.')[0]=='cpu03'")
  reader=reader.replace('load_plan(plan_root,require_release=True)','load_plan(plan_root)')
  reader=reader.replace("yaml.safe_load(recipe_path(task,prepared).read_text())","yaml.safe_load((prepared/task['config_path']).read_text())")
  reader=reader.replace("dimensions['coupled_core_units']==100000","dimensions['coupled_core_units']==20").replace("complete_units=100000","complete_units=20")
  script=archive/'fixture_reader.sbatch';script.write_text(reader)
  fixture_software=work/'reader_software';fixture_software.mkdir()
  manifest=json.loads((plan/'delivery.json').read_text())
  for item in manifest['files']:
   src=plan/item['path'];dst=fixture_software/item['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
   if item['path']=='Libs/scripts/read_pfc_parameter_block.sbatch':
    dst.write_text(reader);item.update(bytes=dst.stat().st_size,sha256=digest(dst))
  (fixture_software/'delivery.json').write_text(json.dumps(manifest,indent=2)+'\n')
  assert subprocess.run(['bash','-n',str(script)]).returncode==0
  env=dict(os.environ,NEURONSIM_PFC_BLOCK_PLAN=str(plan),NEURONSIM_PFC_BLOCK_GROUP='9',NEURONSIM_TASK_SOURCE_JOB=source_job,NEURONSIM_TASK_READER_SOFTWARE=str(fixture_software))
  with (archive/'reader.log').open('x') as log:result=subprocess.run(['bash',str(script)],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=work)
  readback=fake_base/('parameter_block_readback_'+job);report=json.loads((readback/'report.json').read_text())
  assert result.returncode==0 and report['status']=='PASS_PFC_TEN_CONDITION_SOURCE_READBACK',report
  assert report['complete_development_episodes']==10 and len(report['circuits'])==1 and report['complete_joint_blocks']==1
  group=report['circuits'][0];assert group['circuit_index']==52 and group['block_index']==2
  assert group['complete_development_episodes']==10 and group['temporal_bins']==10000 and group['paired_contrast_rows']==5000 and len(group['pairing'])==9
  target=readback/'circuit_052/block_002';responses={}
  for offset,binding in enumerate(group['source_bindings']):
   assert binding['task_index']==90+offset and binding['block_index']==2 and binding['complete_units']==20
   name=binding['condition_id']
   with (target/(name+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
   assert len(rows)==1000 and all(int(r['circuit_index'])==52 and int(r['block_index'])==2 and int(r['task_index'])==90+offset and r['episode_id']==binding['episode_id'] for r in rows)
   responses[name]=rows
   exported=json.loads((target/(name+'_dataset/report.json')).read_text())
   assert exported['complete_episodes']==1 and exported['verification'][0]['all_unit_bin_values_checked']==20000
   assert exported['verification'][0]['control_integral_natively_timed'] is True
  with (target/'paired_contrasts.csv').open(newline='') as f:contrasts=list(csv.DictReader(f))
  assert len(contrasts)==5000
  compared=0
  for row in contrasts:
   i=int(row['bin_index']);d=row['distractor'];cue=responses['cue_'+d][i];omission=responses['omission_'+d][i];ref=responses['cue_none'][i];ref_o=responses['omission_none'][i]
   assert int(row['circuit_index'])==52 and int(row['block_index'])==2
   assert row['cue_episode_id']==cue['episode_id'] and row['omission_episode_id']==omission['episode_id'] and row['reference_cue_episode_id']==ref['episode_id'] and row['reference_omission_episode_id']==ref_o['episode_id']
   for key,value in row.items():
    if key.endswith('_cue_minus_omission'):
     field=key.removesuffix('_cue_minus_omission');assert float(value)==float(cue[field])-float(omission[field]);compared+=1
    if key.endswith('_distractor_interaction'):
     field=key.removesuffix('_distractor_interaction');assert float(value)==(float(cue[field])-float(omission[field]))-(float(ref[field])-float(ref_o[field]));compared+=1
  # Exercise the adapted intake against the same complete engineering outputs.
  intake=(repo/'Libs/scripts/intake_pfc_parameter_block.sbatch').read_text()
  intake=intake.replace("repo=Path('/workspace/NeuronSim')","repo=Path("+repr(str(fake))+")")
  intake=intake.replace("socket.gethostname().split('.')[0]=='cpu04'","socket.gethostname().split('.')[0]=='cpu03'").replace("preflight['hostname']=='cpu04'","preflight['hostname']=='cpu03'")
  intake=intake.replace('load_plan(plan_root,require_release=True)','load_plan(plan_root)')
  intake=intake.replace("yaml.safe_load(recipe_path(task,base/'parameter_block_prepare_613107').read_text())","yaml.safe_load((base/'parameter_block_prepare_613107'/task['config_path']).read_text())")
  intake=intake.replace("dims['coupled_core_units']==100000 and dims['n_e']==80000 and dims['n_i']==20000","dims['coupled_core_units']==20 and dims['n_e']==16 and dims['n_i']==4")
  for old,new in [("complete_units']==100000","complete_units']==20"),("dims['coupled_core_units']==100000","dims['coupled_core_units']==20"),("all_unit_bin_values_checked']==100000000","all_unit_bin_values_checked']==20000"),("circuit_meta['units']==100000","circuit_meta['units']==20"),("complete_units=100000","complete_units=20"),("core_units=100000","core_units=20")]:
   intake=intake.replace(old,new)
  intake=intake.replace("software=base/item.get('reader_snapshot','circuit_reader_prepare_611402')","software=fixture_software")
  intake=intake.replace("digest(base/reader_snapshot/'Libs/scripts/read_pfc_parameter_block.sbatch')","digest(script)")
  intake=intake.replace("if row['path'] in files:","if row['path'] in files and row['path']!='Libs/scripts/read_pfc_parameter_block.sbatch':")
  intake=intake.replace("digest(software/'Libs/scripts/read_pfc_parameter_block.sbatch')","digest(script)")
  intake=intake.replace("job=os.environ['SLURM_JOB_ID']","job=os.environ['SLURM_JOB_ID']\nscript=Path("+repr(str(script))+")\nfixture_software=Path("+repr(str(fixture_software))+")")
  intake_script=archive/'fixture_intake.sbatch';intake_script.write_text(intake)
  intake_env=dict(env,NEURONSIM_INTAKE_READER_JOB=job,NEURONSIM_INTAKE_SOURCE_ARRAY=source_job,NEURONSIM_INTAKE_READER_SNAPSHOT=str(plan))
  with (archive/'intake.log').open('x') as log:accepted=subprocess.run(['bash',str(intake_script)],env=intake_env,stdout=log,stderr=subprocess.STDOUT,cwd=work)
  receipt=json.loads((fake_base/('parameter_block_delivery_intake_'+job+'.json')).read_text())
  assert accepted.returncode==0 and receipt['status']=='PASS_PFC_CIRCUIT_DELIVERY_INTAKE',receipt
  assert len(receipt['task_receipts'])==10 and len(receipt['reader_receipts'])==1
  record['intake_full_group_verified']=True
  # Modify only this disposable fixture's batch binding. The real accepted sources are untouched.
  batch=fake_base/'parameter_block_source/circuit_052/block_002/cue_none/index_090/batches'/('run_'+source_job)/'delivery.json'
  saved=batch.read_bytes();wrong=json.loads(saved)
  wrong['cases'][0]['case_id']=wrong['cases'][0]['case_id'].replace('block_002','block_001')
  rejected_script=archive/'wrong_block_reader.sbatch'
  rejected_script.write_text(reader.replace("job=os.environ['SLURM_JOB_ID']","job=os.environ['SLURM_JOB_ID']+'_wrong_block'"))
  try:
   batch.write_text(json.dumps(wrong,indent=2)+'\n')
   with (archive/'wrong_block.log').open('x') as log:bad=subprocess.run(['bash',str(rejected_script)],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=work)
   rejected=json.loads((fake_base/('parameter_block_readback_'+job+'_wrong_block')/'report.json').read_text())
   assert bad.returncode!=0 and rejected['exit_code']==1 and rejected['circuits']==[],rejected
  finally:batch.write_bytes(saved)
  for name in ['Libs/scripts/run_pfc_parameter_block_source.sbatch','Libs/scripts/read_pfc_parameter_block.sbatch','Libs/scripts/intake_pfc_parameter_block.sbatch']:
   s=(repo/name).read_text();marker=s.split("<<'",1)[1].split("'",1)[0];body=s.split("<<'"+marker+"'\n",1)[1].rsplit('\n'+marker,1)[0];compile(body,name,'exec')
  record.update(status='PASS_PFC_PARAMETER_BLOCK_SOURCE_AND_READER_FIXTURE',complete_engineering_episodes=10,temporal_rows_verified=10000,contrast_rows_verified=5000,contrast_values_recomputed=compared,exports_verified=10,all_unit_bin_values_verified=200000,exact_prefix_pairs=9,wrong_block_source_binding_rejected=True,source_launcher_sha256=digest(plan/'Libs/scripts/run_pfc_parameter_block_source.sbatch'),reader_sha256=digest(plan/'Libs/scripts/read_pfc_parameter_block.sbatch'),intake_sha256=digest(repo/'Libs/scripts/intake_pfc_parameter_block.sbatch'),readback_report_sha256=digest(readback/'report.json'))
  shutil.copy2(readback/'report.json',archive/'reader_report.json')
  code=0
 except Exception as error:record.update(error=str(error),traceback=traceback.format_exc())
 record['exit_code']=code;(archive/'report.json').write_text(json.dumps(record,indent=2)+'\n')
 files=[]
 for p in sorted(archive.iterdir()):
  if p.is_file():files.append(dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)))
 (archive/'delivery.json').write_text(json.dumps(dict(status=record['status'],job_id=job,files=files),indent=2)+'\n')
 print(json.dumps(record))
raise SystemExit(code)
PY
