#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/opt/conda/bin/python3 -B - <<'PYCODE'
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,os,socket,subprocess,traceback,yaml
mode=os.environ.get('NEURONSIM_PFC_BLOCK_INTAKE_MODE','circuit');assert mode in ('circuit','source')
assert socket.gethostname().split('.')[0]==('cpu03' if mode=='source' else 'cpu04') and not os.environ.get('SLURM_JOB_GPUS')
repo=Path('/workspace/NeuronSim');root=repo/'Output/neural_population_dynamics';base=root/'benchmark/pfc_parameter_design'
job=os.environ['SLURM_JOB_ID']
plan_root=Path(os.environ['NEURONSIM_PFC_BLOCK_PLAN'])
import sys
helper_name='Libs/population_dynamics/pfc_parameter_block_execution.py'
helper_manifest=json.loads((plan_root/'delivery.json').read_text())
helper_binding=next(row for row in helper_manifest['files'] if row['path']==helper_name)
helper_path=plan_root/helper_name
assert helper_path.stat().st_size==helper_binding['bytes'] and hashlib.sha256(helper_path.read_bytes()).hexdigest()==helper_binding['sha256']
sys.path.insert(0,str(plan_root/'Libs'))
from population_dynamics.pfc_parameter_block_execution import load_plan,group_rows,identity as episode_identity,recipe_path,source_path
allocation=load_plan(plan_root,require_release=True)
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source_array=os.environ['NEURONSIM_INTAKE_SOURCE_ARRAY']
reader_job=None;reader_delivery_sha=None;reader_report_sha=None
if mode=='circuit':
 group=int(os.environ['NEURONSIM_PFC_BLOCK_GROUP']);selected=group_rows(allocation,group)
 circuit_index=int(selected[0]['circuit_index']);block_index=int(selected[0]['block_index'])
 reader_job=os.environ['NEURONSIM_INTAKE_READER_JOB'];source_array=os.environ['NEURONSIM_INTAKE_SOURCE_ARRAY']
 reader_snapshot=os.environ['NEURONSIM_INTAKE_READER_SNAPSHOT']
 reader_archive=base/('parameter_block_readback_'+reader_job)
 reader_delivery_sha=digest(reader_archive/'delivery.json');reader_report_sha=digest(reader_archive/'report.json')
 reader_delivery=json.loads((reader_archive/'delivery.json').read_text());reader_report=json.loads((reader_archive/'report.json').read_text())
 assert reader_delivery['job_id']==reader_job and reader_delivery['verified'] is True and reader_delivery['status']=='PASS_PFC_TEN_CONDITION_SOURCE_READBACK'
 assert reader_report['job_id']==reader_job and reader_report['exit_code']==0 and reader_report['source_job']==source_array and reader_report['status']==reader_delivery['status']
 assert len(reader_report['circuits'])==1 and reader_report['circuits'][0]['circuit_index']==circuit_index and reader_report['circuits'][0]['block_index']==block_index
 # Authenticate the source-binding report and frozen execution code before reusing
 # its source-file checks. The remaining delivered datasets are checked once below.
 reader_manifest={row['path']:row for row in reader_delivery['files']}
 assert len(reader_manifest)==len(reader_delivery['files'])
 for name in ['report.json','Libs/scripts/read_pfc_parameter_block.sbatch','resource_preflight.json']:
  row=reader_manifest[name]
  assert (reader_archive/name).stat().st_size==row['bytes'] and digest(reader_archive/name)==row['sha256']
 assert digest(reader_archive/'Libs/scripts/read_pfc_parameter_block.sbatch')==digest(base/reader_snapshot/'Libs/scripts/read_pfc_parameter_block.sbatch')
 preflight=json.loads((reader_archive/'resource_preflight.json').read_text())
 assert preflight['status']=='PASS' and preflight['hostname']=='cpu04' and preflight['slurm_job_id']==reader_job
 reader_started=min(datetime.fromisoformat(sample['recorded_utc']).timestamp() for sample in preflight['samples'])
 source_bindings={binding['source_archive']:binding for binding in reader_report['circuits'][0]['source_bindings']}
 assert len(source_bindings)==10
else:
 source_index=int(os.environ['NEURONSIM_PFC_BLOCK_SOURCE_INDEX']);assert 0<=source_index<len(allocation)
 selected=[allocation[source_index]]
 circuit_index=int(selected[0]['circuit_index']);block_index=int(selected[0]['block_index'])
def verified_delivery(archive,status):
 d=json.loads((archive/'delivery.json').read_text());assert d['status']==status
 seen=set();total=0
 reuse=mode=='circuit' and archive!=reader_archive
 if reuse:
  binding=source_bindings[str(archive)]
  assert binding['complete_units']==100000 and binding['export']=='EXPORTED_VERIFIED_PFC_DEVELOPMENT_DATASET'
  assert digest(archive/'delivery.json')==binding['delivery_sha256']
  stat=(archive/'delivery.json').stat()
  assert max(stat.st_mtime,stat.st_ctime)<=reader_started,str(archive/'delivery.json')
 for row in d['files']:
  name=Path(row['path']);assert not name.is_absolute() and '..' not in name.parts and str(name) not in seen
  seen.add(str(name));p=archive/name;stat=p.stat()
  assert stat.st_size==row['bytes'],str(p)
  if reuse:assert max(stat.st_mtime,stat.st_ctime)<=reader_started,str(p)
  else:assert digest(p)==row['sha256'],str(p)
  total+=row['bytes']
 return d,seen,total
def task_accept(item):
 index=item['index'];task=allocation[index];circuit=int(task['circuit_index'])
 source_job=str(item['array_job'])
 receipt=base/f'parameter_block_source_acceptance_{source_job}_{index}.json'
 if receipt.exists():
  r=json.loads(receipt.read_text());assert r['status']=='ACCEPTED_COMPLETE_FULL_SCALE_TASK_SOURCE_ARCHIVE' and r['native_job_id']==item['raw'] and r['episode_id']==task['episode_id'], 'PFC_BLOCK_ACCEPTED_SOURCE_IDENTITY_MISMATCH'
  accepted_origin=source_path(base,task)/('run_'+item['raw'])
  assert r['source_delivery_sha256']==digest(accepted_origin/'delivery.json') and r['source_report_sha256']==digest(accepted_origin/'source_run/report.json')
  return str(receipt)
 identity_record=episode_identity(task);name=task['condition_id']
 expected=yaml.safe_load(recipe_path(task,base/'parameter_block_prepare_613107').read_text())
 origin=source_path(base,task)/('run_'+item['raw'])
 identity=identity_record
 batch=origin.parent/'batches'/('run_'+item['raw'])/'delivery.json';plan=json.loads(batch.read_text())
 assert plan['status']=='COMPLETE' and plan['exit_code']==0 and plan['srun_exit_code']==0 and len(plan['cases'])==1 and plan['job_id']==item['raw']
 case=plan['cases'][0];assert case['node']=='cpu03'
 assert Path(case['archive'])==origin and case['case_id']==identity['episode_id'] and case['source_execution_complete'] is True and case['exit_code']==0
 d=json.loads((origin/'delivery.json').read_text());status=d['status'];assert status=='ARCHIVED_PFC_SOURCE_EXECUTION'
 d,files,total=verified_delivery(origin,status)
 assert d['source_execution_complete'] is True and d['exit_code']==0 and d['job_id']==item['raw'] and d['case_id']==identity['episode_id'] and d['result_status']=='PFC_SOURCE_EXECUTION_COMPLETE'
 config=yaml.safe_load((origin/'source_run/config.yaml').read_text());assert config==expected and config['benchmark_episode_candidate']['identity']==identity
 report=json.loads((origin/'source_run/report.json').read_text());dims=report['dimensions']
 assert report['status']=='PFC_SOURCE_EXECUTION_COMPLETE' and report['source_execution_complete'] is True and report['case_id']==identity['episode_id']
 assert report['completed_case_count']==1 and report['completed_duration_ticks']==100000
 assert dims['coupled_core_units']==100000 and dims['n_e']==80000 and dims['n_i']==20000 and dims['duration_ticks']==100000 and dims['duration_seconds']==5 and dims['dt_seconds']==0.00005
 assert report['seeds']==dict(initial_rng_seed=identity['initial_state_seed'],external_noise_seed=identity['background_seed'])
 assert report['random_stream_policy']=='independent_initial_external'
 assert report['resolved_config']==config and report['config_sha256']==digest(origin/'source_run/config.yaml')==case['config_sha256']
 checks=report['validations'];assert len(checks)==29 and all(v['passed'] is True for v in checks.values())
 for key,filename in dict(drive_sha256='pfc_native_drive.py',executor_sha256='pfc_deterministic_replay.py',kernel_parameterization_sha256='pfc_kernel_parameters.py',runner_sha256='pfc_source_execution.py').items():
  assert digest(origin/'Libs/population_dynamics'/filename)==report['code'][key]==digest(base/'task_source_prepare_610582'/'Libs/population_dynamics'/filename)
 assert report['source_entry_sha256']=='416a914b9a358ae4b685d1a7119d14d777f31c70253f88e01b85d7ede993d720'
 assert report['production_episodes']==0 and report['scientific_fidelity_qualified'] is False
 r=dict(status='ACCEPTED_COMPLETE_FULL_SCALE_TASK_SOURCE_ARCHIVE',slurm_array_job=f'{source_job}_{index}',native_job_id=item['raw'],acceptance_job=job,acceptance_step=os.environ.get('SLURM_STEP_ID'),bulk_source_file_authentication_job=reader_job if mode=='circuit' else job,bulk_source_authentication_reused=mode=='circuit',source_reader_delivery_sha256=reader_delivery_sha,source_reader_report_sha256=reader_report_sha,scheduler_timing_scope='Caller-supplied scheduler elapsed when available; complete batch/source delivered exit codes checked; circuit mode also binds accepted reader.',scheduler_elapsed_seconds=item['elapsed'],circuit_index=circuit,block_index=int(task['block_index']),task_index=index,condition=name,episode_id=identity['episode_id'],archive=str(origin),source_delivery_sha256=digest(origin/'delivery.json'),batch_delivery_sha256=digest(batch),verified_files=len(files),verified_bytes=total,core_units=100000,duration_ms=5000,simulator_cost=report['simulator_cost'],execution_observation=json.loads((origin/'source_run/engine_observation.json').read_text()),config_sha256=report['config_sha256'],source_report_sha256=digest(origin/'source_run/report.json'),source_validation_checks_passed=len(checks),complete_source_episodes=1,task_readback_complete=False,scientific_fidelity_qualified=False,parameter_domain_qualified=False,production_episodes=0,recorded_utc=datetime.now(timezone.utc).isoformat())
 with receipt.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
 return str(receipt)
def reader_accept(item):
 reader_job=str(item['job_id']);circuit_index=item['circuit'];group=item['group']
 archive=base/('parameter_block_readback_'+reader_job);receipt=base/('parameter_block_readback_acceptance_'+reader_job+'.json')
 if receipt.exists():
  r=json.loads(receipt.read_text());assert r['status']=='ACCEPTED_PFC_TEN_CONDITION_TASK_READBACK';return str(receipt)
 d,files,total=verified_delivery(archive,'PASS_PFC_TEN_CONDITION_SOURCE_READBACK')
 file_sha={row['path']:row['sha256'] for row in d['files']}
 assert d['job_id']==reader_job and d['verified'] is True
 report=json.loads((archive/'report.json').read_text())
 assert report['status']==d['status'] and report['job_id']==d['job_id'] and report['source_job']==str(item.get('source_job','610646')) and report['exit_code']==0
 assert report['complete_development_episodes']==10 and report['complete_parameter_points']==1 and len(report['circuits'])==1
 for key in ['scientific_fidelity_qualified','source_probability_law_qualified','parameter_domain_qualified','task_success_assessed','confidence_intervals','p_values']:assert report[key] is False,key
 assert report['new_source_simulations']==report['production_episodes']==0
 software=base/item.get('reader_snapshot','circuit_reader_prepare_611402')
 frozen_delivery=json.loads((software/'delivery.json').read_text())
 for row in frozen_delivery['files']:
  if row['path'].startswith('Libs/') and row['path'] in files:
   assert digest(archive/row['path'])==row['sha256'],row['path']
 assert 'Libs/scripts/read_pfc_parameter_block.sbatch' in files
 assert digest(archive/'Libs/scripts/read_pfc_parameter_block.sbatch')==digest(software/'Libs/scripts/read_pfc_parameter_block.sbatch')
 circuit=report['circuits'][0];assert circuit['circuit_index']==circuit_index and circuit['block_index']==block_index and circuit['complete_development_episodes']==10
 assert circuit['temporal_bins']==10000 and circuit['paired_contrast_rows']==5000 and circuit['independent_noise_blocks']==1
 prepared=base/'parameter_block_prepare_613107'/f'recipes/circuit_{circuit_index:03d}/block_{block_index:03d}'
 design=json.loads((prepared/'report.json').read_text());assert len(design['conditions'])==10 and design['production_eligible'] is False
 assert [item['identity'] for item in design['conditions']]==[episode_identity(row) for row in selected]
 conditions=design['conditions'];names=[c['condition']['condition_id'] for c in conditions]
 assert len(names)==len(set(names))==10
 bindings=circuit['source_bindings'];assert [b['condition_id'] for b in bindings]==names
 sources=[];events=0;all_unit_bin_values=0
 for index,(condition,binding) in enumerate(zip(conditions,bindings)):
  name=names[index];identity=condition['identity'];episode=identity['episode_id']
  task=selected[index];source_index=int(task['task_index'])
  accepted=json.loads((base/f"parameter_block_source_acceptance_{item['source_job']}_{source_index}.json").read_text())
  assert accepted['block_index']==block_index and accepted['task_index']==source_index and binding['episode_id']==episode
  assert accepted['status'] in ['ACCEPTED_FIRST_COMPLETE_FULL_SCALE_TASK_SOURCE_ARCHIVE','ACCEPTED_COMPLETE_FULL_SCALE_TASK_SOURCE_ARCHIVE']
  assert accepted['episode_id']==episode and accepted['archive']==binding['source_archive'] and accepted['source_delivery_sha256']==binding['delivery_sha256']
  origin=Path(binding['source_archive']);assert digest(origin/'delivery.json')==binding['delivery_sha256']
  source_report=json.loads((origin/'source_run/report.json').read_text());assert source_report['source_execution_complete'] is True
  dims=source_report['dimensions'];assert dims['coupled_core_units']==100000 and dims['duration_ticks']==100000 and dims['dt_seconds']==.00005 and dims['duration_seconds']==5
  if 'source_report_sha256' in accepted:assert digest(origin/'source_run/report.json')==accepted['source_report_sha256']
  assert binding['complete_units']==100000 and binding['export']=='EXPORTED_VERIFIED_PFC_DEVELOPMENT_DATASET'
  prefix=Path(f'circuit_{circuit_index:03d}/block_{block_index:03d}/{name}_dataset')
  assert str(prefix/'report.json') in files and str(prefix/'dataset/dataset.json') in files
  exported=json.loads((archive/prefix/'report.json').read_text())
  assert exported['status']==binding['export'] and exported['source_paths']==[str(origin)]
  assert exported['registered_source_cases']==exported['selected_source_cases']==exported['complete_episodes']==1
  assert exported['missing_registered_cases']==[] and len(exported['verification'])==1
  verification=exported['verification'][0]
  assert verification['case_id']==episode and verification['status']=='PASS' and verification['event_values']==binding['events']
  assert verification['all_unit_bin_values_checked']==100000000 and verification['control_integral_natively_timed'] is True and verification['privileged_states_exposed_by_default_window'] is False
  assert exported['source_fidelity_qualified'] is False and exported['scientific_fidelity_qualified'] is False
  assert exported['production_episodes']==exported['source_episodes_added_to_campaign']==exported['new_source_simulations']==0
  catalog_path=archive/prefix/'dataset/dataset.json';assert digest(catalog_path)==exported['dataset_catalog_sha256']
  catalog=json.loads(catalog_path.read_text());assert catalog['complete_episodes']==1 and catalog['source_fidelity_qualified'] is False
  binding_meta=catalog['binding'];assert binding_meta['sample_unit']=='complete_mechanistic_episode' and binding_meta['schema']=='neural_circuit_episodes_v1' and binding_meta['scenario_id']=='pfc_working_memory'
  assert binding_meta['resolved_source_config']==yaml.safe_load((prepared/condition['config_path']).read_text())
  assert binding_meta['config_sha256']==source_report['config_sha256']
  assert len(binding_meta['source_bindings'])==1
  source_binding=binding_meta['source_bindings'][0]
  assert source_binding['case_id']==episode and source_binding['path']==str(origin/'source_run')
  assert source_binding['delivery_sha256']==binding['delivery_sha256'] and source_binding['report_sha256']==digest(origin/'source_run/report.json')
  assert len(catalog['circuits'])==1 and len(catalog['shards'])==1
  circuit_id,circuit_meta=next(iter(catalog['circuits'].items()));assert circuit_meta['units']==100000
  for field in ['parameter_identity','connectivity_identity']:assert circuit_meta['provenance'][field]==binding_meta['resolved_source_config']['benchmark_circuit_candidate'][field]
  shard=catalog['shards'][0];assert len(shard['episodes'])==1
  ep=shard['episodes'][0]
  assert ep['episode_id']==episode and ep['circuit_id']==circuit_id and ep['split']=='development' and ep['duration_ticks']==100000 and ep['dt_ms']==.05 and ep['event_count']==binding['events']
  for payload in [circuit_meta,shard]:
   name_path=prefix/'dataset'/payload['path'];assert str(name_path) in files and file_sha[str(name_path)]==payload['sha256']
  origins_path=Path(f'circuit_{circuit_index:03d}/block_{block_index:03d}/{name}_origins.json');assert str(origins_path) in files
  origins=json.loads((archive/origins_path).read_text());assert len(origins)==1
  assert origins[0]['archive']==str(origin) and origins[0]['case_id']==episode and origins[0]['delivery_sha256']==binding['delivery_sha256']
  csv_path=Path(f'circuit_{circuit_index:03d}/block_{block_index:03d}/{name}.csv');assert str(csv_path) in files
  with (archive/csv_path).open(newline='') as stream:
   rows=list(csv.DictReader(stream))
  assert len(rows)==1000
  for bin_index,row in enumerate(rows):
   assert row['condition_id']==name and row['episode_id']==episode and int(row['bin_index'])==bin_index
   assert int(row['circuit_index'])==circuit_index and int(row['block_index'])==block_index and int(row['task_index'])==source_index
   assert int(row['start_tick'])==100*bin_index and int(row['stop_tick'])==100*(bin_index+1)
   assert float(row['start_ms'])==5*bin_index and float(row['stop_ms'])==5*(bin_index+1)
  events+=binding['events'];all_unit_bin_values+=verification['all_unit_bin_values_checked']
  sources.append(dict(condition_id=name,episode_id=episode,source_archive=str(origin),delivery_sha256=binding['delivery_sha256'],event_count=binding['events'],complete_units=100000,temporal_bins=1000))
 pairings=circuit['pairing'];assert [p['condition_id'] for p in pairings]==names[1:]
 for p in pairings:
  assert p['reference_condition']=='cue_none' and p['initial_and_final_rng_exact'] is True and p['exact_prefix_events']>=0
  expected_tick=2000 if p['condition_id'].startswith('omission_') else 48000 if p['condition_id']=='cue_late_off_target' else 8000
  assert p['first_changed_current_tick']==expected_tick
 pair_path=Path(f'circuit_{circuit_index:03d}/block_{block_index:03d}/paired_contrasts.csv');assert str(pair_path) in files
 with (archive/pair_path).open(newline='') as stream:
  contrasts=list(csv.DictReader(stream))
 expected={(name,i) for name in ['none','aligned','off_target','antipodal','late_off_target'] for i in range(1000)}
 observed={(row['distractor'],int(row['bin_index'])) for row in contrasts}
 assert len(contrasts)==5000 and observed==expected
 for row in contrasts:
  i=int(row['bin_index']);assert int(row['start_tick'])==i*100 and int(row['stop_tick'])==(i+1)*100
  assert int(row['circuit_index'])==circuit_index and int(row['block_index'])==block_index and row['block_id']==selected[0]['block_id']
  pair={r['condition_id']:r['episode_id'] for r in selected}
  assert row['cue_episode_id']==pair['cue_'+row['distractor']] and row['omission_episode_id']==pair['omission_'+row['distractor']]
  assert row['reference_cue_episode_id']==pair['cue_none'] and row['reference_omission_episode_id']==pair['omission_none']
 r=dict(status='ACCEPTED_PFC_TEN_CONDITION_TASK_READBACK',job_id=reader_job,authentication_job=job,archive=str(archive),delivery_sha256=digest(archive/'delivery.json'),report_sha256=digest(archive/'report.json'),verified_files=len(files),verified_bytes=total,complete_development_episodes=10,new_accepted_readback_episodes=10,complete_parameter_points=1,circuit_index=circuit_index,block_index=block_index,independent_noise_blocks=1,temporal_bins=10000,paired_contrast_rows=5000,all_unit_bin_values_checked_by_existing_export=all_unit_bin_values,events_verified_by_existing_export=events,sources=sources,pairing=pairings,scheduler_elapsed_seconds=item['elapsed'],scheduler_timing_scope='Compute-node sacct connection unavailable; caller can later bind management scheduler accounting without repeating data validation.',task_success_assessed=False,source_probability_law_qualified=False,parameter_domain_qualified=False,scientific_fidelity_qualified=False,production_episodes=0,new_source_simulations=0,numerical_reader_and_exports_repeated=False,authentication_scope='All delivered files authenticated. Frozen ten-condition identities, accepted source receipts, catalog maps, complete temporal and contrast coverage, and existing export/pairing results checked. Existing event and full-unit numerical checks reused.',recorded_utc=datetime.now(timezone.utc).isoformat())
 with receipt.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
 return str(receipt)


if mode=='source':
 summary=dict(status='FAILED_PFC_PARAMETER_BLOCK_SOURCE_INTAKE',job_id=job,step_id=os.environ.get('SLURM_STEP_ID'),task_index=source_index,task_receipts=[],reader_receipts=[],new_source_simulations=0,production_episodes=0,failures=[])
 try:
  elapsed=os.environ.get('NEURONSIM_PFC_BLOCK_SOURCE_ELAPSED')
  summary['task_receipts'].append(task_accept(dict(index=source_index,raw=os.environ['NEURONSIM_PFC_BLOCK_SOURCE_RAW_JOB'],elapsed=int(elapsed) if elapsed is not None else None,array_job=source_array)))
  summary['status']='PASS_PFC_PARAMETER_BLOCK_SOURCE_INTAKE'
 except Exception as error:summary['failures'].append(dict(error=str(error),traceback=traceback.format_exc()))
else:
 summary=dict(status='PFC_CIRCUIT_INTAKE_INCOMPLETE',job_id=job,circuit_index=circuit_index,block_index=block_index,reader_job=reader_job,scheduler_success_dependency='afterok:'+reader_job,source_raw_jobs_bound_by='Reader source bindings plus complete batch/source deliveries',task_receipts=[],reader_receipts=[],new_source_simulations=0,production_episodes=0,failures=[])
 try:
  for offset,task in enumerate(selected):
   index=int(task['task_index']);condition=reader_report['circuits'][0]['source_bindings'][offset]
   raw=Path(condition['source_archive']).name.removeprefix('run_')
   summary['task_receipts'].append(task_accept(dict(index=index,raw=raw,elapsed=None,array_job=source_array)))
  summary['reader_receipts'].append(reader_accept(dict(job_id=reader_job,circuit=circuit_index,group=group,source_job=source_array,reader_snapshot=reader_snapshot,elapsed=None)))
  summary['status']='PASS_PFC_CIRCUIT_DELIVERY_INTAKE'
 except Exception as error:
  summary['status']='FAILED_PFC_CIRCUIT_DELIVERY_INTAKE';summary['failures'].append(dict(error=str(error),traceback=traceback.format_exc()))
summary['recorded_utc']=datetime.now(timezone.utc).isoformat()
suffix=job+('_step_'+os.environ['SLURM_STEP_ID'] if 'SLURM_STEP_ID' in os.environ else '')
if mode=='source':suffix+=f'_index_{source_index:03d}'
out=base/(('parameter_block_source_intake_' if mode=='source' else 'parameter_block_delivery_intake_')+suffix+'.json')
with out.open('x') as f:f.write(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
raise SystemExit(0 if not summary['failures'] else 1)
PYCODE
