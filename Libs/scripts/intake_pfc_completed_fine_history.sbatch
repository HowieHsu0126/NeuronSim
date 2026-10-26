#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/opt/conda/bin/python3 -B - <<'PYCODE'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,socket,traceback,yaml
assert socket.gethostname().split('.')[0]=='cpu03' and not os.environ.get('SLURM_JOB_GPUS')
repo=Path('/workspace/NeuronSim');root=repo/'Output/neural_population_dynamics';base=root/'benchmark/pfc_parameter_design'
job=os.environ['SLURM_JOB_ID']
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verified_delivery(archive,status):
 d=json.loads((archive/'delivery.json').read_text())
 assert d['status']==status
 seen=set();total=0
 for row in d['files']:
  name=Path(row['path']);assert not name.is_absolute() and '..' not in name.parts and str(name) not in seen
  seen.add(str(name));p=archive/name
  assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],str(p)
  total+=row['bytes']
 return d,seen,total
def validate_history_source(original,binding,analysis,label,seed):
 assert original['source_execution_complete'] is True
 assert all(value['passed'] is True for value in original['validations'].values())
 rec=analysis['reconstruction'];drive=analysis['reconstruction_drive']
 if label=='native':
  dims=original['dimensions']
  assert original['status']=='PFC_SOURCE_EXECUTION_COMPLETE'
  assert original['seeds']=={'source_rng_seed':seed} and binding['refinement_seeds'] is None
  assert dims['coupled_core_units']==100000 and dims['n_e']==80000 and dims['n_i']==20000
  assert dims['dt_seconds']==.00005 and dims['duration_seconds']==5 and dims['duration_ticks']==100000
  assert drive['seeds']==original['seeds'] and drive['random_stream_policy']==original['random_stream_policy']=='source_global'
  assert drive['source_entry_sha256']==original['source_entry_sha256']
  assert original['inputs_used']['input_mode']=='native_numpy_stream'
  assert original['inputs_used']['ticks_consumed']==100000
  totals={p:original['inputs_used'][p]['events'] for p in ['E','I']}
  assert totals==original['source_external_event_totals']
  assert totals==original['validations']['complete_source_input_consumption']['event_totals']
  assert original['inputs_used']['E']['values']==8000000000 and original['inputs_used']['I']['values']==2000000000
  dt=dims['dt_seconds']
 else:
  assert label in ('half_A','half_B','quarter_A','quarter_B')
  assert original['status']=='COMPLETE_COUPLED_CLOCK_DEVELOPMENT_CASE' and original['coupled_core_units']==100000
  assert original['source_rng_seed']==seed and original['dt_ms']=={'half_A':.025,'half_B':.025,'quarter_A':.0125,'quarter_B':.0125}[label]
  assert original['duration_ms']==5000 and original['completed_duration_ticks']==round(5000/original['dt_ms'])
  assert original['refinement_seeds']==binding['refinement_seeds'] and len(binding['refinement_seeds'])==2
  assert drive['coupling']['coarse_source']['seeds']['source_rng_seed']==seed
  assert drive['coupling']['refinement_seeds']==original['refinement_seeds']
  totals=original['drive']['external_event_totals'];dt=original['drive']['dt_seconds']
  assert totals==original['validations']['complete_source_input_consumption']['event_totals']
 assert rec['ticks_reconstructed']==original['completed_duration_ticks']
 assert rec['source_events_consumed']==original['validations']['canonical_event_schema']['event_count']
 assert drive['ticks_generated']==original['completed_duration_ticks'] and drive['dt_seconds']==dt
 assert drive['external_event_totals']==totals

def history_accept(item):
 index=item['index'];origin=root/f'multicircuit/pfc_source_reproduction/clock_refinement/profile_history/complete/run_610289_{index}'
 receipt=root/f'benchmark/pfc_first_passage_performance/source_history_acceptance_610289_{index}.json'
 if receipt.exists():
  r=json.loads(receipt.read_text());assert r['status']=='ACCEPTED_PFC_ORIGINAL_SOURCE_HISTORY'
  assert r['source']['native_job_id']==item['raw'] and r['source']['source_rng_seed']==item['seed'] and r['source']['source_label']==item['label']
  assert r['source']['delivery_sha256']==digest(origin/'delivery.json') and r['source']['analysis_sha256']==digest(origin/'analysis/report.json')
  return str(receipt)
 d,files,total=verified_delivery(origin,'COMPLETE_PFC_PROFILE_HISTORY_COMPLETE')
 assert d['job_id']==f'610289_{index}' and d['verified'] is True
 report=json.loads((origin/'report.json').read_text());assert report['exit_code']==0 and report['error'] is None and report['status']==d['status'] and report['job_id']==d['job_id'] and report['node']=='cpu03'
 assert all(c['exit_code']==0 for c in report['commands'])
 analysis=json.loads((origin/'analysis/report.json').read_text());config=json.loads((origin/'runtime_config.json').read_text());binding=config['source_binding']
 assert analysis['status']=='COMPLETE_PFC_PROFILE_SOURCE_HISTORY' and analysis['source_rng_seed']==item['seed'] and analysis['source_label']==item['label']
 keys=analysis['completed_profile_keys'];frames=analysis['frame_reports']
 assert keys==analysis['requested_profile_keys'] and len(keys)>0 and len(keys)==len(set(keys))==len(frames) and {f['key'] for f in frames}==set(keys)
 frozen=yaml.safe_load((origin/'Libs/configs/neural_population_pfc_profile_history.yaml').read_text())
 assert {k:v for k,v in config.items() if k!='source_binding'}==frozen
 planned=yaml.safe_load(Path(report['config_source']).read_text());assert frozen==planned
 import sys
 sys.path.insert(0,str(origin/'Libs'))
 from population_dynamics.pfc_profile_history import source_requests
 expected_keys=[request['key'] for request in source_requests(frozen,item['label'],pilot=False,seed=item['seed'])]
 expected_count={'native':35,'half_A':35,'quarter_A':18,'half_B':36,'quarter_B':18}[item['label']]
 assert keys==expected_keys and len(keys)==expected_count and [f['key'] for f in frames]==expected_keys
 source=repo/binding['archive'];assert digest(source/'delivery.json')==binding['delivery_sha256'] and digest(source/binding['case_path']/'report.json')==binding['report_sha256']
 original=json.loads((source/binding['case_path']/'report.json').read_text())
 validate_history_source(original,binding,analysis,item['label'],item['seed'])
 rec=analysis['reconstruction'];drive=analysis['reconstruction_drive']
 assert analysis['source_reconstruction_calls']==1 and analysis['distinct_source_loads']==1 and analysis['tape_validations']==1
 inner=0
 for frame in frames:
  name=Path('analysis')/frame['report'];assert str(name) in files
  f=json.loads((origin/name).read_text())
  assert f['status']=='COMPLETE_PFC_PROFILE_CONDITIONAL_HISTORY' and f['source_rng_seed']==item['seed'] and f['source_label']==item['label']
  assert f['census_classification_matches'] is True and f['importance_estimates_clipped'] is False and f['negative_variance_diagnostics_preserved'] is True
  assert f['inner_trials']==frame['inner_trials'] and f['integrated_histories']==frame['integrated_histories'] and f['sample_mean']==frame['sample_mean']
  assert f['scientific_fidelity_qualified'] is False and f['population_confidence_interval_computed'] is False and f['source_history_population_uncertainty_established'] is False
  inner+=f['inner_trials']
 assert inner==analysis['total_inner_trials'] and analysis['new_full_network_episodes']==0 and analysis['new_independent_source_blocks']==0
 src=dict(job_id=d['job_id'],native_job_id=item['raw'],archive=str(origin),status=d['status'],delivery_sha256=digest(origin/'delivery.json'),report_sha256=digest(origin/'report.json'),analysis_sha256=digest(origin/'analysis/report.json'),verified_files=len(files),verified_bytes=total,wall_seconds=report['wall_seconds'],source_rng_seed=item['seed'],source_label=item['label'],analysis_status=analysis['status'],file_authentication_job=job,scheduler_elapsed_seconds=item['elapsed'],scheduler_elapsed_status='RECORDED' if item['elapsed'] is not None else 'NOT_CAPTURED_BY_INTAKE',complete_profile_mappings=len(keys),reconstruction_seconds=analysis['reconstruction_seconds'],reconstruction=rec,total_inner_trials=inner,distinct_selected_physical_windows=analysis['distinct_selected_physical_windows'],distinct_integrated_references=analysis['distinct_integrated_references'],peak_rss_kib=analysis['peak_rss_kib'],reconstruction_drive=drive,source_binding=binding,total_inner_trials_scope='Sum across profile requests; not independent samples or unique executed trials.')
 r=dict(status='ACCEPTED_PFC_ORIGINAL_SOURCE_HISTORY',authentication_job=job,source=src,new_accepted_history_tasks=1,new_accepted_paired_profiles=0,new_source_law_episodes=0,scientific_fidelity_qualified=False,confirmation_execution_released=False,production_episodes=0,full_task_speedup_vs_same_input_baseline_measured=False,authentication_scope='All delivered bytes and SHA256, original source bindings, reconstructed ticks/events/input totals and complete frame summaries checked; no integration or source simulation repeated.',recorded_utc=datetime.now(timezone.utc).isoformat())
 with receipt.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
 return str(receipt)

index=int(os.environ['NEURONSIM_PFC_HISTORY_SOURCE_INDEX'])
assert 0<=index<160
label=['native','half_A','quarter_A','half_B','quarter_B'][index%5]
# Intake is submitted afterok for the exact source. Authentic source reports
# below remain the acceptance authority; scheduler history can expire before
# serialized intake. An unavailable elapsed value is not an output failure.
elapsed=os.environ.get('NEURONSIM_PFC_HISTORY_SOURCE_ELAPSED')
item=dict(index=index,raw=os.environ['NEURONSIM_PFC_HISTORY_SOURCE_RAW_JOB'],elapsed=int(elapsed) if elapsed is not None else None,label=label,seed=95001+index//5)

summary=dict(status='FAILED_PFC_FINE_SOURCE_HISTORY_INTAKE',job_id=job,history_receipts=[],failures=[],new_source_simulations=0,production_episodes=0)
try:
 summary['history_receipts'].append(history_accept(item))
 summary['status']='PASS_PFC_FINE_SOURCE_HISTORY_INTAKE'
except Exception as error:summary['failures'].append(dict(error=str(error),traceback=traceback.format_exc()))
summary['recorded_utc']=datetime.now(timezone.utc).isoformat()
suffix=job+('_step_'+os.environ['SLURM_STEP_ID'] if 'SLURM_STEP_ID' in os.environ else '')+f'_index_{index:03d}'
with (root/'benchmark/pfc_first_passage_performance'/('fine_source_history_intake_'+suffix+'.json')).open('x') as f:f.write(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
raise SystemExit(0 if not summary['failures'] else 1)
PYCODE
