#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/opt/conda/bin/python3 -B - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,socket,yaml
repo=Path('/workspace/NeuronSim');root=repo/'Output/neural_population_dynamics';job=os.environ['SLURM_JOB_ID']
assert socket.gethostname().split('.')[0]=='cpu03' and not os.environ.get('SLURM_JOB_GPUS')
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verified_delivery(a,status):
 d=json.loads((a/'delivery.json').read_text());assert d['status']==status
 seen=set();total=0
 for row in d['files']:
  n=Path(row['path']);assert not n.is_absolute() and '..' not in n.parts and str(n) not in seen
  seen.add(str(n));p=a/n;assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256'];total+=row['bytes']
 return d,seen,total
def validate_reader_execution(report,reader_job,index,step=None):
 assert report['job_id']==reader_job, 'READER_JOB_IDENTITY'
 if reader_job==f'610296_{index}':
  return
 if '_' in reader_job:
  parts=reader_job.split('_')
  assert len(parts)==2 and all(part.isdigit() for part in parts), 'ACTUAL_ARRAY_JOB_ID'
  assert int(parts[0])>0 and int(parts[1])==index, 'READER_ARRAY_TASK_IDENTITY'
  assert report.get('slurm_step_id') in (None,'batch'), 'READER_ARRAY_BATCH_IDENTITY'
 else:
  assert reader_job.isdigit() and int(reader_job)>0, 'ACTUAL_ALLOCATION_JOB_ID'
  assert step is not None and report.get('slurm_step_id')==step, 'READER_STEP_IDENTITY'

def readback_accept(item):
 index=item['index'];reader_job=os.environ.get('NEURONSIM_PFC_HISTORY_READER_JOB',f'610296_{index}');archive=root/f'multicircuit/pfc_source_reproduction/clock_refinement/profile_history_readback/complete/run_{reader_job}'
 path=root/f'benchmark/pfc_first_passage_performance/paired_history_acceptance_{reader_job}.json'
 if path.exists():
  accepted=json.loads(path.read_text());assert accepted['status']=='ACCEPTED_PFC_PAIRED_PROFILE_HISTORY_READBACK';return str(path)
 d,files,total=verified_delivery(archive,'COMPLETE_PFC_PROFILE_HISTORY_READBACK_COMPLETE')
 assert d['verified'] is True and d['job_id']==reader_job
 report=json.loads((archive/'report.json').read_text());analysis=json.loads((archive/'analysis/report.json').read_text())
 assert report['exit_code']==0 and report['error'] is None and report['status']==d['status'] and report['job_id']==reader_job and report['node']=='cpu04'
 assert all(c['exit_code']==0 for c in report['commands'])
 validate_reader_execution(report,reader_job,index,os.environ.get('NEURONSIM_PFC_HISTORY_READER_STEP'))
 seed=analysis['source_rng_seed']
 assert analysis['status']=='COMPLETE_PFC_SHARED_PROFILE_HISTORY_READBACK' and analysis['readback_group_index']==index
 group=analysis['readback_group']
 assert analysis['readback_group']==group and analysis['original_source_blocks']==1 and analysis['full_source_requests'] is True
 settings=yaml.safe_load((archive/'Libs/configs/neural_population_pfc_profile_history_readback.yaml').read_text())
 planned=yaml.safe_load(Path(report['config_source']).read_text());assert settings==planned
 history_settings=yaml.safe_load((archive/'source_config.yaml').read_text())
 import sys
 sys.path.insert(0,str(archive/'Libs'))
 for module_name in list(sys.modules):
  if module_name=='population_dynamics' or module_name.startswith('population_dynamics.'):del sys.modules[module_name]
 from population_dynamics.pfc_profile_history_readback import readback_requests
 request=readback_requests(settings,history_settings)[index]
 assert request['group_index']==index and request['source_rng_seed']==seed and request['group_name']==group
 pairs=analysis['pairs'];keys=[p['key'] for p in pairs]
 assert keys==request['profile_keys'] and len(keys)==len(set(keys))==analysis['paired_profiles']
 for clock in ['coarse','fine']:
  cost=analysis['source_costs'][clock];expected=request['source_runs'][clock]
  assert cost['history_execution_archive']==expected['archive']
  source_archive=repo/cost['history_execution_archive'];source_delivery=json.loads((source_archive/'delivery.json').read_text())
  source_job=cost['history_execution_job_id'];assert source_delivery['job_id']==source_job and source_delivery['status']=='COMPLETE_PFC_PROFILE_HISTORY_COMPLETE'
  acceptance=json.loads((root/f'benchmark/pfc_first_passage_performance/source_history_acceptance_{source_job}.json').read_text())
  assert acceptance['status']=='ACCEPTED_PFC_ORIGINAL_SOURCE_HISTORY' and acceptance['source']['delivery_sha256']==digest(source_archive/'delivery.json')
  config=json.loads((source_archive/'runtime_config.json').read_text())
  assert cost['source_binding']==config['source_binding']
  original=repo/config['source_binding']['archive']
  assert digest(original/'delivery.json')==config['source_binding']['delivery_sha256'] and digest(original/config['source_binding']['case_path']/'report.json')==config['source_binding']['report_sha256']
  assert cost['source_reconstruction_calls']==1
 moment_total=0;negative=0;undefined=0
 for pair in pairs:
  relative=Path('analysis')/pair['key']/'report.json';assert str(relative) in files
  f=json.loads((archive/relative).read_text());assert f['status']=='COMPLETE_PFC_PROFILE_HISTORY_READBACK' and f['source_rng_seed']==seed
  assert f['statistics']==pair['statistics'] and f['per_clock']==pair['per_clock'] and f['paired_selected_physical_windows']==pair['paired_selected_physical_windows']
  assert f['statistics']['paired_physical_frame_equal'] is True and f['statistics']['negative_variance_diagnostics_preserved'] is True and f['statistics']['importance_estimates_clipped'] is False
  assert f['statistics']['fine_minus_coarse_cdf_le1']==-f['statistics']['fine_minus_coarse_count_ge2']
  assert f['statistics']['scientific_fidelity_qualified'] is False and f['statistics']['population_confidence_interval_computed'] is False
  neg=False;unknown=False
  for clock in ['coarse','fine']:
   info=f['per_clock'][clock];assert info['source_label']==request['source_runs'][clock]['source_label'] and info['source_binding']==analysis['source_costs'][clock]['source_binding']
   assert info['negative_variance_diagnostics_preserved'] is True and info['importance_estimates_clipped'] is False and info['census_classification_matches'] is True
   moment_total+=info['moment_reports_recomputed']
   values=[s['latent_unit_window_variance_diagnostic'] for s in info['stratum_diagnostics']]
   neg=neg or any(value is not None and value<0 for value in values)
   unknown=unknown or any(value is None for value in values)
  negative+=int(neg);undefined+=int(unknown)
 assert analysis['new_neural_trajectories']==analysis['new_independent_source_blocks']==analysis['production_episodes']==0
 assert analysis['scientific_fidelity_qualified'] is False and analysis['source_history_population_uncertainty_established'] is False and analysis['confirmation_execution_released'] is False
 accepted=dict(status='ACCEPTED_PFC_PAIRED_PROFILE_HISTORY_READBACK',authentication_job=job,authentication_step=os.environ.get('SLURM_STEP_ID'),job_id=reader_job,reader_step=report.get('slurm_step_id'),readback_group_index=index,archive=str(archive),delivery_sha256=digest(archive/'delivery.json'),report_sha256=digest(archive/'report.json'),analysis_sha256=digest(archive/'analysis/report.json'),verified_files=len(files),verified_bytes=total,source_rng_seed=seed,readback_group=group,paired_profiles=len(keys),paired_profile_keys=keys,source_costs=analysis['source_costs'],moment_reports_recomputed_by_existing_reader=moment_total,negative_variance_profiles_preserved=negative,undefined_variance_profiles_preserved=undefined,new_accepted_paired_profiles=len(keys),new_source_law_episodes=0,new_source_simulations=0,scientific_fidelity_qualified=False,confirmation_execution_released=False,production_episodes=0,authentication_scope='All delivered bytes authenticated. Exact frozen profile sets, original source receipt/config bindings and per-profile summaries checked. Existing reader moment computations reused without new integration or simulation.',recorded_utc=datetime.now(timezone.utc).isoformat())
 with path.open('x') as f:f.write(json.dumps(accepted,indent=2)+'\n')
 return str(path)

path=readback_accept(dict(index=int(os.environ['NEURONSIM_PFC_HISTORY_READER_INDEX'])))
print(path)
PY
