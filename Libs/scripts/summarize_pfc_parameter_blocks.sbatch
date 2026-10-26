#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/opt/conda/bin/python3 -B - <<'PYCODE'
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,math,os,shutil,socket,sys,tempfile,traceback
import numpy as np
assert socket.gethostname().split('.')[0]=='cpu04' and not os.environ.get('SLURM_JOB_GPUS')
repo=Path('/workspace/NeuronSim');base=repo/'Output/neural_population_dynamics/benchmark/pfc_parameter_design';job=os.environ['SLURM_JOB_ID']
archive=base/('parameter_block_response_summary_'+job);assert not archive.exists()
reader_array=os.environ['NEURONSIM_PFC_BLOCK_READER_ARRAY']
reader_jobs=json.loads(os.environ['NEURONSIM_PFC_BLOCK_READER_JOBS']) if 'NEURONSIM_PFC_BLOCK_READER_JOBS' in os.environ else [reader_array+'_'+str(group) for group in range(10)]
assert isinstance(reader_jobs,list) and len(reader_jobs)==10 and len(set(reader_jobs))==10
assert all(isinstance(value,str) and value.replace('_','').isdecimal() for value in reader_jobs)
requested=list(zip(reader_jobs,[23,23,63,63,74,74,86,86,52,52]))

expected_episodes=10*len(requested)
reuse_specs=[]
record=dict(status='FAILED_PFC_PARAMETER_BLOCK_RESPONSE_SUMMARY',job_id=job,source_readers=[reader for reader,_ in requested],source_circuits=[circuit for _,circuit in requested],complete_development_episodes=0,new_source_simulations=0,production_episodes=0,working_memory_qualified=False,scientific_fidelity_qualified=False,parameter_domain_qualified=False,task_protocol_frozen=False,source_probability_law_qualified=False,confidence_intervals=False,p_values=False,independent_noise_blocks_per_circuit=2,scope='Descriptive complete-task response after measured source cue removal. Authenticate archived current segments and all complete temporal bins. All conditions retained; no success threshold.',boundary_semantics='First complete5ms bin following archived cue-current removal; late-delay4000ms unchanged.')
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
code=1
with tempfile.TemporaryDirectory(prefix='neuronsim-pfc-task-summary-'+job+'.',dir='/tmp') as tmp:
 work=Path(tmp)
 try:
  first=base/'task_boundary_readback_611415'
  manifest=json.loads((first/'delivery.json').read_text())
  for row in manifest['files']:
   if row['path'].startswith('Libs/'):
    src=first/row['path'];dst=work/row['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    assert dst.stat().st_size==row['bytes'] and digest(dst)==row['sha256']
  sys.path.insert(0,str(work/'Libs'))
  from population_dynamics.pfc_temporal_readouts import delay_summary
  from population_dynamics.pfc_delay_analysis import PHASES
  summaries=[];source_receipts=[];seen=set();checked_bytes=0;measured_currents=[];reused={};reused_sources=[];phase_rows=[];temporal_cases=[]
  for spec in reuse_specs:
   src=base/spec['archive'];assert digest(src/'delivery.json')==spec['delivery_sha256'] and digest(src/'report.json')==spec['report_sha256']
   prior=json.loads((src/'report.json').read_text());delivery=json.loads((src/'delivery.json').read_text())
   assert prior['status']=='COMPLETE_PFC_MEASURED_CURRENT_DELAY_SUMMARY' and prior['exit_code']==0 and delivery['verified']
   entry=next(row for row in delivery['files'] if row['path']=='report.json')
   assert entry['sha256']==spec['report_sha256'] and entry['bytes']==(src/'report.json').stat().st_size
   for item in prior['measured_current_segments']:
    assert item['episode_id'] not in reused;reused[item['episode_id']]=item
   reused_sources.append(dict(archive=str(src),report_sha256=spec['report_sha256'],delivery_sha256=spec['delivery_sha256'],complete_development_episodes=prior['complete_development_episodes']))
  for group,(reader,circuit) in enumerate(requested):
   receipt_path=base/('parameter_block_readback_acceptance_'+reader+'.json');receipt=json.loads(receipt_path.read_text())
   assert receipt['status']=='ACCEPTED_PFC_TEN_CONDITION_TASK_READBACK' and receipt['complete_development_episodes']==10 and receipt['circuit_index']==circuit
   block=receipt['block_index'];assert block==group%2+1 and receipt['job_id']==reader
   source=Path(receipt['archive']);assert digest(source/'delivery.json')==receipt['delivery_sha256'] and digest(source/'report.json')==receipt['report_sha256']
   delivery=json.loads((source/'delivery.json').read_text());files={r['path']:r for r in delivery['files']}
   report=json.loads((source/'report.json').read_text());assert [b['condition_id'] for b in report['circuits'][0]['source_bindings']]==[b['condition_id'] for b in receipt['sources']]
   for item in receipt['sources']:
    name=item['condition_id'];episode=item['episode_id'];assert episode not in seen;seen.add(episode)
    relative=f'circuit_{circuit:03d}/block_{block:03d}/{name}.csv';entry=files[relative];dst=work/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/relative,dst)
    assert dst.stat().st_size==entry['bytes'] and digest(dst)==entry['sha256'];checked_bytes+=entry['bytes']
    with dst.open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==1000
    for i,row in enumerate(rows):
     assert row['episode_id']==episode and row['condition_id']==name and int(row['bin_index'])==i and int(row['start_tick'])==i*100 and int(row['stop_tick'])==(i+1)*100
    if episode in reused:
     cached=reused[episode]
     assert cached['condition_id']==name and cached['circuit_index']==circuit and cached['source_archive']==item['source_archive'] and cached['source_delivery_sha256']==item['delivery_sha256'] and cached['late_delay_external_current_zero']
     removal_tick=cached['cue_schedule_deactivation_tick'];removal_ms=cached['cue_schedule_deactivation_ms'];post_start=cached['first_complete_post_cue_bin_ms'];current_summary=cached['segments']
     measured_currents.append(cached)
    else:
     origin=Path(item['source_archive']);d=json.loads((origin/'delivery.json').read_text())
     assert digest(origin/'delivery.json')==item['delivery_sha256']
     source_files={entry['path']:entry for entry in d['files']}
     for relative_source in ['source_run/report.json','source_run/applied_current_observation.npz']:
      entry_source=source_files[relative_source];source_path=origin/relative_source
      assert source_path.stat().st_size==entry_source['bytes'] and digest(source_path)==entry_source['sha256']
     source_report=json.loads((origin/'source_run/report.json').read_text())
     assert source_report['status']=='PFC_SOURCE_EXECUTION_COMPLETE' and source_report['case_id']==episode
     current_check=source_report['validations']['applied_current_source_clock_coverage']
     with np.load(origin/'source_run/applied_current_observation.npz',allow_pickle=False) as stored:
      ticks=stored['segment_start_ticks'];stops=stored['segment_stop_ticks'];currents=stored['current_A'];cue=stored['cue_active'];distractor=stored['distractor_active']
     assert ticks.tolist()==current_check['expected_start_ticks'] and stops.tolist()==current_check['expected_stop_ticks']
     assert currents.shape==(len(ticks),80000) and np.isfinite(currents).all()
     cue_removed=np.flatnonzero((~cue[1:]) & cue[:-1])+1
     assert len(cue_removed)==1
     removal_tick=int(ticks[cue_removed[0]]);removal_ms=removal_tick*.05
     post_start=5*math.ceil(removal_ms/5)
     # Complete delivered current segments must expose all applied inputs,
     # including distractor-only responses and native transition delays.
     current_summary=[]
     for i,(begin,end) in enumerate(zip(ticks,stops)):
      nonzero=int(np.count_nonzero(currents[i]))
      if not cue[i] and not distractor[i]:assert nonzero==0,'PFC_UNCLEARED_EXTERNAL_CURRENT_AFTER_SOURCE_OPERATION'
      current_summary.append(dict(start_tick=int(begin),stop_tick=int(end),cue_active=bool(cue[i]),distractor_active=bool(distractor[i]),nonzero_E_currents=nonzero,max_current_nA=float(currents[i].max()/.000000001)))
     late_segments=[i for i,end in enumerate(stops) if end>80000]
     assert all(np.count_nonzero(currents[i])==0 for i in late_segments)
     measured_currents.append(dict(circuit_index=circuit,block_index=block,episode_id=episode,condition_id=name,source_archive=str(origin),source_delivery_sha256=item['delivery_sha256'],source_current_sha256=source_files['source_run/applied_current_observation.npz']['sha256'],cue_schedule_deactivation_tick=removal_tick,cue_schedule_deactivation_ms=removal_ms,first_complete_post_cue_bin_ms=post_start,late_delay_external_current_zero=True,segments=current_summary))
    for phase,start in [('post_measured_cue',post_start),('late_delay',PHASES['late_delay'][0])]:
     values=delay_summary(rows,start)
     values.update(measured_cue_schedule_deactivation_ms=removal_ms,late_delay_external_current_zero=True)
     summaries.append(dict(circuit_index=circuit,block_index=block,reader_job=reader,episode_id=episode,condition_id=name,phase=phase,**values))
    temporal_cases.append((circuit,block,reader,episode,name,post_start,rows))
   source_receipts.append(dict(reader_job=reader,receipt=str(receipt_path),receipt_sha256=digest(receipt_path),reader_delivery_sha256=receipt['delivery_sha256'],reader_report_sha256=receipt['report_sha256']))
  by_episode={item['episode_id']:item for item in measured_currents}
  by_circuit_condition={(item['circuit_index'],item['block_index'],item['condition_id']):item for item in measured_currents}
  for circuit,block,reader,episode,name,post_start,rows in temporal_cases:
    # Summarize control-free intervals and the measured distractor interval.
    # Undefined angles and negative selectivity remain in their denominators.
    current_summary=by_episode[episode]['segments']
    current_condition=name if not name.endswith('_none') else 'cue_aligned'
    matched=by_circuit_condition[(circuit,block,current_condition)]
    active=[s for s in matched['segments'] if s['distractor_active']]
    if active:
     distractor_start=min(s['start_tick'] for s in active)*.05
     distractor_stop=max(s['stop_tick'] for s in active)*.05
    else:raise ValueError('MEASURED_MATCHED_DISTRACTOR_INTERVAL_REQUIRED')
    intervals=[('autonomous_pre_distractor',post_start,distractor_start),
      ('measured_distractor',5*math.ceil(distractor_start/5),5*math.floor(distractor_stop/5)),
      ('autonomous_post_distractor',5*math.ceil(distractor_stop/5),5000.0),
      ('late_delay',4000.0,5000.0)]
    for phase,start,stop in intervals:
     selected=[row for row in rows if float(row['start_ms'])>=start and float(row['stop_ms'])<=stop]
     assert selected and float(selected[0]['start_ms'])==start and float(selected[-1]['stop_ms'])==stop
     values=delay_summary(selected,start)
     nulls=sum(row['circular_cue_error_deg'] in (None,'') for row in selected)
     assert values['direction_defined_bins']+nulls==len(selected)
     current_segments=[s for s in current_summary if s['start_tick']*.05<stop and s['stop_tick']*.05>start]
     zero=all(s['nonzero_E_currents']==0 for s in current_segments)
     if phase.startswith('autonomous') or phase=='late_delay':assert zero
     phase_rows.append(dict(circuit_index=circuit,block_index=block,reader_job=reader,episode_id=episode,condition_id=name,
       phase=phase,cue_present=name.startswith('cue_'),external_current_zero=zero,
       undefined_direction_bins=nulls,**values))

  assert len(seen)==expected_episodes and len(summaries)==2*expected_episodes and len(phase_rows)==4*expected_episodes
  table=work/'delay_summary.csv'
  with table.open('x',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
  phase_table=work/'task_phase_summary.csv'
  with phase_table.open('x',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(phase_rows[0]));writer.writeheader();writer.writerows(phase_rows)
  record.update(complete_development_episodes=expected_episodes,summary_rows=2*expected_episodes,task_phase_rows=len(phase_rows),task_phases=phase_rows,task_phase_summary_sha256=digest(phase_table),reused_measured_current_sources=reused_sources,reused_current_episodes=len(reused),new_current_episodes_authenticated=expected_episodes-len(reused),cases=summaries,measured_current_segments=measured_currents,source_acceptance_receipts=source_receipts,verified_temporal_csvs=expected_episodes,verified_temporal_csv_bytes=checked_bytes,implementation_sha256=digest(work/'Libs/population_dynamics/pfc_temporal_readouts.py'),summary_sha256=digest(table))
  comparisons=[]
  selected={(row['circuit_index'],row['block_index'],row['condition_id'],row['phase']):row for row in phase_rows}
  fields=['excitatory_spikes','inhibitory_spikes','cue_selectivity_mean_hz','undefined_direction_bins','direction_defined_bins']
  names=['cue_none','cue_aligned','cue_off_target','cue_antipodal','cue_late_off_target','omission_none','omission_aligned','omission_off_target','omission_antipodal','omission_late_off_target']
  phases=['autonomous_pre_distractor','measured_distractor','autonomous_post_distractor','late_delay']
  u=(24.444444444444443-20)/10;v=(1.988888888888889-1.6)/.7
  for name in names:
   for phase in phases:
    for field in fields:
     means={circuit:sum(selected[circuit,block,name,phase][field] for block in [1,2])/2 for circuit in [23,63,74,86,52]}
     for circuit in [23,63,74,86,52]:
      comparisons.append(dict(condition_id=name,phase=phase,field=field,contrast='block2_minus_block1',circuit_index=circuit,value=selected[circuit,2,name,phase][field]-selected[circuit,1,name,phase][field],inference='descriptive_independent_noise_blocks'))
     for contrast,value in [
       ('sigma_corner_difference',.5*(means[23]+means[74]-means[63]-means[86])),
       ('Jpos_corner_difference',.5*(means[23]+means[63]-means[74]-means[86])),
       ('corner_interaction',means[23]-means[63]-means[74]+means[86]),
       ('interior_minus_bilinear_corners',means[52]-(u*v*means[23]+(1-u)*v*means[63]+u*(1-v)*means[74]+(1-u)*(1-v)*means[86]))]:
      comparisons.append(dict(condition_id=name,phase=phase,field=field,contrast=contrast,circuit_index='all',value=value,inference='descriptive_unpaired_circuit_means'))
  comparison_table=work/'parameter_block_comparisons.csv'
  with comparison_table.open('x',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
  assert len(comparisons)==1800
  record.update(status='COMPLETE_PFC_MEASURED_CURRENT_DELAY_SUMMARY',parameter_comparison_rows=len(comparisons),parameter_comparison_sha256=digest(comparison_table),between_circuit_noise_pairing=False,statistical_significance_claim=False,source_episodes_complete=100,stop_rule_reached=True)
  code=0
 except Exception as error:
  record.update(error=str(error),traceback=traceback.format_exc())
 record.update(exit_code=code,recorded_utc=datetime.now(timezone.utc).isoformat())
 (work/'report.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
 archive.mkdir()
 outputs=['report.json']+(['delay_summary.csv','task_phase_summary.csv','parameter_block_comparisons.csv'] if code==0 else [])
 delivered=[]
 for name in outputs:
  shutil.copy2(work/name,archive/name);delivered.append(dict(path=name,bytes=(archive/name).stat().st_size,sha256=digest(archive/name)))
 (archive/'delivery.json').write_text(json.dumps(dict(status=record['status'],job_id=job,verified=True,files=delivered),indent=2)+'\n')
 print(json.dumps(record,indent=2))
raise SystemExit(code)
PYCODE
