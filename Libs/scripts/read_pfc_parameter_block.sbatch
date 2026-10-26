#!/bin/bash
#SBATCH --job-name=NrnSim_PfcBlockReadback
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=none
#SBATCH --time=0
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/NrnSim-PfcBlockReadback-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/opt/conda/bin/python3 -B - <<'PY'
from pathlib import Path
import csv,hashlib,json,os,shutil,socket,sys,tempfile,traceback
import numpy as np
import yaml
repo=Path('/workspace/NeuronSim');job=os.environ['SLURM_JOB_ID']
assert socket.gethostname().split('.')[0]=='cpu04' and not os.environ.get('SLURM_JOB_GPUS')
base=repo/'Output/neural_population_dynamics/benchmark/pfc_parameter_design'
prepared=base/'parameter_block_prepare_613107'
source_job=os.environ['NEURONSIM_TASK_SOURCE_JOB']
plan_root=Path(os.environ['NEURONSIM_PFC_BLOCK_PLAN'])
archive=base/('parameter_block_readback_'+job);assert not archive.exists()
work=Path(tempfile.mkdtemp(prefix='neuronsim-pfc-task-readback-'+job+'.',dir='/tmp'))
record=dict(status='FAILED_PFC_TASK_BOUNDARY_READBACK',job_id=job,source_job=source_job,circuits=[],
            new_source_simulations=0,production_episodes=0,scientific_fidelity_qualified=False,
            confidence_intervals=False,p_values=False,source_probability_law_qualified=False)
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write_rows(p,rows):
 with p.open('x',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
code=1
try:
 software=Path(os.environ['NEURONSIM_TASK_READER_SOFTWARE'])
 manifest=json.loads((software/'delivery.json').read_text())
 for row in manifest['files']:
  name=Path(row['path']);assert not name.is_absolute() and '..' not in name.parts
  p=software/name;target=work/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  assert target.stat().st_size==row['bytes'] and digest(target)==row['sha256']
 sys.path.insert(0,str(work/'Libs'))
 from population_dynamics.pfc_parameter_block_execution import load_plan,group_rows,identity,recipe_path,source_path,complete_source_batch
 from population_dynamics.resource_probe import probe
 from population_dynamics.pfc_source_dataset import _benchmark_registration,export
 from population_dynamics.pfc_scale_analysis import _source_header,_source_values
 from population_dynamics.pfc_temporal_readouts import temporal_rows,RATE_FIELDS,COUNT_FIELDS,MOMENT_FIELDS
 h=probe(work/'resource_preflight.json',4);assert h['status']=='PASS',h['reasons']
 allocation=load_plan(plan_root,require_release=True)
 selected=group_rows(allocation,int(os.environ['NEURONSIM_PFC_BLOCK_GROUP']))
 circuit=int(selected[0]['circuit_index']);block=int(selected[0]['block_index'])
 shutil.copy2(plan_root/'execution.csv',work/'execution.csv')
 for _ in range(1):
  design=json.loads((prepared/f'recipes/circuit_{circuit:03d}/block_{block:03d}/report.json').read_text())
  assert len(design['conditions'])==10 and design['production_eligible'] is False
  assert [item['identity'] for item in design['conditions']]==[identity(row) for row in selected]
  responses={};pairing=[];bindings=[];reference=None
  target=work/f'circuit_{circuit:03d}/block_{block:03d}';target.mkdir(parents=True)
  for offset,condition in enumerate(design['conditions']):
   name=condition['condition']['condition_id'];episode=condition['identity']['episode_id']
   task=selected[offset];assert name==task['condition_id'] and episode==task['episode_id']
   source_base=source_path(base,task)
   batch,source_attempts=complete_source_batch(base,task)
   plan=json.loads((batch/'delivery.json').read_text())
   assert plan['status']=='COMPLETE' and plan['exit_code']==plan['srun_exit_code']==0 and len(plan['cases'])==1 and plan['cases'][0]['case_id']==episode
   assert plan['cases'][0]['node']=='cpu03' and plan['cases'][0]['source_execution_complete'] is True
   origin=Path(plan['cases'][0]['archive'])
   assert origin.parent==source_base and origin.name=='run_'+plan['job_id']
   delivery=json.loads((origin/'delivery.json').read_text())
   assert delivery['source_execution_complete'] is True and delivery['exit_code']==0
   staged=work/'inputs'/f'circuit_{circuit:03d}'/name
   for row in delivery['files']:
    path=Path(row['path']);assert not path.is_absolute() and '..' not in path.parts
    p=origin/path;dst=staged/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    assert dst.stat().st_size==row['bytes'] and digest(dst)==row['sha256']
   shutil.copy2(origin/'delivery.json',staged/'delivery.json')
   config=yaml.safe_load((staged/'source_run/config.yaml').read_text())
   expected=yaml.safe_load(recipe_path(task,prepared).read_text())
   assert config==expected and config['benchmark_episode_candidate']['identity']==condition['identity']
   config,registry,_,dimensions=_benchmark_registration(config)
   config_sha=digest(staged/'source_run/config.yaml')
   status,header=_source_header(staged,config,registry,config_sha)
   assert status['status']=='COMPLETE'
   raw,common,paired=_source_values(header,config,registry,dimensions,config_sha)
   with np.load(staged/'source_run/applied_current_observation.npz',allow_pickle=False) as stored:
    current={key:stored[key] for key in stored.files}
   rows=temporal_rows(raw['spike_ticks'],raw['spike_unit_indices'],n_e=dimensions['n_e'],n_i=dimensions['n_i'],
                      dt_ms=config['dt_ms'],bin_ms=config['readout_bin_ms'],duration_ms=config['duration_ms'])
   assert dimensions['coupled_core_units']==100000 and dimensions['duration_ticks']==100000 and len(rows)==1000
   rows=[dict(circuit_index=circuit,block_index=block,block_id=task['block_id'],task_index=int(task['task_index']),condition_id=name,episode_id=episode,**row) for row in rows]
   write_rows(target/(name+'.csv'),rows);responses[name]=rows
   if reference is None:reference=(raw,common,paired,current)
   else:
    previous,old_common,old_paired,old_current=reference
    assert common==old_common and paired['inputs_used']==old_paired['inputs_used']
    assert np.array_equal(paired['initial'],old_paired['initial'])
    assert set(paired['final_rng'])==set(old_paired['final_rng'])
    assert all(np.array_equal(value,old_paired['final_rng'][key]) for key,value in paired['final_rng'].items())
    boundaries=np.union1d(current['segment_start_ticks'],old_current['segment_start_ticks'])
    first=dimensions['duration_ticks']
    for boundary in boundaries:
     left=np.searchsorted(old_current['segment_start_ticks'],boundary,side='right')-1
     right=np.searchsorted(current['segment_start_ticks'],boundary,side='right')-1
     if not np.array_equal(old_current['current_A'][left],current['current_A'][right]):
      first=int(boundary);break
    masks=[previous['spike_ticks']<first,raw['spike_ticks']<first]
    assert all(np.array_equal(previous[key][masks[0]],raw[key][masks[1]]) for key in ['spike_ticks','spike_unit_indices','spike_seconds'])
    pairing.append(dict(condition_id=name,reference_condition='cue_none',first_changed_current_tick=first,
                        exact_prefix_events=int(masks[0].sum()),initial_and_final_rng_exact=True))
   origin_table=target/(name+'_origins.json')
   origin_table.write_text(json.dumps([dict(archive=str(origin),staged=str(staged),case_id=episode,
                      delivery_sha256=digest(origin/'delivery.json'))])+'\n')
   result=export([staged],staged/'source_run/config.yaml',target/(name+'_dataset'),input_origins=origin_table)
   assert result['status']=='EXPORTED_VERIFIED_PFC_DEVELOPMENT_DATASET' and result['complete_episodes']==1
   bindings.append(dict(circuit_index=circuit,block_index=block,task_index=int(task['task_index']),episode_id=episode,condition_id=name,source_archive=str(origin),delivery_sha256=digest(origin/'delivery.json'),
                        events=len(raw['spike_ticks']),complete_units=100000,export=result['status'],source_attempts=source_attempts))
  contrasts=[]
  for distractor in ['none','aligned','off_target','antipodal','late_off_target']:
   for index in range(1000):
    cue=responses['cue_'+distractor][index];omission=responses['omission_'+distractor][index]
    no_cue=responses['cue_none'][index];no_omission=responses['omission_none'][index]
    row=dict(circuit_index=circuit,block_index=block,block_id=selected[0]['block_id'],cue_episode_id=cue['episode_id'],omission_episode_id=omission['episode_id'],reference_cue_episode_id=no_cue['episode_id'],reference_omission_episode_id=no_omission['episode_id'],distractor=distractor,bin_index=index,start_tick=cue['start_tick'],stop_tick=cue['stop_tick'])
    assert all(r['start_tick']==row['start_tick'] and r['stop_tick']==row['stop_tick'] for r in [omission,no_cue,no_omission])
    for key in (*COUNT_FIELDS,*RATE_FIELDS,*MOMENT_FIELDS):
     row[key+'_cue_minus_omission']=cue[key]-omission[key]
     row[key+'_distractor_interaction']=(cue[key]-omission[key])-(no_cue[key]-no_omission[key])
    contrasts.append(row)
  write_rows(target/'paired_contrasts.csv',contrasts)
  record['circuits'].append(dict(circuit_index=circuit,block_index=block,block_id=selected[0]['block_id'],complete_development_episodes=10,temporal_bins=10000,
                 paired_contrast_rows=5000,source_bindings=bindings,pairing=pairing,independent_noise_blocks=1))
 record.update(status='PASS_PFC_TEN_CONDITION_SOURCE_READBACK',complete_development_episodes=10,complete_joint_blocks=1,
               complete_parameter_points=1,parameter_domain_qualified=False,task_success_assessed=False)
 code=0
except Exception as error:record.update(error=str(error),traceback=traceback.format_exc())
finally:
 record['exit_code']=code;(work/'report.json').write_text(json.dumps(record,indent=2)+'\n')
 archive.mkdir(parents=True);files=[]
 for p in sorted(work.rglob('*')):
  name=p.relative_to(work)
  if not p.is_file() or name.parts[0]=='inputs' or '__pycache__' in name.parts:continue
  dst=archive/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);assert digest(dst)==digest(p)
  files.append(dict(path=str(name),bytes=p.stat().st_size,sha256=digest(dst)))
 (archive/'delivery.json').write_text(json.dumps(dict(status=record['status'],job_id=job,verified=True,files=files),indent=2)+'\n')
 shutil.rmtree(work);print(json.dumps(record))
raise SystemExit(code)
PY
