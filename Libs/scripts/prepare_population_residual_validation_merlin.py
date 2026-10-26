"""Prepare CPU input reconciliation followed by full-unit objective validation."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.name.startswith('mrln_') or len(args.name)>90:
        raise ValueError('Merlin name must start with mrln_ and contain at most90characters')
    previous=args.output/'execution.json'
    if previous.exists() and json.loads(previous.read_text()).get('status')!='PREPARED':
        raise FileExistsError('reuse the recorded execution or choose a distinct recovery output')
    root=args.campaign_root
    checks=root/'native_observation_fitting/objective_checks'
    result=json.loads((checks/'result.json').read_text())
    if result['status']!='PASS' or result['tests']!=16 or any(result[k] for k in ('failures','errors','skipped')):
        raise ValueError('accepted16 objective checks required')
    package=Path(__file__).resolve().parents[1]/'population_dynamics'
    objective='population_dynamics/population_residual_observation.py'
    if hashlib.sha256((package/'population_residual_observation.py').read_bytes()).hexdigest()!=result['sources'][objective]['sha256']:
        raise ValueError('objective changed after accepted validation')
    inputs=root/'native_observation_fitting/real_inputs'
    raw=(inputs/'pilot_manifest.json').read_bytes();manifest=json.loads(raw)
    upload=json.loads((inputs/'pilot_upload/upload.json').read_text())
    if upload['manifest_sha256']!=hashlib.sha256(raw).hexdigest():
        raise ValueError('input upload manifest binding differs')
    known={row['path']:row for row in upload['files']}
    pending=[]
    inline={}
    for index,row in enumerate(manifest['files']):
        if row['path'] in known:
            if any(known[row['path']][key]!=row[key] for key in ('bytes','sha256')):
                raise ValueError('acknowledged file identity differs')
        else:
            attempt=json.loads((inputs/f'pilot_upload/pending_{index:03d}.json').read_text())
            if attempt['status']!='CREATE_OUTCOME_UNKNOWN' or any(attempt[k]!=row[k] for k in ('path','bytes','sha256')):
                raise ValueError('every missing file must have an explicit unresolved upload attempt')
            if row['path'].endswith('.json') and row['bytes'] <= 65536:
                data=Path(row['source']).read_bytes()
                if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                    raise ValueError('inline metadata source differs')
                inline[row['path']]=base64.b64encode(data).decode()
                known[row['path']]={k:row[k] for k in ('path','bytes','sha256')}|{'resolution':'inline_original_metadata','previous_upload_outcome':'unresolved_not_replayed'}
            else:
                pending.append({k:attempt[k] for k in ('path','bytes','sha256','dispatched_at')})
    source=json.loads((checks/'source_upload.json').read_text())
    wrapper=json.loads((root/'merlin_readout_analysis/wrapper_upload.json').read_text())
    driver=(package/'population_residual_observation_validation.py').read_bytes()
    payload=dict(source=source,wrapper=wrapper,manifest_base64=base64.b64encode(raw).decode(),
        known=known,pending=pending,inline=inline,search_directories=sorted({row['hdfs_path'].rsplit('/',1)[0] for row in known.values() if 'hdfs_path' in row}),
        driver=dict(bytes=len(driver),sha256=hashlib.sha256(driver).hexdigest(),content=base64.b64encode(driver).decode()))
    bootstrap='''import base64,hashlib,json,os,pathlib,shutil,subprocess,sys,tempfile,traceback,datetime
root=pathlib.Path(tempfile.mkdtemp(prefix='neuronsim-native-fit-'))
output=root/'results';output.mkdir()
execution={'status':'STAGING','exit_code':1,'run_id':os.environ['RH2_JOB_RUN_ID'],'trial_id':os.environ['ARNOLD_TRIAL_ID'],'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def fetch(uri,path,expected=None):
 path.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['hdfs','dfs','-get',uri,str(path)],check=True)
 if expected:
  with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  if path.stat().st_size!=expected['bytes'] or digest!=expected['sha256']:raise ValueError('input identity differs: '+str(path))
try:
 if shutil.disk_usage(root).free<4*1024**3:raise RuntimeError('native validation requires4GiB scratch')
 fetch(payload['wrapper']['hdfs_path'],root/'publisher.py',payload['wrapper'])
 sys.path.insert(0,str(root))
 from publisher import persist,write_json
 fetch(payload['source']['hdfs_path'],root/'source.json',payload['source'])
 sources=json.loads((root/'source.json').read_bytes())
 for name,row in sources.items():
  data=base64.b64decode(row['content'],validate=True)
  if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('source identity differs: '+name)
  path=root/'Libs'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 row=payload['driver'];data=base64.b64decode(row['content'],validate=True)
 if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('driver identity differs')
 (root/'Libs/population_dynamics/population_residual_observation_validation.py').write_bytes(data)
 raw=base64.b64decode(payload['manifest_base64'],validate=True);(root/'manifest.json').write_bytes(raw);manifest=json.loads(raw)
 resolved=dict(payload['known']);resolutions=[]
 for name,encoded in payload['inline'].items():
  data=base64.b64decode(encoded,validate=True);row=resolved[name]
  if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('inline metadata identity differs')
  target=root/'inputs'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 if payload['pending']:
  sizes={row['bytes'] for row in payload['pending']};candidates=[]
  for directory in payload['search_directories']:
   listing=subprocess.run(['hdfs','dfs','-ls',directory],capture_output=True,text=True,check=True).stdout
   for line in listing.splitlines():
    parts=line.split(maxsplit=7)
    if len(parts)==8 and parts[0].startswith('-') and int(parts[4]) in sizes:candidates.append((parts[7],int(parts[4])))
  for number,(uri,size) in enumerate(candidates):
   wanted=[row for row in payload['pending'] if row['bytes']==size and row['path'] not in resolved]
   if not wanted:continue
   candidate=root/f'candidate_{number}';fetch(uri,candidate)
   with candidate.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
   for row in wanted:
    if digest==row['sha256']:
     target=root/'inputs'/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(candidate,target)
     resolved[row['path']]=dict(row,hdfs_path=uri,resolution='remote_content_hash')
     resolutions.append(resolved[row['path']])
   candidate.unlink()
  missing=[row['path'] for row in payload['pending'] if row['path'] not in resolved]
  write_json(output/'input_reconciliation.json',{'resolved':resolutions,'missing':missing,'candidate_files':len(candidates),'directories':payload['search_directories']})
  if missing:raise RuntimeError('unresolved native inputs: '+','.join(missing))
 for row in manifest['files']:
  target=root/'inputs'/row['path']
  if not target.exists():fetch(resolved[row['path']]['hdfs_path'],target,row)
  else:
   with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
   if target.stat().st_size!=row['bytes'] or digest!=row['sha256']:raise ValueError('resolved input identity differs')
 write_json(output/'resolved_inputs.json',{'files':list(resolved.values()),'manifest_sha256':hashlib.sha256(raw).hexdigest(),'complete':True})
 execution['status']='VALIDATING';execution['driver_sha256']=payload['driver']['sha256']
 environment=dict(os.environ,PYTHONPATH=str(root/'Libs'))
 with (output/'validation.log').open('w') as log:
  completed=subprocess.run([sys.executable,'-m','population_dynamics.population_residual_observation_validation','--inputs',str(root/'inputs'),'--manifest',str(root/'manifest.json'),'--output',str(output),'--threads','8'],env=environment,stdout=log,stderr=subprocess.STDOUT)
 execution.update(status='COMPLETE' if completed.returncode==0 else 'FAILED',exit_code=completed.returncode)
except Exception:
 execution.update(status='FAILED',error=traceback.format_exc())
finally:
 execution['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (output/'execution.json').write_text(json.dumps(execution,indent=2))
 if (root/'publisher.py').exists():
  sys.path.insert(0,str(root));from publisher import persist
  if (output/'validation.json').exists():shutil.copyfile(output/'validation.json',output/'analysis.json')
  receipt=persist(root,output,os.environ['RH2_CHECKPOINT_HDFS'],execution['exit_code']);print(json.dumps(receipt),flush=True)
raise SystemExit(execution['exit_code'])
'''
    entry="#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8\npython3 -m pip install --no-deps scipy==1.16.2 PyYAML==6.0.3\npython3 - <<'NRNSIM_NATIVE_REAL'\npayload="+repr(payload)+"\n"+bootstrap+"NRNSIM_NATIVE_REAL\n"
    request=json.loads((checks/'request.json').read_text());request.pop('request_context',None)
    request.update(name=args.name,description='Reconcile exact native input hashes, then validate three-seed, two-mode residual observation gradients at100000units and500ms. One fitting episode; no model selection or source simulation.')
    request['job_config']['job_template_config']['entrypoint_full_script']=entry
    request['job_config']['job_template_config']['env_map'].update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8')
    request['resource_config']['arnold_resource_config']['roles'][0].update(cpu=8,memory_mb=16384,gpu_type='CPU_ONLY')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    (args.output/'check.json').write_text(json.dumps(dict(check_point='create',node_type='merlin_job_run',job_run_context=request),indent=2)+'\n')
    (args.output/'execution.json').write_text(json.dumps(dict(status='PREPARED',name=args.name,submitted=False,known_files=len(known),pending_files=payload['pending'],driver_sha256=payload['driver']['sha256'],cpu=8,memory_mib=16384,gpu=0),indent=2)+'\n')
    print(json.dumps(dict(prepared=True,known_files=len(known),pending_files=len(pending),request_bytes=len(json.dumps(request)))))


if __name__=='__main__':
    main()
