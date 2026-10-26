"""Prepare a CPU return of comparison metadata from persisted generation output."""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path


BOOTSTRAP = r'''
import hashlib,json,os,pathlib,shutil,subprocess,sys,tarfile,tempfile,traceback
root=pathlib.Path(tempfile.mkdtemp(prefix='native-generation-return-'));output=root/'results';output.mkdir()
report=dict(status='PREPARING',exit_code=1,run_id=os.environ['RH2_JOB_RUN_ID'],trial_id=os.environ['ARNOLD_TRIAL_ID'],
 scope='comparison_metadata_return',large_arrays_returned=False,scientific_computation_repeated=False)
publisher=None
try:
 p=payload['publisher'];subprocess.run(['hdfs','dfs','-get',p['hdfs_path'],str(root/'publisher.py')],check=True)
 def digest(path):
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
  return h.hexdigest()
 assert (root/'publisher.py').stat().st_size==p['bytes'] and digest(root/'publisher.py')==p['sha256']
 sys.path.insert(0,str(root));import publisher
 receipt=payload['receipt'];archive=root/'generation.tar.gz'
 assert receipt['persisted'] is True and receipt['exit_code']==0
 publisher.fetch(receipt['checkpoint_path']+'/results.tar.gz',archive);publisher.verify(archive,{'bytes':receipt['result_bytes'],'sha256':receipt['result_sha256']})
 seen=set();returned=[]
 with tarfile.open(archive,'r|*') as bundle:
  for member in bundle:
   name=pathlib.PurePosixPath(member.name)
   if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0]!='results' or str(name) in seen:raise ValueError('unsafe generation archive member')
   seen.add(str(name))
   if member.isdir():continue
   if not member.isfile():raise ValueError('generation archive must contain regular files')
   relative=pathlib.Path(*name.parts[1:])
   if relative.suffix not in ('.json','.csv','.yaml'):continue
   target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
   with bundle.extractfile(member) as source,target.open('xb') as destination:shutil.copyfileobj(source,destination,1024*1024)
   returned.append(dict(path=relative.as_posix(),bytes=target.stat().st_size,sha256=digest(target)))
 execution=json.loads((output/'execution.json').read_text())
 (output/'execution.json').rename(output/'source_execution.json')
 for row in returned:
  if row['path']=='execution.json':row['path']='source_execution.json'
 delivery=output/'delivery';d=json.loads((delivery/'delivery.json').read_text())
 if execution['run_id']!=payload['source_run'] or str(execution['trial_id'])!=payload['source_trial'] or execution['exit_code']!=0:raise ValueError('source execution identity differs')
 if d['status']!='PASS' or d['exit_code']!=0 or d['training_seed']!=payload['seed'] or set(d['modes'])!={'current','history'}:raise ValueError('complete seed delivery required')
 manifest={row['path']:row for row in d['files']}
 if len(manifest)!=len(d['files']):raise ValueError('duplicate source delivery path')
 for row in returned:
  if row['path'].startswith('delivery/') and row['path']!='delivery/delivery.json':
   relative=row['path'].removeprefix('delivery/');expected=manifest[relative]
   if row['sha256']!=expected['sha256'] or row['bytes']!=expected.get('bytes',expected.get('destination_bytes')):raise ValueError('returned metadata differs from source manifest')
 for mode in ('current','history'):
  result=d['modes'][mode]
  if result['status']!='PASS' or result['exit_code']!=0 or result['verified_origins']!=48 or result['score_rows']!=144 or result['generated_unit_count_values']!=7680000000:raise ValueError('complete full-unit mode replay required')
  for relative,key in ((f'results/{mode}/forecasts/evaluation.json','evaluation_sha256'),(f'results/{mode}/replay.json','replay_sha256')):
   if digest(delivery/relative)!=result[key]:raise ValueError('mode receipt differs from returned metadata')
 report.update(status='VERIFIED_GENERATION_METADATA',exit_code=0,seed=payload['seed'],source_run=payload['source_run'],source_trial=payload['source_trial'],source_archive=receipt,
  source_execution=execution,delivery_sha256=digest(delivery/'delivery.json'),returned_files=returned,complete_modes=list(d['modes']),full_output_persisted=True)
except Exception:report.update(status='FAILED_METADATA_RETURN',error=traceback.format_exc())
finally:
 for name in ('analysis.json','execution.json'):(output/name).write_text(json.dumps(report,indent=2)+'\n')
 if publisher is not None:
  receipt=publisher.persist(root,output,os.environ['RH2_CHECKPOINT_HDFS'].rstrip('/')+'/generation_metadata',report['exit_code'])
  print(json.dumps(dict(report=report,receipt=receipt)),flush=True)
raise SystemExit(report['exit_code'])
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--generation-preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if not args.name.startswith('mrln_') or len(args.name) > 90:
        raise ValueError('valid mrln_ job name required')
    if args.output.exists():
        raise FileExistsError('fresh return preparation required')
    execution = json.loads((args.generation_preparation/'execution.json').read_text())
    receipt = json.loads((args.generation_preparation/'terminal_receipt.json').read_text())
    if receipt.get('persisted') is not True or receipt.get('exit_code') != 0:
        raise ValueError('persisted successful generation receipt required')
    payload = dict(receipt=receipt, source_run=execution['run_id'], source_trial=execution['trial_id'],
        seed=execution['seed'], publisher=json.loads((args.campaign_root/'merlin_readout_analysis/wrapper_upload.json').read_text()))
    encoded = base64.b64encode(gzip.compress(json.dumps(payload).encode(),mtime=0)).decode()
    entry = "#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''\npython3 - <<'RETURN'\nimport base64,gzip,json\npayload=json.loads(gzip.decompress(base64.b64decode("+repr(encoded)+")))\n"+BOOTSTRAP+"\nRETURN\n"
    request = json.loads((args.campaign_root/'native_observation_fitting/generation_checks/request.json').read_text())
    request.pop('request_context', None)
    request.update(name=args.name, description='Verify an existing full generation archive and return its manifest-bound comparison metadata. No training, generation or replay repeated.')
    request['job_config']['job_template_config']['entrypoint_full_script'] = entry
    resource = request['resource_config']['arnold_resource_config'];resource.update(cluster_ids=[20], group_id=271)
    role = resource['roles'][0];role.update(cpu=2,memory_mb=4096,gpu_type='CPU_ONLY',queue_name='compute-49-hl-cloudnative-am-ray.cpu-guarantee');role.pop('gpu',None)
    args.output.mkdir(parents=True)
    for name,value in [('request.json',request),('check.json',dict(check_point='create',node_type='merlin_job_run',job_run_context=request)),
        ('execution.json',dict(status='PREPARED',submitted=False,name=args.name,source_run=payload['source_run'],source_trial=payload['source_trial'],seed=payload['seed']))]:
        (args.output/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='PREPARED',seed=payload['seed'],source_run=payload['source_run'])))


if __name__ == '__main__':
    main()
