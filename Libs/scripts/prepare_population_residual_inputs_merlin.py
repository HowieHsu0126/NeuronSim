"""Prepare Merlin byte admission for native fitting or generation inputs."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path


BOOTSTRAP = r"""
import base64,concurrent.futures,hashlib,json,os,pathlib,shutil,subprocess,sys,tarfile,tempfile,traceback
root=pathlib.Path(tempfile.mkdtemp(prefix='native-residual-inputs-'))
output=root/'results';output.mkdir()
def digest(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def verify(path,row):
 if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:raise ValueError('input identity differs: '+str(path))
def fetch(uri,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['hdfs','dfs','-get',uri,str(path)],check=True)
report={'status':'RUNNING','exit_code':1,'run_id':os.environ['RH2_JOB_RUN_ID'],'trial_id':os.environ['ARNOLD_TRIAL_ID'],'scientific_claim_qualified':False,'source_quality_qualified':False}
try:
 if shutil.disk_usage(root).free<8*1024**3:raise RuntimeError('input admission needs8GiB scratch')
 fetch(payload['publisher']['hdfs_path'],root/'publisher.py')
 verify(root/'publisher.py',payload['publisher'])
 sys.path.insert(0,str(root));import publisher
 raw=base64.b64decode(payload['manifest'],validate=True)
 (root/'manifest.json').write_bytes(raw)
 manifest=json.loads(raw)
 resolutions=[]
 report['reconciled_uploads']=resolutions
 if payload['pending']:
  wanted={row['sha256']:row for row in payload['pending']}
  known_uris={row['hdfs_path'] for row in payload['resolved'].values() if 'hdfs_path' in row}
  candidates=[]
  search={'directories':payload['directories'],'listing_complete':False,'scan_complete':False,'known_uris_excluded':len(known_uris),'tested_candidates':[]}
  report['pending_search']=search
  listed_uris=set();search['directory_counts']={}
  for directory in payload['directories']:
   listing=subprocess.run(['hdfs','dfs','-stat','%n\t%b\t%Y',directory.rstrip('/')+'/*'],capture_output=True,text=True,check=True).stdout
   count=0
   for line in listing.splitlines():
    if not line.strip():continue
    fields=line.split('\t')
    if len(fields)!=3:raise ValueError('native HDFS stat requires name, bytes and modification milliseconds')
    name,size_text,mtime_text=fields
    if pathlib.PurePosixPath(name).name!=name or name in ('.','..') or not size_text.isdecimal() or not mtime_text.isdecimal():raise ValueError('invalid native HDFS stat record')
    uri=directory.rstrip('/')+'/'+name
    if uri in listed_uris:raise ValueError('duplicate native HDFS stat URI')
    listed_uris.add(uri);count+=1
    if uri not in known_uris and int(size_text) in {row['bytes'] for row in wanted.values()}:candidates.append(uri)
   search['directory_counts'][directory]=count
  search.update(listing_complete=True,candidate_count=len(candidates),parsed_objects=len(listed_uris),known_uris_observed=len(known_uris & listed_uris))
  if known_uris-listed_uris:raise RuntimeError('known uploaded inputs are absent from native listing; pending absence is not established')
  for index,uri in enumerate(candidates):
   path=root/('candidate_'+str(index));fetch(uri,path)
   actual=digest(path)
   search['tested_candidates'].append({'hdfs_path':uri,'bytes':path.stat().st_size,'sha256':actual})
   if actual in wanted:
    expected=wanted[actual];verify(path,expected)
    for row in payload['pending']:
     if row['sha256']==actual:
      resolved=dict(row,hdfs_path=uri,resolution='native_content_hash')
      payload['resolved'][row['path']]=resolved;resolutions.append(resolved)
    del wanted[actual]
   path.unlink()
   if not wanted:break
  search.update(scan_complete=len(search['tested_candidates'])==len(candidates),unresolved_sha256=sorted(wanted))
  report['unresolved_uploads']=[row for row in payload['pending'] if row['sha256'] in wanted]
  if wanted:raise RuntimeError('pending upload hashes remain unresolved: '+','.join(wanted))
 objects={}
 for row in manifest['files']:
  relative=pathlib.PurePosixPath(row['path'])
  if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe input path')
  known=payload['resolved'][row['path']]
  if any(known[key]!=row[key] for key in ('bytes','sha256')):raise ValueError('resolved input identity differs')
  objects.setdefault(row['sha256'],known)
 (root/'objects').mkdir()
 def stage(row):
  path=root/'objects'/row['sha256']
  if 'content_base64' in row:path.write_bytes(base64.b64decode(row['content_base64'],validate=True))
  else:fetch(row['hdfs_path'],path)
  verify(path,row)
  return row['sha256']
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for count,_ in enumerate(pool.map(stage,objects.values()),1):
   if count%20==0:print(json.dumps({'verified_unique_files':count,'unique_files':len(objects)}),flush=True)
 for row in manifest['files']:
  target=root/'inputs'/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
  os.link(root/'objects'/row['sha256'],target)
  if target.stat().st_size!=row['bytes']:raise ValueError('linked input size differs')
 archive=root/'inputs.tar.gz'
 inventory=[{key:row[key] for key in ('path','bytes','sha256')} for row in manifest['files']]
 inventory.append({'path':'manifest.json','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 (root/'input_inventory.json').write_text(json.dumps(inventory))
 with tarfile.open(archive,'w:gz',compresslevel=1,dereference=True) as bundle:
  for row in manifest['files']:bundle.add(root/'inputs'/row['path'],arcname=row['path'],recursive=False)
  bundle.add(root/'manifest.json',arcname='manifest.json',recursive=False)
  bundle.add(root/'input_inventory.json',arcname='input_inventory.json',recursive=False)
 destination=os.environ['RH2_CHECKPOINT_HDFS'].rstrip('/')+'/'+payload['destination']
 publisher.hdfs('-mkdir','-p',destination)
 publisher.hdfs('-put',archive,destination+'/inputs.tar.gz')
 report.update(status='VERIFIED_INPUT_BUNDLE',exit_code=0,input_manifest_sha256=hashlib.sha256(raw).hexdigest(),
  logical_files=len(manifest['files']),logical_bytes=sum(row['bytes'] for row in manifest['files']),
  reconciled_uploads=resolutions,unique_files=len(objects),unique_bytes=sum(row['bytes'] for row in objects.values()),
  archive={'hdfs_path':destination+'/inputs.tar.gz','bytes':archive.stat().st_size,'sha256':digest(archive)},
  source_units=manifest['source_units'],training_seeds=manifest['training_seeds'],scope=manifest['scope'])
except Exception:
 report.update(status='FAILED_INPUT_ADMISSION',error=traceback.format_exc())
finally:
 (output/'analysis.json').write_text(json.dumps(report,indent=2))
 (output/'execution.json').write_text(json.dumps({k:report[k] for k in ('status','exit_code','run_id','trial_id')}))
 if (root/'publisher.py').exists():
  sys.path.insert(0,str(root));import publisher
  receipt=publisher.persist(root,output,os.environ['RH2_CHECKPOINT_HDFS'],report['exit_code'])
  print(json.dumps({'report':report,'receipt':receipt}),flush=True)
raise SystemExit(report['exit_code'])
"""


def validate_input_contract(manifest, purpose):
    if (manifest.get("schema") != "native_residual_observation_inputs_v1"
            or manifest.get("training_seeds") != [41, 42, 43]
            or manifest.get("source_units") != 100000):
        raise ValueError("registered native residual inputs required")
    if purpose == "fitting":
        if (manifest.get("scope") != "complete_fitting_and_internal_selection"
                or manifest.get("blocks") != list(range(14))):
            raise ValueError("complete fitting and internal-selection inputs required")
    elif purpose == "generation":
        if (manifest.get("scope") != "native_generation_additional_inputs"
                or manifest.get("blocks") != list(range(14, 18))):
            raise ValueError("separate generation inputs for blocks14 through17 required")
        expected = {f"cohort/{area}/train_block_{block:03d}_familiar_{condition}/{name}"
            for block in range(14, 18) for condition in range(4)
            for area, names in (("train", ("report.json", "spikes.npz", "initial_parameters.npz")),
                                ("train_fields", ("summary.json", "spatial_fields.npz")))
            for name in names}
        config = "generation_config/neural_population_matched_forecast.yaml"
        expected.add(config)
        files = {row["path"]: row for row in manifest["files"]}
        if (len(files) != len(manifest["files"]) or set(files) != expected
                or not manifest.get("forecast_config_sha256")
                or files[config]["sha256"] != manifest["forecast_config_sha256"]):
            raise ValueError("all16 generation episodes and the parent-frozen forecast config required")
    else:
        raise ValueError("unknown native input purpose")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--purpose", choices=("fitting", "generation"), default="fitting")
    parser.add_argument("--reconcile-pending", action="store_true")
    args = parser.parse_args()
    if not args.name.startswith("mrln_") or len(args.name) > 90:
        raise ValueError("a Merlin name with mrln_ prefix and at most90 characters is required")
    if (args.output / "execution.json").exists():
        raise FileExistsError("reconcile the existing admission execution")
    root = args.campaign_root
    fitting = root / "native_observation_fitting"
    manifest_path = fitting / ("real_inputs/full_manifest.json" if args.purpose == "fitting"
                              else "generation_inputs/additional_manifest.json")
    upload_dir = fitting / ("real_inputs/full_upload" if args.purpose == "fitting"
                           else "generation_inputs/transfer")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    validate_input_contract(manifest, args.purpose)
    upload = json.loads((upload_dir / "upload.json").read_text())
    if upload["manifest_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("authenticated input transfer for the declared manifest required")
    resolved = {row["path"]: row for row in upload["files"]}
    if len(resolved) != len(upload["files"]) or not set(resolved) <= {row["path"] for row in manifest["files"]}:
        raise ValueError("resolved input membership differs")
    for row in manifest["files"]:
        if row["path"] not in resolved:
            continue
        if any(resolved[row["path"]][key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("resolved input identity differs")
    missing = [row for row in manifest["files"] if row["path"] not in resolved]
    pending = []
    if missing:
        if not args.reconcile_pending:
            raise ValueError("complete transfer or explicit pending reconciliation required")
        markers = {}
        for path in upload_dir.glob("pending_*.json"):
            row = json.loads(path.read_text())
            if row.get("status") == "CREATE_OUTCOME_UNKNOWN":
                markers[(row["sha256"], row["bytes"])] = row
        for row in missing:
            marker = markers.get((row["sha256"], row["bytes"]))
            if marker is None:
                raise ValueError("unattempted input must be transferred first: " + row["path"])
            pending.append({key: row[key] for key in ("path", "bytes", "sha256")})
    publisher = json.loads((root / "merlin_readout_analysis/wrapper_upload.json").read_text())
    payload = dict(manifest=base64.b64encode(raw).decode(), resolved=resolved, publisher=publisher,
        destination="native_fitting_inputs" if args.purpose == "fitting" else "native_generation_inputs",
        pending=pending, directories=sorted({row["hdfs_path"].rsplit("/", 1)[0]
            for row in resolved.values() if "hdfs_path" in row}))
    encoded = base64.b64encode(gzip.compress(json.dumps(payload).encode(), mtime=0)).decode()
    entry = ("#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1\n"
             "python3 - <<'NATIVE_INPUT_ADMISSION'\nimport base64,gzip,json\npayload=json.loads(gzip.decompress(base64.b64decode("
             + repr(encoded) + ")))\n" + BOOTSTRAP + "\nNATIVE_INPUT_ADMISSION\n")
    request = json.loads((fitting / "fitter_checks/package_recovery/request.json").read_text())
    request.pop("request_context", None)
    request.update(name=args.name, description=f"Verify native {args.purpose} input identities and persist one portable input bundle. Byte admission only; no fitting, new simulation or source qualification.")
    request["job_config"]["job_template_config"]["entrypoint_full_script"] = entry
    request["resource_config"]["arnold_resource_config"]["roles"][0].update(cpu=8, memory_mb=16384, gpu_type="CPU_ONLY")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (args.output / "check.json").write_text(json.dumps(dict(check_point="create",
        node_type="merlin_job_run", job_run_context=request), indent=2) + "\n")
    (args.output / "execution.json").write_text(json.dumps(dict(status="PREPARED", submitted=False,
        name=args.name, purpose=args.purpose, input_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        logical_files=len(manifest["files"]), pending_inputs=pending), indent=2) + "\n")
    print(json.dumps(dict(status="PREPARED", logical_files=len(manifest["files"]))))


if __name__ == "__main__":
    main()
