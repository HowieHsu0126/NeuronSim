"""Prepare a Merlin residual fit from an admitted input bundle and tested source."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path


BOOTSTRAP = r"""
import base64,hashlib,json,os,pathlib,shutil,subprocess,sys,tarfile,tempfile,time,traceback
root=pathlib.Path(tempfile.mkdtemp(prefix='native-residual-fit-'))
output=root/'results';output.mkdir()
started=False
try:
 if shutil.disk_usage(root).free<10*1024**3:raise RuntimeError('native fit staging requires10GiB scratch')
 wrapper=root/'publisher.py'
 subprocess.run(['hdfs','dfs','-get',payload['publisher']['hdfs_path'],str(wrapper)],check=True)
 with wrapper.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
 if wrapper.stat().st_size!=payload['publisher']['bytes'] or actual!=payload['publisher']['sha256']:raise ValueError('publisher identity differs')
 sys.path.insert(0,str(root));import publisher
 admitted=payload['archive']
 bundle={'complete':True,'bytes':admitted['bytes'],'sha256':admitted['sha256'],
         'chunks':[dict(admitted,index=0)]}
 staging=publisher.stage(payload['source'],bundle,root)
 if publisher.fingerprint(root/'inputs/manifest.json')['sha256']!=payload['input_manifest_sha256']:raise ValueError('staged input manifest differs')
 (root/'inputs.tar').unlink()
 if 'config_identity' in payload:publisher.verify(root/'source'/payload['config_relative'],payload['config_identity'])
 for number,row in enumerate(payload['restore']):
  prefix=row['receipt']['checkpoint_path']
  archive=root/('restore_'+str(number)+'.tar.gz')
  publisher.fetch(prefix+'/results.tar.gz',archive)
  publisher.verify(archive,{'bytes':row['receipt']['result_bytes'],'sha256':row['receipt']['result_sha256']})
  relative=pathlib.PurePosixPath(row['relative_cell'])
  if len(relative.parts)!=3 or relative.parts[0] not in ('qualify','refit') or relative.parts[1] not in ('seed_41','seed_42','seed_43') or relative.parts[2] not in ('current','history'):raise ValueError('invalid checkpoint cell')
  cell=output/relative
  if cell.exists():raise ValueError('duplicate restored cell')
  cell.mkdir(parents=True)
  seen=set()
  with tarfile.open(archive,'r:*') as saved:
   for member in saved:
    name=pathlib.PurePosixPath(member.name)
    if not name.parts or name.parts[0]!='results' or name.is_absolute() or '..' in name.parts or member.name in seen:raise ValueError('unsafe checkpoint archive member')
    seen.add(member.name)
    if member.isdir():continue
    if not member.isfile():raise ValueError('checkpoint archive requires regular files')
    target=cell.joinpath(*name.parts[1:]);target.parent.mkdir(parents=True,exist_ok=True)
    with saved.extractfile(member) as source,target.open('xb') as destination:shutil.copyfileobj(source,destination,1024*1024)
  if not all((cell/name).is_file() for name in ('reference.npz','reference.json','epoch_checkpoint.pt')):raise ValueError('incomplete restored checkpoint set')
  archive.unlink()
 environment=dict(os.environ,PYTHONPATH=str(root/'source'),PYTHONFAULTHANDLER='1')
 command=[sys.executable,'-m','population_dynamics.population_residual_observation_execution',
  '--inputs',str(root/'inputs'),'--manifest',str(root/'inputs/manifest.json'),
  '--config',str(root/'source'/payload.get('config_relative','configs/population_residual_observation_fitting.yaml')),
  '--output',str(output),'--publisher',str(wrapper),'--publisher-sha256',payload['publisher']['sha256'],
  '--phase',payload['phase'],'--device',payload['device'],'--threads',str(payload.get('threads',8)),
  '--seeds',*map(str,payload['seeds']),'--modes',*payload['modes']]
 if payload['epoch_limit'] is not None:command+=['--epoch-limit',str(payload['epoch_limit'])]
 (output/'launch.json').write_text(json.dumps({'command':command,'staging':staging,'restored_cells':[row['relative_cell'] for row in payload['restore']]}))
 started=True
 with (output/'fitting.log').open('w') as log:
  process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
  while True:
   code=process.poll()
   telemetry={'scope':'fitting_observation','unix_time':time.time(),'pid':process.pid,'exit_code':code}
   if payload['device']=='cuda':
    query=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    telemetry.update(gpu=query.stdout.strip(),gpu_query_exit_code=query.returncode,gpu_query_error=query.stderr[-1000:])
   with (output/'fitting.log').open('rb') as recent:
    recent.seek(0,2);recent.seek(max(0,recent.tell()-6000));telemetry['log_tail']=recent.read().decode(errors='replace')
   print(json.dumps(telemetry),flush=True)
   if code is not None:break
   try:process.wait(timeout=300)
   except subprocess.TimeoutExpired:pass
 if code and not (output/'execution.json').is_file():
  report={'status':'FAILED_FIT_PROCESS','exit_code':code,'child_exit_code':code,
   'run_id':os.environ['RH2_JOB_RUN_ID'],'trial_id':os.environ['ARNOLD_TRIAL_ID'],
   'fitting_execution_receipt_available':False,'scientific_claim_qualified':False}
  for name in ('analysis.json','execution.json'):(output/name).write_text(json.dumps(report,indent=2))
  receipt=publisher.persist(root,output,os.environ['RH2_CHECKPOINT_HDFS'].rstrip('/')+'/fit_launcher',code)
  print(json.dumps({'report':report,'receipt':receipt}),flush=True)
 raise SystemExit(code)
except Exception:
 report={'status':'FAILED_FIT_STAGING' if not started else 'FAILED_FIT_LAUNCH','exit_code':1,
  'run_id':os.environ['RH2_JOB_RUN_ID'],'trial_id':os.environ['ARNOLD_TRIAL_ID'],'error':traceback.format_exc()}
 (output/'execution.json').write_text(json.dumps(report,indent=2))
 if (root/'publisher.py').exists():
  sys.path.insert(0,str(root));import publisher
  receipt=publisher.persist(root,output,os.environ['RH2_CHECKPOINT_HDFS'].rstrip('/')+'/fit_launcher',1)
  print(json.dumps({'report':report,'receipt':receipt}),flush=True)
 raise
"""



DEFAULT_CONFIG = "configs/population_residual_observation_fitting.yaml"
MEAN_CONFIG = "configs/population_residual_mean_fitting.yaml"


def validated_source_config(checks, config_relative):
    """Select a config inside the source bundle that passed the recorded tests."""
    if config_relative not in (DEFAULT_CONFIG, MEAN_CONFIG):
        raise ValueError("registered source-relative fitting config required")
    result = json.loads((checks / "result.json").read_text())
    if (result.get("status") != "PASS" or result.get("exit_code") != 0
            or result.get("tests", 0) < 1
            or any(result.get(key, 1) for key in ("failures", "errors", "skipped"))):
        raise ValueError("accepted fitting source validation required")
    source = json.loads((checks / "source_upload.json").read_text())
    if not source.get("complete") or not source.get("hdfs_path"):
        raise ValueError("complete tested source upload required")
    identity = result.get("sources", {}).get(config_relative)
    if not isinstance(identity, dict) or set(identity) != {"bytes", "sha256"}:
        raise ValueError("selected config was not included in the tested source bundle")
    if config_relative == MEAN_CONFIG:
        required = {"population_dynamics/" + name for name in (
            "population_residual_mean_objective.py", "native_renewal_expectation.py",
            "population_residual_observation_fitting.py", "population_residual_observation_training.py",
            "population_residual_observation_execution.py", "poisson_trajectory_evaluation.py")}
        if (not required <= set(result["sources"])
                or result.get("source_payload_sha256") != source["sha256"]):
            raise ValueError("mean objective requires the exact tested source payload")
    return source, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign-root", "admission", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--phase", choices=("qualify", "refit"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--modes", choices=("current", "history"), nargs="+", default=["current", "history"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epoch-limit", type=int)
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--resume-progress", type=Path)
    parser.add_argument("--source-checks", type=Path, help="Accepted source checks; immutable input and publisher paths stay under campaign-root")
    parser.add_argument("--config-relative", default=DEFAULT_CONFIG, choices=(DEFAULT_CONFIG, MEAN_CONFIG))
    parser.add_argument("--cluster-id", type=int)
    parser.add_argument("--group-id", type=int)
    parser.add_argument("--queue")
    parser.add_argument("--gpu-type", help="Verified job request enum for the queue, e.g. A100_SXM_80GB")
    args = parser.parse_args()
    if not args.name.startswith("mrln_") or len(args.name) > 90:
        raise ValueError("valid mrln_ job name required")
    if (args.output / "execution.json").exists():
        raise FileExistsError("reconcile the existing fit execution")
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= {41, 42, 43}
            or len(set(args.modes)) != len(args.modes)):
        raise ValueError("distinct registered seeds and modes required")
    if args.epoch_limit is not None and args.epoch_limit < 1:
        raise ValueError("positive optional epoch limit required")
    root = args.campaign_root
    fitting = root / "native_observation_fitting"
    checks = args.source_checks or fitting / "checkpoint_checks/native_route_recovery"
    source, config_identity = validated_source_config(checks, args.config_relative)
    admission = json.loads(args.admission.read_text())
    raw = (fitting / "real_inputs/full_manifest.json").read_bytes()
    if (admission.get("status") != "VERIFIED_INPUT_BUNDLE" or admission["exit_code"] != 0
            or admission["input_manifest_sha256"] != hashlib.sha256(raw).hexdigest()
            or admission["logical_files"] != 462):
        raise ValueError("complete native input admission required")
    restore = {}
    if args.resume_progress:
        progress = json.loads(args.resume_progress.read_text())
        for row in progress["records"]:
            if row["scope"] != "completed_epoch_checkpoint" or row["receipt"]["exit_code"] != 0:
                raise ValueError("verified completed-epoch publication required")
            restore[row["relative_cell"]] = row
    if args.phase == "refit":
        for seed in args.seeds:
            for mode in args.modes:
                row = restore.get(f"qualify/seed_{seed}/{mode}")
                if row is None or not row["fit_export_available"]:
                    raise ValueError("complete same-seed/mode qualification exports required for refit")
    publisher = json.loads((root / "merlin_readout_analysis/wrapper_upload.json").read_text())
    payload = dict(source=source, archive=admission["archive"], publisher=publisher,
        input_manifest_sha256=hashlib.sha256(raw).hexdigest(), phase=args.phase,
        seeds=args.seeds, modes=args.modes, device=args.device, epoch_limit=args.epoch_limit, threads=args.threads,
        restore=list(restore.values()), config_relative=args.config_relative, config_identity=config_identity)
    encoded = base64.b64encode(gzip.compress(json.dumps(payload).encode(), mtime=0)).decode()
    entry = (f"#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS={args.threads} OPENBLAS_NUM_THREADS={args.threads} MKL_NUM_THREADS={args.threads}\n"
        + ("export CUDA_VISIBLE_DEVICES=''\n" if args.device == "cpu" else "")
        + "python3 -m pip install --no-deps scipy==1.16.2 PyYAML==6.0.3\n"
        + "python3 - <<'NATIVE_RESIDUAL_FIT'\nimport base64,gzip,json\npayload=json.loads(gzip.decompress(base64.b64decode("
        + repr(encoded) + ")))\n" + BOOTSTRAP + "\nNATIVE_RESIDUAL_FIT\n")
    request = json.loads((checks / "request.json").read_text())
    request.pop("request_context", None)
    request.update(name=args.name, description="Fit native residual observation dynamics with frozen parent/readout and registered fitting/selection blocks. Preserve all seeds, complete epoch checkpoints and failure records.")
    request["job_config"]["job_template_config"]["entrypoint_full_script"] = entry
    request["job_config"]["job_template_config"]["env_map"].update(OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    resource = request["resource_config"]["arnold_resource_config"]
    role = resource["roles"][0]
    role.update(cpu=8, memory_mb=16384, gpu_type="CPU_ONLY")
    role.pop("gpu", None)
    if args.device == "cuda":
        # CPU validation templates mask CUDA; let the GPU scheduler set visibility.
        request["job_config"]["job_template_config"]["env_map"].pop("CUDA_VISIBLE_DEVICES", None)
        if None in (args.cluster_id, args.group_id, args.queue, args.gpu_type):
            raise ValueError("live GPU cluster, group, queue and GPU type are required")
        resource.update(cluster_ids=[args.cluster_id], group_id=args.group_id)
        role.update(gpu=1, gpu_type=args.gpu_type, queue_name=args.queue)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("request.json", request),
        ("check.json", dict(check_point="create", node_type="merlin_job_run", job_run_context=request)),
        ("execution.json", dict(status="PREPARED", submitted=False, name=args.name,
             phase=args.phase, seeds=args.seeds, modes=args.modes, device=args.device, threads=args.threads,
             epoch_limit=args.epoch_limit, restored_cells=list(restore))),
    ):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(dict(status="PREPARED", name=args.name, device=args.device)))


if __name__ == "__main__":
    main()
