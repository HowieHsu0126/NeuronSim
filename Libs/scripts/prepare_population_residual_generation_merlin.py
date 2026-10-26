"""Prepare, without submitting, one native residual generation run on Merlin."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys

from prepare_population_residual_fitting_merlin import DEFAULT_CONFIG, MEAN_CONFIG, validated_source_config


BOOTSTRAP = r'''
import base64,hashlib,json,os,pathlib,platform,shutil,signal,subprocess,sys,tempfile,time,traceback
root=pathlib.Path(tempfile.mkdtemp(prefix='native-residual-generation-'))
output=root/'delivery'; diagnostics=root/'launcher';diagnostics.mkdir()
identity=dict(run_id=os.environ['RH2_JOB_RUN_ID'],trial_id=os.environ['ARNOLD_TRIAL_ID'],host=platform.node())
report=dict(status='STAGING',child_exit_code=None,scientific_claim_qualified=False,seed=payload['seed'],**identity)
publisher=None;process=None;code=1;published={};publication_errors={}
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.writing')
 temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def stop(signum,frame):
 report['received_signal']=signum
 if process is not None and process.poll() is None:os.killpg(process.pid,signum)
 else:raise InterruptedError('signal '+str(signum))
for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
def publish_modes():
 for mode in ('current','history'):
  path=root/'work/results'/mode/'mode.json'
  if mode in published or mode in publication_errors or not path.is_file():continue
  receipt=json.loads(path.read_text())
  if receipt.get('status') not in ('PASS','FAILED','RESOURCE_BLOCKED'):continue
  snapshot=root/('snapshot_'+mode);snapshot.mkdir()
  shutil.copytree(root/'work/results'/mode,snapshot/'results'/mode)
  for name in ('provenance','Libs'):
   if (root/'work'/name).is_dir():shutil.copytree(root/'work'/name,snapshot/name)
  phase=snapshot/'phases';phase.mkdir()
  for name in (mode+'_evaluate',mode+'_replay'):
   if (root/'work/phases'/name).is_dir():shutil.copytree(root/'work/phases'/name,phase/name)
  write(snapshot/'execution.json',dict(scope='completed_mode',mode=mode,mode_receipt=receipt,**identity))
  try:
   published[mode]=publisher.persist(root,snapshot,prefix+'/modes/'+mode,receipt['exit_code'])
  except Exception:publication_errors[mode]=traceback.format_exc()
  write(diagnostics/'progress.json',dict(published_modes=published,publication_errors=publication_errors,**identity))
  print(json.dumps(dict(scope='native_generation_mode_publication',mode=mode,
   receipt=published.get(mode),error=publication_errors.get(mode),**identity)),flush=True)
try:
 wrapper=root/'publisher.py'
 subprocess.run(['hdfs','dfs','-get',payload['publisher']['hdfs_path'],str(wrapper)],check=True)
 with wrapper.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
 if wrapper.stat().st_size!=payload['publisher']['bytes'] or actual!=payload['publisher']['sha256']:raise ValueError('publisher identity differs')
 sys.path.insert(0,str(root));import publisher
 prefix=os.environ['RH2_CHECKPOINT_HDFS'].rstrip('/')+'/native_generation'
 staging={}
 for name in ('fitting','generation','refits'):
  target=root/name;target.mkdir();staging[name]=publisher.stage(payload['source'],payload['bundles'][name],target)
  bound=target/'inputs'/('input_inventory.json' if name=='refits' else 'manifest.json')
  if publisher.fingerprint(bound)['sha256']!=payload['manifest_sha256'][name]:raise ValueError(name+' staged manifest differs')
  (target/'inputs.tar').unlink()
 repo=root/'repo';repo.mkdir();shutil.move(str(root/'fitting/source'),repo/'Libs')
 for name,row in payload['overlays'].items():
  target=repo/'Libs'/name;target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(base64.b64decode(row['content'],validate=True));publisher.verify(target,row)
 publisher.verify(repo/'Libs'/payload['config_relative'],payload['config_identity'])
 sys.path.insert(0,str(repo/'Libs'))
 from population_dynamics.population_residual_refit_provenance import build_refit_receipt
 if build_refit_receipt(root/'refits/inputs')!=payload['refit_receipt']:raise ValueError('staged six-refit family differs')
 command=[sys.executable,str(repo/'Libs/scripts/run_population_residual_correction.py'),
  '--repo',str(repo),'--native-refits',str(root/'refits/inputs'),
  '--fitting-inputs',str(root/'fitting/inputs'),'--generation-inputs',str(root/'generation/inputs'),
  '--fitting-config',str(repo/'Libs'/payload['config_relative']),
  '--seed',str(payload['seed']),'--device','cuda','--work',str(root/'work'),'--output',str(output)]
 report.update(status='RUNNING',command=command,staging=staging)
 write(diagnostics/'execution.json',report)
 environment=dict(os.environ,PYTHONPATH=str(repo/'Libs'),PYTHONFAULTHANDLER='1')
 with (diagnostics/'generation.log').open('w') as log:
  process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while True:
   code=process.poll();publish_modes()
   if code is not None:break
   try:process.wait(timeout=15)
   except subprocess.TimeoutExpired:pass
 report.update(child_exit_code=code,status='COMPLETE' if code==0 else 'FAILED')
except Exception:
 report.update(status='FAILED_LAUNCHER',error=traceback.format_exc())
 if process is not None and process.poll() is None:
  os.killpg(process.pid,signal.SIGTERM);process.wait()
 if process is not None:report['child_exit_code']=process.returncode
 code=report['child_exit_code'] if report['child_exit_code'] not in (None,0) else 1
finally:
 report.update(exit_code=code,published_modes=published,publication_errors=publication_errors)
 write(diagnostics/'execution.json',report)
 # A runner failure can precede its archive. Always retain launcher diagnostics.
 terminal=root/'terminal';terminal.mkdir()
 if output.is_dir():shutil.move(str(output),terminal/'delivery')
 shutil.copytree(diagnostics,terminal/'launcher')
 write(terminal/'execution.json',report)
 if publisher is not None:
  try:
   receipt=publisher.persist(root,terminal,os.environ['RH2_CHECKPOINT_HDFS'].rstrip('/')+'/native_generation/terminal',code)
   print(json.dumps(dict(scope='native_generation_terminal',report=report,receipt=receipt)),flush=True)
  except Exception:
   report['terminal_publication_error']=traceback.format_exc();write(diagnostics/'execution.json',report)
   print(json.dumps(report),flush=True)
   if code==0:code=1
raise SystemExit(code)
'''


def fingerprint(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


def refit_inventory(refits, inventory):
    """Bind the portable archive inventory to the twelve canonical export files."""
    expected = {f"seed_{seed}/{mode}/{name}" for seed in (41, 42, 43)
                for mode in ("current", "history") for name in ("fit.json", "predictor.npz")}
    rows = {row["path"]: row for row in inventory}
    if len(rows) != len(inventory) or set(rows) != expected:
        raise ValueError("exactly six canonical fit/predictor pairs required")
    for name, row in rows.items():
        if fingerprint(refits / name) != {key: row[key] for key in ("bytes", "sha256")}:
            raise ValueError("refit archive inventory differs from canonical export: " + name)


def single_archive(admission):
    archive = admission["archive"]
    return dict(complete=True, bytes=archive["bytes"], sha256=archive["sha256"], chunks=[dict(archive, index=0)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign-root", "output", "refits-upload", "refits-inventory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-checks", type=Path, help="Accepted fitting source checks; input and publisher paths remain under campaign-root")
    parser.add_argument("--config-relative", default=DEFAULT_CONFIG, choices=(DEFAULT_CONFIG, MEAN_CONFIG))
    parser.add_argument("--refit-root", type=Path, help="Complete six-cell native refits; defaults to native_observation_fitting/refits")
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, choices=(41, 42, 43), required=True)
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--group-id", type=int, required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--gpu-type", required=True, help="Verified request enum, e.g. A100_SXM_80GB")
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=65536)
    args = parser.parse_args(argv)
    if not args.name.startswith("mrln_") or len(args.name) > 90:
        raise ValueError("valid mrln_ job name required")
    if args.output.exists():
        raise FileExistsError("fresh generation preparation output required")
    if args.cpu < 2 or args.memory_mb <= 32768:
        raise ValueError("two CPU threads and more than32GiB allocated memory required")
    root, fitting = args.campaign_root, args.campaign_root / "native_observation_fitting"
    checks = (args.source_checks or fitting / "checkpoint_checks/native_route_recovery").resolve()
    source, config_identity = validated_source_config(checks, args.config_relative)
    accepted = json.loads((checks / "result.json").read_text())
    bundles, digests = {}, {}
    for name, admission_path, manifest_path, count in (
        ("fitting", "input_admission/complete_inputs/analysis.json", "real_inputs/full_manifest.json", 462),
        ("generation", "generation_inputs/complete_admission/analysis.json", "generation_inputs/additional_manifest.json", 81)):
        admission = json.loads((fitting / admission_path).read_text())
        digests[name] = fingerprint(fitting / manifest_path)["sha256"]
        if (admission.get("status") != "VERIFIED_INPUT_BUNDLE" or admission.get("exit_code") != 0
                or admission.get("input_manifest_sha256") != digests[name] or admission.get("logical_files") != count):
            raise ValueError("complete authenticated " + name + " admission required")
        bundles[name] = single_archive(admission)
    sys.path.insert(0, str(args.repo / "Libs"))
    from population_dynamics.population_residual_refit_provenance import build_refit_receipt
    refits = (args.refit_root or fitting / "refits").resolve()
    family = build_refit_receipt(refits)
    if family["shared_binding"]["input_manifest_sha256"] != digests["fitting"]:
        raise ValueError("canonical refits differ from admitted fitting inputs")
    if family["shared_binding"]["config_sha256"] != config_identity["sha256"]:
        raise ValueError("native refits differ from the selected fitting config")
    bound_sources = family["shared_binding"]["sources"]
    if not isinstance(bound_sources, dict) or not bound_sources:
        raise ValueError("native refit source identities required")
    for name, digest in bound_sources.items():
        identity = accepted["sources"].get("population_dynamics/" + name)
        if (Path(name).name != name or not name.endswith(".py")
                or not isinstance(identity, dict) or identity.get("sha256") != digest):
            raise ValueError("native refits differ from the tested fitting sources: " + name)
    inventory = json.loads(args.refits_inventory.read_text())
    refit_inventory(refits, inventory)
    bundles["refits"] = json.loads(args.refits_upload.read_text())
    if bundles["refits"].get("complete") is not True:
        raise ValueError("complete portable refit upload required")
    digests["refits"] = fingerprint(args.refits_inventory)["sha256"]
    overlays = {}
    for name in ("__init__.py", "scripts/run_population_residual_correction.py",
                 "scripts/run_population_trajectory_campaign.py", "population_dynamics/population_residual_refit_provenance.py"):
        path = args.repo / "Libs" / name
        overlays[name] = dict(**fingerprint(path), content=base64.b64encode(path.read_bytes()).decode())
    payload = dict(source=source, publisher=json.loads((root / "merlin_readout_analysis/wrapper_upload.json").read_text()),
                   bundles=bundles, manifest_sha256=digests, overlays=overlays, refit_receipt=family, seed=args.seed,
                   config_relative=args.config_relative, config_identity=config_identity)
    encoded = base64.b64encode(gzip.compress(json.dumps(payload).encode(), mtime=0)).decode()
    entry = ("#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2\n"
        "python3 -m pip install --no-deps scipy==1.16.2 PyYAML==6.0.3\n"
        "python3 - <<'NATIVE_RESIDUAL_GENERATION'\nimport base64,gzip,json\npayload=json.loads(gzip.decompress(base64.b64decode("
        + repr(encoded) + ")))\n" + BOOTSTRAP + "\nNATIVE_RESIDUAL_GENERATION\n")
    request = json.loads((checks / "request.json").read_text())
    request.pop("request_context", None)
    request.update(name=args.name, description="Native residual generation with complete frozen parents, six refits, full-unit matched evaluation and independent all-origin replay.")
    template = request["job_config"]["job_template_config"]
    template["entrypoint_full_script"] = entry
    template["env_map"].update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    # The source checks can use a CPU template; Merlin sets GPU visibility.
    template["env_map"].pop("CUDA_VISIBLE_DEVICES", None)
    resource = request["resource_config"]["arnold_resource_config"]
    resource.update(cluster_ids=[args.cluster_id], group_id=args.group_id)
    resource["roles"][0].update(cpu=args.cpu, memory_mb=args.memory_mb, gpu=1, gpu_type=args.gpu_type, queue_name=args.queue)
    args.output.mkdir(parents=True)
    for name, value in (("request.json", request), ("check.json", dict(check_point="create", node_type="merlin_job_run", job_run_context=request)),
        ("execution.json", dict(status="PREPARED", submitted=False, name=args.name, seed=args.seed,
            manifest_sha256=digests, refit_receipt=family, source=source,
            source_checks=str(checks), config_relative=args.config_relative, config_identity=config_identity,
            refit_root=str(refits), cpu_threads=2,
            source_units=100000, fields=128, episodes=16, origins=48, score_rows=144, joint_draws=16))):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(dict(status="PREPARED", submitted=False, name=args.name, seed=args.seed)))


if __name__ == "__main__":
    main()
