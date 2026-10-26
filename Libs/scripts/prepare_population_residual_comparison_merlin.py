"""Prepare native residual count or field comparisons from completed Merlin outputs."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import inspect
import itertools
import shutil
import tarfile
from pathlib import Path, PurePosixPath

SEEDS = (41, 42, 43)

BOOTSTRAP = r'''
import base64,hashlib,itertools,json,os,pathlib,platform,shutil,subprocess,sys,tarfile,tempfile,traceback
from pathlib import Path,PurePosixPath
root=pathlib.Path(tempfile.mkdtemp(prefix='native-residual-comparison-'))
output=root/'results';publisher=None;code=1
report=dict(status='STAGING',exit_code=None,run_id=os.environ['RH2_JOB_RUN_ID'],
 trial_id=os.environ['ARNOLD_TRIAL_ID'],host=platform.node(),scope=payload.get('scope','comparison_metadata_only'),
 large_arrays_or_checkpoints_read=False,training_repeated=False,generation_repeated=False,replay_repeated=False)
try:
 p=payload['publisher'];wrapper=root/'publisher.py'
 subprocess.run(['hdfs','dfs','-get',p['hdfs_path'],str(wrapper)],check=True)
 with wrapper.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
 if wrapper.stat().st_size!=p['bytes'] or actual!=p['sha256']:raise ValueError('publisher identity differs')
 sys.path.insert(0,str(root));import publisher
 staging=publisher.stage(payload['source'],payload['inputs'],root)
 if publisher.fingerprint(root/'inputs/input_inventory.json')['sha256']!=payload['inventory_sha256']:raise ValueError('comparison input inventory differs')
 for row in json.loads((root/'inputs/input_inventory.json').read_text()):
  publisher.verify(root/'inputs'/row['path'],row)
 for name,row in payload['overlays'].items():
  path=root/'source/population_dynamics'/name
  path.write_bytes(base64.b64decode(row['content'],validate=True));publisher.verify(path,row)
 origins=[]
 for item in payload['metadata']:
  seed=item['seed'];receipt=item['receipt'];expected=item['export']
  archive=root/('seed_'+str(seed)+'.tar.gz')
  publisher.fetch(receipt['checkpoint_path']+'/results.tar.gz',archive)
  publisher.verify(archive,{'bytes':receipt['result_bytes'],'sha256':receipt['result_sha256']})
  target=root/('metadata_'+str(seed));target.mkdir()
  rows={row['path']:row for row in expected['returned_files']}
  seen=set();files=set()
  with tarfile.open(archive,'r:*') as bundle:
   for member in bundle:
    name=pathlib.PurePosixPath(member.name)
    if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0]!='results' or str(name) in seen:raise ValueError('unsafe metadata archive member')
    seen.add(str(name))
    if member.isdir():continue
    if not member.isfile():raise ValueError('regular metadata files required')
    relative=pathlib.Path(*name.parts[1:]).as_posix()
    if relative not in rows and relative not in ('analysis.json','execution.json'):raise ValueError('unlisted metadata file')
    destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
    with bundle.extractfile(member) as source,destination.open('xb') as stream:shutil.copyfileobj(source,stream,1024*1024)
    if relative in rows:publisher.verify(destination,rows[relative])
    files.add(relative)
  if files!=set(rows)|{'analysis.json','execution.json'}:raise ValueError('incomplete metadata export archive')
  exported=json.loads((target/'analysis.json').read_text())
  if any(exported.get(key)!=value for key,value in expected.items()):raise ValueError('CPU export report differs from accepted metadata')
  source_execution=json.loads((target/'source_execution.json').read_text())
  if source_execution.get('run_id')!=expected['source_run'] or str(source_execution.get('trial_id'))!=expected['source_trial'] or source_execution.get('exit_code')!=0:raise ValueError('generation execution identity differs')
  if publisher.fingerprint(target/'delivery/delivery.json')['sha256']!=expected['delivery_sha256']:raise ValueError('accepted delivery differs')
  correction=root/'corrections'/('seed_'+str(seed));correction.parent.mkdir(exist_ok=True)
  shutil.move(str(target/'delivery'),correction)
  origins.append(dict(seed=seed,source_run=expected['source_run'],source_trial=expected['source_trial'],metadata_archive=receipt,delivery_sha256=expected['delivery_sha256']))
  if 'source_archive' in item:
   original=item['source_archive']
   if exported.get('source_archive')!=original:raise ValueError('generation source archive binding differs')
   full_archive=root/('generation_'+str(seed)+'.tar.gz')
   publisher.fetch(original['checkpoint_path']+'/results.tar.gz',full_archive)
   publisher.verify(full_archive,{'bytes':original['result_bytes'],'sha256':original['result_sha256']})
   delivery=json.loads((correction/'delivery.json').read_text())
   arrays=extract_field_arrays(full_archive,correction,delivery['files'],publisher.verify)
   origins[-1].update(source_archive={key:original[key] for key in ('checkpoint_path','result_bytes','result_sha256')},
                     field_array_files=arrays)
   full_archive.unlink()
   report.update(large_arrays_or_checkpoints_read=True,full_unit_samples_extracted=False)

 command=[sys.executable,'-m','population_dynamics.population_residual_correction_analysis',
  '--corrections',str(root/'corrections'),'--parent-campaign',str(root/'inputs/parent'),
  '--eag-comparison',str(root/'inputs/eag/comparison.json'),'--eag-acceptance',str(root/'inputs/eag/acceptance.json'),
  '--output',str(output)]
 if payload.get('scope')=='native_field_comparison':
  command=[sys.executable,'-m','population_dynamics.memory_field_comparison',
   '--native-corrections',str(root/'corrections'),'--parent-campaign',str(root/'inputs/parent'),
   '--baseline-population-campaign',str(root/'inputs/baseline_population'),
   '--braid-comparison',str(root/'inputs/braid'),'--memory-comparison',str(root/'inputs/memory'),
   '--output',str(output)]
 if payload.get('scope')=='native_ldns_comparison':
  command=[sys.executable,'-m','population_dynamics.population_residual_correction_analysis',
   '--corrections',str(root/'corrections'),'--parent-campaign',str(root/'inputs/parent'),
   '--ldns-comparison',str(root/'inputs/ldns'),'--field-comparison',str(root/'inputs/field'),
   '--output',str(output)]
 if payload.get('scope')=='native_forecast_decomposition':
  command=[sys.executable,'-m','population_dynamics.population_native_forecast_decomposition',
   '--corrections',str(root/'corrections'),'--parent-campaign',str(root/'inputs/parent'),
   '--ldns-comparison',str(root/'inputs/ldns'),'--field-comparison',str(root/'inputs/field'),
   '--ldns-forecasts',str(root/'inputs/ldns_forecasts'),
   '--field-scale',str(root/'inputs/field/fitting_field_scale.npz'),'--output',str(output)]
 report.update(command=command,staging=staging,input_origins=origins)
 with (root/'comparison.log').open('w') as log:
  child=subprocess.run(command,env=dict(os.environ,PYTHONPATH=str(root/'source')),stdout=log,stderr=subprocess.STDOUT)
 code=child.returncode;report.update(status='COMPLETE' if code==0 else 'FAILED_COMPARISON',child_exit_code=code)
except Exception:report.update(status='FAILED_COMPARISON_LAUNCHER',error=traceback.format_exc())
finally:
 output.mkdir(exist_ok=True)
 report['exit_code']=code
 (output/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
 if (root/'comparison.log').is_file():shutil.copy2(root/'comparison.log',output/'comparison.log')
 if publisher is not None:
  namespace=payload.get('scope') if payload.get('scope') in ('native_field_comparison','native_ldns_comparison','native_forecast_decomposition') else 'native_comparison'
  receipt=publisher.persist(root,output,os.environ['RH2_CHECKPOINT_HDFS'].rstrip('/')+'/'+namespace,code)
  print(json.dumps(dict(report=report,receipt=receipt)),flush=True)
raise SystemExit(code)
'''



def field_array_paths():
    """The saved scales and origin fields consumed by both correction modes."""
    return {f"results/{mode}/forecasts/{name}" for mode in ("current", "history")
            for name in (["fitting_field_scale.npz"] +
                [f"train_block_{block:03d}_familiar_{template}_origin_{origin}.npz"
                 for block, template, origin in itertools.product(range(14, 18), range(4), (1000, 3000, 5000))])}


def extract_field_arrays(archive, target, files, verify):
    """Extract the 98 manifest-bound field files; leave full-unit samples archived."""
    target = Path(target)
    index = {row["path"]: row for row in files}
    wanted = field_array_paths()
    if len(index) != len(files) or not wanted <= set(index):
        raise ValueError("complete unique field file manifest required")
    selected = {"results/delivery/" + name: name for name in wanted}
    seen, extracted = set(), set()
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or str(name) in seen:
                raise ValueError("unsafe generation archive member")
            seen.add(str(name))
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("regular generation archive files required")
            if str(name) not in selected:
                continue
            relative = selected[str(name)]
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, path.open("xb") as destination:
                shutil.copyfileobj(source, destination, 1024 * 1024)
            verify(path, index[relative])
            extracted.add(relative)
    if extracted != wanted:
        raise ValueError("generation archive lacks required field arrays")
    return [index[name] for name in sorted(extracted)]

def admit_metadata(seed, receipt, export, intake):
    """Require the completed CPU export and its local intake for this seed."""
    rows = export.get("returned_files", [])
    paths = [row["path"] for row in rows]
    modes = intake.get("modes", {})
    if (seed not in SEEDS or receipt.get("persisted") is not True or receipt.get("exit_code") != 0
            or export.get("status") != "VERIFIED_GENERATION_METADATA" or export.get("exit_code") != 0
            or export.get("seed") != seed or intake.get("seed") != seed
            or intake.get("status") != "VERIFIED_GENERATION_METADATA"
            or export.get("scope") != "comparison_metadata_return"
            or export.get("large_arrays_returned") is not False or intake.get("large_arrays_returned") is not False
            or export.get("scientific_computation_repeated") is not False
            or export.get("full_output_persisted") is not True
            or not export.get("source_run") or export["source_run"] != intake.get("source_run")
            or not export.get("source_trial") or export.get("delivery_sha256") != intake.get("delivery_sha256")
            or intake.get("files") != len(rows) or not rows or len(set(paths)) != len(paths)
            or set(export.get("complete_modes", [])) != {"current", "history"}
            or set(modes) != {"current", "history"}
            or any(row.get("status") != "PASS" or row.get("exit_code") != 0 for row in modes.values())):
        raise ValueError(f"complete accepted metadata required for seed {seed}")
    for row in rows:
        path = Path(row["path"])
        if (path.is_absolute() or ".." in path.parts or path.suffix not in (".json", ".csv", ".yaml")
                or type(row.get("bytes")) is not int or row["bytes"] < 0 or not row.get("sha256")):
            raise ValueError("metadata-only relative export paths required")
    index = {row["path"]: row for row in rows}
    if index.get("delivery/delivery.json", {}).get("sha256") != export["delivery_sha256"]:
        raise ValueError("returned delivery identity differs")
    keys = ("status", "exit_code", "seed", "source_run", "source_trial", "scope",
            "large_arrays_returned", "scientific_computation_repeated", "full_output_persisted",
            "delivery_sha256", "returned_files", "complete_modes")
    return dict(seed=seed, receipt=receipt, export={key: export[key] for key in keys})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--generation-root", type=Path, help="Generation directory containing seed_N/metadata_return; defaults to native_observation_fitting/generation")
    parser.add_argument("--source-checks", type=Path, help="Accepted source-check directory; baseline input and publisher paths remain unchanged")
    parser.add_argument("--request-template", type=Path, help="Merlin request JSON used as the CPU comparison template")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    references = parser.add_mutually_exclusive_group()
    references.add_argument("--field-inputs", type=Path, help="Uploaded thirty-file field comparison input bundle")
    references.add_argument("--ldns-inputs", type=Path, help="Uploaded native parent, LDNS and accepted field-linkage metadata")
    references.add_argument("--decomposition-inputs", type=Path, help="Uploaded accepted metadata and saved LDNS field trajectories")
    args = parser.parse_args(argv)
    if not args.name.startswith("mrln_") or len(args.name) > 90:
        raise ValueError("valid mrln_ job name required")
    if args.output.exists():
        raise FileExistsError("fresh comparison preparation required")
    root, fitting = args.campaign_root, args.campaign_root / "native_observation_fitting"
    generation = (args.generation_root or fitting / "generation").resolve()
    checks = (args.source_checks or fitting / "checkpoint_checks/native_route_recovery").resolve()
    request_template = (args.request_template or fitting / "generation_checks/request.json").resolve()
    metadata = []
    for seed in SEEDS:
        directory = generation / f"seed_{seed}" / "metadata_return"
        values = [json.loads((directory / name).read_text()) for name in ("receipt.json", "analysis.json", "intake.json")]
        item = admit_metadata(seed, *values)
        if args.field_inputs is not None or args.decomposition_inputs is not None:
            archive = values[1]["source_archive"]
            if (archive.get("persisted") is not True or archive.get("exit_code") != 0
                    or archive.get("result_bytes", 0) <= 0 or not archive.get("result_sha256")):
                raise ValueError("accepted complete generation archive required")
            item["source_archive"] = archive
        metadata.append(item)
    input_root = args.field_inputs or args.ldns_inputs or args.decomposition_inputs or fitting / "comparison_inputs"
    inventory_path = input_root / "input_inventory.json"
    raw = inventory_path.read_bytes()
    inventory = json.loads(raw)
    expected = {f"parent/seed_{seed}/{name}" for seed in SEEDS
                for name in ("delivery.json", "forecasts/evaluation.json", "forecasts/scores.csv", "replay.json")}
    expected |= {"eag/acceptance.json", "eag/comparison.json", "eag/joint_comparison_all_seeds.csv"}
    if args.field_inputs is not None:
        expected = {path for path in expected if path.startswith("parent/")}
        expected |= {f"parent/seed_{seed}/training/run.json" for seed in SEEDS}
        expected |= {f"baseline_population/seed_{seed}/{name}" for seed in SEEDS
                     for name in ("delivery.json", "training/run.json", "forecasts/fitting_field_scale.npz")}
        expected |= {f"{baseline}/{name}" for baseline, table in (
                     ("braid", "mean_comparison_all_seeds.csv"), ("memory", "field_comparison_all_seeds.csv"))
                     for name in ("delivery.json", "comparison/comparison.json", "comparison/" + table)}
    if args.ldns_inputs is not None or args.decomposition_inputs is not None:
        expected = {path for path in expected if path.startswith("parent/")}
        expected |= {"ldns/" + name for name in ("delivery.json", "source_delivery.json", "source_command.json",
                     "analysis/comparison.json", "analysis/joint_comparison_all_seeds.csv")}
        expected |= {"field/intake.json", "field/comparison.json"}
    if args.decomposition_inputs is not None:
        expected.add("field/fitting_field_scale.npz")
        expected |= {f"ldns_forecasts/seed_{seed}/delivery.json" for seed in SEEDS}
        expected |= {f"ldns_forecasts/seed_{seed}/forecasts/train_block_{block:03d}_familiar_{template}_origin_{origin}.npz"
                     for seed, block, template, origin in itertools.product(SEEDS, range(14, 18), range(4), (1000, 3000, 5000))}
    if len(inventory) != len(expected) or {row["path"] for row in inventory} != expected:
        raise ValueError("complete declared comparison input bundle required")
    inputs = json.loads((input_root / "upload.json").read_text())
    source = json.loads((checks / "source_upload.json").read_text())
    if inputs.get("complete") is not True or source.get("complete") is not True:
        raise ValueError("complete existing source and comparison uploads required")
    if args.source_checks is not None:
        validation = json.loads((checks / "result.json").read_text())
        if (validation.get("status") != "PASS" or validation.get("exit_code") != 0
                or validation.get("tests", 0) < 1
                or any(validation.get(key, 1) for key in ("failures", "errors", "skipped"))
                or not source.get("hdfs_path") or not source.get("sha256")
                or validation.get("source_payload_sha256") != source["sha256"]):
            raise ValueError("explicit source checks require the exact tested source payload")
    overlays = {}
    names = ["population_residual_correction_analysis.py", "population_residual_refit_provenance.py", "provenance.py"]
    if args.field_inputs is not None:
        names.extend(["memory_field_comparison.py", "lif_neural_field.py", "braid_population_comparison.py"])
    if args.decomposition_inputs is not None:
        names.extend(["population_native_forecast_decomposition.py", "trajectory_scores.py", "lif_neural_field.py",
                      "braid_population_comparison.py"])
    for name in names:
        content = (args.repo / "Libs/population_dynamics" / name).read_bytes()
        overlays[name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                              content=base64.b64encode(content).decode())
    payload = dict(source=source, inputs=inputs, metadata=metadata, overlays=overlays,
        inventory_sha256=hashlib.sha256(raw).hexdigest(),
        publisher=json.loads((root / "merlin_readout_analysis/wrapper_upload.json").read_text()))
    payload["scope"] = ("native_forecast_decomposition" if args.decomposition_inputs is not None else
                        "native_field_comparison" if args.field_inputs is not None else
                        "native_ldns_comparison" if args.ldns_inputs is not None else "comparison_metadata_only")
    encoded = base64.b64encode(gzip.compress(json.dumps(payload).encode(), mtime=0)).decode()
    helpers = "\n".join(inspect.getsource(function) for function in (field_array_paths, extract_field_arrays))
    bootstrap = BOOTSTRAP.replace("root=pathlib.Path", helpers + "\nroot=pathlib.Path", 1)
    entry = ("#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2\n"
        "python3 -m pip install --no-deps scipy==1.16.2 PyYAML==6.0.3\n"
        "python3 - <<'NATIVE_COMPARISON'\nimport base64,gzip,json\npayload=json.loads(gzip.decompress(base64.b64decode("
        + repr(encoded) + ")))\n" + bootstrap + "\nNATIVE_COMPARISON\n")
    request = json.loads(request_template.read_text())
    request.pop("request_context", None)
    request.update(name=args.name, description="Complete descriptive native residual comparison from accepted outputs and existing baselines. Preserve all conditions and fairness limits.")
    template = request["job_config"]["job_template_config"]
    template["entrypoint_full_script"] = entry
    template["env_map"].update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    resource = request["resource_config"]["arnold_resource_config"]
    resource.update(cluster_ids=[20], group_id=271)
    resource["roles"][0].update(cpu=4, memory_mb=8192, gpu_type="CPU_ONLY", queue_name="compute-49-hl-cloudnative-am-ray.cpu-guarantee")
    resource["roles"][0].pop("gpu", None)
    args.output.mkdir(parents=True)
    execution = dict(status="PREPARED", submitted=False, name=args.name, training_seeds=list(SEEDS),
                     scope=payload["scope"], metadata=metadata,
                     generation_root=str(generation), source_checks=str(checks), request_template=str(request_template),
                     inventory_sha256=payload["inventory_sha256"],
                     software_sources={name: row["sha256"] for name, row in overlays.items()})
    for name, value in (("request.json", request), ("check.json", dict(check_point="create", node_type="merlin_job_run", job_run_context=request)),
                        ("execution.json", execution)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(dict(status="PREPARED", submitted=False, training_seeds=list(SEEDS))))


if __name__ == "__main__":
    main()
