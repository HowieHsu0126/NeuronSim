#!/usr/bin/env python3
"""Package the V1 admission test closure, then prepare a Merlin CPU request."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import tarfile
import zlib


TESTS = ["test_v1_runtime_qualification.py", "test_v1_glif_production.py",
         "test_benchmark_production.py", "test_benchmark_production_block.py",
         "test_circuit_data.py", "test_benchmark_collection.py"]


def identity(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def write(path, value):
    text = json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(path)
    else:
        path.write_text(text)


def closure(repo):
    base = repo / "Libs/population_dynamics"
    pending = [base / name for name in TESTS] + [base / "v1_runtime_qualification.py"]
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom):
                if node.level == 1:
                    modules = [node.module] if node.module else [row.name for row in node.names]
                elif node.level == 0 and node.module and node.module.startswith("population_dynamics"):
                    module = node.module.removeprefix("population_dynamics").lstrip(".")
                    modules = [module] if module else [row.name for row in node.names]
            elif isinstance(node, ast.Import):
                modules = [row.name.removeprefix("population_dynamics.") for row in node.names
                           if row.name.startswith("population_dynamics.")]
            for module in modules:
                candidate = base / (module.replace(".", "/") + ".py")
                if candidate.is_file():
                    pending.append(candidate)
    found.add(base / "__init__.py")
    return found


def package(args):
    if socket.gethostname().split(".")[0] == "mgmt2":
        raise RuntimeError("PACKAGE_ON_RECEIVE_HOST")
    files = closure(args.repo)
    files.update(args.repo / name for name in [
        "Libs/configs/neural_population_dynamics.yaml",
        "Libs/configs/neural_population_v1_circuit_development.yaml",
        "Libs/configs/neural_population_v1_constructed_visual_development.yaml",
        "Libs/scripts/run_v1_glif_network.sbatch",
        "Libs/scripts/run_v1_source_rates.sbatch",
        "Libs/scripts/run_v1_source_spikes.sbatch",
        "Libs/scripts/run_v1_native_episode.sbatch",
        "Libs/scripts/run_v1_production_block_export.sbatch",
        "Libs/scripts/v1_runtime_admission_validation.py",
        "Libs/scripts/ca3_merlin_remote.py"])
    files.update([args.qualification, args.family / "receipt.json",
                  args.family / "merlin_return_manifest.json", args.pack_results])
    if args.reuse_runtime_qualification:
        files.add(args.reuse_runtime_qualification)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.archive.exists():
        raise FileExistsError("REUSE_OR_RECONCILE_EXISTING_ARCHIVE")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with tarfile.open(args.archive, "w:gz", compresslevel=1) as archive:
        for path in sorted(files):
            if path.is_symlink() or not path.is_file():
                raise ValueError("REGULAR_DECLARED_FILE_REQUIRED:" + str(path))
            manifest[str(path)] = identity(path)
            archive.add(path, arcname=str(path).lstrip("/"), recursive=False)
    job = dict(repo=str(args.repo), original_qualification=str(args.qualification),
               family_receipt=str(args.family / "receipt.json"),
               family_manifest=str(args.family / "merlin_return_manifest.json"),
               input_files=manifest, tests=["Libs/population_dynamics/" + name for name in TESTS])
    if args.reuse_runtime_qualification:
        job["runtime_qualification_source"] = str(args.reuse_runtime_qualification)
    write(args.output / "job_inputs.json", job)
    write(args.output / "preparation.json", dict(status="PACKAGED", files=len(manifest),
          file_bytes=sum(row["bytes"] for row in manifest.values()), archive=str(args.archive),
          archive_identity=identity(args.archive)))
    print(json.dumps(dict(status="PACKAGED", files=len(manifest), archive=identity(args.archive))))


def request(args):
    job = json.loads((args.output / "job_inputs.json").read_text())
    prep = json.loads((args.output / "preparation.json").read_text())
    upload = json.loads((args.output / "upload.json").read_text())
    if (not upload["complete"] or {k: upload[k] for k in ("bytes", "sha256")} != prep["archive_identity"]
            or sum(row["bytes"] for row in upload["chunks"]) != upload["bytes"]):
        raise ValueError("VERIFIED_COMPLETE_UPLOAD_REQUIRED")
    payload = json.loads(args.template.read_text())
    shape = payload["resource_config"]["arnold_resource_config"]["roles"][0]
    if shape["gpu"] != 0 or shape["cpu"] != 4 or shape["memory_mb"] != 16384:
        raise ValueError("VALIDATED_CPU_RESOURCE_SHAPE_REQUIRED")
    job.update(upload=upload, resource_shape=shape)
    scripts = {name: (args.repo / "Libs/scripts" / name).read_text()
               for name in ("v1_runtime_admission_validation.py", "ca3_merlin_remote.py")}
    scripts["pack_results.py"] = args.pack_results.read_text()
    for name, source in scripts.items():
        compile(source, name, "exec")
    encoded = base64.b64encode(zlib.compress(json.dumps(dict(job=job, files=scripts)).encode(), 9)).decode()
    bootstrap = f"""import base64,hashlib,json,os,pathlib,subprocess,sys,zlib
payload=json.loads(zlib.decompress(base64.b64decode({encoded!r})))
trial=os.environ['ARNOLD_TRIAL_ID']; run=os.environ['RH2_JOB_RUN_ID']
work=pathlib.Path('/tmp')/('neuronsim-v1-runtime-validation-'+trial)
work.mkdir(exist_ok=False)
for name,text in payload['files'].items():(work/name).write_text(text)
(work/'job.json').write_text(json.dumps(payload['job']))
raise SystemExit(subprocess.call([sys.executable,str(work/'v1_runtime_admission_validation.py'),'--job-json',str(work/'job.json')]))
"""
    original = payload["job_config"]["job_template_config"]["entrypoint_full_script"]
    prefix = original.split("python3 - <<", 1)[0]
    # pytest is a test dependency; the numerical environment retains its verified pins.
    if "python3 -m pip install pytest==8.4.2\n" not in prefix:
        prefix += "python3 -m pip install pytest==8.4.2\n"
    script = prefix + "python3 - <<'RUNTIME_VALIDATION'\n" + bootstrap + "RUNTIME_VALIDATION\n"
    compile(bootstrap, "runtime_validation_bootstrap", "exec")
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    payload.update(name=args.name, description="Frozen V1 sm80 runtime admission: focused CPU tests and verified returned-family artifact; no new simulations.", request_context={})
    payload["job_config"]["job_template_config"]["entrypoint_full_script"] = script
    write(args.output / "request.json", payload)
    write(args.output / "check.json", dict(check_point="create", node_type="merlin_job_run", job_run_context=payload))
    (args.output / "entrypoint.sh").write_text(script)
    print(json.dumps(dict(status="PREPARED_NOT_SUBMITTED", name=args.name,
                          request_bytes=(args.output / "request.json").stat().st_size)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("package", "request"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--family", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pack-results", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--reuse-runtime-qualification", type=Path)
    args = parser.parse_args()
    (package if args.action == "package" else request)(args)


if __name__ == "__main__":
    main()
