#!/usr/bin/env python3
"""Prepare a Merlin CPU request for the nine completed V1 construction intakes.

This command does not submit jobs. The generated job verifies the bound archive,
executes the existing 16 tests in a child, and then runs the existing intake CLI.
Scientific inputs stay at their archived paths inside the container; acceptance
outputs and diagnostics stay under a new isolated /tmp work directory.
"""
from __future__ import annotations

import argparse
import ast
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zlib

from prepare_ca3_merlin_replay import publish, read_json, validate_upload
from prepare_v1_merlin_readout import cpu_shape


EXPECTED_TEST_COUNT = 16
PYTHON_PACKAGES = (
    "numpy==2.4.2", "pandas==2.3.3", "scipy==1.16.2", "h5py==3.14.0",
    "PyYAML==6.0.3", "psutil==7.1.0", "matplotlib==3.10.7",
    "scikit-image==0.25.2", "sympy==1.14.0",
)
ENTRYPOINT_PREFIX = (
    "#!/bin/bash\nset -euo pipefail\n"
    "export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "
    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
)

TEST_RUNNER = r"""import hashlib,json,os,pathlib,sys,traceback,unittest
report_path=pathlib.Path(sys.argv[1])
report=json.loads(report_path.read_text())
report['child_pid']=os.getpid()
report['python_version']=sys.version
try:
    for name,expected in report['expected_sources'].items():
        data=pathlib.Path(name).read_bytes()
        identity=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        report['tested_sources'][name]=identity
        if identity != expected:
            raise ValueError('V1_INTAKE_TEST_SOURCE_IDENTITY_DIFFERS:'+name)
    suite=unittest.defaultTestLoader.loadTestsFromName(report['test_module'])
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
    report.update(testsRun=result.testsRun,failures=len(result.failures),
                  errors=len(result.errors),skips=len(result.skipped))
    report['success']=(result.wasSuccessful() and result.testsRun==report['expected_tests']
                       and not result.skipped)
except BaseException as error:
    report.update(success=False,runner_error_type=type(error).__name__,runner_error=str(error))
    traceback.print_exc()
finally:
    report['status']='PASS' if report['success'] else 'FAILED'
    temporary=report_path.with_name(report_path.name+'.partial')
    temporary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    temporary.replace(report_path)
raise SystemExit(0 if report['success'] else 1)
"""

RUNTIME = r"""from datetime import datetime,timezone
import importlib.metadata
import json,os,pathlib,socket,subprocess,sys
import ca3_merlin_remote as transport


def require(condition,message):
    if not condition:
        raise ValueError(message)


def identity(path):
    return dict(bytes=path.stat().st_size,sha256=transport.digest(path))


def runtime_tests(job,work,output,environment):
    report_path=output/'runtime_tests.json'
    log_path=output/'runtime_tests.log'
    report=dict(status='NOT_RUN',scope='isolated_child_process',testsRun=0,
        failures=0,errors=0,skips=0,success=False,expected_tests=job['expected_tests'],
        test_module='population_dynamics.test_v1_construction_intake_merlin',
        expected_sources=job['test_sources'],tested_sources={},log_path=str(log_path))
    transport.write_json(report_path,report)
    child=None
    try:
        with log_path.open('x',encoding='utf-8') as log:
            child=subprocess.run([sys.executable,'-B','-c',job['test_runner'],str(report_path)],
                cwd=job['scripts_directory'],env=environment,stdout=log,
                stderr=subprocess.STDOUT,check=False)
    finally:
        report=json.loads(report_path.read_text())
        success=(child is not None and child.returncode==0 and report.get('success') is True
            and report.get('testsRun')==job['expected_tests']
            and all(report.get(key)==0 for key in ('failures','errors','skips'))
            and report.get('tested_sources')==job['test_sources'])
        report.update(status='PASS' if success else 'FAILED',success=success,
                      child_returncode=child.returncode if child is not None else None)
        transport.write_json(report_path,report)
    return report


def intake_command(job,work,intake_output):
    command=list(job['staging_command'])
    command[0]=sys.executable
    command[2]='run'
    for flag in ('--output','--resource-shape'):
        if flag in command:
            index=command.index(flag)
            del command[index:index+2]
    command.extend(['--resource-shape',str(work/'resource_shape.json'),
                    '--output',str(intake_output)])
    return command


def validate_intake(report,intake_output,expected_tasks):
    require(report.get('status')=='PASS_NINE_V1_CONSTRUCTION_INTAKES'
        and report.get('exit_code')==0
        and report.get('validated_recurrent_acceptances')==5
        and report.get('validated_input_acceptances')==4,
        'V1_INTAKE_COMPLETE_NINE_ACCEPTANCE_REPORT_REQUIRED')
    for key in ('source_simulations','production_episodes','canonical_acceptances_written',
                'future_reference_preparations'):
        require(report.get(key)==0,'V1_INTAKE_UNEXPECTED_ACTIVITY:'+key)
    artifacts=report.get('acceptances',[])
    actual=[[row['stage'],row['circuit'],row['source_job_id']] for row in artifacts]
    require(actual==expected_tasks,'V1_INTAKE_ACCEPTANCE_SOURCE_IDENTITIES_DIFFER')
    statuses=dict(recurrent='PASS_RECURRENT_SOURCE_DELIVERY_AND_REPLAY',
                  inputs='PASS_SOURCE_INPUT_DELIVERY_AND_REPLAY')
    for row in artifacts:
        path=pathlib.Path(row['acceptance'])
        require(path.is_absolute() and path.resolve().is_relative_to(intake_output)
                and identity(path)==row['acceptance_identity'],
                'V1_INTAKE_RETURNED_ACCEPTANCE_IDENTITY_DIFFERS')
        acceptance=json.loads(path.read_text())
        require(acceptance.get('status')==statuses[row['stage']]
                and acceptance.get('source_job_id')==row['source_job_id']
                and acceptance.get('source_simulations')==0 and acceptance.get('production_episodes')==0,
                'V1_INTAKE_RETURNED_ACCEPTANCE_SCOPE_DIFFERS')


def main():
    work=pathlib.Path(__file__).resolve().parent
    job=json.loads((work/'job.json').read_text())
    run=os.environ.get('RH2_JOB_RUN_ID')
    trial=os.environ.get('ARNOLD_TRIAL_ID')
    checkpoint=os.environ.get('RH2_CHECKPOINT_HDFS')
    require(run and trial and checkpoint and checkpoint.startswith('hdfs://')
            and socket.gethostname().split('.')[0]!='mgmt2',
            'ACTUAL_MERLIN_INTAKE_RUN_TRIAL_AND_CHECKPOINT_REQUIRED')
    output=work/'results'
    output.mkdir(exist_ok=False)
    target=checkpoint.rstrip('/')+'/v1/construction_intake/'+trial
    receipt=dict(status='PREPARING_MERLIN_V1_CONSTRUCTION_INTAKE',execution_backend='merlin',
        merlin_job_run_id=run,merlin_trial_id=trial,hostname=socket.gethostname(),output=str(output),
        started_utc=datetime.now(timezone.utc).isoformat(),persisted=False,
        resource_shape=job['resource_shape'],input_bindings=job['input_bindings'],
        original_staging_command=job['staging_command'],archive_sha256=job['upload']['sha256'],
        archive_bytes=job['upload']['bytes'],source_simulations=0,production_episodes=0,
        canonical_acceptances_written=0,future_reference_preparations=0)
    transport.write_json(output/'merlin_execution.json',receipt)
    target_ready=False
    exit_code=1
    try:
        subprocess.run(['hdfs','dfs','-mkdir','-p',target],check=True)
        target_ready=True
        require(job['resource_shape']['name']=='head' and os.environ.get('ARNOLD_HEAD_GPU')=='0'
            and os.environ.get('ARNOLD_APPLIED_GPU_NUM')=='0'
            and not os.environ.get('CUDA_VISIBLE_DEVICES'),
            'V1_INTAKE_ACTUAL_MERLIN_CPU_ROLE_REQUIRED')
        cpus=os.environ.get('ARNOLD_HEAD_CPU','')
        require(cpus.isascii() and cpus.isdecimal() and int(cpus)>=job['resource_shape']['cpu'],
                'V1_INTAKE_ACTUAL_CPU_ALLOCATION_BELOW_REQUEST')
        receipt['memory_evidence']=transport.memory_evidence(job['resource_shape'],required_available_gib=4)
        receipt['python_version']=sys.version
        receipt['launcher_sources']={name:identity(work/name) for name in job['launcher_sources']}
        require(receipt['launcher_sources']==job['launcher_sources'],'V1_INTAKE_LAUNCHER_SOURCE_CHANGED')
        setup_command=[sys.executable,'-m','pip','install',*job['python_packages']]
        receipt['environment_setup_command']=setup_command
        with (output/'environment_setup.log').open('x',encoding='utf-8') as log:
            subprocess.run(setup_command,stdout=log,stderr=subprocess.STDOUT,check=True)
        receipt['environment_versions']={name:importlib.metadata.version(name) for name in
            ('numpy','pandas','scipy','h5py','PyYAML','psutil','matplotlib','scikit-image','sympy')}
        stage=work/'source_inputs'
        stage.mkdir()
        archive=transport.download_inputs(dict(upload=job['upload'],input_files=job['input_files']),stage)
        transport.stage_inputs(archive,job['input_files'])
        receipt['verified_staged_input_files']=len(job['input_files'])
        transport.write_json(work/'resource_shape.json',job['resource_shape'])
        environment=dict(os.environ)
        environment.update(PYTHONPATH=os.pathsep.join((job['scripts_directory'],job['libs_directory'])),
            CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(work/'matplotlib'))
        tests=runtime_tests(job,work,output,environment)
        receipt['runtime_tests']={key:tests[key] for key in
            ('status','testsRun','failures','errors','skips','success','child_returncode')}
        for name in ('runtime_tests.json','runtime_tests.log'):
            transport.hdfs_put(output/name,target+'/'+name)
        require(tests['success'],'V1_CONSTRUCTION_INTAKE_RUNTIME_TESTS_FAILED')
        intake_output=output/'intake'
        command=intake_command(job,work,intake_output)
        receipt.update(status='RUNNING_MERLIN_V1_CONSTRUCTION_INTAKE',intake_command=command)
        transport.write_json(output/'merlin_execution.json',receipt)
        with (output/'intake.log').open('x',encoding='utf-8') as log:
            child=subprocess.run(command,cwd=job['scripts_directory'],env=environment,
                stdout=log,stderr=subprocess.STDOUT,check=False)
        receipt['intake_exit_code']=child.returncode
        require(child.returncode==0,'V1_CONSTRUCTION_INTAKE_CHILD_FAILED:'+str(child.returncode))
        report_path=intake_output/'report.json'
        report=json.loads(report_path.read_text())
        validate_intake(report,intake_output,job['expected_tasks'])
        receipt.update(status='PASS_MERLIN_V1_CONSTRUCTION_INTAKE',intake_report=str(report_path),
            intake_report_identity=identity(report_path),validated_recurrent_acceptances=5,
            validated_input_acceptances=4)
        exit_code=0
    except BaseException as error:
        receipt.update(status='FAILED_MERLIN_V1_CONSTRUCTION_INTAKE',error_type=type(error).__name__,
                       error=str(error))
        exit_code=130 if isinstance(error,KeyboardInterrupt) else 1
    finally:
        receipt.update(exit_code=exit_code,finished_utc=datetime.now(timezone.utc).isoformat())
        transport.write_json(output/'merlin_terminal.json',receipt)
        if target_ready:
            try:
                transport.persist(output,work,target,receipt)
            except BaseException as error:
                exit_code=exit_code or 1
                receipt.update(status='FAILED_MERLIN_V1_INTAKE_PERSISTENCE',persisted=False,
                    exit_code=exit_code,persistence_error_type=type(error).__name__,persistence_error=str(error))
                transport.write_json(output/'merlin_terminal.json',receipt)
        print(json.dumps(dict(status=receipt['status'],exit_code=exit_code,persisted=receipt['persisted'],
            merlin_job_run_id=run,merlin_trial_id=trial,output=str(output),source_simulations=0,
            production_episodes=0,canonical_acceptances_written=0)),flush=True)
    return exit_code


if __name__=='__main__':
    raise SystemExit(main())
"""

BOOTSTRAP = r"""import base64,json,os,pathlib,re,subprocess,sys,zlib
payload=json.loads(zlib.decompress(base64.b64decode(PAYLOAD_LITERAL)))
run=os.environ.get('RH2_JOB_RUN_ID')
trial=os.environ.get('ARNOLD_TRIAL_ID')
if not run or not trial or not re.fullmatch(r'[A-Za-z0-9_.-]+',trial):
    raise ValueError('ACTUAL_MERLIN_CPU_RUN_AND_TRIAL_REQUIRED')
work=pathlib.Path('/tmp')/('neuronsim-v1-construction-intake-'+trial)
work.mkdir(exist_ok=False)
for name,text in payload['files'].items():
    (work/name).write_text(text,encoding='utf-8')
(work/'job.json').write_text(json.dumps(payload['job'],indent=2)+'\n',encoding='utf-8')
raise SystemExit(subprocess.call([sys.executable,'-B',str(work/'run_intake.py')]))
"""


def identity_bytes(data: bytes) -> dict:
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def argument(command: list[str], flag: str, required: bool = True) -> str | None:
    if any(item.startswith(flag + "=") for item in command):
        raise ValueError("Use separate argv values for " + flag)
    count = command.count(flag)
    if count == 0 and not required:
        return None
    if count != 1:
        raise ValueError("Exactly one staging argument required: " + flag)
    index = command.index(flag)
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise ValueError("Missing staging argument value: " + flag)
    return command[index + 1]


def bound_source(path: Path, inputs: dict) -> bytes:
    data = path.read_bytes()
    if inputs.get(str(path)) != identity_bytes(data):
        raise ValueError("Source differs from the uploaded intake archive: " + str(path))
    return data


def prepare(args: argparse.Namespace) -> dict:
    if re.fullmatch(r"mrln_[A-Za-z0-9_-]{1,85}", args.name) is None:
        raise ValueError("Use a mrln_ job name with at most 90 characters")
    command_bytes = args.staging_command.read_bytes()
    command = json.loads(command_bytes)
    if (not isinstance(command, list) or not all(isinstance(item, str) and item for item in command)
            or len(command) < 3 or command[2] != "staging"
            or Path(command[1]).name != "v1_construction_intake_merlin.py"
            or not Path(command[1]).is_absolute()):
        raise ValueError("The original Python/script/staging argv is required")
    inputs = read_json(args.input_files)
    upload = read_json(args.upload_manifest)
    validate_upload(upload)
    for name, row in inputs.items():
        path = Path(name)
        if (not path.is_absolute() or ".." in path.parts or type(row.get("bytes")) is not int
                or row["bytes"] < 0 or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is None):
            raise ValueError("Invalid bound intake input identity: " + name)
    for flag in ("--qualification-root", "--construction-root", "--source-root"):
        if not Path(argument(command, flag)).is_absolute():
            raise ValueError("Absolute frozen root required: " + flag)
    for family in ("cohort", "construction"):
        path = argument(command, "--" + family + "-manifest")
        sha = argument(command, "--" + family + "-manifest-sha256")
        if inputs.get(path, {}).get("sha256") != sha:
            raise ValueError("Staging manifest binding differs from uploaded archive: " + family)
    for flag in ("--output", "--resource-shape"):
        argument(command, flag, required=False)
    consumer = Path(command[1])
    scripts_directory, libs_directory = consumer.parent, consumer.parent.parent
    test_path = libs_directory / "population_dynamics/test_v1_construction_intake_merlin.py"
    transport_path = scripts_directory / "ca3_merlin_remote.py"
    closure_helper = scripts_directory / "stage_v1_merlin_block_inputs.py"
    consumer_data = bound_source(consumer, inputs)
    test_data = bound_source(test_path, inputs)
    transport_data = bound_source(transport_path, inputs)
    bound_source(closure_helper, inputs)
    consumer_tree = ast.parse(consumer_data, filename=str(consumer))
    declarations = [node for node in consumer_tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "TASKS" for target in node.targets)]
    if len(declarations) != 1:
        raise ValueError("The bound intake TASKS declaration is required")
    tasks = ast.literal_eval(declarations[0].value)
    if (not isinstance(tasks, tuple) or len(tasks) != 9
            or [row[0] for row in tasks] != ["recurrent"] * 5 + ["inputs"] * 4
            or [row[1] for row in tasks] != [95, 96, 97, 98, 99, 94, 95, 96, 97]):
        raise ValueError("The bound five-recurrent/four-input scope is required")
    test_tree = ast.parse(test_data, filename=str(test_path))
    if sum(isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
           for node in ast.walk(test_tree)) != EXPECTED_TEST_COUNT:
        raise ValueError("The existing sixteen-test suite is required")
    request = deepcopy(read_json(args.resource_template))
    shape = cpu_shape(request)
    if shape.get("name") != "head" or shape["cpu"] != 4 or shape["memory_mb"] != 16384:
        raise ValueError("Use the approved head CPU4/16GiB/zero-GPU template")
    scripts = dict(
        ca3_merlin_remote=transport_data.decode("utf-8"),
        pack_results=args.pack_results.read_bytes().decode("utf-8"), run_intake=RUNTIME)
    scripts = {name + ".py": text for name, text in scripts.items()}
    for name, text in scripts.items():
        ast.parse(text, filename=name)
    ast.parse(TEST_RUNNER, filename="v1_intake_unittest_child.py")
    launcher_sources = {name: identity_bytes(text.encode("utf-8")) for name, text in scripts.items()}
    input_bindings = {label: dict(path=str(path.resolve()), **identity_bytes(path.read_bytes()))
                      for label, path in (("staging_command", args.staging_command),
                          ("input_files", args.input_files), ("upload_manifest", args.upload_manifest))}
    job = dict(staging_command=command, input_files=inputs, upload=upload, resource_shape=shape,
        python_packages=list(PYTHON_PACKAGES),
        scripts_directory=str(scripts_directory), libs_directory=str(libs_directory),
        expected_tasks=tasks, expected_tests=EXPECTED_TEST_COUNT, test_runner=TEST_RUNNER,
        test_sources={name: row for name, row in inputs.items() if Path(name).suffix == ".py"},
        launcher_sources=launcher_sources, input_bindings=input_bindings)
    payload = dict(job=job, files=scripts)
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode("utf-8"), 9)).decode("ascii")
    bootstrap = BOOTSTRAP.replace("PAYLOAD_LITERAL", repr(encoded))
    ast.parse(bootstrap, filename="v1_construction_intake_bootstrap.py")
    entrypoint = (ENTRYPOINT_PREFIX + "python3 - <<'MRLN_V1_INTAKE_BOOTSTRAP'\n" + bootstrap
                  + "MRLN_V1_INTAKE_BOOTSTRAP\n")
    subprocess.run(["bash", "-n"], input=entrypoint, text=True, check=True)
    template = request["job_config"]["job_template_config"]
    template["entrypoint_full_script"] = entrypoint
    template.setdefault("env_map", {}).update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    request["name"] = args.name
    request["description"] = (
        "mrln_ CPU-only intake of five completed recurrent constructions and four completed input constructions. "
        "Run the bound sixteen-test suite before intake. Persist isolated acceptances and provenance; "
        "zero source simulations, production episodes, canonical writes, or future reference preparations.")
    if "request_context" in request:
        request["request_context"].pop("precheck_sid", None)
        if not request["request_context"]:
            del request["request_context"]
    args.output.mkdir(parents=True, exist_ok=True)
    request_path = args.output / (args.name + ".request.json")
    entrypoint_path = args.output / (args.name + ".sh")
    publish(request_path, json.dumps(request, indent=2) + "\n")
    publish(entrypoint_path, entrypoint)
    summary = dict(status="PREPARED_NOT_SUBMITTED", request=str(request_path),
        entrypoint=str(entrypoint_path), resource_shape=shape, input_bindings=input_bindings,
        upload_sha256=upload["sha256"], upload_bytes=upload["bytes"], upload_chunks=len(upload["chunks"]),
        bound_input_files=len(inputs), launcher_sources=launcher_sources, python_packages=list(PYTHON_PACKAGES),
        runtime_tests=dict(status="NOT_RUN", expected_tests=EXPECTED_TEST_COUNT,
                           source_files=len(job["test_sources"]), scope="isolated_child_process"),
        expected_tasks=tasks, required_available_memory_gib=4,
        canonical_acceptances_written=0, source_simulations=0, production_episodes=0,
        future_reference_preparations=0, runtime_unverified=True, current_precheck_required=True,
        validation=dict(consumer_ast=True, test_ast=True, runtime_ast=True, test_runner_ast=True,
                        transport_ast=True, packer_ast=True, bootstrap_ast=True, shell_syntax=True))
    publish(args.output / (args.name + ".prepared.json"), json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("staging-command", "input-files", "upload-manifest", "resource-template", "pack-results", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--name", required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
