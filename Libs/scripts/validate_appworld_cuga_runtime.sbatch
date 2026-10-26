#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_AppWorldCugaGraph
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-appworld-cuga-graph-%j.out
#SBATCH --error=/tmp/neuronsim-appworld-cuga-graph-%j.err
set -euo pipefail
repo=${1:?repository root required}
environment_receipt=${2:?completed CUGA environment.json required}
model_name=${3:?configured model name required}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/opt/conda/bin/python - "$repo" "$environment_receipt" "$model_name" "$0" <<'PY'
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import traceback

repo, receipt_path = map(Path, sys.argv[1:3])
model_name, launcher = sys.argv[3], Path(sys.argv[4])
receipt = json.loads(receipt_path.read_text())
hostname = socket.gethostname().split('.')[0]
if (hostname not in {'cpu03', 'cpu04'} or
        receipt.get('status') != 'CUGA_PACKAGES_PREPARED_NATIVE_INTEGRATION_REQUIRED' or
        receipt.get('source_revision') != 'cbbdf40f8cb99a56a417b3957139a103e7096890' or
        receipt.get('hostname', '').split('.')[0] != hostname):
    raise ValueError('pinned CUGA preparation on this allowed compute node is required')
base = Path('/data/neuronsim/neuronsim_appworld_benchmark')
filesystem = subprocess.check_output(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(base)], text=True).strip()
if filesystem not in {'ext4', 'xfs', 'btrfs', 'ext3', 'ext2'} or shutil.disk_usage(base).free < 512*1024**2:
    raise ValueError('node-local diagnostic storage admission failed')
job = os.environ['SLURM_JOB_ID']
output = repo/'Output/neural_population_dynamics/agentic_extension/benchmark/software_validation'/f'run_{job}'
output.mkdir(exist_ok=False)
shutil.copy2(launcher, output/'executed_launcher.sbatch')
shutil.copy2(receipt_path, output/'cuga_environment.json')
modules = [
    '__init__.py', 'provenance.py', 'resource_probe.py', 'appworld_reference.py',
    'appworld_collaboration.py', 'appworld_action_decoder.py', 'appworld_local_actor.py',
    'appworld_benchmark_protocol.py', 'appworld_benchmark_execution.py', 'appworld_benchmark_runner.py',
    'appworld_benchmark_transport.py', 'appworld_benchmark_development.py',
    'appworld_benchmark_collaboration.py', 'appworld_benchmark_native_tools.py',
    'appworld_benchmark_cuga_channel.py', 'appworld_benchmark_cuga_process.py',
    'appworld_benchmark_cuga_runtime.py', 'appworld_benchmark_cuga_tools.py',
    'appworld_benchmark_cuga_runtime_check.py',
]
report = {
    'status': 'FAILED', 'job_id': job, 'hostname': hostname,
    'observed_utc': datetime.now(timezone.utc).isoformat(),
    'suite': 'installed_cuga_graph_and_callback',
    'source_sha256': {}, 'seed_results': [],
    'evidence_scope': 'real_installed_cuga_graph_construction_and_client_callback_with_synthetic_http_response',
    'native_tasks_executed': 0, 'model_generations': 0, 'graph_loop_executed': False,
}
with tempfile.TemporaryDirectory(prefix=f'cuga_graph_check_{job}_', dir=base) as temporary:
    root = Path(temporary)
    try:
        for name in modules:
            relative = 'Libs/population_dynamics/'+name
            content = (repo/relative).read_bytes()
            ast.parse(content, filename=relative)
            for destination in (root/relative, output/'source'/relative):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            report['source_sha256'][relative] = hashlib.sha256(content).hexdigest()
        environment = dict(os.environ, PYTHONPATH=str(root/'Libs'), TMPDIR=str(root),
                           XDG_CACHE_HOME=str(root/'cache'))
        resource = subprocess.run(
            [sys.executable, '-m', 'population_dynamics.resource_probe',
             '--output', str(root/'resources.json'), '--required-memory-gib', '4'],
            cwd=root, env=environment, capture_output=True, text=True)
        (output/'resources.log').write_text(resource.stdout+resource.stderr)
        if (root/'resources.json').exists():
            shutil.copy2(root/'resources.json', output/'resources.json')
        report['resource_exit_code'] = resource.returncode
        if resource.returncode:
            raise RuntimeError('node-local resource admission failed')
        probe = subprocess.run(
            [receipt['python'], '-c',
             'import sys,json,importlib.util; print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,cuga_origin=importlib.util.find_spec("cuga").origin)))'],
            cwd=root, env=environment, capture_output=True, text=True)
        (output/'python_identity.log').write_text(probe.stdout+probe.stderr)
        if probe.returncode:
            raise RuntimeError('prepared interpreter identity probe failed')
        identity = json.loads(probe.stdout)
        report['python_identity'] = identity
        sys.path.insert(0, str(root/'Libs'))
        from population_dynamics.appworld_benchmark_cuga_process import (
            _bubblewrap_command, _python_runtime, _worker_environment,
        )
        executable, prefix, base_prefix = _python_runtime(receipt['python'], identity['base_prefix'])
        runtime = dict(executable=str(executable), prefix=str(prefix), base_prefix=str(base_prefix))
        blocked = False
        for seed in (41, 42, 43):
            if blocked:
                report['seed_results'].append(dict(seed=seed, status='NOT_RUN',
                                                  reason='preceding_structural_check_failed'))
                continue
            owner = root/f'seed_{seed}'
            sandbox = owner/'sandbox'
            command = _bubblewrap_command(
                executable=executable, prefix=prefix, base_prefix=base_prefix,
                application_lib=root/'Libs', sandbox=sandbox, owner_root=owner,
                protected_roots=[repo, base/'source_42b5bcf3_643755',
                                 Path('/data/neuronsim/neuronsim_models')],
                worker_arguments=(
                    '-m', 'population_dynamics.appworld_benchmark_cuga_runtime_check',
                    '--seed', str(seed), '--model-name', model_name,
                    '--work-root', '/work/check', '--worker-config', '/work/worker_config.json'),
            )
            sandbox.mkdir(parents=True)
            for directory in ('logs', 'dbs', 'cache/fastembed', 'cache/huggingface', 'tmp'):
                (sandbox/directory).mkdir(parents=True, exist_ok=True)
            (sandbox/'worker_config.json').write_text(json.dumps({
                'seed': seed, 'model_name': model_name,
                'expected_packages': receipt['packages'], 'python_runtime': runtime,
            }, indent=2)+'\n')
            worker_environment = _worker_environment(
                prefix=prefix, base_prefix=base_prefix, base_url='http://127.0.0.1:1/v1')
            seed_output = output/f'seed_{seed}'
            seed_output.mkdir()
            shutil.copy2(sandbox/'worker_config.json', seed_output/'worker_config.json')
            with (seed_output/'stdout.log').open('w') as stdout, (seed_output/'stderr.log').open('w') as stderr:
                result = subprocess.run(command, env=worker_environment, stdout=stdout, stderr=stderr)
            check_path = sandbox/'check/cuga_runtime_check.json'
            check = json.loads(check_path.read_text()) if check_path.is_file() else None
            if check is not None:
                shutil.copy2(check_path, seed_output/check_path.name)
            passed = result.returncode == 0 and check is not None and check.get('status') == 'PASS'
            report['seed_results'].append({
                'seed': seed, 'status': 'PASS' if passed else 'FAILED',
                'exit_code': result.returncode, 'command': list(command),
                'stage': None if check is None else check.get('stage'),
                'exception': None if check is None else check.get('exception'),
            })
            blocked = not passed
        if all(row['status'] == 'PASS' for row in report['seed_results']):
            report['status'] = 'PASS'
    except BaseException as error:
        report['exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
sys.exit(0 if report['status'] == 'PASS' else 1)
PY
