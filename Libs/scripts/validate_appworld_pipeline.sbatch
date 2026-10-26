#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_AppWorldPipelineTests
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-appworld-pipeline-tests-%j.out
#SBATCH --error=/tmp/neuronsim-appworld-pipeline-tests-%j.err
set -euo pipefail
repo=${1:?repository root required}
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/opt/conda/bin/python - "$repo" "$0" "${2:-pipeline}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

repo, launcher = map(Path, sys.argv[1:3])
suite = sys.argv[3]
if suite not in {'pipeline', 'native_tools', 'cuga_runtime', 'cuga_channel', 'cuga_process', 'cuga_development'}:
    raise ValueError('validation suite must be pipeline, native_tools, cuga_runtime, cuga_channel, cuga_process or cuga_development')
job = os.environ['SLURM_JOB_ID']
if socket.gethostname().split('.')[0] not in {'cpu03', 'cpu04'}:
    raise RuntimeError('CPU-only validation requires cpu03 or cpu04')
base = Path('/data/neuronsim/neuronsim_appworld_benchmark')
base.mkdir(parents=True, exist_ok=True)
filesystem = subprocess.check_output(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(base)], text=True).strip()
if filesystem not in {'ext4', 'xfs', 'btrfs', 'ext3', 'ext2'} or shutil.disk_usage(base).free < 512 * 1024**2:
    raise RuntimeError('node-local validation storage admission failed')
output = repo / 'Output/neural_population_dynamics/agentic_extension/benchmark/software_validation' / f'run_{job}'
output.mkdir(exist_ok=False)
shutil.copy2(launcher, output / 'executed_launcher.sbatch')
modules = ['__init__.py', 'provenance.py', 'resource_probe.py', 'appworld_reference.py', 'appworld_collaboration.py',
           'appworld_action_decoder.py', 'appworld_local_actor.py', 'appworld_benchmark_protocol.py',
           'appworld_benchmark_execution.py', 'appworld_benchmark_runner.py', 'appworld_benchmark_transport.py', 'appworld_benchmark_development.py',
           'appworld_benchmark_scoring.py', 'appworld_benchmark_model_probe.py', 'appworld_benchmark_model_runtime.py',
           'appworld_benchmark_collaboration.py', 'appworld_benchmark_host.py', 'appworld_benchmark_collaboration_development.py',
           'appworld_benchmark_pipeline.py', 'appworld_benchmark_inference.py']
# Run the pipeline and its previously unexecuted prerequisite suites in one
# allocation. Accepted runner, collaboration and host suites are reused.
tests = ['test_appworld_benchmark_execution.py', 'test_appworld_benchmark_scoring.py', 'test_appworld_benchmark_development.py',
         'test_appworld_benchmark_model_probe.py', 'test_appworld_benchmark_model_runtime.py',
         'test_appworld_benchmark_inference.py', 'test_appworld_benchmark_pipeline.py']
if suite == 'native_tools':
    modules += ['appworld_benchmark_native_tools.py']
    tests = ['test_appworld_benchmark_native_tools.py']
elif suite == 'cuga_runtime':
    modules += ['appworld_benchmark_cuga_runtime.py', 'appworld_benchmark_cuga_tools.py']
    tests = ['test_appworld_benchmark_cuga_runtime.py']
elif suite == 'cuga_channel':
    modules += ['appworld_benchmark_native_tools.py', 'appworld_benchmark_cuga_channel.py']
    tests = ['test_appworld_benchmark_cuga_channel.py']
elif suite == 'cuga_process':
    modules += ['appworld_benchmark_native_tools.py', 'appworld_benchmark_cuga_channel.py',
                'appworld_benchmark_cuga_process.py', 'appworld_benchmark_cuga_runtime.py',
                'appworld_benchmark_cuga_tools.py', 'appworld_benchmark_cuga_receipt.py',
                'test_appworld_benchmark_scoring.py']
    tests = ['test_appworld_benchmark_cuga_process.py', 'test_appworld_benchmark_cuga_receipt.py']
elif suite == 'cuga_development':
    modules += ['appworld_benchmark_native_tools.py', 'appworld_benchmark_cuga_channel.py',
                'appworld_benchmark_cuga_process.py', 'appworld_benchmark_cuga_runtime.py',
                'appworld_benchmark_cuga_tools.py', 'appworld_benchmark_cuga_receipt.py',
                'appworld_benchmark_cuga_development.py', 'test_appworld_benchmark_pipeline.py']
    tests = ['test_appworld_benchmark_cuga_development.py']
modules += tests
inputs = ['Libs/population_dynamics/' + name for name in modules]
inputs += ['Libs/configs/neural_population_appworld_benchmark.yaml', 'Libs/configs/neural_population_appworld_local_actor.yaml',
           'Libs/scripts/run_appworld_development.sbatch', 'Libs/scripts/qualify_appworld_model.sbatch']
shell_scripts = ['run_appworld_development.sbatch', 'qualify_appworld_model.sbatch']
if suite == 'cuga_development':
    inputs += ['Libs/scripts/run_appworld_cuga_development.sbatch']
    shell_scripts.append('run_appworld_cuga_development.sbatch')
with tempfile.TemporaryDirectory(prefix=f'pipeline_tests_{job}_', dir=base) as temporary:
    root = Path(temporary)
    hashes = {}
    for relative in inputs:
        content = (repo / relative).read_bytes()
        for destination in (root / relative, output / 'source' / relative):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        hashes[relative] = hashlib.sha256(content).hexdigest()
    environment = dict(os.environ, PYTHONPATH=str(root / 'Libs'), TMPDIR=str(root), CUDA_VISIBLE_DEVICES='')
    resource_command = [sys.executable, '-m', 'population_dynamics.resource_probe', '--output', str(root/'resources.json'),
                        '--required-memory-gib', '4']
    resource = subprocess.run(resource_command, cwd=root, env=environment, capture_output=True, text=True)
    (root/'resource.log').write_text(resource.stdout + resource.stderr)
    syntax_checks = [subprocess.run(['bash', '-n', str(root/'Libs/scripts'/name)],
                                   cwd=root, env=environment, capture_output=True, text=True)
                     for name in shell_scripts]
    syntax_exit_code = next((check.returncode for check in syntax_checks if check.returncode), 0)
    (root/'shell_syntax.log').write_text(''.join(check.stdout + check.stderr for check in syntax_checks))
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '--junitxml', str(root/'tests.xml'),
               *('Libs/population_dynamics/' + name for name in tests)]
    result = None
    if resource.returncode == 0 and syntax_exit_code == 0:
        with (root/'tests.log').open('w') as log:
            result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
    counts = None
    if (root/'tests.xml').is_file():
        suites = list(ET.parse(root/'tests.xml').getroot().iter('testsuite'))
        counts = {name: sum(int(suite.get(name, '0')) for suite in suites) for name in ('tests', 'errors', 'failures', 'skipped')}
    for name in ('resources.json', 'resource.log', 'shell_syntax.log', 'tests.xml', 'tests.log'):
        if (root/name).is_file():
            shutil.copy2(root/name, output/name)
    passed = result is not None and result.returncode == 0
    receipt = {'status': 'PASS' if passed else 'FAILED', 'job_id': job, 'hostname': socket.gethostname(),
               'observed_utc': datetime.now(timezone.utc).isoformat(), 'resource_exit_code': resource.returncode,
               'shell_syntax_exit_code': syntax_exit_code, 'pytest_exit_code': None if result is None else result.returncode,
               'command': command, 'source_sha256': hashes, 'test_counts': counts,
               'native_tasks_executed': 0, 'model_generations': 0,
               'suite': suite,
               'evidence_scope': ('native_tool_dispatch_and_async_process_ownership_with_synthetic_worlds'
                                  if suite == 'native_tools' else
                                  'cuga_callback_usage_and_graph_observation_with_synthetic_inputs'
                                  if suite == 'cuga_runtime' else
                                  'dedicated_unix_fd_proxy_and_native_owner_cleanup_with_synthetic_clients'
                                  if suite == 'cuga_channel' else
                                  'bubblewrap_process_ownership_and_native_scoring_receipt_with_synthetic_workers'
                                  if suite == 'cuga_process' else
                                  'cuga_native_owner_and_independent_scoring_pipeline_with_synthetic_inputs'
                                  if suite == 'cuga_development' else
                                  'pipeline_prerequisites_and_scenario_inference_with_synthetic_inputs')}
    (output/'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    sys.exit(0 if passed else 1)
PY
