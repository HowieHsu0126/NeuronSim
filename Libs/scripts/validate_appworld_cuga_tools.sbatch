#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_AppWorldCugaTools
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-appworld-cuga-tools-%j.out
#SBATCH --error=/tmp/neuronsim-appworld-cuga-tools-%j.err
set -euo pipefail
repo=${1:?repository root required}
environment_receipt=${2:?completed CUGA environment.json required}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/opt/conda/bin/python - "$repo" "$environment_receipt" "$0" <<'PY'
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

repo, receipt_path, launcher = map(Path, sys.argv[1:])
receipt = json.loads(receipt_path.read_text())
if (receipt.get('status') != 'CUGA_PACKAGES_PREPARED_NATIVE_INTEGRATION_REQUIRED'
        or receipt.get('source_revision') != 'cbbdf40f8cb99a56a417b3957139a103e7096890'
        or receipt.get('hostname', '').split('.')[0] != socket.gethostname().split('.')[0]):
    raise ValueError('completed CUGA preparation on this node is required')
python = Path(receipt['python'])
base = Path('/data/neuronsim/neuronsim_appworld_benchmark')
filesystem = subprocess.check_output(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(base)], text=True).strip()
if filesystem not in {'ext4', 'xfs', 'btrfs', 'ext3', 'ext2'} or shutil.disk_usage(base).free < 512*1024**2:
    raise ValueError('node-local validation storage admission failed')
job = os.environ['SLURM_JOB_ID']
output = repo/'Output/neural_population_dynamics/agentic_extension/benchmark/software_validation'/f'run_{job}'
output.mkdir(exist_ok=False)
shutil.copy2(launcher, output/'executed_launcher.sbatch')
shutil.copy2(receipt_path, output/'cuga_environment.json')
names = ('__init__.py', 'resource_probe.py', 'appworld_benchmark_cuga_tools.py', 'test_appworld_benchmark_cuga_tools.py')
with tempfile.TemporaryDirectory(prefix=f'cuga_tools_tests_{job}_', dir=base) as temporary:
    root = Path(temporary)
    hashes = {}
    for name in names:
        content = (repo/'Libs/population_dynamics'/name).read_bytes()
        ast.parse(content, filename=name)
        for path in (root/'Libs/population_dynamics'/name, output/'source/Libs/population_dynamics'/name):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        hashes['Libs/population_dynamics/'+name] = hashlib.sha256(content).hexdigest()
    environment = dict(os.environ, PYTHONPATH=str(root/'Libs'), TMPDIR=str(root), XDG_CACHE_HOME=str(root/'cache'))
    resource = subprocess.run([str(python), '-m', 'population_dynamics.resource_probe', '--output', str(root/'resources.json'),
                               '--required-memory-gib', '4'], cwd=root, env=environment, capture_output=True, text=True)
    (root/'resources.log').write_text(resource.stdout+resource.stderr)
    binding_code = '''
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
receipt=json.loads(Path(sys.argv[1]).read_text())
actual={name:importlib.metadata.version(name) for name in receipt['packages']}
if actual != receipt['packages']:
    raise ValueError('CUGA prepared packages changed')
source=Path(receipt['source_checkout'])
modules={
 'cuga.backend.activity_tracker.tracker':'src/cuga/backend/activity_tracker/tracker.py',
 'cuga.backend.cuga_graph.nodes.cuga_lite.executors.common.call_api_helper':'src/cuga/backend/cuga_graph/nodes/cuga_lite/executors/common/call_api_helper.py',
 'cuga.backend.tools_env.registry.utils.api_utils':'src/cuga/backend/tools_env/registry/utils/api_utils.py',
 'cuga.backend.tools_env.registry.utils.types':'src/cuga/backend/tools_env/registry/utils/types.py',
}
for module,relative in modules.items():
    installed=Path(inspect.getsourcefile(importlib.import_module(module)))
    if installed.read_bytes() != (source/relative).read_bytes():
        raise ValueError('installed CUGA source differs from prepared checkout: '+module)
print(json.dumps({'packages':actual,'source_modules_compared':list(modules)}))
'''
    binding = None
    result = None
    command = [str(python), '-m', 'unittest', '-v', 'population_dynamics.test_appworld_benchmark_cuga_tools']
    if resource.returncode == 0:
        binding = subprocess.run([str(python), '-c', binding_code, str(receipt_path)], cwd=root,
                                 env=environment, capture_output=True, text=True)
        (root/'source_binding.log').write_text(binding.stdout+binding.stderr)
        if binding.returncode == 0:
            with (root/'tests.log').open('w') as log:
                result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
    passed = result is not None and result.returncode == 0
    for name in ('resources.json', 'resources.log', 'source_binding.log', 'tests.log'):
        if (root/name).is_file():
            shutil.copy2(root/name, output/name)
    report = {
        'status': 'PASS' if passed else 'FAILED', 'job_id': job, 'hostname': socket.gethostname(),
        'recorded_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': hashes,
        'resource_exit_code': resource.returncode, 'source_binding_exit_code': None if binding is None else binding.returncode,
        'test_exit_code': None if result is None else result.returncode, 'command': command,
        'evidence_scope': 'installed_cuga_external_tool_interfaces_with_synthetic_executor',
        'native_appworld_tasks_executed': 0, 'model_generations': 0, 'full_graph_qualified': False,
    }
    (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    sys.exit(0 if passed else 1)
PY
