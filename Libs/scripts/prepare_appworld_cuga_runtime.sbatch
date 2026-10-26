#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_AppWorldCugaSetup
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-appworld-cuga-setup-%j.out
#SBATCH --error=/tmp/neuronsim-appworld-cuga-setup-%j.err
set -euo pipefail
repo=${1:?repository root required}
scratch_root=${2:?node-local scratch root required}
base_python=${3:?existing Python interpreter required}
uv_binary=${4:?uv 0.12.6 executable required}
resume_runtime=${5:-}
revision=cbbdf40f8cb99a56a417b3957139a103e7096890
work="$scratch_root/neuronsim_appworld_benchmark/cuga_cbbdf40f8_${SLURM_JOB_ID}"
archive="$repo/Output/neural_population_dynamics/agentic_extension/benchmark/cuga_runtime/run_${SLURM_JOB_ID}"
case "$(hostname -s)" in
    cpu03|cpu04) ;;
    *) printf "%s\n" CUGA_PREPARATION_REQUIRES_CPU03_OR_CPU04 >&2; exit 1;;
esac
fstype=$(findmnt -T "$scratch_root" -n -o FSTYPE)
case "$fstype" in nfs|nfs4|cifs) printf '%s\n' NODE_LOCAL_SCRATCH_REQUIRED >&2; exit 1;; esac
mkdir -p "$(dirname "$work")" "$(dirname "$archive")"
mkdir "$work" "$archive"
mkdir -p "$work/tmp" "$work/cache" "$work/probe/population_dynamics"
cp "$0" "$archive/executed_launcher.sbatch"
cp "$repo/Libs/population_dynamics/__init__.py" "$repo/Libs/population_dynamics/resource_probe.py" "$work/probe/population_dynamics/"
export PYTHONPATH="$work/probe" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$work/tmp" XDG_CACHE_HOME="$work/cache" UV_CACHE_DIR="$work/cache/uv"
export UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_NO_PROGRESS=1
export GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1
finish() {
    result=$?
    trap - EXIT
    if [ -n "${proxy_forward_pid:-}" ]; then
        kill "$proxy_forward_pid" 2>/dev/null || true
        wait "$proxy_forward_pid" || true
    fi
    for name in resource_admission.json setup.log source_acquisition.json wheel_route.json dependency_check.log package_freeze.txt environment.json; do
        if [ -f "$work/$name" ]; then cp "$work/$name" "$archive/"; fi
    done
    printf '%s\n' "$work" > "$archive/runtime_path.txt"
    printf '%s\n' "$result" > "$archive/exit_code.txt"
    exit "$result"
}
trap finish EXIT
exec > "$work/setup.log" 2>&1
"$base_python" -m population_dynamics.resource_probe --output "$work/resource_admission.json" --required-memory-gib 8
"$base_python" - "$scratch_root" "$uv_binary" <<'PYADMIT'
from pathlib import Path
import shutil
import subprocess
import sys
if shutil.disk_usage(Path(sys.argv[1])).free < 32 * 1024**3:
    raise ValueError('CUGA package preparation requires 32 GiB free node-local storage')
version = subprocess.check_output([sys.argv[2], '--version'], text=True).strip()
if not version.startswith('uv 0.12.6 '):
    raise ValueError('prepared recipe requires the inspected uv 0.12.6 executable')
print(version)
PYADMIT
# Reuse the task-owned submit-host forwarding route from native preparation.
# An inherited loopback proxy requires forwarding from the allocated compute node.
proxy_forward_pid=""
proxy_spec=$("$base_python" - <<'PYPROXY'
import socket
import urllib.request
from urllib.parse import urlsplit
parsed = urlsplit(urllib.request.getproxies().get('https', ''))
if parsed.hostname in {'127.0.0.1', 'localhost', '::1'}:
    if parsed.username is not None or parsed.port is None:
        raise ValueError('UNSUPPORTED_INHERITED_PROXY')
    with socket.socket() as local:
        local.bind(('127.0.0.1', 0))
        print(parsed.port, local.getsockname()[1])
PYPROXY
)
if [ -n "$proxy_spec" ]; then
    read -r remote_proxy_port local_proxy_port <<< "$proxy_spec"
    ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=10 -N \
        -L "127.0.0.1:$local_proxy_port:127.0.0.1:$remote_proxy_port" \
        "${SLURM_SUBMIT_HOST:?submit host required for inherited proxy}" &
    proxy_forward_pid=$!
    "$base_python" - "$local_proxy_port" <<'PYREADY'
import socket
import sys
import time
for _ in range(100):
    try:
        with socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=.1):
            break
    except OSError:
        time.sleep(.1)
else:
    raise RuntimeError('TASK_PROXY_FORWARD_NOT_READY')
PYREADY
    export http_proxy="http://127.0.0.1:$local_proxy_port"
    export https_proxy="$http_proxy" all_proxy="$http_proxy"
    export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy" ALL_PROXY="$http_proxy"
fi
# Direct official PyPI access was verified by source-route check643772.
export no_proxy="${no_proxy:-},localhost,127.0.0.1,::1,pypi.org,.pypi.org,files.pythonhosted.org,.files.pythonhosted.org,download.pytorch.org,.pytorch.org"
export NO_PROXY="$no_proxy"
source_root="$work/cuga-agent"
if [ -n "$resume_runtime" ]; then
    "$base_python" - "$resume_runtime" "$work" "$revision" <<'PYRESUME'
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
previous, work = map(Path, sys.argv[1:3])
filesystem = subprocess.check_output(['findmnt', '-T', str(previous), '-n', '-o', 'FSTYPE'], text=True).strip()
if filesystem not in {'ext4', 'xfs', 'btrfs', 'ext3', 'ext2'}:
    raise ValueError('CUGA resume requires node-local source and cache')
acquisition = json.loads((previous/'source_acquisition.json').read_text())
resource = json.loads((previous/'resource_admission.json').read_text())
if acquisition['source_revision'] != sys.argv[3] or resource['hostname'].split('.')[0] != socket.gethostname().split('.')[0]:
    raise ValueError('CUGA resume source revision or node differs')
if not (previous/'cuga-agent/uv.lock').is_file():
    raise ValueError('CUGA resume has no completed source extraction')
acquisition['resumed_from_runtime'] = str(previous)
(work/'source_acquisition.json').write_text(json.dumps(acquisition, indent=2)+'\n')
print(json.dumps({'reuse_source_and_cache': str(previous)}))
PYRESUME
    source_root="$resume_runtime/cuga-agent"
    export UV_CACHE_DIR="$resume_runtime/cache/uv"
else
# Fetch the fixed commit archive after two Git pack transfers ended early.
# Retain the source identity and the existing inspected-file byte checks.
"$base_python" - "$work" "$revision" <<'PYARCHIVE'
import json
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request
work, revision = Path(sys.argv[1]), sys.argv[2]
url = 'https://codeload.github.com/cuga-project/cuga-agent/tar.gz/' + revision
archive = work / 'source.tar.gz'
# CPU03 direct HTTPS passed source-route probe652711; the forwarded archive
# transfer failed in652703. Keep one transfer and preserve failures.
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(url, timeout=60) as response, archive.open('xb') as output:
    resolved_url = response.url
    shutil.copyfileobj(response, output, length=1024*1024)
root_name = 'cuga-agent-' + revision
package_files = {'pyproject.toml', 'uv.lock', 'README.md', 'LICENSE'}
with tarfile.open(archive, 'r:gz') as source:
    selected = []
    for member in source.getmembers():
        parts = Path(member.name).parts
        if not parts or parts[0] != root_name:
            raise ValueError('CUGA archive root differs from pinned commit')
        relative = '/'.join(parts[1:])
        if relative in package_files or relative.startswith(('src/cuga/', 'src/scripts/')):
            selected.append(member)
    if not selected:
        raise ValueError('CUGA archive contains no selected package source')
    source.extractall(work, members=selected, filter='data')
(work / root_name).rename(work / 'cuga-agent')
report = {'source_revision': revision, 'source_url': url, 'resolved_source_url': resolved_url,
          'archive_bytes': archive.stat().st_size, 'selected_members': len(selected),
          'method': 'official_github_commit_archive', 'transport': 'direct_https'}
(work / 'source_acquisition.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
PYARCHIVE
fi
"$base_python" - "$repo" "$source_root" "$archive" <<'PYSOURCE'
import json
from pathlib import Path
import shutil
import sys
import tomllib
repo, source, archive = map(Path, sys.argv[1:])
project = tomllib.loads((source/'pyproject.toml').read_text())['project']
if project['name'] != 'cuga' or project['version'] != '0.3.2' or project['license'] != 'Apache-2.0':
    raise ValueError('CUGA candidate package or declared license differs from inspected source')
if not (source/'uv.lock').is_file():
    raise ValueError('pinned CUGA source has no dependency lockfile')
inspected = repo/'Output/neural_population_dynamics/agentic_extension/benchmark/baseline_admission/cuga_cbbdf40f8/source'
paths = [
    'pyproject.toml',
    'src/cuga/backend/cuga_graph/graph.py',
    'src/cuga/backend/cuga_graph/nodes/answer/final_answer.py',
    'src/cuga/backend/cuga_graph/nodes/chat/chat.py',
    'src/cuga/backend/cuga_graph/nodes/human_in_the_loop/wait_for_response.py',
    'src/cuga/backend/activity_tracker/tracker.py',
    'src/cuga/configurations/modes/balanced.toml',
    'src/cuga/backend/cuga_graph/nodes/cuga_lite/executors/code_executor.py',
    'src/cuga/backend/cuga_graph/nodes/cuga_lite/executors/common/call_api_helper.py',
    'src/cuga/backend/tools_env/registry/utils/api_utils.py',
    'src/cuga/backend/tools_env/registry/utils/types.py',
]
for relative in paths:
    if (source/relative).read_bytes() != (inspected/relative).read_bytes():
        raise ValueError('CUGA checkout differs from inspected pinned source: '+relative)
for relative in [*paths, 'uv.lock', 'README.md', 'LICENSE']:
    target = archive/'source'/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source/relative, target)
print(json.dumps({'source_files_compared': len(paths), 'package': project['name'], 'version': project['version'], 'license': project['license']}))
PYSOURCE
cd "$source_root"
# Verify the direct route for the actual locked CPU wheel before uv resumes.
# The forwarded TLS stream failed in652714; uv keeps its own wheel hash checks.
"$base_python" - "$source_root/uv.lock" "$work" <<'PYWHEEL'
import json
from pathlib import Path
import sys
import tomllib
import urllib.request
lock = tomllib.loads(Path(sys.argv[1]).read_text())
abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
candidates = [wheel['url'] for package in lock['package'] if package['name'] == 'torch'
              for wheel in package.get('wheels', [])
              if f'-{abi}-{abi}-' in wheel['url'] and 'linux' in wheel['url'] and 'x86_64' in wheel['url']]
if len(candidates) != 1:
    raise ValueError('locked CPU torch wheel selection is ambiguous')
url = candidates[0]
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
request = urllib.request.Request(url, headers={'Range': 'bytes=0-65535'})
with opener.open(request, timeout=30) as response:
    sample = response.read(65537)
    report = {'url': url, 'resolved_url': response.url, 'status': response.status,
              'sample_bytes': len(sample), 'transport': 'direct_https', 'complete_wheel_validated': False}
if not sample:
    raise ValueError('locked CPU wheel route returned no bytes')
(Path(sys.argv[2])/'wheel_route.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
PYWHEEL
# Install from the upstream frozen lock. No agent import, model request,
# AppWorld task, registry startup or benchmark data download occurs here.
"$uv_binary" sync --frozen --no-dev --extra evaluation --no-editable --python "$base_python" --default-index https://pypi.org/simple
"$uv_binary" pip check --python "$source_root/.venv/bin/python" > "$work/dependency_check.log" 2>&1
"$uv_binary" pip freeze --python "$source_root/.venv/bin/python" > "$work/package_freeze.txt"
"$source_root/.venv/bin/python" - "$work" "$revision" "$uv_binary" "$source_root" <<'PYRECEIPT'
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
work = Path(sys.argv[1])
packages = {name: importlib.metadata.version(name) for name in
            ('cuga', 'langchain', 'langchain-core', 'langchain-openai', 'langgraph', 'fastmcp', 'torch', 'browsergym-core')}
if packages['cuga'] != '0.3.2':
    raise ValueError('installed CUGA package version differs from pinned source')
report = {
    'status': 'CUGA_PACKAGES_PREPARED_NATIVE_INTEGRATION_REQUIRED',
    'scope': 'pinned_full_graph_candidate_packages_only', 'formal_eligible': False,
    'source_repository': 'https://github.com/cuga-project/cuga-agent.git', 'source_revision': sys.argv[2],
    'source_checkout': str(Path(sys.argv[4])), 'python': str(Path(sys.argv[4])/'.venv/bin/python'),
    'source_acquisition': json.loads((work/'source_acquisition.json').read_text()),
    'python_version': platform.python_version(), 'packages': packages,
    'uv_version': subprocess.check_output([sys.argv[3], '--version'], text=True).strip(),
    'dependency_resolution': 'upstream_uv_lock_frozen_evaluation_extra_no_dev',
    'job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
    'recorded_at': datetime.now(timezone.utc).isoformat(),
    'agent_modules_imported': False, 'full_graph_constructed': False,
    'model_generations': 0, 'appworld_tasks_executed': 0,
    'registry_started': False, 'native_integration_qualified': False,
}
(work/'environment.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
PYRECEIPT
