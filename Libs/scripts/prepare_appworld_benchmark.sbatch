#!/usr/bin/env bash
#SBATCH --job-name=NrnSim_AppWorldPrepare
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=none
#SBATCH --no-requeue
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/neuronsim-appworld-prepare-%j.out
#SBATCH --error=/tmp/neuronsim-appworld-prepare-%j.err
set -euo pipefail

# Install the admitted candidate into an independent node-local environment.
# This preparation downloads official code/data but executes no benchmark task.
repo=${1:?repository root required}
scratch_root=${2:?node-local scratch root required}
base_python=${3:?base Python required}
revision=42b5bcf3cd334fee33f0c37c02070a9f5807add5
runtime=${4:-"$scratch_root/neuronsim_appworld_benchmark/source_${revision:0:8}_${SLURM_JOB_ID}"}
archive="$repo/Output/neural_population_dynamics/agentic_extension/benchmark/source_admission/run_${SLURM_JOB_ID}"
test_source="$runtime/project"
test -n "${SLURM_JOB_ID:-}"
test "$(hostname -s)" = cpu04
fstype=$(findmnt -T "$scratch_root" -n -o FSTYPE)
case "$fstype" in nfs|nfs4|cifs) printf '%s\n' 'NODE_LOCAL_SCRATCH_REQUIRED' >&2; exit 1;; esac
mkdir -p "$scratch_root/neuronsim_appworld_benchmark" "$(dirname "$archive")"
if [ -n "${4:-}" ]; then
    test -d "$runtime/appworld/.git"
else
    mkdir "$runtime"
fi
mkdir "$archive"
mkdir -p "$test_source/Libs/population_dynamics" "$test_source/Libs/scripts" "$archive/source"
cp "$repo/Libs/__init__.py" "$test_source/Libs/"
cp "$repo/Libs/population_dynamics/__init__.py" "$repo/Libs/population_dynamics/resource_probe.py" "$repo/Libs/population_dynamics/appworld_benchmark_protocol.py" "$repo/Libs/population_dynamics/test_appworld_benchmark_protocol.py" "$test_source/Libs/population_dynamics/"
cp "$repo/Libs/scripts/inspect_appworld_benchmark.py" "$test_source/Libs/scripts/"
cp "$repo/Libs/scripts/prepare_appworld_benchmark.sbatch" "$repo/Libs/scripts/inspect_appworld_benchmark.py" "$repo/Libs/population_dynamics/appworld_benchmark_protocol.py" "$repo/Libs/population_dynamics/test_appworld_benchmark_protocol.py" "$archive/source/"
cp "${BASH_SOURCE[0]}" "$archive/source/executed_launcher.sbatch"
export PYTHONPATH="$test_source:$test_source/Libs"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export PATH="$runtime/bin:$PATH"
export TMPDIR="$runtime/tmp"
export XDG_CACHE_HOME="$runtime/cache"
export PIP_CACHE_DIR="$runtime/cache/pip"
export APPWORLD_CACHE="$runtime/cache/appworld"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$runtime/benchmark"
finish() {
    result=$?
    trap - EXIT
    if [ -n "${proxy_forward_pid:-}" ]; then
        kill "$proxy_forward_pid" 2>/dev/null || true
        wait "$proxy_forward_pid" || true
    fi
    for name in resource_admission.json setup.log package_freeze.txt tests.log environment.json; do
        if [ -f "$runtime/$name" ]; then cp "$runtime/$name" "$archive/"; fi
    done
    if [ -d "$runtime/intake" ]; then cp -a "$runtime/intake/." "$archive/"; fi
    printf '%s\n' "$runtime" > "$archive/runtime_path.txt"
    printf '%s\n' "$result" > "$archive/exit_code.txt"
    exit "$result"
}
trap finish EXIT
exec > "$runtime/setup.log" 2>&1
"$base_python" -m Libs.population_dynamics.resource_probe --output "$runtime/resource_admission.json" --required-memory-gib 8
# The inherited proxy may listen only on the submit host. Keep the forward
# inside this Slurm job and reap it before completion.
proxy_forward_pid=""
proxy_spec=$("$base_python" - <<'PY_PROXY'
import socket
import urllib.request
from urllib.parse import urlsplit

value = urllib.request.getproxies().get("https", "")
parsed = urlsplit(value)
if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
    if parsed.username is not None or parsed.port is None:
        raise ValueError("UNSUPPORTED_INHERITED_PROXY")
    with socket.socket() as local:
        local.bind(("127.0.0.1", 0))
        print(parsed.port, local.getsockname()[1])
PY_PROXY
)
if [ -n "$proxy_spec" ]; then
    read -r remote_proxy_port local_proxy_port <<< "$proxy_spec"
    ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ConnectTimeout=10 -N \
        -L "127.0.0.1:$local_proxy_port:127.0.0.1:$remote_proxy_port" \
        "${SLURM_SUBMIT_HOST:?submit host required for inherited proxy}" &
    proxy_forward_pid=$!
    "$base_python" - "$local_proxy_port" <<'PY_READY'
import socket
import sys
import time
for _ in range(100):
    try:
        with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=.1):
            break
    except OSError:
        time.sleep(.1)
else:
    raise RuntimeError("TASK_PROXY_FORWARD_NOT_READY")
PY_READY
    export http_proxy="http://127.0.0.1:$local_proxy_port"
    export https_proxy="$http_proxy" all_proxy="$http_proxy"
    export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy" ALL_PROXY="$http_proxy"
fi

if ! command -v git-lfs >/dev/null 2>&1; then
    "$base_python" - "$runtime" <<'PY_LFS'
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import urllib.request

runtime = Path(sys.argv[1])
url = "https://github.com/git-lfs/git-lfs/releases/download/v3.8.0/git-lfs-linux-amd64-v3.8.0.tar.gz"
expected = "e455e00f15d9b95661b8d53498ffb0c3367962cf1ec73c31ab7369516cd6ab8d"
# The route closes long full-body transfers. The official asset supports
# byte ranges; bounded segments retain one final published digest check.
expected_size = 5909255
segments = []
for start in range(0, expected_size, 1024 * 1024):
    end = min(start + 1024 * 1024, expected_size) - 1
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        expected_range = f"bytes {start}-{end}/{expected_size}"
        if response.status != 206 or response.headers.get("Content-Range") != expected_range:
            raise ValueError("OFFICIAL_GIT_LFS_ASSET_RANGE_MISMATCH")
        segment = response.read()
    if len(segment) != end - start + 1:
        raise ValueError("OFFICIAL_GIT_LFS_ASSET_RANGE_INCOMPLETE")
    segments.append(segment)
archive = b"".join(segments)
if hashlib.sha256(archive).hexdigest() != expected:
    raise ValueError("OFFICIAL_GIT_LFS_RELEASE_DIGEST_MISMATCH")
with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
    members = [item for item in bundle.getmembers()
               if item.isfile() and Path(item.name).name == "git-lfs"]
    if len(members) != 1:
        raise ValueError("GIT_LFS_RELEASE_BINARY_NOT_UNIQUE")
    payload = bundle.extractfile(members[0])
    if payload is None:
        raise ValueError("GIT_LFS_RELEASE_BINARY_MISSING")
    directory = runtime / "bin"
    directory.mkdir()
    binary = directory / "git-lfs"
    binary.write_bytes(payload.read())
    binary.chmod(0o755)
print("Installed node-local official git-lfs v3.8.0; archive digest verified.")
PY_LFS
    export PATH="$runtime/bin:$PATH"
fi
git lfs version
if [ ! -d "$runtime/appworld/.git" ]; then
    git init "$runtime/appworld"
    git -C "$runtime/appworld" remote add origin https://github.com/StonyBrookNLP/appworld.git
    git -C "$runtime/appworld" lfs install --local
    if [ -n "${https_proxy:-}" ]; then git -C "$runtime/appworld" config http.proxy "$https_proxy"; fi
    git -C "$runtime/appworld" config remote.origin.promisor true
    git -C "$runtime/appworld" config remote.origin.partialclonefilter blob:none
    git -C "$runtime/appworld" sparse-checkout init --cone
    git -C "$runtime/appworld" sparse-checkout set src experiments
    git -C "$runtime/appworld" fetch --depth=1 --filter=blob:none origin "$revision"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$runtime/appworld" checkout --detach FETCH_HEAD
    git -C "$runtime/appworld" lfs pull --include="src/appworld/.source/*" --exclude=""
    test "$(git -C "$runtime/appworld" rev-parse HEAD)" = "$revision"
else
    test "$(git -C "$runtime/appworld" rev-parse HEAD)" = "$revision"
    git -C "$runtime/appworld" diff --quiet HEAD -- src experiments
fi
if [ ! -x "$runtime/venv/bin/python" ]; then "$base_python" -m venv "$runtime/venv"; fi
python="$runtime/venv/bin/python"
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    "$python" -m pip install --index-url https://pypi.org/simple "$runtime/appworld" "$runtime/appworld/experiments[simplified]"
"$python" -m pip check
"$python" -m pip freeze > "$runtime/package_freeze.txt"
"$runtime/venv/bin/appworld" install
"$runtime/venv/bin/appworld" download data --version 0.2.0 --root "$runtime/benchmark"
cd "$test_source"
"$python" -m pytest -q -p no:cacheprovider Libs/population_dynamics/test_appworld_benchmark_protocol.py > "$runtime/tests.log" 2>&1
"$python" -m Libs.scripts.inspect_appworld_benchmark --data-root "$runtime/benchmark" --output "$runtime/intake"
"$python" - "$runtime" "$revision" <<'PY'
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys

runtime, revision = Path(sys.argv[1]), sys.argv[2]
payload = {
    "status": "PREPARED_RUNTIME_AND_SPLIT_METADATA",
    "observed_utc": datetime.now(timezone.utc).isoformat(),
    "job_id": os.environ["SLURM_JOB_ID"],
    "hostname": socket.gethostname(),
    "source_revision": revision,
    "source_checkout": str(runtime / "appworld"),
    "native_python": str(runtime / "venv/bin/python"),
    "data_root": str(runtime / "benchmark"),
    "packages": {name: importlib.metadata.version(name) for name in ("appworld", "appworld-agents")},
    "requested_data_version": "0.2.0",
    "package_freeze_sha256": hashlib.sha256((runtime / "package_freeze.txt").read_bytes()).hexdigest(),
    "task_contents_manually_inspected": False,
    "task_worlds_created": 0,
    "model_generations": 0,
    "formal_evaluation_released": False,
}
(runtime / "environment.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload))
PY
