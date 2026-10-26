#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${NEURONSIM_PYTHON:-python3}"
cd "${PROJECT_ROOT}"
actual_node="${SLURMD_NODENAME:-$(hostname -s)}"
if [[ "$actual_node" != "cpu04" ]]; then
  printf 'Resource policy violation: actual_node=%s\n' "$actual_node" >&2
  exit 64
fi

export PYTHONUNBUFFERED=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
"${PYTHON}" -m pytest -q \
  Libs/tests/test_neuronsim_data_centric_architecture_submit_jobs.py \
  Libs/tests/test_dynamics_model_training_recovery_campaign.py
