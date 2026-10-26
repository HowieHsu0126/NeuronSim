"""Provenance consumed by intake replay and immutable cohort selection."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def freeze_json(path: Path, value: Any) -> None:
    payload = canonical_json(value)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"FROZEN_ARTIFACT_CHANGE_REJECTED:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "pandas", "h5py", "pynwb", "allensdk", "PyYAML", "requests"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not_installed"
    return result


def cpu_execution_provenance() -> dict[str, Any]:
    """Validate a Merlin or legacy Slurm CPU allocation and record its identity.

    Merlin run/trial fields take precedence over Slurm compatibility fields.
    Incomplete Merlin allocations must not fall back to a Slurm identity.
    """
    environment = os.environ

    def positive_cpu_count(field: str) -> int:
        value = environment.get(field, "")
        if not value.isascii() or not value.isdecimal() or int(value) <= 0:
            raise ValueError(f"CPU_EXECUTION_INVALID_FIELD:{field}")
        return int(value)

    if any(field in environment for field in ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID")):
        for field in ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID"):
            value = environment.get(field, "")
            if not value or value != value.strip():
                raise ValueError(f"CPU_EXECUTION_INVALID_FIELD:{field}")
        if environment.get("ARNOLD_MONITOR_ROLE") != "head":
            raise ValueError("CPU_EXECUTION_INVALID_FIELD:ARNOLD_MONITOR_ROLE")
        if environment.get("ARNOLD_APPLIED_GPU_NUM") != "0":
            raise ValueError("CPU_EXECUTION_INVALID_FIELD:ARNOLD_APPLIED_GPU_NUM")
        cpus = positive_cpu_count("ARNOLD_HEAD_CPU")
        backend = dict(backend="merlin", run_id=environment["RH2_JOB_RUN_ID"],
                       trial_id=environment["ARNOLD_TRIAL_ID"])
    else:
        job = environment.get("SLURM_JOB_ID", "")
        if not job.isascii() or not job.isdecimal() or int(job) <= 0:
            raise ValueError("CPU_EXECUTION_INVALID_FIELD:SLURM_JOB_ID")
        node = environment.get("SLURMD_NODENAME")
        if node not in ("cpu03", "cpu04", "cpu05", "cpu06"):
            raise ValueError("CPU_EXECUTION_INVALID_FIELD:SLURMD_NODENAME")
        field = "SLURM_CPUS_PER_TASK" if "SLURM_CPUS_PER_TASK" in environment else "SLURM_CPUS_ON_NODE"
        cpus = positive_cpu_count(field)
        backend = dict(backend="slurm", run_id=job, trial_id=None,
                       slurm_job_id=job, slurm_node=node)
    if environment.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("CPU_EXECUTION_INVALID_FIELD:CUDA_VISIBLE_DEVICES")
    return dict(backend, hostname=socket.gethostname(), allocated_cpus=cpus, allocated_gpus=0)


def runtime_manifest(config: dict[str, Any], command: list[str], exit_code: int) -> dict[str, Any]:
    return {
        "resolved_config": config,
        "command": command,
        "exit_code": exit_code,
        "versions": versions(),
        "slurm": {key: value for key, value in os.environ.items() if key in (
            "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST",
            "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
        )},
        "machine": platform.machine(),
    }
