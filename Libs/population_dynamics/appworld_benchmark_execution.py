"""Report genuine allocation identity without changing the process environment."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import socket
import sys
from urllib.parse import urlsplit

from . import resource_probe


_MERLIN_FIELDS = ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "RH2_CHECKPOINT_HDFS")
_SLURM_NODES = frozenset({"cpu03", "cpu04", "cpu05", "cpu06", "gpu03"})


def execution_identity(require_allocation: bool = False) -> dict[str, str]:
    """Validate allocation metadata, or identify an unallocated observation.

    Historical Slurm metadata remains readable. This function does not authorize
    new Slurm jobs. Merlin identity never uses or produces Slurm identifiers.
    """
    hostname = socket.gethostname()
    if not hostname or any(character.isspace() for character in hostname):
        raise ValueError("the actual hostname is unavailable or invalid")
    short_hostname = hostname.split(".")[0].lower()
    merlin_present = any(name in os.environ for name in _MERLIN_FIELDS)
    slurm_present = "SLURM_JOB_ID" in os.environ
    if merlin_present and slurm_present:
        raise ValueError("Merlin and Slurm allocation identities conflict")
    if merlin_present:
        values = {name: os.environ.get(name, "") for name in _MERLIN_FIELDS}
        if any(not value or any(character.isspace() for character in value)
               for value in values.values()):
            raise ValueError("Merlin requires complete run, trial and checkpoint identity")
        if short_hostname == "mgmt2":
            raise ValueError("Merlin allocation cannot execute on management host mgmt2")
        run_id = values["RH2_JOB_RUN_ID"]
        trial_id = values["ARNOLD_TRIAL_ID"]
        if any("/" in value or "\\" in value or value in {".", ".."}
               for value in (run_id, trial_id)):
            raise ValueError("Merlin run and trial identifiers must be single path components")
        checkpoint = values["RH2_CHECKPOINT_HDFS"]
        parsed = urlsplit(checkpoint)
        segments = parsed.path.rstrip("/").split("/")
        if (not checkpoint.startswith("hdfs://") or parsed.scheme != "hdfs"
                or not parsed.path.startswith("/") or parsed.query or parsed.fragment
                or any(segment in {".", ".."} for segment in segments)
                or segments[-1] != run_id):
            raise ValueError("Merlin checkpoint must be an HDFS URI ending in its run ID")
        return {"execution_backend": "merlin", "hostname": hostname,
                "run_id": run_id, "trial_id": trial_id, "checkpoint_path": checkpoint}
    if slurm_present:
        job_id = os.environ["SLURM_JOB_ID"]
        if not job_id.isascii() or not job_id.isdecimal():
            raise ValueError("historical Slurm job ID must be numeric")
        if short_hostname not in _SLURM_NODES:
            raise ValueError("historical Slurm identity requires an authorized compute node")
        return {"execution_backend": "slurm", "hostname": hostname, "slurm_job_id": job_id}
    if require_allocation:
        raise ValueError("an allocated Merlin or historical Slurm execution is required")
    return {"execution_backend": "unallocated", "hostname": hostname}


def _cgroup_directory(controller: str) -> tuple[int, Path]:
    """Resolve this process's controller membership against its actual mounts."""
    memberships = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, group = line.split(":", 2)
        if controller in controllers.split(","):
            memberships.insert(0, (1, group))
        elif hierarchy == "0" and not controllers:
            memberships.append((2, group))
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        fields, filesystem = before.split(), after.split()
        mounts.append((fields[3], fields[4], filesystem[0], filesystem[2]))
    for version, group in memberships:
        for root, mountpoint, filesystem, options in mounts:
            if filesystem != ("cgroup2" if version == 2 else "cgroup"):
                continue
            if version == 1 and controller not in options.split(","):
                continue
            root = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), root)
            mountpoint = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), mountpoint)
            if ".." in Path(group).parts:
                raise ValueError("current cgroup membership contains parent traversal")
            try:
                relative = Path(group).relative_to(root)
            except ValueError:
                continue
            return version, Path(mountpoint) / relative
    raise FileNotFoundError(f"current {controller} cgroup mount is unavailable")


def _cgroup_integer(path: Path) -> int:
    value = path.read_text().strip()
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"{path}: expected a nonnegative integer, found {value!r}")
    return int(value)


def _memory_cgroup() -> dict:
    version, directory = _cgroup_directory("memory")
    quota_file, current_file = (
        ("memory.max", "memory.current") if version == 2
        else ("memory.limit_in_bytes", "memory.usage_in_bytes")
    )
    quota = _cgroup_integer(directory / quota_file)
    current = _cgroup_integer(directory / current_file)
    # The v1 kernel encodes an unlimited quota as its page-aligned LONG_MAX.
    unlimited_v1 = sys.maxsize // os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PAGE_SIZE")
    if quota <= 0 or (version == 1 and quota >= unlimited_v1):
        raise ValueError(f"{directory / quota_file}: a finite positive memory quota is required")
    return {"cgroup_version": version, "cgroup_path": str(directory),
            "quota_bytes": quota, "current_bytes": current}


def _cpu_cgroup() -> dict:
    try:
        version, directory = _cgroup_directory("cpu")
        if version == 2:
            quota_text, period_text = (directory / "cpu.max").read_text().split()
            unlimited = quota_text == "max"
        else:
            quota_text = (directory / "cpu.cfs_quota_us").read_text().strip()
            period_text = (directory / "cpu.cfs_period_us").read_text().strip()
            unlimited = quota_text == "-1"
    except FileNotFoundError as error:
        return {"status": "UNAVAILABLE", "reason": str(error)}
    period = int(period_text)
    quota = None if unlimited else int(quota_text)
    if period <= 0 or (quota is not None and quota <= 0):
        raise ValueError("cgroup CPU quota and period must be positive or explicitly unlimited")
    return {"status": "UNLIMITED" if unlimited else "FINITE",
            "cgroup_version": version, "cgroup_path": str(directory),
            "quota_us": quota, "period_us": period,
            "quota_cpus": None if quota is None else quota / period}


def _platform_head_capacity() -> dict:
    """Read generated capacity for the supported single-head execution shape."""
    if os.environ.get("ARNOLD_HEAD_NUM") != "1":
        raise ValueError("masked cgroups require ARNOLD_HEAD_NUM=1")
    values = {}
    for name in ("ARNOLD_HEAD_MEM", "ARNOLD_HEAD_CPU"):
        try:
            value = float(os.environ.get(name, ""))
        except ValueError as error:
            raise ValueError(f"{name} requires positive numeric platform capacity") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} requires positive finite platform capacity")
        values[name] = value
    memory_bytes = values["ARNOLD_HEAD_MEM"] * 1024**2
    if not math.isfinite(memory_bytes) or memory_bytes < 1:
        raise ValueError("ARNOLD_HEAD_MEM requires finite positive byte capacity")
    return {"role": "head", "replicas": 1,
            "memory_capacity_bytes": int(memory_bytes),
            "memory_capacity_source": "ARNOLD_HEAD_MEM",
            "cpu_capacity": values["ARNOLD_HEAD_CPU"],
            "cpu_capacity_source": "ARNOLD_HEAD_CPU"}


def probe_resources(output: Path, required_memory_gib: float = 0) -> dict:
    """Persist allocation-aware admission without computing on GPUs."""
    identity = execution_identity(require_allocation=True)
    if identity["execution_backend"] == "slurm":
        return resource_probe.probe(output, required_memory_gib=required_memory_gib)
    if (isinstance(required_memory_gib, bool)
            or not isinstance(required_memory_gib, (int, float))
            or not math.isfinite(required_memory_gib) or required_memory_gib < 0):
        raise ValueError("required_memory_gib must be finite and nonnegative")
    report = {**identity, "status": "BLOCKED", "reasons": [],
              "required_memory_gib": required_memory_gib}
    stage = "HOST_RESOURCE"
    try:
        sample = resource_probe.snapshot()
        report["samples"] = [sample]
        host_available = sample["memory_bytes"]["MemAvailable"]
        root = sample["disk_bytes"]["/"]
        for label, value in (("MemAvailable", host_available), *root.items()):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label}: expected nonnegative byte count")
        if root["total"] <= 0 or max(root["used"], root["free"]) > root["total"]:
            raise ValueError("root disk byte counts are inconsistent")
        if root["free"] < 20 * 1024**3 or root["used"] / root["total"] >= .95:
            report["reasons"].append("ROOT_STORAGE_CRITICAL")
        stage = "MEMORY_CGROUP"
        try:
            memory = _memory_cgroup()
        except FileNotFoundError as error:
            report["scope"] = "platform_role_capacity_and_host_pressure"
            report["cgroup_absence"] = {"type": type(error).__name__, "message": str(error)}
            stage = "PLATFORM_ROLE_CAPACITY"
            capacity = _platform_head_capacity()
            report["platform_role"] = capacity
            report["cpu_capacity"] = capacity["cpu_capacity"]
            report["memory"] = {
                "host_available_bytes": host_available,
                "capacity_bytes": capacity["memory_capacity_bytes"],
                "capacity_source": capacity["memory_capacity_source"],
                "quota_bytes": None, "current_bytes": None,
                "effective_available_bytes": None, "container_usage_status": "UNAVAILABLE",
            }
            if capacity["memory_capacity_bytes"] < required_memory_gib * 1024**3:
                report["reasons"].append("INSUFFICIENT_ROLE_MEMORY_CAPACITY")
            if host_available < required_memory_gib * 1024**3:
                report["reasons"].append("INSUFFICIENT_HOST_AVAILABLE_MEMORY")
        else:
            report["scope"] = "measured_cgroup_and_host_pressure"
            available = min(host_available, max(0, memory["quota_bytes"] - memory["current_bytes"]))
            report["memory"] = {**memory, "host_available_bytes": host_available,
                                "effective_available_bytes": available}
            if available < required_memory_gib * 1024**3:
                report["reasons"].append("INSUFFICIENT_AVAILABLE_MEMORY")
        stage = "CPU_RESOURCE"
        affinity = sorted(os.sched_getaffinity(0))
        if not affinity or any(type(cpu) is not int or cpu < 0 for cpu in affinity):
            raise ValueError("actual CPU affinity is missing or invalid")
        report["cpu_affinity"] = affinity
        report["cpu_quota"] = _cpu_cgroup()
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
        report["reasons"].append(f"INVALID_{stage}_EVIDENCE")
        report["evidence_error"] = {"type": type(error).__name__, "message": str(error)}
    report["status"] = "BLOCKED" if report["reasons"] else "PASS"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
