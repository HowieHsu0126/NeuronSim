"""Allocation provenance checks; all scheduler identities here are test fixtures."""
from __future__ import annotations

import json
import os

import pytest

from population_dynamics import appworld_benchmark_execution as execution


@pytest.fixture(autouse=True)
def allocation_environment(monkeypatch):
    for name in ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "RH2_CHECKPOINT_HDFS", "SLURM_JOB_ID",
                 "ARNOLD_HEAD_CPU", "ARNOLD_HEAD_MEM", "ARNOLD_HEAD_NUM", "ARNOLD_HEAD_GPU"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(execution.socket, "gethostname", lambda: "merlin-worker-7")


def merlin(monkeypatch):
    values = {"RH2_JOB_RUN_ID": "run-123", "ARNOLD_TRIAL_ID": "trial-456",
              "RH2_CHECKPOINT_HDFS": "hdfs://storage/checkpoints/20261023/run-123"}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


@pytest.mark.parametrize("required", [False, True])
def test_complete_merlin_identity_preserves_environment(monkeypatch, required):
    values = merlin(monkeypatch)
    before = dict(os.environ)
    assert execution.execution_identity(require_allocation=required) == {
        "execution_backend": "merlin", "hostname": "merlin-worker-7",
        "run_id": "run-123", "trial_id": "trial-456",
        "checkpoint_path": values["RH2_CHECKPOINT_HDFS"],
    }
    assert dict(os.environ) == before


@pytest.mark.parametrize("name", ["RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "RH2_CHECKPOINT_HDFS"])
@pytest.mark.parametrize("value", [None, "", " "])
def test_incomplete_merlin_identity_rejects_even_for_observation(monkeypatch, name, value):
    merlin(monkeypatch)
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="complete run, trial and checkpoint"):
        execution.execution_identity()


@pytest.mark.parametrize("slurm_id", ["123", ""])
def test_mixed_scheduler_identity_rejects(monkeypatch, slurm_id):
    merlin(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", slurm_id)
    with pytest.raises(ValueError, match="conflict"):
        execution.execution_identity(require_allocation=True)


@pytest.mark.parametrize("hostname", ["mgmt2", "mgmt2.cluster", "MGMT2.cluster"])
def test_merlin_rejects_management_host(monkeypatch, hostname):
    merlin(monkeypatch)
    monkeypatch.setattr(execution.socket, "gethostname", lambda: hostname)
    with pytest.raises(ValueError, match="management host"):
        execution.execution_identity(require_allocation=True)


@pytest.mark.parametrize("checkpoint", [
    "/checkpoints/20261023/run-123", "https://storage/checkpoints/run-123",
    "hdfs:/checkpoints/run-123", "hdfs://storage/checkpoints/run-123-other",
    "hdfs://storage/run-123/checkpoint", "hdfs://storage/checkpoints/run-123?run=run-123",
    "hdfs://storage/checkpoints/run-123#run-123", "hdfs://storage/../run-123",
])
def test_checkpoint_must_bind_exact_run_id(monkeypatch, checkpoint):
    merlin(monkeypatch)
    monkeypatch.setenv("RH2_CHECKPOINT_HDFS", checkpoint)
    with pytest.raises(ValueError, match="HDFS URI ending in its run ID"):
        execution.execution_identity(require_allocation=True)


@pytest.mark.parametrize("name", ["RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID"])
@pytest.mark.parametrize("value", ["../run-123", "a/b", "a\\b", ".", ".."])
def test_merlin_identifiers_cannot_be_paths(monkeypatch, name, value):
    merlin(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="single path components"):
        execution.execution_identity(require_allocation=True)


@pytest.mark.parametrize("hostname", ["cpu03", "cpu04.cluster", "cpu05", "cpu06", "gpu03"])
def test_historical_slurm_identity_uses_only_slurm_fields(monkeypatch, hostname):
    monkeypatch.setattr(execution.socket, "gethostname", lambda: hostname)
    monkeypatch.setenv("SLURM_JOB_ID", "651828")
    before = dict(os.environ)
    assert execution.execution_identity(require_allocation=True) == {
        "execution_backend": "slurm", "hostname": hostname, "slurm_job_id": "651828",
    }
    assert dict(os.environ) == before


@pytest.mark.parametrize("job_id", ["", "job-651828", "651828_1", " 651828", "１２３"])
def test_historical_slurm_requires_numeric_job_id(monkeypatch, job_id):
    monkeypatch.setattr(execution.socket, "gethostname", lambda: "cpu03")
    monkeypatch.setenv("SLURM_JOB_ID", job_id)
    with pytest.raises(ValueError, match="numeric"):
        execution.execution_identity(require_allocation=True)


@pytest.mark.parametrize("hostname", ["mgmt2", "cpu02", "merlin-worker-7"])
def test_historical_slurm_requires_authorized_node(monkeypatch, hostname):
    monkeypatch.setattr(execution.socket, "gethostname", lambda: hostname)
    monkeypatch.setenv("SLURM_JOB_ID", "651828")
    with pytest.raises(ValueError, match="authorized compute node"):
        execution.execution_identity(require_allocation=True)


def test_unallocated_observation_reports_actual_management_hostname(monkeypatch):
    monkeypatch.setattr(execution.socket, "gethostname", lambda: "mgmt2.cluster")
    assert execution.execution_identity() == {
        "execution_backend": "unallocated", "hostname": "mgmt2.cluster",
    }


def test_unallocated_execution_rejects():
    with pytest.raises(ValueError, match="allocated Merlin or historical Slurm"):
        execution.execution_identity(require_allocation=True)


@pytest.fixture
def merlin_resources(monkeypatch):
    merlin(monkeypatch)
    gib = 1024**3
    sample = {"memory_bytes": {"MemAvailable": 64 * gib},
              "disk_bytes": {"/": {"total": 200 * gib, "used": 100 * gib, "free": 100 * gib}}}
    files = {
        "/proc/self/cgroup": "0::/allocation/task\n",
        "/proc/self/mountinfo": "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        "/sys/fs/cgroup/allocation/task/memory.max": str(32 * gib),
        "/sys/fs/cgroup/allocation/task/memory.current": str(8 * gib),
        "/sys/fs/cgroup/allocation/task/cpu.max": "200000 100000\n",
    }
    original_read = execution.Path.read_text

    def read(path, *args, **kwargs):
        name = str(path)
        if name in files:
            return files[name]
        if name.startswith(("/proc/self/", "/sys/fs/cgroup/")):
            raise FileNotFoundError(name)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(execution.Path, "read_text", read)
    monkeypatch.setattr(execution.resource_probe, "snapshot", lambda: sample)
    monkeypatch.setattr(execution.os, "sched_getaffinity", lambda pid: {2, 4})
    return sample, files


def test_merlin_resources_admit_and_persist_actual_allocation(tmp_path, merlin_resources):
    output = tmp_path / "resources.json"
    report = execution.probe_resources(output, required_memory_gib=24)
    assert report["status"] == "PASS"
    assert report["execution_backend"] == "merlin"
    assert report["run_id"] == "run-123" and "slurm_job_id" not in report
    assert report["memory"]["effective_available_bytes"] == 24 * 1024**3
    assert report["cpu_affinity"] == [2, 4]
    assert report["cpu_quota"]["quota_cpus"] == 2
    assert json.loads(output.read_text()) == report


@pytest.mark.parametrize("host,quota,current,expected", [(64, 4, 3, 1), (1, 32, 8, 1), (64, 4, 5, 0)])
def test_merlin_memory_admission_respects_host_and_cgroup_limits(
    tmp_path, merlin_resources, host, quota, current, expected,
):
    sample, files = merlin_resources
    sample["memory_bytes"]["MemAvailable"] = host * 1024**3
    files["/sys/fs/cgroup/allocation/task/memory.max"] = str(quota * 1024**3)
    files["/sys/fs/cgroup/allocation/task/memory.current"] = str(current * 1024**3)
    report = execution.probe_resources(tmp_path / "resources.json", required_memory_gib=2)
    assert report["status"] == "BLOCKED"
    assert report["memory"]["effective_available_bytes"] == expected * 1024**3
    assert "INSUFFICIENT_AVAILABLE_MEMORY" in report["reasons"]


@pytest.mark.parametrize("filename,value", [
    ("memory.max", None), ("memory.max", "max"), ("memory.max", "0"),
    ("memory.max", "-1"), ("memory.max", "nan"), ("memory.current", None),
    ("memory.current", "-1"), ("memory.current", "bad"),
])
def test_invalid_or_missing_memory_cgroup_evidence_is_persisted(
    tmp_path, merlin_resources, filename, value,
):
    _, files = merlin_resources
    name = "/sys/fs/cgroup/allocation/task/" + filename
    if value is None:
        del files[name]
    else:
        files[name] = value
    output = tmp_path / "resources.json"
    report = execution.probe_resources(output)
    assert report["status"] == "BLOCKED"
    expected_reason = ("INVALID_PLATFORM_ROLE_CAPACITY_EVIDENCE" if value is None
                       else "INVALID_MEMORY_CGROUP_EVIDENCE")
    assert expected_reason in report["reasons"]
    assert report["evidence_error"]["message"]
    assert json.loads(output.read_text()) == report


@pytest.mark.parametrize("free,used", [(19, 100), (30, 190)])
def test_merlin_preserves_critical_root_disk_rejection(tmp_path, merlin_resources, free, used):
    sample, _ = merlin_resources
    sample["disk_bytes"]["/"].update(free=free * 1024**3, used=used * 1024**3)
    report = execution.probe_resources(tmp_path / "resources.json")
    assert report["status"] == "BLOCKED"
    assert "ROOT_STORAGE_CRITICAL" in report["reasons"]


@pytest.mark.parametrize("quota", [str(32 * 1024**3), "9223372036854771712"])
def test_v1_memory_and_cpu_controller_paths(tmp_path, merlin_resources, quota):
    _, files = merlin_resources
    files["/proc/self/cgroup"] = "5:memory:/task\n4:cpu,cpuacct:/task\n"
    files["/proc/self/mountinfo"] = (
        "1 0 0:1 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n"
        "2 0 0:2 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n"
    )
    files.update({
        "/sys/fs/cgroup/memory/task/memory.limit_in_bytes": quota,
        "/sys/fs/cgroup/memory/task/memory.usage_in_bytes": str(8 * 1024**3),
        "/sys/fs/cgroup/cpu/task/cpu.cfs_quota_us": "150000",
        "/sys/fs/cgroup/cpu/task/cpu.cfs_period_us": "100000",
    })
    report = execution.probe_resources(tmp_path / "resources.json", required_memory_gib=24)
    if quota == "9223372036854771712":
        assert report["status"] == "BLOCKED"
        assert "INVALID_MEMORY_CGROUP_EVIDENCE" in report["reasons"]
    else:
        assert report["status"] == "PASS"
        assert report["memory"]["cgroup_version"] == 1
        assert report["cpu_quota"]["quota_cpus"] == 1.5


def test_missing_optional_cpu_quota_remains_explicit(tmp_path, merlin_resources):
    _, files = merlin_resources
    del files["/sys/fs/cgroup/allocation/task/cpu.max"]
    report = execution.probe_resources(tmp_path / "resources.json")
    assert report["status"] == "PASS"
    assert report["cpu_quota"]["status"] == "UNAVAILABLE"


def test_historical_slurm_resource_probe_is_delegated_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(execution.socket, "gethostname", lambda: "cpu03")
    monkeypatch.setenv("SLURM_JOB_ID", "651828")
    output = tmp_path / "resources.json"
    calls = []
    delegated = {"status": "PASS", "historical_fixture": True}

    def probe(path, required_memory_gib=0):
        calls.append((path, required_memory_gib))
        return delegated

    monkeypatch.setattr(execution.resource_probe, "probe", probe)
    assert execution.probe_resources(output, required_memory_gib=3) is delegated
    assert calls == [(output, 3)]


def platform_head(monkeypatch):
    for name, value in {"ARNOLD_HEAD_CPU": "1", "ARNOLD_HEAD_MEM": "2048",
                        "ARNOLD_HEAD_NUM": "1", "ARNOLD_HEAD_GPU": "0"}.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def masked_cgroups(merlin_resources):
    sample, files = merlin_resources
    files["/proc/self/mountinfo"] = "1 0 0:1 / /sys/fs/cgroup rw - ext4 /dev/fixture rw\n"
    return sample, files


def test_masked_cgroups_use_capacity_without_inventing_usage(tmp_path, monkeypatch, masked_cgroups):
    platform_head(monkeypatch)
    output = tmp_path / "resources.json"
    report = execution.probe_resources(output, required_memory_gib=1.5)
    assert report["status"] == "PASS"
    assert report["scope"] == "platform_role_capacity_and_host_pressure"
    assert report["run_id"] == "run-123" and report["trial_id"] == "trial-456"
    assert report["memory"]["capacity_bytes"] == 2 * 1024**3
    assert report["memory"]["capacity_source"] == "ARNOLD_HEAD_MEM"
    assert report["memory"]["current_bytes"] is None
    assert report["memory"]["effective_available_bytes"] is None
    assert report["cgroup_absence"]["type"] == "FileNotFoundError"
    assert report["cpu_capacity"] == 1 and report["cpu_affinity"] == [2, 4]
    assert json.loads(output.read_text()) == report


@pytest.mark.parametrize("name,value", [
    ("ARNOLD_HEAD_CPU", None), ("ARNOLD_HEAD_MEM", None), ("ARNOLD_HEAD_NUM", None),
    ("ARNOLD_HEAD_CPU", ""), ("ARNOLD_HEAD_CPU", "bad"), ("ARNOLD_HEAD_CPU", "0"),
    ("ARNOLD_HEAD_MEM", "-1"), ("ARNOLD_HEAD_MEM", "nan"), ("ARNOLD_HEAD_MEM", "inf"),
    ("ARNOLD_HEAD_MEM", "1e308"), ("ARNOLD_HEAD_NUM", "0"), ("ARNOLD_HEAD_NUM", "2"),
    ("ARNOLD_HEAD_NUM", "1.0"),
])
def test_masked_cgroups_reject_invalid_capacity(tmp_path, monkeypatch, masked_cgroups, name, value):
    platform_head(monkeypatch)
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    output = tmp_path / "resources.json"
    report = execution.probe_resources(output, required_memory_gib=1)
    assert report["status"] == "BLOCKED"
    assert "INVALID_PLATFORM_ROLE_CAPACITY_EVIDENCE" in report["reasons"]
    assert report["cgroup_absence"]["message"]
    assert json.loads(output.read_text()) == report


@pytest.mark.parametrize("host_gib,required_gib,reason", [
    (64, 3, "INSUFFICIENT_ROLE_MEMORY_CAPACITY"),
    (0.5, 1, "INSUFFICIENT_HOST_AVAILABLE_MEMORY"),
])
def test_masked_cgroups_check_capacity_and_host_pressure_separately(
    tmp_path, monkeypatch, masked_cgroups, host_gib, required_gib, reason,
):
    platform_head(monkeypatch)
    sample, _ = masked_cgroups
    sample["memory_bytes"]["MemAvailable"] = int(host_gib * 1024**3)
    report = execution.probe_resources(tmp_path / "resources.json", required_memory_gib=required_gib)
    assert report["status"] == "BLOCKED" and reason in report["reasons"]
    assert report["memory"]["effective_available_bytes"] is None


def test_platform_capacity_does_not_hide_malformed_cgroup_evidence(tmp_path, monkeypatch, merlin_resources):
    platform_head(monkeypatch)
    _, files = merlin_resources
    files["/sys/fs/cgroup/allocation/task/memory.current"] = "bad"
    report = execution.probe_resources(tmp_path / "resources.json", required_memory_gib=1)
    assert report["status"] == "BLOCKED"
    assert "INVALID_MEMORY_CGROUP_EVIDENCE" in report["reasons"]
    assert "platform_role" not in report


def test_platform_capacity_does_not_hide_measured_exhaustion(tmp_path, monkeypatch, merlin_resources):
    platform_head(monkeypatch)
    _, files = merlin_resources
    files["/sys/fs/cgroup/allocation/task/memory.max"] = str(32 * 1024**3)
    files["/sys/fs/cgroup/allocation/task/memory.current"] = str(32 * 1024**3)
    report = execution.probe_resources(tmp_path / "resources.json", required_memory_gib=1)
    assert report["status"] == "BLOCKED"
    assert "INSUFFICIENT_AVAILABLE_MEMORY" in report["reasons"]
    assert report["memory"]["effective_available_bytes"] == 0
    assert "platform_role" not in report
