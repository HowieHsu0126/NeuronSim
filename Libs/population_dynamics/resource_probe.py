"""Bounded node-local resource receipt for resuming the circuit campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


def snapshot():
    blocked = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                text = (path / "stat").read_text()
                if text.rsplit(")", 1)[1].split()[0] == "D":
                    blocked.append(dict(pid=int(path.name), command=text.split("(", 1)[1].rsplit(")", 1)[0]))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[key] = int(value.split()[0]) * 1024
    disks = {path: dict(zip(("total", "used", "free"), shutil.disk_usage(path)))
             for path in ("/", "/tmp", "/dev/shm")}
    return dict(recorded_utc=datetime.now(timezone.utc).isoformat(), d_state_count=len(blocked),
                blocked_processes=blocked, memory_bytes=memory, disk_bytes=disks)


def probe(output: Path, required_memory_gib: float = 0):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("NODE_RESOURCE_PROBE_REQUIRES_SLURM")
    samples = [snapshot()]
    telemetry = subprocess.check_output(["vmstat", "1", "4"], text=True)
    if samples[0]["d_state_count"] >= 15:
        time.sleep(30)
        samples.append(snapshot())
    latest = samples[-1]
    root = latest["disk_bytes"]["/"]
    reasons = []
    if root["free"] < 20 * 1024**3 or root["used"] / root["total"] >= .95:
        reasons.append("ROOT_STORAGE_CRITICAL")
    if latest["d_state_count"] >= 15:
        reasons.append("PERSISTENT_UNINTERRUPTIBLE_IO")
    if latest["memory_bytes"]["MemAvailable"] < required_memory_gib * 1024**3:
        reasons.append("INSUFFICIENT_AVAILABLE_MEMORY")
    lines = telemetry.strip().splitlines()
    headings = lines[1].split()
    values = [dict(zip(headings, line.split())) for line in lines[2:]]
    if len(values) >= 3 and all(float(row["wa"]) >= 25 for row in values[-3:]):
        reasons.append("PERSISTENT_IO_WAIT")
    if len(values) >= 3 and all(float(row["si"]) > 0 or float(row["so"]) > 0 for row in values[-3:]):
        reasons.append("PERSISTENT_SWAPPING")
    report = dict(status="PASS" if not reasons else "BLOCKED", reasons=reasons,
                  hostname=socket.gethostname(), slurm_job_id=os.environ["SLURM_JOB_ID"],
                  cpus_requested=os.environ.get("SLURM_CPUS_PER_TASK"), samples=samples,
                  vmstat=telemetry, required_memory_gib=required_memory_gib)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and not reasons:
        report["status"] = "PENDING_GPU_TELEMETRY"
        report["cuda_visible_devices"] = visible
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if visible and not reasons:
        command = [
            "nvidia-smi", "-i", visible, "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"]
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        report["allocated_gpu_query"] = dict(status="RUNNING", pid=child.pid, command=command)
        output.write_text(json.dumps(report, indent=2) + "\n")
        try:
            value, _ = child.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            # A driver query can enter uninterruptible sleep. Signal once and
            # return its identity; an unbounded wait would hide the stop reason.
            child.terminate()
            report["allocated_gpu_query"].update(status="TIMED_OUT_SIGTERM_SENT", exit_code=child.poll())
            reasons.append("GPU_TELEMETRY_UNRESPONSIVE")
        else:
            report["allocated_gpu_query"].update(status="PASS" if child.returncode == 0 else "FAILED",
                                                  exit_code=child.returncode)
            if child.returncode:
                report["allocated_gpu_query"]["output"] = value.strip()
                reasons.append("GPU_TELEMETRY_FAILED")
            else:
                report["allocated_gpu"] = value.strip()
        report["status"] = "PASS" if not reasons else "BLOCKED"
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-memory-gib", type=float, default=0)
    args = parser.parse_args()
    report = probe(args.output, args.required_memory_gib)
    print(json.dumps(report))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
