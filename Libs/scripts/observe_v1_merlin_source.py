#!/usr/bin/env python3
"""Observe one accepted V1 source batch and return its published receipts."""
import argparse
import base64
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SNAPSHOT = '''import base64,json,pathlib,subprocess
root=pathlib.Path('/tmp/neuronsim-v1-sm80-block-validation/source_workspaces')
p=root/'batch_results.json'
v={'batch_report_exists':p.is_file()}
if p.is_file():
 v['batch']=json.loads(p.read_text())
 v['log_tails']={}
 for job in v['batch']['jobs']:
  q=pathlib.Path(job['log'])
  if q.is_file():
   with q.open('rb') as f:
    f.seek(max(0,q.stat().st_size-1200))
    v['log_tails'][job['job_id']]=f.read().decode(errors='replace')
r=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
v['gpu_sample']={'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'sm_activity':'SEPARATE_GRAFANA_METRIC_REQUIRED'}
print('V1_OBSERVATION_B64='+base64.b64encode(json.dumps(v).encode()).decode())
'''


def metric_series(value):
    if isinstance(value, dict):
        if "dps" in value:
            yield value
        else:
            for item in value.values():
                yield from metric_series(item)
    elif isinstance(value, list):
        for item in value:
            yield from metric_series(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", type=Path, required=True)
    parser.add_argument("--merlin-tools", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] == "mgmt2":
        parser.error("Run the observer on the receive host")
    sys.path.insert(0, str(args.merlin_tools))
    io = importlib.import_module("merlin_io")
    args.operation.mkdir(parents=True, exist_ok=True)
    lock = (args.operation / "observer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    observation = dict(pid=os.getpid(), host=socket.gethostname(), status="WAITING_FOR_ACCEPTED_RUN")
    io.atomic_json(args.operation / "monitor.json", observation)
    deadline = time.monotonic() + 600
    while True:
        path = args.operation / "execution.json"
        execution = json.loads(path.read_text()) if path.exists() else {}
        if execution.get("run_sid"):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Accepted execution was not recorded within ten minutes")
        time.sleep(5)
    sid = execution["run_sid"]
    failures = 0
    while True:
        started = time.monotonic()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        observation = dict(pid=os.getpid(), host=socket.gethostname(), run_sid=sid,
                           observed_utc=datetime.now(timezone.utc).isoformat(), errors=[])
        try:
            response = io.http_session().post(
                "https://cluster-api.example.invalid/api/v4/merlin/training/job-run-trials/list",
                json=dict(filter=dict(job_run_sid_any=[sid]), page_size=20), timeout=(15, 45))
            response.raise_for_status()
            value = response.json()
            if value.get("ResponseBase", {}).get("code", 0) or len(value["data"]) != 1:
                raise ValueError("One unambiguous original trial is required")
            trial = value["data"][0]
            if trial["job_run_sid"] != sid:
                raise ValueError("Trial run identity differs")
            tid = str(trial["sid"])
            config = trial["launch_config"]
            checkpoint = config.get("CheckpointHdfs") or config.get("Envs", {}).get("RH2_CHECKPOINT_HDFS")
            safe = {key: trial.get(key) for key in ("sid", "job_run_sid", "status", "start_at", "completed_at")}
            safe["checkpoint_path"] = checkpoint
            io.atomic_json(args.operation / "latest_trial.json", safe)
            observation["trial"] = safe
            failures = 0
            if trial.get("start_at"):
                if not trial.get("completed_at"):
                    try:
                        result = io.call(["job-v2", "runs", "execute-script"], dict(
                            job_run_id=sid, trial_id=tid, backend="webshell", exec_timeout_seconds=30,
                            exec="python3 - <<'V1_SOURCE_OBSERVATION'\n" + SNAPSHOT + "V1_SOURCE_OBSERVATION"), timeout=65)
                        match = re.search(r"V1_OBSERVATION_B64=([A-Za-z0-9+/=]+)", json.dumps(result))
                        if not match:
                            raise ValueError("Source observation marker missing")
                        observation["source"] = json.loads(base64.b64decode(match[1]))
                    except Exception as error:
                        observation["errors"].append(dict(operation="source_snapshot", error=str(error)[:500]))
                try:
                    binding = io.call(["job-v2", "runs", "get-grafana"], dict(trial_id=tid), timeout=45)
                    url = urlsplit(binding["grafana_url"])
                    params = dict(parse_qsl(url.query))
                    end = datetime.fromisoformat(trial["completed_at"].replace("Z", "+00:00")) if trial.get("completed_at") else datetime.now(timezone.utc)
                    params["to"] = str(int(end.timestamp() * 1000))
                    params["from"] = str(int(datetime.fromisoformat(trial["start_at"].replace("Z", "+00:00")).timestamp() * 1000))
                    url = urlunsplit((url.scheme, url.netloc, url.path, urlencode(params), url.fragment))
                    observation["resource_window"] = dict(start=trial["start_at"], end=end.isoformat(), scope="Observed allocation window; not a three-hour mean")
                    for panel, name in ((7, "gpu_duty_cycle"), (113, "gpu_sm_activity")):
                        try:
                            data = io.call(["grafana", "panel-data"], dict(url=url, panel_id=panel), args.operation / (stamp + "_" + name + ".json"), timeout=65)
                            rows = []
                            for row in metric_series(data):
                                if tid not in json.dumps(row.get("tags", {})):
                                    continue
                                points = {key: val for key, val in row["dps"].items() if isinstance(val, (int, float))}
                                if points:
                                    rows.append(dict(metric=row.get("metric"), tags=row.get("tags"), samples=len(points), mean_percent=sum(points.values()) / len(points), maximum_percent=max(points.values()), first_sample=min(points), last_sample=max(points)))
                            observation[name] = rows or "UNVERIFIED"
                        except Exception as error:
                            observation[name] = "UNVERIFIED"
                            observation["errors"].append(dict(operation=name, error=str(error)[:500]))
                except Exception as error:
                    observation["errors"].append(dict(operation="grafana_binding", error=str(error)[:500]))
            returned = []
            for job in execution["source_jobs"]:
                path = args.operation / ("receipt_" + job + ".json")
                try:
                    receipt = json.loads(path.read_text()) if path.exists() else io.preview(checkpoint + "/v1/job_" + job + "/receipt.json")
                    if receipt["merlin_job_run_id"] != sid or str(receipt["merlin_trial_id"]) != tid:
                        raise ValueError("Source receipt identity differs")
                    io.atomic_json(path, receipt)
                    returned.append(dict(job=job, status=receipt["status"], exit_code=receipt["exit_code"], persisted=receipt["persisted"]))
                except Exception as error:
                    observation["errors"].append(dict(operation="receipt_" + job, error=str(error)[:250]))
                    if not trial.get("completed_at"):
                        break
            observation["returned_receipts"] = returned
            observation["status"] = "TERMINAL" if trial.get("completed_at") else "OBSERVING"
        except Exception as error:
            failures += 1
            observation.update(status="OBSERVATION_UNKNOWN", error=str(error)[:500])
        observation["finished_utc"] = datetime.now(timezone.utc).isoformat()
        observation["next_check_seconds"] = 300
        io.atomic_json(args.operation / (stamp + "_observation.json"), observation)
        io.atomic_json(args.operation / "monitor.json", observation)
        print(json.dumps({key: observation[key] for key in ("status", "observed_utc", "errors")}), flush=True)
        if observation["status"] == "TERMINAL":
            return 0 if len(returned) == len(execution["source_jobs"]) and all(row["exit_code"] == 0 and row["persisted"] for row in returned) else 2
        if failures >= 3:
            return 2
        while time.monotonic() - started < 300:
            time.sleep(min(30, 300 - (time.monotonic() - started)))


if __name__ == "__main__":
    raise SystemExit(main())
