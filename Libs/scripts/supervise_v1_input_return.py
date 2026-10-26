#!/usr/bin/env python3
"""Supervise the two verified V1 source returns on the existing receive host."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] == "mgmt2":
        raise RuntimeError("V1_RETURN_SUPERVISION_REQUIRES_RECEIVE_HOST")
    plan = json.loads(args.plan.read_text())
    record_path = Path(plan["execution_record"])
    record = dict(status="STARTING", pid=os.getpid(), host=socket.gethostname(),
                  started_utc=datetime.now(timezone.utc).isoformat(), completed_circuits=[])
    def update(**values):
        record.update(values, observed_utc=datetime.now(timezone.utc).isoformat())
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    update()
    forward = ["ssh", "-N", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=10", "-o", "ExitOnForwardFailure=yes",
               "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
               "-L", f"127.0.0.1:{plan['proxy_port']}:127.0.0.1:{plan['upstream_proxy_port']}",
               plan["control_host"]]
    tunnel = None
    try:
        tunnel = subprocess.Popen(forward, stdin=subprocess.DEVNULL)
        time.sleep(1)
        if tunnel.poll() is not None:
            raise RuntimeError("V1_RETURN_FORWARD_START_FAILED")
        with socket.create_connection(("127.0.0.1", plan["proxy_port"]), timeout=5):
            pass
        update(status="RUNNING", tunnel_pid=tunnel.pid)
        environment = dict(os.environ, MERLIN_HTTP_PROXY=f"http://127.0.0.1:{plan['proxy_port']}",
                           PYTHONDONTWRITEBYTECODE="1")
        for task in plan["returns"]:
            request = json.loads(Path(task["request"]).read_text())
            integration = Path(task["output"]) / "integration.json"
            update(active_circuit=request["circuit"], active_progress=None)
            previous = -1
            while not integration.exists():
                command = [sys.executable, "-u", "-B", plan["receiver"],
                           "--request", task["request"], "--work", task["work"],
                           "--output", task["output"], "--merlin-skill", plan["merlin_skill"],
                           "--migration-tools", plan["migration_tools"],
                           "--workers", "1", "--max-new-chunks", "50"]
                code = subprocess.run(command, env=environment, check=False).returncode
                if code:
                    raise RuntimeError("V1_RETURN_RECEIVER_FAILED:" + str(code))
                if integration.exists():
                    break
                progress = json.loads((Path(task["output"]) / "transfer_progress.json").read_text())
                if progress["status"] != "PARTIAL_CHUNK_RETURN" or progress["cached_chunks"] <= previous:
                    raise RuntimeError("V1_RETURN_NO_CONFIRMED_CHUNK_PROGRESS")
                previous = progress["cached_chunks"]
                update(active_progress=progress)
            result = json.loads(integration.read_text())
            if (result["status"] != "PASS_V1_MERLIN_INPUT_DELIVERY_INTEGRATED"
                    or result["circuit"] != request["circuit"]
                    or result["run_sid"] != request["run_sid"]
                    or result["trial_sid"] != request["trial_sid"]):
                raise RuntimeError("V1_RETURN_INTEGRATION_IDENTITY_DIFFERS")
            record["completed_circuits"].append(request["circuit"])
            update(last_integration=str(integration))
        update(status="COMPLETE_BOTH_INPUT_RETURNS", exit_code=0)
        return 0
    except BaseException as error:
        update(status="FAILED_RETURN_SUPERVISION", error_type=type(error).__name__, error=str(error), exit_code=1)
        return 1
    finally:
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
            tunnel.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
