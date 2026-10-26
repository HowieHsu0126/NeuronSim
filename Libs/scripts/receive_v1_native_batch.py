#!/usr/bin/env python3
"""Return the persisted frozen V1 native-family comparison on a receive host."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merlin-tools", type=Path, required=True)
    parser.add_argument("--migration-tools", type=Path, required=True)
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--control-host", required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] == "mgmt2":
        raise RuntimeError("NATIVE_BATCH_RETURN_REQUIRES_RECEIVE_HOST")
    sys.path[:0] = [str(args.merlin_tools), str(args.migration_tools)]
    # Deployment helpers live outside the project package.
    from merlin_io import atomic_json, collect
    from return_results import integrate

    receipt = json.loads(args.receipt.read_text())
    if (receipt["status"] != "PASS_FROZEN_C0_B2_NATIVE_BATCH_AND_INDEPENDENT_REPLAY"
            or receipt["exit_code"] != 0 or receipt["persisted"] is not True
            or receipt["passed_native_conditions"] != 20
            or receipt["independent_replay"]["passed_conditions"] != 10):
        raise ValueError("COMPLETE_FROZEN_NATIVE_FAMILY_RECEIPT_REQUIRED")
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(status="STARTING", pid=os.getpid(), hostname=socket.gethostname(),
                  run_sid=receipt["run_sid"], trial_sid=receipt["trial_sid"],
                  started_utc=datetime.now(timezone.utc).isoformat())

    def update(**values):
        record.update(values, observed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(args.output / "return_execution.json", record)

    update()
    tunnel = None
    try:
        tunnel = subprocess.Popen([
            "ssh", "-N", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "ConnectTimeout=10", "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            "-L", f"127.0.0.1:{args.proxy_port}:127.0.0.1:18084", args.control_host,
        ], stdin=subprocess.DEVNULL)
        time.sleep(1)
        if tunnel.poll() is not None:
            raise RuntimeError("RETURN_FORWARD_START_FAILED")
        os.environ["MERLIN_HTTP_PROXY"] = f"http://127.0.0.1:{args.proxy_port}"
        update(status="RETURNING", tunnel_pid=tunnel.pid)
        bundle = args.work / "result.tar.gz"
        collect(receipt["checkpoint_path"].rstrip("/") + "/receipt.json",
                bundle, receipt=receipt, workers=1)
        update(status="VERIFYING_AND_INTEGRATING")
        result = integrate(bundle)
        result.update(run_sid=receipt["run_sid"], trial_sid=receipt["trial_sid"],
                      remote_receipt=str(args.receipt), result_sha256=receipt["result_sha256"],
                      result_bytes=receipt["result_bytes"], new_benchmark_episodes=0,
                      production_qualified=False)
        atomic_json(args.output / "integration.json", result)
        update(status="VERIFIED_AND_INTEGRATED", exit_code=0,
               files=result["files"], canonical_files_added=result["added"])
    except BaseException as error:
        update(status="FAILED_RETURN", exit_code=1, error_type=type(error).__name__,
               error=str(error))
        raise
    finally:
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
            tunnel.wait(timeout=10)


if __name__ == "__main__":
    main()
