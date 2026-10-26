#!/usr/bin/env python3
"""Return completed V1 source blocks on the receive host while control work proceeds."""
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
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--merlin-tools", type=Path, required=True)
    parser.add_argument("--migration-tools", type=Path, required=True)
    parser.add_argument("--primary-work", type=Path, required=True)
    parser.add_argument("--replay-work", type=Path, required=True)
    parser.add_argument("--control-host", required=True)
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--transport-receipts", type=Path,
                        help="Optional byte-identical republishing receipts, numbered 0 and 1")
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] == "mgmt2":
        raise RuntimeError("SOURCE_RETURN_REQUIRES_RECEIVE_HOST")
    sys.path[:0] = [str(args.merlin_tools), str(args.migration_tools)]
    # These established deployment helpers are outside the project package.
    from merlin_io import atomic_json, collect
    from return_results import integrate

    controller = args.record / "source_return_controller.json"
    state = dict(status="STARTING", pid=os.getpid(), hostname=socket.gethostname(),
                 started_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(controller, state)
    tunnel = None
    try:
        tunnel = subprocess.Popen([
            "ssh", "-N", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3", "-L",
            f"127.0.0.1:{args.proxy_port}:127.0.0.1:18084", args.control_host,
        ], stdin=subprocess.DEVNULL)
        time.sleep(1)
        if tunnel.poll() is not None:
            raise RuntimeError("SOURCE_RETURN_FORWARD_FAILED")
        os.environ["MERLIN_HTTP_PROXY"] = f"http://127.0.0.1:{args.proxy_port}"
        state["tunnel_pid"] = tunnel.pid
        for index, (label, work) in enumerate((("primary", args.primary_work), ("replay", args.replay_work))):
            receipt = json.loads((args.record / f"{label}_receipt.json").read_text())
            if (receipt["status"] != "PASS_COMPLETE_TEN_CONDITION_SOURCE_INPUT_GLIF_CUDA_BLOCK"
                    or receipt["exit_code"] != 0 or receipt["persisted"] is not True):
                raise ValueError("COMPLETED_SOURCE_RECEIPT_REQUIRED:" + label)
            state.update(status="RETURNING", active_source=label,
                         observed_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(controller, state)
            work.mkdir(parents=True, exist_ok=True)
            bundle = work / "result.tar.gz"
            transport = receipt
            if args.transport_receipts is not None:
                transport = json.loads((args.transport_receipts / f"receipt_{index}.json").read_text())
                if (transport.get("status") != "PASS_BYTE_IDENTICAL_RESULT_TRANSPORT_RECHUNK"
                        or transport.get("origin_receipt") != receipt
                        or any(transport.get(key) != receipt[key]
                               for key in ("result_bytes", "result_sha256"))):
                    raise ValueError("BYTE_IDENTICAL_SOURCE_TRANSPORT_REQUIRED:" + label)
            collect(transport["checkpoint_path"] + "/receipt.json", bundle,
                    receipt=transport, workers=4)
            result = integrate(bundle)
            result.update(run_sid=receipt["merlin_job_run_id"], trial_sid=receipt["merlin_trial_id"],
                          source_conditions=10, new_benchmark_episodes=0,
                          observed_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(args.record / f"{label}_return.json", result)
        state.update(status="BOTH_SOURCE_ARCHIVES_VERIFIED", exit_code=0)
    except BaseException as error:
        state.update(status="RETURN_INCOMPLETE", exit_code=1, error_type=type(error).__name__)
        raise
    finally:
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
            tunnel.wait(timeout=10)
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(controller, state)


if __name__ == "__main__":
    main()
