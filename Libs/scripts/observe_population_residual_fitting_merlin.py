"""Observe one residual fit and return its latest published checkpoints."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import subprocess
import tempfile
import time


def collect_with_curl(io, uri, target, receipt, workers):
    """Fill the existing collector cache with verified native HTTP responses."""
    cache = target.with_name(target.name + ".chunks")
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        index, row = item
        path = cache / f"{index:05d}.part"
        if path.exists():
            return
        session = io.http_session()
        with tempfile.TemporaryDirectory(prefix="native-fit-preview-") as scratch:
            header = Path(scratch) / "headers"
            header.touch(mode=0o600)
            header.write_text("\n".join(key + ": " + value for key, value in session.headers.items())
                + "\nContent-Type: application/json\nCookie: "
                + "; ".join(key + "=" + value for key, value in session.cookies.get_dict().items()) + "\n")
            for attempt in range(3):
                result = subprocess.run(["curl", "--silent", "--show-error", "--http1.1",
                    "--connect-timeout", "20", "--max-time", "90", "--proxy", session.proxies["https"],
                    "--header", "@" + str(header), "--data-binary", "@-",
                    "https://cluster-seed.example.invalid/api/v4/seed/common/hdfs-paths/preview"],
                    input=json.dumps(dict(file_path=receipt["checkpoint_path"] + "/" + row["name"])).encode(),
                    capture_output=True, timeout=100)
                tls_eof = result.returncode == 35 and b"unexpected eof" in result.stderr.lower()
                if (result.returncode not in (28, 52, 56) and not tls_eof) or attempt == 2:
                    break
                print(json.dumps(dict(retry_chunk=index, transport_exit_code=result.returncode,
                                      next_attempt=attempt + 2)), flush=True)
                time.sleep(2 ** attempt)
        chunk_uri = receipt["checkpoint_path"] + "/" + row["name"]
        if result.returncode == 35 and b"unexpected eof" in result.stderr.lower():
            response = io.call(["common", "hdfs-paths", "preview"],
                dict(file_path=chunk_uri), timeout=100)
        elif result.returncode:
            raise RuntimeError("HDFS curl preview failed: " + result.stderr.decode(errors="replace")[-500:])
        else:
            response = json.loads(result.stdout)
        status = response.get("ResponseBase", {})
        if status.get("code") == -2 and "token is expired" in str(status.get("message", "")).lower():
            # The CLI owns cn-seed token refresh. Retry only this explicit expiry
            # response, once, for the same read-only chunk URI.
            response = io.call(["common", "hdfs-paths", "preview"],
                {"file_path": receipt["checkpoint_path"] + "/" + row["name"]}, timeout=100)
        if response.get("ResponseBase", {}).get("code", 0) != 0 or "file_content" not in response:
            raise RuntimeError("HDFS curl preview API failed: " + str(response.get("ResponseBase"))[:500])
        value = json.loads(response["file_content"])
        data = base64.b64decode(value["content_base64"], validate=True)
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("HDFS curl preview chunk identity differs")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        print(json.dumps(dict(downloaded_chunk=index, bytes=row["bytes"])), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fetch, enumerate(receipt["chunks"])):
            pass
    return io.collect(uri, target, receipt=receipt, workers=workers)


def observe(args, io, receiver, pending):
    execution = json.loads(args.execution.read_text())
    run_id = execution["run_id"]
    output = args.execution.parent
    now = datetime.now(timezone.utc)
    observation = dict(observed_at=now.isoformat(), run_id=run_id, errors=[])

    def attempt(label, function):
        try:
            return function()
        except Exception as error:
            observation["errors"].append(dict(operation=label, error=str(error)[-1600:]))
            return None

    def command(parts, payload):
        return io.call(["job-v2", *parts], payload, timeout=60)

    def receive(record, directory):
        receipt = record["receipt"]
        target = directory / "results.tar.gz"
        saved = target.with_suffix(".remote_receipt.json")
        if saved.is_file() and target.is_file() and json.loads(saved.read_text()) == receipt:
            return dict(receipt_uri=record["receipt_uri"], returned=True, reused=True)
        if args.no_collect:
            return dict(receipt_uri=record["receipt_uri"], returned=False, download_deferred=True)
        uri = record["receipt_uri"]
        if uri not in pending:
            if args.curl_return:
                pending[uri] = receiver.submit(collect_with_curl, io, uri, target, receipt, args.workers)
            else:
                pending[uri] = receiver.submit(io.collect, uri, target, receipt=receipt, workers=args.workers)
        if not pending[uri].done():
            return dict(receipt_uri=uri, returned=False, download_running_or_queued=True)
        pending[uri].result()
        return dict(receipt_uri=record["receipt_uri"], returned=True, bytes=receipt["result_bytes"])

    run = attempt("run", lambda: command(["runs", "get"], dict(sid=run_id)))
    if run is not None:
        data = run["data"]
        observation["run"] = {key: data.get(key) for key in ("sid", "name", "status", "created_at", "completed_at")}
    terminal = observation.get("run", {}).get("status") in ("done", "failed", "stopped", "cancelled")
    trials = attempt("trials", lambda: command(["run-trials", "list"], dict(filter=dict(job_run_sid_any=[run_id]), page_size=100)))
    if trials is not None and trials.get("data"):
        trial = max(trials["data"], key=lambda row: int(row["sid"]))
        trial_id = trial["sid"]
        detail = attempt("trial", lambda: command(["run-trials", "get"], dict(sid=trial_id)))
        if detail is not None:
            trial = detail.get("data", detail)
        prefix = trial.get("launch_config", {}).get("CheckpointHdfs")
        observation["trial"] = {key: trial.get(key) for key in ("sid", "job_run_sid", "status", "start_at", "completed_at")}
        observation["trial"]["checkpoint_path"] = prefix
        pods = attempt("pods", lambda: command(["run-trial-pods", "list"], dict(filter=dict(trial_sid_any=[trial_id]), page_size=100)))
        if pods is not None:
            observation["pods"] = [{key: row.get(key) for key in ("sid", "pod_name", "status", "exit_code", "exit_reason")} for row in pods.get("data", [])]
        logs = attempt("logs", lambda: command(["run-trial-tls-logs", "get"], dict(filter=dict(trial_sid=trial_id,
            stream="std", keyword="fitting_observation", timestamp_ms_from=int(time.time()*1000)-900000,
            timestamp_ms_to=int(time.time()*1000)), page_size=100)))
        if logs is not None:
            # Keep workload telemetry and errors; do not retain launcher environments.
            messages = []
            def visit(value):
                if isinstance(value, dict):
                    for child in value.values():
                        visit(child)
                elif isinstance(value, list):
                    for child in value:
                        visit(child)
                elif isinstance(value, str):
                    try:
                        event = json.loads(value)
                    except json.JSONDecodeError:
                        return
                    if isinstance(event, dict) and event.get("scope") == "fitting_observation":
                        messages.append(event)
            visit(logs.get("data", logs))
            observation["workload_logs"] = messages[-20:]
            observation["logs_read"] = True
        if prefix:
            progress = attempt("progress", lambda: io.preview(prefix.rstrip("/") + "/native_fitting/checkpoint_progress.json"))
            if progress is not None:
                if progress["run_id"] != run_id or str(progress["trial_id"]) != str(trial_id):
                    raise ValueError("Checkpoint progress execution identity differs")
                io.atomic_json(output / "checkpoint_progress.json", progress)
                latest = {row["relative_cell"]: row for row in progress["records"]}
                observation["checkpoints"] = []
                for cell, record in latest.items():
                    snapshot_path = record["receipt"]["checkpoint_path"].removeprefix(
                        prefix.rstrip("/") + "/native_fitting/snapshots/")
                    snapshot = snapshot_path.split("/", 1)[0]
                    if not snapshot.isdecimal():
                        raise ValueError("Checkpoint receipt is outside its execution snapshots")
                    # A terminal bundle contains all cells. Avoid downloading the
                    # same remaining checkpoints individually before that bundle.
                    returned = (dict(return_via_terminal_bundle=True) if terminal else
                        attempt("return:" + cell, lambda: receive(record, output / "checkpoint_return" / str(trial_id) / snapshot)))
                    observation["checkpoints"].append(dict(cell=cell, epoch=record["completed_epochs"],
                        resource=record["resource"], recorded_at=record["recorded_at"], return_result=returned))
            if terminal:
                uri = prefix.rstrip("/") + "/native_fitting/terminal/readout_diagnostic/receipt.json"
                receipt = attempt("terminal_receipt", lambda: io.preview(uri))
                if receipt is not None:
                    observation["terminal_receipt"] = receipt
                    if receipt["exit_code"] == 0:
                        observation["terminal_return"] = attempt("terminal_return", lambda: receive(dict(receipt=receipt, receipt_uri=uri), output / "terminal_return"))
    observation["completed_at"] = datetime.now(timezone.utc).isoformat()
    io.atomic_json(output / "observations" / (now.strftime("%Y%m%dT%H%M%S") + ".json"), observation)
    io.atomic_json(output / "observation.json", observation)
    print(json.dumps({key: observation[key] for key in ("observed_at", "run_id", "run", "checkpoints", "terminal_return", "errors")
                      if key in observation}), flush=True)
    return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--helper-dir", type=Path, required=True)
    parser.add_argument("--no-collect", action="store_true", help="Observe without downloading")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--follow", action="store_true", help="Observe every five minutes while a separate worker receives results")
    parser.add_argument("--curl-return", action="store_true", help="Use native curl for result chunks and retain collector validation")
    args = parser.parse_args()
    if platform.node().split(".")[0] == "mgmt2":
        raise RuntimeError("Checkpoint return requires a receive host")
    sys.path.insert(0, str(args.helper_dir))
    import merlin_io as io

    pending = {}
    with ThreadPoolExecutor(max_workers=1) as receiver:
        while True:
            started = time.monotonic()
            observation = observe(args, io, receiver, pending)
            if not args.follow:
                # One-shot callers wait for their requested returns, then record
                # completion without another platform query.
                for uri, future in pending.items():
                    try:
                        future.result()
                    except Exception as error:
                        observation["errors"].append(dict(operation="return:" + uri, error=str(error)[-1600:]))
                io.atomic_json(args.execution.parent / "return_status.json", dict(
                    completed_at=datetime.now(timezone.utc).isoformat(), errors=observation["errors"],
                    returned_receipts=[uri for uri, future in pending.items() if not future.exception()]))
                return int(bool(observation["errors"]))
            if (observation.get("terminal_return") or {}).get("returned"):
                return int(bool(observation["errors"]))
            terminal = observation.get("run", {}).get("status") in ("done", "failed", "stopped", "cancelled")
            # A failed transfer must not stop observation of a running GPU job.
            # Later published checkpoints use distinct receipts and remain eligible.
            if terminal and any(future.done() and future.exception() is not None for future in pending.values()):
                return 1
            if observation.get("terminal_receipt", {}).get("exit_code", 0) != 0:
                return 1
            remaining = max(0, 300 - (time.monotonic() - started))
            while remaining > 0:
                time.sleep(min(30, remaining))
                remaining = max(0, 300 - (time.monotonic() - started))


if __name__ == "__main__":
    raise SystemExit(main())
