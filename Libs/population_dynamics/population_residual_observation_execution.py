"""Publish resumable native residual fits through the existing Merlin transport."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import torch

from .population_residual_observation_fitting import fit, write_json
from .provenance import sha256_file


class CheckpointPublisher:
    """Publish a stable cell at first use, ten-minute boundaries and cell return."""

    def __init__(self, publisher, work, output, checkpoint_prefix, interval_seconds=600):
        self.publisher = publisher
        self.work, self.output = Path(work), Path(output)
        self.prefix = checkpoint_prefix.rstrip("/") + "/native_fitting"
        self.interval = interval_seconds
        self.last = {}
        self.records = []

    def __call__(self, cell, path, epoch, force=False):
        now = time.monotonic()
        relative = cell.relative_to(self.output).as_posix()
        previous = self.last.get(relative)
        if not force and previous is not None and now - previous["time"] < self.interval:
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["epoch"] != epoch:
            raise ValueError("published epoch differs from atomic checkpoint")
        if not all((cell / name).is_file() for name in ("reference.json", "reference.npz")):
            raise ValueError("checkpoint publication requires the reference pair")
        exported = (cell / "fit.json").is_file()
        identity = (epoch, exported)
        if previous is not None and previous["identity"] == identity:
            return
        binding = state["identity"]["binding"]
        resource = dict(cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            if query.returncode:
                raise RuntimeError("GPU telemetry failed: " + query.stderr[-1000:])
            resource.update(allocated_gpu=query.stdout.strip(),
                torch_peak_memory_bytes=torch.cuda.max_memory_allocated())
        record = dict(scope="completed_epoch_checkpoint", phase=binding["phase"],
            seed=binding["seed"], mode=binding["mode"], completed_epochs=epoch,
            selected_epoch=state["selected_epoch"], checkpoint_status=state["status"],
            fit_export_available=exported, source_fit_process_exit_code=None,
            best_validation_loss=state["best_loss"], resource=resource,
            run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ["ARNOLD_TRIAL_ID"],
            host=platform.node(), recorded_at=datetime.now(timezone.utc).isoformat(),
            scientific_claim_qualified=False)
        write_json(cell / "analysis.json", record)
        # Exit zero describes this completed checkpoint publication operation.
        # The source fitting process has not exited; its status is explicit above.
        destination = self.prefix + "/snapshots/" + str(len(self.records)).zfill(4)
        receipt = self.publisher.persist(self.work, cell, destination, 0)
        record.update(receipt_uri=receipt["checkpoint_path"] + "/receipt.json",
                      receipt=receipt, relative_cell=relative)
        self.records.append(record)
        write_json(self.output / "checkpoint_progress.json", dict(records=self.records,
            run_id=record["run_id"], trial_id=record["trial_id"], scientific_claim_qualified=False))
        self.publisher.hdfs("-mkdir", "-p", self.prefix)
        self.publisher.hdfs("-put", "-f", self.output / "checkpoint_progress.json",
                            self.prefix + "/checkpoint_progress.json")
        self.last[relative] = dict(time=time.monotonic(), identity=identity)
        print(json.dumps({key: record[key] for key in
            ("scope", "phase", "seed", "mode", "completed_epochs", "receipt_uri")}), flush=True)


def execute(args, publisher):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = dict(run_id=os.environ["RH2_JOB_RUN_ID"], trial_id=os.environ["ARNOLD_TRIAL_ID"],
                  host=platform.node(), exit_code=1, status="RUNNING",
                  started_at=datetime.now(timezone.utc).isoformat())
    callback = CheckpointPublisher(publisher, output.parent, output, os.environ["RH2_CHECKPOINT_HDFS"])
    try:
        summary = fit(args.inputs, args.manifest, args.config, output, args.phase,
            args.seeds, args.modes, args.device, args.epoch_limit, checkpoint_callback=callback)
        report.update(status="COMPLETE" if summary["complete"] else "PARTIAL", exit_code=0)
    except Exception:
        report.update(status="FAILED", error=traceback.format_exc())
    finally:
        report.update(elapsed_seconds=time.perf_counter() - started,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "execution.json", report)
        receipt = publisher.persist(output.parent, output,
            os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/native_fitting/terminal", report["exit_code"])
        print(json.dumps(dict(execution=report, receipt=receipt)), flush=True)
    return report["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "manifest", "config", "output", "publisher"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--publisher-sha256", required=True)
    parser.add_argument("--phase", choices=("qualify", "refit"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--modes", nargs="+", choices=("current", "history"), default=["current", "history"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--epoch-limit", type=int)
    args = parser.parse_args()
    if (platform.node().split(".")[0] == "mgmt2" or not os.environ.get("RH2_JOB_RUN_ID")
            or not os.environ.get("ARNOLD_TRIAL_ID") or not os.environ.get("RH2_CHECKPOINT_HDFS")):
        raise RuntimeError("actual Merlin execution and persistent checkpoint path required")
    if not 1 <= args.threads <= 8 or (args.epoch_limit is not None and args.epoch_limit < 1):
        raise ValueError("one to8CPU threads and positive optional epoch limit required")
    if sha256_file(args.publisher) != args.publisher_sha256:
        raise ValueError("publisher identity differs")
    spec = importlib.util.spec_from_file_location("native_residual_publisher", args.publisher)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    torch.set_num_threads(args.threads)
    raise SystemExit(execute(args, publisher))


if __name__ == "__main__":
    main()
