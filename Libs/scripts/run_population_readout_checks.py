"""Run focused readout tests on Merlin CPU and persist their actual result."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("modules", nargs="+")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    environment = {key: os.environ[key] for key in (
        "RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "RH2_CHECKPOINT_HDFS")}
    report = dict(status="STARTED", execution_backend="merlin", environment=environment,
                  hostname=platform.node(), python=platform.python_version(),
                  modules=args.modules, source_files=json.loads(args.sources.read_text()))
    started = time.perf_counter()
    log = io.StringIO()
    code = 1
    try:
        import numpy
        import torch
        torch.set_num_threads(1)
        report.update(numpy=numpy.__version__, torch=torch.__version__, torch_threads=1,
                      cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      actual_cpu_count=os.cpu_count())
        suite = unittest.defaultTestLoader.loadTestsFromNames(args.modules)
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        code = 0 if result.wasSuccessful() and result.testsRun > 0 else 1
        report.update(status="PASS" if code == 0 else "FAILED", tests_run=result.testsRun,
                      failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
    except Exception as error:
        import traceback
        traceback.print_exc(file=log)
        report.update(status="FAILED", error_type=type(error).__name__, error=str(error))
    report.update(exit_code=code, elapsed_seconds=time.perf_counter()-started)
    (output / "tests.log").write_text(log.getvalue())
    report["tests_log_sha256"] = hashlib.sha256(log.getvalue().encode()).hexdigest()
    (output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    destination = environment["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/readout_checks"
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", destination], check=True)
    for name in ("tests.log", "result.json"):
        subprocess.run(["hdfs", "dfs", "-put", str(output/name), destination+"/"+name], check=True)
    print(json.dumps(dict(status=report["status"], exit_code=code, result_hdfs=destination+"/result.json",
                         run=environment["RH2_JOB_RUN_ID"], trial=environment["ARNOLD_TRIAL_ID"])), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
