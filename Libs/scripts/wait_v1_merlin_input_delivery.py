#!/usr/bin/env python3
"""Run the existing C99 consumer when this Merlin job completes its source."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

request_path = Path(sys.argv[1])
request = json.loads(request_path.read_text())
work = Path(request["workspace"])
if (request["circuits"] != [99] or request["run_sid"] != os.environ["RH2_JOB_RUN_ID"]
        or request["trial_sid"] != os.environ["ARNOLD_TRIAL_ID"]):
    raise ValueError("V1_C99_WAIT_EXECUTION_IDENTITY_DIFFERS")
delivery_path = work / "circuit_099/source/delivery.json"
while True:
    if delivery_path.exists():
        data = delivery_path.read_bytes()
        try:
            delivery = json.loads(data)
            break
        except json.JSONDecodeError:
            pass  # The producer can still be writing its final JSON record.
    terminal = work / "validation/merlin_terminal.json"
    if terminal.exists():
        raise RuntimeError("V1_SOURCE_TERMINATED_WITHOUT_COMPLETE_C99_DELIVERY")
    time.sleep(15)
if delivery["status"] != "PASS_SOURCE_INPUT_BUILD_REPEATABILITY" or delivery["exit_code"]:
    raise RuntimeError("V1_C99_SOURCE_DELIVERY_FAILED")
request["expected_deliveries"] = {"99": hashlib.sha256(data).hexdigest()}
resolved = request_path.with_name("circuit_099_consumer_request.json")
with resolved.open("x") as stream:
    json.dump(request, stream)
print(json.dumps(dict(status="C99_DELIVERY_READY", delivery_sha256=request["expected_deliveries"]["99"])), flush=True)
raise SystemExit(subprocess.run([sys.executable, "-B", str(work / "v1_merlin_input_intake_remote.py"),
                                "--request", str(resolved)], check=False).returncode)
