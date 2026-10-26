"""Process-level tests. Run on Merlin CPU; no CUDA or dataset access is used."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from v1_merlin_source_batch import run_batch


FAKE_RUNTIME = r'''import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--job-file", required=True)
parser.add_argument("--gpu", type=int, required=True)
args = parser.parse_args()
assert args.gpu == 0
job = json.loads(Path(args.job_file).read_text())
root = Path(__file__).resolve().parent
control = Path(job["control_dir"])
job_id = job["job"]

def update_counter(starting):
    with (control / "counter.json").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        text = stream.read()
        counter = json.loads(text) if text else {
            "active": 0, "peak": 0, "started": [], "finished": []}
        counter["active"] += 1 if starting else -1
        counter["peak"] = max(counter["peak"], counter["active"])
        counter["started" if starting else "finished"].append(job_id)
        stream.seek(0)
        stream.truncate()
        json.dump(counter, stream)
        stream.flush()

def wait_for(path):
    deadline = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("timed out waiting for " + str(path))
        time.sleep(0.01)

update_counter(True)
(control / (job_id + ".started")).write_text("started")
print("start " + job_id, flush=True)
code = 0
try:
    for peer in job.get("wait_for_jobs", []):
        wait_for(control / (peer + ".started"))
    if job.get("behavior") == "fail":
        (control / "failure_observed").write_text("7")
        code = 7
    elif job.get("behavior") == "finish_after_failure":
        wait_for(control / "failure_observed")
        time.sleep(0.2)
    else:
        time.sleep(0.04)
finally:
    (root / "result.tar.gz").write_bytes(job_id.encode())
    (root / "receipt.json").write_text(json.dumps({
        "job_id": job_id, "original_job_file": args.job_file,
        "gpu": args.gpu, "cwd": str(Path.cwd()), "exit_code": code}))
    update_counter(False)
    (control / (job_id + ".finished")).write_text(str(code))
    print("finish " + job_id, flush=True)
sys.exit(code)
'''


class V1SourceBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = self.root / "frozen_runtime"
        self.runtime.mkdir()
        (self.runtime / "v1_remote.py").write_text(FAKE_RUNTIME, encoding="utf-8")
        (self.runtime / "pack_results.py").write_bytes(b"# frozen pack fixture\n")
        self.original_scripts = {
            path.name: path.read_bytes() for path in self.runtime.iterdir()
        }
        self.control = self.root / "control"
        self.control.mkdir()
        self.outputs = self.root / "shared_outputs"
        self.outputs.mkdir()
        self.workspace = self.root / "batch_workspace"
        environment = patch.dict(os.environ, {
            "RH2_JOB_RUN_ID": "fixture_run",
            "ARNOLD_TRIAL_ID": "fixture_trial",
            "ARNOLD_APPLIED_GPU_NUM": "1",
            "RH2_CHECKPOINT_HDFS": "/fixture_only_no_hdfs_calls",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def job(self, job_id: str, *, filename: str | None = None, **values) -> Path:
        payload = {
            "job": job_id, "mode": "source_batch",
            "output": str(self.outputs), "control_dir": str(self.control),
        }
        payload.update(values)
        path = self.root / ((filename or job_id) + ".json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def run_jobs(self, jobs, workers=2):
        return run_batch(
            runtime_dir=self.runtime, job_files=jobs,
            workspace_root=self.workspace, workers=workers,
        )

    def test_concurrency_bound_and_isolated_byte_identical_runtime(self) -> None:
        jobs = [
            self.job("job0", wait_for_jobs=["job1"]),
            self.job("job1", wait_for_jobs=["job0"]),
            self.job("job2"), self.job("job3"),
        ]
        original_jobs = {path: path.read_bytes() for path in jobs}
        result = self.run_jobs(jobs)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["status"], "SUCCEEDED")
        counter = json.loads((self.control / "counter.json").read_text())
        self.assertEqual(counter["peak"], 2)
        self.assertEqual(counter["active"], 0)
        self.assertCountEqual(counter["finished"], [f"job{i}" for i in range(4)])
        for job in result["jobs"]:
            workspace = Path(job["workspace"])
            for name, contents in self.original_scripts.items():
                self.assertEqual((workspace / name).read_bytes(), contents)
            self.assertEqual((workspace / "result.tar.gz").read_text(), job["job_id"])
            receipt = json.loads((workspace / "receipt.json").read_text())
            self.assertEqual(receipt["job_id"], job["job_id"])
            self.assertEqual(receipt["original_job_file"], job["job_file"])
            self.assertEqual(receipt["gpu"], 0)
            self.assertEqual(receipt["cwd"], str(Path.cwd()))
            self.assertIn("finish " + job["job_id"], Path(job["log"]).read_text())
            self.assertEqual((job["status"], job["exit_code"]), ("SUCCEEDED", 0))
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.runtime.iterdir()},
            self.original_scripts,
        )
        self.assertEqual({path: path.read_bytes() for path in jobs}, original_jobs)
        self.assertEqual(
            json.loads((self.workspace / "batch_results.json").read_text()), result
        )

    def test_failure_stops_dispatch_and_waits_for_active_job(self) -> None:
        jobs = [
            self.job("job0", behavior="fail", wait_for_jobs=["job1"]),
            self.job("job1", behavior="finish_after_failure"),
            self.job("job2"),
        ]
        result = self.run_jobs(jobs)
        self.assertEqual((result["status"], result["exit_code"]), ("FAILED", 7))
        self.assertEqual(
            [(job["status"], job["exit_code"]) for job in result["jobs"]],
            [("FAILED", 7), ("SUCCEEDED", 0), ("NOT_STARTED", None)],
        )
        self.assertFalse((self.control / "job2.started").exists())
        self.assertTrue((self.control / "job1.finished").exists())
        counter = json.loads((self.control / "counter.json").read_text())
        self.assertEqual(counter["active"], 0)
        self.assertCountEqual(counter["started"], ["job0", "job1"])
        self.assertIn("not_started_reason", result["jobs"][2])

    def test_invalid_batch_is_rejected_before_writes_or_launch(self) -> None:
        good = self.job("good")
        duplicate = self.job("good", filename="duplicate")
        wrong_mode = self.job("wrong_mode", mode="source_compare")
        bad_id = self.job("../unsafe", filename="bad_id")
        existing = self.job("existing")
        (self.outputs / "run_merlin_fixture_trial_v1_existing").mkdir()
        cases = [
            ([good], 0), ([good], -1), ([good], True),
            ([good, duplicate], 2), ([good, wrong_mode], 2),
            ([good, bad_id], 2), ([good, existing], 2),
        ]
        for jobs, workers in cases:
            with self.subTest(jobs=jobs, workers=workers):
                with self.assertRaises(ValueError):
                    self.run_jobs(jobs, workers)
                self.assertFalse(self.workspace.exists())
                self.assertFalse((self.control / "counter.json").exists())

    def test_existing_workspace_is_preserved_without_launch(self) -> None:
        job = self.job("job0")
        self.workspace.mkdir()
        sentinel = self.workspace / "keep.txt"
        sentinel.write_text("preserved", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "workspace already exists"):
            self.run_jobs([job])
        self.assertEqual(sentinel.read_text(), "preserved")
        self.assertEqual(list(self.workspace.iterdir()), [sentinel])
        self.assertFalse((self.control / "counter.json").exists())


if __name__ == "__main__":
    unittest.main()
