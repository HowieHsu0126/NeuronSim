"""Exercise directory publication and the existing manifest-bound return format."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

import ca3_merlin_remote as transport


@unittest.skipUnless(os.environ.get("RH2_CHECKPOINT_HDFS"), "Requires an actual Merlin CPU trial")
class TestResultPublication(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="neuronsim-transport-validation-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        shutil.copy2(Path(__file__).with_name("pack_results.py"), self.work / "pack_results.py")
        self.output = self.work / "output"
        self.output.mkdir()
        (self.output / "fixture.bin").write_bytes(bytes(range(256)) * 4)
        self.extra = self.work / "additional_output"
        self.extra.mkdir()
        (self.extra / "metadata.json").write_text('{"artifact_role":"transport_fixture"}\n')
        (self.output / "input_reference").symlink_to(self.extra / "metadata.json")
        self.receipt = dict(status="TRANSPORT_FIXTURE", exit_code=0, persisted=False,
                            source_simulations=0, scientific_fidelity_qualified=False)

    def test_real_directory_round_trip_preserves_all_files_and_additional_roots(self):
        target = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/result_publication_validation/fixture"
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
        transport.persist(self.output, self.work, target, self.receipt,
                          chunk_bytes=128, additional_roots=(self.extra,))
        subprocess.run(["hdfs", "dfs", "-get", target + "/receipt.json",
                        str(self.work / "returned_receipt.json")], check=True)
        subprocess.run(["hdfs", "dfs", "-get", target + "/chunks",
                        str(self.work / "returned_chunks")], check=True)
        receipt = json.loads((self.work / "returned_receipt.json").read_text())
        self.assertTrue(receipt["persisted"])
        self.assertGreater(len(receipt["chunks"]), 1)
        reconstructed = bytearray()
        for row in receipt["chunks"]:
            self.assertEqual(Path(row["name"]).parts[0], "chunks")
            value = json.loads((self.work / "returned_chunks" / Path(row["name"]).name).read_text())
            data = base64.b64decode(value["content_base64"], validate=True)
            self.assertEqual(len(data), row["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])
            reconstructed.extend(data)
        self.assertEqual(bytes(reconstructed), (self.work / "result.tar.gz").read_bytes())
        self.assertEqual(hashlib.sha256(reconstructed).hexdigest(), receipt["result_sha256"])
        with tarfile.open(self.work / "result.tar.gz") as archive:
            manifest = json.load(archive.extractfile("merlin_return_manifest.json"))
            self.assertIn(str(self.extra / "metadata.json").lstrip("/"), manifest)
            self.assertNotIn(str(self.output / "input_reference").lstrip("/"), manifest)
            for name, identity in manifest.items():
                data = archive.extractfile(name).read()
                self.assertEqual(len(data), identity["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), identity["sha256"])

    def test_directory_upload_failure_withholds_final_receipt(self):
        calls = []
        def fail_directory(path, destination):
            calls.append(destination)
            if Path(path).is_dir():
                raise RuntimeError("fixture directory publication failure")
        with patch.object(transport, "hdfs_put", side_effect=fail_directory):
            with self.assertRaisesRegex(RuntimeError, "fixture directory publication failure"):
                transport.persist(self.output, self.work, "hdfs://fixture/result", self.receipt, chunk_bytes=128)
        self.assertEqual(len(calls), 2)
        self.assertFalse(any(path.endswith("/receipt.json") for path in calls))
        self.assertFalse(self.receipt["persisted"])
        self.assertFalse((self.work / "receipt.json").exists())
        self.assertEqual(list(self.work.glob("return-chunks-*")), [])


if __name__ == "__main__":
    if (os.environ.get("ARNOLD_HEAD_GPU") != "0"
            or os.environ.get("ARNOLD_APPLIED_GPU_NUM") != "0"):
        raise ValueError("MERLIN_CPU_ALLOCATION_REQUIRED")
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestResultPublication))
    passed = result.wasSuccessful() and result.testsRun == 2 and not result.skipped
    report = dict(status="PASS_RESULT_DIRECTORY_PUBLICATION" if passed else "FAILED_RESULT_DIRECTORY_PUBLICATION",
        merlin_job_run_id=os.environ["RH2_JOB_RUN_ID"], merlin_trial_id=os.environ["ARNOLD_TRIAL_ID"],
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        elapsed_seconds=time.monotonic() - started, source_simulations=0,
        transport_sha256=transport.digest(Path(transport.__file__)),
        test_sha256=transport.digest(Path(__file__)))
    output = Path(__file__).with_name("validation.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    target = os.environ["RH2_CHECKPOINT_HDFS"].rstrip("/") + "/result_publication_validation"
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", target], check=True)
    transport.hdfs_put(output, target + "/validation.json")
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if passed else 1)
