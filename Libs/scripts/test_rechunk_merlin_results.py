"""Exercise result transport against a temporary filesystem-backed HDFS CLI."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import rechunk_merlin_results as rechunk


class TestRechunkMerlinResults(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rechunk-hdfs-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.remote = self.root / "hdfs"
        self.calls = self.root / "hdfs_calls.jsonl"
        binaries = self.root / "bin"
        binaries.mkdir()
        executable = binaries / "hdfs"
        executable.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""\
            import json
            import os
            from pathlib import Path
            import shutil
            import sys
            from urllib.parse import urlsplit

            arguments = sys.argv[1:]
            with Path(os.environ['RECHUNK_TEST_CALLS']).open('a') as stream:
                stream.write(json.dumps(arguments) + '\\n')

            def remote(uri):
                parsed = urlsplit(uri)
                if parsed.scheme != 'hdfs' or parsed.netloc != 'fixture' or '..' in Path(parsed.path).parts:
                    raise SystemExit('unexpected HDFS test path: ' + uri)
                return Path(os.environ['RECHUNK_TEST_HDFS']) / parsed.path.lstrip('/')

            if arguments[:3] == ['dfs', '-mkdir', '-p'] and len(arguments) == 4:
                remote(arguments[3]).mkdir(parents=True, exist_ok=True)
            elif arguments[:2] == ['dfs', '-get'] and len(arguments) == 4:
                destination = Path(arguments[3])
                if destination.exists():
                    raise SystemExit('get destination already exists')
                shutil.copyfile(remote(arguments[2]), destination)
            elif arguments[:2] == ['dfs', '-put'] and len(arguments) == 4:
                source, destination = Path(arguments[2]), remote(arguments[3])
                if destination.exists():
                    raise SystemExit('put destination already exists')
                if source.is_dir():
                    shutil.copytree(source, destination)
                else:
                    shutil.copyfile(source, destination)
            else:
                raise SystemExit('unsupported HDFS test command: ' + repr(arguments))
            """))
        executable.chmod(0o755)
        self.transport_run = os.environ.get("RH2_JOB_RUN_ID", "transport-run-fixture")
        self.transport_trial = os.environ.get("ARNOLD_TRIAL_ID", "transport-trial-fixture")
        environment = patch.dict(os.environ, {
            "PATH": str(binaries) + os.pathsep + os.environ.get("PATH", ""),
            "RECHUNK_TEST_HDFS": str(self.remote),
            "RECHUNK_TEST_CALLS": str(self.calls),
            "RH2_JOB_RUN_ID": self.transport_run,
            "ARNOLD_TRIAL_ID": self.transport_trial,
        })
        environment.start()
        self.addCleanup(environment.stop)

    def origin(self, payload):
        source = self.remote / "source"
        source.mkdir(parents=True)
        (source / "result.tar.gz").write_bytes(payload)
        return dict(status="SOURCE_FIXTURE_COMPLETE", persisted=True, exit_code=0,
                    merlin_job_run_id="source-run-fixture", merlin_trial_id="source-trial-fixture",
                    checkpoint_path="hdfs://fixture/source", result_bytes=len(payload),
                    result_sha256=hashlib.sha256(payload).hexdigest(),
                    source_metadata=dict(scope="opaque-byte-transport-fixture", preserved=[1, 2, 3]))

    def restored_chunks(self, receipt, label):
        target = self.remote / "transport/result_transport" / label
        parts = []
        for row in receipt["chunks"]:
            raw = (target / row["name"]).read_bytes()
            self.assertLess(len(raw), 4 * 1024 * 1024)
            part = base64.b64decode(json.loads(raw)["content_base64"], validate=True)
            self.assertEqual(len(part), row["bytes"])
            self.assertEqual(hashlib.sha256(part).hexdigest(), row["sha256"])
            parts.append(part)
        return b"".join(parts)

    def test_two_mib_chunks_preserve_bundle_origin_and_transport_identity(self):
        payload = bytes(range(256)) * (4 * 1024 * 1024 // 256) + b"final partial chunk"
        origin = self.origin(payload)
        original = deepcopy(origin)
        receipt = rechunk.republish(origin, "large", self.root / "work", "hdfs://fixture/transport",
                                    chunk_bytes=2 * 1024 * 1024)
        target = self.remote / "transport/result_transport/large"
        self.assertEqual(self.restored_chunks(receipt, "large"), payload)
        self.assertEqual((target / "result.tar.gz").read_bytes(), payload)
        self.assertEqual([row["bytes"] for row in receipt["chunks"]],
                         [2 * 1024 * 1024, 2 * 1024 * 1024, len(b"final partial chunk")])
        self.assertEqual(receipt["origin_receipt"], original)
        self.assertEqual(origin, original)
        self.assertEqual(receipt["merlin_job_run_id"], self.transport_run)
        self.assertEqual(receipt["merlin_trial_id"], self.transport_trial)
        self.assertNotEqual(receipt["merlin_job_run_id"], origin["merlin_job_run_id"])
        self.assertNotEqual(receipt["merlin_trial_id"], origin["merlin_trial_id"])
        self.assertEqual(json.loads((target / "receipt.json").read_text()), receipt)
        self.assertEqual(receipt["status"], "PASS_BYTE_IDENTICAL_RESULT_TRANSPORT_RECHUNK")
        self.assertEqual(receipt["source_simulations"], 0)
        self.assertFalse(receipt["scientific_outputs_changed"])

    def test_bad_origin_sha_preserves_failed_local_and_published_receipts(self):
        payload = b"retained source bundle"
        origin = self.origin(payload)
        origin["result_sha256"] = "0" * 64
        work = self.root / "work"
        with self.assertRaisesRegex(ValueError, "ORIGINAL_BUNDLE_IDENTITY_DIFFERS"):
            rechunk.republish(origin, "bad", work, "hdfs://fixture/transport",
                              chunk_bytes=2 * 1024 * 1024)
        local = json.loads((work / "receipt.json").read_text())
        target = self.remote / "transport/result_transport/bad"
        self.assertEqual(json.loads((target / "receipt.json").read_text()), local)
        self.assertEqual(local["status"], "FAILED_RESULT_TRANSPORT_RECHUNK")
        self.assertEqual(local["exit_code"], 1)
        self.assertFalse(local["persisted"])
        self.assertEqual(local["origin_receipt"], origin)
        self.assertEqual(local["error_type"], "ValueError")
        self.assertEqual(local["error"], "ORIGINAL_BUNDLE_IDENTITY_DIFFERS")
        self.assertNotIn("chunks", local)
        self.assertFalse((target / "result.tar.gz").exists())
        self.assertFalse((target / "chunks").exists())
        self.assertEqual((self.remote / "source/result.tar.gz").read_bytes(), payload)

    def test_default_remains_256_kib_with_complete_partial_tail(self):
        payload = bytes(range(256)) * 2048 + b"legacy default tail"
        origin = self.origin(payload)
        receipt = rechunk.republish(origin, "default", self.root / "work", "hdfs://fixture/transport")
        self.assertEqual(receipt["chunk_bytes"], 256 * 1024)
        self.assertEqual([row["bytes"] for row in receipt["chunks"]],
                         [256 * 1024, 256 * 1024, len(b"legacy default tail")])
        self.assertEqual(self.restored_chunks(receipt, "default"), payload)
        self.assertEqual(receipt["result_bytes"], origin["result_bytes"])
        self.assertEqual(receipt["result_sha256"], origin["result_sha256"])

    def test_invalid_sizes_are_rejected_before_local_or_hdfs_writes(self):
        origin = self.origin(b"source stays untouched")
        for index, invalid in enumerate((0, -1, 128 * 1024, 3 * 1024 * 1024, True, 262144.0)):
            work = self.root / f"work_{index}"
            with self.subTest(chunk_bytes=invalid):
                with self.assertRaisesRegex(ValueError, "UNSUPPORTED_RETURN_CHUNK_BYTES"):
                    rechunk.republish(origin, str(index), work, "hdfs://fixture/transport",
                                      chunk_bytes=invalid)
                self.assertFalse(work.exists())
                self.assertFalse(self.calls.exists())
                self.assertFalse((self.remote / "transport").exists())


if __name__ == "__main__":
    unittest.main()
