"""Transport-only admission checks; no model, task, or scorer runs."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "appworld_return_fragments", Path(__file__).parents[1] / "scripts/rechunk_appworld_merlin_return.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReturnFragmentsTests(unittest.TestCase):
    def source(self):
        content = b"verified diagnostic payload"
        reference = {"uri": "hdfs://source/files.json", "bytes": len(content),
                     "sha256": hashlib.sha256(content).hexdigest()}
        return {"status": "PASS_FAILED_SOURCE_DIAGNOSTICS", "exit_code": 0, "persisted": True,
                "source_exit_code": 1, "source_run_id": "original", "source_trial_id": "trial",
                "source_outcome": "failed_diagnostics", "source_status": "FAILED",
                "returned_files": reference, "result_bytes": reference["bytes"],
                "result_sha256": reference["sha256"],
                "bundle": {"uri": "hdfs://source/outputs.tar.gz", "bytes": 10, "sha256": "archive"}}

    def test_verified_failed_source_is_accepted(self):
        source = self.source()
        self.assertEqual(MODULE.validate_source(source), source["returned_files"])

    def test_missing_or_conflicting_receipt_is_rejected(self):
        for key, value in (("persisted", False), ("source_exit_code", 0), ("exit_code", 1),
                           ("result_bytes", 0), ("result_sha256", "different")):
            with self.subTest(key=key):
                source = self.source()
                source[key] = value
                with self.assertRaises(ValueError):
                    MODULE.validate_source(source)

    def test_payload_bytes_are_preserved_and_corruption_is_rejected(self):
        reference = self.source()["returned_files"]
        MODULE.verify_content(b"verified diagnostic payload", reference)
        with self.assertRaises(ValueError):
            MODULE.verify_content(b"modified diagnostic payload", reference)

    def test_archive_transport_preserves_original_identity_and_payload_route(self):
        source = self.source()
        self.assertEqual(MODULE.transport_reference(source, "source_archive"), source["bundle"])
        self.assertEqual(MODULE.transport_reference(source, "payload"), source["returned_files"])
        with self.assertRaises(ValueError):
            MODULE.transport_reference(source, "unknown")

    def test_completed_return_requires_consistent_original_outcome(self):
        source = self.source()
        source.update(status="PASS", source_exit_code=0, source_outcome="completed",
                      source_status="SCORED_DEVELOPMENT_BATCH")
        self.assertEqual(MODULE.transport_reference(source, "source_archive"), source["bundle"])
        source["source_status"] = "FAILED"
        with self.assertRaises(ValueError):
            MODULE.validate_source(source)


if __name__ == "__main__":
    unittest.main()
