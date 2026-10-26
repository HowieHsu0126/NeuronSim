"""Regression checks for the Merlin result boundary, without source simulation."""
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import merlin_io
except ModuleNotFoundError as error:
    if error.name != "merlin_io":
        raise
    raise unittest.SkipTest("Merlin intake tests require the configured transport helper") from error
from population_dynamics import pfc_merlin_intake as intake


class ReturnBoundaryTests(unittest.TestCase):
    def test_native_zero_padded_producer_identity(self):
        task = dict(index=85, run_id="run", trial_id="trial", xie_job="643651_85")
        receipt = dict(task, status="COMPLETE", exit_code=0, persisted=True,
                       compatibility_execution_id="merlin_trial_pfc_0085", hostname="test-host")
        with patch.object(intake, "load_task", side_effect=LookupError("reached consumer")):
            with self.assertRaisesRegex(LookupError, "reached consumer"):
                intake.validate_episode(dict(domain_base="unused"), task, receipt)
            receipt["compatibility_execution_id"] = "merlin_trial_pfc_85"
            with self.assertRaisesRegex(ValueError, "MERLIN_RECEIPT_MISMATCH"):
                intake.validate_episode(dict(domain_base="unused"), task, receipt)

    def test_transfer_failure_retains_good_chunks_and_original_error(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            target, scratch, output = root / "returned/result.tar.gz", root / "scratch", root / "output"
            scratch.mkdir(); output.mkdir()
            task = dict(index=1001, run_id="run", trial_id="trial", xie_job="643652_1",
                        output_bundle=str(target), receipt_uri="fixture/receipt.json")
            rows = [dict(bytes=4, sha256=hashlib.sha256(value).hexdigest()) for value in (b"good", b"next")]
            receipt = dict(index=1001, run_id="run", trial_id="trial", xie_job="643652_1", chunks=rows)
            def fail_collect(uri, local):
                cache = local.with_name(local.name + ".chunks"); cache.mkdir()
                (cache / "00000.part").write_bytes(b"bad!")
                (cache / "00001.part").write_bytes(b"next")
                raise RuntimeError("original transfer failed")
            with patch.object(intake, "preview", return_value=receipt), patch.object(intake, "collect", side_effect=fail_collect):
                with self.assertRaisesRegex(RuntimeError, "original transfer failed"):
                    intake.receive({}, task, output, scratch)
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["error"], "original transfer failed")
            self.assertEqual(failure["preserved_chunks"], 1)
            self.assertEqual(failure["chunk_preservation_errors"][0]["index"], 0)
            self.assertEqual((target.with_name(target.name + ".chunks") / "00001.part").read_bytes(), b"next")
            self.assertFalse(target.with_suffix(".receipt.json").exists())

    def test_chunk_receive_reuses_one_session(self):
        chunks = [b"first", b"second"]
        all_bytes = b"".join(chunks)
        receipt = dict(persisted=True, exit_code=0, checkpoint_path="fixture",
                       result_bytes=len(all_bytes), result_sha256=hashlib.sha256(all_bytes).hexdigest(),
                       chunks=[dict(name=str(n), bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for n,b in enumerate(chunks)])
        sessions = []
        class Session:
            def __enter__(self): return self
            def __exit__(self, *args): return False
        session = Session()
        def preview(uri, record=None, session=None):
            if uri.endswith("receipt.json"): return receipt
            sessions.append(session)
            return dict(content_base64=base64.b64encode(chunks[int(uri.rsplit("/",1)[1])]).decode())
        with tempfile.TemporaryDirectory() as root, patch.object(merlin_io,"http_session",return_value=session) as factory, patch.object(merlin_io,"preview",side_effect=preview):
            target = Path(root) / "result.tar.gz"
            self.assertEqual(merlin_io.collect("fixture/receipt.json", target), receipt)
            self.assertEqual(target.read_bytes(), all_bytes)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(sessions, [session, session])


    def test_cli_preview_retries_transient_failure(self):
        failures = [RuntimeError("Merlin command failed: unexpected EOF"),
                    subprocess.TimeoutExpired(["merlin-cli"], 120)]
        for failure in failures:
            with self.subTest(error=type(failure).__name__), patch.dict(os.environ, {"MERLIN_PREVIEW_TRANSPORT":"cli"}), patch.object(merlin_io,"call",side_effect=[failure,{"file_content":'{"ok":true}'}]) as call, patch.object(merlin_io.time,"sleep"):
                self.assertEqual(merlin_io.preview("fixture/receipt.json"), {"ok":True})
                self.assertEqual(call.call_count, 2)

    def test_cli_preview_stops_at_retry_bound(self):
        with patch.dict(os.environ, {"MERLIN_PREVIEW_TRANSPORT":"cli"}), patch.object(merlin_io,"call",side_effect=RuntimeError("unexpected EOF")) as call, patch.object(merlin_io.time,"sleep"):
            with self.assertRaisesRegex(RuntimeError,"unexpected EOF"):
                merlin_io.preview("fixture/receipt.json")
            self.assertEqual(call.call_count, 3)

    def test_cli_preview_rejects_non_json_fallback_without_retry(self):
        with patch.dict(os.environ, {"MERLIN_PREVIEW_TRANSPORT":"cli"}), patch.object(merlin_io,"call",side_effect=RuntimeError("BAM endpoint returned non-JSON response (HTTP 200)")) as call, patch.object(merlin_io.time,"sleep"):
            with self.assertRaisesRegex(RuntimeError,"non-JSON"):
                merlin_io.preview("fixture/receipt.json")
            self.assertEqual(call.call_count, 1)

if __name__ == "__main__":
    unittest.main()
