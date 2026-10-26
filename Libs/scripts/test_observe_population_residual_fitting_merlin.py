"""Only explicit expired-token responses can invoke the CLI refresh path."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from observe_population_residual_fitting_merlin import collect_with_curl


class ExpiredTokenTests(unittest.TestCase):
    def collect(self, status, replacement=b"verified", *, transport_exit=0, transport_error=b""):
        data = b"verified"
        row = dict(name="chunk_000001.json", bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        receipt = dict(checkpoint_path="hdfs://exact/checkpoint", chunks=[row])
        session = types.SimpleNamespace(headers={}, cookies=Mock(get_dict=lambda: {}), proxies={"https": "http://proxy"})
        io = types.SimpleNamespace(http_session=Mock(return_value=session), collect=Mock(return_value="COLLECTED"),
            call=Mock(return_value=dict(ResponseBase=dict(code=0), file_content=json.dumps(
                dict(content_base64=base64.b64encode(replacement).decode())))))
        result = types.SimpleNamespace(returncode=transport_exit,
            stdout=json.dumps(dict(ResponseBase=status)).encode(), stderr=transport_error)
        with tempfile.TemporaryDirectory() as directory, \
                patch("observe_population_residual_fitting_merlin.subprocess.run", return_value=result), \
                patch("observe_population_residual_fitting_merlin.time.sleep"):
            try:
                value = collect_with_curl(io, "hdfs://receipt", Path(directory) / "result.tar.gz", receipt, 1)
            except Exception as error:
                return io, error
        return io, value

    def test_expiry_refreshes_exact_chunk_once_and_continues(self):
        io, result = self.collect(dict(code=-2, message="token is expired"))
        self.assertEqual(result, "COLLECTED")
        io.call.assert_called_once_with(["common", "hdfs-paths", "preview"],
            {"file_path": "hdfs://exact/checkpoint/chunk_000001.json"}, timeout=100)

    def test_other_authorization_rejections_do_not_refresh(self):
        for status in (dict(code=-2, message="permission denied"), dict(code=-1, message="token is expired")):
            io, error = self.collect(status)
            self.assertIsInstance(error, RuntimeError)
            io.call.assert_not_called()
            io.collect.assert_not_called()

    def test_refreshed_response_still_requires_original_hash(self):
        io, error = self.collect(dict(code=-2, message="token is expired"), replacement=b"tampered")
        self.assertIsInstance(error, ValueError)
        self.assertIn("chunk identity differs", str(error))
        io.call.assert_called_once()
        io.collect.assert_not_called()

    def test_persistent_tls_eof_uses_exact_chunk_cli_read(self):
        io, result = self.collect({}, transport_exit=35,
                                  transport_error=b"SSL routines::unexpected eof while reading")
        self.assertEqual(result, "COLLECTED")
        io.call.assert_called_once_with(["common", "hdfs-paths", "preview"],
            {"file_path": "hdfs://exact/checkpoint/chunk_000001.json"}, timeout=100)

    def test_other_tls_errors_do_not_use_cli_fallback(self):
        io, error = self.collect({}, transport_exit=35, transport_error=b"certificate verification failed")
        self.assertIsInstance(error, RuntimeError)
        io.call.assert_not_called()
        io.collect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
